"""Pins the diagnostic fields and the explicit pool ceiling on the IR client.

These tests exist because the 2026-08 "Connection error" investigation could
not proceed: the error line carried no elapsed, no target, no request id, and
no statement of whether the failing socket was fresh or reused.

⚠️ The failure is reproduced for real, not simulated. `_StaleSocketServer`
answers the first request on a connection and then hangs up on the second
WITHOUT sending a response line -- which is exactly the production signature
(death inside `httpcore.../http11.py _receive_response_headers`, before one
response byte). A mocked transport would have proved only that the logging
call runs; it could not have proved that `connection_state` reports `reused`
on the very shape of failure the field was added to identify.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from src.worker.agents.http_diagnostics import (
    CONNECTION_STATE_NEW,
    CONNECTION_STATE_REUSED,
    FAILURE_EVENT,
    DiagnosticAsyncHTTPTransport,
)

_OK_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"


class _StaleSocketServer:
    """Keep-alive server that dies before response headers on request `fail_on`.

    `fail_on=2` reproduces the stale-pooled-socket case (the connection is
    healthy when httpcore takes it out of the pool, and the upstream hangs up
    only after the request is on the wire). `fail_on=1` is the positive
    control: the same failure on a connection that was just dialled, which
    must report `new` rather than `reused`.
    """

    def __init__(self, fail_on: int) -> None:
        self._fail_on = fail_on
        self._server: asyncio.Server | None = None
        self.url = ""

    async def __aenter__(self) -> _StaleSocketServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen = 0
        try:
            while True:
                try:
                    await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError, ConnectionResetError:
                    return
                seen += 1
                if seen == self._fail_on:
                    # Hang up with no status line at all.
                    return
                writer.write(_OK_RESPONSE)
                await writer.drain()
        finally:
            writer.close()


def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(FAILURE_EVENT)]


@pytest.mark.asyncio
async def test_reused_socket_failure_emits_every_diagnostic_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stale-pooled-socket failure names itself, with a joinable id."""
    async with _StaleSocketServer(fail_on=2) as server:
        transport = DiagnosticAsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(transport=transport) as client:
            first = await client.get(server.url)
            assert first.status_code == 200  # the connection is now pooled

            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="src.worker.agents.http_diagnostics"):
                with pytest.raises(httpx.HTTPError):
                    await client.get(server.url, headers={"X-Correlation-Id": "corr-abc-123"})

    records = _failure_records(caplog)
    assert len(records) == 1, "the failure must be logged exactly once"
    rec = records[0]

    # The measurement the whole investigation turned on.
    assert rec.connection_state == CONNECTION_STATE_REUSED
    # Died before one response byte -- the production signature. Keyed on
    # httpcore's `.complete` event: `.started` fires merely on ENTERING the
    # read and is True even here, so it cannot express this.
    assert rec.response_headers_received is False
    # Elapsed is present, real, and separable from a 600s read budget.
    assert isinstance(rec.elapsed_s, float)
    assert 0.0 <= rec.elapsed_s < 30.0
    # Joins to inference-router's own logs via the header already on the wire.
    assert rec.request_id == "corr-abc-123"
    assert rec.request_id_source == "correlation-id"
    # Target and method, so a burst can be attributed to an upstream.
    assert rec.target_url == server.url
    assert rec.http_method == "GET"
    assert rec.error_type

    # The rendered line carries the fields too, not just the record: the
    # deployed log pipeline reads text.
    message = rec.getMessage()
    for fragment in (
        FAILURE_EVENT,
        "connection_state=reused",
        "response_headers_received=False",
        "request_id=corr-abc-123",
        "request_id_source=correlation-id",
        "elapsed_s=",
        f"target_url={server.url}",
    ):
        assert fragment in message, f"{fragment!r} missing from {message!r}"


@pytest.mark.asyncio
async def test_fresh_socket_failure_reports_new_not_reused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Positive control: the field tracks reality in BOTH directions.

    Without this, `connection_state=reused` could simply be what the code
    always says, and would be worthless as evidence for the pool hypothesis.
    """
    async with _StaleSocketServer(fail_on=1) as server:
        transport = DiagnosticAsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(transport=transport) as client:
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="src.worker.agents.http_diagnostics"):
                with pytest.raises(httpx.HTTPError):
                    await client.get(server.url)

    records = _failure_records(caplog)
    assert len(records) == 1
    assert records[0].connection_state == CONNECTION_STATE_NEW
    assert "connection_state=new" in records[0].getMessage()


@pytest.mark.asyncio
async def test_request_id_without_correlation_header_is_labelled_local(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unjoinable id must say so rather than imply a correlation IR never saw."""
    async with _StaleSocketServer(fail_on=1) as server:
        transport = DiagnosticAsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(transport=transport) as client:
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="src.worker.agents.http_diagnostics"):
                with pytest.raises(httpx.HTTPError):
                    await client.get(server.url)

    rec = _failure_records(caplog)[0]
    assert rec.request_id_source == "local"
    assert rec.request_id  # still present, so the two lines of one run join locally


@pytest.mark.asyncio
async def test_success_path_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """The instrument is error-path only; it cannot scale log volume with traffic."""
    async with _StaleSocketServer(fail_on=99) as server:
        transport = DiagnosticAsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(transport=transport) as client:
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="src.worker.agents.http_diagnostics"):
                assert (await client.get(server.url)).status_code == 200
                assert (await client.get(server.url)).status_code == 200

    assert _failure_records(caplog) == []


@pytest.mark.asyncio
async def test_existing_trace_extension_is_chained_not_clobbered() -> None:
    """Installing our trace hook must not silently disable someone else's."""
    seen: list[str] = []

    async def _caller_trace(name: str, info: object) -> None:
        seen.append(name)

    async with _StaleSocketServer(fail_on=99) as server:
        transport = DiagnosticAsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get(server.url, extensions={"trace": _caller_trace})
            assert resp.status_code == 200

    assert "connection.connect_tcp.started" in seen


def test_pool_ceiling_settings_exist_and_are_bounded() -> None:
    """The ceiling is a real stated number, never None.

    httpcore treats None as unbounded, which is what "no `limits=` at all"
    must never quietly become if someone later makes these Optional.
    """
    from src.settings import get_settings

    settings = get_settings()
    assert isinstance(settings.ir_http_max_connections, int)
    assert isinstance(settings.ir_http_max_keepalive_connections, int)
    assert isinstance(settings.ir_http_keepalive_expiry_s, float)
    assert settings.ir_http_max_keepalive_connections <= settings.ir_http_max_connections


@pytest.mark.asyncio
async def test_agent_run_error_line_carries_elapsed_target_and_request_id(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `PydanticAIAgent.run` error line must be joinable on its own.

    This is the line the 2026-08 investigation actually had in hand -- "Error
    in PydanticAIAgent.run for agent ...: Connection error." -- and it carried
    nothing but the agent name. `request_id` is what joins it downward to the
    `ir_http_request_failed` line (which alone knows `connection_state`) and
    outward to inference-router's own logs.
    """
    from src.api.identity_context import set_correlation_context
    from src.worker.agents.pydantic_ai_agent import (
        PydanticAIAgent,
        PydanticAIAgentConfig,
    )

    config = PydanticAIAgentConfig(
        name="diag_agent",
        model_provider="openai",
        model="gpt-4o",
        db_path="",
        base_url="http://inference-router.example.com/v1",
        api_key="test-key",  # pragma: allowlist secret
        # The owner's keys admit a PROVEN platform tenant only (rulings 2026-10-01/03).
        tenant_id="0195f48b-d000-7000-8000-000000000001",
    )
    # ⚠️ `openai` is EXTERNAL and refused unless the tenant is a proven platform
    # workspace. This test asserts the shape of the error LOG LINE, not egress,
    # so it constructs the agent inside a licence-proven platform context.
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context():
        agent = PydanticAIAgent(config=config)

    async def _boom(*_a: object, **_k: object) -> object:
        # The production exception verbatim: httpx renders EOF-before-headers
        # through the OpenAI SDK as exactly "Connection error."
        raise httpx.ConnectError("Connection error.")

    monkeypatch.setattr(agent, "_run_agent_with_timeout", _boom)
    set_correlation_context("run-xyz-789", None)

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="src.worker.agents.pydantic_ai_agent"):
        with pytest.raises(Exception, match="Connection error"):
            await agent.run("hello")

    records = [r for r in caplog.records if "Error in PydanticAIAgent.run" in r.getMessage()]
    assert len(records) == 1, f"expected one run-error line, got {len(records)}"
    rec = records[0]

    # ⛔ THIS is why the investigation had nothing to join on. The code DID
    # already carry a join key -- `ir_request_id` -- but `extract_error_detail`
    # sources it from the RESPONSE header `x-ir-request-id`, and a failure that
    # dies before one response byte has no response at all. The only key that
    # existed was one that exists only when the request succeeded far enough
    # not to need it. `request_id` is request-side, so it survives.
    assert rec.ir_request_id is None
    assert rec.request_id == "run-xyz-789"
    assert rec.target_base_url == "http://inference-router.example.com/v1"
    assert rec.model_provider == "openai"
    assert isinstance(rec.elapsed_s, float)
    assert rec.agent_name == "diag_agent"

    message = rec.getMessage()
    for fragment in (
        "elapsed_s=",
        "target_base_url=http://inference-router.example.com/v1",
        "model_provider=openai",
        "request_id=run-xyz-789",
    ):
        assert fragment in message, f"{fragment!r} missing from {message!r}"
