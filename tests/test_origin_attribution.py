"""Origin attribution: identity headers outbound, `worker_name` on the event.

⛔ READ THIS BEFORE "FIXING" A CLI AGENT TO EMIT A COST EVENT.
The CLI-agent fleet (`cli_agent`, `claude_code_agent`, `codex_agent`,
`remote_cli_agent`) deliberately emits NO cost events. Its spend is already
metered downstream by inference-router: of worker_management's ~14,840
historical cost events, 14,833 had an exact inference-router twin. A second
emit here is a DOUBLE-BILL, not a fix, and `emit_byo_cost_event_task` refuses a
platform-routed `model_provider` for exactly that reason.

What was genuinely missing is ORIGIN, and these tests pin the two halves of it:

  1. Outbound identity headers. Every httpx call the CLI agents made went out
     bare — no tenant, no subject, no app, no worker — so inference-router
     recorded the spend against worker_management's own service account and
     could not say which app or worker produced it. Same spend, now
     attributable.
  2. `worker_name` on the events WM does emit. billing whitelists it as
     metadata to split one `service_key` ("worker-management") across the many
     workers that fan into it. It used to appear in `cost_events.py` only as a
     COMMENT — never written onto an event — so the field billing keeps for
     this question was always empty.

The omit-when-empty rule is load-bearing and has its own tests below: an empty
header is a PRESENT header, so `Header(default=...)` downstream cannot fire and
inference-router coerces "" to the nil sentinel. An absent header is honest; an
empty one is a claim that the tenant is nothing.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from src.api.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    NIL_TENANT_ID,
    get_identity_headers,
    set_app_id_context,
    set_correlation_context,
    set_identity_context,
)
from src.telemetry import cost_events
from src.worker.agents.codex_agent import CodexAgent, CodexAgentConfig

# An allowlisted executor origin (request_overrides.allowed_executor_origins).
GW = "http://coding-gateway.coding-gateway.svc.cluster.local:8687"

TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
SUBJECT = "subject-42"
LICENSE = "lic-7"
APP_ID = "requirements_management"
CORRELATION = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def _reset_context() -> None:
    set_identity_context("", "", "")
    set_correlation_context("", "")
    set_app_id_context("")


def _set_full_context() -> None:
    set_identity_context(TENANT, LICENSE, SUBJECT)
    set_correlation_context(CORRELATION, "plan")
    set_app_id_context(APP_ID)


@pytest.fixture(autouse=True)
def _clean_identity_context():
    """ContextVars are task-local but a test that sets one would otherwise leak
    it into the next test in the same context. Reset on both sides."""
    _reset_context()
    yield
    _reset_context()


# --------------------------------------------------------------------------
# The canonical builder
# --------------------------------------------------------------------------


def test_identity_headers_are_omitted_not_emptied() -> None:
    """No identity in context ⇒ NO headers at all, never empty-string ones.

    This is the whole reason the helper was rewritten. The previous shape
    returned all five keys unconditionally, so an unresolved tenant went out as
    `X-Tenant-Id: ""` — present, therefore un-defaultable downstream.
    """
    assert get_identity_headers() == {}


def test_nil_sentinel_tenant_is_dropped() -> None:
    """The all-zero uuid is a sentinel, not a customer.

    Forwarding it launders an unattributable call into an attributed-looking
    one, so both the tenant header and its effective-tenant twin are dropped.
    """
    set_identity_context(NIL_TENANT_ID, "", "")
    headers = get_identity_headers()
    assert "X-Tenant-Id" not in headers
    assert EFFECTIVE_TENANT_HEADER not in headers


def test_identity_headers_carry_the_full_contract() -> None:
    """A fully-populated request yields every header the httpx hook also sends."""
    _set_full_context()
    headers = get_identity_headers(worker_name="code_reviewer")
    assert headers == {
        "X-Tenant-Id": TENANT,
        EFFECTIVE_TENANT_HEADER: TENANT,
        "X-License-Id": LICENSE,
        "X-Subject-Id": SUBJECT,
        "X-Correlation-Id": CORRELATION,
        "X-Stage": "plan",
        "X-App-Id": APP_ID,
        "X-Worker-Name": "code_reviewer",
    }


def test_worker_name_is_never_guessed() -> None:
    """An unnamed caller gets NO X-Worker-Name — not a placeholder.

    Same rule as `app_id`: a guessed origin collapses distinct workers into one
    bucket, which is strictly worse than an honest absence.
    """
    _set_full_context()
    assert "X-Worker-Name" not in get_identity_headers()


# --------------------------------------------------------------------------
# Outbound: the CLI agents actually send them
# --------------------------------------------------------------------------


class _CapturingClient:
    """Stand-in for httpx.AsyncClient that records the outbound call."""

    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> _CapturingClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, **kwargs):
        type(self).captured = {"url": url, **kwargs}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"success": True, "output": "done"},
            text="",
        )


def _codex_proxy_agent(name: str = "codex_reviewer") -> CodexAgent:
    return CodexAgent(
        CodexAgentConfig(
            name=name,
            db_path="test.db",
            use_proxy=True,
            proxy_url="http://cli-proxy:9000",
        )
    )


@pytest.mark.asyncio
async def test_proxy_execution_forwards_identity_headers(monkeypatch) -> None:
    """The CLI proxy hop carries tenant/subject/app/worker, not a bare POST.

    ⛔ This asserts headers, NOT a cost event. The proxy hop's spend is metered
    downstream; what it lacked was a name for its origin.
    """
    _set_full_context()
    agent = _codex_proxy_agent()
    monkeypatch.setattr("src.worker.agents.cli_agent.httpx.AsyncClient", _CapturingClient)

    await agent._execute_via_proxy("review this")

    headers = _CapturingClient.captured["headers"]
    assert headers["X-Tenant-Id"] == TENANT
    assert headers["X-Subject-Id"] == SUBJECT
    assert headers["X-App-Id"] == APP_ID
    assert headers[EFFECTIVE_TENANT_HEADER] == TENANT
    # The origin half: which worker inside THIS service asked for the work.
    assert headers["X-Worker-Name"] == "codex_reviewer"


@pytest.mark.asyncio
async def test_proxy_execution_sends_no_empty_headers(monkeypatch) -> None:
    """With no identity resolved the proxy call goes out clean, not with blanks.

    A bare call is honest about being unattributable; a blank-header call tells
    the callee the tenant is the empty string.
    """
    agent = _codex_proxy_agent()
    monkeypatch.setattr("src.worker.agents.cli_agent.httpx.AsyncClient", _CapturingClient)

    await agent._execute_via_proxy("review this")

    headers = _CapturingClient.captured["headers"]
    assert "X-Tenant-Id" not in headers
    assert "X-App-Id" not in headers
    assert all(v for v in headers.values()), f"empty header value in {headers!r}"


@pytest.mark.asyncio
async def test_remote_executor_identity_does_not_clobber_the_bearer(monkeypatch) -> None:
    """coding-gateway gets BOTH credentials: our SA bearer and the run's origin.

    `Authorization` says which SERVICE is calling and is the only thing the
    gateway authenticates on; the identity headers say on WHOSE BEHALF. A merge
    that let an identity key win over the bearer would break auth, so the order
    is pinned here.
    """
    from src.worker.agents import remote_container_executor as rce

    _set_full_context()
    captured: ClassVar[dict[str, object]] = {}

    class _Client(_CapturingClient):
        async def post(self, url, **kwargs):
            captured.update({"url": url, **kwargs})
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"success": True, "output": "ok", "exit_code": 0},
                text="",
            )

    monkeypatch.setattr(rce.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        rce,
        "get_service_authenticator",
        lambda: SimpleNamespace(get_token=AsyncMock(return_value="sa-token")),
    )

    executor = rce.RemoteContainerExecutor(
        rce.RemoteExecutionConfig(executor_endpoint=GW, container_id="c1")
    )
    await executor.execute_agent(cli="claude", prompt="hi", worker_name="remote_reviewer")

    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer sa-token"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Tenant-Id"] == TENANT
    assert headers["X-Worker-Name"] == "remote_reviewer"


# --------------------------------------------------------------------------
# The emitted event names its worker
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_stamps_worker_name_on_the_event(monkeypatch) -> None:
    """`worker_name` reaches the metering stream as a FIELD, not a comment.

    Before this, `worker_name` occurred in cost_events.py exactly once — inside
    the `caller_client_id` comment explaining that billing whitelists it
    separately. billing kept the field; WM never filled it.
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=10,
        output_tokens=5,
        latency_ms=42,
        worker_name="principle_consistency",
    )

    fields = fake_client.xadd.call_args[0][1]
    assert fields["worker_name"] == "principle_consistency"
    # It must not be confused with the other two origin fields.
    assert fields["caller_client_id"] == ""
    assert fields["app_id"] == ""
    assert fields["service_key"] == "worker-management"


@pytest.mark.asyncio
async def test_emit_leaves_worker_name_empty_rather_than_guessing(monkeypatch) -> None:
    """No worker in scope ⇒ empty field. Never a default, never a placeholder."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )

    fields = fake_client.xadd.call_args[0][1]
    assert fields["worker_name"] == ""
    # usage_extra untouched by this change — byo stays the billing discriminator.
    assert json.loads(fields["usage_extra"])["byo"] == "true"


@pytest.mark.asyncio
async def test_emit_task_forwards_worker_name(monkeypatch) -> None:
    """The public entrypoint passes the worker through to the stream fields."""
    captured: ClassVar[dict[str, object]] = {}

    async def _fake_emit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cost_events, "_emit", _fake_emit)
    monkeypatch.setattr(
        cost_events,
        "get_settings",
        lambda: SimpleNamespace(
            metering_redis_url="redis://metering:6379/0",
            metering_costs_stream="improvement-loop:costs",
        ),
    )
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: TENANT)

    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=10,
        output_tokens=5,
        latency_ms=42,
        model_provider="anthropic",
        worker_name="principle_consistency",
    )
    # The entrypoint is fire-and-forget: it schedules `_emit` on the loop and
    # returns. Yield once so the scheduled task actually runs before asserting.
    await asyncio.sleep(0)

    assert captured["worker_name"] == "principle_consistency"
