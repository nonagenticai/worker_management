"""repetition_penalty / frequency_penalty / max_tokens / X-Correlation-Id reach inference-router.

Both inference paths are driven END TO END through their HTTP route into a REAL
pydantic-ai model whose only fake is the socket: the OpenAI SDK client's httpx
transport is swapped for a ``MockTransport`` that stands in for inference-router
and records the JSON body and the headers it received. Nothing between the
request model and the wire is mocked, because "set on a config object, never
reached the body" is exactly the failure being guarded against.

  * ``POST /workers/{name}/message`` → cached ``PydanticAIAgent`` → IR
  * ``POST /chat/completions``       → one-shot ``Agent``           → IR

No token-generating call leaves the process: the transport answers in-memory.

    uv run pytest tests/test_ir_passthrough_sampling_correlation.py -v --no-cov
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS
from tests._platform_tenant import PLATFORM_TENANT, caller_tenant, proven_tenant

AGENT_MODULE = "src.worker.agents.pydantic_ai_agent"
MSG_MODULE = "src.api.workers.message_endpoints"
CHAT_MODULE = "src.api.providers._chat_router"
PINNED_MODEL = "local-llm"
TENANT = "019edc1e-0000-7000-8000-000000000001"
CORRELATION = "550e8400-e29b-41d4-a716-446655440000"
OTHER_CORRELATION = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


class _IRStub:
    """In-memory inference-router: records (body, headers) of every request."""

    def __init__(
        self, *, finish_reason: str = "stop", served: dict[str, Any] | None = None
    ) -> None:
        self.calls: list[tuple[dict[str, Any], httpx.Headers]] = []
        self.finish_reason = finish_reason
        # IR's top-level read-back fields (served_model / served_provider /
        # served_model_root), echoed on the JSON body and on the stream chunks.
        self.served = (
            served
            if served is not None
            else {"served_model": PINNED_MODEL, "served_provider": "pgx-local"}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((body, request.headers))
        if body.get("stream"):
            return self._sse()
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-j2",
                "object": "chat.completion",
                "created": 0,
                "model": PINNED_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": self.finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                **self.served,
            },
        )

    def _sse(self) -> httpx.Response:
        def chunk(delta: dict[str, Any], finish: str | None, **extra: Any) -> str:
            payload = {
                "id": "chatcmpl-j2",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": PINNED_MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }
            return f"data: {json.dumps(payload)}\n\n"

        text = (
            chunk({"role": "assistant", "content": "ok"}, None, **self.served)
            + chunk(
                {},
                self.finish_reason,
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                **self.served,
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=text.encode()
        )

    @property
    def body(self) -> dict[str, Any]:
        assert len(self.calls) == 1, f"expected one IR request, saw {len(self.calls)}"
        return self.calls[0][0]

    @property
    def headers(self) -> httpx.Headers:
        assert len(self.calls) == 1, f"expected one IR request, saw {len(self.calls)}"
        return self.calls[0][1]


def _wire_max_tokens(body: dict[str, Any]) -> int | None:
    """The budget on the wire. pydantic-ai 1.x / openai-python 2.x serialise it as
    ``max_completion_tokens``; IR's ChatCompletionRequest accepts both names
    (``AliasChoices("max_tokens", "max_completion_tokens")``). Exactly one may appear."""
    present = [k for k in ("max_tokens", "max_completion_tokens") if k in body]
    assert len(present) <= 1, present
    return body[present[0]] if present else None


def _real_ir_agent(ir: _IRStub, **agent_kwargs: Any) -> Any:
    from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

    agent = create_pydantic_ai_agent(
        name="j2_probe",
        description="d",
        model_provider="inference_router",
        model=PINNED_MODEL,
        base_url="http://inference-router.invalid",
        tenant_id=TENANT,
        **agent_kwargs,
    )
    # The real OpenAI SDK client the provider built (with WM's outbound identity
    # hook); only its transport — the socket — is swapped.
    agent.agent.model.client._client._transport = httpx.MockTransport(ir)
    return agent


@asynccontextmanager
async def _noop_ctx(*_a: Any, **_kw: Any):
    yield "fake-lock"


_PYDANTIC_CFG = {
    "agent_role": "r",
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": PINNED_MODEL,
    "temperature": 0.0,
}


@pytest.fixture
def message_client():
    """POST /workers/{name}/message with persistence mocked and the cache
    returning whatever agent the test puts in ``holder['agent']``."""
    holder: dict[str, Any] = {}
    with (
        patch(f"{MSG_MODULE}.worker_registry") as registry,
        patch(f"{MSG_MODULE}.worker_cache") as cache,
        patch(f"{MSG_MODULE}.state_manager") as state_mgr,
        patch(f"{MSG_MODULE}.StateDatabase") as state_db_cls,
        patch(f"{MSG_MODULE}.mlx_model_registry") as mlx,
        patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()),
    ):
        state_mgr.atomic_worker_operation = _noop_ctx
        state_mgr.resolve_worker_id = AsyncMock(return_value=("wid-j2", True))
        state_mgr.validate_state_transition = MagicMock(return_value=True)
        state_db = AsyncMock()
        state_db.get_latest_worker_state = AsyncMock(return_value=None)
        state_db.add_worker_state = AsyncMock(return_value=1)
        state_db_cls.return_value = state_db
        cache.get_or_create_worker = AsyncMock(side_effect=lambda *_a, **_k: holder["agent"])
        cache.remove_worker = AsyncMock()
        cache.remove_instance = AsyncMock()
        registry.get_worker = AsyncMock(
            return_value={
                "worker_name": "j2_probe",
                "worker_id": "wid-j2",
                "db_path": "/tmp/j2.db",
                "is_agent": True,
                "config": dict(_PYDANTIC_CFG),
                "status": "active",
            }
        )
        mlx.get_model = AsyncMock(return_value=None)
        mlx.get_instance_for_model = AsyncMock(return_value=None)

        app.dependency_overrides[get_tenant_id] = lambda: TENANT

        @asynccontextmanager
        async def _fake_lifespan(_a: Any):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _fake_lifespan
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, holder, state_db
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original


def _post(client: TestClient, headers: dict[str, str] | None = None, **extra: Any):
    return client.post(
        "/workers/j2_probe/message", json={"message": "hi", **extra}, headers=headers or {}
    )


# ============================================================ /workers/{name}/message


class TestMessagePathReachesIR:
    def test_per_message_values_reach_ir_body_and_headers(self, message_client):
        """The acceptance test: body keys AND the correlation header arrive at IR,
        and usage_metrics records the same values the wire carried.

        MUTATIONS (each turns this red):
          * drop ``apply_body_correlation_id(request.correlation_id)`` in
            send_message ⇒ no X-Correlation-Id at IR;
          * drop the ``repetition_penalty`` key from ``_run_sampling_kwargs`` ⇒
            no repetition_penalty in the body.
        """
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, temperature=0.0)

        resp = _post(
            client,
            seed=7,
            repetition_penalty=1.15,
            frequency_penalty=0.25,
            correlation_id=CORRELATION,
        )

        assert resp.status_code == 200, resp.text
        body, headers = ir.body, ir.headers
        assert body["repetition_penalty"] == 1.15
        assert body["frequency_penalty"] == 0.25
        assert body["seed"] == 7
        assert body["model"] == PINNED_MODEL
        # A worker with no max_tokens on a self-hosted provider still sends the
        # resolved default. (A per-message max_tokens is folded into the config
        # that BUILDS the agent; this fixture hands back a pre-built agent, so
        # that hop is covered by the config test below and test_message_endpoints.)
        assert _wire_max_tokens(body) == DEFAULT_SELF_HOSTED_MAX_TOKENS
        assert headers.get("x-correlation-id") == CORRELATION
        um = resp.json()["usage_metrics"]
        assert um["max_tokens"] == _wire_max_tokens(body)
        assert um["correlation_id"] == CORRELATION
        assert um["repetition_penalty"] == body["repetition_penalty"]
        assert um["frequency_penalty"] == body["frequency_penalty"]
        assert um["seed"] == body["seed"]
        assert um["finish_reason"] == "stop"

    def test_nothing_set_sends_no_penalty_keys_and_no_correlation(self, message_client):
        """Absent ⇒ ABSENT on the wire (not ``null``), and nothing invented."""
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        resp = _post(client)
        assert resp.status_code == 200, resp.text
        assert "repetition_penalty" not in ir.body
        assert "frequency_penalty" not in ir.body
        assert "x-correlation-id" not in ir.headers
        # Control: the body was parsed and carried the ordinary fields.
        assert ir.body["model"] == PINNED_MODEL
        um = resp.json()["usage_metrics"]
        assert um["repetition_penalty"] is None
        assert um["correlation_id"] is None

    def test_worker_config_max_tokens_reaches_ir_and_is_recorded(self, message_client):
        """The agent path resolves max_tokens (``_resolve_max_tokens``): the
        worker's own value is sent, and usage_metrics records what was sent."""
        client, holder, _ = message_client
        ir = _IRStub(finish_reason="length")
        holder["agent"] = _real_ir_agent(ir, max_tokens=1234)
        resp = _post(client)
        assert resp.status_code == 200, resp.text
        assert _wire_max_tokens(ir.body) == 1234
        um = resp.json()["usage_metrics"]
        assert um["max_tokens"] == 1234
        assert um["finish_reason"] == "length"

    def test_penalties_on_a_non_pydantic_worker_are_a_typed_422(self, message_client):
        client, _, state_db = message_client
        from src.api.workers import message_endpoints

        with patch.object(message_endpoints, "_seed_supported", return_value=False):
            resp = _post(client, repetition_penalty=1.1)
        assert resp.status_code == 422
        assert "SAMPLING_OVERRIDE_NOT_SUPPORTED" in resp.text
        state_db.add_worker_state.assert_not_awaited()

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "x"])
    def test_repetition_penalty_is_bounded(self, message_client, bad):
        client, _, _ = message_client
        assert _post(client, repetition_penalty=bad).status_code == 422

    def test_correlation_id_must_be_a_uuid(self, message_client):
        client, _, _ = message_client
        assert _post(client, correlation_id="529894").status_code == 422


# ======================================================= correlation precedence
#
# ENABLE_AUTH=false (as in every endpoint test here) does not install
# AuthenticationMiddleware, which is what copies the inbound X-Correlation-Id
# header into the ContextVar. So header-vs-body precedence is pinned at the
# ContextVar the middleware writes and the IR hook reads, not through the route.


class TestBodyCorrelationPrecedence:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from src.api.identity_context import set_correlation_context

        yield
        set_correlation_context(None, None)

    @staticmethod
    def _resolve(header: str | None, body: str | None) -> str:
        from src.api.identity_context import (
            apply_body_correlation_id,
            get_correlation_id_from_context,
            get_stage_from_context,
            set_correlation_context,
        )

        set_correlation_context(header, "stage-x")
        apply_body_correlation_id(body)
        assert get_stage_from_context() == "stage-x"  # the stage is never clobbered
        return get_correlation_id_from_context()

    def test_body_alone_is_used(self):
        assert self._resolve(None, CORRELATION) == CORRELATION

    def test_a_valid_header_wins_over_the_body(self):
        """Same precedence as graph execution (``first_valid_correlation_id``)."""
        assert self._resolve(OTHER_CORRELATION, CORRELATION) == OTHER_CORRELATION

    def test_a_malformed_header_does_not_beat_a_valid_body(self):
        """The n8n execution id (``529894``) was measured winning over valid
        bodies on 2026-08-14; it must not here either."""
        assert self._resolve("529894", CORRELATION) == CORRELATION

    def test_no_body_leaves_the_header_relay_unchanged(self):
        """CONTROL: without a body value the pre-existing relay is untouched —
        even a non-UUID header is still forwarded as before."""
        assert self._resolve("529894", None) == "529894"


# ================================================================== agent (unit)


@pytest.mark.asyncio
async def test_config_penalties_reach_the_body_and_a_run_value_wins():
    """Worker-config values are sent; a per-run value overrides them for THAT run
    only, and the reasoning keys sharing ``extra_body`` survive the merge.

    MUTATION: replace ``_with_repetition_penalty(...)`` in ``_run_model_settings``
    with a bare ``{"repetition_penalty": ...}`` ⇒ pydantic-ai's shallow merge
    drops ``enable_reasoning`` from the run's body ⇒ red.
    """
    ir = _IRStub()
    agent = _real_ir_agent(
        ir, repetition_penalty=1.1, frequency_penalty=0.1, enable_reasoning=False
    )
    with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
        overridden = await agent.run("hi", repetition_penalty=1.3, frequency_penalty=0.4)
        plain = await agent.run("hi")

    (run_body, _), (plain_body, _) = ir.calls
    assert run_body["repetition_penalty"] == 1.3
    assert run_body["frequency_penalty"] == 0.4
    assert run_body["enable_reasoning"] is False
    assert plain_body["repetition_penalty"] == 1.1
    assert plain_body["frequency_penalty"] == 0.1
    assert plain_body["enable_reasoning"] is False
    assert overridden.usage_metrics.repetition_penalty == 1.3
    assert plain.usage_metrics.repetition_penalty == 1.1


def test_repetition_penalty_is_never_sent_to_an_external_provider():
    """An external OpenAI-compatible API rejects the vLLM-only key."""
    from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

    def settings_for(provider: str) -> dict[str, Any]:
        cfg = PydanticAIAgentConfig(
            name="x",
            model_provider=provider,
            model="m",
            db_path="",
            repetition_penalty=1.2,
            frequency_penalty=0.3,
        )
        return dict(ModelFactory.get_model_settings(cfg, MagicMock()) or {})

    external = settings_for("deepseek")
    assert "extra_body" not in external
    assert external["frequency_penalty"] == 0.3
    # Control: the same config on a self-hosted provider does carry it.
    assert settings_for("inference_router")["extra_body"] == {"repetition_penalty": 1.2}


@pytest.mark.asyncio
async def test_licence_and_correlation_headers_come_from_the_request_identity():
    """The licence IR attributes on is the introspected one on ``X-License-Id``
    (never a body/worker value — see the hook's comment); the correlation id is
    the ContextVar. Both are read at send time from the SAME ContextVars."""
    from src.api.identity_context import set_correlation_context, set_identity_context

    ir = _IRStub()
    agent = _real_ir_agent(ir)
    set_identity_context(TENANT, "lic-j2", "sub-j2")
    set_correlation_context(CORRELATION, "")
    try:
        with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
            result = await agent.run("hi")
    finally:
        set_identity_context(None, None, None)
        set_correlation_context(None, None)
    assert ir.headers.get("x-license-id") == "lic-j2"
    assert ir.headers.get("x-correlation-id") == CORRELATION
    assert result.usage_metrics.correlation_id == CORRELATION


# ============================================================== /chat/completions


@pytest.fixture
def chat_client(monkeypatch):
    # The URLs are never dialled: the transport is replaced before any request.
    monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://inference-router.invalid")
    monkeypatch.setenv("VLLM_URL", "http://vllm.invalid")
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    # Platform-tenant behaviour (any model, X-Provider honoured); the customer
    # rules are pinned in tests/test_tenant_scoped_providers.py.
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        caller_tenant(PLATFORM_TENANT),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client
    app.dependency_overrides.clear()


def _chat_with_stub(client: TestClient, ir: _IRStub, **extra: Any) -> httpx.Response:
    """POST /chat/completions with the REAL ``_build_agent``; only the socket of
    the model it builds is replaced by the IR stub."""
    from src.api.providers import _chat_router

    real_build = _chat_router._build_agent

    async def build_then_stub(req: Any, request: Any) -> Any:
        agent = await real_build(req, request)
        agent.model.client._client._transport = httpx.MockTransport(ir)
        return agent

    headers = extra.pop("headers", None) or {}
    payload = {
        "provider": extra.pop("provider", "inference_router"),
        "model": PINNED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    with patch(f"{CHAT_MODULE}._build_agent", side_effect=build_then_stub):
        return client.post("/chat/completions", json=payload, headers=headers)


# The /chat/completions body origin/main 8bd67b9 sends for a no-options request
# (the handler streams internally, hence stream / stream_options).
GOLDEN_CHAT_BODY: dict[str, Any] = {
    "messages": [{"content": "", "role": "system"}, {"content": "hi", "role": "user"}],
    "model": PINNED_MODEL,
    "stream": True,
    "stream_options": {"include_usage": True},
    "temperature": 0.1,
}


class TestChatCompletionsPathReachesIR:
    @pytest.mark.parametrize("provider", ["inference_router", "vllm"])
    def test_no_optional_fields_sends_the_pre_pr_body_byte_for_byte(self, chat_client, provider):
        """GOLDEN. A caller that sends none of the new fields (the fr chat page)
        gets exactly the body origin/main 8bd67b9 sent: temperature only — no
        default max_tokens (it truncated fr chat at 6000) and no vllm
        ``chat_template_kwargs.enable_thinking=false``. The literal below was
        captured by running this test against origin/main 8bd67b9.

        MUTATION: build the settings with ``ModelFactory.get_model_settings``
        (or add a default max_tokens) ⇒ extra keys ⇒ red.
        """
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir, provider=provider)
        assert resp.status_code == 200, resp.text
        assert ir.body == GOLDEN_CHAT_BODY
        usage = resp.json()["usage"]
        assert usage["max_tokens"] is None
        assert usage["finish_reason"] == "stop"

    def test_an_explicit_max_tokens_keeps_the_worker_path_caps(self, chat_client, monkeypatch):
        """Explicit values still go through ``_resolve_max_tokens``: clamp to the
        self-hosted time budget."""
        from src.worker.agents import pydantic_ai_agent

        monkeypatch.setattr(
            pydantic_ai_agent,
            "_get_settings",
            lambda: MagicMock(self_hosted_max_tokens_ceiling=500),
        )
        ir = _IRStub()
        with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
            resp = _chat_with_stub(chat_client, ir, max_tokens=4000)
        assert resp.status_code == 200, resp.text
        assert _wire_max_tokens(ir.body) == 500

    def test_explicit_values_and_correlation_reach_ir(self, chat_client):
        ir = _IRStub(finish_reason="length")
        resp = _chat_with_stub(
            chat_client,
            ir,
            max_tokens=321,
            seed=3,
            repetition_penalty=1.05,
            frequency_penalty=0.2,
            correlation_id=CORRELATION,
        )
        assert resp.status_code == 200, resp.text
        body, headers = ir.body, ir.headers
        assert _wire_max_tokens(body) == 321
        assert body["seed"] == 3
        assert body["repetition_penalty"] == 1.05
        assert body["frequency_penalty"] == 0.2
        assert headers.get("x-correlation-id") == CORRELATION
        usage = resp.json()["usage"]
        assert usage["max_tokens"] == 321
        assert usage["finish_reason"] == "length"

    def test_nothing_set_sends_no_penalty_seed_or_correlation(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir)
        assert resp.status_code == 200, resp.text
        for key in ("repetition_penalty", "frequency_penalty", "seed"):
            assert key not in ir.body, key
        assert "x-correlation-id" not in ir.headers


@pytest.mark.asyncio
async def test_a_malformed_correlation_is_forwarded_but_never_persisted():
    """usage_metrics.correlation_id stores a validated UUID or null. A malformed
    inbound header (the n8n execution id) is still relayed to IR as before, but
    must never land in worker_states.

    MUTATION: stamp ``get_correlation_id_from_context() or None`` instead of
    ``_uuid_or_none(...)`` ⇒ ``"529894"`` is persisted ⇒ red.
    """
    from src.api.identity_context import set_correlation_context

    ir = _IRStub()
    agent = _real_ir_agent(ir)
    set_correlation_context("529894", "")
    try:
        with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
            result = await agent.run("hi")
    finally:
        set_correlation_context(None, None)
    assert ir.headers.get("x-correlation-id") == "529894"  # relay unchanged
    assert result.usage_metrics.correlation_id is None


# ================================================== served_* read-back pass-through
#
# IR's response names what ACTUALLY served the call: served_provider,
# served_model (vLLM's served name) and served_model_root (the weights behind
# it). Both routes pass all three through, unchanged, as additive top-level
# fields; nothing already in the response changes.

SERVED = {
    "served_provider": "pgx-local",
    "served_model": "local-llm",
    "served_model_root": "RadixArk/Some-Root-NVFP4",
}


class TestServedFieldsPassThrough:
    """Additive keys: null when IR does not report them.

    MUTATION: drop ``served_model_root`` from ``IR_SERVED_FIELDS`` (or the
    ChatCompletionsResponse / MessageResponse field) ⇒ every test here is red."""

    def test_chat_completions_passes_served_fields_through(self, chat_client):
        resp = _chat_with_stub(chat_client, _IRStub(served=SERVED))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key, value in SERVED.items():
            assert body[key] == value, key
        # Additive only: the pre-existing fields are still there, unchanged.
        assert body["output"] == "ok" and body["model"] == PINNED_MODEL
        assert body["provider"] == "inference_router"

    def test_chat_completions_stream_meta_carries_served_fields(self, chat_client):
        resp = _chat_with_stub(chat_client, _IRStub(served=SERVED), stream=True)
        assert resp.status_code == 200, resp.text
        frames = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: {")
        ]
        meta = next(f["meta"] for f in frames if "meta" in f)
        for key, value in SERVED.items():
            assert meta[key] == value, key

    def test_chat_completions_absent_fields_stay_null(self, chat_client):
        resp = _chat_with_stub(chat_client, _IRStub(served={}))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert all(body[k] is None for k in SERVED)

    def test_send_worker_message_passes_served_fields_through(self, message_client):
        client, holder, _ = message_client
        holder["agent"] = _real_ir_agent(_IRStub(served=SERVED))
        resp = _post(client)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key, value in SERVED.items():
            assert body[key] == value, key
            assert body["usage_metrics"][key] == value, key


# ================================================================ X-Provider pin
#
# IR honours X-Provider as a single-provider path (no failover). WM forwards the
# caller's inbound X-Provider, else the worker config's ``ir_provider``; absent
# ⇒ no header and the request is unchanged.


class TestProviderPinForwarded:
    """MUTATION: drop the X-Provider stamp in ``make_identity_event_hook`` ⇒ the
    header tests below go red (the absent-⇒-absent control stays green)."""

    def test_chat_completions_forwards_inbound_x_provider(self, chat_client):
        from src.api.providers import _chat_router

        ir = _IRStub()
        real_build = _chat_router._build_agent

        async def build_then_stub(req: Any, request: Any) -> Any:
            agent = await real_build(req, request)
            agent.model.client._client._transport = httpx.MockTransport(ir)
            return agent

        with patch(f"{CHAT_MODULE}._build_agent", side_effect=build_then_stub):
            resp = chat_client.post(
                "/chat/completions",
                json={
                    "provider": "inference_router",
                    "model": PINNED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-Provider": "pgx-local"},
            )
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "pgx-local"

    def test_chat_completions_without_pin_sends_no_header(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir)
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers
        assert ir.body == GOLDEN_CHAT_BODY  # byte-for-byte what main sends

    def test_malformed_pin_is_dropped(self, chat_client):
        from src.api.identity_context import valid_provider_pin

        assert valid_provider_pin("pgx-local") == "pgx-local"
        assert valid_provider_pin("bad header\\r\\n") == ""
        assert valid_provider_pin("") == ""

    def test_send_worker_message_forwards_inbound_x_provider(self, message_client):
        client, holder, _ = message_client
        # An inbound pin is honoured for a PLATFORM tenant only (ruling 2026-10-01).
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        with proven_tenant(PLATFORM_TENANT):  # fr#1385: a PROVEN platform tenant
            resp = _post(client, headers={"X-Provider": "pgx-local"})
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "pgx-local"

    def test_send_worker_message_drops_a_customer_s_inbound_x_provider(self, message_client):
        """A customer must not pin IR to an upstream (that is how a commercial
        provider behind IR would be reached on the owner's key). ``TENANT`` is
        not a platform tenant. MUTATION: forward the pin unconditionally ⇒ red."""
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        resp = _post(client, headers={"X-Provider": "xai"})
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers

    def test_send_worker_message_uses_worker_config_pin(self, message_client):
        client, holder, _ = message_client
        # A stored ir_provider pin is honoured for a PLATFORM tenant only; a
        # customer's is dropped at worker build (enforce_customer_routing,
        # tests/test_tenant_byo_fix2.py). This test injects the agent directly,
        # so it says which tenant it models rather than inheriting a customer.
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, ir_provider="pgx-local")
        with proven_tenant(PLATFORM_TENANT):  # fr#1385: a PROVEN platform tenant
            resp = _post(client)
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "pgx-local"

    def test_inbound_pin_wins_over_worker_config(self, message_client):
        client, holder, _ = message_client
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, ir_provider="pgx-local")
        with proven_tenant(PLATFORM_TENANT):  # fr#1385: a PROVEN platform tenant
            resp = _post(client, headers={"X-Provider": "other-provider"})
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "other-provider"

    def test_send_worker_message_without_pin_sends_no_header(self, message_client):
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        assert _post(client).status_code == 200
        assert "x-provider" not in ir.headers

    def test_worker_config_ir_provider_is_persisted_and_reaches_the_agent_config(self):
        """The worker-config key is persisted by CRUD and reaches AgentConfig."""
        from src.api.workers.crud_endpoints import _CONFIG_FIELDS
        from src.worker.factory.agent_factory import _create_worker_configs

        assert "ir_provider" in _CONFIG_FIELDS
        _, ac = _create_worker_configs(
            "w",
            [],
            "r",
            "pydantic_ai",
            "inference_router",
            PINNED_MODEL,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ir_provider="pgx-local",
        )
        assert ac.ir_provider == "pgx-local"


class TestProviderPinNeedsProof:
    """fr#1385 (owner ruling 2026-10-03): the X-Provider pin reaches the owner's
    commercial upstreams behind IR, so it takes the SAME predicate as the owner's
    keys — membership AND a request that PROVED the tenant. An asserted platform
    tenant is treated like a customer: pin dropped, local-llm still served.

    MUTATION (chat): honour ``inbound_pin`` when ``refusal is not None`` ⇒ the
    asserted-platform chat test goes red. MUTATION (message): gate the pin on
    ``is_commercial_platform_tenant`` again ⇒ the asserted-platform message test
    goes red.
    """

    def test_chat_asserted_platform_pin_is_dropped_and_local_still_served(self, chat_client):
        ir = _IRStub()
        with caller_tenant(PLATFORM_TENANT, proven=False):
            resp = _chat_with_stub(chat_client, ir, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text  # local-llm still works
        assert ir.calls, "IR was never called"
        assert "x-provider" not in ir.headers

    def test_chat_asserted_platform_non_local_ir_model_is_typed_403(self, chat_client):
        ir = _IRStub()
        with caller_tenant(PLATFORM_TENANT, proven=False):
            resp = _chat_with_stub(chat_client, ir, model="deepseek-v4-pro")
        assert resp.status_code == 403, resp.text
        assert "PROVIDER_NOT_ENABLED_FOR_TENANT" in resp.text
        assert ir.calls == []

    def test_chat_proven_platform_non_local_ir_model_is_allowed(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir, model="deepseek-v4-pro")  # proven fixture
        assert resp.status_code == 200, resp.text

    def test_chat_customer_pin_is_dropped(self, chat_client):
        ir = _IRStub()
        with caller_tenant(TENANT, proven=True):  # proven, but not in the platform set
            resp = _chat_with_stub(chat_client, ir, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers

    def test_message_asserted_platform_pin_is_dropped(self, message_client):
        client, holder, _ = message_client
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        with proven_tenant(PLATFORM_TENANT, proven=False):
            resp = _post(client, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text  # local-llm still served
        assert "x-provider" not in ir.headers

    def test_message_proven_platform_pin_is_forwarded(self, message_client):
        client, holder, _ = message_client
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        with proven_tenant(PLATFORM_TENANT):
            resp = _post(client, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "deepseek"

    # Narrowing (fr#1385 CI fix): only a pin that can reach a COMMERCIAL upstream
    # is withheld from an unproven platform tenant. A pin to owner hardware
    # (pgx-local, provider_catalog.is_on_hardware_ir_pin) is kept. MUTATION: drop
    # every pin again when refused ⇒ the two ``local_pin_is_kept`` tests go red.

    def test_chat_asserted_platform_local_pin_is_kept(self, chat_client):
        ir = _IRStub()
        with caller_tenant(PLATFORM_TENANT, proven=False):
            resp = _chat_with_stub(chat_client, ir, headers={"X-Provider": "pgx-local"})
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "pgx-local"

    def test_message_asserted_platform_local_pin_is_kept(self, message_client):
        client, holder, _ = message_client
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        with proven_tenant(PLATFORM_TENANT, proven=False):
            resp = _post(client, headers={"X-Provider": "pgx-local"})
        assert resp.status_code == 200, resp.text
        assert ir.headers.get("x-provider") == "pgx-local"

    def test_chat_customer_local_pin_is_still_dropped(self, chat_client):
        """Customers: every pin dropped (ruling 2026-10-01) — narrowing is platform-only."""
        ir = _IRStub()
        with caller_tenant(TENANT, proven=True):
            resp = _chat_with_stub(chat_client, ir, headers={"X-Provider": "pgx-local"})
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers

    def test_message_customer_local_pin_is_still_dropped(self, message_client):
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        resp = _post(client, headers={"X-Provider": "pgx-local"})
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers

    @staticmethod
    def _unset_platform_set(monkeypatch) -> None:
        """WM ships before gitops: COMMERCIAL_PLATFORM_TENANT_IDS unset/empty ⇒
        nobody is platform — no crash, no fallback list."""
        from src.settings import reset_settings_cache

        monkeypatch.setenv("COMMERCIAL_PLATFORM_TENANT_IDS", "")
        reset_settings_cache()

    def test_message_unset_platform_set_drops_a_proven_platform_pin(
        self, message_client, monkeypatch
    ):
        self._unset_platform_set(monkeypatch)
        client, holder, _ = message_client
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir)
        with proven_tenant(PLATFORM_TENANT):
            resp = _post(client, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text
        assert "x-provider" not in ir.headers

    def test_chat_unset_platform_set_drops_a_proven_platform_pin(self, chat_client, monkeypatch):
        self._unset_platform_set(monkeypatch)
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir, headers={"X-Provider": "deepseek"})
        assert resp.status_code == 200, resp.text  # customer path: local-llm served
        assert "x-provider" not in ir.headers


# ================================================ sent-sampling read-back (echo)


class _IR503(_IRStub):
    """IR with every provider in the chain down: 503 + classification header."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((body, request.headers))
        return httpx.Response(
            503,
            headers={"X-IR-Error-Classification": "chain_deadline_exceeded"},
            json={"detail": "all providers in chain failed"},
        )


class _IRHang(_IRStub):
    """IR that accepts the request (so the body IS on the wire) and never answers.

    Deterministic, not merely slow: the agent's budget is a REAL
    ``asyncio.timeout`` created with no deadline (see ``_armed_on_receipt``),
    and this stub arms it — ``reschedule(now)`` — only once the request body
    has been recorded. The timeout therefore cannot fire before receipt, no
    matter how loaded the box is (a fixed 0.2 s budget raced the ~170 ms it
    takes the request to land and failed 2/6 cold at load ~4).
    """

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[asyncio.Timeout] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        body = json.loads(request.content)
        self.calls.append((body, request.headers))
        assert len(self.timeouts) == 1, "control: the agent run budget was not captured"
        # Start the budget NOW: the real Timeout expires at the next loop tick
        # and cancels this await, which asyncio converts into TimeoutError.
        self.timeouts[0].reschedule(asyncio.get_running_loop().time())
        await asyncio.Event().wait()  # hang forever; only the timeout ends this
        raise AssertionError("unreachable: the agent run timeout must fire")


# A budget value no real config produces, so the wrapper below captures only
# the agent-run ``asyncio.timeout`` and passes every other caller through.
_ARMED_ON_RECEIPT_BUDGET = 987_654.321


def _armed_on_receipt(ir: _IRHang):
    """Wrap ``asyncio.timeout`` so the agent's budget is created unarmed.

    It is still the REAL ``asyncio.Timeout`` (so the path is ``asyncio.timeout``
    expiry → ``TimeoutError`` → the agent's ``except TimeoutError`` branch);
    only its deadline is deferred until ``_IRHang`` receives the request.
    """
    real_timeout = asyncio.timeout

    def _timeout(delay: float | None) -> asyncio.Timeout:
        if delay == _ARMED_ON_RECEIPT_BUDGET:
            tm = real_timeout(None)
            ir.timeouts.append(tm)
            return tm
        return real_timeout(delay)

    return _timeout


_SAMPLING_KEYS = ("temperature", "seed", "repetition_penalty", "frequency_penalty")


class TestSentSamplingEcho:
    """usage_metrics (and the chat JSON / SSE meta) echo what was SENT to IR.

    Read-back provenance (judge-oracle v2): temperature / seed / max_tokens /
    penalties come from the request body WM built, never from worker config;
    absent from the body ⇒ null; 0 is a value, not an absence.

    MUTATIONS (each measured red, see the PR body):
      * ``token_metrics.seed = self.config.seed`` in
        ``_attach_sampling_provenance`` (echo from config) ⇒
        ``test_worker_echo_is_the_per_message_value_not_the_config`` red;
      * ``effective.get("temperature") or None`` / ``... seed ... or None`` ⇒
        ``test_worker_t0_seed0_echo_as_zero`` red;
      * ``settings.get("temperature") or None`` in ``_sent_sampling`` ⇒
        the chat zero tests red; ``req.repetition_penalty`` instead of the
        settings ⇒ ``test_chat_penalty_not_sent_to_external_provider_echoes_null`` red;
      * drop ``self._stamp_sent_usage_metrics(e, run_sampling)`` ⇒
        ``test_worker_503_still_echoes_what_was_sent`` red;
      * drop ``self._stamp_sent_usage_metrics(timeout_error, run_sampling)`` in
        the ``except TimeoutError`` branch ⇒
        ``test_worker_timeout_still_echoes_what_was_sent`` red.
    """

    def test_worker_t0_seed0_echo_as_zero(self, message_client):
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, temperature=0.0)
        resp = _post(client, seed=0, repetition_penalty=1.05, correlation_id=CORRELATION)
        assert resp.status_code == 200, resp.text
        body = ir.body
        # Control: the zeros really were on the wire.
        assert body["temperature"] == 0 and body["seed"] == 0
        um = resp.json()["usage_metrics"]
        assert um["temperature"] is not None and um["temperature"] == 0
        assert um["seed"] is not None and um["seed"] == 0
        assert um["temperature"] == body["temperature"]
        assert um["seed"] == body["seed"]
        assert um["max_tokens"] == _wire_max_tokens(body)
        assert um["repetition_penalty"] == body["repetition_penalty"] == 1.05
        assert um["frequency_penalty"] is None and "frequency_penalty" not in body

    def test_worker_echo_is_the_per_message_value_not_the_config(self, message_client):
        """Config seed 5, per-message seed 0: 0 went out, so 0 is echoed."""
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, temperature=0.0, seed=5, frequency_penalty=0.5)
        resp = _post(client, seed=0, frequency_penalty=0.0)
        assert resp.status_code == 200, resp.text
        body = ir.body
        assert body["seed"] == 0 and body["frequency_penalty"] == 0
        um = resp.json()["usage_metrics"]
        assert (um["seed"], um["frequency_penalty"]) == (body["seed"], body["frequency_penalty"])
        assert um["seed"] == 0

    def test_worker_503_still_echoes_what_was_sent(self, message_client):
        """The j05 shape: IR 503 ⇒ no result, but the call WAS made with T=0 /
        seed=0, so the error body carries that read-back instead of nothing."""
        client, holder, _ = message_client
        ir = _IR503()
        holder["agent"] = _real_ir_agent(ir, temperature=0.0, max_tokens=512)
        resp = _post(client, seed=0, repetition_penalty=1.05, correlation_id=CORRELATION)
        assert resp.status_code == 503, resp.text
        body = ir.body  # exactly one IR call: the failure is not retried
        err = resp.json()["error"]
        # Additive: the pre-existing envelope keys are unchanged.
        assert err["status"] == 503 and err["code"] == "SERVICE_UNAVAILABLE"
        assert "503" in err["message"]
        um = err["details"]["usage_metrics"]
        assert um["temperature"] == body["temperature"] == 0
        assert um["seed"] == body["seed"] == 0
        assert um["temperature"] is not None and um["seed"] is not None
        assert um["max_tokens"] == _wire_max_tokens(body) == 512
        assert um["repetition_penalty"] == body["repetition_penalty"] == 1.05
        assert um["frequency_penalty"] is None
        assert um["requested_model"] == body["model"] == PINNED_MODEL
        assert um["correlation_id"] == CORRELATION
        # Nothing was measured: no tokens, no finish reason, no served_*.
        assert um["prompt_tokens"] is None and um["completion_tokens"] is None
        assert um["finish_reason"] is None and um["served_provider"] is None

    def test_worker_timeout_still_echoes_what_was_sent(self, message_client):
        """The ``except TimeoutError`` branch: the agent's own run budget fires
        (a real ``asyncio.timeout``, not a synthetic raise) after the request
        reached IR, so the error body still carries the T=0 / seed=0 read-back.

        The budget is armed by the stub on receipt (``_armed_on_receipt``), so
        "timed out before the request landed" is impossible by construction."""
        client, holder, _ = message_client
        ir = _IRHang()
        agent = _real_ir_agent(ir, temperature=0.0, max_tokens=512)
        holder["agent"] = agent
        with (
            patch.object(type(agent), "_agent_run_timeout", lambda _self: _ARMED_ON_RECEIPT_BUDGET),
            patch.object(asyncio, "timeout", _armed_on_receipt(ir)),
        ):
            resp = _post(client, seed=0, repetition_penalty=1.05, correlation_id=CORRELATION)
        assert resp.status_code >= 500, resp.text
        assert len(ir.timeouts) == 1, "control: the agent's own asyncio.timeout was used"
        assert ir.timeouts[0].expired(), "control: the real asyncio.timeout fired"
        assert len(ir.calls) == 1, "control: the request reached IR before the timeout"
        body = ir.calls[0][0]
        err = resp.json()["error"]
        assert "timed out" in err["message"]
        um = err["details"]["usage_metrics"]
        assert um["temperature"] is not None and um["seed"] is not None
        assert um["temperature"] == body["temperature"] == 0
        assert um["seed"] == body["seed"] == 0
        assert um["max_tokens"] == _wire_max_tokens(body) == 512
        assert um["repetition_penalty"] == body["repetition_penalty"] == 1.05
        assert um["correlation_id"] == CORRELATION
        assert um["prompt_tokens"] is None and um["completion_tokens"] is None

    def test_an_error_without_a_run_carries_no_details(self, message_client):
        """Control: a failure that is not a model call (the registry lookup
        raising) renders the envelope exactly as before — no ``details``."""
        client, _holder, _ = message_client
        from src.api.workers import message_endpoints

        with patch.object(
            message_endpoints.worker_registry,
            "get_worker",
            AsyncMock(side_effect=RuntimeError("registry down")),
        ):
            resp = _post(client, seed=0)
        assert resp.status_code >= 500, resp.text
        assert "details" not in resp.json()["error"]

    def test_chat_t0_seed0_echo_as_zero(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(
            chat_client, ir, temperature=0, seed=0, repetition_penalty=1.05, max_tokens=256
        )
        assert resp.status_code == 200, resp.text
        body = ir.body
        assert body["temperature"] == 0 and body["seed"] == 0
        usage = resp.json()["usage"]
        assert usage["temperature"] is not None and usage["temperature"] == 0
        assert usage["seed"] is not None and usage["seed"] == 0
        assert usage["repetition_penalty"] == body["repetition_penalty"] == 1.05
        assert usage["max_tokens"] == _wire_max_tokens(body) == 256
        assert usage["frequency_penalty"] is None

    def test_chat_stream_meta_echoes_t0_seed0(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir, temperature=0, seed=0, stream=True)
        assert resp.status_code == 200, resp.text
        frames = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: {")
        ]
        meta = next(f["meta"] for f in frames if "meta" in f)
        assert meta.get("temperature") == 0 and "temperature" in meta
        assert meta.get("seed") == 0 and "seed" in meta
        assert (meta["temperature"], meta["seed"]) == (ir.body["temperature"], ir.body["seed"])
        # Absent on the wire ⇒ absent from meta.
        assert "repetition_penalty" not in meta and "repetition_penalty" not in ir.body

    def test_chat_nothing_set_echoes_only_the_sent_temperature(self, chat_client):
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir)
        assert resp.status_code == 200, resp.text
        usage = resp.json()["usage"]
        assert usage["temperature"] == ir.body["temperature"] == 0.1
        for key in ("seed", "repetition_penalty", "frequency_penalty"):
            assert usage[key] is None, key
            assert key not in ir.body, key

    def test_chat_penalty_not_sent_to_external_provider_echoes_null(self, chat_client, monkeypatch):
        """repetition_penalty is dropped for a non-self-hosted provider. The
        request asked for 1.05; the wire carried nothing ⇒ null, not 1.05."""

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
        # Every LOCAL provider is self-hosted (it would send the key), so the
        # only way to reach the drop branch is an external provider, admitted
        # here because the chat client runs as a PROVEN platform tenant (owner
        # ruling 2026-10-03). The socket is still the in-memory stub.
        ir = _IRStub()
        resp = _chat_with_stub(chat_client, ir, provider="openai", repetition_penalty=1.05)
        assert resp.status_code == 200, resp.text
        assert "repetition_penalty" not in ir.body
        assert "repetition_penalty" not in (ir.body.get("extra_body") or {})
        assert resp.json()["usage"]["repetition_penalty"] is None


# ======================================================== per-message temperature
#
# merqi's claim judge needs the SAME verdict for the same prompt, whatever
# temperature the worker row was provisioned with. ``temperature`` is a per-RUN
# override exactly like seed: absent ⇒ the call is byte-identical to before.


class TestPerMessageTemperature:
    def test_absent_sends_the_worker_temperature_and_the_pre_pr_body(self, message_client):
        """No temperature in the request ⇒ the wire body is EXACTLY what a plain
        ``agent.run`` sends: the worker's own 0.1, nothing added.

        MUTATION: make ``_run_sampling_kwargs`` always include
        ``temperature`` (e.g. ``request.temperature or 0.0``) ⇒ body carries 0.0
        ⇒ red.
        """
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, temperature=0.1)
        resp = _post(client)
        assert resp.status_code == 200, resp.text
        assert ir.body["temperature"] == 0.1
        assert resp.json()["usage_metrics"]["temperature"] == 0.1

        # Control: the same agent run directly (the pre-PR call) sends the same body.
        direct_ir = _IRStub()
        direct = _real_ir_agent(direct_ir, temperature=0.1)
        with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
            asyncio.run(direct.run("hi"))
        assert ir.body == direct_ir.body

    def test_present_reaches_the_wire_and_usage_metrics(self, message_client):
        """MUTATION: drop ``temperature`` from ``_run_sampling_kwargs`` (or from
        ``_run_model_settings``) ⇒ the body carries the worker's 0.7 ⇒ red."""
        client, holder, _ = message_client
        ir = _IRStub()
        holder["agent"] = _real_ir_agent(ir, temperature=0.7)
        resp = _post(client, temperature=0.0)
        assert resp.status_code == 200, resp.text
        assert ir.body["temperature"] == 0.0
        um = resp.json()["usage_metrics"]
        assert um["temperature"] == 0.0
        assert um["temperature_not_sent"] is None

    def test_on_a_non_pydantic_worker_is_a_typed_422(self, message_client):
        client, _, state_db = message_client
        from src.api.workers import message_endpoints

        with patch.object(message_endpoints, "_seed_supported", return_value=False):
            resp = _post(client, temperature=0.0)
        assert resp.status_code == 422
        assert "SAMPLING_OVERRIDE_NOT_SUPPORTED" in resp.text
        state_db.add_worker_state.assert_not_awaited()

    @pytest.mark.parametrize("bad", [-0.1, 1.01, 2.0, "hot"])
    def test_is_bounded_to_zero_one(self, message_client, bad):
        client, _, _ = message_client
        assert _post(client, temperature=bad).status_code == 422


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_message_request_accepts_temperature_in_bounds(value):
    from src.api.workers.models import MessageRequest

    assert MessageRequest(message="hi", temperature=value).temperature == value


def test_message_request_temperature_defaults_to_none():
    from src.api.workers.models import MessageRequest

    assert MessageRequest(message="hi").temperature is None


@pytest.mark.asyncio
async def test_run_worker_kwargs_absent_vs_present():
    """The exact ``worker.run`` call: absent ⇒ ``run(message)`` with NO kwargs."""
    from src.api.workers.message_endpoints import _run_worker
    from src.api.workers.models import MessageRequest

    worker = MagicMock()
    worker.accepts_run_seed = True
    worker.run = AsyncMock(return_value="r")

    await _run_worker(worker, MessageRequest(message="hi"), "w")
    worker.run.assert_awaited_once_with("hi")

    worker.run.reset_mock()
    await _run_worker(worker, MessageRequest(message="hi", temperature=0.0), "w")
    worker.run.assert_awaited_once_with("hi", temperature=0.0)


@pytest.mark.asyncio
async def test_run_temperature_wins_for_that_run_only():
    """Per run, never baked into the cached agent: the next plain run is back
    to the worker's own temperature."""
    ir = _IRStub()
    agent = _real_ir_agent(ir, temperature=0.4)
    with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
        overridden = await agent.run("hi", temperature=0.0)
        plain = await agent.run("hi")
    (run_body, _), (plain_body, _) = ir.calls
    assert run_body["temperature"] == 0.0
    assert plain_body["temperature"] == 0.4
    assert overridden.usage_metrics.temperature == 0.0
    assert plain.usage_metrics.temperature == 0.4


@pytest.mark.asyncio
async def test_a_model_that_strips_temperature_reports_it_instead_of_echoing_it():
    """When pydantic-ai will strip the key, usage_metrics must not claim it was sent."""
    ir = _IRStub()
    agent = _real_ir_agent(ir, temperature=0.1)
    agent._temperature_unsent_reason = "reasoning model 'o3' rejects sampling parameters"
    with patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()):
        result = await agent.run("hi", temperature=0.0)
    assert result.usage_metrics.temperature is None
    assert result.usage_metrics.temperature_not_sent == (
        "reasoning model 'o3' rejects sampling parameters"
    )


def test_temperature_unsent_reason_per_provider_model():
    """Mirrors pydantic-ai's drop rules: local/OpenAI-compatible chat models and
    Groq send it; OpenAI reasoning models and sampling-less Claude models do not."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    from src.worker.agents.pydantic_ai_agent import _temperature_unsent_reason

    oai = OpenAIProvider(api_key="k")
    assert _temperature_unsent_reason(OpenAIChatModel("local-llm", provider=oai), None) is None
    assert _temperature_unsent_reason(OpenAIChatModel("gpt-4o", provider=oai), None) is None
    assert "reasoning" in (
        _temperature_unsent_reason(OpenAIChatModel("o3", provider=oai), None) or ""
    )
    assert (
        _temperature_unsent_reason(
            GroqModel("llama-3.3-70b", provider=GroqProvider(api_key="k")), None
        )
        is None
    )
    anth = AnthropicProvider(api_key="k")
    assert (
        _temperature_unsent_reason(AnthropicModel("claude-sonnet-4-5", provider=anth), None) is None
    )
    assert (
        _temperature_unsent_reason(AnthropicModel("claude-opus-4-7", provider=anth), None)
        is not None
    )
