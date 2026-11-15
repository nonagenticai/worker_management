"""Tests for POST /chat/completions — stateless streaming + non-streaming."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Environment must be set before importing app
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pytest
from fastapi.testclient import TestClient

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from tests._platform_tenant import PLATFORM_TENANT, caller_tenant


async def _fake_tenant_id(request=None):
    return "test-tenant"


@pytest.fixture
def client():
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    # These tests pin the PLATFORM-tenant behaviour (unchanged by the 2026-10-01
    # ruling); the customer rules live in tests/test_tenant_scoped_providers.py.
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        caller_tenant(PLATFORM_TENANT),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


def _make_run_result(output: str):
    """Build a fake pydantic-ai AgentRunResult with str output + usage stub."""
    result = MagicMock()
    result.output = output
    result.usage = MagicMock(
        return_value=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
    )
    return result


def _make_stream_agent(
    deltas, *, input_tokens=10, output_tokens=20, total_tokens=30, enter_exc=None
):
    """Fake pydantic-ai Agent whose run_stream yields `deltas` (plain-text path).

    The non-streaming handler now internally streams (to observe TTFT + decode
    window for pp_tps/tg_tps), so non-streaming tests mock run_stream like the
    streaming tests. `enter_exc` makes __aenter__ raise (provider-failure path).
    """

    async def fake_stream_text(delta=True):
        for d in deltas:
            yield d

    fake_stream = MagicMock()
    fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
    fake_stream.usage = MagicMock(
        return_value=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens
        )
    )
    fake_stream.all_messages = MagicMock(return_value=[])
    fake_stream_ctx = MagicMock()
    if enter_exc is not None:
        fake_stream_ctx.__aenter__ = AsyncMock(side_effect=enter_exc)
    else:
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_agent = MagicMock()
    fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)
    return fake_agent


def _queue_rejected_exc(retry_after: str | None = "7") -> Exception:
    """An IR queue_rejected failure shaped like pydantic-ai's: the raised error has
    no ``.response``; IR's headers sit one link down the ``__cause__`` chain."""
    headers = {"X-IR-Error-Classification": "queue_rejected"}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    upstream = RuntimeError("Error code: 503 - queue full")
    upstream.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
    outer = RuntimeError("status_code: 503, model_name: m, body: queue full")
    outer.__cause__ = upstream
    return outer


class TestChatCompletionsValidation:
    def test_unknown_provider_returns_404(self, client):
        resp = client.post(
            "/chat/completions",
            json={
                "provider": "not-real",
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404

    def test_empty_messages_rejected(self, client):
        resp = client.post(
            "/chat/completions",
            json={"provider": "vllm", "model": "m", "messages": []},
        )
        assert resp.status_code == 422

    def test_no_user_message_rejected(self, client):
        resp = client.post(
            "/chat/completions",
            json={
                "provider": "vllm",
                "model": "m",
                "messages": [{"role": "assistant", "content": "hi"}],
            },
        )
        assert resp.status_code == 400

    def test_invalid_temperature_rejected(self, client):
        resp = client.post(
            "/chat/completions",
            json={
                "provider": "vllm",
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 99.0,
            },
        )
        assert resp.status_code == 422


class TestChatCompletionsNonStreaming:
    def test_vllm_non_streaming_returns_output(self, client, monkeypatch):
        """Happy path: non-streaming handler internally streams via run_stream."""
        monkeypatch.delenv("VLLM_URL", raising=False)

        fake_agent = _make_stream_agent(["hello ", "there"])

        with (
            patch("src.api.providers._chat_router._build_agent", return_value=fake_agent),
        ):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "local-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "system_prompt": "be terse",
                    "temperature": 0.3,
                    "stream": False,
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["output"] == "hello there"
        assert body["provider"] == "vllm"
        assert body["model"] == "local-llm"
        assert body["usage"]["input_tokens"] == 10
        assert body["usage"]["output_tokens"] == 20
        assert body["usage"]["total_tokens"] == 30
        fake_agent.run_stream.assert_called_once_with("hi")

    def test_last_user_message_used_as_prompt(self, client):
        fake_agent = _make_stream_agent(["ok"])

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "first-reply"},
                        {"role": "user", "content": "second"},
                    ],
                },
            )
        assert resp.status_code == 200
        fake_agent.run_stream.assert_called_once_with("second")

    def test_provider_failure_returns_502(self, client):
        fake_agent = _make_stream_agent([], enter_exc=RuntimeError("upstream died"))

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 502
        assert "upstream died" in resp.json()["error"]["message"]

    def test_queue_rejected_returns_503_with_retry_after(self, client):
        """IR queue_rejected is BUSY: 503 + IR's Retry-After + classification."""
        fake_agent = _make_stream_agent([], enter_exc=_queue_rejected_exc("7"))

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "inference_router",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "7"
        assert resp.headers["X-IR-Error-Classification"] == "queue_rejected"
        assert "Provider busy" in resp.json()["error"]["message"]

    def test_queue_rejected_without_retry_after_uses_setting(self, client):
        """No Retry-After from IR ⇒ the configured fallback, like the worker path."""
        from src.settings import get_settings

        fake_agent = _make_stream_agent([], enter_exc=_queue_rejected_exc(None))

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "inference_router",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == str(get_settings().queue_rejected_retry_after_s)
        assert resp.headers["X-IR-Error-Classification"] == "queue_rejected"

    def test_build_agent_failure_returns_502(self, client):
        """If model factory raises (e.g., missing API key) it becomes 502."""
        with patch(
            "src.api.providers._chat_router._build_agent",
            side_effect=RuntimeError("ANTHROPIC_API_KEY not configured"),
        ):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 502
        assert "ANTHROPIC_API_KEY" in resp.json()["error"]["message"]


class TestChatCompletionsStreaming:
    def test_stream_returns_sse_chunks(self, client):
        """stream=true yields SSE chunks ending with [DONE]."""

        async def fake_stream_text(delta=True):
            yield "hello "
            yield "world"

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        # Chunks are newline-separated "data: {...}" blobs
        lines = [line for line in body.splitlines() if line.startswith("data: ")]
        payloads = [line.removeprefix("data: ") for line in lines]
        assert "[DONE]" in payloads
        # Two delta payloads before [DONE]
        deltas = [json.loads(p)["delta"] for p in payloads if p != "[DONE]" and "delta" in p]
        assert deltas == ["hello ", "world"]

    def test_stream_emits_pp_tg_in_meta_frame(self, client):
        """Final meta frame carries pp_tps + tg_tps floats derived from timings.

        Inserts a sleep between deltas so the inter-delta window exceeds the
        ``MIN_TG_WINDOW_S`` guard (50 ms by default) — without the sleep the
        two yields land in adjacent event-loop ticks and tg is honestly
        dropped (covered by ``test_stream_burst_flush_drops_tg`` below).
        """
        import asyncio

        async def fake_stream_text(delta=True):
            yield "hello "
            await asyncio.sleep(0.08)
            yield "world"

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
        fake_stream.usage = MagicMock(
            return_value=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
        )
        # No ModelResponse available on the fake — keeps resolved_* out of meta
        # so we can isolate the pp/tg assertions.
        fake_stream.all_messages = MagicMock(return_value=[])
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        payloads = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        meta_frames = [json.loads(p) for p in payloads if p != "[DONE]" and '"meta"' in p]
        assert len(meta_frames) == 1, f"expected exactly one meta frame, got {meta_frames}"
        meta = meta_frames[0]["meta"]
        assert meta["input_tokens"] == 10
        assert meta["output_tokens"] == 20
        assert isinstance(meta["pp_tps"], float) and meta["pp_tps"] > 0
        assert isinstance(meta["tg_tps"], float) and meta["tg_tps"] > 0
        # Sanity-bound: 20 tokens over an >=80 ms window → < 250 t/s.
        assert meta["tg_tps"] < 1000, (
            f"tg_tps={meta['tg_tps']} implausibly high — burst-flush guard regressed"
        )

    def test_stream_burst_flush_drops_tg(self, client):
        """When deltas arrive in a sub-50ms burst, tg_tps must be dropped.

        Reproduces the production failure mode where the upstream chain
        (mlx_inference / vLLM → pydantic-ai → asyncio) coalesces flushes:
        two synchronous yields with no `await` in between land in the same
        event-loop tick, giving a (last_delta - first_delta) window of
        microseconds. Pre-fix the meta frame would report
        ``tg_tps`` ≈ 2_421_346.1 for 162 tokens; post-fix the field is
        absent. ``pp_tps`` is unaffected and still emitted.
        """

        async def fake_stream_text(delta=True):
            yield "hello "
            yield "world"

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
        fake_stream.usage = MagicMock(
            return_value=SimpleNamespace(input_tokens=10, output_tokens=162, total_tokens=172)
        )
        fake_stream.all_messages = MagicMock(return_value=[])
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        payloads = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        meta_frames = [json.loads(p) for p in payloads if p != "[DONE]" and '"meta"' in p]
        assert len(meta_frames) == 1
        meta = meta_frames[0]["meta"]
        assert meta["output_tokens"] == 162
        assert "tg_tps" not in meta, (
            f"burst-flush stream should drop tg_tps; got {meta.get('tg_tps')}"
        )
        # pp is computed from (first_delta_t - t0) which is not collapsed by
        # the burst, so it remains present and finite.
        assert "pp_tps" in meta and meta["pp_tps"] > 0

    def test_stream_single_delta_drops_tg(self, client):
        """A single-delta stream has no window to compute tg over — drop it."""

        async def fake_stream_text(delta=True):
            yield "the whole thing in one chunk"

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
        fake_stream.usage = MagicMock(
            return_value=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
        )
        fake_stream.all_messages = MagicMock(return_value=[])
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        payloads = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        meta_frames = [json.loads(p) for p in payloads if p != "[DONE]" and '"meta"' in p]
        assert len(meta_frames) == 1
        meta = meta_frames[0]["meta"]
        assert "tg_tps" not in meta

    def test_stream_failure_emits_error_then_done(self, client):
        """If the upstream stream raises mid-stream, we still terminate cleanly."""

        async def fake_stream_text(delta=True):
            yield "partial"
            raise RuntimeError("stream died")

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        body = resp.text
        assert "stream died" in body
        assert "[DONE]" in body

    def test_stream_queue_rejected_emits_busy_frame_then_done(self, client):
        """queue_rejected mid-stream (200 already sent): the error frame keeps its
        string ``error`` and adds type/classification/retry_after; [DONE] follows."""
        fake_agent = _make_stream_agent([], enter_exc=_queue_rejected_exc("12"))

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "inference_router",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        errors = [f for f in frames if isinstance(f, dict) and "error" in f]
        assert len(errors) == 1
        err = errors[0]
        assert isinstance(err["error"], str)
        assert err["type"] == "provider_busy"
        assert err["classification"] == "queue_rejected"
        assert err["error_code"] == "PROVIDER_BUSY"
        assert err["status"] == 503
        assert err["details"] == {"retry_after": 12}
        assert frames[-1] == "[DONE]"
        assert not any(isinstance(f, dict) and "meta" in f for f in frames)

    def test_stream_non_busy_failure_frame_has_no_busy_keys(self, client):
        """Any other stream error keeps the old frame shape exactly."""
        fake_agent = _make_stream_agent([], enter_exc=RuntimeError("upstream died"))

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        frames = _parse_sse(resp.text)
        errors = [f for f in frames if isinstance(f, dict) and "error" in f]
        assert errors == [{"error": "upstream died", "provider": "vllm"}]
        assert frames[-1] == "[DONE]"

    def test_stream_emits_keepalive_during_silent_gap(self, client, monkeypatch):
        """Silent gap before first delta yields ``: keepalive`` SSE comments.

        Defends against Cloudflare/Envoy/cloudflared idle-stream resets that
        previously surfaced as a ~110s "hang" in fr's chat UI when the upstream
        model was slow to first token.
        """
        import asyncio as _asyncio

        import src.api.providers._chat_router as mod

        # Shrink the heartbeat window so the test runs in <1s.
        monkeypatch.setattr(mod, "SSE_HEARTBEAT_INTERVAL_S", 0.05)

        async def slow_stream_text(delta=True):
            # 0.16s of silence ≥ 3 heartbeat intervals → at least 2 keepalives
            # before the first real delta lands.
            await _asyncio.sleep(0.16)
            yield "hi"

        fake_stream_ctx = MagicMock()
        fake_stream = MagicMock()
        fake_stream.stream_text = lambda delta=True: slow_stream_text(delta=delta)
        fake_stream.usage = MagicMock(
            return_value=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
        )
        fake_stream.all_messages = MagicMock(return_value=[])
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        body = resp.text
        # SSE comment lines start with `:` and are spec-ignored by clients
        # but are real bytes on the wire that defeat intermediary idle resets.
        assert ": keepalive" in body, "expected keepalive SSE comment during silent gap"
        assert body.count(": keepalive") >= 2, (
            f"expected ≥2 keepalive frames in 0.16s gap with 0.05s interval, "
            f"got {body.count(': keepalive')}: {body!r}"
        )
        # Real delta still flows once the upstream produces.
        assert any(
            line.startswith("data: ") and '"delta"' in line and '"hi"' in line
            for line in body.splitlines()
        )
        assert "[DONE]" in body


class TestChatCompletionsIdentityForwarding:
    """`_build_agent` must pull tenant/license/subject from request.state and
    thread them into `PydanticAIAgentConfig` so the inference-router provider
    can attribute LLM cost back to the originating user.
    """

    @pytest.fixture(autouse=True)
    def _openai_api_key(self, monkeypatch):
        # pydantic-ai's Agent(model=...) path may try to inspect a model and
        # construct an OpenAI client when given a non-string mock; ensure a
        # placeholder API key exists so the construction doesn't 500 the test.
        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    @pytest.mark.asyncio
    async def test_build_agent_forwards_request_state_identity(self):
        from fastapi import Request

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)
        request.state.tenant_id = "tenant-zeta"
        request.state.license_id = "lic-9"
        request.state.subject_id = "subj-7"

        req = ChatCompletionsRequest(
            provider="vllm",
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig") as mock_config,
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
        ):
            await _build_agent(req, request)

        kwargs = mock_config.call_args.kwargs
        assert kwargs["tenant_id"] == "tenant-zeta"
        assert kwargs["license_id"] == "lic-9"
        assert kwargs["subject_id"] == "subj-7"

    @pytest.mark.asyncio
    async def test_build_agent_handles_missing_identity_state(self):
        """When middleware hasn't run (tests, public endpoints): identity is None."""
        from fastapi import Request

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)  # state is empty — no middleware ran

        req = ChatCompletionsRequest(
            provider="vllm",
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig") as mock_config,
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
        ):
            await _build_agent(req, request)

        kwargs = mock_config.call_args.kwargs
        assert kwargs["tenant_id"] is None
        assert kwargs["license_id"] is None
        assert kwargs["subject_id"] is None


class TestChatCompletionsOutputSchema:
    """`_build_agent` must thread the request's `output_schema` into the
    pydantic-ai Agent so vLLM (or whatever the underlying provider is)
    receives a structured-output constraint. Regression: previously the field
    was declared on the request model and then silently dropped, which is why
    `output_schema={"type":"boolean"}` returned a 353-token free-form essay
    instead of `true`/`false`.
    """

    @pytest.fixture(autouse=True)
    def _openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    @pytest.mark.asyncio
    async def test_boolean_schema_sets_bool_output_type(self):
        from fastapi import Request

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)

        req = ChatCompletionsRequest(
            provider="vllm",
            model="m",
            messages=[ChatMessage(role="user", content="is the sky blue?")],
            output_schema={"type": "boolean"},
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig"),
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
            patch("src.api.providers._chat_router.Agent") as mock_agent_cls,
        ):
            await _build_agent(req, request)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["output_type"] is bool, (
            f"expected output_type=bool, got {kwargs.get('output_type')!r}"
        )

    @pytest.mark.asyncio
    async def test_no_schema_keeps_default_str_output_type(self):
        from fastapi import Request

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)

        req = ChatCompletionsRequest(
            provider="vllm",
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig"),
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
            patch("src.api.providers._chat_router.Agent") as mock_agent_cls,
        ):
            await _build_agent(req, request)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["output_type"] is str

    @pytest.mark.asyncio
    async def test_array_schema_sets_list_output_type(self):
        from fastapi import Request

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)

        req = ChatCompletionsRequest(
            provider="vllm",
            model="m",
            messages=[ChatMessage(role="user", content="name three primary colours")],
            output_schema={"type": "array", "items": {"type": "string"}},
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig"),
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
            patch("src.api.providers._chat_router.Agent") as mock_agent_cls,
        ):
            await _build_agent(req, request)

        kwargs = mock_agent_cls.call_args.kwargs
        # For non-inference_router providers we hand pydantic-ai the bare
        # generic. `list[str]` compares equal to itself across the helper
        # boundary (typing aliases hash by parameters).
        assert kwargs["output_type"] == list[str]

    @pytest.mark.asyncio
    async def test_inference_router_boolean_uses_native_output(self):
        """For inference_router we wrap in NativeOutput so vLLM receives
        response_format=json_schema. The vLLM xgrammar backend then constrains
        generation server-side to a literal `true`/`false`.
        """
        from fastapi import Request
        from pydantic_ai import NativeOutput

        from src.api.providers._chat_router import (
            ChatCompletionsRequest,
            ChatMessage,
            _build_agent,
        )

        scope = {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/chat/completions",
            "query_string": b"",
        }
        request = Request(scope)

        req = ChatCompletionsRequest(
            provider="inference_router",
            model="local-llm",
            messages=[ChatMessage(role="user", content="is the sky blue?")],
            output_schema={"type": "boolean"},
        )

        with (
            patch("src.api.providers._chat_router.PydanticAIAgentConfig"),
            patch(
                "src.api.providers._chat_router.PydanticAIAgent._initialize_model",
                return_value=MagicMock(),
            ),
            patch("src.api.providers._chat_router.Agent") as mock_agent_cls,
        ):
            await _build_agent(req, request)

        kwargs = mock_agent_cls.call_args.kwargs
        assert isinstance(kwargs["output_type"], NativeOutput)
        # Retries for inference_router are reduced to 2 to avoid retry-storm
        # ×60s past karma's 300s ingestion ceiling — same budget the worker
        # agent uses (pydantic_ai_agent.py:505–509).
        assert kwargs["retries"] == 2


def _build_stream_ctx(deltas: list[str], validated_output: Any) -> MagicMock:
    """Build a fake pydantic-ai stream context for the result-frame tests.

    Mirrors real ``StreamedRunResult`` behaviour:
    - When ``output_type`` is ``str`` (validated_output is a ``str``),
      ``stream_text(delta=True)`` yields ``deltas`` one by one.
    - When ``output_type`` is non-str (validated_output is bool/list/dict),
      ``stream_text(delta=True)`` raises
      ``UserError("stream_text() can only be used with text responses")``
      — matching pydantic-ai 1.x. The producer branches on
      ``output_schema`` to avoid calling stream_text in that case.
    - ``get_output()`` returns ``validated_output`` in both paths.

    ``usage`` and ``all_messages`` are stubbed minimally so the meta path
    doesn't blow up after the stream completes.
    """
    is_text = isinstance(validated_output, str)

    async def fake_stream_text(delta=True):
        if not is_text:
            raise RuntimeError("stream_text() can only be used with text responses")
        for d in deltas:
            yield d

    stream = MagicMock()
    stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
    stream.get_output = AsyncMock(return_value=validated_output)
    stream.usage = MagicMock(
        return_value=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8)
    )
    stream.all_messages = MagicMock(return_value=[])

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _parse_sse(body: str) -> list[Any]:
    """Return the parsed payloads (str ``[DONE]`` or parsed-JSON dict) in order."""
    out: list[Any] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            out.append("[DONE]")
        else:
            out.append(json.loads(payload))
    return out


class TestChatCompletionsResultFrame:
    """Streaming with ``output_schema`` set emits exactly one ``result`` frame
    carrying the validated typed value and the echoed schema, placed between
    the last ``delta`` and the ``meta`` frame. fr's renderer waits on this
    frame to dispatch to the schema-shaped UI component (Pill for boolean,
    bulleted list for array, key/value table for object).

    Without a schema (plain-text default), no ``result`` frame is emitted —
    the deltas already carry the bubble's content unchanged.
    """

    @pytest.mark.parametrize(
        ("schema", "validated"),
        [
            ({"type": "boolean"}, True),
            ({"type": "array", "items": {"type": "string"}}, ["alpha", "beta", "gamma"]),
            (
                {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer"],
                },
                {"answer": "yes", "confidence": 0.9},
            ),
        ],
        ids=["boolean", "array", "object"],
    )
    def test_result_frame_emitted_between_deltas_and_meta(self, client, schema, validated):
        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(
            return_value=_build_stream_ctx(deltas=["raw1", "raw2"], validated_output=validated)
        )

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "go"}],
                    "stream": True,
                    "output_schema": schema,
                },
            )
        assert resp.status_code == 200
        payloads = _parse_sse(resp.text)

        # Frame ordering for typed-output: one ``progress`` heartbeat (the
        # delta path is skipped because pydantic-ai's ``stream_text(delta=True)``
        # only works with text responses), then exactly one ``result``, then
        # exactly one ``meta``, then ``[DONE]``.
        kinds: list[str] = []
        for p in payloads:
            if p == "[DONE]":
                kinds.append("done")
            elif "delta" in p:
                kinds.append("delta")
            elif "progress" in p:
                kinds.append("progress")
            elif "result" in p:
                kinds.append("result")
            elif "meta" in p:
                kinds.append("meta")
            elif "error" in p:
                kinds.append("error")
        compressed = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
        assert compressed == ["progress", "result", "meta", "done"], (
            f"unexpected frame order {kinds!r} for schema={schema!r}"
        )
        # No ``delta`` frames in the typed path — those would be raw JSON
        # characters which the renderer can't usefully display.
        assert "delta" not in kinds, f"typed-output path must not emit delta frames, got {kinds!r}"

        # Exactly one result frame; carries the validated value + echoed schema.
        result_frames = [p for p in payloads if isinstance(p, dict) and "result" in p]
        assert len(result_frames) == 1
        rf = result_frames[0]
        assert rf["result"] == validated
        assert rf["schema"] == schema

    def test_no_result_frame_when_schema_absent(self, client):
        """Plain-text path (output_schema=None) keeps the old frame sequence."""
        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(
            return_value=_build_stream_ctx(deltas=["hello "], validated_output="hello ")
        )

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        payloads = _parse_sse(resp.text)
        assert not any(isinstance(p, dict) and "result" in p for p in payloads), (
            "result frame must not be emitted when output_schema is None"
        )

    def test_result_frame_skipped_when_get_output_raises(self, client):
        """Validation failure on `get_output()` is non-fatal: the stream still
        terminates cleanly without a `result` frame, so the renderer falls
        through to TextResult (which already has the accumulated deltas).
        """
        ctx = _build_stream_ctx(deltas=["raw"], validated_output=None)
        # Override get_output to simulate a validator failure.
        ctx.__aenter__.return_value.get_output = AsyncMock(
            side_effect=ValueError("output failed schema validation")
        )

        fake_agent = MagicMock()
        fake_agent.run_stream = MagicMock(return_value=ctx)

        with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "vllm",
                    "model": "m",
                    "messages": [{"role": "user", "content": "go"}],
                    "stream": True,
                    "output_schema": {"type": "boolean"},
                },
            )
        assert resp.status_code == 200
        payloads = _parse_sse(resp.text)
        assert not any(isinstance(p, dict) and "result" in p for p in payloads)
        # Stream still ends cleanly with [DONE].
        assert payloads[-1] == "[DONE]"
