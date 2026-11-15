"""WS-1 — prompt-replay identity headers on the inference-router outbound hook.

When asset-evaluator drives a prompt replay through WM, send_message reads the
inbound `X-Replay-Mode: true` header and stashes it in a per-request ContextVar.
The cached InferenceRouterOpenAIProvider's httpx event hook must then stamp the
shared replay contract on the outbound completion request so inference-router
writes a `replay_invocation_log` row:

    X-Replay-Mode: true
    X-Worker-Name: <worker>          (already always present)
    X-Worker-Version: <active ver>   (workers.config_revision, baked at build)
    X-Correlation-Id: <caller uuid>  (forwarded from the correlation ContextVar)

Non-replay traffic (the overwhelming majority) must be byte-identical to before
— no X-Replay-Mode / X-Worker-Version headers at all. These tests pin both.

The hook is exercised directly (build provider → set ContextVars → invoke the
event hook with an httpx.Request), matching the idiom in
test_ir_outbound_effective_tenant.py. No DB / network required.
"""

from __future__ import annotations

import httpx
import pytest

from src.api.identity_context import (
    set_correlation_context,
    set_identity_context,
    set_replay_mode_context,
)
from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)
CORRELATION_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def _request_hook(provider: InferenceRouterOpenAIProvider) -> object:
    """Pull the single outbound request event hook off the provider's client."""
    http_client = provider._client._client  # AsyncOpenAI -> httpx.AsyncClient
    hooks = http_client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    return hooks[0]


def _reset_context() -> None:
    set_replay_mode_context(False)
    set_correlation_context("", "")
    set_identity_context("", "", "")


@pytest.mark.asyncio
async def test_replay_mode_stamps_full_identity_on_outbound_ir_request() -> None:
    """Replay-mode ContextVar true + a caller correlation id → the outbound IR
    request carries X-Replay-Mode, X-Worker-Version, X-Worker-Name and the
    caller's X-Correlation-Id (the full WS-1 shared contract)."""
    provider = InferenceRouterOpenAIProvider(
        base_url=IR_URL,
        api_key=API_KEY,
        worker_name="prompt_reasoner",
        worker_version="v7",
    )
    hook = _request_hook(provider)
    try:
        set_identity_context("tenant-1", "lic-1", "subj-1")
        set_correlation_context(CORRELATION_ID, "")
        set_replay_mode_context(True)
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert req.headers["X-Replay-Mode"] == "true"
        assert req.headers["X-Worker-Version"] == "v7"
        assert req.headers["X-Correlation-Id"] == CORRELATION_ID
        # X-Worker-Name is carried unconditionally as an OpenAI-SDK default
        # header (merged at send time, not by this hook), so it rides every
        # request including replays — pinned here at its construction source.
        assert provider._worker_name == "prompt_reasoner"
    finally:
        _reset_context()


@pytest.mark.asyncio
async def test_non_replay_request_omits_replay_headers() -> None:
    """With replay-mode false (default for all normal traffic), the outbound
    request must NOT carry X-Replay-Mode or X-Worker-Version — behaviour is
    unchanged for the overwhelming majority of calls."""
    provider = InferenceRouterOpenAIProvider(
        base_url=IR_URL,
        api_key=API_KEY,
        worker_name="prompt_reasoner",
        worker_version="v7",
    )
    hook = _request_hook(provider)
    try:
        set_correlation_context(CORRELATION_ID, "")
        set_replay_mode_context(False)
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert "X-Replay-Mode" not in req.headers
        assert "X-Worker-Version" not in req.headers
    finally:
        _reset_context()


@pytest.mark.asyncio
async def test_replay_mode_emits_empty_worker_version_when_unknown() -> None:
    """A replay against a worker with no config_revision still stamps the mode;
    X-Worker-Version is '' (never the literal 'None') — header-safe by default."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _request_hook(provider)
    try:
        set_correlation_context(CORRELATION_ID, "")
        set_replay_mode_context(True)
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert req.headers["X-Replay-Mode"] == "true"
        assert req.headers["X-Worker-Version"] == ""
    finally:
        _reset_context()
