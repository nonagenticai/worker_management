"""App-id attribution — outbound X-App-Id on the two inference-router clients.

`app_id IS NULL` is the estate's largest attribution gap ($4,384.57 of $5,081.86
over 30d, measured 2026-09-04 against prod), and inference-router deliberately
does NOT default X-App-Id (it fans in across 19 tenants and cannot guess whose
app a call belongs to — `src/routers/completions.py:158` in that repo reads it
with `default="unknown"`, never invents it here). worker_management is a
caller of inference-router and, until this change, had zero app_id plumbing:
the inbound header was never read into a ContextVar and neither outbound
client forwarded one.

Both real clients share ONE hook implementation (`make_identity_event_hook`,
see `test_the_identity_hook_is_defined_once_not_copied` in
test_outbound_tenant_is_real.py) — `InferenceRouterOpenAIProvider.__init__`
and `ModelFactory._create_vllm_model` both install it. These tests pin the
X-App-Id contract on both installation sites plus the no-default rule.
"""

from __future__ import annotations

import asyncio

import httpx2 as httpx
import pytest

from src.api.identity_context import set_app_id_context, set_correlation_context
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIProvider,
    ModelFactory,
    PydanticAIAgentConfig,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)


def _reset_context() -> None:
    set_app_id_context("")
    set_correlation_context("", "")


def _ir_provider_hook(provider: InferenceRouterOpenAIProvider) -> object:
    """Pull the single outbound request event hook off the provider's client."""
    http_client = provider._client._client  # AsyncOpenAI -> httpx.AsyncClient
    hooks = http_client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    return hooks[0]


def _vllm_hooks(cfg: PydanticAIAgentConfig) -> list:
    model = ModelFactory._create_vllm_model(cfg)
    client = getattr(model._provider, "_http_client", None) or getattr(
        model._provider, "http_client", None
    )
    if client is None or not getattr(client, "event_hooks", {}).get("request"):
        client = model._provider.client._client
    hooks = client.event_hooks["request"]
    assert hooks, "vllm client should install the shared identity hook"
    return hooks


# --------------------------------------- InferenceRouterOpenAIProvider


@pytest.mark.asyncio
async def test_ir_provider_forwards_app_id_from_context() -> None:
    """A caller-stamped X-App-Id in context is forwarded to inference-router."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    try:
        set_app_id_context("requirements_management")
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert req.headers["X-App-Id"] == "requirements_management"
    finally:
        _reset_context()


@pytest.mark.asyncio
async def test_ir_provider_omits_app_id_when_context_empty() -> None:
    """No X-App-Id in context → the header is OMITTED, never sent empty and
    never a guessed default. An absent header lets IR's own `default="unknown"`
    fire; a present empty header would not (Header(default=...) only fires on
    absence)."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    try:
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert "X-App-Id" not in req.headers
    finally:
        _reset_context()


# --------------------------------------- the vllm client is an IR client too


def test_vllm_client_forwards_app_id_from_context() -> None:
    """Same shared hook, same contract, on the second outbound path.

    This is the path `principle_consistency` actually uses (VLLM_URL is
    repointed at inference-router — see test_outbound_tenant_is_real.py).
    """
    cfg = PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm",
        model="local-llm",
        db_path="",
        base_url="http://ir:8150/v1",
        tenant_id="tenant-1",
    )
    hooks = _vllm_hooks(cfg)
    try:
        set_app_id_context("asset_evaluator")
        req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
        for h in hooks:
            asyncio.run(h(req))
        assert req.headers.get("X-App-Id") == "asset_evaluator"
    finally:
        _reset_context()


def test_vllm_client_omits_app_id_when_context_empty() -> None:
    cfg = PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm",
        model="local-llm",
        db_path="",
        base_url="http://ir:8150/v1",
        tenant_id="tenant-1",
    )
    hooks = _vllm_hooks(cfg)
    try:
        req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
        for h in hooks:
            asyncio.run(h(req))
        assert "X-App-Id" not in req.headers
    finally:
        _reset_context()
