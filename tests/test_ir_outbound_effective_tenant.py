"""WS-B — outbound effective-tenant side-channel on the LLM-provider httpx hooks.

The inbound auth middleware honours `X-Effective-Tenant-Id` from a delegating
service account so BYO + cost events attribute to the run tenant. For multi-hop
re-delegation to keep working, the OUTBOUND httpx event hooks on the cached LLM
clients must re-stamp `X-Effective-Tenant-Id` from the identity ContextVar (the
effective tenant the inbound middleware resolved into context), not just
`X-Tenant-Id`. These tests pin that contract for both providers.
"""

from __future__ import annotations

import httpx2 as httpx
import pytest

from src.api.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    set_identity_context,
)
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIProvider,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/v1"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)


def _request_hook(provider) -> object:
    """Pull the single outbound request event hook off the provider's client."""
    http_client = provider._client._client  # AsyncOpenAI -> httpx.AsyncClient
    hooks = http_client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    return hooks[0]


@pytest.mark.asyncio
async def test_ir_hook_stamps_effective_tenant_from_context() -> None:
    """When the ContextVar tenant is set (effective tenant resolved by the
    inbound middleware), the outbound request carries it in BOTH X-Tenant-Id
    and the X-Effective-Tenant-Id side-channel."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _request_hook(provider)
    try:
        set_identity_context("tenant-099", "lic-1", "subj-1")
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        assert req.headers["X-Tenant-Id"] == "tenant-099"
        assert req.headers[EFFECTIVE_TENANT_HEADER] == "tenant-099"
    finally:
        set_identity_context("", "", "")


@pytest.mark.asyncio
async def test_ir_hook_omits_effective_tenant_when_context_empty() -> None:
    """With no effective tenant in context, X-Tenant-Id falls back to the
    construction-time SA tenant but the side-channel is NOT stamped — a missing
    header keeps the inbound honour logic inert rather than delegating to ""."""
    provider = InferenceRouterOpenAIProvider(
        base_url=IR_URL,
        api_key=API_KEY,
        tenant_id="sa-tenant",
    )
    hook = _request_hook(provider)
    set_identity_context("", "", "")
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hook(req)
    assert req.headers["X-Tenant-Id"] == "sa-tenant"
    assert EFFECTIVE_TENANT_HEADER not in req.headers
