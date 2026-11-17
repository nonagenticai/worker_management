"""Platform tenants get a real IR token; unproven tenants fail loud; no placeholder.

Owner ruling 2026-10-06: every tenant that calls gets its own licence and token,
so every call is billed to its own tenant. Census (licences DB, 2026-10-06):
every COMMERCIAL_PLATFORM_TENANT_IDS member is kind=internal with an ACTIVE
inference-router licence in a bundle, so the bundle-token mint works for each.

Before this change a platform tenant's completion went out with the SDK
placeholder ``Authorization: Bearer internal`` (``IR_CLIENT_API_KEY``) — a 401
once inference-router enforces introspection, while looking licensed in any
header dump.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx2 as httpx
import pytest

from src.settings import reset_settings_cache

_IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150"
_COMPLETIONS = f"{_IR_URL}/v1/chat/completions"
_N8N_DEV = "bbbbbbbb-0000-4000-8000-0000000000a5"  # commercial-platform (conftest set)
_CUSTOMER = "01a0b1c2-d3e4-7f56-8a9b-0c1d2e3f4a5b"


def _token_for(tenant_id: str) -> str:
    return f"lt_test-bundle-{tenant_id}"  # pragma: allowlist secret


@pytest.fixture
def minted(monkeypatch):
    from src.api.identity_context import set_identity_context
    from src.services import ir_bundle_tokens

    monkeypatch.setenv("INFERENCE_ROUTER_URL", _IR_URL)
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
    reset_settings_cache()
    ir_bundle_tokens.reset_bundle_token_cache()
    calls: list[str] = []

    async def _fake(tenant_id: str) -> str:
        calls.append(tenant_id)
        return _token_for(tenant_id)

    monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _fake)
    yield calls
    set_identity_context(None, None, None)
    ir_bundle_tokens.reset_bundle_token_cache()
    reset_settings_cache()


def _hook(fallback_tenant_id: str = ""):
    from src.worker.agents.pydantic_ai_agent import make_identity_event_hook

    reset_settings_cache()
    return make_identity_event_hook(
        worker_name="principle_consistency",
        worker_version="1",
        fallback_tenant_id=fallback_tenant_id,
        fallback_license_id="",
        fallback_subject_id="",
    )


def _sdk_request(url: str = _COMPLETIONS) -> httpx.Request:
    """What the OpenAI SDK hands the hook: the IR_CLIENT_API_KEY placeholder."""
    from src.worker.agents.pydantic_ai_agent import IR_CLIENT_API_KEY

    return httpx.Request("POST", url, headers={"Authorization": f"Bearer {IR_CLIENT_API_KEY}"})


def _counter(outcome: str) -> Any:
    from src.services.ir_bundle_tokens import BUNDLE_TOKEN_TOTAL

    return BUNDLE_TOKEN_TOTAL.labels(outcome=outcome)


# ------------------------------------------------------------ platform tenant


@pytest.mark.asyncio
async def test_platform_tenant_completion_carries_its_own_minted_token(minted):
    from src.api.identity_context import set_identity_context
    from src.api.tenant_constants import is_commercial_platform_tenant

    assert is_commercial_platform_tenant(_N8N_DEV)  # the case is really a platform tenant
    set_identity_context(_N8N_DEV, "lic-1", "sub-1", tenant_provenance="license")
    req = _sdk_request()
    await _hook()(req)
    assert req.headers["Authorization"] == f"Bearer {_token_for(_N8N_DEV)}"
    assert req.headers["X-Tenant-Id"] == _N8N_DEV
    assert minted == [_N8N_DEV]


@pytest.mark.asyncio
async def test_platform_tenant_vllm_listing_is_licensed(minted, monkeypatch):
    """VLLM_URL is inference-router in prod: /providers/vllm/models is a router call."""
    from src.api.identity_context import set_identity_context
    from src.api.providers import _router as providers

    monkeypatch.setenv("VLLM_URL", _IR_URL)
    reset_settings_cache()
    set_identity_context(_N8N_DEV, "lic-1", "sub-1", tenant_provenance="license")
    sent: list[dict[str, str] | None] = []

    async def _get(url: str, *, headers: dict[str, str] | None = None, **_: Any) -> dict:
        sent.append(headers)
        return {"data": [{"id": "local-llm"}]}

    monkeypatch.setattr(providers, "_httpx_json_get", _get)
    assert await providers._fetch_vllm_models(_N8N_DEV) == ["local-llm"]
    assert len(sent) == 1 and sent[0] is not None
    assert sent[0]["Authorization"] == f"Bearer {_token_for(_N8N_DEV)}"
    assert sent[0]["X-Worker-Name"] == "worker-management"
    # WM's own model listing carries WM's licences app key.
    assert sent[0]["X-App-Id"] == "workers"


# ----------------------------------------------------------- unproven tenant


@pytest.mark.asyncio
async def test_unproven_delegated_tenant_is_refused_loudly(minted, caplog):
    """Delegated with no verified end-user token (and not an allowlisted IR
    delegating service): no mint, ERROR, counted, and the call is REFUSED
    (IRTenantNotProvenError) instead of going out tokenless."""
    from src.api.identity_context import set_identity_context
    from src.services.ir_bundle_tokens import IRTenantNotProvenError

    set_identity_context(_CUSTOMER, "lic-1", "sub-1", tenant_provenance="delegated")
    not_proven = _counter("tenant_not_proven")
    np_before = not_proven._value.get()
    req = _sdk_request()
    with (
        caplog.at_level(logging.INFO, logger="src.services.ir_bundle_tokens"),
        pytest.raises(IRTenantNotProvenError),
    ):
        await _hook()(req)
    assert minted == []
    assert not_proven._value.get() == np_before + 1
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("tenant_not_proven" in r.getMessage() for r in errors), caplog.text


# ------------------------------------------------- placeholder is impossible


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["true", "false"])
@pytest.mark.parametrize("provenance", ["asserted", "default", "unknown"])
async def test_router_never_receives_the_placeholder_bearer(minted, monkeypatch, flag, provenance):
    """Every no-token path — flag off, unproven, worker fallback tenant, no tenant.

    Each either goes out honestly tokenless (flag off) or is REFUSED before it
    is sent (no tenant ⇒ IRTenantRequiredError; unproven tenant with the flag on
    ⇒ IRTenantNotProvenError). In no case does the placeholder survive.
    """
    from src.api.identity_context import set_identity_context
    from src.services.ir_bundle_tokens import IRTenantNotProvenError, IRTenantRequiredError

    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", flag)
    reset_settings_cache()
    for tenant, fallback in ((_CUSTOMER, ""), (None, _CUSTOMER), (None, "")):
        set_identity_context(tenant, None, None, tenant_provenance=provenance)
        req = _sdk_request()
        try:
            await _hook(fallback)(req)
        except IRTenantRequiredError:
            assert not (tenant or fallback), (tenant, fallback)
            continue
        except IRTenantNotProvenError:
            assert flag == "true", (tenant, fallback)
            continue
        assert flag == "false", (tenant, fallback)
        assert req.headers.get("Authorization") is None, (tenant, fallback)
    assert minted == []


@pytest.mark.asyncio
async def test_minted_token_is_not_dropped_and_other_origins_are_untouched(minted):
    from src.api.identity_context import set_identity_context

    set_identity_context(_CUSTOMER, "lic-1", "sub-1", tenant_provenance="license")
    req = _sdk_request()
    await _hook()(req)
    assert req.headers["Authorization"] == f"Bearer {_token_for(_CUSTOMER)}"
    # Not inference-router: not ours to police, the placeholder stays.
    other = _sdk_request("https://api.deepseek.com/v1/chat/completions")
    await _hook()(other)
    assert other.headers["Authorization"] == "Bearer internal"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Bearer internal", True),
        ("bearer INTERNAL", True),
        ("Bearer EMPTY", True),
        ("Bearer", True),
        ("Bearer   ", True),
        ("Bearer lt_real-token", False),
        ("Basic internal", False),
        (None, False),
    ],
)
def test_is_placeholder_bearer(value, expected):
    from src.services.ir_bundle_tokens import is_placeholder_bearer

    assert is_placeholder_bearer(value) is expected
