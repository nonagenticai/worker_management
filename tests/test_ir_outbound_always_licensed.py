"""Every worker_management -> inference-router request carries a licence token.

inference-router is switching ``INFERENCE_ROUTER_INTROSPECT_MODE`` to
``enforce``: from then on a request with neither ``Authorization: Bearer <licence
token>`` nor the ``X-Tenant-Id: Bearer <token>`` form is a 401. The model
listing used to go out TOKENLESS whenever WM had no tenant — worker-factory
validation on ``POST /workers`` auto-provisioning (crud_endpoints /
message_endpoints called ``create_worker`` with no tenant) — which IR counted as
``outcome="absent", worker="unknown"``. A later fix licensed those as WM's own
``WM_IR_SERVICE_TENANT_ID`` (platform-core). Owner ruling 2026-10-07 retired
that: WM NEVER calls inference-router on its own behalf. The endpoints now pass
the caller's end tenant, and a tenantless listing is REFUSED before anything is
sent (``IRTenantRequiredError``).

The HTTP client is mocked and every request it is asked to send is inspected,
so the assertion is about what reaches the wire, not about a helper's return
value. Each case also asserts that at least one request was captured, so a code
path that stopped calling the router cannot pass vacuously.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from src.settings import reset_settings_cache

_IR_URL = "http://test-inference-router:8150"
_SERVICE_TENANT = "bbbbbbbb-0000-4000-8000-0000000000a5"
_CUSTOMER_TENANT = "01a0b1c2-d3e4-7f56-8a9b-0c1d2e3f4a5b"
# The placeholder the OpenAI SDK clients carry until the identity hook swaps in
# a real token (pydantic_ai_agent.IR_CLIENT_API_KEY). It is NOT a licence.
_PLACEHOLDER_BEARERS = {"Bearer internal", "Bearer EMPTY", "Bearer "}


def _token_for(tenant_id: str) -> str:
    return f"lt_test-bundle-{tenant_id}"  # pragma: allowlist secret


def _assert_licensed(headers: dict[str, str]) -> None:
    auth = headers.get("Authorization", "")
    x_tenant = headers.get("X-Tenant-Id", "")
    licensed = (auth.startswith("Bearer ") and auth not in _PLACEHOLDER_BEARERS) or (
        x_tenant.lower().startswith("bearer ")
    )
    assert licensed, f"outbound inference-router request carries no licence token: {headers}"


class _CapturingClient:
    """Stands in for ``httpx.AsyncClient``; records every request's headers."""

    sent: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _CapturingClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None, **_: Any) -> Any:
        type(self).sent.append({"url": url, "headers": dict(headers or {})})

        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict[str, Any]:
                return {"data": [{"id": "local-llm", "provider_chain": ["vllm"]}]}

        return _Resp()


@pytest.fixture
def wired(monkeypatch):
    from src.api.identity_context import set_identity_context
    from src.services import ir_bundle_tokens
    from src.services.inference_router_registry import InferenceRouterRegistry

    monkeypatch.setenv("INFERENCE_ROUTER_URL", _IR_URL)
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
    # The RETIRED service-tenant env var is set on purpose: it must be ignored.
    monkeypatch.setenv("WM_IR_SERVICE_TENANT_ID", _SERVICE_TENANT)
    reset_settings_cache()
    ir_bundle_tokens.reset_bundle_token_cache()

    minted: list[str] = []

    async def _fake_tenant_token(tenant_id: str) -> str:
        minted.append(tenant_id)
        return _token_for(tenant_id)

    # licenses is out of scope here; the mint itself is pinned by
    # tests/test_ir_bundle_tokens.py.
    monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _fake_tenant_token)
    monkeypatch.setattr("src.api.identity_context.get_license_id_from_context", lambda: None)
    _CapturingClient.sent = []
    with patch("src.services.inference_router_registry.httpx.AsyncClient", _CapturingClient):
        yield InferenceRouterRegistry(), minted
    set_identity_context(None, None, None)
    ir_bundle_tokens.reset_bundle_token_cache()
    reset_settings_cache()


@pytest.mark.asyncio
async def test_tenantless_model_validation_is_refused_not_licensed_as_wm(wired):
    """get_model(model, None) ⇒ IRTenantRequiredError; nothing sent, nothing minted."""
    from src.services.ir_bundle_tokens import IRTenantRequiredError

    registry, minted = wired
    with pytest.raises(IRTenantRequiredError):
        await registry.get_model("local-llm", None)
    assert _CapturingClient.sent == []
    assert minted == []


@pytest.mark.asyncio
async def test_every_router_request_is_licensed_as_the_end_tenant(wired):
    """A proven tenant's listing and refresh: both licensed with ITS OWN token."""
    from src.api.identity_context import set_identity_context

    registry, minted = wired
    set_identity_context(_CUSTOMER_TENANT, "lic-1", "sub-1", tenant_provenance="license")
    await registry.fetch_models(_CUSTOMER_TENANT)
    await registry.refresh_model_cache(_CUSTOMER_TENANT)

    assert len(_CapturingClient.sent) == 2
    for req in _CapturingClient.sent:
        assert req["url"] == f"{_IR_URL}/api/models"
        _assert_licensed(req["headers"])
        assert req["headers"]["X-Worker-Name"] == "worker-management"
        assert req["headers"]["Authorization"] == f"Bearer {_token_for(_CUSTOMER_TENANT)}"
        assert req["headers"]["X-Tenant-Id"] == _CUSTOMER_TENANT
    assert minted == [_CUSTOMER_TENANT, _CUSTOMER_TENANT]
    assert _SERVICE_TENANT not in minted


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "n8n-dev", _SERVICE_TENANT])
async def test_retired_service_tenant_setting_never_rescues_a_tenantless_call(
    wired, monkeypatch, value
):
    """Whatever WM_IR_SERVICE_TENANT_ID holds, a tenantless listing is refused
    and counted -- never sent as WM / platform-core."""
    from src.services.ir_bundle_tokens import (
        TENANTLESS_IR_CALL_REFUSED_TOTAL,
        IRTenantRequiredError,
    )

    registry, minted = wired
    monkeypatch.setenv("WM_IR_SERVICE_TENANT_ID", value)
    reset_settings_cache()
    counter = TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(call="listing", worker="worker-management")
    before = counter._value.get()
    with pytest.raises(IRTenantRequiredError):
        await registry.fetch_models(None)
    assert _CapturingClient.sent == []
    assert counter._value.get() == before + 1
    assert minted == []


@pytest.mark.asyncio
async def test_tenantless_listing_to_another_origin_is_refused_too(wired):
    from src.services.ir_bundle_tokens import IRTenantRequiredError

    registry, minted = wired
    registry.get_base_url = lambda: "http://evil.example:8150"
    with pytest.raises(IRTenantRequiredError):
        await registry.fetch_models(None)
    assert _CapturingClient.sent == []
    assert minted == []
