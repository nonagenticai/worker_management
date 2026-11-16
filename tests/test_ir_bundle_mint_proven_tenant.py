"""A bundle token is minted ONLY for a tenant this request PROVED.

⛔ A privilege keyed on a tenant is open if the tenant can be asserted. With
``WM_IR_BUNDLE_TOKENS`` on, ``ir_authorization`` mints a REAL inference-router
token for the request's tenant. Before this guard, a tokenless trusted-network
caller that named a tenant in a bare ``X-Tenant-Id`` (``tenant_provenance =
asserted``) got WM to act at IR as THAT tenant.

One test per provenance kind: the valid instance mints, the forged / asserted
variant of the same kind makes ZERO licenses calls and is REFUSED
(``IRTenantNotProvenError``) -- never forwarded tokenless. Every refusal is also driven
through both WM->IR paths (``fetch_models`` and the completion identity hook).

The licenses + Keycloak pair is a real ``BundleTokenCache`` behind an
``httpx.MockTransport``, so "minted nothing" is counted at the wire, not at a
patched function.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api.identity_context import (
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_BYPASS,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    TENANT_PROVENANCE_UNKNOWN,
    set_delegating_subject,
    set_identity_context,
)
from src.auth import end_user_token, keycloak_roles
from src.services import ir_bundle_tokens as bt
from src.settings import reset_settings_cache

pytestmark = pytest.mark.asyncio

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150"
COMPLETIONS = f"{IR_URL}/v1/chat/completions"
TENANT = "019edc1e-0000-7000-8000-0000000000d1"
OTHER_TENANT = "019edc1e-0000-7000-8000-0000000000d2"
SA_SUBJECT = "019ffe9f-0000-7000-8000-00000000beef"  # a configured delegator
LICENSES = "http://licenses.licenses.svc.cluster.local:8544"
TOKEN_URL = "https://kc.example/realms/secure-apps/protocol/openid-connect/token"
MINTED = "lt_proven-tenant-bundle-token"  # pragma: allowlist secret
FAKE_SA_JWT = "eyJ.fake-sa.sig"  # pragma: allowlist secret
FAKE_CLIENT_SECRET = "fake-minter-secret"  # pragma: allowlist secret
ISSUER = "https://auth.nonagentic.ai/realms/portal"
KID = "portal-key-mint-guard"


class _Licenses:
    """Counts every licenses bundle-token call (the thing that must not happen)."""

    def __init__(self) -> None:
        self.mints: list[str] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": FAKE_SA_JWT, "expires_in": 300})
        tenant = request.url.path.split("/")[3]
        self.mints.append(tenant)
        return httpx.Response(
            201,
            json={
                "token": MINTED,
                "expires_at": "2099-01-01T00:00:00+00:00",
                "tenant_id": tenant,
                "token_id": "tok-1",
            },
        )


@pytest.fixture
def licenses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
    monkeypatch.setenv("INFERENCE_ROUTER_URL", IR_URL)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    monkeypatch.setenv(
        "KEYCLOAK_JWKS_URL",
        "http://keycloak.keycloak.svc.cluster.local:8080/realms/secure-apps/protocol/openid-connect/certs",
    )
    monkeypatch.delenv("END_USER_JWKS_URL", raising=False)
    monkeypatch.delenv("END_USER_TOKEN_ISSUER", raising=False)
    reset_settings_cache()
    end_user_token._jwks_cache.clear()
    lic = _Licenses()
    bt.reset_bundle_token_cache()
    bt._state["cache"] = bt.BundleTokenCache(
        licenses_url=LICENSES,
        token_url=TOKEN_URL,
        client_id="worker-management-bundle-minter",
        client_secret=FAKE_CLIENT_SECRET,
        transport=httpx.MockTransport(lic.handler),
    )
    yield lic
    bt.reset_bundle_token_cache()
    set_identity_context(None, None, None)
    end_user_token._jwks_cache.clear()
    reset_settings_cache()


@pytest.fixture
def portal_key(monkeypatch: pytest.MonkeyPatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    pub.update({"kid": KID, "alg": "RS256", "use": "sig"})

    async def fake_fetch(_url: str) -> dict:
        return {"keys": [pub]}

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", fake_fetch)
    return key


def _end_user_jwt(key: Any, tenant: str) -> str:
    claims = {
        "sub": "end-user-1",
        "iss": ISSUER,
        "exp": int(time.time()) + 600,
        "tenant_id": tenant,
        "typ": "Bearer",
        "azp": "fr-frontend",
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


async def _delegate(tenant: str, end_user_token_value: str | None, delegator: str = SA_SUBJECT):
    """What the middleware's delegated branch does: stamp, then verify the JWT."""
    from types import SimpleNamespace

    set_identity_context(tenant, None, delegator, tenant_provenance=TENANT_PROVENANCE_DELEGATED)
    set_delegating_subject(delegator)
    headers = {}
    if end_user_token_value is not None:
        headers[end_user_token.END_USER_TOKEN_HEADER] = f"Bearer {end_user_token_value}"
    req = SimpleNamespace(headers=headers, state=SimpleNamespace())
    await end_user_token.stamp_end_user_proof(req, tenant, delegator)


def _not_proven() -> float:
    return bt.BUNDLE_TOKEN_TOTAL.labels(outcome="tenant_not_proven")._value.get()


# ------------------------------------------------------------------ per kind


async def test_license_proven_tenant_mints(licenses: _Licenses) -> None:
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    assert await bt.ir_authorization(COMPLETIONS, TENANT) == f"Bearer {MINTED}"
    assert licenses.mints == [TENANT]


async def test_license_for_one_tenant_never_mints_for_another(licenses: _Licenses) -> None:
    """Forged variant of ``license``: a licence proves ONE tenant. A different
    tenant (graph state, a body field, a worker-config fallback) proves nothing."""
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    before = _not_proven()
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, OTHER_TENANT)
    assert licenses.mints == []
    assert _not_proven() == before + 1


async def test_delegated_with_verified_end_user_token_mints(
    licenses: _Licenses, portal_key: Any
) -> None:
    await _delegate(TENANT, _end_user_jwt(portal_key, TENANT))
    assert await bt.ir_authorization(COMPLETIONS, TENANT) == f"Bearer {MINTED}"
    assert licenses.mints == [TENANT]


@pytest.mark.parametrize(
    "variant",
    ["header_only", "token_for_other_tenant", "token_signed_by_another_key", "delegator_removed"],
)
async def test_delegated_without_a_verified_tenant_claim_mints_nothing(
    licenses: _Licenses, portal_key: Any, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Forged variants of ``delegated``: the tenant came from the
    X-Effective-Tenant-Id HEADER and no signed claim WM validated backs it."""
    if variant == "header_only":
        await _delegate(TENANT, None)
    elif variant == "token_for_other_tenant":
        await _delegate(TENANT, _end_user_jwt(portal_key, OTHER_TENANT))
    elif variant == "token_signed_by_another_key":
        forger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        await _delegate(TENANT, _end_user_jwt(forger, TENANT))
    else:
        await _delegate(TENANT, _end_user_jwt(portal_key, TENANT))
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "someone-else")
    before = _not_proven()
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, TENANT)
    assert licenses.mints == []
    assert _not_proven() == before + 1


@pytest.mark.parametrize(
    "provenance",
    [
        TENANT_PROVENANCE_ASSERTED,
        TENANT_PROVENANCE_DEFAULT,
        TENANT_PROVENANCE_BYPASS,
        TENANT_PROVENANCE_UNKNOWN,
        # Not a known value: coerced to `unknown` by set_identity_context. A
        # provenance nobody listed as proven must never mint (fail closed).
        "end_user_token",
    ],
)
async def test_unproven_provenance_mints_nothing(licenses: _Licenses, provenance: str) -> None:
    set_identity_context(TENANT, None, None, tenant_provenance=provenance)
    before = _not_proven()
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, TENANT)
    assert licenses.mints == []
    assert _not_proven() == before + 1


async def test_no_request_context_mints_nothing(licenses: _Licenses) -> None:
    set_identity_context(None, None, None)
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, TENANT)
    assert licenses.mints == []


async def test_a_token_cached_for_a_proven_request_is_not_served_to_an_asserting_one(
    licenses: _Licenses,
) -> None:
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    assert await bt.ir_authorization(COMPLETIONS, TENANT) == f"Bearer {MINTED}"
    set_identity_context(TENANT, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, TENANT)
    assert licenses.mints == [TENANT]


async def test_flag_off_counts_nothing(licenses: _Licenses, monkeypatch) -> None:
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "false")
    reset_settings_cache()
    set_identity_context(TENANT, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    before = _not_proven()
    assert await bt.ir_authorization(COMPLETIONS, TENANT) is None
    assert _not_proven() == before


# ------------------------------------------- both WM->IR paths, asserted tenant


_RealAsyncClient = httpx.AsyncClient


async def _listing_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Drive ``fetch_models`` with IR mocked but the licenses mint left REAL.

    Patching ``httpx.AsyncClient`` in the registry module patches it for every
    importer (one module object), so the cache's own client is re-pinned to the
    real class with its MockTransport.
    """
    from src.services.inference_router_registry import InferenceRouterRegistry

    monkeypatch.setattr(
        bt.BundleTokenCache,
        "_client",
        lambda self: _RealAsyncClient(transport=self._transport, timeout=5.0),
    )
    registry = InferenceRouterRegistry()
    registry.get_base_url = lambda: IR_URL
    with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "local-llm"}]}
        client.get = AsyncMock(return_value=resp)
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        try:
            await registry.fetch_models(TENANT)
        except bt.IRTenantNotProvenError:
            client.get.assert_not_called()  # refused BEFORE any router call
            raise
        return client.get.call_args.kwargs["headers"]


async def test_fetch_models_for_an_asserted_tenant_is_refused(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_identity_context(TENANT, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    with pytest.raises(bt.IRTenantNotProvenError):
        await _listing_headers(monkeypatch)
    assert licenses.mints == []


async def test_fetch_models_for_a_proven_tenant_sends_the_token(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_identity_context(TENANT, None, None, tenant_provenance=TENANT_PROVENANCE_LICENSE)
    headers = await _listing_headers(monkeypatch)
    assert headers["Authorization"] == f"Bearer {MINTED}"
    assert licenses.mints == [TENANT]


def _hook():
    from src.worker.agents.pydantic_ai_agent import make_identity_event_hook

    reset_settings_cache()
    return make_identity_event_hook(
        worker_name="w",
        worker_version="1",
        fallback_tenant_id="",
        fallback_license_id="",
        fallback_subject_id="",
    )


async def _send_through_hook(hook) -> list[httpx.Request]:
    """Send one completion through a real httpx client carrying ``hook``;
    return every request that actually reached the (mock) router transport."""
    reached: list[httpx.Request] = []

    def _router(request: httpx.Request) -> httpx.Response:
        reached.append(request)
        return httpx.Response(200, json={})

    async with _RealAsyncClient(
        transport=httpx.MockTransport(_router), event_hooks={"request": [hook]}
    ) as client:
        await client.post(COMPLETIONS, headers={"Authorization": "Bearer internal"}, json={})
    return reached


async def test_completion_hook_for_an_asserted_tenant_is_refused_and_never_sent(
    licenses: _Licenses,
) -> None:
    set_identity_context(TENANT, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    reached: list[httpx.Request] = []
    with pytest.raises(bt.IRTenantNotProvenError):
        reached = await _send_through_hook(_hook())
    assert reached == []
    assert licenses.mints == []


async def test_completion_hook_worker_fallback_tenant_is_not_proven(
    licenses: _Licenses,
) -> None:
    """A tenant taken from worker config, not the request, never mints."""
    from src.worker.agents.pydantic_ai_agent import make_identity_event_hook

    set_identity_context(None, None, None)
    reset_settings_cache()
    hook = make_identity_event_hook(
        worker_name="w",
        worker_version="1",
        fallback_tenant_id=TENANT,
        fallback_license_id="",
        fallback_subject_id="",
    )
    req = httpx.Request("POST", COMPLETIONS, headers={"Authorization": "Bearer internal"})
    with pytest.raises(bt.IRTenantNotProvenError):
        await hook(req)
    assert licenses.mints == []


async def test_completion_hook_for_a_proven_tenant_sends_the_token(licenses: _Licenses) -> None:
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    req = httpx.Request("POST", COMPLETIONS, headers={"Authorization": "Bearer internal"})
    await _hook()(req)
    assert req.headers["Authorization"] == f"Bearer {MINTED}"
    assert licenses.mints == [TENANT]


# ------------------------------------------- platform-set membership (fr#1385)
# Owner ruling 2026-10-06: a commercial-platform tenant is minted for like any
# other tenant — but only when PROVEN. Membership grants nothing by itself: an
# asserted platform tenant still gets no token.

from tests._platform_tenant import EXECUTION_TENANT

PLATFORM_0002 = EXECUTION_TENANT  # in the conftest gitops commercial set


async def test_proven_platform_tenant_gets_its_own_bundle_token(licenses: _Licenses) -> None:
    set_identity_context(
        PLATFORM_0002, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE
    )
    assert await bt.ir_authorization(COMPLETIONS, PLATFORM_0002) == f"Bearer {MINTED}"
    assert licenses.mints == [PLATFORM_0002]


async def test_asserted_platform_tenant_gets_no_bundle_token(licenses: _Licenses) -> None:
    set_identity_context(PLATFORM_0002, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, PLATFORM_0002)
    assert licenses.mints == []


async def test_unset_platform_set_treats_every_tenant_as_a_customer(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WM ships before gitops: COMMERCIAL_PLATFORM_TENANT_IDS unset/empty ⇒ no
    platform tenant, no crash, no fallback list. A PROVEN ``…0002`` then follows
    the customer rule (mint for a proven tenant); an asserted one still gets none."""
    monkeypatch.setenv("COMMERCIAL_PLATFORM_TENANT_IDS", "")
    reset_settings_cache()
    set_identity_context(
        PLATFORM_0002, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE
    )
    assert await bt.ir_authorization(COMPLETIONS, PLATFORM_0002) == f"Bearer {MINTED}"
    assert licenses.mints == [PLATFORM_0002]
    set_identity_context(PLATFORM_0002, None, None, tenant_provenance=TENANT_PROVENANCE_ASSERTED)
    bt.reset_bundle_token_cache()
    with pytest.raises(bt.IRTenantNotProvenError):
        await bt.ir_authorization(COMPLETIONS, PLATFORM_0002)


# ------------------------------------- delegating services (WM_DELEGATING_SUBJECTS)
# Live finding 2026-10-07: kag/karma (licence subject 01a0138e-…) calls WM for
# END tenants (merqi, nonagentic-gtm) it names in X-Effective-Tenant-Id, with no
# X-End-User-Token. Mirror of the router's AUTH_DELEGATING_SUBJECTS: an
# allowlisted delegating subject proves the END tenant FOR THE IR TOKEN ONLY,
# so WM mints that end tenant's token (billing goes to the end tenant).

KAG_SUBJECT = "01a0138e-e7b9-712c-b140-cf20d7b9e644"
END_TENANT = "019edc1e-0000-7000-8000-0000000000e1"


async def test_allowlisted_delegating_service_mints_the_end_tenant_token(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", KAG_SUBJECT)
    monkeypatch.setenv(
        "WM_DELEGATING_SUBJECTS", f"8382f0b3-c082-4f97-8d47-120ce5a18e05,{KAG_SUBJECT}"
    )
    reset_settings_cache()
    await _delegate(END_TENANT, None, delegator=KAG_SUBJECT)
    assert await bt.ir_authorization(COMPLETIONS, END_TENANT) == f"Bearer {MINTED}"
    # Minted for the END tenant, never the service account's own tenant.
    assert licenses.mints == [END_TENANT]


async def test_allowlisted_delegating_service_completion_reaches_router_as_end_tenant(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", KAG_SUBJECT)
    monkeypatch.setenv("WM_DELEGATING_SUBJECTS", KAG_SUBJECT)
    reset_settings_cache()
    await _delegate(END_TENANT, None, delegator=KAG_SUBJECT)
    reached = await _send_through_hook(_hook())
    assert len(reached) == 1
    assert reached[0].headers["Authorization"] == f"Bearer {MINTED}"
    assert reached[0].headers["X-Tenant-Id"] == END_TENANT
    assert licenses.mints == [END_TENANT]


async def test_non_allowlisted_delegator_naming_a_tenant_is_refused_no_router_call(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delegator WM honours for re-scoping (AUTH_DELEGATING_SUBJECTS) but NOT on
    WM_DELEGATING_SUBJECTS, with no end-user token: refused, counted, ERROR, and
    the router never sees the call."""
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", KAG_SUBJECT)
    monkeypatch.setenv("WM_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
    reset_settings_cache()
    await _delegate(END_TENANT, None, delegator=KAG_SUBJECT)
    before = _not_proven()
    reached: list[httpx.Request] = []
    with pytest.raises(bt.IRTenantNotProvenError):
        reached = await _send_through_hook(_hook())
    assert reached == []
    assert licenses.mints == []
    assert _not_proven() == before + 1


async def test_ir_delegation_does_not_widen_the_owner_key_gate(
    licenses: _Licenses, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WM_DELEGATING_SUBJECTS is scoped to the IR-token decision: the generic
    tenant proof (owner commercial keys, BYO) still refuses header-only delegation."""
    from src.auth.tenant_proof import tenant_proof_refusal

    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", KAG_SUBJECT)
    monkeypatch.setenv("WM_DELEGATING_SUBJECTS", KAG_SUBJECT)
    reset_settings_cache()
    await _delegate(END_TENANT, None, delegator=KAG_SUBJECT)
    assert tenant_proof_refusal(END_TENANT, ir_token=True) is None
    assert tenant_proof_refusal(END_TENANT) == "delegated_without_verified_end_user_token"
