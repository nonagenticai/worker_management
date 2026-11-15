"""Endpoint tests for /tenant-config/modality (masking, gating, resolve)."""

from __future__ import annotations

import os
import secrets
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("WM_SECRET_KEY", secrets.token_hex(32))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import tenant_config
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.worker_management_api import app
from src.auth import (
    TenantContext,
    get_byo_mode,
    get_identity_proven,
    get_tenant_id,
    get_tenant_proven,
)
from src.security.secret_box import encrypt_secret

TENANT = "019edc1e-0000-7000-8000-000000000001"
PLATFORM_TENANT = "0195f48b-d000-7000-8000-000000000001"  # n8n-dev, a platform tenant

# Fake provider credential for the tests. Referenced as a NAME (not a string
# literal after an `apiKey:` key) so the detect-secrets keyword plugin — which
# WM's CI archetype runs — does not flag it. The value is meaningless.
DUMMY = "provider-cred-placeholder"


class _FakeDB:
    """In-memory stand-in for TenantModalityConfigDB keyed by (tenant, modality, role)."""

    def __init__(self) -> None:
        self.rows: dict[tuple, dict] = {}

    async def get_config(self, tenant_id, modality, role="__default__"):
        return self.rows.get((tenant_id, modality, role))

    async def upsert_config(self, *, tenant_id, modality, role="__default__", **cols):
        import datetime

        self.rows[(tenant_id, modality, role)] = {
            "tenant_id": tenant_id,
            "modality": modality,
            "role": role,
            "updated_at": datetime.datetime(2026, 7, 7, tzinfo=datetime.UTC),
            **cols,
        }

    async def delete_config(self, tenant_id, modality, role="__default__"):
        self.rows.pop((tenant_id, modality, role), None)

    async def list_configs(self, tenant_id, modality=None):
        rows = [
            r
            for (t, m, _role), r in self.rows.items()
            if t == tenant_id and (modality is None or m == modality)
        ]
        return sorted(rows, key=lambda r: (r["modality"], r["role"] != "__default__", r["role"]))


async def _fake_tenant_id(request=None):
    return TENANT


_HOLDER: dict[str, Any] = {}


async def _fake_identity_proven() -> bool:
    """Stand in for ``get_identity_proven``, which needs the real middleware.

    ⛔ THE DEFAULT HERE IS A TEST CONVENIENCE AND NOTHING ELSE. These tests run
    with ``ENABLE_AUTH=false``, so ``AuthenticationMiddleware`` — the only code
    that ever stamps ``request.state.identity_proven`` — is not installed, and
    every request would otherwise be unproven. Modelling the legitimate
    token-bearing S2S backend (the caller these resolve tests were written for)
    means saying so explicitly.

    The PRODUCTION default is the opposite: absent attribute ⇒ unproven ⇒ the
    key is withheld. ``test_resolve_withholds_key_when_the_dependency_is_absent``
    pins that by removing this override entirely.
    """
    return bool(_HOLDER.get("identity_proven", True))


async def _fake_tenant_proven() -> bool:
    """Stand in for ``get_tenant_proven``. Same test-only default as its twin.

    ⛔ THE DEFAULT IS True ONLY BECAUSE THE MIDDLEWARE IS ABSENT. In production
    the two flags come from different places on the request and can DISAGREE —
    a delegating service account is ``identity_proven=True,
    tenant_proven=False`` — which is precisely why they are two overrides here
    and not one. ``_as_delegating`` below is the case that would silently
    vanish if this were derived from ``identity_proven``.
    """
    return bool(_HOLDER.get("tenant_proven", True))


def _as_unproven() -> None:
    """Make subsequent requests look like the tokenless trusted-network caller.

    Both flags go False: that branch proved neither who the caller is nor which
    tenant it may act for.
    """
    _HOLDER["identity_proven"] = False
    _HOLDER["tenant_proven"] = False


def _as_delegating() -> None:
    """Make subsequent requests look like a DELEGATING service account.

    The split case, and the whole reason for the second flag: a verified
    licence token (``identity_proven=True``) whose request was then re-scoped to
    a different tenant by ``X-Effective-Tenant-Id`` — a header the middleware
    validates on spelling alone, so the tenant is the caller's own assertion
    (``tenant_proven=False``).
    """
    _HOLDER["identity_proven"] = True
    _HOLDER["tenant_proven"] = False


def _as_owner(monkeypatch) -> None:
    """Make subsequent requests carry the verified managed-BYO owner role.

    In production that role reaches the caller-roles ContextVar only from a
    signature-verified, azp-allowlisted ``X-Caller-Authorization`` SA JWT.
    """
    from src.security import managed_byo

    monkeypatch.setattr(
        managed_byo,
        "get_caller_roles_from_context",
        lambda: (managed_byo.MANAGED_BYO_ADMIN_ROLE,),
    )


@pytest.fixture
def client(monkeypatch):
    fake = _FakeDB()
    _HOLDER["db"] = fake
    _HOLDER["identity_proven"] = True
    _HOLDER["tenant_proven"] = True
    monkeypatch.setattr(tenant_config._modality_router, "_db", fake)
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    app.dependency_overrides[get_identity_proven] = _fake_identity_proven
    app.dependency_overrides[get_tenant_proven] = _fake_tenant_proven
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


def test_get_unconfigured(client):
    resp = client.get("/tenant-config/modality/tts")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "configured": False,
        "provider": None,
        "model": None,
        "baseUrl": None,
        "params": {},
        "hasKey": False,
        "ownerProvided": False,
        "updatedAt": None,
    }


def test_unknown_modality_404(client):
    assert client.get("/tenant-config/modality/bogus").status_code == 404
    assert client.get("/tenant-config/modality/bogus/catalog").status_code == 404
    assert client.get("/tenant-config/modality/bogus/resolve").status_code == 404


def test_catalog_lists_providers(client):
    body = client.get("/tenant-config/modality/tts/catalog").json()
    keys = {p["key"] for p in body["providers"]}
    assert keys == {"local", "openai", "elevenlabs", "self_hosted"}


def test_catalog_byo_required_includes_self_hosted(client):
    # self_hosted is external → a byo_required tenant may point at a local/rent-GPU
    # endpoint; the non-external "local" preset stays hidden.
    app.dependency_overrides[get_byo_mode] = lambda: "byo_required"
    body = client.get("/tenant-config/modality/tts/catalog").json()
    keys = {p["key"] for p in body["providers"]}
    assert "local" not in keys
    assert keys == {"openai", "elevenlabs", "self_hosted"}


def test_catalog_byo_required_hides_local(client):
    app.dependency_overrides[get_byo_mode] = lambda: "byo_required"
    body = client.get("/tenant-config/modality/asr/catalog").json()
    keys = {p["key"] for p in body["providers"]}
    assert "local" not in keys
    assert keys == {"openai", "deepgram", "self_hosted"}


@pytest.mark.parametrize(
    ("modality", "expected_keys"),
    [
        ("asr", {"openai", "deepgram", "self_hosted"}),
        ("image", {"openai", "self_hosted"}),
        ("embedding", {"openai", "openai_compatible", "self_hosted"}),
    ],
)
def test_catalog_all_modalities_offer_self_hosted_under_byo(client, modality, expected_keys):
    # Phase B: asr / image / embedding each gain a self_hosted preset so a
    # byo_required tenant can point at the owner GPU (or a rent-GPU box).
    app.dependency_overrides[get_byo_mode] = lambda: "byo_required"
    body = client.get(f"/tenant-config/modality/{modality}/catalog").json()
    keys = {p["key"] for p in body["providers"]}
    assert "local" not in keys
    assert keys == expected_keys


@pytest.mark.parametrize(
    ("modality", "base_url"),
    [
        ("asr", "http://asr.example.com:9201"),
        ("image", "http://image-gen.example.com:9204"),
        ("embedding", "http://mlx-ingestion-backend.example.com:9120"),
    ],
)
def test_self_hosted_put_and_resolve_per_modality(client, modality, base_url):
    # PUT without baseUrl → 422 (requires_base_url); with baseUrl → resolve returns
    # a configured, keyless BYO route carrying the base_url. The preset's own
    # owner-GPU default is accepted for a PLATFORM tenant only (a customer is
    # refused it: test_customer_self_hosted_owner_gpu_default_is_refused).
    app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
    missing = client.put(
        f"/tenant-config/modality/{modality}", json={"provider": "self_hosted", "model": "x"}
    )
    assert missing.status_code == 422, missing.text
    assert "base_url" in missing.text

    resp = client.put(
        f"/tenant-config/modality/{modality}",
        json={"provider": "self_hosted", "model": "x", "baseUrl": base_url},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["baseUrl"] == base_url
    body = client.get(f"/tenant-config/modality/{modality}/resolve").json()
    assert body["configured"] is True
    assert body["provider"] == "self_hosted"
    assert body["base_url"] == base_url
    assert body["byo"] is True
    assert body["api_key"] is None


def test_catalog_platform_only_is_local_only(client):
    app.dependency_overrides[get_byo_mode] = lambda: "platform_only"
    body = client.get("/tenant-config/modality/image/catalog").json()
    assert [p["key"] for p in body["providers"]] == ["local"]


def test_put_sets_and_masks_key(client):
    resp = client.put(
        "/tenant-config/modality/tts",
        json={"provider": "elevenlabs", "model": "eleven_turbo_v2_5", "apiKey": DUMMY},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hasKey"] is True
    assert body["provider"] == "elevenlabs"
    assert "apiKey" not in body and "api_key" not in body
    assert DUMMY not in resp.text


def test_put_byo_required_rejects_local(client):
    app.dependency_overrides[get_byo_mode] = lambda: "byo_required"
    resp = client.put("/tenant-config/modality/tts", json={"provider": "local", "model": "kokoro"})
    assert resp.status_code == 403, resp.text


def test_put_platform_only_rejects_external(client):
    app.dependency_overrides[get_byo_mode] = lambda: "platform_only"
    resp = client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    assert resp.status_code == 403, resp.text


def test_put_unknown_provider_422(client):
    resp = client.put(
        "/tenant-config/modality/tts", json={"provider": "nope", "model": "x", "apiKey": DUMMY}
    )
    assert resp.status_code == 422


def test_put_requires_key_for_openai(client):
    resp = client.put("/tenant-config/modality/tts", json={"provider": "openai", "model": "tts-1"})
    assert resp.status_code == 422


def test_vlm_anthropic_preset_is_external():
    # The catalog preset is the allowlist: without it, PUT 422s and resolve
    # ignores the row. Its provider field is what the gateway's VLM route gates
    # its Anthropic BYO adapter on (SUPPORTED_CLOUD_PROVIDERS).
    from src.worker.agents.modality_catalog import get_preset

    preset = get_preset("vlm", "anthropic")
    assert preset is not None
    assert preset["provider"] == "anthropic"
    assert preset["is_external"] is True
    assert preset["requires_api_key"] is True
    assert preset["requires_base_url"] is False


def test_vlm_anthropic_put_and_resolve_reachable(client):
    # End-to-end reachability of the gateway's Anthropic VLM adapter: a tenant
    # configures provider=anthropic and resolve returns a byo route whose
    # provider ("anthropic") the gateway routes to its Messages-API adapter.
    resp = client.put(
        "/tenant-config/modality/vlm",
        json={"provider": "anthropic", "model": "claude-opus-4-8", "apiKey": DUMMY},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hasKey"] is True
    assert body["provider"] == "anthropic"
    assert DUMMY not in resp.text  # key never echoed

    resolved = client.get("/tenant-config/modality/vlm/resolve").json()
    assert resolved["configured"] is True
    assert resolved["provider"] == "anthropic"
    assert resolved["byo"] is True
    assert resolved["cost_provider"] == "anthropic"
    assert resolved["api_key"] == DUMMY  # decrypted for the S2S backend


def test_put_requires_base_url_for_compatible(client):
    resp = client.put(
        "/tenant-config/modality/embedding",
        json={"provider": "openai_compatible", "model": "bge", "apiKey": DUMMY},
    )
    assert resp.status_code == 422


def test_put_self_hosted_requires_base_url(client):
    # self_hosted has requires_base_url=True → PUT with a model but no baseUrl is
    # rejected specifically for the missing base_url.
    resp = client.put(
        "/tenant-config/modality/tts", json={"provider": "self_hosted", "model": "kokoro"}
    )
    assert resp.status_code == 422, resp.text
    assert "base_url" in resp.text


def test_resolve_self_hosted_returns_base_url_and_byo(client):
    app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
    resp = client.put(
        "/tenant-config/modality/tts",
        json={
            "provider": "self_hosted",
            "model": "kokoro",
            "baseUrl": "http://tts.example.com:9202",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["baseUrl"] == "http://tts.example.com:9202"
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is True
    assert body["provider"] == "self_hosted"
    assert body["base_url"] == "http://tts.example.com:9202"
    assert body["byo"] is True  # is_external=True → resolved as a BYO route
    assert body["api_key"] is None  # requires_api_key=False


def test_delete(client):
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    assert client.get("/tenant-config/modality/tts").json()["configured"] is True
    assert client.delete("/tenant-config/modality/tts").status_code == 204
    assert client.get("/tenant-config/modality/tts").json()["configured"] is False


def test_resolve_returns_decrypted_key_and_byo(client):
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1-hd", "apiKey": DUMMY},
    )
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "tts-1-hd"
    assert body["api_key"] == DUMMY  # decrypted round-trip
    assert body["cost_provider"] == "openai"
    assert body["byo"] is True


def test_resolve_local_row_is_not_byo(client):
    client.put("/tenant-config/modality/tts", json={"provider": "local", "model": "kokoro"})
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is True
    assert body["byo"] is False
    assert body["cost_provider"] == "kokoro"


def test_resolve_no_config_local_allowed_falls_open(client):
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is False
    assert body["byo"] is False


def test_resolve_no_config_byo_required_409(client):
    app.dependency_overrides[get_byo_mode] = lambda: "byo_required"
    resp = client.get("/tenant-config/modality/tts/resolve")
    assert resp.status_code == 409, resp.text


def test_resolve_platform_only_ignores_external_row(client):
    # Store an external row (as local_allowed), then resolve under platform_only.
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    app.dependency_overrides[get_byo_mode] = lambda: "platform_only"
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is False
    assert body["byo"] is False


def test_resolve_product_override_tightens_to_byo_required_409(client, monkeypatch):
    # Tenant-wide posture is the default (local_allowed → would fall open), but the
    # PER-PRODUCT override for this product is byo_required, so resolving with
    # ?product must 409 (no config row) — the per-product posture wins.
    mock = AsyncMock(return_value="byo_required")
    monkeypatch.setattr(tenant_config._modality_router, "_lookup_tenant_byo_mode", mock)
    resp = client.get("/tenant-config/modality/tts/resolve?product=prod-a")
    assert resp.status_code == 409, resp.text
    mock.assert_awaited_once()


def test_resolve_product_override_platform_only_ignores_external_row(client, monkeypatch):
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    # Tenant-wide stays local_allowed; the per-product override is platform_only,
    # so the stored external row is ignored for this product.
    monkeypatch.setattr(
        tenant_config._modality_router,
        "_lookup_tenant_byo_mode",
        AsyncMock(return_value="platform_only"),
    )
    body = client.get("/tenant-config/modality/tts/resolve?product=prod-a").json()
    assert body["configured"] is False
    assert body["byo"] is False


def test_resolve_without_product_skips_per_product_lookup(client, monkeypatch):
    # No ?product → the per-product lookup must NOT run; tenant-wide posture applies.
    mock = AsyncMock(return_value="byo_required")
    monkeypatch.setattr(tenant_config._modality_router, "_lookup_tenant_byo_mode", mock)
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is False and body["byo"] is False
    mock.assert_not_awaited()


def test_tier_falls_back_to_default(client):
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    # A tier with no row resolves the default row.
    body = client.get("/tenant-config/modality/tts/resolve?tier=high").json()
    assert body["configured"] is True
    assert body["provider"] == "openai"


def test_owner_provided_put_masks_and_resolves(client, monkeypatch):
    # PUT a managed-BYO (owner-provided) external key → masked GET + resolve both
    # surface ownerProvided/owner_provided True. D5: the decrypted OWNER key is
    # NOT returned on the tenant's own token, only to the verified gateway.
    _as_owner(monkeypatch)
    resp = client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY, "ownerProvided": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ownerProvided"] is True
    assert client.get("/tenant-config/modality/tts").json()["ownerProvided"] is True
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["byo"] is True
    assert body["owner_provided"] is True
    assert body["api_key"] is None
    monkeypatch.setattr(
        tenant_config._modality_router,
        "get_caller_client_id_from_context",
        lambda: "modality-gateway",
    )
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["api_key"] == DUMMY


def test_owner_provided_defaults_false(client):
    # A tenant-supplied BYO key (no ownerProvided flag) resolves owner_provided False.
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY},
    )
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["byo"] is True
    assert body["owner_provided"] is False


def test_owner_provided_preserved_on_edit(client, monkeypatch):
    # The owner sets a managed key; the TENANT then edits the model without the
    # flag → allowed, and the mark is preserved.
    with monkeypatch.context() as m:
        _as_owner(m)
        client.put(
            "/tenant-config/modality/tts",
            json={"provider": "openai", "model": "tts-1", "apiKey": DUMMY, "ownerProvided": True},
        )
    resp = client.put(
        "/tenant-config/modality/tts", json={"provider": "openai", "model": "tts-1-hd"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ownerProvided"] is True
    # Only the owner may clear the flag (a tenant is refused: TestManagedRowLocked).
    _as_owner(monkeypatch)
    resp = client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1-hd", "ownerProvided": False},
    )
    assert resp.json()["ownerProvided"] is False


def test_owner_provided_ignored_for_local_row(client, monkeypatch):
    # owner_provided is a BYO/external concept; a non-external local row never
    # resolves as owner-provided even if the flag was somehow stored.
    _as_owner(monkeypatch)
    client.put(
        "/tenant-config/modality/tts",
        json={"provider": "local", "model": "kokoro", "ownerProvided": True},
    )
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["byo"] is False
    assert body["owner_provided"] is False


# ─────────────────────────────────────────────────────────────────────────────
# /resolve must not hand a decrypted BYO provider key to a caller whose
# identity was never proven.
#
# ⛔ WHY THESE EXIST. `/tenant-config/modality/{modality}/resolve` is the ONLY
# route in this service that returns a tenant's plaintext provider credential
# in a response body. `AuthenticationMiddleware` admits a tokenless request
# from anywhere in TRUSTED_CIDRS (the pod network) and lets it name its own
# tenant in a bare X-Tenant-Id header, checked on spelling alone; the `workers`
# namespace is ambient L4 with no waypoint, so there is no mTLS peer identity
# either. Before this gate, any pod in the cluster could read any tenant's
# provider key by asserting that tenant's uuid in a header.
#
# ⭐ Every assertion below is on the ABSENCE of the key and on the counter —
# never on a key's value. The one place a key value is asserted is the
# proven-caller test, which must keep passing or the fix has broken the
# legitimate S2S path instead of narrowing it.
# ─────────────────────────────────────────────────────────────────────────────


class _RecordingMetrics:
    """Captures record_modality_key_withheld calls; everything else is inert."""

    def __init__(self) -> None:
        self.withheld: list[tuple[str, str]] = []

    def record_modality_key_withheld(self, modality: str, *, reason: str) -> None:
        self.withheld.append((modality, reason))

    def __getattr__(self, _name):  # pragma: no cover - inert for other counters
        def _noop(*_args, **_kwargs) -> None:
            return None

        return _noop


@pytest.fixture
def metrics():
    rec = _RecordingMetrics()
    with patch.object(tenant_config._modality_router, "get_metrics_client", lambda: rec):
        yield rec


def _configure_byo_key(client) -> None:
    resp = client.put(
        "/tenant-config/modality/tts",
        json={"provider": "openai", "model": "tts-1-hd", "apiKey": DUMMY},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["hasKey"] is True


def test_resolve_gives_the_key_to_a_proven_caller(client, metrics):
    # The legitimate S2S backend: it presented a licence token, the middleware
    # introspected it active, identity_proven is True. Nothing is withheld and
    # nothing is counted — this is the control for every test below it.
    _configure_byo_key(client)
    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["configured"] is True
    assert body["provider"] == "openai"
    assert body["api_key"] == DUMMY
    assert body["hasKey"] is True
    assert body["keyWithheld"] is False
    assert metrics.withheld == []


def test_resolve_withholds_the_key_from_an_unproven_caller(client, metrics):
    # The tokenless trusted-network caller. It gets the FULL resolution — the
    # provider/model/base_url half was never secret and refusing it outright
    # would break callers that do not need a key — with api_key stripped and
    # hasKey telling it a key exists that it was not handed.
    _configure_byo_key(client)
    _as_unproven()

    resp = client.get("/tenant-config/modality/tts/resolve")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["configured"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "tts-1-hd"
    assert body["byo"] is True
    assert body["cost_provider"] == "openai"

    assert body["api_key"] is None
    assert body["hasKey"] is True
    assert body["keyWithheld"] is True
    # Belt and braces: the secret must not appear anywhere in the payload,
    # including a field somebody adds later.
    assert DUMMY not in resp.text

    assert metrics.withheld == [("tts", "unproven_identity")]


def test_resolve_withholds_the_key_from_a_delegating_caller_on_an_adopted_tenant(client, metrics):
    """⛔ THE RESIDUAL PATH. A verified token is not entitlement to a tenant.

    The delegating service account presented a real licence token — WHO it is
    was proven — and then re-scoped this request to somebody else's tenant with
    ``X-Effective-Tenant-Id``, which the middleware validates on spelling
    alone. Gating on ``identity_proven`` by itself returned the ADOPTED
    tenant's decrypted key to it. Everything else about delegation is
    unchanged: the resolution is still for the adopted tenant, still says
    ``byo``, still names provider and model. Only the credential is withheld,
    and it is counted under its OWN reason so this population never hides
    inside the tokenless one.
    """
    _configure_byo_key(client)
    _as_delegating()

    resp = client.get("/tenant-config/modality/tts/resolve")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Not a refusal — the delegating subject keeps working for everything it
    # does today. If this half ever flips to a 401/409, the fix has broken the
    # path instead of narrowing it.
    assert body["configured"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "tts-1-hd"
    assert body["byo"] is True

    assert body["api_key"] is None
    assert body["hasKey"] is True
    assert body["keyWithheld"] is True
    # Belt and braces: the secret must not appear anywhere in the payload.
    assert DUMMY not in resp.text

    assert metrics.withheld == [("tts", "adopted_tenant")]


def test_resolve_still_gives_the_key_for_the_delegators_own_tenant(client, metrics):
    """The same subject, NOT delegating, must still be handed the key.

    The control for the test above. Without it, "withhold from a delegating
    subject" and "withhold from every token caller" are indistinguishable, and
    the fix could be a blanket refusal that happens to pass the adopted-tenant
    assertion.
    """
    _configure_byo_key(client)
    _HOLDER["identity_proven"] = True
    _HOLDER["tenant_proven"] = True

    body = client.get("/tenant-config/modality/tts/resolve").json()
    assert body["api_key"] == DUMMY
    assert body["hasKey"] is True
    assert body["keyWithheld"] is False
    assert metrics.withheld == []


def test_withhold_reason_separates_adopted_from_unproven_and_indeterminate(metrics):
    """The three reasons are three populations with three different fixes.

    Asserted on ``_withhold_reason`` directly so the mapping is pinned even for
    combinations no live branch produces yet. ``None`` must require BOTH flags:
    that is what makes any future context shape fail closed.
    """
    reason = tenant_config._modality_router._withhold_reason

    def _ctx(identity: bool, tenant: bool) -> TenantContext:
        return TenantContext(
            tenant_id=TENANT,
            subject_id="",
            tenant_name="",
            visibility="tenant",
            identity_proven=identity,
            tenant_proven=tenant,
        )

    assert reason(_ctx(True, True)) is None
    assert reason(_ctx(True, False)) == "adopted_tenant"
    assert reason(_ctx(False, False)) == "unproven_identity"
    # Identity unproven wins over the tenant flag: the tokenless branch stamps
    # both False and must keep reporting as itself, not as a delegator.
    assert reason(_ctx(False, True)) == "unproven_identity"

    # A PARTIAL shape — carries the old flag, not the new one. This is what a
    # rollback to the previous release looks like to anything that imports
    # this module, and it must NOT be read as entitled on the strength of the
    # half it has.
    class _OldShape:
        tenant_id = TENANT
        identity_proven = True

    assert reason(_OldShape()) == "indeterminate"


def test_resolve_withholds_key_when_the_dependency_is_absent(monkeypatch, metrics):
    # ⭐ FAIL CLOSED BY OMISSION. No get_identity_proven override at all and no
    # AuthenticationMiddleware (ENABLE_AUTH=false), so request.state carries no
    # identity_proven attribute — exactly the shape of a deployment running
    # without auth, or a future branch that forgets to stamp. The real
    # dependency's False default must still withhold.
    fake = _FakeDB()
    monkeypatch.setattr(tenant_config._modality_router, "_db", fake)
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    try:
        with (
            patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
            patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            _configure_byo_key(c)
            resp = c.get("/tenant-config/modality/tts/resolve")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["configured"] is True
            assert body["api_key"] is None
            assert body["hasKey"] is True
            assert body["keyWithheld"] is True
            assert DUMMY not in resp.text
            assert metrics.withheld == [("tts", "unproven_identity")]
    finally:
        app.dependency_overrides.clear()


def test_key_for_caller_withholds_and_counts_indeterminate(metrics):
    # A context that does not carry the flag at all (an older shape, a
    # hand-built object, anything that is not a TenantContext). It must
    # withhold, and it must be counted APART from the tokenless callers so a
    # regression in the context shape does not hide inside their population.
    class _ContextWithoutTheFlag:
        tenant_id = TENANT

    withheld = tenant_config._modality_router._key_for_caller(
        _ContextWithoutTheFlag(), "tts", DUMMY
    )
    assert withheld is None
    assert metrics.withheld == [("tts", "indeterminate")]


def test_key_for_caller_returns_none_without_counting_when_there_is_no_key(metrics):
    # No key configured is not a withholding. Counting it would flood the
    # migration instrument with rows that need no migration.
    ctx = TenantContext(
        tenant_id=TENANT,
        subject_id="",
        tenant_name="",
        visibility="tenant",
        identity_proven=False,
    )
    assert tenant_config._modality_router._key_for_caller(ctx, "tts", None) is None
    assert metrics.withheld == []


def test_telemetry_failure_cannot_hand_out_the_key():
    # ⛔ The counter is best-effort; the withholding is not. A metrics backend
    # that raises must never be able to turn a withheld secret into a returned
    # one.
    class _BrokenMetrics:
        def record_modality_key_withheld(self, *a, **k):
            raise RuntimeError("metrics backend down")

    ctx = TenantContext(
        tenant_id=TENANT,
        subject_id="",
        tenant_name="",
        visibility="tenant",
        identity_proven=False,
    )
    with patch.object(
        tenant_config._modality_router, "get_metrics_client", lambda: _BrokenMetrics()
    ):
        assert tenant_config._modality_router._key_for_caller(ctx, "tts", DUMMY) is None


def test_tenant_context_defaults_to_unproven():
    # The dataclass defaults are half of the fail-closed story; the other half
    # is get_identity_proven / get_tenant_proven's getattr defaults, pinned
    # above. BOTH fields must default False: a `tenant_proven` defaulting True
    # would hand the key to every hand-built context in the codebase.
    ctx = TenantContext(tenant_id=TENANT, subject_id="", tenant_name="", visibility="tenant")
    assert ctx.identity_proven is False
    assert ctx.tenant_proven is False


# ─────────────────────────────────────────────────────────────────────────────
# THE WIRE — AuthenticationMiddleware -> request.state -> get_tenant_proven ->
#            TenantContext.tenant_proven -> _withhold_reason -> response body
#
# ⛔ NOTHING IN THIS SECTION MAY REGISTER A dependency_overrides ENTRY, AND THAT
# RESTRICTION IS THE ENTIRE POINT OF THE SECTION. Both ENDS of this fix were
# already tested and the WIRE BETWEEN THEM WAS NOT: the endpoint tests above
# override `get_identity_proven` / `get_tenant_proven`, so they never read
# `request.state` at all, and `TestIdentityProvenStamp` in
# tests/test_auth_middleware.py asserts on `request.state` and stops there.
#
# ⭐ MEASURED CONSEQUENCE, not a worry. Swapping ONE line in
# `src/auth/dependencies.py::get_tenant_proven` —
#
#     -    return bool(getattr(request.state, "tenant_proven", False))
#     +    return bool(getattr(request.state, "identity_proven", False))
#
# — hands a delegating service account the ADOPTED tenant's decrypted key, and
# left every one of the 133 tests in these two files GREEN. A dependency
# override is a wire cut: it satisfies the consumer without the producer, so no
# number of tests on either side can see a mismatch between them.
#
# These tests run the real middleware, the real `get_tenant_context` and the
# real router in one pass, so the flag has to survive every hop.
# ─────────────────────────────────────────────────────────────────────────────

# svc-kag's subject, as configured in AUTH_DELEGATING_SUBJECTS on the deployed
# worker-management (verified 2026-09-12 three agreeing ways: the running pod's
# env, the live Deployment, and gitops `origin/main`
# clusters/strato/services/workers/deployment.yaml). It is a public subject
# uuid, not a credential.
SVC_KAG_SUBJECT = "01a03c8f-16a3-76a6-98d9-79987ee191e3"
# The tenant svc-kag's own licence introspects to.
SVC_KAG_OWN_TENANT = "22222222-2222-4222-8222-222222222222"
# The tenant it adopts via X-Effective-Tenant-Id — i.e. the victim.
ADOPTED_TENANT = "44444444-4444-4444-8444-444444444444"


def _wire_app() -> FastAPI:
    """A FastAPI app with the REAL middleware in front of the REAL router.

    Deliberately not `src.api.worker_management_api.app`: that one is built
    once at import with ``ENABLE_AUTH=false`` (set at the top of this module),
    so it carries no ``AuthenticationMiddleware`` and can never stamp
    ``request.state.identity_proven`` / ``tenant_proven`` at all.
    """
    wire = FastAPI()
    wire.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    wire.include_router(tenant_config._modality_router.router)
    return wire


def _seed_byo_key_row(fake: _FakeDB, tenant: str) -> None:
    """Give ``tenant`` a tts row that HAS a key, without going through a route.

    Seeded directly rather than via ``PUT`` so the write path cannot silently
    become part of what these tests prove. ``encrypt_secret`` is the same
    function the PUT handler uses, so ``decrypt_secret`` on the read path sees
    exactly the production shape.
    """
    import datetime

    fake.rows[(tenant, "tts", "__default__")] = {
        "tenant_id": tenant,
        "modality": "tts",
        "role": "__default__",
        "provider": "openai",
        "model": "tts-1-hd",
        "base_url": None,
        "api_key_encrypted": encrypt_secret(DUMMY),
        "owner_provided": False,
        "cost_source": None,
        "updated_at": datetime.datetime(2026, 9, 12, tzinfo=datetime.UTC),
    }


def _delegating_introspection() -> dict[str, Any]:
    return {"active": True, "tenant_id": SVC_KAG_OWN_TENANT, "subject_id": SVC_KAG_SUBJECT}


def test_wire_withholds_the_adopted_tenants_key_with_no_dependency_overrides(monkeypatch, metrics):
    """⛔ THE MUTANT THIS EXISTS FOR: ``get_tenant_proven`` reading the wrong flag.

    svc-kag presents a real licence token (so ``identity_proven`` is True and
    should be) and re-scopes the request to somebody else's tenant with
    ``X-Effective-Tenant-Id``, which ``_effective_tenant`` validates on
    SPELLING alone. Every hop is real: the middleware decides ``delegated`` and
    stamps ``request.state.tenant_proven = not delegated``; ``get_tenant_proven``
    reads that attribute off ``request.state``; ``get_tenant_context`` hangs it
    on ``TenantContext.tenant_proven``; ``_withhold_reason`` reads the field;
    ``_key_for_caller`` drops the credential and labels the population.

    A mismatch anywhere along that chain reddens this test and nothing else.
    """
    fake = _FakeDB()
    _seed_byo_key_row(fake, ADOPTED_TENANT)
    monkeypatch.setattr(tenant_config._modality_router, "_db", fake)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SVC_KAG_SUBJECT)

    wire = _wire_app()
    # The contract of this section, asserted rather than narrated.
    assert wire.dependency_overrides == {}

    with (
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=_delegating_introspection(),
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
        TestClient(wire, raise_server_exceptions=False) as c,
    ):
        resp = c.get(
            "/tenant-config/modality/tts/resolve",
            headers={
                "X-Tenant-Id": "Bearer sa-token",
                "X-Effective-Tenant-Id": ADOPTED_TENANT,
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # ⭐ The adoption really happened. Without this the test could pass because
    # delegation silently stopped working — the row it resolved would then be
    # svc-kag's own (absent) one, and "no key returned" would mean nothing.
    assert body["configured"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "tts-1-hd"
    assert body["byo"] is True

    # Withheld, not refused.
    assert body["api_key"] is None
    assert body["hasKey"] is True
    assert body["keyWithheld"] is True
    assert DUMMY not in resp.text

    # Its OWN population, so the "present a token FOR THAT TENANT" fix is never
    # counted with the tokenless "present a token" one.
    assert metrics.withheld == [("tts", "adopted_tenant")]


def test_wire_still_hands_the_key_to_the_same_subject_on_its_own_tenant(monkeypatch, metrics):
    """The control, run through the same wire and the same subject.

    Without it, "the wire carries ``tenant_proven`` correctly" and "the wire
    withholds from everybody" are indistinguishable — a blanket refusal, or a
    ``get_tenant_proven`` hard-wired to False, would satisfy the test above.
    Same service account, same token, NO ``X-Effective-Tenant-Id``: nothing is
    adopted, so the tenant came out of the introspected licence and the key
    must still flow.
    """
    fake = _FakeDB()
    _seed_byo_key_row(fake, SVC_KAG_OWN_TENANT)
    monkeypatch.setattr(tenant_config._modality_router, "_db", fake)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SVC_KAG_SUBJECT)

    wire = _wire_app()
    assert wire.dependency_overrides == {}

    with (
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=_delegating_introspection(),
        ),
        TestClient(wire, raise_server_exceptions=False) as c,
    ):
        resp = c.get(
            "/tenant-config/modality/tts/resolve",
            headers={"X-Tenant-Id": "Bearer sa-token"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["api_key"] == DUMMY
    assert body["hasKey"] is True
    assert body["keyWithheld"] is False
    assert metrics.withheld == []


def test_wire_withholds_from_the_tokenless_trusted_network_caller(monkeypatch, metrics):
    """The third population, also end to end and also with no overrides.

    ``test_resolve_withholds_key_when_the_dependency_is_absent`` above covers
    the shape where NOTHING stamped the flags. This one covers the shape where
    the middleware stamped them **False** — a different fact reaching the same
    verdict — and pins that it is labelled ``unproven_identity`` rather than
    ``adopted_tenant`` all the way out at the response.
    """
    fake = _FakeDB()
    _seed_byo_key_row(fake, ADOPTED_TENANT)
    monkeypatch.setattr(tenant_config._modality_router, "_db", fake)

    wire = _wire_app()
    assert wire.dependency_overrides == {}

    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=True),
        TestClient(wire, raise_server_exceptions=False) as c,
    ):
        resp = c.get(
            "/tenant-config/modality/tts/resolve",
            headers={"X-Tenant-Id": ADOPTED_TENANT},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["configured"] is True
    assert body["api_key"] is None
    assert body["hasKey"] is True
    assert body["keyWithheld"] is True
    assert DUMMY not in resp.text
    assert metrics.withheld == [("tts", "unproven_identity")]


def test_getattr_defaults_hold_for_a_direct_caller(metrics):
    """⚠️ THE ONE THING THAT PINS THE ``getattr`` DEFAULTS, AND IT IS NOT THE ENDPOINT.

    ⛔ CORRECTED 2026-09-12. The PR body claimed ``_is_proven``'s ``getattr``
    default was killed by ``test_key_for_caller_withholds_and_counts_indeterminate``.
    It is not, and re-running the mutant says so: ``_withhold_reason`` opens
    with a ``hasattr`` guard that returns ``indeterminate`` BEFORE either helper
    is reached, so through the route the default is unreachable and flipping it
    to ``True`` changes no endpoint behaviour at all. The same restructure made
    the twin default on ``_is_tenant_proven`` equally unreachable from there.

    The defaults are still worth keeping and still worth proving — they are the
    contract for the DIRECT callers (the resolve handler's own log line calls
    both helpers, and any future importer may) — so this test calls them
    directly, which is the only place the default can fire. Flipping either
    ``False`` to ``True`` reddens exactly this test.
    """
    is_proven = tenant_config._modality_router._is_proven
    is_tenant_proven = tenant_config._modality_router._is_tenant_proven

    class _NothingStamped:
        tenant_id = TENANT

    class _IdentityOnly:
        """A rollback shape: the older release's field, not the newer one."""

        tenant_id = TENANT
        identity_proven = True

    assert is_proven(_NothingStamped()) is False
    assert is_tenant_proven(_NothingStamped()) is False
    # The half it HAS must not be read as the half it lacks.
    assert is_proven(_IdentityOnly()) is True
    assert is_tenant_proven(_IdentityOnly()) is False
    # Nothing above is a withholding decision, so nothing may be counted.
    assert metrics.withheld == []
