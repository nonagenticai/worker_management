"""End-user proof accepts a token from ANY configured realm (2026-11-06).

WHY THIS FILE EXISTS
--------------------
Realm-per-app (G1, 2026-10-22) moved fr's USERS from realm ``portal`` to realm
``fr``. The verifier pinned a single realm (``END_USER_REALM = "portal"``), so
every fr → worker_management call carrying ``X-End-User-Token`` failed with
``invalid:UnknownKidError`` — measured live 2026-11-06: fr chat
``GET /api/assets/workers/providers`` → WM ``GET /providers`` 401 → Cloudflare
502, for EVERY tenant, not one.

The fix must not weaken the check: a token verifies only against the JWKS of the
realm it was minted in, and its ``iss`` must equal that realm's issuer exactly.
These tests pin both halves — the accepted set, and that a token from one realm
cannot satisfy another.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.auth import end_user_token as eut
from src.settings import get_settings

ISSUER_ORIGIN = "https://auth.nonagentic.ai"
SECURE_APPS_JWKS = (
    "http://keycloak.keycloak.svc.cluster.local:8080"
    "/realms/secure-apps/protocol/openid-connect/certs"
)


def _reset(monkeypatch: pytest.MonkeyPatch, realms: str | None = None) -> None:
    monkeypatch.setenv("KEYCLOAK_JWKS_URL", SECURE_APPS_JWKS)
    monkeypatch.delenv("END_USER_JWKS_URL", raising=False)
    monkeypatch.delenv("END_USER_TOKEN_ISSUER", raising=False)
    if realms is None:
        monkeypatch.delenv("END_USER_TOKEN_REALMS", raising=False)
    else:
        monkeypatch.setenv("END_USER_TOKEN_REALMS", realms)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_accepts_portal_and_fr(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    sources = eut.end_user_sources()
    assert [s[0] for s in sources] == ["portal", "fr"]
    assert sources[0][1] == f"{ISSUER_ORIGIN}/realms/portal"
    assert sources[1][1] == f"{ISSUER_ORIGIN}/realms/fr"
    # Each realm gets ITS OWN key set — never the other realm's.
    assert sources[0][2].endswith("/realms/portal/protocol/openid-connect/certs")
    assert sources[1][2].endswith("/realms/fr/protocol/openid-connect/certs")


def test_single_realm_config_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch, realms="portal")
    assert [s[0] for s in eut.end_user_sources()] == ["portal"]


def test_empty_realms_setting_falls_back_to_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misconfiguration must never widen the trust set.
    _reset(monkeypatch, realms="")
    assert [s[0] for s in eut.end_user_sources()] == ["portal"]


def _keypair(label: str) -> tuple[object, dict]:
    """A fresh RSA keypair whose ``kid`` is DISTINCT per key.

    Keycloak mints a random ``kid`` per realm key, so a realm-``fr`` token's kid
    is ABSENT from the portal key set and the verifier raises
    ``_UnknownKidError`` — the exact failure measured live (``invalid:
    UnknownKidError``). Deriving the kid from the JWK dict's length gave every
    key ``kid-3``, so the negative cases below failed on SIGNATURE and never
    exercised that path.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk["kid"] = f"kid-{label}"
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return key, jwk


def _token(key: object, kid: str, realm: str) -> str:
    now = int(time.time())
    claims = {
        "iss": f"{ISSUER_ORIGIN}/realms/{realm}",
        "sub": "a6d29e5c-0000-0000-0000-000000000000",
        "azp": "fr-frontend",
        "typ": "Bearer",
        "tenant_id": "aaaaaaaa-0000-4000-8000-000000000003",
        "iat": now,
        "exp": now + 300,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


@pytest.mark.asyncio
async def test_fr_token_now_verifies_and_foreign_token_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset(monkeypatch)
    portal_key, portal_jwk = _keypair("portal")
    fr_key, fr_jwk = _keypair("fr")
    stranger_key, stranger_jwk = _keypair("stranger")

    key_sets = {
        f"{ISSUER_ORIGIN}/realms/portal": {"keys": [portal_jwk]},
        f"{ISSUER_ORIGIN}/realms/fr": {"keys": [fr_jwk]},
    }

    async def fake_get_jwks(url: str, *, force_refresh: bool = False) -> dict:
        if "/realms/portal/" in url:
            return {"keys": [portal_jwk]}
        if "/realms/fr/" in url:
            return {"keys": [fr_jwk]}
        return {"keys": []}

    assert key_sets  # the issuer prefixes above are what _verify pins
    monkeypatch.setattr(eut, "_get_jwks", fake_get_jwks)

    fr_payload, fr_reason = await eut._verified_payload(
        "Bearer " + _token(fr_key, str(fr_jwk["kid"]), "fr")
    )
    assert fr_reason == ""
    assert fr_payload is not None
    assert fr_payload["iss"].endswith("/realms/fr")

    portal_payload, portal_reason = await eut._verified_payload(
        "Bearer " + _token(portal_key, str(portal_jwk["kid"]), "portal")
    )
    assert portal_reason == ""
    assert portal_payload is not None

    # A key neither realm publishes must still be refused.
    _, bad_reason = await eut._verified_payload(
        "Bearer " + _token(stranger_key, str(stranger_jwk["kid"]), "fr")
    )
    assert bad_reason == "invalid:UnknownKidError", bad_reason

    # MUTATION PROOF: with only realm `portal` trusted (the pre-fix behaviour),
    # the very same fr token is refused — with the kid-not-found failure that
    # production actually raised, not a signature mismatch.
    _reset(monkeypatch, realms="portal")
    _, pinned_reason = await eut._verified_payload(
        "Bearer " + _token(fr_key, str(fr_jwk["kid"]), "fr")
    )
    assert pinned_reason == "invalid:UnknownKidError", pinned_reason
