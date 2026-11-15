"""Caller identity must come from a verified JWT, never from a request header.

These tests drive the REAL ASGI path — a request goes through
``AuthenticationMiddleware``, which calls the real
``verified_caller_identity``, which runs a real RS256 signature check. The
only thing stubbed is the network fetch of the JWKS document; the verification
itself is genuine.

That distinction is the point. The pre-existing tests in
``test_quality_score_and_correlation_endpoint.py`` call ``set_caller_context``
directly, so they exercise the endpoint's branch logic while proving nothing
about where the caller's roles came from — they stayed green throughout the
period the header was forgeable. A test of this fix that stubbed
``verified_caller_identity`` or ``set_caller_context`` would have the same
blind spot, and would stay green if someone reinstated a
``verified or header`` fallback.
"""

from __future__ import annotations

import json
import time
from urllib.parse import urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.identity_context import (
    get_caller_client_id_from_context,
    get_caller_roles_from_context,
)
from src.auth import keycloak_roles

_KID = "test-key-1"
_TOKEN_URL = "https://keycloak.invalid/realms/secure-apps/protocol/openid-connect/token"
_ALLOWED_CLIENT = "improvement-loop-service"


@pytest.fixture
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def jwks(rsa_key):
    """A JWKS document containing only our test signing key."""
    pub_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key()))
    pub_jwk["kid"] = _KID
    pub_jwk["alg"] = "RS256"
    pub_jwk["use"] = "sig"
    return {"keys": [pub_jwk]}


# Keycloak's `iss` is its FRONTEND url, which is NOT the host we fetch keys from.
# Measured live 2026-08-02 on a real improvement-loop-service SA token: the token
# said `https://auth.nonagentic.ai/realms/secure-apps` while WM's KEYCLOAK_JWKS_URL
# was `http://keycloak.keycloak.svc.cluster.local:8080/realms/secure-apps/...`.
# The test issuer deliberately uses a DIFFERENT host from _TOKEN_URL so that any
# future attempt to pin the full issuer URL fails here instead of in production.
_ISSUER = "https://auth.example.invalid/realms/secure-apps"


def _mint(
    rsa_key,
    *,
    azp: str,
    roles: list[str],
    kid: str = _KID,
    exp_delta: int | None = 3600,
    iss: str | None = _ISSUER,
) -> str:
    """Mint a test SA token. ``exp_delta=None`` omits `exp`; ``iss=None`` omits `iss`."""
    claims: dict = {"sub": "svc", "azp": azp, "realm_access": {"roles": roles}}
    if exp_delta is not None:
        claims["exp"] = int(time.time()) + exp_delta
    if iss is not None:
        claims["iss"] = iss
    return jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture(autouse=True)
def _caller_identity_env(monkeypatch, jwks):
    """Configure caller-identity and reset the module-level JWKS cache."""
    from src.settings import get_settings

    monkeypatch.setenv("KEYCLOAK_TOKEN_URL", _TOKEN_URL)
    monkeypatch.delenv("KEYCLOAK_JWKS_URL", raising=False)
    monkeypatch.delenv("CALLER_ALLOWED_CLIENTS", raising=False)
    get_settings.cache_clear()

    monkeypatch.setattr(keycloak_roles, "_jwks_cache", None, raising=False)
    monkeypatch.setattr(keycloak_roles, "_jwks_cache_ts", 0.0, raising=False)

    async def _fake_fetch(url: str) -> dict:
        assert url.endswith("/protocol/openid-connect/certs"), url
        return jwks

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", _fake_fetch)
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    """A minimal app carrying the REAL middleware.

    ``bypass_auth=True`` skips license introspection only. The caller-identity
    block runs before it, so the derivation under test is fully exercised.
    """
    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return {
            "roles": list(get_caller_roles_from_context()),
            "client_id": get_caller_client_id_from_context(),
        }

    app.add_middleware(AuthenticationMiddleware, bypass_auth=True)
    return TestClient(app)


# --------------------------------------------------------------------------
# The falsification test: this is the vulnerability, stated directly.
# --------------------------------------------------------------------------


def test_forged_caller_role_header_is_ignored(client):
    """A caller-supplied x-caller-roles header grants nothing.

    On main this returns ["service-internal"], which
    graph_sessions_correlation_api trades for an unscoped cross-tenant read.
    """
    resp = client.get(
        "/probe",
        headers={
            "x-caller-roles": "service-internal,license-admin",
            "x-caller-client-id": "innocent-service",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"roles": [], "client_id": ""}


def test_forged_client_id_cannot_frame_another_service(client):
    """The audit actor is never caller-chosen.

    graph_sessions_correlation_api logs caller_client_id as the actor of a
    cross-tenant read. While that came from a request header, an attacker
    could stamp an innocent service as the actor of their own breach.
    """
    resp = client.get("/probe", headers={"x-caller-client-id": "improvement-loop-service"})
    assert resp.json()["client_id"] == ""


# --------------------------------------------------------------------------
# The green half: the mechanism must actually work, or the ETL stays broken.
# --------------------------------------------------------------------------


def test_verified_sa_jwt_grants_roles(client, rsa_key):
    """An allowlisted SA presenting a signed JWT gets its roles.

    This is the path improvement-loop-service's exec_quality ETL needs. It
    fails on main, where nothing ever populated these roles for this service.
    """
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    assert resp.json() == {"roles": ["service-internal"], "client_id": _ALLOWED_CLIENT}


def test_roles_come_from_the_token_not_the_header(client, rsa_key):
    """A valid JWT does not let an attacker top up their roles via the header."""
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    resp = client.get(
        "/probe",
        headers={
            "x-caller-authorization": f"Bearer {token}",
            "x-caller-roles": "license-admin",
        },
    )
    assert resp.json()["roles"] == ["service-internal"]


# --------------------------------------------------------------------------
# False-positive controls.
# --------------------------------------------------------------------------


def test_verified_but_non_allowlisted_client_gets_nothing(client, rsa_key):
    """A human dashboard token must not become an internal service.

    `secure-apps` is not an SA-only realm — billing_service authenticates
    human dashboard users against it requiring `license-admin`, which is one
    of the three roles graph_sessions_correlation_api treats as
    internal-service. Without the azp allowlist, this request would win an
    unscoped cross-tenant read.
    """
    token = _mint(rsa_key, azp="billing-dashboard-user", roles=["license-admin"])
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    assert resp.json() == {"roles": [], "client_id": ""}


def _assert_caller_token_401(resp, failure_class: str, token: str = "") -> None:
    assert resp.status_code == 401, resp.text
    err = resp.json()["error"]
    assert err["code"] == "CALLER_TOKEN_INVALID"
    assert err["failure_class"] == failure_class
    assert failure_class in err["message"]
    if token:
        assert token not in resp.text, "the rejected token must never be echoed"


def test_unsigned_and_wrongly_signed_tokens_are_rejected(client, rsa_key):
    """Signature verification is real, not a decode — and a failure is a 401."""
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = _mint(other_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    _assert_caller_token_401(
        client.get("/probe", headers={"x-caller-authorization": f"Bearer {forged}"}),
        "InvalidSignatureError",
        forged,
    )

    garbage = client.get("/probe", headers={"x-caller-authorization": "Bearer not-a-jwt"})
    _assert_caller_token_401(garbage, "DecodeError", "not-a-jwt")


# --------------------------------------------------------------------------
# Change (B), 2026-09-30: a PRESENT caller token that fails validation is a
# 401. Absent header, JWKS unreachable and valid-but-not-allowlisted keep
# their fail-open behaviour. Context: worker_management#242 (NOT its fix).
# --------------------------------------------------------------------------


def test_forged_token_with_valid_kid_and_bad_signature_is_401(client, rsa_key):
    """A token claiming OUR kid but signed by another key: 401, not a silent downgrade."""
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = _mint(attacker_key, azp=_ALLOWED_CLIENT, roles=["service-internal"], kid=_KID)
    assert jwt.get_unverified_header(forged)["kid"] == _KID
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {forged}"})
    _assert_caller_token_401(resp, "InvalidSignatureError", forged)


def test_expired_caller_token_is_401(client, rsa_key):
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"], exp_delta=-60)
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    _assert_caller_token_401(resp, "ExpiredSignatureError", token)


def test_valid_token_from_non_allowlisted_marketing_client_is_not_401(client, rsa_key):
    """marketing-service-account sends a VALID token ~4x/day; it must keep being served."""
    token = _mint(rsa_key, azp="marketing-service-account", roles=["uma_authorization"])
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"roles": [], "client_id": ""}


def test_absent_caller_header_is_unchanged(client):
    resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json() == {"roles": [], "client_id": ""}
    # An empty header value is treated as absent, not as a malformed token.
    empty = client.get("/probe", headers={"X-Caller-Authorization": ""})
    assert empty.status_code == 200
    assert empty.json() == {"roles": [], "client_id": ""}


def test_non_bearer_caller_header_is_401_malformed(client):
    resp = client.get("/probe", headers={"X-Caller-Authorization": "Basic Zm9vOmJhcg=="})
    _assert_caller_token_401(resp, "MalformedHeader", "Zm9vOmJhcg==")


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: httpx.ConnectError("keycloak down"),
        lambda: jwt.PyJWKClientConnectionError("keycloak down"),
        lambda: httpx.HTTPStatusError(
            "503", request=httpx.Request("GET", "http://k"), response=httpx.Response(503)
        ),
    ],
    ids=["ConnectError", "PyJWKClientConnectionError", "HTTP503"],
)
def test_jwks_unreachable_is_not_401(client, rsa_key, monkeypatch, exc_factory):
    """Keycloak down is an outage, not a verdict on the token: serve, grant nothing."""

    async def _boom(url: str) -> dict:
        raise exc_factory()

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", _boom)
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"roles": [], "client_id": ""}


def test_jwks_with_no_usable_keys_is_not_401(client, rsa_key, monkeypatch):
    """A broken key set from Keycloak is the key source failing, not the token."""

    async def _empty(url: str) -> dict:
        return {"keys": []}

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", _empty)
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text


def test_unknown_kid_triggers_one_refresh_then_denies(client, rsa_key, monkeypatch):
    """A kid absent even after one refetch is an invalid token (401), not an outage."""
    calls = {"n": 0}
    original = keycloak_roles._fetch_jwks

    async def _counting(url: str) -> dict:
        calls["n"] += 1
        return await original(url)

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", _counting)
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"], kid="rotated-away")
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    _assert_caller_token_401(resp, "UnknownKidError", token)
    assert calls["n"] == 2, "should refetch once on an unknown kid before giving up"


def test_jwks_unreachable_denies_privilege_but_serves_the_request(client, rsa_key, monkeypatch):
    """Fail closed on the privilege, open on the request.

    A 503 here would take down every public /workers/ call the moment Keycloak
    blinked — strictly worse than the ETL skipping an enrichment pass.
    """

    async def _boom(url: str) -> dict:
        raise httpx.ConnectError("keycloak down")

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", _boom)
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"])
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["roles"] == []


def test_no_caller_header_at_all_is_the_ordinary_case(client):
    """Every existing caller sends nothing here and must be unaffected."""
    resp = client.get("/probe")
    assert resp.json() == {"roles": [], "client_id": ""}


def test_token_without_exp_is_rejected(client, rsa_key):
    """A token minted without `exp` would never expire.

    PyJWT only checks exp when the claim is PRESENT, so verify_exp alone does
    not cover this — the verifier requires the claim.
    """
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"], exp_delta=None)
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    _assert_caller_token_401(resp, "MissingRequiredClaimError", token)


def test_expired_token_is_rejected(client, rsa_key):
    """An expired but correctly signed SA token is refused with a 401."""
    token = _mint(rsa_key, azp=_ALLOWED_CLIENT, roles=["service-internal"], exp_delta=-60)
    resp = client.get("/probe", headers={"x-caller-authorization": f"Bearer {token}"})
    _assert_caller_token_401(resp, "ExpiredSignatureError", token)


# --------------------------------------------------------------------------
# Realm pinning (added 2026-08-02). The `azp` allowlist stays the primary
# control; these close the "same Keycloak, different realm" case.
# --------------------------------------------------------------------------


def test_token_from_another_realm_on_the_same_keycloak_is_rejected(client, rsa_key):
    """`secure-apps` is the configured realm; a `portal` token must not grant roles.

    `portal` is the realm real customers authenticate against, so a portal token
    reaching the service-internal path would be a privilege escalation from an
    ordinary end user, not merely a misconfiguration.
    """
    tok = _mint(
        rsa_key,
        azp="improvement-loop-service",
        roles=["service-internal"],
        iss="https://auth.example.invalid/realms/portal",
    )
    r = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {tok}"})
    _assert_caller_token_401(r, "InvalidIssuerError", tok)


def test_token_without_iss_is_rejected(client, rsa_key):
    """Without `require: ["iss"]` the realm check above would simply not run."""
    tok = _mint(rsa_key, azp="improvement-loop-service", roles=["service-internal"], iss=None)
    r = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {tok}"})
    _assert_caller_token_401(r, "MissingRequiredClaimError", tok)


def test_issuer_host_may_differ_from_the_jwks_host(client, rsa_key):
    """⚠️ Non-vacuity control for the two tests above — and a production regression guard.

    Keycloak's `iss` is its frontend url; the JWKS URL is the in-cluster Service. They
    do NOT match in production. If the two rejections above were achieved by pinning the
    full issuer, this test goes red — which is exactly what would have happened to every
    genuine SA token in prod.
    """
    assert urlsplit(_ISSUER).netloc != urlsplit(_TOKEN_URL).netloc
    tok = _mint(rsa_key, azp="improvement-loop-service", roles=["service-internal"])
    r = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {tok}"})
    assert r.json()["client_id"] == "improvement-loop-service"
    assert "service-internal" in r.json()["roles"]


def test_expected_realm_is_derived_from_the_jwks_url():
    """The realm is read out of the configured URL, not configured separately."""
    assert (
        keycloak_roles._expected_realm(
            "http://keycloak.keycloak.svc.cluster.local:8080"
            "/realms/secure-apps/protocol/openid-connect/certs"
        )
        == "secure-apps"
    )
    # No /realms/ segment -> no realm -> the check is skipped rather than crashing.
    assert keycloak_roles._expected_realm("https://example.invalid/certs") == ""


def test_small_clock_skew_on_fresh_token_is_not_401(client, rsa_key):
    """Keycloak's clock a few seconds ahead (iat/nbf in the future) must not 401 a fresh token."""
    now = int(time.time())
    claims = {
        "sub": "svc",
        "azp": _ALLOWED_CLIENT,
        "realm_access": {"roles": ["service-internal"]},
        "iss": _ISSUER,
        "iat": now + 5,
        "nbf": now + 5,
        "exp": now + 3600,
    }
    token = jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": _KID})
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text


def test_valid_token_without_azp_is_not_401(client, rsa_key):
    """A correctly signed token that carries no azp keeps today's behaviour (empty identity)."""
    claims = {
        "sub": "svc",
        "realm_access": {"roles": ["service-internal"]},
        "iss": _ISSUER,
        "exp": int(time.time()) + 3600,
    }
    token = jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": _KID})
    resp = client.get("/probe", headers={"X-Caller-Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"roles": [], "client_id": ""}
