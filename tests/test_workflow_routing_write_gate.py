"""worker_management#230: only a verified ``service-internal`` SA may repoint
the FLEET-WIDE workflow routing pointer.

Drives the REAL ``AuthenticationMiddleware`` and the REAL RS256 verification in
``src/auth/keycloak_roles.py`` (only the JWKS fetch and licence introspection
are stubbed), so the azp allowlist and the role source are production's, not a
patched ContextVar.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api import auth_middleware
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.workflow_routing import repository
from src.api.workflow_routing.routes import ROUTING_WRITER_ROLE
from src.api.workflow_routing.routes import router as routing_router
from src.auth import keycloak_roles
from tests._production_error_handlers import install_production_exception_handlers

_KID = "routing-gate-test-key"
_ISSUER = "https://auth.example.invalid/realms/secure-apps"
ALLOWED_SA = "improvement-loop-service"
TENANT = "bbbbbbbb-0000-4000-8000-0000000000a5"
WF = "Zz9Yy8Xx7Ww6Vv5U"  # a 16-char n8n workflow id (migration 036)
ASSET = "cell:READY"

ROW: dict[str, Any] = {
    "asset_key": ASSET,
    "active_workflow_id": WF,
    "prior_workflow_id": None,
    "promoted_at": "2026-09-29T00:00:00+00:00",
    "observation_window_until": "2026-09-29T00:00:00+00:00",
    "rollback_thresholds": {},
    "card_type": None,
    "list": None,
}
# repository.rollback additionally returns the pre-swap active (#338).
ROLLBACK_ROW: dict[str, Any] = {**ROW, "rolled_back_from": "Qq1Rr2Ss3Tt4Uu5V"}

WRITES: list[tuple[str, dict[str, Any]]] = [
    (
        "/api/workflow-routing/promote",
        {"asset_key": ASSET, "new_workflow_id": WF, "observation_window_minutes": 0},
    ),
    ("/api/workflow-routing/rollback", {"asset_key": ASSET}),
    ("/api/workflow-routing/extend", {"asset_key": ASSET, "additional_minutes": 5}),
    ("/api/workflow-routing/clear-prior", {"asset_key": ASSET}),
]
WRITE_IDS = [p.rsplit("/", 1)[1] for p, _ in WRITES]


@pytest.fixture
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _mint(key, *, azp: str, roles: list[str]) -> str:
    claims = {
        "sub": "svc",
        "azp": azp,
        "realm_access": {"roles": roles},
        "exp": int(time.time()) + 3600,
        "iss": _ISSUER,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": _KID})


@pytest.fixture(autouse=True)
def _env(monkeypatch, rsa_key):
    from src.settings import get_settings

    pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key()))
    pub.update(kid=_KID, alg="RS256", use="sig")
    monkeypatch.setenv(
        "KEYCLOAK_TOKEN_URL",
        "https://keycloak.invalid/realms/secure-apps/protocol/openid-connect/token",
    )
    monkeypatch.delenv("KEYCLOAK_JWKS_URL", raising=False)
    monkeypatch.setenv("CALLER_ALLOWED_CLIENTS", f"{ALLOWED_SA},claude-asset-bootstrap")
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT", raising=False)
    # Default ON for this file: these tests exercise the service-internal
    # write gate (worker_management#230), not the pointer-creation gate
    # (worker_management follow-up, 2026-09-30). The latter's own on/off
    # behaviour is covered by TestRoutingPointersEnabledGate below, which
    # overrides this per-test.
    monkeypatch.setenv("WORKFLOW_ROUTING_POINTERS_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(keycloak_roles, "_jwks_cache", None, raising=False)
    monkeypatch.setattr(keycloak_roles, "_jwks_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", AsyncMock(return_value={"keys": [pub]}))
    monkeypatch.setattr(
        auth_middleware, "_lookup_tenant_byo_mode", AsyncMock(return_value="platform")
    )
    monkeypatch.setattr(
        auth_middleware,
        "_introspect_license",
        AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT,
                "license_id": "lic-customer",
                "subject_id": "subj-customer",
                "byo_mode": "local_allowed",
            }
        ),
    )
    yield
    get_settings.cache_clear()


@pytest.fixture
def repo():
    with (
        patch.object(repository, "promote", new_callable=AsyncMock, return_value=ROW) as promote,
        patch.object(
            repository, "rollback", new_callable=AsyncMock, return_value=ROLLBACK_ROW
        ) as rb,
        patch.object(
            repository, "extend_observation_window", new_callable=AsyncMock, return_value=ROW
        ) as ext,
        patch.object(repository, "clear_prior", new_callable=AsyncMock, return_value=ROW) as cp,
        patch.object(
            repository, "get_active_by_asset_key", new_callable=AsyncMock, return_value=ROW
        ) as ga,
        patch.object(repository, "list_observed", new_callable=AsyncMock, return_value=[ROW]) as lo,
    ):
        yield {
            "promote": promote,
            "rollback": rb,
            "extend": ext,
            "clear-prior": cp,
            "active": ga,
            "observed": lo,
        }


def _app() -> FastAPI:
    # The production exception handlers are installed so every body asserted in
    # this file is the one a client actually receives (``{"error": {...}}``),
    # not FastAPI's default ``{"detail": ...}``.
    app = install_production_exception_handlers(FastAPI())
    app.include_router(routing_router)
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    return app


@pytest.fixture
def internet_client(repo):
    """An external peer: reaches the app only through the edge, with a licence."""
    return TestClient(_app(), client=("203.0.113.7", 0), raise_server_exceptions=False)


@pytest.fixture
def cluster_client(repo):
    """An in-cluster peer inside TRUSTED_CIDRS: the SA caller's branch."""
    return TestClient(_app(), client=("127.0.0.1", 0), raise_server_exceptions=False)


LICENCE = {"X-Tenant-Id": "Bearer lic-token-of-any-customer"}


def _sa(key, *, azp: str = ALLOWED_SA, roles: list[str] | None = None) -> dict[str, str]:
    token = _mint(key, azp=azp, roles=[ROUTING_WRITER_ROLE] if roles is None else roles)
    return {"X-Caller-Authorization": f"Bearer {token}", "X-Tenant-Id": TENANT}


def test_role_is_service_internal():
    assert ROUTING_WRITER_ROLE == "service-internal"


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_licence_holder_is_403_on_every_write(internet_client, repo, path, body):
    resp = internet_client.post(path, json=body, headers=LICENCE)
    assert resp.status_code == 403, resp.text
    assert "requires_role:service-internal" in resp.text
    for mock in repo.values():
        mock.assert_not_awaited()


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_tokenless_trusted_network_caller_is_403_on_every_write(cluster_client, repo, path, body):
    resp = cluster_client.post(path, json=body, headers={"X-Tenant-Id": TENANT})
    assert resp.status_code == 403, resp.text
    for mock in repo.values():
        mock.assert_not_awaited()


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_service_internal_sa_can_write(cluster_client, repo, rsa_key, path, body):
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    repo[path.rsplit("/", 1)[1]].assert_awaited_once()


def test_service_internal_sa_alongside_a_licence_can_write(internet_client, repo, rsa_key):
    headers = {**_sa(rsa_key), **LICENCE}
    path, body = WRITES[0]
    assert internet_client.post(path, json=body, headers=headers).status_code == 200


def test_prompt_asset_bootstrap_role_is_not_accepted(cluster_client, repo, rsa_key):
    path, body = WRITES[0]
    resp = cluster_client.post(
        path,
        json=body,
        headers=_sa(rsa_key, azp="claude-asset-bootstrap", roles=["prompt-asset-bootstrap"]),
    )
    assert resp.status_code == 403, resp.text
    repo["promote"].assert_not_awaited()


def test_service_internal_from_a_non_allowlisted_azp_is_403(cluster_client, repo, rsa_key):
    path, body = WRITES[0]
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, azp="some-other-client"))
    assert resp.status_code == 403, resp.text
    repo["promote"].assert_not_awaited()


def test_forged_x_caller_roles_header_is_not_a_role_source(cluster_client, repo):
    path, body = WRITES[0]
    resp = cluster_client.post(
        path,
        json=body,
        headers={"X-Tenant-Id": TENANT, "x-caller-roles": ROUTING_WRITER_ROLE},
    )
    assert resp.status_code == 403, resp.text


def test_reads_stay_open_to_a_platform_licence_holder(internet_client, repo):
    active = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
    )
    observed = internet_client.get("/api/workflow-routing/observed", headers=LICENCE)
    assert active.status_code == 200, active.text
    assert observed.status_code == 200, observed.text


# ---------------------------------------------------------------------------
# Pointer-creation gate (rollback-watch timer retired; see routes.py comment
# above ``_require_routing_pointers_enabled``). ONLY /promote creates a row,
# so ONLY /promote is gated — /rollback, /extend, /clear-prior must be
# unaffected regardless of the flag.
# ---------------------------------------------------------------------------

PROMOTE_PATH, PROMOTE_BODY = WRITES[0]
NON_PROMOTE_WRITES = WRITES[1:]
NON_PROMOTE_IDS = WRITE_IDS[1:]


class TestRoutingPointersEnabledGate:
    def test_promote_with_flag_unset_is_409_and_creates_no_row(
        self, monkeypatch, cluster_client, repo, rsa_key
    ):
        from src.settings import get_settings

        monkeypatch.delenv("WORKFLOW_ROUTING_POINTERS_ENABLED", raising=False)
        get_settings.cache_clear()
        resp = cluster_client.post(PROMOTE_PATH, json=PROMOTE_BODY, headers=_sa(rsa_key))
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["message"]["error"] == "routing_pointers_disabled"
        repo["promote"].assert_not_awaited()

    def test_promote_with_flag_true_succeeds(self, monkeypatch, cluster_client, repo, rsa_key):
        from src.settings import get_settings

        monkeypatch.setenv("WORKFLOW_ROUTING_POINTERS_ENABLED", "true")
        get_settings.cache_clear()
        resp = cluster_client.post(PROMOTE_PATH, json=PROMOTE_BODY, headers=_sa(rsa_key))
        assert resp.status_code == 200, resp.text
        repo["promote"].assert_awaited_once()

    def test_unauthorised_caller_still_gets_401_403_when_flag_unset(
        self, monkeypatch, internet_client, repo
    ):
        """Writer-role dependency runs FIRST: an unlicensed/unauthorised caller
        must never learn, via a 409, whether pointer creation is enabled."""
        from src.settings import get_settings

        monkeypatch.delenv("WORKFLOW_ROUTING_POINTERS_ENABLED", raising=False)
        get_settings.cache_clear()
        resp = internet_client.post(PROMOTE_PATH, json=PROMOTE_BODY, headers=LICENCE)
        assert resp.status_code == 403, resp.text
        assert "requires_role:service-internal" in resp.text
        repo["promote"].assert_not_awaited()

    @pytest.mark.parametrize(("path", "body"), NON_PROMOTE_WRITES, ids=NON_PROMOTE_IDS)
    def test_non_promote_writes_unaffected_by_flag_unset(
        self, monkeypatch, cluster_client, repo, rsa_key, path, body
    ):
        from src.settings import get_settings

        monkeypatch.delenv("WORKFLOW_ROUTING_POINTERS_ENABLED", raising=False)
        get_settings.cache_clear()
        resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
        assert resp.status_code == 200, resp.text
        repo[path.rsplit("/", 1)[1]].assert_awaited_once()
