"""The ``prompt-asset-bootstrap`` role: ``initial_create`` and NOTHING else.

The ``claude-asset-bootstrap`` Keycloak SA (ncn#1780) creates INITIAL prompt
assets. It carries ONLY the realm role ``prompt-asset-bootstrap`` — never
``service-internal``, which licenses and improvement-loop-service accept on role
alone. worker_management maps the new role on exactly one move.

These tests drive the REAL ``AuthenticationMiddleware`` and the REAL RS256
verification in ``src/auth/keycloak_roles.py`` (only the JWKS network fetch is
stubbed), on the tokenless trusted-network branch an in-cluster SA caller lands
on: peer ``127.0.0.1``, a bare ``X-Tenant-Id`` uuid, no licence. So the azp
allowlist and the role source are the production ones, not a patched ContextVar.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.tenant_constants import SYSTEM_TENANT_ID

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api import auth_middleware
from src.api import graph_sessions_correlation_api as correlation
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.prompts import repository
from src.api.prompts.routes import PROMPT_BOOTSTRAP_ROLE
from src.api.prompts.routes import router as prompts_router
from src.auth import keycloak_roles

_KID = "bootstrap-test-key"
_TOKEN_URL = "https://keycloak.invalid/realms/secure-apps/protocol/openid-connect/token"
_ISSUER = "https://auth.example.invalid/realms/secure-apps"
BOOTSTRAP_SA = "claude-asset-bootstrap"
# The tenant the tool's workers must land in (infer calls resolve there today).
TENANT = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
PUT_BODY = {"system": "You are a keyword picker.", "question": "Pick keywords"}
# A Planka card id (REALIGN R7: every release names its board card).
CARD = "1234567890123456789"


def _row(**overrides):
    row = {
        "prompt_id": "kw",
        "version": 1,
        "sha256": "0" * 64,
        "system_prompt_sha256": "0" * 64,
        "question": "Pick keywords",
        "system": "You are a keyword picker.",
        "template": None,
        "output_schema": None,
        "worker_names": [],
        "purpose_tags": [],
        "git_sha": None,
        "status": "DRAFT",
        "published_at": "2026-09-28T00:00:00+00:00",
        "activated_at": None,
        "config_revision": "kw@1",
    }
    row.update(overrides)
    return row


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

    monkeypatch.setenv("KEYCLOAK_TOKEN_URL", _TOKEN_URL)
    monkeypatch.delenv("KEYCLOAK_JWKS_URL", raising=False)
    monkeypatch.setenv("CALLER_ALLOWED_CLIENTS", f"improvement-loop-service,{BOOTSTRAP_SA}")
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(keycloak_roles, "_jwks_cache", None, raising=False)
    monkeypatch.setattr(keycloak_roles, "_jwks_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", AsyncMock(return_value={"keys": [pub]}))
    # The tokenless branch looks up byo_mode over HTTP; not under test here.
    monkeypatch.setattr(
        auth_middleware, "_lookup_tenant_byo_mode", AsyncMock(return_value="platform")
    )
    yield
    get_settings.cache_clear()


@pytest.fixture
def repo():
    with (
        patch.object(repository, "upsert_version", new_callable=AsyncMock) as upsert,
        patch.object(repository, "transition", new_callable=AsyncMock) as transition,
        patch.object(repository, "activate", new_callable=AsyncMock) as activate,
    ):
        upsert.return_value = (_row(), True)
        transition.return_value = (_row(status="PUBLISHED"), "DRAFT")
        activate.return_value = (_row(status="PUBLISHED"), None, True)
        yield {"upsert": upsert, "transition": transition, "activate": activate}


@pytest.fixture
def client(repo):
    # worker_management#336's fr card lookup is proved in
    # tests/test_prompt_release_card_verify.py; here it confirms every card.
    from src.api.prompts import release_card

    app = FastAPI()
    app.include_router(prompts_router)
    app.include_router(correlation.router)

    @app.get("/probe-tenant")
    async def probe(request: Request):
        return {"tenant_id": request.state.tenant_id}

    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    # A real peer inside TRUSTED_CIDRS: the in-cluster SA caller's branch.
    with patch.object(release_card, "verify_release_card", new_callable=AsyncMock):
        yield TestClient(app, client=("127.0.0.1", 0), raise_server_exceptions=False)


def _headers(token: str) -> dict[str, str]:
    return {"X-Caller-Authorization": f"Bearer {token}", "X-Tenant-Id": TENANT}


def _bootstrap(rsa_key, azp: str = BOOTSTRAP_SA) -> dict[str, str]:
    return _headers(_mint(rsa_key, azp=azp, roles=[PROMPT_BOOTSTRAP_ROLE]))


def _status(client, headers, status: str, reason: str | None = None, card: str | None = CARD):
    body = {"status": status} | ({"reason": reason} if reason else {})
    body |= {"actor_card_id": card} if card is not None else {}
    return client.post("/prompts/kw/versions/1/status", json=body, headers=headers)


def test_role_constant_is_the_keycloak_realm_role_name():
    assert PROMPT_BOOTSTRAP_ROLE == "prompt-asset-bootstrap"


def test_bootstrap_token_resolves_the_asserted_tenant_without_a_licence(client, rsa_key):
    """Flow (c)/(d): an SA token with no licence lands on the tokenless branch,
    which takes the tenant from the bare X-Tenant-Id uuid."""
    resp = client.get("/probe-tenant", headers=_bootstrap(rsa_key))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"tenant_id": TENANT}


def test_bootstrap_can_put_a_draft_and_initial_create_it(client, repo, rsa_key):
    headers = _bootstrap(rsa_key)
    put = client.put("/prompts/kw/versions/1", json=PUT_BODY, headers=headers)
    assert put.status_code == 200, put.text
    assert repo["upsert"].await_args.kwargs["actor"].sub == BOOTSTRAP_SA
    # 035: the bootstrap SA is a system writer — privileged, so it may write a
    # version of a prompt another tenant owns (it never changes the owner).
    assert repo["upsert"].await_args.kwargs["privileged"] is True
    # A system SA's new prompt is SYSTEM-owned, not owned by its incidental
    # licence tenant (the AA conveyor tenant): U6 ownership, migration 035.
    assert repo["upsert"].await_args.kwargs["owner_tenant_id"] == SYSTEM_TENANT_ID
    assert repo["upsert"].await_args.kwargs["owner_tenant_id"] != TENANT

    resp = _status(client, headers, "PUBLISHED", "initial_create")
    assert resp.status_code == 200, resp.text
    call = repo["transition"].await_args
    assert call.args == ("kw", 1, "PUBLISHED")
    assert call.kwargs["reason"] == "initial_create"
    assert call.kwargs["privileged"] is True
    # The ledger's actor_sub is the verified azp, i.e. the SA's own name.
    assert (call.kwargs["actor"].sub, call.kwargs["actor"].kind) == (
        BOOTSTRAP_SA,
        "service_account",
    )


def test_bootstrap_initial_create_without_a_card_is_422(client, repo, rsa_key):
    """REALIGN R7: NO exemption for the bootstrap SA — its initial_create is a
    release, so it names its board card like every other HTTP caller."""
    resp = _status(client, _bootstrap(rsa_key), "PUBLISHED", "initial_create", card=None)
    assert resp.status_code == 422, resp.text
    assert resp.json() == {"detail": {"error": "prompt_release_card_required"}}
    repo["transition"].assert_not_awaited()


def test_bootstrap_initial_create_passes_its_card_to_the_ledger(client, repo, rsa_key):
    resp = _status(client, _bootstrap(rsa_key), "PUBLISHED", "initial_create")
    assert resp.status_code == 200, resp.text
    assert repo["transition"].await_args.kwargs["actor_card_id"] == CARD


@pytest.mark.parametrize("reason", [None, "initial_create"])
def test_bootstrap_is_403_on_discard(client, repo, rsa_key, reason):
    resp = _status(client, _bootstrap(rsa_key), "DISCARDED", reason)
    assert resp.status_code == 403, resp.text
    assert "requires_role:service-internal" in resp.text
    assert PROMPT_BOOTSTRAP_ROLE not in resp.text
    repo["transition"].assert_not_awaited()


def test_bootstrap_is_403_on_activate(client, repo, rsa_key):
    resp = client.post("/prompts/kw/versions/1/activate", headers=_bootstrap(rsa_key))
    assert resp.status_code == 403, resp.text
    repo["activate"].assert_not_awaited()


def test_bootstrap_does_not_get_the_unscoped_correlation_read(client, rsa_key):
    """That route never 403s (it falls back to the caller's tenant scope); the
    privilege is the UNSCOPED cross-tenant read, which must not be taken."""
    with (
        patch.object(
            correlation.GraphExecutionSessionDB,
            "fetch_sessions_by_correlation_id",
            new_callable=AsyncMock,
            return_value=[],
        ) as scoped,
        patch.object(
            correlation.GraphExecutionSessionDB,
            "fetch_sessions_by_correlation_id_unscoped",
            new_callable=AsyncMock,
            return_value=[],
        ) as unscoped,
    ):
        client.get(
            "/api/graph/sessions/by-correlation/00000000-0000-4000-8000-000000000001",
            headers=_bootstrap(rsa_key),
        )
    unscoped.assert_not_awaited()
    scoped.assert_awaited_once()
    assert scoped.await_args.kwargs["tenant_id"] == TENANT


def test_bootstrap_token_with_a_non_allowlisted_azp_is_403(client, repo, rsa_key):
    resp = _status(
        client, _bootstrap(rsa_key, azp="some-other-client"), "PUBLISHED", "initial_create"
    )
    assert resp.status_code == 403, resp.text
    repo["transition"].assert_not_awaited()


def test_service_internal_still_initial_creates(client, repo, rsa_key):
    """Control: the existing admin path is unchanged."""
    headers = _headers(_mint(rsa_key, azp="improvement-loop-service", roles=["service-internal"]))
    assert _status(client, headers, "PUBLISHED", "initial_create").status_code == 200


@pytest.mark.parametrize(("target", "previous"), [("CANARY", "DRAFT"), ("PUBLISHED", "CANARY")])
def test_bootstrap_still_moves_into_canary_and_published(client, repo, rsa_key, target, previous):
    """U6 step 4 gates CANARY/PUBLISHED on a prompt role; the bootstrap SA's
    verified ``prompt-asset-bootstrap`` role (real RS256 + azp allowlist) passes."""
    repo["transition"].return_value = (_row(status=target), previous)
    resp = _status(client, _bootstrap(rsa_key), target)
    assert resp.status_code == 200, resp.text
    assert repo["transition"].await_args.args == ("kw", 1, target)


@pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
def test_a_platform_caller_without_an_sa_jwt_cannot_move_into_canary_or_published(
    client, repo, target
):
    """Same platform tenant, no verified SA role: the router gate admits it, the
    publish-role gate refuses it."""
    resp = _status(client, {"X-Tenant-Id": TENANT}, target)
    assert resp.status_code == 403, resp.text
    assert resp.json() == {"detail": {"error": "prompt_publish_role_required"}}
    repo["transition"].assert_not_awaited()
