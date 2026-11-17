"""G-15: the workflow RELEASE writes and their gates (DESIGN D4, phase-10 REVIEW).

/seed-active, /release, /promote-canary and /discard-canary each need BOTH the
verified ``service-internal`` SA role AND a caller tenant in
``RELEASE_TENANT_IDS`` (…0001 system, …0006 kata — the successor to the v3
workflow-improvement stream …0002 that fr deleted 2026-11-03) —
narrower than the platform set. Every refusal happens BEFORE the repository is
reached. phase-10 asks for a test and a mutation control PER GATE, so each
refusal test below is parametrized over the four routes: removing
``Depends(_require_release_tenant)`` from ONE route's decorator reds exactly
that route's ``test_customer_tenant_with_the_role_is_403`` /
``test_platform_tenant_outside_the_release_set_is_403`` ids.

Drives the REAL ``AuthenticationMiddleware`` and RS256 verification (only the
JWKS fetch and licence introspection are stubbed), as
``test_workflow_routing_write_gate.py`` does, so the role source, the tenant on
the SA path and the captured ``X-AA-Trace-Id`` are production's.
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
from src.api.tenant_constants import platform_tenant_ids, release_tenant_ids
from src.api.workflow_routing import repository
from src.api.workflow_routing.routes import (
    ROUTING_WRITER_ROLE,
    ROUTING_WRITER_SCOPED_ROLE,
)
from src.api.workflow_routing.routes import router as routing_router
from src.auth import keycloak_roles
from tests._production_error_handlers import install_production_exception_handlers, wire_error

_KID = "routing-release-test-key"
_ISSUER = "https://auth.example.invalid/realms/secure-apps"
ALLOWED_SA = "improvement-loop-service"

SYSTEM = "aaaaaaaa-0000-4000-8000-000000000001"
IMPROVE = "aaaaaaaa-0000-4000-8000-000000000006"  # kata, successor to …0002
EXECUTION = "aaaaaaaa-0000-4000-8000-000000000003"  # platform, NOT a release tenant
CONVEYOR = "bbbbbbbb-0000-4000-8000-0000000000a5"  # platform, NOT a release tenant
CUSTOMER = "aaaaaaaa-0000-4000-8000-00000000c001"

NAME = "AA Activity - Reasoning"
ASSET = f"activity:{NAME}"
WF_A = "AbCdEfGh12345678"
WF_B = "Zz9Yy8Xx7Ww6Vv5U"
TRACE = "trace-g15-release-test"

ROW: dict[str, Any] = {
    "asset_key": ASSET,
    "card_type": None,
    "list": None,
    "active_workflow_id": WF_A,
    "prior_workflow_id": None,
    "promoted_at": "2026-10-01T00:00:00+00:00",
    "observation_window_until": None,
    "rollback_thresholds": {},
    "canary_workflow_id": WF_B,
    "canary_pct": 10,
    "canary_id": "cn-1",
    "released_by_card_id": "card-1",
    "released_by_tenant_id": IMPROVE,
    "trace_id": TRACE,
    "promoted_trace_id": None,
    "released_at": "2026-10-01T00:00:00+00:00",
}

SEED = ("/api/workflow-routing/seed-active", {"asset_key": ASSET, "active_workflow_id": WF_A})
RELEASE = (
    "/api/workflow-routing/release",
    {
        "asset_key": ASSET,
        "canary_workflow_id": WF_B,
        "canary_pct": 10,
        "canary_id": "cn-1",
        "released_by_card_id": "card-1",
    },
)
PROMOTE = (
    "/api/workflow-routing/promote-canary",
    {"asset_key": ASSET, "canary_id": "cn-1", "card_id": "card-1"},
)
DISCARD = (
    "/api/workflow-routing/discard-canary",
    {"asset_key": ASSET, "canary_id": "cn-1", "card_id": "card-1"},
)
WRITES = [SEED, RELEASE, PROMOTE, DISCARD]
WRITE_IDS = [p.rsplit("/", 1)[1] for p, _ in WRITES]
# The three writes that also check the RELEASABLE SET (discard does not).
ALLOWLISTED = [SEED, RELEASE, PROMOTE]
ALLOWLISTED_IDS = [p.rsplit("/", 1)[1] for p, _ in ALLOWLISTED]
TRACED = [SEED, RELEASE]
TRACED_IDS = [p.rsplit("/", 1)[1] for p, _ in TRACED]

REPO_FN = {
    "seed-active": "seed_active",
    "release": "release",
    "promote-canary": "promote_canary",
    "discard-canary": "discard_canary",
}


def _name(path: str) -> str:
    return path.rsplit("/", 1)[1]


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
    monkeypatch.setenv("CALLER_ALLOWED_CLIENTS", ALLOWED_SA)
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT", raising=False)
    monkeypatch.setenv("WM_WF_RELEASE_ALLOWLIST", json.dumps([NAME]))
    # Default ON for this file: these tests exercise the release gates, not the
    # pointer-creation flag (George 2026-09-30). That flag's own on/off
    # behaviour on /seed-active and /release is covered by the
    # "pointer-creation flag" section below, which overrides this per test.
    monkeypatch.setenv("WORKFLOW_ROUTING_POINTERS_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(keycloak_roles, "_jwks_cache", None, raising=False)
    monkeypatch.setattr(keycloak_roles, "_jwks_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", AsyncMock(return_value={"keys": [pub]}))
    monkeypatch.setattr(
        auth_middleware, "_lookup_tenant_byo_mode", AsyncMock(return_value="platform")
    )
    # The licence belongs to a RELEASE tenant on purpose: the licence-only test
    # then proves the ROLE half refuses on its own, with the tenant half passing.
    monkeypatch.setattr(
        auth_middleware,
        "_introspect_license",
        AsyncMock(
            return_value={
                "active": True,
                "tenant_id": IMPROVE,
                "license_id": "lic-improve",
                "subject_id": "subj-improve",
                "byo_mode": "local_allowed",
            }
        ),
    )
    yield
    get_settings.cache_clear()


def _set_allowlist(monkeypatch, value: str | None) -> None:
    from src.settings import get_settings

    if value is None:
        monkeypatch.delenv("WM_WF_RELEASE_ALLOWLIST", raising=False)
    else:
        monkeypatch.setenv("WM_WF_RELEASE_ALLOWLIST", value)
    get_settings.cache_clear()


@pytest.fixture
def repo():
    ok = (repository.OK, ROW)
    with (
        patch.object(repository, "seed_active", new_callable=AsyncMock, return_value=ok) as seed,
        patch.object(repository, "release", new_callable=AsyncMock, return_value=ok) as rel,
        patch.object(repository, "promote_canary", new_callable=AsyncMock, return_value=ok) as pc,
        patch.object(repository, "discard_canary", new_callable=AsyncMock, return_value=ok) as dc,
        patch.object(
            repository, "get_active_by_asset_key", new_callable=AsyncMock, return_value=ROW
        ) as ga,
        patch.object(repository, "promote", new_callable=AsyncMock, return_value=ROW) as legacy,
    ):
        yield {
            "seed-active": seed,
            "release": rel,
            "promote-canary": pc,
            "discard-canary": dc,
            "active": ga,
            "promote": legacy,
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


LICENCE = {"X-Tenant-Id": "Bearer lic-token-of-the-improvement-stream"}


def _sa(
    key, *, tenant: str = IMPROVE, trace: str | None = TRACE, roles: list[str] | None = None
) -> dict[str, str]:
    token = _mint(key, azp=ALLOWED_SA, roles=[ROUTING_WRITER_ROLE] if roles is None else roles)
    headers = {"X-Caller-Authorization": f"Bearer {token}", "X-Tenant-Id": tenant}
    if trace is not None:
        headers["X-AA-Trace-Id"] = trace
    return headers


def _assert_untouched(repo) -> None:
    for mock in repo.values():
        mock.assert_not_awaited()


# The bodies as a client receives them, through production's http_exception_handler.
FORBIDDEN = wire_error(403, {"error": "platform_routing_forbidden"}, "FORBIDDEN")
NOT_RELEASABLE = wire_error(403, {"error": "asset_not_releasable"}, "FORBIDDEN")


# ---------------------------------------------------------------------------
# The release tenant set
# ---------------------------------------------------------------------------


def test_release_tenants_are_exactly_system_and_the_improvement_stream():
    assert {SYSTEM, IMPROVE} == release_tenant_ids()
    assert release_tenant_ids() <= platform_tenant_ids()


# ---------------------------------------------------------------------------
# Gate 1 — the service-internal role (reused #230 gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_licence_only_is_403_on_every_release_write(internet_client, repo, path, body):
    """A licence of a RELEASE tenant, with a trace, but no SA role → 403."""
    resp = internet_client.post(path, json=body, headers={**LICENCE, "X-AA-Trace-Id": TRACE})
    assert resp.status_code == 403, resp.text
    assert "requires_role:service-internal" in resp.text
    _assert_untouched(repo)


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_tokenless_release_tenant_is_403_on_every_release_write(cluster_client, repo, path, body):
    """The role half refuses on its own: release tenant asserted, no SA JWT."""
    resp = cluster_client.post(
        path, json=body, headers={"X-Tenant-Id": IMPROVE, "X-AA-Trace-Id": TRACE}
    )
    assert resp.status_code == 403, resp.text
    assert "requires_role:service-internal" in resp.text
    _assert_untouched(repo)


# ---------------------------------------------------------------------------
# Gate 2 — the release tenant. THE per-route mutation controls.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_customer_tenant_with_the_role_is_403(cluster_client, repo, rsa_key, path, body):
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=CUSTOMER))
    assert resp.status_code == 403, resp.text
    assert resp.json() == FORBIDDEN
    _assert_untouched(repo)


@pytest.mark.parametrize("tenant", [EXECUTION, CONVEYOR], ids=["0003", "conveyor"])
@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_platform_tenant_outside_the_release_set_is_403(
    cluster_client, repo, rsa_key, path, body, tenant
):
    """Narrower than platform (phase-10 Q3): …0003 is platform and still refused."""
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=tenant))
    assert resp.status_code == 403, resp.text
    assert resp.json() == FORBIDDEN
    _assert_untouched(repo)


@pytest.mark.parametrize("tenant", [IMPROVE, SYSTEM], ids=["0002", "0001"])
@pytest.mark.parametrize(("path", "body"), WRITES, ids=WRITE_IDS)
def test_release_tenant_with_the_role_succeeds(cluster_client, repo, rsa_key, path, body, tenant):
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=tenant))
    assert resp.status_code == 200, resp.text
    repo[_name(path)].assert_awaited_once()


def test_an_uppercase_release_tenant_is_accepted(cluster_client, repo, rsa_key):
    path, body = RELEASE
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=IMPROVE.upper()))
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Gate 3 — the pointer-creation flag (WORKFLOW_ROUTING_POINTERS_ENABLED,
# George 2026-09-30). /seed-active creates a row and /release arms a canary:
# both CREATE pointer state, so both are gated. /promote-canary and
# /discard-canary only mutate an existing row and are not: discard must stay
# available to disarm. The flag runs after both auth gates.
# ---------------------------------------------------------------------------

CREATING = [SEED, RELEASE]
CREATING_IDS = [_name(p) for p, _ in CREATING]
MUTATING = [PROMOTE, DISCARD]
MUTATING_IDS = [_name(p) for p, _ in MUTATING]
POINTERS_DISABLED = "routing_pointers_disabled"


def _set_pointers_flag(monkeypatch, value: str | None) -> None:
    from src.settings import get_settings

    if value is None:
        monkeypatch.delenv("WORKFLOW_ROUTING_POINTERS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("WORKFLOW_ROUTING_POINTERS_ENABLED", value)
    get_settings.cache_clear()


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_flag_unset_refuses_pointer_creation_with_409(
    monkeypatch, cluster_client, repo, rsa_key, path, body
):
    _set_pointers_flag(monkeypatch, None)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["message"]["error"] == POINTERS_DISABLED
    _assert_untouched(repo)


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_flag_unset_licence_only_caller_still_gets_403_not_409(
    monkeypatch, internet_client, repo, path, body
):
    """The role check runs first: the flag state is never disclosed to it."""
    _set_pointers_flag(monkeypatch, None)
    resp = internet_client.post(path, json=body, headers={**LICENCE, "X-AA-Trace-Id": TRACE})
    assert resp.status_code == 403, resp.text
    assert "requires_role:service-internal" in resp.text
    assert POINTERS_DISABLED not in resp.text
    _assert_untouched(repo)


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_flag_unset_tokenless_caller_still_gets_403_not_409(
    monkeypatch, cluster_client, repo, path, body
):
    _set_pointers_flag(monkeypatch, None)
    resp = cluster_client.post(
        path, json=body, headers={"X-Tenant-Id": IMPROVE, "X-AA-Trace-Id": TRACE}
    )
    assert resp.status_code == 403, resp.text
    assert POINTERS_DISABLED not in resp.text
    _assert_untouched(repo)


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_flag_unset_customer_tenant_still_gets_403_not_409(
    monkeypatch, cluster_client, repo, rsa_key, path, body
):
    """The release-tenant check also runs before the flag."""
    _set_pointers_flag(monkeypatch, None)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=CUSTOMER))
    assert resp.status_code == 403, resp.text
    assert resp.json() == FORBIDDEN
    _assert_untouched(repo)


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_flag_true_allows_pointer_creation(monkeypatch, cluster_client, repo, rsa_key, path, body):
    _set_pointers_flag(monkeypatch, "true")
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    repo[_name(path)].assert_awaited_once()


@pytest.mark.parametrize(("path", "body"), MUTATING, ids=MUTATING_IDS)
def test_flag_unset_does_not_gate_promote_or_discard_canary(
    monkeypatch, cluster_client, repo, rsa_key, path, body
):
    _set_pointers_flag(monkeypatch, None)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    repo[_name(path)].assert_awaited_once()


# ---------------------------------------------------------------------------
# Trace: decision 4 (release) and Q6 (seed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace", [None, "   "], ids=["absent", "blank"])
@pytest.mark.parametrize(("path", "body"), TRACED, ids=TRACED_IDS)
def test_missing_trace_is_400(cluster_client, repo, rsa_key, path, body, trace):
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, trace=trace))
    assert resp.status_code == 400, resp.text
    assert resp.json() == wire_error(400, {"error": "trace_id_required"}, "BAD_REQUEST")
    _assert_untouched(repo)


def test_release_stamps_trace_tenant_and_card(cluster_client, repo, rsa_key):
    path, body = RELEASE
    assert cluster_client.post(path, json=body, headers=_sa(rsa_key)).status_code == 200
    kwargs = repo["release"].await_args.kwargs
    assert kwargs == {
        "asset_key": ASSET,
        "canary_workflow_id": WF_B,
        "canary_pct": 10,
        "canary_id": "cn-1",
        "released_by_card_id": "card-1",
        "tenant_id": IMPROVE,
        "trace_id": TRACE,
    }


def test_seed_stamps_trace_and_tenant(cluster_client, repo, rsa_key):
    path, body = SEED
    assert cluster_client.post(path, json=body, headers=_sa(rsa_key)).status_code == 200
    kwargs = repo["seed-active"].await_args.kwargs
    assert kwargs == {
        "asset_key": ASSET,
        "active_workflow_id": WF_A,
        "tenant_id": IMPROVE,
        "trace_id": TRACE,
    }


def test_promote_canary_does_not_require_a_trace_but_stamps_one(cluster_client, repo, rsa_key):
    path, body = PROMOTE
    assert cluster_client.post(path, json=body, headers=_sa(rsa_key, trace=None)).status_code == 200
    assert repo["promote-canary"].await_args.kwargs["trace_id"] is None
    assert cluster_client.post(path, json=body, headers=_sa(rsa_key)).status_code == 200
    assert repo["promote-canary"].await_args.kwargs["trace_id"] == TRACE


# ---------------------------------------------------------------------------
# The releasable set (WM_WF_RELEASE_ALLOWLIST) — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "asset_key",
    [
        "activity:AA Activity - Not In The Set",
        f"pattern:{NAME}",  # listed name, wrong map
        f"cell:{NAME}",
    ],
)
@pytest.mark.parametrize(("path", "body"), ALLOWLISTED, ids=ALLOWLISTED_IDS)
def test_a_non_releasable_asset_is_403(cluster_client, repo, rsa_key, path, body, asset_key):
    resp = cluster_client.post(path, json={**body, "asset_key": asset_key}, headers=_sa(rsa_key))
    assert resp.status_code == 403, resp.text
    assert resp.json() == NOT_RELEASABLE
    _assert_untouched(repo)


@pytest.mark.parametrize(
    "allowlist",
    [None, "", "[]", "not json", json.dumps({NAME: True}), json.dumps([NAME, 3])],
    ids=["unset", "empty", "empty-list", "invalid-json", "not-a-list", "non-string-member"],
)
@pytest.mark.parametrize(("path", "body"), ALLOWLISTED, ids=ALLOWLISTED_IDS)
def test_an_unset_or_broken_allowlist_refuses_everything(
    monkeypatch, cluster_client, repo, rsa_key, path, body, allowlist
):
    _set_allowlist(monkeypatch, allowlist)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 403, resp.text
    assert resp.json() == NOT_RELEASABLE
    _assert_untouched(repo)


def test_discard_is_not_allowlisted_so_a_dropped_asset_can_still_be_disarmed(
    monkeypatch, cluster_client, repo, rsa_key
):
    _set_allowlist(monkeypatch, None)
    path, body = DISCARD
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    repo["discard-canary"].assert_awaited_once()


# ---------------------------------------------------------------------------
# Body validation: 16-char n8n ids, asset_key shape, pct 1..50
# ---------------------------------------------------------------------------

BAD_IDS = ["A" * 15, "A" * 17, "aaaaaaaa-0000-4000-8000-0000000000ff", "abcdefgh-jklmnop"]
BAD_ID_IDS = ["15-chars", "17-chars", "uuid", "non-alnum"]


@pytest.mark.parametrize("bad", BAD_IDS, ids=BAD_ID_IDS)
def test_seed_rejects_a_non_16_char_id(cluster_client, repo, rsa_key, bad):
    path, body = SEED
    resp = cluster_client.post(path, json={**body, "active_workflow_id": bad}, headers=_sa(rsa_key))
    assert resp.status_code == 422, resp.text
    _assert_untouched(repo)


@pytest.mark.parametrize("bad", BAD_IDS, ids=BAD_ID_IDS)
def test_release_rejects_a_non_16_char_id(cluster_client, repo, rsa_key, bad):
    path, body = RELEASE
    resp = cluster_client.post(path, json={**body, "canary_workflow_id": bad}, headers=_sa(rsa_key))
    assert resp.status_code == 422, resp.text
    _assert_untouched(repo)


@pytest.mark.parametrize(("pct", "status"), [(0, 422), (1, 200), (50, 200), (51, 422)])
def test_release_pct_is_bounded_1_to_50(cluster_client, repo, rsa_key, pct, status):
    path, body = RELEASE
    resp = cluster_client.post(path, json={**body, "canary_pct": pct}, headers=_sa(rsa_key))
    assert resp.status_code == status, resp.text


@pytest.mark.parametrize("bad_key", ["AA Activity - Reasoning", "activity:", "other:x"])
def test_a_malformed_asset_key_is_422(cluster_client, repo, rsa_key, bad_key):
    path, body = DISCARD
    resp = cluster_client.post(path, json={**body, "asset_key": bad_key}, headers=_sa(rsa_key))
    assert resp.status_code == 422, resp.text
    _assert_untouched(repo)


# ---------------------------------------------------------------------------
# 409 / 404 mapping (the repository decides atomically; routes map outcomes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "outcome", "status", "error"),
    [
        (*SEED, repository.ACTIVE_CONFLICT, 409, "active_conflict"),
        (*SEED, repository.NOT_FOUND, 409, "concurrent_seed"),
        (*RELEASE, repository.CANARY_ARMED, 409, "canary_already_armed"),
        (*RELEASE, repository.CANARY_IS_ACTIVE, 409, "canary_is_active"),
        (*RELEASE, repository.NOT_FOUND, 404, "no_pointer"),
        (*PROMOTE, repository.NO_CANARY, 409, "no_canary"),
        (*PROMOTE, repository.CANARY_ID_MISMATCH, 409, "canary_id_mismatch"),
        (*PROMOTE, repository.NOT_FOUND, 404, "no_pointer"),
        (*DISCARD, repository.NO_CANARY, 409, "no_canary"),
        (*DISCARD, repository.CANARY_ID_MISMATCH, 409, "canary_id_mismatch"),
    ],
    ids=[
        "seed-different-active",
        "seed-concurrent",
        "release-armed",
        "release-canary-is-active",
        "release-not-seeded",
        "promote-no-canary",
        "promote-stale-canary-id",
        "promote-no-pointer",
        "discard-no-canary",
        "discard-stale-canary-id",
    ],
)
def test_refusal_outcomes_map_to_409_or_404(
    cluster_client, repo, rsa_key, path, body, outcome, status, error
):
    row = None if outcome == repository.NOT_FOUND else ROW
    repo[_name(path)].return_value = (outcome, row)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == status, resp.text
    code = {404: "NOT_FOUND", 409: "CONFLICT"}[status]
    assert resp.json() == wire_error(status, {"error": error}, code)


def test_a_different_card_can_discard_a_foreign_canary(cluster_client, repo, rsa_key):
    """Discard is deliberately NOT card-bound: a new release must be able to
    clear a foreign (e.g. rejected) armed canary first, or one stuck canary
    blocks every later release. The card id never reaches the repository
    (whose UPDATE has no card predicate — pinned against real Postgres in
    tests/integration/test_workflow_routing_repository_pg.py)."""
    repo["discard-canary"].return_value = (repository.OK, {**ROW, "canary_workflow_id": None})
    path, body = DISCARD
    resp = cluster_client.post(path, json={**body, "card_id": "card-2"}, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    repo["discard-canary"].assert_awaited_once_with(asset_key=ASSET, canary_id="cn-1")


def test_seed_replay_with_the_same_active_is_a_200(cluster_client, repo, rsa_key):
    repo["seed-active"].return_value = (repository.UNCHANGED, {**ROW, "canary_workflow_id": None})
    path, body = SEED
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_workflow_id"] == WF_A


# ---------------------------------------------------------------------------
# GET /active shape (read: licence auth only)
# ---------------------------------------------------------------------------


def test_get_active_returns_the_row_with_a_nested_canary(internet_client, repo):
    resp = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["asset_key"] == ASSET
    assert body["active_workflow_id"] == WF_A
    assert body["prior_workflow_id"] is None
    assert body["canary"] == {
        "workflow_id": WF_B,
        "pct": 10,
        "canary_id": "cn-1",
        "released_by_card_id": "card-1",
    }
    assert "canary_workflow_id" not in body, "the canary is nested, never flat"
    repo["active"].assert_awaited_once_with(ASSET)


def test_get_active_canary_is_null_when_none_is_armed(internet_client, repo):
    repo["active"].return_value = {
        **ROW,
        "canary_workflow_id": None,
        "canary_pct": None,
        "canary_id": None,
        "released_by_card_id": None,
    }
    resp = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["canary"] is None


def test_get_active_is_404_when_there_is_no_pointer(internet_client, repo):
    repo["active"].return_value = None
    resp = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
    )
    assert resp.status_code == 404, resp.text


def test_get_active_legacy_list_only_maps_to_cell(internet_client, repo):
    resp = internet_client.get(
        "/api/workflow-routing/active", params={"list": "doing"}, headers=LICENCE
    )
    assert resp.status_code == 200, resp.text
    repo["active"].assert_awaited_once_with("cell:doing")


def test_get_active_with_card_type_is_400(internet_client, repo):
    resp = internet_client.get(
        "/api/workflow-routing/active",
        params={"card_type": "c", "list": "doing"},
        headers=LICENCE,
    )
    assert resp.status_code == 400, resp.text
    assert resp.json() == wire_error(400, {"error": "asset_key_required"}, "BAD_REQUEST")
    repo["active"].assert_not_awaited()


# ---------------------------------------------------------------------------
# Strict tenant separation on the READS (owner ruling 2026-09-30). A pointer row
# is system data; a non-platform tenant must get EXACTLY what a genuine miss
# returns — same status, bytes and headers — and the repository is never read.
# Mutation control: make ``_may_read_routing`` return True unconditionally and
# every test in this block that names CUSTOMER goes RED.
# ---------------------------------------------------------------------------


def _licence_of(monkeypatch, tenant: str) -> None:
    monkeypatch.setattr(
        auth_middleware,
        "_introspect_license",
        AsyncMock(
            return_value={
                "active": True,
                "tenant_id": tenant,
                "license_id": f"lic-{tenant}",
                "subject_id": f"subj-{tenant}",
                "byo_mode": "local_allowed",
            }
        ),
    )


def _wire(resp) -> tuple[int, bytes, list[tuple[str, str]]]:
    return resp.status_code, resp.content, sorted(resp.headers.items())


# Two ways a customer tenant reaches these reads: an introspected licence over
# the edge, and an asserted ``X-Tenant-Id`` on the tokenless trusted network.
CUSTOMER_CALLERS = ["licence", "asserted"]


def _customer_get(internet_client, cluster_client, monkeypatch, how: str, path: str, **kw):
    if how == "licence":
        _licence_of(monkeypatch, CUSTOMER)
        return internet_client.get(path, headers=LICENCE, **kw)
    return cluster_client.get(path, headers={"X-Tenant-Id": CUSTOMER}, **kw)


@pytest.mark.parametrize("params", [{"asset_key": ASSET}, {"list": "doing"}], ids=["asset", "cell"])
@pytest.mark.parametrize("how", CUSTOMER_CALLERS)
def test_customer_active_read_is_byte_identical_to_a_genuine_miss(
    internet_client, cluster_client, repo, monkeypatch, how, params
):
    # The genuine miss: a platform caller, no pointer row.
    repo["active"].return_value = None
    miss = internet_client.get("/api/workflow-routing/active", params=params, headers=LICENCE)
    assert miss.status_code == 404, miss.text
    repo["active"].reset_mock()

    # The customer: a row EXISTS (and carries card/tenant/trace ids).
    repo["active"].return_value = ROW
    resp = _customer_get(
        internet_client,
        cluster_client,
        monkeypatch,
        how,
        "/api/workflow-routing/active",
        params=params,
    )
    assert _wire(resp) == _wire(miss)
    assert b"card-1" not in resp.content and TRACE.encode() not in resp.content
    repo["active"].assert_not_awaited()


@pytest.mark.parametrize("how", CUSTOMER_CALLERS)
def test_customer_observed_read_is_byte_identical_to_nothing_observed(
    internet_client, cluster_client, repo, monkeypatch, how
):
    with patch.object(repository, "list_observed", new_callable=AsyncMock, return_value=[]) as lo:
        miss = internet_client.get("/api/workflow-routing/observed", headers=LICENCE)
        assert miss.status_code == 200 and miss.json() == [], miss.text
        lo.reset_mock()
        lo.return_value = [ROW]
        resp = _customer_get(
            internet_client, cluster_client, monkeypatch, how, "/api/workflow-routing/observed"
        )
        assert _wire(resp) == _wire(miss)
        lo.assert_not_awaited()


@pytest.mark.parametrize("tenant", sorted(platform_tenant_ids()), ids=lambda t: t.rsplit("-", 1)[1])
def test_every_platform_tenant_still_reads(internet_client, repo, monkeypatch, tenant):
    _licence_of(monkeypatch, tenant)
    with patch.object(repository, "list_observed", new_callable=AsyncMock, return_value=[ROW]):
        active = internet_client.get(
            "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
        )
        observed = internet_client.get("/api/workflow-routing/observed", headers=LICENCE)
    assert active.status_code == 200, active.text
    assert active.json()["canary"]["released_by_card_id"] == "card-1"
    assert observed.status_code == 200, observed.text
    assert [r["asset_key"] for r in observed.json()] == [ASSET]


def test_platform_tenant_on_the_asserted_path_still_reads(cluster_client, repo):
    resp = cluster_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers={"X-Tenant-Id": SYSTEM}
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Legacy writes never touch a releasable pointer (they keep only the service
# role, so writing an ``activity:`` key there would bypass the release gates).
# ---------------------------------------------------------------------------

LEGACY_ACTIVITY_WRITES = [
    (
        "/api/workflow-routing/promote",
        {
            "asset_key": "activity:AA Activity - Reasoning",
            "new_workflow_id": "AbCdEfGhIjKlMnOp",
            "observation_window_minutes": 0,
        },
    ),
    (
        "/api/workflow-routing/rollback",
        {"asset_key": "activity:AA Activity - Reasoning", "reason": "x"},
    ),
    (
        "/api/workflow-routing/extend",
        {"asset_key": "activity:AA Activity - Reasoning", "additional_minutes": 5},
    ),
    ("/api/workflow-routing/clear-prior", {"asset_key": "activity:AA Activity - Reasoning"}),
]


@pytest.mark.parametrize(
    ("path", "body"),
    LEGACY_ACTIVITY_WRITES,
    ids=[p.rsplit("/", 1)[-1] for p, _ in LEGACY_ACTIVITY_WRITES],
)
def test_legacy_writes_refuse_a_releasable_asset(cluster_client, repo, rsa_key, path, body):
    with (
        patch.object(repository, "rollback", new_callable=AsyncMock) as rb,
        patch.object(repository, "extend_observation_window", new_callable=AsyncMock) as ex,
        patch.object(repository, "clear_prior", new_callable=AsyncMock) as cp,
    ):
        resp = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=SYSTEM))
        assert resp.status_code == 403, resp.text
        assert resp.json() == wire_error(403, {"error": "use_release_path"}, "FORBIDDEN")
        for m in (rb, ex, cp):
            m.assert_not_awaited()
    _assert_untouched(repo)


def test_legacy_promote_still_writes_a_cell_pointer(cluster_client, repo, rsa_key):
    resp = cluster_client.post(
        "/api/workflow-routing/promote",
        json={
            "asset_key": "cell:doing",
            "new_workflow_id": "AbCdEfGhIjKlMnOp",
            "observation_window_minutes": 0,
        },
        headers=_sa(rsa_key, tenant=SYSTEM),
    )
    assert resp.status_code == 200, resp.text
    repo["promote"].assert_awaited_once()


def test_legacy_writes_refuse_a_pattern_asset_too(cluster_client, repo, rsa_key):
    # Allow-list of cell:, not a deny of activity: (audit finding 3).
    resp = cluster_client.post(
        "/api/workflow-routing/promote",
        json={
            "asset_key": "pattern:AA Pattern - Doing Orchestrate",
            "new_workflow_id": "AbCdEfGhIjKlMnOp",
            "observation_window_minutes": 0,
        },
        headers=_sa(rsa_key, tenant=SYSTEM),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json() == wire_error(403, {"error": "use_release_path"}, "FORBIDDEN")
    repo["promote"].assert_not_awaited()


@pytest.mark.parametrize(
    "bad", ["activity:AA\x00Activity", "activity:AA Activity\x1f", "cell:\x7f"]
)
def test_get_active_rejects_control_characters_as_4xx(internet_client, repo, bad):
    # A NUL reaching Postgres would 500 (audit finding 5).
    resp = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": bad}, headers=LICENCE
    )
    assert 400 <= resp.status_code < 500, resp.text
    repo["active"].assert_not_awaited()


@pytest.mark.parametrize("field", ["canary_id", "card_id"])
def test_canary_actions_reject_control_characters(cluster_client, repo, rsa_key, field):
    body = {"asset_key": "activity:AA Activity - Reasoning", "canary_id": "c1", "card_id": "123"}
    body[field] = "x\x00y"
    resp = cluster_client.post(
        "/api/workflow-routing/discard-canary", json=body, headers=_sa(rsa_key, tenant=SYSTEM)
    )
    assert resp.status_code == 422, resp.text
    repo["discard-canary"].assert_not_awaited()


def test_the_scoped_routing_writer_role_is_accepted(cluster_client, repo, rsa_key):
    # The dedicated n8n client holds ONLY workflow-routing-writer (one service = one identity).
    headers = _sa(rsa_key, tenant=IMPROVE, roles=[ROUTING_WRITER_SCOPED_ROLE])
    resp = cluster_client.post(
        "/api/workflow-routing/discard-canary",
        json={"asset_key": "activity:AA Activity - Reasoning", "canary_id": "c1", "card_id": "123"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


def test_an_unrelated_role_is_still_refused(cluster_client, repo, rsa_key):
    headers = _sa(rsa_key, tenant=IMPROVE, roles=["prompt-asset-bootstrap"])
    resp = cluster_client.post(
        "/api/workflow-routing/discard-canary",
        json={"asset_key": "activity:AA Activity - Reasoning", "canary_id": "c1", "card_id": "123"},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text
    _assert_untouched(repo)


# ---------------------------------------------------------------------------
# The REAL wire shape. Production (``src.api.worker_management_api:app``)
# renders every HTTPException through ``http_exception_handler`` as
# ``{"error": {"code", "message", "status"}}``; ``_app()`` above installs those
# same handlers (copied off the production app), so these pins are what a
# client receives. A bare ``FastAPI()`` would answer ``{"detail": ...}`` here.
# ---------------------------------------------------------------------------

_POINTERS_DISABLED_MESSAGE = (
    "creating workflow routing pointers is disabled until an event-driven "
    "rollback verdict exists (rollback-watch timer retired); set "
    "WORKFLOW_ROUTING_POINTERS_ENABLED=true only together with that verdict"
)


def test_the_test_app_carries_productions_http_exception_handler():
    """Guard for every body pin in this file: the handler is production's, not FastAPI's."""
    from fastapi import HTTPException
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.error_handlers import http_exception_handler

    app = _app()
    assert app.exception_handlers[HTTPException] is http_exception_handler
    assert app.exception_handlers[StarletteHTTPException] is http_exception_handler


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_wire_flag_unset_409_body(monkeypatch, cluster_client, repo, rsa_key, path, body):
    _set_pointers_flag(monkeypatch, None)
    resp = cluster_client.post(path, json=body, headers=_sa(rsa_key))
    assert resp.status_code == 409, resp.text
    assert resp.json() == {
        "error": {
            "code": "CONFLICT",
            "message": {"error": POINTERS_DISABLED, "message": _POINTERS_DISABLED_MESSAGE},
            "status": 409,
        }
    }
    assert "detail" not in resp.json()
    _assert_untouched(repo)


_REQUIRES_ROLE = wire_error(
    403, f"requires_role:{ROUTING_WRITER_ROLE}|{ROUTING_WRITER_SCOPED_ROLE}", "FORBIDDEN"
)


@pytest.mark.parametrize(("path", "body"), CREATING, ids=CREATING_IDS)
def test_wire_unauthorised_caller_403_body(
    monkeypatch, internet_client, cluster_client, repo, rsa_key, path, body
):
    _set_pointers_flag(monkeypatch, None)
    licence_only = internet_client.post(
        path, json=body, headers={**LICENCE, "X-AA-Trace-Id": TRACE}
    )
    tokenless = cluster_client.post(
        path, json=body, headers={"X-Tenant-Id": IMPROVE, "X-AA-Trace-Id": TRACE}
    )
    wrong_role = cluster_client.post(path, json=body, headers=_sa(rsa_key, roles=["other"]))
    customer = cluster_client.post(path, json=body, headers=_sa(rsa_key, tenant=CUSTOMER))
    for resp in (licence_only, tokenless, wrong_role):
        assert resp.status_code == 403, resp.text
        assert resp.json() == _REQUIRES_ROLE
    assert customer.status_code == 403, customer.text
    assert customer.json() == {
        "error": {
            "code": "FORBIDDEN",
            "message": {"error": "platform_routing_forbidden"},
            "status": 403,
        }
    }
    _assert_untouched(repo)


@pytest.mark.parametrize("how", CUSTOMER_CALLERS)
def test_wire_active_miss_and_customer_refusal_are_the_same_404_body(
    internet_client, cluster_client, repo, monkeypatch, how
):
    repo["active"].return_value = None
    miss = internet_client.get(
        "/api/workflow-routing/active", params={"asset_key": ASSET}, headers=LICENCE
    )
    repo["active"].return_value = ROW
    refused = _customer_get(
        internet_client,
        cluster_client,
        monkeypatch,
        how,
        "/api/workflow-routing/active",
        params={"asset_key": ASSET},
    )
    expected = {
        "error": {
            "code": "NOT_FOUND",
            "message": "No routing pointer for this asset_key",
            "status": 404,
        }
    }
    assert miss.status_code == 404 and miss.json() == expected, miss.text
    assert refused.status_code == 404 and refused.json() == expected, refused.text
    assert _wire(refused) == _wire(miss)


@pytest.mark.parametrize("how", CUSTOMER_CALLERS)
def test_wire_customer_observed_is_200_empty_list(
    internet_client, cluster_client, repo, monkeypatch, how
):
    with patch.object(
        repository, "list_observed", new_callable=AsyncMock, return_value=[ROW]
    ) as lo:
        resp = _customer_get(
            internet_client, cluster_client, monkeypatch, how, "/api/workflow-routing/observed"
        )
    assert resp.status_code == 200, resp.text
    assert resp.content == b"[]"
    lo.assert_not_awaited()
