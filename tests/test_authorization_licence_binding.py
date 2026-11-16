"""Licence from ``Authorization`` + tenant binding (port of requirements_management #222/#223).

The istio ingress gateway validates ``Authorization: Bearer <licence>``, leaves it
in place and injects a PLAIN ``X-Tenant-Id: <uuid>``. This middleware used to read
the licence only from ``X-Tenant-Id: Bearer``, so such a request -- and any in-mesh
pod that sent ``Authorization: Bearer <licence of A>`` + ``X-Tenant-Id: B`` -- was
admitted on the TOKENLESS trusted-network branch as whatever the plain header said.

Pinned here, each with the mutation that turns it red:

- the licence is read from ``Authorization`` (X-Tenant-Id Bearer keeps precedence);
- with it, a non-delegating plain ``X-Tenant-Id`` / ``X-Effective-Tenant-Id`` that
  names another tenant is 403 -- the bypass is CLOSED;
- a delegating subject's plain header is a delegation request;
- a non-licence in ``Authorization`` (e.g. a Keycloak JWT) is a hard 401, never a
  fall-through to network trust (George 2026-10-14);
- OBSERVE: a tokenless caller (coding-ai, fr workers.ts shape) is served exactly as
  before, and an ``X-Tenant-Id: Bearer`` caller's mismatching
  ``X-Effective-Tenant-Id`` is still served as the licence tenant until
  WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT is on.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.api.auth_middleware as mw
from src.api.auth_middleware import AuthenticationMiddleware
from src.settings import reset_settings_cache
from src.telemetry.attribution_metrics import LICENCE_TENANT_MISMATCH_TOTAL

LICENCE_TENANT = "aaaaaaaa-0000-4000-8000-0000000000aa"
OTHER_TENANT = "aaaaaaaa-0000-4000-8000-0000000000bb"
SUBJECT = "01a0dd02-0000-7000-8000-000000000001"  # NOT delegating
DELEGATOR = "019ffe9f-0000-7000-8000-00000000f00d"
LICENCE_ID = "lic-0001"
TRUSTED_PEER = ("10.42.1.7", 54321)
UNTRUSTED_PEER = ("203.0.113.9", 54321)
ENFORCE_ENV = "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENFORCE_ENV, raising=False)
    monkeypatch.delenv("REQUIRE_AUTHENTICATED_TENANT", raising=False)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", DELEGATOR)
    mw._assertion_log_seen.clear()
    reset_settings_cache()
    yield
    mw._assertion_log_seen.clear()
    reset_settings_cache()


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, object]:
        return {
            "tenant_id": request.state.tenant_id,
            "license_id": request.state.license_id,
        }

    return application


def _intro(subject: str = SUBJECT, *, active: bool = True) -> dict[str, Any]:
    if not active:
        return {"active": False, "reason": "token_not_found"}
    return {
        "active": True,
        "tenant_id": LICENCE_TENANT,
        "subject_id": subject,
        "license_id": LICENCE_ID,
        "byo_mode": "local_allowed",
    }


def _call(
    app: FastAPI,
    headers: dict[str, str],
    *,
    intro: dict[str, Any] | None = None,
    peer: tuple[str, int] = TRUSTED_PEER,
    introspect: AsyncMock | None = None,
) -> tuple[int, dict, AsyncMock]:
    spy = introspect or AsyncMock(return_value=intro if intro is not None else _intro())
    with (
        patch.object(mw, "_introspect_license", spy),
        patch.object(
            mw, "_lookup_tenant_byo_mode", new_callable=AsyncMock, return_value="local_allowed"
        ),
        patch.object(mw, "get_metrics_client", return_value=MagicMock()),
        TestClient(app, client=peer, raise_server_exceptions=False) as client,
    ):
        resp = client.get("/api/probe", headers=headers)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body, spy


def _mismatch(carrier: str, header: str, outcome: str) -> float:
    return LICENCE_TENANT_MISMATCH_TOTAL.labels(
        carrier=carrier, header=header, outcome=outcome
    )._value.get()


# --------------------------------------------------------------------------- carrier


class TestLicenceIsReadFromAuthorization:
    """MUTATION: drop the ``Authorization`` read in ``_extract_license_token`` ⇒
    every case here lands on the tokenless branch (license_id None / 401 off-mesh)."""

    def test_gateway_shape_is_authenticated_as_the_licence(self, app):
        status, body, spy = _call(
            app, {"Authorization": "Bearer lic-a", "X-Tenant-Id": LICENCE_TENANT}
        )
        assert status == 200
        assert body == {"tenant_id": LICENCE_TENANT, "license_id": LICENCE_ID}
        spy.assert_awaited_once_with("lic-a")

    def test_no_tenant_header_runs_as_the_licence_tenant(self, app):
        status, body, _ = _call(app, {"Authorization": "Bearer lic-a"})
        assert status == 200
        assert body["tenant_id"] == LICENCE_TENANT

    def test_scheme_is_case_insensitive(self, app):
        status, _, spy = _call(app, {"Authorization": "bEaReR lic-a"})
        assert status == 200
        spy.assert_awaited_once_with("lic-a")

    def test_off_mesh_caller_with_a_licence_is_admitted(self, app):
        status, body, _ = _call(app, {"Authorization": "Bearer lic-a"}, peer=UNTRUSTED_PEER)
        assert status == 200
        assert body["license_id"] == LICENCE_ID

    def test_x_tenant_id_bearer_keeps_precedence(self, app):
        status, _, spy = _call(
            app, {"X-Tenant-Id": "Bearer lic-x", "Authorization": "Bearer lic-a"}
        )
        assert status == 200
        spy.assert_awaited_once_with("lic-x")

    def test_same_tenant_spelled_in_upper_case_is_not_a_mismatch(self, app):
        status, body, _ = _call(
            app, {"Authorization": "Bearer lic-a", "X-Tenant-Id": LICENCE_TENANT.upper()}
        )
        assert status == 200
        assert body["tenant_id"] == LICENCE_TENANT


class TestNonLicenceInAuthorizationIsAHard401:
    """George 2026-10-14: a Keycloak JWT in ``Authorization`` is 401, never a silent
    fall-through to the trusted-network branch. MUTATION: on an inactive
    Authorization-borne token, fall through to network trust ⇒ 200 as OTHER_TENANT."""

    def test_inactive_authorization_token_from_the_mesh_is_401(self, app):
        status, body, _ = _call(
            app,
            {"Authorization": "Bearer eyJhbGciOi.keycloak.jwt", "X-Tenant-Id": OTHER_TENANT},
            intro=_intro(active=False),
        )
        assert status == 401
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"

    def test_introspection_outage_is_503_not_a_tokenless_admit(self, app):
        status, _, _ = _call(
            app,
            {"Authorization": "Bearer lic-a", "X-Tenant-Id": OTHER_TENANT},
            introspect=AsyncMock(return_value=None),
        )
        assert status == 503


# --------------------------------------------------------------------------- binding


class TestTheBypassIsClosedForAuthorizationLicenceCallers:
    """MUTATION: delete the ``_refuse_foreign_tenant_claims`` call ⇒ these are 200
    as the licence tenant (silent downgrade) instead of 403."""

    def test_plain_tenant_naming_another_tenant_is_403(self, app):
        before = _mismatch("authorization", "x_tenant_id", "refused")
        status, body, _ = _call(app, {"Authorization": "Bearer lic-a", "X-Tenant-Id": OTHER_TENANT})
        assert status == 403
        assert body["error"]["code"] == "TENANT_MISMATCH"
        assert _mismatch("authorization", "x_tenant_id", "refused") == before + 1

    def test_plain_tenant_name_that_is_not_the_licence_tenant_is_403(self, app):
        status, _, _ = _call(app, {"Authorization": "Bearer lic-a", "X-Tenant-Id": "n8n-dev"})
        assert status == 403

    def test_effective_tenant_naming_another_tenant_is_403(self, app):
        before = _mismatch("authorization", "x_effective_tenant_id", "refused")
        status, _, _ = _call(
            app,
            {
                "Authorization": "Bearer lic-a",
                "X-Tenant-Id": LICENCE_TENANT,
                "X-Effective-Tenant-Id": OTHER_TENANT,
            },
        )
        assert status == 403
        assert _mismatch("authorization", "x_effective_tenant_id", "refused") == before + 1

    def test_effective_tenant_equal_to_the_licence_tenant_is_a_noop(self, app):
        status, body, _ = _call(
            app, {"Authorization": "Bearer lic-a", "X-Effective-Tenant-Id": LICENCE_TENANT}
        )
        assert status == 200
        assert body["tenant_id"] == LICENCE_TENANT

    def test_a_tenant_starting_with_bearer_is_still_a_claim(self, app):
        status, _, _ = _call(app, {"Authorization": "Bearer lic-a", "X-Tenant-Id": "bearerco"})
        assert status == 403


class TestDelegatingSubjectWithAuthorizationLicence:
    """MUTATION: drop ``eff_raw = header_eff or plain_tenant`` (use the header only)
    ⇒ the plain-header delegation runs as the service-account tenant."""

    def test_plain_tenant_is_a_delegation_request(self, app):
        status, body, _ = _call(
            app,
            {"Authorization": "Bearer lic-fr", "X-Tenant-Id": OTHER_TENANT},
            intro=_intro(DELEGATOR),
        )
        assert status == 200
        assert body["tenant_id"] == OTHER_TENANT
        # The service account's licence is not the customer's: never stamped.
        assert body["license_id"] is None

    def test_effective_header_wins_over_the_plain_header(self, app):
        status, body, _ = _call(
            app,
            {
                "Authorization": "Bearer lic-fr",
                "X-Tenant-Id": LICENCE_TENANT,
                "X-Effective-Tenant-Id": OTHER_TENANT,
            },
            intro=_intro(DELEGATOR),
        )
        assert status == 200
        assert body["tenant_id"] == OTHER_TENANT

    def test_plain_header_equal_to_own_tenant_is_not_delegation(self, app):
        status, body, _ = _call(
            app,
            {"Authorization": "Bearer lic-fr", "X-Tenant-Id": LICENCE_TENANT},
            intro=_intro(DELEGATOR),
        )
        assert status == 200
        assert body == {"tenant_id": LICENCE_TENANT, "license_id": LICENCE_ID}

    def test_own_tenant_in_another_spelling_is_not_a_bad_delegation(self, app):
        """Upper-case own uuid from a delegator: no re-scope, licence stays stamped."""
        status, body, _ = _call(
            app,
            {"Authorization": "Bearer lic-fr", "X-Tenant-Id": LICENCE_TENANT.upper()},
            intro=_intro(DELEGATOR),
        )
        assert status == 200
        assert body == {"tenant_id": LICENCE_TENANT, "license_id": LICENCE_ID}


# --------------------------------------------------------------------------- observe


class TestObserveModeLeavesExistingCallersUnchanged:
    """WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT unset (the deployed state)."""

    def test_tokenless_bare_tenant_is_served_as_before(self, app):
        """coding-ai per-request shape: bare tenant, no Authorization."""
        status, body, spy = _call(app, {"X-Tenant-Id": OTHER_TENANT})
        assert status == 200
        assert body == {"tenant_id": OTHER_TENANT, "license_id": None}
        spy.assert_not_awaited()

    def test_empty_authorization_bearer_is_no_licence(self, app):
        status, body, spy = _call(app, {"X-Tenant-Id": OTHER_TENANT, "Authorization": "Bearer "})
        assert status == 200
        assert body["tenant_id"] == OTHER_TENANT
        spy.assert_not_awaited()

    def test_non_bearer_authorization_is_ignored(self, app):
        status, body, spy = _call(
            app, {"X-Tenant-Id": OTHER_TENANT, "Authorization": "Basic dXNlcjpwdw=="}
        )
        assert status == 200
        assert body["tenant_id"] == OTHER_TENANT
        spy.assert_not_awaited()

    def test_fr_workers_shape_is_unchanged(self, app):
        """fr workers.ts: X-Tenant-Id Bearer <svc licence> + X-Effective-Tenant-Id, no
        Authorization -- delegated exactly as before."""
        status, body, spy = _call(
            app,
            {"X-Tenant-Id": "Bearer lic-fr", "X-Effective-Tenant-Id": OTHER_TENANT},
            intro=_intro(DELEGATOR),
        )
        assert status == 200
        assert body["tenant_id"] == OTHER_TENANT
        spy.assert_awaited_once_with("lic-fr")

    def test_x_tenant_id_bearer_mismatching_effective_tenant_is_still_served(self, app):
        """MUTATION: make ``refuse`` unconditional ⇒ 403 here and the live
        promo-* caller (subject 01a0dd02-…) breaks."""
        before = _mismatch("x_tenant_id", "x_effective_tenant_id", "served")
        status, body, _ = _call(
            app, {"X-Tenant-Id": "Bearer lic-a", "X-Effective-Tenant-Id": OTHER_TENANT}
        )
        assert status == 200
        assert body["tenant_id"] == LICENCE_TENANT
        assert _mismatch("x_tenant_id", "x_effective_tenant_id", "served") == before + 1


class TestEnforceMode:
    def test_x_tenant_id_bearer_mismatching_effective_tenant_is_403(self, app, monkeypatch):
        monkeypatch.setenv(ENFORCE_ENV, "true")
        reset_settings_cache()
        status, _, _ = _call(
            app, {"X-Tenant-Id": "Bearer lic-a", "X-Effective-Tenant-Id": OTHER_TENANT}
        )
        assert status == 403

    def test_tokenless_bare_tenant_is_401(self, app, monkeypatch):
        monkeypatch.setenv(ENFORCE_ENV, "true")
        reset_settings_cache()
        status, _, _ = _call(app, {"X-Tenant-Id": OTHER_TENANT})
        assert status == 401


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("Bearer", None),
        ("Bearer   ", None),
        ("Bearer tok", "tok"),
        ("bearer  tok ", "tok"),
        ("Basic tok", None),
        ("Bearertok", None),
    ],
)
def test_bearer_value(raw: str | None, expected: str | None) -> None:
    assert mw._bearer_value(raw) == expected
