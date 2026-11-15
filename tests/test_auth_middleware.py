"""Tests for src.api.auth_middleware module.

Covers:
- _is_trusted_network: IP trust checks against TRUSTED_NETWORKS CIDRs
- _is_public_endpoint: public path matching
- _extract_license_token: token extraction from X-Tenant-Id header only
- _unauthorized_response: 401 response structure
- AuthenticationMiddleware.dispatch: integration flows (bypass, public, trusted, token auth)
- get_current_user: request.state extraction and 401 on missing user
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import (
    AuthenticationMiddleware,
    _is_trusted_network,
    get_current_user,
)
from src.api.tenant_constants import BYPASS_TENANT_UUID, TENANT_UUIDS

# ============================================================================
# _is_trusted_network
# ============================================================================


class TestIsTrustedNetwork:
    """Tests for the _is_trusted_network helper."""

    def test_loopback_ipv4_trusted(self):
        assert _is_trusted_network("127.0.0.1") is True

    def test_class_a_private_trusted(self):
        assert _is_trusted_network("10.0.0.1") is True

    def test_class_c_private_trusted(self):
        assert _is_trusted_network("192.168.1.1") is True

    def test_class_b_private_172_trusted(self):
        assert _is_trusted_network("172.16.5.10") is True

    def test_public_ip_not_trusted(self):
        assert _is_trusted_network("8.8.8.8") is False

    def test_none_not_trusted(self):
        assert _is_trusted_network(None) is False

    def test_empty_string_not_trusted(self):
        assert _is_trusted_network("") is False

    def test_invalid_ip_not_trusted(self):
        assert _is_trusted_network("not-an-ip") is False

    def test_ipv6_loopback_trusted(self):
        assert _is_trusted_network("::1") is True

    def test_public_ipv6_not_trusted(self):
        assert _is_trusted_network("2001:db8::1") is False


# ============================================================================
# _is_public_endpoint
# ============================================================================


class TestIsPublicEndpoint:
    """Tests for AuthenticationMiddleware._is_public_endpoint."""

    @pytest.fixture()
    def middleware(self):
        app = FastAPI()
        return AuthenticationMiddleware(app)

    @pytest.mark.parametrize(
        "path",
        [
            "/health",
            "/healthz",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/ready",
            "/readyz",
            "/metrics",
            "/",
        ],
    )
    def test_public_endpoints_return_true(self, middleware, path):
        assert middleware._is_public_endpoint(path) is True

    def test_docs_subpath_public(self, middleware):
        assert middleware._is_public_endpoint("/docs/oauth2-redirect") is True

    def test_health_subpath_public(self, middleware):
        assert middleware._is_public_endpoint("/health/live") is True

    def test_static_prefix_public(self, middleware):
        assert middleware._is_public_endpoint("/static/style.css") is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/workers",
            "/api/v1/graphs",
            "/workers",
            "/login",
        ],
    )
    def test_private_endpoints_return_false(self, middleware, path):
        assert middleware._is_public_endpoint(path) is False

    def test_trailing_slash_stripped(self, middleware):
        assert middleware._is_public_endpoint("/health/") is True


# ============================================================================
# _extract_license_token
# ============================================================================


class TestExtractLicenseToken:
    """Tests for AuthenticationMiddleware._extract_license_token."""

    @pytest.fixture()
    def middleware(self):
        app = FastAPI()
        return AuthenticationMiddleware(app)

    @staticmethod
    def _make_request(headers: dict | None = None) -> Request:
        """Build a minimal ASGI Request with given headers."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/test",
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        }
        return Request(scope)

    def test_extract_from_x_tenant_id_bearer(self, middleware):
        req = self._make_request({"X-Tenant-Id": "Bearer my-license-token-123"})
        assert middleware._extract_license_token(req) == "my-license-token-123"

    def test_authorization_bearer_is_a_licence_source(self, middleware):
        """The ingress gateway passes the validated licence in Authorization (2026-10-14)."""
        req = self._make_request({"Authorization": "Bearer auth-token-456"})
        assert middleware._extract_license_token(req) == "auth-token-456"

    def test_x_tenant_id_takes_precedence_over_authorization(self, middleware):
        """X-Tenant-Id: Bearer wins when both carry a token."""
        req = self._make_request(
            {
                "X-Tenant-Id": "Bearer tenant-token",
                "Authorization": "Bearer auth-token",
            }
        )
        assert middleware._extract_license_token(req) == "tenant-token"

    def test_returns_none_when_no_headers(self, middleware):
        req = self._make_request({})
        assert middleware._extract_license_token(req) is None

    def test_returns_none_when_not_bearer_format(self, middleware):
        req = self._make_request({"X-Tenant-Id": "plain-value"})
        assert middleware._extract_license_token(req) is None

    def test_authorization_scheme_is_case_insensitive(self, middleware):
        """RFC 9110: the auth scheme is case-insensitive."""
        req = self._make_request({"Authorization": "BEARER upper-case-token"})
        assert middleware._extract_license_token(req) == "upper-case-token"


# ============================================================================
# _unauthorized_response
# ============================================================================


class TestUnauthorizedResponse:
    """Tests for AuthenticationMiddleware._unauthorized_response."""

    @pytest.fixture()
    def middleware(self):
        app = FastAPI()
        return AuthenticationMiddleware(app)

    def test_status_code_401(self, middleware):
        resp = middleware._unauthorized_response("bad token", reason="missing_token")
        assert resp.status_code == 401

    def test_www_authenticate_header(self, middleware):
        resp = middleware._unauthorized_response("bad token", reason="missing_token")
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_body_error_structure(self, middleware):
        resp = middleware._unauthorized_response(
            "Invalid or expired license token", reason="license_inactive"
        )
        body = json.loads(resp.body.decode())
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert body["error"]["message"] == "Invalid or expired license token"
        assert body["error"]["status"] == 401


# ============================================================================
# Integration: dispatch via TestClient
# ============================================================================


class TestMiddlewareDispatchIntegration:
    """Integration tests exercising the full middleware dispatch through TestClient."""

    @staticmethod
    def _build_app(bypass_auth: bool = False) -> FastAPI:
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, bypass_auth=bypass_auth)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/api/workers")
        async def workers(request: Request):
            from src.api.identity_context import (
                get_app_id_from_context,
                get_correlation_id_from_context,
                get_license_id_from_context,
                get_stage_from_context,
                get_subject_id_from_context,
                get_tenant_id_from_context,
            )

            return {
                "user": request.state.user,
                "tenant_id": request.state.tenant_id,
                "subject_id": getattr(request.state, "subject_id", "MISSING"),
                "license_id": getattr(request.state, "license_id", "MISSING"),
                "ctx_tenant_id": get_tenant_id_from_context(),
                "ctx_license_id": get_license_id_from_context(),
                "ctx_subject_id": get_subject_id_from_context(),
                "ctx_correlation_id": get_correlation_id_from_context(),
                "ctx_stage": get_stage_from_context(),
                "ctx_app_id": get_app_id_from_context(),
            }

        return app

    # --- bypass_auth --------------------------------------------------

    def test_bypass_auth_sets_test_user(self):
        app = self._build_app(bypass_auth=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/workers")
        assert resp.status_code == 200
        data = resp.json()
        # user_id stays the readable label — it is an identity string, not a
        # tenant column. tenant_id must be a real uuid: it is written straight
        # into a `uuid` column (migration 024), so the literal "test_user" this
        # used to assert would make every bypass_auth test that touches Postgres
        # die with 22P02.
        assert data["user"]["user_id"] == "test_user"
        assert data["tenant_id"] == BYPASS_TENANT_UUID
        assert uuid.UUID(data["tenant_id"]).version == 5

    # --- public endpoints ---------------------------------------------

    def test_public_endpoint_skips_auth(self):
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    # --- trusted network, no token ------------------------------------

    def test_trusted_network_no_token_uses_default_tenant(self):
        """Patch _is_trusted_network to simulate a trusted caller."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] is not None

    def test_trusted_network_with_plain_x_tenant_id(self):
        """When X-Tenant-Id is NOT Bearer format, no token extracted.
        Trusted network uses the plain header value as tenant."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": TENANT_UUIDS["george-dev"]})
        assert resp.status_code == 200
        # Resolved to the tenant's uuid, not stored verbatim. `george-dev` is
        # deliberately NOT the default tenant, so this still discriminates from
        # test_trusted_network_no_token_uses_default_tenant above.
        assert resp.json()["tenant_id"] == TENANT_UUIDS["george-dev"]

    def test_correlation_headers_read_into_context(self):
        """X-Correlation-Id + X-Stage are stashed in ContextVars for cost attribution."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": TENANT_UUIDS["george-dev"],
                    "X-Correlation-Id": "run-123",
                    "X-Stage": "script_generation",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctx_correlation_id"] == "run-123"
        assert data["ctx_stage"] == "script_generation"

    def test_correlation_absent_defaults_empty(self):
        """No correlation headers → empty ContextVars (opt-in, backward-compatible)."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": TENANT_UUIDS["george-dev"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctx_correlation_id"] == ""
        assert data["ctx_stage"] == ""

    def test_app_id_header_read_into_context(self):
        """X-App-Id is relayed verbatim into the app_id ContextVar."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": TENANT_UUIDS["george-dev"],
                    "X-App-Id": "requirements_management",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["ctx_app_id"] == "requirements_management"

    def test_app_id_absent_defaults_empty(self):
        """No X-App-Id header → empty ContextVar. worker_management never
        invents an app_id — an absent header must stay absent, not become a
        guess that collapses every un-stamped caller into one bucket."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": TENANT_UUIDS["george-dev"]})
        assert resp.status_code == 200
        assert resp.json()["ctx_app_id"] == ""

    # --- untrusted network, no token ----------------------------------

    def test_untrusted_no_token_returns_401(self):
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=False):
            resp = client.get("/api/workers")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"

    # --- valid license token ------------------------------------------

    def test_valid_license_token_authenticates(self):
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        # Live introspect response shape: distinct tenant_id + subject_id.
        # Pre-2026-04-27 this test sent only `subject_id` and relied on a
        # buggy fallback in middleware that conflated the two.
        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-1",
        }

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer valid-license-token"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["tenant_id"] == "11111111-1111-4111-8111-111111111111"
        assert data["tenant_id"] == "11111111-1111-4111-8111-111111111111"

    def test_effective_tenant_honored_for_delegating_subject(self, monkeypatch):
        """A trusted service account (subject in AUTH_DELEGATING_SUBJECTS) may
        re-scope the request to X-Effective-Tenant-Id so BYO + cost attribution
        follow the run's real tenant — and byo_mode re-resolves for it."""
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "22222222-2222-4222-8222-222222222222",
            "subject_id": sa_subject,
            "byo_mode": "platform_only",
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            patch(
                "src.api.auth_middleware._lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "44444444-4444-4444-8444-444444444444",
                    "X-Subject-Id": "end-user-7",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "44444444-4444-4444-8444-444444444444"
        assert data["ctx_tenant_id"] == "44444444-4444-4444-8444-444444444444"
        # run's own subject forwarded for per-user metering
        assert data["subject_id"] == "end-user-7"

    def test_delegation_without_a_forwarded_licence_does_not_keep_the_service_accounts(
        self, monkeypatch
    ):
        """⛔ Attributed-and-wrong is worse than visibly unattributed.

        The tenant has just been re-scoped to the CUSTOMER, but the introspected
        licence belongs to the SERVICE ACCOUNT. Keeping it stamps customer spend
        on someone else's entitlement, and nothing downstream can tell: billing
        counts the row as attributed, the invoice line names the wrong licence,
        and the wrong quota pool drains.

        Measured 2026-09-02, when WORKER_MANAGEMENT_LICENSE_TOKEN was first set on
        `requirements` and this branch went live for the conflict-reconciler:
        rows whose licence-owner != billed-tenant went from 11 (12h before) to
        3,411 (11.5h after). 3,397 carried svc-requirements' "Coding Module"
        licence across NINE tenants that do not own it, drawing 10,771,443 tokens
        against a licence whose quota_monthly is 1,000,000.

        `None` lets billing's `resolve_license(tenant_id, service_key)` find a
        licence the ADOPTED tenant actually owns, and where none exists the row
        stays unattributed and visible instead of silently misattributed.
        """
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "22222222-2222-4222-8222-222222222222",
            "subject_id": sa_subject,
            # The service account's OWN licence, as introspection returns it.
            "license_id": "01a0138c-ffd9-73f4-beb9-d57fcacdb7dc",
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            patch(
                "src.api.auth_middleware._lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "44444444-4444-4444-8444-444444444444",
                    # deliberately NO X-License-Id: the reconciler sweeper runs
                    # outside any request and has no customer licence in scope.
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        # The tenant was adopted...
        assert data["tenant_id"] == "44444444-4444-4444-8444-444444444444"
        # ...so the service account's licence must NOT ride along with it.
        assert data["license_id"] is None
        assert data["ctx_license_id"] in (None, "")

    def test_delegation_with_a_forwarded_licence_still_uses_it(self, monkeypatch):
        """The control: the working path must be untouched.

        Without this, the test above could be passing because delegation stopped
        producing a licence AT ALL — which would break every caller that does
        forward one. `fwd_license` must still win, exactly as before.
        """
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "22222222-2222-4222-8222-222222222222",
            "subject_id": sa_subject,
            "license_id": "01a0138c-ffd9-73f4-beb9-d57fcacdb7dc",
        }
        customer_licence = "99999999-9999-4999-8999-999999999999"
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            patch(
                "src.api.auth_middleware._lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "44444444-4444-4444-8444-444444444444",
                    "X-License-Id": customer_licence,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "44444444-4444-4444-8444-444444444444"
        assert data["license_id"] == customer_licence

    def test_effective_tenant_unresolvable_is_a_400_not_a_silent_adoption(self, monkeypatch):
        """⛔ The delegation header is a SECOND mouth on the tenant pipe.

        It was ``.strip()``ed and adopted with no validation, so a delegating
        service account could re-scope a request to any string — ``sales``, or
        ``'; DROP--`` — and that value reached ``scope_clause`` and the database.
        ``tenant_id`` is a ``uuid`` column now, so the same input becomes a 22P02
        blamed on Postgres rather than on the caller. Found only AFTER the
        tokenless header had been "closed": one entry point is not the boundary.
        """
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "22222222-2222-4222-8222-222222222222",
            "subject_id": sa_subject,
        }
        for bogus in ("sales", "'; DROP--", "n8n-dev-typo", ".."):
            with patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ):
                resp = client.get(
                    "/api/workers",
                    headers={
                        "X-Tenant-Id": "Bearer sa-token",
                        "X-Effective-Tenant-Id": bogus,
                    },
                )
            assert resp.status_code == 400, f"{bogus!r} was adopted with {resp.status_code}"
            assert resp.json()["error"]["code"] == "INVALID_TENANT"
            # Non-vacuity: it must not have silently fallen back to the SA's own
            # tenant either, which would be a quiet cross-tenant read.
            assert "tenant_id" not in resp.json()

    def test_effective_tenant_accepts_a_known_name_and_resolves_it(self, monkeypatch):
        """A NAME still delegates — resolved to its uuid, never stored as a name.

        ⛔ THIS PINS A KNOWN-OPEN DEFECT, DELIBERATELY. `n8n-dev` resolves through
        the PORTAL REALM's map to `0195f48b-…0001` (`n8n-dev-portal-realm`, the
        SEED tenant) while the licences registry has `n8n-dev` as
        `019de842-c92a-7e37-8fdd-eb0d8c5396f9` — the same misattribution as issue
        #264, on the delegation header instead of the tokenless one.

        ⚠️ I briefly inverted this test to assert a 400, and that was reverted on
        evidence rather than on second thoughts. The tokenless door could be
        closed because its log carries the RAW header value, so Loki proved only
        1 of 1,501 assertions in 30d was name-valued. THIS door logged only the
        resolved uuid until 2026-09-16, so a name-sender and a uuid-sender were
        byte-identical: of 72,351 delegated requests in the same window, 58,710
        (81.1%) resolved to a uuid a name could equally have produced. Closing it
        would 400 an unknown share of that, and the five delegating subjects in
        the logs are not covered by the three-repo sender enumeration.

        ⭐ The exit condition is mechanical: `_effective_tenant`'s call site now
        logs `raw=`. Once a full retention window has passed, group by the raw
        value; if the name count is 0, make it strict and invert this test.

        The value stays `george-dev` here ONLY because this test now asserts the
        permissive path, where every name behaves alike. The discriminating
        fixture (`n8n-dev`, where the registries disagree) lives in
        tests/test_tokenless_tenant_name_is_refused.py.
        """
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value={
                    "active": True,
                    "tenant_id": "22222222-2222-4222-8222-222222222222",
                    "subject_id": sa_subject,
                },
            ),
            patch(
                "src.api.auth_middleware._lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "george-dev",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == TENANT_UUIDS["george-dev"]

    def test_introspected_tenant_that_is_not_a_uuid_is_refused(self):
        """Licences is trusted, but it still returns a string into a uuid column.

        A malformed value would fail every statement of the request with 22P02 and
        read as a database fault rather than as bad introspection data.
        """
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={"active": True, "tenant_id": "not-a-uuid", "subject_id": None},
        ):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": "Bearer t"})
        assert resp.status_code == 401
        assert "tenant_id" not in resp.json()

    def test_effective_tenant_ignored_for_non_delegating_subject(self, monkeypatch):
        """A customer (non-delegating subject) cannot spoof another tenant."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "33333333-3333-4333-8333-333333333333",
            "subject_id": "some-customer-subject",
            "byo_mode": "local_allowed",
        }
        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer customer-token",
                    "X-Effective-Tenant-Id": "55555555-5555-4555-8555-555555555555",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "33333333-3333-4333-8333-333333333333"

    def test_effective_tenant_rejected_header_logs_warning(self, monkeypatch, caplog):
        """A non-delegating subject that sends an effective-tenant header is logged
        (spoof attempt or a caller misconfigured against AUTH_DELEGATING_SUBJECTS)."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "33333333-3333-4333-8333-333333333333",
            "subject_id": "some-customer-subject",
            "byo_mode": "local_allowed",
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            caplog.at_level("WARNING", logger="src.api.auth_middleware"),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer customer-token",
                    "X-Effective-Tenant-Id": "55555555-5555-4555-8555-555555555555",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "33333333-3333-4333-8333-333333333333"
        assert "effective-tenant header rejected" in caplog.text

    def test_effective_tenant_rejected_warning_names_the_path_and_the_stamped_tenant(
        self, monkeypatch, caplog
    ):
        """#242 — ``subject=`` alone was not actionable.

        ⭐ MEASURED: this fires 17x over 7 days, ALL from one subject, most
        recently 2026-09-16T17:20Z. A single misconfigured caller, and the two
        facts needed to find it were the missing ones: WHICH route it hits, and
        WHICH tenant the request actually ran as. The pair "header wanted X,
        request ran as Y" is the whole diagnosis, and only Y was recoverable
        before — X is the header the caller sent and Y is what the introspected
        licence stamped instead.

        ⛔ STILL NOT A 401. Refusing is blocked on a Keycloak change; this asserts
        the response is untouched so a later session cannot mistake the extra
        fields for the refusal having landed.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "33333333-3333-4333-8333-333333333333",
            "subject_id": "some-customer-subject",
            "byo_mode": "local_allowed",
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            caplog.at_level("WARNING", logger="src.api.auth_middleware"),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer customer-token",
                    "X-Effective-Tenant-Id": "55555555-5555-4555-8555-555555555555",
                },
            )

        rejected = [
            rec for rec in caplog.records if "effective-tenant header rejected" in rec.getMessage()
        ]
        assert len(rejected) == 1
        rendered = rejected[0].getMessage()
        assert "subject=some-customer-subject" in rendered, "the pre-existing field must survive"
        assert "path=/api/workers" in rendered, "WHICH route the misconfigured caller hits"
        assert "stamped_tenant=33333333-3333-4333-8333-333333333333" in rendered, (
            "the tenant the request actually ran as — the introspected one wins here"
        )
        # ⛔ Behaviour unchanged: no 401, and the introspected tenant still wins.
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "33333333-3333-4333-8333-333333333333"

    def test_effective_tenant_equal_tenant_header_no_warning(self, monkeypatch, caplog):
        """A non-delegating subject whose effective-tenant header equals its OWN
        introspected tenant is a benign no-op (the S2S hook stamps the header
        unconditionally) ⇒ no warning. Only a DIFFERENT asserted tenant is a real
        re-scope attempt; warning on the equal case would be fleet-wide log spam."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "33333333-3333-4333-8333-333333333333",
            "subject_id": "some-customer-subject",
            "byo_mode": "local_allowed",
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            caplog.at_level("WARNING", logger="src.api.auth_middleware"),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer customer-token",
                    "X-Effective-Tenant-Id": "33333333-3333-4333-8333-333333333333",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "33333333-3333-4333-8333-333333333333"
        assert "effective-tenant header rejected" not in caplog.text

    def test_effective_tenant_inert_without_config(self, monkeypatch):
        """Default (no AUTH_DELEGATING_SUBJECTS) ⇒ side-channel inert."""
        monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "22222222-2222-4222-8222-222222222222",
            "subject_id": "8382f0b3-c082-4f97-8d47-120ce5a18e05",
        }
        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "44444444-4444-4444-8444-444444444444",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "22222222-2222-4222-8222-222222222222"

    def test_authorization_licence_alone_authenticates_off_mesh(self):
        """An Authorization licence is introspected, even from an untrusted network."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-1",
        }

        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            patch("src.api.auth_middleware._is_trusted_network", return_value=False),
        ):
            resp = client.get(
                "/api/workers",
                headers={"Authorization": "Bearer valid-license-token"},
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "11111111-1111-4111-8111-111111111111"

    # --- invalid license token ----------------------------------------

    def test_invalid_license_token_returns_401(self):
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={"active": False},
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer bad-token"},
            )
        assert resp.status_code == 401

    # --- introspection transiently unavailable (None) -> retryable 503 ---

    def test_introspect_none_returns_503_retryable(self):
        """A None result means introspection could not reach a verdict (licenses
        slow/unreachable, cold stale cache) — NOT a rejected token. It must be a
        retryable 503, never a 401, so a transient blip does not hard-fail a
        long run's tail LLM stage (the growth_bundle 401 class)."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer some-token"},
            )
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"
        assert resp.json()["error"]["code"] == "INTROSPECTION_UNAVAILABLE"

    # --- subject_id propagation (Phase 5a per-user metering) ----------

    def test_introspect_with_subject_id_stashes_on_request_state(self):
        """Phase 5a: subject_id from introspect must land on request.state.subject_id."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-xyz",
            "license_id": "lic-1",
        }

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer valid-license-token"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == "user-uuid-xyz"

    def test_introspect_with_null_subject_id_stashes_none(self):
        """Backwards compat: introspect with null/missing subject_id → request.state.subject_id is None."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        # Token bound to a tenant but no Subject row (e.g. M2M license)
        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": None,
        }

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer m2m-token"},
            )
        assert resp.status_code == 200
        data = resp.json()
        # Must be JSON null, NOT the literal string "None"
        assert data["subject_id"] is None

    def test_trusted_network_path_leaves_subject_id_none(self):
        """Trusted-network shortcut has no introspected user identity."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers")
        assert resp.status_code == 200
        assert resp.json()["subject_id"] is None

    def test_bypass_auth_leaves_subject_id_none(self):
        """bypass_auth (test mode) sets a stub user but no real subject_id."""
        app = self._build_app(bypass_auth=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/workers")
        assert resp.status_code == 200
        assert resp.json()["subject_id"] is None

    # --- license_id + identity ContextVars (wiring follow-up) ---------

    def test_introspect_with_license_id_stashes_on_request_state(self):
        """license_id from introspect must land on request.state.license_id."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-xyz",
            "license_id": "lic-uuid-1",
        }

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer valid-license-token"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["license_id"] == "lic-uuid-1"

    def test_introspect_populates_identity_context_vars(self):
        """ContextVars must mirror request.state values for cross-async propagation."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-xyz",
            "license_id": "lic-uuid-1",
        }

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer valid-license-token"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctx_tenant_id"] == "11111111-1111-4111-8111-111111111111"
        assert data["ctx_license_id"] == "lic-uuid-1"
        assert data["ctx_subject_id"] == "user-uuid-xyz"

    def test_trusted_network_populates_context_vars_with_empty_subject(self):
        """Trusted network: ContextVar tenant set, license/subject empty."""
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": TENANT_UUIDS["george-dev"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctx_tenant_id"] == TENANT_UUIDS["george-dev"]
        assert data["ctx_license_id"] == ""
        assert data["ctx_subject_id"] == ""

    def test_bypass_auth_populates_context_vars_with_test_user(self):
        """bypass_auth: tenant ContextVar = test_user, others empty."""
        app = self._build_app(bypass_auth=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctx_tenant_id"] == BYPASS_TENANT_UUID
        assert data["ctx_license_id"] == ""
        assert data["ctx_subject_id"] == ""

    # --- active token missing tenant context --------------------------

    def test_active_token_missing_tenant_returns_401(self):
        app = self._build_app(bypass_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={"active": True},
        ):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "Bearer token-no-tenant"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert "tenant context" in body["error"]["message"].lower()


# ============================================================================
# get_current_user
# ============================================================================


class TestGetCurrentUser:
    """Tests for the get_current_user dependency helper."""

    def test_returns_user_when_set(self):
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)
        request.state.user = {"user_id": "u1", "tenant_id": "t1"}
        result = get_current_user(request)
        assert result == {"user_id": "u1", "tenant_id": "t1"}

    def test_raises_401_when_not_set(self):
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)
        with pytest.raises(Exception) as exc_info:
            get_current_user(request)
        assert getattr(exc_info.value, "status_code", None) == 401


# ============================================================================
# _introspect_license (unit)
# ============================================================================


class TestIntrospectLicense:
    """Tests for the _introspect_license async helper with mocked httpx."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_successful_introspection_returns_data(self):
        from src.api.auth_middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "subject_id": "t-1"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await _introspect_license("good-token")

        assert result == {"active": True, "subject_id": "t-1"}

    @pytest.mark.asyncio(loop_scope="session")
    async def test_introspection_caches_active_result(self):
        from src.api.auth_middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "subject_id": "cached-t"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result1 = await _introspect_license("cache-token")
            result2 = await _introspect_license("cache-token")

        assert result1 == result2
        # post called only once; second call hits cache
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_introspection_http_error_returns_none(self):
        from src.api.auth_middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await _introspect_license("fail-token")

        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_introspection_exception_returns_none(self):
        from src.api.auth_middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await _introspect_license("error-token")

        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_inactive_token_not_cached(self):
        from src.api.auth_middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await _introspect_license("inactive-token")
            await _introspect_license("inactive-token")

        # Called twice because inactive results are NOT cached
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_transient_failure_serves_stale_last_known_good(self):
        """Stale-while-error: once a token has validated ACTIVE, a later transient
        introspect failure (positive cache expired, licenses briefly unreachable)
        serves the last-known-good result instead of returning None (which would
        become a spurious 503/401). This is the durable fix for the growth_bundle
        401 class — a long run's tail stage keeps working across a licenses blip."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
        )

        _license_cache.clear()
        _license_stale_cache.clear()

        good = {"active": True, "tenant_id": "t-stale", "subject_id": "s-1"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = good
        ok_client = AsyncMock()
        ok_client.__aenter__ = AsyncMock(return_value=ok_client)
        ok_client.__aexit__ = AsyncMock(return_value=False)
        ok_client.post = AsyncMock(return_value=ok_resp)

        # 1) A successful introspection warms both the positive + stale caches.
        with patch("src.api.auth_middleware.httpx.AsyncClient", return_value=ok_client):
            first = await _introspect_license("stale-token")
        assert first == good

        # Simulate the 5-min positive cache expiring (stale 15-min cache remains).
        _license_cache.clear()

        # 2) Now licenses is transiently unreachable → must serve the stale
        #    last-known-good, NOT None.
        boom_client = AsyncMock()
        boom_client.__aenter__ = AsyncMock(return_value=boom_client)
        boom_client.__aexit__ = AsyncMock(return_value=False)
        boom_client.post = AsyncMock(side_effect=Exception("connection refused"))
        with patch("src.api.auth_middleware.httpx.AsyncClient", return_value=boom_client):
            served = await _introspect_license("stale-token")
        assert served == good, "transient failure must serve stale last-known-good"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_transient_failure_cold_stale_returns_none(self):
        """A transient failure with NO last-known-good (cold stale cache) returns
        None → the middleware maps it to a retryable 503, never a 401."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
        )

        _license_cache.clear()
        _license_stale_cache.clear()

        boom_client = AsyncMock()
        boom_client.__aenter__ = AsyncMock(return_value=boom_client)
        boom_client.__aexit__ = AsyncMock(return_value=False)
        boom_client.post = AsyncMock(side_effect=Exception("timeout"))
        with patch("src.api.auth_middleware.httpx.AsyncClient", return_value=boom_client):
            result = await _introspect_license("never-seen-token")
        assert result is None

    # ----- token_not_found rotation-race grace -----

    @staticmethod
    def _client(*, status=200, json_body=None):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = json_body or {}
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=resp)
        return client

    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_not_found_warm_stale_serves_last_known_good(self):
        """token_not_found on a token seen active moments ago serves warm stale."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        good = {"active": True, "tenant_id": "t-rot", "subject_id": "s-rot"}
        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=self._client(json_body=good),
        ):
            await _introspect_license("rot-token")
        _license_cache.clear()  # positive cache gone, stale still warm
        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=self._client(json_body={"active": False, "reason": "token_not_found"}),
        ):
            served = await _introspect_license("rot-token")
        assert served == good

    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_not_found_cold_replica_returns_none(self):
        """Cold replica (no warm stale) on token_not_found returns None → 503."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=self._client(json_body={"active": False, "reason": "token_not_found"}),
        ):
            result = await _introspect_license("cold-token")
        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_not_found_past_window_is_authoritative(self):
        """After the grace window (from first sighting), token_not_found is the
        authoritative inactive verdict → the dict (→ 401 at dispatch)."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        tnf = {"active": False, "reason": "token_not_found"}
        with patch("src.api.auth_middleware.time.monotonic", side_effect=[0.0, 10_000.0]):
            with patch(
                "src.api.auth_middleware.httpx.AsyncClient",
                return_value=self._client(json_body=tnf),
            ):
                first = await _introspect_license("gone-token")
                second = await _introspect_license("gone-token")
        assert first is None  # within window
        assert second == tnf  # expired → authoritative

    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_invalid_is_not_graced(self):
        """The grace is reason-specific: token_invalid (revoked/expired) is NOT
        stale-served even with a warm stale entry — it stays authoritative."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        good = {"active": True, "tenant_id": "t-was", "subject_id": "s-was"}
        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=self._client(json_body=good),
        ):
            await _introspect_license("revk-token")
        _license_cache.clear()  # stale still warm
        revoked = {"active": False, "reason": "token_invalid"}
        with patch(
            "src.api.auth_middleware.httpx.AsyncClient",
            return_value=self._client(json_body=revoked),
        ):
            result = await _introspect_license("revk-token")
        assert result == revoked  # NOT stale-served

    @pytest.mark.asyncio(loop_scope="session")
    async def test_persistent_token_not_found_does_not_reopen_window(self):
        """Regression: a permanently-missing token settles to authoritative (401)
        after the window, not re-opening a fresh grace each ~stale-TTL. The expired
        branch re-pins the ORIGINAL first-sighting, so the stored value stays t0."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        tnf = {"active": False, "reason": "token_not_found"}
        with patch("src.api.auth_middleware.time.monotonic", side_effect=[0.0, 200.0, 400.0]):
            with patch(
                "src.api.auth_middleware.httpx.AsyncClient",
                return_value=self._client(json_body=tnf),
            ):
                r1 = await _introspect_license("perma-gone")
                r2 = await _introspect_license("perma-gone")
                r3 = await _introspect_license("perma-gone")
        assert r1 is None
        assert r2 == tnf
        assert r3 == tnf
        assert list(_license_token_not_found_since.values()) == [0.0]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_grace_zero_disables_token_not_found_transient(self):
        """LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS=0 restores instant authoritative."""
        from src.api.auth_middleware import (
            _introspect_license,
            _license_cache,
            _license_stale_cache,
            _license_token_not_found_since,
        )

        _license_cache.clear()
        _license_stale_cache.clear()
        _license_token_not_found_since.clear()

        tnf = {"active": False, "reason": "token_not_found"}
        with patch("src.api.auth_middleware._TOKEN_NOT_FOUND_GRACE", 0.0):
            with patch(
                "src.api.auth_middleware.httpx.AsyncClient",
                return_value=self._client(json_body=tnf),
            ):
                result = await _introspect_license("cold-token")
        assert result == tnf


# ============================================================================
# request.state.identity_proven — which auth branch actually ran
# ============================================================================


class TestIdentityProvenStamp:
    """The middleware must record WHICH branch admitted the request.

    ⛔ WHY THIS IS A SEPARATE FACT FROM ``tenant_id``. Both auth branches
    produce a tenant_id, so a handler reading only that cannot tell a caller
    that presented a verified licence token from one that was admitted on
    TRUSTED_CIDRS membership and then NAMED ITS OWN TENANT in a bare header.
    ``/tenant-config/modality/{modality}/resolve`` returns a decrypted BYO
    provider key and must be able to tell those two apart — keyed off the
    branch, never off the peer CIDR, which is exactly what failed to establish
    identity in the first place.
    """

    @staticmethod
    def _build_app():
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, bypass_auth=False)

        @app.get("/api/workers")
        async def _probe(request: Request):
            # getattr with a sentinel, not a default of False: this test must be
            # able to tell "stamped False" from "never stamped".
            return {
                "identity_proven": getattr(request.state, "identity_proven", "UNSTAMPED"),
                "tenant_proven": getattr(request.state, "tenant_proven", "UNSTAMPED"),
                "tenant_id": getattr(request.state, "tenant_id", "UNSTAMPED"),
            }

        return app

    def test_verified_license_token_is_proven(self):
        app = self._build_app()
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": "11111111-1111-4111-8111-111111111111",
            "subject_id": "user-uuid-1",
        }
        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ):
            resp = client.get("/api/workers", headers={"X-Tenant-Id": "Bearer valid-license-token"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["identity_proven"] is True
        # Not delegating: the tenant came out of the introspected licence, so
        # BOTH facts are proven.
        assert resp.json()["tenant_proven"] is True

    def test_tokenless_trusted_network_assertion_is_not_proven(self):
        # The live shape: a pod on the trusted network asserting a tenant uuid
        # in a bare header. Served exactly as before — the stamp changes no
        # outcome here — but recorded as unproven.
        app = self._build_app()
        client = TestClient(app, raise_server_exceptions=False)
        with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
            resp = client.get(
                "/api/workers",
                headers={"X-Tenant-Id": "11111111-1111-4111-8111-111111111111"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["identity_proven"] is False
        assert resp.json()["tenant_proven"] is False

    def test_bypass_auth_is_not_proven(self):
        # bypass_auth is a test affordance. Switching auth off must not be a
        # way to make an unproven caller look proven.
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, bypass_auth=True)

        @app.get("/api/workers")
        async def _probe(request: Request):
            return {
                "identity_proven": getattr(request.state, "identity_proven", "UNSTAMPED"),
                "tenant_proven": getattr(request.state, "tenant_proven", "UNSTAMPED"),
            }

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/workers")
        assert resp.status_code == 200, resp.text
        assert resp.json()["identity_proven"] is False
        assert resp.json()["tenant_proven"] is False

    def test_delegated_tenant_is_identity_proven_but_not_tenant_proven(self, monkeypatch):
        """⛔ THE ONE CASE WHERE THE TWO FLAGS DISAGREE, AND THE WHOLE POINT.

        A delegating service account presents a real licence token, so
        ``identity_proven`` is True and SHOULD be — it really did prove who it
        is. It then re-scopes the request to a DIFFERENT tenant with
        ``X-Effective-Tenant-Id``, which ``_effective_tenant`` validates on
        spelling alone. The adopted tenant is therefore the caller's own
        assertion and must be stamped unproven, or
        ``_modality_router._key_for_caller`` hands over that tenant's decrypted
        provider key.

        The probe also asserts the adoption actually HAPPENED (``tenant_id`` is
        the victim's, not the service account's). Without that, a regression
        that stopped honouring delegation altogether would satisfy the
        ``tenant_proven is False`` assertion for the wrong reason.
        """
        sa_subject = "01a03c8f-16a3-76a6-98d9-79987ee191e3"
        sa_tenant = "22222222-2222-4222-8222-222222222222"
        adopted_tenant = "44444444-4444-4444-8444-444444444444"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)

        app = self._build_app()
        client = TestClient(app, raise_server_exceptions=False)
        introspect_result = {
            "active": True,
            "tenant_id": sa_tenant,
            "subject_id": sa_subject,
        }
        with (
            patch(
                "src.api.auth_middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=introspect_result,
            ),
            patch(
                "src.api.auth_middleware._lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
        ):
            resp = client.get(
                "/api/workers",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": adopted_tenant,
                },
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Delegation still works — this must not become a refusal.
        assert body["tenant_id"] == adopted_tenant
        # It proved WHO it is...
        assert body["identity_proven"] is True
        # ...and not WHICH tenant it may act for.
        assert body["tenant_proven"] is False
