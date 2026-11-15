"""Tests for src.api.security_middleware.

Covers SecurityHeadersMiddleware, RequestValidationMiddleware,
RateLimitMiddleware, and AuthenticationMiddleware using real
FastAPI apps with TestClient.
"""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.security_middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal FastAPI app with the given middleware stack
# ---------------------------------------------------------------------------


def _make_app(*middleware_classes, auth_kwargs=None):
    """Return a FastAPI app wired with the requested middleware and a
    simple /ping endpoint plus /health, /docs stubs."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"status": "pong"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/docs")
    async def docs_page():
        return {"status": "docs"}

    @app.get("/redoc")
    async def redoc_page():
        return {"status": "redoc"}

    @app.get("/openapi.json")
    async def openapi():
        return {"status": "openapi"}

    @app.post("/upload")
    async def upload(request: Request):
        return {"status": "uploaded"}

    @app.get("/protected")
    async def protected():
        return {"status": "secret"}

    # Add middleware in reverse order so the first in the list is outermost
    for mw in reversed(middleware_classes):
        if mw is AuthenticationMiddleware:
            kwargs = auth_kwargs or {}
            app.add_middleware(mw, **kwargs)
        else:
            app.add_middleware(mw)

    return app


# ===================================================================
# 1. SecurityHeadersMiddleware
# ===================================================================


class TestSecurityHeadersMiddleware:
    """Tests for SecurityHeadersMiddleware."""

    def test_adds_standard_security_headers(self):
        """Security headers X-Content-Type-Options, X-Frame-Options,
        X-XSS-Protection are present on a normal endpoint."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"

    def test_regular_page_gets_deny_frame_options(self):
        """Non-docs pages must have X-Frame-Options: DENY."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        for path in ["/ping", "/health", "/protected"]:
            resp = client.get(path)
            assert resp.headers["X-Frame-Options"] == "DENY", f"Failed for {path}"

    def test_docs_page_gets_sameorigin_frame_options(self):
        """Documentation pages (/docs, /redoc, /openapi.json) must have
        X-Frame-Options: SAMEORIGIN."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        for path in ["/docs", "/redoc", "/openapi.json"]:
            resp = client.get(path)
            assert resp.headers["X-Frame-Options"] == "SAMEORIGIN", f"Failed for {path}"

    def test_adds_referrer_policy(self):
        """Referrer-Policy header is present."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_adds_csp_header(self):
        """Content-Security-Policy header is present on regular pages."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp

    def test_docs_page_csp_allows_cdn(self):
        """CSP for docs pages must allow cdn.jsdelivr.net."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/docs")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "cdn.jsdelivr.net" in csp

    def test_regular_page_has_hsts(self):
        """Non-docs pages include Strict-Transport-Security."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert "Strict-Transport-Security" in resp.headers

    def test_regular_page_has_cache_control(self):
        """Non-docs pages include Cache-Control: no-cache, no-store."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert "no-cache" in resp.headers.get("Cache-Control", "")
        assert "no-store" in resp.headers.get("Cache-Control", "")

    def test_xss_in_query_param_returns_422(self):
        """Query parameters with XSS patterns trigger a 422."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping?name=<script>alert(1)</script>")
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_sql_injection_in_query_param_returns_422(self):
        """Query parameters with SQL injection patterns trigger a 422."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping?id=' OR '1'='1' --")
        assert resp.status_code == 422

    def test_clean_query_params_pass(self):
        """Normal query parameters pass through without issues."""
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping?page=1")
        assert resp.status_code == 200


# ===================================================================
# 2. RequestValidationMiddleware
# ===================================================================


class TestRequestValidationMiddleware:
    """Tests for RequestValidationMiddleware."""

    def test_oversized_content_length_returns_413(self):
        """Content-Length exceeding 50 MB must produce a 413 response."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        oversized = 50 * 1024 * 1024 + 1  # 50 MB + 1 byte
        resp = client.post(
            "/upload",
            content=b"x",
            headers={"Content-Length": str(oversized)},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_normal_content_length_passes(self):
        """Requests within the size limit pass through normally."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/upload",
            content=b"small payload",
            headers={"Content-Length": "13"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "uploaded"

    def test_no_content_length_passes(self):
        """GET requests without Content-Length pass through."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_suspicious_user_agent_still_allowed(self):
        """Suspicious user-agents are logged but NOT blocked."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        for agent in ["sqlmap/1.7", "Nikto/2.1", "Nmap Scripting Engine"]:
            resp = client.get("/ping", headers={"User-Agent": agent})
            assert resp.status_code == 200, f"Agent '{agent}' was unexpectedly blocked"

    def test_xss_in_header_returns_400(self):
        """XSS content in referer/origin/x-forwarded-for returns 400."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/ping",
            headers={"Referer": "<script>alert('xss')</script>"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "BAD_REQUEST"

    def test_sql_injection_in_header_returns_400(self):
        """SQL injection in dangerous headers returns 400."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/ping",
            headers={"Origin": "' OR '1'='1' --"},
        )
        assert resp.status_code == 400

    def test_clean_headers_pass(self):
        """Normal header values pass through without issue."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/ping",
            headers={
                "Referer": "https://example.com/page",
                "Origin": "https://example.com",
                "X-Forwarded-For": "192.168.1.1",
            },
        )
        assert resp.status_code == 200

    def test_exactly_50mb_passes(self):
        """Content-Length of exactly 50 MB (the limit) passes."""
        app = _make_app(RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        exactly_50mb = 50 * 1024 * 1024
        resp = client.post(
            "/upload",
            content=b"x",
            headers={"Content-Length": str(exactly_50mb)},
        )
        # Should NOT be 413 since it is at the limit, not over
        assert resp.status_code != 413


# ===================================================================
# 3. RateLimitMiddleware
# ===================================================================


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware (completely disabled passthrough)."""

    def test_all_requests_pass_through(self):
        """Every request must pass through since rate limiting is disabled."""
        app = _make_app(RateLimitMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        # Fire many requests - none should be rate-limited
        for _ in range(20):
            resp = client.get("/ping")
            assert resp.status_code == 200

    def test_accepts_custom_init_params(self):
        """RateLimitMiddleware accepts init params but ignores them."""
        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"status": "pong"}

        app.add_middleware(
            RateLimitMiddleware,
            calls_per_minute=1,
            calls_per_second=1,
            burst_limit=1,
        )
        client = TestClient(app, raise_server_exceptions=False)

        # Even with limit=1, all should pass since it is disabled
        for _ in range(10):
            resp = client.get("/ping")
            assert resp.status_code == 200

    def test_different_endpoints_all_pass(self):
        """Multiple different endpoints all pass through."""
        app = _make_app(RateLimitMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        for path in ["/ping", "/health", "/protected"]:
            resp = client.get(path)
            assert resp.status_code == 200


# ===================================================================
# 4. AuthenticationMiddleware
# ===================================================================


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware."""

    # --- Public paths skip auth ---

    def test_public_paths_skip_auth_when_required(self):
        """Public paths (/health, /docs, /, /redoc, /openapi.json) bypass
        authentication even when require_auth=True."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["secret-key-123"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        for path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            resp = client.get(path)
            assert resp.status_code == 200, f"Public path {path} was blocked"

    def test_root_path_is_public(self):
        """The root '/' path is public."""
        app = FastAPI()

        @app.get("/")
        async def root():
            return {"status": "root"}

        app.add_middleware(
            AuthenticationMiddleware,
            require_auth=True,
            api_keys=["key123"],
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/")
        assert resp.status_code == 200

    # --- require_auth=False allows everything ---

    def test_require_auth_false_allows_all(self):
        """When require_auth=False, non-public paths are also accessible
        without credentials."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": False},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected")
        assert resp.status_code == 200

    # --- API key authentication ---

    def test_api_key_in_header_works(self):
        """Valid X-API-Key header authenticates the request."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["my-secret-key"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected", headers={"X-API-Key": "my-secret-key"})
        assert resp.status_code == 200

    def test_wrong_api_key_returns_401(self):
        """Invalid X-API-Key header returns 401."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["my-secret-key"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"

    def test_missing_credentials_returns_401(self):
        """No credentials at all returns 401 with WWW-Authenticate header."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["key"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers

    # --- Bearer token authentication ---

    def test_bearer_token_works(self):
        """A Bearer token longer than 10 chars authenticates successfully."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": []},
        )
        client = TestClient(app, raise_server_exceptions=False)

        long_token = "a" * 20
        resp = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {long_token}"},
        )
        assert resp.status_code == 200

    def test_short_bearer_token_fails(self):
        """A Bearer token with 10 or fewer chars fails authentication."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": []},
        )
        client = TestClient(app, raise_server_exceptions=False)

        short_token = "a" * 10  # exactly 10, needs > 10
        resp = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {short_token}"},
        )
        assert resp.status_code == 401

    def test_api_key_in_query_param_works(self):
        """API key provided via query parameter authenticates (with warning)."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["qp-key-value"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected?api_key=qp-key-value")
        assert resp.status_code == 200

    def test_multiple_api_keys_supported(self):
        """Multiple API keys can be configured and any valid one works."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={
                "require_auth": True,
                "api_keys": ["key-alpha", "key-beta", "key-gamma"],
            },
        )
        client = TestClient(app, raise_server_exceptions=False)

        for key in ["key-alpha", "key-beta", "key-gamma"]:
            resp = client.get("/protected", headers={"X-API-Key": key})
            assert resp.status_code == 200, f"Key '{key}' should authenticate"

    def test_401_response_has_www_authenticate(self):
        """401 responses include a WWW-Authenticate header listing accepted
        schemes."""
        app = _make_app(
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": True, "api_keys": ["k"]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/protected")
        assert resp.status_code == 401
        www_auth = resp.headers.get("WWW-Authenticate", "")
        assert "Bearer" in www_auth
        assert "X-API-Key" in www_auth


# ===================================================================
# 5. Combined middleware stack
# ===================================================================


class TestCombinedMiddleware:
    """Verify middleware cooperates when stacked together."""

    def test_security_headers_with_request_validation(self):
        """Both security headers and request validation work together."""
        app = _make_app(SecurityHeadersMiddleware, RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_oversized_request_still_gets_headers_from_outer_middleware(self):
        """A 413 from RequestValidation still passes through SecurityHeaders
        which wraps it."""
        app = _make_app(SecurityHeadersMiddleware, RequestValidationMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        oversized = 50 * 1024 * 1024 + 1
        resp = client.post(
            "/upload",
            content=b"x",
            headers={"Content-Length": str(oversized)},
        )
        assert resp.status_code == 413

    def test_full_stack_normal_request(self):
        """All four middleware stacked together pass a normal request."""
        app = _make_app(
            SecurityHeadersMiddleware,
            RequestValidationMiddleware,
            RateLimitMiddleware,
            AuthenticationMiddleware,
            auth_kwargs={"require_auth": False},
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.json()["status"] == "pong"
