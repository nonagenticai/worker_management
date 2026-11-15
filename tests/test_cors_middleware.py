"""Tests for CORS middleware (src/api/cors_middleware.py).

Covers SecureCORSMiddleware dispatch logic, preflight handling,
CORSSecurityValidator, and configuration helpers.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from src.api.cors_middleware import (
    ALLOWED_HEADERS,
    ALLOWED_METHODS,
    ALLOWED_ORIGINS_DEVELOPMENT,
    ALLOWED_ORIGINS_PRODUCTION,
    CORSSecurityValidator,
    SecureCORSMiddleware,
    get_cors_middleware_config,
    setup_cors_middleware,
)

# ---------------------------------------------------------------------------
# Helpers - build a tiny FastAPI app with the middleware under test
# ---------------------------------------------------------------------------


def _make_app(
    allowed_origins=None,
    allowed_origin_regex=None,
    is_production=False,
    allowed_methods=None,
    allowed_headers=None,
) -> FastAPI:
    """Create a minimal FastAPI app with SecureCORSMiddleware attached."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return JSONResponse({"ok": True})

    @app.post("/data")
    async def post_data():
        return JSONResponse({"created": True})

    app.add_middleware(
        SecureCORSMiddleware,
        allowed_origins=allowed_origins or ["https://allowed.example.com"],
        allowed_origin_regex=allowed_origin_regex,
        allowed_methods=allowed_methods or list(ALLOWED_METHODS),
        allowed_headers=allowed_headers or list(ALLOWED_HEADERS),
        allow_credentials=True,
        max_age=600,
        is_production=is_production,
    )

    return app


@pytest.fixture
def client():
    """TestClient with a simple allowed origin list."""
    return TestClient(_make_app())


@pytest.fixture
def prod_client():
    """TestClient configured for production mode."""
    return TestClient(
        _make_app(
            allowed_origins=["https://prod.example.com"],
            is_production=True,
        )
    )


# ======================================================================
# 1. No Origin header -> response passes through without CORS headers
# ======================================================================


class TestNoOriginHeader:
    """Requests without an Origin header must not receive CORS headers."""

    def test_get_without_origin(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers
        assert resp.json() == {"ok": True}

    def test_post_without_origin(self, client):
        resp = client.post("/data")
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers


# ======================================================================
# 2. Allowed origin -> CORS headers added
# ======================================================================


class TestAllowedOrigin:
    """When the origin is on the allowed list, CORS headers must appear."""

    def test_allowed_origin_adds_allow_origin(self, client):
        origin = "https://allowed.example.com"
        resp = client.get("/ping", headers={"Origin": origin})
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == origin

    def test_allowed_origin_adds_vary(self, client):
        origin = "https://allowed.example.com"
        resp = client.get("/ping", headers={"Origin": origin})
        assert "Origin" in resp.headers.get("vary", "")

    def test_allowed_origin_adds_credentials(self, client):
        origin = "https://allowed.example.com"
        resp = client.get("/ping", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_allowed_origin_adds_expose_headers(self, client):
        origin = "https://allowed.example.com"
        resp = client.get("/ping", headers={"Origin": origin})
        assert "access-control-expose-headers" in resp.headers


# ======================================================================
# 3. Disallowed origin -> only Vary header, no Allow-Origin
# ======================================================================


class TestDisallowedOrigin:
    """When the origin is NOT allowed, only a Vary header should appear."""

    def test_disallowed_origin_no_allow_origin(self, client):
        resp = client.get("/ping", headers={"Origin": "https://evil.example.com"})
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_disallowed_origin_has_vary(self, client):
        resp = client.get("/ping", headers={"Origin": "https://evil.example.com"})
        assert "Origin" in resp.headers.get("vary", "")

    def test_disallowed_origin_no_credentials(self, client):
        resp = client.get("/ping", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-credentials" not in resp.headers

    def test_disallowed_origin_production_rejects_localhost(self, prod_client):
        """Production mode must not auto-allow localhost."""
        resp = prod_client.get("/ping", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in resp.headers


# ======================================================================
# 4. Preflight OPTIONS -> 200 with full CORS headers
# ======================================================================


class TestPreflightAllowed:
    """OPTIONS with an allowed origin must return 200 and full CORS headers."""

    def test_preflight_status_200(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200

    def test_preflight_returns_allow_origin(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers["access-control-allow-origin"] == origin

    def test_preflight_returns_allow_methods(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        methods = resp.headers["access-control-allow-methods"]
        assert "GET" in methods
        assert "POST" in methods

    def test_preflight_returns_allow_headers(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-headers" in resp.headers

    def test_preflight_returns_max_age(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers["access-control-max-age"] == "600"

    def test_preflight_returns_credentials(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers["access-control-allow-credentials"] == "true"

    def test_preflight_returns_vary(self, client):
        origin = "https://allowed.example.com"
        resp = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        vary = resp.headers.get("vary", "")
        assert "Origin" in vary


# ======================================================================
# 5. Preflight with disallowed origin -> 403
# ======================================================================


class TestPreflightDisallowedOrigin:
    """OPTIONS from a disallowed origin must return 403."""

    def test_preflight_disallowed_origin_returns_403(self, client):
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 403

    def test_preflight_disallowed_origin_has_vary(self, client):
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "Origin" in resp.headers.get("vary", "")

    def test_preflight_disallowed_origin_no_allow_origin(self, client):
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in resp.headers


# ======================================================================
# 6. Preflight with disallowed method -> 405
# ======================================================================


class TestPreflightDisallowedMethod:
    """OPTIONS requesting a method not in the allowed set must return 405."""

    def test_preflight_patch_not_allowed_returns_405(self, client):
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://allowed.example.com",
                "Access-Control-Request-Method": "PATCH",
            },
        )
        assert resp.status_code == 405

    def test_preflight_disallowed_method_body(self, client):
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://allowed.example.com",
                "Access-Control-Request-Method": "PATCH",
            },
        )
        assert b"Method not allowed" in resp.content


# ======================================================================
# 7. CORSSecurityValidator
# ======================================================================


class TestCORSSecurityValidator:
    """CORSSecurityValidator.validate_origin_security edge cases."""

    def test_rejects_ftp_scheme(self):
        assert CORSSecurityValidator.validate_origin_security("ftp://files.example.com") is False

    def test_rejects_empty_string(self):
        assert CORSSecurityValidator.validate_origin_security("") is False

    def test_rejects_ip_address_origin(self):
        assert CORSSecurityValidator.validate_origin_security("http://192.168.1.1") is False

    def test_rejects_ipv6_origin(self):
        assert CORSSecurityValidator.validate_origin_security("http://[::1]") is False

    def test_rejects_tk_tld(self):
        assert CORSSecurityValidator.validate_origin_security("https://evil.tk") is False

    def test_rejects_ml_tld(self):
        assert CORSSecurityValidator.validate_origin_security("https://evil.ml") is False

    def test_rejects_ga_tld(self):
        assert CORSSecurityValidator.validate_origin_security("https://evil.ga") is False

    def test_rejects_cf_tld(self):
        assert CORSSecurityValidator.validate_origin_security("https://evil.cf") is False

    def test_rejects_long_hostname(self):
        long_host = "a" * 254 + ".com"
        assert CORSSecurityValidator.validate_origin_security(f"https://{long_host}") is False

    def test_accepts_valid_https(self):
        assert CORSSecurityValidator.validate_origin_security("https://app.example.com") is True

    def test_accepts_valid_http(self):
        assert CORSSecurityValidator.validate_origin_security("http://app.example.com") is True

    def test_rejects_no_hostname(self):
        assert CORSSecurityValidator.validate_origin_security("http://") is False

    def test_rejects_data_scheme(self):
        assert CORSSecurityValidator.validate_origin_security("data:text/html,<h1>hi</h1>") is False


# ======================================================================
# 8. get_cors_middleware_config returns correct dict
# ======================================================================


class TestGetCorsMiddlewareConfig:
    """get_cors_middleware_config must return appropriate dicts."""

    def test_production_origins(self):
        cfg = get_cors_middleware_config(is_production=True)
        assert cfg["allow_origins"] == ALLOWED_ORIGINS_PRODUCTION

    def test_development_origins(self):
        cfg = get_cors_middleware_config(is_production=False)
        assert cfg["allow_origins"] == ALLOWED_ORIGINS_DEVELOPMENT

    def test_common_fields_present(self):
        cfg = get_cors_middleware_config(is_production=False)
        assert cfg["allow_credentials"] is True
        assert cfg["allow_methods"] == ALLOWED_METHODS
        assert cfg["allow_headers"] == ALLOWED_HEADERS
        assert cfg["max_age"] == 600

    def test_expose_headers_present(self):
        cfg = get_cors_middleware_config(is_production=True)
        assert "expose_headers" in cfg
        assert isinstance(cfg["expose_headers"], list)


# ======================================================================
# Extra: regex origin matching and localhost dev-mode fallback
# ======================================================================


class TestOriginRegexAndLocalhostFallback:
    """Additional dispatch-level integration tests."""

    def test_regex_origin_match(self):
        app = _make_app(
            allowed_origins=[],
            allowed_origin_regex=r"https://.*\.myapp\.io",
        )
        c = TestClient(app)
        resp = c.get("/ping", headers={"Origin": "https://tenant.myapp.io"})
        assert resp.headers["access-control-allow-origin"] == "https://tenant.myapp.io"

    def test_localhost_allowed_in_dev_mode(self):
        app = _make_app(
            allowed_origins=[],
            is_production=False,
        )
        c = TestClient(app)
        resp = c.get("/ping", headers={"Origin": "http://localhost:9999"})
        assert resp.headers["access-control-allow-origin"] == "http://localhost:9999"

    def test_localhost_blocked_in_production(self):
        app = _make_app(
            allowed_origins=["https://prod.example.com"],
            is_production=True,
        )
        c = TestClient(app)
        resp = c.get("/ping", headers={"Origin": "http://localhost:9999"})
        assert "access-control-allow-origin" not in resp.headers


# ======================================================================
# setup_cors_middleware smoke test
# ======================================================================


class TestSetupCorsMiddleware:
    """Ensure setup_cors_middleware attaches middleware without errors."""

    def test_setup_default_dev(self):
        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        setup_cors_middleware(app, is_production=False)
        c = TestClient(app)
        resp = c.get(
            "/ping",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_setup_custom_origins(self):
        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        setup_cors_middleware(
            app,
            is_production=True,
            custom_origins=["https://custom.example.com"],
        )
        c = TestClient(app)
        resp = c.get(
            "/ping",
            headers={"Origin": "https://custom.example.com"},
        )
        assert resp.headers["access-control-allow-origin"] == "https://custom.example.com"
