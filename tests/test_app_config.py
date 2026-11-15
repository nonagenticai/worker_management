"""Tests for src.api.app_config module.

Covers:
- Health endpoint returns 200 with status=healthy
- HealthCheckFilter filters /health log records
- HealthCheckFilter passes non-health log records
- create_secure_app returns a FastAPI instance
- App has routes registered
- Global exception handler returns 500 with INTERNAL_ERROR code
"""

import logging
import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registered_paths(app: FastAPI) -> set[str]:
    """Return the set of registered route paths.

    FastAPI >= 0.137 wraps ``include_router`` results in ``_IncludedRouter``
    objects that have no ``.path`` attribute, so deriving paths from the
    OpenAPI schema (stable across versions) is required for included routers.
    """
    paths = set(app.openapi().get("paths", {}).keys())
    for route in app.routes:
        p = getattr(route, "path", None)
        if p:
            paths.add(p)
    return paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_app():
    """Create a fresh FastAPI app via create_secure_app with auth disabled."""
    import src.api.app_config as app_cfg

    with (
        patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "false",
                "ENABLE_RATE_LIMITING": "false",
                "ENVIRONMENT": "production",
            },
        ),
        patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
        patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
    ):
        yield app_cfg.create_secure_app()


@pytest.fixture()
def client(fresh_app):
    """TestClient wrapping the fresh app."""
    return TestClient(fresh_app, raise_server_exceptions=False)


@pytest.fixture()
def dev_app():
    """App created in development mode (exception details exposed)."""
    import src.api.app_config as app_cfg

    with (
        patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "false",
                "ENABLE_RATE_LIMITING": "false",
                "ENVIRONMENT": "development",
            },
        ),
        patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
        patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
    ):
        yield app_cfg.create_secure_app()


@pytest.fixture()
def dev_client(dev_app):
    """TestClient in development mode."""
    return TestClient(dev_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status_healthy(self, client: TestClient) -> None:
        """Response body must include status=healthy."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_contains_version(self, client: TestClient) -> None:
        """Response body must include a version string."""
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_contains_environment(self, client: TestClient) -> None:
        """Response body must include the environment field."""
        data = client.get("/health").json()
        assert "environment" in data

    def test_health_respects_api_version_env(self) -> None:
        """API_VERSION env var should be reflected in the health response.

        The module-level API_VERSION constant is captured at import time.
        We must also patch the module attribute so create_secure_app picks
        it up when building the health endpoint closure.
        """
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_AUTH": "false",
                    "ENABLE_RATE_LIMITING": "false",
                },
            ),
            patch.object(app_cfg, "API_VERSION", "42.0.0"),
        ):
            app = app_cfg.create_secure_app()
            tc = TestClient(app, raise_server_exceptions=False)
            data = tc.get("/health").json()
            assert data["version"] == "42.0.0"


# ---------------------------------------------------------------------------
# 2 & 3. HealthCheckFilter
# ---------------------------------------------------------------------------


class TestHealthCheckFilter:
    """Tests for HealthCheckFilter logging filter."""

    def _make_record(self, msg: str) -> logging.LogRecord:
        """Helper to create a LogRecord with a given message."""
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_filters_health_path(self) -> None:
        """Records containing '/health' should be filtered out (return False)."""
        from src.api.app_config import HealthCheckFilter

        f = HealthCheckFilter()
        record = self._make_record("GET /health HTTP/1.1 200")
        assert f.filter(record) is False

    def test_filters_quoted_health_path(self) -> None:
        """Records containing '"/health' should also be filtered out."""
        from src.api.app_config import HealthCheckFilter

        f = HealthCheckFilter()
        record = self._make_record('127.0.0.1 - "GET /health HTTP/1.1" 200')
        assert f.filter(record) is False

    def test_passes_non_health_record(self) -> None:
        """Records that do NOT mention /health should pass (return True)."""
        from src.api.app_config import HealthCheckFilter

        f = HealthCheckFilter()
        record = self._make_record("GET /workers HTTP/1.1 200")
        assert f.filter(record) is True

    def test_passes_empty_message(self) -> None:
        """An empty log message should pass the filter."""
        from src.api.app_config import HealthCheckFilter

        f = HealthCheckFilter()
        record = self._make_record("")
        assert f.filter(record) is True

    def test_passes_unrelated_message(self) -> None:
        """A completely unrelated log message should pass."""
        from src.api.app_config import HealthCheckFilter

        f = HealthCheckFilter()
        record = self._make_record("Application startup complete")
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# 4. create_secure_app returns FastAPI instance
# ---------------------------------------------------------------------------


class TestCreateSecureApp:
    """Tests for the create_secure_app factory function."""

    def test_returns_fastapi_instance(self, fresh_app: FastAPI) -> None:
        """create_secure_app must return a FastAPI object."""
        assert isinstance(fresh_app, FastAPI)

    def test_app_title(self, fresh_app: FastAPI) -> None:
        """App title should be 'Worker Management API'."""
        assert fresh_app.title == "Worker Management API"

    def test_docs_disabled_by_default(self) -> None:
        """When DEBUG and ENABLE_DOCS are false, docs endpoints are None."""
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_AUTH": "false",
                    "ENABLE_RATE_LIMITING": "false",
                },
            ),
            patch.object(app_cfg, "DEBUG", False),
            patch.object(app_cfg, "ENABLE_DOCS", False),
        ):
            app = app_cfg.create_secure_app()
            assert app.docs_url is None
            assert app.redoc_url is None

    def test_docs_enabled_when_debug(self) -> None:
        """When DEBUG=true, docs endpoints should be available."""
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_AUTH": "false",
                    "ENABLE_RATE_LIMITING": "false",
                },
            ),
            patch.object(app_cfg, "DEBUG", True),
        ):
            app = app_cfg.create_secure_app()
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"


# ---------------------------------------------------------------------------
# 5. App has routes registered
# ---------------------------------------------------------------------------


class TestAppRoutes:
    """Tests verifying routes are registered on the app."""

    def test_routes_not_empty(self, fresh_app: FastAPI) -> None:
        """The application must have at least one route registered."""
        assert len(fresh_app.routes) > 0

    def test_health_route_exists(self, fresh_app: FastAPI) -> None:
        """A route matching '/health' must exist."""
        assert "/health" in _registered_paths(fresh_app)

    def test_workers_route_exists(self, fresh_app: FastAPI) -> None:
        """The /workers route from the main router must be present."""
        assert "/workers" in _registered_paths(fresh_app)


# ---------------------------------------------------------------------------
# 6. Global exception handler
# ---------------------------------------------------------------------------


class TestGlobalExceptionHandler:
    """Tests for the global exception handler.

    Note: the SecurityHeadersMiddleware wraps the ASGI stack and catches
    unhandled exceptions before FastAPI's global exception handler can see
    them. Its catch-all returns a generic 500 with code INTERNAL_ERROR.
    We therefore assert on the middleware's response shape, which is the
    actual production behaviour.
    """

    def test_exception_handler_returns_500(self, client: TestClient, fresh_app: FastAPI) -> None:
        """An unhandled exception should result in a 500 response."""

        @fresh_app.get("/test-crash")
        async def crash():
            raise RuntimeError("something broke")

        response = client.get("/test-crash")
        assert response.status_code == 500

    def test_exception_handler_returns_internal_error_code(
        self, client: TestClient, fresh_app: FastAPI
    ) -> None:
        """The 500 body must contain error.code == 'INTERNAL_ERROR'."""

        @fresh_app.get("/test-crash-code")
        async def crash_code():
            raise RuntimeError("kaboom")

        response = client.get("/test-crash-code")
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["status"] == 500

    def test_production_hides_details(self, client: TestClient, fresh_app: FastAPI) -> None:
        """In production, the error message must NOT leak the exception string.

        The SecurityHeadersMiddleware intercepts unhandled exceptions and
        returns a generic message. Verify the actual detail is hidden.
        """

        @fresh_app.get("/test-crash-prod")
        async def crash_prod():
            raise RuntimeError("secret internal detail")

        response = client.get("/test-crash-prod")
        body = response.json()
        # Neither the middleware nor the global handler should expose the real error
        assert "secret internal detail" not in body["error"]["message"]

    def test_development_shows_generic_error(
        self, dev_client: TestClient, dev_app: FastAPI
    ) -> None:
        """Even in development mode, the middleware returns a generic 500.

        The SecurityHeadersMiddleware catches exceptions before FastAPI's
        global handler can add dev-mode details. The response still has the
        correct INTERNAL_ERROR structure.
        """

        @dev_app.get("/test-crash-dev")
        async def crash_dev():
            raise ValueError("visible detail for devs")

        response = dev_client.get("/test-crash-dev")
        body = response.json()
        assert response.status_code == 500
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["status"] == 500
