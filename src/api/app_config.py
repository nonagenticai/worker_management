"""Application configuration and security setup for Worker Management API.

This module configures all security middleware, authentication, and
production-ready settings for the FastAPI application.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.settings import get_settings
from src.utils.log_format import JsonLogFormatter

# Import security middleware
from .auth_middleware import AuthenticationMiddleware
from .rate_limiting import RateLimitMiddleware

# Import routers
from .router import api_router as main_router
from .security_middleware import (
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)
from .tenant_constants import default_tenant_uuid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
# ⚠️ DOCS ONLY — a second raw read of this env var, deliberately left unresolved.
# auth_middleware._default_tenant_id() is the tenant-bearing copy: it runs the
# same env through resolve_tenant_id (which now refuses the nil uuid) and raises
# at boot if it is unusable. This copy's only consumer is the OpenAPI X-Tenant-Id
# example below (~line 178), so it can never scope a request or reach the
# database — a bad value renders a bad example, nothing more.
# ⛔ If you ever make this feed a request tenant, resolve it first.
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID") or default_tenant_uuid()

# Security configuration
ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTH", "true").lower() == "true"
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting Worker Management API v{API_VERSION}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(
        f"Security features: Auth={ENABLE_AUTHENTICATION}, RateLimit={ENABLE_RATE_LIMITING}"
    )

    rate_limit_middleware = None
    if ENABLE_RATE_LIMITING:
        for middleware in app.user_middleware:
            # Starlette's user_middleware exposes _MiddlewareFactory[P], which
            # is callable but not typed as `type` — cast to the runtime form
            # ty expects on issubclass.
            if hasattr(middleware, "cls") and issubclass(
                cast("type", middleware.cls), RateLimitMiddleware
            ):
                rate_limit_middleware = middleware.cls
                break

        if rate_limit_middleware and hasattr(rate_limit_middleware, "startup"):
            startup_fn = cast("Any", rate_limit_middleware).startup
            await startup_fn()

    yield

    # Shutdown
    logger.info("Shutting down Worker Management API")

    if rate_limit_middleware and hasattr(rate_limit_middleware, "shutdown"):
        shutdown_fn = cast("Callable[[], Any]", rate_limit_middleware.shutdown)
        await shutdown_fn()


def create_secure_app() -> FastAPI:
    """Create FastAPI application with comprehensive security configuration.

    Returns:
        FastAPI: Configured application instance with security middleware, CORS, error handlers, and routers
    """
    # Create FastAPI app
    docs_enabled = DEBUG or ENABLE_DOCS

    app = FastAPI(
        title="Worker Management API",
        description="Secure API for managing workers, graphs, and execution workflows",
        version=API_VERSION,
        debug=DEBUG,
        lifespan=lifespan,
        # Security configurations
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # 1. CORS Middleware (must be first)
    # Internal service - use standard permissive CORS like other services
    # (nlp_service, ingestion_service, kag_orchestrator, etc.)
    app.add_middleware(
        cast("Any", CORSMiddleware),
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # 2. Security Headers Middleware (critical security layer)
    app.add_middleware(cast("Any", SecurityHeadersMiddleware))

    # 3. Request Validation Middleware (input validation)
    app.add_middleware(cast("Any", RequestValidationMiddleware))

    # 4. Rate Limiting Middleware
    if ENABLE_RATE_LIMITING:
        try:
            # Try distributed rate limiting first
            app.add_middleware(cast("Any", RateLimitMiddleware), redis_url=REDIS_URL)
            logger.info("Enabled distributed rate limiting with Redis")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiting: {e}")

    # 5. Authentication Middleware (if enabled)
    if ENABLE_AUTHENTICATION:
        app.add_middleware(
            cast("Any", AuthenticationMiddleware),
            bypass_auth=False,  # Set to True for testing
        )
        logger.info("Authentication middleware enabled")

    # Add Bearer security scheme so Swagger shows the Authorize button (when docs are enabled)
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        from .auth_middleware import PUBLIC_ENDPOINTS

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = openapi_schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.setdefault(
            "BearerAuth",
            {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "License",
                "description": (
                    "Paste the raw bearer token. Swagger UI sends it as "
                    "Authorization: Bearer <token>."
                ),
            },
        )

        # Reusable tenant header parameter
        parameters = components.setdefault("parameters", {})
        parameters.setdefault(
            "TenantIdHeader",
            {
                "name": "X-Tenant-ID",
                "in": "header",
                "required": False,
                "description": "Tenant identifier (defaults to configured DEFAULT_TENANT_ID).",
                "schema": {"type": "string", "default": DEFAULT_TENANT_ID},
                "example": DEFAULT_TENANT_ID,
            },
        )

        # Attach tenant header parameter to all operations if not present
        for path, path_item in openapi_schema.get("paths", {}).items():
            normalized = path.rstrip("/") or "/"
            is_public = normalized in PUBLIC_ENDPOINTS or normalized.startswith(
                ("/docs", "/redoc", "/static", "/openapi.json", "/health")
            )

            for operation in list(path_item.values()):
                if not isinstance(operation, dict):
                    continue
                if not is_public:
                    operation.setdefault("security", [{"BearerAuth": []}])
                else:
                    operation.pop("security", None)
                op_params = operation.setdefault("parameters", [])
                has_tenant = any(
                    p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header"
                    for p in op_params
                )
                if not has_tenant:
                    op_params.append({"$ref": "#/components/parameters/TenantIdHeader"})

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    cast("Any", app).openapi = custom_openapi

    # Include routers
    app.include_router(main_router)  # Main API endpoints

    # Global exception handler for security
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> Any:  # noqa: ARG001
        """Global exception handler to prevent information disclosure."""
        if isinstance(exc, HTTPException):
            raise exc

        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        # Don't expose internal details in production
        if ENVIRONMENT == "production":
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "status": 500,
                    }
                },
            )
        else:
            # In development, show more details
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": f"Internal server error: {exc!s}",
                        "status": 500,
                    }
                },
            )

    # Health check endpoint (always available)
    @app.get("/health", operation_id="app_health_check")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": API_VERSION, "environment": ENVIRONMENT}

    logger.info("Secure FastAPI application created successfully")
    return app


class HealthCheckFilter(logging.Filter):
    """Filter out health check endpoint logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for health check requests to skip logging them."""
        # Check if the log message contains health endpoint references
        message = record.getMessage()
        return not ("/health" in message or '"/health' in message)


def configure_logging() -> None:
    """Configure logging for the application.

    Sets up log levels, formats, and filters including health check endpoint filtering.

    Honours LOG_FORMAT. It was set to `json` on the deployment while NOTHING in
    src/ read it, so this service emitted plain text while w2v_orchestrator and
    requirements both emitted JSON -- a knob that silently did nothing, which is
    the same class of defect as the metric that shipped under the wrong name.
    """
    log_level = logging.DEBUG if DEBUG else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if get_settings().log_format.lower() == "json":
        json_handler = logging.StreamHandler()
        json_handler.setFormatter(JsonLogFormatter())
        root = logging.getLogger()
        # basicConfig already installed a text handler; replace rather than add,
        # or every record is emitted twice and the log doubles in size.
        root.handlers = [json_handler]

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Add health check filter to uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())

    if ENVIRONMENT == "production":
        # In production, reduce verbosity
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# Create the application instance
configure_logging()
app = create_secure_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_config:app", host="0.0.0.0", port=8887, reload=DEBUG, log_level="info")  # nosec B104 — container requires binding to all interfaces  # noqa: S104
