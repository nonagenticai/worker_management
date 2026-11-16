"""Worker Management API main application module.

This module sets up the FastAPI application for the worker management system,
configures database initialization, error handling, and includes all API routers.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, cast

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.error_handlers import (
    APIError,
    api_error_handler,
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from src.api.prompts import repository as prompt_repository
from src.api.router import api_router
from src.api.security_middleware import (
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.tenant_constants import default_tenant_uuid
from src.api.workers.prompt_drift import start_drift_task
from src.db.db_config import close_pg_pool, init_pg_pool, run_migrations
from src.services.tenant_llm_audit_retention import start_audit_retention_task
from src.services.worker_states_retention import start_retention_task
from src.settings import get_settings
from src.telemetry import get_metrics_client
from src.telemetry.metrics_server import start_metrics_server, stop_metrics_server

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Configure logging with health endpoint filter
class HealthCheckFilter(logging.Filter):
    """Filter out health check endpoint logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for health check requests to skip logging them."""
        # Check if the log message contains health endpoint references
        message = record.getMessage()
        return not ("/health" in message or '"/health' in message)


from src.utils.log_format import JsonLogFormatter

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# THIS is the module the container actually starts:
#   CMD ["python","-m","uvicorn","src.api.worker_management_api:app", ...]
#
# `api/app_config.py` has its own configure_logging() and its own module-level
# call, and it is NOT on this path -- nothing here imports it. #176 wired
# LOG_FORMAT into that one, which is why the deployed pod kept emitting plain
# text with LOG_FORMAT=json set and settings.log_format == "json". Verified on
# the pod: the setting was read correctly; the handler was simply installed in a
# module that never loads. Bounding the payload worked in the same PR because
# that code IS on this path.
#
# So the format switch lives here, next to the basicConfig it has to override.
if os.getenv("LOG_FORMAT", "text").lower() == "json":
    _json_handler = logging.StreamHandler()
    _json_handler.setFormatter(JsonLogFormatter())
    # REPLACE, do not append: basicConfig already installed a text handler and
    # adding a second one emits every record twice, doubling the very log volume
    # #175 and #176 exist to reduce.
    logging.getLogger().handlers = [_json_handler]

logger = logging.getLogger(__name__)

# Add health check filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckFilter())

# Load environment variables (but don't override existing ones)
load_dotenv(override=False)

# Now proceed with OpenTelemetry and FastAPI imports with conditional loading
# Initialize OpenTelemetry components with conditional imports
otelemetry_available = False
tracer = None
FastAPIInstrumentor = None

try:
    from opentelemetry import trace  # isort:skip
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # isort:skip
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # isort:skip
    from opentelemetry.sdk.trace import TracerProvider  # isort:skip
    from opentelemetry.sdk.trace.export import (  # isort:skip
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    try:
        from opentelemetry.instrumentation.fastapi import (  # isort:skip
            FastAPIInstrumentor,
        )

        otelemetry_available = True
        logger.info("OpenTelemetry FastAPI instrumentation available")
    except ImportError as e:
        logger.warning(f"OpenTelemetry FastAPI instrumentation not available: {e}")

        # Create a no-op FastAPIInstrumentor class
        class NoOpFastAPIInstrumentor:
            """No-op replacement when the OpenTelemetry FastAPI instrumentor is unavailable."""

            @staticmethod
            def instrument_app(app: Any) -> None:  # noqa: ARG004, RUF100, D102
                logger.info("OpenTelemetry instrumentation disabled - no-op instrumentation")

        FastAPIInstrumentor: Any = NoOpFastAPIInstrumentor

except ImportError as e:
    logger.warning(f"OpenTelemetry core modules not available: {e}")

    # Create no-op implementations
    class NoOpTrace:
        """No-op replacement for opentelemetry.trace when OpenTelemetry is unavailable."""

        def get_tracer(self, name: str) -> NoOpTrace:  # noqa: ARG002, RUF100, D102
            return self

        def set_tracer_provider(self, provider: Any) -> None:  # noqa: D102
            pass

    class NoOpFastAPIInstrumentor:
        """No-op replacement when the OpenTelemetry FastAPI instrumentor is unavailable."""

        @staticmethod
        def instrument_app(app: Any) -> None:  # noqa: ARG004, RUF100, D102
            logger.info("OpenTelemetry not available - no instrumentation applied")

    trace: Any = NoOpTrace()
    FastAPIInstrumentor: Any = NoOpFastAPIInstrumentor

# --- OpenTelemetry Tracer Setup ---
# Check if OpenTelemetry is explicitly disabled
otel_disabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"

# Only set up tracing if OpenTelemetry is fully available and not disabled
if otelemetry_available and not otel_disabled:
    try:
        # Get OTLP endpoint from environment
        otlp_endpoint = os.getenv("OTLP_TRACES_ENDPOINT")

        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: "worker-management-service"})
        )

        # Only add OTLP exporter if endpoint is explicitly configured
        if otlp_endpoint:
            try:
                # Test connection first with a short timeout
                import httpx

                test_url = otlp_endpoint.replace("/v1/traces", "/")
                try:
                    httpx.get(test_url, timeout=2)
                    logger.debug(f"Successfully connected to OTLP endpoint: {test_url}")
                except Exception as conn_err:
                    logger.warning(
                        f"Cannot reach Grafana at {test_url}: {conn_err}. "
                        f"Traces will not be exported, but service will continue normally."
                    )
                    raise  # Re-raise to skip adding the processor

                # Create exporter with timeout configuration
                exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    timeout=5,  # 5 second timeout for export
                )
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=2048,
                    schedule_delay_millis=5000,
                    max_export_batch_size=512,
                    export_timeout_millis=30000,
                )
                provider.add_span_processor(processor)
                logger.info(f"OpenTelemetry OTLP exporter configured for {otlp_endpoint}")
            except Exception as e:
                logger.warning(
                    f"Grafana unavailable - OTLP exporter disabled. Service continues without tracing. Error: {e}"
                )
                # Don't add the OTLP processor, service will work without it

        # Add console exporter for local debugging if no OTLP endpoint or debug enabled
        if not otlp_endpoint or os.getenv("OTEL_DEBUG", "false").lower() == "true":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            logger.info("OpenTelemetry console exporter configured for debugging")

        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry tracing configured successfully")
    except Exception as e:
        logger.warning(
            f"Failed to configure OpenTelemetry tracing: {e}. Service continues without tracing."
        )
        # Ensure we have a basic tracer even if setup fails
        tracer = None
elif otel_disabled:
    logger.info("OpenTelemetry tracing explicitly disabled via OTEL_SDK_DISABLED")
    tracer = None
else:
    logger.info("OpenTelemetry tracing disabled - modules not available")
    tracer = None
# --- End Tracer Setup ---

# --- FastAPI and other application imports ---
# (Imports moved to top of file)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    # Startup
    #
    # The DEDICATED plaintext Prometheus listener, on its OWN port — bound
    # FIRST, before anything that can fail. Deliberately NOT a /metrics route on
    # the API port: see src/telemetry/metrics_server.py for why relaxing 8887
    # would expose the worker-creating API to the whole pod CIDR.
    #
    # Binding it before init_pg_pool() is on purpose: a pod that cannot reach
    # Postgres is exactly the pod whose metrics an operator most wants, and this
    # call is fail-soft so it can never itself be the reason startup dies.
    #
    # ⚠️ get_metrics_client() is what BUILDS the MeterProvider that the reader
    # is attached to. Calling it here forces that construction — and the zero
    # series it pre-materialises — at startup rather than on the first request
    # that happens to record something.
    start_metrics_server(app, get_settings().metrics_port)
    get_metrics_client()
    # The WM->inference-router prometheus_client counters live in modules that
    # used to be imported lazily (first IR call), so their series were absent
    # until first use. Import them here and materialise every label set at 0.
    from src.api.auth_middleware import preinit_actor_missing_metrics
    from src.services.ir_bundle_tokens import preinit_tenantless_refusal_metrics
    from src.worker.agents.pydantic_ai_agent import preinit_ir_identity_metrics

    preinit_tenantless_refusal_metrics()
    preinit_ir_identity_metrics()
    preinit_actor_missing_metrics()
    from src.telemetry.attribution_metrics import preinit_attribution_metrics

    preinit_attribution_metrics()

    try:
        # Initialize PostgreSQL connection pool
        await init_pg_pool()
        logger.info("PostgreSQL connection pool initialized successfully")
    except Exception:
        logger.exception("Failed to initialize databases")
        raise

    # Apply schema migrations from migrations/*.sql. Idempotent — every
    # migration uses ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS,
    # so re-running on every pod start is safe. Skipping this is what
    # caused prod 500 on GET /workers (workers.tenant_id missing because
    # the gitops ConfigMap-based init container only carried 001-007).
    #
    # Non-strict mode by default: a forward safety belt so a single bad
    # migration logs and keeps going instead of crashing pod startup. As of
    # the 001 removal the chain is fully Postgres-clean (001 was the only
    # SQLite-flavored file; its objects already live in postgres_schema.sql),
    # so a strict run is green too. Set WORKER_MGMT_MIGRATIONS_STRICT=1 to
    # fail-fast instead.
    if os.environ.get("WORKER_MGMT_SKIP_MIGRATIONS", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            applied, failed = await run_migrations()
            logger.info("Schema migrations: applied=%s failed=%s", applied, failed)
        except psycopg.Error, OSError, RuntimeError:
            logger.exception("Failed to apply schema migrations")
            raise

    # Once per process: is migration 031 applied? The runner above is
    # non-strict in production, so a failed 031 would otherwise surface only as
    # a CheckViolation 500 on the first prompt write. The result is cached and
    # read by the prompt write routes (503 while ``missing_031``) and /health.
    try:
        state = await prompt_repository.probe_lifecycle_schema()
        log = logger.error if state == prompt_repository.SCHEMA_MISSING_031 else logger.info
        log("prompt lifecycle schema: %s", state)
    except Exception:
        logger.warning("prompt lifecycle schema probe failed; state stays unknown", exc_info=True)

    # Start the worker_states retention sweep. It does its first pass BEFORE
    # its first sleep (see retention_loop's docstring), so a pod that rolls
    # more often than the interval still reaps.
    retention_task = start_retention_task()
    # tenant_llm_config_audit retention (WM-5): daily, same loop shape.
    audit_retention_task = start_audit_retention_task()

    # Prompt-binding drift metric (#314): config.system_prompt vs the bound
    # PUBLISHED version, refreshed in the background (first refresh before the
    # first sleep), never per message. None when WM_PROMPT_DRIFT_INTERVAL_S=0.
    drift_task = start_drift_task()

    yield

    # Shutdown
    logger.info("Application shutting down")
    for task in (retention_task, audit_retention_task, drift_task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    try:
        await close_pg_pool()
        logger.info("PostgreSQL connection pool closed successfully")
    except Exception:
        logger.exception("Error closing PostgreSQL pool")
    stop_metrics_server(app)


app = FastAPI(
    title="Worker Management API",
    description=(
        "Multi-tenant API for creating LLM-powered workers, sending messages to agents, "
        "tracking worker state, and executing DAG workflow graphs. "
        "Supports five agent types (pydantic-ai, claude-code, codex, remote-claude, remote-codex) "
        # No provider count here on purpose: the previous "six model providers"
        # had already drifted (it omitted anthropic, google, vllm and vlm_local)
        # and adding xai would only have made it wrong by a different number.
        # ModelProvider in worker.agents.pydantic_ai_agent is the source of truth.
        "and model providers spanning hosted APIs (OpenAI, Anthropic, DeepSeek, Groq, "
        "Cohere, Google, xAI) plus self-hosted backends (vLLM, MLX, local VLM, "
        "inference-router). "
        "Workers produce structured output validated against JSON Schema. "
        "All data is tenant-isolated via license-token authentication."
    ),
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url=None,  # We provide a custom docs route to inject tenant header
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {
            "name": "Workers",
            "description": "Operations with workers and agents. Workers can be simple task processors or AI agents.",
        },
        {
            "name": "Graphs",
            "description": "Operations with workflow graphs. Graphs define connections between workers and tasks.",
        },
        {
            "name": "Graph Execution",
            "description": "Execute workflow graphs with specified parameters and monitor execution.",
        },
        {
            "name": "Tool Registry",
            "description": "Manage custom tools and functions that can be used by workers.",
        },
        {
            "name": "Worker States",
            "description": "Track and manage worker execution states and history.",
        },
        {
            "name": "Agents",
            "description": "Operations for managing agent workload information and availability.",
        },
    ],
    responses={
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "example": "Invalid UUID format. Expected UUID, got: invalid-uuid",
                            }
                        },
                    }
                }
            },
        },
        404: {
            "description": "Not Found",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string", "example": "Worker not found"}},
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "example": "Internal server error",
                            }
                        },
                    }
                }
            },
        },
    },
)

# Add CORS middleware (must be first for proper CORS handling)
raw_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8921,http://localhost:3000,http://127.0.0.1:3000",
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    cast("Any", CORSMiddleware),
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Add security middleware (order matters - these run in reverse order of addition)
app.add_middleware(cast("Any", SecurityHeadersMiddleware))
app.add_middleware(cast("Any", RequestValidationMiddleware))

# Optional authentication middleware (enabled by default, toggle with ENABLE_AUTH=false)
if os.getenv("ENABLE_AUTH", "true").lower() == "true":
    try:
        app.add_middleware(cast("Any", AuthenticationMiddleware), bypass_auth=False)
        logger.info("Authentication middleware enabled")
    except Exception:
        logger.exception("Failed to enable authentication middleware")
# Rate limiting COMPLETELY DISABLED for schemathesis testing
# app.add_middleware(
#     RateLimitMiddleware,
#     calls_per_minute=2000,  # 2000 calls per minute per IP for production
#     calls_per_second=100,   # 100 calls per second per IP
#     burst_limit=200         # 200 burst limit for handling traffic spikes
# )
# Register exception handlers
app.add_exception_handler(RequestValidationError, cast("Any", validation_exception_handler))
app.add_exception_handler(HTTPException, cast("Any", http_exception_handler))
# Also handle Starlette HTTPException (in case of internal routing)
from starlette.exceptions import HTTPException as StarletteHTTPException

app.add_exception_handler(StarletteHTTPException, cast("Any", http_exception_handler))
app.add_exception_handler(APIError, cast("Any", api_error_handler))
# Register general exception handler last to catch all unhandled exceptions
app.add_exception_handler(Exception, general_exception_handler)

# Instrument the FastAPI app with OpenTelemetry (must be done before including routers)
if FastAPIInstrumentor:
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI app instrumented with OpenTelemetry")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI app with OpenTelemetry: {e}")
else:
    logger.info("OpenTelemetry FastAPI instrumentation not available")

# Add SQLAlchemy and httpx instrumentation for full observability
if otelemetry_available and not otel_disabled:
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumented with OpenTelemetry")
    except ImportError:
        logger.warning("SQLAlchemy instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to instrument SQLAlchemy: {e}")

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("httpx instrumented with OpenTelemetry")
    except ImportError:
        logger.warning("httpx instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to instrument httpx: {e}")


# Custom Swagger UI with default tenant header
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> Any:
    """Serve the Swagger UI page with a default X-Tenant-ID request header injected."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Worker Management API Docs",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "requestInterceptor": (
                "function(req){req.headers['X-Tenant-ID']='"
                + default_tenant_uuid()
                + "';return req;}"
            ),
        },
    )


# Include the main API router
app.include_router(api_router)


# Add Bearer security scheme so Swagger shows the Authorize button
def custom_openapi() -> dict[str, Any]:
    """Return the OpenAPI schema with a Bearer security scheme injected for Swagger."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

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
            "description": "Enter license token as 'Bearer <token>'",
        },
    )
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return openapi_schema


cast("Any", app).openapi = custom_openapi


# Add root endpoint for health check
@app.get("/", operation_id="root")
async def root(request: Request) -> dict[str, str]:
    """Root endpoint for health check with query parameter validation."""
    # Validate query parameters for security
    if request.url.query:
        from urllib.parse import parse_qsl

        from src.api.security_validation import (
            SecurityValidationError,
            validate_query_parameters,
        )

        try:
            # Parse and validate query parameters
            parsed_params = dict(parse_qsl(request.url.query, keep_blank_values=True))
            validate_query_parameters(parsed_params)
        except SecurityValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid query parameters: {e!s}") from e

    return {"message": "Worker Management API is running", "version": "1.0.0"}


@app.get("/health", operation_id="health_check")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint with query parameter validation."""
    # Validate query parameters for security
    if request.url.query:
        from urllib.parse import parse_qsl

        from src.api.security_validation import (
            SecurityValidationError,
            validate_query_parameters,
        )

        try:
            # Parse and validate query parameters
            parsed_params = dict(parse_qsl(request.url.query, keep_blank_values=True))
            validate_query_parameters(parsed_params)
        except SecurityValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid query parameters: {e!s}") from e

    # ``prompts_read_requires_published`` is the LIVE value of the prompt
    # lifecycle flag (P4 contract §3), so D7 and the grid can key on what this
    # pod actually does rather than on a manifest that may not have rolled.
    return {
        "status": "healthy",
        "service": "worker-management-api",
        "prompts_read_requires_published": get_settings().prompts_read_requires_published,
        # ready | missing_031 | unknown — the startup probe of migration 031
        # (prompt write routes answer 503 while it reads missing_031).
        "prompt_lifecycle_schema": prompt_repository.lifecycle_schema_state(),
    }


@app.get("/health/live", operation_id="liveness_check")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready", operation_id="readiness_check")
async def readiness_check() -> Any:
    """Kubernetes readiness probe with database connectivity check."""
    from src.db.db_config import get_pg_pool

    checks = {}

    # Check database connectivity
    try:
        pool = get_pg_pool()
        if pool is None:
            checks["database"] = False
        else:
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT 1 AS ok")
                checks["database"] = cur.rowcount != 0
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        checks["database"] = False

    # All checks must pass
    all_healthy = all(checks.values())

    if not all_healthy:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}
