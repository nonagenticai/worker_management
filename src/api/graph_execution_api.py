"""Graph execution API module.

This module provides REST endpoints for executing graphs in both standard and stepwise modes.
It handles graph execution requests, parameter validation, and error handling for the workflow system.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.error_handlers import (
    BadRequestError,
    InternalServerError,
    NotFoundError,
)
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_GRAPH_EXECUTION_EXECUTE,
    ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE,
    ROUTE_GRAPH_EXECUTION_SESSION_STATUS,
    ROUTE_GRAPH_EXECUTION_SESSIONS_LIST,
    tokenless_tenant_gate,
)
from src.db.graph_execution_db import (
    GraphExecutionSessionDB,
    first_valid_correlation_id,
)
from src.worker.graphs.graph_models import ExecutionRequest, ExecutionResponse
from src.worker.graphs.pydantic_graph_execution_service import (
    GraphNotFoundError,
    get_execution_service,
)


def _get_db_name() -> str:
    try:
        url = make_url(os.getenv("DATABASE_URL", ""))
        return url.database or "unknown"
    except Exception:
        return "unknown"


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/execution", tags=["Graph Execution"])


@router.post(
    "/graphs/{graph_id}/execute",
    response_model=ExecutionResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 404, 405, 422, 500]}
    },
    operation_id="execute_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_EXECUTION_EXECUTE))],
)
async def execute_graph(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: ExecutionRequest,
    graph_id: str = Path(
        ...,
        description="UUID of the graph to execute",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-Id"),
) -> ExecutionResponse:
    """Execute a graph with specified parameters for a specific tenant.

    Args:
        request: The execution request containing variables and configuration
        graph_id: UUID of the graph to execute
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        ExecutionResponse: Result of the graph execution including outputs and status

    Raises:
        HTTPException: 400 for validation errors, 404 for graph not found,
                      405 for method not allowed, 422 for request validation errors,
                      500 for execution errors
    """
    try:
        # Validate UUID format since we now use str type
        try:
            UUID(graph_id)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Invalid UUID format for graph_id: {graph_id}"
            ) from None

        service = get_execution_service()

        # Header wins on disagreement (per spec §5.2.1) — but only among values
        # that could actually BE the join key. A plain `header or body` is a
        # presence test, so a malformed header (the n8n execution id, prod
        # 2026-08-14) beat a valid body and the row lost its key regardless.
        effective_correlation_id = first_valid_correlation_id(
            x_correlation_id, request.correlation_id
        )

        # Execute the graph
        result = await service.execute_graph(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            graph_id=graph_id,
            request=request,
            variables=request.variables,
            start_node=request.start_node,
            visibility=ctx.visibility,
            correlation_id=effective_correlation_id,
        )

        return result

    except GraphNotFoundError as e:
        # Explicit and typed, ahead of the `except ValueError` substring branch
        # below. That branch still matches this message, but a 404 must not
        # depend on the phrase "not found" surviving in an error string.
        #
        # Until issue #211 this was unreachable: the service swallowed the raise
        # and returned `ExecutionResponse(status="failed")`, so the endpoint
        # answered 200 for a graph that does not exist.
        logger.warning("Graph execution: graph %s not found for this scope", graph_id)
        raise NotFoundError("Graph", graph_id) from e
    except ValidationError as e:
        logger.exception("Graph execution validation error")
        error_str = str(e)
        # Handle specific validation errors with better messages
        if "Start node must be a valid UUID format" in error_str and request.start_node:
            raise HTTPException(
                status_code=422,
                detail=f"Validation failed: body -> start_node: Value error, Start node must be a valid UUID format, got: '{request.start_node}'",
            ) from e
        # This should not happen as Pydantic validates the request
        raise BadRequestError(f"Validation error: {error_str}") from e
    except ValueError as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Graph", graph_id) from e
        # Handle specific validation errors
        if "Start node must be a valid UUID format" in error_str and request.start_node:
            raise HTTPException(
                status_code=422,
                detail=f"Validation failed: body -> start_node: Value error, Start node must be a valid UUID format, got: '{request.start_node}'",
            ) from e
        logger.exception("Graph execution validation error")
        raise BadRequestError(error_str) from e
    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Graph", graph_id) from e
        logger.error(f"Graph execution error: {e}", exc_info=True)
        raise InternalServerError(f"Execution error: {error_str}") from e


@router.post(
    "/graphs/{graph_id}/execute/stepwise",
    response_model=ExecutionResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 404, 405, 422, 500]}
    },
    operation_id="execute_graph_stepwise",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE))],
)
async def execute_graph_stepwise(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: ExecutionRequest,
    graph_id: str = Path(
        ...,
        description="UUID of the graph to execute step by step",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-Id"),
) -> ExecutionResponse:
    """Execute a graph step by step (controlled execution) for a specific tenant.

    Args:
        request: The execution request containing variables and configuration
        graph_id: UUID of the graph to execute in stepwise mode
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        ExecutionResponse: Result of the stepwise graph execution

    Raises:
        HTTPException: 400 for validation errors, 404 for graph not found,
                      405 for method not allowed, 422 for request validation errors,
                      500 for execution errors
    """
    try:
        # Validate UUID format since we now use str type
        try:
            UUID(graph_id)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Invalid UUID format for graph_id: {graph_id}"
            ) from None

        service = get_execution_service()

        # Header wins on disagreement (per spec §5.2.1) — but only among values
        # that could actually BE the join key. A plain `header or body` is a
        # presence test, so a malformed header (the n8n execution id, prod
        # 2026-08-14) beat a valid body and the row lost its key regardless.
        effective_correlation_id = first_valid_correlation_id(
            x_correlation_id, request.correlation_id
        )

        # Start stepwise execution
        result = await service.execute_graph_stepwise(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            graph_id=graph_id,
            variables=request.variables,
            start_node=request.start_node,
            visibility=ctx.visibility,
            correlation_id=effective_correlation_id,
        )

        return result

    except GraphNotFoundError as e:
        # Explicit and typed, ahead of the `except ValueError` substring branch
        # below. That branch still matches this message, but a 404 must not
        # depend on the phrase "not found" surviving in an error string.
        #
        # Until issue #211 this was unreachable: the service swallowed the raise
        # and returned `ExecutionResponse(status="failed")`, so the endpoint
        # answered 200 for a graph that does not exist.
        logger.warning("Stepwise execution: graph %s not found for this scope", graph_id)
        raise NotFoundError("Graph", graph_id) from e
    except ValidationError as e:
        logger.exception("Stepwise execution validation error")
        error_str = str(e)
        # Handle specific validation errors with better messages
        if "Start node must be a valid UUID format" in error_str and request.start_node:
            raise HTTPException(
                status_code=422,
                detail=f"Validation failed: body -> start_node: Value error, Start node must be a valid UUID format, got: '{request.start_node}'",
            ) from e
        # This should not happen as Pydantic validates the request
        raise BadRequestError(f"Validation error: {error_str}") from e
    except ValueError as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Graph", graph_id) from e
        # Handle specific validation errors
        if "Start node must be a valid UUID format" in error_str and request.start_node:
            raise HTTPException(
                status_code=422,
                detail=f"Validation failed: body -> start_node: Value error, Start node must be a valid UUID format, got: '{request.start_node}'",
            ) from e
        logger.exception("Stepwise execution validation error")
        raise BadRequestError(error_str) from e
    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Graph", graph_id) from e
        logger.error(f"Stepwise execution error: {e}", exc_info=True)
        raise InternalServerError(f"Execution error: {error_str}") from e


# Session status endpoint
@router.get(
    "/sessions/{session_id}/status",
    response_model=ExecutionResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 404, 405, 422, 500]}
    },
    operation_id="get_session_status",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_EXECUTION_SESSION_STATUS))],
)
async def get_session_status(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    session_id: str = Path(
        ...,
        description="Session ID to check status for",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> ExecutionResponse:
    """Get the status of a graph execution session for a specific tenant.

    Args:
        session_id: The session ID to check status for
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        ExecutionResponse: Current status and details of the execution session

    Raises:
        HTTPException: 400 for validation errors, 404 if session not found,
                      405 for method not allowed, 422 for request validation errors,
                      500 for other errors
    """
    try:
        service = get_execution_service()

        # Get session status from the service (async since Phase 1 — hits DB
        # to surface inference telemetry fields on completed sessions).
        session_status = await service.get_session_status(ctx.tenant_id, ctx.subject_id, session_id)

        if session_status is None:
            raise NotFoundError("Session", session_id)

        # Convert to ExecutionResponse format
        return ExecutionResponse(
            session_id=session_id,
            status=session_status.get("status", "unknown"),
            current_node=session_status.get("current_node"),
            node_outputs=session_status.get("node_outputs", {}),
            completed_nodes=session_status.get("completed_nodes", []),
            failed_nodes=session_status.get("failed_nodes", []),
            final_output=session_status.get("final_output"),
            error_message=session_status.get("error_message"),
            started_at=session_status.get("started_at", datetime.now(UTC)),
            updated_at=session_status.get("updated_at", datetime.now(UTC)),
            chosen_provider=session_status.get("chosen_provider"),
            fallback_count=session_status.get("fallback_count"),
            prompt_tokens=session_status.get("prompt_tokens"),
            completion_tokens=session_status.get("completion_tokens"),
            validation_passed=session_status.get("validation_passed"),
            quality_score=session_status.get("quality_score"),
        )

    except ValueError as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Session", session_id) from e
        logger.exception("Session status validation error")
        raise BadRequestError(error_str) from e
    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise NotFoundError("Session", session_id) from e
        logger.error(f"Session status error: {e}", exc_info=True)
        raise InternalServerError(f"Session status error: {error_str}") from e


# Session list endpoint — used by model-evaluator's eval-set sampler to
# pull production traces for a workflow under a tenant + time window.
# See `Eval-tuple payload sourcing` deferral in Plan 5 close-out.
@router.get(
    "/sessions/list",
    response_model=list[dict[str, Any]],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 405, 422, 500]}
    },
    operation_id="list_execution_sessions",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_EXECUTION_SESSIONS_LIST))],
)
async def list_execution_sessions(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str | None = Query(
        default=None,
        description="Filter to sessions for a specific graph (workflow) UUID.",
    ),
    status: str | None = Query(
        default=None,
        description="Filter by session status (e.g. 'completed', 'failed').",
    ),
    validation_passed: bool | None = Query(  # noqa: FBT001, RUF100
        default=None,
        description="Filter to sessions whose final-response validation passed (or failed).",
    ),
    correlation_id: str | None = Query(
        default=None,
        description="Filter to sessions tagged with this correlation/trace ID.",
    ),
    since: datetime | None = Query(  # noqa: B008, RUF100
        default=None,
        description="Lower bound on created_at (inclusive, ISO-8601).",
    ),
    until: datetime | None = Query(  # noqa: B008, RUF100
        default=None,
        description="Upper bound on created_at (exclusive, ISO-8601).",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of sessions to return (1-500, default 100).",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description=(
            "Number of rows to skip; combine with ``limit`` to page through "
            "matching sessions ordered by ``created_at`` DESC."
        ),
    ),
) -> list[dict[str, Any]]:
    """List execution sessions visible under (tenant_id, subject_id) with filters.

    Each row carries the persisted session columns including ``session_id``,
    ``request_payload``, ``final_response``, ``correlation_id``,
    ``validation_passed``, ``status``, ``created_at``, and ``completed_at``.
    ``request_payload`` / ``final_response`` are the input/output shapes the
    eval-set sampler needs to redact and persist as ``eval_tuples``; the
    model-evaluator side maps them through the existing redact + insert path
    (``POST /eval-sets/{id}/refresh-from-prod``). ``session_id`` and
    ``correlation_id`` are persisted onto the eval_tuple as provenance.

    Filters compose with AND. ``since`` / ``until`` bracket ``created_at``
    (closed lower bound, open upper bound). UUIDs are validated when supplied.

    Visibility: returns tenant-visible rows the caller would see directly. With
    no ``X-Subject-Id`` forwarded, ``visibility='user'`` rows belonging to
    other subjects are excluded by design — refresh-from-wm cannot reach
    user-private traces.
    """
    if graph_id is not None:
        try:
            UUID(graph_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid UUID format for graph_id: {graph_id}",
            ) from None

    if since is not None and until is not None and not (since < until):
        raise HTTPException(
            status_code=400,
            detail="`since` must be strictly less than `until`.",
        )

    try:
        rows = await GraphExecutionSessionDB().list_sessions(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            graph_id=graph_id,
            status=status,
            validation_passed=validation_passed,
            correlation_id=correlation_id,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error("list_execution_sessions error: %s", e, exc_info=True)
        raise InternalServerError(f"List sessions error: {e}") from e

    # Stringify UUID + datetime for the JSON response. asyncpg returns native
    # types; FastAPI handles datetime, but UUID columns come back as
    # uuid.UUID and need explicit string conversion for stable client parsing.
    out: list[dict[str, Any]] = []
    for r in rows:
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, UUID):
                clean[k] = str(v)
            elif isinstance(v, datetime):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        out.append(clean)
    return out


# Health check endpoint
@router.get(
    "/health",
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="execution_health",
)
async def execution_health_check() -> dict[str, Any]:
    """Health check for execution service.

    Returns:
        dict: Health status information including engine type and timestamp

    Raises:
        HTTPException: 405 for method not allowed, 500 if health check fails
    """
    try:
        get_execution_service()

        return {
            "status": "healthy",
            "execution_engine": "pydantic_ai",
            "database_name": _get_db_name(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.exception("Health check error")
        raise InternalServerError(f"Health check failed: {e!s}") from e
