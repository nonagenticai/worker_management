"""Worker state management endpoints."""

import logging
from typing import Annotated, Any
from urllib.parse import unquote

from fastapi import Depends, HTTPException, Query
from fastapi import Path as PathParam

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.workers._globals import state_manager, worker_cache, worker_registry
from src.api.workers._router import router
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_WORKERS_ALL_STATES,
    ROUTE_WORKERS_ALL_STATES_DELETE,
    ROUTE_WORKERS_INSTANCE_CLEANUP,
    ROUTE_WORKERS_STATE_LATEST,
    ROUTE_WORKERS_STATES_DELETE,
    ROUTE_WORKERS_WORKER_STATES,
    tokenless_tenant_gate,
)
from src.db.workers_state_db import MAX_WORKER_STATES_LIMIT, StateDatabase

logger = logging.getLogger(__name__)


@router.get(
    "/states",
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="list_all_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_ALL_STATES))],
)
async def get_all_worker_states(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, Any]:
    """Get all worker state records across all workers for a specific tenant.

    Retrieves all state records from all workers for the specified tenant. This is useful
    for system-wide monitoring, debugging, or analytics. Use with caution in
    production as it may return a large amount of data.

    Args:
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        Dict containing:
            - states: List of state records, capped at MAX_WORKER_STATES_LIMIT
            - total: TRUE number of state records in scope (a real COUNT(*),
              which may exceed len(states))
            - returned: number of records actually in ``states``
            - truncated: True when ``total`` exceeds ``returned`` -- i.e. this
              response does NOT contain everything

    Raises:
        HTTPException: 500 if database query fails

    Note:
        The read is CAPPED at MAX_WORKER_STATES_LIMIT records; the underlying
        query is bounded in SQL. This endpoint takes no pagination parameters,
        so a caller that needs more than the cap must use the paginated
        ``GET /workers_states`` endpoint (limit/offset) or the per-worker state
        queries. ``truncated`` tells you when that applies.

    Examples:
        Response:
        ```json
        {
            "states": [
                {
                    "id": 1,
                    "worker_id": "123e4567-e89b-12d3-a456-426614174000",
                    "worker_name": "data_processor",
                    "state_type": "completed",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "worker_state": {"status": "completed"}
                },
                ...
            ],
            "total": 150,
            "returned": 150,
            "truncated": false
        }
        ```
    """
    try:
        db = StateDatabase()
        # get_worker_states is bounded; ask for the ceiling explicitly so this
        # endpoint keeps its previous shape for every tenant under the cap.
        states = await db.get_worker_states(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            limit=MAX_WORKER_STATES_LIMIT,
        )
        # ``total`` is a real COUNT over the same scope, NOT len(states).
        # Reporting the page length as the total would tell a tenant holding
        # more than the cap that there are exactly ``cap`` records -- a silent
        # truncation presenting itself as a complete answer, which is worse
        # than either the old unbounded read or an honest cap.
        total = await db.count_worker_states(tenant_id=ctx.tenant_id, subject_id=ctx.subject_id)
        return {
            "states": states,
            "total": total,
            "returned": len(states),
            "truncated": total > len(states),
        }
    except Exception as e:
        logger.exception("Error retrieving all worker states")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve worker states: {e!s}"
        ) from e


@router.delete(
    "/states",
    status_code=204,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="delete_all_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_ALL_STATES_DELETE))],
)
async def delete_all_worker_states(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> None:
    """Delete all worker state records for a specific tenant while leaving workers intact.

    Removes all historical state data from all workers for the specified tenant. This is
    a destructive operation that cannot be undone. The workers themselves remain
    registered and functional - only their state history is cleared.

    Args:
        tenant_id: Tenant identifier for multi-tenancy

    Use cases:
    - Clearing old data to free up database space
    - Resetting the system for testing
    - Compliance with data retention policies

    Returns:
        None (204 No Content on success)

    Raises:
        HTTPException: 500 if database operation fails

    Warning:
        This operation is irreversible. All state history for this tenant will be permanently lost.

    Note:
        Returns 204 No Content on successful deletion. The number of deleted
        records is logged but not returned to the client.
    """
    try:
        db = StateDatabase()
        states_deleted = await db.delete_all_worker_states(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id
        )
        logger.info(
            "Deleted %d worker state records for tenant=%s subject=%s",
            states_deleted,
            ctx.tenant_id,
            ctx.subject_id,
        )
        return  # 204 No Content
    except Exception as e:
        logger.exception("Error deleting all worker states")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete all worker states: {e!s}"
        ) from e


@router.get(
    "/{worker_name}/states",
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_WORKER_STATES))],
)
async def get_worker_state_history(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    worker_name: str = PathParam(
        ...,
        description="Name of the worker to get states for",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
    limit: int = Query(10, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get state history for a specific worker for a specific tenant.

    Retrieves the historical state records for a worker, ordered by timestamp
    (most recent first). This is useful for debugging, monitoring worker activity,
    and understanding the worker's processing history.

    Args:
        worker_name: Name of the worker to get state history for
        limit: Maximum number of state records to return (1-100, default 10)
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        List of state records, each containing:
            - id: State record ID
            - worker_id: Worker's unique ID
            - worker_name: Worker's name
            - is_agent: Whether worker is an agent
            - state_type: Type of state (created, processing, completed, error)
            - timestamp: When the state was recorded
            - metadata: Optional metadata
            - worker_state: The actual state data

    Raises:
        HTTPException: 422 if worker name is invalid
        HTTPException: 404 if worker not found
        HTTPException: 500 if database query fails

    Examples:
        GET /workers/data_processor/states?limit=5

        Returns last 5 state records for the data_processor worker
    """
    # Check if this is actually a misrouted path or invalid name
    if (
        worker_name.lower() in ["states", "message", "null"]
        or "\x00" in worker_name
        or not worker_name.strip()
        or worker_name == "null"
    ):
        raise HTTPException(status_code=422, detail="Invalid worker name")

    # Use atomic worker ID resolution
    worker_id, in_registry = await state_manager.resolve_worker_id(
        ctx.tenant_id, ctx.subject_id, worker_name, worker_registry
    )

    db = StateDatabase()
    try:
        # Primary lookup with resolved worker_id
        states = await db.get_worker_states_by_id(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id, limit=limit
        )

        # If no states found and worker_id != worker_name, try with worker_name
        if not states and worker_id != worker_name:
            states = await db.get_worker_states_by_id(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                worker_id=worker_name,
                limit=limit,
            )

        # If still no states found, perform comprehensive check
        if not states:  # noqa: RUF100, SIM102
            # Check if worker exists in registry or has any states
            if not in_registry:
                # Check if any states exist for this worker
                all_states = await db.get_worker_states(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    limit=MAX_WORKER_STATES_LIMIT,
                )
                worker_exists = any(
                    state.get("worker_name") == worker_name
                    or state.get("worker_id") == worker_name
                    or state.get("worker_id") == worker_id
                    for state in all_states
                )

                if not worker_exists:
                    raise HTTPException(status_code=404, detail="Worker not found")

        logger.debug(f"Retrieved {len(states)} states for worker {worker_name} (ID: {worker_id})")
        return states
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception("Error retrieving worker states")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/{worker_name}/state/latest",
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [404, 422, 500]}},
    operation_id="get_latest_worker_state",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_STATE_LATEST))],
)
async def get_latest_worker_state(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    worker_name: str = PathParam(
        ...,
        description="Name of the worker to get latest state for",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> dict[str, Any]:
    """Get the most recent state for a worker for a specific tenant.

    Retrieves only the latest state record for a worker. This is useful for
    checking the current status or the result of the most recent operation.

    Args:
        worker_name: Name of the worker to get latest state for
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        Dict containing the latest state record with:
            - id: State record ID
            - worker_id: Worker's unique ID
            - worker_name: Worker's name
            - is_agent: Whether worker is an agent
            - state_type: Type of state (created, processing, completed, error)
            - timestamp: When the state was recorded
            - metadata: Optional metadata
            - worker_state: The actual state data including status and any results

    Raises:
        HTTPException: 404 if worker not found or no state exists
        HTTPException: 500 if database query fails

    Examples:
        Response for a completed task:
        ```json
        {
            "id": 42,
            "worker_id": "123e4567-e89b-12d3-a456-426614174000",
            "worker_name": "data_processor",
            "is_agent": true,
            "state_type": "completed",
            "timestamp": "2024-01-15T10:30:00Z",
            "worker_state": {
                "status": "completed",
                "message": "Process the dataset",
                "response": "Dataset processed successfully. 1000 records analyzed."
            }
        }
        ```
    """
    # Validate worker name - reject reserved names
    if worker_name.lower() in [
        "null",
        "undefined",
        "none",
        "false",
        "true",
        "",
        "states",
        "message",
    ]:
        raise HTTPException(status_code=422, detail="Invalid worker name")

    # Use atomic worker ID resolution
    worker_id, in_registry = await state_manager.resolve_worker_id(
        ctx.tenant_id, ctx.subject_id, worker_name, worker_registry
    )

    db = StateDatabase()
    try:
        # Primary lookup with resolved worker_id
        state = await db.get_latest_worker_state(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id
        )

        # If no state found and worker_id != worker_name, try with worker_name
        if not state and worker_id != worker_name:
            state = await db.get_latest_worker_state(
                tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_name
            )

        # If still no state found, perform comprehensive check
        if not state:  # noqa: RUF100, SIM102
            # Check if worker exists in registry or has any states
            if not in_registry:
                # Check if any states exist for this worker
                all_states = await db.get_worker_states(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    limit=MAX_WORKER_STATES_LIMIT,
                )
                worker_exists = any(
                    state_record.get("worker_name") == worker_name
                    or state_record.get("worker_id") == worker_name
                    or state_record.get("worker_id") == worker_id
                    for state_record in all_states
                )

                if not worker_exists:
                    raise HTTPException(status_code=404, detail="Worker not found")

        # Worker exists but no state found
        if not state:
            raise HTTPException(status_code=404, detail="No state found for worker")

        logger.debug(f"Retrieved latest state for worker {worker_name} (ID: {worker_id})")
        return state
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception("Error retrieving latest worker state")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/{worker_name}/states",
    status_code=204,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="delete_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_STATES_DELETE))],
)
async def delete_worker_states(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    worker_name: str = PathParam(
        ...,
        description="Name of the worker whose states to delete",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> None:
    """Delete all state records for a specific worker for a specific tenant.

    Removes all historical state records for a worker from the state database.
    The worker itself remains registered and functional - only its state history
    is cleared. This is useful for cleaning up old state data.

    Args:
        worker_name: Name of the worker whose states to delete.
                    Must be 1-100 characters.
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        None (204 No Content on success)

    Raises:
        HTTPException: 422 if worker name is empty or invalid
        HTTPException: 404 if worker not found
        HTTPException: 500 if database operation fails

    Note:
        Returns 204 No Content on successful deletion. The number of deleted
        records is logged but not returned to the client.
    """
    try:
        # Validate worker_name input
        if not worker_name:
            raise HTTPException(status_code=422, detail="Worker name cannot be empty")

        # Check if this is actually a misrouted path
        if worker_name.lower() in ["states", "message", "null"]:
            raise HTTPException(status_code=422, detail="Invalid worker name")

        # URL decode the worker name to handle special characters
        worker_name = unquote(worker_name)

        # Get worker data to retrieve worker_id
        worker_data = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
        )
        if not worker_data:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker_id = worker_data.get("worker_id", worker_name)

        # Delete worker states
        db = StateDatabase()
        states_deleted = await db.delete_worker_state(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id
        )
        logger.info(f"Deleted {states_deleted} state records for worker {worker_name}")
        return  # 204 No Content

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception(f"Error deleting worker states for {worker_name}")
        raise HTTPException(status_code=500, detail=f"Failed to delete worker states: {e!s}") from e


@router.delete(
    "/{worker_name}/instance",
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [404, 422, 500]}},
    operation_id="cleanup_worker_instance",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_INSTANCE_CLEANUP))],
)
async def cleanup_worker_instance(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],  # noqa: ARG001
    worker_name: str = PathParam(
        ...,
        description="Name of the worker instance to cleanup",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> dict[str, str]:
    """Force cleanup of a worker instance from the cache.

    This endpoint forcefully removes a worker instance from the in-memory cache
    without deleting the worker registration. This is useful for:
    - Clearing stuck or unhealthy worker instances
    - Forcing recreation of a worker on next use
    - Freeing memory from unused worker instances

    The worker registration remains intact, so the worker can be recreated
    automatically on the next request.

    Args:
        worker_name: Name of the worker instance to cleanup.
                    Must be valid Unicode identifier (no control chars, commas, or whitespace)

    Returns:
        Dict containing:
            - status: Confirmation message "Worker instance cleaned up"

    Raises:
        HTTPException: 422 if worker name is invalid (empty, reserved word,
                      invalid characters, too long)
        HTTPException: 500 if cleanup fails

    Note:
        This operation only affects the cached instance, not the worker registration.
        Reserved names like 'null', 'undefined', 'true', 'false' are rejected.
        Maximum length is 200 bytes when UTF-8 encoded.

    Examples:
        DELETE /workers/data_processor/instance

        Response:
        ```json
        {
            "status": "Worker instance cleaned up"
        }
        ```
    """
    # URL decode the worker name to handle special characters
    worker_name = unquote(worker_name)

    # List of reserved/invalid worker names
    INVALID_WORKER_NAMES = [
        "{}",
        "[]",
        "()",
        "<>",
        "",
        "null",
        "null,null",
        "false",
        "true",
        "none",
        "undefined",
        "nan",
        "infinity",
    ]

    # Check if this is actually a misrouted path or invalid name
    # Be more strict about what we accept - reject reserved words and suspicious patterns
    if (
        worker_name.lower() in [name.lower() for name in INVALID_WORKER_NAMES]
        or not worker_name.strip()
        or len(worker_name.strip()) == 0
        or "\x00" in worker_name
        or "," in worker_name  # Reject comma-separated values
    ):
        raise HTTPException(status_code=422, detail="Invalid worker name")

    # Additional pattern validation for valid worker names - Unicode-aware
    # Accept numeric worker names like '0' - they are valid
    if len(worker_name) > 0:
        # Check first character: must be Unicode letter, digit, or underscore
        if not (worker_name[0].isalnum() or worker_name[0] == "_"):
            raise HTTPException(
                status_code=422,
                detail="Worker name must start with a Unicode letter, digit, or underscore",
            )

        # Check remaining characters: letters, digits, underscores, hyphens
        for char in worker_name[1:]:
            if not (char.isalnum() or char in "_-"):
                raise HTTPException(
                    status_code=422,
                    detail="Worker name can only contain Unicode letters, digits, underscores, and hyphens",
                )

    # Check for extremely long names (over 200 chars)
    if len(worker_name.encode("utf-8")) > 200:
        raise HTTPException(status_code=422, detail="Worker name too long")

    await worker_cache.remove_worker(worker_name)
    return {"status": "Worker instance cleaned up"}
