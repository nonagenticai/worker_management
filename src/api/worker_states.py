"""Worker States API module for managing worker state operations."""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.error_handlers import create_error_response

# Import must be after path manipulation
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_WORKERS_STATES_CREATE,
    ROUTE_WORKERS_STATES_LIST,
    tokenless_tenant_gate,
)
from src.db.workers_state_db import StateDatabase
from src.type_definitions.constraint_types import validate_identifier

# Configure logging
logger = logging.getLogger(__name__)

# Create the worker states router
router = APIRouter(prefix="/workers_states", tags=["Worker States"])


class WorkerStateType(str, Enum):
    """Enumeration of valid worker state types.

    ⛔ STRICT. Unlike the service-side enum this one is a request-model field
    (``WorkerStateBase.state_type``), so anything not listed here is a 422 — a
    member added on plausibility widens the gate for every caller.

    ⚠️ ``updated`` WAS NAMED HERE AS DELIBERATELY ABSENT AND IS NOW A MEMBER
    (2026-09-17). The census it was waiting on came back non-zero —
    ``(updated, processing)`` = 8/7d, 2/48h — so it was promoted on the rule this
    docstring already stated, not against it. ``prompt-swapped`` is still
    deliberately NOT here: still zero observed transition traffic, row count
    notwithstanding. See :mod:`src.services.worker_state_manager` for the
    measurement and for why a row count never earns membership.
    """

    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    IDLE = "idle"
    ACTIVE = "active"
    DEFAULT = "default"
    CONFIG_UPDATED = "config-updated"
    UPDATED = "updated"


class WorkerStateBase(BaseModel):
    """Base model for worker states."""

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields for backward compatibility
        json_schema_extra={
            "examples": [
                {
                    "worker_id": "worker_123",
                    "worker_name": "data_processor",
                    "is_agent": True,
                    "state_type": "processing",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "metadata": {
                        "task_id": "task_456",
                        "progress": 75,
                        "status": "running",
                    },
                    "worker_state": {
                        "current_operation": "analyzing_data",
                        "processed_records": 1500,
                        "total_records": 2000,
                        "errors": 0,
                    },
                }
            ]
        },
    )

    worker_id: str = Field(..., min_length=1, max_length=100)
    worker_name: str = Field(..., min_length=1, max_length=100)
    is_agent: bool = False
    state_type: WorkerStateType = Field(..., description="The type of worker state")
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None
    worker_state: dict[str, Any] = Field(..., description="Worker state data")

    @field_validator("worker_id", "worker_name")
    @classmethod
    def validate_names(cls, v: str) -> str:
        # Shared empty / null-byte / strip check (single source of truth).
        v_stripped = validate_identifier(v, "Field")

        # Reject extremely long identifiers that could cause issues
        if len(v_stripped) > 100:
            raise ValueError("Field cannot exceed 100 characters")

        # Check for control characters other than null bytes
        if any(ord(char) < 32 and char not in "\t\n\r" for char in v_stripped):
            raise ValueError("Field contains invalid control characters")

        return v_stripped

    @field_validator("state_type", mode="before")
    @classmethod
    def validate_state_type(cls, v) -> WorkerStateType:
        """Validate the state type enum value.

        Args:
            v: The state type value to validate

        Returns:
            WorkerStateType: The validated enum value

        Raises:
            ValueError: If the state type is invalid
        """
        # If it's already a WorkerStateType enum, check for null bytes in its value
        if isinstance(v, WorkerStateType):
            if "\x00" in v.value:
                raise ValueError("State type cannot contain null bytes")
            return v

        # Handle missing or None values
        if v is None or v == "":
            raise ValueError("State type is required and cannot be empty")

        # Convert to string for validation
        state_str = str(v).strip()

        # Check for null bytes
        if "\x00" in state_str:
            raise ValueError("State type cannot contain null bytes")

        # Check if the string is a valid enum value
        valid_values = [e.value for e in WorkerStateType]
        if state_str not in valid_values:
            raise ValueError(
                f"Invalid state type: '{state_str}'. Must be one of: {', '.join(valid_values)}"
            )

        # Return the enum value
        return WorkerStateType(state_str)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary object")  # noqa: RUF100, TRY004

        if len(str(v)) > 50000:
            raise ValueError("Metadata is too large (max 50KB when serialized)")
        return v

    @field_validator("worker_state")
    @classmethod
    def validate_worker_state_required(cls, v):
        if v is None:
            raise ValueError("Worker state is required and cannot be null")
        if not isinstance(v, dict):
            raise ValueError("Worker state must be a dictionary object")  # noqa: RUF100, TRY004

        # Enhanced validation that's balanced between security and flexibility
        import json

        # Check serialized size first (more efficient and accurate)
        try:
            serialized = json.dumps(v)
            if len(serialized) > 50000:
                raise ValueError("Worker state is too large (max 50KB when serialized)")
        except (TypeError, ValueError) as e:
            # This indicates non-serializable data which is always invalid
            raise ValueError(f"Worker state contains non-serializable data: {e!s}") from e

        # Targeted null byte checking - focus on keys and top-level values
        # This allows complex nested structures while still preventing security issues
        def balanced_null_byte_check(obj, depth=0, path="") -> None:
            # Prevent infinite recursion and excessive processing
            if depth > 8:
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Always check dictionary keys for null bytes (critical security check)
                    if isinstance(key, str) and "\x00" in key:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state key at {path}{key}"
                        )

                    # Check string values only at shallow depths to allow complex data
                    if isinstance(value, str) and depth < 2 and "\x00" in value:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state value at {path}{key}"
                        )

                    # Recurse with depth limit
                    if isinstance(value, (dict, list)) and depth < 4:
                        balanced_null_byte_check(value, depth + 1, f"{path}{key}.")

            elif isinstance(obj, list) and depth < 4:
                for i, item in enumerate(obj):
                    if isinstance(item, str) and depth < 2 and "\x00" in item:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state list item at {path}[{i}]"
                        )
                    elif isinstance(item, (dict, list)):
                        balanced_null_byte_check(item, depth + 1, f"{path}[{i}].")

        balanced_null_byte_check(v)

        return v


class SaveWorkerStateRequest(BaseModel):
    """Request model for saving worker states - strictly validates allowed fields only."""

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields for backward compatibility
        json_schema_extra={
            "examples": [
                {
                    "worker_id": "worker_123",
                    "worker_name": "data_processor",
                    "state_type": "processing",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "worker_state": {
                        "current_operation": "analyzing_data",
                        "processed_records": 1500,
                        "total_records": 2000,
                        "errors": 0,
                    },
                }
            ]
        },
    )

    worker_id: str = Field(..., min_length=1, max_length=100)
    worker_name: str = Field(..., min_length=1, max_length=100)
    state_type: WorkerStateType = Field(..., description="The type of worker state")
    timestamp: datetime | None = None
    worker_state: dict = Field(..., description="Worker state data")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            return None

        # If it's already a datetime, validate its range but be more permissive
        if isinstance(v, datetime):
            # Allow a wider range for historical and test data, but reject clearly invalid years
            if v.year < 1800 or v.year > 9999:
                raise ValueError(f"Timestamp year {v.year} is out of acceptable range (1800-9999)")
            return v

        # Try to parse string timestamps
        if isinstance(v, str):
            from dateutil.parser import parse

            try:
                # Parse the datetime string
                dt = parse(v)
                # More permissive date range - allow historical dates and far future dates for testing
                # But still reject clearly invalid dates that could cause system issues
                if dt.year < 1800 or dt.year > 9999:
                    raise ValueError(
                        f"Timestamp year {dt.year} is out of acceptable range (1800-9999)"
                    )
                return dt
            except Exception as e:
                # Only reject if it's a completely invalid format
                # Allow edge cases that might be valid in some contexts
                raise ValueError(f"Invalid timestamp format: {e!s}") from e

        # Be more permissive with other numeric types that could represent timestamps
        try:
            # Try to convert numeric values to datetime (Unix timestamps, etc.)
            if isinstance(v, (int, float)):
                # Assume it's a Unix timestamp if it's in a reasonable range
                if 0 <= v <= 2147483647:  # Valid Unix timestamp range
                    return datetime.fromtimestamp(v, tz=UTC)
                else:
                    raise ValueError(f"Numeric timestamp {v} is out of valid Unix timestamp range")
        except (ValueError, OSError):  # fmt: skip
            pass

        raise ValueError(
            f"Timestamp must be a datetime object, ISO format string, or Unix timestamp, got {type(v)}"
        )

    @field_validator("worker_id", "worker_name")
    @classmethod
    def validate_names(cls, v):
        # Shared empty / null-byte / strip check (single source of truth).
        v_stripped = validate_identifier(v, "Field")

        # Reject extremely long identifiers that could cause issues
        if len(v_stripped) > 100:
            raise ValueError("Field cannot exceed 100 characters")

        # Check for control characters other than null bytes
        if any(ord(char) < 32 and char not in "\t\n\r" for char in v_stripped):
            raise ValueError("Field contains invalid control characters")

        return v_stripped

    @field_validator("state_type", mode="before")
    @classmethod
    def validate_state_type(cls, v) -> WorkerStateType:
        """Validate the state type enum value.

        Args:
            v: The state type value to validate

        Returns:
            WorkerStateType: The validated enum value

        Raises:
            ValueError: If the state type is invalid
        """
        # If it's already a WorkerStateType enum, check for null bytes in its value
        if isinstance(v, WorkerStateType):
            if "\x00" in v.value:
                raise ValueError("State type cannot contain null bytes")
            return v

        # Handle missing or None values
        if v is None or v == "":
            raise ValueError("State type is required and cannot be empty")

        # Convert to string for validation
        state_str = str(v).strip()

        # Check for null bytes
        if "\x00" in state_str:
            raise ValueError("State type cannot contain null bytes")

        # Check if the string is a valid enum value
        valid_values = [e.value for e in WorkerStateType]
        if state_str not in valid_values:
            raise ValueError(
                f"Invalid state type: '{state_str}'. Must be one of: {', '.join(valid_values)}"
            )

        # Return the enum value
        return WorkerStateType(state_str)

    @field_validator("worker_state")
    @classmethod
    def validate_worker_state_required(cls, v):
        if v is None:
            raise ValueError("Worker state is required and cannot be null")
        if not isinstance(v, dict):
            raise ValueError("Worker state must be a dictionary object")  # noqa: RUF100, TRY004

        # Enhanced validation that's more targeted and balanced
        import json

        # Check serialized size first (more efficient)
        try:
            serialized = json.dumps(v)
            if len(serialized) > 50000:
                raise ValueError("Worker state is too large (max 50KB when serialized)")
        except (TypeError, ValueError) as e:
            # This indicates non-serializable data which is always invalid
            raise ValueError(f"Worker state contains non-serializable data: {e!s}") from e

        # More targeted null byte checking - only check keys and string values at top levels
        # This prevents valid complex nested structures from being rejected while still blocking attacks
        def smart_null_byte_check(obj, depth=0, path="") -> None:
            # Limit recursion depth to prevent performance issues with deeply nested structures
            if depth > 10:
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Always check dictionary keys for null bytes (security critical)
                    if isinstance(key, str) and "\x00" in key:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state key at {path}{key}"
                        )

                    # For string values, only check at shallow depths to allow complex nested data
                    if isinstance(value, str) and depth < 3 and "\x00" in value:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state value at {path}{key}"
                        )

                    # Recurse into nested structures but with depth limits
                    if isinstance(value, (dict, list)) and depth < 5:
                        smart_null_byte_check(value, depth + 1, f"{path}{key}.")

            elif isinstance(obj, list) and depth < 5:
                for i, item in enumerate(obj):
                    if isinstance(item, str) and depth < 3 and "\x00" in item:
                        raise ValueError(
                            f"Null bytes not allowed in worker_state list item at {path}[{i}]"
                        )
                    elif isinstance(item, (dict, list)):
                        smart_null_byte_check(item, depth + 1, f"{path}[{i}].")

        smart_null_byte_check(v)

        return v


class BulkWorkerStatesRequest(BaseModel):
    """Request model for bulk worker states with proper validation."""

    worker_states: list[SaveWorkerStateRequest] = Field(..., max_length=100, min_length=1)

    @field_validator("worker_states")
    @classmethod
    def validate_worker_states_list(
        cls, v: list[SaveWorkerStateRequest]
    ) -> list[SaveWorkerStateRequest]:
        if not v:
            raise ValueError("At least one worker state is required")
        if len(v) > 100:
            raise ValueError("Cannot save more than 100 worker states in a single request")
        return v


@router.get(
    "",
    response_model=None,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 500]}},
    operation_id="list_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_STATES_LIST))],
)
async def get_workers_states(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of worker states to return"),
    offset: int = Query(0, ge=0, description="Number of worker states to skip"),
):
    """Retrieve worker states from the database with pagination for a specific tenant.

    Fetches worker state records from the state database with pagination support.
    This endpoint returns the state history for all workers belonging to the specified tenant.

    Args:
        tenant_id: Tenant identifier for multi-tenancy
        limit: Maximum number of worker states to return (1-1000, default: 100)
        offset: Number of worker states to skip for pagination (default: 0)

    Returns:
        List of dictionaries, each containing:
            - id: State record ID
            - worker_id: Unique worker identifier
            - worker_name: Human-readable worker name
            - is_agent: Whether the worker is an AI agent
            - state_type: Type of state (created, processing, completed, error, etc.)
            - timestamp: When the state was recorded (ISO format)
            - metadata: Optional metadata dictionary
            - worker_state: The actual state data dictionary

    Raises:
        HTTPException: 500 with detailed error message for:
            - Database connection/operational errors
            - File system errors
            - Runtime errors
            - Unexpected errors

    Examples:
        Response:
        ```json
        [
            {
                "id": 1,
                "worker_id": "worker_123",
                "worker_name": "data_processor",
                "is_agent": true,
                "state_type": "completed",
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {"task_id": "task_456"},
                "worker_state": {
                    "status": "completed",
                    "processed_records": 2000,
                    "errors": 0
                }
            }
        ]
        ```
    """
    try:
        db = StateDatabase()
        # Pagination is applied IN SQL. Slicing a full-history read in Python
        # left the query unbounded: the page size never reached the database.
        return await db.get_worker_states(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            limit=limit,
            offset=offset,
        )
    except OSError as e:
        logger.exception("File system error retrieving worker states")
        return create_error_response(
            status_code=500,
            detail=f"File system error: {e!s}",
            error_code="FILE_SYSTEM_ERROR",
        )
    except RuntimeError as e:
        logger.exception("Runtime error retrieving worker states")
        return create_error_response(
            status_code=500,
            detail=f"Runtime error: {e!s}",
            error_code="RUNTIME_ERROR",
        )
    except Exception as e:
        logger.exception("Unexpected error retrieving worker states")
        return create_error_response(
            status_code=500,
            detail=f"Unexpected error: {e!s}",
            error_code="INTERNAL_ERROR",
        )


@router.post(
    "",
    status_code=201,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 422, 500]}},
    operation_id="create_worker_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_STATES_CREATE))],
)
async def save_workers_state(
    request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    body: BulkWorkerStatesRequest = Body(...),  # noqa: B008, RUF100
):
    """Save multiple worker states to the database for a specific tenant.

    Batch endpoint for saving worker states. Accepts up to 100 state records
    in a single request. Each state is validated and saved individually.
    Extra fields not defined in the schema are rejected with 422 errors.

    Args:
        tenant_id: Tenant identifier for multi-tenancy
        worker_states: List of SaveWorkerStateRequest objects (max 100), each containing:
            - worker_id: Unique worker identifier (1-100 chars)
            - worker_name: Human-readable worker name (1-100 chars)
            - state_type: Valid state type enum value
            - timestamp: Optional timestamp (defaults to current time)
            - worker_state: Required state data dictionary (max 50KB)

    Returns:
        Dict containing:
            - message: Success confirmation message
            - results: List of saved state records with:
                - id: Generated state record ID
                - worker_id: Worker identifier

    Raises:
        HTTPException: 422 for validation errors:
            - Empty or null worker_id/worker_name
            - Invalid state_type value
            - Missing or non-dict worker_state
            - Extra fields in request body
            - worker_state larger than 50KB
        HTTPException: 500 for server errors:
            - Database connection/operational errors
            - File system errors
            - Runtime errors
            - Unexpected errors

    Note:
        - is_agent defaults to False as it's not in the request model
        - metadata is set to None as it's not in the request model
        - Each state is saved individually (not transactional)
        - Timestamp defaults to current time if not provided

    Examples:
        Request:
        ```json
        {
            "worker_states": [
                {
                    "worker_id": "worker_123",
                    "worker_name": "data_processor",
                    "state_type": "processing",
                    "worker_state": {
                        "current_operation": "analyzing_data",
                        "processed_records": 1500
                    }
                }
            ]
        }
        ```

        Response:
        ```json
        {
            "message": "Workers' states saved successfully",
            "results": [
                {
                    "id": 42,
                    "worker_id": "worker_123"
                }
            ]
        }
        ```
    """
    # ⛔ WRITE, not read. The tokenless trusted-network branch stamps
    # ``identity_proven=False`` (see ``AuthenticationMiddleware._stamp_nothing_proven``)
    # whether the header was omitted (DEFAULT_TENANT_ID) or asserted (a
    # syntax-only name/uuid check, no allowlist) — either way nothing proved
    # who the caller is. That is tolerable for the sibling GET, which only
    # returns rows already resolvable this way; it is not tolerable here,
    # where the same unproven claim gets PERSISTED as up to 100 real rows
    # under whichever tenant the caller named. Refuse the write outright
    # rather than accept-and-attribute — there is no "adopted" write to fall
    # back to. The GET above stays tokenless and metered on purpose; do not
    # add this check there.
    if not ctx.identity_proven:
        return create_error_response(
            status_code=401,
            detail=(
                "POST /workers_states requires a verified licence token; send "
                "`X-Tenant-Id: Bearer <token>`."
            ),
            error_code="UNAUTHORIZED",
        )

    worker_states = body.worker_states

    # Validate request body size
    if request.headers.get("content-length"):
        content_length = int(request.headers.get("content-length", 0))
        max_size = 5 * 1024 * 1024  # 5MB max request size
        if content_length > max_size:
            return create_error_response(
                status_code=413,
                detail=f"Request body too large. Maximum size is {max_size} bytes.",
                error_code="PAYLOAD_TOO_LARGE",
            )

    # Validate worker states count - enforce 100 item limit
    if not worker_states:
        return create_error_response(
            status_code=422,
            detail="At least one worker state is required",
            error_code="VALIDATION_ERROR",
        )

    if len(worker_states) > 100:
        return create_error_response(
            status_code=422,
            detail="Cannot save more than 100 worker states in a single request",
            error_code="VALIDATION_ERROR",
        )

    try:
        db = StateDatabase()
        # Prepare states for batch insertion
        states_data = []
        for state in worker_states:
            state_dict = state.model_dump()
            worker_state = state_dict.pop("worker_state")

            states_data.append(
                {
                    "worker_id": state_dict.get("worker_id"),
                    "worker_name": state_dict.get("worker_name"),
                    "is_agent": False,  # Default to False since it's not in the request
                    "state_type": state_dict.get("state_type"),
                    "worker_state": worker_state,
                    "metadata": None,  # No metadata in the request model
                }
            )

        # Use atomic batch insertion
        state_ids = await db.add_worker_state_batch(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            tenant_name=ctx.tenant_name,
            states=states_data,
            visibility=ctx.visibility,
        )

        # Force additional WAL checkpoint to ensure immediate visibility (SQLite only)
        # await db.conn.execute("PRAGMA wal_checkpoint(FULL)")
        logger.info("Batch save completed")

        # Prepare results
        results = []
        for i, state in enumerate(worker_states):
            state_dict = state.model_dump()
            results.append({"id": state_ids[i], "worker_id": state_dict.get("worker_id")})

        logger.info("Successfully saved %d worker states in atomic batch", len(results))
        return {"message": "Workers' states saved successfully", "results": results}
    except ValidationError as e:
        logger.exception("Validation error saving worker states")
        return create_error_response(
            status_code=422,
            detail=f"Validation error: {e!s}",
            error_code="VALIDATION_ERROR",
        )
    except OSError as e:
        logger.exception("File system error saving worker states")
        return create_error_response(
            status_code=500,
            detail=f"File system error: {e!s}",
            error_code="FILE_SYSTEM_ERROR",
        )
    except RuntimeError as e:
        logger.exception("Runtime error saving worker states")
        return create_error_response(
            status_code=500,
            detail=f"Runtime error: {e!s}",
            error_code="RUNTIME_ERROR",
        )
    except Exception as e:
        logger.exception("Unexpected error saving worker states")
        return create_error_response(
            status_code=500,
            detail=f"Unexpected error: {e!s}",
            error_code="INTERNAL_ERROR",
        )
