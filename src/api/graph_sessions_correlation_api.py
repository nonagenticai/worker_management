"""Session enrichment endpoint indexed by correlation_id.

Plan 5 (AA observability redesign) needs a stable, typed path for the
``metering_service`` ETL to enrich ``exec_quality.{validation_passed,
quality_score, retry_count}`` — three columns left NULL today because the
data lives in the separate ``worker_management`` database. This module
exposes ``GET /api/graph/sessions/by-correlation/{cid}`` returning the
per-session typed columns plus a SUM(retry_count) rollup from
``node_execution_logs``.

Mounted under the ``/api/`` prefix per Movies/CLAUDE.md; the legacy
``/execution/...`` routes in ``graph_execution_api.py`` stay alongside.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.error_handlers import InternalServerError, NotFoundError
from src.api.identity_context import (
    get_caller_client_id_from_context,
    get_caller_roles_from_context,
)
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import ROUTE_GRAPH_SESSIONS_CORRELATION, tokenless_tenant_gate
from src.db.graph_execution_db import GraphExecutionSessionDB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/graph", tags=["Graph Sessions"])

# Internal-service roles allowed to read sessions across tenant boundaries via
# this endpoint. Mirrors the precedent set in
# ``licenses/auth/dependencies.py:93-108`` (``INTERNAL_SERVICE_ROLES`` + the
# ``require_internal_service()`` dep) where the licenses service solved the
# same cross-tenant-read problem when metering needed enrichment data. The
# roles travel via the ``x-caller-roles`` header license-authz injects after
# verifying the upstream Keycloak SA token (see
# ``src/api/auth_middleware.py:139-142`` for ingestion).
# The shared internal-service role was rotated `metering-internal` ->
# `service-internal` fleet-wide; the legacy `metering-internal` was retired
# once every SA carried `service-internal`.
_INTERNAL_SERVICE_ROLES: frozenset[str] = frozenset(
    {"service-internal", "onboarding-internal", "license-admin"}
)


class SessionByCorrelationResponse(BaseModel):
    """Typed response row for the by-correlation endpoint.

    Mirrors the columns the metering ETL writes into ``exec_quality``;
    ``retry_count_sum`` is the per-session SUM across
    ``node_execution_logs.retry_count``.
    """

    session_id: str
    correlation_id: str
    tenant_id: str
    subject_id: str | None = None
    status: str
    validation_passed: bool | None = None
    quality_score: float | None = None
    retry_count_sum: int = 0
    created_at: datetime
    completed_at: datetime | None = None


@router.get(
    "/sessions/by-correlation/{correlation_id}",
    response_model=list[SessionByCorrelationResponse],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_sessions_by_correlation_id",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_SESSIONS_CORRELATION))],
)
async def get_sessions_by_correlation_id(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    correlation_id: str = Path(
        ...,
        description=(
            "End-to-end correlation/trace UUID propagated from n8n / "
            "fr-backend via the X-Correlation-Id header."
        ),
    ),
) -> list[SessionByCorrelationResponse]:
    """Return all sessions tagged with ``correlation_id`` under the caller's scope.

    Used by the metering_service ``exec_quality`` ETL to enrich the three
    columns currently left NULL (``validation_passed``, ``quality_score``,
    ``retry_count``). Returns ``[]`` → HTTP 404 so the ETL can branch on
    a single status code; 200 with an empty list would force the ETL to
    inspect the body to disambiguate "not yet" from "service error".

    Visibility (two paths):

    * **In-tenant callers** (default): tenant_id-restricted; user-private
      rows belonging to other subjects are filtered out. Cross-tenant
      requests for an existing correlation_id return 404 (not 403) to
      avoid leaking existence per the service-isolation invariant.

    * **Internal-service callers** (caller in ``_INTERNAL_SERVICE_ROLES``,
      e.g. metering-internal): scope filter is bypassed; the row is
      returned regardless of which tenant owns it. Every cross-tenant
      read is audit-logged with caller client_id + observed tenant_id.
    """
    try:
        UUID(correlation_id)
    except ValueError as exc:
        raise NotFoundError("session by correlation_id", correlation_id) from exc

    caller_roles = get_caller_roles_from_context()
    is_internal_service = any(r in _INTERNAL_SERVICE_ROLES for r in caller_roles)

    try:
        db = GraphExecutionSessionDB()
        if is_internal_service:
            rows = await db.fetch_sessions_by_correlation_id_unscoped(
                correlation_id=correlation_id,
            )
        else:
            rows = await db.fetch_sessions_by_correlation_id(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                correlation_id=correlation_id,
            )
    except Exception as exc:
        logger.error(
            "fetch_sessions_by_correlation_id error for cid=%s: %s",
            correlation_id,
            exc,
            exc_info=True,
        )
        raise InternalServerError(f"sessions-by-correlation error: {exc}") from exc

    if not rows:
        raise NotFoundError("session by correlation_id", correlation_id)

    if is_internal_service:
        # Audit cross-tenant reads so misuse of the internal-service role
        # bypass is discoverable from logs alone. caller client_id comes
        # from license-authz's `x-caller-client-id` header; tenant_id is
        # the per-row owner (may differ across rows for a multi-tenant
        # correlation_id, hence the comprehension).
        observed_tenants = sorted({str(r.get("tenant_id") or "") for r in rows})
        logger.info(
            "cross-tenant by-correlation read: caller=%s roles=%s cid=%s tenants=%s rows=%d",
            get_caller_client_id_from_context() or "<unknown>",
            ",".join(caller_roles),
            correlation_id,
            ",".join(observed_tenants),
            len(rows),
        )

    out: list[SessionByCorrelationResponse] = []
    for r in rows:
        out.append(
            SessionByCorrelationResponse(
                session_id=str(r["session_id"]),
                correlation_id=str(r["correlation_id"]),
                tenant_id=str(r["tenant_id"]),
                subject_id=(str(r["subject_id"]) if r.get("subject_id") is not None else None),
                status=str(r["status"]),
                validation_passed=_as_optional_bool(r.get("validation_passed")),
                quality_score=_as_optional_float(r.get("quality_score")),
                retry_count_sum=int(r.get("retry_count_sum") or 0),
                created_at=_as_datetime(r["created_at"]),
                completed_at=_as_optional_datetime(r.get("completed_at")),
            )
        )
    return out


def _as_optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _as_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _as_datetime(value)
