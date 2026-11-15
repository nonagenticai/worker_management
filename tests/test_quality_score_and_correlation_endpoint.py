"""Unit tests for Plan 5 PR 1: quality_score persistence + by-correlation endpoint.

Two concerns covered here so the diff stays surgical:

  1. ``GraphExecutionSessionDB.upsert_finalized_session`` accepts and persists
     ``quality_score`` (NUMERIC(5,4)) added by migration 018, including the
     ``ON CONFLICT SET`` path.
  2. ``GET /api/graph/sessions/by-correlation/{cid}`` returns the
     enrichment-ready row shape consumed by the metering_service exec_quality
     ETL: per-session typed columns plus a SUM(retry_count) rollup from
     node_execution_logs.

Tests are unit-level: the Postgres connection is mocked via ``get_pg_connection``.
The integration story (cross-service propagation from n8n → fr → worker_management
→ metering) is covered separately by the metering_service ETL test suite once
PR 2 lands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.auth import TenantContext
from src.db.graph_execution_db import (
    ExecutionOutputs,
    GraphExecutionSessionDB,
    InferenceTelemetry,
    ScopeContext,
)

CID = "11111111-1111-1111-1111-111111111111"


def _ctx(tenant_id: str, subject_id: str) -> TenantContext:
    """Build the TenantContext the migrated endpoint now expects."""
    return TenantContext(
        tenant_id=tenant_id, subject_id=subject_id, tenant_name="", visibility="tenant"
    )


@pytest.fixture
def mock_pg_upsert():
    """Mock get_pg_connection for upsert tests — fetchrow returns a row dict."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "session_id": "sess-1",
            "graph_id": "graph-1",
            "tenant_id": "tenant-A",
            "subject_id": "user-1",
            "correlation_id": CID,
            "quality_score": 0.8500,
        }
    )
    with patch("src.db.graph_execution_db.get_pg_connection") as m:
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        cm.__aexit__.return_value = None
        m.return_value = cm
        yield conn


@pytest.fixture
def mock_pg_fetch():
    """Mock get_pg_connection for fetch tests — fetch returns a list of rows."""
    conn = AsyncMock()
    with patch("src.db.graph_execution_db.get_pg_connection") as m:
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        cm.__aexit__.return_value = None
        m.return_value = cm
        yield conn


# --------------------------------------------------------------------------
# (1) quality_score persistence via upsert_finalized_session
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_finalized_session_passes_quality_score_to_insert(mock_pg_upsert):
    """quality_score should appear in the INSERT param list and the SQL column list."""
    db = GraphExecutionSessionDB()
    await db.upsert_finalized_session(
        ScopeContext(tenant_id="tenant-A", subject_id="user-1", session_id="sess-1"),
        "graph-1",
        "completed",
        outputs=ExecutionOutputs(node_outputs={}, completed_nodes=[], failed_nodes=[]),
        telemetry=InferenceTelemetry(validation_passed=True, quality_score=0.85),
    )

    sql, *params = mock_pg_upsert.fetchrow.await_args.args
    sql_lower = sql.lower()

    assert "quality_score" in sql_lower
    # ON CONFLICT path must update the column too — otherwise a later
    # finalize-call carrying a refreshed score would be dropped on the floor.
    assert "quality_score = excluded.quality_score" in sql_lower
    assert 0.85 in params


@pytest.mark.asyncio
async def test_upsert_finalized_session_quality_score_optional(mock_pg_upsert):
    """Caller may omit quality_score — None is forwarded, INSERT still runs."""
    db = GraphExecutionSessionDB()
    await db.upsert_finalized_session(
        ScopeContext(tenant_id="tenant-A", subject_id="user-1", session_id="sess-2"),
        "graph-1",
        "completed",
        outputs=ExecutionOutputs(node_outputs={}, completed_nodes=[], failed_nodes=[]),
    )
    assert mock_pg_upsert.fetchrow.await_count == 1
    _sql, *params = mock_pg_upsert.fetchrow.await_args.args
    # quality_score is bound positionally; with the value omitted, the
    # bound list must contain None at its slot rather than raising.
    assert None in params


# --------------------------------------------------------------------------
# (2) fetch_sessions_by_correlation_id — DB helper for the new endpoint
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_sessions_by_correlation_id_uses_scope_clause_and_correlation_predicate(
    mock_pg_fetch,
):
    """Query must filter by both scope (tenant/visibility) AND correlation_id,
    and roll up retry_count from node_execution_logs.
    """
    mock_pg_fetch.fetch = AsyncMock(
        return_value=[
            {
                "session_id": "sess-1",
                "correlation_id": CID,
                "tenant_id": "tenant-A",
                "subject_id": "user-1",
                "status": "completed",
                "validation_passed": True,
                "quality_score": 0.92,
                "created_at": "2026-05-18T00:00:00",
                "completed_at": "2026-05-18T00:00:30",
                "retry_count_sum": 3,
            }
        ]
    )

    db = GraphExecutionSessionDB()
    rows = await db.fetch_sessions_by_correlation_id(
        tenant_id="tenant-A",
        subject_id="user-1",
        correlation_id=CID,
    )

    assert len(rows) == 1
    assert rows[0]["retry_count_sum"] == 3
    assert rows[0]["quality_score"] == 0.92
    assert rows[0]["validation_passed"] is True

    sql, *params = mock_pg_fetch.fetch.await_args.args
    sql_lower = sql.lower()
    assert "correlation_id = $3" in sql_lower
    assert "tenant_id = $1" in sql_lower
    assert "sum(n.retry_count)" in sql_lower
    assert "node_execution_logs" in sql_lower
    assert "tenant-A" in params
    assert CID in params


@pytest.mark.asyncio
async def test_fetch_sessions_by_correlation_id_returns_empty_list_when_no_rows(
    mock_pg_fetch,
):
    """Helper returns [] when no rows match — the API layer translates to 404."""
    mock_pg_fetch.fetch = AsyncMock(return_value=[])
    db = GraphExecutionSessionDB()
    rows = await db.fetch_sessions_by_correlation_id(
        tenant_id="tenant-A",
        subject_id="user-1",
        correlation_id=CID,
    )
    assert rows == []


# --------------------------------------------------------------------------
# (3) End-to-end through the FastAPI endpoint
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_returns_404_for_invalid_uuid():
    """Path-param that is not a UUID returns 404 (not 422), matching the contract
    docstring — keep the ETL's error-handling branch single-status.
    """
    from src.api.error_handlers import APIError
    from src.api.graph_sessions_correlation_api import get_sessions_by_correlation_id

    with pytest.raises(APIError) as exc_info:
        await get_sessions_by_correlation_id(
            ctx=_ctx("tenant-A", "user-1"),
            correlation_id="not-a-uuid",
        )
    # NotFoundError inherits from APIError; status_code is 404.
    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_endpoint_returns_404_when_db_returns_empty(monkeypatch):
    """Empty DB result → 404 so the ETL can branch on status code alone."""
    from src.api import graph_sessions_correlation_api as mod
    from src.api.error_handlers import APIError

    async def _empty(self, **_kwargs):
        return []

    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id",
        _empty,
    )

    with pytest.raises(APIError) as exc_info:
        await mod.get_sessions_by_correlation_id(
            ctx=_ctx("tenant-A", "user-1"),
            correlation_id=CID,
        )
    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_endpoint_returns_typed_rows_on_happy_path(monkeypatch):
    """Happy path — typed response model populated from DB row dict."""
    from datetime import UTC, datetime

    from src.api import graph_sessions_correlation_api as mod

    created = datetime(2026, 5, 18, tzinfo=UTC)
    completed = datetime(2026, 5, 18, 0, 0, 30, tzinfo=UTC)

    async def _rows(self, **_kwargs):
        return [
            {
                "session_id": "sess-1",
                "correlation_id": CID,
                "tenant_id": "tenant-A",
                "subject_id": "user-1",
                "status": "completed",
                "validation_passed": True,
                "quality_score": 0.92,
                "retry_count_sum": 4,
                "created_at": created,
                "completed_at": completed,
            }
        ]

    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id",
        _rows,
    )

    out = await mod.get_sessions_by_correlation_id(
        ctx=_ctx("tenant-A", "user-1"),
        correlation_id=CID,
    )

    assert len(out) == 1
    row = out[0]
    assert row.session_id == "sess-1"
    assert row.correlation_id == CID
    assert row.validation_passed is True
    assert row.quality_score == pytest.approx(0.92)
    assert row.retry_count_sum == 4
    assert row.completed_at == completed


# --------------------------------------------------------------------------
# (4) Internal-service role bypass (metering-internal et al.)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_rejects_retired_metering_internal_role(monkeypatch, caplog):
    """Retirement regression: after the fleet rename, the legacy
    `metering-internal` role no longer grants the cross-tenant unscoped-read
    bypass — such a caller falls through to the SCOPED helper and triggers no
    audit-log line. The bypass is now keyed on `service-internal`.
    """
    import logging
    from datetime import UTC, datetime

    from src.api import graph_sessions_correlation_api as mod

    scoped_called = {"hit": False}
    unscoped_called = {"hit": False}

    async def _scoped(self, **_kwargs):
        scoped_called["hit"] = True
        # Return the caller's own in-tenant row so the endpoint does not 404.
        return [
            {
                "session_id": "sess-mine",
                "correlation_id": CID,
                "tenant_id": "tenant-MINE",
                "subject_id": "user-MINE",
                "status": "completed",
                "validation_passed": True,
                "quality_score": 0.9,
                "retry_count_sum": 0,
                "created_at": datetime(2026, 5, 18, tzinfo=UTC),
                "completed_at": datetime(2026, 5, 18, 0, 0, 5, tzinfo=UTC),
            }
        ]

    async def _unscoped(self, *, correlation_id):
        unscoped_called["hit"] = True
        return []

    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id",
        _scoped,
    )
    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id_unscoped",
        _unscoped,
    )

    # ContextVar shim — set caller roles + client_id as if license-authz
    # had injected them via x-caller-roles / x-caller-client-id headers.
    from src.api.identity_context import set_caller_context

    set_caller_context("metering-service", ("metering-internal",))

    caplog.set_level(logging.INFO, logger=mod.logger.name)

    await mod.get_sessions_by_correlation_id(
        ctx=_ctx("tenant-MINE", "user-MINE"),
        correlation_id=CID,
    )

    # Legacy role no longer bypasses: scoped path taken, unscoped NOT hit.
    assert scoped_called["hit"] is True
    assert unscoped_called["hit"] is False

    # No cross-tenant audit line for a non-internal caller.
    audit = next(
        (r for r in caplog.records if "cross-tenant by-correlation read" in r.message),
        None,
    )
    assert audit is None, "retired metering-internal must not trigger the cross-tenant path"


@pytest.mark.asyncio
async def test_endpoint_uses_unscoped_helper_with_service_internal_role(monkeypatch):
    """Rotation: a caller carrying only the new shared `service-internal` role
    gets the same cross-tenant unscoped-read bypass as metering-internal, so
    the internal ETL path keeps working once Keycloak renames the role.
    """
    from datetime import UTC, datetime

    from src.api import graph_sessions_correlation_api as mod
    from src.api.identity_context import set_caller_context

    scoped_called = {"hit": False}
    unscoped_called = {"hit": False}

    async def _scoped(self, **_kwargs):
        scoped_called["hit"] = True
        return []

    async def _unscoped(self, *, correlation_id):
        unscoped_called["hit"] = True
        assert correlation_id == CID
        return [
            {
                "session_id": "sess-y",
                "correlation_id": CID,
                "tenant_id": "tenant-OTHER",
                "subject_id": "user-z",
                "status": "completed",
                "validation_passed": True,
                "quality_score": 0.8,
                "retry_count_sum": 1,
                "created_at": datetime(2026, 5, 18, tzinfo=UTC),
                "completed_at": datetime(2026, 5, 18, 0, 0, 10, tzinfo=UTC),
            }
        ]

    monkeypatch.setattr(mod.GraphExecutionSessionDB, "fetch_sessions_by_correlation_id", _scoped)
    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id_unscoped",
        _unscoped,
    )

    set_caller_context("some-service", ("service-internal",))

    out = await mod.get_sessions_by_correlation_id(
        ctx=_ctx("tenant-MINE", "user-MINE"),
        correlation_id=CID,
    )

    assert unscoped_called["hit"] is True
    assert scoped_called["hit"] is False
    assert out[0].tenant_id == "tenant-OTHER"


@pytest.mark.asyncio
async def test_endpoint_does_not_log_audit_when_caller_is_in_tenant(monkeypatch, caplog):
    """In-tenant callers (no internal role) hit the scoped helper and do NOT
    trigger the audit-log line. Keeps log volume sane for normal traffic.
    """
    import logging
    from datetime import UTC, datetime

    from src.api import graph_sessions_correlation_api as mod
    from src.api.identity_context import set_caller_context

    async def _scoped(self, **_kwargs):
        return [
            {
                "session_id": "sess-1",
                "correlation_id": CID,
                "tenant_id": "tenant-MINE",
                "subject_id": "user-MINE",
                "status": "completed",
                "validation_passed": True,
                "quality_score": 0.9,
                "retry_count_sum": 0,
                "created_at": datetime(2026, 5, 18, tzinfo=UTC),
                "completed_at": datetime(2026, 5, 18, 0, 0, 5, tzinfo=UTC),
            }
        ]

    monkeypatch.setattr(
        mod.GraphExecutionSessionDB,
        "fetch_sessions_by_correlation_id",
        _scoped,
    )

    # No internal-service role.
    set_caller_context("some-app", ())

    caplog.set_level(logging.INFO, logger=mod.logger.name)

    out = await mod.get_sessions_by_correlation_id(
        ctx=_ctx("tenant-MINE", "user-MINE"),
        correlation_id=CID,
    )

    assert len(out) == 1
    audit = [r for r in caplog.records if "cross-tenant by-correlation read" in r.message]
    assert audit == [], "audit log fired for in-tenant caller; should only fire on bypass"
