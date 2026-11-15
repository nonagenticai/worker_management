"""Unit tests for correlation_id propagation through GraphExecutionSessionDB.

The end-to-end story (n8n / fr-backend → worker_management → metering_service)
joins all three sources by a single `correlation_id` value carried in the
`X-Correlation-Id` header. This file pins the worker_management contract:

  * GraphExecutionSessionDB.create_session passes correlation_id into the
    INSERT.
  * GraphExecutionSessionDB.upsert_finalized_session passes correlation_id
    and uses COALESCE on the conflict path so a later upsert without the
    value cannot blank out an earlier-set id.

The DB call is mocked — these are unit tests for the SQL contract, not
integration tests. The integration story is covered by Task 13 in the plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.graph_execution_db import (
    ExecutionOutputs,
    GraphExecutionSessionDB,
    ScopeContext,
)


@pytest.fixture
def mock_pg():
    """Mock get_pg_connection async context manager + connection."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "session_id": "sess-1",
            "graph_id": "graph-1",
            "tenant_id": "tenant-A",
            "subject_id": "user-1",
            "correlation_id": "11111111-1111-1111-1111-111111111111",
        }
    )
    with patch("src.db.graph_execution_db.get_pg_connection") as m:
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        cm.__aexit__.return_value = None
        m.return_value = cm
        yield conn


@pytest.mark.asyncio
async def test_create_session_passes_correlation_id_to_insert(mock_pg):
    cid = "11111111-1111-1111-1111-111111111111"
    db = GraphExecutionSessionDB()
    await db.create_session(
        tenant_id="tenant-A",
        subject_id="user-1",
        session_id="sess-1",
        graph_id="graph-1",
        correlation_id=cid,
    )
    assert mock_pg.fetchrow.await_count == 1
    sql, *params = mock_pg.fetchrow.await_args.args
    assert "correlation_id" in sql.lower()
    # The correlation_id is the 11th positional parameter in the bound list
    # (see the create_session implementation). Look it up by value to keep
    # the test resilient to small column-order tweaks.
    assert cid in params, params


@pytest.mark.asyncio
async def test_create_session_correlation_id_optional(mock_pg):
    """Caller may omit correlation_id — None is forwarded, not an error."""
    db = GraphExecutionSessionDB()
    await db.create_session(
        tenant_id="tenant-A",
        subject_id="user-1",
        session_id="sess-2",
        graph_id="graph-1",
    )
    assert mock_pg.fetchrow.await_count == 1


@pytest.mark.asyncio
async def test_upsert_finalized_session_uses_coalesce_on_conflict(mock_pg):
    """The ON CONFLICT SET clause must COALESCE the correlation_id so that
    a later upsert call without the value does not blank out an earlier-set id.
    """
    db = GraphExecutionSessionDB()
    await db.upsert_finalized_session(
        ScopeContext(tenant_id="tenant-A", subject_id="user-1", session_id="sess-1"),
        "graph-1",
        "completed",
        outputs=ExecutionOutputs(node_outputs={}, completed_nodes=[], failed_nodes=[]),
        correlation_id=None,
    )
    sql, *_ = mock_pg.fetchrow.await_args.args
    sql_lower = sql.lower()
    # Must reference the column AND COALESCE on the conflict path.
    assert "correlation_id" in sql_lower
    assert "coalesce(excluded.correlation_id" in sql_lower.replace(" ", "")
