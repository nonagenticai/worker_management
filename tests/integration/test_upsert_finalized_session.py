"""Integration tests for ``GraphExecutionSessionDB.upsert_finalized_session``.

These exercise the real Postgres 18 ``INSERT ... ON CONFLICT (session_id)
DO UPDATE`` statement, matching the schema's ``UNIQUE (session_id)``
constraint (see ``postgres_schema.sql`` line 92).

They are marked ``@pytest.mark.integration`` and gated on ``DATABASE_URL``
exactly like the existing telemetry-columns integration tests.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.graph_execution_db import (
    ExecutionOutputs,
    GraphExecutionSessionDB,
    InferenceTelemetry,
    ScopeContext,
)

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    """Module-scoped PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture
async def disposable_graph(pg_pool):
    """Create a throwaway graph row under a unique tenant_id.

    Yields ``(tenant_id, graph_id)``. CASCADE on ``fk_graph_exec`` cleans up
    any session rows created by the test.
    """
    tenant_id = tenant_uuid(f"int-upsert-{uuid.uuid4().hex[:12]}")
    graph_id = str(uuid.uuid4())
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO graphs
                (id, name, tenant_id, tenant_name, subject_id, nodes, edges, variables)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            graph_id,
            f"upsert-test-graph-{graph_id[:8]}",
            tenant_id,
            tenant_id,  # tenant_name (denormalized; mirrors migration 011 backfill)
            tenant_id,
            [],
            [],
            {},
        )
    try:
        yield tenant_id, graph_id
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM graphs WHERE id = $1", graph_id)


class TestUpsertFinalizedSession:
    async def test_fresh_insert_with_full_inference_telemetry(self, disposable_graph):
        """First-time upsert writes a new row populated with all 7 fields."""
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        request_payload = {
            "model": "local-llm",
            "messages": [{"role": "user", "content": "hi"}],
        }
        final_response = {
            "id": "resp-42",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        }

        row = await db.upsert_finalized_session(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            graph_id,
            "completed",
            outputs=ExecutionOutputs(
                node_outputs={"n1": "ok"},
                completed_nodes=["n1"],
                failed_nodes=[],
                node_errors=None,
                worker_assignments={"n1": "w1"},
                final_response=final_response,
            ),
            telemetry=InferenceTelemetry(
                chosen_provider="Qwen3-4B-AWQ",
                fallback_count=0,
                prompt_tokens=42,
                completion_tokens=17,
                validation_passed=True,
            ),
            error_message=None,
            request_payload=request_payload,
        )
        assert row is not None
        assert str(row["session_id"]) == session_id
        assert row["status"] == "completed"

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["chosen_provider"] == "Qwen3-4B-AWQ"
        assert fetched["fallback_count"] == 0
        assert fetched["prompt_tokens"] == 42
        assert fetched["completion_tokens"] == 17
        assert fetched["validation_passed"] is True
        assert fetched["request_payload"] == request_payload
        assert fetched["final_response"] == final_response
        assert fetched["completed_at"] is not None
        assert fetched["node_outputs"] == {"n1": "ok"}
        assert fetched["completed_nodes"] == ["n1"]

    async def test_upsert_overwrites_existing_row(self, disposable_graph):
        """Calling create_session first, then upsert, exercises the ON CONFLICT path."""
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        await db.create_session(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            session_id=session_id,
            graph_id=graph_id,
            status="running",
        )

        row = await db.upsert_finalized_session(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            graph_id,
            "completed",
            outputs=ExecutionOutputs(
                node_outputs={"n1": "final"},
                completed_nodes=["n1"],
                failed_nodes=[],
            ),
            telemetry=InferenceTelemetry(
                chosen_provider="vllm",
                fallback_count=1,
                validation_passed=False,
            ),
        )
        assert row is not None
        assert row["status"] == "completed"

        # Only one row — ON CONFLICT updated rather than inserted.
        async with get_pg_connection() as conn:
            count_row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM graph_execution_sessions "
                "WHERE tenant_id = $1 AND session_id = $2",
                tenant_id,
                session_id,
            )
        assert count_row is not None
        assert count_row["n"] == 1

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["status"] == "completed"
        assert fetched["chosen_provider"] == "vllm"
        assert fetched["fallback_count"] == 1
        assert fetched["validation_passed"] is False
        assert fetched["node_outputs"] == {"n1": "final"}

    async def test_total_failure_projection(self, disposable_graph):
        """All providers failed: chosen_provider=None, fallback_count=N."""
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        await db.upsert_finalized_session(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            graph_id,
            "failed",
            outputs=ExecutionOutputs(
                node_outputs={},
                completed_nodes=[],
                failed_nodes=["n1"],
            ),
            telemetry=InferenceTelemetry(
                chosen_provider=None,
                fallback_count=3,
                prompt_tokens=None,
                completion_tokens=None,
                validation_passed=None,
            ),
            error_message="all providers failed",
        )

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["chosen_provider"] is None
        assert fetched["fallback_count"] == 3
        assert fetched["prompt_tokens"] is None
        assert fetched["completion_tokens"] is None
        assert fetched["validation_passed"] is None
        assert fetched["status"] == "failed"
        assert fetched["error_message"] == "all providers failed"

    async def test_non_inference_graph_leaves_columns_null(self, disposable_graph):
        """Non-inference graph: all 7 telemetry kwargs default to None → NULL."""
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        await db.upsert_finalized_session(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            graph_id,
            "completed",
            outputs=ExecutionOutputs(
                node_outputs={"n1": "out"},
                completed_nodes=["n1"],
                failed_nodes=[],
            ),
        )

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        for col in (
            "chosen_provider",
            "fallback_count",
            "prompt_tokens",
            "completion_tokens",
            "validation_passed",
            "request_payload",
            "final_response",
        ):
            assert fetched[col] is None, f"expected NULL for {col}, got {fetched[col]!r}"

    async def test_cross_tenant_upsert_raises_permission_error(self, disposable_graph):
        """A second creator upserting an existing session_id must raise and leave the original row untouched.

        The schema's ``UNIQUE (session_id)`` constraint means the ON CONFLICT
        target can match rows owned by a different ``tenant_id``. Without a
        tenant guard on the UPDATE, creator B could overwrite creator A's
        finalized row simply by reusing the session_id. This test pins the
        raise-on-mismatch contract so the tenant boundary cannot silently
        regress.
        """
        creator_alice, graph_id = disposable_graph
        # Second disposable tenant reusing the same graph row so the FK holds
        # (both sessions refer to the same graph; the schema does not require
        # graph ownership to match session ownership).
        creator_bob = tenant_uuid(f"int-upsert-bob-{uuid.uuid4().hex[:12]}")
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        # Alice creates the session first.
        await db.create_session(
            tenant_id=creator_alice,
            subject_id=creator_alice,
            session_id=session_id,
            graph_id=graph_id,
            status="running",
        )

        # Bob tries to finalize the same session_id. Must raise.
        with pytest.raises(PermissionError, match="different tenant or user"):
            await db.upsert_finalized_session(
                ScopeContext(tenant_id=creator_bob, subject_id=creator_bob, session_id=session_id),
                graph_id,
                "completed",
                outputs=ExecutionOutputs(
                    node_outputs={"pwned": "yes"},
                    completed_nodes=["pwned"],
                    failed_nodes=[],
                ),
                telemetry=InferenceTelemetry(
                    chosen_provider="evil",
                    fallback_count=0,
                    validation_passed=True,
                ),
            )

        # Alice's row is untouched: same tenant_id, status still "running",
        # no node_outputs from Bob, no inference telemetry from Bob.
        fetched = await db.get_session(
            tenant_id=creator_alice, subject_id=creator_alice, session_id=session_id
        )
        assert fetched is not None
        assert fetched["tenant_id"] == creator_alice
        assert fetched["status"] == "running"
        assert fetched["chosen_provider"] is None
        # node_outputs in create_session is left as DB default (NULL / {}).
        assert fetched.get("node_outputs") in (None, {})

        # Bob cannot see the row at all via his creator scope.
        assert (
            await db.get_session(
                tenant_id=creator_bob, subject_id=creator_bob, session_id=session_id
            )
            is None
        )
