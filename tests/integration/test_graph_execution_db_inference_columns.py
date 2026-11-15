"""Integration tests for the inference telemetry columns on
``graph_execution_sessions`` (Phase 1 inference-graph, migration 009).

These tests exercise the DB-layer against a REAL PostgreSQL 18 instance,
not the in-memory fake used by ``tests/test_graph_execution_db_unit.py``.

They are marked ``@pytest.mark.integration`` and are skipped by default
(see ``pyproject.toml`` addopts). To run locally against a local/CI
Postgres 18:

    DATABASE_URL=postgresql://user:pw@host:5432/worker_management_ci \
    POSTGRES_SSLMODE=disable \
    uv run pytest tests/integration/test_graph_execution_db_inference_columns.py \
        -v --no-cov -m integration

The suite is self-contained: it creates a disposable ``graphs`` row and a
disposable ``graph_execution_sessions`` row under a throwaway ``tenant_id``,
writes telemetry via ``update_session_status``, then reads it back via
``get_session`` and asserts round-trip equality including nested JSONB shape.

Migration policy: if the target DB was bootstrapped from ``postgres_schema.sql``
(which we updated in this task), the 7 columns already exist. If the DB is
stuck on an older schema missing the columns, the test applies migration 009
idempotently before running. Both paths are covered.
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


MIGRATION_009_SQL = """
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS chosen_provider TEXT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS fallback_count INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS prompt_tokens INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS completion_tokens INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS validation_passed BOOLEAN;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS request_payload JSONB;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS final_response JSONB;
CREATE INDEX IF NOT EXISTS graph_execution_sessions_graph_id_created_at_idx
    ON graph_execution_sessions (graph_id, created_at DESC);
"""


INFERENCE_TELEMETRY_COLUMNS: dict[str, str] = {
    "chosen_provider": "text",
    "fallback_count": "integer",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "validation_passed": "boolean",
    "request_payload": "jsonb",
    "final_response": "jsonb",
}


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    """Module-scoped PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
    require_database_url()
    # Default to non-SSL for local dev / CI unless caller overrides.
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
async def ensure_migration_009(pg_pool):
    """Apply migration 009 idempotently before each test.

    If the schema was bootstrapped from ``postgres_schema.sql`` the columns
    already exist and each statement is a no-op thanks to IF NOT EXISTS.
    If an older DB is missing the columns, this brings it up to date.
    """
    async with get_pg_connection() as conn:
        # psycopg v3 supports multi-statement execute when autocommit is on.
        for stmt in MIGRATION_009_SQL.strip().split(";\n"):
            sql = stmt.strip()
            if sql:
                await conn.execute(sql)


@pytest_asyncio.fixture
async def disposable_graph(ensure_migration_009):
    """Create a throwaway graph row under a unique tenant_id.

    Yields ``(tenant_id, graph_id)``. Cascaded deletes on
    ``graph_execution_sessions`` clean up session rows we create per test.
    """
    tenant_id = tenant_uuid(f"int-test-{uuid.uuid4().hex[:12]}")
    graph_id = str(uuid.uuid4())
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO graphs
                (id, name, tenant_id, tenant_name, subject_id, nodes, edges, variables)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            graph_id,
            f"int-test-graph-{graph_id[:8]}",
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
            # CASCADE on fk_graph_exec cleans up sessions too.
            await conn.execute("DELETE FROM graphs WHERE id = $1", graph_id)


class TestMigration009Schema:
    async def test_all_seven_columns_exist_with_correct_types(self, ensure_migration_009):
        """information_schema must report all 7 columns with expected data types."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'graph_execution_sessions'
                  AND column_name = ANY($1::text[])
                """,
                list(INFERENCE_TELEMETRY_COLUMNS.keys()),
            )
        by_name = {r["column_name"]: r for r in rows}
        for col, expected_type in INFERENCE_TELEMETRY_COLUMNS.items():
            assert col in by_name, f"Column {col!r} missing after migration 009"
            assert by_name[col]["data_type"].lower() == expected_type, (
                f"Column {col!r} has wrong type: "
                f"expected {expected_type!r}, got {by_name[col]['data_type']!r}"
            )
            assert by_name[col]["is_nullable"].upper() == "YES", (
                f"Column {col!r} must be nullable (no defaults)"
            )

    async def test_composite_index_exists(self, ensure_migration_009):
        """The trainer-pull index must exist after migration 009."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'graph_execution_sessions'
                  AND indexname = 'graph_execution_sessions_graph_id_created_at_idx'
                """,
            )
        assert row is not None, "composite (graph_id, created_at DESC) index missing"

    async def test_migration_is_idempotent(self, ensure_migration_009):
        """Re-applying migration 009 must not error (IF NOT EXISTS guards)."""
        async with get_pg_connection() as conn:
            for stmt in MIGRATION_009_SQL.strip().split(";\n"):
                sql = stmt.strip()
                if sql:
                    await conn.execute(sql)


class TestInferenceTelemetryRoundTrip:
    async def test_write_all_seven_columns_and_read_back(self, disposable_graph):
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

        await db.create_session(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            session_id=session_id,
            graph_id=graph_id,
        )
        updated = await db.update_session_status(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            "completed",
            outputs=ExecutionOutputs(final_response=final_response),
            telemetry=InferenceTelemetry(
                chosen_provider="vllm",
                fallback_count=1,
                prompt_tokens=42,
                completion_tokens=17,
                validation_passed=True,
            ),
            request_payload=request_payload,
        )
        assert updated is not None

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["chosen_provider"] == "vllm"
        assert fetched["fallback_count"] == 1
        assert fetched["prompt_tokens"] == 42
        assert fetched["completion_tokens"] == 17
        assert fetched["validation_passed"] is True
        assert fetched["request_payload"] == request_payload
        assert fetched["final_response"] == final_response

    async def test_partial_update_leaves_other_columns_null(self, disposable_graph):
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        await db.create_session(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            session_id=session_id,
            graph_id=graph_id,
        )
        await db.update_session_status(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            "running",
            telemetry=InferenceTelemetry(chosen_provider="Qwen3-4B-AWQ"),
        )

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["chosen_provider"] == "Qwen3-4B-AWQ"
        for col in (
            "fallback_count",
            "prompt_tokens",
            "completion_tokens",
            "validation_passed",
            "request_payload",
            "final_response",
        ):
            assert fetched[col] is None, f"expected NULL for {col}, got {fetched[col]!r}"

    async def test_jsonb_nested_round_trip(self, disposable_graph):
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        nested_payload = {
            "model": "local-llm",
            "metadata": {
                "tenant": "n8n-dev",
                "trace": {"id": "t-1", "sampled": True, "depth": 3},
            },
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.7,
        }
        nested_response = {
            "id": "resp-nested",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

        await db.create_session(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            session_id=session_id,
            graph_id=graph_id,
        )
        await db.update_session_status(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            "completed",
            outputs=ExecutionOutputs(final_response=nested_response),
            request_payload=nested_payload,
        )

        fetched = await db.get_session(
            tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id
        )
        assert fetched is not None
        assert fetched["request_payload"] == nested_payload
        assert fetched["final_response"] == nested_response
        # Spot-check deep values to catch silent dict-identity regressions.
        assert fetched["request_payload"]["metadata"]["trace"]["depth"] == 3
        assert fetched["final_response"]["usage"]["total_tokens"] == 15

    async def test_list_sessions_surfaces_inference_columns(self, disposable_graph):
        tenant_id, graph_id = disposable_graph
        db = GraphExecutionSessionDB()
        session_id = str(uuid.uuid4())

        await db.create_session(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            session_id=session_id,
            graph_id=graph_id,
        )
        await db.update_session_status(
            ScopeContext(tenant_id=tenant_id, subject_id=tenant_id, session_id=session_id),
            "completed",
            telemetry=InferenceTelemetry(
                chosen_provider="claude",
                fallback_count=2,
                validation_passed=True,
            ),
        )

        rows = await db.list_sessions(tenant_id=tenant_id, subject_id=tenant_id)
        assert len(rows) == 1
        assert rows[0]["chosen_provider"] == "claude"
        assert rows[0]["fallback_count"] == 2
        assert rows[0]["validation_passed"] is True
