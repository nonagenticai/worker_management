"""A non-UUID correlation_id must cost the join key, not the whole session row.

Why this exists
---------------
`graph_execution_sessions.correlation_id` is declared UUID (migration 012), but
the value arrives from the `X-Correlation-Id` HTTP header, which any caller can
set to anything. When a caller sends a non-UUID, postgres rejects the bind for
that ONE column and the ENTIRE INSERT fails::

    invalid input syntax for type uuid: "527162"
    CONTEXT: unnamed portal parameter $24

so status, tokens, provider, quality score -- the whole row -- is lost, and
`with_retry` burns three attempts on an error that can never succeed.

Measured on production 2026-08-14, this had been happening silently for six
weeks:

===========================================  ==================
`graph_execution_sessions` total rows        830
newest `created_at`                          **2026-06-28**
rows in the previous 24h                     **0**
uuid-cast failures in the live pod, 1 hour   10
===========================================  ==================

That is also why the #170 canary could never find a new session row.

The column type is NOT the bug and must not be widened: the fleet join key is
UUID everywhere it lands -- `billing.billing_cost_events`,
`metering.{outcome_event,exec_quality,cost_by_correlation}` and
`inference_router.replay_invocation_log` are all UUID columns. Widening only
here would break those joins while still failing to make "527162" joinable.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_correlation_id_must_not_destroy_the_session.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.graph_execution_db import GraphExecutionSessionDB

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

# The exact shape observed failing in production: an n8n execution id.
BAD_CORRELATION_ID = "527162"


@pytest_asyncio.fixture
async def pg_pool():
    require_database_url()
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


async def _session_row(session_id: str):
    async with get_pg_connection() as conn:
        return await conn.fetchrow(
            "SELECT session_id, status, correlation_id FROM graph_execution_sessions "
            "WHERE session_id = $1",
            session_id,
        )


async def _make_graph(tenant_id: str) -> str:
    """Insert a parent graph row.

    ``fk_graph_exec`` is ``graph_execution_sessions.graph_id -> graphs(id) ON
    DELETE CASCADE``, so a session cannot be written against an invented
    graph_id. Without this the tests would fail on the FK and never reach the
    correlation_id behaviour under test.
    """
    graph_id = str(uuid.uuid4())
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO graphs (id, name, tenant_id, tenant_name, subject_id,
                                visibility, description, nodes, edges, variables,
                                created_at, updated_at)
            -- `$3::uuid` is required now that `tenant_id` is a uuid column:
            -- the same bind parameter also feeds varchar columns here, and
            -- Postgres refuses to deduce two types for one parameter
            -- (`AmbiguousParameter: inconsistent types deduced for $3`).
            VALUES ($1, $2, $3::uuid, $3, $3, 'tenant', '', '[]'::jsonb, '[]'::jsonb,
                    '{}'::jsonb, now(), now())
            """,
            graph_id,
            f"g-{graph_id[:8]}",
            tenant_id,
        )
    return graph_id


async def test_non_uuid_correlation_id_still_persists_the_session(pg_pool):
    """The regression: before the fix this raised and wrote nothing at all."""
    db = GraphExecutionSessionDB()
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    session_id = str(uuid.uuid4())

    await db.create_session(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        session_id=session_id,
        graph_id=await _make_graph(tenant_id),
        correlation_id=BAD_CORRELATION_ID,
    )

    row = await _session_row(session_id)
    assert row is not None, (
        "the session row was not persisted at all -- one malformed optional "
        "field destroyed the entire execution record"
    )
    assert row["correlation_id"] is None, (
        "a non-UUID must be dropped to NULL, not stored; the column is UUID and "
        "the fleet join key is UUID everywhere"
    )


async def test_valid_uuid_correlation_id_is_preserved(pg_pool):
    """The guard: the fix must not throw away legitimate join keys."""
    db = GraphExecutionSessionDB()
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    session_id = str(uuid.uuid4())
    good = str(uuid.uuid4())

    await db.create_session(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        session_id=session_id,
        graph_id=await _make_graph(tenant_id),
        correlation_id=good,
    )

    row = await _session_row(session_id)
    assert row is not None
    assert str(row["correlation_id"]) == good


async def test_fetch_by_non_uuid_correlation_returns_empty_not_500(pg_pool):
    """A non-UUID lookup matches nothing; it must not raise."""
    db = GraphExecutionSessionDB()
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")

    rows = await db.fetch_sessions_by_correlation_id(
        tenant_id=tenant_id, subject_id=tenant_id, correlation_id=BAD_CORRELATION_ID
    )
    assert rows == []

    unscoped = await db.fetch_sessions_by_correlation_id_unscoped(correlation_id=BAD_CORRELATION_ID)
    assert unscoped == []


async def test_non_uuid_filter_matches_nothing_rather_than_everything(pg_pool):
    """The dangerous failure mode: a dropped filter would widen, not narrow.

    A caller that asks to narrow by correlation_id and instead receives EVERY
    session in its tenant is worse than an error. The unsatisfiable predicate
    keeps the narrowing honest.
    """
    db = GraphExecutionSessionDB()
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")

    await db.create_session(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        session_id=str(uuid.uuid4()),
        graph_id=await _make_graph(tenant_id),
        correlation_id=str(uuid.uuid4()),
    )

    filtered = await db.list_sessions(
        tenant_id=tenant_id, subject_id=tenant_id, correlation_id=BAD_CORRELATION_ID
    )
    assert filtered == [], (
        "a non-UUID correlation filter returned rows -- the clause was dropped "
        "and the query widened to the whole tenant"
    )

    unfiltered = await db.list_sessions(tenant_id=tenant_id, subject_id=tenant_id)
    assert len(unfiltered) == 1, "sanity: the tenant really does have one session"
