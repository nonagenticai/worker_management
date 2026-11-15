"""Rescue rate must be answerable in SQL, not by grepping logs.

Why this exists
---------------
When an LLM stage fails transiently, `send_message` recreates the worker and
retries (`message_endpoints.py`, the `for attempt in range(1, llm_retry_max_attempts)`
loop). On success it logged::

    Worker %s recovered on LLM retry %d/%d

and that was the ONLY record. So "how often does a retry actually save a
request?" could only be answered by parsing Loki -- which retains ~16 days here
(memory `loki-history-starts-at-the-worker4-migration`), while `worker_states`
is the durable record and holds 892k rows.

Writing the attempt number onto the completed state makes the rate a real
ratio. Both paths write it: 0 on the primary run, >= 1 on the rescue, so the
denominator exists::

    SELECT count(*) FILTER (WHERE (worker_state->>'retry_attempt')::int > 0)
         / NULLIF(count(*), 0)::float AS rescue_rate
    FROM worker_states WHERE state_type = 'completed';

No migration: `worker_state` is JSONB.

These tests assert the SHAPE the query depends on, against a real database --
a unit test against the in-memory fake would not prove the JSONB round-trip or
that the `->>` cast works.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_rescue_rate_is_queryable.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.workers_state_db import StateDatabase

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


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


async def _completed(db: StateDatabase, *, tenant_id: str, name: str, retry_attempt: int) -> None:
    await db.add_worker_state(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_id=name,
        worker_name=name,
        is_agent=True,
        state_type="completed",
        worker_state={
            "status": "completed",
            "message": "m",
            "retry_attempt": retry_attempt,
            "output": "ok",
        },
    )


async def test_retry_attempt_round_trips_as_queryable_jsonb(pg_pool):
    """The value must survive the write and be castable to int in SQL."""
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    db = StateDatabase()
    await _completed(db, tenant_id=tenant_id, name="primary", retry_attempt=0)
    await _completed(db, tenant_id=tenant_id, name="rescued", retry_attempt=3)

    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT worker_name, (worker_state->>'retry_attempt')::int AS attempt "
            "FROM worker_states WHERE tenant_id = $1 ORDER BY worker_name",
            tenant_id,
        )

    got = {r["worker_name"]: r["attempt"] for r in rows}
    assert got == {"primary": 0, "rescued": 3}, got


async def test_rescue_rate_query_computes(pg_pool):
    """The actual ratio the task asked for -- 1 rescue in 4 completions = 0.25."""
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    db = StateDatabase()
    for i in range(3):
        await _completed(db, tenant_id=tenant_id, name=f"p{i}", retry_attempt=0)
    await _completed(db, tenant_id=tenant_id, name="r0", retry_attempt=2)

    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT count(*) FILTER (WHERE (worker_state->>'retry_attempt')::int > 0)::float
                 / NULLIF(count(*), 0)::float AS rescue_rate,
                   count(*) AS completions
            FROM worker_states
            WHERE tenant_id = $1 AND state_type = 'completed'
            """,
            tenant_id,
        )

    assert row["completions"] == 4
    assert row["rescue_rate"] == pytest.approx(0.25), row["rescue_rate"]


async def test_rows_without_the_key_do_not_break_the_query(pg_pool):
    """Historical rows predate the field; the query must tolerate them.

    892k rows already exist without `retry_attempt`. `->>` yields NULL there,
    and NULL is neither > 0 nor counted by the FILTER, so old rows land in the
    denominator only -- they are genuinely completions of unknown retry status.
    A cast that raised on NULL would make the whole query unusable, so this is
    the case worth pinning.
    """
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    db = StateDatabase()
    # a legacy-shaped row: no retry_attempt key at all
    await db.add_worker_state(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_id="legacy",
        worker_name="legacy",
        is_agent=True,
        state_type="completed",
        worker_state={"status": "completed", "message": "m", "output": "ok"},
    )
    await _completed(db, tenant_id=tenant_id, name="rescued", retry_attempt=1)

    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT count(*) FILTER (WHERE (worker_state->>'retry_attempt')::int > 0) AS rescued,
                   count(*) AS completions
            FROM worker_states
            WHERE tenant_id = $1 AND state_type = 'completed'
            """,
            tenant_id,
        )

    assert (row["rescued"], row["completions"]) == (1, 2)
