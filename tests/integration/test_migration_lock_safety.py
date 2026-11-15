"""Proof that a blocked migration fails fast instead of hanging the pod boot.

Regression guard for the migration convoy / doom loop (2026-08-08)
-----------------------------------------------------------------
``run_migrations()`` ran the whole 23-file chain on every pod boot with
``lock_timeout = 0`` -- block forever. Migrations 010 and 019 take ACCESS
EXCLUSIVE across 11 tables. A pod SIGKILLed mid-migration left its
PgBouncer-backed server connection alive and mid-transaction, still holding
ACCESS EXCLUSIVE; the next pod blocked on it and was itself killed by the
startupProbe at 300s (failureThreshold 30 x periodSeconds 10) plus
``terminationGracePeriodSeconds`` 60 -> 359s, exit 137, leaking another blocked
backend. Observed in production: one backend held the lock for 2h29m52s and 60
distinct backends unblocked within 10ms of its death.

The load-bearing test here holds ACCESS EXCLUSIVE on ``tools`` from a second
connection and asserts the migration run *returns* -- fast, with the blocked
files recorded as lock-timeout failures -- rather than hanging until the probe
fires. Against pre-fix code it does not return at all.

Also covers migration 002, which had never once succeeded in production: see
``migrations/002_fix_tool_tags.sql`` for the full drift story (production
``tools.tags`` is ``text[]``; ``postgres_schema.sql`` says ``JSONB``).

Run locally, in the shape CI actually uses -- note the ``+psycopg`` driver
suffix the archetype exports, which is a SQLAlchemy URL and NOT a libpq DSN
(see ``pg_guard.pg_conninfo``)::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_migration_lock_safety.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import psycopg
import pytest
import pytest_asyncio

from src.db.db_config import (
    _MIGRATION_LOCK_TIMEOUT,
    close_pg_pool,
    init_pg_pool,
)

from .pg_guard import (
    PG_CONNECT_ERRORS,
    apply_migrations,
    pg_conninfo,
    require_database_url,
    unreachable_pg,
)

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_MIGRATION_002 = _MIGRATIONS_DIR / "002_fix_tool_tags.sql"

# Files that touch `tools` and can therefore queue behind an ACCESS EXCLUSIVE
# lock on it. Each burns at most one `lock_timeout` before being recorded as a
# failure and stepped over, so the whole run is bounded by roughly
# len(_TOOLS_MIGRATIONS) * lock_timeout.
_TOOLS_MIGRATIONS = (
    "002_fix_tool_tags.sql",
    "010_per_user_within_tenant_isolation.sql",
    "011_add_tenant_name.sql",
    "019_drop_creator_id.sql",
)

# The production budget this fix exists to stay inside: startupProbe
# failureThreshold 30 x periodSeconds 10. Exceeding it is what SIGKILLed the
# pod and grew the convoy.
_STARTUP_PROBE_BUDGET_S = 300.0

# Generous upper bound for the blocked run: every `tools` migration waiting a
# full lock_timeout, plus slack for the ~19 unblocked files and fixture setup.
# The point is not the exact number -- it is that the run is *bounded at all*
# and finishes far inside the probe budget.
_MAX_BLOCKED_RUN_S = 90.0


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
async def blocker():
    """A second connection holding ACCESS EXCLUSIVE on ``tools``.

    Deliberately a raw connection rather than one from the pool under test, so
    the lock is genuinely held by a different backend -- the same shape as the
    orphaned PgBouncer server connection that caused the incident.

    ``autocommit=False`` is load-bearing twice over. It keeps the ``LOCK
    TABLE`` inside an open transaction, and the pool under test is opened
    ``autocommit=True`` (``init_pg_pool``'s ``conn_kwargs``), so borrowing a
    pool connection instead would release the lock the instant it was taken and
    the test would assert against nothing.
    """
    conn = await psycopg.AsyncConnection.connect(pg_conninfo(), autocommit=False)
    try:
        await conn.execute("LOCK TABLE tools IN ACCESS EXCLUSIVE MODE")
        # Prove the lock is held by a *different* backend than any pool
        # connection, so a future refactor cannot quietly neuter the test.
        cur = await conn.execute("SELECT pg_backend_pid()")
        row = await cur.fetchone()
        assert row is not None
        blocker_pid = row[0]
        cur = await conn.execute(
            "SELECT count(*) FROM pg_locks l JOIN pg_class c ON c.oid = l.relation "
            "WHERE c.relname = 'tools' AND l.mode = 'AccessExclusiveLock' "
            "AND l.granted AND l.pid = %s",
            (blocker_pid,),
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == 1, (
            f"blocker backend {blocker_pid} does not hold a granted "
            f"AccessExclusiveLock on `tools`; the test would assert nothing"
        )
        yield conn
    finally:
        await conn.rollback()
        await conn.close()


# `timeout` is what makes this test red rather than hung against pre-fix code:
# without lock_timeout the run blocks indefinitely on the first `tools`
# migration, so the guard below never gets to make its assertion.
@pytest.mark.timeout(180)
async def test_blocked_migration_fails_fast_instead_of_hanging(pg_pool, blocker):
    """A migration blocked on ACCESS EXCLUSIVE times out; it does not hang.

    Pre-fix (`lock_timeout = 0`) this never returns and dies on the 180s
    pytest timeout, mirroring the pod dying on its 300s startupProbe.
    """
    started = time.monotonic()
    applied, failed = await apply_migrations(strict=False)
    elapsed = time.monotonic() - started

    # 1. It returned at all, and inside the budget that used to kill the pod.
    assert elapsed < _MAX_BLOCKED_RUN_S, (
        f"blocked migration run took {elapsed:.1f}s (bound {_MAX_BLOCKED_RUN_S}s). "
        f"With lock_timeout unset this never returns and the startupProbe "
        f"SIGKILLs the pod at {_STARTUP_PROBE_BUDGET_S}s."
    )
    assert elapsed < _STARTUP_PROBE_BUDGET_S

    # 2. The blocked files are recorded as failures carrying a lock-timeout
    #    message -- i.e. they took the non-strict handler, not a hang.
    failed_names = {name for name, _ in failed}
    blocked = failed_names & set(_TOOLS_MIGRATIONS)
    assert blocked, (
        f"expected at least one `tools` migration to report a lock timeout; "
        f"failed={failed} applied={len(applied)}"
    )
    for name, msg in failed:
        if name in _TOOLS_MIGRATIONS:
            assert "lock timeout" in msg.lower(), (
                f"{name} failed for a reason other than the lock timeout: {msg}"
            )

    # 3. The run genuinely continued past the blocked files instead of aborting
    #    on the first one -- that is what lets the pod finish booting.
    assert applied, "expected unblocked migrations to still apply"


@pytest.mark.timeout(180)
async def test_lock_timeout_error_is_caught_by_the_non_strict_handler(pg_pool, blocker):
    """SQLSTATE 55P03 is a psycopg.Error, so `except psycopg.Error` catches it.

    The whole fix rests on the existing non-strict handler absorbing a lock
    timeout. Pin the exception hierarchy so a psycopg upgrade cannot quietly
    turn a fail-fast back into a crash-on-boot.
    """
    assert issubclass(psycopg.errors.LockNotAvailable, psycopg.Error)

    conn = await psycopg.AsyncConnection.connect(pg_conninfo(), autocommit=True)
    try:
        await conn.execute(f"SET lock_timeout = '{_MIGRATION_LOCK_TIMEOUT}'")
        with pytest.raises(psycopg.errors.LockNotAvailable) as excinfo:
            # LOCK TABLE is only legal inside a transaction block.
            async with conn.transaction():
                await conn.execute("LOCK TABLE tools IN ACCESS EXCLUSIVE MODE")
    finally:
        await conn.close()

    assert excinfo.value.sqlstate == "55P03"


async def test_migration_002_applies_to_prod_shaped_text_array_tags(pg_pool):
    """Migration 002 applies cleanly when `tools.tags` is `text[]` with NULLs.

    This is production's actual shape. Pre-fix the file said
    ``UPDATE tools SET tags = '[]'``, which PostgreSQL rejects against `text[]`
    at PLAN time -- ``ERROR: malformed array literal: "[]"`` -- even though the
    WHERE matches zero rows on a NOT NULL column. Here we force the harder
    case and seed real NULLs so the backfill has something to do.
    """
    sql_text = _MIGRATION_002.read_text(encoding="utf-8")
    conn = await psycopg.AsyncConnection.connect(pg_conninfo(), autocommit=True)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS mig002_probe CASCADE")
        await conn.execute("CREATE SCHEMA mig002_probe")
        # Replicate production's drifted column: text[], nullable, with NULLs.
        await conn.execute(
            "CREATE TABLE mig002_probe.tools ("
            "  id bigint PRIMARY KEY,"
            "  name varchar(255) NOT NULL,"
            "  tags text[] DEFAULT '{}'::text[]"
            ")"
        )
        await conn.execute(
            "INSERT INTO mig002_probe.tools (id, name, tags) VALUES "
            "(1, 'has-nulls', NULL), (2, 'has-tags', ARRAY['a','b']), (3, 'also-null', NULL)"
        )
        # Resolve the migration's unqualified `tools` to the prod-shaped table.
        await conn.execute("SET search_path TO mig002_probe, public")

        # The assertion: this no longer raises.
        await conn.execute(sql_text)

        cur = await conn.execute("SELECT id, tags FROM mig002_probe.tools ORDER BY id")
        rows = await cur.fetchall()

        assert [r[1] for r in rows] == [[], ["a", "b"], []], (
            f"NULL tags should be backfilled to the empty array, got {rows}"
        )
        assert all(r[1] is not None for r in rows), f"no tags may remain NULL: {rows}"

        # 002 also asserts NOT NULL; the backfill must have made that possible.
        cur = await conn.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema='mig002_probe' AND table_name='tools' AND column_name='tags'"
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == "NO", "002 should leave tags NOT NULL"
    finally:
        await conn.execute("DROP SCHEMA IF EXISTS mig002_probe CASCADE")
        await conn.close()


async def test_migration_chain_reports_no_failure_for_002(pg_pool):
    """The unblocked chain records zero failures for 002.

    In production `failed` contained 002 on every single boot (26 occurrences
    in Loki over 7 days) because the malformed array literal re-raised forever
    and 002 never recorded itself in `schema_migrations`.
    """
    applied, failed = await apply_migrations(strict=False)

    failed_names = {name for name, _ in failed}
    assert "002_fix_tool_tags.sql" not in failed_names, (
        f"migration 002 must apply cleanly; failures were {failed}"
    )
    assert failed == [], f"the chain should be failure-free, got {failed}"
    assert "002_fix_tool_tags.sql" in applied
