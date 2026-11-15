"""Migration-chain proof against a real PostgreSQL 18.

Runs the **whole** ``migrations/*.sql`` chain in ``strict=True`` mode on top of
a database already bootstrapped from ``postgres_schema.sql`` (CI applies it via
``scripts/init_db.py``; locally run that script first — see below). The chain
is expected to be fully Postgres-clean and idempotent now that migration 001
(the lone SQLite-flavored file) has been removed: every remaining file uses
``IF NOT EXISTS`` / ``information_schema`` guards, so on the end-state schema
they all no-op without error.

This is the regression guard for the ``creator_id → tenant_id`` schema rewrite
(worker_management #57/#58): it proves migrations 010 and 019 no-op against the
honest base schema and that ``creator_id`` is gone for good.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_migration_chain.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

from src.db import db_config
from src.db.db_config import (
    close_pg_pool,
    get_pg_connection,
    init_pg_pool,
    run_migrations,
)

from .pg_guard import (
    PG_CONNECT_ERRORS,
    apply_migrations,
    require_database_url,
    unreachable_pg,
)

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

# Tables whose creator_id → tenant_id rewrite migrations 010/019 codify.
_OWNED_TABLES = ("workers", "graphs", "tools", "skills", "graph_execution_sessions")


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


async def test_migration_chain_runs_clean_and_idempotent(pg_pool):
    """Strict run applies every remaining migration with zero failures, twice."""
    applied1, failed1 = await apply_migrations()

    # strict=True re-raises on the first error, so reaching here already means
    # nothing failed — assert it explicitly for a clear failure message.
    assert failed1 == [], f"strict migration run reported failures: {failed1}"
    assert applied1, "expected the migration chain to apply at least one file"

    # 001 is gone; the canonical tenant-isolation migrations are present.
    assert not any(name.startswith("001_") for name in applied1), (
        f"migration 001 should be deleted but appeared in {applied1}"
    )
    assert "010_per_user_within_tenant_isolation.sql" in applied1
    assert "019_drop_creator_id.sql" in applied1

    # Idempotent, and now cheap with it. This used to assert
    # `applied2 == applied1` -- i.e. the whole chain re-executes every run, and
    # is merely harmless because each file is written with IF NOT EXISTS. That
    # is a weaker property than it looks: on production those re-runs still
    # demanded ACCESS EXCLUSIVE on live tables, and 8 of 26 files failed on
    # every single pod boot with `canceling statement due to lock timeout`.
    #
    # With the `schema_migration_files` ledger the second run must apply
    # nothing EXCEPT the files that declare `migrate:always`. Asserting an exact
    # set rather than equality with `applied1` is the stronger claim: equality
    # passes whether or not the ledger works.
    #
    # The expected set is derived from the files themselves, not hardcoded, so
    # this keeps covering whatever is added tomorrow. Today it is exactly 028,
    # the INVALID-index reaper, which is a control loop: recorded once and
    # skipped thereafter it would never reap anything again.
    applied2, failed2 = await run_migrations(strict=True)
    assert failed2 == [], f"second strict run reported failures: {failed2}"

    always = sorted(
        path.name
        for path in sorted((Path(__file__).resolve().parents[2] / "migrations").glob("*.sql"))
        if db_config._declares_directive(
            path.read_text(encoding="utf-8"), db_config._MIGRATION_ALWAYS_DIRECTIVE
        )
    )
    assert sorted(applied2) == always, (
        f"a second run must apply exactly the `migrate:always` files and skip "
        f"every other already-applied one. It re-ran {sorted(applied2)}; "
        f"expected {always}. More than that means the ledger is not skipping "
        f"and every boot re-takes ACCESS EXCLUSIVE on live tables."
    )


async def test_end_state_schema_is_tenant_native(pg_pool):
    """After the chain, owned tables are tenant-native with no creator_id."""
    await apply_migrations()

    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY($1::text[])
              AND column_name IN ('creator_id', 'tenant_id', 'subject_id', 'visibility')
            """,
            list(_OWNED_TABLES),
        )

    by_table: dict[str, set[str]] = {}
    for row in rows:
        by_table.setdefault(row["table_name"], set()).add(row["column_name"])

    for table in _OWNED_TABLES:
        cols = by_table.get(table, set())
        assert "creator_id" not in cols, f"{table}.creator_id should be gone"
        assert {"tenant_id", "subject_id", "visibility"} <= cols, (
            f"{table} missing tenant-native columns; found {cols}"
        )
