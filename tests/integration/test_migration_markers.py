"""The runner's two file-level directives, proved against a real PostgreSQL 18.

#202 added the ledger and the ``migrate:no-transaction`` directive. Both are
claims about what the DATABASE does, and #202's only coverage of the directive
is against ``_FakeConn``, which never parses the SQL it is handed. That fake
cannot see the two facts this module measures:

* ``CREATE INDEX CONCURRENTLY`` genuinely succeeds through the runner when the
  directive is present, and genuinely FAILS when it is absent -- the half that
  proves the directive is doing the work rather than being inert.
* a **multi-statement** no-transaction file cannot run at all. psycopg sends it
  through the simple-query protocol, which wraps the whole string in an implicit
  transaction, and PostgreSQL answers ``DROP INDEX CONCURRENTLY cannot run
  inside a transaction block``. The drop-then-create pair that #202 documented
  and unit-tested is exactly that shape.

It then covers the two things this branch adds on top of #202:

* ``migrate:always`` -- a file the ledger must never retire, because its value
  is continuous re-assertion rather than a one-time change.
* migration 028, which reaps an index left ``indisvalid = false`` by a failed
  concurrent build. ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` sees only the
  NAME, skips, and reports success, so without 028 the migration is recorded as
  applied, permanently, while the object it was written to create does nothing.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql://user:pw@127.0.0.1:PORT/worker_management_test"  # pragma: allowlist secret
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_migration_markers.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

from src.db import db_config
from src.db.db_config import (
    close_pg_pool,
    get_pg_connection,
    init_pg_pool,
    run_migrations,
)

from .pg_guard import PG_CONNECT_ERRORS, forget_all_migrations, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

_ALWAYS = db_config._MIGRATION_ALWAYS_DIRECTIVE
_NO_TX = db_config._MIGRATION_NO_TRANSACTION_DIRECTIVE
_028 = "028_reap_invalid_worker_states_indexes.sql"


def _migration_paths() -> list[Path]:
    """Every shipped migration file, read from the repo, not a hardcoded list.

    Derived rather than enumerated on purpose: a hand-listed set silently stops
    covering the file somebody adds tomorrow.
    """
    return sorted((Path(__file__).resolve().parents[2] / "migrations").glob("*.sql"))


@pytest_asyncio.fixture
async def pg_pool():
    """PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
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


async def _forget_files(*filenames: str) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            "DELETE FROM schema_migration_files WHERE filename = ANY($1::text[])",
            list(filenames),
        )


async def _index_validity(index_name: str) -> Optional[bool]:
    """``None`` if the index does not exist, else its ``pg_index.indisvalid``.

    Three-valued on purpose. "Absent" and "present but invalid" are different
    facts, and an assertion that cannot tell them apart would pass against a
    reap that never ran because the index was never built in the first place.
    """
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT i.indisvalid
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = $1
            """,
            index_name,
        )
    return None if row is None else bool(row["indisvalid"])


# ---------------------------------------------------------------------------
# 1. `migrate:always` survives the ledger.
# ---------------------------------------------------------------------------


async def test_a_recorded_always_file_still_runs_while_its_neighbours_are_skipped(pg_pool):
    """The ledger retires ordinary files and must never retire an `always` one.

    Both halves in one assertion on purpose. "028 ran again" alone would pass
    against a runner whose ledger was broken and re-ran everything, which is the
    lock storm #202 removed. What has to be true is that the SECOND run applies
    exactly the `migrate:always` set and nothing else.
    """
    await forget_all_migrations()
    first, failed = await run_migrations(strict=True)
    assert failed == [], f"first run reported failures: {failed}"
    assert _028 in first, "028 did not apply on a clean ledger"

    second, failed = await run_migrations(strict=True)
    assert failed == [], f"second run reported failures: {failed}"

    expected = sorted(
        path.name
        for path in _migration_paths()
        if db_config._declares_directive(path.read_text(encoding="utf-8"), _ALWAYS)
    )
    assert sorted(second) == expected, (
        f"a second run must apply exactly the `{_ALWAYS}` files. Got {sorted(second)}, "
        f"expected {expected}. Too many means the ledger is not skipping and every "
        f"boot re-takes ACCESS EXCLUSIVE; too few means an `{_ALWAYS}` control loop "
        f"was silently retired into a one-time historical event."
    )
    assert expected, "no shipped file declares the directive, so this test is vacuous"


async def test_a_file_that_merely_mentions_a_directive_does_not_declare_it(pg_pool, tmp_path: Path):
    """Prose about a directive must not arm it.

    This is not hypothetical in this repository: migration headers here run to
    dozens of lines and discuss the runner's own mechanics by name. 027 already
    quotes `CREATE INDEX CONCURRENTLY IF NOT EXISTS` and explains the INVALID
    trap; 028 has to name `migrate:always` to explain why it carries it.

    Under a substring test over the header, the file below -- which says it is
    NOT a no-transaction file -- would be run in autocommit with no lock_timeout
    guard and no rollback. It is written the way a real author would write it.
    """
    name = "903_mentions_only.sql"
    (tmp_path / name).write_text(
        f"-- {name}\n"
        f"-- Unlike 028, this file is deliberately NOT {_ALWAYS}, and it must\n"
        f"-- never be marked {_NO_TX} either: it is ordinary transactional DDL.\n"
        "CREATE TABLE IF NOT EXISTS zz_mentions_only (a int);\n",
        encoding="utf-8",
    )
    try:
        applied, failed = await run_migrations(migrations_dir=tmp_path, strict=True)
        assert failed == [], f"{failed}"
        assert applied == [name]
        assert not db_config._wants_no_transaction((tmp_path / name).read_text()), (
            "a file whose comment MENTIONS the no-transaction directive was read "
            "as DECLARING it; it would run with no lock_timeout guard and no rollback"
        )
        assert not db_config._runs_always((tmp_path / name).read_text()), (
            "a file whose comment MENTIONS `always` was read as declaring it; it "
            "would re-take its lock on every single boot, forever"
        )
    finally:
        await _forget_files(name)
        async with get_pg_connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS zz_mentions_only")


# ---------------------------------------------------------------------------
# 2. CREATE INDEX CONCURRENTLY through the runner, on a real backend.
# ---------------------------------------------------------------------------

_CIC_TABLE = "zz_cic_target"
_CIC_STATEMENT = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {{index}} ON {_CIC_TABLE} (a);\n"


@pytest_asyncio.fixture
async def cic_table(pg_pool):
    """A real table to build a real concurrent index on."""
    async with get_pg_connection() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_CIC_TABLE}")
        await conn.execute(f"CREATE TABLE {_CIC_TABLE} (a int)")
        await conn.execute(f"INSERT INTO {_CIC_TABLE} (a) SELECT generate_series(1, 50)")
    try:
        yield
    finally:
        async with get_pg_connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {_CIC_TABLE}")


async def test_concurrently_succeeds_when_the_file_declares_no_transaction(
    cic_table, tmp_path: Path
):
    """The marked file builds a VALID index -- not merely "did not raise"."""
    name = "901_cic_marked.sql"
    index = "zz_cic_target_marked_idx"
    (tmp_path / name).write_text(
        f"-- {name}\n-- {_NO_TX}\n" + _CIC_STATEMENT.format(index=index),
        encoding="utf-8",
    )
    try:
        applied, failed = await run_migrations(migrations_dir=tmp_path, strict=True)
        assert failed == [], f"a marked CONCURRENTLY file failed: {failed}"
        assert applied == [name]
        assert await _index_validity(index) is True, (
            "the index is absent or indisvalid=false -- a CONCURRENTLY build that "
            "reports success but leaves nothing usable is the exact "
            "green-signal-for-a-result-that-never-happened shape 028 exists for"
        )
    finally:
        await _forget_files(name)
        async with get_pg_connection() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {index}")


async def test_concurrently_fails_when_the_marker_is_absent(cic_table, tmp_path: Path):
    """The negative half: the SAME statement, unmarked, is refused by Postgres.

    Without this the positive test above proves nothing -- a runner that ignored
    the directive entirely and never opened a transaction would pass it. This is
    what pins the directive to an observable difference in behaviour.
    """
    name = "902_cic_unmarked.sql"
    index = "zz_cic_target_unmarked_idx"
    (tmp_path / name).write_text(
        f"-- {name}\n-- deliberately not marked\n" + _CIC_STATEMENT.format(index=index),
        encoding="utf-8",
    )
    try:
        applied, failed = await run_migrations(migrations_dir=tmp_path, strict=False)
        assert applied == [], (
            "an unmarked CREATE INDEX CONCURRENTLY was reported as applied; the "
            "runner is not wrapping unmarked files in a transaction, so the "
            "directive is inert and proves nothing"
        )
        assert [f[0] for f in failed] == [name]
        assert "transaction" in failed[0][1].lower(), (
            f"expected Postgres to refuse CONCURRENTLY inside a transaction block; "
            f"got {failed[0][1]!r}"
        )
        assert await _index_validity(index) is None, "the unmarked build left an index behind"
    finally:
        await _forget_files(name)
        async with get_pg_connection() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {index}")


async def test_a_multi_statement_no_transaction_file_is_refused_with_the_real_reason(
    cic_table, tmp_path: Path
):
    """The drop-then-create pair cannot run, and the server's error says the wrong thing.

    psycopg sends a multi-statement string through the simple-query protocol,
    which wraps the whole string in an implicit transaction. PostgreSQL then
    answers `DROP INDEX CONCURRENTLY cannot run inside a transaction block` --
    a message pointing at transactions, when the cause is the semicolon.

    A file in that shape can never apply, so left to the server it would be
    logged as a routine migration failure on every boot, forever. That is
    precisely how 026 and 027 went unnoticed for weeks. The guard names the
    real cause before the statement reaches a backend.
    """
    name = "904_cic_two_statements.sql"
    index = "zz_cic_target_two_stmt_idx"
    (tmp_path / name).write_text(
        f"-- {name}\n"
        f"-- {_NO_TX}\n"
        f"DROP INDEX CONCURRENTLY IF EXISTS {index};\n"
        f"CREATE INDEX CONCURRENTLY {index} ON {_CIC_TABLE} (a);\n",
        encoding="utf-8",
    )
    try:
        applied, failed = await run_migrations(migrations_dir=tmp_path, strict=False)
        assert applied == [], "a file that cannot execute was reported as applied"
        assert [f[0] for f in failed] == [name]
        assert "exactly one" in failed[0][1], (
            f"the failure must name the statement count, which is the actual "
            f"cause; got {failed[0][1]!r}"
        )
        assert await _index_validity(index) is None

        # And the reason the guard exists: without it, the server's own answer
        # blames transactions rather than the semicolon.
        with pytest.raises(Exception) as exc:
            async with get_pg_connection() as conn:
                await conn.execute(
                    f"DROP INDEX CONCURRENTLY IF EXISTS {index}; "
                    f"CREATE INDEX CONCURRENTLY {index} ON {_CIC_TABLE} (a);"
                )
        assert "cannot run inside a transaction block" in str(exc.value), (
            f"the premise of this guard is that the raw pair is refused by the "
            f"server; it was not. Got: {exc.value}"
        )
    finally:
        await _forget_files(name)
        async with get_pg_connection() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {index}")


# ---------------------------------------------------------------------------
# 3. Migration 028 reaps an invalid index.
# ---------------------------------------------------------------------------
#
# HOW THE PRECONDITION IS MANUFACTURED (deterministically, and honestly):
# a `CREATE UNIQUE INDEX CONCURRENTLY` over data that violates the uniqueness
# fails in its validation phase and leaves the index behind with
# `indisvalid = false`. That is a GENUINELY failed concurrent build with a real
# invalid index as its residue -- not a catalog UPDATE pretending to be one,
# and not a raced cancellation that may or may not land.

_INVALID_INDEX = "zz_worker_states_invalid_probe_idx"


async def test_028_reaps_an_index_left_invalid_by_a_failed_concurrent_build(pg_pool):
    """An `indisvalid = false` index on worker_states is dropped by the chain.

    Measured against postgres:18 on 2026-08-28: over an index forced invalid,
    `CREATE INDEX CONCURRENTLY IF NOT EXISTS` returns statusmessage
    "CREATE INDEX" and leaves `indisvalid` false. So without 028 the migration
    that owns that name is recorded as applied, permanently, while every query
    plan ignores the object it was written to create.
    """
    tenant = tenant_uuid("migration-markers-invalid-index")
    worker = "markers-invalid-index-probe"

    async with get_pg_connection() as conn:
        await conn.execute(f"DROP INDEX IF EXISTS {_INVALID_INDEX}")
        await conn.execute(
            """
            INSERT INTO worker_states
                (tenant_id, subject_id, visibility, tenant_name, worker_id,
                 worker_name, is_agent, state_type, worker_state, timestamp)
            SELECT $1, $2, 'tenant', 'markers-invalid-index', $3, $4, true,
                   'processing', '{"probe": true}'::jsonb,
                   now() - (g || ' seconds')::interval
            FROM generate_series(1, 2) AS g
            """,
            tenant,
            tenant,
            worker,
            worker,
        )

    try:
        with pytest.raises(Exception) as built:  # driver-specific error class
            async with get_pg_connection() as conn:
                await conn.execute(
                    f"CREATE UNIQUE INDEX CONCURRENTLY {_INVALID_INDEX} "
                    f"ON worker_states (worker_id)"
                )
        assert "duplicate" in str(built.value).lower(), (
            f"the probe build failed for an unexpected reason: {built.value}"
        )
        assert await _index_validity(_INVALID_INDEX) is False, (
            "could not manufacture an invalid index, so this test would pass "
            "vacuously -- check whether the duplicate rows were actually seeded"
        )

        # 028 is `migrate:always`, so it runs whatever the ledger already holds.
        applied, failed = await run_migrations(strict=True)
        assert failed == [], f"{failed}"
        assert _028 in applied, (
            f"028 did not run. It must keep `{_ALWAYS}` or it becomes a one-shot "
            f"that reads like protection while providing none"
        )
        assert await _index_validity(_INVALID_INDEX) is None, (
            "the invalid index survived the chain: `IF NOT EXISTS` will now skip "
            "its name forever while reporting success"
        )
    finally:
        async with get_pg_connection() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {_INVALID_INDEX}")
            await conn.execute(
                "DELETE FROM worker_states WHERE tenant_id = $1 AND worker_id = $2",
                tenant,
                worker,
            )
