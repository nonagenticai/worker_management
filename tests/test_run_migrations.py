"""Unit tests for ``src.db.db_config.run_migrations``.

This is the regression suite for the prod 500 on ``GET /workers``:

    psycopg.errors.UndefinedColumn: column "tenant_id" does not exist

Root cause: the gitops-managed init-container ConfigMap
``worker-management-migrations`` only carried migrations 001-007. Migrations
008, 009, and 010 (the one that adds ``workers.tenant_id``) were committed
to ``migrations/`` in this repo but never reached the live database.

Fix: stop relying on the external ConfigMap. Ship ``migrations/*.sql`` inside
the application image and apply them in lifespan startup. These tests pin
that behavior:

* Migration files are discovered, sorted, and applied once each.
* Empty / missing directory does not raise.
* A pool override is honored (so production code can pass an explicit
  pool without leaning on module global state).
* Failures bubble up so CI / lifespan can fail-fast.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest

from src.db import db_config

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool


class _FakeConn:
    """Minimal async psycopg connection stand-in.

    Captures every executed SQL string so tests can assert ordering and
    content without standing up a real Postgres.
    """

    def __init__(self, *, already_applied: set[str] | None = None) -> None:
        self.executed: list[str] = []
        # If set, the next *migration* execute raises this exception.
        self.raise_on_next: Exception | None = None
        # Matches production: `init_pg_pool` opens this pool with
        # `autocommit=True` (db_config conn_kwargs). Defaulting the fake to
        # False would let a "restore autocommit" bug pass here and only show up
        # against a real pool.
        self.autocommit = True
        # Filenames the ledger reports as already applied, plus whatever the
        # run records as it goes.
        self.ledger: set[str] = set(already_applied or ())
        self.open_transactions = 0
        self.max_open_transactions = 0

    # A statement PostgreSQL will not run inside a transaction block.
    _CONCURRENTLY = re.compile(r"\bCONCURRENTLY\b", re.IGNORECASE)

    @staticmethod
    def _is_guard(sql: str) -> bool:
        """True for bookkeeping SQL that is not a migration file's own body.

        Covers the per-file ``SET LOCAL`` guards and every statement touching
        ``schema_migration_files``. These must not consume ``raise_on_next``:
        if they did, the strict test below would "pass" by exploding on the
        guard instead of on a migration, which is a vacuous green.
        """
        head = sql.lstrip().upper()
        return head.startswith(("SET ", "RESET ")) or "SCHEMA_MIGRATION_FILES" in head

    def _refuse_what_postgres_refuses(self, sql: str) -> None:
        """Raise what a real backend raises, instead of recording the string.

        ⛔ This replaces a guard that could NEVER FIRE. The previous version
        tracked a derived flag::

            self._in_no_transaction_file = value and not self._pool_default

        with ``_pool_default = True``, so the expression was ``value and False``
        -- permanently ``False``. The ``assert not self._in_no_transaction_file``
        in ``transaction()`` was unreachable, and the docstring of
        ``test_a_no_transaction_file_runs_in_autocommit`` claimed it was what
        made a regression "fail loudly". It could not fail at all.

        The flag was derived from autocommit, and autocommit carries no signal
        here: ``init_pg_pool`` already opens the pool ``autocommit=True``
        (``db_config`` conn_kwargs), so ``_apply_outside_transaction``'s
        ``set_autocommit(True)`` is a no-op the fake cannot observe. Making the
        fake default to ``autocommit=False`` would restore the signal and
        re-introduce a different defect this same double already caused.

        So model the BACKEND instead of the runner's internals. PostgreSQL
        rejects CONCURRENTLY inside a transaction block, and psycopg sends a
        multi-statement string over the simple-query protocol, which wraps it in
        an implicit one -- both measured against postgres:18 on 2026-08-28. A
        fake that knows only those two rules catches every way the runner can
        get this wrong, without knowing anything about how the runner decides.
        """
        if not self._CONCURRENTLY.search(sql):
            return
        if self.open_transactions > 0:
            raise psycopg.errors.ActiveSqlTransaction(
                "CREATE INDEX CONCURRENTLY cannot run inside a transaction block"
            )
        body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
        if len([stmt for stmt in body.split(";") if stmt.strip()]) > 1:
            raise psycopg.errors.ActiveSqlTransaction(
                "DROP INDEX CONCURRENTLY cannot run inside a transaction block"
            )

    @asynccontextmanager
    async def transaction(self):
        # psycopg's transaction() is legal on an autocommit connection -- it
        # issues a real BEGIN/COMMIT. What must never happen is the runner
        # opening one around a `migrate:no-transaction` file, and that is
        # enforced in `execute` against the STATEMENT, not here against a flag.
        self.open_transactions += 1
        self.max_open_transactions = max(self.max_open_transactions, self.open_transactions)
        try:
            yield self
        finally:
            self.open_transactions -= 1

    async def set_autocommit(self, value: bool) -> None:
        assert self.open_transactions == 0, "toggled autocommit inside a transaction"
        self.autocommit = value

    async def execute(self, sql: str, params: Any = None, *args: Any) -> Any:
        if self.raise_on_next is not None and not self._is_guard(sql):
            exc, self.raise_on_next = self.raise_on_next, None
            raise exc
        self._refuse_what_postgres_refuses(sql)
        self.executed.append(sql)
        upper = sql.upper()
        if "INSERT INTO SCHEMA_MIGRATION_FILES" in upper and params:
            self.ledger.add(params[0])
        # DICTS, not tuples: the real pool sets `row_factory=dict_row`
        # (`init_pg_pool` conn_kwargs). This fake returned tuples and so
        # happily passed a `row[0]` lookup that raises `KeyError: 0` against
        # every real connection -- 12 integration tests caught what 18 green
        # unit tests did not. Model the row shape, not a convenient one.
        rows = (
            [{"filename": name} for name in sorted(self.ledger)]
            if "SELECT FILENAME FROM SCHEMA_MIGRATION_FILES" in upper
            else []
        )
        return _FakeCursor(rows)

    @property
    def migrations_executed(self) -> list[str]:
        """Executed SQL with the guards and ledger bookkeeping filtered out."""
        return [sql for sql in self.executed if not self._is_guard(sql)]


class _FakeCursor:
    """Result stand-in: ``_applied_filenames`` awaits ``execute(...).fetchall()``."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[dict[str, str]]:
        return self._rows


class _FakePool:
    """Pool stand-in whose ``connection()`` yields a single shared ``_FakeConn``."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self._conn


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Three ordered migration files mimicking the real layout."""
    (tmp_path / "001_first.sql").write_text(
        "ALTER TABLE workers ADD COLUMN IF NOT EXISTS first_col TEXT;\n"
    )
    (tmp_path / "002_second.sql").write_text(
        "ALTER TABLE workers ADD COLUMN IF NOT EXISTS second_col TEXT;\n"
    )
    # The migration that fixes the production bug.
    (tmp_path / "010_per_user_within_tenant_isolation.sql").write_text(
        "ALTER TABLE workers ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255);\n"
        "ALTER TABLE workers ADD COLUMN IF NOT EXISTS subject_id VARCHAR(255);\n"
        "CREATE INDEX IF NOT EXISTS idx_workers_tenant_subject "
        "ON workers(tenant_id, subject_id);\n"
    )
    # A non-SQL file should be ignored.
    (tmp_path / "README.md").write_text("not a migration\n")
    # An empty SQL file should be skipped silently.
    (tmp_path / "003_empty.sql").write_text("   \n")
    return tmp_path


@pytest.mark.asyncio
async def test_run_migrations_applies_files_in_order(migrations_dir: Path) -> None:
    """All non-empty .sql files are executed exactly once, in lexicographic order."""
    conn = _FakeConn()
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", pool),
    )

    # Empty file (003_empty.sql) is skipped; README.md is not a .sql file.
    assert applied == [
        "001_first.sql",
        "002_second.sql",
        "010_per_user_within_tenant_isolation.sql",
    ]
    assert failed == []

    # Three files * (some statements each) — assert tenant_id ALTER actually ran.
    joined = "\n".join(conn.executed)
    assert "tenant_id VARCHAR(255)" in joined, (
        "Migration 010 (the prod fix) must execute and add workers.tenant_id; "
        f"executed SQL was: {joined!r}"
    )
    assert "first_col" in joined
    assert "second_col" in joined


@pytest.mark.asyncio
async def test_run_migrations_returns_empty_list_when_no_files(tmp_path: Path) -> None:
    """A directory with no .sql files is a no-op, not an error."""
    conn = _FakeConn()
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=tmp_path,
        pool=cast("AsyncConnectionPool", pool),
    )

    assert applied == []
    assert failed == []
    assert conn.executed == []


@pytest.mark.asyncio
async def test_run_migrations_returns_empty_list_when_dir_missing(
    tmp_path: Path,
) -> None:
    """A non-existent migrations dir does not crash startup."""
    conn = _FakeConn()
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=tmp_path / "does-not-exist",
        pool=cast("AsyncConnectionPool", pool),
    )

    assert applied == []
    assert failed == []
    assert conn.executed == []


@pytest.mark.asyncio
async def test_every_migration_is_guarded_inside_its_own_transaction(
    migrations_dir: Path,
) -> None:
    """Each file gets ``SET LOCAL`` guards in the SAME transaction as its DDL.

    Regression guard for the lock convoy of 2026-08-27. The guards used to be
    session-level ``SET``s issued once before the loop. That is unsafe here:
    ``DATABASE_URL`` points at PgBouncer in ``pool_mode = transaction``, which
    pins a server connection only for the length of a transaction, and 15 of
    the migration files carried their own ``COMMIT;`` that ended it mid-chain.
    Measured consequence: migration 011 held ACCESS EXCLUSIVE on ``workers``
    and blocked on ``worker_states`` for 67 minutes with no timeout firing,
    stalling 19 sessions behind it.

    A ``SET LOCAL`` inside the same transaction as the DDL cannot come adrift
    that way -- a transaction never migrates between backends. So the assertion
    is specifically that the guards appear per-file and *inside* a transaction,
    not once at the top.
    """
    conn = _FakeConn()
    pool = _FakePool(conn)

    await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", pool),
    )

    lock_guards = [sql for sql in conn.executed if sql.startswith("SET LOCAL lock_timeout")]
    migrations = conn.migrations_executed
    assert len(lock_guards) == len(migrations), (
        f"expected one SET LOCAL lock_timeout per migration; "
        f"got {len(lock_guards)} guards for {len(migrations)} migrations"
    )
    assert all(f"'{db_config._MIGRATION_LOCK_TIMEOUT}'" in sql for sql in lock_guards), (
        f"guards must carry the configured timeout: {lock_guards}"
    )

    # Every guard must immediately precede the migration it protects, so no
    # file can run outside the window its own guard opens.
    for migration in migrations:
        idx = conn.executed.index(migration)
        window = conn.executed[max(0, idx - 2) : idx]
        assert any(sql.startswith("SET LOCAL lock_timeout") for sql in window), (
            f"migration ran without a preceding SET LOCAL guard: {migration!r}"
        )

    # The old session-level form must not come back: it is what broke.
    assert not any(
        sql.startswith(("SET lock_timeout", "RESET lock_timeout")) for sql in conn.executed
    ), "session-level lock_timeout is unsafe through a transaction-mode pooler"


@pytest.mark.asyncio
async def test_a_files_own_begin_commit_is_stripped(tmp_path: Path) -> None:
    """The runner owns the transaction boundaries; the file's own are removed.

    A file's ``COMMIT;`` executing inside the runner's transaction is what
    returned the connection to PgBouncer mid-chain. 668 "there is already a
    transaction in progress" warnings in the worker4 log are the fingerprint.
    """
    (tmp_path / "001_wrapped.sql").write_text(
        "-- a comment\nBEGIN;\n\nALTER TABLE workers ADD COLUMN IF NOT EXISTS c TEXT;\n\nCOMMIT;\n"
    )
    conn = _FakeConn()
    pool = _FakePool(conn)

    await db_config.run_migrations(
        migrations_dir=tmp_path,
        pool=cast("AsyncConnectionPool", pool),
    )

    body = conn.migrations_executed[0]
    assert "BEGIN;" not in body.upper(), f"outer BEGIN survived: {body!r}"
    assert "COMMIT;" not in body.upper(), f"outer COMMIT survived: {body!r}"
    assert "ADD COLUMN IF NOT EXISTS c TEXT" in body, f"body was mangled: {body!r}"


def test_stripping_leaves_a_do_block_begin_alone() -> None:
    """``DO $$ BEGIN ... END $$`` must survive: it is not a transaction boundary.

    The strip is anchored to the start and end of the file precisely so a
    PL/pgSQL block cannot be mistaken for one. Migrations 011, 013 and 020 all
    carry both, so getting this wrong corrupts them silently.
    """
    sql = "BEGIN;\nDO $$\nBEGIN\n  RAISE NOTICE 'hi';\nEND $$;\nCOMMIT;\n"
    out = db_config._strip_outer_transaction(sql)
    assert out.count("BEGIN") == 1, f"the DO block's BEGIN was stripped: {out!r}"
    assert "END $$;" in out
    assert not out.upper().startswith("BEGIN;")
    assert not out.upper().rstrip().endswith("COMMIT;")


def test_stripping_is_a_no_op_for_an_unwrapped_file() -> None:
    """12 of the 27 files have no outer transaction; they must pass through."""
    sql = "ALTER TABLE workers ADD COLUMN IF NOT EXISTS c TEXT;"
    assert db_config._strip_outer_transaction(sql) == sql


@pytest.mark.asyncio
async def test_an_applied_migration_is_skipped_on_the_next_run(
    migrations_dir: Path,
) -> None:
    """The ledger stops an applied file re-taking ACCESS EXCLUSIVE every boot.

    Measured before this: 8 of 26 files failed on EVERY pod start with
    ``canceling statement due to lock timeout``, 7 of them already-applied
    no-ops, burning ~160s of boot on locks that were never going to be granted.
    """
    conn = _FakeConn()
    pool = _FakePool(conn)

    first, _ = await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", pool),
    )
    assert first, "first run should apply something"

    second_conn = _FakeConn(already_applied=conn.ledger)
    second, failed = await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", _FakePool(second_conn)),
    )

    assert second == [], f"already-applied files must not re-run; got {second}"
    assert failed == []
    assert second_conn.migrations_executed == [], (
        f"no migration DDL should be issued on a clean second run; "
        f"got {second_conn.migrations_executed}"
    )


@pytest.mark.asyncio
async def test_a_failed_migration_is_not_recorded_as_applied(
    migrations_dir: Path,
) -> None:
    """A file that failed must be retried next boot, not skipped forever.

    The ledger INSERT rides in the same transaction as the DDL precisely so a
    rollback takes the bookkeeping with it. Recording a failure as applied is
    the one way this change could hide a genuinely unapplied migration.
    """
    conn = _FakeConn()
    conn.raise_on_next = psycopg.errors.LockNotAvailable("lock timeout")
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", pool),
    )

    assert len(failed) == 1, f"expected exactly one failure; got {failed}"
    failed_name = failed[0][0]
    assert failed_name not in applied
    assert failed_name not in conn.ledger, (
        f"{failed_name} failed but was recorded as applied — it would never retry"
    )


@pytest.mark.asyncio
async def test_a_no_transaction_file_runs_in_autocommit(tmp_path: Path) -> None:
    """``migrate:no-transaction`` is what makes CREATE INDEX CONCURRENTLY possible.

    PostgreSQL refuses CONCURRENTLY inside a transaction block, which is why
    migrations 026 and 027 were written as plain ``CREATE INDEX`` and then
    failed on every boot for weeks against a table too busy to grant SHARE.
    ``_FakeConn`` raises `ActiveSqlTransaction` when a CONCURRENTLY statement
    reaches it inside an open transaction -- the same error a real backend
    gives -- so a regression that wrapped this path fails loudly here rather
    than at runtime on prod. (This sentence used to credit an assert on
    ``_in_no_transaction_file``; that flag was permanently ``False`` and the
    assert could never fire. See ``_refuse_what_postgres_refuses``.)

    ⚠️ ONE statement. This fixture was originally the drop-then-create pair, and
    that pair cannot run: psycopg sends a multi-statement string through the
    simple-query protocol, which wraps it in an implicit transaction, and the
    server answers `DROP INDEX CONCURRENTLY cannot run inside a transaction
    block`. The fake could not see it -- it never parses the SQL it is handed --
    so the pattern type-checked, unit-tested green, and would have failed on the
    first real boot. Measured against postgres:18 on 2026-08-28, and now
    enforced by `_no_transaction_violation` and proved on a real backend in
    tests/integration/test_migration_markers.py.
    """
    (tmp_path / "029_concurrent.sql").write_text(
        "-- migrate:no-transaction\nCREATE INDEX CONCURRENTLY idx_demo ON workers (tenant_id);\n"
    )
    conn = _FakeConn()
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=tmp_path,
        pool=cast("AsyncConnectionPool", pool),
    )

    assert applied == ["029_concurrent.sql"], f"failed={failed}"
    assert "CONCURRENTLY" in conn.migrations_executed[0]
    # Restored to the POOL DEFAULT (True), not forced to False: `init_pg_pool`
    # opens this pool with autocommit=True, so forcing False would hand back a
    # silently different connection.
    assert conn.autocommit is True, "autocommit was not restored to the pool default"
    # No SET LOCAL: there is no transaction to scope one to, and CONCURRENTLY
    # takes only SHARE UPDATE EXCLUSIVE so it does not block writers anyway.
    assert not any(sql.startswith("SET LOCAL") for sql in conn.executed)


@pytest.mark.asyncio
async def test_autocommit_is_restored_when_a_concurrent_file_fails(
    tmp_path: Path,
) -> None:
    """A failure inside the autocommit path must not leak autocommit to the pool."""
    (tmp_path / "029_concurrent.sql").write_text(
        "-- migrate:no-transaction\nCREATE INDEX CONCURRENTLY idx_demo ON workers (x);\n"
    )
    conn = _FakeConn()
    conn.raise_on_next = psycopg.errors.LockNotAvailable("boom")
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=tmp_path,
        pool=cast("AsyncConnectionPool", pool),
    )

    assert applied == []
    assert len(failed) == 1
    assert conn.autocommit is True, (
        "the failure path left autocommit off; the pool hands this connection "
        "to unrelated callers that expect the autocommit=True it was opened with"
    )


@pytest.mark.asyncio
async def test_run_migrations_strict_propagates_failure(migrations_dir: Path) -> None:
    """``strict=True`` aborts on first failure and re-raises (fail-fast path)."""
    conn = _FakeConn()
    conn.raise_on_next = psycopg.errors.SyntaxError("boom")
    pool = _FakePool(conn)

    with pytest.raises(psycopg.errors.SyntaxError):
        await db_config.run_migrations(
            migrations_dir=migrations_dir,
            pool=cast("AsyncConnectionPool", pool),
            strict=True,
        )


@pytest.mark.asyncio
async def test_run_migrations_lenient_collects_failures_and_continues(
    migrations_dir: Path,
) -> None:
    """Non-strict mode (production default) logs failures and keeps going.

    This is the legacy init-container behavior. Without it, a single broken
    migration would block a later one (e.g. migration 010 — the production
    fix for ``workers.tenant_id``). Historically migration 001 carried SQLite
    syntax (``CREATE TRIGGER IF NOT EXISTS``) and forced this mode; 001 has
    since been removed, but lenient mode stays as a forward safety belt.
    """
    conn = _FakeConn()
    # First migration explodes; subsequent ones must still run.
    conn.raise_on_next = psycopg.errors.SyntaxError("boom on first")
    pool = _FakePool(conn)

    applied, failed = await db_config.run_migrations(
        migrations_dir=migrations_dir,
        pool=cast("AsyncConnectionPool", pool),
        strict=False,
    )

    assert applied == [
        "002_second.sql",
        "010_per_user_within_tenant_isolation.sql",
    ]
    assert len(failed) == 1
    assert failed[0][0] == "001_first.sql"
    assert "boom on first" in failed[0][1]

    # Crucially: the prod-fix migration still ran.
    joined = "\n".join(conn.executed)
    assert "tenant_id VARCHAR(255)" in joined


@pytest.mark.asyncio
async def test_run_migrations_requires_initialized_pool(
    migrations_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling without a pool and without ``init_pg_pool()`` raises clearly."""
    monkeypatch.setattr(db_config, "_pg_pool", None, raising=False)

    with pytest.raises(RuntimeError, match="pool not initialized"):
        await db_config.run_migrations(migrations_dir=migrations_dir)


def test_resolve_migrations_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``WORKER_MGMT_MIGRATIONS_DIR`` overrides the repo-relative default."""
    monkeypatch.setenv("WORKER_MGMT_MIGRATIONS_DIR", str(tmp_path))
    assert db_config._resolve_migrations_dir() == tmp_path


def test_resolve_migrations_dir_default_points_at_repo_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default resolution lands on the repo's ``migrations/`` directory.

    This is what ships in the production image (Dockerfile copies
    ``migrations/`` to ``/app/migrations``).
    """
    monkeypatch.delenv("WORKER_MGMT_MIGRATIONS_DIR", raising=False)
    resolved = db_config._resolve_migrations_dir()
    assert resolved.name == "migrations"
    # The committed migration that fixes prod must be present in the
    # resolved directory — guards against ``COPY migrations/`` being
    # dropped from the Dockerfile or files being deleted from the repo.
    expected = resolved / "010_per_user_within_tenant_isolation.sql"
    assert expected.exists(), (
        f"Expected canonical migration file at {expected}; Dockerfile or repo layout has drifted."
    )


def test_repo_migration_010_adds_workers_tenant_id() -> None:
    """The migration file shipped with the repo really does add tenant_id.

    Belt-and-braces against someone editing migration 010 in a way that
    silently drops the ``workers.tenant_id`` ALTER while leaving the file
    in place.
    """
    repo_root = Path(__file__).resolve().parent.parent
    sql = (repo_root / "migrations" / "010_per_user_within_tenant_isolation.sql").read_text()
    assert "ALTER TABLE workers" in sql
    assert "tenant_id" in sql
    assert "ADD COLUMN IF NOT EXISTS tenant_id" in sql, (
        "Migration 010 must add workers.tenant_id idempotently — this is the "
        "exact prod bug the fix addresses."
    )


@pytest.mark.asyncio
async def test_lifespan_uses_run_migrations_after_pool_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan startup must call ``run_migrations`` after ``init_pg_pool``.

    Pins the wiring so a future refactor can't silently regress to "init the
    pool but skip migrations" — which is the whole class of bug we just fixed.
    """
    # tests/conftest.py sets WORKER_MGMT_SKIP_MIGRATIONS=1 so the rest of the
    # mocking-heavy unit suite doesn't have to provide a fake pool. This test
    # is the one place where we explicitly want the migration call to fire.
    monkeypatch.delenv("WORKER_MGMT_SKIP_MIGRATIONS", raising=False)

    from src.api import worker_management_api

    init_mock = AsyncMock()
    migrate_mock = AsyncMock(return_value=(["010_per_user_within_tenant_isolation.sql"], []))
    close_mock = AsyncMock()

    call_order: list[str] = []
    init_mock.side_effect = lambda *a, **kw: call_order.append("init")  # noqa: ARG005
    migrate_mock.side_effect = lambda *a, **kw: (  # noqa: ARG005
        call_order.append("migrate") or (["010_per_user_within_tenant_isolation.sql"], [])
    )

    app = MagicMock()

    # monkeypatch.setattr handles cleanup automatically and is friendlier to
    # type checkers than rebinding module attributes by hand.
    monkeypatch.setattr(worker_management_api, "init_pg_pool", init_mock)
    monkeypatch.setattr(worker_management_api, "run_migrations", migrate_mock)
    monkeypatch.setattr(worker_management_api, "close_pg_pool", close_mock)

    async with worker_management_api.lifespan(app):
        pass

    assert call_order == ["init", "migrate"], (
        f"Lifespan must initialize the pool BEFORE running migrations; observed order: {call_order}"
    )
    init_mock.assert_awaited_once()
    migrate_mock.assert_awaited_once()
    close_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# Directive detection: a file that MENTIONS a directive must not DECLARE it.
# ---------------------------------------------------------------------------
#
# `_wants_no_transaction` was a substring test over the first 40 lines:
#
#     return _MIGRATION_NO_TRANSACTION_DIRECTIVE in head
#
# Not currently triggered -- 0 of 28 shipped files contain the string in their
# header under either form -- so this is a latent defect, not a live one. It is
# worth closing anyway, because migration headers in this repository routinely
# discuss the runner's own mechanics by name: 027 quotes
# `CREATE INDEX CONCURRENTLY IF NOT EXISTS` and explains the INVALID trap, and
# 028 has to name its own directive to justify carrying it. A file that says it
# does NOT use the directive is armed by a substring test, and then runs with no
# lock_timeout guard and no rollback.


_MENTIONS_BUT_DOES_NOT_DECLARE = (
    "-- 999_ordinary.sql\n"
    "-- This file does NOT use migrate:no-transaction: it is ordinary\n"
    "-- transactional DDL and wants the runner's lock_timeout guard.\n"
    "ALTER TABLE workers ADD COLUMN IF NOT EXISTS zz TEXT;\n"
)

_DECLARES = (
    "-- 999_concurrent.sql\n-- migrate:no-transaction\nCREATE INDEX CONCURRENTLY i ON t (a);\n"
)


def test_a_header_that_mentions_the_directive_does_not_arm_it() -> None:
    """The substring form armed this file; the anchored form must not.

    This is the assertion that fails against `directive in head` and passes
    against `re.fullmatch` on a whole comment line.
    """
    assert not db_config._wants_no_transaction(_MENTIONS_BUT_DOES_NOT_DECLARE), (
        "a file whose header says it does NOT use the directive was read as "
        "declaring it; it would run in autocommit with no lock_timeout guard "
        "and no rollback"
    )


def test_a_header_that_declares_the_directive_still_arms_it() -> None:
    """The other half: anchoring must not break real declarations.

    Without this, `return False` would satisfy the test above.
    """
    assert db_config._wants_no_transaction(_DECLARES)


def test_the_directive_is_only_read_from_the_header() -> None:
    """A declaration below the header window is not a declaration.

    Preserved from the substring form deliberately -- the directive changes how
    the whole file is executed, so it has to be visible where a reader looks.
    """
    buried = "-- x\n" * 45 + "-- migrate:no-transaction\nSELECT 1;\n"
    assert not db_config._wants_no_transaction(buried)


def test_an_indented_or_spaced_declaration_is_still_a_declaration() -> None:
    """Anchoring is on the stripped line, not on column zero."""
    assert db_config._wants_no_transaction("--   migrate:no-transaction  \nSELECT 1;\n")
    assert db_config._runs_always("  --  migrate:always\nSELECT 1;\n")


def test_a_multi_statement_no_transaction_file_is_rejected_before_execution() -> None:
    """psycopg wraps a multi-statement string in an implicit transaction.

    So the drop-then-create pair cannot run at all, and PostgreSQL's own answer
    (`DROP INDEX CONCURRENTLY cannot run inside a transaction block`) blames
    transactions rather than the semicolon. Measured against postgres:18 on
    2026-08-28; proved end-to-end in tests/integration/test_migration_markers.py.
    """
    pair = (
        "-- migrate:no-transaction\n"
        "DROP INDEX CONCURRENTLY IF EXISTS idx_foo;\n"
        "CREATE INDEX CONCURRENTLY idx_foo ON foo (bar);\n"
    )
    problem = db_config._no_transaction_violation(pair)
    assert problem is not None and "exactly one" in problem, problem

    single = "-- migrate:no-transaction\nCREATE INDEX CONCURRENTLY idx_foo ON foo (bar);\n"
    assert db_config._no_transaction_violation(single) is None, (
        "the one-statement form is the supported shape and must be accepted"
    )


def test_a_dollar_quoted_no_transaction_file_is_refused_rather_than_split() -> None:
    """A `$$ ... $$` body may contain semicolons, so statement counting is unsafe."""
    body = "-- migrate:no-transaction\nDO $x$ BEGIN PERFORM 1; PERFORM 2; END $x$;\n"
    problem = db_config._no_transaction_violation(body)
    assert problem is not None and "dollar-quoted" in problem, problem
