"""Migration 033 (prompt_version_transitions): up, backfill, idempotence, down.

Same harness as the 031 test: the real 030 + 031 files build the ledger in a
private schema inside ONE transaction, the real 033 (and its real down file)
run against it, and the whole transaction is rolled back. Nothing touches the
shared ``public`` tables.

Run locally:
    DATABASE_URL=postgresql://… uv run pytest \
        tests/integration/test_migration_033_prompt_transitions.py -m pg -v --no-cov
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from .pg_guard import PG_CONNECT_ERRORS, pg_conninfo, require_database_url, unreachable_pg

pytestmark = pytest.mark.pg

_ROOT = Path(__file__).resolve().parents[2] / "migrations"
_UP_030 = _ROOT / "030_prompt_versions.sql"
_UP_031 = _ROOT / "031_prompt_version_lifecycle.sql"
_UP_033 = _ROOT / "033_prompt_version_transitions.sql"
_DOWN_033 = _ROOT / "down" / "033_prompt_version_transitions.down.sql"
_SCHEMA = "p4_mig033"


def _body(path: Path) -> str:
    sql = path.read_text()
    sql = re.sub(r"^\s*BEGIN\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)
    return re.sub(r"^\s*COMMIT\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)


@pytest.fixture
def conn():
    require_database_url()
    try:
        c = psycopg.connect(pg_conninfo(), row_factory=dict_row)
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        c.execute(f"CREATE SCHEMA {_SCHEMA}")
        c.execute(f"SET LOCAL search_path = {_SCHEMA}")
        c.execute(
            "CREATE TABLE schema_migration_files (filename TEXT PRIMARY KEY, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        c.execute(_body(_UP_030))
        for pid, version, status, activated in (
            ("a", 1, "retired", False),
            ("a", 2, "active", True),
            ("b", 1, "retired", False),
        ):
            c.execute(
                "INSERT INTO prompt_versions (prompt_id, version, sha256, system_prompt_sha256, "
                "system, status, activated_at) VALUES (%s, %s, %s, %s, %s, %s, "
                "CASE WHEN %s THEN now() ELSE NULL END)",
                (pid, version, "a" * 64, "b" * 64, f"{pid} v{version}", status, activated),
            )
        c.execute(_body(_UP_031))
        yield c
    finally:
        c.rollback()
        c.close()


def _ledger(conn) -> list[tuple]:
    rows = conn.execute(
        "SELECT prompt_id, version, from_status, to_status, reason, actor_sub, actor_kind "
        "FROM prompt_version_transitions ORDER BY prompt_id, version, id"
    ).fetchall()
    return [tuple(r.values()) for r in rows]


def _table_exists(conn) -> bool:
    return (
        conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL AS e", (f"{_SCHEMA}.prompt_version_transitions",)
        ).fetchone()["e"]
        is True
    )


BACKFILL = [
    ("a", 1, None, "DRAFT", "backfill_033_creation", None, "migration"),
    ("a", 2, None, "DRAFT", "backfill_033_creation", None, "migration"),
    ("b", 1, None, "DRAFT", "backfill_033_creation", None, "migration"),
]


def test_033_creates_the_ledger_and_backfills_one_creation_row_per_version(conn):
    conn.execute(_body(_UP_033))
    assert _ledger(conn) == BACKFILL
    at = conn.execute(
        "SELECT bool_and(t.at = p.published_at) AS same FROM prompt_version_transitions t "
        "JOIN prompt_versions p USING (prompt_id, version)"
    ).fetchone()["same"]
    assert at is True, "the creation row is dated when the version was published"
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s "
        "AND indexname = 'prompt_version_transitions_version_at_idx'",
        (_SCHEMA,),
    ).fetchone()
    assert idx is not None
    assert "(prompt_id, version, at)" in idx["indexdef"]


def test_033_is_idempotent(conn):
    conn.execute(_body(_UP_033))
    conn.execute(
        "INSERT INTO prompt_version_transitions "
        "(prompt_id, version, from_status, to_status, reason, actor_sub, actor_kind) "
        "VALUES ('a', 2, 'DRAFT', 'PUBLISHED', 'initial_create', 'sa', 'service_account')"
    )
    before = _ledger(conn)
    conn.execute(_body(_UP_033))
    conn.execute(_body(_UP_033))
    assert _ledger(conn) == before, "a re-run must neither re-backfill nor drop rows"


def test_033_fk_and_checks_are_real(conn):
    conn.execute(_body(_UP_033))
    for bad in (
        # no such version
        "('zz', 1, NULL, 'DRAFT', NULL, NULL, 'migration')",
        # unknown status
        "('a', 1, 'DRAFT', 'LIVE', NULL, NULL, 'migration')",
        # unknown actor kind
        "('a', 1, 'DRAFT', 'CANARY', NULL, 'x', 'header')",
        # reason over 200
        f"('a', 1, 'DRAFT', 'CANARY', '{'r' * 201}', NULL, 'migration')",
    ):
        conn.execute("SAVEPOINT s")
        with pytest.raises(psycopg.errors.IntegrityError):
            conn.execute(
                "INSERT INTO prompt_version_transitions "
                "(prompt_id, version, from_status, to_status, reason, actor_sub, actor_kind) "
                f"VALUES {bad}"
            )
        conn.execute("ROLLBACK TO SAVEPOINT s")


_APPEND_ONLY_STMTS = (
    "UPDATE prompt_version_transitions SET reason = 'x'",
    "DELETE FROM prompt_version_transitions",
    "TRUNCATE prompt_version_transitions",
    "TRUNCATE prompt_versions CASCADE",
)


def _assert_append_only(conn) -> None:
    for stmt in _APPEND_ONLY_STMTS:
        conn.execute("SAVEPOINT s")
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
            conn.execute(stmt)
        conn.execute("ROLLBACK TO SAVEPOINT s")


def test_033_is_append_only_with_no_opt_out(conn):
    conn.execute(_body(_UP_033))
    _assert_append_only(conn)
    # The retired GUC opt-out opens NOTHING: any client could set it.
    conn.execute("SET LOCAL prompt_version_transitions.allow_delete = 'on'")
    _assert_append_only(conn)
    conn.execute("SET prompt_version_transitions.allow_delete = 'on'")
    _assert_append_only(conn)
    fn = conn.execute(
        "SELECT pg_get_functiondef(p.oid) AS src FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = %s AND p.proname = 'prompt_version_transitions_append_only'",
        (_SCHEMA,),
    ).fetchone()
    assert "current_setting" not in fn["src"]
    assert [r[0] for r in _ledger(conn)] == ["a", "a", "b"]


def test_033_the_owner_can_disable_the_triggers_for_one_transaction(conn):
    """Test teardown's path: ownership (which can already DROP the table)."""
    conn.execute(_body(_UP_033))
    conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
    conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id = 'b'")
    conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
    assert [r[0] for r in _ledger(conn)] == ["a", "a"]
    _assert_append_only(conn)


_TRUNCATE_TRG_COUNT = (
    "SELECT count(*) AS n FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = %s "
    "AND t.tgname = 'prompt_version_transitions_no_truncate_trg'"
)


def test_033_truncate_trigger_survives_a_rerun_and_down_drops_it(conn):
    conn.execute(_body(_UP_033))
    conn.execute(_body(_UP_033))
    assert conn.execute(_TRUNCATE_TRG_COUNT, (_SCHEMA,)).fetchone()["n"] == 1
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="TRUNCATE"):
        conn.execute("TRUNCATE prompt_version_transitions")
    conn.execute("ROLLBACK TO SAVEPOINT s")
    conn.execute(_body(_DOWN_033))
    assert conn.execute(_TRUNCATE_TRG_COUNT, (_SCHEMA,)).fetchone()["n"] == 0


def test_033_down_drops_the_ledger_and_its_runner_row_then_up_reapplies(conn):
    conn.execute(_body(_UP_033))
    conn.execute(
        "INSERT INTO schema_migration_files (filename) VALUES ('033_prompt_version_transitions.sql')"
    )
    conn.execute(_body(_DOWN_033))
    assert not _table_exists(conn)
    fn = conn.execute(
        "SELECT count(*) AS n FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = %s AND p.proname = 'prompt_version_transitions_append_only'",
        (_SCHEMA,),
    ).fetchone()["n"]
    assert fn == 0
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM schema_migration_files "
            "WHERE filename = '033_prompt_version_transitions.sql'"
        ).fetchone()["n"]
        == 0
    )
    # prompt_versions untouched by the round trip
    assert conn.execute("SELECT count(*) AS n FROM prompt_versions").fetchone()["n"] == 3
    # down is itself re-runnable, and up re-applies cleanly after it
    conn.execute(_body(_DOWN_033))
    conn.execute(_body(_UP_033))
    assert _ledger(conn) == BACKFILL
