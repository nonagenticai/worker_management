"""Migration 034 (worker_prompt_binding): up, idempotence, constraints, down.

Same harness as the 033 test: the real 034 (and its real down file) run in a
private schema inside ONE transaction, which is rolled back. Nothing touches the
shared ``public`` tables. 034 depends on nothing earlier (no FK by design — see
its header), so no prior migration is replayed.

Run locally:
    DATABASE_URL=postgresql://… uv run pytest \
        tests/integration/test_migration_034_worker_prompt_binding.py -m pg -v --no-cov
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
_UP_034 = _ROOT / "034_worker_prompt_binding.sql"
_DOWN_034 = _ROOT / "down" / "034_worker_prompt_binding.down.sql"
_SCHEMA = "u6_mig034"
_TENANT = "0195f48b-d000-7000-8000-000000000003"


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
        yield c
    finally:
        c.rollback()
        c.close()


def _table_exists(conn) -> bool:
    return (
        conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL AS e", (f"{_SCHEMA}.worker_prompt_binding",)
        ).fetchone()["e"]
        is True
    )


def _insert(conn, worker: str = "worker-a", prompt_id: str = "worker-a") -> None:
    conn.execute(
        "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
        "VALUES (%s, %s, %s, 'adopt_live_worker_prompts')",
        (_TENANT, worker, prompt_id),
    )


def test_034_creates_the_table_with_its_pk_default_and_index(conn):
    conn.execute(_body(_UP_034))
    assert _table_exists(conn)
    _insert(conn)
    row = conn.execute("SELECT * FROM worker_prompt_binding").fetchone()
    assert row["tenant_id"] == _TENANT
    assert row["prompt_id"] == "worker-a"
    assert row["bound_by"] == "adopt_live_worker_prompts"
    assert row["bound_at"] is not None, "bound_at defaults to now()"
    pk = conn.execute(
        "SELECT pg_get_constraintdef(c.oid) AS d FROM pg_constraint c "
        "WHERE c.conrelid = to_regclass(%s) AND c.contype = 'p'",
        (f"{_SCHEMA}.worker_prompt_binding",),
    ).fetchone()
    assert pk["d"] == "PRIMARY KEY (tenant_id, worker_name)"
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s "
        "AND indexname = 'worker_prompt_binding_prompt_id_idx'",
        (_SCHEMA,),
    ).fetchone()
    assert idx is not None
    assert "(prompt_id)" in idx["indexdef"]


def test_034_one_binding_per_tenant_worker(conn):
    conn.execute(_body(_UP_034))
    _insert(conn)
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert(conn, prompt_id="some-other-prompt")
    conn.execute("ROLLBACK TO SAVEPOINT s")


def test_034_not_null_columns_are_real(conn):
    conn.execute(_body(_UP_034))
    for cols, vals in (
        ("worker_name, prompt_id, bound_by", "'w', 'p', 'x'"),
        ("tenant_id, prompt_id, bound_by", f"'{_TENANT}', 'p', 'x'"),
        ("tenant_id, worker_name, bound_by", f"'{_TENANT}', 'w', 'x'"),
        ("tenant_id, worker_name, prompt_id", f"'{_TENANT}', 'w', 'p'"),
    ):
        conn.execute("SAVEPOINT s")
        with pytest.raises(psycopg.errors.NotNullViolation):
            conn.execute(f"INSERT INTO worker_prompt_binding ({cols}) VALUES ({vals})")
        conn.execute("ROLLBACK TO SAVEPOINT s")


def test_034_has_no_fk_to_prompt_versions(conn):
    """Integrity is checked in code (prompt_source.py): a binding may name a
    prompt that has no row at all, and that must not be refused here."""
    conn.execute(_body(_UP_034))
    _insert(conn, prompt_id="not-in-any-ledger")
    fks = conn.execute(
        "SELECT count(*) AS n FROM pg_constraint WHERE conrelid = to_regclass(%s) "
        "AND contype = 'f'",
        (f"{_SCHEMA}.worker_prompt_binding",),
    ).fetchone()["n"]
    assert fks == 0


def test_034_is_idempotent(conn):
    conn.execute(_body(_UP_034))
    _insert(conn)
    conn.execute(_body(_UP_034))
    conn.execute(_body(_UP_034))
    assert conn.execute("SELECT count(*) AS n FROM worker_prompt_binding").fetchone()["n"] == 1


def test_034_down_drops_the_table_and_its_runner_row_then_up_reapplies(conn):
    conn.execute(_body(_UP_034))
    _insert(conn)
    conn.execute(
        "INSERT INTO schema_migration_files (filename) VALUES ('034_worker_prompt_binding.sql')"
    )
    conn.execute(_body(_DOWN_034))
    assert not _table_exists(conn)
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM schema_migration_files "
            "WHERE filename = '034_worker_prompt_binding.sql'"
        ).fetchone()["n"]
        == 0
    )
    # down is itself re-runnable, and up re-applies cleanly after it
    conn.execute(_body(_DOWN_034))
    conn.execute(_body(_UP_034))
    assert _table_exists(conn)
    assert conn.execute("SELECT count(*) AS n FROM worker_prompt_binding").fetchone()["n"] == 0
