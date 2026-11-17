"""Migration 035 (prompt_owner): backfill, the RAISE guard, the DB guarantee, up/down/up.

Same harness as the 033/034 tests: the real 030, 031, 034 and 035 files (and
035's real down file) run in a private schema inside ONE transaction, which is
rolled back. Nothing touches the shared ``public`` tables.

THE guarantee under test: a ``worker_prompt_binding`` whose tenant is not the
prompt's owner cannot be inserted (``ForeignKeyViolation``). Drop the binding
FK from 035 and ``test_035_refuses_a_cross_tenant_binding`` goes red.

Run locally:
    DATABASE_URL=postgresql://… uv run pytest \
        tests/integration/test_migration_035_prompt_owner.py -m pg -v --no-cov
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
_UP_034 = _ROOT / "034_worker_prompt_binding.sql"
_UP_035 = _ROOT / "035_prompt_owner.sql"
_DOWN_035 = _ROOT / "down" / "035_prompt_owner.down.sql"
_SCHEMA = "u6_mig035"
SYSTEM = "aaaaaaaa-0000-4000-8000-000000000001"
TENANT_B = "aaaaaaaa-0000-4000-8000-000000000003"


def _body(path: Path) -> str:
    sql = path.read_text()
    sql = re.sub(r"^\s*BEGIN\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)
    return re.sub(r"^\s*COMMIT\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)


def _version(conn, pid: str, version: int, owner: str | None = None) -> None:
    cols = "prompt_id, version, sha256, system_prompt_sha256, system"
    vals = "%s, %s, %s, %s, %s"
    args: list[object] = [pid, version, "a" * 64, "b" * 64, f"{pid} v{version}"]
    if owner is not None:
        cols += ", owner_tenant_id"
        vals += ", %s"
        args.append(owner)
    conn.execute(f"INSERT INTO prompt_versions ({cols}) VALUES ({vals})", args)


@pytest.fixture
def conn():
    """A private schema holding 030+031+034 and three pre-035 versions (a@1, a@2, b@1)."""
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
        c.execute(_body(_UP_031))
        c.execute(_body(_UP_034))
        for pid, version in (("a", 1), ("a", 2), ("b", 1)):
            _version(c, pid, version)
        yield c
    finally:
        c.rollback()
        c.close()


def _bind(conn, tenant: str, prompt_id: str, worker: str = "worker-x") -> None:
    conn.execute(
        "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
        "VALUES (%s, %s, %s, 'test')",
        (tenant, worker, prompt_id),
    )


def _refused(conn, exc_type: type[Exception], fn, *args) -> None:
    conn.execute("SAVEPOINT s")
    with pytest.raises(exc_type):
        fn(conn, *args)
    conn.execute("ROLLBACK TO SAVEPOINT s")


def _exists(conn, name: str) -> bool:
    return bool(
        conn.execute("SELECT to_regclass(%s) IS NOT NULL AS e", (f"{_SCHEMA}.{name}",)).fetchone()[
            "e"
        ]
    )


def test_035_backfills_every_existing_prompt_to_the_system_tenant(conn):
    conn.execute(_body(_UP_035))
    owners = conn.execute(
        "SELECT prompt_id, owner_tenant_id FROM prompt_owner ORDER BY prompt_id"
    ).fetchall()
    assert [(r["prompt_id"], r["owner_tenant_id"]) for r in owners] == [
        ("a", SYSTEM),
        ("b", SYSTEM),
    ]
    rows = conn.execute("SELECT DISTINCT owner_tenant_id FROM prompt_versions").fetchall()
    assert [r["owner_tenant_id"] for r in rows] == [SYSTEM]
    col = conn.execute(
        "SELECT is_nullable FROM information_schema.columns WHERE table_schema = %s "
        "AND table_name = 'prompt_versions' AND column_name = 'owner_tenant_id'",
        (_SCHEMA,),
    ).fetchone()
    assert col["is_nullable"] == "NO"


def test_035_raises_when_a_version_is_left_without_an_owner(conn):
    """Simulate a backfill that misses a prompt: a pre-created prompt_owner (the
    same DDL 035 uses, so its IF NOT EXISTS keeps it) whose trigger silently
    drops 'b'. The DO block must refuse the whole migration."""
    conn.execute(
        "CREATE TABLE prompt_owner (prompt_id TEXT PRIMARY KEY, owner_tenant_id TEXT NOT NULL, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
        "CONSTRAINT prompt_owner_owner_prompt_key UNIQUE (owner_tenant_id, prompt_id))"
    )
    conn.execute(
        "CREATE FUNCTION skip_b() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN IF NEW.prompt_id = 'b' THEN RETURN NULL; END IF; RETURN NEW; END $$"
    )
    conn.execute(
        "CREATE TRIGGER skip_b BEFORE INSERT ON prompt_owner FOR EACH ROW EXECUTE FUNCTION skip_b()"
    )
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.RaiseException) as exc:
        conn.execute(_body(_UP_035))
    conn.execute("ROLLBACK TO SAVEPOINT s")
    assert "035_prompt_owner refused" in str(exc.value)
    assert "prompt_id b has a version without an owner" in str(exc.value)


def test_035_refuses_a_cross_tenant_binding(conn):
    """THE guarantee: tenant B cannot bind a worker to a prompt owned by the system tenant."""
    conn.execute(_body(_UP_035))
    _refused(conn, psycopg.errors.ForeignKeyViolation, _bind, TENANT_B, "a")
    assert conn.execute("SELECT count(*) AS n FROM worker_prompt_binding").fetchone()["n"] == 0


def test_035_accepts_a_same_tenant_binding(conn):
    conn.execute(_body(_UP_035))
    _bind(conn, SYSTEM, "a")
    conn.execute(
        "INSERT INTO prompt_owner (prompt_id, owner_tenant_id) VALUES ('c', %s)", (TENANT_B,)
    )
    _bind(conn, TENANT_B, "c")
    assert conn.execute("SELECT count(*) AS n FROM worker_prompt_binding").fetchone()["n"] == 2


def test_035_every_version_carries_its_prompts_owner(conn):
    conn.execute(_body(_UP_035))
    # A version filed under another tenant than the prompt's owner is refused.
    _refused(conn, psycopg.errors.ForeignKeyViolation, _version, "a", 3, TENANT_B)
    # A brand-new prompt with no owner row cannot get a version at all.
    _refused(conn, psycopg.errors.NotNullViolation, _version, "new", 1)
    # Rollout bridge: a pre-035 insert (no owner column) of an EXISTING prompt
    # is filled with that prompt's owner, never another.
    _version(conn, "a", 3)
    row = conn.execute(
        "SELECT owner_tenant_id FROM prompt_versions WHERE prompt_id = 'a' AND version = 3"
    ).fetchone()
    assert row["owner_tenant_id"] == SYSTEM


def test_035_is_idempotent(conn):
    conn.execute(_body(_UP_035))
    _bind(conn, SYSTEM, "a")
    conn.execute(_body(_UP_035))
    conn.execute(_body(_UP_035))
    assert conn.execute("SELECT count(*) AS n FROM prompt_owner").fetchone()["n"] == 2
    fks = conn.execute(
        "SELECT conname FROM pg_constraint WHERE contype = 'f' AND connamespace = %s::regnamespace "
        "ORDER BY conname",
        (_SCHEMA,),
    ).fetchall()
    assert [r["conname"] for r in fks] == [
        "prompt_versions_owner_fk",
        "worker_prompt_binding_owner_fk",
    ]


def test_035_up_down_up(conn):
    conn.execute(_body(_UP_035))
    conn.execute("INSERT INTO schema_migration_files (filename) VALUES ('035_prompt_owner.sql')")
    conn.execute(_body(_DOWN_035))
    assert not _exists(conn, "prompt_owner")
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM information_schema.columns WHERE table_schema = %s "
            "AND table_name = 'prompt_versions' AND column_name = 'owner_tenant_id'",
            (_SCHEMA,),
        ).fetchone()["n"]
        == 0
    )
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM schema_migration_files "
            "WHERE filename = '035_prompt_owner.sql'"
        ).fetchone()["n"]
        == 0
    )
    # Versions survive the down; a cross-tenant binding is now possible again.
    assert conn.execute("SELECT count(*) AS n FROM prompt_versions").fetchone()["n"] == 3
    conn.execute(_body(_DOWN_035))  # down is re-runnable
    conn.execute(_body(_UP_035))
    assert _exists(conn, "prompt_owner")
    _refused(conn, psycopg.errors.ForeignKeyViolation, _bind, TENANT_B, "a")
