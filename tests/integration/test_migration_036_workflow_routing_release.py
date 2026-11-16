"""Migration 036 (workflow_routing_release): up, fail-closed refusal, CHECKs, down.

Same harness as the 034 test: the real 014 (the shape 036 alters), the real 036
and its real down file run in a private schema inside ONE transaction, which is
rolled back. Nothing touches the shared ``public`` tables.

Run locally:
    DATABASE_URL=postgresql://… uv run pytest \
        tests/integration/test_migration_036_workflow_routing_release.py -m pg -v --no-cov
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from .pg_guard import PG_CONNECT_ERRORS, pg_conninfo, require_database_url, unreachable_pg

pytestmark = pytest.mark.pg

_ROOT = Path(__file__).resolve().parents[2] / "migrations"
_UP_014 = _ROOT / "014_workflow_routing_pointer.sql"
_UP_036 = _ROOT / "036_workflow_routing_release.sql"
_DOWN_036 = _ROOT / "down" / "036_workflow_routing_release.down.sql"
_SCHEMA = "g15_mig036"
_TABLE = f"{_SCHEMA}.workflow_routing_pointer"

ASSET = "activity:AA Activity - Reasoning"
WF_A = "AbCdEfGh12345678"
WF_B = "Zz9Yy8Xx7Ww6Vv5U"
WF_C = "Qq1Rr2Ss3Tt4Uu5V"


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
        # 014 still records itself in the legacy ledger.
        c.execute("CREATE TABLE schema_migrations (version TEXT PRIMARY KEY, description TEXT)")
        c.execute(_body(_UP_014))
        yield c
    finally:
        c.rollback()
        c.close()


def _insert(conn, **cols: Any) -> None:
    row = {"asset_key": ASSET, "active_workflow_id": WF_A, **cols}
    names = ", ".join(row)
    marks = ", ".join(["%s"] * len(row))
    conn.execute(
        f"INSERT INTO workflow_routing_pointer ({names}) VALUES ({marks})",
        tuple(row.values()),
    )


def _rejects(conn, error: type[Exception], **cols: Any) -> None:
    conn.execute("SAVEPOINT s")
    with pytest.raises(error):
        _insert(conn, **cols)
    conn.execute("ROLLBACK TO SAVEPOINT s")


def _column_type(conn, column: str) -> str:
    return conn.execute(
        "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
        "WHERE attrelid = to_regclass(%s) AND attname = %s AND NOT attisdropped",
        (_TABLE, column),
    ).fetchone()["t"]


def _pk(conn) -> str:
    return conn.execute(
        "SELECT pg_get_constraintdef(c.oid) AS d FROM pg_constraint c "
        "WHERE c.conrelid = to_regclass(%s) AND c.contype = 'p'",
        (_TABLE,),
    ).fetchone()["d"]


def _index_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = %s "
        "AND tablename = 'workflow_routing_pointer'",
        (_SCHEMA,),
    ).fetchall()
    return {r["indexname"] for r in rows}


def _count(conn) -> int:
    return conn.execute("SELECT count(*) AS n FROM workflow_routing_pointer").fetchone()["n"]


def test_036_up_on_an_empty_table_gives_the_release_shape(conn):
    conn.execute(_body(_UP_036))
    assert _pk(conn) == "PRIMARY KEY (asset_key)"
    for col in ("active_workflow_id", "prior_workflow_id", "canary_workflow_id"):
        assert _column_type(conn, col) == "text", col
    assert _column_type(conn, "canary_pct") == "smallint"
    assert _column_type(conn, "released_at") == "timestamp with time zone"
    # The legacy key and the clock are nullable; a seeded row needs neither.
    _insert(conn)
    row = conn.execute("SELECT * FROM workflow_routing_pointer").fetchone()
    assert row["card_type"] is None
    assert row["list"] is None
    assert row["observation_window_until"] is None
    assert row["canary_workflow_id"] is None
    # Both 014 indexes were dropped; only the active index is rebuilt.
    names = _index_names(conn)
    assert "workflow_routing_pointer_active" in names
    assert "workflow_routing_pointer_observation" not in names


def test_036_refuses_when_a_row_exists_and_changes_nothing(conn):
    """Fail closed: a 014-shape row makes the WHOLE file raise (D3 DO block)."""
    conn.execute(
        "INSERT INTO workflow_routing_pointer "
        "(card_type, list, active_workflow_id, observation_window_until) "
        "VALUES ('c', 'READY', 'aaaaaaaa-0000-4000-8000-0000000000ff', now())"
    )
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.RaiseException, match="not empty"):
        conn.execute(_body(_UP_036))
    conn.execute("ROLLBACK TO SAVEPOINT s")
    assert _column_type(conn, "active_workflow_id") == "uuid"
    assert _pk(conn) == "PRIMARY KEY (card_type, list)"
    assert _count(conn) == 1


def test_036_is_rerunnable_on_an_empty_table_after_014_re_runs(conn):
    """pg_guard.apply_migrations forgets the ledger and re-runs the chain
    strict: 014 (re-creating its observation index) then 036 must both pass."""
    conn.execute(_body(_UP_036))
    conn.execute(_body(_UP_014))
    conn.execute(_body(_UP_036))
    assert _pk(conn) == "PRIMARY KEY (asset_key)"
    assert "workflow_routing_pointer_observation" not in _index_names(conn)


def test_036_canary_columns_are_all_or_none(conn):
    conn.execute(_body(_UP_036))
    check = psycopg.errors.CheckViolation
    _rejects(conn, check, canary_workflow_id=WF_B)  # no pct / id / card
    _rejects(conn, check, canary_workflow_id=WF_B, canary_pct=10, canary_id="cn-1")
    _rejects(conn, check, canary_pct=10, canary_id="cn-1", released_by_card_id="card-1")
    _insert(
        conn,
        canary_workflow_id=WF_B,
        canary_pct=10,
        canary_id="cn-1",
        released_by_card_id="card-1",
    )
    assert _count(conn) == 1


@pytest.mark.parametrize(("pct", "ok"), [(0, False), (1, True), (50, True), (51, False)])
def test_036_canary_pct_is_1_to_50(conn, pct, ok):
    conn.execute(_body(_UP_036))
    cols = {
        "canary_workflow_id": WF_B,
        "canary_pct": pct,
        "canary_id": "cn-1",
        "released_by_card_id": "card-1",
    }
    if ok:
        _insert(conn, **cols)
        assert _count(conn) == 1
    else:
        _rejects(conn, psycopg.errors.CheckViolation, **cols)
        assert _count(conn) == 0


def test_036_canary_equal_to_active_is_rejected(conn):
    conn.execute(_body(_UP_036))
    _rejects(
        conn,
        psycopg.errors.CheckViolation,
        canary_workflow_id=WF_A,
        canary_pct=10,
        canary_id="cn-1",
        released_by_card_id="card-1",
    )


@pytest.mark.parametrize(
    "column",
    ["active_workflow_id", "prior_workflow_id"],
)
@pytest.mark.parametrize(
    "bad",
    ["A" * 15, "A" * 17, "aaaaaaaa-0000-4000-8000-0000000000ff"],
    ids=["15", "17", "uuid"],
)
def test_036_workflow_ids_must_be_16_char_n8n_ids(conn, column, bad):
    conn.execute(_body(_UP_036))
    _rejects(conn, psycopg.errors.CheckViolation, **{column: bad})


def test_036_asset_key_is_the_primary_key(conn):
    conn.execute(_body(_UP_036))
    _insert(conn)
    _rejects(conn, psycopg.errors.UniqueViolation, active_workflow_id=WF_C)
    _rejects(conn, psycopg.errors.NotNullViolation, asset_key=None)


def test_036_down_restores_014_then_up_reapplies(conn):
    conn.execute(_body(_UP_036))
    _insert(conn)
    conn.execute(
        "INSERT INTO schema_migration_files (filename) VALUES ('036_workflow_routing_release.sql')"
    )
    conn.execute(_body(_DOWN_036))
    assert _pk(conn) == "PRIMARY KEY (card_type, list)"
    assert _column_type(conn, "active_workflow_id") == "uuid"
    assert _column_type(conn, "prior_workflow_id") == "uuid"
    cols = {
        r["attname"]
        for r in conn.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid = to_regclass(%s) "
            "AND attnum > 0 AND NOT attisdropped",
            (_TABLE,),
        ).fetchall()
    }
    assert cols == {
        "card_type",
        "list",
        "active_workflow_id",
        "prior_workflow_id",
        "promoted_at",
        "observation_window_until",
        "rollback_thresholds",
    }
    assert _index_names(conn) >= {
        "workflow_routing_pointer_active",
        "workflow_routing_pointer_observation",
    }
    assert _count(conn) == 0, "down deletes every 036 row (none fits 014)"
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM schema_migration_files "
            "WHERE filename = '036_workflow_routing_release.sql'"
        ).fetchone()["n"]
        == 0
    )
    # 014's NOT NULLs are back.
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.NotNullViolation):
        conn.execute(
            "INSERT INTO workflow_routing_pointer (list, active_workflow_id, "
            "observation_window_until) VALUES ('READY', gen_random_uuid(), now())"
        )
    conn.execute("ROLLBACK TO SAVEPOINT s")
    # down is itself re-runnable, and up re-applies cleanly after it
    conn.execute(_body(_DOWN_036))
    conn.execute(_body(_UP_036))
    assert _pk(conn) == "PRIMARY KEY (asset_key)"
    _insert(conn)
    assert _count(conn) == 1
