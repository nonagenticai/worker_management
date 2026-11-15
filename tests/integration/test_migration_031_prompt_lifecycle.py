"""Migration 031 (prompt version lifecycle): up, down, and the refusal.

Each test builds the PRE-031 ledger from the real ``030_prompt_versions.sql`` in
a private schema inside ONE transaction, runs the real 031 file (and the real
down file) against it, asserts, and rolls the whole transaction back. Nothing
touches the shared ``public.prompt_versions`` the API tests use, and nothing
survives the test.

What each part proves:

1. **The mapping** — active→PUBLISHED, retired & never activated→DRAFT,
   retired & activated once→ARCHIVED — on a row set containing all three.
2. **The one-active index is gone** and two PUBLISHED can coexist; the CHECK is
   TRANSITIONAL (still admits the legacy spellings an old pod writes mid-rollout;
   032 tightens it); the default is DRAFT.
3. **Idempotent** — a re-run on the end state changes nothing (the startup
   runner re-runs the chain on a pre-ledger database).
4. **Precondition** — a prompt with two ``active`` rows (possible only once the
   030 index is dropped by hand) makes 031 raise and change nothing. Negative
   control: the same file applies cleanly once the duplicate is resolved.
5. **Down** — the highest PUBLISHED per prompt becomes active, everything else
   retired, the index/CHECK/default come back, 031's ledger row is removed.

Run locally:
    DATABASE_URL=postgresql://… uv run pytest \
        tests/integration/test_migration_031_prompt_lifecycle.py -m pg -v --no-cov
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
_DOWN_031 = _ROOT / "down" / "031_prompt_version_lifecycle.down.sql"


def _body(path: Path) -> str:
    """The file without its own BEGIN/COMMIT — the test owns the transaction."""
    sql = path.read_text()
    sql = re.sub(r"^\s*BEGIN\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)
    return re.sub(r"^\s*COMMIT\s*;\s*$", "", sql, flags=re.MULTILINE | re.IGNORECASE)


@pytest.fixture
def conn():
    """A connection whose one transaction is rolled back, scoped to a private schema."""
    require_database_url()
    try:
        c = psycopg.connect(pg_conninfo(), row_factory=dict_row)
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        c.execute("CREATE SCHEMA p4_mig031")
        # SET LOCAL: transaction-scoped, never leaks past the rollback.
        c.execute("SET LOCAL search_path = p4_mig031")
        c.execute(
            "CREATE TABLE schema_migration_files (filename TEXT PRIMARY KEY, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        c.execute(_body(_UP_030))
        yield c
    finally:
        c.rollback()
        c.close()


def _seed(conn, rows):
    for prompt_id, version, status, activated in rows:
        conn.execute(
            "INSERT INTO prompt_versions (prompt_id, version, sha256, system_prompt_sha256, "
            "system, status, activated_at) VALUES (%s, %s, %s, %s, %s, %s, "
            "CASE WHEN %s THEN now() ELSE NULL END)",
            (prompt_id, version, "a" * 64, "b" * 64, f"{prompt_id} v{version}", status, activated),
        )


def _statuses(conn) -> dict[tuple[str, int], str]:
    rows = conn.execute("SELECT prompt_id, version, status FROM prompt_versions").fetchall()
    return {(r["prompt_id"], r["version"]): r["status"] for r in rows}


def _indexes(conn) -> set[str]:
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'p4_mig031' "
        "AND tablename = 'prompt_versions'"
    ).fetchall()
    return {r["indexname"] for r in rows}


def _check_def(conn) -> str:
    row = conn.execute(
        "SELECT pg_get_constraintdef(oid) AS d FROM pg_constraint "
        "WHERE conname = 'prompt_versions_status_check' "
        "AND conrelid = 'p4_mig031.prompt_versions'::regclass"
    ).fetchone()
    return row["d"]


def _default(conn) -> str:
    return conn.execute(
        "SELECT column_default AS d FROM information_schema.columns WHERE "
        "table_schema = 'p4_mig031' AND table_name = 'prompt_versions' AND column_name = 'status'"
    ).fetchone()["d"]


# prompt_id, version, legacy status, activated_at set?
LEGACY = (
    ("a", 1, "retired", True),  # was live once        -> ARCHIVED
    ("a", 2, "active", True),  # live                   -> PUBLISHED
    ("a", 3, "retired", False),  # published, never live -> DRAFT
    ("b", 1, "retired", False),  # the prod shape (41/41) -> DRAFT
)
EXPECTED_UP = {
    ("a", 1): "ARCHIVED",
    ("a", 2): "PUBLISHED",
    ("a", 3): "DRAFT",
    ("b", 1): "DRAFT",
}


def test_031_maps_every_legacy_status(conn):
    _seed(conn, LEGACY)
    conn.execute(_body(_UP_031))
    assert _statuses(conn) == EXPECTED_UP


def test_031_drops_the_one_active_index_and_allows_several_published(conn):
    assert "prompt_versions_one_active_idx" in _indexes(conn), "030 baseline missing"
    _seed(conn, LEGACY)
    conn.execute(_body(_UP_031))
    assert "prompt_versions_one_active_idx" not in _indexes(conn)
    # Two PUBLISHED for one prompt is now legal (the resolver picks the highest).
    conn.execute("UPDATE prompt_versions SET status = 'PUBLISHED' WHERE prompt_id = 'a'")
    assert list(_statuses(conn).values()).count("PUBLISHED") == 3


def test_031_check_and_default(conn):
    conn.execute(_body(_UP_031))
    check = _check_def(conn)
    for status in ("DRAFT", "CANARY", "PUBLISHED", "ARCHIVED", "DISCARDED"):
        assert status in check
    assert "DRAFT" in _default(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():  # savepoint, so the outer transaction survives
            _seed(conn, [("c", 1, "nonsense", False)])


def test_031_check_is_transitional_so_an_old_pod_can_still_write(conn):
    """Mid-rollout a pre-031 pod still inserts 'retired' (PUT) and writes
    'active'/'retired' (activate). The 031 CHECK must admit them; the follow-up
    032 tightens it once no old pod runs."""
    conn.execute(_body(_UP_031))
    check = _check_def(conn)
    assert "'active'" in check and "'retired'" in check
    assert "032_prompt_version_lifecycle_tighten" in _UP_031.read_text()
    # main's PUT and main's activate, verbatim shapes, against the 031 schema.
    _seed(conn, [("old", 1, "retired", False), ("old", 2, "retired", False)])
    conn.execute(
        "UPDATE prompt_versions SET status = 'retired' "
        "WHERE prompt_id = 'old' AND status = 'active' AND version <> 2"
    )
    conn.execute(
        "UPDATE prompt_versions SET status = 'active', activated_at = now() "
        "WHERE prompt_id = 'old' AND version = 2"
    )
    assert _statuses(conn) == {("old", 1): "retired", ("old", 2): "active"}


def test_down_with_mixed_legacy_active_and_published_keeps_the_highest_active(conn):
    """Mixed window rows: an old pod's 'active' next to new code's PUBLISHED.
    The down must treat 'active' as PUBLISHED — otherwise it can leave two
    'active' rows (the one-active index then fails) or pick the lower one."""
    _seed(conn, [(p, v, "retired", False) for p in ("m", "n") for v in (1, 2)])
    conn.execute(_body(_UP_031))
    for pid, legacy_v, published_v in (("m", 1, 2), ("n", 2, 1)):
        conn.execute(
            "UPDATE prompt_versions SET status = 'active' WHERE prompt_id = %s AND version = %s",
            (pid, legacy_v),
        )
        conn.execute(
            "UPDATE prompt_versions SET status = 'PUBLISHED' WHERE prompt_id = %s AND version = %s",
            (pid, published_v),
        )
    conn.execute(_body(_DOWN_031))
    assert _statuses(conn) == {
        ("m", 1): "retired",
        ("m", 2): "active",
        ("n", 1): "retired",
        ("n", 2): "active",
    }


def test_031_is_idempotent(conn):
    _seed(conn, LEGACY)
    conn.execute(_body(_UP_031))
    conn.execute(_body(_UP_031))
    assert _statuses(conn) == EXPECTED_UP


def test_031_refuses_two_active_rows_for_one_prompt_and_changes_nothing(conn):
    conn.execute("DROP INDEX prompt_versions_one_active_idx")
    _seed(
        conn, [("dup", 1, "active", True), ("dup", 2, "active", True), ("ok", 1, "retired", False)]
    )
    before = _statuses(conn)
    with pytest.raises(psycopg.errors.RaiseException, match="dup"):
        with conn.transaction():
            conn.execute(_body(_UP_031))
    assert _statuses(conn) == before, "a refused 031 must write nothing"
    assert "'active'" in _check_def(conn), "a refused 031 must not swap the CHECK"
    # Negative control: resolve the duplicate and the same file applies.
    conn.execute(
        "UPDATE prompt_versions SET status = 'retired' WHERE prompt_id = 'dup' AND version = 1"
    )
    conn.execute(_body(_UP_031))
    assert _statuses(conn) == {("dup", 1): "ARCHIVED", ("dup", 2): "PUBLISHED", ("ok", 1): "DRAFT"}


def test_down_restores_030_with_the_highest_published_active(conn):
    _seed(conn, LEGACY)
    conn.execute(_body(_UP_031))
    conn.execute(
        "INSERT INTO schema_migration_files (filename) VALUES ('031_prompt_version_lifecycle.sql')"
    )
    # Post-031 states the down must collapse: two PUBLISHED on one prompt, plus
    # every other lifecycle status.
    conn.execute(
        "UPDATE prompt_versions SET status = 'PUBLISHED' WHERE prompt_id = 'a' AND version = 3"
    )
    _seed(conn, [("c", 1, "CANARY", False), ("c", 2, "DISCARDED", False)])

    conn.execute(_body(_DOWN_031))

    assert _statuses(conn) == {
        ("a", 1): "retired",  # ARCHIVED
        ("a", 2): "retired",  # PUBLISHED but not the highest
        ("a", 3): "active",  # highest PUBLISHED
        ("b", 1): "retired",  # DRAFT
        ("c", 1): "retired",  # CANARY
        ("c", 2): "retired",  # DISCARDED
    }
    assert "prompt_versions_one_active_idx" in _indexes(conn)
    assert "prompt_versions_status_idx" not in _indexes(conn)
    assert "'active'" in _check_def(conn) and "PUBLISHED" not in _check_def(conn)
    assert "retired" in _default(conn)
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM schema_migration_files "
            "WHERE filename = '031_prompt_version_lifecycle.sql'"
        ).fetchone()["n"]
        == 0
    ), "the down must remove 031's ledger row so the runner re-applies it"

    # Round trip: up again lands on a valid 031 state.
    conn.execute(_body(_UP_031))
    assert _statuses(conn)[("a", 3)] == "PUBLISHED"
    assert _statuses(conn)[("a", 2)] == "ARCHIVED"  # it had been activated
