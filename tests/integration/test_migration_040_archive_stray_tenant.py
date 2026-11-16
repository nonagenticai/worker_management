"""Migration 040 archives, then deletes, rows keyed by licence subject 8382f0b3.

Proof for ``migrations/040_archive_stray_tenant_8382f0b3.sql`` (and its down
file). Each test gets a THROWAWAY database whose ``public`` schema and every
table are owned by a NON-superuser role, and the whole chain runs AS that role
-- the shape of prod's ``worker_mgmt_user`` (owner == app == migrating role,
measured 2026-10-14). A superuser passes every privilege check, so the
"archive is unreadable by the app role" assertion would prove nothing as one.

Asserted:
* every stray row (tenant_id = the subject), including an FK child reached only
  through the fresh schema's ``graph_states -> graphs`` cascade, lands in
  ``archive.<table>_8382f0b3`` row-for-row and is gone from ``public``;
* NEGATIVE CONTROL: rows of OTHER tenants whose ``subject_id`` is the stray id
  survive byte-for-byte -- the match is ``tenant_id`` only;
* re-running (ledger forgotten) changes nothing; an empty database no-ops and
  creates no ``archive`` schema;
* the app/owner role gets 42501 on the archive (positive control: it reads
  ``public.workers``), and so does an unrelated role;
* it REFUSES, changing nothing, on: another tenant's FK child of a stray parent;
  an FK child table with no tenant_id; the stray id in a table outside the set;
  stray rows when an archive copy already exists;
* the down file restores ``public`` byte-for-byte and drops the schema.

Run locally (DATABASE_URL = a SUPERUSER on a throwaway postgres:18, because the
fixture creates a database and two roles; POSTGRES_SSLMODE=disable)::

    uv run pytest tests/integration/test_migration_040_archive_stray_tenant.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
import pytest_asyncio
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.db.db_config import run_migrations

from .pg_guard import PG_CONNECT_ERRORS, pg_conninfo, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

REPO = Path(__file__).resolve().parents[2]
FILE = "040_archive_stray_tenant_8382f0b3.sql"
DOWN = REPO / "migrations" / "down" / "040_archive_stray_tenant_8382f0b3.down.sql"
STRAY = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
TENANT_A = "bbbbbbbb-0000-4000-8000-0000000000a5"
TENANT_B = "01a1113b-674f-72b0-a84b-dc746d39afca"
SUFFIX = "_8382f0b3"
TABLES = ("workers", "graphs", "worker_states", "graph_states")
G_STRAY = "00000000-0000-4000-8000-0000000000a1"
G_KEEP = "00000000-0000-4000-8000-0000000000b1"


@dataclass
class Db:
    admin: psycopg.AsyncConnection  # superuser, in the throwaway database
    owner: psycopg.AsyncConnection  # the non-superuser owner/app role
    pool: AsyncConnectionPool  # what run_migrations uses, as the owner
    owner_role: str
    outsider_role: str


@pytest_asyncio.fixture
async def db() -> AsyncIterator[Db]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    tag = uuid.uuid4().hex[:10]
    name, owner, outsider = f"wm040_{tag}", f"wm040_owner_{tag}", f"wm040_out_{tag}"
    try:
        server = await psycopg.AsyncConnection.connect(pg_conninfo(), autocommit=True)
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    await server.execute(f'CREATE DATABASE "{name}"')
    await server.execute(f'CREATE ROLE "{owner}" NOLOGIN NOSUPERUSER NOBYPASSRLS')
    await server.execute(f'CREATE ROLE "{outsider}" NOLOGIN NOSUPERUSER NOBYPASSRLS')
    await server.execute(f'GRANT CONNECT, CREATE ON DATABASE "{name}" TO "{owner}"')
    in_db = make_conninfo(pg_conninfo(), dbname=name)
    as_owner = make_conninfo(in_db, options=f"-c role={owner}")
    admin = await psycopg.AsyncConnection.connect(in_db, autocommit=True, row_factory=dict_row)
    await admin.execute(f'ALTER SCHEMA public OWNER TO "{owner}"')
    owner_conn = await psycopg.AsyncConnection.connect(
        as_owner, autocommit=True, row_factory=dict_row
    )
    await owner_conn.execute((REPO / "postgres_schema.sql").read_text())
    pool = AsyncConnectionPool(
        conninfo=as_owner,
        min_size=1,
        max_size=2,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    )
    await pool.open()
    try:
        yield Db(admin, owner_conn, pool, owner, outsider)
    finally:
        await pool.close()
        await owner_conn.close()
        await admin.close()
        await server.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,)
        )
        await server.execute(f'DROP DATABASE IF EXISTS "{name}"')
        await server.execute(f'DROP ROLE IF EXISTS "{owner}"')
        await server.execute(f'DROP ROLE IF EXISTS "{outsider}"')
        await server.close()


async def _rows(conn: psycopg.AsyncConnection, sql: str, *args: Any) -> list[Any]:
    cur = await conn.execute(sql, args or None)
    return await cur.fetchall()


async def _snapshot(conn: psycopg.AsyncConnection, schema: str, table: str, where: str) -> list:
    if (await _rows(conn, "SELECT to_regclass(%s) AS r", f"{schema}.{table}"))[0]["r"] is None:
        return []
    rows = await _rows(
        conn, f"SELECT to_jsonb(t)::text AS j FROM {schema}.{table} t WHERE {where} ORDER BY 1"
    )
    return [r["j"] for r in rows]


async def _state(conn: psycopg.AsyncConnection) -> dict[str, tuple[list, list]]:
    """table -> (stray rows, every other row), as canonical json text."""
    stray = f"tenant_id = '{STRAY}'::uuid"
    other = f"tenant_id <> '{STRAY}'::uuid"
    return {
        t: (await _snapshot(conn, "public", t, stray), await _snapshot(conn, "public", t, other))
        for t in TABLES
    }


async def _archive(conn: psycopg.AsyncConnection) -> dict[str, list]:
    return {t: await _snapshot(conn, "archive", t + SUFFIX, "true") for t in TABLES}


async def _has_archive_schema(conn: psycopg.AsyncConnection) -> bool:
    rows = await _rows(conn, "SELECT 1 FROM pg_namespace WHERE nspname = 'archive'")
    return bool(rows)


async def _chain(d: Db) -> list[str]:
    applied, failed = await run_migrations(pool=d.pool, strict=True)
    assert failed == []
    return applied


async def _rerun_040(d: Db) -> list[str]:
    await d.owner.execute("DELETE FROM schema_migration_files WHERE filename = %s", (FILE,))
    return await _chain(d)


async def _seed(d: Db) -> None:
    c = d.owner

    async def worker(wid: str, tenant: str, subject: str) -> None:
        await c.execute(
            "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (wid, wid, tenant, tenant, subject),
        )

    async def state(wid: str, tenant: str, subject: str, n: int) -> None:
        await c.execute(
            "INSERT INTO worker_states (worker_id, worker_name, tenant_id, tenant_name, "
            "subject_id, state_type, worker_state, timestamp) "
            "VALUES (%s, %s, %s, %s, %s, 'updated', '{}', now() + make_interval(secs => %s))",
            (wid, wid, tenant, tenant, subject, n),
        )

    await worker("w-stray-1", STRAY, STRAY)
    await worker("w-stray-2", STRAY, "some-other-subject")
    # NEGATIVE CONTROLS: real tenants' rows created BY the stray subject.
    await worker("w-a-by-stray-subject", TENANT_A, STRAY)
    await worker("w-b", TENANT_B, "b-user")
    await state("w-stray-1", STRAY, STRAY, 1)
    await state("w-stray-1", STRAY, STRAY, 2)
    await state("w-stray-2", STRAY, STRAY, 3)
    await state("w-a-by-stray-subject", TENANT_A, STRAY, 4)
    await state("w-b", TENANT_B, "b-user", 5)
    for gid, tenant, subject in ((G_STRAY, STRAY, STRAY), (G_KEEP, TENANT_B, STRAY)):
        await c.execute(
            "INSERT INTO graphs (id, name, tenant_id, tenant_name, subject_id, nodes, edges) "
            "VALUES (%s, %s, %s, %s, %s, '[]', '[]')",
            (gid, gid, tenant, tenant, subject),
        )
        # graph_states -> graphs is ON DELETE CASCADE in the fresh schema.
        await c.execute(
            "INSERT INTO graph_states (graph_id, tenant_id, tenant_name, subject_id, state_data) "
            "VALUES (%s, %s, %s, %s, '{}')",
            (gid, tenant, tenant, subject),
        )


async def test_empty_database_noops_and_creates_no_archive(db: Db) -> None:
    applied = await _chain(db)
    assert FILE in applied
    assert not await _has_archive_schema(db.owner)
    await _rerun_040(db)
    assert not await _has_archive_schema(db.owner)


async def test_only_stray_rows_are_archived_then_deleted(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    before = await _state(db.owner)
    # The corpus really landed (a deletion test with no rows passes for nothing).
    assert [len(before[t][0]) for t in TABLES] == [2, 1, 3, 1]
    assert [len(before[t][1]) for t in TABLES] == [2, 1, 2, 1]

    assert FILE in await _rerun_040(db)

    after = await _state(db.owner)
    for t in TABLES:
        assert after[t][0] == [], f"stray rows remain in {t}"
        assert after[t][1] == before[t][1], f"non-stray rows of {t} changed"
    # The control row authored by the stray subject in tenant A is still there.
    kept = await _rows(db.owner, "SELECT worker_id FROM workers WHERE subject_id = %s", STRAY)
    assert [r["worker_id"] for r in kept] == ["w-a-by-stray-subject"]

    await db.admin.execute("SET ROLE NONE")  # superuser reads the archive
    archived = await _archive(db.admin)
    assert archived == {t: before[t][0] for t in TABLES}

    # Idempotent: a forgotten ledger row re-runs 040, which now finds nothing.
    await _rerun_040(db)
    assert await _state(db.owner) == after
    assert await _archive(db.admin) == archived


async def test_archive_is_unreadable_by_the_app_role_and_public(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    await _rerun_040(db)
    # Positive control: the same role reads its own live table.
    assert (await _rows(db.owner, "SELECT count(*) AS n FROM public.workers"))[0]["n"] == 2
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await db.owner.execute(f"SELECT * FROM archive.workers{SUFFIX}")
    await db.admin.execute(f'SET ROLE "{db.outsider_role}"')
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await db.admin.execute(f"SELECT * FROM archive.workers{SUFFIX}")
    finally:
        await db.admin.execute("RESET ROLE")


async def _assert_refused(db: Db, match: str) -> None:
    before = await _state(db.owner)
    with pytest.raises(psycopg.Error, match=match):
        await _rerun_040(db)
    assert await _state(db.owner) == before
    assert not await _has_archive_schema(db.owner)


async def test_refuses_when_another_tenants_fk_child_hangs_off_a_stray_parent(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    await db.owner.execute(
        "INSERT INTO graph_states (graph_id, tenant_id, tenant_name, subject_id, state_data) "
        "VALUES (%s, %s, 'a', 'a-user', '{}')",
        (G_STRAY, TENANT_A),
    )
    await _assert_refused(db, "another tenant")


async def test_refuses_an_fk_child_table_without_tenant_id(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    await db.owner.execute(
        "CREATE TABLE extra_child (id int, worker_id text REFERENCES workers(worker_id) "
        "ON DELETE CASCADE)"
    )
    await _assert_refused(db, "lack a uuid tenant_id")


async def test_refuses_the_stray_id_outside_the_archived_set(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    await db.owner.execute(
        "INSERT INTO tenant_llm_config (tenant_id, provider) VALUES (%s, 'local')", (STRAY,)
    )
    await _assert_refused(db, "outside the archived set")


async def test_refuses_to_mix_into_an_existing_archive_copy(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    await _rerun_040(db)
    await db.admin.execute("SET ROLE NONE")
    archived = await _archive(db.admin)
    await db.owner.execute(
        "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id) "
        "VALUES ('w-stray-late', 'w-stray-late', %s, 'x', 'x')",
        (STRAY,),
    )
    before = await _state(db.owner)
    with pytest.raises(psycopg.Error, match="already exists"):
        await _rerun_040(db)
    assert await _state(db.owner) == before
    assert await _archive(db.admin) == archived


async def test_down_restores_public_and_drops_the_archive(db: Db) -> None:
    await _chain(db)
    await _seed(db)
    before = await _state(db.owner)
    await _rerun_040(db)
    await db.owner.execute(DOWN.read_text())
    assert await _state(db.owner) == before
    assert not await _has_archive_schema(db.owner)
    ledger = await _rows(db.owner, "SELECT 1 FROM schema_migration_files WHERE filename = %s", FILE)
    assert ledger == []
