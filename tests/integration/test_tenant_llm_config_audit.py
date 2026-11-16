"""Migration 037 + the customer-visible BYO-config audit, against real PostgreSQL 18.

Owner ruling 2026-10-02: when the owner sets, tests or removes a customer
tenant's BYO key, the CUSTOMER must be able to see it. The unit suite
(``tests/test_owner_key_test.py::TestCustomerVisibleAudit``) drives the real
middleware over an in-memory fake; what only a database can prove is here:

* 037 applies, its CHECKs refuse a plaintext-shaped ``key_id`` and unknown
  actions / actor kinds, and its down file removes it cleanly (up/down/up).
* The audit row is written IN THE SAME TRANSACTION as the config change: when
  the audit insert fails, the PUT / DELETE does not commit.
* ``GET /tenant-config/llm/audit`` is scoped by SQL to the caller's tenant,
  newest first, capped at 200, and neither the table nor the response holds
  the key, a hash of the plaintext key, or the actor ``sub``.

Run locally:
    DATABASE_URL=postgresql+psycopg://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql+psycopg://… uv run pytest \
        tests/integration/test_tenant_llm_config_audit.py -m pg -v --no-cov
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import httpx
import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from psycopg.rows import dict_row

from src.auth import TenantContext, get_tenant_context
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import (
    PG_CONNECT_ERRORS,
    apply_migrations,
    pg_conninfo,
    require_database_url,
    unreachable_pg,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.pg

_ROOT = Path(__file__).resolve().parents[2] / "migrations"
_UP_037 = _ROOT / "037_tenant_llm_config_audit.sql"
_DOWN_037 = _ROOT / "down" / "037_tenant_llm_config_audit.down.sql"
_UP_038 = _ROOT / "038_tenant_llm_config_audit_test_refused.sql"
_DOWN_038 = _ROOT / "down" / "038_tenant_llm_config_audit_test_refused.down.sql"
_SCHEMA = "u6_mig037"

TENANT_A = "aaaaaaaa-0000-4000-8000-0000000037aa"
TENANT_B = "aaaaaaaa-0000-4000-8000-0000000037bb"
OWNER_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
OWNER_SUB = "owner-sub-037-should-never-be-stored"
KEY = "sk-tenant-a-plaintext-037-key"  # pragma: allowlist secret


# ── Migration 037 in a private schema (sync, rolled back) ────────────────────


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


def _exists(c) -> bool:
    row = c.execute(
        "SELECT to_regclass(%s) IS NOT NULL AS e", (f"{_SCHEMA}.tenant_llm_config_audit",)
    ).fetchone()
    return bool(row["e"])


def _insert(c, **over):
    row = {
        "tenant_id": TENANT_A,
        "role": "__default__",
        "action": "set_key",
        "actor_kind": "platform_operator",
        "provider": "openai",
        "key_id": "0123456789ab",
    }
    row.update(over)
    c.execute(
        "INSERT INTO tenant_llm_config_audit (tenant_id, role, action, actor_kind, provider, "
        "key_id) VALUES (%(tenant_id)s, %(role)s, %(action)s, %(actor_kind)s, %(provider)s, "
        "%(key_id)s)",
        row,
    )


def test_037_up_down_up(conn):
    conn.execute(_body(_UP_037))
    assert _exists(conn)
    _insert(conn)
    conn.execute("INSERT INTO schema_migration_files (filename) VALUES (%s)", (_UP_037.name,))
    conn.execute(_body(_DOWN_037))
    assert not _exists(conn)
    left = conn.execute("SELECT count(*) AS n FROM schema_migration_files").fetchone()
    assert left["n"] == 0
    conn.execute(_body(_UP_037))
    conn.execute(_body(_UP_037))  # re-runnable
    assert _exists(conn)
    cols = {
        r["column_name"]: r["data_type"]
        for r in conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'tenant_llm_config_audit'",
            (_SCHEMA,),
        ).fetchall()
    }
    assert cols == {
        "id": "bigint",
        "tenant_id": "uuid",
        "role": "text",
        "action": "text",
        "actor_kind": "text",
        "provider": "text",
        "key_id": "text",
        "created_at": "timestamp with time zone",
    }
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s "
        "AND indexname = 'idx_tenant_llm_config_audit_tenant'",
        (_SCHEMA,),
    ).fetchone()
    assert idx and "(tenant_id, created_at DESC, id DESC)" in idx["indexdef"]


@pytest.mark.parametrize(
    "over",
    [
        {"key_id": KEY},  # a plaintext key can never land in key_id
        {"key_id": "0123456789AB"},
        {"key_id": "0123456789abc"},
        {"action": "delete"},
        {"actor_kind": "owner"},
    ],
)
def test_037_checks_refuse_bad_rows(conn, over):
    conn.execute(_body(_UP_037))
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, **over)
    conn.execute("ROLLBACK TO SAVEPOINT s")
    _insert(conn, key_id="none")  # the control: a good row still lands


def test_038_admits_test_refused_and_nothing_else(conn):
    """WM-4: 037 refuses ``test_refused``; 038 admits it, still refuses garbage;
    the down file narrows the CHECK back (deleting those rows) and 038 re-applies."""
    conn.execute(_body(_UP_037))
    conn.execute("SAVEPOINT s")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, action="test_refused")  # 037 alone: refused
    conn.execute("ROLLBACK TO SAVEPOINT s")
    conn.execute(_body(_UP_038))
    conn.execute(_body(_UP_038))  # re-runnable
    _insert(conn, action="test_refused")
    _insert(conn, action="test")
    conn.execute("SAVEPOINT t")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, action="test_refusedx")
    conn.execute("ROLLBACK TO SAVEPOINT t")
    conn.execute("INSERT INTO schema_migration_files (filename) VALUES (%s)", (_UP_038.name,))
    conn.execute(_body(_DOWN_038))
    left = conn.execute(
        "SELECT action, count(*) AS n FROM tenant_llm_config_audit GROUP BY action"
    ).fetchall()
    assert {r["action"]: r["n"] for r in left} == {"test": 1}
    gone = conn.execute(
        "SELECT count(*) AS n FROM schema_migration_files WHERE filename = %s", (_UP_038.name,)
    ).fetchone()
    assert gone["n"] == 0
    conn.execute("SAVEPOINT u")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, action="test_refused")  # narrowed again
    conn.execute("ROLLBACK TO SAVEPOINT u")
    conn.execute(_body(_UP_038))
    _insert(conn, action="test_refused")


# ── The router against the real database ────────────────────────────────────


@pytest_asyncio.fixture
async def pg_pool() -> AsyncIterator[None]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    await apply_migrations()
    await _purge()
    try:
        yield
    finally:
        await _purge()
        await close_pg_pool()


async def _purge() -> None:
    async with get_pg_connection() as c:
        await c.execute("DROP TRIGGER IF EXISTS t037_break ON tenant_llm_config_audit")
        await c.execute("DROP FUNCTION IF EXISTS t037_break()")
        for table in ("tenant_llm_config_audit", "tenant_llm_config"):
            await c.execute(
                f"DELETE FROM {table} WHERE tenant_id = ANY($1::uuid[])",
                [TENANT_A, TENANT_B],
            )


def _app(tenant: dict[str, str]) -> FastAPI:
    from src.api.tenant_config._router import _require_licence_or_delegated_caller, router

    app = FastAPI()
    app.include_router(router)
    # No middleware here, so provenance is ``unknown`` and the audit route would
    # 401. Its caller gate is proven through the real middleware in
    # tests/test_owner_key_test.py::TestAuditRefusesTokenless; this file proves SQL.
    app.dependency_overrides[_require_licence_or_delegated_caller] = lambda: None
    app.dependency_overrides[get_tenant_context] = lambda: TenantContext(
        tenant_id=tenant["id"],
        subject_id=tenant["id"],
        tenant_name="t",
        visibility="tenant",
        byo_mode="local_allowed",
    )
    return app


@pytest_asyncio.fixture
async def api(pg_pool, monkeypatch):
    """(client, tenant holder, identity holder) over the real router + real DB."""
    from src.api.tenant_config import _router as tc

    monkeypatch.setenv("WM_SECRET_KEY", "a" * 64)
    from src.settings import reset_settings_cache

    reset_settings_cache()
    tenant = {"id": TENANT_A}
    # (sub, token_tenant, super_admin, delegator) of the verified end-user JWT.
    ident = {"v": (OWNER_SUB, OWNER_TENANT, True, "svc")}
    monkeypatch.setattr(tc, "get_end_user_identity", lambda: ident["v"])
    monkeypatch.setattr(tc, "tenant_key_entitled", lambda _t: True)
    tc._key_test_cooldown.clear()
    transport = httpx.ASGITransport(app=_app(tenant))
    async with httpx.AsyncClient(transport=transport, base_url="http://wm") as client:
        yield client, tenant, ident
    tc._key_test_cooldown.clear()
    reset_settings_cache()


async def _audit_rows() -> list[dict]:
    async with get_pg_connection() as c:
        return await c.fetch(
            "SELECT * FROM tenant_llm_config_audit WHERE tenant_id = ANY($1::uuid[]) ORDER BY id",
            [TENANT_A, TENANT_B],
        )


@pytest.mark.asyncio
async def test_writes_are_audited_and_hold_no_key_and_no_sub(api):
    client, _tenant, ident = api
    put = await client.put(
        "/tenant-config/llm", json={"provider": "openai", "model": "gpt-4o", "apiKey": KEY}
    )
    assert put.status_code == 200, put.text
    with patch(
        "src.api.providers._router._httpx_json_get",
        new=AsyncMock(return_value={"data": [{"id": "gpt-4o"}]}),
    ):
        test = await client.post("/tenant-config/llm/test", json={})
    assert test.status_code == 200, test.text
    ident["v"] = ("tenant-user", TENANT_A, False, "svc")  # the customer's own admin
    put = await client.put("/tenant-config/llm", json={"provider": "openai", "model": "gpt-4.1"})
    assert put.status_code == 200, put.text
    ident["v"] = ("other-user", TENANT_B, False, "svc")  # foreign, not super_admin
    put = await client.put("/tenant-config/llm", json={"provider": "openai", "model": "gpt-4.2"})
    assert put.status_code == 200, put.text
    ident["v"] = ("", "", False, "")  # a service, no end-user JWT
    assert (await client.delete("/tenant-config/llm")).status_code == 204

    rows = await _audit_rows()
    assert [(r["action"], r["actor_kind"]) for r in rows] == [
        ("set_key", "platform_operator"),
        ("test", "platform_operator"),
        ("update", "workspace_admin"),
        ("update", "service"),
        ("remove", "service"),
    ]
    assert len({r["key_id"] for r in rows}) == 1 and rows[0]["key_id"] != "none"

    resp = await client.get("/tenant-config/llm/audit")
    assert resp.status_code == 200, resp.text
    entries = resp.json()["entries"]
    assert [e["action"] for e in entries] == ["remove", "update", "update", "test", "set_key"]
    plain_hash = hashlib.sha256(KEY.encode("utf-8")).hexdigest()
    stored = json.dumps(rows, default=str)
    for blob in (stored, resp.text):
        assert KEY not in blob
        assert plain_hash[:12] not in blob
        assert OWNER_SUB not in blob and "tenant-user" not in blob and "other-user" not in blob


@pytest.mark.asyncio
async def test_tenant_a_never_reads_tenant_b(api):
    client, tenant, _ident = api
    tenant["id"] = TENANT_B
    put = await client.put(
        "/tenant-config/llm", json={"provider": "openai", "model": "b-model", "apiKey": KEY}
    )
    assert put.status_code == 200, put.text
    tenant["id"] = TENANT_A
    resp = await client.get("/tenant-config/llm/audit")
    assert resp.status_code == 200, resp.text
    assert resp.json()["entries"] == []
    tenant["id"] = TENANT_B
    resp = await client.get("/tenant-config/llm/audit")
    assert [e["action"] for e in resp.json()["entries"]] == ["set_key"]


@pytest.mark.asyncio
async def test_limit_is_capped_in_sql_and_newest_first(api):
    client, _tenant, _ident = api
    async with get_pg_connection() as c:
        await c.execute(
            "INSERT INTO tenant_llm_config_audit (tenant_id, role, action, actor_kind, provider, "
            "key_id, created_at) SELECT $1::uuid, '__default__', 'test', 'service', 'p' || g, "
            "'none', now() - make_interval(secs => 300 - g) FROM generate_series(1, 250) g",
            TENANT_A,
        )
    from src.db.tenant_llm_config_db import TenantLLMConfigDB

    assert len(await TenantLLMConfigDB().list_audit(TENANT_A, 10_000)) == 200
    resp = await client.get("/tenant-config/llm/audit?limit=1000")
    entries = resp.json()["entries"]
    assert len(entries) == 200
    assert entries[0]["provider"] == "p250" and entries[-1]["provider"] == "p51"
    resp = await client.get("/tenant-config/llm/audit")
    assert len(resp.json()["entries"]) == 50


@pytest.mark.asyncio
async def test_a_failed_audit_insert_rolls_the_change_back(api):
    """Same transaction: no audit row ⇒ no config change, for PUT and DELETE."""
    client, _tenant, _ident = api
    put = await client.put(
        "/tenant-config/llm", json={"provider": "openai", "model": "gpt-4o", "apiKey": KEY}
    )
    assert put.status_code == 200, put.text
    async with get_pg_connection() as c:
        await c.execute(
            "CREATE FUNCTION t037_break() RETURNS trigger LANGUAGE plpgsql AS "
            "$$ BEGIN RAISE EXCEPTION 't037 audit unavailable'; END $$"
        )
        await c.execute(
            "CREATE TRIGGER t037_break BEFORE INSERT ON tenant_llm_config_audit "
            "FOR EACH ROW EXECUTE FUNCTION t037_break()"
        )
    with pytest.raises(psycopg.errors.RaiseException):
        await client.put("/tenant-config/llm", json={"provider": "openai", "model": "gpt-CHANGED"})
    with pytest.raises(psycopg.errors.RaiseException):
        await client.delete("/tenant-config/llm")
    async with get_pg_connection() as c:
        row = await c.fetchrow(
            "SELECT model FROM tenant_llm_config WHERE tenant_id = $1 AND role = '__default__'",
            TENANT_A,
        )
    assert row is not None and row["model"] == "gpt-4o"
