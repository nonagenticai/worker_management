"""``prompt_drift``'s binding query (#314) on real PG 18, end to end.

Real migrations, real rows: two platform workers are adopted by
``scripts/adopt_live_worker_prompts.py`` (v1 PUBLISHED = their live text, plus a
binding each), then ONE worker's ``config.system_prompt`` is rewritten the way an
n8n ensure-node would. A third binding names a prompt that does not exist.

Every fixture row uses a per-test ``worker-drifttest-<hex>-`` prefix and is
purged afterwards; assertions read only those rows and their label sets, so the
result is exact whatever else the database holds.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://user:pw@127.0.0.1:PORT/worker_management_ci"  # pragma: allowlist secret
    export POSTGRES_SSLMODE=disable
    uv run pytest tests/integration/test_prompt_drift_sql.py -m pg -v --no-cov
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from prometheus_client import REGISTRY

from scripts import adopt_live_worker_prompts as job
from src.api.workers import prompt_drift
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import PG_CONNECT_ERRORS, apply_migrations, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.pg

P3 = "0195f48b-d000-7000-8000-000000000003"  # platform (george-dev)


@pytest_asyncio.fixture(scope="module")
async def pg_pool() -> AsyncIterator[None]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    await apply_migrations()
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture
async def prefix(pg_pool) -> AsyncIterator[str]:
    pfx = f"worker-drifttest-{uuid.uuid4().hex[:10]}-"
    prompt_drift._reset_for_tests()
    try:
        yield pfx
    finally:
        prompt_drift._reset_for_tests()
        await _purge(pfx)


async def _purge(pfx: str) -> None:
    like = pfx + "%"
    async with get_pg_connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM worker_prompt_binding WHERE worker_name LIKE $1", like)
        await conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id LIKE $1", like)
        await conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_versions WHERE prompt_id LIKE $1", like)
        await conn.execute("DELETE FROM prompt_owner WHERE prompt_id LIKE $1", like)
        await conn.execute("DELETE FROM workers WHERE worker_name LIKE $1", like)


async def _worker(worker_name: str, system: str) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id, "
            "visibility, is_agent, status, config) "
            "VALUES ($1, $2, $3::uuid, $4, $4, 'tenant', false, 'active', $5::jsonb)",
            f"wid-{uuid.uuid4().hex}",
            worker_name,
            P3,
            P3,
            json.dumps({"system_prompt": system}),
        )


def _drift(worker: str) -> float | None:
    return REGISTRY.get_sample_value(
        "wm_prompt_binding_drift", {"tenant_id": P3, "worker_name": worker}
    )


@pytest.mark.asyncio
async def test_two_bindings_one_drifted_one_unresolved(prefix):
    steady, drifted, orphan = prefix + "steady", prefix + "drifted", prefix + "orphan"
    await _worker(steady, "You are steady.")
    await _worker(drifted, "You are drifted, as adopted.")
    report = await job.adopt(apply=True, name_like=prefix + "%")
    assert report.conflicts == []

    # An ensure-node rewrites ONE config after adoption; its PUBLISHED v1 stays.
    async with get_pg_connection() as conn:
        await conn.execute(
            "UPDATE workers SET config = jsonb_set(config, '{system_prompt}', $1::jsonb) "
            "WHERE worker_name = $2",
            json.dumps("You are what the ensure-node wrote last."),
            drifted,
        )
        # A binding whose prompt has no PUBLISHED version (and no worker row).
        # Migration 035's FK needs the prompt to be OWNED by the binding's
        # tenant, so the prompt exists as an owner row with no versions at all.
        await conn.execute(
            "INSERT INTO prompt_owner (prompt_id, owner_tenant_id) VALUES ($1, $2)",
            prefix + "no-such-prompt",
            P3,
        )
        await conn.execute(
            "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
            "VALUES ($1, $2, $3, 'test')",
            P3,
            orphan,
            prefix + "no-such-prompt",
        )

    rows = [r for r in await prompt_drift.read_bindings() if r["worker_name"].startswith(prefix)]
    by_name = {r["worker_name"]: r for r in rows}
    assert set(by_name) == {steady, drifted, orphan}
    assert by_name[steady]["published_status"] == "PUBLISHED"
    assert by_name[steady]["published_version"] == 1
    assert by_name[orphan]["published_system"] is None
    assert by_name[orphan]["worker_found"] is False

    snap = prompt_drift.compute_drift(rows)
    assert snap.drift == {(P3, steady): 0, (P3, drifted): 1}
    assert snap.drift_total == 1
    assert snap.unresolved == 1
    assert snap.checked == 3

    # And through the exported series (the whole table, so label-scoped reads).
    assert await prompt_drift.refresh_once() is not None
    assert _drift(steady) == 0
    assert _drift(drifted) == 1
    assert _drift(orphan) is None
    assert (REGISTRY.get_sample_value("wm_prompt_binding_drift_total") or 0) >= 1
