"""U6 config MIRROR (LEDGER C2-32) on real PG 18.

With ``WM_PROMPT_SOURCE=published`` every prompt-version transition into or out
of PUBLISHED rewrites ``workers.config.system_prompt`` of each BOUND platform
worker to the RESOLVED text (highest PUBLISHED, re-read from the table), in the
SAME transaction as the status move. Proved here:

* publish v2 -> config == v2, every other config key unchanged, drift 0;
* archive the top -> config falls back to the NEXT-highest PUBLISHED, recomputed
  (never the version just moved); activate of a lower version likewise; nothing
  PUBLISHED left -> config untouched;
* a rolled-back transition leaves BOTH the status and the config unchanged;
* two concurrent transitions on one prompt serialise: the config ends at the
  resolved winner (deterministic interleaving, not luck);
* a CUSTOMER-tenant worker and an UNBOUND worker are never touched, even when
  their config text equals the old text;
* ``WM_PROMPT_SOURCE=legacy`` mirrors nothing (a publish is not a deploy).

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://user:pw@127.0.0.1:PORT/worker_management_ci"  # pragma: allowlist secret
    export POSTGRES_SSLMODE=disable
    uv run pytest tests/integration/test_prompt_config_mirror.py -m pg -v --no-cov
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from prometheus_client import REGISTRY

from src.api.prompts import repository
from src.api.prompts.actor import Actor
from src.api.prompts.hashing import sha256_text, version_payload_sha256
from src.api.workers import prompt_drift
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.settings import reset_settings_cache

from .pg_guard import PG_CONNECT_ERRORS, apply_migrations, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

P1 = "aaaaaaaa-0000-4000-8000-000000000001"  # platform (system tenant)
CUSTOMER = "7a3c1e5d-0000-4000-8000-00000000c0de"  # not in PLATFORM_TENANT_IDS
ACTOR = Actor(sub="test-mirror", kind="migration")

V1 = "You are the mirror test worker, version one."
V2 = "You are the mirror test worker, version two."
V3 = "You are the mirror test worker, version three."
EXTRA = {"model": "local-llm", "temperature": 0.2, "tools": ["a", "b"]}


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


@pytest.fixture
def published_flag(monkeypatch) -> Any:
    monkeypatch.setenv("WM_PROMPT_SOURCE", "published")
    reset_settings_cache()
    yield
    monkeypatch.delenv("WM_PROMPT_SOURCE", raising=False)
    reset_settings_cache()


@pytest_asyncio.fixture
async def prefix(pg_pool) -> AsyncIterator[str]:
    pfx = f"worker-mirrortest-{uuid.uuid4().hex[:10]}-"
    try:
        yield pfx
    finally:
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


async def _worker(name: str, system: str, tenant: str = P1) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id, "
            "visibility, is_agent, status, config) "
            "VALUES ($1, $2, $3::uuid, $4, $4, 'tenant', false, 'active', $5::jsonb)",
            f"wid-{uuid.uuid4().hex}",
            name,
            tenant,
            tenant,
            json.dumps({"system_prompt": system, **EXTRA}),
        )


async def _bind(name: str, pid: str, tenant: str = P1) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
            "VALUES ($1, $2, $3, 'test')",
            tenant,
            name,
            pid,
        )


async def _draft(pid: str, version: int, system: str, tenant: str = P1) -> None:
    payload = {
        "question": None,
        "system": system,
        "template": None,
        "output_schema": None,
        "worker_names": [],
        "purpose_tags": [],
    }
    await repository.upsert_version(
        prompt_id=pid,
        version=version,
        sha256=version_payload_sha256(**payload),
        system_prompt_sha256=sha256_text(system),
        git_sha=None,
        actor=ACTOR,
        owner_tenant_id=tenant,
        **payload,
    )


async def _move(pid: str, version: int, to: str, **kw: Any) -> None:
    row, _ = await repository.transition(pid, version, to, actor=ACTOR, **kw)
    assert row is not None


async def _publish(pid: str, version: int) -> None:
    await _move(pid, version, repository.CANARY)
    await _move(pid, version, repository.PUBLISHED)


async def _seed(pid: str, versions: dict[int, str], tenant: str = P1) -> None:
    """v1 via initial_create; each further version DRAFT -> CANARY -> PUBLISHED."""
    for v, text in versions.items():
        await _draft(pid, v, text, tenant)
    first, *rest = sorted(versions)
    await _move(pid, first, repository.PUBLISHED, reason=repository.INITIAL_CREATE)
    for v in rest:
        await _publish(pid, v)


async def _config(name: str) -> dict[str, Any]:
    async with get_pg_connection() as conn:
        cfg = await conn.fetchval("SELECT config FROM workers WHERE worker_name = $1", name)
    return json.loads(cfg) if isinstance(cfg, str) else dict(cfg)


async def _status(pid: str, version: int) -> str:
    async with get_pg_connection() as conn:
        return await conn.fetchval(
            "SELECT status FROM prompt_versions WHERE prompt_id = $1 AND version = $2",
            pid,
            version,
        )


def _synced(pid: str) -> float:
    return REGISTRY.get_sample_value("wm_prompt_config_synced_total", {"prompt_id": pid}) or 0.0


async def test_publish_mirrors_the_new_text_and_nothing_else(prefix, published_flag):
    pid, cpid = prefix + "prompt", prefix + "customer-prompt"
    bound, unbound, customer = prefix + "bound", prefix + "unbound", prefix + "customer"
    await _worker(bound, V1)
    await _worker(unbound, V1)  # same OLD text, but no binding
    await _worker(customer, V1, tenant=CUSTOMER)  # same OLD text, bound, customer
    await _seed(pid, {1: V1})
    await _seed(cpid, {1: V1}, tenant=CUSTOMER)
    await _bind(bound, pid)
    await _bind(customer, cpid, tenant=CUSTOMER)
    await _draft(pid, 2, V2)
    await _draft(cpid, 2, V2, tenant=CUSTOMER)

    before = _synced(pid)
    await _publish(pid, 2)
    await _publish(cpid, 2)

    assert await _config(bound) == {"system_prompt": V2, **EXTRA}
    assert await _config(unbound) == {"system_prompt": V1, **EXTRA}
    assert await _config(customer) == {"system_prompt": V1, **EXTRA}
    assert _synced(pid) == before + 1
    assert _synced(cpid) == 0

    rows = [r for r in await prompt_drift.read_bindings() if r["worker_name"] == bound]
    snap = prompt_drift.compute_drift(rows)
    assert snap.drift == {(P1, bound): 0}
    assert snap.drift_total == 0


async def test_archive_falls_back_to_the_next_highest_published(prefix, published_flag):
    pid, w = prefix + "prompt", prefix + "w"
    await _worker(w, V1)
    await _seed(pid, {1: V1})
    await _bind(w, pid)
    await _draft(pid, 2, V2)
    await _draft(pid, 3, V3)
    await _publish(pid, 2)
    await _publish(pid, 3)
    assert (await _config(w))["system_prompt"] == V3

    # Archive the TOP: the mirror is recomputed from the table (v2), never the
    # text of the version just moved (v3).
    await _move(pid, 3, repository.ARCHIVED)
    assert (await _config(w))["system_prompt"] == V2

    # DISCARD of the new top (PUBLISHED -> DISCARDED) also removes it.
    await _move(pid, 2, repository.DISCARDED)
    assert (await _config(w))["system_prompt"] == V1

    # Re-publish v3 then roll back with the deprecated activate of v1: it
    # archives v3, and the mirror follows to v1.
    await _move(pid, 3, repository.PUBLISHED)
    assert (await _config(w))["system_prompt"] == V3
    _, previous, changed = await repository.activate(pid, 1, actor=ACTOR)
    assert (previous, changed) == (3, True)
    assert await _config(w) == {"system_prompt": V1, **EXTRA}

    # Nothing PUBLISHED left: the config is left exactly as it was.
    await _move(pid, 1, repository.ARCHIVED)
    assert await _config(w) == {"system_prompt": V1, **EXTRA}


async def test_canary_does_not_change_the_mirror(prefix, published_flag):
    pid, w = prefix + "prompt", prefix + "w"
    await _worker(w, V1)
    await _seed(pid, {1: V1})
    await _bind(w, pid)
    # An inert config edit after the sync stays (CANARY is never resolved, so a
    # CANARY move must not rewrite the config).
    async with get_pg_connection() as conn:
        await conn.execute(
            "UPDATE workers SET config = jsonb_set(config, '{system_prompt}', $1::jsonb) "
            "WHERE worker_name = $2",
            json.dumps("edited by hand"),
            w,
        )
    await _draft(pid, 2, V2)
    await _move(pid, 2, repository.CANARY)
    assert (await _config(w))["system_prompt"] == "edited by hand"


async def test_a_rolled_back_transition_changes_neither_status_nor_config(prefix, published_flag):
    pid, w = prefix + "prompt", prefix + "w"
    await _worker(w, V1)
    await _seed(pid, {1: V1})
    await _bind(w, pid)
    await _draft(pid, 2, V2)
    await _move(pid, 2, repository.CANARY)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with get_pg_connection() as conn, conn.transaction():
            await repository.transition(pid, 2, repository.PUBLISHED, actor=ACTOR, conn=conn)
            # Inside the transaction the mirror is already written ...
            inside = await conn.fetchval(
                "SELECT config ->> 'system_prompt' FROM workers WHERE worker_name = $1", w
            )
            assert inside == V2
            raise _Boom  # ... a failure before commit

    assert await _status(pid, 2) == repository.CANARY
    assert await _config(w) == {"system_prompt": V1, **EXTRA}


async def test_concurrent_transitions_serialise_to_the_resolved_winner(prefix, published_flag):
    """A publishes v3 (uncommitted); B archives v2 on the SAME prompt meanwhile.

    B must wait for A, then resolve with A's commit visible -> v3. Without the
    per-prompt lock B resolves v1 from its own snapshot, blocks only on the
    worker row A wrote, and overwrites A's v3 with v1 after A commits.
    """
    pid, w = prefix + "prompt", prefix + "w"
    await _worker(w, V1)
    await _seed(pid, {1: V1})
    await _bind(w, pid)
    await _draft(pid, 2, V2)
    await _publish(pid, 2)
    await _draft(pid, 3, V3)
    await _move(pid, 3, repository.CANARY)
    assert (await _config(w))["system_prompt"] == V2

    a_written, release_a = asyncio.Event(), asyncio.Event()
    b_pid: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    async def tx_a() -> None:
        async with get_pg_connection() as conn, conn.transaction():
            await repository.transition(pid, 3, repository.PUBLISHED, actor=ACTOR, conn=conn)
            a_written.set()
            await release_a.wait()

    async def tx_b() -> None:
        async with get_pg_connection() as conn:
            b_pid.set_result(int(await conn.fetchval("SELECT pg_backend_pid()")))
            async with conn.transaction():
                await repository.transition(pid, 2, repository.ARCHIVED, actor=ACTOR, conn=conn)

    task_a = asyncio.create_task(tx_a())
    await asyncio.wait_for(a_written.wait(), 10)
    task_b = asyncio.create_task(tx_b())
    backend = await asyncio.wait_for(b_pid, 10)

    # Deterministic: wait until B is BLOCKED on a lock while A is still open.
    async def _b_waits_on_lock() -> None:
        while True:
            async with get_pg_connection() as conn:
                kind = await conn.fetchval(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1", backend
                )
            if kind == "Lock":
                return
            await asyncio.sleep(0.02)

    try:
        await asyncio.wait_for(_b_waits_on_lock(), 10)
    finally:
        release_a.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), 20)

    assert await _status(pid, 3) == repository.PUBLISHED
    assert await _status(pid, 2) == repository.ARCHIVED
    resolved = await repository.get_active(pid, highest_published=True)
    assert resolved is not None
    assert resolved["system"] == V3
    assert (await _config(w))["system_prompt"] == resolved["system"]


async def test_legacy_flag_mirrors_nothing(prefix, monkeypatch):
    monkeypatch.setenv("WM_PROMPT_SOURCE", "legacy")
    reset_settings_cache()
    try:
        pid, w = prefix + "prompt", prefix + "w"
        await _worker(w, V1)
        await _seed(pid, {1: V1})
        await _bind(w, pid)
        await _draft(pid, 2, V2)
        await _publish(pid, 2)
        assert (await _config(w))["system_prompt"] == V1
    finally:
        monkeypatch.delenv("WM_PROMPT_SOURCE", raising=False)
        reset_settings_cache()
