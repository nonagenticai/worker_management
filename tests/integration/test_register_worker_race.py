"""Real-Postgres proof of ``WorkerRegistryDB.register_worker`` idempotency.

``register_worker`` does a check-then-insert against the
``unique_tenant_worker_name (tenant_id, worker_name)`` constraint. Two hardened
behaviors are pinned here against a live Postgres 18:

* Concurrent same-owner creates of one worker_name -> exactly one row, every
  caller sees the same worker_id, no ``UniqueViolation`` surfaces (get-or-create
  race recovery).
* A second subject creating a worker_name already owned as another subject's
  ``visibility='user'`` worker -> :class:`WorkerNameConflictError`, the original
  row untouched (the lookup identity is narrower than the constraint identity).

Marked ``@pytest.mark.pg`` and gated on ``DATABASE_URL`` exactly like the other
pure-Postgres integration tests. CI applies ``postgres_schema.sql`` via
``scripts/init_db.py`` before this runs.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_register_worker_race.py -m pg -v --no-cov
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.worker_registry_db import WorkerNameConflictError, WorkerRegistryDB

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    """Module-scoped PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture
async def tenant(pg_pool):
    """A throwaway tenant id; cleans up its workers on teardown."""
    tenant_id = tenant_uuid(f"int-wreg-{uuid.uuid4().hex[:12]}")
    try:
        yield tenant_id
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workers WHERE tenant_id = $1", tenant_id)


async def _count_rows(tenant_id: str, worker_name: str) -> int:
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS n FROM workers WHERE tenant_id = $1 AND worker_name = $2",
            tenant_id,
            worker_name,
        )
    assert row is not None
    return int(row["n"])


class TestRegisterWorkerRace:
    async def test_concurrent_same_owner_creates_one_row_same_id(self, tenant):
        """5 concurrent identical registers -> one row, one shared worker_id, no 500."""
        db = WorkerRegistryDB()
        worker_name = "race-worker"

        async def register():
            return await db.register_worker(
                tenant_id=tenant,
                subject_id=tenant,
                worker_name=worker_name,
                db_path="/tmp/x",
                is_agent=False,
                config={"model": "gpt-4"},
                visibility="tenant",
            )

        # A barrier makes all coroutines reach the INSERT window together,
        # maximizing the chance every-but-one loses the create race.
        results = await asyncio.gather(*(register() for _ in range(5)))

        worker_ids = {str(r["worker_id"]) for r in results}
        assert len(worker_ids) == 1, f"callers disagreed on worker_id: {worker_ids}"
        assert await _count_rows(tenant, worker_name) == 1

    async def test_sequential_reregister_is_idempotent(self, tenant):
        """Re-registering the same name returns the same worker_id and keeps one row."""
        db = WorkerRegistryDB()
        worker_name = "idem-worker"

        first = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name=worker_name,
            db_path="/tmp/a",
            is_agent=False,
            config={"v": 1},
            visibility="tenant",
        )
        second = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name=worker_name,
            db_path="/tmp/b",
            is_agent=True,
            config={"v": 2},
            visibility="tenant",
        )

        assert str(second["worker_id"]) == str(first["worker_id"])
        assert await _count_rows(tenant, worker_name) == 1

    async def test_cross_subject_user_worker_raises_conflict(self, tenant):
        """A worker_name owned as subject A's private worker conflicts for subject B.

        The initial scoped SELECT for B misses A's ``visibility='user'`` row, but
        the INSERT trips ``unique_tenant_worker_name``. Recovery re-SELECT (still
        B-scoped) finds nothing -> WorkerNameConflictError, and A's row is intact.
        """
        db = WorkerRegistryDB()
        worker_name = "shared-name"
        alice, bob = "alice-subj", "bob-subj"

        alice_row = await db.register_worker(
            tenant_id=tenant,
            subject_id=alice,
            worker_name=worker_name,
            db_path="/tmp/alice",
            is_agent=False,
            config={"owner": "alice"},
            visibility="user",
        )

        with pytest.raises(WorkerNameConflictError) as exc_info:
            await db.register_worker(
                tenant_id=tenant,
                subject_id=bob,
                worker_name=worker_name,
                db_path="/tmp/bob",
                is_agent=False,
                config={"owner": "bob"},
                visibility="user",
            )
        assert exc_info.value.worker_name == worker_name

        # Exactly one row remains, and it is still Alice's.
        assert await _count_rows(tenant, worker_name) == 1
        surviving = await db.get_worker(tenant_id=tenant, subject_id=alice, worker_name=worker_name)
        assert surviving is not None
        assert str(surviving["worker_id"]) == str(alice_row["worker_id"])
        assert surviving["config"] == {"owner": "alice"}
        # Bob cannot see it at all under his own scope.
        assert (
            await db.get_worker(tenant_id=tenant, subject_id=bob, worker_name=worker_name) is None
        )
