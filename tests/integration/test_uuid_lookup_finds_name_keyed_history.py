"""A lookup BY UUID must find the history written BEFORE the worker registered.

Why this exists
---------------
``WorkerStateManager.resolve_worker_id`` deliberately falls back to using the
worker_name AS the worker_id for anything not yet in the registry ("legacy
compatibility"). When such a worker later registers and is minted a UUID, its
earlier rows keep the NAME in ``worker_states.worker_id``.

``get_worker_states_by_id`` used to bind the SAME caller value to both columns
(``worker_id = $3 OR worker_name = $4``). A lookup BY NAME therefore found
everything — ``worker_name`` is populated on 100% of rows — but a lookup BY
UUID, which is what every caller uses once the worker IS registered, matched
neither column on the older rows. Half the worker's history vanished from the
read with no error.

``_rekey_name_scoped_states`` repairs the split at registration, but only where
the historical rows share the registering worker's tenant AND subject: its
UPDATE carries the same ``scope_clause``. Measured against production
2026-08-14, that covers **24 of the 396** split rows — one worker,
``icp-analysis``. The other 372 were written under a tenant/subject that
matches no registered worker of that name (56 of them under the all-zeros
sentinel tenant ``00000000-0000-4000-8000-000000000001``), so no future
registration can ever reach them. Re-keying alone cannot close this; the read
has to tolerate both shapes.

The isolation test below is the load-bearing half. The fix adds a disjunct to a
tenant-scoped, security-relevant query, so "it returns more rows" is only
correct if it returns strictly more rows *from the caller's own tenant*.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_uuid_lookup_finds_name_keyed_history.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.workers_state_db import StateDatabase

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def pg_pool():
    require_database_url()
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


async def _register_worker(*, tenant_id: str, worker_id: str, worker_name: str) -> None:
    """Insert the registry row directly.

    The production resolution path is a plain lookup on ``workers``; going
    through the full registration API would drag in config/schema validation
    that has nothing to do with what is under test here.
    """
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO workers (worker_id, worker_name, tenant_id, subject_id,
                                 tenant_name, visibility, is_agent, status)
            -- `$3::uuid` is required now that `tenant_id` is a uuid column:
            -- the same bind parameter also feeds varchar columns here, and
            -- Postgres refuses to deduce two types for one parameter
            -- (`AmbiguousParameter: inconsistent types deduced for $3`).
            VALUES ($1, $2, $3::uuid, $3, $3, 'tenant', false, 'active')
            """,
            worker_id,
            worker_name,
            tenant_id,
        )


async def _split_history(db: StateDatabase, *, tenant_id: str, worker_id: str, worker_name: str):
    """Write one pre-registration (name-keyed) row and one post-registration row."""
    await db.add_worker_state(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_id=worker_name,  # the legacy fallback: NAME stored as the id
        worker_name=worker_name,
        is_agent=False,
        state_type="processing",
        worker_state={"half": "before-registration"},
    )
    await db.add_worker_state(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_id=worker_id,  # after registration: the minted UUID
        worker_name=worker_name,
        is_agent=False,
        state_type="completed",
        worker_state={"half": "after-registration"},
    )


async def test_lookup_by_uuid_returns_both_halves_of_the_history(pg_pool):
    """The regression. Before the fix this returned 1 row, silently."""
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    worker_name = f"w-{uuid.uuid4().hex[:8]}"
    worker_id = str(uuid.uuid4())
    db = StateDatabase()

    await _register_worker(tenant_id=tenant_id, worker_id=worker_id, worker_name=worker_name)
    await _split_history(db, tenant_id=tenant_id, worker_id=worker_id, worker_name=worker_name)

    rows = await db.get_worker_states_by_id(
        tenant_id=tenant_id, subject_id=tenant_id, worker_id=worker_id, limit=10
    )

    halves = sorted(r["worker_state"]["half"] for r in rows)
    assert halves == ["after-registration", "before-registration"], (
        f"lookup by UUID returned {halves}. The name-keyed half of this worker's "
        f"history is invisible to every caller that holds the UUID."
    )


async def test_lookup_by_name_still_returns_both_halves(pg_pool):
    """Guard the path that already worked — the fix must not narrow it."""
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    worker_name = f"w-{uuid.uuid4().hex[:8]}"
    worker_id = str(uuid.uuid4())
    db = StateDatabase()

    await _register_worker(tenant_id=tenant_id, worker_id=worker_id, worker_name=worker_name)
    await _split_history(db, tenant_id=tenant_id, worker_id=worker_id, worker_name=worker_name)

    rows = await db.get_worker_states_by_id(
        tenant_id=tenant_id, subject_id=tenant_id, worker_id=worker_name, limit=10
    )

    assert len(rows) == 2, f"lookup by name regressed to {len(rows)} row(s)"


async def test_name_resolution_does_not_cross_tenants(pg_pool):
    """The isolation guard: resolving a name must not widen the tenant scope.

    Two tenants register the SAME worker_name — legitimate and common, since
    ``unique_tenant_worker_name`` is unique per (tenant_id, worker_name), not
    globally. Tenant A asks for its own UUID. It must see only its own rows,
    never tenant B's name-keyed history, which shares the resolved name.
    """
    shared_name = f"w-{uuid.uuid4().hex[:8]}"
    tenant_a = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    tenant_b = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    uuid_a = str(uuid.uuid4())
    uuid_b = str(uuid.uuid4())
    db = StateDatabase()

    await _register_worker(tenant_id=tenant_a, worker_id=uuid_a, worker_name=shared_name)
    await _register_worker(tenant_id=tenant_b, worker_id=uuid_b, worker_name=shared_name)
    await _split_history(db, tenant_id=tenant_a, worker_id=uuid_a, worker_name=shared_name)
    await _split_history(db, tenant_id=tenant_b, worker_id=uuid_b, worker_name=shared_name)

    rows = await db.get_worker_states_by_id(
        tenant_id=tenant_a, subject_id=tenant_a, worker_id=uuid_a, limit=50
    )

    assert len(rows) == 2, f"expected tenant A's 2 rows, got {len(rows)}"
    assert {r["tenant_id"] for r in rows} == {tenant_a}, (
        "name resolution leaked rows across tenants — the added disjunct must "
        "stay inside scope_clause, never beside it"
    )


async def test_unregistered_name_lookup_is_unaffected(pg_pool):
    """A worker that never registered has no registry row to resolve.

    The subquery yields NULL and the new disjunct is simply never true; the
    caller still finds its history by name.
    """
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    worker_name = f"w-{uuid.uuid4().hex[:8]}"
    db = StateDatabase()

    await db.add_worker_state(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_id=worker_name,
        worker_name=worker_name,
        is_agent=False,
        state_type="processing",
        worker_state={"half": "never-registered"},
    )

    rows = await db.get_worker_states_by_id(
        tenant_id=tenant_id, subject_id=tenant_id, worker_id=worker_name, limit=10
    )

    assert len(rows) == 1
    assert rows[0]["worker_state"]["half"] == "never-registered"
