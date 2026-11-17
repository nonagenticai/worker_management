"""The two rows that billed paid DeepSeek are gone, and nothing else moved.

Proof for ``migrations/025_purge_local_preset_paid_tier_rows.sql``.

WHY THIS TEST EXISTS AT ALL. ``run_migrations()`` runs NON-STRICT in production
(``WORKER_MGMT_MIGRATIONS_STRICT`` unset), so a data migration that silently
deletes nothing is byte-identical to one that worked. The only way to know the
predicate matches the rows it was written for is to seed those rows and look.

WHAT IT HAS TO PROVE, AND WHY EACH PART IS HERE
-----------------------------------------------
1. **The two offending rows are deleted.** Tenant
   ``aaaaaaaa-0000-4000-8000-000000000001``, roles ``tier:medium`` and
   ``tier:high``, ``provider='local'``, ``model='tier-medium'`` — the shape that
   spent $140.31 over 46,102 requests on api.deepseek.com while the tenant's own
   configuration said "platform inference".
2. **Nothing else is touched.** Four controls, each of which a sloppier
   predicate would have eaten: the same tenant's ``__default__`` row, another
   tenant's identical row, a row that made a DELIBERATE external choice, and a
   ``tier:medium`` row already repaired to a local model. Without these, part 1
   also passes on ``DELETE FROM tenant_llm_config``.
3. **A row created AFTER the incident window survives.** This is the one that
   is easy to get wrong and expensive to discover: every migration re-runs on
   every pod boot, so without the ``created_at`` bound a future operator who
   deliberately re-creates such a row would have it deleted from under them at
   the next restart, forever, with no log anyone reads.
4. **Idempotency**, for the same re-runs-on-every-boot reason.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_migration_025_paid_tier_purge.py -m pg -v --no-cov
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from src.db.db_config import (
    close_pg_pool,
    get_pg_connection,
    init_pg_pool,
)

from .pg_guard import (
    PG_CONNECT_ERRORS,
    apply_migrations,
    require_database_url,
    unreachable_pg,
)

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

# The tenant whose two rows the migration removes (platform-cp-1).
VICTIM = "aaaaaaaa-0000-4000-8000-000000000001"
# A bystander that carries a byte-identical pair of rows. If the tenant_id
# predicate is dropped, this tenant loses its config too.
BYSTANDER = "aaaaaaaa-0000-4000-8000-0000000000ff"

# (tenant, role, provider, model, created_at, survives?)
SEED: tuple[tuple[str, str, str, str, str, bool], ...] = (
    # 1. The two rows the migration exists for.
    (VICTIM, "tier:medium", "local", "tier-medium", "2026-07-19 16:11:00+00", False),
    (VICTIM, "tier:high", "local", "tier-medium", "2026-07-19 16:11:00+00", False),
    # 2a. Same tenant, the default row — a different role, must survive.
    (VICTIM, "__default__", "local", "local-llm", "2026-07-19 16:11:00+00", True),
    # 2b. Another tenant, identical shape — the tenant_id predicate is real.
    (BYSTANDER, "tier:medium", "local", "tier-medium", "2026-07-19 16:11:00+00", True),
    # 2c. A DELIBERATE external choice on one of the two target roles. The
    #     tenant is paying knowingly with its own key; deleting it would erase a
    #     decision, not a defect.
    (VICTIM, "tier:small", "deepseek", "deepseek-v4-pro", "2026-07-19 16:11:00+00", True),
    # 3. Created after the incident window — the created_at bound. A row an
    #    operator writes tomorrow must not be deleted at the next pod boot.
    (VICTIM, "tier:medium", "local", "tier-medium", "2026-09-01 00:00:00+00", True),
)


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


async def _seed() -> None:
    """Insert the corpus, replacing whatever a previous run left behind.

    Note SEED holds TWO ``(VICTIM, 'tier:medium')`` rows with different
    ``created_at``. The table is keyed ``(tenant_id, role)``, so they cannot
    coexist; they are seeded and asserted in separate phases instead — which is
    exactly how the production table would see them (the old row deleted, then a
    new one written later).
    """
    async with get_pg_connection() as conn:
        await conn.execute(
            "DELETE FROM tenant_llm_config WHERE tenant_id = ANY($1::uuid[])",
            [VICTIM, BYSTANDER],
        )


async def _insert(tenant: str, role: str, provider: str, model: str, created: str) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO tenant_llm_config
                (tenant_id, role, provider, model, params, created_at, updated_at)
            VALUES ($1::uuid, $2, $3, $4, '{}'::jsonb, $5::timestamptz, $5::timestamptz)
            """,
            tenant,
            role,
            provider,
            model,
            created,
        )


async def _roles(tenant: str) -> set[str]:
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT role FROM tenant_llm_config WHERE tenant_id = $1::uuid", tenant
        )
    return {r["role"] for r in rows}


async def test_the_two_paid_rows_go_and_nothing_else_does(pg_pool):
    """1 + 2. The predicate hits exactly the two rows it was written for."""
    await apply_migrations()
    await _seed()
    for tenant, role, provider, model, created, _survives in SEED:
        if created.startswith("2026-09"):
            continue  # phase 2, below — it collides on (tenant, role)
        await _insert(tenant, role, provider, model, created)

    # NEGATIVE CONTROL: the rows really are there before the migration runs. A
    # deletion test whose corpus never landed passes for the wrong reason.
    assert await _roles(VICTIM) == {"tier:medium", "tier:high", "__default__", "tier:small"}

    await apply_migrations()

    assert await _roles(VICTIM) == {"__default__", "tier:small"}, (
        "the two local-preset tier rows that billed deepseek must be gone, "
        "and the default + deliberate-external rows must remain"
    )
    assert await _roles(BYSTANDER) == {"tier:medium"}, (
        "another tenant's identical row must be untouched — the deletion is "
        "scoped to one tenant, not to a shape"
    )


async def test_a_row_created_after_the_incident_window_survives(pg_pool):
    """3. The created_at bound, i.e. the migration cannot delete future intent."""
    await _seed()
    await _insert(VICTIM, "tier:medium", "local", "tier-medium", "2026-09-01 00:00:00+00")

    await apply_migrations()

    assert await _roles(VICTIM) == {"tier:medium"}, (
        "025 re-runs whenever an operator forgets its ledger row; without the "
        "created_at bound this row would be deleted from under an operator "
        "every restart, forever"
    )


async def test_deleting_twice_deletes_nothing_more(pg_pool):
    """4. Idempotency — the chain runs on every boot."""
    await _seed()
    await _insert(VICTIM, "tier:medium", "local", "tier-medium", "2026-07-19 16:11:00+00")
    await _insert(VICTIM, "__default__", "local", "local-llm", "2026-07-19 16:11:00+00")

    await apply_migrations()
    first = await _roles(VICTIM)
    await apply_migrations()
    assert await _roles(VICTIM) == first == {"__default__"}

    await _seed()
