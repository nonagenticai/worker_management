"""Real-Postgres proof that ``list_workers``/``count_workers`` actually filter.

The unit suite cannot prove this. ``tests/db_test_helper.py``'s in-memory fake
parses only ``col = $N`` predicates out of a WHERE clause and silently ignores
everything else, so a ``starts_with(...)`` filter would be dropped by the fake
and a prefix test written against it would pass whether the SQL filters or not —
the same shape of instrument as the defect being fixed. These assertions
therefore run against a real PostgreSQL 18 (CI's ephemeral ``postgres:18``
sidecar, via the archetype's ``tests/integration -m pg`` pass).

What was measured on the live service 2026-09-23, and what these pin:

* ``GET /workers``, ``?prefix=aa``, ``?prefix=zzz_nonexistent_prefix`` and
  ``?owner_tenant_id=<zero uuid>`` returned four identical md5s — the filter was
  not applied at all.
* ``?limit=500`` returned 165 rows against ``?limit=100``'s 100, i.e. the default
  dropped 65 workers with no field saying so. Cross-checked: the ``workers``
  table holds exactly 165 rows for that tenant.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_list_workers_prefix_filter.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.worker_registry_db import WorkerRegistryDB

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

# Two cohorts that share no prefix, plus a decoy whose name CONTAINS the cohort
# prefix without starting with it. The decoy is what separates a real prefix
# match from a substring match.
COHORT_A = ["fr_planner_alpha", "fr_planner_beta", "fr_planner_gamma"]
COHORT_B = ["kag_ingest_one", "kag_ingest_two"]
# "legacy_…" contains the cohort prefix without starting with it (substring vs
# prefix). "kagXingest_three" differs from the prefix "kag_ingest_" in exactly
# one position, where the prefix has "_" — under LIKE that "_" is a wildcard and
# would match the "X", so this row is the discriminator between a LIKE-based
# implementation and a literal one.
DECOY = ["legacy_fr_planner_copy", "kagXingest_three"]
ALL_NAMES = COHORT_A + COHORT_B + DECOY


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
async def seeded_tenant(pg_pool):
    """A throwaway tenant holding ALL_NAMES; drops its rows on teardown."""
    tenant_id = tenant_uuid(f"int-listfilter-{uuid.uuid4().hex[:12]}")
    db = WorkerRegistryDB()
    for i, name in enumerate(ALL_NAMES):
        await db.register_worker(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            worker_name=name,
            db_path="/tmp/listfilter.db",
            is_agent=False,
            config={},
            worker_id=f"lf-{i}-{uuid.uuid4().hex[:8]}",
        )
    try:
        yield tenant_id, db
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workers WHERE tenant_id = $1", tenant_id)


class TestPrefixIsARealWhere:
    async def test_seed_is_what_the_test_thinks_it_is(self, seeded_tenant):
        """Positive control. Without this, every assertion below could be passing
        on an empty table — an unfiltered list and a filtered list of a table that
        holds nothing are both ``[]``."""
        tenant_id, db = seeded_tenant
        everything = await db.list_workers(tenant_id=tenant_id, subject_id=tenant_id)
        assert sorted(everything) == sorted(ALL_NAMES)
        assert len(everything) == 7
        assert await db.count_workers(tenant_id=tenant_id, subject_id=tenant_id) == 7

    async def test_prefix_returns_only_the_cohort(self, seeded_tenant):
        tenant_id, db = seeded_tenant

        cohort = await db.list_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
        )

        assert sorted(cohort) == sorted(COHORT_A)
        assert len(cohort) == 3
        # The measured defect, stated as an exclusion: an unfiltered read of this
        # tenant returns 6 names, so a filter that returns 6 is not a filter.
        for other in COHORT_B:
            assert other not in cohort

    async def test_prefix_is_a_prefix_not_a_substring(self, seeded_tenant):
        """``legacy_fr_planner_copy`` contains the prefix but does not start with
        it, so it must not be in the cohort."""
        tenant_id, db = seeded_tenant

        cohort = await db.list_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
        )

        assert "legacy_fr_planner_copy" not in cohort

    async def test_underscore_is_literal_not_a_wildcard(self, seeded_tenant):
        """``_`` in a caller's prefix must mean an underscore, not "any char".

        A ``LIKE``-based implementation returns 3 here (the two real
        ``kag_ingest_*`` rows plus ``kagXingest_three``, whose ``X`` the ``_``
        would match); a literal one returns exactly the 2 real rows. Worker
        names are full of underscores — the cohort op derives prefixes like
        ``fr_planner_`` — so this is not a corner case, it is every call."""
        tenant_id, db = seeded_tenant

        cohort = await db.list_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="kag_ingest_"
        )

        assert sorted(cohort) == sorted(COHORT_B)
        assert len(cohort) == 2
        assert "kagXingest_three" not in cohort
        assert (
            await db.count_workers(tenant_id=tenant_id, subject_id=tenant_id, prefix="kag_ingest_")
            == 2
        )

    async def test_percent_in_a_prefix_is_literal(self, seeded_tenant):
        """Under ``LIKE`` a bare ``%`` matches everything. Here it must match
        nothing, because no seeded name contains one."""
        tenant_id, db = seeded_tenant

        assert await db.list_workers(tenant_id=tenant_id, subject_id=tenant_id, prefix="%") == []
        assert await db.count_workers(tenant_id=tenant_id, subject_id=tenant_id, prefix="%") == 0

    async def test_unmatched_prefix_returns_empty_not_everything(self, seeded_tenant):
        tenant_id, db = seeded_tenant

        assert (
            await db.list_workers(
                tenant_id=tenant_id, subject_id=tenant_id, prefix="zzz_nonexistent_prefix"
            )
            == []
        )

    async def test_prefix_is_case_insensitive_like_get_worker(self, seeded_tenant):
        tenant_id, db = seeded_tenant

        cohort = await db.list_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="FR_PLANNER_"
        )

        assert sorted(cohort) == sorted(COHORT_A)


class TestCountDescribesTheSamePopulationAsThePage:
    async def test_count_matches_the_filtered_page(self, seeded_tenant):
        tenant_id, db = seeded_tenant

        total = await db.count_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
        )
        page = await db.list_workers(
            tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
        )

        assert total == 3
        assert total == len(page)

    async def test_count_exceeds_a_truncated_page_by_the_right_amount(self, seeded_tenant):
        """The 165-vs-100 shape, at this fixture's scale: 7 exist, a page of 2
        returns 2, and the total says 7 so the caller can tell."""
        tenant_id, db = seeded_tenant

        total = await db.count_workers(tenant_id=tenant_id, subject_id=tenant_id)
        page = await db.list_workers(tenant_id=tenant_id, subject_id=tenant_id, limit=2)

        assert total == 7
        assert len(page) == 2
        assert total - len(page) == 5

    async def test_count_ignores_offset_and_limit(self, seeded_tenant):
        tenant_id, db = seeded_tenant

        page = await db.list_workers(tenant_id=tenant_id, subject_id=tenant_id, limit=2, offset=5)

        assert len(page) == 2
        assert await db.count_workers(tenant_id=tenant_id, subject_id=tenant_id) == 7

    async def test_another_tenants_rows_are_in_neither(self, seeded_tenant):
        """A prefix filter must narrow within the tenant scope, never widen it."""
        tenant_id, db = seeded_tenant
        other = tenant_uuid(f"int-listfilter-other-{uuid.uuid4().hex[:12]}")
        await db.register_worker(
            tenant_id=other,
            subject_id=other,
            worker_name="fr_planner_intruder",
            db_path="/tmp/listfilter.db",
            is_agent=False,
            config={},
            worker_id=f"lf-x-{uuid.uuid4().hex[:8]}",
        )
        try:
            cohort = await db.list_workers(
                tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
            )
            assert "fr_planner_intruder" not in cohort
            assert len(cohort) == 3
            assert (
                await db.count_workers(
                    tenant_id=tenant_id, subject_id=tenant_id, prefix="fr_planner_"
                )
                == 3
            )
        finally:
            async with get_pg_connection() as conn:
                await conn.execute("DELETE FROM workers WHERE tenant_id = $1", other)
