"""``tenant_id`` is a real Postgres ``uuid`` on every tenant-scoped table.

Proof for migration ``024_tenant_id_is_a_uuid.sql``.

WHAT THIS HAS TO PROVE, AND WHY EACH PART IS HERE
-------------------------------------------------
The migration is not just an ``ALTER TYPE``. It rewrites live tenant
identifiers, so the tests that matter are the ones that would catch it
*silently doing the wrong thing*:

1. **The column really is ``uuid``** on all 12 tables — the postcondition the
   migration itself raises on, asserted independently of the migration.
2. **No row changed tenant.** Every legacy literal maps 1:1 to a distinct uuid,
   so two rows that were in different tenants before are in different tenants
   after, and two that were together stay together. This is the isolation
   guarantee; without it the flip could quietly merge tenants.
3. **The mapping is the fleet's mapping.** The uuid the SQL derives must equal
   ``uuid.uuid5(TENANT_NAMESPACE, label)`` computed in Python. If SQL and Python
   ever disagree, two services derive different uuids for the same legacy label
   and the fleet stops joining — a defect that would be invisible inside any
   single service.
4. **The label survives.** ``tenant_name`` was a byte-identical copy of
   ``tenant_id`` on all 952,808 production rows; after the flip it must carry
   the legacy label, and must NOT clobber a label an operator already set.
5. **Idempotency**, because ``run_migrations()`` re-runs every file on every pod
   boot.
6. **A non-uuid literal can no longer be stored at all** — the negative control.
   Without it, 1-5 could all pass on a column that had simply been emptied.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_tenant_id_is_a_uuid.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.api.tenant_constants import TENANT_NAMESPACE, derive_tenant_uuid
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

# The 12 tables migration 024 flips. Hardcoded on purpose: deriving this list
# from information_schema would make the test agree with whatever the database
# happens to contain, which is precisely the failure it is meant to catch.
TENANT_TABLES = (
    "workers",
    "worker_states",
    "graphs",
    "graph_states",
    "graph_execution_sessions",
    "node_execution_logs",
    "execution_events",
    "tools",
    "workload",
    "skills",
    "tenant_llm_config",
    "tenant_modality_config",
)

# Two legacy literals that really are in production (measured 2026-08-25:
# `n8n-dev` 88,092 rows, `sales` 18) plus a snowflake id, which is the other
# shape that got in. They must stay in DIFFERENT tenants after the flip.
LEGACY_A = "n8n-dev"
LEGACY_B = "sales"
LEGACY_C = "1769980520185202330"


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    """Module-scoped PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
    require_database_url()
    import os

    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


async def _column_types() -> dict[str, str]:
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT table_name, data_type
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND column_name = 'tenant_id'
              AND table_name = ANY($1::text[])
            """,
            list(TENANT_TABLES),
        )
    return {r["table_name"]: r["data_type"] for r in rows}


async def test_every_tenant_column_is_uuid(pg_pool):
    """1. The end state, asserted independently of the migration's own check."""
    await apply_migrations()

    types = await _column_types()
    assert types, "expected the tenant-scoped tables to exist"
    not_uuid = {t: d for t, d in types.items() if d != "uuid"}
    assert not_uuid == {}, f"tenant_id is still not uuid on: {not_uuid}"


async def test_sql_derivation_matches_python(pg_pool):
    """3. SQL and Python must agree, or the fleet stops joining.

    The migration derives ids with ``uuid_generate_v5`` inside Postgres; every
    other service derives them with ``uuid.uuid5`` in Python. This asserts the
    two are the same function over the same namespace — the single assumption
    that makes one legacy label mean one tenant across services.
    """
    await apply_migrations()

    labels = [LEGACY_A, LEGACY_B, LEGACY_C, "__system_workflow__", "auto"]
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT l AS label, uuid_generate_v5($1::uuid, l)::text AS derived "
            "FROM unnest($2::text[]) AS l",
            str(TENANT_NAMESPACE),
            labels,
        )

    for row in rows:
        expected = str(uuid.uuid5(TENANT_NAMESPACE, row["label"]))
        assert row["derived"] == expected, (
            f"SQL and Python disagree for {row['label']!r}: {row['derived']} != {expected}"
        )
        # And the shared helper agrees with both, so callers cannot drift either.
        assert derive_tenant_uuid(row["label"]) == expected


async def test_remap_is_one_to_one_and_preserves_isolation(pg_pool):
    """2 + 4 + 6. The load-bearing test.

    Seeds rows under three DIFFERENT legacy tenants (plus one that already holds
    a real label), runs the flip, and asserts every row kept its own tenant, got
    its label back, and that a non-uuid can no longer be written at all.
    """
    await apply_migrations()

    marker = uuid.uuid4().hex[:8]
    seeded = {
        f"iso-a-{marker}": (LEGACY_A, LEGACY_A),
        f"iso-b-{marker}": (LEGACY_B, LEGACY_B),
        f"iso-c-{marker}": (LEGACY_C, LEGACY_C),
        # tenant_name already carries a real operator-set label: the migration
        # must NOT overwrite it with the legacy value.
        f"iso-d-{marker}": ("auto", "a-real-human-label"),
    }

    async with get_pg_connection() as conn:
        # The column is uuid by now, so seeding the legacy shape requires going
        # back to text for the duration of the check. This is what production
        # looked like before 024, which is the only state that can exercise it.
        await conn.execute("ALTER TABLE workers ALTER COLUMN tenant_id TYPE varchar(255)")
        try:
            for worker_id, (tenant, label) in seeded.items():
                await conn.execute(
                    "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, "
                    "subject_id, visibility, config) "
                    "VALUES ($1, $2, $3, $4, $3, 'tenant', '{}')",
                    worker_id,
                    worker_id,
                    tenant,
                    label,
                )
        except Exception:
            await conn.execute(
                "ALTER TABLE workers ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid"
            )
            raise

    # Re-run the chain: 024 finds `workers` pending again and repairs it.
    applied, failed = await apply_migrations()
    assert failed == [], f"migration chain reported failures: {failed}"
    assert "024_tenant_id_is_a_uuid.sql" in applied

    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT worker_id, tenant_id::text AS tenant_id, tenant_name FROM workers "
            "WHERE worker_id = ANY($1::text[])",
            list(seeded),
        )
        by_id = {r["worker_id"]: r for r in rows}

        assert len(by_id) == len(seeded), "the flip lost rows"

        # -- 2. one legacy tenant -> one uuid, and DIFFERENT legacy tenants stay
        #       different. This is the isolation guarantee.
        for worker_id, (tenant, _label) in seeded.items():
            assert by_id[worker_id]["tenant_id"] == derive_tenant_uuid(tenant)
        distinct_after = {by_id[w]["tenant_id"] for w in seeded}
        distinct_before = {t for t, _ in seeded.values()}
        assert len(distinct_after) == len(distinct_before), (
            "the flip merged tenants that were distinct before it ran"
        )

        # -- 4. the label survives, and an existing one is not clobbered.
        assert by_id[f"iso-a-{marker}"]["tenant_name"] == LEGACY_A
        assert by_id[f"iso-b-{marker}"]["tenant_name"] == LEGACY_B
        assert by_id[f"iso-d-{marker}"]["tenant_name"] == "a-real-human-label"

        # -- the crosswalk records every remap, so this is reversible.
        cross = await conn.fetch(
            "SELECT legacy_value, tenant_uuid::text AS tenant_uuid, "
            "merge_target::text AS merge_target FROM tenant_id_crosswalk "
            "WHERE legacy_value = ANY($1::text[])",
            [t for t, _ in seeded.values()],
        )
        assert {c["legacy_value"] for c in cross} == distinct_before
        for entry in cross:
            assert entry["tenant_uuid"] == derive_tenant_uuid(entry["legacy_value"])

        # -- `merge_target` names the REGISTERED tenant a legacy label spells, for
        #    the labels the portal realm actually mints. It is the value a future
        #    merge acts on, so a wrong one would move 88,092 production rows into
        #    the wrong tenant — and nothing else in the schema would notice.
        #    `n8n-dev` must name the uuid Keycloak mints for it
        #    (realm-patch-script-configmap.yaml:111-113); a label the realm does
        #    not know must name NOTHING rather than guess.
        by_label = {c["legacy_value"]: c for c in cross}
        assert by_label[LEGACY_A]["merge_target"] == "aaaaaaaa-0000-4000-8000-000000000001"
        assert by_label[LEGACY_B]["merge_target"] is None
        assert by_label[LEGACY_C]["merge_target"] is None
        # ...and it is NOT applied: the rows still sit on their derived uuid.
        assert by_label[LEGACY_A]["tenant_uuid"] != by_label[LEGACY_A]["merge_target"]

        # -- 6. NEGATIVE CONTROL. Without this the assertions above would still
        #       pass on a column that had merely been emptied of dirt.
        with pytest.raises(Exception) as exc_info:
            await conn.execute(
                "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, "
                "subject_id, visibility, config) "
                f"VALUES ('neg-{marker}', 'neg-{marker}', 'n8n-dev', 'n8n-dev', "
                "'s', 'tenant', '{}')"
            )
        assert "uuid" in str(exc_info.value).lower(), (
            f"expected a uuid type error, got: {exc_info.value}"
        )

        await conn.execute("DELETE FROM workers WHERE worker_id = ANY($1::text[])", list(seeded))


async def test_non_canonical_uuid_spellings_converge_not_split(pg_pool):
    """A valid uuid written a different way must NOT become a second tenant.

    ⛔ This test exists because a mutation survived without it, and because two
    adversarial reproductions found the underlying defect in the first draft,
    which classified values with a regex:

    * ``AAAAAAAA-…`` PASSES ``^[0-9a-fA-F]{8}-…{12}$``, so it was treated as
      "already a uuid" and left alone — then ``::uuid`` lowercased it during the
      ALTER and it collided with the canonical row on
      ``unique_tenant_worker_name``. The ALTER died inside the index rebuild and,
      because production runs the migration runner non-strict, the file was
      logged and stepped over on EVERY boot: the flip would silently never have
      happened.
    * ``aaaaaaaad0007000…`` (no dashes) and ``{aaaaaaaa-…}`` (braces) FAIL that
      regex but cast fine, so they were handed a fabricated UUIDv5 and split off
      from the tenant they literally spell — the exact property the migration
      promises it preserves.

    ``pg_input_is_valid(…, 'uuid')`` replaced the regex: it answers the only
    question that matters, which is what the ALTER's own cast will do.
    """
    await apply_migrations()

    canonical = "aaaaaaaa-0000-4000-8000-000000000003"
    marker = uuid.uuid4().hex[:8]
    spellings = {
        f"canon-{marker}": canonical,
        f"upper-{marker}": canonical.upper(),
        f"nodash-{marker}": canonical.replace("-", ""),
        f"braces-{marker}": "{" + canonical + "}",
    }

    async with get_pg_connection() as conn:
        await conn.execute("ALTER TABLE workers ALTER COLUMN tenant_id TYPE varchar(255)")
        try:
            for worker_id, spelling in spellings.items():
                await conn.execute(
                    "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, "
                    "subject_id, visibility, config) "
                    "VALUES ($1, $2, $3, 'x', 's', 'tenant', '{}')",
                    worker_id,
                    worker_id,
                    spelling,
                )
        except Exception:
            await conn.execute(
                "ALTER TABLE workers ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid"
            )
            raise

    _applied, failed = await apply_migrations()
    assert failed == [], f"migration chain reported failures: {failed}"

    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT worker_id, tenant_id::text AS tenant_id FROM workers "
            "WHERE worker_id = ANY($1::text[])",
            list(spellings),
        )
        got = {r["worker_id"]: r["tenant_id"] for r in rows}
        assert len(got) == len(spellings), "the flip lost rows"

        # All four spellings are ONE tenant, and it is the tenant they spell --
        # not a derived uuid, which would mean the migration invented a tenant.
        assert set(got.values()) == {canonical}, (
            f"non-canonical uuid spellings split into separate tenants: {got}"
        )
        # And none of them was routed through the derived-label path.
        cross = await conn.fetch(
            "SELECT legacy_value FROM tenant_id_crosswalk WHERE legacy_value = ANY($1::text[])",
            list(spellings.values()),
        )
        assert cross == [], f"a castable uuid was treated as a legacy label: {cross}"

        await conn.execute("DELETE FROM workers WHERE worker_id = ANY($1::text[])", list(spellings))


async def test_a_spelling_collision_raises_before_anything_is_altered(pg_pool):
    """The pre-flight guard: fail loudly and early, never inside the index rebuild.

    Two rows holding ONE uuid under two spellings, with the same ``worker_name``,
    both collapse onto the same ``(tenant_id, worker_name)`` when ``::uuid``
    normalises them. Without the guard the ALTER dies inside the unique-index
    rebuild with ``could not create unique index`` — which names neither the rows
    nor the remedy — and because production runs the migration runner NON-STRICT
    that error is logged and stepped over on every boot, so the flip silently
    never happens. This asserts the guard fires, names the constraint, and leaves
    the schema untouched.
    """
    await apply_migrations()

    marker = uuid.uuid4().hex[:8]
    canonical = "aaaaaaaa-0000-4000-8000-000000000003"
    shared_name = f"collide-{marker}"

    async with get_pg_connection() as conn:
        await conn.execute("ALTER TABLE workers ALTER COLUMN tenant_id TYPE varchar(255)")
        for worker_id, spelling in (
            (f"c1-{marker}", canonical),
            (f"c2-{marker}", canonical.upper()),
        ):
            await conn.execute(
                "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, "
                "subject_id, visibility, config) "
                "VALUES ($1, $2, $3, 'x', 's', 'tenant', '{}')",
                worker_id,
                shared_name,
                spelling,
            )

    try:
        with pytest.raises(Exception) as exc_info:
            await apply_migrations()

        message = str(exc_info.value)
        assert "024 cannot flip" in message, message
        assert "unique_tenant_worker_name" in message, message
        # The remedy has to be in the message: an operator reading a pod log is
        # the only audience this error ever gets.
        assert "tenant_id::uuid::text" in message, message

        # Nothing was altered — the guard runs before the first ALTER.
        types = await _column_types()
        assert types["workers"] != "uuid", "the guard fired only AFTER altering the table"
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workers WHERE worker_name = $1", shared_name)
        await apply_migrations()

    assert (await _column_types())["workers"] == "uuid"


async def test_flip_is_idempotent(pg_pool):
    """5. ``run_migrations()`` re-runs every file on every pod boot."""
    await apply_migrations()
    before = await _column_types()

    _applied, failed = await apply_migrations()
    assert failed == [], f"second run reported failures: {failed}"

    after = await _column_types()
    assert after == before
    assert all(d == "uuid" for d in after.values())
