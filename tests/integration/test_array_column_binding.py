"""The two `text[]` columns must be declared AND bound as arrays, not jsonb.

Why this exists
---------------
A full `pg_dump -s` diff of production (`worker_management` against its
production Postgres) against `postgres_schema.sql` on 2026-08-10
found 56 column-type divergences. Exactly two of them are the jsonb-vs-text[]
class, and they are the only class where **production is right and the file is
the stale artifact**:

    tools.tags     : file `JSONB NOT NULL`  ->  prod `text[] NOT NULL DEFAULT '{}'::text[]`
    skills.domains : file `JSONB NOT NULL`  ->  prod `text[] NOT NULL DEFAULT '{}'::text[]`

`tools.tags` was already known — `migrations/002_fix_tool_tags.sql` documents
that it raised `malformed array literal: "[]"` in production on every pod boot.
`skills.domains` is the same bug one table over, and it was invisible because
CI bootstraps from `postgres_schema.sql`, so CI's `domains` column was `jsonb`
and happily accepted the `Jsonb`-wrapped list that production rejects.

The write path is the load-bearing half. `db_config._array_param_indices`
disambiguates a bare Python `list` ONLY by looking for an explicit array cast
on the placeholder; with no cast the list is wrapped in `Jsonb()` and postgres
answers::

    ERROR: column "domains" is of type text[] but expression is of type jsonb

so `SkillsDB.add_skill` could never have written a row to the real database.

Note what these tests do NOT do: they do not touch the other 54 type
divergences, the 46 missing indexes or the 24 missing constraints. On those,
production is the side that drifted, and asserting the current production shape
would codify the damage.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_array_column_binding.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.skills_db import SkillsDB

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


@pytest.mark.parametrize(
    ("table", "column"),
    [("tools", "tags"), ("skills", "domains")],
)
async def test_declared_column_is_a_text_array_like_production(pg_pool, table, column):
    """The bootstrapped schema must have production's type, or CI is blind.

    This is the assertion that makes the write test below meaningful: if the CI
    column were still `jsonb`, a `Jsonb`-wrapped bind would pass here and fail
    in production — exactly the gap that hid this bug.
    """
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
            """,
            table,
            column,
        )
    assert row is not None, f"{table}.{column} does not exist in the bootstrapped schema"
    assert (row["data_type"], row["udt_name"]) == ("ARRAY", "_text"), (
        f"{table}.{column} is {row['data_type']}/{row['udt_name']}, but production is "
        f"text[] NOT NULL DEFAULT '{{}}'::text[]. Declaring jsonb here makes CI accept "
        f"a Jsonb-wrapped bind that the real database rejects."
    )


async def test_add_skill_writes_a_real_array_and_reads_it_back(pg_pool):
    """The regression: `SkillsDB.add_skill` must survive a `text[]` column.

    Without the `::text[]` casts in skills_db.py this raises
    ``DatatypeMismatch: column "domains" is of type text[] but expression is of
    type jsonb``.
    """
    worker_id = str(uuid.uuid4())
    tenant_id = tenant_uuid(f"t-{uuid.uuid4().hex[:12]}")
    skill_name = f"s-{uuid.uuid4().hex[:8]}"
    domains = ["nlp", "knowledge-graph", "o'quote"]  # embedded quote: real array quoting

    # The bootstrapped schema declares `fk_skill_worker` (production has no
    # foreign keys at all — one of the 24 constraint divergences), so the parent
    # row is required here even though it would not be in prod.
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id)
            -- `$3::uuid` is required now that `tenant_id` is a uuid column:
            -- the same bind parameter also feeds varchar columns here, and
            -- Postgres refuses to deduce two types for one parameter
            -- (`AmbiguousParameter: inconsistent types deduced for $3`).
            VALUES ($1, $2, $3::uuid, $3, $4)
            """,
            worker_id,
            f"w-{uuid.uuid4().hex[:8]}",
            tenant_id,
            f"u-{uuid.uuid4().hex[:8]}",
        )

    db = SkillsDB()
    try:
        skill = await db.add_skill(
            tenant_id=tenant_id,
            subject_id=f"u-{uuid.uuid4().hex[:8]}",
            worker_id=worker_id,
            skill_name=skill_name,
            domains=domains,
            skill_level="expert",
        )
        assert skill.domains == domains

        # Read back through the raw driver: proves the value landed as a native
        # PG array (element-wise indexable), not as a jsonb blob or a string.
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT domains, array_length(domains, 1) AS n, domains[1] AS first
                FROM skills WHERE tenant_id = $1 AND worker_id = $2 AND skill_name = $3
                """,
                tenant_id,
                worker_id,
                skill_name,
            )
        assert row is not None
        assert list(row["domains"]) == domains
        assert row["n"] == len(domains)
        assert row["first"] == "nlp"

        # The UPDATE branch of the upsert carries its own cast — cover it too.
        updated = ["reasoning"]
        await db.add_skill(
            tenant_id=tenant_id,
            subject_id=f"u-{uuid.uuid4().hex[:8]}",
            worker_id=worker_id,
            skill_name=skill_name,
            domains=updated,
            skill_level="advanced",
        )
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                "SELECT domains FROM skills WHERE tenant_id = $1 AND worker_id = $2 "
                "AND skill_name = $3",
                tenant_id,
                worker_id,
                skill_name,
            )
        assert row is not None
        assert list(row["domains"]) == updated
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM skills WHERE tenant_id = $1", tenant_id)
            await conn.execute("DELETE FROM workers WHERE tenant_id = $1", tenant_id)
