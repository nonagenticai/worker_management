"""WM-5: tenant_llm_config_audit retention, against real PostgreSQL 18.

``prune_audit_once`` deletes rows older than the window — every tenant, in
bounded batches — and keeps everything younger. MUTATION: drop the
``created_at < cutoff`` predicate ⇒ the young rows go ⇒ red; never delete
(pre-WM-5) ⇒ the old rows stay ⇒ red.

Run locally:
    DATABASE_URL=postgresql+psycopg://… uv run pytest \
        tests/integration/test_tenant_llm_audit_retention.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import PG_CONNECT_ERRORS, apply_migrations, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.pg

TENANT_A = "0195f48b-d000-7000-8000-0000000005aa"
TENANT_B = "0195f48b-d000-7000-8000-0000000005bb"
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)


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
        await c.execute(
            "DELETE FROM tenant_llm_config_audit WHERE tenant_id = ANY($1::uuid[])",
            [TENANT_A, TENANT_B],
        )


async def _row(tenant: str, age_days: int, action: str = "test") -> None:
    async with get_pg_connection() as c:
        await c.execute(
            "INSERT INTO tenant_llm_config_audit "
            "(tenant_id, role, action, actor_kind, provider, key_id, created_at) "
            "VALUES ($1, '__default__', $2, 'service', 'openai', 'none', $3)",
            tenant,
            action,
            NOW - timedelta(days=age_days),
        )


async def _ages() -> dict[str, list[int]]:
    async with get_pg_connection() as c:
        rows = await c.fetch(
            "SELECT tenant_id::text AS t, created_at FROM tenant_llm_config_audit "
            "WHERE tenant_id = ANY($1::uuid[]) ORDER BY created_at",
            [TENANT_A, TENANT_B],
        )
    out: dict[str, list[int]] = {}
    for r in rows:
        out.setdefault(r["t"], []).append((NOW - r["created_at"]).days)
    return out


@pytest.mark.asyncio
async def test_prunes_only_rows_older_than_the_window_across_tenants(pg_pool):
    from src.services.tenant_llm_audit_retention import prune_audit_once

    for age in (900, 401, 399, 10, 0):
        await _row(TENANT_A, age)
    for age in (500, 30):
        await _row(TENANT_B, age, action="set_key")
    result = await prune_audit_once(retention_days=400, now=NOW)
    assert result.deleted >= 3  # other tests' rows are young; only ours are old
    assert result.exhausted is True
    assert await _ages() == {TENANT_A: [399, 10, 0], TENANT_B: [30]}
    again = await prune_audit_once(retention_days=400, now=NOW)
    assert again.deleted == 0


@pytest.mark.asyncio
async def test_batches_are_bounded_and_a_backlog_carries_over(pg_pool):
    from src.services.tenant_llm_audit_retention import prune_audit_once

    for age in (700, 600, 500, 450, 1):
        await _row(TENANT_A, age)
    first = await prune_audit_once(retention_days=400, batch_size=1, max_batches=2, now=NOW)
    assert (first.deleted, first.batches, first.exhausted) == (2, 2, False)
    assert len((await _ages())[TENANT_A]) == 3
    rest = await prune_audit_once(retention_days=400, batch_size=10, now=NOW)
    assert rest.exhausted is True
    assert (await _ages())[TENANT_A] == [1]


@pytest.mark.asyncio
async def test_default_window_is_400_days(pg_pool):
    from src.services.tenant_llm_audit_retention import prune_audit_once
    from src.settings import get_settings

    assert get_settings().tenant_llm_audit_retention_days == 400
    await _row(TENANT_A, 401)
    await _row(TENANT_A, 399)
    await prune_audit_once(now=NOW)
    assert (await _ages())[TENANT_A] == [399]
