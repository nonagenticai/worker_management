"""G-15 repository writes against real Postgres (the unit tests mock the repository).

Pins the audit finding that ONLY the card which released a canary may promote
it, while discard stays open to any card (a new release must be able to clear
a foreign, rejected canary first).
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from src.api.workflow_routing import repository
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import PG_CONNECT_ERRORS, apply_migrations, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

ACTIVE = "AbCdEfGhIjKlMnOp"
CANARY = "QrStUvWxYzAbCdEf"
TENANT = "aaaaaaaa-0000-4000-8000-000000000002"


@pytest_asyncio.fixture
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
async def armed(pg_pool) -> AsyncIterator[str]:
    """A seeded pointer with a canary released by card 'card-a'."""
    key = f"activity:AA Activity - G15 Test {uuid.uuid4().hex[:8]}"
    outcome, _ = await repository.seed_active(
        asset_key=key, active_workflow_id=ACTIVE, tenant_id=TENANT, trace_id="t-seed"
    )
    assert outcome == repository.OK
    outcome, _ = await repository.release(
        asset_key=key,
        canary_workflow_id=CANARY,
        canary_pct=10,
        canary_id="c1",
        released_by_card_id="card-a",
        tenant_id=TENANT,
        trace_id="t-release",
    )
    assert outcome == repository.OK
    try:
        yield key
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workflow_routing_pointer WHERE asset_key = $1", key)


async def test_only_the_releasing_card_can_promote(armed):
    outcome, row = await repository.promote_canary(
        asset_key=armed, canary_id="c1", card_id="card-b", trace_id="t"
    )
    assert outcome != repository.OK
    current = await repository.get_active_by_asset_key(asset_key=armed)
    assert current is not None
    assert current["active_workflow_id"] == ACTIVE
    assert current["canary_workflow_id"] == CANARY

    outcome, row = await repository.promote_canary(
        asset_key=armed, canary_id="c1", card_id="card-a", trace_id="t"
    )
    assert outcome == repository.OK
    assert row is not None
    assert row["active_workflow_id"] == CANARY
    assert row["prior_workflow_id"] == ACTIVE
    assert row["canary_workflow_id"] is None


async def test_any_card_can_discard_a_foreign_canary(armed):
    outcome, row = await repository.discard_canary(asset_key=armed, canary_id="c1")
    assert outcome == repository.OK
    assert row is not None
    assert row["active_workflow_id"] == ACTIVE
    assert row["canary_workflow_id"] is None


async def test_rollback_returns_the_pre_swap_active_as_rolled_back_from(pg_pool):
    """#338: ``RETURNING`` sees prior already NULL, so the webhook's
    ``rolled_back_from`` must be the pre-swap active read in the same statement."""
    key = f"cell:g15-rollback-{uuid.uuid4().hex[:8]}"
    try:
        first = await repository.promote(
            asset_key=key,
            new_workflow_id=ACTIVE,
            observation_window_minutes=0,
            rollback_thresholds={},
        )
        assert first is not None
        second = await repository.promote(
            asset_key=key,
            new_workflow_id=CANARY,
            observation_window_minutes=0,
            rollback_thresholds={},
        )
        assert second is not None
        assert second["prior_workflow_id"] == ACTIVE

        row = await repository.rollback(asset_key=key)
        assert row is not None
        assert row["asset_key"] == key
        assert row["active_workflow_id"] == ACTIVE
        assert row["prior_workflow_id"] is None
        assert row["rolled_back_from"] == CANARY

        # The guards are intact: no prior left ⇒ refused (None), nothing written.
        assert await repository.rollback(asset_key=key) is None
        current = await repository.get_active_by_asset_key(asset_key=key)
        assert current is not None
        assert current["active_workflow_id"] == ACTIVE
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workflow_routing_pointer WHERE asset_key = $1", key)


async def test_rollback_is_refused_while_a_canary_is_armed(armed):
    async with get_pg_connection() as conn:
        await conn.execute(
            "UPDATE workflow_routing_pointer SET prior_workflow_id = $2 WHERE asset_key = $1",
            armed,
            "PqRsTuVwXyZaBcDe",
        )
    assert await repository.rollback(asset_key=armed) is None
    current = await repository.get_active_by_asset_key(asset_key=armed)
    assert current is not None
    assert current["active_workflow_id"] == ACTIVE
