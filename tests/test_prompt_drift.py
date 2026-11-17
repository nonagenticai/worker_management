"""U6 #314: the prompt-binding drift metric (``src/api/workers/prompt_drift.py``).

The database is stubbed at ``prompt_drift.get_pg_connection``: the fake
connection answers the ``to_regclass`` probe and returns the joined binding rows
the real query would. The query itself is proved against real Postgres in
``tests/integration/test_prompt_drift_sql.py``. Everything else — the
comparison, the gauges, the label removal, the error latch, the task start — is
the real code, read back from the real prometheus ``REGISTRY``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from src.api.workers import prompt_drift
from src.settings import reset_settings_cache

if TYPE_CHECKING:
    from types import EllipsisType

P1 = "aaaaaaaa-0000-4000-8000-000000000001"
P3 = "aaaaaaaa-0000-4000-8000-000000000003"
CUSTOMER = "aaaaaaaa-0000-4000-8000-000000000100"  # cdmsfirst

PUBLISHED = "You are worker-a, published version 1."
REWRITTEN = "You are what the n8n ensure-node wrote last."


def _row(
    worker: str = "worker-a",
    *,
    tenant: str = P3,
    config: str | None = PUBLISHED,
    published: str | None = PUBLISHED,
    status: str | None = "PUBLISHED",
    worker_found: bool = True,
    owner: str | EllipsisType | None = ...,
) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "worker_name": worker,
        "prompt_id": worker,
        # migration 035: by default the binding's tenant owns its prompt.
        "owner_tenant_id": tenant if owner is ... else owner,
        "worker_found": worker_found,
        "config_system": config,
        "published_version": None if published is None else 1,
        "published_status": None if published is None else status,
        "published_system": published,
    }


def _db(rows: list[dict[str, Any]] | Exception, *, table: bool = True):
    """Patch the pool with a fake connection answering the two reads."""
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=table)
    if isinstance(rows, Exception):
        conn.fetch = AsyncMock(side_effect=rows)
    else:
        conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _ctx():
        yield conn

    return patch.object(prompt_drift, "get_pg_connection", _ctx), conn


def _drift(tenant: str, worker: str) -> float | None:
    return REGISTRY.get_sample_value(
        "wm_prompt_binding_drift", {"tenant_id": tenant, "worker_name": worker}
    )


def _sample(name: str) -> float | None:
    return REGISTRY.get_sample_value(name)


async def _refresh(rows: list[dict[str, Any]] | Exception, *, table: bool = True):
    patcher, conn = _db(rows, table=table)
    with patcher:
        result = await prompt_drift.refresh_once()
    return result, conn


@pytest.fixture(autouse=True)
def _clean():
    prompt_drift._reset_for_tests()
    yield
    prompt_drift._reset_for_tests()


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_equal_config_and_published_is_zero():
    snap, _ = await _refresh([_row()])
    assert snap is not None
    assert _drift(P3, "worker-a") == 0
    assert _sample("wm_prompt_binding_drift_total") == 0
    assert _sample("wm_prompt_binding_unresolved_total") == 0
    assert _sample("wm_prompt_binding_checked_total") == 1
    assert (_sample("wm_prompt_drift_last_success_timestamp_seconds") or 0) > 0


@pytest.mark.asyncio
async def test_config_rewritten_is_drift_one_and_total_one():
    await _refresh([_row(config=REWRITTEN), _row("worker-b")])
    assert _drift(P3, "worker-a") == 1
    assert _drift(P3, "worker-b") == 0
    assert _sample("wm_prompt_binding_drift_total") == 1
    assert _sample("wm_prompt_binding_checked_total") == 2


@pytest.mark.asyncio
async def test_drift_clears_when_the_config_converges_back():
    await _refresh([_row(config=REWRITTEN)])
    assert _drift(P3, "worker-a") == 1
    await _refresh([_row()])
    assert _drift(P3, "worker-a") == 0
    assert _sample("wm_prompt_binding_drift_total") == 0


@pytest.mark.asyncio
async def test_a_whitespace_difference_is_drift():
    """Mutation control: a comparator that stripped or normalised would read 0
    here, while the runtime sends different bytes. sha256_text has no
    normalisation, and neither may this comparison."""
    await _refresh([_row(config=PUBLISHED + "\n")])
    assert _drift(P3, "worker-a") == 1


@pytest.mark.asyncio
async def test_a_config_with_no_system_prompt_is_drift():
    await _refresh([_row(config=None)])
    assert _drift(P3, "worker-a") == 1


# ---------------------------------------------------------------------------
# unresolved bindings and tenancy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_published_version_is_unresolved_not_drift():
    await _refresh([_row(published=None), _row("worker-b")])
    assert _drift(P3, "worker-a") is None, "an unresolved binding carries no drift series"
    assert _sample("wm_prompt_binding_unresolved_total") == 1
    assert _sample("wm_prompt_binding_drift_total") == 0
    assert _sample("wm_prompt_binding_checked_total") == 2


@pytest.mark.asyncio
async def test_a_legacy_active_winner_is_unresolved_like_the_resolver():
    """get_active(highest_published=True) can return an 'active' row; the
    resolver answers None for it, so the metric must too."""
    await _refresh([_row(status="active")])
    assert _drift(P3, "worker-a") is None
    assert _sample("wm_prompt_binding_unresolved_total") == 1


@pytest.mark.asyncio
async def test_a_missing_worker_row_is_unresolved():
    await _refresh([_row(worker_found=False, config=None)])
    assert _drift(P3, "worker-a") is None
    assert _sample("wm_prompt_binding_unresolved_total") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [P1, None], ids=["owned_by_another_tenant", "no_owner"])
async def test_a_prompt_owned_by_another_tenant_is_unresolved_not_drift(owner):
    """035: the resolver refuses a cross-tenant (or ownerless) prompt and runs the
    config text, so the metric must count it unresolved and export no series —
    even when the texts differ."""
    await _refresh([_row(owner=owner, config=REWRITTEN), _row("worker-b")])
    assert _drift(P3, "worker-a") is None
    assert _sample("wm_prompt_binding_unresolved_total") == 1
    assert _sample("wm_prompt_binding_drift_total") == 0
    assert _sample("wm_prompt_binding_checked_total") == 2


@pytest.mark.asyncio
async def test_a_customer_binding_is_never_exported():
    await _refresh([_row(tenant=CUSTOMER, config=REWRITTEN), _row(tenant=P1)])
    assert _drift(CUSTOMER, "worker-a") is None
    assert _drift(P1, "worker-a") == 0
    assert _sample("wm_prompt_binding_checked_total") == 1
    assert _sample("wm_prompt_binding_drift_total") == 0


# ---------------------------------------------------------------------------
# series lifecycle and failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_removed_binding_series_is_gone():
    await _refresh([_row(config=REWRITTEN), _row("worker-b")])
    assert _drift(P3, "worker-a") == 1
    await _refresh([_row("worker-b")])
    assert _drift(P3, "worker-a") is None, "a stale 1 must not linger"
    assert _drift(P3, "worker-b") == 0
    assert _sample("wm_prompt_binding_drift_total") == 0


@pytest.mark.asyncio
async def test_a_db_error_keeps_previous_values_and_counts_it(caplog):
    await _refresh([_row(config=REWRITTEN), _row(published=None)])
    stamp = _sample("wm_prompt_drift_last_success_timestamp_seconds")
    before = _sample("wm_prompt_drift_refresh_errors_total") or 0.0

    with caplog.at_level(logging.WARNING, logger=prompt_drift.logger.name):
        snap, _ = await _refresh(RuntimeError("pool down"))
        assert snap is None
        snap, _ = await _refresh(RuntimeError("pool still down"))
        assert snap is None

    assert _sample("wm_prompt_drift_refresh_errors_total") == before + 2
    assert _drift(P3, "worker-a") == 1
    assert _sample("wm_prompt_binding_drift_total") == 1
    assert _sample("wm_prompt_binding_unresolved_total") == 1
    assert _sample("wm_prompt_drift_last_success_timestamp_seconds") == stamp
    warnings = [r for r in caplog.records if "drift refresh failed" in r.getMessage()]
    assert len(warnings) == 1, "logged once per process, counted every time"


@pytest.mark.asyncio
async def test_a_missing_binding_table_is_zero_bindings():
    snap, conn = await _refresh([_row(config=REWRITTEN)], table=False)
    assert snap is not None
    assert snap.checked == 0
    conn.fetch.assert_not_awaited()
    assert _sample("wm_prompt_binding_checked_total") == 0
    assert _sample("wm_prompt_binding_drift_total") == 0


# ---------------------------------------------------------------------------
# the refresher task
# ---------------------------------------------------------------------------


@pytest.fixture
def interval(monkeypatch):
    def _set(value: str) -> None:
        monkeypatch.setenv("WM_PROMPT_DRIFT_INTERVAL_S", value)
        reset_settings_cache()

    yield _set
    reset_settings_cache()


@pytest.mark.asyncio
async def test_interval_zero_does_not_start_the_refresher(interval):
    interval("0")
    with patch.object(prompt_drift, "drift_loop") as loop:
        assert prompt_drift.start_drift_task() is None
    loop.assert_not_called()


@pytest.mark.asyncio
async def test_the_refresher_runs_before_its_first_sleep_and_cancels(interval):
    interval("3600")
    patcher, conn = _db([_row(config=REWRITTEN)])
    with patcher:
        task = prompt_drift.start_drift_task()
        assert task is not None
        for _ in range(50):
            if _drift(P3, "worker-a") is not None:
                break
            await asyncio.sleep(0)
        assert _drift(P3, "worker-a") == 1, "first refresh happens before the 3600s sleep"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert conn.fetch.await_count == 1
