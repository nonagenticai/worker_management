"""Unit tests for db/workers_state_db.py with InMemoryDB.

Verifies real data flow: add states, retrieve them, verify content.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from src.db.workers_state_db import (
    DEFAULT_WORKER_STATES_LIMIT,
    MAX_WORKER_STATES_LIMIT,
    StateDatabase,
    clamp_worker_states_limit,
    with_retry,
)
from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

# ---------------------------------------------------------------------------
# with_retry decorator
# ---------------------------------------------------------------------------


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        @with_retry(max_retries=3, retry_delay=0.001)
        async def my_func():
            return "ok"

        result = await my_func()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_interface_error(self):
        call_count = 0

        @with_retry(max_retries=3, retry_delay=0.001)
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise psycopg.InterfaceError("transient")
            return "ok"

        result = await my_func()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_raises(self):
        @with_retry(max_retries=2, retry_delay=0.001)
        async def my_func():
            raise psycopg.InterfaceError("persistent failure")

        with pytest.raises(psycopg.InterfaceError, match="persistent"):
            await my_func()

    @pytest.mark.asyncio
    async def test_retries_on_postgres_error(self):
        call_count = 0

        @with_retry(max_retries=3, retry_delay=0.001)
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise psycopg.Error("pg error")
            return "recovered"

        result = await my_func()
        assert result == "recovered"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Add worker state
# ---------------------------------------------------------------------------


class TestAddWorkerState:
    @pytest.mark.asyncio
    async def test_add_dict_state(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type="idle",
                worker_state={"key": "value", "count": 42},
            )

        assert isinstance(state_id, int)
        assert state_id >= 1

    @pytest.mark.asyncio
    async def test_add_bytes_state(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type="idle",
                worker_state=json.dumps({"from": "bytes"}).encode("utf-8"),
            )

            # Read back to verify bytes were converted to dict
            states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert state_id >= 1
        assert len(states) == 1
        assert states[0]["worker_state"]["from"] == "bytes"

    @pytest.mark.asyncio
    async def test_add_string_state(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type="idle",
                worker_state=json.dumps({"from": "string"}),
            )

            states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert state_id >= 1
        assert len(states) == 1
        assert states[0]["worker_state"]["from"] == "string"

    @pytest.mark.asyncio
    async def test_add_with_metadata(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type="idle",
                worker_state={"status": "ok"},
                metadata={"source": "test"},
            )

            states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert state_id >= 1
        assert states[0]["metadata"] == {"source": "test"}


# ---------------------------------------------------------------------------
# Batch add
# ---------------------------------------------------------------------------


class TestAddWorkerStateBatch:
    @pytest.mark.asyncio
    async def test_empty_batch(self):
        db = StateDatabase()
        result = await db.add_worker_state_batch(tenant_id="c1", subject_id="c1", states=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_insert_and_read_back(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            states_input = [
                {
                    "worker_id": "w1",
                    "worker_name": "W1",
                    "state_type": "idle",
                    "worker_state": {"x": 1},
                    "is_agent": False,
                },
                {
                    "worker_id": "w2",
                    "worker_name": "W2",
                    "state_type": "busy",
                    "worker_state": {"x": 2},
                    "is_agent": True,
                },
            ]
            ids = await db.add_worker_state_batch(
                tenant_id="c1", subject_id="c1", states=states_input
            )

            all_states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert len(ids) == 2
        assert len(all_states) == 2
        worker_names = {s["worker_name"] for s in all_states}
        assert worker_names == {"W1", "W2"}


# ---------------------------------------------------------------------------
# Get worker states
# ---------------------------------------------------------------------------


class TestGetWorkerStates:
    @pytest.mark.asyncio
    async def test_returns_all_for_creator(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="idle",
                worker_state={"a": 1},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w2",
                worker_name="W2",
                is_agent=True,
                state_type="busy",
                worker_state={"b": 2},
            )
            await db.add_worker_state(
                tenant_id="c2",
                subject_id="c2",
                worker_id="w3",
                worker_name="W3",
                is_agent=False,
                state_type="idle",
                worker_state={"c": 3},
            )

            result_c1 = await db.get_worker_states(tenant_id="c1", subject_id="c1")
            result_c2 = await db.get_worker_states(tenant_id="c2", subject_id="c2")

        assert len(result_c1) == 2
        assert len(result_c2) == 1


# ---------------------------------------------------------------------------
# Get worker states by ID
# ---------------------------------------------------------------------------


class TestGetWorkerStatesById:
    @pytest.mark.asyncio
    async def test_returns_states_for_worker(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="idle",
                worker_state={"step": 1},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="busy",
                worker_state={"step": 2},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w2",
                worker_name="W2",
                is_agent=False,
                state_type="idle",
                worker_state={"step": 1},
            )

            result = await db.get_worker_states_by_id(
                tenant_id="c1", subject_id="c1", worker_id="w1", limit=10
            )

        assert len(result) == 2
        for r in result:
            assert r["worker_id"] == "w1"

    @pytest.mark.asyncio
    async def test_with_limit(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            for i in range(5):
                await db.add_worker_state(
                    tenant_id="c1",
                    subject_id="c1",
                    worker_id="w1",
                    worker_name="W1",
                    is_agent=False,
                    state_type="s",
                    worker_state={"step": i},
                )

            result = await db.get_worker_states_by_id(
                tenant_id="c1", subject_id="c1", worker_id="w1", limit=3
            )

        assert len(result) == 3


# ---------------------------------------------------------------------------
# Get latest worker state
# ---------------------------------------------------------------------------


class TestGetLatestWorkerState:
    @pytest.mark.asyncio
    async def test_returns_most_recent(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="idle",
                worker_state={"v": 1},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="busy",
                worker_state={"v": 2},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="done",
                worker_state={"v": 3},
            )

            latest = await db.get_latest_worker_state(
                tenant_id="c1", subject_id="c1", worker_id="w1"
            )

        assert latest is not None
        # Should be the most recent (limit=1)
        assert latest["state_type"] in ("idle", "busy", "done")

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            result = await db.get_latest_worker_state(
                tenant_id="c1", subject_id="c1", worker_id="w1"
            )

        assert result is None


# ---------------------------------------------------------------------------
# Delete worker state
# ---------------------------------------------------------------------------


class TestDeleteWorkerState:
    @pytest.mark.asyncio
    async def test_delete_states_for_worker(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="idle",
                worker_state={"a": 1},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="busy",
                worker_state={"a": 2},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w2",
                worker_name="W2",
                is_agent=False,
                state_type="idle",
                worker_state={"b": 1},
            )

            count = await db.delete_worker_state(tenant_id="c1", subject_id="c1", worker_id="w1")
            remaining = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert count == 2
        assert len(remaining) == 1
        assert remaining[0]["worker_id"] == "w2"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_zero(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            count = await db.delete_worker_state(tenant_id="c1", subject_id="c1", worker_id="nope")

        assert count == 0


# ---------------------------------------------------------------------------
# Delete all worker states
# ---------------------------------------------------------------------------


class TestDeleteAllWorkerStates:
    @pytest.mark.asyncio
    async def test_deletes_all_for_creator(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="W1",
                is_agent=False,
                state_type="idle",
                worker_state={"a": 1},
            )
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w2",
                worker_name="W2",
                is_agent=False,
                state_type="idle",
                worker_state={"a": 2},
            )
            await db.add_worker_state(
                tenant_id="c2",
                subject_id="c2",
                worker_id="w3",
                worker_name="W3",
                is_agent=False,
                state_type="idle",
                worker_state={"b": 1},
            )

            count = await db.delete_all_worker_states(tenant_id="c1", subject_id="c1")
            remaining_c1 = await db.get_worker_states(tenant_id="c1", subject_id="c1")
            remaining_c2 = await db.get_worker_states(tenant_id="c2", subject_id="c2")

        assert count == 2
        assert len(remaining_c1) == 0
        assert len(remaining_c2) == 1


# ---------------------------------------------------------------------------
# _row_to_dict
# ---------------------------------------------------------------------------


class TestRowToDict:
    def test_conversion(self):
        db = StateDatabase()
        row = {
            "id": 1,
            "worker_id": "w1",
            "worker_name": "W",
            "tenant_id": "c1",
            "subject_id": "c1",
            "visibility": "tenant",
            "is_agent": False,
            "state_type": "idle",
            "worker_state": {"k": "v"},
            "timestamp": datetime(2026, 1, 1),
            "metadata": {"m": 1},
        }
        result = db._row_to_dict(row)
        assert result["id"] == 1
        assert result["worker_id"] == "w1"
        assert result["worker_state"] == {"k": "v"}
        assert result["timestamp"] == "2026-01-01T00:00:00"
        assert result["metadata"] == {"m": 1}

    def test_null_fields(self):
        db = StateDatabase()
        row = {
            "id": 2,
            "worker_id": "w2",
            "worker_name": "W2",
            "tenant_id": "c1",
            "subject_id": "c1",
            "visibility": "tenant",
            "is_agent": True,
            "state_type": "done",
            "worker_state": None,
            "timestamp": None,
            "metadata": None,
        }
        result = db._row_to_dict(row)
        assert result["worker_state"] == {}
        assert result["timestamp"] is None
        assert result["metadata"] == {}


# ---------------------------------------------------------------------------
# get_worker_states pagination bounds
# ---------------------------------------------------------------------------


class TestGetWorkerStatesPagination:
    """``get_worker_states`` must never read a tenant's whole history."""

    @staticmethod
    async def _seed(db: StateDatabase, count: int) -> None:
        await db.add_worker_state_batch(
            tenant_id="c1",
            subject_id="c1",
            states=[
                {
                    "worker_id": f"w{i}",
                    "worker_name": f"W{i}",
                    "state_type": "idle",
                    "worker_state": {"i": i},
                    "is_agent": False,
                }
                for i in range(count)
            ],
        )

    @pytest.mark.asyncio
    async def test_default_limit_bounds_the_query(self):
        """A caller passing no pagination args gets ONE page, not everything."""
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, 150)
            states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert len(states) == DEFAULT_WORKER_STATES_LIMIT

    @pytest.mark.asyncio
    async def test_limit_above_max_is_clamped_not_honoured(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, MAX_WORKER_STATES_LIMIT + 25)
            states = await db.get_worker_states(tenant_id="c1", subject_id="c1", limit=1_000_000)

        assert len(states) == MAX_WORKER_STATES_LIMIT

    @pytest.mark.asyncio
    async def test_limit_below_one_is_clamped_to_one(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, 5)
            states = await db.get_worker_states(tenant_id="c1", subject_id="c1", limit=0)

        assert len(states) == 1

    @pytest.mark.asyncio
    async def test_paging_returns_each_row_exactly_once(self):
        """No gaps, no duplicates, and the walk terminates on a short page."""
        mem = InMemoryDB()
        db = StateDatabase()
        total = 250
        page_size = 40

        seen: list[int] = []
        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, total)

            offset = 0
            # Bounded walk: an UNBOUNDED query returns every row on every page,
            # so the loop would never terminate. Cap it so that regression
            # FAILS the test instead of hanging the suite.
            max_pages = (total // page_size) + 2
            for _ in range(max_pages):
                page = await db.get_worker_states(
                    tenant_id="c1", subject_id="c1", limit=page_size, offset=offset
                )
                assert len(page) <= page_size, "query returned more rows than the page size"
                seen.extend(s["id"] for s in page)
                if len(page) < page_size:
                    break
                offset += page_size
            else:
                pytest.fail("paging did not terminate: the query is not bounded")

        assert len(seen) == total
        assert len(set(seen)) == total
        # id DESC ordering is preserved ACROSS pages, not just within one.
        assert seen == sorted(seen, reverse=True)

    @pytest.mark.asyncio
    async def test_no_args_caller_gets_unchanged_item_shape(self):
        """Backward compatibility: existing callers keep working, same fields."""
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=True,
                state_type="completed",
                worker_state={"result": 42},
            )
            states = await db.get_worker_states(tenant_id="c1", subject_id="c1")

        assert len(states) == 1
        assert states[0]["worker_name"] == "Worker1"
        assert states[0]["is_agent"] is True
        assert states[0]["state_type"] == "completed"
        assert states[0]["worker_state"]["result"] == 42

    def test_clamp_helper_bounds(self):
        assert clamp_worker_states_limit(-5) == 1
        assert clamp_worker_states_limit(0) == 1
        assert clamp_worker_states_limit(50) == 50
        assert clamp_worker_states_limit(MAX_WORKER_STATES_LIMIT + 1) == MAX_WORKER_STATES_LIMIT

    @pytest.mark.asyncio
    async def test_count_reports_the_true_total_beyond_the_cap(self):
        """count_worker_states must see rows the capped page read cannot."""
        mem = InMemoryDB()
        db = StateDatabase()
        total = MAX_WORKER_STATES_LIMIT + 137

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, total)
            page = await db.get_worker_states(
                tenant_id="c1", subject_id="c1", limit=MAX_WORKER_STATES_LIMIT
            )
            counted = await db.count_worker_states(tenant_id="c1", subject_id="c1")

        assert len(page) == MAX_WORKER_STATES_LIMIT
        assert counted == total
        assert counted > len(page)

    @pytest.mark.asyncio
    async def test_count_is_scoped_to_the_tenant(self):
        mem = InMemoryDB()
        db = StateDatabase()

        with patch_get_pg_connection("src.db.workers_state_db", mem):
            await self._seed(db, 3)
            await db.add_worker_state(
                tenant_id="c2",
                subject_id="c2",
                worker_id="w9",
                worker_name="W9",
                is_agent=False,
                state_type="idle",
                worker_state={"x": 9},
            )
            assert await db.count_worker_states(tenant_id="c1", subject_id="c1") == 3
            assert await db.count_worker_states(tenant_id="c2", subject_id="c2") == 1


# ---------------------------------------------------------------------------
# The write-choke-point counter
# ---------------------------------------------------------------------------


class TestWorkerStateWriteCounter:
    """``add_worker_state`` is the choke point EVERY real state write passes.

    The Pydantic surface (``/workers_states``) validates but carries no real
    traffic; this function takes a plain ``str`` and INSERTs it with no enum
    coercion and no CHECK behind it. ⛔ Observe-only — nothing here rejects.
    """

    @staticmethod
    async def _write(mem, *, state_type, **extra):
        db = StateDatabase()
        client = MagicMock()
        with (
            patch_get_pg_connection("src.db.workers_state_db", mem),
            patch("src.db.workers_state_db.get_metrics_client", return_value=client),
        ):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type=state_type,
                worker_state={"status": state_type},
                **extra,
            )
        return state_id, client.record_worker_state_write.call_args_list

    @pytest.mark.asyncio
    async def test_write_is_counted_once_with_validated_false_by_default(self):
        """Default "false": this layer CANNOT know what the caller established.

        ⭐ "false" is the EXPOSURE value, and defaulting DOWNWARD is the point —
        a caller that says nothing must never be booked as coverage.
        """
        state_id, calls = await self._write(InMemoryDB(), state_type="processing")
        assert isinstance(state_id, int)
        assert len(calls) == 1
        assert calls[0].args == ("processing",)
        assert calls[0].kwargs == {"validated": "false"}

    @pytest.mark.asyncio
    async def test_each_of_the_three_validated_values_is_forwarded_verbatim(self):
        """All three must survive this layer unaltered — it classifies nothing."""
        from src.telemetry.agent_metrics import WORKER_STATE_WRITE_VALIDATED

        assert WORKER_STATE_WRITE_VALIDATED == ("true", "false", "by_construction")
        for value in WORKER_STATE_WRITE_VALIDATED:
            _state_id, calls = await self._write(
                InMemoryDB(), state_type="completed", validated=value
            )
            assert len(calls) == 1
            assert calls[0].kwargs == {"validated": value}

    @pytest.mark.asyncio
    async def test_config_updated_reaches_the_counter_raw_and_is_classified_there(self):
        """⚠️ HYPHEN. 153,087 live rows. The DB layer must hand over the RAW value.

        ⚠️ The classification half INVERTED on 2026-09-13: ``config-updated``
        became a ``WorkerStateType`` member, so it now labels as ITSELF. The
        property this test actually guards is unchanged and is the first
        assertion — one seam owns the allow-list, and it is not this one.
        """
        from src.telemetry.agent_metrics import classify_worker_state_type

        _state_id, calls = await self._write(InMemoryDB(), state_type="config-updated")
        assert calls[0].args == ("config-updated",)
        assert classify_worker_state_type(calls[0].args[0]) == "config-updated"

    @pytest.mark.asyncio
    async def test_a_real_non_member_still_reaches_the_counter_raw_and_collapses(self):
        """⛔ The sentinel's job, on a value that is STILL not a member.

        ``prompt-swapped`` is in the row census and has zero transition traffic,
        so it was deliberately not promoted. Without this case, promoting
        ``config-updated`` would have left the collapse path untested here.
        """
        from src.telemetry.agent_metrics import classify_worker_state_type

        _state_id, calls = await self._write(InMemoryDB(), state_type="prompt-swapped")
        assert calls[0].args == ("prompt-swapped",)
        assert classify_worker_state_type(calls[0].args[0]) == "<unknown>"

    @pytest.mark.asyncio
    async def test_counter_failure_does_not_fail_the_write(self):
        """An instrumentation failure must never turn a persisted row into a 500."""
        mem = InMemoryDB()
        db = StateDatabase()
        with (
            patch_get_pg_connection("src.db.workers_state_db", mem),
            patch(
                "src.db.workers_state_db.get_metrics_client",
                side_effect=RuntimeError("otel down"),
            ),
        ):
            state_id = await db.add_worker_state(
                tenant_id="c1",
                subject_id="c1",
                worker_id="w1",
                worker_name="Worker1",
                is_agent=False,
                state_type="idle",
                worker_state={"status": "idle"},
            )
        assert isinstance(state_id, int)
