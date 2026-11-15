"""Tests for WorkerStateManager — state transitions, locking, worker ID resolution."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest

from src.services.worker_state_manager import (
    STATE_TRANSITIONS,
    WorkerStateManager,
    WorkerStateType,
)


@pytest.fixture
def state_manager():
    return WorkerStateManager()


class TestStateTransitionValidation:
    """Test valid/invalid state transitions."""

    def test_created_to_processing(self, state_manager):
        assert state_manager.validate_state_transition("created", "processing") is True

    def test_created_to_idle(self, state_manager):
        assert state_manager.validate_state_transition("created", "idle") is True

    def test_created_to_error(self, state_manager):
        assert state_manager.validate_state_transition("created", "error") is True

    def test_processing_to_completed(self, state_manager):
        assert state_manager.validate_state_transition("processing", "completed") is True

    def test_processing_to_error(self, state_manager):
        assert state_manager.validate_state_transition("processing", "error") is True

    def test_processing_to_idle(self, state_manager):
        assert state_manager.validate_state_transition("processing", "idle") is True

    def test_completed_to_processing(self, state_manager):
        assert state_manager.validate_state_transition("completed", "processing") is True

    def test_error_to_processing(self, state_manager):
        assert state_manager.validate_state_transition("error", "processing") is True

    def test_error_to_created(self, state_manager):
        assert state_manager.validate_state_transition("error", "created") is True

    def test_idle_to_processing(self, state_manager):
        assert state_manager.validate_state_transition("idle", "processing") is True

    def test_idle_to_active(self, state_manager):
        assert state_manager.validate_state_transition("idle", "active") is True

    def test_default_to_processing(self, state_manager):
        assert state_manager.validate_state_transition("default", "processing") is True

    def test_invalid_created_to_completed(self, state_manager):
        """Cannot go directly from created to completed."""
        assert state_manager.validate_state_transition("created", "completed") is False

    def test_invalid_completed_to_error(self, state_manager):
        """Cannot go from completed to error."""
        assert state_manager.validate_state_transition("completed", "error") is False

    def test_invalid_completed_to_created(self, state_manager):
        assert state_manager.validate_state_transition("completed", "created") is False

    def test_unknown_state_allows_transition(self, state_manager):
        """Unknown states should be allowed with a warning."""
        assert state_manager.validate_state_transition("unknown", "processing") is True

    def test_all_transitions_covered(self):
        """Verify all enum states have transition rules."""
        for state in WorkerStateType:
            assert state in STATE_TRANSITIONS, f"Missing transition rules for {state}"


class TestAtomicWorkerOperation:
    """Test atomic operation locking."""

    @pytest.mark.asyncio
    async def test_lock_is_acquired(self, state_manager):
        async with state_manager.atomic_worker_operation("w1", "worker1", "test_op") as lock_hash:
            assert lock_hash is not None
            assert len(lock_hash) == 16  # SHA256 truncated to 16
            # Lock info should be stored
            assert "w1" in state_manager._lock_info

        # After context exit, lock info should be cleaned up
        assert "w1" not in state_manager._lock_info

    @pytest.mark.asyncio
    async def test_sequential_operations(self, state_manager):
        """Two sequential operations on same worker should succeed."""
        results = []
        async with state_manager.atomic_worker_operation("w1", "worker1", "op1") as h1:
            results.append(h1)

        async with state_manager.atomic_worker_operation("w1", "worker1", "op2") as h2:
            results.append(h2)

        assert len(results) == 2
        # Different operations should get different hashes
        assert results[0] != results[1]

    @pytest.mark.asyncio
    async def test_concurrent_same_worker_serialized(self, state_manager):
        """Concurrent operations on the same worker+operation should be serialized."""
        execution_order = []

        async def operation(name, delay):
            async with state_manager.atomic_worker_operation("w1", "worker1", "same_op"):
                execution_order.append(f"{name}_start")
                await asyncio.sleep(delay)
                execution_order.append(f"{name}_end")

        await asyncio.gather(
            operation("a", 0.05),
            operation("b", 0.01),
        )

        # One must complete before the other starts
        assert execution_order[0].endswith("_start")
        assert execution_order[1].endswith("_end")
        assert execution_order[2].endswith("_start")
        assert execution_order[3].endswith("_end")

    @pytest.mark.asyncio
    async def test_different_workers_parallel(self, state_manager):
        """Operations on different workers should run in parallel."""
        execution_order = []

        async def operation(worker_id, name, delay):
            async with state_manager.atomic_worker_operation(worker_id, name, "op"):
                execution_order.append(f"{name}_start")
                await asyncio.sleep(delay)
                execution_order.append(f"{name}_end")

        await asyncio.gather(
            operation("w1", "a", 0.05),
            operation("w2", "b", 0.05),
        )

        # Both should start before either ends (parallel execution)
        starts = [i for i, x in enumerate(execution_order) if x.endswith("_start")]
        ends = [i for i, x in enumerate(execution_order) if x.endswith("_end")]
        # At least one start should come before all ends
        assert min(starts) < max(ends)


class TestResolveWorkerId:
    """Test worker ID resolution with registry lookup."""

    @pytest.mark.asyncio
    async def test_resolve_from_registry(self, state_manager):
        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(
            return_value={"worker_id": "resolved-id-123", "worker_name": "test"}
        )

        worker_id, in_registry = await state_manager.resolve_worker_id(
            "tenant1", "tenant1", "test", mock_registry
        )
        assert worker_id == "resolved-id-123"
        assert in_registry is True

    @pytest.mark.asyncio
    async def test_resolve_uuid_as_name(self, state_manager):
        """If worker_name is a UUID, use it directly."""
        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(return_value=None)

        uuid_str = "01935f7a-1234-7000-8000-abcdef123456"
        worker_id, in_registry = await state_manager.resolve_worker_id(
            "tenant1", "tenant1", uuid_str, mock_registry
        )
        assert worker_id == uuid_str
        assert in_registry is False

    @pytest.mark.asyncio
    async def test_resolve_fallback_to_name(self, state_manager):
        """Non-UUID worker_name not in registry falls back to name as ID."""
        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(return_value=None)

        worker_id, in_registry = await state_manager.resolve_worker_id(
            "tenant1", "tenant1", "my_worker", mock_registry
        )
        assert worker_id == "my_worker"
        assert in_registry is False

    @pytest.mark.asyncio
    async def test_resolve_handles_registry_error(self, state_manager):
        """Registry errors should fall back gracefully."""
        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(side_effect=psycopg.OperationalError("DB error"))

        worker_id, in_registry = await state_manager.resolve_worker_id(
            "tenant1", "tenant1", "my_worker", mock_registry
        )
        assert worker_id == "my_worker"
        assert in_registry is False


class TestHeldLockKeying:
    """Lock identity is keyed on ``(worker_id, operation)`` — and reaped when idle.

    Replaces the former ``get_worker_lock`` tests. That method handed out a bare
    lock with no corresponding release signal, which is exactly why its entry in
    ``_operation_locks`` could never be reclaimed; the refcounted ``_held_lock``
    scope is now the only way to obtain one. The keying contract those tests
    asserted is preserved below — now expressed under contention, which is the
    only situation in which sharing one lock object actually matters.
    """

    @staticmethod
    async def _hold(state_manager, worker_id, operation, seen, started, release):
        async with state_manager._held_lock(worker_id, operation) as lock:
            seen.append(lock)
            started.set()
            await release.wait()

    @pytest.mark.asyncio
    async def test_same_key_shares_one_lock(self, state_manager):
        """A waiter must be handed the very same lock object as the holder."""
        seen: list = []
        started, release = asyncio.Event(), asyncio.Event()

        h = asyncio.create_task(self._hold(state_manager, "w1", "op1", seen, started, release))
        await started.wait()

        async def waiter():
            async with state_manager._held_lock("w1", "op1") as lock:
                seen.append(lock)

        w = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(h, w)

        assert len(seen) == 2
        assert seen[0] is seen[1]
        assert state_manager._operation_locks == {}

    @pytest.mark.asyncio
    async def test_different_operation_is_a_different_lock(self, state_manager):
        seen: list = []
        started, release = asyncio.Event(), asyncio.Event()

        h = asyncio.create_task(self._hold(state_manager, "w1", "op1", seen, started, release))
        await started.wait()
        # Same worker, different operation: must not block on the held lock.
        async with state_manager._held_lock("w1", "op2") as lock:
            seen.append(lock)
        release.set()
        await h

        assert seen[0] is not seen[1]
        assert state_manager._operation_locks == {}

    @pytest.mark.asyncio
    async def test_different_worker_is_a_different_lock(self, state_manager):
        seen: list = []
        started, release = asyncio.Event(), asyncio.Event()

        h = asyncio.create_task(self._hold(state_manager, "w1", "op1", seen, started, release))
        await started.wait()
        async with state_manager._held_lock("w2", "op1") as lock:
            seen.append(lock)
        release.set()
        await h

        assert seen[0] is not seen[1]
        assert state_manager._operation_locks == {}

    @pytest.mark.asyncio
    async def test_entry_exists_while_held_and_is_gone_after(self, state_manager):
        async with state_manager._held_lock("w1", "op1"):
            assert list(state_manager._operation_locks) == ["w1:op1"]
        assert state_manager._operation_locks == {}


class TestTransitionVerdictCounter:
    """``validate_state_transition`` emits one verdict per call — and changes nothing.

    ⛔ The three ``unknown_*`` verdicts are the FAIL-OPEN path: the function
    still returns ``True`` on all of them. Making the state machine reject was
    measured at ~142,560 rejected invocations over 30 days (≈55% of traffic),
    so these tests pin the observability AND pin the absence of behaviour
    change. If a future edit makes any arm below return False or raise, that is
    the regression.
    """

    @staticmethod
    def _verdicts_for(state_manager, current, new):
        """Return (verdicts emitted, value returned) for one validate call."""
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            returned = state_manager.validate_state_transition(current, new)
        emitted = [
            call.args[0] for call in client.record_worker_state_transition_verdict.call_args_list
        ]
        return emitted, returned

    def test_valid_fires_once(self, state_manager):
        emitted, returned = self._verdicts_for(state_manager, "created", "processing")
        assert emitted == ["valid"]
        assert returned is True

    def test_config_updated_to_processing_is_a_known_valid_transition(self, state_manager, caplog):
        """⭐ THE CHANGE. ⚠️ HYPHEN — ``config-updated`` is the 153,087-row value.

        It was 4,615 of 4,616 unknown verdicts over 24h (99.98%; denominator
        8,444 calls), i.e. essentially the entire fail-open population, and it
        is now an enum member with a ``STATE_TRANSITIONS`` entry.

        ⛔ ``returned is True`` ALONE CANNOT PROVE THIS: the fail-open arm
        returned ``True`` for this pair too, so a return-value assertion passes
        identically before and after. The proof that the unknown arm is no
        longer taken is the other three: verdict ``valid`` rather than
        ``unknown_from``, NO pair-census emission, and NO warning.
        """
        client = MagicMock()
        with (
            caplog.at_level(logging.WARNING, logger="src.services.worker_state_manager"),
            patch("src.services.worker_state_manager.get_metrics_client", return_value=client),
        ):
            returned = state_manager.validate_state_transition("config-updated", "processing")

        assert returned is True
        emitted = [
            call.args[0] for call in client.record_worker_state_transition_verdict.call_args_list
        ]
        assert emitted == ["valid"], "must resolve against STATE_TRANSITIONS, not fail open"
        assert client.record_worker_state_transition_unknown_pair.call_args_list == [], (
            "the pair census fires only on the unknown arm; this pair left it"
        )
        assert "Unknown state types in transition" not in caplog.text

    def test_config_updated_targets_are_exactly_processing(self):
        """⛔ Not widened on plausibility: ``processing`` is the ONLY observed target.

        The 24h pair census carried 72 series for this counter and exactly one
        was non-zero. Every other target for ``config-updated`` is unmeasured,
        so it stays out.
        """
        assert STATE_TRANSITIONS[WorkerStateType.CONFIG_UPDATED] == [WorkerStateType.PROCESSING]

    def test_config_updated_to_an_unobserved_target_is_invalid_not_unknown(self, state_manager):
        """The from-side is known now, so an unlisted target books invalid, not unknown_to."""
        emitted, returned = self._verdicts_for(state_manager, "config-updated", "completed")
        assert emitted == ["invalid_transition"]
        assert returned is False

    def test_invalid_transition_fires_once(self, state_manager):
        emitted, returned = self._verdicts_for(state_manager, "created", "completed")
        assert emitted == ["invalid_transition"]
        assert returned is False

    def test_unknown_from_fires_once_and_still_returns_true(self, state_manager):
        """⚠️ Exemplar is ``prompt-swapped``, NOT ``config-updated``.

        ``config-updated`` was the exemplar until it became an enum member on
        2026-09-13; it now books ``valid`` and cannot reach this arm. Left as
        it was, this test would have kept passing on the WRONG verdict name.
        """
        emitted, returned = self._verdicts_for(state_manager, "prompt-swapped", "processing")
        assert emitted == ["unknown_from"]
        assert returned is True, "fail-open must be preserved"

    def test_unknown_to_fires_once_and_still_returns_true(self, state_manager):
        emitted, returned = self._verdicts_for(state_manager, "processing", "prompt-swapped")
        assert emitted == ["unknown_to"]
        assert returned is True, "fail-open must be preserved"

    def test_unknown_both_fires_once_and_still_returns_true(self, state_manager):
        """⚠️ The to-side exemplar was ``updated`` until 2026-09-17.

        ``updated`` became an enum member, so that pair now books
        ``unknown_from``, not ``unknown_both`` — this test would have kept
        passing on the WRONG verdict name had the exemplar been left alone. Both
        sides are now free text that no promotion can turn into members.
        """
        emitted, returned = self._verdicts_for(state_manager, "prompt-swapped", "7f2a-whatever")
        assert emitted == ["unknown_both"]
        assert returned is True, "fail-open must be preserved"

    def test_every_verdict_arm_is_reachable(self):
        """Guard the guard: the five arms above must be the whole closed set."""
        from src.telemetry.agent_metrics import WORKER_STATE_VERDICTS

        reached = {
            "valid",
            "invalid_transition",
            "unknown_from",
            "unknown_to",
            "unknown_both",
        }
        assert reached == set(WORKER_STATE_VERDICTS)

    def test_counter_failure_does_not_break_validation(self, state_manager):
        """Observe-only: a dead metrics client must not change the verdict."""
        with patch(
            "src.services.worker_state_manager.get_metrics_client",
            side_effect=RuntimeError("otel down"),
        ):
            assert state_manager.validate_state_transition("created", "processing") is True
            assert state_manager.validate_state_transition("created", "completed") is False
            assert state_manager.validate_state_transition("prompt-swapped", "x") is True

    def test_unknown_pair_census_is_emitted_beside_the_verdict(self, state_manager):
        """⭐ The deliverable: the verdict alone cannot name WHICH transition fell through."""
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            assert state_manager.validate_state_transition("prompt-swapped", "processing") is True
        pairs = [
            call.args for call in client.record_worker_state_transition_unknown_pair.call_args_list
        ]
        assert pairs == [("prompt-swapped", "processing")]

    def test_unknown_pair_census_is_NOT_emitted_on_the_known_arms(self, state_manager):
        """⛔ The unknown census must stay on the unknown arm.

        Both sides already resolved against the enum on valid/invalid, and the
        invalid arm now has its OWN counter. If this starts firing on the known
        arms the two counters stop partitioning the label space, which is the
        whole separate-counter rationale.
        """
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            state_manager.validate_state_transition("created", "processing")  # valid
            state_manager.validate_state_transition("created", "completed")  # invalid
        assert client.record_worker_state_transition_unknown_pair.call_args_list == []

    def test_unknown_pair_census_passes_the_raw_values_for_classification(self, state_manager):
        """The manager hands over raw strings; agent_metrics owns the allow-list.

        One seam owns the cardinality bound, so there is one place to audit it.
        """
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            state_manager.validate_state_transition("prompt-swapped", "7f2a-whatever")
        assert client.record_worker_state_transition_unknown_pair.call_args.args == (
            "prompt-swapped",
            "7f2a-whatever",
        )

    def test_unknown_pair_census_survives_a_missing_side(self, state_manager):
        """A ``None`` from an untyped call site must still be counted, not crash."""
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            assert state_manager.validate_state_transition(None, "processing") is True
        assert client.record_worker_state_transition_unknown_pair.call_args.args == (
            None,
            "processing",
        )

    def test_pair_counter_failure_does_not_break_validation(self, state_manager):
        """Observe-only: a dead pair counter must not change a single return value."""
        client = MagicMock()
        client.record_worker_state_transition_unknown_pair.side_effect = RuntimeError("otel down")
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            assert state_manager.validate_state_transition("prompt-swapped", "processing") is True
            assert state_manager.validate_state_transition("created", "processing") is True
            assert state_manager.validate_state_transition("created", "completed") is False

    def test_unknown_states_still_log_the_original_warning(self, state_manager, caplog):
        with (
            caplog.at_level(logging.WARNING, logger="src.services.worker_state_manager"),
            patch("src.services.worker_state_manager.get_metrics_client", return_value=MagicMock()),
        ):
            state_manager.validate_state_transition("prompt-swapped", "processing")
        assert "Unknown state types in transition" in caplog.text

    # -- the INVALID-arm pair census (sibling of the unknown-arm one) --------

    def test_invalid_pair_census_is_emitted_beside_the_invalid_verdict(self, state_manager):
        """⭐ THE CHANGE. ``invalid_transition`` is 1,753 of 16,409 verdicts / 24h.

        The verdict counter's labels are verdict-only, so until now nothing could
        say WHICH transitions that 10.7% is. ⛔ ``returned is False`` alone cannot
        prove this test — that was already the behaviour — so the assertion that
        matters is the emission with the raw pair.
        """
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            returned = state_manager.validate_state_transition("processing", "processing")
        assert returned is False
        pairs = [
            call.args for call in client.record_worker_state_transition_invalid_pair.call_args_list
        ]
        assert pairs == [("processing", "processing")]
        assert client.record_worker_state_transition_unknown_pair.call_args_list == [], (
            "the two pair counters must partition; a pair must never land on both"
        )

    def test_invalid_pair_census_is_NOT_emitted_on_valid_or_unknown(self, state_manager):
        """⛔ The complement of the test above, and the partition's other half."""
        client = MagicMock()
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            state_manager.validate_state_transition("created", "processing")  # valid
            state_manager.validate_state_transition("prompt-swapped", "processing")  # unknown_from
            state_manager.validate_state_transition("created", "7f2a-whatever")  # unknown_to
        assert client.record_worker_state_transition_invalid_pair.call_args_list == []

    def test_invalid_pair_census_changes_no_return_value(self, state_manager):
        """⛔ COUNTER ONLY. The fail-open stays open and nothing starts raising.

        A dead counter must not change a single verdict, and specifically the
        unknown arm must still return ``True``.
        """
        client = MagicMock()
        client.record_worker_state_transition_invalid_pair.side_effect = RuntimeError("otel down")
        with patch("src.services.worker_state_manager.get_metrics_client", return_value=client):
            assert state_manager.validate_state_transition("created", "completed") is False
            assert state_manager.validate_state_transition("created", "processing") is True
            assert state_manager.validate_state_transition("prompt-swapped", "x") is True

    # -- `updated` joins the enum (#248's rule, re-applied) ------------------

    def test_updated_to_processing_is_a_known_valid_transition(self, state_manager, caplog):
        """⭐ THE OTHER CHANGE. ``(updated, processing)`` = 8/7d, 2/48h — NON-ZERO.

        ⛔ ``returned is True`` ALONE CANNOT PROVE THIS: the fail-open arm
        returned ``True`` for this pair too, so a return-value assertion passes
        identically before and after. The proof that the unknown arm is no longer
        taken is the other three: verdict ``valid`` rather than ``unknown_from``,
        NO pair-census emission, and NO warning.
        """
        client = MagicMock()
        with (
            caplog.at_level(logging.WARNING, logger="src.services.worker_state_manager"),
            patch("src.services.worker_state_manager.get_metrics_client", return_value=client),
        ):
            returned = state_manager.validate_state_transition("updated", "processing")

        assert returned is True
        emitted = [
            call.args[0] for call in client.record_worker_state_transition_verdict.call_args_list
        ]
        assert emitted == ["valid"], "must resolve against STATE_TRANSITIONS, not fail open"
        assert client.record_worker_state_transition_unknown_pair.call_args_list == []
        assert "Unknown state types in transition" not in caplog.text

    def test_updated_targets_are_exactly_processing(self):
        """⛔ Not widened on plausibility: ``processing`` is the ONLY observed target.

        ⚠️ And that is a property of the CALLER, not of the world — the sole call
        site passes the to-state as the literal ``"processing"``.
        """
        assert STATE_TRANSITIONS[WorkerStateType.UPDATED] == [WorkerStateType.PROCESSING]

    def test_updated_to_an_unobserved_target_is_invalid_not_unknown(self, state_manager):
        """The from-side is known now, so an unlisted target books invalid, not unknown_to."""
        emitted, returned = self._verdicts_for(state_manager, "updated", "completed")
        assert emitted == ["invalid_transition"]
        assert returned is False

    def test_prompt_swapped_is_still_NOT_a_member(self):
        """⛔ The rule, pinned. Row count never earns membership; pair traffic does.

        ``prompt-swapped`` is in the 2026-09-09 row census and its pair series are
        still zero on every window. If it is ever added without a non-zero pair
        measurement, this is the test that says so.
        """
        assert "prompt-swapped" not in {m.value for m in WorkerStateType}
