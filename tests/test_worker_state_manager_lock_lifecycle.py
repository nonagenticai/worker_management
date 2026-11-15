"""Lifecycle of ``WorkerStateManager._operation_locks``.

Two properties are pinned here and they pull in opposite directions:

1. **Nothing accumulates.** ``_operation_locks`` is an unbounded, process-global
   dict keyed by a caller-supplied path parameter
   (``GET /workers/{worker_name}/states`` -> :meth:`resolve_worker_id`, which
   runs BEFORE any 404 check). Before the refcounted lifecycle, one request per
   distinct name added one ``asyncio.Lock`` that was never removed.
2. **Mutual exclusion still holds.** The naive repair -- dropping the entry when
   an operation finishes -- lets a coroutine that is holding or awaiting the
   lock be bypassed by the next caller, who creates a fresh lock and walks
   straight into the critical section. That trades a memory leak for a
   correctness bug.

A test for (1) alone is satisfied by deleting the locking entirely, so (1) and
(2) must both be asserted, and the overlap detector used for (2) must be shown
to be falsifiable -- see ``test_overlap_detector_can_fail``.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.services.worker_state_manager import WorkerStateManager


@pytest.fixture
def state_manager():
    return WorkerStateManager()


class TestNoUnboundedGrowth:
    """Property 1: distinct keys must not accumulate entries."""

    @pytest.mark.asyncio
    async def test_distinct_workers_do_not_accumulate_locks(self, state_manager):
        """500 sequential operations on 500 distinct workers leave nothing behind."""
        for i in range(500):
            async with state_manager.atomic_worker_operation(f"t1:a{i:06d}", f"a{i:06d}", "op"):
                pass

        assert len(state_manager._operation_locks) == 0, (
            f"{len(state_manager._operation_locks)} lock entries survived 500 completed "
            "operations on distinct workers; the dict grows without bound"
        )

    @pytest.mark.asyncio
    async def test_resolve_worker_id_does_not_accumulate_locks(self, state_manager):
        """The reachable HTTP path: one lookup per caller-supplied worker name.

        ``GET /workers/{worker_name}/states`` reaches :meth:`resolve_worker_id`
        for any name matching ``^[a-zA-Z0-9_-]+$`` before it can 404, so a
        single token can mint one entry per request.
        """
        registry = AsyncMock()
        registry.get_worker = AsyncMock(return_value=None)

        for i in range(200):
            await state_manager.resolve_worker_id("t1", "s1", f"a{i:06d}", registry)

        assert len(state_manager._operation_locks) == 0, (
            f"{len(state_manager._operation_locks)} lock entries survived 200 resolutions of "
            "distinct, non-existent worker names"
        )

    @pytest.mark.asyncio
    async def test_entry_is_reaped_even_when_the_body_raises(self, state_manager):
        """A failing operation must not pin its entry."""
        for i in range(50):
            with pytest.raises(RuntimeError):
                async with state_manager.atomic_worker_operation(f"t1:b{i}", f"b{i}", "op"):
                    raise RuntimeError("boom")

        assert len(state_manager._operation_locks) == 0


class TestMutualExclusionStillHolds:
    """Property 2: the repair must not weaken the critical section."""

    @staticmethod
    async def _run_overlap_probe(enter_critical_section, n=50):
        """Run ``n`` bodies through ``enter_critical_section`` and report peak overlap.

        Each body yields to the event loop twice while "inside", so two bodies
        that are genuinely allowed to interleave WILL be observed overlapping.
        Without those yields the body is atomic under asyncio regardless of the
        lock, and the probe could never fail.
        """
        concurrent = 0
        peak = 0

        async def body(i):
            nonlocal concurrent, peak
            async with enter_critical_section(i):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                concurrent -= 1

        await asyncio.gather(*(body(i) for i in range(n)))
        return peak

    @pytest.mark.asyncio
    async def test_concurrent_same_key_never_overlaps(self, state_manager):
        """50 concurrent operations on ONE key serialize: peak overlap is exactly 1."""

        def enter(_i):
            return state_manager.atomic_worker_operation("t1:w1", "w1", "op")

        peak = await self._run_overlap_probe(enter)
        assert peak == 1, f"{peak} coroutines were inside the critical section at once"
        assert len(state_manager._operation_locks) == 0

    @pytest.mark.asyncio
    async def test_overlap_detector_can_fail(self):
        """Positive control: the SAME probe against no locking must report overlap.

        Without this, ``test_concurrent_same_key_never_overlaps`` would also
        pass against a fix that removed mutual exclusion altogether.
        """
        import contextlib

        @contextlib.asynccontextmanager
        async def unlocked(_i):
            yield

        peak = await self._run_overlap_probe(unlocked)
        assert peak > 1, "the overlap detector cannot observe overlap; it proves nothing"

    @pytest.mark.asyncio
    async def test_late_arrival_cannot_bypass_an_active_waiter(self, state_manager):
        """A caller arriving AFTER the first holder exits must still queue.

        ⭐ THIS IS THE TEST THAT KILLS THE NAIVE REPAIR. Simply dropping the
        entry when the operation that created it finishes passes every
        accumulation test in this file AND
        ``test_concurrent_same_key_never_overlaps`` -- the latter because
        ``asyncio.gather`` makes all its contenders arrive before anyone exits,
        so each already holds a reference to the one lock object and no
        newcomer ever consults the dict.

        The defect needs a STAGGERED arrival: holder exits and removes the
        entry while a waiter is still inside the critical section, so the next
        caller finds an empty dict, mints a fresh lock and enters beside it.
        """
        inside = 0
        peak = 0
        holder_in, holder_go = asyncio.Event(), asyncio.Event()
        waiter_in, waiter_go = asyncio.Event(), asyncio.Event()

        async def critical(entered: asyncio.Event | None, go: asyncio.Event | None):
            nonlocal inside, peak
            async with state_manager.atomic_worker_operation("t1:w1", "w1", "op"):
                inside += 1
                peak = max(peak, inside)
                if entered is not None:
                    entered.set()
                if go is not None:
                    await go.wait()
                else:
                    await asyncio.sleep(0)
                inside -= 1

        holder = asyncio.create_task(critical(holder_in, holder_go))
        await holder_in.wait()

        waiter = asyncio.create_task(critical(waiter_in, waiter_go))
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # waiter is now queued inside lock.acquire()

        holder_go.set()
        await waiter_in.wait()  # holder has exited; waiter is INSIDE

        latecomer = asyncio.create_task(critical(None, None))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        waiter_go.set()
        await asyncio.gather(holder, waiter, latecomer)

        assert peak == 1, (
            f"{peak} coroutines were inside the critical section at once: a late arrival "
            "was handed a different lock object and bypassed an active waiter"
        )
        assert len(state_manager._operation_locks) == 0

    @pytest.mark.asyncio
    async def test_distinct_keys_still_run_in_parallel(self, state_manager):
        """The repair must not collapse per-worker locks into one global lock."""

        def enter(i):
            return state_manager.atomic_worker_operation(f"t1:w{i}", f"w{i}", "op")

        peak = await self._run_overlap_probe(enter, n=10)
        assert peak > 1, "operations on DISTINCT workers serialized; locking is too coarse"
        assert len(state_manager._operation_locks) == 0

    @pytest.mark.asyncio
    async def test_waiters_share_the_holders_lock_object(self, state_manager):
        """While one caller holds and others wait, there is exactly ONE entry.

        This is the property the naive "delete after use" repair breaks: a
        waiter must never be handed a different lock object than the holder.
        """
        started = asyncio.Event()
        release = asyncio.Event()
        entries_seen = []

        async def holder():
            async with state_manager.atomic_worker_operation("t1:w1", "w1", "op"):
                started.set()
                await release.wait()

        async def waiter():
            async with state_manager.atomic_worker_operation("t1:w1", "w1", "op"):
                pass

        h = asyncio.create_task(holder())
        await started.wait()
        ws = [asyncio.create_task(waiter()) for _ in range(3)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Holder + 3 waiters, one key => one entry, and it must still be there.
        entries_seen.append(dict(state_manager._operation_locks))
        assert len(entries_seen[0]) == 1, (
            "a contended key must map to exactly one entry while it is contended"
        )

        release.set()
        await asyncio.gather(h, *ws)
        assert len(state_manager._operation_locks) == 0

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_corrupt_the_refcount(self, state_manager):
        """A waiter cancelled mid-await must decrement, not strand or double-free."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with state_manager.atomic_worker_operation("t1:w1", "w1", "op"):
                started.set()
                await release.wait()

        async def waiter():
            async with state_manager.atomic_worker_operation("t1:w1", "w1", "op"):
                pass

        h = asyncio.create_task(holder())
        await started.wait()
        doomed = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed

        release.set()
        await h
        assert len(state_manager._operation_locks) == 0

        # And the lock must still work afterwards.
        def enter(_i):
            return state_manager.atomic_worker_operation("t1:w1", "w1", "op")

        peak = await TestMutualExclusionStillHolds._run_overlap_probe(enter, n=10)
        assert peak == 1
        assert len(state_manager._operation_locks) == 0
