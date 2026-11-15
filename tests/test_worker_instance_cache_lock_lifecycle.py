"""Lifecycle of ``WorkerInstanceCache._worker_locks``.

Three properties are pinned here, and the first two pull in opposite directions:

1. **Nothing accumulates.** ``_worker_locks`` is a process-global dict whose keys
   are CALLER-ENUMERABLE: :func:`_cache_key` folds in ``override_key``, a
   fingerprint of per-message fields the caller supplies (``system_prompt``,
   ``instruction``, ``output_schema``, ``output_validator``, ``model_override``,
   ``max_tokens``, ``enable_reasoning``). Before the refcounted lifecycle, one
   request per distinct override minted one ``asyncio.Lock`` that was never
   removed.
2. **Mutual exclusion still holds.** The naive repair -- dropping the entry when
   the operation that created it finishes -- lets a coroutine that is holding or
   awaiting the lock be bypassed by the next caller, who mints a fresh lock and
   walks into the critical section. For this cache the critical section is
   create-or-reuse-or-evict of one instance slot, so that trades a memory leak
   for duplicate instantiation and eviction of a live instance.
3. **All four public entry points share ONE lock per key.**
   ``get_or_create_worker``, ``remove_worker``, ``remove_instance`` and
   ``invalidate_tenant`` must mutually exclude. Three of them used to read the
   dict with a raw ``.get()`` and fall back to an unlocked cleanup; once entries
   are reaped that fallback would be the only branch ever taken.

A test for (1) alone is satisfied by deleting the locking entirely, so (1) and
(2) must both be asserted, and the overlap detector used for (2) must be shown
to be falsifiable -- see ``test_overlap_detector_can_fail``.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.workers._globals import WorkerInstanceCache


@pytest.fixture
def cache():
    return WorkerInstanceCache()


async def _make_fake_worker(*_args, **_kwargs):
    """``_create_worker_instance`` stand-in: a fresh instance per call."""
    return _fake_worker()


def _fake_worker():
    """A worker object the cache will treat as healthy and cleanable."""
    worker = MagicMock()
    worker.cleanup = AsyncMock()
    worker.check_health = AsyncMock(return_value=True)
    del worker.config  # no MCP-server branch
    return worker


class TestNoUnboundedGrowth:
    """Property 1: distinct keys must not accumulate entries."""

    @pytest.mark.asyncio
    async def test_distinct_override_keys_do_not_accumulate_locks(self, cache, monkeypatch):
        """500 creates on 500 caller-minted override keys leave nothing behind.

        This is the reachable HTTP shape: ``POST /workers/{name}/message`` with a
        per-request ``system_prompt`` produces a distinct ``override_key``, hence
        a distinct cache key, hence (before the fix) a permanent lock.
        """
        monkeypatch.setattr(
            cache, "_create_worker_instance", AsyncMock(side_effect=_make_fake_worker)
        )
        for i in range(500):
            await cache.get_or_create_worker(
                "w1",
                {"worker_name": "w1", "config": {}},
                tenant_id="t1",
                override_key=f"{i:016x}",
            )

        assert len(cache._worker_locks) == 0, (
            f"{len(cache._worker_locks)} lock entries survived 500 creates on distinct "
            "caller-supplied override keys; the dict grows without bound"
        )
        # The instance slots themselves are a SEPARATE, still-unbounded leak;
        # this fix deliberately does not close it. Pin that so a later reader
        # cannot mistake a bounded lock dict for a bounded cache.
        assert len(cache._workers) == 500

    @pytest.mark.asyncio
    async def test_entry_is_reaped_even_when_creation_raises(self, cache, monkeypatch):
        """A failing create must not pin its entry."""
        monkeypatch.setattr(
            cache, "_create_worker_instance", AsyncMock(side_effect=RuntimeError("boom"))
        )
        for i in range(50):
            with pytest.raises(RuntimeError):
                await cache.get_or_create_worker(
                    "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1", override_key=str(i)
                )

        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_remove_worker_does_not_accumulate_locks(self, cache):
        for i in range(100):
            cache._workers[f"t1:w1:_:ov-{i}"] = _fake_worker()
        await cache.remove_worker("w1", tenant_id="t1")
        assert cache._workers == {}
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_remove_instance_does_not_accumulate_locks(self, cache):
        for i in range(100):
            key = f"t1:w1:_:ov-{i}"
            cache._workers[key] = _fake_worker()
            await cache.remove_instance(
                "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1", override_key=str(i)
            )
        assert cache._workers == {}
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_invalidate_tenant_does_not_accumulate_locks(self, cache):
        for i in range(100):
            cache._workers[f"t1:w{i}:_"] = _fake_worker()
        await cache.invalidate_tenant("t1")
        assert cache._workers == {}
        assert len(cache._worker_locks) == 0


class TestEveryPublicPathTakesTheSameKeyLock:
    """Property 3: all four entry points block on the ONE entry for their key.

    ``remove_worker`` used to skip any key with no lock entry, and
    ``remove_instance`` / ``invalidate_tenant`` used to fall through to an
    UNLOCKED ``_cleanup_worker``. With reaped entries "no lock entry" is the
    resting state of every key, so these pin that eviction still happens AND
    still happens under the lock a concurrent create holds.

    Each test holds the key's entry itself and asserts the public method cannot
    make progress -- a method that reverted to a raw ``.get()`` would find the
    entry (it exists, we are holding it) but a method that reverted to the
    unlocked fallback, or that skipped lockless keys, completes immediately and
    fails here.
    """

    @pytest.mark.asyncio
    async def test_remove_worker_evicts_a_key_that_has_no_pre_existing_lock(self, cache):
        worker = _fake_worker()
        cache._workers["t1:w1:_"] = worker
        assert cache._worker_locks == {}  # resting state after the fix

        await cache.remove_worker("w1", tenant_id="t1")

        assert "t1:w1:_" not in cache._workers, (
            "remove_worker skipped a slot because it had no lock entry; after reaping "
            "that is every slot, so the method would evict nothing at all"
        )
        worker.cleanup.assert_awaited_once()

    @staticmethod
    async def _assert_blocks_until_released(cache, key, coro_factory, *, evicts=True):
        """Run ``coro_factory()`` while ``key``'s lock is held; it must not finish."""
        cache._workers[key] = _fake_worker()
        async with cache._held_lock(key):
            assert cache._worker_locks[key].users == 1
            task = asyncio.create_task(coro_factory())
            for _ in range(6):
                await asyncio.sleep(0)
            assert not task.done(), (
                f"{coro_factory} completed while the key's lock was held; it is not "
                "routing through the shared per-key lock"
            )
            # The blocked caller must be COUNTED, or the entry could be reaped
            # out from under it when we release.
            assert cache._worker_locks[key].users == 2, (
                f"users={cache._worker_locks[key].users}; the waiter was not counted"
            )
        await task
        if evicts:
            assert key not in cache._workers
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_remove_worker_blocks_on_the_shared_key_lock(self, cache):
        await self._assert_blocks_until_released(
            cache, "t1:w1:_", lambda: cache.remove_worker("w1", tenant_id="t1")
        )

    @pytest.mark.asyncio
    async def test_remove_instance_blocks_on_the_shared_key_lock(self, cache):
        await self._assert_blocks_until_released(
            cache,
            "t1:w1:_",
            lambda: cache.remove_instance(
                "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1"
            ),
        )

    @pytest.mark.asyncio
    async def test_invalidate_tenant_blocks_on_the_shared_key_lock(self, cache):
        await self._assert_blocks_until_released(
            cache, "t1:w1:_", lambda: cache.invalidate_tenant("t1")
        )

    @staticmethod
    async def _assert_evictor_excludes_a_later_create(cache, evict_factory):
        """An eviction that starts from the RESTING state must still exclude a create.

        ⭐ THIS IS THE TEST THAT KILLS THE ``.get()``-OR-UNLOCKED REPAIR, and the
        sibling ``_assert_blocks_until_released`` probes do NOT: those hold the
        key's entry themselves, so a mutant that locks only "if an entry already
        exists" finds one and looks correct. The defect lives in the opposite
        case -- the resting state, where reaping means NO entry exists, which
        after this fix is every key that is not actively in use.

        Staged here: the evictor is mid-``worker.cleanup()`` (a real suspension
        point) when a create for the same key arrives. Unlocked, the create
        completes and installs a replacement, and then ``_cleanup_worker``'s
        ``finally: del self._workers[key]`` deletes that brand-new instance --
        the cache is left EMPTY and the creator is holding an object nothing
        will ever reuse.
        """
        key = "t1:w1:_"
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def slow_cleanup():
            cleanup_started.set()
            await finish_cleanup.wait()

        doomed = _fake_worker()
        doomed.cleanup = AsyncMock(side_effect=slow_cleanup)
        cache._workers[key] = doomed
        assert cache._worker_locks == {}, "precondition: the resting state has no entry"

        replacement = _fake_worker()
        evictor = asyncio.create_task(evict_factory())
        # Bounded wait: an evictor that skips the slot entirely (e.g. the old
        # ``if cache_key in self._worker_locks`` guard, which after reaping
        # matches nothing) never reaches ``cleanup()``, and an unbounded wait
        # here would hang the suite instead of naming the defect.
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        except TimeoutError:
            evictor.cancel()
            pytest.fail(
                "the evictor never reached worker.cleanup(): it skipped the slot, so the "
                "instance was never evicted at all"
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cache, "_create_worker_instance", AsyncMock(return_value=replacement))
            creator = asyncio.create_task(
                cache.get_or_create_worker(
                    "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1"
                )
            )
            for _ in range(6):
                await asyncio.sleep(0)
            assert not creator.done(), (
                "a create completed while an eviction was in flight: the eviction took no "
                "lock because the key had no entry, which after reaping is the normal state"
            )

            finish_cleanup.set()
            await asyncio.gather(evictor, creator)

        assert cache._workers.get(key) is replacement, (
            "the eviction deleted the replacement a concurrent create had installed"
        )
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_remove_instance_excludes_a_create_that_arrives_mid_eviction(self, cache):
        await self._assert_evictor_excludes_a_later_create(
            cache,
            lambda: cache.remove_instance(
                "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1"
            ),
        )

    @pytest.mark.asyncio
    async def test_invalidate_tenant_excludes_a_create_that_arrives_mid_eviction(self, cache):
        await self._assert_evictor_excludes_a_later_create(
            cache, lambda: cache.invalidate_tenant("t1")
        )

    @pytest.mark.asyncio
    async def test_remove_worker_excludes_a_create_that_arrives_mid_eviction(self, cache):
        await self._assert_evictor_excludes_a_later_create(
            cache, lambda: cache.remove_worker("w1", tenant_id="t1")
        )

    @pytest.mark.asyncio
    async def test_get_or_create_worker_blocks_on_the_shared_key_lock(self, cache, monkeypatch):
        monkeypatch.setattr(
            cache, "_create_worker_instance", AsyncMock(side_effect=_make_fake_worker)
        )
        await self._assert_blocks_until_released(
            cache,
            "t1:w1:_",
            lambda: cache.get_or_create_worker(
                "w1", {"worker_name": "w1", "config": {}}, tenant_id="t1"
            ),
            evicts=False,
        )


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
    async def test_concurrent_same_key_never_overlaps(self, cache):
        peak = await self._run_overlap_probe(lambda _i: cache._held_lock("k1"))
        assert peak == 1, f"{peak} coroutines were inside the critical section at once"
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_overlap_detector_can_fail(self):
        """Positive control: the SAME probe against no locking must report overlap.

        Without this, ``test_concurrent_same_key_never_overlaps`` would also pass
        against a "fix" that removed mutual exclusion altogether.
        """

        @contextlib.asynccontextmanager
        async def unlocked(_i):
            yield

        peak = await self._run_overlap_probe(unlocked)
        assert peak > 1, "the overlap detector cannot observe overlap; it proves nothing"

    @pytest.mark.asyncio
    async def test_distinct_keys_still_run_in_parallel(self, cache):
        """The repair must not collapse per-key locks into one global lock."""
        peak = await self._run_overlap_probe(lambda i: cache._held_lock(f"k{i}"), n=10)
        assert peak > 1, "operations on DISTINCT keys serialized; locking is too coarse"
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_late_arrival_cannot_bypass_an_active_waiter(self, cache):
        """A caller arriving AFTER the first holder exits must still queue.

        ⭐ THIS IS THE TEST THAT KILLS THE NAIVE REPAIR. Dropping the entry when
        the operation that created it finishes passes every accumulation test in
        this file AND ``test_concurrent_same_key_never_overlaps`` -- the latter
        because ``asyncio.gather`` makes all its contenders arrive before anyone
        exits, so each already holds a reference to the one lock object and no
        newcomer ever consults the dict.

        The defect needs a STAGGERED arrival: the holder exits and drops the
        entry while a waiter is still inside the critical section, so the next
        caller finds an empty dict, mints a fresh lock and enters beside it.
        """
        inside = 0
        peak = 0
        holder_in, holder_go = asyncio.Event(), asyncio.Event()
        waiter_in, waiter_go = asyncio.Event(), asyncio.Event()

        async def critical(entered: asyncio.Event | None, go: asyncio.Event | None):
            nonlocal inside, peak
            async with cache._held_lock("k1"):
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
        for _ in range(3):
            await asyncio.sleep(0)  # waiter is now queued inside lock.acquire()

        holder_go.set()
        await waiter_in.wait()  # holder has exited; waiter is INSIDE

        latecomer = asyncio.create_task(critical(None, None))
        for _ in range(4):
            await asyncio.sleep(0)

        waiter_go.set()
        await asyncio.gather(holder, waiter, latecomer)

        assert peak == 1, (
            f"{peak} coroutines were inside the critical section at once: a late arrival "
            "was handed a different lock object and bypassed an active waiter"
        )
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_waiters_share_the_holders_lock_object(self, cache):
        """While one caller holds and others wait, there is exactly ONE entry."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with cache._held_lock("k1"):
                started.set()
                await release.wait()

        async def waiter():
            async with cache._held_lock("k1"):
                pass

        h = asyncio.create_task(holder())
        await started.wait()
        ws = [asyncio.create_task(waiter()) for _ in range(3)]
        for _ in range(3):
            await asyncio.sleep(0)

        assert len(cache._worker_locks) == 1, (
            "a contended key must map to exactly one entry while it is contended"
        )
        assert cache._worker_locks["k1"].users == 4, (
            f"users={cache._worker_locks['k1'].users}; the holder plus 3 waiters must all "
            "be counted, or the entry can be reaped out from under a waiter"
        )

        release.set()
        await asyncio.gather(h, *ws)
        assert len(cache._worker_locks) == 0


class TestCancellation:
    """A cancelled caller must leave no entry, and must not strand a peer."""

    @pytest.mark.asyncio
    async def test_cancelled_holder_leaves_no_entry(self, cache):
        started = asyncio.Event()

        async def holder():
            async with cache._held_lock("k1"):
                started.set()
                await asyncio.Event().wait()  # never fires

        h = asyncio.create_task(holder())
        await started.wait()
        h.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h

        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_cancelled_holder_with_a_queued_waiter(self, cache):
        """Cancelling the holder must hand the lock to the waiter, then reap at zero."""
        started = asyncio.Event()
        waiter_ran = asyncio.Event()

        async def holder():
            async with cache._held_lock("k1"):
                started.set()
                await asyncio.Event().wait()

        async def waiter():
            async with cache._held_lock("k1"):
                waiter_ran.set()

        h = asyncio.create_task(holder())
        await started.wait()
        w = asyncio.create_task(waiter())
        for _ in range(3):
            await asyncio.sleep(0)
        assert cache._worker_locks["k1"].users == 2

        h.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h
        await w

        assert waiter_ran.is_set(), "the waiter was stranded by the holder's cancellation"
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_corrupt_the_refcount(self, cache):
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with cache._held_lock("k1"):
                started.set()
                await release.wait()

        async def waiter():
            async with cache._held_lock("k1"):
                pass

        h = asyncio.create_task(holder())
        await started.wait()
        doomed = asyncio.create_task(waiter())
        for _ in range(3):
            await asyncio.sleep(0)
        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed

        release.set()
        await h
        assert len(cache._worker_locks) == 0

        # And the lock must still work afterwards.
        peak = await TestMutualExclusionStillHolds._run_overlap_probe(
            lambda _i: cache._held_lock("k1"), n=10
        )
        assert peak == 1
        assert len(cache._worker_locks) == 0

    @pytest.mark.asyncio
    async def test_timeout_on_a_waiter_leaves_no_entry(self, cache):
        """``asyncio.timeout`` around a contended acquire must still decrement."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with cache._held_lock("k1"):
                started.set()
                await release.wait()

        h = asyncio.create_task(holder())
        await started.wait()

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01), cache._held_lock("k1"):
                pass  # pragma: no cover - unreachable, the acquire times out

        assert cache._worker_locks["k1"].users == 1, (
            "the timed-out waiter did not decrement; its reference is stranded forever"
        )

        release.set()
        await h
        assert len(cache._worker_locks) == 0
