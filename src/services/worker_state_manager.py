"""Worker state management service.

Provides state transition validation and per-worker operation locking to ensure
atomic operations across concurrent requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

import psycopg

from src.telemetry import get_metrics_client
from src.telemetry.agent_metrics import (
    WORKER_STATE_VERDICT_INVALID,
    WORKER_STATE_VERDICT_UNKNOWN_BOTH,
    WORKER_STATE_VERDICT_UNKNOWN_FROM,
    WORKER_STATE_VERDICT_UNKNOWN_TO,
    WORKER_STATE_VERDICT_VALID,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.db.worker_registry_db import WorkerRegistryDB

logger = logging.getLogger(__name__)


# ⚠️ THIS ENUM IS STILL NARROWER THAN THE SET OF VALUES THAT REACHES THIS MODULE.
# ``worker_states.state_type`` is ``text NOT NULL`` with no CHECK, so the column
# holds whatever a caller writes. A live census (1,092,122 rows, 2026-09-09)
# found three non-members: ``config-updated`` (153,087 rows — note the HYPHEN),
# ``updated`` (3,448) and ``prompt-swapped``.
#
# ``config-updated`` is a member as of 2026-09-13, and that was step 2 of the
# sanctioned order counter -> enum -> decide -> close-fail-open-LAST. The
# evidence step is now CLOSED: over 24h the pair census below carried 72 series
# and EXACTLY ONE was non-zero — ``(config-updated, processing)`` at 4,615 of
# 4,616 unknown verdicts, 99.98%, against a denominator of 8,444 calls
# (unknown_from 4,616 · valid 3,378 · invalid_transition 450). Its
# ``STATE_TRANSITIONS`` entry is therefore ``[PROCESSING]`` and nothing else.
#
# ⚠️ "Only one target observed" IS A PROPERTY OF THE CALLER, NOT OF THE WORLD.
# The sole caller, ``src.api.workers.message_endpoints``, passes the to-state as
# the literal ``"processing"``, so the census was STRUCTURALLY INCAPABLE of
# observing any other target for any from-state. Read the entry as "the only
# target reachable through today's one call site". A second call site would
# invalidate that reading, not merely add to it.
#
# ⚠️ ``updated`` WAS IN THAT PROHIBITION AND IS NOW A MEMBER (2026-09-17). The
# text here used to read: *"``updated`` and ``prompt-swapped`` are DELIBERATELY
# ABSENT ... their pair series are ZERO on every one of the 72"*. Re-measured on
# ``worker_management_worker_state_transition_unknown_pairs_total``:
#
#     (updated, processing)   8 over 7d,   2 over 48h
#
# NON-ZERO, on both windows. The old sentence was true when it was written and
# is false now, and the rule it stated — *"membership is earned by non-zero PAIR
# traffic, never by a row count"* — is the rule that PROMOTES it. That rule is
# unchanged and is restated below; only the measurement moved.
#
# ⚠️ 8/7d IS SMALL, AND SMALL IS STILL NON-ZERO. The bar this enum uses is
# "observed transitioning", not "observed often": a pair that reaches the
# fail-open arm at all is a pair the state machine has no rule for, and the
# volume only decides how loud the finding is, never whether it is one. The
# entry is ``[PROCESSING]`` and nothing else, for exactly the reason
# ``CONFIG_UPDATED``'s is — and with the same ⚠️ above it: the sole call site
# passes the to-state as the literal ``"processing"``, so the census was
# STRUCTURALLY INCAPABLE of observing another target. Read it as "the only
# target reachable through today's one call site".
#
# ⛔ ``prompt-swapped`` REMAINS DELIBERATELY ABSENT, and it is the whole reason
# this is a rule rather than a habit. It is in the ROW census and its pair series
# are still ZERO on every window — a row proves a value was once persisted, not
# that a transition out of it ever reached this function. Membership is earned by
# non-zero PAIR traffic, never by a row count. Adding it would only widen the
# STRICT enum in :mod:`src.api.worker_states`, whose ``state_type`` field turns
# an unlisted value into a 422.
#
# ⛔ THE FAIL-OPEN IN ``validate_state_transition`` IS STILL OPEN, and closing it
# is the remaining step — not this one. But do not read the ~142,560 / 30 days
# figure quoted elsewhere in this repo as the cost of closing it, for TWO
# independent reasons.
#
# ⭐ 1. THIS CHANGE ALREADY REMOVED ~99.98% OF IT. That figure counts the
# fail-open population, and the unknown arm carried ~4,616 verdicts/24h ≈
# 138,480/30d — essentially all of it. 4,615 of those 4,616 were
# ``(config-updated, processing)``, which now resolves through
# ``STATE_TRANSITIONS`` and books ``valid``. Residual traversals of the arm are
# on the order of ~30/30d. ⛔ Do NOT carry the pre-change number forward into
# the ``decide`` rung: the population it described is gone. (The two counts come
# from different windows, so treat ≈138,480-of-142,560 as an order-of-magnitude
# match, not an identity.)
#
# ⚠️ 2. IT IS NOT A REJECTION COUNT, and the word "rejected" attached to it in
# five other places in this repo is wrong. NOTHING IS REFUSED TODAY: the sole
# caller does ``if current_state and not validate_state_transition(...)`` and
# the body is a ``logger.warning`` — the ``add_worker_state`` write that follows
# is OUTSIDE that branch and runs unconditionally. So the figure counts
# TRAVERSALS of the fail-open arm, not writes that would have been refused.
# ⛔ The ``decide`` rung must re-derive the real blast radius from the caller's
# behaviour before closing the arm; a traversal count is an upper bound on
# nothing, because closing the arm only changes a return value that this caller
# currently discards.
#
# ⚠️ Unexplained, recorded rather than smoothed over: the verdict counter read
# 4,616 unknown while the pair counter read 4,615 emissions on the same window.
# HYPOTHESIS ONLY — ``increase()`` extrapolation across scrape boundaries. It
# has not been confirmed, and a real off-by-one in the pair arm is not excluded.
class WorkerStateType(str, Enum):
    """Valid worker state types with transition rules."""

    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    IDLE = "idle"
    ACTIVE = "active"
    DEFAULT = "default"
    CONFIG_UPDATED = "config-updated"
    UPDATED = "updated"


# Valid state transitions
STATE_TRANSITIONS: dict[WorkerStateType, list[WorkerStateType]] = {
    WorkerStateType.CREATED: [
        WorkerStateType.PROCESSING,
        WorkerStateType.IDLE,
        WorkerStateType.ERROR,
    ],
    WorkerStateType.PROCESSING: [
        WorkerStateType.COMPLETED,
        WorkerStateType.ERROR,
        WorkerStateType.IDLE,
    ],
    WorkerStateType.COMPLETED: [WorkerStateType.PROCESSING, WorkerStateType.IDLE],
    WorkerStateType.ERROR: [
        WorkerStateType.PROCESSING,
        WorkerStateType.IDLE,
        WorkerStateType.CREATED,
    ],
    WorkerStateType.IDLE: [WorkerStateType.PROCESSING, WorkerStateType.ACTIVE],
    WorkerStateType.ACTIVE: [
        WorkerStateType.PROCESSING,
        WorkerStateType.IDLE,
        WorkerStateType.COMPLETED,
    ],
    WorkerStateType.DEFAULT: [WorkerStateType.PROCESSING, WorkerStateType.IDLE],
    # ⭐ [PROCESSING] and nothing else — the 24h pair census found exactly one
    # non-zero target for this state. Do not widen it on plausibility.
    WorkerStateType.CONFIG_UPDATED: [WorkerStateType.PROCESSING],
    # ⭐ Same rule, same evidence shape, three orders of magnitude smaller:
    # ``(updated, processing)`` = 8/7d, 2/48h, and no other target on any
    # window. Do not widen it on plausibility either.
    WorkerStateType.UPDATED: [WorkerStateType.PROCESSING],
}


@dataclass
class WorkerLockInfo:
    """Information about worker locks for coordination."""

    worker_id: str
    worker_name: str
    lock_hash: str
    timestamp: datetime
    operation: str


@dataclass
class _LockEntry:
    """One per-``(worker_id, operation)`` lock plus its in-flight user count.

    ``users`` counts the current holder AND everyone queued behind it, so it is
    non-zero for exactly as long as the entry is load-bearing.
    """

    lock: asyncio.Lock
    users: int = 0


class WorkerStateManager:
    """Worker state management with per-``(worker_id, operation)`` locking.

    ⛔ NOT THREAD-SAFE — this docstring used to claim it was, which is the
    dangerous kind of wrong: it would license someone to call this from a
    :class:`~concurrent.futures.ThreadPoolExecutor` or a second event loop, and
    nothing here would stop them. The real guarantee is **single-event-loop
    atomicity**. ``_operation_locks`` is a plain :class:`dict` and
    :class:`asyncio.Lock` is not thread-safe; what makes
    :meth:`_held_lock` correct is that its get/create/increment block contains
    no ``await``, so no other coroutine on the same loop can be scheduled
    part-way through it. Under the single uvicorn event loop this service runs
    on today, that is sufficient.

    What would break it: running two :class:`WorkerStateManager` callers on
    different threads or different event loops (``asyncio.to_thread``, a
    ``run_in_executor`` hop, ``asyncio.run`` on a worker thread), or inserting
    an ``await`` into that block. Either makes ``users += 1`` a read-modify-write
    that can lose an update, and the entry can then be reaped while still held.
    If threads ever become real here, the fix is a ``threading.Lock`` around the
    dict plus a lock primitive that is valid across loops — not a wider
    ``asyncio.Lock``, which is not one.
    """

    def __init__(self) -> None:
        self._operation_locks: dict[str, _LockEntry] = {}
        self._lock_info: dict[str, WorkerLockInfo] = {}

    def _get_lock_key(self, worker_id: str, operation: str) -> str:
        """Generate a unique lock key for worker operation."""
        return f"{worker_id}:{operation}"

    @asynccontextmanager
    async def _held_lock(self, worker_id: str, operation: str) -> AsyncIterator[asyncio.Lock]:
        """Hold the per-``(worker_id, operation)`` lock, reaping it when idle.

        ⭐ REFERENCE COUNTED, AND IT HAS TO BE. ``_operation_locks`` is a
        process-global dict keyed by a caller-supplied path parameter:
        ``GET /workers/{worker_name}/states`` reaches :meth:`resolve_worker_id`
        BEFORE any 404 check, so one token walking ``a000001``…``a999999`` used
        to mint one permanent :class:`asyncio.Lock` per request — monotonic RSS
        growth until the pod is OOM-killed.

        ⛔ DO NOT "simplify" this to deleting the entry when an operation
        finishes. A coroutine may be holding this lock, and others may be queued
        inside ``lock.acquire()``; dropping the entry lets the next caller
        create a FRESH lock for the same key and enter the critical section
        alongside them. That converts a memory leak into a lost-update bug,
        which is strictly worse. ``users`` is what makes removal safe: it counts
        the holder plus every waiter, and the entry is removed only at zero —
        at which point the lock is provably unlocked, unwaited, and carries no
        state, so a later caller re-creating it is indistinguishable.

        Reference counting is used rather than a
        :class:`weakref.WeakValueDictionary` (whose eviction would depend on
        every caller's reference discipline, and on GC timing, neither of which
        is assertable) or an LRU (which is still unbounded while entries are
        held, and whose eviction of an "idle" lock races anyone who has already
        been handed it).

        The get/create/increment below contains no ``await``, so under asyncio
        it is atomic against other coroutines on the same loop: two callers
        cannot race into creating two entries for one key. The decrement is in
        ``finally`` so it survives the body raising, the acquire being
        cancelled, and the task being cancelled while holding.
        """
        lock_key = self._get_lock_key(worker_id, operation)
        entry = self._operation_locks.get(lock_key)
        if entry is None:
            entry = _LockEntry(lock=asyncio.Lock())
            self._operation_locks[lock_key] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield entry.lock
        finally:
            entry.users -= 1
            # ``is entry`` asserts an invariant rather than preventing a race:
            # no successor can be installed under this key while ``users > 0``,
            # because a newcomer that finds this entry increments it instead of
            # replacing it, and the only removal is this branch at zero. So the
            # check is UNFALSIFIABLE as the code stands -- mutating it away does
            # not fail any test, and should not be expected to. It is kept
            # because it is free, and because it fails loudly-by-omission the
            # day the invariant stops holding: a second removal path, an entry
            # replaced in place rather than incremented, or an ``await`` between
            # the decrement and the delete (which would let a reaper and a
            # newcomer interleave) all make a successor reachable here.
            if entry.users <= 0 and self._operation_locks.get(lock_key) is entry:
                del self._operation_locks[lock_key]

    def validate_state_transition(self, current_state: str, new_state: str) -> bool:
        """Validate if state transition is allowed based on STATE_TRANSITIONS.

        ⛔ FAILS OPEN, DELIBERATELY, AND MUST CONTINUE TO. An unparseable state
        on either side logs a warning and returns ``True`` — and every caller
        treats a ``False`` as a warning too, never a refusal. Making this
        reject was measured at ~142,560 rejected invocations over 30 days,
        ≈55% of all traffic. The counter below is the whole change: it makes
        the verdict distribution readable so that decision can eventually be
        taken on data. It does not alter a single return value.
        """
        # Resolve each side independently so the fail-open path can say WHICH
        # side was unknown. Previously the first ValueError short-circuited both.
        try:
            current: WorkerStateType | None = WorkerStateType(current_state)
        except ValueError:
            current = None
        try:
            new: WorkerStateType | None = WorkerStateType(new_state)
        except ValueError:
            new = None

        if current is None or new is None:
            if current is None and new is None:
                verdict = WORKER_STATE_VERDICT_UNKNOWN_BOTH
            elif current is None:
                verdict = WORKER_STATE_VERDICT_UNKNOWN_FROM
            else:
                verdict = WORKER_STATE_VERDICT_UNKNOWN_TO
            self._record_transition_verdict(verdict)
            # ⭐ And the (from, to) PAIR, which the verdict alone cannot carry.
            # Observe-only, on this arm only, and placed BEFORE the untouched
            # warning+return below so no control-flow edge moves.
            self._record_transition_unknown_pair(current_state, new_state)
            # If state types are not in enum, allow transition but log warning
            logger.warning("Unknown state types in transition: %s -> %s", current_state, new_state)
            return True

        allowed = new in STATE_TRANSITIONS.get(current, [])
        self._record_transition_verdict(
            WORKER_STATE_VERDICT_VALID if allowed else WORKER_STATE_VERDICT_INVALID
        )
        if not allowed:
            # ⭐ And the (from, to) PAIR on the REFUSED arm, which the verdict
            # alone cannot carry — the same gap the unknown arm's census closed,
            # on an arm that is 10.7% of all traffic (invalid_transition 1,753 vs
            # valid 14,656 over 24h) and has never been broken down at all.
            #
            # ⛔ COUNTER ONLY. Placed AFTER the verdict and BEFORE the untouched
            # `return allowed`, so no control-flow edge moves, nothing is raised,
            # and the fail-open above stays open. Closing it is a separate rung
            # and a cheap one to defer: the sole caller
            # (`message_endpoints.py:569`) discards this bool and writes the row
            # unconditionally, so a refusal here would change nothing today.
            self._record_transition_invalid_pair(current_state, new_state)
        return allowed

    @staticmethod
    def _record_transition_verdict(verdict: str) -> None:
        """Emit one transition verdict. Swallows everything — observe-only.

        Instrumentation must never be able to turn a state write into a failure;
        that is the whole point of a change that is forbidden to alter
        behaviour.
        """
        try:
            get_metrics_client().record_worker_state_transition_verdict(verdict)
        except Exception:  # pragma: no cover - defensive; observe-only
            logger.debug("worker state transition verdict counter failed", exc_info=True)

    @staticmethod
    def _record_transition_unknown_pair(current_state: str, new_state: str) -> None:
        """Emit one ``(from, to)`` pair for the fail-open arm. Observe-only.

        ⭐ THIS IS THE PRECONDITION FOR EVER ADDING TO :class:`WorkerStateType`.
        The verdict counter says how many calls fell through; it does not say
        WHICH transition did, because its labels are verdict-only. Adding
        ``config-updated``/``updated`` as members turns every one of those calls
        from a fail-open ``True`` into a ``STATE_TRANSITIONS`` lookup, so any
        observed pair without an entry silently becomes ``False``. Until this
        census is populated that risk is unmeasured — which is exactly why the
        enum is NOT being changed in the same commit that adds this.

        Swallows everything, for the same reason
        :meth:`_record_transition_verdict` does: instrumentation must never be
        able to turn a state write into a failure.
        """
        try:
            get_metrics_client().record_worker_state_transition_unknown_pair(
                current_state, new_state
            )
        except Exception:  # pragma: no cover - defensive; observe-only
            logger.debug("worker state transition pair counter failed", exc_info=True)

    @staticmethod
    def _record_transition_invalid_pair(current_state: str, new_state: str) -> None:
        """Emit one ``(from, to)`` pair for the REFUSED arm. Observe-only.

        The sibling of :meth:`_record_transition_unknown_pair`, on the arm that
        one is forbidden to fire on. Both sides resolved against the enum here,
        so the pair is guaranteed to be labellable — no ``other``/``none``
        sentinel is reachable from this call site.

        ⭐ WHY IT EXISTS. ``invalid_transition`` is 1,753 of 16,409 verdicts over
        24h — 10.7% of traffic the state machine declares illegal — and the
        verdict counter's labels are verdict-only, so nothing today can say WHICH
        transitions those are. The one candidate explanation on record
        (``processing -> processing``, left undecided when #248 shipped) is
        unfalsifiable without this.

        ⛔ It does NOT make the verdict a refusal, and must not be read as a step
        that did. ``validate_state_transition`` returns the same ``False`` it
        always returned, and the sole caller still discards it.

        Swallows everything, for the same reason the other two recorders do:
        instrumentation must never be able to turn a state write into a failure.
        """
        try:
            get_metrics_client().record_worker_state_transition_invalid_pair(
                current_state, new_state
            )
        except Exception:  # pragma: no cover - defensive; observe-only
            logger.debug("worker state transition invalid pair counter failed", exc_info=True)

    @asynccontextmanager
    async def atomic_worker_operation(
        self, worker_id: str, worker_name: str, operation: str
    ) -> AsyncIterator[str]:
        """Context manager for atomic worker operations with proper locking."""
        lock_hash = hashlib.sha256(
            f"{worker_id}:{operation}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        async with self._held_lock(worker_id, operation):
            # Store lock info for debugging
            self._lock_info[worker_id] = WorkerLockInfo(
                worker_id=worker_id,
                worker_name=worker_name,
                lock_hash=lock_hash,
                timestamp=datetime.now(),
                operation=operation,
            )
            try:
                yield lock_hash
            finally:
                # Clean up lock info
                self._lock_info.pop(worker_id, None)

    async def resolve_worker_id(
        self,
        tenant_id: str,
        subject_id: str,
        worker_name: str,
        registry: WorkerRegistryDB,
    ) -> tuple[str, bool]:
        """Resolve worker ID with (tenant_id, subject_id)-aware lookup."""
        lock_key = f"{tenant_id}:{worker_name}"
        async with self.atomic_worker_operation(lock_key, worker_name, "resolve_id"):
            try:
                # Primary lookup: Check registry with (tenant, subject) scoping
                worker_data = await registry.get_worker(
                    tenant_id=tenant_id, subject_id=subject_id, worker_name=worker_name
                )
                if worker_data and worker_data.get("worker_id"):
                    worker_id = worker_data["worker_id"]
                    logger.debug(
                        "Worker ID resolved from registry: tenant=%s, name=%s -> %s",
                        tenant_id,
                        worker_name,
                        worker_id,
                    )
                    return worker_id, True

                # Secondary lookup: Check if worker_name is actually a UUID
                try:
                    UUID(worker_name)
                    logger.debug(
                        "Worker name is UUID, using as worker_id (tenant=%s): %s",
                        tenant_id,
                        worker_name,
                    )
                    return worker_name, False
                except ValueError:
                    pass

                # Fallback: Use worker_name as worker_id (legacy compatibility)
                logger.debug(
                    "Using worker_name as worker_id (tenant=%s): %s",
                    tenant_id,
                    worker_name,
                )
                return worker_name, False

            except psycopg.Error, ValueError, KeyError, AttributeError:
                logger.exception(
                    "Error resolving worker ID for tenant=%s, name=%s",
                    tenant_id,
                    worker_name,
                )
                return worker_name, False


__all__ = [
    "STATE_TRANSITIONS",
    "WorkerLockInfo",
    "WorkerStateManager",
    "WorkerStateType",
]
