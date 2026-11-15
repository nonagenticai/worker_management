"""The await-free invariant ``WorkerInstanceCache._held_lock`` rests on.

``tests/test_worker_instance_cache_lock_lifecycle.py`` pins what ``_held_lock``
DOES. This file pins the rule that makes it correct, which until now lived only
in prose — the class docstring says "neither its get/create/increment block nor
its decrement/reap block contains an ``await``", and nothing could fail when
that stopped being true.

Two blocks, two different reasons, one shared root cause: the reap
(``entry.users -= 1`` / ``del``) deliberately runs WITHOUT ``_locks_lock``, so
the two blocks are not mutually excluded by anything except the absence of a
suspension point between them.

1. **The increment.** ``_locks_lock`` does NOT protect the get/create/increment
   block from the reaper, because the reaper never takes ``_locks_lock``. Let a
   caller suspend after reading the entry and before ``users += 1`` and the
   holder's reap can run in that window: it sees ``users == 0``, deletes the
   entry, and the suspended caller then increments and locks an ORPHAN, while
   the next arrival mints a fresh entry and enters the critical section beside
   it. That is duplicate instantiation / eviction of a live instance, i.e. the
   exact failure the refcount exists to prevent, reintroduced.
   ⇒ ``TestArrivalWindow``, a deterministic sweep over arrival offsets.

2. **The reap.** An ``await`` in the ``finally`` is a cancellation delivery
   point. A task that is cancelled AGAIN while unwinding (a TaskGroup or
   lifespan shutdown that re-cancels, an outer ``asyncio.timeout``) raises at
   that await, skips ``users -= 1``, and strands the entry forever — the leak
   this whole change exists to fix.
   ⇒ ``TestReapIsAwaitFree``.

3. **Both, structurally.** ⚠️ MEASURED, and it is why (1) and (2) are not
   sufficient on their own: restoring ``async with self._locks_lock`` around the
   ``finally`` — the single edit the class docstring most explicitly forbids —
   is a BEHAVIOURALLY EQUIVALENT mutant today. An uncontended
   :class:`asyncio.Lock` acquire returns on a fast path without yielding
   (verified on CPython 3.14: a task doing ``async with Lock(): pass``
   completes within one event-loop step), and ``_locks_lock`` is never held
   across a suspension, so it is never contended, so that ``await`` never
   actually suspends and never delivers a cancellation. No behavioural test can
   kill it. It is nonetheless one edit away from lethal — it becomes a real
   suspension the moment anything makes ``_locks_lock`` contended.
   ⇒ ``TestStructuralInvariant`` asserts the rule directly on the AST, which
   catches the currently-equivalent edit as well as the lethal ones.

Every probe here carries a positive control that must FAIL in the same file, for
the reason recorded in the sibling file: an exclusion probe that cannot observe
the defect it is aimed at proves nothing, and this branch has already produced
one such probe.
"""

import ast
import asyncio
import contextlib
import inspect
import textwrap
from contextlib import asynccontextmanager

import pytest

from src.api.workers._globals import WorkerInstanceCache, _LockEntry

KEY = "tenant-a:worker-a:standard"


class _ReplicaCache:
    """A line-for-line copy of ``_held_lock``'s locking with ONE seeded defect.

    The control for each probe must come from the class of change the probe
    guards against, not from "no locking at all" — an unlocked context manager
    overlaps under any probe whatsoever and therefore demonstrates nothing about
    whether THIS defect is observable.
    """

    def __init__(self, defect: str) -> None:
        self._worker_locks: dict[str, _LockEntry] = {}
        self._locks_lock = asyncio.Lock()
        self._defect = defect

    @asynccontextmanager
    async def _held_lock(self, cache_key: str):
        async with self._locks_lock:
            entry = self._worker_locks.get(cache_key)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock())
                self._worker_locks[cache_key] = entry
                if self._defect == "await_in_create_branch":
                    await asyncio.sleep(0)
            if self._defect == "await_before_increment":
                await asyncio.sleep(0)
            entry.users += 1
        try:
            async with entry.lock:
                yield entry.lock
        finally:
            if self._defect == "await_in_reap":
                await asyncio.sleep(0)
            entry.users -= 1
            if entry.users <= 0 and self._worker_locks.get(cache_key) is entry:
                del self._worker_locks[cache_key]


# --------------------------------------------------------------------------- #
# 1. the arrival window
# --------------------------------------------------------------------------- #


async def _arrival_offset_scenario(cache, before_release: int, before_latecomer: int) -> int:
    """Stage holder / victim / latecomer at one fixed pair of arrival offsets.

    Returns the peak number of coroutines simultaneously inside the critical
    section. Correct locking makes that 1 at every offset.

    The offsets are counted in ``await asyncio.sleep(0)`` — event-loop steps —
    not seconds. Nothing here consults the clock or the RNG, so the interleaving
    is a pure function of ``(before_release, before_latecomer)`` and the sweep
    below is reproducible rather than probabilistic.

    * ``holder`` takes the key and parks, so ``users == 1``.
    * ``victim`` arrives and is advanced ``before_release`` steps. Correct code
      has no suspension point in the get/create/increment block, so the victim
      is already counted and queued inside ``lock.acquire()``. Broken code
      leaves it parked mid-block with the entry read but not yet counted.
    * the holder is then released and reaps. With the victim uncounted, the reap
      sees ``users == 0`` and deletes the live entry.
    * ``latecomer`` arrives ``before_latecomer`` steps later, finds the dict
      empty, mints a second lock for the same key and walks in.
    """
    inside = 0
    peak = 0
    holder_in = asyncio.Event()
    holder_go = asyncio.Event()

    async def body():
        nonlocal inside, peak
        async with cache._held_lock(KEY):
            inside += 1
            peak = max(peak, inside)
            # Yield while "inside": without this the body is atomic under
            # asyncio and two coroutines could never be *observed* together,
            # whatever the locking did.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            inside -= 1

    async def holder():
        nonlocal inside, peak
        async with cache._held_lock(KEY):
            inside += 1
            peak = max(peak, inside)
            holder_in.set()
            await holder_go.wait()
            inside -= 1

    tasks = [asyncio.create_task(holder())]
    await holder_in.wait()

    tasks.append(asyncio.create_task(body()))
    for _ in range(before_release):
        await asyncio.sleep(0)

    holder_go.set()
    for _ in range(before_latecomer):
        await asyncio.sleep(0)

    tasks.append(asyncio.create_task(body()))
    await asyncio.gather(*tasks)
    return peak


async def _sweep(make_cache, span: int = 8) -> tuple[int, list[tuple[int, int]]]:
    """Run every arrival-offset pair in ``range(span) x range(span)``.

    Returns ``(worst peak, offsets that overlapped)``. A fresh cache per
    scenario, so one scenario's residue cannot mask or manufacture the next's.
    """
    worst = 0
    offenders: list[tuple[int, int]] = []
    for a in range(span):
        for b in range(span):
            cache = make_cache()
            peak = await _arrival_offset_scenario(cache, a, b)
            if peak > 1:
                offenders.append((a, b))
            worst = max(worst, peak)
            assert not cache._worker_locks, (
                f"offsets ({a}, {b}) left {len(cache._worker_locks)} entry(ies) behind"
            )
    return worst, offenders


class TestArrivalWindow:
    """``users += 1`` must not be separated from the dict read by a suspension."""

    @pytest.mark.asyncio
    async def test_no_arrival_offset_admits_a_second_caller(self):
        """⭐ The core invariant, made falsifiable.

        ``_locks_lock`` is not what protects this block — the reaper does not
        take it. The absence of an ``await`` is.
        """
        worst, offenders = await _sweep(WorkerInstanceCache)
        assert worst == 1, (
            f"{worst} coroutines were inside the critical section at once, at arrival "
            f"offsets {offenders}: a caller was suspended between reading the lock entry "
            "and counting itself into it, so the holder's reap deleted an entry that was "
            "still wanted and the next arrival minted a second lock for the same key"
        )

    @pytest.mark.asyncio
    async def test_arrival_offset_sweep_can_fail(self):
        """Positive control, drawn from the defect class, not from 'no locking'.

        The replica is ``_held_lock`` with one ``await asyncio.sleep(0)`` moved
        into the get/create/increment block and nothing else changed. If the
        sweep cannot make THIS overlap, it cannot catch the real thing either
        and ``test_no_arrival_offset_admits_a_second_caller`` is decoration.
        """
        worst, offenders = await _sweep(lambda: _ReplicaCache("await_before_increment"))
        assert worst > 1, (
            "the arrival-offset sweep could not observe overlap even against a "
            "_held_lock replica whose increment IS separated from the dict read by an "
            "await; the sweep proves nothing about the real implementation"
        )
        assert offenders, "worst peak > 1 but no offending offset was recorded"

    @pytest.mark.asyncio
    async def test_the_replica_is_otherwise_faithful(self):
        """The control differs from the subject ONLY in the seeded defect.

        Without this, ``test_arrival_offset_sweep_can_fail`` could be passing
        because the replica is broken in some unrelated way.
        """
        worst, offenders = await _sweep(lambda: _ReplicaCache("none"))
        assert worst == 1, (
            f"the undefected replica overlapped at {offenders}; it is not a faithful "
            "copy, so the defected replica does not isolate the seeded defect"
        )


# --------------------------------------------------------------------------- #
# 2. the reap
# --------------------------------------------------------------------------- #


async def _strand_probe(cache, redeliveries: int = 4) -> dict:
    """Cancel a holder, then keep cancelling while it unwinds. Return the residue.

    One ``cancel()`` is not enough to expose an ``await`` in the ``finally``:
    the first :exc:`~asyncio.CancelledError` is consumed where it is delivered
    (the parked ``wait()``), so the unwind that follows starts with no pending
    cancellation. It is the SECOND cancellation — a TaskGroup tearing down, a
    lifespan shutdown re-cancelling on its grace timer, an outer
    ``asyncio.timeout`` — that lands on whatever the ``finally`` awaits.

    Correct code has no suspension point in the ``finally``, so the unwind
    completes inside a single event-loop step and every later ``cancel()`` is a
    no-op against an already-finished task.
    """
    started = asyncio.Event()

    async def holder():
        async with cache._held_lock(KEY):
            started.set()
            await asyncio.Event().wait()  # parked forever; only cancellation ends it

    task = asyncio.create_task(holder())
    await started.wait()
    task.cancel()
    for _ in range(redeliveries):
        await asyncio.sleep(0)
        if task.done():
            break
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return dict(cache._worker_locks)


class TestReapIsAwaitFree:
    """``users -= 1`` must not sit behind a suspension point."""

    @pytest.mark.asyncio
    async def test_re_cancellation_during_unwind_cannot_strand_the_entry(self):
        residue = await _strand_probe(WorkerInstanceCache())
        assert residue == {}, (
            f"{len(residue)} entry(ies) stranded: {residue}. A cancellation was delivered "
            "inside the reap, so `users -= 1` never ran and the key is now permanently "
            "resident -- the unbounded-growth defect this change exists to fix"
        )

    @pytest.mark.asyncio
    async def test_strand_probe_can_fail(self):
        """Positive control: the same probe against a reap that awaits."""
        residue = await _strand_probe(_ReplicaCache("await_in_reap"))
        assert residue, (
            "the re-cancellation probe saw no stranded entry even against a _held_lock "
            "replica that awaits inside its `finally`; the probe cannot detect the defect "
            "it is aimed at"
        )

    @pytest.mark.asyncio
    async def test_the_replica_reap_is_otherwise_faithful(self):
        assert await _strand_probe(_ReplicaCache("none")) == {}, (
            "the undefected replica stranded an entry; it is not a faithful copy"
        )


# --------------------------------------------------------------------------- #
# 2b. the entry block, before the try even starts
# --------------------------------------------------------------------------- #


async def _cancel_before_entry(cache, offset: int) -> dict:
    """Cancel a caller ``offset`` event-loop steps after it starts. Return the residue.

    ⚠️ The get/create/increment block runs BEFORE the ``try:``, so anything that
    suspends inside it has no ``finally`` behind it at all. An entry that has
    been inserted into the dict but not yet counted is unreachable by every
    other code path — the reap only removes an entry it holds a reference to —
    so a cancellation landing in that window strands it with ``users == 0``
    forever. Being inside ``_locks_lock`` does not help: the entry is already
    published in the dict.

    This is the half of the invariant the arrival-offset sweep cannot see. That
    sweep needs a second caller to observe overlap, and the window on the create
    branch is only open while ``_locks_lock`` is held, so no second caller can
    get in to observe anything.
    """

    async def caller():
        async with cache._held_lock(KEY):
            await asyncio.sleep(0)

    task = asyncio.create_task(caller())
    for _ in range(offset):
        await asyncio.sleep(0)
        if task.done():
            break
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return dict(cache._worker_locks)


async def _cancel_sweep(make_cache, span: int = 8) -> list[tuple[int, dict]]:
    """Cancel at every step offset in ``range(span)``; return the offsets that leaked."""
    leaks = []
    for offset in range(span):
        cache = make_cache()
        residue = await _cancel_before_entry(cache, offset)
        if residue:
            leaks.append((offset, residue))
    return leaks


class TestCancellationBeforeEntry:
    """Nothing in the get/create/increment block may suspend: there is no finally yet."""

    @pytest.mark.asyncio
    async def test_no_cancellation_offset_strands_an_uncounted_entry(self):
        leaks = await _cancel_sweep(WorkerInstanceCache)
        assert leaks == [], (
            f"cancellation stranded an entry at offsets {[o for o, _ in leaks]}: the entry "
            "was published into `_worker_locks` and then the caller was interrupted before "
            "`users += 1` and before the `try:`, so no reap will ever remove it"
        )

    @pytest.mark.asyncio
    async def test_cancellation_sweep_can_fail(self):
        """Positive control: the replica awaits between the insert and the increment."""
        leaks = await _cancel_sweep(lambda: _ReplicaCache("await_in_create_branch"))
        assert leaks, (
            "the cancellation sweep saw no stranded entry even against a _held_lock replica "
            "that awaits after publishing the entry and before counting it"
        )

    @pytest.mark.asyncio
    async def test_the_replica_entry_block_is_otherwise_faithful(self):
        assert await _cancel_sweep(lambda: _ReplicaCache("none")) == [], (
            "the undefected replica stranded an entry; it is not a faithful copy"
        )


# --------------------------------------------------------------------------- #
# 3. the rule itself
# --------------------------------------------------------------------------- #

_SUSPENDING = (ast.Await, ast.AsyncWith, ast.AsyncFor)


def _await_free_violations(source: str) -> list[str]:
    """Report every suspension point in ``_held_lock``'s two await-free blocks.

    ⚠️ Raises :exc:`AssertionError` rather than returning ``[]`` when it cannot
    find the blocks. A structural check that quietly inspects nothing reports
    "compliant" for the wrong reason, and that is indistinguishable from a pass.

    Args:
        source: the source text of an ``async def _held_lock`` (dedented).

    Returns:
        One human-readable line per offending node; empty means compliant.
    """
    tree = ast.parse(textwrap.dedent(source))
    funcs = [
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_held_lock"
    ]
    assert len(funcs) == 1, f"expected exactly one `async def _held_lock`, found {len(funcs)}"
    func = funcs[0]

    def _stmts(nodes):
        return [n for stmt in nodes for n in ast.walk(stmt)]

    def _touches_users(nodes, op):
        return any(
            isinstance(n, ast.AugAssign)
            and isinstance(n.op, op)
            and isinstance(n.target, ast.Attribute)
            and n.target.attr == "users"
            for n in nodes
        )

    # The entry block is identified by the statement it must contain, not by
    # position or by being the only `_locks_lock` block -- a reap that has been
    # re-locked under `_locks_lock` adds a second one, and that edit is exactly
    # what this check exists to report rather than to crash on.
    entry_blocks = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.AsyncWith)
        and any(
            isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == "_locks_lock"
            for item in n.items
        )
        and _touches_users(_stmts(n.body), ast.Add)
    ]
    assert len(entry_blocks) == 1, (
        f"expected exactly one `async with self._locks_lock:` block containing "
        f"`entry.users += 1`, found {len(entry_blocks)}; either the increment left the "
        "block or the checker is not looking at what it claims to check"
    )
    reap_blocks = [n for n in ast.walk(func) if isinstance(n, ast.Try) and n.finalbody]
    assert len(reap_blocks) == 1, f"expected exactly one `try/finally`, found {len(reap_blocks)}"

    entry_nodes = _stmts(entry_blocks[0].body)
    reap_nodes = _stmts(reap_blocks[0].finalbody)
    assert _touches_users(reap_nodes, ast.Sub), (
        "`entry.users -= 1` is not inside the `finally:` block"
    )

    violations = []
    for label, nodes in (("get/create/increment", entry_nodes), ("decrement/reap", reap_nodes)):
        for node in nodes:
            if isinstance(node, _SUSPENDING):
                violations.append(f"line {node.lineno}: {type(node).__name__} in the {label} block")
    return violations


def _held_lock_source() -> str:
    """The real ``_held_lock`` source, read through the decorator."""
    return textwrap.dedent(inspect.getsource(WorkerInstanceCache._held_lock))


def _with_reap_relocked(source: str) -> str:
    """Put ``async with self._locks_lock:`` back around the whole ``finally`` body.

    Built by re-indenting the real reap block rather than by matching a literal,
    so the control cannot silently stop being the edit it claims to be.
    """
    lines = source.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == "finally:"]
    assert len(hits) == 1, f"expected one `finally:`, found {len(hits)}"
    head = hits[0]
    indent = lines[head][: len(lines[head]) - len(lines[head].lstrip())]
    body = lines[head + 1 :]
    assert body and body[0].startswith(indent + "    "), "the reap block is not where expected"
    relocked = [f"{indent}    async with self._locks_lock:\n"] + [
        ("    " + line if line.strip() else line) for line in body
    ]
    return "".join(lines[: head + 1] + relocked)


def _with_await_injected(source: str, anchor: str) -> str:
    """Insert ``await asyncio.sleep(0)`` immediately above ``anchor``'s line."""
    lines = source.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == anchor]
    assert len(hits) == 1, f"anchor {anchor!r} appears {len(hits)} times; cannot inject"
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i, f"{indent}await asyncio.sleep(0)\n")
    return "".join(lines)


class TestStructuralInvariant:
    """The rule in the class docstring, asserted on the code rather than narrated.

    This is what covers the edit behaviour cannot: putting ``async with
    self._locks_lock`` back around the ``finally``. That produces no observable
    difference today (an uncontended ``asyncio.Lock`` acquire never yields), so
    it survives every behavioural probe above while removing the property those
    probes exist to protect.
    """

    def test_neither_block_contains_a_suspension_point(self):
        violations = _await_free_violations(_held_lock_source())
        assert violations == [], (
            "WorkerInstanceCache._held_lock has gained a suspension point in a block its "
            "correctness requires to be atomic: "
            + "; ".join(violations)
            + ". The reap runs WITHOUT `_locks_lock`, so these two blocks are mutually "
            "exclusive only while neither can be interrupted. See the class docstring and "
            "TestArrivalWindow / TestReapIsAwaitFree for what breaks."
        )

    @pytest.mark.parametrize(
        ("anchor", "block"),
        [("entry.users += 1", "get/create/increment"), ("entry.users -= 1", "decrement/reap")],
    )
    def test_the_check_reports_an_injected_await(self, anchor, block):
        """Positive control on the REAL source, minimally perturbed.

        The control input is ``_held_lock`` itself with one line added, so a
        clean verdict on the unperturbed source cannot be an artefact of the
        checker walking the wrong function or the wrong block.
        """
        violations = _await_free_violations(_with_await_injected(_held_lock_source(), anchor))
        assert violations, f"the checker did not notice an await injected above `{anchor}`"
        assert any(block in v for v in violations), (
            f"the checker flagged {violations}, but not in the {block} block"
        )

    def test_the_check_reports_an_async_with_in_the_reap(self):
        """The specific edit the docstring forbids and behaviour cannot catch."""
        source = _with_reap_relocked(_held_lock_source())
        violations = _await_free_violations(source)
        assert any("decrement/reap" in v for v in violations), (
            "re-locking the reap under `_locks_lock` was not reported. That edit is "
            "behaviourally equivalent today only because an uncontended asyncio.Lock "
            "acquire does not yield; this check is the only thing that can catch it."
        )

    def test_the_check_refuses_to_pass_when_it_cannot_find_the_blocks(self):
        """A blinded checker must fail, not return 'no violations'."""
        with pytest.raises(AssertionError):
            _await_free_violations("async def _held_lock(self, k):\n    yield\n")
