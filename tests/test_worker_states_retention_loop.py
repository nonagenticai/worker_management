"""The retention loop must do its work BEFORE its first sleep.

Why this test exists
--------------------
The only prior background loop in this service
(`pydantic_graph_execution_service._periodic_cleanup`) opens its `try` with
`await asyncio.sleep(300)`. A loop shaped that way completes zero passes in a
pod that rolls more often than its interval: the work is always scheduled and
never done, and nothing errors, so the absence is invisible. With a 1-hour
retention interval that is a very easy pod to be.

These are unit tests -- no database -- because the ordering is a property of
the loop, not of the SQL.
"""

from __future__ import annotations

import asyncio

import pytest

from src.services.worker_states_retention import RetentionPolicy, retention_loop


@pytest.mark.asyncio
async def test_the_first_pass_runs_before_the_first_sleep():
    """Cancel during the first sleep; the sweep must already have run once."""
    calls: list[str] = []

    async def sweep() -> None:
        calls.append("swept")

    task = asyncio.create_task(retention_loop(sweep=sweep, interval_seconds=3600))
    # Yield enough for the loop to reach its first `await asyncio.sleep`.
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == ["swept"], (
        "the loop slept before sweeping -- a pod rolling faster than the "
        "interval would never complete a single pass"
    )


@pytest.mark.asyncio
async def test_a_failing_pass_backs_off_instead_of_spinning():
    """The sleep stays on the error path, so a broken pass cannot hot-loop."""
    attempts = 0
    slept: list[float] = []

    async def sweep() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)
        await real_sleep(0)

    task = asyncio.create_task(
        retention_loop(sweep=sweep, interval_seconds=3600, error_backoff_seconds=60)
    )
    try:
        asyncio.sleep = fake_sleep  # type: ignore[assignment]
        for _ in range(30):
            await real_sleep(0)
    finally:
        asyncio.sleep = real_sleep  # type: ignore[assignment]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert attempts > 1, "the loop stopped retrying after one failure"
    assert slept, "a failing pass did not sleep -- it would spin on the database"
    assert set(slept) == {60}, (
        f"a failing pass must back off by error_backoff_seconds, got {sorted(set(slept))}"
    )


def test_the_analytics_window_can_never_be_shorter_than_the_general_one():
    """An inverted config must not delete terminal rows the general rule keeps.

    That is the one direction of misconfiguration that destroys data nothing
    else can rebuild, so it is clamped rather than obeyed.
    """
    from datetime import UTC, datetime

    policy = RetentionPolicy(
        retention_days=90,
        analytics_retention_days=7,  # nonsense: shorter than the general window
        min_per_worker=10,
        batch_size=100,
        max_batches_per_pass=10,
    )
    now = datetime(2026, 8, 27, tzinfo=UTC)
    cutoff, analytics_cutoff = policy.cutoffs(now)

    assert analytics_cutoff <= cutoff, (
        "an inverted config would delete terminal rows the general window keeps"
    )


@pytest.mark.asyncio
async def test_a_zero_keep_guard_is_refused():
    """min_per_worker=0 would allow deleting a worker's ONLY row."""
    from src.services.worker_states_retention import sweep_once

    policy = RetentionPolicy(
        retention_days=30,
        analytics_retention_days=180,
        min_per_worker=0,
        batch_size=100,
        max_batches_per_pass=10,
    )
    with pytest.raises(ValueError, match="min_per_worker"):
        await sweep_once(policy)


# ---------------------------------------------------------------------------
# The pass deadline: a bound with no caller is the failure mode.
# ---------------------------------------------------------------------------
#
# On 2026-08-27 a single batch of this sweep ran for 34 minutes and took
# worker-management down. A row-count bound (`batch_size`,
# `max_batches_per_pass`) does not bound TIME, and the incident was a time
# problem. So `sweep_once` grew `pass_deadline_seconds` -- and a bound that
# `retention_loop` never passes is worse than no bound at all: it is present in
# the signature, exercised by its own unit test, and never once applied in
# production. That exact shape has shipped in this service before, which is why
# the binding itself is asserted here and not just the behaviour.


@pytest.mark.asyncio
async def test_the_loop_binds_a_pass_deadline_to_its_default_runner(monkeypatch):
    """`retention_loop()` with no injected sweep must pass a real deadline."""
    from src.services import worker_states_retention as retention

    seen: dict[str, object] = {}

    async def fake_sweep_once(*args, **kwargs):
        seen["called"] = True
        seen.update(kwargs)

    monkeypatch.setattr(retention, "sweep_once", fake_sweep_once)

    task = asyncio.create_task(retention.retention_loop(interval_seconds=3600))
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen.get("called"), "the default runner never called sweep_once"
    deadline = seen.get("pass_deadline_seconds")
    assert deadline is not None, (
        "retention_loop's default runner passed no pass_deadline_seconds -- the "
        "time bound is inert in production, which is exactly how it shipped "
        "the first time"
    )
    assert deadline == 3600 * retention._PASS_DEADLINE_FRACTION, (
        f"the deadline must be derived from the loop interval so it keeps "
        f"bounding when the interval is lowered; got {deadline}"
    )
    assert deadline < 3600, (
        "a deadline at or above the interval cannot stop a pass overlapping its own next tick"
    )
