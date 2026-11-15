"""The retention sweep must delete old rows AND keep the newest per worker.

Why this is a `pg` test and not a unit test
-------------------------------------------
The entire policy is expressed in one SQL statement -- the age cutoffs, the
terminal-vs-transient split, and the correlated per-worker keep-guard. A unit
test against a fake would assert that Python called a string, which is exactly
the vacuous shape that has been caught repeatedly in this fleet. The only thing
worth pinning is what the database actually deletes, so these run against a
real Postgres via the `pg` marker (CI selects them with
`integration_test_select: "tests/integration -m pg"`).

Rows are INSERTed directly rather than through `StateDatabase.add_worker_state`
because that method stamps `datetime.now(UTC)` unconditionally -- there is no
way to seed a 100-day-old row through the public API.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_worker_states_retention.py -m pg -v --no-cov
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.services.worker_states_retention import RetentionPolicy, sweep_once

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]

NOW = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)

# Deliberately small so the assertions name exact rows rather than ranges.
POLICY = RetentionPolicy(
    retention_days=30,
    analytics_retention_days=180,
    min_per_worker=2,
    batch_size=100,
    max_batches_per_pass=50,
)


@pytest_asyncio.fixture
async def pg_pool():
    require_database_url()
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture(autouse=True)
async def _no_leftovers_from_earlier_runs(pg_pool):
    """Remove rows this module seeded in an earlier run, before each test.

    Not hygiene -- a correctness precondition, and it fails in only one
    direction. The sweep is GLOBAL and walks a forward-only `id` watermark
    from 0 (see `sweep_once`), so every row already in the table is examined
    before the rows a test seeds. A test that bounds its pass with a small
    budget therefore spends that budget on other runs' leftovers and never
    reaches its own rows.

    Measured on a real postgres:18 before this fixture existed: the module
    leaves 26 rows per run, and on the 17th consecutive run against the same
    database `test_batching_completes_a_backlog_larger_than_one_batch` (budget
    50 batches x 5 rows = 250) returned

        SweepResult(deleted=0, batches=50, exhausted=False, examined=250)

    -- 250 examined, 0 of them its own. Runs 1-16 passed, so this is a
    test-pollution cliff, not a flake and not a defect in the sweep.

    CI never hit it because each job gets a fresh ephemeral `postgres:18`
    container; a developer re-running the file against one long-lived local
    database does, which is precisely the person a red suite misleads most.

    Scoped by `tenant_name`, which `_seed` stamps as 'retention-test' on every
    row it writes and which nothing else in `tests/` or `src/` uses. A blanket
    `DELETE FROM worker_states` would also work today and would silently start
    eating another test file's fixtures the day one is added.
    """
    async with get_pg_connection() as conn:
        await conn.execute("DELETE FROM worker_states WHERE tenant_name = $1", "retention-test")
    yield


async def _seed(
    *, tenant_id: str, worker_id: str, state_type: str, age_days: float, tag: str
) -> int:
    """Insert one row at a chosen age and return its id."""
    ts = NOW - timedelta(days=age_days)
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO worker_states
                (tenant_id, subject_id, visibility, tenant_name, worker_id,
                 worker_name, is_agent, state_type, worker_state, timestamp)
            VALUES ($1, $2, 'tenant', 'retention-test', $3, $4, true, $5, $6, $7)
            RETURNING id
            """,
            tenant_id,
            tenant_id,
            worker_id,
            worker_id,
            state_type,
            {"tag": tag},
            ts,
        )
    assert row is not None
    return int(row["id"])


async def _surviving_tags(tenant_id: str) -> set[str]:
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT worker_state->>'tag' AS tag FROM worker_states WHERE tenant_id = $1",
            tenant_id,
        )
    return {r["tag"] for r in rows}


async def test_old_rows_go_and_the_newest_per_worker_stay(pg_pool):
    """The headline contract, in one assertion.

    Six `processing` rows for one worker, every one of them older than the
    30-day window. Five are expired by age; the two NEWEST must survive anyway,
    because `min_per_worker=2` outranks the age rule -- `get_latest_worker_state`
    reads the newest row whatever its type, so deleting it would roll the
    worker's reported status backwards.

    This one test is sensitive in BOTH directions: if the sweep deletes nothing
    the `old-*` tags remain and it fails; if the keep-guard is dropped the
    `keep-*` tags vanish and it fails.
    """
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    for age, tag in (
        (200.0, "old-200"),
        (150.0, "old-150"),
        (100.0, "old-100"),
        (90.0, "old-90"),
        (80.0, "keep-80"),
        (70.0, "keep-70"),
    ):
        await _seed(
            tenant_id=tenant_id,
            worker_id=worker_id,
            state_type="processing",
            age_days=age,
            tag=tag,
        )

    result = await sweep_once(POLICY, now=NOW)

    assert result.deleted == 4, result
    assert await _surviving_tags(tenant_id) == {"keep-80", "keep-70"}


async def test_terminal_rows_inside_the_analytics_window_survive(pg_pool):
    """`completed`/`error` outlive the 30-day window up to the analytics window.

    They are the numerator and denominator of the rescue-rate query, which is
    the documented reason this table is the durable record at all (Loki keeps
    ~16 days). A sweep that treated them like `processing` would silently make
    that ratio unanswerable.
    """
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    # All at 100 days: past the 30d general window, inside the 180d analytics one.
    for age, st, tag in (
        (104.0, "completed", "term-completed-1"),
        (103.0, "error", "term-error"),
        (102.0, "completed", "term-completed-2"),
        (101.0, "processing", "transient-1"),
        (100.0, "processing", "transient-2"),
    ):
        await _seed(tenant_id=tenant_id, worker_id=worker_id, state_type=st, age_days=age, tag=tag)

    await sweep_once(POLICY, now=NOW)

    # The two `processing` rows are the two NEWEST rows of this worker, so the
    # min_per_worker=2 guard keeps them -- nothing is deleted here at all, and
    # that is the point: the guard is evaluated per worker, not per state type.
    assert await _surviving_tags(tenant_id) == {
        "term-completed-1",
        "term-error",
        "term-completed-2",
        "transient-1",
        "transient-2",
    }


async def test_terminal_rows_past_the_analytics_window_are_deleted(pg_pool):
    """The analytics window is a real bound, not an exemption."""
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    for age, tag in (
        (400.0, "ancient-1"),
        (300.0, "ancient-2"),
        (250.0, "ancient-3"),
        (20.0, "recent-1"),
        (10.0, "recent-2"),
    ):
        await _seed(
            tenant_id=tenant_id,
            worker_id=worker_id,
            state_type="completed",
            age_days=age,
            tag=tag,
        )

    await sweep_once(POLICY, now=NOW)

    assert await _surviving_tags(tenant_id) == {"recent-1", "recent-2"}


async def test_rows_inside_the_retention_window_are_never_touched(pg_pool):
    """Nothing younger than the window may be deleted, whatever its type."""
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    for age, tag in ((25.0, "fresh-1"), (20.0, "fresh-2"), (5.0, "fresh-3"), (1.0, "fresh-4")):
        await _seed(
            tenant_id=tenant_id,
            worker_id=worker_id,
            state_type="processing",
            age_days=age,
            tag=tag,
        )

    result = await sweep_once(POLICY, now=NOW)

    assert result.deleted == 0
    assert await _surviving_tags(tenant_id) == {"fresh-1", "fresh-2", "fresh-3", "fresh-4"}


async def test_the_keep_guard_is_per_worker_not_global(pg_pool):
    """Two workers each keep their own newest rows.

    A guard keyed on the wrong column -- or a global `ORDER BY id DESC LIMIT n`
    -- would protect the newest rows of the busiest worker and strip a quiet
    worker bare. `worker_id` alone is not sufficient either: it is an opaque
    history key with no FK, so the guard is keyed on (tenant_id, worker_id).
    """
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    busy = f"busy-{uuid.uuid4().hex[:8]}"
    quiet = f"quiet-{uuid.uuid4().hex[:8]}"

    # The quiet worker's rows are ALL older than the busy worker's.
    for age, tag in ((300.0, "quiet-old"), (290.0, "quiet-keep-1"), (280.0, "quiet-keep-2")):
        await _seed(
            tenant_id=tenant_id, worker_id=quiet, state_type="processing", age_days=age, tag=tag
        )
    for age, tag in ((60.0, "busy-old"), (50.0, "busy-keep-1"), (40.0, "busy-keep-2")):
        await _seed(
            tenant_id=tenant_id, worker_id=busy, state_type="processing", age_days=age, tag=tag
        )

    await sweep_once(POLICY, now=NOW)

    assert await _surviving_tags(tenant_id) == {
        "quiet-keep-1",
        "quiet-keep-2",
        "busy-keep-1",
        "busy-keep-2",
    }


async def test_the_sweep_does_not_cross_into_another_tenant(pg_pool):
    """The sweep is cross-tenant by design, but age still governs each row.

    There is no RLS on this table, so nothing in the database stops a sweep
    from deleting another tenant's rows. This pins that a neighbouring tenant's
    IN-WINDOW rows survive a pass that deletes this tenant's expired ones.
    """
    victim = tenant_uuid(f"ret-victim-{uuid.uuid4().hex[:12]}")
    neighbour = tenant_uuid(f"ret-neighbour-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    for age, tag in ((300.0, "v-old"), (290.0, "v-keep-1"), (280.0, "v-keep-2")):
        await _seed(
            tenant_id=victim, worker_id=worker_id, state_type="processing", age_days=age, tag=tag
        )
    for age, tag in ((5.0, "n-fresh-1"), (4.0, "n-fresh-2")):
        await _seed(
            tenant_id=neighbour,
            worker_id=worker_id,
            state_type="processing",
            age_days=age,
            tag=tag,
        )

    await sweep_once(POLICY, now=NOW)

    assert await _surviving_tags(victim) == {"v-keep-1", "v-keep-2"}
    assert await _surviving_tags(neighbour) == {"n-fresh-1", "n-fresh-2"}


async def test_batching_completes_a_backlog_larger_than_one_batch(pg_pool):
    """A backlog bigger than `batch_size` still drains, in several statements.

    This is what keeps the DELETE off a long lock: 12 expired rows at 5 per
    statement is 3 delete batches, not one transaction spanning the backlog.
    """
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    worker_id = f"w-{uuid.uuid4().hex[:8]}"

    for i in range(14):
        await _seed(
            tenant_id=tenant_id,
            worker_id=worker_id,
            state_type="processing",
            age_days=300.0 - i,
            tag=f"r{i:02d}",
        )

    policy = RetentionPolicy(
        retention_days=30,
        analytics_retention_days=180,
        min_per_worker=2,
        batch_size=5,
        max_batches_per_pass=50,
    )
    result = await sweep_once(policy, now=NOW)

    assert result.deleted == 12, result
    assert result.batches > 1, "a 12-row backlog at 5/batch must take several statements"
    # The two newest (largest `i`, smallest age) survive.
    assert await _surviving_tags(tenant_id) == {"r12", "r13"}


async def test_guard_protected_rows_at_low_ids_do_not_block_the_pass(pg_pool):
    """A permanently-ineligible row must not stall every later batch.

    The sweep bounds each batch on the primary key BEFORE applying the
    per-worker keep-guard -- that is what keeps the batch off a table-wide scan
    (see `_DELETE_BATCH_SQL`). The cost of that ordering is that guard-protected
    rows now appear as *candidates*, so a batch that re-ran the same predicate
    would re-select the same head-of-line rows forever: a worker with fewer than
    `min_per_worker` rows is ineligible for all time, and if it also holds the
    lowest `id` it would occupy every batch while the deletable rows behind it
    were never reached. The forward-only `id` watermark is what prevents that.

    Shape: one blocker worker with a single ancient row (never deletable at
    `min_per_worker=2`, and seeded first so it owns the lowest id), then a
    second worker whose backlog sits entirely behind it. With `batch_size=1`
    the blocker consumes the whole first batch, so the pass only reaches the
    backlog if the watermark advanced past it.
    """
    tenant_id = tenant_uuid(f"ret-{uuid.uuid4().hex[:12]}")
    blocker = f"blocker-{uuid.uuid4().hex[:8]}"
    backlog = f"backlog-{uuid.uuid4().hex[:8]}"

    # Seeded FIRST => lowest id. One row only => can never satisfy min_per_worker=2.
    await _seed(
        tenant_id=tenant_id,
        worker_id=blocker,
        state_type="processing",
        age_days=300.0,
        tag="blocker",
    )
    for i in range(5):
        await _seed(
            tenant_id=tenant_id,
            worker_id=backlog,
            state_type="processing",
            age_days=299.0 - i,
            tag=f"b{i}",
        )

    # `batch_size=1` so the blocker occupies an entire batch on its own. The
    # budget is deliberately generous: this suite shares one database and the
    # sweep is cross-tenant by design, so earlier tests' guard-protected
    # leftovers are legitimate candidates sitting at lower ids. They must not
    # starve the budget, and the assertions below are scoped to this tenant so
    # they cannot be perturbed by them either.
    policy = RetentionPolicy(
        retention_days=30,
        analytics_retention_days=180,
        min_per_worker=2,
        batch_size=1,
        max_batches_per_pass=500,
    )
    result = await sweep_once(policy, now=NOW)

    assert result.exhausted, (
        "the pass hit its batch budget instead of reaching the end of the "
        "candidate range -- the watermark is not advancing"
    )
    # b0..b2 are deletable; b3/b4 are the two newest of `backlog` and are held
    # by the keep-guard, as is the blocker's only row. Without the watermark
    # every batch would re-select the lowest-id candidate, `b0..b2` would
    # survive, and this set would be the full six.
    assert await _surviving_tags(tenant_id) == {"blocker", "b3", "b4"}
    # `examined` counts each candidate once, so it is at least this tenant's
    # six. A batch that re-read its head-of-line row would drive it to the
    # 500-batch budget with nothing deleted.
    assert result.examined >= 6, result
    assert result.examined < policy.max_batches_per_pass, result


# ---------------------------------------------------------------------------
# The per-batch statement_timeout, measured on the real backend.
# ---------------------------------------------------------------------------


async def test_the_batch_connection_is_time_bounded_and_hands_the_bound_back(pg_pool):
    """`SET statement_timeout` really lands, and `RESET` really clears it.

    Both halves are load-bearing and neither can be proved against a fake. The
    SET is the seat belt for the 2026-08-27 incident, where one batch ran for
    34 minutes. The RESET is what stops that 30s bound leaking: this connection
    returns to a pool fronting PgBouncer in `pool_mode = transaction`, which
    does not run DISCARD ALL, so an un-RESET `SET` genuinely follows the server
    connection onto unrelated query traffic.

    The value is read back from `pg_settings` on the SAME backend that ran the
    SET, so this asserts what PostgreSQL is actually enforcing rather than that
    Python sent a string.

    "Clears it" means back to *this connection's own default*, which is not
    necessarily 0. `RESET` restores the session default -- postgresql.conf, the
    role/database default, or whatever the client passed in `PGOPTIONS` at
    startup. CI runs with `PGOPTIONS=-c statement_timeout=300000`, so a test
    that hard-codes 0 asserts the environment, not the RESET. The baseline is
    therefore read off the same backend before the sweep touches it.
    """
    from contextlib import asynccontextmanager

    from src.services import worker_states_retention as retention

    observed: list[tuple[str, int]] = []
    baseline: list[int] = []
    real_conn = retention.get_pg_connection

    class _Watcher:
        """Delegates everything, and reads the live GUC after each SET/RESET."""

        def __init__(self, conn):
            self._conn = conn

        async def _live_timeout(self) -> int:
            return int(
                await self._conn.fetchval(
                    "SELECT setting FROM pg_settings WHERE name = 'statement_timeout'"
                )
            )

        async def execute(self, sql, *args):
            result = await self._conn.execute(sql, *args)
            head = " ".join(sql.split()[:2]).upper()
            if head in {"SET STATEMENT_TIMEOUT", "RESET STATEMENT_TIMEOUT"}:
                observed.append((head, await self._live_timeout()))
            return result

        async def fetchrow(self, sql, *args):
            return await self._conn.fetchrow(sql, *args)

    @asynccontextmanager
    async def _watching():
        async with real_conn() as conn:
            watcher = _Watcher(conn)
            baseline.append(await watcher._live_timeout())
            yield watcher

    policy = RetentionPolicy(
        retention_days=30,
        analytics_retention_days=180,
        min_per_worker=2,
        batch_size=1,
        max_batches_per_pass=1,
    )
    retention.get_pg_connection = _watching  # type: ignore[assignment]
    try:
        await sweep_once(policy, now=NOW)
    finally:
        retention.get_pg_connection = real_conn  # type: ignore[assignment]

    assert baseline, "the sweep never opened a connection, so nothing was measured"
    assert baseline[0] != retention._BATCH_STATEMENT_TIMEOUT_MS, (
        f"this backend's default statement_timeout is already "
        f"{retention._BATCH_STATEMENT_TIMEOUT_MS}ms, so the RESET half of this "
        f"test cannot distinguish a restored bound from a leaked one -- run it "
        f"against a backend whose default differs from the batch bound"
    )

    assert observed == [
        ("SET STATEMENT_TIMEOUT", retention._BATCH_STATEMENT_TIMEOUT_MS),
        ("RESET STATEMENT_TIMEOUT", baseline[0]),
    ], (
        f"expected the batch connection to be bounded at "
        f"{retention._BATCH_STATEMENT_TIMEOUT_MS}ms and then handed back at its "
        f"own default of {baseline[0]}ms; the backend reported {observed}"
    )
