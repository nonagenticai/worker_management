"""Bounded retention sweep for the ``worker_states`` history table.

Why this exists
---------------
``worker_states`` is append-only and nothing has ever removed a row from it.
Measured on production 2026-08-27: **3088 MB total / 840 MB heap / 988,948
rows**, i.e. ~2.2 GB (73%) is TOASTed ``worker_state`` JSONB. It grows by
roughly 100k rows/month (857,075 rows in ``postgres_schema.sql``'s header,
953k at migration 024 on 2026-08-25, 988,948 now).

What the retention window is derived FROM
-----------------------------------------
Not from the row count -- from the payload distribution, because the two point
in opposite directions. Measured on the read-only standby, ``sum(octet_length(
worker_state::text))`` per age bucket (NOT ``pg_column_size``, which on a
TOASTed column reports the 18-byte pointer rather than the datum)::

    <7d       49,071 rows    517 MB     ~10.5 KB/row
    7-30d    109,380 rows   1101 MB     ~10.1 KB/row
    30-60d   236,320 rows   1602 MB      ~6.8 KB/row
    60-90d   109,460 rows    921 MB      ~8.4 KB/row
    90-180d  484,717 rows    458 MB      ~0.9 KB/row

The oldest bucket is **49% of the rows and 10% of the bytes**. A naive
"delete everything older than 90 days" therefore throws away half the history
-- including everything the analytics queries below depend on -- and reclaims
almost nothing. The bytes are in the 30-60d band.

Split by ``state_type`` instead (same measurement)::

    completed       434,675 rows   2166 MB    avg 5225 B
    processing      478,775 rows   2102 MB    avg 4605 B
    error            40,917 rows    310 MB    avg 7946 B
    config-updated   29,180 rows    798 kB    avg   28 B
    (updated / idle / created / prompt-swapped: < 100 kB combined)

``processing`` is a transient intermediate state -- it is superseded the moment
a terminal state is written for that worker -- and it carries 2.1 GB. That is
the reclaimable mass, and removing it costs no documented consumer anything.

The policy
----------
Three rules, applied together:

1. **Keep everything newer than ``worker_state_retention_days`` (default 30).**
   30 days is not arbitrary. ``worker_states`` is deliberately the durable
   record for retry telemetry *because* Loki here retains only ~16 days
   (``message_endpoints.py``, the rescue-rate note), and the
   ``_DETERMINISTIC_FAILURE_MARKERS`` analysis in ``workers/_helpers.py`` was
   derived from a **30-day** window over ``state_type='error'``. 30 days is the
   shortest window that keeps both of those honest.

2. **Beyond that, keep terminal rows (``completed``/``error``) until
   ``worker_state_analytics_retention_days`` (default 180).** These are the
   numerator and denominator of the rescue-rate ratio. 180 days is chosen from
   the same distribution: the 90-180d band costs only 458 MB, so keeping a full
   six months of analytics history is nearly free, while capping unbounded
   growth. The oldest row today is ~172 days old, so this rule deletes nothing
   yet -- it exists to bound the future, not to reclaim the present.

3. **Always keep the newest ``worker_state_retention_min_per_worker``
   (default 10) rows per (tenant_id, worker_id), regardless of age or type.**
   This is a correctness guard, not a nicety. ``get_latest_worker_state``
   derives a worker's *current* status from its newest row whatever the type,
   and ``crud_endpoints`` / ``message_endpoints`` gate state transitions on it.
   Deleting the newest ``processing`` row of a long-running or dormant worker
   would silently roll its reported status back to an older terminal state.
   10 also covers the default page size of ``get_worker_states_by_id``
   (``limit=10``), so a dormant worker's history never reads as empty.

Access pattern this was checked against
---------------------------------------
Every read of ``worker_states`` in ``src/`` is newest-first with a hard cap, or
an aggregate ``COUNT``: ``get_worker_states`` (``ORDER BY id DESC LIMIT``,
clamped to 1000), ``get_worker_states_by_id`` (``ORDER BY timestamp DESC
LIMIT``, <= 100), ``get_latest_worker_state`` (limit 1), and
``count_worker_states``. There is no audit export, no history walker, and no
analytics code path. The only things with a claim on old rows are the two
offline operator queries quoted above, which rules 1 and 2 exist to protect.

Tenancy
-------
There is **no row-level security in this repo** -- verified zero ``CREATE
POLICY`` / ``ROW LEVEL SECURITY`` across ``migrations/``,
``postgres_schema.sql`` and ``src/``, and migration 024 says so explicitly at
its type-flip step. Tenant scoping is application-level via ``scope_clause()``.
So this sweep needs no ``SECURITY DEFINER`` resolver and no tenant GUC; it is
deliberately and explicitly **cross-tenant**, which is why every predicate it
uses is tenant-agnostic and the per-worker keep-guard is keyed on
``(tenant_id, worker_id)`` rather than ``worker_id`` alone -- ``worker_id`` is
an opaque, non-unique history key (the FK to ``workers`` was removed
2026-08-13 precisely because 16,505 rows have no matching worker).

What this does NOT do
---------------------
``DELETE`` marks rows dead; it does not return space to the operating system.
Autovacuum will reclaim the space for *reuse* by this table, which is what
bounds growth. Returning the ~2 GB to the filesystem requires a ``VACUUM FULL``
or ``pg_repack``, both of which take heavy locks and are an operator decision,
not something a background loop should ever do on its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Optional

from src.db.db_config import get_pg_connection
from src.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


# --- Why this sweep is bounded in wall-clock, not only in rows --------------
#
# On 2026-08-27 a single batch of this sweep ran for 34 minutes and took
# worker-management down. It was a seq scan of a 990,198-row / 840 MB table
# because `idx_worker_states_timestamp` did not exist -- migration 026 had
# timed out on a lock on every boot since it was written. The sweep held
# `worker_states` long enough for a queued ACCESS EXCLUSIVE (migration 011) to
# stack ten application queries behind it; both pods went 0/1 and every request
# returned `PoolTimeout: couldn't get a connection after 30.00 sec`.
#
# The index is the cure and it now exists. These bounds are the SEAT BELT: a
# row-count bound (`batch_size`, `max_batches_per_pass`) does not bound TIME,
# and the incident was a time problem. A missing index, a bad plan, a bloated
# table or a slow disk all reappear as a long statement, and this sweep must
# never again be able to sit across an hour boundary into its own next tick.
#
# 30s per batch: with the index a batch is milliseconds, so this is ~3 orders
# of magnitude of headroom and only ever fires on a genuinely pathological
# plan.
_BATCH_STATEMENT_TIMEOUT_MS = 30_000

# 900s per pass, against a 3600s loop interval: a pass can consume at most a
# quarter of its own period, so consecutive passes can never overlap. Chosen as
# a fraction of the interval rather than an absolute, because the interval is
# configurable and an absolute would silently stop bounding if it were lowered.
_PASS_DEADLINE_FRACTION = 0.25

# State types that are the durable analytics record. Everything else is a
# transient or bookkeeping state and is eligible for the short window.
TERMINAL_STATE_TYPES: tuple[str, ...] = ("completed", "error")


# One batch, in three bounded steps.
#
# WHY THIS SHAPE (measured on production 2026-08-27, before this fix)
# -------------------------------------------------------------------
# The previous form put the keep-guard subquery in the WHERE of an unbounded
# scan and relied on `ORDER BY s.id LIMIT $4` to stop early. It does not stop
# early: a sort cannot emit its first row until its input is complete, so the
# guard was evaluated for EVERY row in the table before the LIMIT applied.
# `EXPLAIN` on prod, verbatim:
#
#     Limit  (cost=119261460.73..119261463.23 rows=1000 width=4)
#       ->  Sort  (cost=119261460.73..119261822.86 rows=144849 width=4)
#             Sort Key: s.id
#             ->  Seq Scan on worker_states s  (cost=0.00..119253518.82 ...)
#                   Filter: (... AND ((SubPlan 1) >= 10))
#
# A cost of 1.19e8 for ONE batch of 1000. Measured: `EXPLAIN (ANALYZE)` of that
# statement ran **>5 minutes without returning a single batch** and was
# cancelled. At 200 batches per pass on an hourly loop, the sweep would never
# complete a pass and would hold a seq scan over the busiest table in the
# service permanently open against prod.
#
# `idx_worker_states_timestamp` (migration 026) does not rescue it: the
# `timestamp < cutoff` predicate matches ~84% of the table, so the planner
# correctly refuses an index scan for it. The bound has to come from the
# primary key instead.
#
# So: bound FIRST on `id` (append-only, so id order IS age order), then apply
# the guard to that bounded set only.
#
#   $6  exclusive id watermark -- see the cursor note in `sweep_once`.
#
# `$5` is deliberately reused (the `LIMIT` inside the correlated subquery and
# the `>=` comparison must be the same number) -- the connection wrapper
# translates `$N` to NAMED psycopg placeholders precisely so a reused number
# binds to the same argument.
#
# The keep-guard is still "does this row have at least `min_keep` NEWER
# siblings" rather than a `row_number()` window -- but it is only now actually
# bounded, because `idx_worker_states_tenant_worker_id_id` (migration 027)
# carries `id` as its third column. With the two-column index the plan was
# `Index Scan -> Filter(id > s.id) -> Sort -> Limit 10`: it had to read and
# sort EVERY row of a (tenant_id, worker_id) pair. The busiest pair on prod
# holds 131,507 rows. With `id` in the index the same subquery is an
# index-only scan that stops after `min_keep` entries, which is what the
# previous revision of this comment claimed already happened.
_DELETE_BATCH_SQL = """
WITH candidates AS (
    SELECT s.id, s.tenant_id, s.worker_id
    FROM worker_states s
    WHERE s.id > $6
      AND s.timestamp < $1
      AND (s.state_type <> ALL($2::text[]) OR s.timestamp < $3)
    ORDER BY s.id
    LIMIT $4
),
doomed AS (
    SELECT c.id
    FROM candidates c
    WHERE (
            SELECT count(*)
            FROM (
                SELECT 1
                FROM worker_states n
                WHERE n.tenant_id = c.tenant_id
                  AND n.worker_id = c.worker_id
                  AND n.id > c.id
                ORDER BY n.id
                LIMIT $5
            ) AS newer
    ) >= $5
),
removed AS (
    DELETE FROM worker_states t
    USING doomed d
    WHERE t.id = d.id
    RETURNING t.id
)
SELECT (SELECT count(*) FROM removed)      AS deleted,
       (SELECT count(*) FROM candidates)   AS examined,
       (SELECT max(id)  FROM candidates)   AS max_examined_id
"""


@dataclass(frozen=True)
class RetentionPolicy:
    """The resolved window this sweep will enforce."""

    retention_days: int
    analytics_retention_days: int
    min_per_worker: int
    batch_size: int
    max_batches_per_pass: int

    @classmethod
    def from_settings(cls) -> RetentionPolicy:
        """Resolve the policy from the service `Settings`."""
        s = get_settings()
        return cls(
            retention_days=s.worker_state_retention_days,
            analytics_retention_days=s.worker_state_analytics_retention_days,
            min_per_worker=s.worker_state_retention_min_per_worker,
            batch_size=s.worker_state_retention_batch_size,
            max_batches_per_pass=s.worker_state_retention_max_batches_per_pass,
        )

    def cutoffs(self, now: datetime) -> tuple[datetime, datetime]:
        """(general cutoff, analytics cutoff) as absolute instants.

        The analytics window can never be SHORTER than the general one: a
        misconfiguration that inverted them would delete terminal rows the
        general rule was still keeping, which is the one direction of error
        that loses data nothing else can rebuild.
        """
        analytics_days = max(self.analytics_retention_days, self.retention_days)
        return (
            now - timedelta(days=self.retention_days),
            now - timedelta(days=analytics_days),
        )


@dataclass(frozen=True)
class SweepResult:
    """What one pass actually did."""

    deleted: int
    batches: int
    exhausted: bool
    """True when the pass ran out of CANDIDATE rows rather than out of budget.

    Deliberately not "ran out of deletable rows". A batch can legitimately
    delete zero while still making progress -- every candidate it examined was
    held back by the per-worker keep-guard -- and treating that as exhaustion
    would end the pass on the first such batch and never reach the deletable
    rows behind it.
    """

    examined: int = 0
    """Candidate rows inspected. ``examined - deleted`` were guard-protected."""

    timed_out: bool = False
    """True when the pass ended on a time bound rather than on its row budget.

    Kept distinct from ``exhausted`` on purpose. ``exhausted`` means the whole
    eligible range was walked; ``timed_out`` means it was not, and the two must
    never be conflated -- an operator reading "exhausted" concludes there is
    nothing left to delete, which after a timeout is the opposite of the truth.
    """


async def sweep_once(
    policy: Optional[RetentionPolicy] = None,
    *,
    now: Optional[datetime] = None,
    pass_deadline_seconds: Optional[float] = None,
) -> SweepResult:
    """Delete one pass worth of expired rows, in bounded batches.

    Each batch is its own statement, so the row locks it takes are released
    between batches instead of being held across a million-row delete. The pass
    stops at ``max_batches_per_pass`` even if rows remain -- the next pass picks
    up where this one left off, and a backlog that needs more than one pass is a
    log line rather than a transaction that runs for an hour.

    Batches walk forward on an ``id`` watermark rather than re-running the same
    predicate. Without it the pass cannot make progress past guard-protected
    rows: a worker with fewer than ``min_per_worker`` rows total is permanently
    ineligible, sits at a low ``id``, and so is re-selected as a candidate by
    every batch forever -- the same head-of-line rows examined 200 times while
    the deletable rows behind them are never reached. ``worker_states`` is
    append-only with a ``BIGSERIAL`` key, so ascending ``id`` is ascending age
    and a forward-only cursor can never skip a deletable row.
    """
    policy = policy or RetentionPolicy.from_settings()
    if policy.min_per_worker < 1:
        # A guard of 0 would allow deleting a worker's ONLY row, which is what
        # `get_latest_worker_state` reads. Refuse rather than silently obey.
        raise ValueError("worker_state_retention_min_per_worker must be >= 1")

    cutoff, analytics_cutoff = policy.cutoffs(now or datetime.now(UTC))

    deleted = 0
    examined = 0
    batches = 0
    exhausted = False
    after_id = 0
    deadline = None
    if pass_deadline_seconds is not None:
        deadline = asyncio.get_running_loop().time() + pass_deadline_seconds
    timed_out = False

    for _ in range(policy.max_batches_per_pass):
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            # Stop on the pass deadline. NOT `exhausted`: the eligible range was
            # not walked, and reporting exhaustion here would tell an operator
            # the opposite of the truth.
            timed_out = True
            logger.warning(
                "worker_states retention pass hit its %.0fs deadline after %d "
                "batch(es); ending the pass so it cannot overlap the next tick",
                pass_deadline_seconds,
                batches,
            )
            break
        async with get_pg_connection() as conn:
            # SET, then RESET in the finally. Deliberately not SET LOCAL: the
            # pool is autocommit=True, so each statement is its own implicit
            # transaction and a LOCAL setting would be discarded before the
            # statement it is meant to bound. This connection returns to a pool
            # fronting PgBouncer in transaction mode, which does not run
            # DISCARD ALL, so the RESET is what stops a 30s statement_timeout
            # leaking onto unrelated query traffic -- the same reasoning, and
            # the same hazard, as run_migrations() in src/db/db_config.py.
            try:
                await conn.execute(f"SET statement_timeout = {_BATCH_STATEMENT_TIMEOUT_MS}")
                row = await conn.fetchrow(
                    _DELETE_BATCH_SQL,
                    cutoff,
                    list(TERMINAL_STATE_TYPES),
                    analytics_cutoff,
                    policy.batch_size,
                    policy.min_per_worker,
                    after_id,
                )
            except Exception as exc:
                # A statement_timeout arrives here. Ending the pass is correct:
                # the batch is bounded on `after_id`, so the next pass resumes
                # from the same watermark rather than re-walking from zero.
                timed_out = True
                # `.exception`: a statement_timeout and a genuine query error
                # arrive at the same handler, and only the traceback tells them
                # apart. The message names the timeout because that is the
                # expected cause -- the traceback is what catches the case where
                # it is not.
                logger.exception(
                    "worker_states retention batch exceeded %dms and was "
                    "cancelled (%s); ending the pass. The sweep is bounded in "
                    "time on purpose -- investigate the plan, do not raise "
                    "this timeout",
                    _BATCH_STATEMENT_TIMEOUT_MS,
                    str(exc).strip().splitlines()[0],
                )
                break
            finally:
                with contextlib.suppress(Exception):
                    await conn.execute("RESET statement_timeout")
        batches += 1
        if row is None:
            # The statement is a plain SELECT over three CTEs and always
            # returns exactly one row, so this is unreachable in practice. Stop
            # rather than treat it as exhaustion: `exhausted` is what tells the
            # operator the eligible range was fully walked, and a driver that
            # returned nothing has not proved that.
            logger.warning("worker_states retention batch returned no row; ending pass early")
            break
        batch_examined = int(row["examined"])
        if batch_examined == 0:
            # No candidate rows left at or beyond the watermark: this pass has
            # walked the whole eligible range.
            exhausted = True
            break
        deleted += int(row["deleted"])
        examined += batch_examined
        # `max_examined_id` is non-NULL whenever `examined > 0`.
        after_id = int(row["max_examined_id"])

    logger.info(
        "worker_states retention pass: deleted=%d examined=%d batches=%d "
        "exhausted=%s timed_out=%s cutoff=%s analytics_cutoff=%s min_per_worker=%d",
        deleted,
        examined,
        batches,
        exhausted,
        timed_out,
        cutoff.isoformat(),
        analytics_cutoff.isoformat(),
        policy.min_per_worker,
    )
    return SweepResult(
        deleted=deleted,
        batches=batches,
        exhausted=exhausted,
        examined=examined,
        timed_out=timed_out,
    )


async def retention_loop(
    *,
    sweep: Optional[Callable[[], Awaitable[object]]] = None,
    interval_seconds: Optional[int] = None,
    error_backoff_seconds: int = 60,
    label: str = "worker_states",
) -> None:
    """Run :func:`sweep_once` forever, doing the work BEFORE the first sleep.

    The ordering is the whole point. The one prior background loop in this
    service (``pydantic_graph_execution_service._periodic_cleanup``) sleeps
    first, so it reaps nothing for the first five minutes after start. A pod
    that rolls more often than the interval running a sleep-first loop never
    completes a single pass -- the work is scheduled, never done, and the
    absence is invisible because nothing errors. So: sweep, THEN sleep.

    The sleep stays on the error path too, so a pass that keeps failing backs
    off instead of spinning on the database.
    """
    settings = get_settings()
    interval = interval_seconds or settings.worker_state_retention_interval_seconds

    # The deadline is derived HERE and bound to the default runner. `sweep_once`
    # defaults `pass_deadline_seconds` to None -- unbound -- so a call with no
    # arguments would have shipped the bound completely inert: present in the
    # signature, exercised by its unit test, and never once applied in
    # production. An injected `sweep` (tests) keeps full control and gets no
    # deadline imposed on it.
    runner = sweep or functools.partial(
        sweep_once, pass_deadline_seconds=interval * _PASS_DEADLINE_FRACTION
    )

    while True:
        try:
            await runner()
        except asyncio.CancelledError:
            logger.info("%s retention loop cancelled", label)
            raise
        except Exception:
            logger.exception("%s retention pass failed; backing off", label)
            await asyncio.sleep(error_backoff_seconds)
            continue
        await asyncio.sleep(interval)


def start_retention_task() -> Optional[asyncio.Task[None]]:
    """Start the loop as a background task, or return None when disabled."""
    settings = get_settings()
    if not settings.worker_state_retention_enabled:
        logger.info("worker_states retention is disabled (WORKER_STATE_RETENTION_ENABLED)")
        return None
    task = asyncio.create_task(retention_loop())
    logger.info(
        "worker_states retention started: keep %dd (terminal %dd), "
        "min %d rows/worker, %d rows/batch, every %ds",
        settings.worker_state_retention_days,
        settings.worker_state_analytics_retention_days,
        settings.worker_state_retention_min_per_worker,
        settings.worker_state_retention_batch_size,
        settings.worker_state_retention_interval_seconds,
    )
    return task
