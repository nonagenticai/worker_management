"""Worker states database module (PostgreSQL with multi-tenancy).

Per-user-within-tenant: every operation takes (tenant_id, subject_id) and the
SELECT/UPDATE/DELETE WHERE clause goes through ``scope_clause()`` so users
within the same tenant can have private state rows that the rest of the
tenant cannot see.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast
from uuid import UUID

import psycopg

from src.api.tenant_constants import tenant_label
from src.db.db_config import get_pg_connection
from src.repositories.visibility import scope_clause
from src.telemetry import get_metrics_client
from src.telemetry.agent_metrics import WORKER_STATE_WRITE_VALIDATED_FALSE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

R = TypeVar("R")

logger = logging.getLogger("worker_state_db")

# Bounds for :meth:`StateDatabase.get_worker_states`. ``worker_states`` is
# append-only history; an unbounded read is what makes a big tenant's list
# endpoint unserveable. DEFAULT is what a caller that passes nothing gets; MAX
# is the ceiling a caller cannot exceed.
DEFAULT_WORKER_STATES_LIMIT = 100
MAX_WORKER_STATES_LIMIT = 1000


def clamp_worker_states_limit(limit: int) -> int:
    """Clamp ``limit`` into ``[1, MAX_WORKER_STATES_LIMIT]``."""
    return 1 if limit < 1 else min(limit, MAX_WORKER_STATES_LIMIT)


def with_retry(
    max_retries: int = 3, retry_delay: float = 0.1
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Decorator for retrying database operations on failure."""

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (psycopg.Error, psycopg.InterfaceError) as e:
                    last_exception = e
                    logger.warning(
                        "Database operation failed, retrying (%d/%d): %s",
                        attempt + 1,
                        max_retries,
                        str(e),
                    )
                    await asyncio.sleep(retry_delay * (2**attempt))
            logger.error(
                "Database operation failed after %d retries: %s",
                max_retries,
                str(last_exception),
            )
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Database operation failed but no exception was captured")

        return wrapper

    return decorator


class StateDatabase:
    """Database to store worker states in PostgreSQL with per-user-within-tenant scoping."""

    @with_retry()
    async def add_worker_state(  # noqa: PLR0913 -- one keyword arg per persisted worker_states column, plus the observe-only ``validated`` label
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        worker_id: str,
        worker_name: str,
        is_agent: bool,
        state_type: str,
        worker_state: dict[str, Any] | bytes | str,
        metadata: Optional[dict[str, Any]] = None,
        visibility: str = "tenant",
        validated: str = WORKER_STATE_WRITE_VALIDATED_FALSE,
    ) -> int:
        """Add new worker state. Returns the new state row id.

        Args:
            validated: What the CALLER knows about the transition into the state
                being written. ⛔ A string from the closed set
                ``WORKER_STATE_WRITE_VALIDATED``, NOT a bool — a bool here is a
                bug and would silently label ``"false"``:

                * ``"true"`` — the caller actually ran
                  ``WorkerStateManager.validate_state_transition``. The
                  ``processing`` write in ``message_endpoints`` passes this,
                  because the validator there genuinely only runs
                  ``if current_state``.
                * ``"false"`` — ⭐ THE EXPOSURE NUMBER. No call ran and nothing
                  proves the transition good. The default, because this function
                  cannot know: it receives a plain ``str`` and INSERTs it, with
                  no enum coercion and no CHECK constraint behind it, so a write
                  that never touched the state machine is indistinguishable here
                  from one that did. The four error/rescue writes in
                  ``message_endpoints`` and every ``crud_endpoints`` write take
                  this default, correctly.
                * ``"by_construction"`` — ⛔ NOT exposure, and must never be
                  summed into it. No call ran, but the from-state is known at
                  the site and the transition is provably valid. The
                  ``completed`` write is exactly this: the handler wrote the
                  ``state_type="processing"`` row itself moments earlier and
                  PROCESSING -> COMPLETED is the first entry of
                  ``STATE_TRANSITIONS[PROCESSING]``.

                It exists purely to label
                ``worker_management_worker_state_writes_total``. ⛔ It gates
                nothing and must never gate anything — see the block comment on
                that counter for the ≈55%-of-traffic measurement.
        """
        effective_subject = subject_id or tenant_id
        logger.info(
            "Adding worker state: tenant_id=%s, subject_id=%s, worker_name=%s, worker_id=%s",
            tenant_id,
            effective_subject,
            worker_name,
            worker_id,
        )

        async with get_pg_connection() as conn:
            timestamp = datetime.now(UTC).replace(tzinfo=None)

            if isinstance(worker_state, bytes):
                state_data = json.loads(worker_state.decode("utf-8"))
            elif isinstance(worker_state, str):
                state_data = json.loads(worker_state)
            else:
                state_data = worker_state

            result = await conn.fetchrow(
                """
                INSERT INTO worker_states
                (tenant_id, subject_id, visibility,
                 tenant_name, worker_id, worker_name, is_agent, state_type, worker_state, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                str(worker_id),
                worker_name,
                is_agent,
                state_type,
                state_data,
                timestamp,
                metadata,
            )

            state_id = int(cast("dict[str, Any]", result)["id"])
            logger.debug("Worker state added successfully with ID %d", state_id)

            # Count the row AFTER the INSERT succeeded, so the counter's
            # denominator is the same population a census of ``worker_states``
            # returns. ``with_retry`` re-enters this function on failure, so a
            # retried write is still counted exactly once — at the attempt that
            # actually persisted. Observe-only, and swallows everything: an
            # instrumentation failure must never turn a persisted state into a
            # raised exception.
            try:
                get_metrics_client().record_worker_state_write(state_type, validated=validated)
            except Exception:  # pragma: no cover - defensive; observe-only
                logger.debug("worker state write counter failed", exc_info=True)

            return state_id

    @with_retry()
    async def add_worker_state_batch(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        states: list[dict[str, Any]],
        visibility: str = "tenant",
    ) -> list[int]:
        """Add multiple worker states atomically. Returns list of new ids."""
        effective_subject = subject_id or tenant_id
        logger.info(
            "Adding batch of %d worker states for tenant=%s subject=%s",
            len(states),
            tenant_id,
            effective_subject,
        )

        if not states:
            return []

        async with get_pg_connection() as conn:
            async with conn.transaction():
                state_ids = []
                base_timestamp = datetime.now(UTC).replace(tzinfo=None)

                for i, state in enumerate(states):
                    timestamp = base_timestamp + timedelta(microseconds=i * 1000)

                    worker_state = state["worker_state"]
                    if isinstance(worker_state, bytes):
                        state_data = json.loads(worker_state.decode("utf-8"))
                    elif isinstance(worker_state, str):
                        state_data = json.loads(worker_state)
                    else:
                        state_data = worker_state

                    try:
                        metadata = state.get("metadata")
                        result = await conn.fetchrow(
                            """
                            INSERT INTO worker_states
                            (tenant_id, subject_id, visibility,
                             tenant_name, worker_id, worker_name, is_agent, state_type, worker_state, timestamp, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            RETURNING id
                            """,
                            tenant_id,
                            effective_subject,
                            visibility,
                            tenant_label(tenant_id, tenant_name),
                            str(state["worker_id"]),
                            state["worker_name"],
                            state.get("is_agent", False),
                            state["state_type"],
                            state_data,
                            timestamp,
                            metadata,
                        )

                        state_id = cast("dict[str, Any]", result)["id"]
                        state_ids.append(state_id)
                        logger.debug("Added worker state with ID %d", state_id)

                    except psycopg.Error:
                        logger.exception(
                            "Failed to insert state %d for worker %s",
                            i,
                            state["worker_id"],
                        )
                        raise

                logger.info("Successfully added batch of %d worker states", len(state_ids))
                return state_ids

    @with_retry()
    async def get_worker_states(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        limit: int = DEFAULT_WORKER_STATES_LIMIT,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve one PAGE of worker states visible under (tenant_id, subject_id).

        The read is bounded in SQL. ``worker_states`` is an append-only history
        table -- ~1.3GB in production -- so an unqualified read of a busy
        tenant's rows is unbounded in both time and memory. ``limit`` is clamped
        to ``MAX_WORKER_STATES_LIMIT``: a caller cannot ask for everything.

        Ordering stays ``id DESC`` (BIGSERIAL PK, newest first), so a caller
        that passes no pagination arguments now gets the newest
        ``DEFAULT_WORKER_STATES_LIMIT`` rows instead of the whole history. Page
        through with ``offset``; a page shorter than ``limit`` is the last one.
        """
        limit = clamp_worker_states_limit(limit)
        offset = max(0, offset)
        logger.info(
            "Retrieving worker states for tenant=%s subject=%s limit=%d offset=%d",
            tenant_id,
            subject_id,
            limit,
            offset,
        )

        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, worker_id, worker_name, tenant_id, subject_id, visibility,
                       is_agent, state_type,
                       worker_state, timestamp, metadata
                FROM worker_states
                WHERE {scope_clause(1, 2)}
                ORDER BY id DESC
                LIMIT $3 OFFSET $4
                """,
                tenant_id,
                subject_id or tenant_id,
                limit,
                offset,
            )

            return [self._row_to_dict(row) for row in rows]

    @with_retry()
    async def count_worker_states(self, *, tenant_id: str, subject_id: str) -> int:
        """The TRUE number of worker states visible under (tenant_id, subject_id).

        ``get_worker_states`` is capped, so ``len(page)`` cannot be reported as
        a total: a tenant with more rows than the cap would be told there are
        exactly ``cap`` -- a silent truncation announcing itself as complete.
        This counts over the SAME scope predicate the page read uses, so the
        two numbers are comparable and a caller can see it was truncated.
        """
        async with get_pg_connection() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM worker_states WHERE {scope_clause(1, 2)}",
                tenant_id,
                subject_id or tenant_id,
            )
        return int(total or 0)

    @staticmethod
    async def _registered_worker_name(tenant_id: str, worker_id: str) -> Optional[str]:
        """The registry name for ``worker_id``, or None when there is nothing to resolve.

        Only UUID-shaped values are looked up. A caller passing a NAME already
        matches every row through the ``worker_name`` column, so the extra query
        would buy nothing; skipping it keeps the by-name path at exactly the one
        statement it has always been.

        Resolution is unambiguous: ``unique_tenant_worker_name`` makes
        (tenant_id, worker_name) unique -- verified live at 0 duplicates across
        691 workers -- and the lookup is pinned to the caller's tenant, so the
        name it returns can only ever be that tenant's. That matters because the
        SAME worker_name legitimately exists in many tenants.

        Returns None for a name, or for a UUID absent from the registry; the
        caller binds that to a ``worker_name = $6`` term, and ``= NULL`` is
        never true, so the disjunct simply drops out.
        """
        try:
            UUID(worker_id)
        except ValueError:
            return None

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                "SELECT worker_name FROM workers WHERE tenant_id = $1 AND worker_id = $2",
                tenant_id,
                worker_id,
            )
        return None if row is None else cast("Optional[str]", row["worker_name"])

    @with_retry()
    async def get_worker_states_by_id(
        self, *, tenant_id: str, subject_id: str, worker_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve states for a specific worker, scoped to (tenant_id, subject_id)."""
        logger.info(
            "Retrieving states for tenant=%s subject=%s worker=%s with limit %d",
            tenant_id,
            subject_id,
            worker_id,
            limit,
        )

        # Resolve the worker's NAME so the query can also match rows keyed by it.
        #
        # ``WorkerStateManager.resolve_worker_id`` deliberately falls back to
        # using the worker_name AS the worker_id for anything not yet in the
        # registry. When such a worker later registers and is minted a UUID, its
        # earlier rows keep the NAME in worker_id. The query below binds the
        # caller's value to both columns, so a lookup BY NAME finds everything
        # (worker_name is populated on 100% of rows), but a lookup BY UUID --
        # what every caller uses once the worker IS registered -- matches
        # neither column on those older rows. Half the worker's history simply
        # disappears from the read, with no error.
        #
        # ``_rekey_name_scoped_states`` repairs the split at registration, but
        # only where the historical rows share the registering worker's tenant
        # AND subject: its UPDATE carries the same scope_clause. Measured
        # against production 2026-08-14, that covers 24 of the 396 split rows
        # (one worker, icp-analysis). The other 372 were written under a
        # tenant/subject matching no registered worker of that name -- 56 under
        # the all-zeros sentinel tenant 00000000-0000-4000-8000-000000000001 --
        # so no future registration can ever reach them. Re-keying cannot be the
        # whole answer; the READ has to tolerate both shapes.
        resolved_name = await self._registered_worker_name(tenant_id, str(worker_id))

        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, worker_id, worker_name, tenant_id, subject_id, visibility,
                       is_agent, state_type,
                       worker_state, timestamp, metadata
                FROM worker_states
                WHERE {scope_clause(1, 2)}
                  AND (worker_id = $3 OR worker_name = $4 OR worker_name = $6)
                ORDER BY timestamp DESC
                LIMIT $5
                """,
                tenant_id,
                subject_id or tenant_id,
                str(worker_id),
                worker_id,
                limit,
                resolved_name,
            )

            logger.info("Retrieved %d rows from database for worker %s", len(rows), worker_id)

            results = []
            for i, row in enumerate(rows):
                try:
                    result_dict = self._row_to_dict(row)
                    results.append(result_dict)
                    logger.debug(
                        "Converted row %d: state_type=%s, timestamp=%s",
                        i,
                        result_dict.get("state_type"),
                        result_dict.get("timestamp"),
                    )
                except KeyError, ValueError, TypeError:
                    logger.exception("Error converting row %d to dict", i)
                    continue

            dropped = len(rows) - len(results)
            if dropped:
                # Surface silent truncation: callers receive fewer rows than the
                # DB returned because some rows failed conversion.
                logger.warning(
                    "Dropped %d/%d malformed worker_state row(s) for worker %s during conversion",
                    dropped,
                    len(rows),
                    worker_id,
                )
            logger.info(
                "Successfully converted %d/%d rows to dictionaries",
                len(results),
                len(rows),
            )
            return results

    @with_retry()
    async def get_latest_worker_state(
        self, *, tenant_id: str, subject_id: str, worker_id: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve the latest state for a specific worker."""
        logger.info(
            "Retrieving latest state for tenant=%s subject=%s worker=%s",
            tenant_id,
            subject_id,
            worker_id,
        )

        states = await self.get_worker_states_by_id(
            tenant_id=tenant_id, subject_id=subject_id, worker_id=worker_id, limit=1
        )
        return states[0] if states else None

    @with_retry()
    async def delete_worker_state(self, *, tenant_id: str, subject_id: str, worker_id: str) -> int:
        """Delete all states for a specific worker visible+mutable under scope."""
        logger.info(
            "Deleting all states for tenant=%s subject=%s worker=%s",
            tenant_id,
            subject_id,
            worker_id,
        )

        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM worker_states WHERE {scope_clause(1, 2)} AND worker_id = $3",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
                str(worker_id),
            )

            deleted_count = int(result.split()[-1]) if result else 0
            logger.info("Deleted %d worker states", deleted_count)
            return deleted_count

    @with_retry()
    async def delete_all_worker_states(self, *, tenant_id: str, subject_id: str) -> int:
        """Delete all worker states visible+mutable under scope (never global)."""
        logger.info("Deleting all worker states for tenant=%s subject=%s", tenant_id, subject_id)

        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM worker_states WHERE {scope_clause(1, 2)}",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
            )

            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(
                "Successfully deleted %d worker state(s) for tenant=%s subject=%s",
                deleted_count,
                tenant_id,
                subject_id,
            )
            return deleted_count

    def _row_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": row["id"],
            "worker_id": str(row["worker_id"]),
            "worker_name": row["worker_name"],
            "tenant_id": row.get("tenant_id"),
            "subject_id": row.get("subject_id"),
            "visibility": row.get("visibility", "tenant"),
            "is_agent": row["is_agent"],
            "state_type": row["state_type"],
            "worker_state": row["worker_state"] or {},
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "metadata": row["metadata"] or {},
        }
