"""Graph execution database module (PostgreSQL with multi-tenancy).

Per-user-within-tenant: every operation takes (tenant_id, subject_id) and the
SELECT/UPDATE/DELETE WHERE clause goes through ``scope_clause()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Optional, cast
from uuid import UUID

from src.api.tenant_constants import tenant_label
from src.db.db_config import get_pg_connection
from src.db.graph_state_db import GraphStateDatabase, _clean_item
from src.db.workers_state_db import with_retry
from src.repositories.visibility import scope_clause

logger = logging.getLogger(__name__)


def _coerce_correlation_id(value: Optional[str]) -> Optional[str]:
    """Return ``value`` if it is a UUID, else None -- never raise.

    ``graph_execution_sessions.correlation_id`` is declared UUID (migration 012)
    and the fleet-wide join key really is a UUID everywhere it lands:
    ``billing.billing_cost_events``, ``metering.{outcome_event,exec_quality,
    cost_by_correlation,...}`` and ``inference_router.replay_invocation_log`` are
    all UUID columns. Widening this column to text would only break those joins,
    so the type stays.

    But the value arrives from the ``X-Correlation-Id`` HTTP header, which any
    caller can set to anything. When a caller sends a non-UUID, postgres rejects
    the bind for that ONE column and the ENTIRE INSERT fails:

        invalid input syntax for type uuid: "527162"
        CONTEXT: unnamed portal parameter $24

    Because the whole statement dies, the session row -- status, tokens,
    provider, quality score, the lot -- is never persisted, and
    ``with_retry`` then burns three attempts on an error that can never
    succeed. Measured on prod 2026-08-14: ``graph_execution_sessions`` had
    **830 rows with a newest created_at of 2026-06-28** and **0 rows in 24h**,
    while the live pod logged 10 of these failures in a single hour. The table
    has been effectively dead for six weeks, which is also why the #170 canary
    could never find a new session row.

    One malformed optional field must not destroy the whole record. Dropping it
    to NULL costs only the join key on that row -- and that join was already
    impossible, since no UUID-typed counterpart could hold "527162" either.

    The WARNING is deliberate and carries the offending value: the real fix is
    upstream, in whichever caller is putting an n8n execution id in a header the
    fleet contract says is a UUID, and this log is what identifies it.
    """
    if value is None:
        return None
    try:
        UUID(str(value))
    except ValueError, AttributeError, TypeError:
        logger.warning(
            "Dropping non-UUID correlation_id %r: graph_execution_sessions.correlation_id "
            "is UUID (migration 012) and the fleet join key is UUID everywhere. The session "
            "row is persisted WITHOUT the join key rather than lost entirely. Fix the caller.",
            value,
        )
        return None
    return str(value)


def first_valid_correlation_id(*candidates: Optional[str]) -> Optional[str]:
    """Return the first candidate that is actually a UUID, else ``None``.

    The join key arrives by two carriers — the ``X-Correlation-Id`` header and
    the request body's ``correlation_id`` — and the API layer used to resolve
    them with a plain ``header or body``. That is a PRESENCE test, not a
    VALIDITY test, so a present-but-malformed header beat a perfectly good body
    value and the row lost its join key anyway.

    Measured on prod 2026-08-14: the n8n Workers node stamped the n8n execution
    id (a decimal string) into the header on every call, so ``header or body``
    always selected the garbage. Loki, namespace=workers: 57 drops carrying 57
    distinct decimal ids (527423..529894), 1:1 with the 57
    ``graph_execution_sessions`` rows written that day — every one NULL. The
    n8n-side guard (n8n_custom_nodes #1056) had by then made the BODY carry a
    valid UUID, and it could never be reached: 27 of those 57 drops landed after
    it went live. n8n-nodes-workers #29 fixed the header at source; this makes
    the server stop depending on every caller getting it right.

    Precedence is unchanged in spirit — the header still wins on disagreement
    (spec §5.2.1) — but it now wins only among values that COULD be the join
    key. An unusable value is not a disagreement, it is an absence.

    Each rejected candidate is logged by ``_coerce_correlation_id``, so the
    caller that sent garbage is still named. Resolving here rather than at the
    DB layer does not add a second warning: what this returns is valid or None,
    and neither logs again downstream.
    """
    for candidate in candidates:
        coerced = _coerce_correlation_id(candidate)
        if coerced is not None:
            return coerced
    return None


@dataclass(frozen=True)
class ScopeContext:
    """Tenancy + identity scope shared by every session/log persistence call.

    ``tenant_name``/``visibility`` default to the same values the individual
    column params used to default to, so methods that don't persist them simply
    ignore the extra fields.
    """

    tenant_id: str
    subject_id: str
    session_id: str
    tenant_name: str = ""
    visibility: str = "tenant"


@dataclass(frozen=True)
class ExecutionOutputs:
    """The execution-result columns of a graph session (all optional on update)."""

    node_outputs: Optional[dict[str, Any]] = None
    completed_nodes: Optional[list[str]] = None
    failed_nodes: Optional[list[str]] = None
    node_errors: Optional[dict[str, Any]] = None
    worker_assignments: Optional[dict[str, Any]] = None
    final_response: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class InferenceTelemetry:
    """Inference telemetry columns (provider choice, token counts, quality)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    quality_score: Optional[float] = None
    chosen_provider: Optional[str] = None
    fallback_count: Optional[int] = None
    validation_passed: Optional[bool] = None


@dataclass(frozen=True)
class NodeExecutionRecord:
    """A single node-execution log row (everything but the tenancy scope)."""

    node_id: str
    worker_name: str
    task_template: str
    resolved_task: str
    output: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    timeout_seconds: int = 300


class GraphExecutionSessionDB:
    """Database operations for graph execution sessions, node logs, and events."""

    # -- Session CRUD ------------------------------------------------------

    @with_retry()
    async def create_session(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        session_id: str,
        graph_id: str,
        execution_mode: str = "automatic",
        status: str = "initialized",
        variables: Optional[dict[str, Any]] = None,
        visibility: str = "tenant",
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new execution session.

        `correlation_id` (UUID string) is the end-to-end join key propagated
        from n8n / fr-backend (X-Correlation-Id header). It is persisted onto
        the row so the metering_service.exec_quality ETL can join Loki entries
        ↔ cost_events ↔ graph_execution_sessions on a single value.
        """
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        variables_clean = _clean_item(variables) if variables else None
        effective_subject = subject_id or tenant_id

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO graph_execution_sessions
                    (session_id, graph_id, tenant_id, subject_id, visibility,
                     tenant_name, execution_mode, status,
                     variables, correlation_id, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING *
                """,
                str(session_id),
                str(graph_id),
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                execution_mode,
                status,
                variables_clean,
                _coerce_correlation_id(correlation_id),
                timestamp,
                timestamp,
            )

            logger.info(
                "Created execution session: tenant=%s subject=%s session_id=%s, graph_id=%s, correlation_id=%s",
                tenant_id,
                effective_subject,
                session_id,
                graph_id,
                correlation_id,
            )
            return _clean_item(dict(cast("dict[str, Any]", row)))

    @with_retry()
    async def get_session(
        self, *, tenant_id: str, subject_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        """Get session by ID."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM graph_execution_sessions
                WHERE {scope_clause(1, 2)} AND session_id = $3
                """,
                tenant_id,
                subject_id or tenant_id,
                str(session_id),
            )

            if not row:
                return None
            return _clean_item(dict(cast("dict[str, Any]", row)))

    @with_retry()
    async def update_session_status(
        self,
        scope: ScopeContext,
        status: str,
        *,
        outputs: Optional[ExecutionOutputs] = None,
        telemetry: Optional[InferenceTelemetry] = None,
        error_message: Optional[str] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Update session status (and optional fields) when visible+mutable under scope."""
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        out = outputs if outputs is not None else ExecutionOutputs()
        tel = telemetry if telemetry is not None else InferenceTelemetry()

        # WHERE positional plan: $1=tenant_id, $2=subject_id, $3=session_id,
        # $4=status, $5=updated_at, then optional fields starting at $6.
        set_parts = ["status = $4", "updated_at = $5"]
        params: list[Any] = [
            scope.tenant_id,
            scope.subject_id or scope.tenant_id,
            str(scope.session_id),
            status,
            timestamp,
        ]
        idx = 6

        # (column, value, needs JSON cleaning). Only non-None values are persisted,
        # so callers update an arbitrary subset of columns in one round-trip.
        optional_columns: list[tuple[str, Any, bool]] = [
            ("node_outputs", out.node_outputs, True),
            ("completed_nodes", out.completed_nodes, True),
            ("failed_nodes", out.failed_nodes, True),
            ("node_errors", out.node_errors, True),
            ("worker_assignments", out.worker_assignments, True),
            ("error_message", error_message, False),
            ("chosen_provider", tel.chosen_provider, False),
            ("fallback_count", tel.fallback_count, False),
            ("prompt_tokens", tel.prompt_tokens, False),
            ("completion_tokens", tel.completion_tokens, False),
            ("validation_passed", tel.validation_passed, False),
            ("quality_score", tel.quality_score, False),
            ("request_payload", request_payload, True),
            ("final_response", out.final_response, True),
        ]
        for column, value, needs_clean in optional_columns:
            if value is not None:
                set_parts.append(f"{column} = ${idx}")
                params.append(_clean_item(value) if needs_clean else value)
                idx += 1

        if status in ("completed", "failed", "cancelled"):
            set_parts.append(f"completed_at = ${idx}")
            params.append(timestamp)
            idx += 1

        set_clause = ", ".join(set_parts)
        query = f"""
            UPDATE graph_execution_sessions
            SET {set_clause}
            WHERE {scope_clause(1, 2)} AND session_id = $3
            RETURNING *
        """

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(query, *params)

            if not row:
                return None

            logger.info(
                "Updated session status: tenant=%s subject=%s session_id=%s, status=%s",
                scope.tenant_id,
                scope.subject_id,
                scope.session_id,
                status,
            )
            return _clean_item(dict(cast("dict[str, Any]", row)))

    @with_retry()
    async def upsert_finalized_session(  # noqa: PLR0913 -- one keyword arg per persisted session column; grouping them into a struct would only move the same fields behind another name
        self,
        scope: ScopeContext,
        graph_id: str,
        status: str,
        *,
        outputs: ExecutionOutputs,
        telemetry: Optional[InferenceTelemetry] = None,
        error_message: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        execution_mode: str = "automatic",
        request_payload: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Upsert a finalized session row.

        ``ON CONFLICT (session_id) DO UPDATE`` is guarded by a tenant_id
        equality check; cross-tenant overwrites raise PermissionError.
        Within the same tenant, user-visible rows owned by another subject
        are similarly refused (the WHERE predicate bars the UPDATE).

        ``correlation_id`` is the end-to-end join key (see migrations/012).
        It is preserved across upserts via COALESCE so a later finalize-call
        without the value does not blank out an earlier-set id.

        ``quality_score`` (NUMERIC(5,4), added in migration 018) is the
        validator-LLM-worker's numeric quality signal; sibling to
        ``validation_passed``. Consumed downstream by the metering ETL via
        the ``GET /api/graph/sessions/by-correlation/{cid}`` endpoint.
        """
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        terminal_statuses = ("completed", "failed", "cancelled")
        completed_at = timestamp if status in terminal_statuses else None
        # This method is called ONCE, at finalize. Stamping created_at with the
        # finalize time makes every INSERT-path row satisfy
        # completed_at == created_at, so the session's duration is structurally
        # zero and no latency question can be answered from this table. Measured
        # 2026-08-10 on prod: 830 of 830 rows, zero exceptions. The caller knows
        # the real start (the graph's own start_time) and passes it here; falling
        # back to `timestamp` preserves the old behaviour when it is unknown.
        # created_at is deliberately absent from the ON CONFLICT DO UPDATE SET, so
        # a later upsert never overwrites the original start.
        created_at = started_at.replace(tzinfo=None) if started_at is not None else timestamp
        effective_subject = scope.subject_id or scope.tenant_id
        tel = telemetry if telemetry is not None else InferenceTelemetry()

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO graph_execution_sessions (
                    session_id, graph_id, tenant_id, subject_id, visibility,
                    tenant_name, execution_mode, status,
                    variables, node_outputs, completed_nodes, failed_nodes,
                    node_errors, worker_assignments, error_message,
                    chosen_provider, fallback_count, prompt_tokens,
                    completion_tokens, validation_passed, quality_score,
                    request_payload, final_response, correlation_id,
                    created_at, updated_at, completed_at
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8,
                    $9, $10, $11, $12,
                    $13, $14, $15,
                    $16, $17, $18,
                    $19, $20, $21,
                    $22, $23, $24,
                    $25, $26, $27
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    graph_id = EXCLUDED.graph_id,
                    execution_mode = EXCLUDED.execution_mode,
                    status = EXCLUDED.status,
                    variables = EXCLUDED.variables,
                    node_outputs = EXCLUDED.node_outputs,
                    completed_nodes = EXCLUDED.completed_nodes,
                    failed_nodes = EXCLUDED.failed_nodes,
                    node_errors = EXCLUDED.node_errors,
                    worker_assignments = EXCLUDED.worker_assignments,
                    error_message = EXCLUDED.error_message,
                    chosen_provider = EXCLUDED.chosen_provider,
                    fallback_count = EXCLUDED.fallback_count,
                    prompt_tokens = EXCLUDED.prompt_tokens,
                    completion_tokens = EXCLUDED.completion_tokens,
                    validation_passed = EXCLUDED.validation_passed,
                    quality_score = EXCLUDED.quality_score,
                    request_payload = EXCLUDED.request_payload,
                    final_response = EXCLUDED.final_response,
                    correlation_id = COALESCE(EXCLUDED.correlation_id, graph_execution_sessions.correlation_id),
                    updated_at = EXCLUDED.updated_at,
                    completed_at = EXCLUDED.completed_at
                WHERE graph_execution_sessions.tenant_id = EXCLUDED.tenant_id
                  AND (graph_execution_sessions.visibility = 'tenant'
                       OR graph_execution_sessions.subject_id = EXCLUDED.subject_id)
                RETURNING *
                """,
                str(scope.session_id),
                str(graph_id),
                scope.tenant_id,
                effective_subject,
                scope.visibility,
                tenant_label(scope.tenant_id, scope.tenant_name),
                execution_mode,
                status,
                _clean_item(variables) if variables is not None else None,
                _clean_item(outputs.node_outputs),
                _clean_item(outputs.completed_nodes),
                _clean_item(outputs.failed_nodes),
                _clean_item(outputs.node_errors) if outputs.node_errors is not None else None,
                _clean_item(outputs.worker_assignments)
                if outputs.worker_assignments is not None
                else None,
                error_message,
                tel.chosen_provider,
                tel.fallback_count,
                tel.prompt_tokens,
                tel.completion_tokens,
                tel.validation_passed,
                tel.quality_score,
                _clean_item(request_payload) if request_payload is not None else None,
                _clean_item(outputs.final_response) if outputs.final_response is not None else None,
                _coerce_correlation_id(correlation_id),
                created_at,
                timestamp,
                completed_at,
            )

            if row is None:
                logger.warning(
                    "Refusing cross-scope upsert_finalized_session: tenant_id=%s, subject=%s, session_id=%s",
                    scope.tenant_id,
                    effective_subject,
                    scope.session_id,
                )
                raise PermissionError(
                    f"session {scope.session_id} belongs to a different tenant or user"
                )

            logger.info(
                "Upserted finalized session: tenant=%s subject=%s session_id=%s, status=%s",
                scope.tenant_id,
                effective_subject,
                scope.session_id,
                status,
            )
            return _clean_item(dict(row))

    @with_retry()
    async def delete_session(self, *, tenant_id: str, subject_id: str, session_id: str) -> bool:
        """Delete a session and its associated logs/events (cascading)."""
        async with get_pg_connection() as conn:
            _sql = f"DELETE FROM graph_execution_sessions WHERE {scope_clause(1, 2)} AND session_id = $3"  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
            result = await conn.execute(
                _sql,
                tenant_id,
                subject_id or tenant_id,
                str(session_id),
            )

            deleted_count = int(result.split()[-1]) if result else 0
            success = deleted_count > 0

            if success:
                logger.info(
                    "Deleted session: tenant=%s subject=%s session_id=%s",
                    tenant_id,
                    subject_id,
                    session_id,
                )
            return success

    @with_retry()
    async def list_sessions(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        graph_id: Optional[str] = None,
        status: Optional[str] = None,
        validation_passed: Optional[bool] = None,
        correlation_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions visible under (tenant_id, subject_id) with optional filters.

        Filters compose with AND. ``since`` / ``until`` bracket ``created_at``
        (closed lower bound, open upper bound). ``offset`` skips that many
        matching rows (combine with ``limit`` for paging by the caller).
        Used by model-evaluator's eval-set sampler to pull production traces
        for a workflow under a time window — see Plans 3+4 close-out,
        "Eval-tuple payload sourcing" deferral.
        """
        # Positional plan: $1=tenant_id, $2=subject_id, then optional filters,
        # then LIMIT, then OFFSET.
        where_parts = [scope_clause(1, 2)]
        params: list[Any] = [tenant_id, subject_id or tenant_id]
        idx = 3

        if graph_id is not None:
            where_parts.append(f"graph_id = ${idx}")
            params.append(str(graph_id))
            idx += 1

        if status is not None:
            where_parts.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        if validation_passed is not None:
            where_parts.append(f"validation_passed = ${idx}")
            params.append(validation_passed)
            idx += 1

        if correlation_id is not None:
            # A non-UUID filter must match NOTHING, and must not be dropped.
            # Binding it raises `invalid input syntax for type uuid` (a 500 after
            # three futile retries); silently omitting the clause would be worse
            # still -- the caller asked to narrow and would get EVERY session
            # back. No row can hold a non-UUID in a UUID column, so an
            # unsatisfiable predicate is the honest answer.
            if _coerce_correlation_id(correlation_id) is None:
                where_parts.append("false")
            else:
                where_parts.append(f"correlation_id = ${idx}")
                params.append(correlation_id)
                idx += 1

        # The column is TIMESTAMPTZ; a naive datetime would be bound as
        # session-local TZ and could shift the boundary by hours. Coerce
        # naive inputs to UTC explicitly; tz-aware inputs pass through.
        if since is not None:
            where_parts.append(f"created_at >= ${idx}")
            params.append(since if since.tzinfo is not None else since.replace(tzinfo=UTC))
            idx += 1

        if until is not None:
            where_parts.append(f"created_at < ${idx}")
            params.append(until if until.tzinfo is not None else until.replace(tzinfo=UTC))
            idx += 1

        where_clause = " AND ".join(where_parts)
        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        query = f"""
            SELECT * FROM graph_execution_sessions
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_idx}
            OFFSET ${offset_idx}
        """

        async with get_pg_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [_clean_item(dict(r)) for r in rows]

    # SQL column list shared by the tenant-scoped + unscoped by-correlation
    # helpers, kept in one place so the typed Pydantic response model and the
    # two SELECTs cannot drift.
    _SESSIONS_BY_CORRELATION_COLS = """
        session_id,
        correlation_id,
        tenant_id,
        subject_id,
        status,
        validation_passed,
        quality_score,
        created_at,
        completed_at,
        COALESCE(
            (SELECT SUM(n.retry_count)
               FROM node_execution_logs n
              WHERE n.session_id = graph_execution_sessions.session_id),
            0
        ) AS retry_count_sum
    """

    @with_retry()
    async def fetch_sessions_by_correlation_id(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        correlation_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch sessions for ``correlation_id`` under the caller's tenant scope.

        Returns the typed columns the metering ETL needs to populate
        ``exec_quality.{validation_passed, quality_score, retry_count}``
        per-session. ``retry_count`` is summed across all
        ``node_execution_logs`` rows for the session — the per-session
        scalar matches how the ETL aggregates its other columns.

        Scope: tenant_id-restricted; visibility='user' rows belonging to
        other subjects are filtered out. Returns ``[]`` (not 404) when no
        rows match — the API layer translates empty-result to HTTP 404.

        See :meth:`fetch_sessions_by_correlation_id_unscoped` for the
        cross-tenant variant used by internal-service callers (e.g. the
        metering ETL) per the ``metering-internal`` role bypass.
        """
        # No UUID column can hold a non-UUID, so "no rows" is the true answer.
        # Binding it instead raises `invalid input syntax for type uuid`, which
        # `with_retry` turns into three futile attempts and a 500 -- where the
        # API layer's documented contract is empty-result -> HTTP 404.
        if _coerce_correlation_id(correlation_id) is None:
            return []

        _sql = f"""
            SELECT {self._SESSIONS_BY_CORRELATION_COLS}
              FROM graph_execution_sessions
             WHERE {scope_clause(1, 2)}
               AND correlation_id = $3
             ORDER BY created_at DESC
        """  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py

        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                _sql,
                tenant_id,
                subject_id or tenant_id,
                correlation_id,
            )
            return [_clean_item(dict(r)) for r in rows]

    @with_retry()
    async def fetch_sessions_by_correlation_id_unscoped(
        self,
        *,
        correlation_id: str,
    ) -> list[dict[str, Any]]:
        """Cross-tenant variant of :meth:`fetch_sessions_by_correlation_id`.

        Intended for internal-service callers (metering ETL, future
        license-authz / onboarding flows) that need to enrich data across
        tenant boundaries. The API layer must gate this on the caller
        holding an internal-service role per the existing pattern in
        ``licenses/auth/dependencies.py`` (``INTERNAL_SERVICE_ROLES``).
        Direct callers in tests bypass that gate, hence the helper's
        narrow public surface: same SELECT columns as the scoped helper,
        WHERE drops only the scope_clause.

        Audit logging belongs at the API layer (it knows the caller
        identity); this helper is intentionally identity-agnostic.
        """
        # Same reasoning as the scoped variant: a non-UUID matches nothing.
        if _coerce_correlation_id(correlation_id) is None:
            return []

        _sql = f"""
            SELECT {self._SESSIONS_BY_CORRELATION_COLS}
              FROM graph_execution_sessions
             WHERE correlation_id = $1
             ORDER BY created_at DESC
        """  # nosec B608 — parameterized; only the column list is interpolated.

        async with get_pg_connection() as conn:
            rows = await conn.fetch(_sql, correlation_id)
            return [_clean_item(dict(r)) for r in rows]

    # -- Node Execution Logs -----------------------------------------------

    @with_retry()
    async def add_node_execution_log(
        self, scope: ScopeContext, record: NodeExecutionRecord
    ) -> dict[str, Any]:
        """Log a node execution."""
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        effective_subject = scope.subject_id or scope.tenant_id

        start_time = record.start_time
        end_time = record.end_time
        if start_time and start_time.tzinfo:
            start_time = start_time.replace(tzinfo=None)
        if end_time and end_time.tzinfo:
            end_time = end_time.replace(tzinfo=None)

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO node_execution_logs
                    (session_id, tenant_id, subject_id, visibility,
                     tenant_name, node_id, worker_name, task_template,
                     resolved_task, output, error_message, start_time, end_time,
                     retry_count, timeout_seconds, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING *
                """,
                str(scope.session_id),
                scope.tenant_id,
                effective_subject,
                scope.visibility,
                tenant_label(scope.tenant_id, scope.tenant_name),
                record.node_id,
                record.worker_name,
                record.task_template,
                record.resolved_task,
                record.output,
                record.error_message,
                start_time,
                end_time,
                record.retry_count,
                record.timeout_seconds,
                timestamp,
            )

            logger.debug(
                "Added node execution log: session_id=%s, node_id=%s, worker=%s",
                scope.session_id,
                record.node_id,
                record.worker_name,
            )
            return _clean_item(dict(cast("dict[str, Any]", row)))

    @with_retry()
    async def get_session_logs(
        self, *, tenant_id: str, subject_id: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Get all node execution logs for a session."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM node_execution_logs
                WHERE {scope_clause(1, 2)} AND session_id = $3
                ORDER BY created_at ASC
                """,
                tenant_id,
                subject_id or tenant_id,
                str(session_id),
            )
            return [_clean_item(dict(r)) for r in rows]

    @with_retry()
    async def update_node_execution_log(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        log_id: int,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
        end_time: Optional[datetime] = None,
        retry_count: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Update an existing node execution log when visible+mutable under scope."""
        # Positional plan: $1=tenant_id, $2=subject_id, $3=log_id, then optional fields.
        set_parts: list[str] = []
        params: list[Any] = [tenant_id, subject_id or tenant_id, log_id]
        idx = 4

        if output is not None:
            set_parts.append(f"output = ${idx}")
            params.append(output)
            idx += 1

        if error_message is not None:
            set_parts.append(f"error_message = ${idx}")
            params.append(error_message)
            idx += 1

        if end_time is not None:
            if end_time.tzinfo:
                end_time = end_time.replace(tzinfo=None)
            set_parts.append(f"end_time = ${idx}")
            params.append(end_time)
            idx += 1

        if retry_count is not None:
            set_parts.append(f"retry_count = ${idx}")
            params.append(retry_count)
            idx += 1

        if not set_parts:
            async with get_pg_connection() as conn:
                row = await conn.fetchrow(
                    f"SELECT * FROM node_execution_logs WHERE {scope_clause(1, 2)} AND id = $3",
                    tenant_id,
                    subject_id or tenant_id,
                    log_id,
                )
                return _clean_item(dict(cast("dict[str, Any]", row))) if row else None

        set_clause = ", ".join(set_parts)
        query = f"""
            UPDATE node_execution_logs
            SET {set_clause}
            WHERE {scope_clause(1, 2)} AND id = $3
            RETURNING *
        """

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(query, *params)
            if not row:
                return None
            return _clean_item(dict(cast("dict[str, Any]", row)))

    # -- Execution Events --------------------------------------------------

    @with_retry()
    async def add_execution_event(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        session_id: str,
        event_type: str,
        node_id: Optional[str] = None,
        event_data: Optional[dict[str, Any]] = None,
        visibility: str = "tenant",
    ) -> dict[str, Any]:
        """Add an execution event."""
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        event_data_clean = _clean_item(event_data) if event_data else None
        effective_subject = subject_id or tenant_id

        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO execution_events
                    (session_id, tenant_id, subject_id, visibility,
                     tenant_name, event_type, node_id, event_data, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
                """,
                str(session_id),
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                event_type,
                node_id,
                event_data_clean,
                timestamp,
            )

            logger.debug(
                "Added execution event: session_id=%s, type=%s, node=%s",
                session_id,
                event_type,
                node_id,
            )
            return _clean_item(dict(cast("dict[str, Any]", row)))

    @with_retry()
    async def get_session_events(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        session_id: str,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all events for a session, optionally filtered by type."""
        async with get_pg_connection() as conn:
            if event_type:
                rows = await conn.fetch(
                    f"""
                    SELECT * FROM execution_events
                    WHERE {scope_clause(1, 2)} AND session_id = $3 AND event_type = $4
                    ORDER BY timestamp ASC
                    """,
                    tenant_id,
                    subject_id or tenant_id,
                    str(session_id),
                    event_type,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT * FROM execution_events
                    WHERE {scope_clause(1, 2)} AND session_id = $3
                    ORDER BY timestamp ASC
                    """,
                    tenant_id,
                    subject_id or tenant_id,
                    str(session_id),
                )
            return [_clean_item(dict(r)) for r in rows]


# -- Backward compatibility re-exports ------------------------------------

GraphExecutionDatabase = GraphStateDatabase

__all__ = [
    "GraphExecutionDatabase",
    "GraphExecutionSessionDB",
    "GraphStateDatabase",
]
