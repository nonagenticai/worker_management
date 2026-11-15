"""Pydantic Graph Execution Service — primary execution service using pydantic-graph patterns."""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid7

import psycopg

# OpenTelemetry tracing
try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = None

try:
    from pydantic_graph import Graph
except ImportError:
    # Fallback for missing pydantic_graph dependency. Generic over the same three
    # parameters as `pydantic_graph.graph.Graph` so the annotations below stay valid in
    # both branches — a bare `class Graph:` made them a union of a generic and a
    # non-generic type, which the type checker rejects as "cannot be specialized".
    class Graph[StateT, DepsT, RunEndT]:
        """No-op replacement when the pydantic_graph package is unavailable."""


# Optional DB persistence for execution sessions
try:
    from src.db.graph_execution_db import GraphExecutionDatabase, GraphExecutionSessionDB
except ImportError:
    GraphExecutionDatabase: type | None = None
    GraphExecutionSessionDB: type | None = None

from src.db.graph_state_db import GraphStateDatabase
from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import (
    EdgeSchema,
    ExecutionRequest,
    ExecutionResponse,
    NodeSchema,
)
from src.worker.graphs.pydantic_graph_state import WorkflowState

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(UTC)


class GraphNotFoundError(ValueError):
    """The addressed graph does not exist under the caller's (tenant, subject) scope.

    This is a REQUEST error, not an execution outcome, and the distinction is the
    whole point of the type. ``execute_graph`` deliberately swallows the failures
    that happen *while a graph runs* -- a bad node schema, a worker that raises --
    and reports them as ``ExecutionResponse(status="failed")`` over HTTP 200,
    because in those cases a graph really was executed and the body carries the
    partial result the caller needs (``node_outputs``, ``completed_nodes``,
    ``failed_nodes``) plus a session id it can fetch later.

    None of that is true when the graph does not exist: nothing ran, there are no
    node outputs, and the session id would reference a graph that is not there.
    Reporting it as a 200 made 1,501 consecutive failures (2026-08-27 onwards,
    issue #211) indistinguishable from success to a caller running with n8n's
    ``onError=continueRegularOutput`` -- an exec-quality probe that had not
    executed once and never said so.

    Subclasses ``ValueError`` so every existing ``except ValueError`` on this path
    keeps behaving as it did; the routers additionally catch this type explicitly
    so their 404 no longer depends on the phrase "not found" appearing in a
    message string.
    """

    def __init__(self, graph_id: str) -> None:
        self.graph_id = graph_id
        # Message text preserved verbatim from the untyped `ValueError` this
        # replaces: the routers' pre-existing substring branch still matches it,
        # so the typed branch is a tightening rather than a swap.
        super().__init__("Graph %s not found" % graph_id)


class PydanticGraphExecutionService:
    """Service for executing graphs using pydantic_graph orchestration.

    This is the primary execution service. It handles:
    - Graph execution via pydantic_graph
    - DB persistence of execution sessions
    - OTel tracing spans
    - Session status tracking
    - Cache management
    """

    def __init__(self, execution_db: Any | None = None) -> None:
        """Initialize the pydantic-graph execution service with timeout and cache management."""
        self.active_sessions: dict[str, Graph[WorkflowState, None, Any]] = {}
        self._graph_cache: dict[str, Graph[WorkflowState, None, Any]] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_max_size = 100  # Maximum cached graphs
        self._execution_timeout = 300  # 5 minutes execution timeout
        self._session_cleanup_tasks: dict[str, asyncio.Task] = {}
        self._session_ttl = 3600  # 1 hour session TTL
        self._session_timestamps: dict[str, float] = {}
        self._cleanup_task: asyncio.Task | None = None

        # Optional DB persistence
        self.execution_db = execution_db
        if self.execution_db is None and GraphExecutionDatabase is not None:
            try:
                self.execution_db = GraphExecutionDatabase()
            except Exception:
                logger.warning(
                    "Failed to initialize GraphExecutionDatabase, DB persistence disabled"
                )
                self.execution_db = None

        # Sessions DB for the finalize projection into graph_execution_sessions
        # (writes the 7 inference telemetry columns in addition to standard
        # session fields). Failures never block execution — see _persist_session.
        self.session_db: Any | None = None
        if GraphExecutionSessionDB is not None:
            try:
                self.session_db = GraphExecutionSessionDB()
            except Exception:
                logger.warning(
                    "Failed to initialize GraphExecutionSessionDB, session "
                    "finalize projection disabled"
                )
                self.session_db = None

        logger.info(
            "PydanticGraphExecutionService initialized (db_persistence=%s, session_finalize=%s)",
            self.execution_db is not None,
            self.session_db is not None,
        )

    async def _get_graph_from_database(
        self, tenant_id: str, subject_id: str, graph_id: str
    ) -> dict[str, Any] | None:
        """Get graph data from database under (tenant_id, subject_id) scope."""
        try:
            db = GraphStateDatabase()
            result = await db.get_graph(
                tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
            )
            return result if result is None else dict(result)
        except Exception:
            logger.exception(
                "Failed to get graph %s for tenant=%s subject=%s from database",
                graph_id,
                tenant_id,
                subject_id,
            )
            return None

    async def execute_graph(  # noqa: PLR0915 -- graph execution orchestrates state init, FSM dispatch, persistence, telemetry, cleanup; each phase is a short block
        self,
        tenant_id: str,
        subject_id: str,
        graph_id: str,
        request: Any | None = None,
        variables: dict[str, Any] | None = None,
        start_node: str | None = None,
        visibility: str = "tenant",
        correlation_id: str | None = None,
    ) -> ExecutionResponse:
        """Execute graph using pydantic-graph patterns under (tenant_id, subject_id) scope."""
        # Outer OTel span covering the full execution lifecycle
        span_cm = (
            tracer.start_as_current_span(
                "graph_execution",
                attributes={
                    "graph.id": graph_id,
                    "graph.tenant_id": tenant_id,
                    "graph.subject_id": subject_id,
                    "graph.start_node": start_node or "default",
                },
            )
            if tracer
            else nullcontext()
        )

        with span_cm as span:
            session_id = str(uuid7())
            # Single source of truth for when this execution began. Captured BEFORE
            # the try so the failure paths below -- which can fire before
            # `initial_state` exists -- can still persist a real start time. Without
            # it, `_persist_session` let the DB layer stamp created_at at finalize
            # time, making completed_at == created_at for every row (measured
            # 2026-08-10: 830 of 830 in prod) and every session duration zero.
            execution_started_at = utc_now()

            try:
                # Convert legacy request to ExecutionRequest if needed
                if (
                    request
                    and hasattr(request, "variables")
                    and isinstance(request, ExecutionRequest)
                ):
                    execution_request = request
                elif request and hasattr(request, "variables"):
                    execution_request = ExecutionRequest(
                        variables=getattr(request, "variables", variables or {}),
                        start_node=getattr(request, "start_node", start_node),
                    )
                else:
                    execution_request = ExecutionRequest(
                        variables=variables or {},
                        start_node=start_node,
                    )

                logger.info(
                    "Executing graph %s for tenant=%s subject=%s via pydantic_graph backend",
                    graph_id,
                    tenant_id,
                    subject_id,
                )

                # Get graph data from database
                graph_data = await self._get_graph_from_database(tenant_id, subject_id, graph_id)

                if not graph_data:
                    raise GraphNotFoundError(graph_id)

                # Convert raw database data to proper schema objects
                node_schemas = [NodeSchema(**node_data) for node_data in graph_data["nodes"]]
                edge_schemas = [EdgeSchema(**edge_data) for edge_data in graph_data["edges"]]

                # Create graph via factory
                graph_factory = GraphFactory()
                pg_graph = graph_factory.create_graph(
                    nodes=node_schemas,
                    edges=edge_schemas,
                    name=graph_data.get("name", f"graph_{graph_id}"),
                    validate=True,
                )

                # Create initial state
                execution_variables = variables or execution_request.variables or {}
                initial_state = WorkflowState(
                    session_id=session_id,
                    graph_id=graph_id,
                    variables=execution_variables,
                    node_outputs={},
                    completed_nodes=[],
                    failed_nodes=[],
                    start_time=execution_started_at,
                    end_time=None,
                    error_message=None,
                    # Thread run identity so the OFFICIAL WorkerNode can forward it
                    # as a raw X-Tenant-Id/X-Subject-Id on its loopback call, letting
                    # the auth middleware enforce the tenant's byo_required posture on
                    # graph-executed LLM stages (not the DEFAULT_TENANT_ID fallback).
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                )

                # Find start node
                try:
                    start_node_instance = graph_factory.create_start_node(
                        graph=pg_graph, start_node_id=start_node
                    )
                except Exception as e:
                    logger.exception("Failed to create start node")
                    if hasattr(pg_graph, "_nodes") and pg_graph._nodes:
                        try:
                            nodes_seq = cast("Any", pg_graph._nodes)
                            start_node_instance = nodes_seq[0]()
                            logger.info("Using first node as fallback start node")
                        except Exception:
                            logger.exception(
                                "Fallback start node creation also failed",
                            )
                            raise RuntimeError(f"Cannot create start node: {e}") from None
                    else:
                        raise RuntimeError(
                            f"Cannot create start node and no fallback available: {e}"
                        ) from e

                # Execute using pydantic_graph native execution pattern
                logger.info("Starting pydantic_graph execution with session %s", session_id)
                logger.info("Start node: %s", start_node_instance.__class__.__name__)
                logger.info("Initial variables: %s", execution_variables)

                try:
                    # Inner tracing span for the graph run itself
                    if tracer:
                        with tracer.start_as_current_span(
                            "pydantic_graph.run",
                            attributes={
                                "graph.session_id": session_id,
                                "graph.id": graph_id,
                                "graph.start_node": start_node_instance.__class__.__name__,
                                "graph.node_count": len(graph_data.get("nodes", [])),
                                "graph.edge_count": len(graph_data.get("edges", [])),
                            },
                        ) as run_span:
                            graph_run_result = await asyncio.wait_for(
                                pg_graph.run(
                                    start_node=start_node_instance,
                                    state=initial_state,
                                    deps=None,
                                ),
                                timeout=self._execution_timeout,
                            )
                            run_span.set_attribute("graph.completed", True)  # noqa: FBT003, RUF100
                    else:
                        graph_run_result = await asyncio.wait_for(
                            pg_graph.run(
                                start_node=start_node_instance,
                                state=initial_state,
                                deps=None,
                            ),
                            timeout=self._execution_timeout,
                        )
                except TimeoutError:
                    logger.exception(
                        "Graph execution timed out after %d seconds",
                        self._execution_timeout,
                    )
                    timeout_response = ExecutionResponse(
                        session_id=session_id,
                        status="failed",
                        started_at=initial_state.start_time or utc_now(),
                        updated_at=utc_now(),
                        completed_nodes=initial_state.completed_nodes,
                        failed_nodes=initial_state.failed_nodes,
                        node_outputs=initial_state.node_outputs,
                        final_output=None,
                        error_message=f"Execution timed out after {self._execution_timeout} seconds",
                    )

                    # Persist timeout to DB
                    await self._persist_session(
                        tenant_id=tenant_id,
                        subject_id=subject_id,
                        graph_id=graph_id,
                        session_id=session_id,
                        status="failed",
                        node_outputs=initial_state.node_outputs,
                        completed_nodes=initial_state.completed_nodes,
                        failed_nodes=initial_state.failed_nodes,
                        error_message=timeout_response.error_message,
                        final_output=None,
                        framework_state=initial_state.framework_state,
                        visibility=visibility,
                        correlation_id=correlation_id,
                        started_at=execution_started_at,
                    )

                    if span:
                        span.set_attribute("graph.status", "failed")
                        span.set_attribute(
                            "graph.error", timeout_response.error_message or "timeout"
                        )

                    return timeout_response

                # Store session for monitoring
                self.active_sessions[session_id] = pg_graph
                self._session_timestamps[session_id] = time.time()
                self._ensure_cleanup_running()

                # Extract results from GraphRunResult
                final_result = (
                    graph_run_result.output if hasattr(graph_run_result, "output") else None
                )
                final_state = (
                    graph_run_result.state if hasattr(graph_run_result, "state") else initial_state
                )

                # Update final state
                final_state.end_time = utc_now()

                # Build execution metadata
                self._build_execution_metadata(final_state, session_id)

                # Determine status
                status = "completed" if final_result and not final_state.error_message else "failed"

                logger.info("pydantic_graph execution completed: %s", status)
                logger.info("Final result type: %s", type(final_result))
                logger.info("Node outputs: %d nodes", len(final_state.node_outputs))

                response = ExecutionResponse(
                    session_id=session_id,
                    status=status,
                    started_at=final_state.start_time or utc_now(),
                    updated_at=final_state.end_time or utc_now(),
                    completed_nodes=final_state.completed_nodes,
                    failed_nodes=final_state.failed_nodes,
                    node_outputs=final_state.node_outputs,
                    final_output=final_result,
                    error_message=final_state.error_message,
                )

                # Persist session to DB
                await self._persist_session(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    graph_id=graph_id,
                    session_id=response.session_id,
                    status=response.status,
                    node_outputs=response.node_outputs,
                    completed_nodes=response.completed_nodes,
                    failed_nodes=response.failed_nodes,
                    error_message=response.error_message,
                    final_output=response.final_output,
                    framework_state=final_state.framework_state,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    started_at=execution_started_at,
                )

                if span:
                    span.set_attribute("graph.status", response.status)
                    span.set_attribute("graph.completed_nodes", len(response.completed_nodes))
                    span.set_attribute("graph.failed_nodes", len(response.failed_nodes))

                return response

            except GraphNotFoundError:
                # MUST precede the broad clause below -- `GraphNotFoundError` is a
                # `ValueError`, so the swallow would otherwise claim it first and
                # return HTTP 200 (issue #211).
                #
                # Nothing executed, so there is no execution to report and no
                # session worth persisting: writing one would file 1,501 FAILED
                # `graph_execution_sessions` rows pointing at a graph_id absent
                # from `graphs`, which is what production accumulated. Re-raise and
                # let the router answer 404.
                if span:
                    span.set_attribute("graph.status", "not_found")
                # WARNING, not exception(): a caller addressing a graph it cannot
                # see is a client error, not a service fault. The identity is on
                # the line because the scope, not the id, is what usually differs
                # -- issue #211's graph existed under a different tenant.
                logger.warning(
                    "Graph %s not found for tenant=%s subject=%s; responding HTTP 404",
                    graph_id,
                    tenant_id,
                    subject_id,
                )
                raise

            except (ValueError, TypeError, RuntimeError) as e:
                # Reached only when a graph that DOES exist failed while running.
                # 200 + status="failed" is the deliberate contract here: the body
                # carries the partial result and a real session id. The graph and
                # session are named so this ERROR line is greppable on its own,
                # rather than only inside exc_info.
                logger.exception(
                    "Graph execution failed for graph=%s session=%s tenant=%s",
                    graph_id,
                    session_id,
                    tenant_id,
                )

                if span:
                    span.set_attribute("graph.status", "failed")
                    span.set_attribute("graph.error", str(e))
                    span.record_exception(e)

                # Persist failed session to DB
                await self._persist_session(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    graph_id=graph_id,
                    session_id=session_id,
                    status="failed",
                    node_outputs={},
                    completed_nodes=[],
                    failed_nodes=[],
                    error_message=f"Execution error: {e!s}",
                    final_output=None,
                    framework_state=None,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    started_at=execution_started_at,
                )

                return ExecutionResponse(
                    session_id=session_id,
                    status="failed",
                    node_outputs={},
                    completed_nodes=[],
                    failed_nodes=[],
                    error_message=f"Execution error: {e!s}",
                    started_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )

            except Exception as e:
                logger.exception("pydantic_graph execution failed")
                logger.exception("Exception details: %s", type(e).__name__)
                import traceback

                logger.exception("Full traceback: %s", traceback.format_exc())

                if span:
                    span.set_attribute("graph.status", "failed")
                    span.set_attribute("graph.error", str(e))
                    span.record_exception(e)

                # Persist failed session to DB
                await self._persist_session(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    graph_id=graph_id,
                    session_id=session_id,
                    status="failed",
                    node_outputs={},
                    completed_nodes=[],
                    failed_nodes=[],
                    error_message=str(e),
                    final_output=None,
                    framework_state=None,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    started_at=execution_started_at,
                )

                return ExecutionResponse(
                    session_id=session_id,
                    status="failed",
                    started_at=utc_now(),
                    updated_at=utc_now(),
                    completed_nodes=[],
                    failed_nodes=[],
                    node_outputs={},
                    final_output=None,
                    error_message=str(e),
                )

    async def execute_graph_stepwise(
        self,
        tenant_id: str,
        subject_id: str,
        graph_id: str,
        variables: dict[str, Any] | None = None,
        start_node: str | None = None,
        visibility: str = "tenant",
        correlation_id: str | None = None,
    ) -> ExecutionResponse:
        """Start stepwise execution of a graph (delegates to regular execution).

        Note: This delegates to regular execution since step-by-step control
        has been removed from the API.
        """
        logger.info("execute_graph_stepwise delegating to regular execution")
        return await self.execute_graph(
            tenant_id=tenant_id,
            subject_id=subject_id,
            graph_id=graph_id,
            variables=variables,
            start_node=start_node,
            visibility=visibility,
            correlation_id=correlation_id,
        )

    async def _persist_session(  # noqa: PLR0913 -- one arg per persisted session column
        self,
        tenant_id: str,
        subject_id: str,
        graph_id: str,
        session_id: str,
        status: str,
        node_outputs: dict[str, Any],
        completed_nodes: list[str],
        failed_nodes: list[str],
        error_message: str | None,
        final_output: Any | None,
        framework_state: dict[str, Any] | None = None,
        visibility: str = "tenant",
        correlation_id: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """Persist execution session to DB if available. Never raises.

        Writes to two separate tables:
        1. ``graph_states`` (via ``execution_db.add_graph_state``) — the
           existing JSON-blob store used by other code paths in
           ``graph_operations.py``. Not removed or modified.
        2. ``graph_execution_sessions`` (via
           ``session_db.upsert_finalized_session``) — the first-class session
           row with per-column inference telemetry projected from
           ``framework_state['inference']``. New in Phase 1 inference-graph.

        Both writes are independent and wrapped in their own error handlers so
        one failing never blocks the other or propagates to the caller.
        """
        # 1) Legacy JSON-blob write — preserved verbatim.
        if self.execution_db is not None:
            try:
                await self.execution_db.add_graph_state(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    graph_id=graph_id,
                    state_payload={
                        "session_id": session_id,
                        "status": status,
                        "node_outputs": node_outputs,
                        "completed_nodes": completed_nodes,
                        "failed_nodes": failed_nodes,
                        "error_message": error_message,
                        "final_output": final_output,
                        "event_type": "execution_complete",
                    },
                    visibility=visibility,
                )
            except Exception:
                logger.warning(
                    "Failed to persist execution session %s to DB",
                    session_id,
                    exc_info=True,
                )

        # 2) Finalize projection into graph_execution_sessions. Project the
        # inference bag (populated by Task 3's InferenceProviderNode and
        # Task 4's EvalSubgraphNode) into first-class columns.
        if self.session_db is None:
            return

        inference_bag: dict[str, Any] = {}
        if framework_state:
            candidate = framework_state.get(WorkflowState.INFERENCE_KEY)
            if isinstance(candidate, dict):
                inference_bag = candidate

        attempts_raw = inference_bag.get("attempts", [])
        attempts: list[dict[str, Any]] = [a for a in attempts_raw if isinstance(a, dict)]
        chosen_provider = inference_bag.get("chosen_provider")

        fallback_count: int | None
        prompt_tokens: int | None
        completion_tokens: int | None

        if chosen_provider is not None:
            fallback_count = max(0, len(attempts) - 1)
            winning = next(
                (a for a in reversed(attempts) if a.get("status") == "ok"),
                None,
            )
            prompt_tokens = winning.get("tokens_prompt") if winning else None
            completion_tokens = winning.get("tokens_completion") if winning else None
        elif attempts:
            # All providers attempted, none succeeded.
            fallback_count = len(attempts)
            prompt_tokens = None
            completion_tokens = None
        else:
            # Non-inference graph: the bag is empty / absent.
            fallback_count = None
            prompt_tokens = None
            completion_tokens = None

        validation_passed = inference_bag.get("validation_passed")
        quality_score_raw = inference_bag.get("quality_score")
        request_payload_raw = inference_bag.get("request_payload")
        final_response_raw = inference_bag.get("final_response")

        quality_score = (
            float(quality_score_raw) if isinstance(quality_score_raw, (int, float)) else None
        )
        request_payload = request_payload_raw if isinstance(request_payload_raw, dict) else None
        final_response = final_response_raw if isinstance(final_response_raw, dict) else None

        from src.db.graph_execution_db import (
            ExecutionOutputs,
            InferenceTelemetry,
            ScopeContext,
        )

        try:
            await self.session_db.upsert_finalized_session(
                ScopeContext(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    visibility=visibility,
                ),
                graph_id,
                status,
                outputs=ExecutionOutputs(
                    node_outputs=node_outputs,
                    completed_nodes=completed_nodes,
                    failed_nodes=failed_nodes,
                    final_response=final_response,
                ),
                telemetry=InferenceTelemetry(
                    chosen_provider=chosen_provider,
                    fallback_count=fallback_count,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    validation_passed=validation_passed,
                    quality_score=quality_score,
                ),
                error_message=error_message,
                request_payload=request_payload,
                correlation_id=correlation_id,
                started_at=started_at,
            )
        except (psycopg.Error, OSError) as exc:
            # Narrow catch — connectivity/DB errors only. Anything else
            # (e.g. programmer errors in the projection code) should surface
            # in tests rather than being silently swallowed.
            logger.warning(
                "Failed to upsert finalized session %s into graph_execution_sessions: %s",
                session_id,
                exc,
            )

    async def get_session_status(
        self, tenant_id: str, subject_id: str, session_id: str
    ) -> dict[str, Any] | None:
        """Get session status from the sessions DB (plus in-memory fallback).

        Primary path queries ``graph_execution_sessions`` to return the full
        finalized row — including the 5 inference telemetry fields surfaced
        by ``ExecutionResponse``:

        - ``chosen_provider``
        - ``fallback_count``
        - ``prompt_tokens``
        - ``completion_tokens``
        - ``validation_passed``

        Fallback path (used when no sessions DB is available or the session
        is still in-flight) returns a minimal dict built from
        ``active_sessions`` with ``status="running"`` — without the DB row
        there is no evidence of a terminal outcome, so the session must not
        be reported as ``completed``.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            subject_id: Subject identifier for per-user scoping (empty == tenant-only)
            session_id: Session ID to check

        Returns:
            Session status dictionary or None if not found.
        """
        # Prefer the DB row so the 5 new inference fields are populated
        # for completed/failed sessions.
        if self.session_db is not None:
            try:
                row = await self.session_db.get_session(
                    tenant_id=tenant_id, subject_id=subject_id, session_id=session_id
                )
            except (psycopg.Error, OSError) as exc:
                logger.warning(
                    "Failed to query session %s for telemetry fields: %s",
                    session_id,
                    exc,
                )
                row = None
            if row is not None:
                return {
                    "session_id": str(row.get("session_id", session_id)),
                    "status": row.get("status", "unknown"),
                    "completed_nodes": row.get("completed_nodes") or [],
                    "failed_nodes": row.get("failed_nodes") or [],
                    "node_outputs": row.get("node_outputs") or {},
                    "current_node": None,
                    "final_output": None,
                    "error_message": row.get("error_message"),
                    "started_at": row.get("created_at") or utc_now(),
                    "updated_at": row.get("updated_at") or utc_now(),
                    "chosen_provider": row.get("chosen_provider"),
                    "fallback_count": row.get("fallback_count"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                    "validation_passed": row.get("validation_passed"),
                    "quality_score": row.get("quality_score"),
                }

        # Fallback: the session is known to this process but has no finalized
        # row in ``graph_execution_sessions`` (no sessions DB configured, the
        # persist has not landed yet, or the session is still in-flight).
        #
        # The only authoritative record of completion is that DB row. Without
        # it we have no evidence the graph finished, so we must NOT claim
        # ``completed`` — that told every poller "finished, and it produced
        # nothing" (``final_output`` is None here) about a graph that may
        # still be running. ``running`` is the honest answer: known session,
        # no terminal outcome recorded. It is part of the documented status
        # vocabulary on ``ExecutionResponse`` ("running" / "completed" /
        # "failed") and matches ``WorkflowState._get_overall_status``.
        if session_id in self.active_sessions:
            return {
                "session_id": session_id,
                "status": "running",
                "completed_nodes": [],
                "failed_nodes": [],
                "node_outputs": {},
                "current_node": None,
                "final_output": None,
                "error_message": None,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            }
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        pg_stats = self.get_cache_stats()
        return {
            "active_sessions": len(self.active_sessions),
            "pydantic_graph_stats": pg_stats,
        }

    def get_available_engines(self) -> list[str]:
        """Get available execution engines."""
        return ["simple", "pydantic_graph"]

    def _build_execution_metadata(
        self, final_state: WorkflowState, session_id: str
    ) -> dict[str, Any]:
        """Build execution metadata from the final workflow state."""
        metadata: dict[str, Any] = {
            "execution_type": "pydantic_graph_documented_pattern",
            "session_id": session_id,
            "implementation": "native_pydantic_graph_run_method",
        }

        metadata.update(
            {
                "nodes_executed": len(final_state.completed_nodes),
                "failed_nodes": len(final_state.failed_nodes),
                "total_processed": len(final_state.completed_nodes) + len(final_state.failed_nodes),
                "history_available": False,
            }
        )

        if final_state.start_time:
            metadata["execution_start"] = final_state.start_time.isoformat()
        if final_state.end_time:
            metadata["execution_end"] = final_state.end_time.isoformat()

        return metadata

    def _get_cache_key(self, nodes: Any, edges: Any) -> str:
        """Generate cache key from nodes and edges."""
        nodes_data = []
        edges_data = []

        for node in nodes:
            if hasattr(node, "model_dump"):
                nodes_data.append(node.model_dump(mode="json"))
            else:
                nodes_data.append(node)

        for edge in edges:
            if hasattr(edge, "model_dump"):
                edges_data.append(edge.model_dump(mode="json"))
            else:
                edges_data.append(edge)

        cache_data = {"nodes": nodes_data, "edges": edges_data}
        cache_str = json.dumps(cache_data, sort_keys=True)
        # MD5 used for cache key deduplication only, not for security purposes
        return hashlib.md5(cache_str.encode(), usedforsecurity=False).hexdigest()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_graphs": len(self._graph_cache),
            "active_sessions": len(self.active_sessions),
            "cache_ttl_seconds": self._cache_ttl,
        }

    def clear_cache(self) -> None:
        """Clear all cached graphs and sessions."""
        self._graph_cache.clear()
        self._cache_timestamps.clear()
        self.active_sessions.clear()
        logger.info("PydanticGraphExecutionService caches cleared")

    def _cleanup_expired_cache_entries(self) -> None:
        """Remove expired cache entries based on TTL."""
        current_time = utc_now()
        expired_keys = []

        for key, timestamp in self._cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._graph_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        if expired_keys:
            logger.info("Cleaned up %d expired cache entries", len(expired_keys))

    def _enforce_cache_size_limit(self) -> None:
        """Enforce maximum cache size by removing oldest entries."""
        if len(self._graph_cache) <= self._cache_max_size:
            return

        sorted_items = sorted(self._cache_timestamps.items(), key=lambda x: x[1])
        items_to_remove = len(self._graph_cache) - self._cache_max_size

        for i in range(items_to_remove):
            key = sorted_items[i][0]
            self._graph_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        logger.info("Enforced cache size limit: removed %d oldest entries", items_to_remove)

    def _ensure_cleanup_running(self) -> None:
        """Ensure the periodic session cleanup task is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
            except RuntimeError:
                pass  # No event loop yet

    async def _periodic_cleanup(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                now = time.time()
                expired = [
                    sid
                    for sid, ts in self._session_timestamps.items()
                    if now - ts > self._session_ttl
                ]
                for sid in expired:
                    logger.info("Cleaning up expired session: %s", sid)
                    await self.cleanup_session(sid)
                if expired:
                    logger.info("Cleaned up %d expired sessions", len(expired))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Session cleanup error: %s", e)

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        self.active_sessions.pop(session_id, None)
        self._session_timestamps.pop(session_id, None)

        cleanup_task = self._session_cleanup_tasks.pop(session_id, None)
        if cleanup_task and not cleanup_task.done():
            cleanup_task.cancel()

        logger.info("Cleaned up session: %s", session_id)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# The adapter class name used by many tests and the API
GraphExecutionService = PydanticGraphExecutionService

# Global service instance
_EXECUTION_SERVICE: PydanticGraphExecutionService | None = None


def get_execution_service() -> PydanticGraphExecutionService:
    """Get the global graph execution service instance."""
    global _EXECUTION_SERVICE  # noqa: PLW0603 -- lazy-init singleton
    if _EXECUTION_SERVICE is None:
        _EXECUTION_SERVICE = PydanticGraphExecutionService()
    return _EXECUTION_SERVICE
