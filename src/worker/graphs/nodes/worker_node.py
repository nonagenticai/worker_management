"""Worker node implementation for pydantic-graph.

This module provides the WorkerNode class that executes tasks using registered workers/agents
following the official pydantic-graph patterns for node definition and execution.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_graph import BaseNode, GraphRunContext

from src.worker.graphs.pydantic_graph_state import WorkflowState

logger = logging.getLogger(__name__)

# Key a worker may emit in its output dict to report a continuous quality score.
# When present, it is lifted into the inference bag via ``set_quality_score`` so
# the orchestrator finalize path persists it onto
# ``graph_execution_sessions.quality_score`` (consumed downstream by the metering
# exec_quality ETL → governor-effectiveness ``governed_avg_quality``).
QUALITY_SCORE_KEY = "quality_score"


def _normalise_quality_score(raw: Any) -> float | None:
    """Coerce a worker-emitted ``quality_score`` into a [0.0, 1.0] float.

    The ``qa_validator`` worker's output field is a free-form "number"; this
    defends against scale drift (0–1 vs 0–100) before it reaches the
    NUMERIC(5,4) column:

    - non-numeric / NaN / inf / bool → ``None`` (caller skips, bag stays NULL);
    - value in ``[0.0, 1.0]`` → used as-is;
    - value in ``(1.0, 100.0]`` → treated as a percentage → divided by 100;
    - anything else (``< 0`` or ``> 100``) → clamped to the nearest bound.

    Returns:
        The normalised score in [0.0, 1.0], or ``None`` when ``raw`` is not a
        usable numeric quality score.
    """
    # bool is a subclass of int — a True/False is not a quality score.
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except TypeError, ValueError, OverflowError:
        # OverflowError: an int too large for float() (e.g. a pathological worker
        # emitting a giant integer quality_score) must yield None, not escape and
        # abort finalize on the shared gateway.
        return None
    if not math.isfinite(value):
        return None

    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 100.0:
        return value / 100.0
    # Out of every recognised range — clamp to the nearest valid bound.
    return 0.0 if value < 0.0 else 1.0


def record_quality_score(
    state: WorkflowState,
    output: Any,
    *,
    node_id: str = "",
    worker_name: str = "",
) -> bool:
    """Lift a numeric ``quality_score`` from a worker-output dict into the bag.

    Calls ``state.set_quality_score`` so the orchestrator finalize path persists
    it onto ``graph_execution_sessions.quality_score``.

    Shared by BOTH worker-execution paths so the score is lifted identically
    regardless of engine:
      - the standalone :class:`WorkerNode` here (via ``_maybe_record_quality``), and
      - the inline ``WorkerNode`` built by ``graph_factory._create_worker_node_class``
        (the class the ``/execution/graphs/{id}/execute`` engine actually runs).

    No-op unless ``output`` is a dict carrying a usable numeric ``quality_score``;
    workers that don't emit the field (procedure_selector, template_preparer, …)
    leave the bag untouched. Returns ``True`` iff a score was recorded.
    """
    if not isinstance(output, dict) or QUALITY_SCORE_KEY not in output:
        return False
    raw = output[QUALITY_SCORE_KEY]
    score = _normalise_quality_score(raw)
    if score is None:
        logger.debug(
            "WorkerNode %s: ignoring non-numeric quality_score %r from worker %s",
            node_id,
            raw,
            worker_name,
        )
        return False
    if score != raw:
        logger.debug("WorkerNode %s: normalised quality_score %r -> %s", node_id, raw, score)
    state.set_quality_score(score)
    return True


@dataclass
class WorkerNode(BaseNode[WorkflowState, None, Any]):
    """CORRECT pydantic-graph node following official documentation.

    This node executes tasks using registered workers/agents and follows
    the official pydantic-graph patterns for node definition and execution.
    """

    node_id: str
    worker_name: str
    task_template: str
    timeout: int = 300
    retry_count: int = 0
    framework_config: dict = field(default_factory=dict)

    async def run(self, ctx: GraphRunContext[WorkflowState]) -> Any:
        """Execute worker task following CORRECT pydantic-graph pattern.

        NOTE: Return type is determined by next node class in GraphFactory
        This method will be overridden by GraphFactory to return correct next node or End
        """
        start_time = time.time()
        logger.info(f"Executing WorkerNode {self.node_id} with worker {self.worker_name}")

        try:
            # Resolve task template with current state
            resolved_task = self._resolve_template(ctx.state.variables, ctx.state.node_outputs)

            # Get worker from registry and execute
            result = await self._execute_worker_task(
                resolved_task,
                tenant_id=ctx.state.tenant_id,
                subject_id=ctx.state.subject_id,
            )

            # Store result in state
            output = self._extract_result_data(result)
            ctx.state.set_node_output(self.node_id, output)

            # Lift a validator-emitted quality score into the inference bag so
            # the finalize path persists it (only fires for workers that return
            # a numeric ``quality_score`` — e.g. ``qa_validator``; all others
            # are untouched).
            self._maybe_record_quality(ctx, output)

            # Calculate execution time
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            logger.info(f"WorkerNode {self.node_id} completed in {execution_time:.2f}s")

            # Return output - routing handled by GraphFactory
            return output

        except Exception as e:
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            # Check if we should retry
            current_retries = ctx.state.increment_retry_count(self.node_id)
            if current_retries <= self.retry_count:
                logger.warning(
                    f"WorkerNode {self.node_id} failed, retrying ({current_retries}/{self.retry_count}): {e!s}"
                )
                # Recursively retry by calling run again
                return await self.run(ctx)
            else:
                # Mark as failed and raise
                error_msg = (
                    f"WorkerNode {self.node_id} failed after {current_retries} retries: {e!s}"
                )
                logger.exception(error_msg)
                ctx.state.mark_node_failed(self.node_id, error_msg)
                raise Exception(error_msg) from e

    def _maybe_record_quality(self, ctx: GraphRunContext[WorkflowState], output: Any) -> None:
        """Lift a numeric ``quality_score`` from worker output into the inference bag.

        No-op unless ``output`` is a dict carrying a usable numeric
        ``quality_score`` (see :func:`_normalise_quality_score`). Workers that
        don't emit the field (procedure_selector, template_preparer, …) leave
        the bag untouched, so ``graph_execution_sessions.quality_score`` stays
        NULL for them — exactly as before this change.
        """
        record_quality_score(ctx.state, output, node_id=self.node_id, worker_name=self.worker_name)

    async def _execute_worker_task(
        self,
        resolved_task: str,
        *,
        tenant_id: str | None = None,
        subject_id: str | None = None,
    ) -> Any:
        """Execute the task using the worker from cache/registry."""
        try:
            from src.api.identity_context import (
                get_subject_id_from_context,
                get_tenant_id_from_context,
            )
            from src.api.workers import worker_cache
            from src.db.worker_registry_db import WorkerRegistryDB

            # Get worker data from registry.
            registry = WorkerRegistryDB()

            # ⛔ This used to read `tenant_id="system"`, described as "the global
            # system tenant". There is no such tenant: measured against
            # production 2026-08-25, the complete set of non-uuid tenant_id
            # literals is 16 values and `system` is not one of them, and nothing
            # in src/ has ever written it. So this lookup could only ever return
            # None and fall straight into the ValueError below — the branch was
            # dead in the sense that matters, silently, for every graph run that
            # reached it.
            #
            # It also discarded the caller's scope, and `worker_data["config"]`
            # carries the BYO LLM key that `worker_cache.get_or_create_worker`
            # bakes into a shared cached agent. A shared pool keyed on a tenant
            # that does not exist is not isolation; it only looked harmless
            # because it never returned a row.
            #
            # Resolve in the EXECUTING tenant's scope instead: state-threaded
            # identity first, the request ContextVar as fallback (graph execution
            # is a plain `await` inside the request coroutine —
            # graph_execution_api.py:98 — so the ContextVar is live).
            # `ctx.state.tenant_id` is a genuine Optional and is None for
            # non-API callers, so this is `or`, not a getattr default.
            effective_tenant = tenant_id or get_tenant_id_from_context()
            if not effective_tenant:
                # `tenant_id` is a `uuid` column since migration 024, so an empty
                # string here would surface as a 22P02 blamed on the database
                # rather than on the missing identity that actually caused it.
                raise ValueError(
                    f"Cannot resolve worker {self.worker_name}: no tenant in graph "
                    "state or request context"
                )

            # subject_id="" is deliberate and unchanged: scope_clause collapses
            # it to tenant_id, i.e. tenant-visible rows plus the tenant's own.
            worker_data = await registry.get_worker(
                tenant_id=effective_tenant,
                subject_id=subject_id or get_subject_id_from_context() or "",
                worker_name=self.worker_name,
            )
            if not worker_data:
                raise ValueError(f"Worker {self.worker_name} not found in registry")

            # Get or create worker instance from cache.
            #
            # `tenant_id=effective_tenant` is passed EXPLICITLY even though
            # get_or_create_worker falls back to worker_data.get("tenant_id"):
            # `effective_tenant` is the tenant this execution is FOR (graph state,
            # else the request ContextVar) and it is already proven non-empty six
            # lines above, whereas the registry row's own tenant_id is whatever the
            # row was stored with. This is a live LLM path -- it is the one that
            # ends in an outbound inference-router call -- so an unresolved tenant
            # here becomes a nil-sentinel cost event in billing.
            worker = await worker_cache.get_or_create_worker(
                self.worker_name, worker_data, tenant_id=effective_tenant
            )

            # Execute with pydantic-ai agent context if available
            if worker_data.get("is_agent", False):
                result = await self._execute_pydantic_ai_agent(worker, resolved_task)
            else:
                result = await worker.run(resolved_task)

            return result

        except ImportError as e:
            logger.exception("Failed to import worker dependencies")
            raise Exception(f"Worker system not available: {e}") from e
        except Exception:
            logger.exception(f"Worker execution failed for {self.worker_name}")
            raise

    async def _execute_pydantic_ai_agent(self, worker: Any, task: str) -> Any:
        """Execute with enhanced pydantic-ai context."""
        try:
            # Prepare enhanced context for pydantic-ai agent
            enhanced_context = {
                "task": task,
                "graph_context": {
                    "session_id": (worker.state.session_id if hasattr(worker, "state") else None),
                    "graph_id": (worker.state.graph_id if hasattr(worker, "state") else None),
                    "node_id": self.node_id,
                    "variables": (worker.state.variables if hasattr(worker, "state") else {}),
                    "previous_outputs": (
                        worker.state.node_outputs if hasattr(worker, "state") else {}
                    ),
                    "completed_nodes": (
                        worker.state.completed_nodes if hasattr(worker, "state") else []
                    ),
                },
                "node_config": self.framework_config,
            }

            # Execute with enhanced context if the worker supports it
            if hasattr(worker, "run_with_context"):
                result = await worker.run_with_context(enhanced_context)
            else:
                # Fallback to standard execution
                result = await worker.run(task)

            return result

        except Exception:
            logger.exception("Pydantic-AI agent execution failed")
            # Fallback to regular worker execution
            return await worker.run(task)

    def _resolve_template(self, variables: dict, node_outputs: dict) -> str:
        """Resolve template variables with current state."""
        template = self.task_template or ""

        # Replace global variables: {variable_name}
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            template = template.replace(placeholder, str(value))

        # Replace node outputs: {node_id.output}
        for node_id, output in node_outputs.items():
            placeholder = f"{{{node_id}.output}}"
            template = template.replace(placeholder, str(output))

        # Handle special placeholders
        template = template.replace("{session_id}", variables.get("session_id", ""))
        template = template.replace("{graph_id}", variables.get("graph_id", ""))

        return template

    def _extract_result_data(self, result: Any) -> Any:
        """Extract data from various result types."""
        try:
            # Handle pydantic-ai agent results
            if hasattr(result, "output"):
                return result.output
            elif hasattr(result, "data"):
                return result.data

            # Handle WorkerRunResult from non-agent workers
            elif hasattr(result, "success") and hasattr(result, "message"):
                if result.success:
                    return getattr(result, "data", result.message)
                else:
                    error_msg = getattr(result, "error", result.message)
                    raise ValueError(f"Worker execution failed: {error_msg}")

            # Handle direct results
            elif isinstance(result, (dict, list, str, int, float, bool)):
                return result

            # Handle other objects by converting to string
            else:
                return str(result)

        except Exception as e:
            logger.exception("Failed to extract result data")
            # Return a safe representation
            return {
                "error": "Failed to extract result",
                "raw_result_type": str(type(result)),
                "extraction_error": str(e),
            }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"WorkerNode(id={self.node_id}, worker={self.worker_name})"
