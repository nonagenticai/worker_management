"""Function node implementation for pydantic-graph.

This module provides the FunctionNode class that executes a server-side registered function
following the official pydantic-graph patterns for node definition and execution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic_graph import BaseNode, GraphRunContext

from src.security.request_overrides import FunctionCodeNotSupportedError, has_validator_code
from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Server-side functions a graph ``function`` node may name. Empty: nothing in
#: prod uses function nodes (0 of 42 graphs, 2026-09-22). Register here, in code
#: under review, never from a request.
FUNCTION_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {}


@dataclass
class FunctionNode(BaseNode[WorkflowState, None, Any]):
    """CORRECT pydantic-graph node following official documentation.

    Executes Python functions (inline or imported) and follows
    the official pydantic-graph patterns for node definition and execution.
    """

    node_id: str
    function_name: str | None = None
    function_code: str | None = None
    timeout: int = 300
    retry_count: int = 0

    def __post_init__(self) -> None:
        """Validate configuration: code is refused, a name must be registered."""
        if has_validator_code(self.function_code):
            raise FunctionCodeNotSupportedError(
                f"FunctionNode {self.node_id}: function_code is not supported; "
                "request-supplied Python is never executed"
            )
        if not self.function_name:
            raise ValueError("FunctionNode must have a registered function_name")

    async def run(self, ctx: GraphRunContext[WorkflowState]) -> Any:
        """Execute function following CORRECT pydantic-graph pattern.

        NOTE: Return type is determined by next node class in GraphFactory
        This method will be overridden by GraphFactory to return correct next node or End
        """
        start_time = time.time()
        logger.info(f"Executing FunctionNode {self.node_id}")

        try:
            # Only a server-side registered function can run; never inline code.
            if has_validator_code(self.function_code):
                raise FunctionCodeNotSupportedError(
                    f"FunctionNode {self.node_id}: function_code is not supported"
                )
            result = await self._execute_registered_function(ctx)

            # Store result in state
            ctx.state.set_node_output(self.node_id, result)

            # Calculate execution time
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            logger.info(f"FunctionNode {self.node_id} completed in {execution_time:.2f}s")

            # Return result - routing handled by GraphFactory
            return result

        except FunctionCodeNotSupportedError:
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            # Check if we should retry
            current_retries = ctx.state.increment_retry_count(self.node_id)
            if current_retries <= self.retry_count:
                logger.warning(
                    f"FunctionNode {self.node_id} failed, retrying ({current_retries}/{self.retry_count}): {e!s}"
                )
                return await self.run(ctx)
            else:
                error_msg = (
                    f"FunctionNode {self.node_id} failed after {current_retries} retries: {e!s}"
                )
                logger.exception(error_msg)
                ctx.state.mark_node_failed(self.node_id, error_msg)
                raise Exception(error_msg) from e

    async def _execute_registered_function(self, ctx: GraphRunContext[WorkflowState]) -> Any:
        """Execute a function from the server-side :data:`FUNCTION_REGISTRY`.

        ``function_name`` used to be ``importlib.import_module``-ed as
        ``module.function`` (any installed module, any callable). Only names a
        server-side module registered are callable now; the registry is empty
        until one does.
        """
        try:
            func = FUNCTION_REGISTRY.get(self.function_name or "")
            if func is None:
                raise ValueError(f"Function '{self.function_name}' is not registered")

            execution_data = self._build_execution_context(ctx.state)
            if inspect.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(execution_data), timeout=self.timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, execution_data),  # copies contextvars
                    timeout=self.timeout,
                )
            return result

        except TimeoutError:
            raise Exception(f"Function execution timed out after {self.timeout} seconds") from None
        except Exception as e:
            logger.exception("Registered function execution failed")
            raise Exception(f"Registered function execution failed: {e}") from e

    def _build_execution_context(self, state: WorkflowState) -> dict[str, Any]:
        """Build execution context from current state."""
        return {
            "variables": state.variables.copy(),
            "node_outputs": state.node_outputs.copy(),
            "session_id": state.session_id,
            "graph_id": state.graph_id,
            "node_id": self.node_id,
            "completed_nodes": state.completed_nodes.copy(),
            "failed_nodes": state.failed_nodes.copy(),
            "current_time": datetime.now(UTC).isoformat(),
            **state.variables,  # Unpack variables for direct access
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"FunctionNode(id={self.node_id}, function={self.function_name})"
