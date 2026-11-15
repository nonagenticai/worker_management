"""Workflow executor node implementation for pydantic-graph.

This module provides the WorkflowExecutorNode class that orchestrates the execution of
WorkerNodes, FunctionNodes, and ConditionNodes in the correct order based on graph edges.
"""

import ast
import logging
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.nodes.condition_node import ConditionNode
from src.worker.graphs.nodes.function_node import FunctionNode
from src.worker.graphs.nodes.worker_node import WorkerNode
from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutorNode(BaseNode[None, None, Any]):
    """Single node that executes an entire workflow using our node implementations.

    This node orchestrates the execution of WorkerNodes, FunctionNodes, and ConditionNodes
    in the correct order based on graph edges, while still using pydantic_graph's framework.
    """

    workflow_nodes: dict[str, dict[str, Any]]
    workflow_edges: list[dict[str, Any]]
    session_id: str
    graph_id: str
    variables: dict[str, Any]
    start_node_id: str | None = None

    def __post_init__(self) -> None:
        """Initialize workflow state."""
        self.node_outputs = {}
        self.completed_nodes = []
        self.failed_nodes = []
        self.start_time = datetime.now()

    async def run(self, ctx: GraphRunContext[None, None]) -> End[Any]:
        """Execute the entire workflow using our node implementations."""
        logger.info(f"Starting workflow execution with {len(self.workflow_nodes)} nodes")

        try:
            # Determine starting node
            current_node_id = self.start_node_id or self._find_start_node()
            if current_node_id is None:
                raise ValueError("No start node found in workflow")
            execution_order = []

            while current_node_id:
                logger.info(f"Executing node: {current_node_id}")
                execution_order.append(current_node_id)

                # Execute the current node
                result = await self._execute_node(current_node_id)

                # Store result in our internal state
                self.node_outputs[current_node_id] = result
                self.completed_nodes.append(current_node_id)

                # Find next node
                current_node_id = self._find_next_node(current_node_id)

                # Prevent infinite loops
                if len(execution_order) > 20:
                    logger.warning("Execution stopped after 20 nodes to prevent infinite loop")
                    break

            logger.info(f"Workflow completed. Executed nodes: {execution_order}")

            # Return the final result
            final_output = {
                "workflow_completed": True,
                "execution_order": execution_order,
                "node_outputs": self.node_outputs,
                "total_nodes_executed": len(execution_order),
            }

            return End(final_output)

        except Exception as e:
            error_msg = f"Workflow execution failed: {e!s}"
            logger.exception(error_msg)

            error_output = {
                "workflow_completed": False,
                "error": error_msg,
                "execution_order": (execution_order if "execution_order" in locals() else []),
                "node_outputs": self.node_outputs,
            }

            return End(error_output)

    async def _execute_node(self, node_id: str) -> Any:
        """Execute a specific node based on its type."""
        node_config = self.workflow_nodes[node_id]
        node_type = node_config["node_type"]

        try:
            if node_type == "worker":
                node = WorkerNode(
                    node_id=node_id,
                    worker_name=node_config["worker_name"],
                    task_template=node_config.get("task_template", ""),
                    timeout=node_config.get("timeout", 300),
                    retry_count=node_config.get("retry_count", 0),
                    framework_config=node_config.get("framework_config", {}),
                )

                # Execute worker task directly
                resolved_task = self._resolve_template(node_config.get("task_template", ""))
                result = await node._execute_worker_task(resolved_task)
                return node._extract_result_data(result)

            elif node_type == "function":
                function_node = FunctionNode(
                    node_id=node_id,
                    function_code=node_config["data"]["function_code"],
                    timeout=node_config.get("timeout", 60),
                    retry_count=node_config.get("retry_count", 0),
                )

                # Create WorkflowState with current execution context
                workflow_state = WorkflowState(
                    session_id=self.session_id,
                    graph_id=self.graph_id,
                    variables=self.variables.copy(),
                    node_outputs=self.node_outputs.copy(),
                    completed_nodes=self.completed_nodes.copy(),
                    failed_nodes=self.failed_nodes.copy(),
                )

                # Create GraphRunContext for FunctionNode
                function_ctx = GraphRunContext(state=workflow_state, deps=None)

                # Execute using the proper pydantic_graph interface
                result = await function_node.run(function_ctx)

                # Handle the result directly
                return result

            elif node_type == "condition":
                condition_node = ConditionNode(
                    node_id=node_id,
                    condition=node_config["data"]["condition"],
                    retry_count=node_config.get("retry_count", 0),
                )

                # Create WorkflowState with current execution context
                workflow_state = WorkflowState(
                    session_id=self.session_id,
                    graph_id=self.graph_id,
                    variables=self.variables.copy(),
                    node_outputs=self.node_outputs.copy(),
                    completed_nodes=self.completed_nodes.copy(),
                    failed_nodes=self.failed_nodes.copy(),
                )

                # Create GraphRunContext for ConditionNode
                condition_ctx = GraphRunContext(state=workflow_state, deps=None)

                # Execute using the proper pydantic_graph interface
                result = await condition_node.run(condition_ctx)

                # Handle the result directly
                return result

            else:
                raise ValueError(f"Unknown node type: {node_type}")

        except Exception:
            logger.exception(f"Failed to execute node {node_id}")
            raise

    def _resolve_template(self, template: str) -> str:
        """Resolve template variables with current state."""
        if not template:
            return ""

        # Replace global variables: {variable_name}
        for var_name, value in self.variables.items():
            placeholder = f"{{{var_name}}}"
            template = template.replace(placeholder, str(value))

        # Replace node outputs: {node_id.output}
        for node_id, output in self.node_outputs.items():
            placeholder = f"{{{node_id}.output}}"
            template = template.replace(placeholder, str(output))

        # Handle special placeholders
        template = template.replace("{session_id}", self.session_id)
        template = template.replace("{graph_id}", self.graph_id)

        return template

    def _find_start_node(self) -> str | None:
        """Find the starting node (node with no incoming edges)."""
        target_nodes = {edge["target"] for edge in self.workflow_edges}
        start_candidates = [
            node_id for node_id in self.workflow_nodes if node_id not in target_nodes
        ]

        if not start_candidates:
            # If all nodes have incoming edges (circular), just pick the first one
            start_candidates = list(self.workflow_nodes.keys())

        return start_candidates[0] if start_candidates else None

    def _find_next_node(self, current_node_id: str) -> str | None:
        """Find the next node to execute based on edges and conditions."""
        next_edges = [edge for edge in self.workflow_edges if edge["source"] == current_node_id]

        if not next_edges:
            return None

        # Check if the current node is a condition node
        current_node = self.workflow_nodes.get(current_node_id)
        if current_node and current_node.get("node_type") == "condition":
            # Get the condition result from node outputs
            condition_result = self.node_outputs.get(current_node_id)

            if isinstance(condition_result, bool):
                # Route based on condition result - first edge for True, second for False
                if condition_result and len(next_edges) > 0:
                    return next_edges[0].get("target")
                elif not condition_result and len(next_edges) > 1:
                    return next_edges[1].get("target")
                elif len(next_edges) > 0:
                    # Fallback to first edge if only one available
                    return next_edges[0].get("target")

        # For edges with explicit conditions, evaluate them
        for edge in next_edges:
            edge_condition = edge.get("condition")
            if edge_condition:
                try:
                    if self._evaluate_edge_condition(edge_condition):
                        return edge.get("target")
                except Exception as e:
                    logger.warning(f"Edge condition evaluation failed: {e}")
                    continue
            else:
                # No condition means this edge is always valid
                return edge.get("target")

        # Fallback to first available edge if no conditions matched
        if next_edges:
            return next_edges[0].get("target")

        return None

    def _evaluate_edge_condition(self, condition: str) -> bool:
        """Evaluate edge condition using safe evaluation similar to ConditionNode.

        Args:
            condition: The condition string to evaluate

        Returns:
            bool: Result of condition evaluation

        Raises:
            Exception: If condition evaluation fails
        """
        try:
            # Resolve template variables in condition
            resolved_condition = self._resolve_template(condition)

            logger.debug(f"Evaluating edge condition: {resolved_condition}")

            # Use safe AST evaluation
            result = self._safe_eval_condition(resolved_condition)

            # Ensure boolean result
            return bool(result)

        except Exception as e:
            logger.exception("Edge condition evaluation failed")
            raise Exception(f"Edge condition evaluation failed: {e}") from e

    def _safe_eval_condition(self, expression: str) -> Any:
        """Safely evaluate condition expression using AST.

        Args:
            expression: The expression to evaluate

        Returns:
            Any: Result of expression evaluation

        Raises:
            ValueError: If expression is invalid or evaluation fails
        """
        try:
            # Parse the expression
            node = ast.parse(expression, mode="eval")

            # Evaluate using safe AST evaluator
            return self._ast_eval_condition(node.body)

        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}") from e

    def _ast_eval_condition(self, node: ast.AST) -> Any:
        """Recursively evaluate AST nodes safely for condition evaluation.

        Args:
            node: AST node to evaluate

        Returns:
            Any: Result of node evaluation

        Raises:
            TypeError: If unsupported AST node type encountered
            NameError: If variable not found
        """
        # Safe operators
        ops: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.BitAnd: operator.and_,
        }

        # Safe comparison operators
        comp_ops: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }

        # Safe boolean operators
        bool_ops: dict[type[ast.boolop], Callable[[Any, Any], Any]] = {
            ast.And: lambda x, y: x and y,
            ast.Or: lambda x, y: x or y,
        }

        # Safe unary operators
        unary_ops: dict[type[ast.unaryop], Callable[[Any], Any]] = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
            ast.Not: operator.not_,
            ast.Invert: operator.invert,
        }

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            # Look up variables
            var_name = node.id
            if var_name in self.variables:
                return self.variables[var_name]
            elif var_name == "True":
                return True
            elif var_name == "False":
                return False
            elif var_name == "None":
                return None
            else:
                raise NameError(f"Variable '{var_name}' not found")
        elif isinstance(node, ast.BinOp):
            left = self._ast_eval_condition(node.left)
            right = self._ast_eval_condition(node.right)
            op = ops.get(type(node.op))
            if op is None:
                raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._ast_eval_condition(node.operand)
            op_func = unary_ops.get(type(node.op))
            if op_func is None:
                raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_func(operand)
        elif isinstance(node, ast.Compare):
            left = self._ast_eval_condition(node.left)
            for comp_op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._ast_eval_condition(comparator)
                comp_func = comp_ops.get(type(comp_op))
                if comp_func is None:
                    raise TypeError(f"Unsupported comparison operator: {type(comp_op).__name__}")
                if not comp_func(left, right):
                    return False
                left = right  # For chained comparisons
            return True
        elif isinstance(node, ast.BoolOp):
            values = [self._ast_eval_condition(value) for value in node.values]
            op = bool_ops.get(type(node.op))
            if op is None:
                raise TypeError(f"Unsupported boolean operator: {type(node.op).__name__}")
            result = values[0]
            for value in values[1:]:
                result = op(result, value)
            return result
        else:
            raise TypeError(f"Unsupported AST node type: {type(node).__name__}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"WorkflowExecutorNode(nodes={len(self.workflow_nodes)}, edges={len(self.workflow_edges)})"
