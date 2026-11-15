"""Condition node implementation for pydantic-graph.

This module provides the ConditionNode class that evaluates conditions for branching logic
using safe AST evaluation following the official pydantic-graph patterns.
"""

import ast
import logging
import operator
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class ConditionNode(BaseNode[WorkflowState, None, bool]):
    """CORRECT pydantic-graph node following official documentation.

    Evaluates conditions for branching logic using safe AST evaluation
    and follows the official pydantic-graph patterns for node definition.
    """

    node_id: str
    condition: str
    retry_count: int = 0

    async def run(
        self, ctx: GraphRunContext[WorkflowState]
    ) -> BaseNode[WorkflowState, None, Any] | End[bool]:
        """Evaluate condition following CORRECT pydantic-graph pattern.

        NOTE: Return type is determined by next node class in GraphFactory
        This method will be overridden by GraphFactory to return correct next node or End
        """
        start_time = time.time()
        logger.info(f"Executing ConditionNode {self.node_id} with condition: {self.condition}")

        try:
            # Evaluate the condition
            result = self._evaluate_condition(ctx.state)

            # Store result in state
            ctx.state.set_node_output(self.node_id, result)

            # Calculate execution time
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            logger.info(
                f"ConditionNode {self.node_id} evaluated to {result} in {execution_time:.2f}s"
            )

            # Return End with boolean result - routing handled by GraphFactory
            return End(result)

        except Exception as e:
            execution_time = time.time() - start_time
            ctx.state.set_node_duration(self.node_id, execution_time)

            # Check if we should retry
            current_retries = ctx.state.increment_retry_count(self.node_id)
            if current_retries <= self.retry_count:
                logger.warning(
                    f"ConditionNode {self.node_id} failed, retrying ({current_retries}/{self.retry_count}): {e!s}"
                )
                return await self.run(ctx)
            else:
                error_msg = (
                    f"ConditionNode {self.node_id} failed after {current_retries} retries: {e!s}"
                )
                logger.exception(error_msg)
                ctx.state.mark_node_failed(self.node_id, error_msg)
                raise RuntimeError(error_msg) from e

    def _evaluate_condition(self, state: WorkflowState) -> bool:
        """Evaluate condition using safe AST evaluation."""
        try:
            # Resolve template variables in condition
            resolved_condition = self._resolve_condition_template(state)

            logger.debug(f"Evaluating resolved condition: {resolved_condition}")

            # Use safe AST evaluation
            result = self._safe_eval(resolved_condition, state)

            # Ensure boolean result
            return bool(result)

        except Exception as e:
            logger.exception("Condition evaluation failed")
            raise ValueError(f"Condition evaluation failed: {e}") from e

    def _resolve_condition_template(self, state: WorkflowState) -> str:
        """Resolve template variables in condition string."""
        condition = self.condition

        # Build variables context
        variables = {
            "session_id": state.session_id,
            "graph_id": state.graph_id,
            "node_id": self.node_id,
            "completed_count": len(state.completed_nodes),
            "failed_count": len(state.failed_nodes),
            **state.variables,
        }

        # Replace template variables
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            if isinstance(value, str):
                condition = condition.replace(placeholder, f"'{value}'")
            else:
                condition = condition.replace(placeholder, str(value))

        # Replace node outputs: {node_id.output}
        for node_id, output in state.node_outputs.items():
            placeholder = f"{{{node_id}.output}}"
            if isinstance(output, str):
                condition = condition.replace(placeholder, f"'{output}'")
            elif isinstance(output, (int, float, bool)):
                condition = condition.replace(placeholder, str(output))
            else:
                # For complex objects, convert to string
                condition = condition.replace(placeholder, f"'{output!s}'")

        # Handle special placeholders
        condition = condition.replace("{session_id}", f"'{variables.get('session_id', '')}'")
        condition = condition.replace("{graph_id}", f"'{variables.get('graph_id', '')}'")
        condition = condition.replace("{node_id}", f"'{self.node_id}'")
        condition = condition.replace("{completed_count}", str(variables.get("completed_count", 0)))
        condition = condition.replace("{failed_count}", str(variables.get("failed_count", 0)))

        return condition

    def _safe_eval(self, expression: str, state: WorkflowState) -> Any:
        """Safely evaluate expression using AST."""
        try:
            # Parse the expression
            node = ast.parse(expression, mode="eval")

            # Evaluate using safe AST evaluator
            return self._ast_eval(node.body, state)

        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}") from e

    def _ast_eval(self, node: ast.AST, state: WorkflowState) -> Any:
        """Recursively evaluate AST nodes safely."""
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
            if var_name in state.variables:
                return state.variables[var_name]
            elif var_name == "True":
                return True
            elif var_name == "False":
                return False
            elif var_name == "None":
                return None
            else:
                raise NameError(f"Variable '{var_name}' not found")
        elif isinstance(node, ast.BinOp):
            left = self._ast_eval(node.left, state)
            right = self._ast_eval(node.right, state)
            op = ops.get(type(node.op))
            if op is None:
                raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._ast_eval(node.operand, state)
            op_func = unary_ops.get(type(node.op))
            if op_func is None:
                raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_func(operand)
        elif isinstance(node, ast.Compare):
            left = self._ast_eval(node.left, state)
            for comp_op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._ast_eval(comparator, state)
                comp_func = comp_ops.get(type(comp_op))
                if comp_func is None:
                    raise TypeError(f"Unsupported comparison operator: {type(comp_op).__name__}")
                if not comp_func(left, right):
                    return False
                left = right  # For chained comparisons
            return True
        elif isinstance(node, ast.BoolOp):
            values = [self._ast_eval(value, state) for value in node.values]
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
        return f"ConditionNode(id={self.node_id}, condition='{self.condition}')"
