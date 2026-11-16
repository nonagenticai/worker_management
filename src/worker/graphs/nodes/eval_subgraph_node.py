"""EvalSubgraphNode — validate the winning inference response.

Phase 1 scope
-------------
For the Phase 1 inference-graph, "evaluation" is *JSON-schema
validation* of the winning response stashed in
``framework_state['inference']['final_response']`` by the upstream
provider chain. A full recursive subgraph executor (that could run an
entire sub-DAG of evaluator nodes) is deferred to a later phase — this
node is deliberately the minimum that satisfies the plan's contract
("Writes ``validation_passed`` to ``framework_state['inference']``")
without pulling in a full graph-in-graph runner.

The node never raises out to the caller: every recognised validation
outcome (including an invalid *configured* schema) is recorded in state
and the chain continues (or ends) gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jsonschema
from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from src.worker.graphs.graph_models import NodeSchema

logger = logging.getLogger(__name__)


def _get_eval_config(schema: NodeSchema) -> dict[str, Any]:
    """Read and normalise the eval config from ``schema.framework_config``.

    Raises:
        ValueError: if ``validation_mode`` is unsupported or the
            required ``json_schema`` field is missing for that mode.
    """
    raw = schema.framework_config or {}
    validation_mode = raw.get("validation_mode", "json_schema")
    if validation_mode != "json_schema":
        raise ValueError(
            f"Eval-subgraph node '{schema.name}' only supports "
            f"validation_mode='json_schema' in Phase 1 (got '{validation_mode}')"
        )

    json_schema = raw.get("json_schema")
    if not isinstance(json_schema, dict) or not json_schema:
        raise ValueError(
            f"Eval-subgraph node '{schema.name}' missing 'json_schema' in framework_config"
        )

    return {
        "validation_mode": validation_mode,
        "json_schema": json_schema,
    }


def create_eval_subgraph_node_class(
    schema: NodeSchema,
    next_node_ids: list[str],
    _all_nodes: dict[str, NodeSchema],
    _edge_map: dict[str, list[dict[str, Any]]],
    _next_edges: list[dict[str, Any]] | None = None,
) -> type[BaseNode[WorkflowState, None, Any]]:
    """Build a per-schema ``EvalSubgraphNode`` class.

    The ``_all_nodes`` / ``_edge_map`` / ``_next_edges`` parameters are
    accepted to match the factory-dispatch signature in
    :class:`GraphFactory` but are not needed here — this node has a
    single logical successor and routes via ``_next_node_ids`` +
    ``_node_class_registry`` only.
    """
    eval_config = _get_eval_config(schema)

    @dataclass
    class EvalSubgraphNode(BaseNode[WorkflowState, None, Any]):
        """Per-schema eval-subgraph node.

        Reads ``framework_state['inference']['final_response']`` and
        validates it against the configured JSON schema, setting
        ``validation_passed`` in the inference bag.
        """

        _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
        _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
        _eval_config: dict[str, Any] = field(default_factory=lambda: dict(eval_config), init=False)

        def _next_successor_cls(
            self,
        ) -> type[BaseNode[WorkflowState, None, Any]] | None:
            """Return the first registered successor class, or None."""
            registry = getattr(type(self), "_node_class_registry", {})
            for next_id in self._next_node_ids:
                cls_ = registry.get(next_id)
                if cls_ is not None:
                    return cls_
            return None

        async def run(
            self, ctx: GraphRunContext[WorkflowState, None]
        ) -> BaseNode[WorkflowState, None, Any] | End[Any]:
            """Validate the winning response, record result, hand off."""
            bag = ctx.state.framework_state.get(WorkflowState.INFERENCE_KEY) or {}
            final_response = bag.get("final_response")

            if final_response is None:
                # Provider chain already decided this run failed — just
                # record validation_passed=False and pass through without
                # overriding the upstream outcome. We write directly to
                # ``node_outputs`` rather than going through
                # ``set_node_output`` so that we don't flip the node's
                # status to "completed" for what is semantically a
                # pass-through failure signal.
                logger.info(
                    "EvalSubgraphNode %s: no final_response present; marking validation_passed=False",
                    self._node_schema_id,
                )
                ctx.state.set_validation_result(False)
                ctx.state.node_outputs[self._node_schema_id] = {
                    "validation_passed": False,
                    "error": "no final_response",
                }
                successor = self._next_successor_cls()
                if successor is not None:
                    return successor()
                return End(final_response)

            json_schema = self._eval_config["json_schema"]

            try:
                jsonschema.validate(instance=final_response, schema=json_schema)
            except jsonschema.ValidationError as e:
                logger.info(
                    "EvalSubgraphNode %s: response failed JSON schema: %s",
                    self._node_schema_id,
                    e.message,
                )
                ctx.state.set_validation_result(False)
                # Record output without flipping status to completed;
                # then mark_node_failed for the authoritative status.
                ctx.state.node_outputs[self._node_schema_id] = {
                    "validation_passed": False,
                    "error": str(e.message),
                }
                ctx.state.mark_node_failed(self._node_schema_id, str(e.message))
            except jsonschema.SchemaError as e:
                logger.exception(
                    "EvalSubgraphNode %s: configured schema is invalid: %s",
                    self._node_schema_id,
                    e.message,
                )
                ctx.state.set_validation_result(False)
                ctx.state.node_outputs[self._node_schema_id] = {
                    "validation_passed": False,
                    "error": f"invalid schema: {e.message}",
                }
                ctx.state.mark_node_failed(self._node_schema_id, f"invalid schema: {e.message}")
            else:
                logger.info(
                    "EvalSubgraphNode %s: response passed JSON schema validation",
                    self._node_schema_id,
                )
                ctx.state.set_validation_result(True)
                ctx.state.set_node_output(
                    self._node_schema_id,
                    {"validation_passed": True},
                )

            successor = self._next_successor_cls()
            if successor is not None:
                return successor()
            return End(final_response)

    EvalSubgraphNode.__name__ = f"EvalSubgraphNode_{schema.id}".replace("-", "_")
    EvalSubgraphNode.__qualname__ = EvalSubgraphNode.__name__
    return EvalSubgraphNode
