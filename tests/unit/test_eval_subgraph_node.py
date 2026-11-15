"""Unit tests for EvalSubgraphNode (Task 4, Phase 1 inference-graph).

These tests cover:

1. Valid response — passes JSON schema, ``validation_passed=True``.
2. Invalid response — fails, ``validation_passed=False``, node marked failed.
3. Invalid schema — ``SchemaError`` caught, ``validation_passed=False``.
4. Missing final_response — pass-through failure without overriding upstream.
5. Model validation — NodeSchema rejects missing ``json_schema`` config.

The node has no external HTTP dependency — tests use real state and a
real factory.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import EdgeSchema, NodeSchema
from src.worker.graphs.pydantic_graph_state import WorkflowState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SIMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["status", "count"],
}


def _make_eval_node(
    json_schema: dict[str, Any] | None = None,
    name: str = "eval_phase1",
) -> NodeSchema:
    return NodeSchema(
        name=name,
        node_type="eval_subgraph",
        framework_config={
            "validation_mode": "json_schema",
            "json_schema": json_schema if json_schema is not None else _SIMPLE_SCHEMA,
        },
    )


def _make_trailing_function_node() -> NodeSchema:
    return NodeSchema(
        name="after_eval",
        node_type="function",
        function_name="noop",
        data={},
    )


def _make_state_with_response(final_response: Any) -> WorkflowState:
    state = WorkflowState(
        session_id="sess-1",
        graph_id="graph-1",
        variables={},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )
    if final_response is not None:
        state.mark_inference_success(
            provider="test-provider",
            response=final_response,
            tokens_prompt=0,
            tokens_completion=0,
        )
    return state


class _FakeCtx:
    def __init__(self, state: WorkflowState) -> None:
        self.state = state


def _make_ctx(state: WorkflowState) -> GraphRunContext[WorkflowState]:
    return cast("GraphRunContext[WorkflowState]", cast("Any", _FakeCtx(state)))


def _build_registry(
    nodes: list[NodeSchema],
    edges: list[EdgeSchema] | None = None,
) -> dict[str, type[BaseNode[WorkflowState, None, Any]]]:
    factory = GraphFactory(worker_manager=None)
    node_map = {str(n.id): n for n in nodes}
    edge_map = factory._build_edge_map(edges or [])
    classes = factory._create_node_classes(nodes, node_map, edge_map)
    return {cast("Any", cls())._node_schema_id: cls for cls in classes}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalSubgraphNodeValid:
    @pytest.mark.asyncio
    async def test_valid_response_sets_passed_true_and_ends(self) -> None:
        eval_node = _make_eval_node()
        registry = _build_registry([eval_node])
        cls_eval = registry[str(eval_node.id)]

        state = _make_state_with_response({"status": "ok", "count": 3})
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        bag = state.framework_state["inference"]
        assert bag["validation_passed"] is True
        assert state.node_outputs[str(eval_node.id)] == {"validation_passed": True}
        # Node must NOT be in failed list.
        assert str(eval_node.id) not in state.failed_nodes

    @pytest.mark.asyncio
    async def test_valid_response_hands_off_to_successor(self) -> None:
        eval_node = _make_eval_node()
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [eval_node, next_node],
            [EdgeSchema(source=eval_node.id, target=next_node.id)],
        )
        cls_eval = registry[str(eval_node.id)]
        cls_next = registry[str(next_node.id)]

        state = _make_state_with_response({"status": "ok", "count": 1})
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, cls_next)
        assert state.framework_state["inference"]["validation_passed"] is True


class TestEvalSubgraphNodeInvalidResponse:
    @pytest.mark.asyncio
    async def test_invalid_response_records_failure(self) -> None:
        eval_node = _make_eval_node()
        registry = _build_registry([eval_node])
        cls_eval = registry[str(eval_node.id)]

        # 'count' must be an integer; provide a string to trigger failure.
        state = _make_state_with_response({"status": "ok", "count": "three"})
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        bag = state.framework_state["inference"]
        assert bag["validation_passed"] is False

        output = state.node_outputs[str(eval_node.id)]
        assert output["validation_passed"] is False
        assert "error" in output
        assert len(output["error"]) > 0

        assert str(eval_node.id) in state.failed_nodes
        assert state.node_status[str(eval_node.id)] == "failed"

    @pytest.mark.asyncio
    async def test_missing_required_field_records_failure(self) -> None:
        eval_node = _make_eval_node()
        registry = _build_registry([eval_node])
        cls_eval = registry[str(eval_node.id)]

        # Missing 'count' — schema requires it.
        state = _make_state_with_response({"status": "ok"})
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        assert state.framework_state["inference"]["validation_passed"] is False
        assert str(eval_node.id) in state.failed_nodes

    @pytest.mark.asyncio
    async def test_invalid_response_still_returns_successor_when_present(self) -> None:
        """Validation failure must still hand off to the next node.

        The eval node only short-circuits with ``End`` when there is no
        successor; when a successor is wired up the chain must continue
        so downstream nodes can react to ``validation_passed=False``.
        """
        eval_node = _make_eval_node()
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [eval_node, next_node],
            [EdgeSchema(source=eval_node.id, target=next_node.id)],
        )
        cls_eval = registry[str(eval_node.id)]
        cls_next = registry[str(next_node.id)]

        # 'count' is required to be an integer — give a string to fail.
        state = _make_state_with_response({"status": "ok", "count": "three"})
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert not isinstance(result, End)
        assert isinstance(result, cls_next)

        bag = state.framework_state["inference"]
        assert bag["validation_passed"] is False
        assert str(eval_node.id) in state.failed_nodes
        assert state.node_status[str(eval_node.id)] == "failed"


class TestEvalSubgraphNodeInvalidSchema:
    @pytest.mark.asyncio
    async def test_invalid_schema_records_failure_without_raising(self) -> None:
        # "type": "not-a-real-type" is an invalid JSON Schema construct.
        bad_schema = {"type": "not-a-real-type"}
        eval_node = _make_eval_node(json_schema=bad_schema, name="eval_bad_schema")
        registry = _build_registry([eval_node])
        cls_eval = registry[str(eval_node.id)]

        state = _make_state_with_response({"anything": 1})
        instance = cls_eval()
        # Must not raise.
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        bag = state.framework_state["inference"]
        assert bag["validation_passed"] is False
        output = state.node_outputs[str(eval_node.id)]
        assert output["validation_passed"] is False
        assert "invalid schema" in output["error"]
        assert str(eval_node.id) in state.failed_nodes


class TestEvalSubgraphNodeNoFinalResponse:
    @pytest.mark.asyncio
    async def test_no_final_response_passes_through_as_failed(self) -> None:
        eval_node = _make_eval_node()
        registry = _build_registry([eval_node])
        cls_eval = registry[str(eval_node.id)]

        # No final_response stashed — mimics all-providers-failed upstream.
        state = WorkflowState(
            session_id="sess-1",
            graph_id="graph-1",
            variables={},
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
        )
        instance = cls_eval()
        result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        bag = state.framework_state["inference"]
        assert bag["validation_passed"] is False

        # The node records its own output, but it should NOT mark itself
        # as failed in node_status — the upstream chain already decided
        # the outcome. (In practice, upstream sets error_message itself
        # via the provider-fail path.)
        output = state.node_outputs[str(eval_node.id)]
        assert output["validation_passed"] is False
        assert output["error"] == "no final_response"


class TestEvalSubgraphNodeConfigValidation:
    def test_missing_json_schema_raises_at_model_validation(self) -> None:
        with pytest.raises(ValueError, match="json_schema"):
            NodeSchema(
                name="eval_bad",
                node_type="eval_subgraph",
                framework_config={"validation_mode": "json_schema"},
            )

    def test_unsupported_validation_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="validation_mode"):
            NodeSchema(
                name="eval_bad_mode",
                node_type="eval_subgraph",
                framework_config={
                    "validation_mode": "regex",
                    "json_schema": {"type": "object"},
                },
            )
