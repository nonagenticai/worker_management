"""Tests for src.worker.graphs.graph_models module.

Covers NodeSchema, EdgeSchema, GraphCreateRequest, GraphUpdateRequest,
ExecutionRequest, and ExecutionResponse Pydantic models with validation.
"""

from datetime import UTC
from typing import Any, cast
from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError

from src.worker.graphs.graph_models import (
    EdgeSchema,
    ExecutionRequest,
    ExecutionResponse,
    GraphCreateRequest,
    GraphUpdateRequest,
    NodeSchema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker_node(**overrides):
    """Create a minimal valid worker NodeSchema dict."""
    defaults = {
        "name": "my_worker",
        "node_type": "worker",
        "worker_name": "some_worker",
    }
    defaults.update(overrides)
    return defaults


def _make_function_node(**overrides):
    """Create a minimal valid function NodeSchema dict."""
    defaults = {
        "name": "my_func",
        "node_type": "function",
        "function_name": "do_something",
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# NodeSchema Tests
# ===========================================================================


class TestNodeSchemaCreation:
    """Test basic NodeSchema creation and defaults."""

    def test_valid_worker_node(self):
        node = NodeSchema(**_make_worker_node())
        assert node.name == "my_worker"
        assert node.node_type == "worker"
        assert node.worker_name == "some_worker"
        assert node.timeout == 300
        assert node.retry_count == 0
        assert node.dependencies == []
        assert node.data == {}

    def test_auto_uuid_generated(self):
        node = NodeSchema(**_make_worker_node())
        assert isinstance(node.id, UUID)

    def test_two_nodes_get_different_uuids(self):
        n1 = NodeSchema(**_make_worker_node())
        n2 = NodeSchema(**_make_worker_node())
        assert n1.id != n2.id

    def test_explicit_uuid_preserved(self):
        explicit_id = uuid7()
        node = NodeSchema(**_make_worker_node(id=str(explicit_id)))
        assert node.id == explicit_id

    def test_defaults_for_optional_fields(self):
        node = NodeSchema(**_make_worker_node())
        assert node.function_name is None
        assert node.task_template is None
        assert node.framework_config == {}


class TestNodeSchemaNameValidation:
    """Test NodeSchema.validate_name (empty / whitespace)."""

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError, match=r"(?i)empty|cannot"):
            NodeSchema(**_make_worker_node(name=""))

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError, match=r"(?i)empty|cannot"):
            NodeSchema(**_make_worker_node(name="   "))

    def test_valid_name_accepted(self):
        node = NodeSchema(**_make_worker_node(name="process_data"))
        assert node.name == "process_data"


class TestNodeSchemaWorkerFunctionNameValidation:
    """Test validate_names for worker_name / function_name fields."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "simple",
            "with_underscore",
            "CamelCase",
            "has123numbers",
            "_leading_underscore",
            "with-hyphen",
            "a",
            "A1_b2-c3",
        ],
    )
    def test_valid_worker_names_accepted(self, valid_name):
        node = NodeSchema(**_make_worker_node(worker_name=valid_name))
        assert node.worker_name == valid_name

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "has space",
            "special!char",
            "dot.name",
            "slash/name",
            "colon:name",
            "@atSign",
            "-leading-hyphen",
        ],
    )
    def test_invalid_worker_names_rejected(self, invalid_name):
        with pytest.raises(ValidationError, match=r"(?i)name must start|pattern"):
            NodeSchema(**_make_worker_node(worker_name=invalid_name))

    def test_null_byte_rejected(self):
        with pytest.raises(ValidationError, match="null bytes"):
            NodeSchema(**_make_worker_node(worker_name="bad\x00name"))

    def test_exceeds_200_chars_rejected(self):
        long_name = "a" * 201
        with pytest.raises(ValidationError, match="200 characters"):
            NodeSchema(**_make_worker_node(worker_name=long_name))

    def test_exactly_200_chars_accepted(self):
        name_200 = "a" * 200
        node = NodeSchema(**_make_worker_node(worker_name=name_200))
        assert node.worker_name == name_200

    def test_empty_string_converted_to_none(self):
        """Empty string on optional name fields becomes None,
        which then fails model_validator for worker type
        because worker_name is required for worker nodes."""
        with pytest.raises(ValidationError, match="worker_name"):
            NodeSchema(**_make_worker_node(worker_name=""))

    def test_none_worker_name_accepted_for_function_node(self):
        node = NodeSchema(**_make_function_node(worker_name=None))
        assert node.worker_name is None

    def test_function_name_validation_same_rules(self):
        """function_name follows the same validate_names rules."""
        node = NodeSchema(**_make_function_node(function_name="valid_func"))
        assert node.function_name == "valid_func"

        with pytest.raises(ValidationError, match=r"(?i)name must start|pattern"):
            NodeSchema(**_make_function_node(function_name="bad name!"))


# ===========================================================================
# NodeSchema - node type validation
# ===========================================================================


class TestNodeSchemaNodeType:
    """Test node_type validation and model_validator consistency checks."""

    def test_invalid_node_type_rejected(self):
        with pytest.raises(ValidationError, match="Invalid node_type"):
            NodeSchema(**_make_worker_node(node_type="invalid_type"))

    def test_worker_without_worker_name_rejected(self):
        with pytest.raises(ValidationError, match="worker_name"):
            NodeSchema(name="bad_worker", node_type="worker")

    def test_function_node_with_function_name(self):
        node = NodeSchema(**_make_function_node())
        assert node.node_type == "function"
        assert node.function_name == "do_something"

    def test_condition_node_needs_condition_logic(self):
        with pytest.raises(ValidationError, match="condition logic"):
            NodeSchema(name="my_cond", node_type="condition")


# ===========================================================================
# NodeSchema - extra="forbid"
# ===========================================================================


class TestNodeSchemaExtraForbid:
    """Config extra=forbid rejects unknown fields."""

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match=r"extra_field|Extra inputs"):
            NodeSchema(**_make_worker_node(extra_field="nope"))

    def test_multiple_unknown_fields_rejected(self):
        with pytest.raises(ValidationError):
            NodeSchema(**_make_worker_node(foo="bar", baz=123))


# ===========================================================================
# EdgeSchema Tests
# ===========================================================================


class TestEdgeSchema:
    """Test EdgeSchema creation and validation."""

    def test_valid_edge_creation(self):
        src_id = uuid7()
        tgt_id = uuid7()
        edge = EdgeSchema(source=src_id, target=tgt_id)
        assert edge.source == src_id
        assert edge.target == tgt_id
        assert edge.condition is None
        assert edge.metadata is None

    def test_edge_with_condition(self):
        edge = EdgeSchema(
            source=uuid7(),
            target=uuid7(),
            condition="result.status == 'success'",
        )
        assert edge.condition is not None
        assert "success" in edge.condition

    def test_edge_with_metadata(self):
        edge = EdgeSchema(
            source=uuid7(),
            target=uuid7(),
            metadata={"priority": 1, "label": "on_success"},
        )
        assert edge.metadata is not None
        assert edge.metadata["priority"] == 1

    def test_source_required(self):
        with pytest.raises(ValidationError, match="source"):
            EdgeSchema(target=uuid7())

    def test_target_required(self):
        with pytest.raises(ValidationError, match="target"):
            EdgeSchema(source=uuid7())

    def test_extra_forbid_on_edge(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            EdgeSchema(
                source=uuid7(),
                target=uuid7(),
                **cast("dict[str, Any]", {"bogus": "value"}),
            )


# ===========================================================================
# GraphCreateRequest Tests
# ===========================================================================


class TestGraphCreateRequest:
    """Test GraphCreateRequest creation and validation."""

    def _minimal_graph_data(self):
        """Return minimal valid GraphCreateRequest dict."""
        return {
            "name": "test_graph",
            "nodes": [_make_worker_node()],
        }

    def test_valid_creation_with_nodes(self):
        graph = GraphCreateRequest(**self._minimal_graph_data())
        assert graph.name == "test_graph"
        assert len(graph.nodes) == 1
        assert graph.edges == []

    def test_valid_creation_with_nodes_and_edges(self):
        node1 = _make_worker_node(name="node_a", worker_name="w1")
        node2 = _make_worker_node(name="node_b", worker_name="w2")

        # First create nodes to get their IDs
        n1 = NodeSchema(**node1)
        n2 = NodeSchema(**node2)

        # Pydantic accepts dicts and coerces to NodeSchema/EdgeSchema; cast
        # so ty doesn't reject the heterogeneous list literal.
        graph = GraphCreateRequest(
            name="pipeline",
            description="A test pipeline",
            nodes=cast(
                "Any",
                [
                    {**node1, "id": str(n1.id)},
                    {**node2, "id": str(n2.id)},
                ],
            ),
            edges=cast("Any", [{"source": str(n1.id), "target": str(n2.id)}]),
        )
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_name_is_required(self):
        with pytest.raises(ValidationError, match="name"):
            GraphCreateRequest(nodes=[_make_worker_node()])

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            GraphCreateRequest(name="", nodes=[_make_worker_node()])

    def test_no_nodes_rejected(self):
        """Graph must contain at least one node."""
        with pytest.raises(ValidationError, match="at least one node"):
            GraphCreateRequest(name="empty_graph", nodes=[])

    def test_description_optional_defaults_empty(self):
        graph = GraphCreateRequest(**self._minimal_graph_data())
        assert graph.description == ""


# ===========================================================================
# GraphUpdateRequest Tests
# ===========================================================================


class TestGraphUpdateRequest:
    """Test GraphUpdateRequest - all fields optional."""

    def test_all_fields_optional(self):
        update = GraphUpdateRequest()
        assert update.name is None
        assert update.description is None
        assert update.nodes is None
        assert update.edges is None

    def test_partial_update_name_only(self):
        update = GraphUpdateRequest(name="new_name")
        assert update.name == "new_name"
        assert update.nodes is None

    def test_partial_update_nodes_only(self):
        update = GraphUpdateRequest(nodes=[NodeSchema(**_make_worker_node())])
        assert update.nodes is not None
        assert len(update.nodes) == 1


# ===========================================================================
# ExecutionRequest Tests
# ===========================================================================


class TestExecutionRequest:
    """Test ExecutionRequest creation and validation."""

    def test_minimal_creation(self):
        req = ExecutionRequest()
        assert req.variables == {}
        assert req.start_node is None

    def test_with_variables_and_start_node(self):
        req = ExecutionRequest(
            variables={"input_file": "data.csv"},
            start_node="some_node_id",
        )
        assert req.variables is not None
        assert req.variables["input_file"] == "data.csv"
        assert req.start_node == "some_node_id"

    def test_extra_forbid(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            ExecutionRequest(**cast("dict[str, Any]", {"unknown_field": "nope"}))


# ===========================================================================
# ExecutionResponse Tests
# ===========================================================================


class TestExecutionResponse:
    """Test ExecutionResponse creation and serialization round-trip."""

    def _sample_response_data(self):
        from datetime import datetime

        now = datetime.now(UTC)
        return {
            "session_id": "sess-001",
            "status": "completed",
            "current_node": "node_a",
            "node_outputs": {"node_a": {"result": 42}},
            "completed_nodes": ["node_a"],
            "failed_nodes": [],
            "final_output": {"answer": 42},
            "error_message": None,
            "started_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

    def test_valid_creation(self):
        resp = ExecutionResponse(**self._sample_response_data())
        assert resp.session_id == "sess-001"
        assert resp.status == "completed"
        assert resp.node_outputs["node_a"]["result"] == 42

    def test_round_trip_serialization(self):
        """model_dump -> model_validate should produce equivalent object."""
        original = ExecutionResponse(**self._sample_response_data())
        dumped = original.model_dump()
        restored = ExecutionResponse.model_validate(dumped)

        assert restored.session_id == original.session_id
        assert restored.status == original.status
        assert restored.node_outputs == original.node_outputs
        assert restored.completed_nodes == original.completed_nodes
        assert restored.failed_nodes == original.failed_nodes
        assert restored.final_output == original.final_output
        assert restored.started_at == original.started_at
        assert restored.updated_at == original.updated_at


# ===========================================================================
# ExecutionRequest round-trip
# ===========================================================================


class TestExecutionRequestRoundTrip:
    """Round-trip serialization for ExecutionRequest."""

    def test_round_trip(self):
        original = ExecutionRequest(
            variables={"key": "value", "nested": {"a": 1}},
            start_node="node_abc",
        )
        dumped = original.model_dump()
        restored = ExecutionRequest.model_validate(dumped)

        assert restored.variables == original.variables
        assert restored.start_node == original.start_node


# ===========================================================================
# NodeSchema - timeout / retry_count validators
# ===========================================================================


class TestNodeSchemaTimeoutRetry:
    """Edge-case coverage for timeout and retry_count validators."""

    def test_timeout_below_minimum_rejected(self):
        with pytest.raises(ValidationError, match="Timeout must be between"):
            NodeSchema(**_make_worker_node(timeout=0))

    def test_timeout_above_maximum_rejected(self):
        with pytest.raises(ValidationError, match="Timeout must be between"):
            NodeSchema(**_make_worker_node(timeout=3601))

    def test_retry_count_negative_rejected(self):
        with pytest.raises(ValidationError, match="Retry count must be between"):
            NodeSchema(**_make_worker_node(retry_count=-1))

    def test_retry_count_above_max_rejected(self):
        with pytest.raises(ValidationError, match="Retry count must be between"):
            NodeSchema(**_make_worker_node(retry_count=11))

    def test_valid_boundary_values(self):
        node = NodeSchema(**_make_worker_node(timeout=1, retry_count=0))
        assert node.timeout == 1
        assert node.retry_count == 0

        node2 = NodeSchema(**_make_worker_node(timeout=3600, retry_count=10))
        assert node2.timeout == 3600
        assert node2.retry_count == 10
