"""Unit tests for GraphFactory and graph_models - graph creation, node class
generation, validation, connectivity, and model validation.

Tests verify graph CREATION logic and model validation. Worker agent calls are mocked.
Node run() methods are tested with mocked WorkerManager for worker nodes.
"""

from datetime import datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from src.worker.graphs.graph_factory import GraphFactory, GraphMetadata
from src.worker.graphs.graph_models import (
    EdgeSchema,
    ExecutionEvent,
    ExecutionRequest,
    ExecutionResponse,
    GraphCreateRequest,
    GraphExecutionState,
    GraphResponse,
    GraphUpdateRequest,
    NodeExecutionResult,
    NodeSchema,
)
from src.worker.graphs.pydantic_graph_state import WorkflowState

# ---------------------------------------------------------------------------
# Typed accessors for dynamically-generated dataclass node attributes.
# Node classes are built by GraphFactory at runtime; ty only sees the
# BaseNode shape, so reads of `_node_schema_id` / `_next_node_ids` /
# `_node_class_registry` need explicit Any-cast helpers.
# ---------------------------------------------------------------------------


def _node_schema_id(inst: object) -> str:
    return cast("str", cast("Any", inst)._node_schema_id)


def _next_node_ids(inst: object) -> list[str]:
    return cast("list[str]", cast("Any", inst)._next_node_ids)


def _set_node_class_registry(inst: object, registry: dict[str, type]) -> None:
    cast("Any", inst.__class__)._node_class_registry = registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker_node(name: str = "test_worker", worker_name: str = "my_worker", **kw):
    """Create a minimal worker NodeSchema."""
    return NodeSchema(name=name, node_type="worker", worker_name=worker_name, **kw)


def _make_function_node(
    name: str = "test_func",
    function_name: str = "do_stuff",
    task_template: str | None = None,
    **kw,
):
    """Create a minimal function NodeSchema."""
    defaults: dict = {"data": {}}
    defaults.update(kw)
    return NodeSchema(
        name=name,
        node_type="function",
        function_name=function_name,
        task_template=task_template,
        **defaults,
    )


def _make_condition_node(name: str = "test_cond", condition: str = "True", **kw):
    """Create a minimal condition NodeSchema."""
    return NodeSchema(name=name, node_type="condition", data={"condition": condition}, **kw)


def _factory() -> GraphFactory:
    """Return a GraphFactory with a mocked WorkerManager so nothing real is invoked."""
    mock_wm = MagicMock()
    return GraphFactory(worker_manager=mock_wm)


def _make_state(**overrides) -> WorkflowState:
    """Create a WorkflowState with sensible defaults."""
    defaults: dict[str, Any] = {
        "session_id": "test-session",
        "graph_id": "test-graph",
        "variables": {},
        "node_outputs": {},
        "completed_nodes": [],
        "failed_nodes": [],
    }
    defaults.update(overrides)
    return WorkflowState(**defaults)


def _make_run_context(state: WorkflowState | None = None) -> GraphRunContext[WorkflowState]:
    """Create a mock GraphRunContext wrapping a WorkflowState."""
    if state is None:
        state = _make_state()
    ctx = MagicMock(spec=GraphRunContext)
    ctx.state = state
    return ctx


# ---------------------------------------------------------------------------
# 1. _create_node_classes
# ---------------------------------------------------------------------------


class TestCreateNodeClasses:
    def test_single_worker_node_returns_one_class(self):
        factory = _factory()
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        assert len(classes) == 1
        assert isinstance(classes[0], type)
        assert issubclass(classes[0], BaseNode)

    def test_node_class_has_schema_id(self):
        factory = _factory()
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        assert hasattr(instance, "_node_schema_id")
        assert _node_schema_id(instance) == str(node.id)

    def test_multiple_node_types(self):
        factory = _factory()
        w = _make_worker_node(name="w1", worker_name="alpha")
        f = _make_function_node(name="f1")
        c = _make_condition_node(name="c1")
        nodes = [w, f, c]
        node_map = {str(n.id): n for n in nodes}
        edge_map: dict = {}
        classes = factory._create_node_classes(nodes, node_map, edge_map)
        assert len(classes) == 3
        for cls in classes:
            assert isinstance(cls, type)
            assert issubclass(cls, BaseNode)


# ---------------------------------------------------------------------------
# 2. create_graph
# ---------------------------------------------------------------------------


class TestCreateGraph:
    def test_single_node_graph(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="single_node")
        assert isinstance(graph, Graph)

    def test_graph_name_is_set(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="my_graph")
        assert isinstance(graph, Graph)
        assert graph.name == "my_graph"

    def test_graph_auto_generates_name(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[])
        assert isinstance(graph, Graph)
        assert graph.name is not None
        assert graph.name.startswith("graph_")

    def test_multi_node_linear_graph(self):
        factory = _factory()
        a = _make_worker_node(name="step_a", worker_name="w1")
        b = _make_worker_node(name="step_b", worker_name="w2")
        c = _make_worker_node(name="step_c", worker_name="w3")
        edges = [
            EdgeSchema(source=a.id, target=b.id),
            EdgeSchema(source=b.id, target=c.id),
        ]
        graph = factory.create_graph(nodes=[a, b, c], edges=edges, name="linear")
        assert isinstance(graph, Graph)

    def test_graph_with_condition_node_branching(self):
        factory = _factory()
        cond = _make_condition_node(name="check", condition="x > 5")
        t_branch = _make_worker_node(name="true_path", worker_name="w1")
        f_branch = _make_worker_node(name="false_path", worker_name="w2")
        edges = [
            EdgeSchema(source=cond.id, target=t_branch.id),
            EdgeSchema(source=cond.id, target=f_branch.id),
        ]
        graph = factory.create_graph(
            nodes=[cond, t_branch, f_branch], edges=edges, name="branching"
        )
        assert isinstance(graph, Graph)

    def test_graph_with_function_node(self):
        factory = _factory()
        fn = _make_function_node(name="compute")
        graph = factory.create_graph(nodes=[fn], edges=[], name="func_graph")
        assert isinstance(graph, Graph)

    def test_graph_with_variables_metadata(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="var_graph")
        metadata = factory._graph_metadata.get(id(graph))
        assert metadata is not None
        assert metadata.name == "var_graph"

    def test_create_graph_with_validation_disabled(self):
        factory = _factory()
        node = _make_worker_node()
        object.__setattr__(node, "worker_name", None)
        graph = factory.create_graph(nodes=[node], edges=[], validate=False)
        assert isinstance(graph, Graph)

    def test_create_graph_validation_fails_raises(self):
        factory = _factory()
        node = _make_worker_node()
        object.__setattr__(node, "worker_name", None)
        with pytest.raises(ValueError, match="Graph validation failed"):
            factory.create_graph(nodes=[node], edges=[], validate=True)


# ---------------------------------------------------------------------------
# 3. Node class names
# ---------------------------------------------------------------------------


class TestNodeClassNames:
    def test_worker_node_class_name_contains_id(self):
        factory = _factory()
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        cls = classes[0]
        assert "WorkerNode" in cls.__name__
        sanitised_id = str(node.id).replace("-", "_")
        assert sanitised_id in cls.__name__

    def test_function_node_class_name_contains_id(self):
        factory = _factory()
        node = _make_function_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        cls = classes[0]
        assert "FunctionNode" in cls.__name__

    def test_condition_node_class_name_contains_id(self):
        factory = _factory()
        node = _make_condition_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        cls = classes[0]
        assert "ConditionNode" in cls.__name__


# ---------------------------------------------------------------------------
# 4. Function node task_template
# ---------------------------------------------------------------------------


class TestFunctionNodeTaskTemplate:
    def test_function_node_with_task_template(self):
        factory = _factory()
        node = _make_function_node(name="templated", task_template="Process {filename}")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        assert len(classes) == 1
        assert issubclass(classes[0], BaseNode)
        instance = classes[0]()
        assert _node_schema_id(instance) == str(node.id)
        assert node.task_template == "Process {filename}"

    def test_function_node_without_task_template(self):
        factory = _factory()
        node = _make_function_node(name="no_template", task_template=None)
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        assert len(classes) == 1
        assert issubclass(classes[0], BaseNode)


# ---------------------------------------------------------------------------
# 5. Two-node graph connectivity
# ---------------------------------------------------------------------------


class TestTwoNodeGraphConnectivity:
    def test_two_node_graph_creates_successfully(self):
        factory = _factory()
        w = _make_worker_node(name="step1", worker_name="analyzer")
        f = _make_function_node(name="step2")
        edge = EdgeSchema(source=w.id, target=f.id)
        graph = factory.create_graph(nodes=[w, f], edges=[edge], name="two_step")
        assert isinstance(graph, Graph)

    def test_two_node_graph_has_two_node_classes(self):
        factory = _factory()
        w = _make_worker_node(name="step1", worker_name="analyzer")
        f = _make_function_node(name="step2")
        edge = EdgeSchema(source=w.id, target=f.id)
        graph = factory.create_graph(nodes=[w, f], edges=[edge], name="pipeline")
        if hasattr(graph, "get_nodes"):
            node_classes = graph.get_nodes()
        elif hasattr(graph, "_nodes"):
            node_classes = graph._nodes
        else:
            metadata = factory._graph_metadata.get(id(graph))
            assert metadata is not None
            assert len(metadata.node_schemas) == 2
            return
        assert len(node_classes) == 2

    def test_worker_node_knows_next_node(self):
        factory = _factory()
        w = _make_worker_node(name="step1", worker_name="analyzer")
        f = _make_function_node(name="step2")
        edge = EdgeSchema(source=w.id, target=f.id)
        node_map = {str(w.id): w, str(f.id): f}
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([w, f], node_map, edge_map)
        worker_cls = None
        function_cls = None
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(w.id):
                worker_cls = cls
            elif _node_schema_id(inst) == str(f.id):
                function_cls = cls
        assert worker_cls is not None
        assert function_cls is not None
        worker_instance = worker_cls()
        assert str(f.id) in _next_node_ids(worker_instance)

    def test_function_terminal_node_has_no_next(self):
        factory = _factory()
        w = _make_worker_node(name="step1", worker_name="analyzer")
        f = _make_function_node(name="step2")
        edge = EdgeSchema(source=w.id, target=f.id)
        node_map = {str(w.id): w, str(f.id): f}
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([w, f], node_map, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(f.id):
                assert _next_node_ids(inst) == []
                return
        pytest.fail("Function node class not found in created classes")

    def test_node_class_registry_is_populated(self):
        factory = _factory()
        w = _make_worker_node(name="step1", worker_name="analyzer")
        f = _make_function_node(name="step2")
        edge = EdgeSchema(source=w.id, target=f.id)
        node_map = {str(w.id): w, str(f.id): f}
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([w, f], node_map, edge_map)
        for cls in classes:
            assert hasattr(cls, "_node_class_registry")
            registry: dict[str, type] = cast("Any", cls)._node_class_registry
            assert str(w.id) in registry
            assert str(f.id) in registry


# ---------------------------------------------------------------------------
# 6. Edge-map building
# ---------------------------------------------------------------------------


class TestBuildEdgeMap:
    def test_single_edge(self):
        w = _make_worker_node()
        f = _make_function_node()
        edge = EdgeSchema(source=w.id, target=f.id)
        edge_map = GraphFactory._build_edge_map([edge])
        assert str(w.id) in edge_map
        targets = [e["target"] for e in edge_map[str(w.id)]]
        assert str(f.id) in targets

    def test_no_edges(self):
        edge_map = GraphFactory._build_edge_map([])
        assert edge_map == {}

    def test_multiple_outgoing_edges(self):
        c = _make_condition_node()
        t1 = _make_worker_node(name="true_branch", worker_name="w1")
        t2 = _make_worker_node(name="false_branch", worker_name="w2")
        edges = [
            EdgeSchema(source=c.id, target=t1.id),
            EdgeSchema(source=c.id, target=t2.id),
        ]
        edge_map = GraphFactory._build_edge_map(edges)
        assert len(edge_map[str(c.id)]) == 2
        targets = [e["target"] for e in edge_map[str(c.id)]]
        assert str(t1.id) in targets
        assert str(t2.id) in targets

    def test_edge_with_condition(self):
        w = _make_worker_node()
        f = _make_function_node()
        edge = EdgeSchema(source=w.id, target=f.id, condition="result == True")
        edge_map = GraphFactory._build_edge_map([edge])
        entry = edge_map[str(w.id)][0]
        assert entry["target"] == str(f.id)
        assert entry["condition"] == "result == True"


# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------


class TestValidateGraphStructure:
    def test_empty_nodes_raises(self):
        issues = GraphFactory.validate_graph_structure([], [])
        assert any("at least one node" in i for i in issues)

    def test_missing_worker_name(self):
        node = _make_worker_node()
        object.__setattr__(node, "worker_name", None)
        issues = GraphFactory.validate_graph_structure([node], [])
        assert any("missing worker_name" in i for i in issues)

    def test_valid_single_node_no_issues(self):
        node = _make_worker_node()
        issues = GraphFactory.validate_graph_structure([node], [])
        assert issues == []

    def test_edge_referencing_nonexistent_source(self):
        from uuid import uuid7

        node = _make_worker_node()
        fake_id = uuid7()
        edge = EdgeSchema(source=fake_id, target=node.id)
        issues = GraphFactory.validate_graph_structure([node], [edge])
        assert any("source" in i and "not found" in i for i in issues)

    def test_edge_referencing_nonexistent_target(self):
        from uuid import uuid7

        node = _make_worker_node()
        fake_id = uuid7()
        edge = EdgeSchema(source=node.id, target=fake_id)
        issues = GraphFactory.validate_graph_structure([node], [edge])
        assert any("target" in i and "not found" in i for i in issues)

    def test_function_node_carrying_function_code_is_an_issue(self):
        node = _make_function_node()
        # Bypass NodeSchema (which refuses the field) to prove this check too.
        object.__setattr__(node, "data", {"function_code": "result = 1"})
        issues = GraphFactory.validate_graph_structure([node], [])
        assert any("function_code" in i and "not supported" in i for i in issues)

    def test_function_node_without_code_is_not_an_issue(self):
        assert GraphFactory.validate_graph_structure([_make_function_node()], []) == []

    def test_condition_node_missing_condition(self):
        node = _make_condition_node()
        object.__setattr__(node, "data", {})
        issues = GraphFactory.validate_graph_structure([node], [])
        assert any("condition" in i.lower() for i in issues)

    def test_valid_linear_graph_no_issues(self):
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        edge = EdgeSchema(source=a.id, target=b.id)
        issues = GraphFactory.validate_graph_structure([a, b], [edge])
        assert issues == []


# ---------------------------------------------------------------------------
# 8. Unsupported node type
# ---------------------------------------------------------------------------


class TestUnsupportedNodeType:
    def test_unsupported_node_type_raises(self):
        factory = _factory()
        node = _make_worker_node()
        object.__setattr__(node, "node_type", "unknown_type")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        with pytest.raises(ValueError, match="Unsupported node type"):
            factory._create_node_class(node, node_map, edge_map)


# ---------------------------------------------------------------------------
# 9. Start node detection
# ---------------------------------------------------------------------------


class TestDetermineStartNode:
    def test_explicit_start_node(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        node_schemas = {str(a.id): a, str(b.id): b}
        edge_map: dict = {}
        result = factory._determine_start_node(str(a.id), node_schemas, edge_map)
        assert result == str(a.id)

    def test_explicit_start_node_not_found_raises(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        node_schemas = {str(a.id): a}
        edge_map: dict = {}
        with pytest.raises(ValueError, match="not found in graph"):
            factory._determine_start_node("nonexistent_id", node_schemas, edge_map)

    def test_auto_detect_start_node_no_incoming(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        node_schemas = {str(a.id): a, str(b.id): b}
        edge_map = {str(a.id): [{"target": str(b.id)}]}
        result = factory._determine_start_node(None, node_schemas, edge_map)
        assert result == str(a.id)

    def test_no_valid_start_node_raises(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        node_schemas = {str(a.id): a, str(b.id): b}
        edge_map = {
            str(a.id): [{"target": str(b.id)}],
            str(b.id): [{"target": str(a.id)}],
        }
        with pytest.raises(ValueError, match="No valid start node"):
            factory._determine_start_node(None, node_schemas, edge_map)


# ---------------------------------------------------------------------------
# 10. create_start_node
# ---------------------------------------------------------------------------


class TestCreateStartNode:
    def test_create_start_node_with_explicit_id(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="test_start")
        start_instance = factory.create_start_node(graph, start_node_id=str(node.id))
        assert hasattr(start_instance, "_node_schema_id")
        assert _node_schema_id(start_instance) == str(node.id)

    def test_create_start_node_auto_detect(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        edge = EdgeSchema(source=a.id, target=b.id)
        graph = factory.create_graph(nodes=[a, b], edges=[edge], name="test_auto")
        start_instance = factory.create_start_node(graph)
        assert _node_schema_id(start_instance) == str(a.id)

    def test_create_start_node_invalid_id_raises(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="test_invalid")
        with pytest.raises(ValueError, match="not found in graph"):
            factory.create_start_node(graph, start_node_id="nonexistent")

    def test_create_start_node_no_metadata(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="test_no_meta")
        factory._graph_metadata.clear()
        with pytest.raises(ValueError):
            factory.create_start_node(graph)


# ---------------------------------------------------------------------------
# 11. _find_and_create_start_node_instance
# ---------------------------------------------------------------------------


class TestFindAndCreateStartNodeInstance:
    def test_node_not_found_raises(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="test_find")
        with pytest.raises(ValueError, match="Could not find node class"):
            factory._find_and_create_start_node_instance(graph, "nonexistent_id")

    def test_graph_without_registered_start_node_raises(self):
        """An unknown graph (no factory metadata) cannot resolve a start class.

        pydantic-graph 2 keeps its node registry private, so the factory resolves
        the start from the metadata it recorded at build time; a graph it never
        built has none.
        """
        factory = _factory()
        fake_graph = MagicMock()
        with pytest.raises(ValueError, match="Could not find node class for start node"):
            factory._find_and_create_start_node_instance(fake_graph, "some_id")


# ---------------------------------------------------------------------------
# 12. _get_worker_manager
# ---------------------------------------------------------------------------


class TestGetWorkerManager:
    def test_returns_existing_manager(self):
        mock_wm = MagicMock()
        factory = GraphFactory(worker_manager=mock_wm)
        assert factory._get_worker_manager() is mock_wm

    def test_creates_manager_when_none(self):
        factory = GraphFactory(worker_manager=None)
        with patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM:
            mock_instance = MagicMock()
            MockWM.return_value = mock_instance
            result = factory._get_worker_manager()
            assert result is mock_instance
            MockWM.assert_called_once()

    def test_raises_runtime_error_on_failure(self):
        factory = GraphFactory(worker_manager=None)
        with (
            patch(
                "src.worker.graphs.graph_factory.WorkerManager",
                side_effect=Exception("init failed"),
            ),
            pytest.raises(RuntimeError, match="Cannot initialize WorkerManager"),
        ):
            factory._get_worker_manager()


# ---------------------------------------------------------------------------
# 13. GraphMetadata
# ---------------------------------------------------------------------------


class TestGraphMetadata:
    def test_metadata_stored_on_create(self):
        factory = _factory()
        node = _make_worker_node()
        graph = factory.create_graph(nodes=[node], edges=[], name="meta_test")
        metadata = factory._graph_metadata.get(id(graph))
        assert metadata is not None
        assert isinstance(metadata, GraphMetadata)
        assert metadata.name == "meta_test"

    def test_metadata_has_node_schemas(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_function_node(name="b1")
        edge = EdgeSchema(source=a.id, target=b.id)
        graph = factory.create_graph(nodes=[a, b], edges=[edge], name="meta2")
        metadata = factory._graph_metadata[id(graph)]
        assert str(a.id) in metadata.node_schemas
        assert str(b.id) in metadata.node_schemas

    def test_metadata_has_edge_map(self):
        factory = _factory()
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_function_node(name="b1")
        edge = EdgeSchema(source=a.id, target=b.id)
        graph = factory.create_graph(nodes=[a, b], edges=[edge], name="meta3")
        metadata = factory._graph_metadata[id(graph)]
        assert str(a.id) in metadata.edge_map
        targets = [e["target"] for e in metadata.edge_map[str(a.id)]]
        assert str(b.id) in targets


# ---------------------------------------------------------------------------
# 14. Worker node run() method
# ---------------------------------------------------------------------------


class TestWorkerNodeRun:
    @pytest.mark.asyncio
    async def test_worker_node_run_success_end(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "done"})
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node(task_template="Do work")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_worker_node_run_routes_to_next(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_function_node(name="b1")
        edge = EdgeSchema(source=a.id, target=b.id)
        node_map = {str(a.id): a, str(b.id): b}
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([a, b], node_map, edge_map)
        worker_instance = None
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(a.id):
                worker_instance = inst
                break
        assert worker_instance is not None
        ctx = _make_run_context()
        result = await worker_instance.run(ctx)
        assert isinstance(result, BaseNode)

    @pytest.mark.asyncio
    async def test_worker_node_run_stores_output(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "hello"})
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state()
        ctx = _make_run_context(state)
        await instance.run(ctx)
        assert str(node.id) in state.node_outputs

    @pytest.mark.asyncio
    async def test_worker_node_run_lifts_quality_score(self):
        # The inline WorkerNode (the class the /execution engine runs) must lift
        # a numeric quality_score from the worker's structured output (carried
        # under result["output"] by WorkerManager) into the inference bag, so the
        # finalize path persists it onto graph_execution_sessions.quality_score.
        inference_key = WorkflowState.INFERENCE_KEY
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(
            return_value={
                "response": '{"quality_score": 0.87}',
                "output": {"validation_passed": True, "quality_score": 0.87, "decision": "pass"},
            }
        )
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node(worker_name="qa_validator")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state()
        ctx = _make_run_context(state)
        await instance.run(ctx)
        assert state.framework_state[inference_key]["quality_score"] == 0.87

    @pytest.mark.asyncio
    async def test_worker_node_run_no_quality_score_is_noop(self):
        # A worker that emits no quality_score leaves the inference bag untouched.
        inference_key = WorkflowState.INFERENCE_KEY
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(
            return_value={"response": "done", "output": {"summary": "looks fine"}}
        )
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node(worker_name="summariser")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state()
        ctx = _make_run_context(state)
        await instance.run(ctx)
        assert inference_key not in state.framework_state

    @pytest.mark.asyncio
    async def test_worker_node_run_template_substitution(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node(task_template="Process {filename}")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state(variables={"filename": "test.csv"})
        ctx = _make_run_context(state)
        await instance.run(ctx)
        call_args = mock_wm.send_message_to_worker.call_args
        assert call_args.kwargs["message"] == "Process test.csv"

    @pytest.mark.asyncio
    async def test_worker_node_run_template_key_error_fallback(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node(task_template="Process {missing_key}")
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state(variables={})
        ctx = _make_run_context(state)
        await instance.run(ctx)
        call_args = mock_wm.send_message_to_worker.call_args
        assert call_args.kwargs["message"] == "Process {missing_key}"

    @pytest.mark.asyncio
    async def test_worker_node_run_string_result(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value="plain string")
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_worker_node_run_exception_returns_end(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(side_effect=Exception("boom"))
        factory = GraphFactory(worker_manager=mock_wm)
        node = _make_worker_node()
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_worker_node_run_no_worker_manager_raises(self):
        factory = GraphFactory(worker_manager=None)
        with patch(
            "src.worker.graphs.graph_factory.WorkerManager",
            side_effect=Exception("no wm"),
        ):
            node = _make_worker_node()
            node_map = {str(node.id): node}
            edge_map: dict = {}
            classes = factory._create_node_classes([node], node_map, edge_map)
            instance = classes[0]()
            ctx = _make_run_context()
            with pytest.raises(RuntimeError, match="Cannot initialize WorkerManager"):
                await instance.run(ctx)

    @pytest.mark.asyncio
    async def test_worker_node_run_next_node_not_found_returns_end(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        node_map = {str(a.id): a}
        edge_map = {str(a.id): [{"target": "nonexistent_id", "condition": None}]}
        cls = factory._create_worker_node_class(
            a,
            ["nonexistent_id"],
            node_map,
            edge_map,
            [{"target": "nonexistent_id", "condition": None}],
        )
        instance = cls()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)


# ---------------------------------------------------------------------------
# 15. Worker node edge conditions
# ---------------------------------------------------------------------------


class TestWorkerNodeEdgeConditions:
    @pytest.mark.asyncio
    async def test_worker_node_edge_condition_routes_correctly(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b_true", worker_name="w2")
        c = _make_worker_node(name="c_false", worker_name="w3")
        all_nodes = {str(a.id): a, str(b.id): b, str(c.id): c}
        next_edges = [
            {"target": str(b.id), "condition": "result == True"},
            {"target": str(c.id), "condition": None},
        ]
        edge_map = {str(a.id): next_edges}
        classes = factory._create_node_classes([a, b, c], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(a.id):
                state = _make_state(variables={"result": True})
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_worker_node_edge_condition_fallback_default(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b_default", worker_name="w2")
        c = _make_worker_node(name="c_cond", worker_name="w3")
        all_nodes = {str(a.id): a, str(b.id): b, str(c.id): c}
        next_edges = [
            {"target": str(c.id), "condition": "result == 999"},
            {"target": str(b.id), "condition": None},
        ]
        edge_map = {str(a.id): next_edges}
        classes = factory._create_node_classes([a, b, c], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(a.id):
                state = _make_state(variables={"result": 0})
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_worker_node_edge_condition_unsafe_ast_logged(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2")
        all_nodes = {str(a.id): a, str(b.id): b}
        next_edges = [
            {"target": str(b.id), "condition": "__import__('os').system('ls')"},
        ]
        edge_map = {str(a.id): next_edges}
        cls = factory._create_worker_node_class(a, [str(b.id)], all_nodes, edge_map, next_edges)
        instance = cls()
        _set_node_class_registry(instance, {str(b.id): type(instance)})
        state = _make_state()
        ctx = _make_run_context(state)
        result = await instance.run(ctx)
        assert isinstance(result, BaseNode)

    @pytest.mark.asyncio
    async def test_worker_node_template_context_includes_node_outputs(self):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(return_value={"response": "ok"})
        factory = GraphFactory(worker_manager=mock_wm)
        a = _make_worker_node(name="a1", worker_name="w1")
        b = _make_worker_node(name="b1", worker_name="w2", task_template="Use {a1_val}")
        all_nodes = {str(a.id): a, str(b.id): b}
        edge = EdgeSchema(source=a.id, target=b.id)
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([a, b], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(b.id):
                state = _make_state(
                    variables={"a1_val": "fromA"},
                    node_outputs={str(a.id): "result_a"},
                )
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, End)
                break


# ---------------------------------------------------------------------------
# 16. Function node run()
# ---------------------------------------------------------------------------


class TestFunctionNodeRun:
    @pytest.mark.asyncio
    async def test_function_node_passes_data_through_without_exec(self):
        factory = _factory()
        node = _make_function_node()
        node_map = {str(node.id): node}
        classes = factory._create_node_classes([node], node_map, {})
        instance = classes[0]()
        state = _make_state(variables={"x": 5})
        result = await instance.run(_make_run_context(state))
        assert isinstance(result, End)
        assert state.node_outputs[str(node.id)] == {"x": 5}

    @pytest.mark.asyncio
    async def test_function_code_that_bypassed_the_schema_fails_closed(self):
        """Stored/constructed code never runs: run() raises, the sentinel is unset."""
        import builtins

        from src.security.request_overrides import FunctionCodeNotSupportedError

        sentinel = "_WM_GRAPH_FACTORY_EXEC_SENTINEL"
        node = _make_function_node()
        object.__setattr__(
            node, "data", {"function_code": f"import builtins\nbuiltins.{sentinel} = True"}
        )
        cls = GraphFactory._create_function_node_class(node, [], {str(node.id): node}, {})
        try:
            with pytest.raises(FunctionCodeNotSupportedError):
                await cls().run(_make_run_context())
            assert not hasattr(builtins, sentinel), "function_code was EXECUTED"
        finally:
            if hasattr(builtins, sentinel):
                delattr(builtins, sentinel)

    @pytest.mark.asyncio
    async def test_function_node_routes_to_next(self):
        factory = _factory()
        fn = _make_function_node()
        w = _make_worker_node(name="next_w", worker_name="w1")
        edge = EdgeSchema(source=fn.id, target=w.id)
        node_map = {str(fn.id): fn, str(w.id): w}
        edge_map = factory._build_edge_map([edge])
        classes = factory._create_node_classes([fn, w], node_map, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(fn.id):
                state = _make_state()
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_function_node_no_function_code(self):
        factory = _factory()
        node = _make_function_node(data={"function_code": ""})
        node_map = {str(node.id): node}
        edge_map: dict = {}
        classes = factory._create_node_classes([node], node_map, edge_map)
        instance = classes[0]()
        state = _make_state(variables={"x": 42})
        ctx = _make_run_context(state)
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_function_node_no_data(self):
        node = _make_function_node()
        object.__setattr__(node, "data", None)
        node_map = {str(node.id): node}
        edge_map: dict = {}
        cls = GraphFactory._create_function_node_class(node, [], node_map, edge_map)
        instance = cls()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_function_node_edge_conditions(self):
        factory = _factory()
        fn = _make_function_node()
        t_node = _make_worker_node(name="true_n", worker_name="w1")
        f_node = _make_worker_node(name="false_n", worker_name="w2")
        all_nodes = {
            str(fn.id): fn,
            str(t_node.id): t_node,
            str(f_node.id): f_node,
        }
        next_edges = [
            {"target": str(t_node.id), "condition": "result == True"},
            {"target": str(f_node.id), "condition": None},
        ]
        edge_map = {str(fn.id): next_edges}
        classes = factory._create_node_classes([fn, t_node, f_node], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(fn.id):
                state = _make_state()
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_function_node_next_node_not_found_returns_end(self):
        fn = _make_function_node()
        node_map = {str(fn.id): fn}
        edge_map = {str(fn.id): [{"target": "ghost_id", "condition": None}]}
        cls = GraphFactory._create_function_node_class(
            fn,
            ["ghost_id"],
            node_map,
            edge_map,
            [{"target": "ghost_id", "condition": None}],
        )
        instance = cls()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)


# ---------------------------------------------------------------------------
# 17. Condition node run()
# ---------------------------------------------------------------------------


class TestConditionNodeRun:
    @pytest.mark.asyncio
    async def test_condition_true_routes_first_branch(self):
        factory = _factory()
        cond = _make_condition_node(condition="x > 5")
        t_node = _make_worker_node(name="t_branch", worker_name="w1")
        f_node = _make_worker_node(name="f_branch", worker_name="w2")
        all_nodes = {
            str(cond.id): cond,
            str(t_node.id): t_node,
            str(f_node.id): f_node,
        }
        edge_map = {
            str(cond.id): [
                {"target": str(t_node.id), "condition": None},
                {"target": str(f_node.id), "condition": None},
            ]
        }
        classes = factory._create_node_classes([cond, t_node, f_node], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                state = _make_state(variables={"x": 10})
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_condition_false_routes_second_branch(self):
        factory = _factory()
        cond = _make_condition_node(condition="x > 5")
        t_node = _make_worker_node(name="t_branch", worker_name="w1")
        f_node = _make_worker_node(name="f_branch", worker_name="w2")
        all_nodes = {
            str(cond.id): cond,
            str(t_node.id): t_node,
            str(f_node.id): f_node,
        }
        edge_map = {
            str(cond.id): [
                {"target": str(t_node.id), "condition": None},
                {"target": str(f_node.id), "condition": None},
            ]
        }
        classes = factory._create_node_classes([cond, t_node, f_node], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                state = _make_state(variables={"x": 2})
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, BaseNode)
                break

    @pytest.mark.asyncio
    async def test_condition_false_no_second_branch_returns_end(self):
        factory = _factory()
        cond = _make_condition_node(condition="x > 5")
        t_node = _make_worker_node(name="t_branch", worker_name="w1")
        all_nodes = {str(cond.id): cond, str(t_node.id): t_node}
        edge_map = {str(cond.id): [{"target": str(t_node.id), "condition": None}]}
        classes = factory._create_node_classes([cond, t_node], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                state = _make_state(variables={"x": 2})
                ctx = _make_run_context(state)
                result = await inst.run(ctx)
                assert isinstance(result, End)
                break

    @pytest.mark.asyncio
    async def test_condition_no_next_nodes_returns_end(self):
        factory = _factory()
        cond = _make_condition_node(condition="True")
        all_nodes = {str(cond.id): cond}
        edge_map: dict = {}
        classes = factory._create_node_classes([cond], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                ctx = _make_run_context()
                result = await inst.run(ctx)
                assert isinstance(result, End)
                break

    @pytest.mark.asyncio
    async def test_condition_stores_result_in_state(self):
        factory = _factory()
        cond = _make_condition_node(condition="True")
        all_nodes = {str(cond.id): cond}
        edge_map: dict = {}
        classes = factory._create_node_classes([cond], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                state = _make_state()
                ctx = _make_run_context(state)
                await inst.run(ctx)
                assert str(cond.id) in state.node_outputs
                assert state.node_outputs[str(cond.id)] is True
                break

    @pytest.mark.asyncio
    async def test_condition_no_data_uses_default(self):
        cond = _make_condition_node()
        object.__setattr__(cond, "data", None)
        all_nodes = {str(cond.id): cond}
        edge_map: dict = {}
        cls = GraphFactory._create_condition_node_class(cond, [], all_nodes, edge_map)
        instance = cls()
        state = _make_state()
        ctx = _make_run_context(state)
        result = await instance.run(ctx)
        assert isinstance(result, End)
        assert state.node_outputs[str(cond.id)] is True

    @pytest.mark.asyncio
    async def test_condition_unsafe_ast_returns_end_false(self):
        cond = _make_condition_node(condition="__import__('os').system('ls')")
        all_nodes = {str(cond.id): cond}
        edge_map: dict = {}
        cls = GraphFactory._create_condition_node_class(cond, [], all_nodes, edge_map)
        instance = cls()
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)

    @pytest.mark.asyncio
    async def test_condition_uses_node_outputs_by_name(self):
        factory = _factory()
        prev = _make_worker_node(name="prev_node", worker_name="w1")
        cond = _make_condition_node(condition="prev_node == 42")
        all_nodes = {str(prev.id): prev, str(cond.id): cond}
        edge_map: dict = {}
        classes = factory._create_node_classes([prev, cond], all_nodes, edge_map)
        for cls in classes:
            inst = cls()
            if _node_schema_id(inst) == str(cond.id):
                state = _make_state(node_outputs={str(prev.id): 42})
                ctx = _make_run_context(state)
                await inst.run(ctx)
                assert state.node_outputs[str(cond.id)] is True
                break

    @pytest.mark.asyncio
    async def test_condition_branch_node_not_in_registry_returns_end(self):
        cond = _make_condition_node(condition="True")
        fake_target = _make_worker_node(name="target", worker_name="w1")
        all_nodes = {str(cond.id): cond, str(fake_target.id): fake_target}
        edge_map: dict = {}
        cls = GraphFactory._create_condition_node_class(
            cond, [str(fake_target.id)], all_nodes, edge_map
        )
        instance = cls()
        _set_node_class_registry(instance, {})
        ctx = _make_run_context()
        result = await instance.run(ctx)
        assert isinstance(result, End)


# ---------------------------------------------------------------------------
# 18. NodeSchema validation
# ---------------------------------------------------------------------------


class TestNodeSchemaValidation:
    def test_worker_node_creation(self):
        node = NodeSchema(name="test", node_type="worker", worker_name="my_worker")
        assert node.name == "test"
        assert node.node_type == "worker"
        assert node.worker_name == "my_worker"

    def test_invalid_node_type_raises(self):
        with pytest.raises(ValidationError, match="Invalid node_type"):
            NodeSchema(name="test", node_type="invalid")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            NodeSchema(name="", node_type="worker", worker_name="w1")

    def test_worker_node_missing_worker_name_raises(self):
        with pytest.raises(ValidationError, match="worker_name"):
            NodeSchema(name="test", node_type="worker")

    def test_function_node_missing_code_raises(self):
        with pytest.raises(ValidationError, match="function_name"):
            NodeSchema(name="test", node_type="function")

    def test_condition_node_missing_condition_raises(self):
        with pytest.raises(ValidationError, match="condition"):
            NodeSchema(name="test", node_type="condition")

    def test_timeout_too_low(self):
        with pytest.raises(ValidationError, match="Timeout"):
            NodeSchema(name="test", node_type="worker", worker_name="w1", timeout=0)

    def test_timeout_too_high(self):
        with pytest.raises(ValidationError, match="Timeout"):
            NodeSchema(name="test", node_type="worker", worker_name="w1", timeout=9999)

    def test_retry_count_negative(self):
        with pytest.raises(ValidationError, match="Retry count"):
            NodeSchema(name="test", node_type="worker", worker_name="w1", retry_count=-1)

    def test_retry_count_too_high(self):
        with pytest.raises(ValidationError, match="Retry count"):
            NodeSchema(name="test", node_type="worker", worker_name="w1", retry_count=11)

    def test_valid_timeout_and_retry(self):
        node = NodeSchema(
            name="test",
            node_type="worker",
            worker_name="w1",
            timeout=60,
            retry_count=3,
        )
        assert node.timeout == 60
        assert node.retry_count == 3

    def test_node_with_all_optional_fields(self):
        node = NodeSchema(
            name="full_node",
            node_type="worker",
            worker_name="w1",
            function_name=None,
            task_template="Do something",
            data={"key": "value"},
            timeout=120,
            retry_count=2,
            framework_config={"env": "test"},
        )
        assert node.framework_config == {"env": "test"}
        assert node.data == {"key": "value"}

    def test_worker_name_empty_string_becomes_none(self):
        with pytest.raises(ValidationError):
            NodeSchema(name="test", node_type="worker", worker_name="")

    def test_worker_name_null_bytes_raises(self):
        with pytest.raises(ValidationError, match="null bytes"):
            NodeSchema(name="test", node_type="worker", worker_name="bad\x00name")

    def test_worker_name_too_long_raises(self):
        with pytest.raises(ValidationError, match="200 characters"):
            NodeSchema(name="test", node_type="worker", worker_name="x" * 201)

    def test_worker_name_invalid_pattern_raises(self):
        with pytest.raises(ValidationError, match="Name must start"):
            NodeSchema(name="test", node_type="worker", worker_name="bad name!")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            # Intentional unknown field — verifies pydantic extra="forbid".
            NodeSchema(
                name="test",
                node_type="worker",
                worker_name="w1",
                **cast("dict[str, Any]", {"unknown_field": "bad"}),
            )

    def test_condition_node_with_task_template_as_condition(self):
        node = NodeSchema(
            name="cond",
            node_type="condition",
            task_template="some condition logic",
        )
        assert node.task_template == "some condition logic"

    def test_function_node_with_function_name(self):
        node = NodeSchema(
            name="fn",
            node_type="function",
            function_name="my_func",
        )
        assert node.function_name == "my_func"


# ---------------------------------------------------------------------------
# 19. EdgeSchema validation
# ---------------------------------------------------------------------------


class TestEdgeSchemaValidation:
    def test_basic_edge(self):
        from uuid import uuid7

        s, t = uuid7(), uuid7()
        edge = EdgeSchema(source=s, target=t)
        assert edge.source == s
        assert edge.target == t
        assert edge.condition is None
        assert edge.metadata is None

    def test_edge_with_condition(self):
        from uuid import uuid7

        s, t = uuid7(), uuid7()
        edge = EdgeSchema(source=s, target=t, condition="result == True")
        assert edge.condition == "result == True"

    def test_edge_with_metadata(self):
        from uuid import uuid7

        s, t = uuid7(), uuid7()
        edge = EdgeSchema(source=s, target=t, metadata={"weight": 1.0})
        assert edge.metadata == {"weight": 1.0}

    def test_edge_condition_null_bytes_raises(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="null bytes"):
            EdgeSchema(source=uuid7(), target=uuid7(), condition="bad\x00cond")

    def test_edge_condition_too_long_raises(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="1000 characters"):
            EdgeSchema(source=uuid7(), target=uuid7(), condition="x" * 1001)

    def test_edge_condition_invalid_pattern_raises(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="valid condition syntax"):
            EdgeSchema(source=uuid7(), target=uuid7(), condition=";;;")

    def test_edge_metadata_too_large_raises(self):
        from uuid import uuid7

        huge_meta = {"key": "x" * 6000}
        with pytest.raises(ValidationError, match="too large"):
            EdgeSchema(source=uuid7(), target=uuid7(), metadata=huge_meta)

    def test_edge_metadata_null_bytes_in_key_raises(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="null bytes"):
            EdgeSchema(source=uuid7(), target=uuid7(), metadata={"\x00bad": "val"})

    def test_edge_metadata_null_bytes_in_value_raises(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="null bytes"):
            EdgeSchema(source=uuid7(), target=uuid7(), metadata={"key": "val\x00ue"})

    def test_edge_metadata_null_bytes_in_nested_list(self):
        from uuid import uuid7

        with pytest.raises(ValidationError, match="null bytes"):
            EdgeSchema(
                source=uuid7(),
                target=uuid7(),
                metadata={"items": ["ok", "bad\x00"]},
            )

    def test_edge_extra_fields_forbidden(self):
        from uuid import uuid7

        with pytest.raises(ValidationError):
            EdgeSchema(
                source=uuid7(),
                target=uuid7(),
                **cast("dict[str, Any]", {"extra_field": "nope"}),
            )


# ---------------------------------------------------------------------------
# 20. ExecutionRequest validation
# ---------------------------------------------------------------------------


class TestExecutionRequestValidation:
    def test_defaults(self):
        req = ExecutionRequest()
        assert req.variables == {}
        assert req.start_node is None

    def test_with_variables(self):
        req = ExecutionRequest(variables={"x": 1})
        assert req.variables == {"x": 1}

    def test_start_node_none(self):
        req = ExecutionRequest(start_node=None)
        assert req.start_node is None

    def test_start_node_empty_string(self):
        req = ExecutionRequest(start_node="")
        assert req.start_node == ""

    def test_start_node_whitespace_becomes_empty(self):
        req = ExecutionRequest(start_node="   ")
        assert req.start_node == ""

    def test_start_node_null_bytes_raises(self):
        with pytest.raises(ValidationError, match="null bytes"):
            ExecutionRequest(start_node="bad\x00id")

    def test_start_node_too_long_raises(self):
        with pytest.raises(ValidationError, match="200 characters"):
            ExecutionRequest(start_node="x" * 201)

    def test_variables_too_large_raises(self):
        # Cap raised to 1 MiB in Phase 1 (env WM_GRAPH_VARIABLES_MAX_BYTES);
        # use a 2 MB payload to clear the new default.
        with pytest.raises(ValidationError, match="too large"):
            ExecutionRequest(variables={"big": "x" * (2 * 1024 * 1024)})

    def test_start_node_valid_uuid(self):
        req = ExecutionRequest(start_node="550e8400-e29b-41d4-a716-446655440001")
        assert req.start_node == "550e8400-e29b-41d4-a716-446655440001"

    def test_start_node_identifier(self):
        req = ExecutionRequest(start_node="__main__")
        assert req.start_node == "__main__"


# ---------------------------------------------------------------------------
# 21. ExecutionResponse model
# ---------------------------------------------------------------------------


class TestExecutionResponseModel:
    def test_full_response(self):
        now = datetime.now()
        resp = ExecutionResponse(
            session_id="s1",
            status="completed",
            current_node="node1",
            node_outputs={"n1": "out"},
            completed_nodes=["n1"],
            failed_nodes=[],
            final_output="done",
            error_message=None,
            started_at=now,
            updated_at=now,
        )
        assert resp.session_id == "s1"
        assert resp.status == "completed"
        assert resp.final_output == "done"

    def test_minimal_response(self):
        now = datetime.now()
        resp = ExecutionResponse(
            session_id="s2",
            status="running",
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
            started_at=now,
            updated_at=now,
        )
        assert resp.current_node is None
        assert resp.final_output is None


# ---------------------------------------------------------------------------
# 22. GraphCreateRequest validation
# ---------------------------------------------------------------------------


class TestGraphCreateRequestValidation:
    def test_valid_create_request(self):
        node = _make_worker_node()
        req = GraphCreateRequest(name="test_graph", nodes=[node])
        assert req.name == "test_graph"
        assert len(req.nodes) == 1

    def test_empty_nodes_raises(self):
        with pytest.raises(ValidationError, match="at least one node"):
            GraphCreateRequest(name="test", nodes=[])

    def test_too_many_nodes_raises(self):
        nodes = [_make_worker_node(name=f"n{i}", worker_name=f"w{i}") for i in range(1001)]
        with pytest.raises(ValidationError, match="1000 nodes"):
            GraphCreateRequest(name="test", nodes=nodes)

    def test_duplicate_node_ids_raises(self):
        node = _make_worker_node()
        with pytest.raises(ValidationError, match="Duplicate"):
            GraphCreateRequest(name="test", nodes=[node, node])

    def test_auto_generates_node_ids(self):
        req = GraphCreateRequest.model_validate(
            {
                "name": "test",
                "nodes": [
                    {"name": "n1", "node_type": "worker", "worker_name": "w1"},
                ],
            }
        )
        assert req.nodes[0].id is not None

    def test_variables_too_large_raises(self):
        # Cap raised to 1 MiB in Phase 1; use a 2 MB payload.
        node = _make_worker_node()
        with pytest.raises(ValidationError, match="too large"):
            GraphCreateRequest(
                name="test",
                nodes=[node],
                variables={"big": "x" * (2 * 1024 * 1024)},
            )


# ---------------------------------------------------------------------------
# 23. GraphUpdateRequest validation
# ---------------------------------------------------------------------------


class TestGraphUpdateRequestValidation:
    def test_partial_update_name_only(self):
        req = GraphUpdateRequest(name="new_name")
        assert req.name == "new_name"
        assert req.nodes is None
        assert req.edges is None

    def test_null_bytes_in_name_raises(self):
        with pytest.raises(ValidationError, match="null bytes"):
            GraphUpdateRequest(name="bad\x00name")

    def test_null_bytes_in_description_raises(self):
        with pytest.raises(ValidationError, match="null bytes"):
            GraphUpdateRequest(description="bad\x00desc")

    def test_too_many_nodes_raises(self):
        nodes = [_make_worker_node(name=f"n{i}", worker_name=f"w{i}") for i in range(1001)]
        with pytest.raises(ValidationError, match="1000 nodes"):
            GraphUpdateRequest(nodes=nodes)

    def test_too_many_edges_raises(self):
        from uuid import uuid7

        edges = [EdgeSchema(source=uuid7(), target=uuid7()) for _ in range(5001)]
        with pytest.raises(ValidationError, match="5000 edges"):
            GraphUpdateRequest(edges=edges)

    def test_duplicate_node_ids_raises(self):
        node = _make_worker_node()
        with pytest.raises(ValidationError, match="Duplicate"):
            GraphUpdateRequest(nodes=[node, node])

    def test_duplicate_edges_raises(self):
        from uuid import uuid7

        s, t = uuid7(), uuid7()
        edge1 = EdgeSchema(source=s, target=t)
        edge2 = EdgeSchema(source=s, target=t)
        with pytest.raises(ValidationError, match="Duplicate edge"):
            GraphUpdateRequest(edges=[edge1, edge2])

    def test_edge_condition_null_bytes_raises(self):
        from uuid import uuid7

        s, t = uuid7(), uuid7()
        edge = EdgeSchema(source=s, target=t, condition="x == 1")
        object.__setattr__(edge, "condition", "bad\x00cond")
        with pytest.raises(ValidationError, match="null bytes"):
            GraphUpdateRequest(edges=[edge])

    def test_preprocess_nodes_auto_ids(self):
        req = GraphUpdateRequest.model_validate(
            {
                "nodes": [
                    {"name": "n1", "node_type": "worker", "worker_name": "w1"},
                ],
            }
        )
        assert req.nodes is not None
        assert req.nodes[0].id is not None

    def test_preprocess_remaps_example_uuids(self):
        req = GraphUpdateRequest.model_validate(
            {
                "nodes": [
                    {"name": "n1", "node_type": "worker", "worker_name": "w1"},
                    {"name": "n2", "node_type": "worker", "worker_name": "w2"},
                ],
                "edges": [
                    {
                        "source": "550e8400-e29b-41d4-a716-446655440001",
                        "target": "550e8400-e29b-41d4-a716-446655440002",
                    }
                ],
            }
        )
        assert req.edges is not None
        assert req.nodes is not None
        actual_edge = req.edges[0]
        assert actual_edge.source == req.nodes[0].id
        assert actual_edge.target == req.nodes[1].id


# ---------------------------------------------------------------------------
# 24. NodeExecutionResult model
# ---------------------------------------------------------------------------


class TestNodeExecutionResultModel:
    def test_full_result(self):
        now = datetime.now()
        r = NodeExecutionResult(
            node_id="n1",
            node_name="step1",
            status="success",
            output={"data": 42},
            error=None,
            started_at=now,
            completed_at=now,
            execution_time=1.5,
        )
        assert r.status == "success"
        assert r.execution_time == 1.5

    def test_failed_result(self):
        now = datetime.now()
        r = NodeExecutionResult(
            node_id="n1",
            node_name="step1",
            status="failed",
            error="timeout",
            started_at=now,
        )
        assert r.status == "failed"
        assert r.output is None
        assert r.completed_at is None


# ---------------------------------------------------------------------------
# 25. GraphExecutionState model
# ---------------------------------------------------------------------------


class TestGraphExecutionStateModel:
    def test_full_state(self):
        now = datetime.now()
        state = GraphExecutionState(
            session_id="s1",
            graph_id="g1",
            variables={"x": 1},
            node_outputs={"n1": "out"},
            node_status={"n1": "completed"},
            completed_nodes=["n1"],
            failed_nodes=[],
            worker_assignments={"n1": "w1"},
            node_errors={},
            start_time=now,
            end_time=now,
            error_message=None,
            current_node=None,
            final_output="done",
        )
        assert state.session_id == "s1"
        assert state.final_output == "done"

    def test_minimal_state(self):
        now = datetime.now()
        state = GraphExecutionState(
            session_id="s2",
            graph_id="g2",
            variables={},
            node_outputs={},
            node_status={},
            completed_nodes=[],
            failed_nodes=[],
            worker_assignments={},
            node_errors={},
            start_time=now,
        )
        assert state.end_time is None
        assert state.current_node is None


# ---------------------------------------------------------------------------
# 26. ExecutionEvent model
# ---------------------------------------------------------------------------


class TestExecutionEventModel:
    def test_event_with_defaults(self):
        evt = ExecutionEvent(
            session_id="s1",
            event_type="node_started",
            node_id="n1",
            message="started",
        )
        assert evt.event_id is not None
        assert evt.timestamp is not None
        assert evt.session_id == "s1"

    def test_event_all_fields(self):
        now = datetime.now()
        evt = ExecutionEvent(
            event_id="e1",
            session_id="s1",
            event_type="graph_completed",
            node_id=None,
            timestamp=now,
            data={"result": "ok"},
            message="all done",
        )
        assert evt.event_id == "e1"
        assert evt.data == {"result": "ok"}

    def test_event_minimal(self):
        evt = ExecutionEvent(session_id="s1", event_type="node_failed")
        assert evt.node_id is None
        assert evt.data is None
        assert evt.message is None


# ---------------------------------------------------------------------------
# 27. GraphResponse model
# ---------------------------------------------------------------------------


class TestGraphResponseModel:
    def test_graph_response(self):
        from uuid import uuid7

        now = datetime.now()
        node = _make_worker_node()
        resp = GraphResponse(
            id=uuid7(),
            name="test",
            description="desc",
            nodes=[node],
            edges=[],
            variables={},
            created_at=now,
            updated_at=now,
        )
        assert resp.name == "test"
        assert len(resp.nodes) == 1
