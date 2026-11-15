from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import ValidationError
from pydantic_graph import Graph

from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import (
    EdgeSchema,
    ExecutionEvent,
    ExecutionRequest,
    ExecutionResponse,
    GraphCreateRequest,
    GraphExecutionState,
    GraphUpdateRequest,
    NodeExecutionResult,
    NodeSchema,
)
from src.worker.graphs.pydantic_graph_state import WorkflowState
from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

NODE_ID_1 = UUID("00000000-0000-0000-0000-000000000001")
NODE_ID_2 = UUID("00000000-0000-0000-0000-000000000002")
NODE_ID_3 = UUID("00000000-0000-0000-0000-000000000003")
NODE_ID_4 = UUID("00000000-0000-0000-0000-000000000004")
CREATOR = "creator-1"


def _worker(name="w1", worker_name="agent1", **kw):
    return NodeSchema(name=name, node_type="worker", worker_name=worker_name, **kw)


def _func(name="f1", function_name="do_stuff", **kw):
    defaults: dict[str, Any] = {"data": {}}
    defaults.update(kw)
    return NodeSchema(name=name, node_type="function", function_name=function_name, **defaults)


def _cond(name="c1", condition="True", **kw):
    return NodeSchema(name=name, node_type="condition", data={"condition": condition}, **kw)


def _edge(src, tgt, condition=None):
    return EdgeSchema(source=src, target=tgt, condition=condition)


def _factory():
    return GraphFactory(worker_manager=MagicMock())


def _state(**overrides):
    defaults: dict[str, Any] = dict(
        session_id="s1",
        graph_id="g1",
        variables={},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )
    defaults.update(overrides)
    return WorkflowState(**defaults)


# ============================================================================
# NodeSchema validation
# ============================================================================


class TestNodeSchemaValidation:
    def test_worker_requires_worker_name(self):
        with pytest.raises(ValidationError, match="worker_name"):
            NodeSchema(name="bad", node_type="worker")

    def test_function_requires_code_or_name(self):
        with pytest.raises(ValidationError):
            NodeSchema(name="bad", node_type="function")

    def test_condition_requires_condition_data(self):
        with pytest.raises(ValidationError):
            NodeSchema(name="bad", node_type="condition")

    def test_invalid_node_type_and_empty_name(self):
        with pytest.raises(ValidationError, match="node_type"):
            NodeSchema(name="bad", node_type="banana", worker_name="x")
        with pytest.raises(ValidationError, match="name"):
            NodeSchema(name="", node_type="worker", worker_name="x")

    def test_timeout_and_retry_bounds(self):
        with pytest.raises(ValidationError, match="Timeout"):
            _worker(timeout=0)
        with pytest.raises(ValidationError, match="Timeout"):
            _worker(timeout=9999)
        with pytest.raises(ValidationError, match="Retry"):
            _worker(retry_count=-1)
        with pytest.raises(ValidationError, match="Retry"):
            _worker(retry_count=11)

    def test_valid_worker_round_trips(self):
        n = _worker(id=NODE_ID_1, name="proc", worker_name="agent1", timeout=60, retry_count=3)
        assert n.id == NODE_ID_1
        assert n.timeout == 60
        assert n.retry_count == 3
        with pytest.raises(ValidationError):
            # Intentionally unknown field — verifies pydantic extra="forbid".
            NodeSchema(
                name="bad",
                node_type="worker",
                worker_name="x",
                **cast("dict[str, Any]", {"bogus": "field"}),
            )


# ============================================================================
# EdgeSchema validation
# ============================================================================


class TestEdgeSchemaValidation:
    def test_valid_edge_and_condition(self):
        e = _edge(NODE_ID_1, NODE_ID_2)
        assert e.source == NODE_ID_1 and e.condition is None
        e2 = _edge(NODE_ID_1, NODE_ID_2, condition="result.status == 'ok'")
        assert e2.condition == "result.status == 'ok'"

    def test_null_byte_and_length_limit(self):
        with pytest.raises(ValidationError, match="null bytes"):
            _edge(NODE_ID_1, NODE_ID_2, condition="bad\x00cond")
        with pytest.raises(ValidationError, match="1000"):
            _edge(NODE_ID_1, NODE_ID_2, condition="x" * 1001)


# ============================================================================
# GraphCreateRequest / GraphUpdateRequest validation
# ============================================================================


class TestGraphRequestValidation:
    def test_create_requires_node_and_rejects_duplicates(self):
        with pytest.raises(ValidationError, match="at least one node"):
            GraphCreateRequest(name="empty", nodes=[], edges=[])
        n = _worker(id=NODE_ID_1)
        with pytest.raises(ValidationError, match="Duplicate"):
            GraphCreateRequest(name="dup", nodes=[n, n])

    def test_create_valid_and_node_limit(self):
        n = _worker(id=NODE_ID_1)
        req = GraphCreateRequest(name="pipeline", nodes=[n])
        assert req.name == "pipeline" and len(req.nodes) == 1
        nodes = [_worker(name=f"w{i}", worker_name=f"a{i}") for i in range(1001)]
        with pytest.raises(ValidationError, match="1000"):
            GraphCreateRequest(name="big", nodes=nodes)

    def test_update_partial_and_null_byte_rejection(self):
        req = GraphUpdateRequest(name="new_name")
        assert req.name == "new_name" and req.nodes is None
        with pytest.raises(ValidationError, match="null bytes"):
            GraphUpdateRequest(name="bad\x00name")

    def test_update_duplicate_edges_rejected(self):
        e = _edge(NODE_ID_1, NODE_ID_2)
        with pytest.raises(ValidationError, match="Duplicate edge"):
            GraphUpdateRequest(edges=[e, e])


# ============================================================================
# ExecutionRequest / ExecutionResponse / auxiliary models
# ============================================================================


class TestExecutionModels:
    def test_execution_request_defaults_and_validation(self):
        req = ExecutionRequest()
        assert req.variables == {} and req.start_node is None
        req2 = ExecutionRequest(variables={"file": "data.csv"}, start_node="node_1")
        assert req2.variables is not None
        assert req2.variables["file"] == "data.csv"
        with pytest.raises(ValidationError, match="null bytes"):
            ExecutionRequest(start_node="bad\x00node")
        req3 = ExecutionRequest(start_node="")
        assert req3.start_node == ""

    def test_execution_response_and_auxiliary_models(self):
        now = datetime.now(UTC)
        resp = ExecutionResponse(
            session_id="s1",
            status="completed",
            node_outputs={"n1": "done"},
            completed_nodes=["n1"],
            failed_nodes=[],
            final_output="result",
            started_at=now,
            updated_at=now,
        )
        assert resp.status == "completed" and resp.final_output == "result"

        r = NodeExecutionResult(
            node_id="n1",
            node_name="step1",
            status="success",
            output={"count": 42},
            started_at=now,
            completed_at=now,
            execution_time=1.5,
        )
        assert r.execution_time == 1.5

        s = GraphExecutionState(
            session_id="s1",
            graph_id="g1",
            variables={"x": 1},
            node_outputs={},
            node_status={},
            completed_nodes=[],
            failed_nodes=[],
            worker_assignments={},
            node_errors={},
            start_time=now,
        )
        assert s.graph_id == "g1"

        ev = ExecutionEvent(session_id="s1", event_type="node_started", node_id="n1")
        assert ev.event_id and ev.session_id == "s1"


# ============================================================================
# WorkflowState
# ============================================================================


class TestWorkflowState:
    def test_node_output_lifecycle(self):
        s = _state()
        s.set_node_output("n1", {"count": 5})
        assert s.get_node_output("n1") == {"count": 5}
        assert "n1" in s.completed_nodes and s.node_status["n1"] == "completed"
        s.set_node_output("n1", "updated")
        assert s.completed_nodes.count("n1") == 1
        assert s.get_node_output("n1") == "updated"

    def test_mark_node_failed_and_first_error_wins(self):
        s = _state()
        s.mark_node_failed("n1", "boom")
        assert "n1" in s.failed_nodes and s.node_errors["n1"] == "boom"
        assert s.error_message == "boom"
        s.mark_node_failed("n2", "second")
        assert s.error_message == "boom"

    def test_mark_running_and_retry(self):
        s = _state()
        s.mark_node_running("n1")
        assert s.current_node == "n1" and s.node_status["n1"] == "running"
        assert s.increment_retry_count("n1") == 1
        assert s.increment_retry_count("n1") == 2

    def test_duration_and_progress(self):
        s = _state()
        s.set_node_duration("n1", 3.14)
        assert s.node_durations["n1"] == 3.14
        assert s.get_progress_percentage(0) == 100.0
        s.set_node_output("n1", "ok")
        assert s.get_progress_percentage(4) == 25.0
        s.mark_node_failed("n2")
        assert s.get_progress_percentage(4) == 50.0
        assert not s.is_execution_complete(4)

    def test_overall_status_transitions(self):
        s = _state()
        assert s._get_overall_status() == "pending"
        s.mark_node_running("n1")
        assert s._get_overall_status() == "running"
        s.end_time = datetime.now(UTC)
        assert s._get_overall_status() == "completed"
        s.mark_node_failed("n1", "kaboom")
        assert s._get_overall_status() == "failed"

    def test_completed_with_errors_status(self):
        s = _state()
        s.mark_node_failed("n1")
        s.end_time = datetime.now(UTC)
        assert s._get_overall_status() == "completed_with_errors"

    def test_execution_summary(self):
        s = _state()
        s.start_time = datetime.now(UTC)
        s.set_node_output("n1", "ok")
        s.mark_node_failed("n2", "timeout")
        s.end_time = s.start_time + timedelta(seconds=5)
        summary = s.get_execution_summary()
        assert summary["completed_nodes"] == 1
        assert summary["failed_nodes"] == 1
        assert summary["has_errors"] is True
        assert summary["execution_time"] is not None

    def test_to_dict_and_from_dict_roundtrip(self):
        s = _state()
        s.start_time = datetime(2025, 1, 1, tzinfo=UTC)
        s.end_time = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
        s.set_node_output("n1", {"x": 1})
        s.mark_node_failed("n2", "err")
        s.set_node_duration("n1", 2.5)
        s.worker_assignments["n1"] = "agent1"
        s.tenant_id = "tenant-xyz"
        s.subject_id = "subject-abc"
        d = s.to_dict()
        restored = WorkflowState.from_dict(d)
        assert restored.session_id == s.session_id
        assert restored.node_outputs == s.node_outputs
        assert restored.failed_nodes == s.failed_nodes
        assert restored.node_durations == s.node_durations
        # Run identity survives the serialization round-trip (needed for
        # stepwise/durable graph runs to keep enforcing BYO posture).
        assert restored.tenant_id == "tenant-xyz"
        assert restored.subject_id == "subject-abc"
        restored2 = WorkflowState.from_dict({"session_id": "s1", "graph_id": "g1"})
        assert restored2.start_time is None
        # Absent identity ⇒ None (byte-identical fallback to DEFAULT_TENANT_ID).
        assert restored2.tenant_id is None
        assert restored2.subject_id is None


# ============================================================================
# GraphFactory.validate_graph_structure
# ============================================================================


class TestValidateGraphStructure:
    def test_empty_and_missing_edge_references(self):
        issues = GraphFactory.validate_graph_structure([], [])
        assert any("at least one node" in i for i in issues)

        n = _worker(id=NODE_ID_1)
        issues = GraphFactory.validate_graph_structure([n], [_edge(NODE_ID_2, NODE_ID_1)])
        assert any("source" in i for i in issues)
        issues = GraphFactory.validate_graph_structure([n], [_edge(NODE_ID_1, NODE_ID_2)])
        assert any("target" in i for i in issues)

    def test_node_type_specific_missing_config(self):
        w = _worker(id=NODE_ID_1)
        object.__setattr__(w, "worker_name", None)
        assert any("worker_name" in i for i in GraphFactory.validate_graph_structure([w], []))

        f = _func(id=NODE_ID_1)
        object.__setattr__(f, "data", {"function_code": "result = 1"})
        assert any("function_code" in i for i in GraphFactory.validate_graph_structure([f], []))

        c = _cond(id=NODE_ID_1)
        object.__setattr__(c, "data", {})
        assert any("condition" in i for i in GraphFactory.validate_graph_structure([c], []))

    def test_valid_graph_returns_no_issues(self):
        n1 = _worker(id=NODE_ID_1)
        n2 = _worker(id=NODE_ID_2, name="w2", worker_name="a2")
        assert GraphFactory.validate_graph_structure([n1, n2], [_edge(NODE_ID_1, NODE_ID_2)]) == []


# ============================================================================
# GraphFactory.create_graph
# ============================================================================


class TestGraphFactoryCreateGraph:
    def test_creates_real_graph_with_metadata(self):
        n1 = _worker(id=NODE_ID_1)
        n2 = _func(id=NODE_ID_2, name="f1")
        factory = _factory()
        graph = factory.create_graph([n1, n2], [_edge(NODE_ID_1, NODE_ID_2)], name="test_graph")
        assert isinstance(graph, Graph)
        meta = factory._graph_metadata.get(id(graph))
        assert meta is not None and meta.name == "test_graph"
        assert str(NODE_ID_1) in meta.node_schemas

    def test_validation_toggle(self):
        n = _worker(id=NODE_ID_1)
        object.__setattr__(n, "worker_name", None)
        factory = _factory()
        graph = factory.create_graph([n], [], validate=False)
        assert isinstance(graph, Graph)
        with pytest.raises(ValueError, match="Graph validation failed"):
            _factory().create_graph([], [])

    def test_unsupported_node_type_raises(self):
        n = _worker(id=NODE_ID_1)
        object.__setattr__(n, "node_type", "magic")
        with pytest.raises(ValueError, match="Unsupported node type"):
            _factory().create_graph([n], [], validate=False)

    def test_graph_with_all_node_types(self):
        n1 = _worker(id=NODE_ID_1)
        n2 = _func(id=NODE_ID_2, name="f1")
        n3 = _cond(id=NODE_ID_3, name="c1", condition="result == True")
        n4 = _worker(id=NODE_ID_4, name="w_true", worker_name="a4")
        edges = [
            _edge(NODE_ID_1, NODE_ID_2),
            _edge(NODE_ID_2, NODE_ID_3),
            _edge(NODE_ID_3, NODE_ID_4),
            _edge(NODE_ID_3, NODE_ID_1),
        ]
        graph = _factory().create_graph([n1, n2, n3, n4], edges, validate=False)
        assert isinstance(graph, Graph)

    def test_auto_generated_name(self):
        factory = _factory()
        graph = factory.create_graph([_worker(id=NODE_ID_1)], [])
        meta = factory._graph_metadata.get(id(graph))
        assert meta.name.startswith("graph_")


# ============================================================================
# GraphFactory.create_start_node
# ============================================================================


class TestStartNode:
    def test_explicit_and_auto_detected(self):
        n1 = _worker(id=NODE_ID_1, name="root")
        n2 = _worker(id=NODE_ID_2, name="leaf", worker_name="a2")
        factory = _factory()
        graph = factory.create_graph([n1, n2], [_edge(NODE_ID_1, NODE_ID_2)])
        start = factory.create_start_node(graph, start_node_id=str(NODE_ID_1))
        assert start._node_schema_id == str(NODE_ID_1)
        auto = factory.create_start_node(graph)
        assert auto._node_schema_id == str(NODE_ID_1)

    def test_invalid_start_node_raises(self):
        factory = _factory()
        graph = factory.create_graph([_worker(id=NODE_ID_1)], [])
        with pytest.raises(ValueError, match="not found"):
            factory.create_start_node(graph, start_node_id="nonexistent")


# ============================================================================
# GraphFactory._build_edge_map
# ============================================================================


class TestBuildEdgeMap:
    def test_edge_map_structure(self):
        edges = [_edge(NODE_ID_1, NODE_ID_2, condition="x > 0"), _edge(NODE_ID_1, NODE_ID_3)]
        result = GraphFactory._build_edge_map(edges)
        assert len(result[str(NODE_ID_1)]) == 2
        assert result[str(NODE_ID_1)][0]["condition"] == "x > 0"
        assert result[str(NODE_ID_1)][1]["condition"] is None


# ============================================================================
# Graph validation logic (cycle, orphan, unreachable, self-loop, conditions, templates)
# ============================================================================


def _run_validation(nodes, edges, variables=None):
    import re
    from collections import deque

    variables = variables or {}
    node_ids = {str(n.id) for n in nodes}
    node_map = {str(n.id): n for n in nodes}
    outgoing = {nid: [] for nid in node_ids}
    incoming = {nid: [] for nid in node_ids}
    errors, warnings = [], []

    for edge in edges:
        src, tgt = str(edge.source), str(edge.target)
        if src not in node_ids:
            errors.append({"type": "invalid_edge_source", "node_id": src})
        if tgt not in node_ids:
            errors.append({"type": "invalid_edge_target", "node_id": tgt})
        if src in node_ids and tgt in node_ids:
            outgoing[src].append(tgt)
            incoming[tgt].append(src)
        if src == tgt and src in node_ids:
            warnings.append({"type": "self_loop", "node_id": src})

    start_nodes = [nid for nid in node_ids if not incoming[nid]]
    if len(nodes) > 1:
        for nid in node_ids:
            if not incoming[nid] and not outgoing[nid]:
                warnings.append({"type": "orphaned_nodes", "node_id": nid})

    reachable = set(start_nodes)
    queue = deque(start_nodes)
    while queue:
        for neighbor in outgoing.get(queue.popleft(), []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    for nid in node_ids - reachable:
        warnings.append({"type": "unreachable_node", "node_id": nid})

    for nid, node in node_map.items():
        if node.node_type == "condition" and len(outgoing[nid]) < 2:
            errors.append({"type": "condition_missing_branches", "node_id": nid})

    tmpl_pat = re.compile(r"\{\{\s*(\w+)\s*\}\}")
    gvars = set(variables.keys())
    for nid, node in node_map.items():
        if node.task_template:
            undefined = set(tmpl_pat.findall(node.task_template)) - gvars
            if undefined:
                warnings.append({"type": "undefined_template_variable", "node_id": nid})

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in node_ids}
    parent = {nid: None for nid in node_ids}
    cycles = []

    def dfs(u):
        color[u] = GRAY
        for v in outgoing.get(u, []):
            if color[v] == GRAY:
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    p = parent.get(cur)
                    if p is None:
                        break
                    cur = p
                cycle.append(v)
                cycle.reverse()
                cycles.append(cycle)
            elif color[v] == WHITE:
                parent[v] = u
                dfs(v)
        color[u] = BLACK

    for nid in node_ids:
        if color[nid] == WHITE:
            dfs(nid)
    for c in cycles:
        errors.append({"type": "circular_dependency", "cycle_path": c})

    return {"valid": not errors, "errors": errors, "warnings": warnings}


class TestCycleDetection:
    def test_two_and_three_node_cycles(self):
        n1, n2 = _worker(id=NODE_ID_1, name="a"), _worker(id=NODE_ID_2, name="b", worker_name="a2")
        r = _run_validation([n1, n2], [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_2, NODE_ID_1)])
        assert not r["valid"]
        assert any(e["type"] == "circular_dependency" for e in r["errors"])

        n3 = _worker(id=NODE_ID_3, name="c", worker_name="a3")
        r2 = _run_validation(
            [n1, n2, n3],
            [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_2, NODE_ID_3), _edge(NODE_ID_3, NODE_ID_1)],
        )
        assert any(e["type"] == "circular_dependency" for e in r2["errors"])

    def test_acyclic_graph_passes(self):
        n1, n2, n3 = (
            _worker(id=NODE_ID_1, name="a"),
            _worker(id=NODE_ID_2, name="b", worker_name="a2"),
            _worker(id=NODE_ID_3, name="c", worker_name="a3"),
        )
        r = _run_validation(
            [n1, n2, n3],
            [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_2, NODE_ID_3)],
        )
        assert not any(e["type"] == "circular_dependency" for e in r["errors"])


class TestOrphansAndUnreachable:
    def test_orphan_detected_in_multi_node_graph(self):
        n1 = _worker(id=NODE_ID_1, name="connected")
        n2 = _worker(id=NODE_ID_2, name="orphan", worker_name="a2")
        n3 = _worker(id=NODE_ID_3, name="connected_too", worker_name="a3")
        r = _run_validation([n1, n2, n3], [_edge(NODE_ID_1, NODE_ID_3)])
        assert any(
            w["type"] == "orphaned_nodes" and w["node_id"] == str(NODE_ID_2) for w in r["warnings"]
        )

    def test_single_node_not_orphan(self):
        r = _run_validation([_worker(id=NODE_ID_1)], [])
        assert not any(w["type"] == "orphaned_nodes" for w in r["warnings"])

    def test_unreachable_cycle_island(self):
        n1, n2 = (
            _worker(id=NODE_ID_1, name="start"),
            _worker(id=NODE_ID_2, name="end", worker_name="a2"),
        )
        n3, n4 = (
            _worker(id=NODE_ID_3, name="isle_a", worker_name="a3"),
            _worker(id=NODE_ID_4, name="isle_b", worker_name="a4"),
        )
        r = _run_validation(
            [n1, n2, n3, n4],
            [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_3, NODE_ID_4), _edge(NODE_ID_4, NODE_ID_3)],
        )
        unreachable_ids = {w["node_id"] for w in r["warnings"] if w["type"] == "unreachable_node"}
        assert str(NODE_ID_3) in unreachable_ids and str(NODE_ID_4) in unreachable_ids


class TestSelfLoopAndConditionBranches:
    def test_self_loop_warning(self):
        r = _run_validation([_worker(id=NODE_ID_1)], [_edge(NODE_ID_1, NODE_ID_1)])
        assert any(w["type"] == "self_loop" for w in r["warnings"])

    def test_condition_branch_validation(self):
        c = _cond(id=NODE_ID_1, condition="x > 0")
        n2 = _worker(id=NODE_ID_2, name="t", worker_name="a2")
        n3 = _worker(id=NODE_ID_3, name="f", worker_name="a3")
        r1 = _run_validation([c, n2], [_edge(NODE_ID_1, NODE_ID_2)])
        assert any(e["type"] == "condition_missing_branches" for e in r1["errors"])
        r2 = _run_validation(
            [c, n2, n3], [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_1, NODE_ID_3)]
        )
        assert not any(e["type"] == "condition_missing_branches" for e in r2["errors"])


class TestTemplateVariableValidation:
    def test_undefined_and_defined_vars(self):
        n = _worker(id=NODE_ID_1, task_template="Process {{filename}} with {{mode}}")
        r1 = _run_validation([n], [], variables={"filename": "data.csv"})
        assert any(w["type"] == "undefined_template_variable" for w in r1["warnings"])
        r2 = _run_validation([n], [], variables={"filename": "data.csv", "mode": "fast"})
        assert not any(w["type"] == "undefined_template_variable" for w in r2["warnings"])


# ============================================================================
# Graph structure classification
# ============================================================================


def _classify(nodes, edges):
    if not nodes:
        return "empty"
    if len(nodes) == 1:
        return "single_node"
    if not edges:
        return "disconnected"
    conns = {}
    for e in edges:
        s, t = str(e.source), str(e.target)
        conns.setdefault(s, {"in": 0, "out": 0})
        conns.setdefault(t, {"in": 0, "out": 0})
        conns[s]["out"] += 1
        conns[t]["in"] += 1
    branching = any(c["out"] > 1 for c in conns.values())
    merging = any(c["in"] > 1 for c in conns.values())
    if branching and merging:
        return "complex"
    if branching:
        return "branching"
    if merging:
        return "merging"
    if len(edges) == len(nodes) - 1:
        return "linear"
    return "complex"


class TestStructureClassification:
    def test_all_structure_types(self):
        assert _classify([], []) == "empty"
        assert _classify([_worker(id=NODE_ID_1)], []) == "single_node"
        n1 = _worker(id=NODE_ID_1)
        n2 = _worker(id=NODE_ID_2, name="w2", worker_name="a2")
        assert _classify([n1, n2], []) == "disconnected"
        n3 = _worker(id=NODE_ID_3, name="w3", worker_name="a3")
        assert (
            _classify([n1, n2, n3], [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_2, NODE_ID_3)])
            == "linear"
        )
        assert (
            _classify([n1, n2, n3], [_edge(NODE_ID_1, NODE_ID_2), _edge(NODE_ID_1, NODE_ID_3)])
            == "branching"
        )
        assert (
            _classify([n1, n2, n3], [_edge(NODE_ID_1, NODE_ID_3), _edge(NODE_ID_2, NODE_ID_3)])
            == "merging"
        )

    def test_complex_structure(self):
        n1, n2, n3, n4 = (
            _worker(id=NODE_ID_1),
            _worker(id=NODE_ID_2, name="w2", worker_name="a2"),
            _worker(id=NODE_ID_3, name="w3", worker_name="a3"),
            _worker(id=NODE_ID_4, name="w4", worker_name="a4"),
        )
        edges = [
            _edge(NODE_ID_1, NODE_ID_2),
            _edge(NODE_ID_1, NODE_ID_3),
            _edge(NODE_ID_2, NODE_ID_4),
            _edge(NODE_ID_3, NODE_ID_4),
        ]
        assert _classify([n1, n2, n3, n4], edges) == "complex"


# ============================================================================
# Graph operations CRUD with InMemoryDB
# ============================================================================


class TestGraphOperationsCRUD:
    @pytest.mark.asyncio
    async def test_create_get_list_roundtrip(self):
        from src.worker.graphs.graph_operations import (
            create_graph_service,
            get_graph_service,
            list_graphs_service,
        )

        mem = InMemoryDB()
        n1 = _worker(id=NODE_ID_1)
        n2 = _worker(id=NODE_ID_2, name="w2", worker_name="a2")
        with patch_get_pg_connection("src.db.graph_state_db", mem):
            c1 = await create_graph_service(
                CREATOR, CREATOR, GraphCreateRequest(name="g1", nodes=[n1])
            )
            await create_graph_service(CREATOR, CREATOR, GraphCreateRequest(name="g2", nodes=[n2]))
            assert c1.name == "g1" and len(c1.nodes) == 1
            fetched = await get_graph_service(CREATOR, CREATOR, str(c1.id))
            assert fetched.name == "g1"
            graphs = await list_graphs_service(CREATOR, CREATOR)
            assert len(graphs) == 2

    @pytest.mark.asyncio
    async def test_update_graph(self):
        from src.worker.graphs.graph_operations import create_graph_service, update_graph_service

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.graph_state_db", mem):
            created = await create_graph_service(
                CREATOR, CREATOR, GraphCreateRequest(name="orig", nodes=[_worker(id=NODE_ID_1)])
            )
            updated = await update_graph_service(
                CREATOR, CREATOR, str(created.id), GraphUpdateRequest(name="renamed")
            )
            assert updated.name == "renamed"

    @pytest.mark.asyncio
    async def test_delete_graph_and_delete_all(self):
        from fastapi import HTTPException

        from src.worker.graphs.graph_operations import (
            create_graph_service,
            delete_all_graphs_service,
            delete_graph_service,
            get_graph_service,
            list_graphs_service,
        )

        mem = InMemoryDB()
        n1, n2 = _worker(id=NODE_ID_1), _worker(id=NODE_ID_2, name="w2", worker_name="a2")
        with patch_get_pg_connection("src.db.graph_state_db", mem):
            c1 = await create_graph_service(
                CREATOR, CREATOR, GraphCreateRequest(name="g1", nodes=[n1])
            )
            await create_graph_service(CREATOR, CREATOR, GraphCreateRequest(name="g2", nodes=[n2]))
            await delete_graph_service(CREATOR, CREATOR, str(c1.id))
            with pytest.raises(HTTPException) as exc:
                await get_graph_service(CREATOR, CREATOR, str(c1.id))
            assert exc.value.status_code == 404
            result = await delete_all_graphs_service(CREATOR, CREATOR)
            assert result["deleted_count"] == 1
            assert len(await list_graphs_service(CREATOR, CREATOR)) == 0

    @pytest.mark.asyncio
    async def test_get_and_delete_nonexistent_raise_404(self):
        from fastapi import HTTPException

        from src.worker.graphs.graph_operations import delete_graph_service, get_graph_service

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.graph_state_db", mem):
            with pytest.raises(HTTPException) as exc:
                await get_graph_service(CREATOR, CREATOR, "00000000-0000-0000-0000-000000000099")
            assert exc.value.status_code == 404
            with pytest.raises(HTTPException) as exc:
                await delete_graph_service(CREATOR, CREATOR, "00000000-0000-0000-0000-000000000099")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        from src.worker.graphs.graph_operations import create_graph_service, list_graphs_service

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await create_graph_service(
                "tenant_a",
                "tenant_a",
                GraphCreateRequest(name="a_graph", nodes=[_worker(id=NODE_ID_1)]),
            )
            assert len(await list_graphs_service("tenant_a", "tenant_a")) == 1
            assert len(await list_graphs_service("tenant_b", "tenant_b")) == 0


# ============================================================================
# PydanticGraphExecutionService
# ============================================================================


class TestPydanticGraphExecutionService:
    def test_init_and_stats(self):
        from src.worker.graphs.pydantic_graph_execution_service import PydanticGraphExecutionService

        svc = PydanticGraphExecutionService(execution_db=None)
        assert svc.active_sessions == {}
        stats = svc.get_stats()
        assert stats["active_sessions"] == 0
        assert "pydantic_graph" in svc.get_available_engines()

    def test_cache_operations(self):
        from src.worker.graphs.pydantic_graph_execution_service import PydanticGraphExecutionService

        svc = PydanticGraphExecutionService(execution_db=None)
        assert svc.get_cache_stats()["cached_graphs"] == 0
        n = _worker(id=NODE_ID_1)
        key1 = svc._get_cache_key([n], [_edge(NODE_ID_1, NODE_ID_2)])
        key2 = svc._get_cache_key([n], [_edge(NODE_ID_1, NODE_ID_2)])
        assert key1 == key2

        svc._graph_cache["old"] = "graph"
        svc._cache_timestamps["old"] = datetime.now(UTC) - timedelta(seconds=600)
        svc._cleanup_expired_cache_entries()
        assert "old" not in svc._graph_cache

        svc._cache_max_size = 2
        for i in range(5):
            svc._graph_cache[f"k{i}"] = f"v{i}"
            svc._cache_timestamps[f"k{i}"] = datetime.now(UTC) - timedelta(seconds=5 - i)
        svc._enforce_cache_size_limit()
        assert len(svc._graph_cache) <= 2

    @pytest.mark.asyncio
    async def test_session_management(self):
        from src.worker.graphs.pydantic_graph_execution_service import PydanticGraphExecutionService

        svc = PydanticGraphExecutionService(execution_db=None)
        # Force DB-less path so the in-memory fallback branch runs.
        svc.session_db = None
        assert await svc.get_session_status("c1", "c1", "nope") is None
        svc.active_sessions["s1"] = MagicMock()
        status = await svc.get_session_status("c1", "c1", "s1")
        assert status is not None
        # No finalized DB row => no evidence the graph finished. It must be
        # reported as running, never as completed-with-no-output.
        assert status["session_id"] == "s1" and status["status"] == "running"
        svc.clear_cache()
        assert len(svc.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_session(self):
        from src.worker.graphs.pydantic_graph_execution_service import PydanticGraphExecutionService

        svc = PydanticGraphExecutionService(execution_db=None)
        svc.active_sessions["s1"] = MagicMock()
        svc._session_timestamps["s1"] = 12345.0
        await svc.cleanup_session("s1")
        assert "s1" not in svc.active_sessions

    def test_build_execution_metadata(self):
        from src.worker.graphs.pydantic_graph_execution_service import PydanticGraphExecutionService

        svc = PydanticGraphExecutionService(execution_db=None)
        s = _state()
        s.start_time = datetime.now(UTC)
        s.end_time = datetime.now(UTC)
        s.set_node_output("n1", "ok")
        s.mark_node_failed("n2")
        meta = svc._build_execution_metadata(s, "s1")
        assert meta["nodes_executed"] == 1 and meta["failed_nodes"] == 1

    def test_backward_compat_alias(self):
        from src.worker.graphs.pydantic_graph_execution_service import (
            GraphExecutionService,
            PydanticGraphExecutionService,
        )

        assert GraphExecutionService is PydanticGraphExecutionService
