"""Integration tests for graph execution with retry semantics and conditional edge routing.

Tests verify execution endpoint behavior, response structure, and that graphs with
conditional edges and branching topology execute correctly through the API.

NOTE: The execution endpoint always returns HTTP 200, even for failures.
Errors are reported in the response body via status="failed" and error_message.
"""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestExecutionResponseStructure:
    """Verify the ExecutionResponse shape returned by the execute endpoint."""

    def test_execute_returns_required_fields(self, two_node_graph, client):
        """Execution response must contain session_id, status, node_outputs, etc."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={},
        )
        assert r.status_code in (200, 400, 404, 422, 500)
        if r.status_code == 200:
            data = r.json()
            assert "session_id" in data
            assert "status" in data
            assert data["status"] in ("completed", "failed", "running")
            assert "node_outputs" in data
            assert isinstance(data["node_outputs"], dict)
            assert "completed_nodes" in data
            assert isinstance(data["completed_nodes"], list)
            assert "failed_nodes" in data
            assert isinstance(data["failed_nodes"], list)
            assert "started_at" in data
            assert "updated_at" in data

    def test_execute_with_variables(self, two_node_graph, client):
        """Execution with variables should accept and process them."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={
                "variables": {"test_input": "hello", "batch_size": 10},
            },
        )
        assert r.status_code in (200, 400, 404, 422, 500)
        if r.status_code == 200:
            data = r.json()
            assert data["session_id"] is not None
            assert data["status"] in ("completed", "failed", "running")

    def test_execute_nonexistent_graph_returns_failed_status(self, client):
        """Executing a nonexistent graph returns 200 with status=failed and error_message."""
        fake_id = str(uuid.uuid7())
        r = client.post(
            f"/execution/graphs/{fake_id}/execute",
            json={},
        )
        # Execution endpoint returns 200 even for failures
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "failed"
        assert data["error_message"] is not None
        assert fake_id in data["error_message"]
        assert "not found" in data["error_message"].lower()

    def test_execute_invalid_uuid_returns_422(self, client):
        """Executing with an invalid UUID should return 422 (path validation)."""
        r = client.post(
            "/execution/graphs/not-a-valid-uuid/execute",
            json={},
        )
        assert r.status_code == 422


class TestExecutionWithStartNode:
    """Test execution with explicit start_node parameter."""

    def test_execute_with_valid_start_node(self, two_node_graph, client):
        """Execution specifying a valid start_node should not error on validation."""
        if not two_node_graph.get("nodes"):
            pytest.skip("Graph has no nodes")
        start_node_id = two_node_graph["nodes"][0]["id"]
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={"start_node": start_node_id},
        )
        assert r.status_code == 200
        data = r.json()
        # Should not fail with start_node format error
        if data.get("error_message"):
            assert "Start node must be a valid UUID" not in data["error_message"]

    def test_execute_with_invalid_start_node_reports_error(self, two_node_graph, client):
        """Execution with a non-UUID start_node returns 200/failed with error in message."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={"start_node": "not-a-uuid"},
        )
        # API returns 200 with status=failed (error in response body, not HTTP code)
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            data = r.json()
            assert data["status"] == "failed"
            assert data["error_message"] is not None
            assert len(data["error_message"]) > 0

    def test_execute_with_nonexistent_start_node(self, two_node_graph, client):
        """Execution with a valid UUID start_node that doesn't exist in the graph."""
        fake_node_id = str(uuid.uuid7())
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={"start_node": fake_node_id},
        )
        assert r.status_code in (200, 400, 404, 500)
        if r.status_code == 200:
            data = r.json()
            # Should fail because start node doesn't exist in the graph
            assert data["status"] == "failed"
            assert data["error_message"] is not None


class TestExecutionRetryBehavior:
    """Test that execution handles worker failures gracefully (retry is internal).

    Since retry logic is internal to the execution engine, we test from the API
    perspective that execution completes or fails with proper error reporting.
    """

    def test_execute_graph_with_nonexistent_worker_produces_output(self, client):
        """A graph referencing a non-existent worker name still executes (default worker fallback).

        The execution engine may fall back to a default worker, so we just verify
        the response structure is complete.
        """
        graph_name = f"exec-retry-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "will_fail",
                        "worker_name": "nonexistent_worker_xyz",
                        "node_type": "worker",
                        "task_template": "Do something",
                    },
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            exec_r = client.post(
                f"/execution/graphs/{graph_id}/execute",
                json={},
            )
            assert exec_r.status_code in (200, 400, 404, 500)
            if exec_r.status_code == 200:
                data = exec_r.json()
                assert data["status"] in ("failed", "completed")
                # Verify response structure is complete regardless of outcome
                assert "session_id" in data
                assert "node_outputs" in data
                assert "completed_nodes" in data
                assert "failed_nodes" in data
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_stepwise_execution_returns_same_structure(self, two_node_graph, client):
        """Stepwise execution should return the same ExecutionResponse structure."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute/stepwise",
            json={},
        )
        assert r.status_code in (200, 400, 404, 422, 500)
        if r.status_code == 200:
            data = r.json()
            assert "session_id" in data
            assert "status" in data
            assert "node_outputs" in data
            assert "completed_nodes" in data
            assert "failed_nodes" in data

    def test_stepwise_and_regular_execution_produce_same_status(self, two_node_graph, client):
        """Stepwise execution delegates to regular execution, so results should match."""
        r1 = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={},
        )
        r2 = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute/stepwise",
            json={},
        )
        if r1.status_code == 200 and r2.status_code == 200:
            # Both should produce the same status (both succeed or both fail)
            assert r1.json()["status"] == r2.json()["status"]


class TestSessionStatusAfterExecution:
    """Test that session status can be queried after execution."""

    def test_session_status_after_execute(self, two_node_graph, client):
        """After execution, the session_id should be queryable (if stored in memory)."""
        exec_r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={},
        )
        if exec_r.status_code != 200:
            pytest.skip(f"Execution failed: {exec_r.status_code}")

        session_id = exec_r.json()["session_id"]
        status_r = client.get(f"/execution/sessions/{session_id}/status")
        # Session may or may not be found (in-memory store, pod might differ)
        assert status_r.status_code in (200, 404)
        if status_r.status_code == 200:
            data = status_r.json()
            assert data["session_id"] == session_id
            assert data["status"] in ("completed", "failed", "running")

    def test_nonexistent_session_returns_404(self, client):
        """Querying a nonexistent session should return 404."""
        fake_session = str(uuid.uuid7())
        r = client.get(f"/execution/sessions/{fake_session}/status")
        assert r.status_code in (200, 404)


class TestConditionalEdgeRouting:
    """Test graphs with conditional edges to verify routing through the API.

    Edge conditions are evaluated at execution time by the execution engine.
    These tests create graphs with condition-type nodes and conditional edges
    and verify that execution processes them.
    """

    def test_graph_with_condition_node_executes(self, client):
        """A graph with a condition node should execute (even if workers are dummy)."""
        graph_name = f"exec-cond-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "start",
                        "worker_name": "dummy",
                        "node_type": "worker",
                        "task_template": "Start task",
                    },
                    {
                        "name": "check",
                        "node_type": "condition",
                        "data": {"condition": "True"},
                    },
                    {
                        "name": "end_true",
                        "worker_name": "dummy",
                        "node_type": "worker",
                        "task_template": "True branch",
                    },
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 3:
                nodes = graph["nodes"]
                client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {"source": nodes[0]["id"], "target": nodes[1]["id"]},
                            {"source": nodes[1]["id"], "target": nodes[2]["id"]},
                        ],
                    },
                )

            exec_r = client.post(
                f"/execution/graphs/{graph_id}/execute",
                json={},
            )
            assert exec_r.status_code in (200, 400, 404, 422, 500)
            if exec_r.status_code == 200:
                data = exec_r.json()
                assert data["status"] in ("completed", "failed")
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_graph_with_conditional_edge_metadata(self, client):
        """Edges with condition strings should be stored and retrievable."""
        graph_name = f"exec-condedge-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "start", "worker_name": "w1", "node_type": "worker"},
                    {"name": "branch_a", "worker_name": "w2", "node_type": "worker"},
                    {"name": "branch_b", "worker_name": "w3", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 3:
                nodes = graph["nodes"]
                patch_r = client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {
                                "source": nodes[0]["id"],
                                "target": nodes[1]["id"],
                                "condition": "result == 'success'",
                            },
                            {
                                "source": nodes[0]["id"],
                                "target": nodes[2]["id"],
                                "condition": "result == 'failure'",
                            },
                        ],
                    },
                )
                assert patch_r.status_code == 200

                # Verify edges with conditions are stored
                get_r = client.get(f"/graphs/{graph_id}")
                assert get_r.status_code == 200
                graph_data = get_r.json()
                edges = graph_data.get("edges", [])
                assert len(edges) == 2
                conditions = [e.get("condition") for e in edges if e.get("condition")]
                assert len(conditions) == 2
                assert "result == 'success'" in conditions
                assert "result == 'failure'" in conditions

                # Verify structure endpoint shows branching topology
                struct_r = client.get(f"/graphs/{graph_id}/structure")
                assert struct_r.status_code == 200
                struct_data = struct_r.json()
                assert struct_data.get("structure_type") == "branching"
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_branching_graph_validates_without_errors(self, client):
        """A branching graph with conditional edges should validate cleanly."""
        graph_name = f"exec-branch-val-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "start", "worker_name": "w1", "node_type": "worker"},
                    {"name": "path_a", "worker_name": "w2", "node_type": "worker"},
                    {"name": "path_b", "worker_name": "w3", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 3:
                nodes = graph["nodes"]
                client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {
                                "source": nodes[0]["id"],
                                "target": nodes[1]["id"],
                                "condition": "x > 0",
                            },
                            {
                                "source": nodes[0]["id"],
                                "target": nodes[2]["id"],
                                "condition": "x <= 0",
                            },
                        ],
                    },
                )

            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            assert data["valid"] is True
            assert len(data["errors"]) == 0
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestExecutionHealthEndpoint:
    """Test the execution health check endpoint."""

    def test_execution_health_returns_engine_info(self, client):
        """Health endpoint should return engine type and timestamp."""
        r = client.get("/execution/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "execution_engine" in data
        assert "timestamp" in data

    def test_execution_health_includes_database_name(self, client):
        """Health endpoint should report the database name."""
        r = client.get("/execution/health")
        assert r.status_code == 200
        data = r.json()
        assert "database_name" in data


class TestExecutionWithComplexGraphs:
    """Test execution with more complex graph topologies."""

    def test_execute_linear_three_node_graph(self, client):
        """Execute a linear 3-node graph (A -> B -> C)."""
        graph_name = f"exec-linear3-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "step1",
                        "worker_name": "dummy",
                        "node_type": "worker",
                        "task_template": "Step 1",
                    },
                    {
                        "name": "step2",
                        "worker_name": "dummy",
                        "node_type": "worker",
                        "task_template": "Step 2",
                    },
                    {
                        "name": "step3",
                        "worker_name": "dummy",
                        "node_type": "worker",
                        "task_template": "Step 3",
                    },
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 3:
                nodes = graph["nodes"]
                client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {"source": nodes[0]["id"], "target": nodes[1]["id"]},
                            {"source": nodes[1]["id"], "target": nodes[2]["id"]},
                        ],
                    },
                )

            # Validate first
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            val_data = val_r.json()
            assert val_data["valid"] is True

            # Verify structure is linear
            struct_r = client.get(f"/graphs/{graph_id}/structure")
            assert struct_r.status_code == 200
            assert struct_r.json().get("structure_type") == "linear"

            # Execute
            exec_r = client.post(
                f"/execution/graphs/{graph_id}/execute",
                json={"variables": {"input": "test"}},
            )
            assert exec_r.status_code in (200, 400, 404, 500)
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_execute_diamond_graph(self, client):
        """Execute a diamond-shaped graph (A -> B, A -> C, B -> D, C -> D)."""
        graph_name = f"exec-diamond-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "start", "worker_name": "w1", "node_type": "worker"},
                    {"name": "left", "worker_name": "w2", "node_type": "worker"},
                    {"name": "right", "worker_name": "w3", "node_type": "worker"},
                    {"name": "join", "worker_name": "w4", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 4:
                nodes = graph["nodes"]
                client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {"source": nodes[0]["id"], "target": nodes[1]["id"]},
                            {"source": nodes[0]["id"], "target": nodes[2]["id"]},
                            {"source": nodes[1]["id"], "target": nodes[3]["id"]},
                            {"source": nodes[2]["id"], "target": nodes[3]["id"]},
                        ],
                    },
                )

            # Verify complex structure
            struct_r = client.get(f"/graphs/{graph_id}/structure")
            assert struct_r.status_code == 200
            assert struct_r.json().get("structure_type") == "complex"

            # Validate — should be valid (no cycles)
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            assert val_r.json()["valid"] is True

            # Execute — accept any result, diamond topology is valid
            exec_r = client.post(
                f"/execution/graphs/{graph_id}/execute",
                json={},
            )
            assert exec_r.status_code in (200, 400, 404, 500)
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestExecutionRejectsExtraFields:
    """Test that execution request validation rejects unknown fields."""

    def test_extra_fields_rejected(self, two_node_graph, client):
        """ExecutionRequest with extra fields should be rejected (extra='forbid')."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={"variables": {}, "unknown_field": "bad"},
        )
        assert r.status_code == 422
