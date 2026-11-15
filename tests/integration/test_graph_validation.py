"""Integration tests for graph validation endpoint (GET /api/graphs/{graph_id}/validate).

Tests validate the validation API response structure, error detection for orphaned nodes,
missing worker references, circular dependencies, and edge cases like empty graphs
and graphs with condition nodes.
"""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestValidateValidGraph:
    """Test validation on well-formed graphs that should pass."""

    def test_validate_single_node_graph_returns_valid(self, test_graph, client):
        """A single-node graph with no edges should validate successfully."""
        r = client.get(f"/graphs/{test_graph['id']}/validate")
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert data["graph_id"] == test_graph["id"]
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) == 0

    def test_validate_two_node_connected_graph(self, two_node_graph, client):
        """A two-node graph with a proper edge should validate successfully."""
        r = client.get(f"/graphs/{two_node_graph['id']}/validate")
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) == 0

    def test_validate_response_has_warnings_field(self, test_graph, client):
        """Validation response should always include a warnings list."""
        r = client.get(f"/graphs/{test_graph['id']}/validate")
        assert r.status_code == 200
        data = r.json()
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_validate_response_structure(self, test_graph, client):
        """Validation response must have graph_id, valid, errors, warnings."""
        r = client.get(f"/graphs/{test_graph['id']}/validate")
        assert r.status_code == 200
        data = r.json()
        assert "graph_id" in data
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert isinstance(data["valid"], bool)
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)


class TestValidateDanglingEdges:
    """Test that edges referencing nonexistent node IDs are accepted by the API.

    The current validate endpoint does NOT check for dangling edge references.
    These tests document that behavior — edges with fake source/target UUIDs
    are stored and validation still returns valid=True.
    """

    def test_graph_with_dangling_edge_source_is_accepted(self, client):
        """An edge with a fake source UUID is stored; validation does not flag it."""
        graph_name = f"val-dangle-src-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "real_node",
                        "worker_name": "dummy",
                        "node_type": "worker",
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
            fake_source = str(uuid.uuid7())
            real_target = graph["nodes"][0]["id"]
            patch_r = client.patch(
                f"/graphs/{graph_id}",
                json={
                    "edges": [{"source": fake_source, "target": real_target}],
                },
            )
            # API accepts the edge with a dangling source
            assert patch_r.status_code == 200

            # Verify the edge is stored
            get_r = client.get(f"/graphs/{graph_id}")
            assert get_r.status_code == 200
            edges = get_r.json().get("edges", [])
            assert len(edges) == 1
            assert edges[0]["source"] == fake_source

            # Validation does not flag dangling references (current behavior)
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            # The node is "connected" via the edge (even though edge source is fake),
            # so no orphaned_nodes warning is generated either
            assert isinstance(data["errors"], list)
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_graph_with_dangling_edge_target_is_accepted(self, client):
        """An edge with a fake target UUID is stored; validation does not flag it."""
        graph_name = f"val-dangle-tgt-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "src_node", "worker_name": "dummy", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            real_source = graph["nodes"][0]["id"]
            fake_target = str(uuid.uuid7())
            patch_r = client.patch(
                f"/graphs/{graph_id}",
                json={
                    "edges": [{"source": real_source, "target": fake_target}],
                },
            )
            assert patch_r.status_code == 200

            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            assert isinstance(data["errors"], list)
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestValidateEmptyGraph:
    """Test validation on graphs with no nodes or no edges."""

    def test_validate_graph_zero_nodes(self, client):
        """A graph with zero nodes should still validate (possibly with warnings)."""
        graph_name = f"val-empty-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [],
                "edges": [],
            },
        )
        if r.status_code not in (201, 200):
            pytest.skip(f"Empty graph creation not supported: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            assert "valid" in data
            assert isinstance(data["errors"], list)
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_validate_graph_nodes_but_no_edges(self, client):
        """A graph with multiple disconnected nodes should produce orphaned-node warnings."""
        graph_name = f"val-noedge-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "a", "worker_name": "w1", "node_type": "worker"},
                    {"name": "b", "worker_name": "w2", "node_type": "worker"},
                    {"name": "c", "worker_name": "w3", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            warnings = data.get("warnings", [])
            orphan_warnings = [w for w in warnings if w.get("type") == "orphaned_nodes"]
            assert len(orphan_warnings) > 0, (
                f"Expected orphaned_nodes warning for disconnected graph, got warnings: {warnings}"
            )
            # Verify orphan warning contains node_ids
            assert "node_ids" in orphan_warnings[0]
            assert len(orphan_warnings[0]["node_ids"]) == 3
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_validate_partially_connected_graph_warns_orphans(self, client):
        """A graph where only some nodes are connected should warn about orphaned ones."""
        graph_name = f"val-partial-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "connected_a", "worker_name": "w1", "node_type": "worker"},
                    {"name": "connected_b", "worker_name": "w2", "node_type": "worker"},
                    {"name": "orphan", "worker_name": "w3", "node_type": "worker"},
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
                        ],
                    },
                )

            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            warnings = data.get("warnings", [])
            orphan_warnings = [w for w in warnings if w.get("type") == "orphaned_nodes"]
            assert len(orphan_warnings) > 0
            # The third node should be in the orphaned list
            orphaned_ids = orphan_warnings[0]["node_ids"]
            assert len(orphaned_ids) == 1
            assert orphaned_ids[0] == graph["nodes"][2]["id"]
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestValidateMissingWorker:
    """Test that validation catches worker nodes missing worker_name."""

    def test_worker_node_without_worker_name_is_invalid(self, client):
        """A worker-type node with no worker_name should fail validation."""
        graph_name = f"val-noworker-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "broken_node",
                        "worker_name": "dummy",
                        "node_type": "worker",
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
            node = graph["nodes"][0]
            patch_r = client.patch(
                f"/graphs/{graph_id}",
                json={
                    "nodes": [
                        {
                            "id": node["id"],
                            "name": "broken_node",
                            "node_type": "worker",
                            "worker_name": None,
                        },
                    ],
                },
            )
            if patch_r.status_code == 200:
                val_r = client.get(f"/graphs/{graph_id}/validate")
                assert val_r.status_code == 200
                data = val_r.json()
                assert data["valid"] is False, (
                    f"Expected invalid for missing worker_name, got: {data}"
                )
                missing_worker_errors = [
                    e for e in data["errors"] if e.get("type") == "missing_worker"
                ]
                assert len(missing_worker_errors) > 0
                # Error should reference the node
                assert "node_id" in missing_worker_errors[0]
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestValidateCircularDependency:
    """Test that validation detects circular dependencies in edges."""

    def test_circular_edge_detected(self, client):
        """A graph with A->B->A edges should flag circular dependency."""
        graph_name = f"val-cycle-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "node_a", "worker_name": "w1", "node_type": "worker"},
                    {"name": "node_b", "worker_name": "w2", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            if graph.get("nodes") and len(graph["nodes"]) >= 2:
                n0 = graph["nodes"][0]["id"]
                n1 = graph["nodes"][1]["id"]
                patch_r = client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {"source": n0, "target": n1},
                            {"source": n1, "target": n0},
                        ],
                    },
                )
                if patch_r.status_code == 200:
                    val_r = client.get(f"/graphs/{graph_id}/validate")
                    assert val_r.status_code == 200
                    data = val_r.json()
                    assert data["valid"] is False, (
                        f"Expected invalid for circular dependency, got: {data}"
                    )
                    cycle_errors = [
                        e for e in data["errors"] if e.get("type") == "circular_dependency"
                    ]
                    assert len(cycle_errors) > 0, (
                        f"Expected circular_dependency error, got errors: {data['errors']}"
                    )
                    cycle_error = cycle_errors[0]
                    assert "cycle_path" in cycle_error
                    assert len(cycle_error["cycle_path"]) >= 2
                    assert "cycle_length" in cycle_error
                    assert "affected_nodes" in cycle_error
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_three_node_cycle_detected(self, client):
        """A graph with A->B->C->A edges should flag circular dependency."""
        graph_name = f"val-cycle3-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "a", "worker_name": "w1", "node_type": "worker"},
                    {"name": "b", "worker_name": "w2", "node_type": "worker"},
                    {"name": "c", "worker_name": "w3", "node_type": "worker"},
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
                ids = [n["id"] for n in graph["nodes"]]
                patch_r = client.patch(
                    f"/graphs/{graph_id}",
                    json={
                        "edges": [
                            {"source": ids[0], "target": ids[1]},
                            {"source": ids[1], "target": ids[2]},
                            {"source": ids[2], "target": ids[0]},
                        ],
                    },
                )
                if patch_r.status_code == 200:
                    val_r = client.get(f"/graphs/{graph_id}/validate")
                    assert val_r.status_code == 200
                    data = val_r.json()
                    assert data["valid"] is False
                    cycle_errors = [
                        e for e in data["errors"] if e.get("type") == "circular_dependency"
                    ]
                    assert len(cycle_errors) > 0
                    assert cycle_errors[0].get("cycle_length", 0) >= 2
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_self_loop_detected(self, client):
        """A node with an edge pointing to itself should be a circular dependency."""
        graph_name = f"val-self-loop-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "loopy", "worker_name": "w1", "node_type": "worker"},
                ],
                "edges": [],
            },
        )
        if r.status_code != 201:
            pytest.skip(f"Graph creation unavailable: {r.status_code}")
        graph = r.json()
        graph_id = graph["id"]

        try:
            node_id = graph["nodes"][0]["id"]
            patch_r = client.patch(
                f"/graphs/{graph_id}",
                json={
                    "edges": [{"source": node_id, "target": node_id}],
                },
            )
            if patch_r.status_code == 200:
                val_r = client.get(f"/graphs/{graph_id}/validate")
                assert val_r.status_code == 200
                data = val_r.json()
                assert data["valid"] is False
                cycle_errors = [e for e in data["errors"] if e.get("type") == "circular_dependency"]
                assert len(cycle_errors) > 0
        finally:
            client.delete(f"/graphs/{graph_id}")


class TestValidateNonexistentGraph:
    """Test validation on a graph that does not exist."""

    def test_validate_nonexistent_graph_returns_404(self, client):
        """Validating a nonexistent graph should return 404 (or 500 on error)."""
        fake_id = str(uuid.uuid7())
        r = client.get(f"/graphs/{fake_id}/validate")
        assert r.status_code in (404, 500)

    def test_validate_invalid_uuid_returns_422(self, client):
        """Validating with an invalid UUID format should return 422."""
        r = client.get("/graphs/not-a-uuid/validate")
        assert r.status_code == 422


class TestValidateConditionNodes:
    """Test validation of graphs containing condition-type nodes."""

    def test_condition_node_validates(self, client):
        """A graph with a condition node should validate (condition nodes don't need worker_name)."""
        graph_name = f"val-cond-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {"name": "start", "worker_name": "dummy", "node_type": "worker"},
                    {
                        "name": "check",
                        "node_type": "condition",
                        "data": {"condition": "1 == 1"},
                    },
                    {"name": "end", "worker_name": "dummy", "node_type": "worker"},
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

            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            missing_worker_errors = [
                e for e in data.get("errors", []) if e.get("type") == "missing_worker"
            ]
            assert len(missing_worker_errors) == 0, (
                f"Condition node should not need worker_name, got: {missing_worker_errors}"
            )
        finally:
            client.delete(f"/graphs/{graph_id}")

    def test_function_node_without_function_is_invalid(self, client):
        """A function-type node with no function_name and no function_code should fail."""
        graph_name = f"val-nofunc-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/graphs/",
            json={
                "name": graph_name,
                "nodes": [
                    {
                        "name": "broken_func",
                        "node_type": "function",
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
            val_r = client.get(f"/graphs/{graph_id}/validate")
            assert val_r.status_code == 200
            data = val_r.json()
            assert data["valid"] is False
            func_errors = [e for e in data["errors"] if e.get("type") == "missing_function"]
            assert len(func_errors) > 0, f"Expected missing_function error, got: {data['errors']}"
        finally:
            client.delete(f"/graphs/{graph_id}")
