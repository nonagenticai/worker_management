"""Integration tests for graph structure and validation."""

import pytest

pytestmark = pytest.mark.integration


class TestGraphStructure:
    def test_get_structure_single_node(self, test_graph, client):
        r = client.get(f"/graphs/{test_graph['id']}/structure")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data or "structure" in data or isinstance(data, dict)

    def test_get_structure_two_nodes(self, two_node_graph, client):
        r = client.get(f"/graphs/{two_node_graph['id']}/structure")
        assert r.status_code == 200


class TestGraphValidation:
    def test_validate_graph(self, test_graph, client):
        r = client.get(f"/graphs/{test_graph['id']}/validate")
        assert r.status_code == 200

    def test_validate_two_node_graph(self, two_node_graph, client):
        r = client.get(f"/graphs/{two_node_graph['id']}/validate")
        assert r.status_code == 200
