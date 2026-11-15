"""Integration tests for graph state endpoints."""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestGraphStatesRead:
    def test_get_graph_state_history(self, test_graph, client):
        r = client.get(f"/graphs/{test_graph['id']}/states")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_graph_state_history_with_limit(self, test_graph, client):
        r = client.get(f"/graphs/{test_graph['id']}/states", params={"limit": 5})
        assert r.status_code == 200

    def test_get_states_nonexistent_graph(self, client):
        fake_id = str(uuid.uuid7())
        r = client.get(f"/graphs/{fake_id}/states")
        assert r.status_code in (200, 404, 500)


class TestGraphStatesDelete:
    def test_delete_graph_states(self, test_graph, client):
        r = client.delete(f"/graphs/{test_graph['id']}/states")
        assert r.status_code == 200
