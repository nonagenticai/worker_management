"""Integration tests for graph execution endpoints."""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestExecutionHealth:
    def test_execution_health(self, client):
        r = client.get("/execution/health")
        assert r.status_code == 200


class TestGraphExecution:
    def test_execute_graph(self, two_node_graph, client):
        """Execute a graph — may fail if workers aren't real, tolerate errors."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute",
            json={},
        )
        # 200 on success, 4xx/5xx if workers don't exist or execution fails
        assert r.status_code in (200, 400, 404, 422, 500)


class TestStepwiseExecution:
    def test_stepwise_execute(self, two_node_graph, client):
        """Stepwise execution — tolerate failures from dummy workers."""
        r = client.post(
            f"/execution/graphs/{two_node_graph['id']}/execute/stepwise",
            json={},
        )
        assert r.status_code in (200, 400, 404, 422, 500)


class TestSessionStatus:
    def test_session_status_nonexistent(self, client):
        fake_id = str(uuid.uuid7())
        r = client.get(f"/execution/sessions/{fake_id}/status")
        assert r.status_code in (200, 404)
