"""Integration tests for per-worker state endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestWorkerStatesRead:
    def test_get_worker_state_history(self, test_worker, client):
        r = client.get(f"/workers/{test_worker['name']}/states")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_worker_state_history_with_limit(self, test_worker, client):
        r = client.get(f"/workers/{test_worker['name']}/states", params={"limit": 5})
        assert r.status_code == 200

    def test_get_latest_worker_state(self, test_worker, client):
        r = client.get(f"/workers/{test_worker['name']}/state/latest")
        # May be 200 (has state) or 404 (no state yet)
        assert r.status_code in (200, 404)

    def test_get_states_nonexistent_worker(self, client):
        r = client.get("/workers/nonexistent-xyz-999/states")
        assert r.status_code in (200, 404)


class TestWorkerStatesDelete:
    def test_delete_worker_states(self, test_worker, client):
        r = client.delete(f"/workers/{test_worker['name']}/states")
        assert r.status_code in (200, 204)


class TestWorkerInstanceCleanup:
    def test_cleanup_worker_instance(self, test_worker, client):
        r = client.delete(f"/workers/{test_worker['name']}/instance")
        assert r.status_code in (200, 204)
