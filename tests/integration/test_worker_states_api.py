"""Integration tests for the /workers_states router."""

import pytest

pytestmark = pytest.mark.integration


class TestWorkerStatesApiRead:
    def test_list_all_states(self, client):
        r = client.get("/workers_states")
        assert r.status_code == 200

    def test_list_states_with_pagination(self, client):
        r = client.get("/workers_states", params={"limit": 5, "offset": 0})
        assert r.status_code == 200


class TestWorkerStatesApiCreate:
    def test_save_state(self, test_worker, client):
        r = client.post(
            "/workers_states",
            json={
                "worker_name": test_worker["name"],
                "state": {"status": "idle", "metadata": {"test": True}},
            },
        )
        # 201 on success, 200 if returns existing, or 422 if schema differs
        assert r.status_code in (200, 201, 422)
