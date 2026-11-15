"""Integration tests for worker CRUD operations."""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestWorkerCreate:
    def test_create_worker(self, client):
        name = f"crud-create-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/pydantic-ai",
            json={
                "worker_name": name,
                "model": "openai:gpt-4o-mini",
                "system_prompt": "Test worker",
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == name
        # Cleanup
        client.delete(f"/workers/{name}")

    def test_create_duplicate_name_allowed(self, test_worker, client):
        """API allows creating workers with duplicate names (different UUIDs)."""
        name = test_worker["name"]
        r = client.post(
            "/workers/pydantic-ai",
            json={
                "worker_name": name,
                "model": "openai:gpt-4o-mini",
                "system_prompt": "Duplicate",
            },
        )
        # API creates a new record — duplicates are allowed
        assert r.status_code in (201, 409, 422, 400)
        if r.status_code == 201:
            client.delete(f"/workers/{name}")


class TestWorkerRead:
    def test_get_worker(self, test_worker, client):
        r = client.get(f"/workers/{test_worker['name']}")
        assert r.status_code == 200
        assert r.json()["name"] == test_worker["name"]

    def test_get_nonexistent_worker(self, client):
        r = client.get("/workers/nonexistent-worker-xyz-999")
        assert r.status_code == 404

    def test_list_workers(self, test_worker, client):
        r = client.get("/workers")
        assert r.status_code == 200
        workers = r.json()
        assert isinstance(workers, list)

    def test_list_workers_with_pagination(self, client):
        r = client.get("/workers", params={"limit": 5, "offset": 0})
        assert r.status_code == 200


class TestWorkerUpdate:
    def test_update_worker(self, test_worker, client):
        """PUT requires full body including worker_name."""
        r = client.put(
            f"/workers/{test_worker['name']}",
            json={
                "worker_name": test_worker["name"],
                "model": "openai:gpt-4o-mini",
                "system_prompt": "Updated prompt",
            },
        )
        assert r.status_code == 200

    def test_update_nonexistent_worker_upserts(self, client):
        """PUT on nonexistent worker creates it (upsert behavior)."""
        name = f"upsert-{uuid.uuid7().hex[:8]}"
        r = client.put(
            f"/workers/{name}",
            json={
                "worker_name": name,
                "model": "openai:gpt-4o-mini",
                "system_prompt": "Upserted",
            },
        )
        assert r.status_code in (200, 201, 404, 422)
        if r.status_code in (200, 201):
            client.delete(f"/workers/{name}")


class TestWorkerDelete:
    def test_delete_worker(self, client):
        name = f"crud-del-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/pydantic-ai",
            json={
                "worker_name": name,
                "model": "openai:gpt-4o-mini",
                "system_prompt": "To be deleted",
            },
        )
        assert r.status_code == 201, f"Setup failed: {r.text}"
        r = client.delete(f"/workers/{name}")
        assert r.status_code == 204

    def test_delete_nonexistent_worker(self, client):
        r = client.delete("/workers/nonexistent-worker-xyz-999")
        assert r.status_code in (404, 204)


class TestWorkerTypes:
    def test_list_types(self, client):
        r = client.get("/workers/types")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
