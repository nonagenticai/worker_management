"""Tests for /workers state management endpoints (src/api/workers/state_endpoints.py).

Covers: GET/DELETE /workers/states, GET/DELETE /workers/{name}/states,
GET /workers/{name}/state/latest, DELETE /workers/{name}/instance.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from src.db.workers_state_db import MAX_WORKER_STATES_LIMIT

TENANT = "test-tenant-state"
WORKER_ID = "wid-state-1234"
_MP = "src.api.workers.state_endpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(resp):
    body = resp.json()
    return body.get("error", {}).get("message", body.get("detail", ""))


SAMPLE_STATE = {
    "id": 1,
    "worker_id": WORKER_ID,
    "worker_name": "test_worker",
    "is_agent": True,
    "state_type": "completed",
    "timestamp": "2025-01-15T10:00:00",
    "metadata": None,
    "worker_state": {"status": "completed", "result": 42},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.state_manager") as mock_state_mgr,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
    ):
        mock_state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))

        mock_db = AsyncMock()
        mock_db.get_worker_states = AsyncMock(return_value=[])
        mock_db.count_worker_states = AsyncMock(return_value=0)
        mock_db.get_worker_states_by_id = AsyncMock(return_value=[])
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.delete_all_worker_states = AsyncMock(return_value=0)
        mock_db.delete_worker_state = AsyncMock(return_value=0)
        MockStateDB.return_value = mock_db

        mock_registry.get_worker = AsyncMock(return_value=None)
        mock_cache.remove_worker = AsyncMock()

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "state_manager": mock_state_mgr,
            "state_db": mock_db,
        }


@pytest.fixture
def client(deps):
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


# ===========================================================================
# GET /workers/states
# ===========================================================================


class TestGetAllWorkerStates:
    def test_returns_states_with_total(self, client, deps):
        deps["state_db"].get_worker_states = AsyncMock(return_value=[SAMPLE_STATE, SAMPLE_STATE])
        deps["state_db"].count_worker_states = AsyncMock(return_value=2)
        resp = client.get("/workers/states")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["states"]) == 2
        assert body["states"][0]["worker_id"] == WORKER_ID
        assert body["truncated"] is False
        assert body["returned"] == 2

    def test_truncation_is_visible_when_scope_exceeds_the_cap(self, client, deps):
        """A capped page must NOT report itself as the complete total.

        Reporting len(states) as ``total`` tells a tenant holding 1500 records
        that there are exactly 1000, and a consumer paging on ``total`` stops
        early and never learns it was truncated.
        """
        true_total = MAX_WORKER_STATES_LIMIT + 500
        capped_page = [SAMPLE_STATE] * MAX_WORKER_STATES_LIMIT
        deps["state_db"].get_worker_states = AsyncMock(return_value=capped_page)
        deps["state_db"].count_worker_states = AsyncMock(return_value=true_total)

        resp = client.get("/workers/states")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["states"]) == MAX_WORKER_STATES_LIMIT
        assert body["returned"] == MAX_WORKER_STATES_LIMIT
        # total is the TRUE count, not the page length
        assert body["total"] == true_total
        assert body["truncated"] is True
        # ...and the count is taken over the same scope as the page read
        deps["state_db"].count_worker_states.assert_awaited_once_with(
            tenant_id=TENANT, subject_id=""
        )

    def test_empty(self, client, deps):
        resp = client.get("/workers/states")
        assert resp.status_code == 200
        body = resp.json()
        assert body["states"] == []
        assert body["total"] == 0
        assert body["truncated"] is False

    def test_db_error(self, client, deps):
        deps["state_db"].get_worker_states = AsyncMock(side_effect=Exception("db down"))
        resp = client.get("/workers/states")
        assert resp.status_code == 500


# ===========================================================================
# DELETE /workers/states
# ===========================================================================


class TestDeleteAllWorkerStates:
    def test_success(self, client, deps):
        deps["state_db"].delete_all_worker_states = AsyncMock(return_value=5)
        resp = client.delete("/workers/states")
        assert resp.status_code == 204

    def test_db_error(self, client, deps):
        deps["state_db"].delete_all_worker_states = AsyncMock(side_effect=Exception("db err"))
        resp = client.delete("/workers/states")
        assert resp.status_code == 500


# ===========================================================================
# GET /workers/{worker_name}/states
# ===========================================================================


class TestGetWorkerStateHistory:
    def test_found_via_worker_id(self, client, deps):
        deps["state_db"].get_worker_states_by_id = AsyncMock(return_value=[SAMPLE_STATE])
        resp = client.get("/workers/test_worker/states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["state_type"] == "completed"

    def test_fallback_to_worker_name(self, client, deps):
        """First lookup by worker_id returns empty, fallback to name."""
        deps["state_db"].get_worker_states_by_id = AsyncMock(side_effect=[[], [SAMPLE_STATE]])
        resp = client.get("/workers/test_worker/states")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_not_found(self, client, deps):
        """Worker not in registry and no states → 404."""
        deps["state_manager"].resolve_worker_id = AsyncMock(
            return_value=("test_worker", False)  # not in registry
        )
        deps["state_db"].get_worker_states_by_id = AsyncMock(side_effect=[[], []])
        deps["state_db"].get_worker_states = AsyncMock(return_value=[])
        resp = client.get("/workers/test_worker/states")
        assert resp.status_code == 404

    def test_reserved_name_rejected(self, client, deps):
        # "null" is a reserved name
        resp = client.get("/workers/null/states")
        assert resp.status_code == 422

    def test_limit_param(self, client, deps):
        deps["state_db"].get_worker_states_by_id = AsyncMock(return_value=[SAMPLE_STATE])
        resp = client.get("/workers/test_worker/states?limit=5")
        assert resp.status_code == 200
        deps["state_db"].get_worker_states_by_id.assert_awaited_once_with(
            tenant_id=TENANT, subject_id="", worker_id=WORKER_ID, limit=5
        )

    def test_db_error(self, client, deps):
        deps["state_db"].get_worker_states_by_id = AsyncMock(side_effect=RuntimeError("db crashed"))
        resp = client.get("/workers/test_worker/states")
        assert resp.status_code == 500


# ===========================================================================
# GET /workers/{worker_name}/state/latest
# ===========================================================================


class TestGetLatestWorkerState:
    def test_found(self, client, deps):
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value=SAMPLE_STATE)
        resp = client.get("/workers/test_worker/state/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state_type"] == "completed"
        assert body["worker_state"]["result"] == 42

    def test_worker_exists_no_state(self, client, deps):
        """Worker in registry but no state records → 404 'No state found'."""
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value=None)
        resp = client.get("/workers/test_worker/state/latest")
        assert resp.status_code == 404
        assert "No state found" in _msg(resp)

    def test_worker_not_found(self, client, deps):
        """Worker not in registry and no states → 404 'Worker not found'."""
        deps["state_manager"].resolve_worker_id = AsyncMock(return_value=("test_worker", False))
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value=None)
        deps["state_db"].get_worker_states = AsyncMock(return_value=[])
        resp = client.get("/workers/test_worker/state/latest")
        assert resp.status_code == 404
        assert "Worker not found" in _msg(resp)

    def test_reserved_names_rejected(self, client, deps):
        for name in ["null", "undefined", "none", "false", "true"]:
            resp = client.get(f"/workers/{name}/state/latest")
            assert resp.status_code == 422, f"'{name}' should be rejected"


# ===========================================================================
# DELETE /workers/{worker_name}/states
# ===========================================================================


class TestDeleteWorkerStates:
    def test_success(self, client, deps):
        deps["registry"].get_worker = AsyncMock(
            return_value={"worker_id": WORKER_ID, "worker_name": "test_worker"}
        )
        resp = client.delete("/workers/test_worker/states")
        assert resp.status_code == 204

    def test_not_found(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        resp = client.delete("/workers/test_worker/states")
        assert resp.status_code == 404

    def test_reserved_name(self, client, deps):
        resp = client.delete("/workers/null/states")
        assert resp.status_code == 422


# ===========================================================================
# DELETE /workers/{worker_name}/instance
# ===========================================================================


class TestCleanupWorkerInstance:
    def test_success(self, client, deps):
        resp = client.delete("/workers/test_worker/instance")
        assert resp.status_code == 200
        assert resp.json()["status"] == "Worker instance cleaned up"
        deps["cache"].remove_worker.assert_awaited_once_with("test_worker")

    def test_invalid_names(self, client, deps):
        for name in ["null", "false", "true", "undefined", "none"]:
            resp = client.delete(f"/workers/{name}/instance")
            assert resp.status_code == 422, f"'{name}' should be rejected"

    def test_numeric_name_valid(self, client, deps):
        resp = client.delete("/workers/0/instance")
        assert resp.status_code == 200

    def test_name_too_long(self, client, deps):
        long_name = "a" * 201
        resp = client.delete(f"/workers/{long_name}/instance")
        assert resp.status_code == 422
