"""POST /workers must return 409 (not 500) when register_worker reports a
genuine cross-subject worker_name conflict.

The registry raises :class:`WorkerNameConflictError` when the blind INSERT trips
``unique_tenant_worker_name`` on a row the caller cannot see (another subject's
``visibility='user'`` worker). The endpoint maps that to a 409 ConflictError;
without the mapping the broad ``except Exception`` turned it into a 500.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from src.db.worker_registry_db import WorkerNameConflictError

TENANT = "test-tenant-conflict"
_MP = "src.api.workers.crud_endpoints"


@pytest.fixture
def conflict_deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):
        mock_registry.register_worker = AsyncMock(
            side_effect=WorkerNameConflictError(TENANT, "taken_name")
        )
        mock_cache.remove_worker = AsyncMock()

        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        MockStateDB.return_value = mock_db

        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/conflict.db"
        mock_create.return_value = fake_worker

        yield mock_registry


@pytest.fixture
def client(conflict_deps):
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


def test_post_workers_conflict_returns_409(client):
    resp = client.post(
        "/workers",
        json={
            "worker_name": "taken_name",
            "model_provider": "openai",
            "model": "gpt-4",
        },
    )
    assert resp.status_code == 409, resp.text
    body = resp.json()
    # error_handlers.create_error_response wraps 409s as {"error": {"code": "CONFLICT", ...}}
    assert body["error"]["code"] == "CONFLICT"
    assert "taken_name" in body["error"]["message"]


def test_post_pydantic_ai_worker_conflict_returns_409(client):
    """Convenience endpoints delegate to create_worker_endpoint, so they inherit the mapping."""
    resp = client.post(
        "/workers/pydantic-ai",
        json={
            "worker_name": "taken_name",
            "model": "openai:gpt-4o-mini",
            "system_prompt": "hi",
        },
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "CONFLICT"
