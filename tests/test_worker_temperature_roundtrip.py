"""Roundtrip test: PUT a worker with temperature=0.1, GET it back, assert
temperature is preserved verbatim. Regression test for the 2026-04-30
audit finding: PUTs against /workers/<name> returned ``temperature: null``
in the response body even when the request set ``temperature=0.1``.
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

TENANT = "test-tenant-temp-roundtrip"
WORKER_ID = "wid-temp-roundtrip-1"
_MP = "src.api.workers.crud_endpoints"


def _existing_worker_data(*, worker_name: str, config: dict) -> dict:
    """Mirror what worker_registry.get_worker returns from the DB."""
    return {
        "worker_name": worker_name,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/temp-roundtrip.db",
        "is_agent": True,
        "config": config,
        "status": "active",
    }


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):
        # Captured config from register_worker — the GET path will read it back.
        captured: dict = {"config": None}

        stored: dict = {"config": {}, "is_agent": True, "config_revision": None}

        async def _register_worker(**kwargs):
            captured["config"] = kwargs.get("config")
            # Model register_worker's real return contract, not just the id:
            # the function reports what the row now HOLDS (merged config, and
            # is_agent / config_revision preserved when the caller passes None),
            # and the handler builds its response and its worker_states audit
            # metadata from that. A double returning only {"worker_id": ...}
            # lets a handler regression that echoes the caller's nulls pass.
            stored["config"] = {**stored["config"], **(kwargs.get("config") or {})}
            if kwargs.get("is_agent") is not None:
                stored["is_agent"] = bool(kwargs["is_agent"])
            if kwargs.get("config_revision") is not None:
                stored["config_revision"] = kwargs["config_revision"]
            return {"worker_id": WORKER_ID, **stored}

        mock_registry.register_worker = AsyncMock(side_effect=_register_worker)
        mock_registry.get_worker = AsyncMock(return_value=None)
        mock_registry.list_workers = AsyncMock(return_value=[])

        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        MockStateDB.return_value = mock_db

        mock_cache.remove_worker = AsyncMock()

        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/temp-roundtrip.db"
        mock_create.return_value = fake_worker

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "state_db": mock_db,
            "create_worker": mock_create,
            "captured": captured,
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


def _payload(**overrides):
    base = {
        "worker_name": "test-temp-roundtrip",
        "agent_role": "Roundtrip test worker",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "deepseek",
        "temperature": 0.1,
        "system_prompt": "Echo the input.",
    }
    base.update(overrides)
    return base


def test_temperature_roundtrips_through_put_then_get(client, deps):
    # PUT branch: pretend the worker already exists so we go down the
    # update (not auto-create) path.
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(worker_name="test-temp-roundtrip", config={})
    )

    put_resp = client.put("/workers/test-temp-roundtrip", json=_payload())
    assert put_resp.status_code == 200, put_resp.text
    put_body = put_resp.json()
    assert put_body.get("temperature") == pytest.approx(0.1), (
        f"PUT response dropped temperature: {put_body}"
    )

    # Sanity-check that register_worker actually persisted it into the
    # config dict that the storage layer would write to JSONB.
    captured_config = deps["captured"]["config"]
    assert captured_config is not None, "register_worker was never called"
    assert captured_config.get("temperature") == pytest.approx(0.1), (
        f"register_worker was called with the wrong config: {captured_config}"
    )

    # GET branch: feed the captured config back through get_worker.
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(
            worker_name="test-temp-roundtrip", config=captured_config
        )
    )
    get_resp = client.get("/workers/test-temp-roundtrip")
    assert get_resp.status_code == 200, get_resp.text
    get_body = get_resp.json()
    assert get_body.get("temperature") == pytest.approx(0.1), (
        f"GET response after PUT lost temperature: {get_body}"
    )


def test_temperature_omitted_persists_as_none(client, deps):
    """Negative case: PUT without temperature must round-trip as null,
    not as the auto-create default of 0.1."""
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(worker_name="test-temp-omitted", config={})
    )

    payload = _payload(worker_name="test-temp-omitted")
    payload.pop("temperature")
    put_resp = client.put("/workers/test-temp-omitted", json=payload)
    assert put_resp.status_code == 200, put_resp.text

    captured_config = deps["captured"]["config"]
    assert captured_config is not None
    assert captured_config.get("temperature") is None, captured_config

    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(worker_name="test-temp-omitted", config=captured_config)
    )
    get_body = client.get("/workers/test-temp-omitted").json()
    assert get_body.get("temperature") is None, get_body


def test_temperature_roundtrips_through_post_create(client, deps):
    """POST /workers must echo temperature in its response, mirroring the
    PUT fix shipped in 3787de3. crud_endpoints.py:186 hand-constructs the
    WorkerResponse the same way PUT did and was missing the temperature kwarg."""
    deps["registry"].get_worker = AsyncMock(return_value=None)

    post_resp = client.post("/workers", json=_payload(worker_name="test-temp-post"))
    assert post_resp.status_code in (200, 201), post_resp.text
    body = post_resp.json()
    assert body.get("temperature") == pytest.approx(0.1), (
        f"POST response dropped temperature: {body}"
    )
