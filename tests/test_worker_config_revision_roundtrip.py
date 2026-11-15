"""Roundtrip test: PUT a worker with config_revision="rX", GET it back,
assert the value is preserved verbatim. Closes the gap discovered 2026-05-04
where the seeder's FR_AGENT_CONFIG_REVISION constant was sent but silently
dropped by WM (no schema field, no column, no persistence) — making the
"lockstep TS/Python constants" rule unenforceable in prod.

Mirrors the structure of test_worker_temperature_roundtrip.py.
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

TENANT = "test-tenant-config-rev-roundtrip"
WORKER_ID = "wid-config-rev-roundtrip-1"
_MP = "src.api.workers.crud_endpoints"


def _existing_worker_data(
    *, worker_name: str, config: dict, config_revision: str | None = None
) -> dict:
    """Mirror what worker_registry.get_worker / _row_to_dict returns from the DB."""
    return {
        "worker_name": worker_name,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/config-rev-roundtrip.db",
        "is_agent": True,
        "config": config,
        "status": "active",
        "config_revision": config_revision,
    }


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):
        # Capture both `config` and `config_revision` from register_worker so
        # tests can assert what hit the DB layer separately from the response.
        captured: dict = {"config": None, "config_revision": "<unset>"}

        stored: dict = {"config": {}, "is_agent": True, "config_revision": None}

        async def _register_worker(**kwargs):
            captured["config"] = kwargs.get("config")
            captured["config_revision"] = kwargs.get("config_revision", "<missing>")
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
        fake_worker.db_path = "/tmp/config-rev-roundtrip.db"
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
        "worker_name": "test-config-rev-roundtrip",
        "agent_role": "Roundtrip test worker",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "deepseek",
        "temperature": 0.1,
        "system_prompt": "Echo the input.",
        "config_revision": "2026-05-04-test-r1",
    }
    base.update(overrides)
    return base


def test_config_revision_roundtrips_through_put_then_get(client, deps):
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(worker_name="test-config-rev-roundtrip", config={})
    )

    put_resp = client.put("/workers/test-config-rev-roundtrip", json=_payload())
    assert put_resp.status_code == 200, put_resp.text
    put_body = put_resp.json()
    assert put_body.get("config_revision") == "2026-05-04-test-r1", (
        f"PUT response dropped config_revision: {put_body}"
    )

    # register_worker must have received config_revision as a top-level
    # kwarg, NOT buried inside config — that's the whole point of the fix.
    assert deps["captured"]["config_revision"] == "2026-05-04-test-r1", (
        f"register_worker did not receive config_revision kwarg: {deps['captured']}"
    )

    # GET branch: feed the persisted revision back through get_worker.
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(
            worker_name="test-config-rev-roundtrip",
            config=deps["captured"]["config"],
            config_revision="2026-05-04-test-r1",
        )
    )
    get_resp = client.get("/workers/test-config-rev-roundtrip")
    assert get_resp.status_code == 200, get_resp.text
    get_body = get_resp.json()
    assert get_body.get("config_revision") == "2026-05-04-test-r1", (
        f"GET response after PUT lost config_revision: {get_body}"
    )


def test_config_revision_bumps_on_update(client, deps):
    """A second PUT with a newer revision must overwrite, not merge."""
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(
            worker_name="test-config-rev-bump",
            config={},
            config_revision="2026-05-04-test-r1",
        )
    )
    put_resp = client.put(
        "/workers/test-config-rev-bump",
        json=_payload(worker_name="test-config-rev-bump", config_revision="2026-05-04-test-r2"),
    )
    assert put_resp.status_code == 200, put_resp.text
    assert put_resp.json().get("config_revision") == "2026-05-04-test-r2"
    assert deps["captured"]["config_revision"] == "2026-05-04-test-r2"


def test_config_revision_omitted_persists_as_none(client, deps):
    """Negative case: PUT without config_revision must round-trip as null."""
    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(worker_name="test-config-rev-omitted", config={})
    )

    payload = _payload(worker_name="test-config-rev-omitted")
    payload.pop("config_revision")
    put_resp = client.put("/workers/test-config-rev-omitted", json=payload)
    assert put_resp.status_code == 200, put_resp.text

    assert deps["captured"]["config_revision"] is None, deps["captured"]

    deps["registry"].get_worker = AsyncMock(
        return_value=_existing_worker_data(
            worker_name="test-config-rev-omitted",
            config=deps["captured"]["config"],
            config_revision=None,
        )
    )
    get_body = client.get("/workers/test-config-rev-omitted").json()
    assert get_body.get("config_revision") is None, get_body


def test_config_revision_roundtrips_through_post_create(client, deps):
    """POST /workers must echo config_revision in its response."""
    deps["registry"].get_worker = AsyncMock(return_value=None)

    post_resp = client.post("/workers", json=_payload(worker_name="test-config-rev-post"))
    assert post_resp.status_code in (200, 201), post_resp.text
    body = post_resp.json()
    assert body.get("config_revision") == "2026-05-04-test-r1", (
        f"POST response dropped config_revision: {body}"
    )
    assert deps["captured"]["config_revision"] == "2026-05-04-test-r1"
