"""Regression tests for temperature roundtrip on /workers endpoints.

Background
----------
A live trace against the deployed service showed that PUT /workers/{name}
returned ``temperature: null`` in its immediate response body even when the
request body and the persisted JSONB row both held the supplied value
(e.g. 0.1). The GET handler at ``crud_endpoints.py:421`` already echoed
``config.get("temperature")`` correctly, so the bug was strictly in the
PUT response construction (``crud_endpoints.py:723`` block) which forgot
to pass ``temperature=request.temperature`` to ``WorkerResponse``.

Commit ``3787de3 fix(workers): include temperature in PUT /workers/{name}
response`` added that single keyword. These tests pin the contract so the
field cannot silently fall off the response again — and exercise the same
contract for POST /workers (the non-update create path) and for GET, so
the full PUT-then-GET roundtrip a caller observes is locked down.

Test pattern follows ``tests/test_crud_endpoints.py`` (FastAPI TestClient
with ``worker_registry`` / ``worker_cache`` / ``StateDatabase`` /
``create_worker`` mocked at the module path).
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

TENANT = "test-tenant-temp"
WORKER_ID = "wid-temp-1234"
_MP = "src.api.workers.crud_endpoints"


def _make_worker_data(*, temperature: float | None = 0.1) -> dict:
    """Return a worker_registry row with the given temperature in config."""
    return {
        "worker_name": "test_worker",
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        "config": {
            "agent_type": "pydantic_ai",
            "model_provider": "inference_router",
            "model": "deepseek",
            "system_prompt": "trace test",
            "instruction": None,
            "temperature": temperature,
            "max_tokens": None,
        },
        "status": "active",
    }


def _payload(*, temperature: float | None = 0.1, **overrides) -> dict:
    base = {
        "worker_name": "test_worker",
        "agent_role": "assistant",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "deepseek",
        "system_prompt": "trace test",
        "temperature": temperature,
    }
    base.update(overrides)
    return base


def register_worker_double(worker_id):
    """A ``register_worker`` stand-in that models the function's real return.

    Since the config merge moved into the UPDATE statement, ``register_worker``
    reports what the row now HOLDS — the merged ``config``, plus ``is_agent`` /
    ``config_revision`` preserved when the caller passes ``None`` — and the
    handlers build both the response body and the ``worker_states`` audit
    metadata from that. A double returning only ``{"worker_id": ...}`` lets a
    handler regression that echoes the caller's own nulls back at it pass
    unnoticed.

    Keeps its own row per worker_name so repeated calls accumulate the way the
    upsert does. It deliberately does NOT consult ``get_worker``: tests drive
    that mock with their own ``side_effect`` sequences, and reading it here
    would consume their entries.

    ``worker_id`` steers the handler's upsert detection: equal to the generated
    id means "new", different means "existing", ``None`` means "no row".
    """
    rows: dict[str, dict] = {}

    async def _register_worker(**kwargs):
        row = rows.setdefault(
            kwargs.get("worker_name"),
            {"config": {}, "is_agent": False, "config_revision": None},
        )
        row["config"] = {**row["config"], **(kwargs.get("config") or {})}
        if kwargs.get("is_agent") is not None:
            row["is_agent"] = bool(kwargs["is_agent"])
        if kwargs.get("config_revision") is not None:
            row["config_revision"] = kwargs["config_revision"]
        return {"worker_id": worker_id, **row}

    return AsyncMock(side_effect=_register_worker)


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):
        mock_registry.get_worker = AsyncMock(return_value=None)
        mock_registry.register_worker = register_worker_double(WORKER_ID)
        mock_registry.list_workers = AsyncMock(return_value=[])

        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        MockStateDB.return_value = mock_db

        mock_cache.remove_worker = AsyncMock()

        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/created.db"
        mock_create.return_value = fake_worker

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "state_db": mock_db,
            "create_worker": mock_create,
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


# ---------------------------------------------------------------------------
# PUT /workers/{name}: response must echo temperature
# ---------------------------------------------------------------------------


class TestPutResponseIncludesTemperature:
    """Regression: PUT response must NOT silently drop ``temperature``.

    Pre-fix (commit < 3787de3) the WorkerResponse() constructor in the PUT
    handler omitted the temperature kwarg, and Pydantic defaulted it to
    None on the wire even though the request body and the persisted JSONB
    config both held the real value.
    """

    def test_put_response_echoes_temperature(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        resp = client.put("/workers/test_worker", json=_payload(temperature=0.1))
        assert resp.status_code == 200
        body = resp.json()
        assert body["temperature"] == 0.1, (
            f"PUT response dropped temperature; got {body.get('temperature')!r} "
            "(this is the regression that 3787de3 fixed)"
        )

    def test_put_response_echoes_zero_temperature(self, client, deps):
        """``0.0`` is falsy — easy to lose to ``if temperature: ...`` checks."""
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(temperature=0.0))
        resp = client.put("/workers/test_worker", json=_payload(temperature=0.0))
        assert resp.status_code == 200
        assert resp.json()["temperature"] == 0.0

    def test_put_response_temperature_none_when_omitted(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(temperature=None))
        payload = _payload()
        payload.pop("temperature")
        resp = client.put("/workers/test_worker", json=payload)
        assert resp.status_code == 200
        assert resp.json()["temperature"] is None

    def test_put_passes_temperature_to_registry(self, client, deps):
        """Persistence side: register_worker's config dict must carry
        the supplied temperature so the next GET returns it."""
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        resp = client.put("/workers/test_worker", json=_payload(temperature=0.42))
        assert resp.status_code == 200
        deps["registry"].register_worker.assert_awaited_once()
        config_arg = deps["registry"].register_worker.await_args.kwargs["config"]
        assert config_arg["temperature"] == 0.42


# ---------------------------------------------------------------------------
# GET /workers/{name}: response must echo persisted temperature
# ---------------------------------------------------------------------------


class TestGetResponseIncludesTemperature:
    def test_get_response_echoes_persisted_temperature(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(temperature=0.7))
        resp = client.get("/workers/test_worker")
        assert resp.status_code == 200
        assert resp.json()["temperature"] == 0.7

    def test_get_response_temperature_null_when_unset(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(temperature=None))
        resp = client.get("/workers/test_worker")
        assert resp.status_code == 200
        assert resp.json()["temperature"] is None


# ---------------------------------------------------------------------------
# Roundtrip: PUT then GET returns the same temperature
# ---------------------------------------------------------------------------


class TestTemperatureRoundtrip:
    """The full caller-observable contract: PUT(temp=X) → GET returns temp=X.

    Drives the registry mock so that ``get_worker`` after PUT returns the
    config that PUT just persisted (which is what the live Postgres-backed
    registry does in production).
    """

    def test_pydantic_ai_temperature_roundtrip(self, client, deps):
        # PUT phase: existing worker present so PUT goes down the update branch
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(temperature=0.1))

        put_resp = client.put("/workers/test_worker", json=_payload(temperature=0.1))
        assert put_resp.status_code == 200
        assert put_resp.json()["temperature"] == 0.1

        # Capture the config that PUT persisted — production's Postgres
        # registry would now return this on the next get_worker() call.
        persisted_config = deps["registry"].register_worker.await_args.kwargs["config"]
        assert persisted_config["temperature"] == 0.1

        # GET phase: registry now returns a row whose config matches what
        # PUT persisted (mirroring the live JSONB roundtrip).
        deps["registry"].get_worker = AsyncMock(
            return_value={
                "worker_name": "test_worker",
                "worker_id": WORKER_ID,
                "db_path": "/tmp/test.db",
                "is_agent": True,
                "config": persisted_config,
                "status": "updated",
            }
        )
        get_resp = client.get("/workers/test_worker")
        assert get_resp.status_code == 200
        assert get_resp.json()["temperature"] == 0.1
