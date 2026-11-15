"""POST / PUT / GET /workers must echo the config the row STORES.

Regression for prod 2026-09-24: ``POST /workers`` (an upsert) returned
``max_tokens: null`` while the row held the requested value, so every n8n infer
atomic's worker_config_drift postcondition reported ``drift``. POST and PUT each
hand-built their ``WorkerResponse`` and omitted ``max_tokens`` (and tool_names,
output_schema, output_validator, mcp_config); GET did not. All three now share
``crud_endpoints._worker_response``.

The registry double models ``register_worker``'s real contract: it merges the
body over the stored config (``config || $4``) and returns what the row holds.
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
from src.auth import get_identity_proven, get_tenant_id, get_tenant_proven

TENANT = "test-tenant-stored-config"
WORKER_ID = "wid-stored-config-1"
NAME = "test-stored-config"
_MP = "src.api.workers.crud_endpoints"


@pytest.fixture
def row():
    """The one stored row every route reads and writes."""
    return {"config": {}, "is_agent": True, "config_revision": None, "exists": False}


@pytest.fixture
def deps(row):
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):

        async def _register_worker(**kwargs):
            row["config"] = {**row["config"], **(kwargs.get("config") or {})}
            row["exists"] = True
            return {
                "worker_id": WORKER_ID,
                "worker_name": NAME,
                "db_path": "/tmp/stored-config.db",
                "is_agent": row["is_agent"],
                "config": dict(row["config"]),
                "output_schema": None,
                "output_validator": None,
                "config_revision": row["config_revision"],
            }

        async def _get_worker(**_kwargs):
            if not row["exists"]:
                return None
            return {
                "worker_name": NAME,
                "worker_id": WORKER_ID,
                "db_path": "/tmp/stored-config.db",
                "is_agent": row["is_agent"],
                "config": dict(row["config"]),
                "status": "active",
            }

        mock_registry.register_worker = AsyncMock(side_effect=_register_worker)
        mock_registry.get_worker = AsyncMock(side_effect=_get_worker)

        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        MockStateDB.return_value = mock_db
        mock_cache.remove_worker = AsyncMock()

        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/stored-config.db"
        mock_create.return_value = fake_worker
        yield mock_registry


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
        "worker_name": NAME,
        "agent_role": "stored-config test worker",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "deepseek",
        "temperature": 0.2,
        "max_tokens": 1234,
        "tool_names": ["web_search"],
        "system_prompt": "Echo the input.",
    }
    base.update(overrides)
    return base


def test_post_create_echoes_stored_max_tokens_and_temperature(client, row):
    resp = client.post("/workers", json=_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert row["config"]["max_tokens"] == 1234  # the row holds it...
    # ...and the response says so. Pre-fix this was None: the POST builder
    # never passed max_tokens.
    assert body["max_tokens"] == 1234, body
    assert body["temperature"] == pytest.approx(0.2), body
    assert body["tool_names"] == ["web_search"], body


def test_post_upsert_echoes_new_stored_max_tokens_and_get_agrees(client, row):
    assert client.post("/workers", json=_payload()).status_code == 201

    resp = client.post("/workers", json=_payload(max_tokens=4321, temperature=0.7))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "updated", body
    assert row["config"]["max_tokens"] == 4321
    assert body["max_tokens"] == 4321, body
    assert body["temperature"] == pytest.approx(0.7), body

    get_body = client.get(f"/workers/{NAME}").json()
    for field in ("max_tokens", "temperature", "tool_names", "model", "model_provider"):
        assert get_body[field] == body[field], (field, get_body, body)


def test_post_partial_upsert_echoes_kept_max_tokens(client, row):
    """An omitted field keeps its stored value — the echo must report the KEPT
    value, not the body's silence."""
    assert client.post("/workers", json=_payload()).status_code == 201
    partial = _payload()
    partial.pop("max_tokens")
    body = client.post("/workers", json=partial).json()
    assert row["config"]["max_tokens"] == 1234
    assert body["max_tokens"] == 1234, body


def test_put_echoes_stored_max_tokens(client, row):
    row["exists"] = True  # PUT update branch, not auto-create
    resp = client.put(f"/workers/{NAME}", json=_payload(max_tokens=2048))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["max_tokens"] == 2048, body
    assert body["temperature"] == pytest.approx(0.2), body
    assert client.get(f"/workers/{NAME}").json()["max_tokens"] == 2048


def test_post_and_put_never_echo_the_override_secret(client, row):
    """The upsert echo carries the stored CONFIG, never a credential the caller
    sent: an upsert response lands in the caller's execution log (n8n stores
    every node output), so echoing api_key_override / mcp_config copies a
    credential into a second store. Runs with a PROVEN identity: the one case
    where GET would return the key, so this is about the upsert rule."""
    app.dependency_overrides[get_identity_proven] = lambda: True
    canary = "override-canary-" + "value"  # not a key; built so no scanner reads it as one
    resp = client.post("/workers", json=_payload(api_key_override=canary))
    assert resp.status_code == 201, resp.text  # a 4xx envelope would pass vacuously
    body = resp.json()
    assert row["config"].get("api_key_override") == canary  # stored...
    assert body.get("api_key_override") is None, body  # ...never echoed
    assert body.get("mcp_config") is None, body
    assert canary not in resp.text

    row["exists"] = True
    put = client.put(f"/workers/{NAME}", json=_payload(api_key_override=canary))
    assert put.status_code == 200, put.text
    assert put.json().get("api_key_override") is None, put.json()
    assert canary not in put.text


# (identity_proven, tenant_proven, secrets returned?)
#   same-tenant proven caller         -> returned (the control)
#   delegating service account        -> withheld: identity proven, tenant ADOPTED
#                                        via X-Effective-Tenant-Id (any tenant)
#   tokenless / unproven caller       -> withheld
_SECRET_READERS = [
    pytest.param(True, True, True, id="same-tenant-proven"),
    pytest.param(True, False, False, id="other-tenant-service-account"),
    pytest.param(False, False, False, id="unproven"),
]


@pytest.mark.parametrize(("identity", "tenant", "returned"), _SECRET_READERS)
def test_get_returns_secrets_only_to_a_caller_proven_for_this_tenant(
    client, row, identity, tenant, returned
):
    """mcp_config (can hold an auth dict), api_key_override and base_url_override
    are returned by GET only to a caller that proved its identity AND the
    worker's tenant. A delegating service account is identity-proven but holds a
    tenant it merely asserted, so it must be withheld exactly like an unproven
    caller. The same-tenant half is the control: without it a GET that never
    returns these fields would pass the other halves vacuously."""
    canary = "mcp-canary-" + "value"  # not a key; built so no scanner reads it as one
    key_canary = "override-canary-" + "value"
    url_canary = "http://user:" + "urlcanary" + "@byo.example/v1"
    mcp = {"mcpServers": {"s": {"url": "http://mcp.example/sse", "auth": {"token": canary}}}}
    row["config"] = {
        "max_tokens": 1234,
        "mcp_config": mcp,
        "api_key_override": key_canary,
        "base_url_override": url_canary,
    }
    row["exists"] = True
    app.dependency_overrides[get_identity_proven] = lambda: identity
    app.dependency_overrides[get_tenant_proven] = lambda: tenant
    resp = client.get(f"/workers/{NAME}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["max_tokens"] == 1234, body  # the rest of the row is intact
    if returned:
        assert body.get("mcp_config") == mcp, body
        assert body.get("api_key_override") == key_canary, body
        assert body.get("base_url_override") == url_canary, body
    else:
        assert body.get("mcp_config") is None, body
        assert body.get("api_key_override") is None, body
        assert body.get("base_url_override") is None, body
        for c in (canary, key_canary, "urlcanary"):
            assert c not in resp.text


def test_validation_422_never_echoes_the_submitted_mcp_auth(client):
    """A 422 body must not carry the caller's submitted input: an invalid
    mcp_config's auth token would otherwise land in every log storing the 422."""
    canary = "mcp-canary-" + "value"
    resp = client.post(
        "/workers", json=_payload(max_tokens="not-an-int", mcp_config={"auth": {"token": canary}})
    )
    assert resp.status_code == 422, resp.text
    assert canary not in resp.text
