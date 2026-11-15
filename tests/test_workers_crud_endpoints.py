"""Round-trip tests for /workers CRUD endpoints.

Focus: Plan B — GET /workers/{name} must return the full round-trippable
worker config (temperature, tool_names, output_schema, base_url_override, etc.)
so a client can re-POST the response to recreate the same worker.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_identity_proven, get_tenant_id, get_tenant_proven

TENANT = "test-tenant-roundtrip"
WORKER_ID = "wid-roundtrip-0001"
_MP = "src.api.workers.crud_endpoints"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roundtrip_deps():
    """Mocks that emulate a real register → get round trip via a shared backing store."""
    store: dict[str, dict] = {}  # keyed by worker_name

    async def _register_worker(
        *,
        tenant_id: str,
        subject_id: str,
        worker_name: str,
        worker_id: str,
        db_path: str,
        is_agent: bool,
        config: dict,
        visibility: str = "tenant",
        **_kwargs,
    ) -> dict:
        # Pull output_schema/output_validator out of config into their own
        # columns, matching how the real registry persists them.
        cfg = dict(config)
        output_schema = cfg.get("output_schema")
        output_validator = cfg.get("output_validator")
        store[worker_name] = {
            "worker_id": worker_id,
            "worker_name": worker_name,
            "tenant_id": tenant_id,
            "subject_id": subject_id,
            "visibility": visibility,
            "db_path": db_path,
            "is_agent": is_agent,
            "status": "created",
            "config": cfg,
            "output_schema": output_schema,
            "output_validator": output_validator,
        }
        return {"worker_id": None}  # new creation semantics

    async def _get_worker(*, tenant_id: str, subject_id: str, worker_name: str, **_kwargs):
        return store.get(worker_name)

    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
    ):
        mock_registry.register_worker = AsyncMock(side_effect=_register_worker)
        mock_registry.get_worker = AsyncMock(side_effect=_get_worker)
        mock_registry.list_workers = AsyncMock(return_value=[])

        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        MockStateDB.return_value = mock_db

        mock_cache.remove_worker = AsyncMock()

        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/roundtrip.db"
        mock_create.return_value = fake_worker

        yield {
            "registry": mock_registry,
            "state_db": mock_db,
            "create_worker": mock_create,
            "store": store,
        }


@pytest.fixture
def client(roundtrip_deps):
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    # This fixture runs with ENABLE_AUTH=false, i.e. NO middleware at all —
    # not even the bypass_auth branch — so `request.state.identity_proven`
    # is never set and `get_identity_proven` falls back to its fail-closed
    # default of False (see src/auth/dependencies.py). That is correct for a
    # genuinely tokenless caller, but this fixture is simulating a caller
    # that just created the worker it is about to read back — a proven,
    # same-tenant round trip — so assert that identity explicitly rather
    # than let it fall through the unproven default meant for a different
    # population entirely (added alongside the api_key_override redaction
    # in GET /workers/{worker_name}; see crud_endpoints.get_worker_status).
    # Same-tenant means tenant_proven too: identity alone is also what a
    # delegating service account holds for a tenant it merely adopted, and GET
    # withholds the overrides from that caller (crud_endpoints._secrets_withheld_reason).
    app.dependency_overrides[get_identity_proven] = lambda: True
    app.dependency_overrides[get_tenant_proven] = lambda: True

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


@pytest.fixture
def unproven_client(roundtrip_deps):
    """Same stack as `client`, but leaves `get_identity_proven` at its real
    fail-closed default (False) — i.e. the tokenless trusted-network shape
    `_trusted_network_tenant` produces, header-omitted or header-asserted.
    """
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
# Round-trip test
# ---------------------------------------------------------------------------


class TestGetWorkerRoundTrip:
    def test_get_returns_full_roundtrip_config(self, client, roundtrip_deps):
        """POST a worker with all round-trip fields, then GET and assert every field is preserved."""
        payload = {
            "worker_name": "roundtrip_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "openai",
            "model": "gpt-4",
            "temperature": 0.2,
            "system_prompt": "You are a helper.",
            "tool_names": ["calc", "search"],
            "output_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            # output_validator CODE is refused (422) since the exec path was
            # removed; the empty string is still accepted and stored as None.
            "output_validator": "",
            # A string mcp_config (file path) is refused since 2026-09-22; a
            # preset is named via mcp_config_preset and read back as mcp_config.
            "mcp_config_preset": "default",
            "base_url_override": "http://inference-router.inference-router.svc.cluster.local:8150/api",
            "api_key_override": "sk-test-override",
        }

        post_resp = client.post("/workers", json=payload)
        assert post_resp.status_code == 201, post_resp.text

        get_resp = client.get("/workers/roundtrip_worker")
        assert get_resp.status_code == 200, get_resp.text
        body = get_resp.json()

        # Core identity
        assert body["name"] == "roundtrip_worker"
        assert body["is_agent"] is True
        assert body["agent_type"] == "pydantic_ai"
        assert body["model_provider"] == "openai"
        assert body["model"] == "gpt-4"
        assert body["system_prompt"] == "You are a helper."

        # Round-trip fields
        assert body["temperature"] == 0.2
        assert body["tool_names"] == ["calc", "search"]
        assert body["output_schema"] == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        assert body["output_validator"] is None
        assert body["mcp_config"] == "default"
        assert (
            body["base_url_override"]
            == "http://inference-router.inference-router.svc.cluster.local:8150/api"
        )
        assert body["api_key_override"] == "sk-test-override"

    def test_get_missing_optional_fields_returns_defaults(self, client, roundtrip_deps):
        """Worker registered without round-trip fields should GET back with sane defaults."""
        payload = {
            "worker_name": "bare_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "mlx",
            "model": "mlx",
        }

        post_resp = client.post("/workers", json=payload)
        assert post_resp.status_code == 201, post_resp.text

        get_resp = client.get("/workers/bare_worker")
        assert get_resp.status_code == 200, get_resp.text
        body = get_resp.json()

        assert body["temperature"] is None
        assert body["tool_names"] == []
        assert body["output_schema"] is None
        assert body["output_validator"] is None
        assert body["mcp_config"] is None
        assert body["base_url_override"] is None
        assert body["api_key_override"] is None


class TestGetWorkerStatusRedactsSecretFromUnprovenCaller:
    """GET /workers/{worker_name} must withhold `api_key_override` from a
    caller whose identity was never proven (the tokenless trusted-network
    branch — header omitted, stamping DEFAULT_TENANT_ID, or header asserted
    and checked on spelling alone; both stamp `identity_proven=False`, see
    `AuthenticationMiddleware._stamp_nothing_proven`).
    """

    def test_unproven_caller_gets_no_api_key(self, unproven_client, roundtrip_deps):
        payload = {
            "worker_name": "secret_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "openai",
            "model": "gpt-4",
            "base_url_override": "http://inference-router.inference-router.svc.cluster.local:8150/api",
            "api_key_override": "sk-should-not-leak",  # pragma: allowlist secret
        }
        post_resp = unproven_client.post("/workers", json=payload)
        assert post_resp.status_code == 201, post_resp.text

        get_resp = unproven_client.get("/workers/secret_worker")
        assert get_resp.status_code == 200, get_resp.text
        body = get_resp.json()

        # BOTH secrets are withheld.
        assert body["api_key_override"] is None
        # ⛔ base_url_override is withheld TOO. An earlier version of this test
        # asserted it survived, on the reasoning that every write site stores a
        # plain endpoint URL. That reasoning was wrong: this very request wrote
        # the field straight from caller input via POST /workers, and the model
        # validator rejects only the empty string — see the credential-bearing
        # case below. A test that pins the old assumption would have made the
        # leak the spec.
        assert body["base_url_override"] is None
        # ... but this is still a narrow redaction, not a refusal or a partial
        # response: the rest of the record is intact and the status is 200.
        assert get_resp.status_code == 200
        assert body["name"] == "secret_worker"
        assert body["model"] == "gpt-4"

    def test_unproven_caller_gets_no_credential_bearing_base_url(
        self, unproven_client, roundtrip_deps
    ):
        """A base_url carrying userinfo is a credential, and must not leak.

        This is the case that refuted the original "base_url_override is always
        a plain endpoint URL" reasoning. The write path now refuses it (userinfo is never allowlisted).
        """
        payload = {
            "worker_name": "byo_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "openai",
            "model": "gpt-4",
            "base_url_override": "https://user:hunter2@byo.example.com/v1",  # pragma: allowlist secret
        }
        # Since the base_url_override allowlist landed the write path DOES stop
        # it: userinfo is refused outright, so the credential is never stored.
        resp = unproven_client.post("/workers", json=payload)
        assert resp.status_code == 422, resp.text
        types = {e["type"] for e in resp.json()["validation_errors"]}
        assert types == {"base_url_override_not_allowed"}
        assert unproven_client.get("/workers/byo_worker").status_code == 404

    def test_unproven_caller_gets_no_api_key_even_when_absent(
        self, unproven_client, roundtrip_deps
    ):
        """No key stored -> still None, and no spurious warning about withholding one."""
        payload = {
            "worker_name": "keyless_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "mlx",
            "model": "mlx",
        }
        post_resp = unproven_client.post("/workers", json=payload)
        assert post_resp.status_code == 201, post_resp.text

        get_resp = unproven_client.get("/workers/keyless_worker")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["api_key_override"] is None


class TestUpdateWorkerPathParamValidation:
    """Regression: PUT /workers/{worker_name} path param must reject reserved names.

    The body validator on WorkerCreateRequest has rejected 'undefined' since
    2025-07-29, but the update_worker URL path arg was only checked against
    the alnum/_/- regex — letting `PUT /workers/undefined` reach the
    auto-create branch and write an orphan row (live audit 2026-05-13).
    """

    @pytest.mark.parametrize("reserved_name", ["undefined", "null", "select", "drop"])
    def test_update_worker_reserved_path_param_returns_422(
        self, client, roundtrip_deps, reserved_name
    ):
        resp = client.put(
            f"/workers/{reserved_name}",
            json={
                "worker_name": "anything_legit",
                "agent_role": "assistant",
                "agent_type": "pydantic_ai",
                "model_provider": "mlx",
                "model": "mlx",
            },
        )
        assert resp.status_code == 422, resp.text
        assert "reserved" in resp.text.lower()
        assert reserved_name not in roundtrip_deps["store"]


class TestRegisterWorkerLogRedaction:
    """POST /workers must never log `api_key_override` (or other config
    values) at INFO -- only identifying fields (worker_name, agent_type, the
    set of config keys). See crud_endpoints.register_worker's log call just
    after `_validated_config_payload`.
    """

    def test_api_key_override_not_logged(self, client, roundtrip_deps, caplog):
        # Not named `secret`/`password`/`api_key`/etc: detect-secrets'
        # KeywordDetector flags `<denylisted keyword> = "<literal>"`
        # assignments regardless of the literal's content (substring match,
        # so even `api_key_value` would still trip it), so a keyword-shaped
        # identifier here would need a pragma even though this is plainly a
        # test fixture.
        override_value = "sk-SECRET123"
        payload = {
            "worker_name": "secret_worker",
            "agent_role": "assistant",
            "agent_type": "pydantic_ai",
            "model_provider": "openai",
            "model": "gpt-4",
            "api_key_override": override_value,
        }

        with caplog.at_level(logging.INFO, logger="src.api.workers.crud_endpoints"):
            post_resp = client.post("/workers", json=payload)
        assert post_resp.status_code == 201, post_resp.text

        assert override_value not in caplog.text
        # The fix must still log something useful: identifying fields, not
        # a blanket suppression of the whole registration log line.
        assert "secret_worker" in caplog.text
        assert "api_key_override" in caplog.text  # the KEY name, not its value
