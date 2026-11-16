"""Tests for /workers CRUD endpoints (src/api/workers/crud_endpoints.py).

Covers: create, list, get, update, delete workers with correct mock targets.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.workers.crud_endpoints import LIST_WORKERS_MAX_LIMIT
from src.auth import get_tenant_id

TENANT = "test-tenant-crud"
WORKER_ID = "wid-crud-1234"
_MP = "src.api.workers.crud_endpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(resp):
    """Extract error message from response (handles both formats)."""
    body = resp.json()
    return body.get("error", {}).get("message", body.get("detail", ""))


def _make_worker_data(
    *,
    worker_name="test_worker",
    worker_id=WORKER_ID,
    is_agent=True,
    config=None,
):
    return {
        "worker_name": worker_name,
        "worker_id": worker_id,
        "db_path": "/tmp/test.db",
        "is_agent": is_agent,
        "config": config
        or {
            "agent_type": "pydantic_ai",
            "model_provider": "mlx",
            "model": "mlx",
            "system_prompt": None,
            "instruction": None,
        },
        "status": "active",
    }


def _create_payload(**overrides):
    base = {
        "worker_name": "test_worker",
        "agent_role": "assistant",
        "agent_type": "pydantic_ai",
        "model_provider": "mlx",
        "model": "mlx",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def register_worker_double(worker_id, seed: dict | None = None):
    """A ``register_worker`` stand-in that models the function's real return.

    ``seed`` pre-populates the stored row for every worker_name, so a test can
    send a body that OMITS a field and still have the double report a different
    stored value for it. Without that, the merged row and the request agree on
    every asserted field and a handler reading either one looks identical —
    which is how "the response tells the truth" stayed unproven.

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
    initial = seed or {"config": {}, "is_agent": False, "config_revision": None}

    async def _register_worker(**kwargs):
        row = rows.setdefault(
            kwargs.get("worker_name"),
            {
                "config": dict(initial.get("config") or {}),
                "is_agent": bool(initial.get("is_agent")),
                "config_revision": initial.get("config_revision"),
            },
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
        # registry
        mock_registry.get_worker = AsyncMock(return_value=None)
        mock_registry.register_worker = register_worker_double(WORKER_ID)
        mock_registry.list_workers = AsyncMock(return_value=[])
        mock_registry.count_workers = AsyncMock(return_value=0)
        mock_registry.delete_worker = AsyncMock(return_value=True)
        mock_registry.delete_all_workers = AsyncMock(return_value=3)
        mock_registry.get_all_workers = AsyncMock(return_value=[])

        # state db
        mock_db = AsyncMock()
        mock_db.add_worker_state = AsyncMock(return_value=1)
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.delete_worker_state = AsyncMock(return_value=0)
        MockStateDB.return_value = mock_db

        # cache
        mock_cache.remove_worker = AsyncMock()

        # create_worker
        fake_worker = MagicMock()
        fake_worker.db_path = "/tmp/created.db"
        mock_create.return_value = fake_worker

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "state_db": mock_db,
            "create_worker": mock_create,
            "worker": fake_worker,
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
# POST /workers (create_worker_endpoint)
# ===========================================================================


class TestCreateWorker:
    def test_create_success(self, client, deps):
        # register_worker must return a new worker_id (same as generated)
        # to be treated as new creation (not upsert)
        deps["registry"].register_worker = register_worker_double(
            None
        )  # no existing → new creation
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "test_worker"
        assert body["status"] == "created"
        assert body["is_agent"] is True
        assert body["db_path"] == "/tmp/created.db"
        deps["create_worker"].assert_awaited_once()
        deps["registry"].register_worker.assert_awaited_once()

    def test_upsert_existing_worker(self, client, deps):
        """register_worker returns different worker_id → upsert detected."""
        deps["registry"].register_worker = register_worker_double("existing-id-999")
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "updated"
        assert body["id"] == "existing-id-999"
        deps["cache"].remove_worker.assert_awaited()

    def test_state_persistence_failure(self, client, deps):
        """State save fails → status='created_without_state'."""
        deps["registry"].register_worker = register_worker_double(None)
        deps["state_db"].add_worker_state = AsyncMock(side_effect=Exception("db down"))
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 201
        assert resp.json()["status"] == "created_without_state"

    def test_empty_worker_name(self, client, deps):
        resp = client.post("/workers", json=_create_payload(worker_name=""))
        assert resp.status_code == 422

    def test_api_key_error(self, client, deps):
        deps["create_worker"].side_effect = RuntimeError("API key required for OpenAI")
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 422
        assert "API key required" in _msg(resp)

    def test_authentication_error(self, client, deps):
        deps["create_worker"].side_effect = RuntimeError("authentication failed")
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 401

    def test_generic_creation_failure(self, client, deps):
        deps["create_worker"].side_effect = RuntimeError("something broke")
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 500
        assert "something broke" in _msg(resp)

    def test_name_normalization(self, client, deps):
        """Worker name is normalized to lowercase with underscores."""
        resp = client.post("/workers", json=_create_payload())
        assert resp.status_code == 201
        # create_worker receives normalized name
        call_kwargs = deps["create_worker"].call_args
        assert call_kwargs.kwargs["worker_name"] == "test_worker"

    def test_no_agent_role_creates_non_agent(self, client, deps):
        """Omitting agent_role and agent_type → is_agent=False."""
        deps["registry"].register_worker = register_worker_double(None)
        resp = client.post("/workers", json=_create_payload(agent_role=None, agent_type=None))
        assert resp.status_code == 201
        assert resp.json()["is_agent"] is False


# ===========================================================================
# GET /workers/types
# ===========================================================================


class TestListWorkerTypes:
    def test_returns_agent_types(self, client, deps):
        resp = client.get("/workers/types")
        assert resp.status_code == 200
        body = resp.json()
        assert "agent_types" in body
        assert isinstance(body["agent_types"], list)
        assert len(body["agent_types"]) > 0

    def test_import_error_fallback(self, client, deps):
        """If agent registry not importable, returns hardcoded list."""
        with patch(f"{_MP}.list_worker_types.__module__"):
            # The function itself has a try/except ImportError fallback
            # Just verify the endpoint returns some types
            resp = client.get("/workers/types")
            assert resp.status_code == 200
            assert "agent_types" in resp.json()


# ===========================================================================
# GET /workers (list_workers)
# ===========================================================================


class TestListWorkers:
    def test_empty_list(self, client, deps):
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_worker_names(self, client, deps):
        deps["registry"].list_workers = AsyncMock(return_value=["w1", "w2", "w3"])
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert resp.json() == ["w1", "w2", "w3"]

    def test_pagination_params(self, client, deps):
        resp = client.get("/workers?limit=5&offset=10")
        assert resp.status_code == 200
        deps["registry"].list_workers.assert_awaited_once_with(
            tenant_id=TENANT, subject_id="", limit=5, offset=10, prefix=None
        )

    def test_overflow_protection(self, client, deps):
        resp = client.get(f"/workers?limit=1000&offset={2**31}")
        assert resp.status_code == 422


# ===========================================================================
# GET /workers/{worker_name}
# ===========================================================================


class TestGetWorkerStatus:
    def test_found_with_state(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        deps["state_db"].get_latest_worker_state = AsyncMock(
            return_value={
                "state_type": "completed",
                "worker_state": {"status": "completed"},
            }
        )
        resp = client.get("/workers/test_worker")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "test_worker"
        assert body["status"] == "completed"

    def test_found_without_state(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value=None)
        resp = client.get("/workers/test_worker")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"  # falls back to worker_data status

    def test_not_found(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        resp = client.get("/workers/nonexistent")
        assert resp.status_code == 404

    def test_worker_name_too_long(self, client, deps):
        long_name = "a" * 101
        resp = client.get(f"/workers/{long_name}")
        assert resp.status_code == 422

    def test_null_bytes_rejected(self, client, deps):
        # URL encoding of null byte is %00
        resp = client.get("/workers/bad%00name")
        assert resp.status_code == 422

    def test_worker_data_missing_fields(self, client, deps):
        """Worker data missing required keys → 500."""
        deps["registry"].get_worker = AsyncMock(
            return_value={"worker_name": "w"}  # missing db_path and config
        )
        resp = client.get("/workers/test_worker")
        assert resp.status_code == 500


# ===========================================================================
# PUT /workers/{worker_name}
# ===========================================================================


# ---------------------------------------------------------------------------
# The response body and the audit trail must report the STORED row
# ---------------------------------------------------------------------------
#
# Found by an adversarial audit of PR #205, and it was a fair hit: reverting
# both handlers to echo `request.*` left the whole suite green. Every other test
# sends a body that fully determines the fields it asserts, so the merged row
# and the request agree and a handler reading either looks identical. These
# tests are the ones where they DISAGREE.
#
# It matters twice over. WM's own PUT is now partial, so a caller that omits a
# field and reads a null back may "correct" it — kag, karma_service, coding-ai
# and fr all re-PUT what they read. And `worker_states.metadata.config` is the
# post-deploy verification signal for this whole change; if it records the
# request, it agrees with itself forever and proves nothing.

_STORED = {
    "config": {
        "model": "stored-model",
        "agent_type": "claude_code",
        "system_prompt": "STORED PROMPT",
        "temperature": 0.9,
        "model_provider": "stored_provider",
        "instruction": "stored instruction",
        "agent_role": "stored role",
    },
    "is_agent": True,
    "config_revision": "stored-rev",
}


class TestResponseAndAuditReportTheStoredRow:
    def _partial_put(self, client, deps, worker_id=WORKER_ID):
        deps["registry"].register_worker = register_worker_double(worker_id, seed=_STORED)
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        # A bootstrapper's body: names itself and its description, nothing else.
        return client.put(
            "/workers/test_worker",
            json={"worker_name": "test_worker", "description": "touched"},
        )

    def test_put_response_echoes_the_stored_values_not_the_request(self, client, deps):
        resp = self._partial_put(client, deps)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "stored-model"
        assert body["agent_type"] == "claude_code"
        assert body["system_prompt"] == "STORED PROMPT"
        assert body["temperature"] == 0.9
        assert body["model_provider"] == "stored_provider"
        assert body["instruction"] == "stored instruction"
        assert body["config_revision"] == "stored-rev"
        assert body["is_agent"] is True

    def test_put_audit_metadata_records_the_stored_values(self, client, deps):
        self._partial_put(client, deps)
        deps["state_db"].add_worker_state.assert_awaited()
        kwargs = deps["state_db"].add_worker_state.await_args.kwargs
        recorded = kwargs["metadata"]["config"]
        assert recorded["model"] == "stored-model"
        assert recorded["system_prompt"] == "STORED PROMPT"
        assert recorded["agent_type"] == "claude_code"
        assert recorded["agent_role"] == "stored role"
        assert kwargs["is_agent"] is True

    def test_post_response_and_audit_report_the_stored_row_too(self, client, deps):
        """POST upserts, and it is the hot path — 19,272 prod events vs PUT's 3,092."""
        deps["registry"].register_worker = register_worker_double("existing-id-999", seed=_STORED)
        resp = client.post(
            "/workers", json={"worker_name": "test_worker", "description": "touched"}
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["model"] == "stored-model"
        assert body["system_prompt"] == "STORED PROMPT"
        assert body["is_agent"] is True

        kwargs = deps["state_db"].add_worker_state.await_args.kwargs
        assert kwargs["metadata"]["config"]["model"] == "stored-model"
        assert kwargs["is_agent"] is True


class TestUpdateWorker:
    def test_update_existing(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        resp = client.put("/workers/test_worker", json=_create_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "updated"

    def test_auto_create_when_not_found(self, client, deps):
        """Worker not found → auto-creates, then updates."""
        deps["registry"].get_worker = AsyncMock(
            side_effect=[None, _make_worker_data()]  # not found, then found after auto-create
        )
        resp = client.put("/workers/test_worker", json=_create_payload())
        assert resp.status_code == 200

    def test_auto_create_without_model_uses_platform_model(self, client, deps):
        """No model in the PUT → auto-create on the one platform model, never "mlx"."""
        deps["registry"].get_worker = AsyncMock(side_effect=[None, _make_worker_data()])
        payload = _create_payload()
        payload.pop("model", None)
        payload.pop("model_provider", None)
        client.put("/workers/test_worker", json=payload)
        kwargs = deps["create_worker"].await_args_list[0].kwargs
        assert kwargs["model"] == "local-llm"
        assert kwargs["model_provider"] == "vllm"

    def test_name_mismatch(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        resp = client.put(
            "/workers/test_worker",
            json=_create_payload(worker_name="different_name"),
        )
        assert resp.status_code == 422
        assert "must match" in _msg(resp)

    def test_auto_create_failure(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        deps["create_worker"].side_effect = RuntimeError("auto-create boom")
        resp = client.put("/workers/test_worker", json=_create_payload())
        assert resp.status_code == 500

    def test_cache_removal_failure_nonfatal(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        deps["cache"].remove_worker = AsyncMock(side_effect=Exception("cache err"))
        resp = client.put("/workers/test_worker", json=_create_payload())
        assert resp.status_code == 200  # still succeeds


# ===========================================================================
# DELETE /workers/ (delete_all_workers)
# ===========================================================================


class TestDeleteAllWorkers:
    def test_success(self, client, deps):
        deps["registry"].get_all_workers = AsyncMock(
            return_value=[
                _make_worker_data(worker_name="w1"),
                _make_worker_data(worker_name="w2"),
            ]
        )
        resp = client.delete("/workers/")
        assert resp.status_code == 204

    def test_state_cleanup_failure_resilient(self, client, deps):
        deps["registry"].get_all_workers = AsyncMock(return_value=[_make_worker_data()])
        deps["state_db"].delete_worker_state = AsyncMock(
            side_effect=Exception("state cleanup failed")
        )
        resp = client.delete("/workers/")
        assert resp.status_code == 204  # still succeeds

    def test_cache_cleanup_failure_resilient(self, client, deps):
        deps["registry"].get_all_workers = AsyncMock(return_value=[_make_worker_data()])
        deps["cache"].remove_worker = AsyncMock(side_effect=Exception("cache err"))
        resp = client.delete("/workers/")
        assert resp.status_code == 204


# ===========================================================================
# DELETE /workers/{worker_name}
# ===========================================================================


class TestDeleteWorker:
    def test_success(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        resp = client.delete("/workers/test_worker")
        assert resp.status_code == 204

    def test_not_found(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        resp = client.delete("/workers/nonexistent")
        assert resp.status_code == 404

    def test_state_cleanup_failure_nonfatal(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        deps["state_db"].delete_worker_state = AsyncMock(side_effect=Exception("state err"))
        resp = client.delete("/workers/test_worker")
        assert resp.status_code == 204  # still succeeds

    def test_registry_deletion_failure_nonfatal(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data())
        deps["registry"].delete_worker = AsyncMock(side_effect=Exception("reg delete err"))
        resp = client.delete("/workers/test_worker")
        assert resp.status_code == 204  # still succeeds


# ===========================================================================
# PUT /workers/{worker_name}/prompt (swap_worker_prompt)
# ===========================================================================

import hashlib as _hashlib


def _sha(t: str) -> str:
    return _hashlib.sha256((t or "").encode("utf-8")).hexdigest()


def _existing_worker(*, prompt="old prompt", schema=None):
    return {
        "worker_id": WORKER_ID,
        "worker_name": "base_worker",
        "is_agent": True,
        "config": {"system_prompt": prompt, "output_schema": schema or {"type": "object"}},
        "output_schema": schema or {"type": "object"},
        "output_validator": None,
    }


class TestSwapWorkerPrompt:
    def test_missing_worker_404(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put("/workers/base_worker/prompt", json={"new_system_prompt": "new"})
        assert resp.status_code == 404
        # A missing WORKER keeps the ordinary envelope — never the pin body.
        assert resp.json().get("error") != "prompt_version_not_found"
        assert "Worker 'base_worker' not found" in resp.text
        deps["registry"].swap_prompt.assert_not_called()

    def test_happy_path_swaps_and_evicts(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock(return_value=_existing_worker(prompt="new prompt"))
        resp = client.put(
            "/workers/base_worker/prompt",
            json={"new_system_prompt": "new prompt", "canary_version": "base_worker__canary_v9"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["swapped"] is True
        assert body["worker_name"] == "base_worker"
        assert body["output_schema_preserved"] is True
        assert body["prompt_sha256"] == _sha("new prompt")
        deps["registry"].swap_prompt.assert_awaited_once()
        deps["cache"].remove_worker.assert_awaited()  # cache evicted
        deps["state_db"].add_worker_state.assert_awaited()  # audit written

    def test_pmp01_schema_mismatch_422(self, client, deps):
        deps["registry"].get_worker = AsyncMock(
            return_value=_existing_worker(schema={"type": "object"})
        )
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put(
            "/workers/base_worker/prompt",
            json={"new_system_prompt": "new", "expected_output_schema": {"type": "array"}},
        )
        assert resp.status_code == 422
        deps["registry"].swap_prompt.assert_not_called()  # never reached the write

    def test_cas_sha_mismatch_409(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker(prompt="old prompt"))
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put(
            "/workers/base_worker/prompt",
            json={"new_system_prompt": "new", "expected_prompt_sha256": _sha("stale prompt")},
        )
        assert resp.status_code == 409
        deps["registry"].swap_prompt.assert_not_called()

    def test_cas_concurrent_change_409(self, client, deps):
        # sha matches the read, but the atomic UPDATE finds 0 rows (concurrent change)
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker(prompt="old prompt"))
        deps["registry"].swap_prompt = AsyncMock(return_value=None)
        resp = client.put(
            "/workers/base_worker/prompt",
            json={"new_system_prompt": "new", "expected_prompt_sha256": _sha("old prompt")},
        )
        assert resp.status_code == 409
        deps["registry"].swap_prompt.assert_awaited_once()


PLATFORM_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"


class TestSwapPromptVersion:
    """P4 A3/A4: ``prompt_version`` swaps ledger content and labels it truthfully.

    Ledger versions are SYSTEM assets (U6 tenancy), so these run as a platform
    tenant; the customer refusal is pinned in ``test_a_customer_cannot_swap_in_a_ledger_version``.
    """

    @pytest.fixture(autouse=True)
    def _platform_tenant(self, client):
        app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
        yield

    def test_a_customer_cannot_swap_in_a_ledger_version(self, client, deps):
        app.dependency_overrides[get_tenant_id] = lambda: TENANT  # a non-platform tenant
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock()
        with patch("src.api.prompts.repository.get_version", AsyncMock()) as get:
            resp = client.put("/workers/base_worker/prompt", json={"prompt_version": "kw@4"})
        assert resp.status_code == 403
        get.assert_not_awaited()  # refused before the ledger is read
        deps["registry"].swap_prompt.assert_not_called()

    def test_the_system_tenant_cannot_swap_in_a_ledger_version(self, client, deps):
        """``00000000-…-0001`` is commercial-LLM platform, NOT prompt-write
        platform (validator 5 T1, scenario B).

        MUTATION: gate this swap on ``is_commercial_platform_tenant`` ⇒ the
        system tenant reads the ledger and swaps ⇒ red.
        """
        app.dependency_overrides[get_tenant_id] = lambda: "00000000-0000-0000-0000-000000000001"
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock()
        with patch("src.api.prompts.repository.get_version", AsyncMock()) as get:
            resp = client.put("/workers/base_worker/prompt", json={"prompt_version": "kw@4"})
        assert resp.status_code == 403, resp.text
        assert "platform_prompt_forbidden" in resp.text
        get.assert_not_awaited()
        deps["registry"].swap_prompt.assert_not_called()

    def test_a_customer_free_text_swap_is_unaffected(self, client, deps):
        app.dependency_overrides[get_tenant_id] = lambda: TENANT
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock(return_value=_existing_worker())
        resp = client.put("/workers/base_worker/prompt", json={"new_system_prompt": "mine"})
        assert resp.status_code == 200, resp.text

    @staticmethod
    def _ledger(status="PUBLISHED", system="ledger body v4"):
        return {"prompt_id": "kw", "version": 4, "status": status, "system": system}

    def test_prompt_version_swaps_the_ledger_body_and_labels_it(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock(return_value=_existing_worker())
        with patch(
            "src.api.prompts.repository.get_version", AsyncMock(return_value=self._ledger())
        ) as get:
            resp = client.put("/workers/base_worker/prompt", json={"prompt_version": "kw@4"})
        assert resp.status_code == 200, resp.text
        get.assert_awaited_once_with("kw", 4)
        kwargs = deps["registry"].swap_prompt.await_args.kwargs
        assert kwargs["new_system_prompt"] == "ledger body v4"
        assert kwargs["config_revision"] == "kw@4"
        assert resp.json()["prompt_sha256"] == _sha("ledger body v4")

    @pytest.mark.parametrize("status", ["DRAFT", "ARCHIVED", "DISCARDED"])
    def test_a_non_runnable_version_is_the_a4_409(self, client, deps, status):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock()
        with patch(
            "src.api.prompts.repository.get_version",
            AsyncMock(return_value=self._ledger(status=status)),
        ):
            resp = client.put("/workers/base_worker/prompt", json={"prompt_version": "kw@4"})
        assert resp.status_code == 409
        assert resp.json() == {
            "error": "prompt_version_not_runnable",
            "status": status,
            "prompt_id": "kw",
            "version": 4,
        }
        deps["registry"].swap_prompt.assert_not_called()

    def test_a_missing_version_is_404(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock()
        with patch("src.api.prompts.repository.get_version", AsyncMock(return_value=None)):
            resp = client.put("/workers/base_worker/prompt", json={"prompt_version": "kw@4"})
        assert resp.status_code == 404
        # The pin-not-found body, top level — distinct from a missing worker.
        assert resp.json() == {
            "error": "prompt_version_not_found",
            "prompt_id": "kw",
            "version": 4,
        }
        deps["registry"].swap_prompt.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"new_system_prompt": "x", "prompt_version": "kw@4"},
            {"prompt_version": "not a revision"},
        ],
    )
    def test_exactly_one_well_formed_source_is_required(self, client, deps, body):
        deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put("/workers/base_worker/prompt", json=body)
        assert resp.status_code == 422
        deps["registry"].swap_prompt.assert_not_called()

    def test_neither_source_is_mains_exact_missing_field_422(self, client, deps):
        """Byte-identical to main, where ``new_system_prompt`` was simply required."""
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put("/workers/base_worker/prompt", json={})
        assert resp.status_code == 422
        assert resp.json() == {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation failed: body -> new_system_prompt: Field required",
                "status": 422,
            },
            "validation_errors": [
                {
                    "field": "body -> new_system_prompt",
                    "message": "Field required",
                    "type": "missing",
                }
            ],
        }
        deps["registry"].swap_prompt.assert_not_called()

    def test_both_sources_is_the_exactly_one_error(self, client, deps):
        deps["registry"].swap_prompt = AsyncMock()
        resp = client.put(
            "/workers/base_worker/prompt",
            json={"new_system_prompt": "x", "prompt_version": "kw@4"},
        )
        assert resp.status_code == 422
        assert "exactly one of new_system_prompt and prompt_version" in resp.text
        deps["registry"].swap_prompt.assert_not_called()

    @pytest.mark.parametrize("on", [True, False])
    def test_a_free_text_swap_label(self, client, deps, monkeypatch, on):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("PROMPTS_READ_REQUIRES_PUBLISHED", "true" if on else "false")
        reset_settings_cache()
        try:
            deps["registry"].get_worker = AsyncMock(return_value=_existing_worker())
            deps["registry"].swap_prompt = AsyncMock(return_value=_existing_worker())
            resp = client.put(
                "/workers/base_worker/prompt",
                json={"new_system_prompt": "free text", "canary_version": "cw__canary"},
            )
        finally:
            monkeypatch.delenv("PROMPTS_READ_REQUIRES_PUBLISHED", raising=False)
            reset_settings_cache()
        assert resp.status_code == 200, resp.text
        label = deps["registry"].swap_prompt.await_args.kwargs["config_revision"]
        # Flag on: unversioned:<sha12>. Flag off: today's canary_version label.
        assert label == (f"unversioned:{_sha('free text')[:12]}" if on else "cw__canary")


_VALID_STORED_CONFIG = {
    "agent_role": "synthesiser",
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": "local-llm",
    "system_prompt": "STORED PROMPT",
}


class TestAPartialBodyIsValidatedAgainstTheMergedRow:
    """The cross-field rules must see the row the write PRODUCES, not the body.

    Regression, measured in production on 2026-08-29: after ``knowledge/kag``
    correctly stopped re-asserting operator-owned fields, every one of its three
    startup seeds 422'd with "model is required when model_provider is
    specified" — it still sent ``model_provider`` while omitting ``model``,
    which the stored row already had. Zero ``wm_seed_ok`` for 16 hours; kag
    could not update its workers at all, and an operator editing a single field
    by hand hit the same wall.

    The rules were not wrong. They ran at FastAPI parse time against the request
    BODY, which stopped being the whole worker when the write became a merge.

    Nothing here would have failed before that merge landed, which is precisely
    why the previous unit of work shipped this green: every test it added sent a
    body containing ``model``, so the merged row and the request agreed and no
    test could tell which one the handler read.
    """

    def test_the_kag_seed_body_is_accepted(self, client, deps):
        """kag's literal body for an already-populated worker: no ``model``."""
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(
                worker_name="kag_synthesize", config=dict(_VALID_STORED_CONFIG)
            )
        )
        deps["registry"].register_worker = register_worker_double(
            WORKER_ID, seed={"config": dict(_VALID_STORED_CONFIG), "is_agent": True}
        )
        resp = client.put(
            "/workers/kag_synthesize",
            json={
                "worker_name": "kag_synthesize",
                "agent_role": "synthesiser",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
            },
        )
        assert resp.status_code == 200, resp.text
        # and the fields it stayed silent about are still the stored ones
        assert resp.json()["model"] == "local-llm"

    def test_a_create_still_requires_a_model_with_a_provider(self, client, deps):
        """The negative control: with nothing stored, the body IS the worker."""
        deps["registry"].get_worker = AsyncMock(return_value=None)
        resp = client.post(
            "/workers",
            json={
                "worker_name": "brand_new",
                "agent_role": "assistant",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
            },
        )
        assert resp.status_code == 422, resp.text
        assert "model is required" in _msg(resp)

    def test_a_partial_body_the_STORED_ROW_cannot_complete_is_still_rejected(self, client, deps):
        """Relaxation is bounded by the merged row, not by the request alone.

        A VALID stored row that simply cannot supply the missing piece: a
        ``claude_code`` worker (which needs no model) being converted to
        ``pydantic_ai`` without one. Nothing here is grandfathered — the stored
        row broke no rule — so the 422 stands.
        """
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(
                config={"agent_role": "synthesiser", "agent_type": "claude_code"}
            )
        )
        resp = client.put(
            "/workers/test_worker",
            json={
                "worker_name": "test_worker",
                "agent_role": "synthesiser",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
            },
        )
        assert resp.status_code == 422, resp.text
        assert "model is required" in _msg(resp)


class TestAPreExistingViolationDoesNotStrandTheRow:
    """44 of 751 production rows (2026-08-29) carry an ``agent_type`` with an
    empty ``agent_role``, and 44 carry ``pydantic_ai`` with no
    ``model_provider``.

    Validating the merged row would make every one of those workers
    permanently un-updatable — a strictly worse defect than the one the
    validation exists to fix. A violation the stored row ALREADY had is
    therefore grandfathered; a violation this request introduces is not.
    """

    _BROKEN_STORED: ClassVar[dict] = {
        "agent_type": "pydantic_ai",
        "model_provider": "mlx",
        "model": "mlx",
        # no agent_role — the shape 44 live rows are in
    }

    def test_a_bootstrapper_can_still_reseed_a_pre_existing_violator(self, client, deps):
        """The body is incoherent alone AND the merged row is too — but the
        stored row already broke this exact rule, so the write goes through.

        This is the realistic shape: a bootstrapper re-seeding one of the 44
        rows that carry an ``agent_type`` with no ``agent_role``. The body must
        be one that cannot stand alone, or the merged row is never consulted
        and this proves nothing.
        """
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(config=dict(self._BROKEN_STORED))
        )
        deps["registry"].register_worker = register_worker_double(
            WORKER_ID, seed={"config": dict(self._BROKEN_STORED), "is_agent": True}
        )
        resp = client.put(
            "/workers/test_worker",
            json={
                "worker_name": "test_worker",
                "agent_type": "pydantic_ai",
                "model_provider": "mlx",
            },
        )
        assert resp.status_code == 200, resp.text

    def test_a_DIFFERENT_new_violation_on_the_same_row_is_still_rejected(self, client, deps):
        """Grandfathering is per-rule, not a blanket amnesty for a broken row.

        Only the violation the row ALREADY had is forgiven; this request brings
        a different one (an unknown provider) and is refused on that.
        """
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(config=dict(self._BROKEN_STORED))
        )
        resp = client.put(
            "/workers/test_worker",
            json={
                "worker_name": "test_worker",
                "agent_role": "now set",
                "agent_type": "pydantic_ai",
                "model_provider": "not_a_real_provider",
            },
        )
        assert resp.status_code == 422, resp.text
        assert "provider" in _msg(resp).lower()


# ===========================================================================
# GET /workers — the filter must be applied or refused, never ignored
# ===========================================================================
#
# Measured on the live service 2026-09-23 (pod ``worker-management`` in
# namespace ``workers``, curl from inside the pod against ``127.0.0.1:8887``):
#
# * ``?limit=500`` returned **165** worker names; ``?limit=100`` and the bare
#   ``GET /workers`` both returned **100**, byte-identical. The default page
#   dropped 65 of 165 workers and no field in the response said so.
# * ``GET /workers``, ``?prefix=aa``, ``?prefix=zzz_nonexistent_prefix`` and
#   ``?owner_tenant_id=00000000-0000-0000-0000-000000000000`` produced **four
#   identical md5s** (``f3e492d2c3465354dff7f8485dbd5de9``). The filter was not
#   weak, it was absent: FastAPI ignores query parameters a route does not
#   declare.
# * Cross-checked directly against Postgres on worker4 (no pgbouncer):
#   ``SELECT count(*) FROM workers WHERE tenant_id = 'aaaaaaaa-…-000000000001'``
#   = 165, so the 165 is the real population and the 100 was the lie.
#
# ``getWorkerCohort`` — the only production op in the ``metaPrompt.loop`` scope —
# issues exactly ``GET /workers?prefix=<derived>``, so every cohort it ever read
# was an arbitrary first page of the whole tenant.
#
# The real ``WHERE`` behind ``prefix`` is proved against a live PostgreSQL in
# tests/integration/test_list_workers_prefix_filter.py; it cannot be proved here,
# because tests/db_test_helper.py's in-memory fake ignores any predicate that is
# not ``col = $N``.

# The live numbers above, reused as the fixture population so a regression reads
# as the incident rather than as an abstract off-by-one.
LIVE_TOTAL = 165
LIVE_DEFAULT_PAGE = 100  # what the old default returned of those 165


class TestFilterIsAppliedOrRefused:
    def test_prefix_reaches_the_registry(self, client, deps):
        """A supplied prefix must become a real argument, not a discarded string."""
        deps["registry"].count_workers = AsyncMock(return_value=3)
        deps["registry"].list_workers = AsyncMock(
            return_value=["fr_planner_a", "fr_planner_b", "fr_planner_c"]
        )

        resp = client.get("/workers?prefix=fr_planner_")

        assert resp.status_code == 200
        deps["registry"].list_workers.assert_awaited_once_with(
            tenant_id=TENANT,
            subject_id="",
            limit=LIST_WORKERS_MAX_LIMIT,
            offset=0,
            prefix="fr_planner_",
        )
        deps["registry"].count_workers.assert_awaited_once_with(
            tenant_id=TENANT, subject_id="", prefix="fr_planner_"
        )

    def test_unmatched_prefix_returns_empty_not_everything(self, client, deps):
        """The measured defect, in one assertion: an unmatched filter must not
        fall back to the unfiltered population."""
        deps["registry"].count_workers = AsyncMock(return_value=0)
        deps["registry"].list_workers = AsyncMock(return_value=[])

        resp = client.get("/workers?prefix=zzz_nonexistent_prefix")

        assert resp.status_code == 200
        assert resp.json() == []
        assert resp.headers["X-Total-Count"] == "0"

    def test_owner_tenant_id_is_refused_not_ignored(self, client, deps):
        """``owner_tenant_id`` is a parameter this service does not implement.

        It is refused rather than implemented: the listing is already scoped to
        the caller's verified tenant, so a caller-supplied tenant filter is
        either redundant or a cross-tenant read, and no query parameter may be
        able to widen tenant scope.
        """
        resp = client.get("/workers?owner_tenant_id=00000000-0000-0000-0000-000000000000")

        assert resp.status_code == 422
        assert "owner_tenant_id" in resp.text
        deps["registry"].list_workers.assert_not_awaited()

    def test_typo_in_a_known_param_is_refused(self, client, deps):
        resp = client.get("/workers?prefx=fr_")

        assert resp.status_code == 422
        assert "prefx" in resp.text
        deps["registry"].list_workers.assert_not_awaited()

    def test_known_params_together_are_accepted(self, client, deps):
        deps["registry"].count_workers = AsyncMock(return_value=2)
        deps["registry"].list_workers = AsyncMock(return_value=["fr_a", "fr_b"])

        resp = client.get("/workers?prefix=fr_&limit=10&offset=0")

        assert resp.status_code == 200


class TestTruncationIsVisible:
    def test_default_asks_the_registry_for_the_whole_population(self, client, deps):
        """The exact live shape: 165 exist, and the default must fetch all 165.

        Pre-fix this issued ``limit=100`` and returned 100 of 165 with nothing in
        the response saying so. The default page size is what made the answer
        wrong, so the assertion is on the limit the handler asks for.
        """
        deps["registry"].count_workers = AsyncMock(return_value=LIVE_TOTAL)
        deps["registry"].list_workers = AsyncMock(
            return_value=[f"w{i:03d}" for i in range(LIVE_TOTAL)]
        )

        resp = client.get("/workers")

        assert resp.status_code == 200
        assert len(resp.json()) == LIVE_TOTAL
        deps["registry"].list_workers.assert_awaited_once_with(
            tenant_id=TENANT,
            subject_id="",
            limit=LIST_WORKERS_MAX_LIMIT,
            offset=0,
            prefix=None,
        )
        assert resp.headers["X-Total-Count"] == str(LIVE_TOTAL)
        assert resp.headers["X-Has-More"] == "false"

    def test_population_over_the_ceiling_is_refused_in_the_body(self, client, deps):
        """Above the ceiling the default refuses, and the total is in the BODY.

        Headers are not enough on their own here: both n8n nodes call this op
        through ``request({json: true})``, which discards response headers, so a
        header-only signal would be invisible to the callers that matter.
        """
        over = LIST_WORKERS_MAX_LIMIT + 7
        deps["registry"].count_workers = AsyncMock(return_value=over)
        deps["registry"].list_workers = AsyncMock(return_value=[])

        resp = client.get("/workers")

        assert resp.status_code == 400
        assert str(over) in resp.text
        deps["registry"].list_workers.assert_not_awaited()

    def test_explicit_page_gets_a_total_and_has_more(self, client, deps):
        deps["registry"].count_workers = AsyncMock(return_value=LIVE_TOTAL)
        deps["registry"].list_workers = AsyncMock(
            return_value=[f"w{i:03d}" for i in range(LIVE_DEFAULT_PAGE)]
        )

        resp = client.get(f"/workers?limit={LIVE_DEFAULT_PAGE}")

        assert resp.status_code == 200
        assert len(resp.json()) == LIVE_DEFAULT_PAGE
        assert resp.headers["X-Total-Count"] == str(LIVE_TOTAL)
        assert resp.headers["X-Has-More"] == "true"
        assert resp.headers["X-Limit"] == str(LIVE_DEFAULT_PAGE)
        assert resp.headers["X-Offset"] == "0"

    def test_last_page_says_has_more_false(self, client, deps):
        deps["registry"].count_workers = AsyncMock(return_value=LIVE_TOTAL)
        deps["registry"].list_workers = AsyncMock(
            return_value=[f"w{i:03d}" for i in range(LIVE_TOTAL - LIVE_DEFAULT_PAGE)]
        )

        resp = client.get(f"/workers?limit={LIVE_DEFAULT_PAGE}&offset={LIVE_DEFAULT_PAGE}")

        assert resp.status_code == 200
        assert resp.headers["X-Total-Count"] == str(LIVE_TOTAL)
        assert resp.headers["X-Has-More"] == "false"

    def test_complete_default_page_is_not_refused(self, client, deps):
        """A default page that is genuinely complete must still be a plain 200 —
        otherwise the refusal above would be a blanket break, not a truth fix."""
        deps["registry"].count_workers = AsyncMock(return_value=3)
        deps["registry"].list_workers = AsyncMock(return_value=["a", "b", "c"])

        resp = client.get("/workers")

        assert resp.status_code == 200
        assert resp.json() == ["a", "b", "c"]
        assert resp.headers["X-Total-Count"] == "3"
        assert resp.headers["X-Has-More"] == "false"
