"""Per-call prompt-version pin on ``POST /workers/{name}/message`` (P4 contract §4).

What is proved here, and against what:

* The worker cache is the REAL :class:`WorkerInstanceCache`; only the agent
  BUILDER is stubbed (it builds from ``worker_data["config"]`` exactly as
  ``_create_worker_instance`` does). So "the cache is not polluted" is measured
  on the real cache — slot keys and instance identity — not on a mock's calls.
* The ledger read is stubbed at ``prompt_pin.prompt_repository.get_version``;
  what that function returns for each status is proved against real Postgres in
  ``tests/integration/test_prompt_versions_api.py``.

The worker is labelled ``config_revision = "kw@1"`` and carries its own prompt
``OWN``. The card stamp pins ``kw@3`` whose ledger body is ``PINNED``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.prompts.hashing import sha256_text
from src.api.worker_management_api import app
from src.api.workers import prompt_pin
from src.api.workers._globals import WorkerInstanceCache
from src.auth import get_tenant_id
from src.settings import reset_settings_cache
from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

# A PLATFORM tenant (n8n-dev): a NON-platform tenant's pin is never applied
# (``tenant_forbidden``, proved in test_message_prompt_source.py), so every pin
# behaviour below is a platform caller's.
TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
WORKER = "pin_worker"
WORKER_ID = "wid-pin-1"
_MP = "src.api.workers.message_endpoints"

OWN = "You are the worker's OWN prompt."
PINNED = "You are prompt kw at version 3."
STAMP = {"prompt_versions": {"kw": 3}}


@asynccontextmanager
async def _noop_ctx(*_a, **_kw):
    yield "fake-lock"


def _worker_data() -> dict[str, Any]:
    # A FRESH dict every call, as WorkerRegistryDB.get_worker returns.
    return {
        "worker_name": WORKER,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        "config": {
            "agent_type": "pydantic_ai",
            "model_provider": "mlx",
            "model": "mlx",
            "temperature": 0.1,
            "system_prompt": OWN,
        },
        "status": "active",
        "config_revision": "kw@1",
    }


def _ledger_row(status: str, version: int = 3, system: str = PINNED) -> dict[str, Any]:
    return {
        "prompt_id": "kw",
        "version": version,
        "status": status,
        "system": system,
        "system_prompt_sha256": sha256_text(system),
    }


async def _build(_self, worker_name, worker_data, **_kw):
    """Stand-in for ``_create_worker_instance``: builds from ``worker_data`` the way
    the real one does, in the real object shape (``instance.config.system_prompt``)."""
    config = worker_data.get("config") or {}
    instance = MagicMock()
    run_result = MagicMock()
    run_result.output = f"ran with: {config.get('system_prompt')}"
    run_result.data = run_result.output
    run_result.usage_metrics = None
    run_result.session_id = None
    run_result.cost_usd = None
    run_result.duration_ms = None
    instance.run = AsyncMock(return_value=run_result)
    instance.config = PydanticAIAgentConfig(
        name=worker_name,
        model_provider="mlx",
        model="mlx",
        db_path="/tmp/test.db",
        system_prompt=config.get("system_prompt"),
    )
    del instance.system_prompt  # no real agent has a top-level one
    del instance.check_health  # a MagicMock's would read as unhealthy
    return instance


@pytest.fixture
def flag(monkeypatch):
    def _set(on: bool) -> None:
        monkeypatch.setenv("PROMPTS_READ_REQUIRES_PUBLISHED", "true" if on else "false")
        reset_settings_cache()

    yield _set
    monkeypatch.delenv("PROMPTS_READ_REQUIRES_PUBLISHED", raising=False)
    reset_settings_cache()


@pytest.fixture
def env():
    cache = WorkerInstanceCache()
    with (
        patch(f"{_MP}.worker_registry") as registry,
        patch(f"{_MP}.worker_cache", cache),
        patch(f"{_MP}.state_manager") as state_mgr,
        patch(f"{_MP}.StateDatabase") as state_db_cls,
        patch.object(WorkerInstanceCache, "_create_worker_instance", _build),
        patch.object(prompt_pin.prompt_repository, "get_version", new_callable=AsyncMock) as get,
    ):
        state_mgr.atomic_worker_operation = _noop_ctx
        state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        state_mgr.validate_state_transition = MagicMock(return_value=True)
        db = AsyncMock()
        db.get_latest_worker_state = AsyncMock(return_value=None)
        db.add_worker_state = AsyncMock(return_value=1)
        state_db_cls.return_value = db
        registry.get_worker = AsyncMock(side_effect=lambda **_kw: _worker_data())
        # The ledger: kw@1 is the worker's OWN prompt (so its label verifies,
        # A3); kw@3 is the pinned version, PUBLISHED unless a test says otherwise.
        ledger: dict[tuple[str, int], dict[str, Any] | None] = {
            ("kw", 1): _ledger_row("PUBLISHED", version=1, system=OWN),
            ("kw", 3): _ledger_row("PUBLISHED"),
        }
        get.side_effect = lambda pid, ver: ledger.get((pid, ver))
        yield {
            "cache": cache,
            "db": db,
            "get_version": get,
            "ledger": ledger,
            "registry": registry,
        }


@pytest.fixture
def client(env):
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(_a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _send(client, **extra):
    return client.post(f"/workers/{WORKER}/message", json={"message": "Hello", **extra})


def _completed_states(db) -> list[dict[str, Any]]:
    return [
        c.kwargs["worker_state"]
        for c in db.add_worker_state.await_args_list
        if c.kwargs.get("state_type") == "completed"
    ]


class TestPinnedExecution:
    def test_flag_on_stamp_runs_the_pinned_version_and_leaves_the_cache_clean(
        self, client, env, flag
    ):
        flag(True)
        cache = env["cache"]

        # 1. An un-stamped call seeds the worker's own slot.
        before = _send(client)
        assert before.status_code == 200, before.text
        assert before.json()["output"] == f"ran with: {OWN}"
        assert len(cache._workers) == 1
        (own_key, own_instance) = next(iter(cache._workers.items()))

        # 2. The stamped call runs kw@3's content and says so.
        pinned = _send(client, **STAMP)
        assert pinned.status_code == 200, pinned.text
        body = pinned.json()
        assert body["output"] == f"ran with: {PINNED}"
        meta = body["metadata"]
        assert meta["worker_config_revision"] == "kw@3"
        assert (meta["prompt_id"], meta["prompt_version"]) == ("kw", 3)
        assert meta["system_prompt_sha256"] == sha256_text(PINNED)
        assert meta["system_prompt_source"] == "instance"
        # Reads: the un-stamped call's A3 label check (kw@1), then the pin (kw@3)
        # — and nothing more (an honoured pin needs no second read).
        assert [c.args for c in env["get_version"].await_args_list] == [("kw", 1), ("kw", 3)]
        assert meta["prompt_pin"] == "honoured"
        # …and records it on the completed row.
        assert _completed_states(env["db"])[-1]["worker_config_revision"] == "kw@3"

        # 3. The worker's own slot was neither evicted nor rewritten; the pin
        #    lives in a slot of its own.
        assert cache._workers[own_key] is own_instance
        assert own_instance.config.system_prompt == OWN
        assert len(cache._workers) == 2

        # 4. The next UN-stamped call runs the worker's own prompt, on the
        #    very instance it ran on before.
        after = _send(client)
        assert after.status_code == 200
        assert after.json()["output"] == f"ran with: {OWN}"
        assert after.json()["metadata"]["worker_config_revision"] == "kw@1"
        assert after.json()["metadata"]["system_prompt_sha256"] == sha256_text(OWN)
        assert after.json()["metadata"]["prompt_pin"] == "ignored:no_stamp"
        assert own_instance.run.await_count == 2
        # Flag on: the verified (A3) revision is recorded on every completed row.
        assert _completed_states(env["db"])[-1]["worker_config_revision"] == "kw@1"

    def test_a_canary_version_is_runnable(self, client, env, flag):
        flag(True)
        env["ledger"][("kw", 3)] = _ledger_row("CANARY")
        resp = _send(client, **STAMP)
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["worker_config_revision"] == "kw@3"

    @pytest.mark.parametrize("status", ["DRAFT", "ARCHIVED", "DISCARDED"])
    def test_a_non_runnable_pin_is_409_before_anything_runs(self, client, env, flag, status):
        flag(True)
        env["ledger"][("kw", 3)] = _ledger_row(status)
        resp = _send(client, **STAMP)
        assert resp.status_code == 409, resp.text
        # A4: the stable body, at the TOP level (not the error envelope, not detail).
        assert resp.json() == {
            "error": "prompt_version_not_runnable",
            "status": status,
            "prompt_id": "kw",
            "version": 3,
        }
        # Refused before the processing row and before any agent was built.
        env["db"].add_worker_state.assert_not_awaited()
        assert env["cache"]._workers == {}

    def test_an_unpublished_pin_is_404_with_a_top_level_body(self, client, env, flag):
        flag(True)
        env["ledger"][("kw", 3)] = None
        resp = _send(client, **STAMP)
        assert resp.status_code == 404
        # The whole body, at the TOP level — not under ``detail``, not the envelope.
        assert resp.json() == {
            "error": "prompt_version_not_found",
            "prompt_id": "kw",
            "version": 3,
        }
        env["db"].add_worker_state.assert_not_awaited()
        assert env["cache"]._workers == {}

    def test_the_worker_not_found_404_is_unchanged(self, client, env, flag):
        """A missing WORKER keeps its ordinary error envelope, so a consumer can
        tell it apart from a missing prompt VERSION by ``body["error"]``."""
        flag(True)
        env["registry"].get_worker = AsyncMock(return_value=None)
        resp = _send(client, require_existing=True, **STAMP)
        assert resp.status_code == 404
        body = resp.json()
        assert body.get("error") != "prompt_version_not_found"
        assert "prompt_id" not in body
        assert "does not exist and was NOT auto-created" in resp.text
        env["get_version"].assert_not_awaited()

    def test_a_pin_plus_a_per_message_system_prompt_runs_the_override(self, client, env, flag):
        """movies-70: never a 422. The caller's explicit override runs, the pin
        is not applied (no ledger read at all), and the revision is the truthful
        ``unversioned:<sha12>`` of what ran."""
        flag(True)
        override = "something else"
        resp = _send(client, system_prompt=override, **STAMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["output"] == f"ran with: {override}"
        meta = body["metadata"]
        assert meta["prompt_pin"] == "ignored:message_override"
        assert meta["worker_config_revision"] == f"unversioned:{sha256_text(override)[:12]}"
        assert (meta["prompt_id"], meta["prompt_version"]) == (None, None)
        assert meta["system_prompt_sha256"] == sha256_text(override)
        assert _completed_states(env["db"])[-1]["worker_config_revision"] == (
            f"unversioned:{sha256_text(override)[:12]}"
        )
        env["get_version"].assert_not_awaited()

    @pytest.mark.parametrize("status", ["DRAFT", None])
    def test_the_override_wins_even_over_an_unrunnable_or_missing_pin(
        self, client, env, flag, status
    ):
        flag(True)
        env["ledger"][("kw", 3)] = _ledger_row(status) if status else None
        resp = _send(client, system_prompt="override", **STAMP)
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["prompt_pin"] == "ignored:message_override"

    def test_flag_off_a_pin_plus_a_per_message_system_prompt_keeps_the_flag_off_rule(
        self, client, env, flag
    ):
        flag(False)
        plain = _send(client, system_prompt="override")
        stamped = _send(client, system_prompt="override", **STAMP)
        assert stamped.status_code == plain.status_code == 200
        assert "prompt_pin" not in plain.json()["metadata"]
        stamped_body = stamped.json()
        assert stamped_body["metadata"].pop("prompt_pin") == "ignored:flag_off"
        # No revision change: identical to the un-stamped override call.
        assert stamped_body == plain.json()
        env["get_version"].assert_not_awaited()

    def test_a_stamp_for_another_prompt_is_ignored(self, client, env, flag):
        flag(True)
        resp = _send(client, prompt_versions={"some-other-prompt": 3})
        assert resp.status_code == 200
        assert resp.json()["output"] == f"ran with: {OWN}"
        assert resp.json()["metadata"]["worker_config_revision"] == "kw@1"
        assert resp.json()["metadata"]["prompt_pin"] == "ignored:no_matching_prompt"
        # The only ledger read is the A3 label check of the worker's own kw@1.
        env["get_version"].assert_awaited_once_with("kw", 1)

    def test_a_worker_without_a_revision_label_ignores_the_stamp(self, client, env, flag):
        flag(True)
        data = _worker_data()
        del data["config_revision"]
        with patch(f"{_MP}.worker_registry.get_worker", AsyncMock(return_value=data)):
            resp = _send(client, **STAMP)
        assert resp.status_code == 200
        assert resp.json()["output"] == f"ran with: {OWN}"
        # No label → nothing to verify: the revision is unversioned (A3).
        meta = resp.json()["metadata"]
        assert meta["worker_config_revision"] == f"unversioned:{sha256_text(OWN)[:12]}"
        assert meta["prompt_pin"] == "label_content_mismatch"
        env["get_version"].assert_not_awaited()


class TestMultiEntryStamp:
    """A1: entries naming prompts this worker does not run are IGNORED, never 422."""

    def test_a_matching_entry_among_others_is_honoured(self, client, env, flag):
        flag(True)
        resp = _send(client, prompt_versions={"kw": 3, "other-prompt": 9, "third": 1})
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"] == f"ran with: {PINNED}"
        assert resp.json()["metadata"]["prompt_pin"] == "honoured"
        env["get_version"].assert_awaited_once_with("kw", 3)

    def test_a_non_matching_map_plus_a_system_prompt_proceeds(self, client, env, flag):
        flag(True)
        resp = _send(
            client, system_prompt="caller prompt", prompt_versions={"other-prompt": 9, "x": 2}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"] == "ran with: caller prompt"
        meta = resp.json()["metadata"]
        # The per-message prompt is not kw@1's content: the label is NOT echoed (A3).
        assert meta["worker_config_revision"] == (
            f"unversioned:{sha256_text('caller prompt')[:12]}"
        )
        assert meta["prompt_pin"] == "label_content_mismatch"


class TestTruthfulRevision:
    """A3: the label is reported only when the content that ran IS that version."""

    def test_a_label_whose_ledger_content_differs_is_unversioned(self, client, env, flag):
        flag(True)
        env["ledger"][("kw", 1)] = _ledger_row("PUBLISHED", version=1, system="NOT OWN")
        resp = _send(client)
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["worker_config_revision"] == f"unversioned:{sha256_text(OWN)[:12]}"
        assert (meta["prompt_id"], meta["prompt_version"]) == (None, None)
        assert meta["prompt_pin"] == "label_content_mismatch"
        assert meta["system_prompt_sha256"] == sha256_text(OWN)
        assert _completed_states(env["db"])[-1]["worker_config_revision"] == (
            f"unversioned:{sha256_text(OWN)[:12]}"
        )

    def test_a_label_missing_from_the_ledger_is_unversioned(self, client, env, flag):
        flag(True)
        env["ledger"][("kw", 1)] = None
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"].startswith("unversioned:")
        assert meta["prompt_pin"] == "label_content_mismatch"

    def test_a_ledger_read_failure_is_unversioned_not_a_500(self, client, env, flag):
        flag(True)
        env["get_version"].side_effect = RuntimeError("db down")
        resp = _send(client)
        assert resp.status_code == 200
        assert resp.json()["metadata"]["prompt_pin"] == "label_content_mismatch"

    def test_a_verified_label_is_echoed(self, client, env, flag):
        flag(True)
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"] == "kw@1"
        assert (meta["prompt_id"], meta["prompt_version"]) == ("kw", 1)
        assert meta["prompt_pin"] == "ignored:no_stamp"

    def test_flag_off_keeps_the_label_echo_and_adds_nothing(self, client, env, flag):
        flag(False)
        env["ledger"][("kw", 1)] = _ledger_row("PUBLISHED", version=1, system="NOT OWN")
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"] == "kw@1"
        assert "prompt_pin" not in meta
        assert "worker_config_revision" not in _completed_states(env["db"])[-1]
        env["get_version"].assert_not_awaited()


class TestRetryPathCarriesTheVerdict:
    def test_a_recovered_retry_reports_prompt_pin_and_the_pinned_revision(self, client, env, flag):
        """A2 says EVERY execution response — the recreate-and-retry return too."""
        flag(True)
        from src.api.workers import message_endpoints

        real_run = message_endpoints._run_worker
        calls = {"n": 0}

        async def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection refused")
            return await real_run(*args, **kwargs)

        with (
            patch(f"{_MP}._run_worker", _flaky),
            patch(f"{_MP}._backoff_delay", return_value=0.0),
        ):
            resp = _send(client, **STAMP)
        assert resp.status_code == 200, resp.text
        assert calls["n"] == 2
        meta = resp.json()["metadata"]
        assert meta["prompt_pin"] == "honoured"
        assert meta["worker_config_revision"] == "kw@3"
        assert _completed_states(env["db"])[-1]["worker_config_revision"] == "kw@3"


class TestOnePinErrorShape:
    def test_the_409_and_both_404s_share_one_top_level_shape(self, client, env, flag):
        """movies-70: every pin error — the A4 409, the message-path 404 and the
        swap-path 404 — is ``{"error", "prompt_id", "version"}`` at the TOP level,
        the 409 adding only ``status``."""
        flag(True)
        env["ledger"][("kw", 3)] = _ledger_row("DRAFT")
        not_runnable = _send(client, **STAMP)
        env["ledger"][("kw", 3)] = None
        msg_missing = _send(client, **STAMP)
        with patch("src.api.workers.crud_endpoints.worker_registry") as crud_registry:
            crud_registry.get_worker = AsyncMock(side_effect=lambda **_kw: _worker_data())
            crud_registry.swap_prompt = AsyncMock()
            swap_missing = client.put(f"/workers/{WORKER}/prompt", json={"prompt_version": "kw@3"})
            crud_registry.swap_prompt.assert_not_called()

        assert (not_runnable.status_code, msg_missing.status_code, swap_missing.status_code) == (
            409,
            404,
            404,
        )
        base = {"error", "prompt_id", "version"}
        assert set(msg_missing.json()) == set(swap_missing.json()) == base
        assert set(not_runnable.json()) == base | {"status"}
        assert (
            msg_missing.json()
            == swap_missing.json()
            == {
                "error": "prompt_version_not_found",
                "prompt_id": "kw",
                "version": 3,
            }
        )
        assert not_runnable.json()["error"] == "prompt_version_not_runnable"
        for resp in (not_runnable, msg_missing, swap_missing):
            assert isinstance(resp.json()["version"], int)
            assert isinstance(resp.json()["prompt_id"], str)


class TestFlagOffIsByteIdentical:
    @pytest.mark.parametrize("status", ["PUBLISHED", "DRAFT"])
    def test_the_stamp_is_ignored(self, client, env, flag, status):
        """Flag off: same response bytes, same state row, same slot, no ledger
        read — even for a stamp that would be REFUSED with the flag on."""
        flag(False)
        env["ledger"][("kw", 3)] = _ledger_row(status)
        plain = _send(client)
        stamped = _send(client, **STAMP)
        assert stamped.status_code == plain.status_code == 200
        # A2 overrides byte-identity for ONE key: a stamped flag-off call says
        # ``prompt_pin: ignored:flag_off``; an un-stamped one adds nothing.
        assert "prompt_pin" not in plain.json()["metadata"]
        stamped_body = stamped.json()
        assert stamped_body["metadata"].pop("prompt_pin") == "ignored:flag_off"
        assert stamped_body == plain.json()
        states = _completed_states(env["db"])
        assert states[0] == states[1]
        assert len(env["cache"]._workers) == 1
        env["get_version"].assert_not_awaited()


class TestFingerprint:
    def test_an_unpinned_request_keeps_its_exact_key(self):
        """``prompt_pin=None`` must hash exactly as before — the key of every
        existing slot is unchanged."""
        from src.api.workers.message_endpoints import _override_cache_fingerprint
        from src.api.workers.models import MessageRequest

        plain = MessageRequest(message="m")
        prompted = MessageRequest(message="m", instruction="x")
        assert _override_cache_fingerprint(plain) is None
        assert _override_cache_fingerprint(plain, None) is None
        assert _override_cache_fingerprint(prompted, None) == _override_cache_fingerprint(prompted)
        assert _override_cache_fingerprint(plain, "kw@3") not in (None, "")
        assert _override_cache_fingerprint(plain, "kw@3") != _override_cache_fingerprint(
            plain, "kw@4"
        )


class TestStampValidation:
    @pytest.mark.parametrize(
        "stamp", [{"../x": 1}, {"kw": 0}, {"kw": -1}, {"kw": "three"}, {"kw": True}]
    )
    def test_a_malformed_stamp_is_422(self, client, env, stamp):
        assert _send(client, prompt_versions=stamp).status_code == 422
        env["get_version"].assert_not_awaited()


class TestHealth:
    @pytest.mark.parametrize("on", [True, False])
    def test_health_reports_the_live_flag(self, client, flag, on):
        flag(on)
        body = client.get("/health").json()
        assert body["prompts_read_requires_published"] is on
        assert body["status"] == "healthy"

    @pytest.mark.parametrize("state", ["ready", "missing_031", "unknown"])
    def test_health_reports_the_cached_031_probe(self, client, state):
        from src.api.prompts import repository

        before = repository.lifecycle_schema_state()
        repository.set_lifecycle_schema_state(state)
        try:
            assert client.get("/health").json()["prompt_lifecycle_schema"] == state
        finally:
            repository.set_lifecycle_schema_state(before)
