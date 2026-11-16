"""U6: a platform worker runs its bound PUBLISHED prompt version (migration 034).

What is proved here, and against what:

* The worker cache is the REAL :class:`WorkerInstanceCache`; only the agent
  BUILDER is stubbed, building from ``worker_data["config"]`` exactly as
  ``_create_worker_instance`` does — so "the published text ran" is measured on
  the instance that ran, not on a mock's call args.
* The binding read is stubbed at ``prompt_source.get_pg_connection`` and the
  ledger at ``prompt_repository.get_active_on`` (the resolver reads on the
  binding read's connection, so the config mirror can reuse it inside a
  transition's transaction) / ``get_version``. Migration 034's
  table itself is proved against real Postgres in
  ``tests/integration/test_migration_034_worker_prompt_binding.py``.

The worker's config carries ``CONFIG``; it is bound to prompt ``p`` whose
PUBLISHED v1 is ``V1``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.prompts import repository as prompt_repository
from src.api.prompts.hashing import sha256_text
from src.api.worker_management_api import app
from src.api.workers import prompt_source
from src.api.workers._globals import WorkerInstanceCache
from src.auth import get_tenant_id
from src.settings import get_settings, reset_settings_cache
from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

PLATFORM_TENANT = "aaaaaaaa-0000-4000-8000-000000000003"  # george-dev / EXECUTION
CUSTOMER_TENANT = "aaaaaaaa-0000-4000-8000-000000000100"  # cdmsfirst
WORKER = "worker-src-test"
WORKER_ID = "wid-src-1"
_MP = "src.api.workers.message_endpoints"

CONFIG = "You are the CONFIG prompt."
REWRITTEN = "You are what the n8n ensure-node wrote last."
V1 = "You are prompt p, published version 1."
V2 = "You are prompt p, published version 2."
CANARY_TEXT = "You are prompt p, CANARY version 2."
_FLAG_VARS = ("WM_PROMPT_SOURCE", "WM_PROMPT_CANARY_ROUTABLE", "PROMPTS_READ_REQUIRES_PUBLISHED")


@asynccontextmanager
async def _noop_ctx(*_a, **_kw):
    yield "fake-lock"


def _worker_data(system_prompt: str = CONFIG) -> dict[str, Any]:
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
            "system_prompt": system_prompt,
        },
        "status": "active",
    }


def _ledger_row(version: int, system: str, status: str = "PUBLISHED") -> dict[str, Any]:
    return {
        "prompt_id": "p",
        "version": version,
        "status": status,
        "system": system,
        "system_prompt_sha256": sha256_text(system),
    }


def _sample(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


@pytest.fixture
def settings(monkeypatch):
    def _set(
        *, source: str = "published", canary_routable: bool = False, pins: bool = False
    ) -> None:
        monkeypatch.setenv("WM_PROMPT_SOURCE", source)
        monkeypatch.setenv("WM_PROMPT_CANARY_ROUTABLE", "true" if canary_routable else "false")
        monkeypatch.setenv("PROMPTS_READ_REQUIRES_PUBLISHED", "true" if pins else "false")
        reset_settings_cache()
        prompt_source.clear_cache()

    yield _set
    for var in _FLAG_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_settings_cache()
    prompt_source.clear_cache()


@pytest.fixture
def env():
    cache = WorkerInstanceCache()
    order: list[str] = []

    async def _build(_self, worker_name, worker_data, **_kw):
        config = worker_data.get("config") or {}
        instance = MagicMock()
        run_result = MagicMock()
        run_result.output = f"ran with: {config.get('system_prompt')}"
        run_result.data = run_result.output
        run_result.usage_metrics = None
        run_result.session_id = None
        run_result.cost_usd = None
        run_result.duration_ms = None

        async def _run(*_a, **_k):
            order.append("llm_call")
            return run_result

        instance.run = AsyncMock(side_effect=_run)
        instance.config = PydanticAIAgentConfig(
            name=worker_name,
            model_provider="mlx",
            model="mlx",
            db_path="/tmp/test.db",
            system_prompt=config.get("system_prompt"),
        )
        del instance.system_prompt
        del instance.check_health
        return instance

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"prompt_id": "p", "owner_tenant_id": PLATFORM_TENANT})

    @asynccontextmanager
    async def _pg():
        yield conn

    attribution = MagicMock(side_effect=lambda *_a, **_k: order.append("attribution"))

    with (
        patch(f"{_MP}.worker_registry") as registry,
        patch(f"{_MP}.worker_cache", cache),
        patch(f"{_MP}.state_manager") as state_mgr,
        patch(f"{_MP}.StateDatabase") as state_db_cls,
        patch(f"{_MP}.set_prompt_attribution", attribution),
        patch.object(WorkerInstanceCache, "_create_worker_instance", _build),
        patch.object(prompt_source, "get_pg_connection", _pg),
        patch.object(prompt_repository, "get_active_on", new_callable=AsyncMock) as get_active,
        patch.object(prompt_repository, "get_version", new_callable=AsyncMock) as get_version,
    ):
        state_mgr.atomic_worker_operation = _noop_ctx
        state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        state_mgr.validate_state_transition = MagicMock(return_value=True)
        db = AsyncMock()
        db.get_latest_worker_state = AsyncMock(return_value=None)
        db.add_worker_state = AsyncMock(return_value=1)
        state_db_cls.return_value = db
        registry.get_worker = AsyncMock(side_effect=lambda **_kw: _worker_data())
        get_active.return_value = _ledger_row(1, V1)
        ledger: dict[tuple[str, int], dict[str, Any] | None] = {("p", 1): _ledger_row(1, V1)}
        get_version.side_effect = lambda pid, ver: ledger.get((pid, ver))
        yield {
            "attribution": attribution,
            "conn": conn,
            "db": db,
            "get_active": get_active,
            "get_version": get_version,
            "ledger": ledger,
            "order": order,
            "registry": registry,
        }


def _client_for(tenant: str):
    app.dependency_overrides[get_tenant_id] = lambda: tenant

    @asynccontextmanager
    async def _fake_lifespan(_a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    return original


@pytest.fixture
def client(env):
    original = _client_for(PLATFORM_TENANT)
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


@pytest.fixture
def customer_client(env):
    original = _client_for(CUSTOMER_TENANT)
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _send(client, **extra):
    return client.post(f"/workers/{WORKER}/message", json={"message": "Hello", **extra})


def _completed_metadata(db) -> dict[str, Any]:
    rows = [
        c.kwargs["metadata"]
        for c in db.add_worker_state.await_args_list
        if c.kwargs.get("state_type") == "completed"
    ]
    return rows[-1]


class TestPublishedSource:
    def test_a_bound_platform_worker_runs_the_published_version(self, client, env, settings):
        """(a) flag published + platform tenant + binding + PUBLISHED v1."""
        settings()
        resp = _send(client)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["output"] == f"ran with: {V1}"
        meta = body["metadata"]
        assert meta["prompt_version_id"] == "p@1"
        assert meta["prompt_source"] == "published"
        assert meta["prompt_body_sha256"] == sha256_text(V1)
        # The digest echo describes the same text (one instance-first read).
        assert meta["system_prompt_sha256"] == sha256_text(V1)
        assert meta["worker_config_revision"] == "p@1"
        # The binding was read for (tenant, worker), and only PUBLISHED resolved.
        env["conn"].fetchrow.assert_awaited_once()
        assert env["conn"].fetchrow.await_args.args[1:] == (PLATFORM_TENANT, WORKER)
        env["get_active"].assert_awaited_once_with(env["conn"], "p", highest_published=True)
        # The worker_states row carries the same attribution.
        row = _completed_metadata(env["db"])
        assert row["prompt_version_id"] == "p@1"
        assert row["prompt_body_sha256"] == sha256_text(V1)
        assert row["prompt_source"] == "published"
        # The outbound hook was told BEFORE the LLM call.
        env["attribution"].assert_called_once_with("p@1", sha256_text(V1))
        assert env["order"] == ["attribution", "llm_call"]

    def test_the_second_writer_cannot_override_the_published_version(self, client, env, settings):
        """(b) n8n ensure-node rewrites config.system_prompt: v1 still runs."""
        settings()
        env["registry"].get_worker = AsyncMock(
            side_effect=lambda **_kw: _worker_data(system_prompt=REWRITTEN)
        )
        body = _send(client).json()
        assert body["output"] == f"ran with: {V1}"
        assert body["metadata"]["prompt_body_sha256"] == sha256_text(V1)
        assert body["metadata"]["prompt_body_sha256"] != sha256_text(REWRITTEN)
        assert body["metadata"]["prompt_version_id"] == "p@1"

    def test_a_new_release_runs_after_the_cache_clears(self, client, env, settings):
        """(c) publish v2 → the next resolution runs v2, in a fresh slot."""
        settings()
        first = _send(client).json()
        assert first["output"] == f"ran with: {V1}"

        env["get_active"].return_value = _ledger_row(2, V2)
        env["ledger"][("p", 2)] = _ledger_row(2, V2)
        # Inside the TTL the cached v1 still answers…
        assert _send(client).json()["metadata"]["prompt_version_id"] == "p@1"
        # …and after it the release is picked up.
        prompt_source.clear_cache()
        second = _send(client).json()
        assert second["output"] == f"ran with: {V2}"
        assert second["metadata"]["prompt_version_id"] == "p@2"
        assert second["metadata"]["prompt_body_sha256"] == sha256_text(V2)

    def test_a_new_release_runs_once_the_ttl_expires_without_clearing(self, client, env, settings):
        """(c') NO clear_cache: the resolver's clock passing the TTL alone picks up v2.

        The test above clears the cache by hand, so a TTL that never expired
        would still pass it. Here only ``prompt_source.time`` (the clock the
        resolver reads) moves.
        """
        settings()
        ttl = get_settings().wm_prompt_resolve_ttl_s
        assert ttl > 0
        clock = MagicMock()
        clock.monotonic.return_value = 1000.0
        with patch.object(prompt_source, "time", clock):
            first = _send(client).json()
            assert first["metadata"]["prompt_version_id"] == "p@1"
            assert env["get_active"].await_count == 1

            env["get_active"].return_value = _ledger_row(2, V2)
            env["ledger"][("p", 2)] = _ledger_row(2, V2)
            # Just short of the TTL: the cached v1 answers, no new read.
            clock.monotonic.return_value = 1000.0 + ttl - 0.001
            cached = _send(client).json()
            assert cached["output"] == f"ran with: {V1}"
            assert cached["metadata"]["prompt_version_id"] == "p@1"
            assert cached["metadata"]["prompt_body_sha256"] == sha256_text(V1)
            assert env["get_active"].await_count == 1

            # Past the TTL: re-resolved, v2 runs.
            clock.monotonic.return_value = 1000.0 + ttl + 0.001
            second = _send(client).json()
        assert env["get_active"].await_count == 2
        assert second["output"] == f"ran with: {V2}"
        assert second["metadata"]["prompt_version_id"] == "p@2"
        assert second["metadata"]["prompt_body_sha256"] == sha256_text(V2)

    def test_a_customer_tenant_is_never_resolved(self, customer_client, env, settings):
        """(d) a binding row exists, but the tenant gate runs before any read."""
        settings()
        body = _send(customer_client).json()
        assert body["output"] == f"ran with: {CONFIG}"
        assert body["metadata"]["prompt_source"] == "legacy_config"
        assert body["metadata"]["prompt_version_id"] is None
        env["conn"].fetchrow.assert_not_awaited()
        env["get_active"].assert_not_awaited()

    def test_flag_legacy_runs_the_config_text(self, client, env, settings):
        """(e) default flag: today's behaviour, plus the attribution keys."""
        settings(source="legacy")
        body = _send(client).json()
        assert body["output"] == f"ran with: {CONFIG}"
        meta = body["metadata"]
        assert meta["prompt_version_id"] is None
        assert meta["prompt_source"] == "legacy_config"
        assert meta["prompt_body_sha256"] == sha256_text(CONFIG)
        assert meta["worker_config_revision"] is None
        env["conn"].fetchrow.assert_not_awaited()
        row = _completed_metadata(env["db"])
        assert row == {
            "prompt_version_id": None,
            "prompt_body_sha256": sha256_text(CONFIG),
            "prompt_source": "legacy_config",
            "canary_selected_by": None,
        }
        env["attribution"].assert_called_once_with(None, sha256_text(CONFIG))

    def test_a_resolver_db_error_falls_back_to_the_config_prompt(self, client, env, settings):
        """(g) fail-safe: no exception, legacy text, counted."""
        settings()
        env["conn"].fetchrow = AsyncMock(side_effect=RuntimeError("relation does not exist"))
        before = _sample("wm_prompt_resolve_errors_total")
        resp = _send(client)
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"] == f"ran with: {CONFIG}"
        assert resp.json()["metadata"]["prompt_source"] == "legacy_config"
        assert _sample("wm_prompt_resolve_errors_total") == before + 1

    @pytest.mark.parametrize(
        "owner",
        ["aaaaaaaa-0000-4000-8000-000000000001", None],
        ids=["owned_by_another_platform_tenant", "no_owner"],
    )
    def test_a_prompt_owned_by_another_tenant_is_refused(self, client, env, settings, owner):
        """035 defense in depth: tenant A never runs a prompt owned by tenant B,
        even if a binding row says so (the FK should make the row impossible)."""
        settings()
        env["conn"].fetchrow = AsyncMock(return_value={"prompt_id": "p", "owner_tenant_id": owner})
        before = _sample("wm_prompt_binding_cross_tenant_refused_total")
        errors_before = _sample("wm_prompt_resolve_errors_total")
        resp = _send(client)
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"] == f"ran with: {CONFIG}"
        assert resp.json()["metadata"]["prompt_source"] == "legacy_config"
        assert _sample("wm_prompt_binding_cross_tenant_refused_total") == before + 1
        assert _sample("wm_prompt_resolve_errors_total") == errors_before, "a refusal, not an error"
        # The other tenant's ledger is never read.
        env["get_active"].assert_not_awaited()

    def test_the_owner_comparison_ignores_case(self, client, env, settings):
        settings()
        env["conn"].fetchrow = AsyncMock(
            return_value={"prompt_id": "p", "owner_tenant_id": PLATFORM_TENANT.upper()}
        )
        resp = _send(client)
        assert resp.json()["output"] == f"ran with: {V1}"

    def test_a_per_message_system_prompt_still_wins(self, client, env, settings):
        """(h) request.system_prompt outranks the published version, unchanged."""
        settings()
        body = _send(client, system_prompt="caller prompt").json()
        assert body["output"] == "ran with: caller prompt"
        assert body["metadata"]["prompt_source"] == "request"
        assert body["metadata"]["prompt_version_id"] is None
        assert body["metadata"]["prompt_body_sha256"] == sha256_text("caller prompt")
        env["get_active"].assert_not_awaited()


class TestCanaryHold:
    STAMP: ClassVar[dict[str, dict[str, int]]] = {"prompt_versions": {"p": 2}}

    def test_a_canary_pin_is_held_when_not_routable(self, client, env, settings):
        """(f) CANARY pin + routable off → PUBLISHED runs, canary_held, counted."""
        settings(pins=True, canary_routable=False)
        env["ledger"][("p", 2)] = _ledger_row(2, CANARY_TEXT, status="CANARY")
        before = _sample("wm_prompt_canary_held_total")
        resp = _send(client, **self.STAMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["output"] == f"ran with: {V1}"
        meta = body["metadata"]
        assert meta["prompt_pin"] == "canary_held"
        assert meta["prompt_source"] == "published"
        assert meta["prompt_version_id"] == "p@1"
        assert meta["worker_config_revision"] == "p@1"
        assert _sample("wm_prompt_canary_held_total") == before + 1
        row = _completed_metadata(env["db"])
        assert row["prompt_pin"] == "canary_held"
        assert row["prompt_version_id"] == "p@1"

    def test_a_canary_pin_runs_when_routable(self, client, env, settings):
        """(f) routable on → the CANARY runs and is reported as such."""
        settings(pins=True, canary_routable=True)
        env["ledger"][("p", 2)] = _ledger_row(2, CANARY_TEXT, status="CANARY")
        before = _sample("wm_prompt_canary_held_total")
        body = _send(client, **self.STAMP).json()
        assert body["output"] == f"ran with: {CANARY_TEXT}"
        meta = body["metadata"]
        assert meta["prompt_source"] == "canary"
        assert meta["canary_selected_by"] == "pin"  # "router" reserved for G-14
        assert meta["prompt_version_id"] == "p@2"
        assert meta["prompt_pin"] == "honoured"
        assert meta["prompt_body_sha256"] == sha256_text(CANARY_TEXT)
        assert _sample("wm_prompt_canary_held_total") == before

    def test_a_published_pin_is_reported_as_pin(self, client, env, settings):
        settings(pins=True)
        env["ledger"][("p", 2)] = _ledger_row(2, V2)
        body = _send(client, **self.STAMP).json()
        assert body["output"] == f"ran with: {V2}"
        assert body["metadata"]["prompt_source"] == "pin"
        assert body["metadata"]["prompt_version_id"] == "p@2"

    def test_legacy_source_never_holds_a_canary_pin(self, settings):
        """The hold is part of the published rollout: legacy keeps the P4 contract."""
        from src.api.workers.prompt_pin import PromptPin, canary_held

        settings(source="legacy")
        pin = PromptPin(prompt_id="p", version=2, status="CANARY", system=CANARY_TEXT)
        before = _sample("wm_prompt_canary_held_total")
        assert canary_held(pin) is False
        assert _sample("wm_prompt_canary_held_total") == before

    def test_a_customer_canary_pin_is_held_even_when_routable(self, settings):
        """Tenant separation: a non-platform tenant never runs a CANARY version."""
        from src.api.workers.prompt_pin import PromptPin, canary_held

        settings(canary_routable=True)
        pin = PromptPin(prompt_id="p", version=2, status="CANARY", system=CANARY_TEXT)
        before = _sample("wm_prompt_canary_held_total")
        assert canary_held(pin, CUSTOMER_TENANT) is True
        assert canary_held(pin, None) is True  # unknown tenant fails closed
        assert _sample("wm_prompt_canary_held_total") == before + 2
        # Control: the same pin under the same flags RUNS for a platform tenant.
        assert canary_held(pin, PLATFORM_TENANT) is False
        assert canary_held(pin, PLATFORM_TENANT.upper()) is False
        assert _sample("wm_prompt_canary_held_total") == before + 2


class TestCustomerPinIsForbidden:
    """Strict tenant separation (owner ruling 2026-09-30): a NON-platform
    tenant's stamp is never applied, PUBLISHED or CANARY, in either prompt
    source, and no ledger row is read for it. Not a 403: the customer's worker
    runs its own config text, and the call says ``tenant_forbidden``.

    The worker is LABELLED ``p@1`` and the stamp names ``p@2`` (PUBLISHED), so
    without the gate the pin WOULD apply — the platform control proves it."""

    STAMP: ClassVar[dict[str, dict[str, int]]] = {"prompt_versions": {"p": 2}}

    @staticmethod
    def _label_worker(env) -> None:
        def _labelled(**_kw) -> dict[str, Any]:
            data = _worker_data()
            data["config_revision"] = "p@1"
            return data

        env["registry"].get_worker = AsyncMock(side_effect=_labelled)
        env["ledger"][("p", 2)] = _ledger_row(2, V2)

    @pytest.mark.parametrize("source", ["legacy", "published"])
    @pytest.mark.parametrize("status", ["PUBLISHED", "CANARY"])
    def test_a_customer_pin_is_ignored_and_reads_no_ledger(
        self, customer_client, env, settings, source, status
    ):
        settings(source=source, pins=True, canary_routable=True)
        self._label_worker(env)
        env["ledger"][("p", 2)] = _ledger_row(2, V2, status=status)
        before = _sample("wm_prompt_pin_forbidden_total")
        resp = _send(customer_client, **self.STAMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["output"] == f"ran with: {CONFIG}"
        meta = body["metadata"]
        assert meta["prompt_pin"] == "tenant_forbidden"
        assert meta["prompt_version_id"] is None
        assert meta["prompt_source"] == "legacy_config"
        # The truthful revision names the content that ran, never a ledger version.
        assert meta["worker_config_revision"] == f"unversioned:{sha256_text(CONFIG)[:12]}"
        assert _sample("wm_prompt_pin_forbidden_total") == before + 1
        env["get_version"].assert_not_awaited()
        env["get_active"].assert_not_awaited()
        env["conn"].fetchrow.assert_not_awaited()
        row = _completed_metadata(env["db"])
        assert row["prompt_pin"] == "tenant_forbidden"
        assert row["prompt_version_id"] is None

    def test_a_customer_stamp_naming_another_prompt_is_also_forbidden(
        self, customer_client, env, settings
    ):
        """Any stamp counts, so a refused customer cannot probe which ids exist."""
        settings(source="legacy", pins=True)
        before = _sample("wm_prompt_pin_forbidden_total")
        body = _send(customer_client, prompt_versions={"no-such-prompt": 9}).json()
        assert body["output"] == f"ran with: {CONFIG}"
        assert body["metadata"]["prompt_pin"] == "tenant_forbidden"
        assert _sample("wm_prompt_pin_forbidden_total") == before + 1
        env["get_version"].assert_not_awaited()

    def test_control_the_same_pin_runs_for_a_platform_tenant(self, client, env, settings):
        settings(source="legacy", pins=True)
        self._label_worker(env)
        before = _sample("wm_prompt_pin_forbidden_total")
        body = _send(client, **self.STAMP).json()
        assert body["output"] == f"ran with: {V2}"
        assert body["metadata"]["prompt_pin"] == "honoured"
        assert body["metadata"]["prompt_version_id"] == "p@2"
        assert _sample("wm_prompt_pin_forbidden_total") == before

    def test_flag_off_keeps_the_flag_off_rule_and_counts_nothing(
        self, customer_client, env, settings
    ):
        settings(source="legacy", pins=False)
        self._label_worker(env)
        before = _sample("wm_prompt_pin_forbidden_total")
        body = _send(customer_client, **self.STAMP).json()
        assert body["output"] == f"ran with: {CONFIG}"
        assert body["metadata"]["prompt_pin"] == "ignored:flag_off"
        assert _sample("wm_prompt_pin_forbidden_total") == before
        env["get_version"].assert_not_awaited()

    def test_pin_forbidden_fails_closed_on_an_unknown_tenant(self, settings):
        from src.api.workers.prompt_pin import pin_forbidden

        settings(pins=True)
        stamp = {"p": 2}
        assert pin_forbidden(stamp, None) is True
        assert pin_forbidden(stamp, CUSTOMER_TENANT) is True
        assert pin_forbidden(stamp, PLATFORM_TENANT) is False
        assert pin_forbidden(stamp, PLATFORM_TENANT.upper()) is False
        assert pin_forbidden(None, CUSTOMER_TENANT) is False
        assert pin_forbidden({}, CUSTOMER_TENANT) is False


class TestResolveCacheBound:
    def test_the_cache_never_exceeds_its_cap(self, settings, monkeypatch):
        """``worker_name`` is caller-controlled: distinct names must not grow the cache."""
        import asyncio

        settings()
        monkeypatch.setattr(prompt_source, "CACHE_MAX_ENTRIES", 8)
        monkeypatch.setattr(prompt_source, "_read", AsyncMock(return_value=None))

        async def _drive() -> None:
            for i in range(50):
                await prompt_source.resolve_worker_prompt(PLATFORM_TENANT, f"worker-flood-{i}")
                assert len(prompt_source._cache) <= 8

        asyncio.run(_drive())
        assert len(prompt_source._cache) == 8
        # Oldest-first: only the 8 most recent names survive.
        assert {k[1] for k in prompt_source._cache} == {f"worker-flood-{i}" for i in range(42, 50)}
