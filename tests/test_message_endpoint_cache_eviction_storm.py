"""Regression tests for the per-message cache-eviction retry storm.

Two defects, one endpoint (``src/api/workers/message_endpoints.py``):

1. A request carrying a per-message override (``model_override`` /
   ``output_schema`` / ``max_tokens``) used to call
   ``worker_cache.remove_worker(...)``, which evicts by ``tenant:worker:``
   PREFIX. ``_cleanup_worker`` calls ``PydanticAIAgent.cleanup()``, which closes
   the model's ``AsyncOpenAI`` httpx pool. So one request closed the connection
   pool of every CONCURRENT sibling run on another variant of the same worker;
   those siblings raised ``APIConnectionError("Connection error.")``, which
   ``_is_connection_error`` scores transient, which dropped them into the retry
   ladder for a fault that was never theirs. The eviction was also redundant:
   all four overrides are already in ``_override_cache_fingerprint``, so an
   override request keys into its own cache slot regardless.

2. The retry ladder bounded the attempt COUNT and not the elapsed TIME.

These tests drive the real ``WorkerInstanceCache`` (only instance CREATION is
stubbed) so the cache-key arithmetic under test is the production one.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest
from openai import APIConnectionError

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.workers._globals import WorkerInstanceCache
from src.auth import get_tenant_id
from src.settings import get_settings, reset_settings_cache

TENANT = "tenant-evict-storm"
WORKER = "video_synthesis"
WORKER_ID = "wid-evict-1"
_MP = "src.api.workers.message_endpoints"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeHTTPClient:
    """Stand-in for ``AsyncOpenAI`` — the thing ``cleanup()`` actually closes."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeModel:
    def __init__(self) -> None:
        self.client = FakeHTTPClient()


class FakeAgent:
    """Agent whose ``run`` fails exactly the way a closed httpx pool fails.

    ``cleanup()`` mirrors ``PydanticAIAgent.cleanup`` (close the model's client),
    and ``run()`` mirrors the openai SDK, which raises ``APIConnectionError`` on
    a client that has been closed underneath an in-flight request.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.model = FakeModel()
        self.run_calls = 0
        self.cleanup_calls = 0
        # Optional coordination hooks, set by individual tests.
        self.entered_run: asyncio.Event | None = None
        self.release_run: asyncio.Event | None = None

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        await self.model.client.close()

    async def run(self, _message: Any) -> Any:
        self.run_calls += 1
        if self.entered_run is not None:
            self.entered_run.set()
        if self.release_run is not None:
            await self.release_run.wait()
        if self.model.client.closed:
            raise APIConnectionError(
                request=httpx.Request("POST", "http://inference-router.example.com/v1/chat")
            )
        resp = MagicMock()
        resp.output = f"ok:{self.label}"
        resp.data = resp.output
        resp.usage_metrics = None
        resp.session_id = None
        resp.cost_usd = None
        resp.duration_ms = None
        return resp


@asynccontextmanager
async def _noop_ctx(*_a: Any, **_kw: Any):
    yield "fake-lock"


def _worker_data() -> dict[str, Any]:
    return {
        "worker_name": WORKER,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/evict.db",
        "is_agent": True,
        "config": {
            "agent_type": "pydantic_ai",
            "model_provider": "inference_router",
            "model": "local-llm",
        },
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Fixture: real cache, stubbed instance creation
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Patch the endpoint's collaborators but keep a REAL WorkerInstanceCache."""
    cache = WorkerInstanceCache()
    created: list[FakeAgent] = []

    async def _create_instance(_self: Any, worker_name: str, *_a: Any, **_kw: Any) -> FakeAgent:
        agent = FakeAgent(f"{worker_name}#{len(created)}")
        created.append(agent)
        return agent

    with (
        patch(f"{_MP}.worker_cache", cache),
        patch(f"{_MP}.worker_registry") as registry,
        patch(f"{_MP}.state_manager") as state_mgr,
        patch(f"{_MP}.StateDatabase") as StateDB,
        patch(f"{_MP}.mlx_model_registry") as mlx,
        patch.object(WorkerInstanceCache, "_create_worker_instance", _create_instance),
    ):
        state_mgr.atomic_worker_operation = _noop_ctx
        state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        state_mgr.validate_state_transition = MagicMock(return_value=True)

        db = AsyncMock()
        db.get_latest_worker_state = AsyncMock(return_value=None)
        db.add_worker_state = AsyncMock(return_value=1)
        StateDB.return_value = db

        registry.get_worker = AsyncMock(side_effect=lambda *_a, **_kw: _worker_data())
        registry.register_worker = AsyncMock(return_value={"worker_id": WORKER_ID})

        mlx.get_model = AsyncMock(return_value={"model": "alt-model"})
        mlx.get_instance_for_model = AsyncMock(return_value="http://mlx-host:8080")

        yield {"cache": cache, "created": created, "state_db": db, "registry": registry}


@asynccontextmanager
async def _asgi_client():
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(_a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _post(client: httpx.AsyncClient, **extra: Any):
    return client.post(f"/workers/{WORKER}/message", json={"message": "Hello", **extra})


# ---------------------------------------------------------------------------
# THE load-bearing test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override_kwargs", "label"),
    [
        ({"output_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}, "schema"),
        ({"model_override": "alt-model"}, "model"),
        ({"max_tokens": 4096}, "tokens"),
    ],
)
async def test_override_request_does_not_break_a_concurrent_sibling(env, override_kwargs, label):
    """A per-message override must not close a concurrent run's http client.

    Timeline: the victim (no overrides) is parked inside ``agent.run()``; the
    override request then runs start-to-finish; the victim is released. Before
    the fix the override request evicted by prefix, ``cleanup()`` closed the
    victim's pool, and the victim's in-flight call raised
    ``APIConnectionError`` -> transient -> retry ladder.
    """
    backoff_calls: list[int] = []

    def _spy_backoff(attempt: int, **_kw: Any) -> float:
        backoff_calls.append(attempt)
        return 0.0

    async with _asgi_client() as client:
        # Prime the base cache slot so victim + override request start from a
        # genuinely SHARED cache, exactly as the live pod does.
        prime = await _post(client)
        assert prime.status_code == 200, prime.text
        assert len(env["created"]) == 1, "priming call must have created the shared instance"
        victim_agent = env["created"][0]

        victim_agent.entered_run = asyncio.Event()
        victim_agent.release_run = asyncio.Event()

        with patch(f"{_MP}._backoff_delay", _spy_backoff):
            victim = asyncio.create_task(_post(client))
            await asyncio.wait_for(victim_agent.entered_run.wait(), timeout=5.0)

            # Non-vacuity: the victim is genuinely mid-run on the SHARED instance.
            assert victim_agent.run_calls == 2, (
                "victim must be running on the primed shared agent, "
                f"run_calls={victim_agent.run_calls}"
            )
            assert not victim_agent.model.client.closed

            evictor = await _post(client, **override_kwargs)
            assert evictor.status_code == 200, evictor.text

            victim_agent.release_run.set()
            victim_resp = await asyncio.wait_for(victim, timeout=10.0)

    # The actual defect, asserted first so a regression reports the CAUSE.
    assert not victim_agent.model.client.closed, (
        f"[{label}] the override request closed the concurrent run's httpx client"
    )
    assert victim_agent.cleanup_calls == 0, (
        f"[{label}] concurrent run's agent was cleaned up mid-flight "
        f"({victim_agent.cleanup_calls} cleanup calls)"
    )
    assert backoff_calls == [], (
        f"[{label}] innocent concurrent run entered the transient-retry ladder: "
        f"attempts={backoff_calls}"
    )
    assert victim_agent.run_calls == 2, (
        f"[{label}] concurrent run was re-executed by the retry ladder "
        f"(run_calls={victim_agent.run_calls}, expected 2: prime + victim)"
    )
    assert victim_resp.status_code == 200, victim_resp.text
    assert victim_resp.json()["output"] == f"ok:{victim_agent.label}"

    # Non-vacuity floor: the override really did take a DIFFERENT cache slot and
    # both slots are live — i.e. this test exercised two concurrent instances and
    # the override was still honoured after the eviction was removed.
    keys = list(env["cache"]._workers)
    assert len(keys) == 2, f"expected a base slot + an override slot, got {keys}"
    assert any(":ov-" in k for k in keys), f"override never got its own slot: {keys}"
    assert len(env["created"]) == 2, "override request must have built its own instance"


@pytest.mark.asyncio
async def test_override_request_reuses_its_own_slot_across_calls(env):
    """The override slot is a cache slot, not a rebuild-every-time path.

    Guards the other half of deleting the eviction: two identical override
    requests must share one instance (that is what makes the eviction
    redundant), while a plain request keeps the base instance.
    """
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    async with _asgi_client() as client:
        assert (await _post(client)).status_code == 200
        assert (await _post(client, output_schema=schema)).status_code == 200
        assert (await _post(client, output_schema=schema)).status_code == 200
        assert (await _post(client)).status_code == 200

    assert len(env["created"]) == 2, (
        "expected exactly 2 instances (base + one override slot), "
        f"got {[a.label for a in env['created']]}"
    )
    base, override = env["created"]
    assert base.run_calls == 2, f"base slot not reused: run_calls={base.run_calls}"
    assert override.run_calls == 2, f"override slot not reused: run_calls={override.run_calls}"
    assert base.cleanup_calls == 0, "base instance was evicted by an override request"


@pytest.mark.asyncio
async def test_remove_instance_evicts_only_the_callers_slot():
    """``remove_instance`` is the scoped counterpart of the prefix-wide ``remove_worker``."""
    cache = WorkerInstanceCache()
    created: list[FakeAgent] = []

    async def _create_instance(_self: Any, worker_name: str, *_a: Any, **_kw: Any) -> FakeAgent:
        agent = FakeAgent(f"{worker_name}#{len(created)}")
        created.append(agent)
        return agent

    with patch.object(WorkerInstanceCache, "_create_worker_instance", _create_instance):
        base = await cache.get_or_create_worker(WORKER, _worker_data(), tenant_id=TENANT)
        ov = await cache.get_or_create_worker(
            WORKER, _worker_data(), tenant_id=TENANT, override_key="deadbeef"
        )
        # Non-vacuity: two distinct live slots before we evict anything.
        assert base is not ov
        assert len(cache._workers) == 2, list(cache._workers)

        await cache.remove_instance(
            WORKER, _worker_data(), tenant_id=TENANT, override_key="deadbeef"
        )

    assert ov.cleanup_calls == 1, "target slot was not cleaned up"
    assert ov.model.client.closed is True
    assert base.cleanup_calls == 0, "scoped eviction touched the sibling slot"
    assert base.model.client.closed is False, "scoped eviction closed the sibling's http client"
    assert list(cache._workers) == [f"{TENANT}:{WORKER}:_"], list(cache._workers)


def test_llm_retry_deadline_setting_exists_with_a_sane_default():
    """The cap must be configurable and default to something below the agent ladder."""
    reset_settings_cache()
    s = get_settings()
    assert s.llm_retry_deadline_s == 600.0
    # Non-vacuity: the deadline is meaningfully tighter than the unbounded ladder it
    # replaces (max_attempts x the per-attempt agent cap).
    unbounded = s.llm_retry_max_attempts * s.agent_run_timeout_s
    assert s.llm_retry_deadline_s < unbounded


# ---------------------------------------------------------------------------
# Retry-ladder wall-clock cap
# ---------------------------------------------------------------------------


@pytest.fixture
def slow_env(env):
    """``env`` with an agent whose run always fails transiently after a delay."""
    per_attempt_s = 0.25

    async def _slow_failing_run(_message: Any) -> Any:
        await asyncio.sleep(per_attempt_s)
        raise APIConnectionError(
            request=httpx.Request("POST", "http://inference-router.example.com/v1/chat")
        )

    async def _create_instance(_self: Any, worker_name: str, *_a: Any, **_kw: Any) -> FakeAgent:
        agent = FakeAgent(f"{worker_name}#slow{len(env['created'])}")
        agent.run = _slow_failing_run  # type: ignore[method-assign]
        env["created"].append(agent)
        return agent

    with patch.object(WorkerInstanceCache, "_create_worker_instance", _create_instance):
        yield {**env, "per_attempt_s": per_attempt_s}


@pytest.mark.asyncio
async def test_retry_ladder_stops_at_the_wall_clock_deadline(slow_env, monkeypatch):
    """A 7-deep ladder must stop on elapsed time, not run all 6 retries."""
    deadline_s = 0.4  # < per-attempt 0.25s x 6 retries, so the count cannot be the bound
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("LLM_RETRY_DEADLINE_S", str(deadline_s))
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_JITTER_RATIO", "0.0")
    reset_settings_cache()
    try:
        settings = get_settings()
        # Non-vacuity: the ladder really is configured 7 deep for this run.
        assert settings.llm_retry_max_attempts == 7
        max_retries = settings.llm_retry_max_attempts - 1

        async with _asgi_client() as client:
            resp = await _post(client)
    finally:
        reset_settings_cache()

    attempts = len(slow_env["created"])
    # Non-vacuity: at least the primary run plus one retry actually happened.
    assert attempts >= 2, f"ladder never retried at all (instances={attempts})"
    assert attempts < 1 + max_retries, (
        f"ladder ran the full attempt count ({attempts} instances = 1 primary + "
        f"{max_retries} retries) instead of stopping at the {deadline_s}s deadline"
    )

    # The surfaced error must be the real one — not a 2xx, not empty, not generic.
    # An unreachable upstream is 502 BAD_GATEWAY, not 500 INTERNAL_ERROR; the caller's
    # retry gate is `>= 500`, so this is unchanged for the n8n Workers node. Asserting
    # the gate as well as the code keeps that invariant explicit.
    assert resp.status_code >= 500, f"expected the genuine failure to surface, got {resp.text}"
    assert resp.status_code == 502, f"expected the genuine failure to surface, got {resp.text}"
    body = resp.json()
    detail = body.get("error", {}).get("message") or body.get("detail") or ""
    assert detail.strip(), "surfaced an empty error message"
    assert "Connection error" in detail, (
        f"surfaced a fabricated message instead of the cause: {detail!r}"
    )


@pytest.mark.asyncio
async def test_exhausted_ladder_never_returns_an_empty_success(slow_env, monkeypatch):
    """Ladder exhaustion is a 500 with the real cause — never a schema-valid 200."""
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LLM_RETRY_DEADLINE_S", "30.0")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_JITTER_RATIO", "0.0")
    reset_settings_cache()
    try:
        async with _asgi_client() as client:
            resp = await _post(client)
    finally:
        reset_settings_cache()

    # Non-vacuity: the ladder ran to exhaustion (primary + 2 retries).
    assert len(slow_env["created"]) == 3, (
        f"ladder did not exhaust its 3 attempts (instances={len(slow_env['created'])})"
    )

    # An exhausted ladder against an unreachable upstream is 502 BAD_GATEWAY, not 500
    # INTERNAL_ERROR. The property this test protects — a real error, never a
    # synthesised success — is the `>= 500` gate plus the assertions below.
    assert resp.status_code >= 500, f"exhausted ladder returned {resp.status_code}: {resp.text}"
    assert resp.status_code == 502, f"exhausted ladder returned {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "output" not in body, f"exhausted ladder synthesised a success payload: {body}"
    detail = body.get("error", {}).get("message") or body.get("detail") or ""
    assert "Connection error" in detail, f"lost the underlying error: {detail!r}"

    # And no 'completed' state was written for a run that never completed.
    written = [
        c.kwargs.get("state_type") for c in slow_env["state_db"].add_worker_state.await_args_list
    ]
    assert "completed" not in written, f"recorded a completed state for a failed run: {written}"
    assert "error" in written, f"never recorded the error state: {written}"
