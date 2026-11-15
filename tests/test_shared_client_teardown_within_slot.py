"""Regression tests: a cleanup must not sever the transport of an in-flight sibling.

THE DEFECT. One cached worker instance is shared by every CONCURRENT request that
lands on the same cache slot, and they all share the ONE ``httpx`` connection pool
owned by that instance's ``AsyncOpenAI``. ``PydanticAIAgent.cleanup()`` closed that
pool unconditionally, and the retry ladder in ``message_endpoints.py`` invokes
cleanup on every transient error ("recreating"). So a run that hit a real fault
tore the sockets out from under the siblings still awaiting ``agent.run()``.

Measured on the deployed #209 diagnostics, in the window this fix targets:

* ``connection_state``: new 4,740 / reused 12 / unknown 0 — the exact INVERSE of a
  stale-keepalive fault, which is why pool/keepalive tuning is not the fix.
* 100% ``response_headers_received=False``, 100% ``error_type=ReadError``, and the
  ``ReadError`` message was EMPTY — a LOCAL close, not a peer RST.
* ``Cleaning up agent`` -> next ``ir_http_request_failed``: p50 4.81 ms, 511 of 512
  within 50 ms. A negative control anchored on an equally frequent unrelated INFO
  line gave p50 340 ms and 0 within 3 ms.
* 534 cleanups across only 10 distinct cache keys in 5 minutes, one key evicted 155
  times — so the collisions are WITHIN a slot, not across the variants that #208's
  ``remove_instance`` already separated.
* inference-router logged 0 errors and routed 2,681 of 2,682 in the window; it has
  no code path that closes an inbound connection.

THE FIX under test has two halves:

1. ``PydanticAIAgent`` refcounts in-flight runs. ``cleanup()`` retires the instance
   and the LAST run out closes the pool. The close is NOT removed — it is
   load-bearing against a pool leak (``TestCleanupClosesHttpClient`` in
   ``test_pydantic_ai_agent.py`` pins that) — it is merely deferred to a moment when
   nobody is using the transport.
2. ``WorkerInstanceCache.remove_instance(..., expected=<the instance that failed>)``
   is a compare-and-evict, so N runs failing together retire ONE generation instead
   of each evicting the replacement the previous one just built.

⛔ These tests use a REAL ``PydanticAIAgent`` with a REAL ``AsyncOpenAI`` wherever
the assertion is about pool state, so ``client.is_closed()`` is the production
object's own answer and not a mock that could hide the wrong attribute path.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.workers._globals import WorkerInstanceCache
from src.auth import get_tenant_id
from src.settings import get_settings, reset_settings_cache
from src.worker.agents.pydantic_ai_agent import PydanticAIAgent, PydanticAIAgentConfig

TENANT = "tenant-within-slot"
WORKER = "video_synthesis"
WORKER_ID = "wid-within-slot-1"
_MP = "src.api.workers.message_endpoints"


def _real_agent(name: str = "within_slot_agent") -> PydanticAIAgent:
    """A real agent whose ``agent.model.client`` is a real ``AsyncOpenAI``.

    ⚠️ ``openai`` is an EXTERNAL provider and is refused unless the tenant is a
    PROVEN platform workspace (owner ruling 2026-10-03). This test is about httpx
    pool teardown, not about egress, so it constructs inside a licence-proven
    platform context only.
    """
    config = PydanticAIAgentConfig(
        name=name,
        model_provider="openai",
        model="gpt-4o",
        db_path="",
        base_url="http://inference-router.example.com:8150/api",
        api_key="test-key",  # pragma: allowlist secret
        tenant_id="0195f48b-d000-7000-8000-000000000001",
    )
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context():
        return PydanticAIAgent(config=config)


def _fake_run_result() -> Any:
    """Minimal stand-in for pydantic-ai's ``AgentRunResult`` on the success path."""
    result = MagicMock()
    result.output = "done"
    result.usage = MagicMock(return_value=None)
    return result


# ---------------------------------------------------------------------------
# THE reproducer: agent level, real pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_leaves_an_in_flight_siblings_pool_open() -> None:
    """A cleanup issued while a sibling run is mid-flight must not close the pool.

    This is the production timeline: two requests share one cached instance, one
    of them hits a genuine upstream fault and its retry ladder cleans the instance
    up, and the other is still inside ``agent.run()`` waiting on bytes.
    """
    agent = _real_agent()
    client = agent.agent.model.client  # real AsyncOpenAI
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: dict[str, Any] = {}

    async def _parked_run(*_a: object, **_k: object) -> Any:
        entered.set()
        await release.wait()
        # What the sibling's socket layer would see at the moment it resumes.
        observed["closed_mid_flight"] = client.is_closed()
        return _fake_run_result()

    agent._run_agent_with_timeout = _parked_run  # type: ignore[method-assign]

    sibling = asyncio.create_task(agent.run("hello"))
    await asyncio.wait_for(entered.wait(), timeout=5.0)

    # Non-vacuity: the sibling really is parked inside the run, on a live pool.
    # Deliberately expressed WITHOUT the new refcount attribute so that on the
    # unfixed tree this test fails on the defect assertion below, not on a probe
    # for a mechanism that does not exist yet.
    assert not sibling.done(), "sibling finished before the evictor ran"
    assert client.is_closed() is False

    # The evictor: exactly what `_cleanup_worker` does on the retry path.
    await agent.cleanup()

    assert client.is_closed() is False, (
        "cleanup() closed the httpx connection pool of a concurrently in-flight run"
    )

    release.set()
    await asyncio.wait_for(sibling, timeout=5.0)

    assert observed["closed_mid_flight"] is False, (
        "the in-flight sibling resumed onto a closed connection pool"
    )
    # The other half of the contract: the close is deferred, never skipped.
    assert client.is_closed() is True, (
        "deferred close never happened after the last in-flight run left — pool leaked"
    )


@pytest.mark.asyncio
async def test_deferred_close_waits_for_the_last_of_several_runs() -> None:
    """With N runs in flight, the pool closes when the Nth leaves, not the first."""
    agent = _real_agent("multi_run_agent")
    client = agent.agent.model.client
    entered = asyncio.Event()
    gates = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    order: list[int] = []
    seq = iter(range(3))

    async def _parked_run(*_a: object, **_k: object) -> Any:
        idx = next(seq)
        order.append(idx)
        if len(order) == 3:
            entered.set()
        await gates[idx].wait()
        return _fake_run_result()

    agent._run_agent_with_timeout = _parked_run  # type: ignore[method-assign]

    tasks = [asyncio.create_task(agent.run("hello")) for _ in range(3)]
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    assert not any(t.done() for t in tasks), "runs finished before the evictor ran"

    await agent.cleanup()
    assert client.is_closed() is False, "cleanup() closed the pool with 3 runs in flight"

    gates[0].set()
    await asyncio.wait_for(tasks[0], timeout=5.0)
    assert client.is_closed() is False, "pool closed after the FIRST of 3 runs finished"

    gates[1].set()
    await asyncio.wait_for(tasks[1], timeout=5.0)
    assert client.is_closed() is False, "pool closed after the SECOND of 3 runs finished"

    gates[2].set()
    await asyncio.wait_for(tasks[2], timeout=5.0)
    assert client.is_closed() is True, "pool never closed after the last run finished"


@pytest.mark.asyncio
async def test_cleanup_with_nothing_in_flight_closes_immediately() -> None:
    """The anti-leak contract is unchanged when there is no sibling to protect.

    Guards the direction the fix could plausibly be over-applied in: deferring
    ALWAYS (or dropping the close) would reintroduce the pool leak the close was
    added for — an instance is rebuilt on every transient-error recreate, every
    failed health check and every tenant config invalidation.
    """
    agent = _real_agent("idle_agent")
    client = agent.agent.model.client
    assert client.is_closed() is False
    await agent.cleanup()
    assert client.is_closed() is True, "an idle agent's pool was not released on cleanup"
    # Idempotent: a second cleanup (or a stray release) must not explode.
    await agent.cleanup()
    assert client.is_closed() is True


@pytest.mark.asyncio
async def test_a_failing_run_neither_closes_the_pool_nor_pins_it_open() -> None:
    """The error path is the one that actually happens here — both directions of it.

    ``httpx.ReadError("")`` with an EMPTY message is verbatim the production
    failure: a LOCAL close, surfaced through the OpenAI SDK as
    ``APIConnectionError("Connection error.")``.

    Two properties in one timeline: the failing run must not take the shared pool
    down with it while a sibling is still using it (the defect), and it must still
    give its transport reference back so the deferred close can eventually fire
    (the leak the close exists to prevent).
    """
    agent = _real_agent("failing_agent")
    client = agent.agent.model.client
    both_in = asyncio.Event()
    arrived = {"n": 0}
    release_failer = asyncio.Event()
    release_survivor = asyncio.Event()

    async def _failing(*_a: object, **_k: object) -> Any:
        await both_in.wait()
        await release_failer.wait()
        raise httpx.ReadError("")

    async def _surviving(*_a: object, **_k: object) -> Any:
        await both_in.wait()
        await release_survivor.wait()
        return _fake_run_result()

    async def _dispatch(*a: object, **k: object) -> Any:
        arrived["n"] += 1
        mine = _failing if arrived["n"] == 1 else _surviving
        if arrived["n"] == 2:
            both_in.set()
        return await mine(*a, **k)

    agent._run_agent_with_timeout = _dispatch  # type: ignore[method-assign]

    failer = asyncio.create_task(agent.run("hello"))
    survivor = asyncio.create_task(agent.run("hello"))
    await asyncio.wait_for(both_in.wait(), timeout=5.0)

    await agent.cleanup()

    release_failer.set()
    with pytest.raises(httpx.ReadError):
        await asyncio.wait_for(failer, timeout=5.0)

    assert client.is_closed() is False, (
        "the pool was closed while a sibling run was still in flight "
        "(a failing run must not take the shared transport down with it)"
    )

    release_survivor.set()
    await asyncio.wait_for(survivor, timeout=5.0)
    assert client.is_closed() is True, (
        "the failing run never released its transport reference — the deferred "
        "close can no longer fire and the pool is leaked"
    )


# ---------------------------------------------------------------------------
# Same defect, driven through the production cache path
# ---------------------------------------------------------------------------


def _worker_data() -> dict[str, Any]:
    return {
        "worker_name": WORKER,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/within-slot.db",
        "is_agent": True,
        "config": {
            "agent_type": "pydantic_ai",
            "model_provider": "inference_router",
            "model": "local-llm",
        },
        "status": "active",
    }


@pytest.mark.asyncio
async def test_remove_instance_does_not_sever_a_run_in_flight_on_that_slot() -> None:
    """``remove_instance`` on the caller's OWN slot still had a sibling inside it.

    #208 scoped the eviction from a ``tenant:worker:`` prefix down to one exact
    slot, which fixed collisions ACROSS variants. It cannot help here: the
    colliding runs share the very slot being evicted. This drives the real
    ``WorkerInstanceCache`` against a real ``PydanticAIAgent``.
    """
    cache = WorkerInstanceCache()
    agent = _real_agent("cache_slot_agent")
    client = agent.agent.model.client
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _parked_run(*_a: object, **_k: object) -> Any:
        entered.set()
        await release.wait()
        return _fake_run_result()

    agent._run_agent_with_timeout = _parked_run  # type: ignore[method-assign]

    async def _create_instance(*_a: Any, **_kw: Any) -> PydanticAIAgent:
        return agent

    with patch.object(WorkerInstanceCache, "_create_worker_instance", _create_instance):
        cached = await cache.get_or_create_worker(WORKER, _worker_data(), tenant_id=TENANT)
        assert cached is agent
        sibling = asyncio.create_task(cached.run("hello"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        # Non-vacuity: one slot, one instance, one run parked inside it.
        assert list(cache._workers) == [f"{TENANT}:{WORKER}:_"], list(cache._workers)
        assert not sibling.done(), "sibling finished before the eviction"

        await cache.remove_instance(WORKER, _worker_data(), tenant_id=TENANT)

    assert client.is_closed() is False, (
        "remove_instance closed the httpx pool of a run in flight on that same slot"
    )
    release.set()
    await asyncio.wait_for(sibling, timeout=5.0)
    assert client.is_closed() is True, "the evicted instance's pool was never released"


@pytest.mark.asyncio
async def test_remove_instance_skips_a_slot_that_holds_a_newer_instance() -> None:
    """Compare-and-evict: a late failure must not retire someone else's replacement."""
    cache = WorkerInstanceCache()
    created: list[MagicMock] = []

    async def _create_instance(*_a: Any, **_kw: Any) -> MagicMock:
        inst = MagicMock()
        inst.cleanup = AsyncMock()
        inst.check_health = AsyncMock(return_value=True)
        created.append(inst)
        return inst

    with patch.object(WorkerInstanceCache, "_create_worker_instance", _create_instance):
        first = await cache.get_or_create_worker(WORKER, _worker_data(), tenant_id=TENANT)
        # The first failing run retires the generation it ran on.
        await cache.remove_instance(WORKER, _worker_data(), tenant_id=TENANT, expected=first)
        second = await cache.get_or_create_worker(WORKER, _worker_data(), tenant_id=TENANT)
        # Non-vacuity: the slot genuinely holds a DIFFERENT object now.
        assert second is not first
        assert len(created) == 2

        # A sibling that also failed on `first` now reaches its own eviction.
        await cache.remove_instance(WORKER, _worker_data(), tenant_id=TENANT, expected=first)

    assert second.cleanup.await_count == 0, (
        "a stale eviction retired the replacement instance a sibling was running on"
    )
    assert list(cache._workers) == [f"{TENANT}:{WORKER}:_"], list(cache._workers)
    assert first.cleanup.await_count == 1, "the failed generation was never retired"


# ---------------------------------------------------------------------------
# Amplification: N concurrent failures must retire ONE generation, not N
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Endpoint-level stand-in. Only the eviction arithmetic is under test here."""

    def __init__(self, label: str, state: dict[str, Any]) -> None:
        self.label = label
        self.fails = False
        self.run_calls = 0
        self.cleanup_calls = 0
        self._state = state

    async def cleanup(self) -> None:
        self.cleanup_calls += 1

    async def run(self, _message: Any) -> Any:
        self.run_calls += 1
        if self.fails:
            # Park until BOTH concurrent requests are inside this instance, so the
            # failures are genuinely simultaneous — that is the live shape.
            self._state["parked"] += 1
            if self._state["parked"] >= 2:
                self._state["both_in"].set()
            await self._state["both_in"].wait()
            raise APIConnectionError(
                request=httpx.Request("POST", "http://inference-router:8150/api/chat/completions")
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


@pytest.mark.asyncio
async def test_simultaneous_failures_on_one_slot_rebuild_the_instance_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs failing together must produce ONE replacement, not a rebuild chain.

    Without compare-and-evict the second failure evicts the replacement the first
    failure just built and is already running on — which is how 10 cache keys
    absorbed 534 cleanups in 5 minutes, one of them 155 times. Every one of those
    extra evictions discards a healthy pool and forces fresh TCP connects, which
    is the mechanism behind ``connection_state`` reading 99.75% new.
    """
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY_S", "0.0")
    monkeypatch.setenv("LLM_RETRY_JITTER_RATIO", "0.0")
    reset_settings_cache()
    try:
        assert get_settings().llm_retry_max_attempts == 2

        cache = WorkerInstanceCache()
        created: list[_FakeAgent] = []
        state: dict[str, Any] = {"parked": 0, "both_in": asyncio.Event()}

        async def _create_instance(
            _self: Any, worker_name: str, *_a: Any, **_kw: Any
        ) -> _FakeAgent:
            agent = _FakeAgent(f"{worker_name}#{len(created)}", state)
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

            async with _asgi_client() as client:
                prime = await client.post(f"/workers/{WORKER}/message", json={"message": "Hi"})
                assert prime.status_code == 200, prime.text
                assert len(created) == 1, "priming call must have built the shared instance"
                shared = created[0]
                shared.fails = True

                a, b = await asyncio.gather(
                    client.post(f"/workers/{WORKER}/message", json={"message": "Hi"}),
                    client.post(f"/workers/{WORKER}/message", json={"message": "Hi"}),
                )
    finally:
        reset_settings_cache()

    # Non-vacuity: both requests really did fail simultaneously on the SAME instance
    # and really did enter the retry ladder.
    assert shared.run_calls == 3, f"prime + 2 concurrent runs expected, got {shared.run_calls}"
    assert len(created) >= 2, "the ladder never rebuilt the instance at all"

    assert len(created) == 2, (
        "each concurrent failure rebuilt the slot: expected 1 replacement for 2 "
        f"simultaneous failures, got instances={[a_.label for a_ in created]}"
    )
    replacement = created[1]
    assert replacement.cleanup_calls == 0, (
        "the replacement instance was torn down while the sibling retry was running "
        f"on it ({replacement.cleanup_calls} cleanup calls)"
    )
    assert replacement.run_calls == 2, (
        f"both retries should have shared the one replacement, got {replacement.run_calls}"
    )
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
