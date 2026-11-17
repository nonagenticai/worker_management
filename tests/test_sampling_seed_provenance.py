"""Seed + sampling provenance: what a worker SENDS, and what it RECORDS it sent.

Determinism of an LLM worker cannot be proven unless (1) a fixed ``seed`` and a
pinned model can reach inference-router, and (2) every completed call records
the model / temperature / seed it actually requested. Before this, usage_metrics
carried served_model / served_provider / tokens only.

⛔ The wire tests drive a REAL ``PydanticAIAgent`` (real pydantic-ai ``Agent``,
real ``InferenceRouterOpenAIChatModel``, real OpenAI SDK client) and read the
JSON body off the captured ``httpx.Request``. A seed set on a config object that
never reaches the request body is exactly the failure being guarded against, so
nothing between the config and the socket is mocked.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest
from pydantic import ValidationError

from src.api.workers._globals import WorkerInstanceCache

MODULE = "src.worker.agents.pydantic_ai_agent"
PINNED_MODEL = "local-llm"
TENANT = "019edc1e-0000-7000-8000-000000000001"


def _ir_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-seed",
            "object": "chat.completion",
            "created": 0,
            "model": PINNED_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            # IR's non-standard routing echo (see test_served_model_routing.py).
            "served_model": PINNED_MODEL,
            "served_provider": "pgx-local",
        },
    )


async def _run_against_captured_ir(**agent_kwargs: Any) -> tuple[dict[str, Any], Any]:
    """Build a real inference_router agent, run it once, return (body, result)."""
    from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

    bodies: list[dict[str, Any]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ir_response(request)

    kwargs: dict[str, Any] = {
        "name": "seed_probe",
        "description": "d",
        "model_provider": "inference_router",
        "model": PINNED_MODEL,
        "base_url": "http://inference-router.invalid",
        **agent_kwargs,
    }
    agent = create_pydantic_ai_agent(**kwargs)
    # The real OpenAI SDK client the provider built; only its transport is swapped.
    agent.agent.model.client._client._transport = httpx.MockTransport(_capture)

    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        result = await agent.run("hi")

    assert len(bodies) == 1, f"expected exactly one outbound request, saw {len(bodies)}"
    return bodies[0], result


# --------------------------------------------------------------------------- wire


@pytest.mark.asyncio
async def test_seed_reaches_the_outbound_request_body():
    """MUTATION: drop the ``agent_model_settings["seed"] = ...`` line in
    ModelFactory.get_model_settings ⇒ ``seed`` is absent from the body ⇒ red."""
    body, _ = await _run_against_captured_ir(seed=1234, temperature=0.3)
    assert body["seed"] == 1234
    assert body["temperature"] == 0.3
    assert body["model"] == PINNED_MODEL


@pytest.mark.asyncio
async def test_a_worker_with_no_seed_sends_no_seed_key():
    """None must mean ABSENT, not ``"seed": null`` — a null seed is a different
    request to an OpenAI-compatible server than no seed at all."""
    body, _ = await _run_against_captured_ir(temperature=0.3)
    assert "seed" not in body
    # Control: the same request DID carry the other sampling field, so an
    # absent seed is not just an empty/unparsed body.
    assert body["temperature"] == 0.3


@pytest.mark.asyncio
async def test_seed_zero_is_sent_not_dropped():
    """0 is a real seed. A truthiness check (``if config.seed:``) would drop it."""
    body, _ = await _run_against_captured_ir(seed=0)
    assert body["seed"] == 0


# ------------------------------------------------------------------ usage_metrics


@pytest.mark.asyncio
async def test_usage_metrics_record_requested_model_temperature_and_seed():
    """MUTATION: make ``_attach_sampling_provenance`` a no-op (or read seed from
    anywhere but the sent settings) ⇒ seed/temperature/requested_model are None ⇒ red."""
    body, result = await _run_against_captured_ir(seed=77, temperature=0.25)
    um = result.usage_metrics
    assert um is not None
    assert um.seed == 77
    assert um.temperature == 0.25
    assert um.requested_model == PINNED_MODEL
    # Recorded == sent: the provenance must agree with the wire, field by field.
    assert (um.seed, um.temperature, um.requested_model) == (
        body["seed"],
        body["temperature"],
        body["model"],
    )
    # served_* are kept, not displaced.
    assert um.served_model == PINNED_MODEL
    assert um.served_provider == "pgx-local"
    assert um.total_tokens == 4


@pytest.mark.asyncio
async def test_usage_metrics_seed_is_null_when_no_seed_was_sent():
    body, result = await _run_against_captured_ir(temperature=0.3)
    um = result.usage_metrics
    assert "seed" not in body
    assert um.seed is None
    assert um.temperature == 0.3


@pytest.mark.asyncio
async def test_usage_metrics_temperature_is_the_effective_value_after_defaulting():
    """A worker whose stored temperature is null is sent agent_factory's 0.7
    default. Recording the stored null would claim "no temperature" for a call
    that carried one — so the recorded value must be the one on the wire.

    Drives the real factory hop (``_create_agent_worker``) that applies the default.
    """
    from src.worker.agents.agent_registry import create_agent as real_create_agent
    from src.worker.factory.agent_factory import (
        AgentConfig,
        WorkerConfig,
        _create_agent_worker,
    )

    captured: dict[str, Any] = {}

    def _spy(agent_type: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_create_agent(agent_type, **kwargs)

    agent_config = AgentConfig(
        agent_role="r",
        agent_type="pydantic_ai",
        model_provider="inference_router",
        model=PINNED_MODEL,
        temperature=None,
        seed=5,
    )
    with patch("src.worker.agents.agent_registry.create_agent", side_effect=_spy):
        worker = await _create_agent_worker(
            WorkerConfig(
                worker_name="seed_probe",
                tool_names=[],
                description="d",
                deps=[],
                base_url_override="http://inference-router.invalid",
                api_key_override=None,
                tools_dir_override=None,
                agent_config=agent_config,
            ),
            agent_config,
            [],
            "",
            [],
        )
    assert captured["temperature"] == 0.7
    assert captured["seed"] == 5

    from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

    agent = worker
    assert isinstance(agent, PydanticAIAgent), f"unexpected worker shape {type(worker)!r}"
    bodies: list[dict[str, Any]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ir_response(request)

    agent.agent.model.client._client._transport = httpx.MockTransport(_capture)
    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        result = await agent.run("hi")
    assert bodies[0]["temperature"] == 0.7
    assert result.usage_metrics.temperature == 0.7
    assert bodies[0]["seed"] == 5
    assert result.usage_metrics.seed == 5


def test_provenance_is_recorded_even_when_usage_is_unavailable():
    """A completed call with no token usage still records what it asked for."""
    from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

    agent = object.__new__(PydanticAIAgent)
    agent._sampling_provenance = {"requested_model": "m", "temperature": 0.1, "seed": 9}
    um = agent._attach_sampling_provenance(None)
    assert um is not None
    assert (um.requested_model, um.temperature, um.seed) == ("m", 0.1, 9)
    assert um.total_tokens is None


@pytest.mark.asyncio
async def test_unmeasured_usage_is_not_booked_as_zero_tokens():
    """Token extraction found nothing (``_extract_token_usage`` → None). The caller
    still gets provenance, but the metrics recorder must see ``token_usage=None``.

    MUTATION: attach provenance before the metrics/BYO-cost branches (which gate
    on ``token_metrics`` truthiness) ⇒ they see a non-None object ⇒ zero-token
    metrics (and, for BYO, a zero-token cost_event) ⇒ red.
    """
    from src.worker.agents.pydantic_ai_agent import PydanticAIAgent, create_pydantic_ai_agent

    agent = create_pydantic_ai_agent(
        name="seed_probe",
        description="d",
        model_provider="inference_router",
        model=PINNED_MODEL,
        base_url="http://inference-router.invalid",
        seed=3,
    )
    agent.agent.model.client._client._transport = httpx.MockTransport(_ir_response)
    metrics = MagicMock()
    with (
        patch(f"{MODULE}.get_metrics_client", return_value=metrics),
        patch.object(PydanticAIAgent, "_extract_token_usage", return_value=None),
    ):
        result = await agent.run("hi")

    assert metrics.record_execution.call_args.kwargs["token_usage"] is None
    um = result.usage_metrics
    assert um is not None
    assert um.seed == 3
    assert um.requested_model == PINNED_MODEL
    assert um.total_tokens is None


async def _byo_run(*, usage_measured: bool) -> MagicMock:
    """Run a BYO-costed agent once; return the patched cost-event emitter."""
    from src.worker.agents.pydantic_ai_agent import PydanticAIAgent, create_pydantic_ai_agent
    from src.worker.factory.llm_config_resolver import BYO_COST_SOURCE

    agent = create_pydantic_ai_agent(
        name="seed_probe",
        description="d",
        model_provider="inference_router",
        model=PINNED_MODEL,
        base_url="http://inference-router.invalid",
        seed=3,
    )
    agent.config.cost_source = BYO_COST_SOURCE
    agent.config.cost_provider = "deepseek"
    # Only EXTERNAL calls are metered by WM (inference-router meters its own),
    # so the run must look external at the emit gate; the wire stays mocked.
    agent.config.model_provider = "deepseek"
    agent.agent.model.client._client._transport = httpx.MockTransport(_ir_response)
    emit = MagicMock()
    patches = [
        patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
        patch("src.telemetry.cost_events.emit_byo_cost_event_task", new=emit),
    ]
    if not usage_measured:
        patches.append(patch.object(PydanticAIAgent, "_extract_token_usage", return_value=None))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = await agent.run("hi")
    assert result.usage_metrics is not None
    assert result.usage_metrics.seed == 3
    return emit


@pytest.mark.asyncio
async def test_an_unmeasured_run_books_no_byo_cost_event():
    """Provenance CREATES a metrics object when usage was unmeasured. The BYO
    cost_event branch gates on ``token_metrics`` truthiness, so this is safe
    only because provenance is attached at the return, after that branch.

    MUTATION: attach provenance right after the routing stamp (before the BYO
    branch) ⇒ a zero-token cost_event is emitted ⇒ red.
    """
    emit = await _byo_run(usage_measured=False)
    emit.assert_not_called()


@pytest.mark.asyncio
async def test_control_a_measured_byo_run_does_book_a_cost_event():
    """CONTROL. Without it the test above passes vacuously if the BYO branch
    never fires at all under this setup."""
    emit = await _byo_run(usage_measured=True)
    emit.assert_called_once()


# ------------------------------------------------------------------ per-run seed


@pytest.mark.asyncio
async def test_a_per_run_seed_reaches_the_body_without_rebuilding_the_agent():
    """``run(prompt, seed=)`` goes out as the body's ``seed`` on the SAME agent
    (its settings untouched), and wins over the agent's config seed.

    MUTATION: drop ``**self._run_kwargs(run_seed)`` from ``agent.run`` ⇒ the body
    carries the config seed (5), not the run seed ⇒ red.
    """
    from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

    bodies: list[dict[str, Any]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ir_response(request)

    agent = create_pydantic_ai_agent(
        name="seed_probe",
        description="d",
        model_provider="inference_router",
        model=PINNED_MODEL,
        base_url="http://inference-router.invalid",
        temperature=0.0,
        seed=5,
    )
    agent.agent.model.client._client._transport = httpx.MockTransport(_capture)
    pyd_agent = agent.agent
    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        results = [await agent.run("hi", seed=s) for s in range(5)]
        plain = await agent.run("hi")

    assert agent.agent is pyd_agent  # same agent object throughout
    assert [b["seed"] for b in bodies] == [0, 1, 2, 3, 4, 5]
    assert all(b["temperature"] == 0.0 for b in bodies)  # agent settings still merged in
    assert [r.usage_metrics.seed for r in results] == [0, 1, 2, 3, 4]
    assert plain.usage_metrics.seed == 5


@pytest.mark.asyncio
async def test_five_per_message_seeds_occupy_exactly_one_real_cache_slot():
    """Real ``WorkerInstanceCache``, keyed exactly as send_message keys it.

    MUTATION: put ``"seed"`` back into ``_override_cache_fingerprint`` ⇒ five
    slots and five builds ⇒ red.
    """
    from src.api.workers.message_endpoints import _override_cache_fingerprint
    from src.api.workers.models import MessageRequest

    cache = WorkerInstanceCache()
    create = AsyncMock(side_effect=lambda **_kw: object())
    cfg = {"agent_role": "r", "agent_type": "pydantic_ai", "model_provider": "inference_router"}
    with (
        patch("src.api.workers._globals.create_worker", new=create),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda c, _tid, **_kw: c),
        ),
    ):
        instances = [
            await cache.get_or_create_worker(
                "w",
                {"worker_name": "w", "config": dict(cfg)},
                tenant_id=TENANT,
                override_key=_override_cache_fingerprint(MessageRequest(message="m", seed=s)),
            )
            for s in range(5)
        ]
    assert create.await_count == 1
    assert len(cache._workers) == 1
    assert len({id(i) for i in instances}) == 1


# ------------------------------------------------------------------ config / chain


def test_seed_rejects_bool_and_string():
    """Strict: a stored ``true`` or ``"42"`` must fail loudly, not become 1 / 42."""
    from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

    for bad in (True, "42", 1.5):
        with pytest.raises(ValidationError):
            PydanticAIAgentConfig(
                name="x", model_provider="inference_router", model="m", db_path="", seed=bad
            )


@pytest.mark.asyncio
async def test_agent_factory_forwards_seed_only_when_set_and_only_for_pydantic_ai():
    """Other agent factories do not accept the kwarg — forwarding would 500."""
    from src.worker.factory.agent_factory import AgentConfig, WorkerConfig, _create_agent_worker

    seen: dict[str, dict[str, Any]] = {}

    def _capture(agent_type: str, **kwargs: Any) -> object:
        seen[agent_type] = kwargs
        return object()

    cases = (("pydantic_ai", 11, True), ("pydantic_ai", None, False), ("claude_code", 11, False))
    for agent_type, seed, expect in cases:
        seen.clear()
        ac = AgentConfig(agent_role="r", agent_type=agent_type, seed=seed)
        with patch("src.worker.agents.agent_registry.create_agent", side_effect=_capture):
            await _create_agent_worker(
                WorkerConfig(
                    worker_name="w",
                    tool_names=[],
                    description="d",
                    deps=[],
                    base_url_override=None,
                    api_key_override=None,
                    tools_dir_override=None,
                    agent_config=ac,
                ),
                ac,
                [],
                "",
                [],
            )
        assert ("seed" in seen[agent_type]) is expect, (agent_type, seed)
        if expect:
            assert seen[agent_type]["seed"] == seed


@pytest.mark.parametrize(("config_seed", "expected"), [(42, 42), (None, None)])
@pytest.mark.asyncio
async def test_cache_reads_seed_from_worker_config(config_seed, expected):
    """The cache is the hop where the stored config becomes create_worker kwargs;
    if it never reads ``seed`` back out, every layer downstream is inert."""
    cfg: dict[str, Any] = {
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": PINNED_MODEL,
    }
    if config_seed is not None:
        cfg["seed"] = config_seed
    cache = WorkerInstanceCache()
    create = AsyncMock(return_value=object())
    with (
        patch("src.api.workers._globals.create_worker", new=create),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda c, _tid, **_kw: c),
        ),
    ):
        await cache.get_or_create_worker("w", {"worker_name": "w", "config": cfg}, tenant_id=TENANT)
    assert create.call_args.kwargs["seed"] == expected
