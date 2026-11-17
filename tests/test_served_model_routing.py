"""Tests for issue #137: emitting the RESOLVED model on BYO cost events.

WM sends inference-router a routing alias (e.g. ``tier-medium``) as the OpenAI
``model`` field. IR resolves it server-side and echoes the result back as two
non-standard body fields, ``served_model`` / ``served_provider``. Billing prices
a cost_event on (provider, served_model) exactly once at flush time, so an event
carrying the alias — or a routing name as its provider — has no
``provider_pricing`` row and its ``cost_usd`` stays NULL forever.

Covered here:
- the ``_process_provider_details`` override lifts IR's extra fields onto
  ``ModelResponse.provider_details`` (hand-built ChatCompletion, no network);
- the streamed path does the same via ``InferenceRouterStreamedResponse``, which the
  ``_streamed_response_cls`` hook selects (``_process_provider_details`` is never
  called on a streamed run, so it needs its own hook);
- ``resolve_billable_provider`` never returns a routing name.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import ModelProfile

from src.worker.agents.provider_catalog import ROUTING_PROVIDERS
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIChatModel,
    PydanticAIAgent,
    PydanticAIAgentConfig,
    resolve_billable_provider,
)

AGENT_MODULE = "src.worker.agents.pydantic_ai_agent"


def _bare_model() -> InferenceRouterOpenAIChatModel:
    """An uninitialised instance, enough to exercise _process_provider_details.

    The hook only reads its argument: the base implementation delegates to the
    module-level ``_map_provider_details(response.choices[0])`` and the override
    reads ``response.model_extra``. Neither touches instance state, so we skip
    __init__ (which would demand a provider + HTTP client) while keeping a real
    instance so the zero-arg ``super()`` inside the override binds.
    """
    return object.__new__(InferenceRouterOpenAIChatModel)


def _completion(**extra) -> ChatCompletion:
    """A minimal ChatCompletion, optionally carrying IR's non-standard fields.

    The OpenAI SDK's BaseModel is ``extra="allow"``, so unknown top-level keys
    survive onto ``model_extra`` — which is exactly the mechanism the override
    relies on.
    """
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "tier-medium",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            **extra,
        }
    )


def test_extra_fields_survive_openai_sdk_parsing():
    """Guards the assumption the override is built on: extra="allow" keeps them."""
    completion = _completion(served_model="deepseek-v4-pro", served_provider="deepseek")
    assert completion.model_extra["served_model"] == "deepseek-v4-pro"
    assert completion.model_extra["served_provider"] == "deepseek"


def test_process_provider_details_surfaces_served_fields():
    """IR's served_model / served_provider land in provider_details."""
    completion = _completion(served_model="deepseek-v4-pro", served_provider="deepseek")
    details = _bare_model()._process_provider_details(completion)
    assert details["served_model"] == "deepseek-v4-pro"
    assert details["served_provider"] == "deepseek"
    # The base implementation's own keys are preserved, not clobbered.
    assert details["finish_reason"] == "stop"


def test_process_provider_details_omits_absent_fields():
    """An IR predating the companion PR returns neither field — degrade cleanly."""
    details = _bare_model()._process_provider_details(_completion())
    assert details is None or "served_model" not in details


def test_process_provider_details_omits_null_fields():
    """Explicit nulls are dropped, not passed through as None values."""
    details = _bare_model()._process_provider_details(
        _completion(served_model=None, served_provider=None)
    )
    assert details is None or "served_model" not in details


def _chunk(**extra) -> ChatCompletionChunk:
    """A minimal ChatCompletionChunk, optionally carrying IR's non-standard fields.

    Streaming echoes ``served_model`` / ``served_provider`` on the chunk body, the
    same ``extra="allow"`` mechanism the non-streaming ``_completion`` relies on.
    """
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "tier-medium",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant", "content": "hi"},
                }
            ],
            **extra,
        }
    )


async def _drive_stream(*chunks: ChatCompletionChunk) -> dict:
    """Run real chunks through the class the model's streamed hook selects.

    Resolves the class via ``InferenceRouterOpenAIChatModel._streamed_response_cls``
    — the exact hook pydantic-ai's ``_process_streamed_response`` uses — so if that
    override is dropped this drives plain ``OpenAIStreamedResponse`` and the
    served_* keys never appear. Returns the accumulated ``provider_details``.
    """

    async def _source():
        for chunk in chunks:
            yield chunk

    streamed_cls = _bare_model()._streamed_response_cls
    response = streamed_cls(
        model_request_parameters=ModelRequestParameters(),
        _model_name="tier-medium",
        _model_profile=ModelProfile(),
        _response=PeekableAsyncStream(_source()),
        _provider_name="inference_router",
        # pydantic-ai 2.69 made the provider's model-id namespace an init field.
        _model_id_namespace="inference_router",
        _provider_url="http://inference-router.invalid",
    )
    async for _ in response._get_event_iterator():
        pass
    return response.provider_details or {}


@pytest.mark.asyncio
async def test_streamed_response_preserves_served_model_and_provider():
    """The streamed path must surface IR's fields too (they are body-level)."""
    details = await _drive_stream(
        _chunk(served_model="deepseek-v4-pro", served_provider="deepseek")
    )
    assert details["served_model"] == "deepseek-v4-pro"
    assert details["served_provider"] == "deepseek"
    # The base implementation's own keys are preserved, not clobbered.
    assert details["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streamed_response_omits_absent_fields():
    """An IR predating the companion PR streams neither field — degrade cleanly."""
    details = await _drive_stream(_chunk())
    assert "served_model" not in details
    assert "served_provider" not in details


@pytest.mark.asyncio
async def test_streamed_and_non_streamed_provider_details_agree():
    """Both paths must yield the same served_* details for the same IR answer."""
    streamed = await _drive_stream(
        _chunk(served_model="deepseek-v4-pro", served_provider="deepseek")
    )
    non_streamed = _bare_model()._process_provider_details(
        _completion(served_model="deepseek-v4-pro", served_provider="deepseek")
    )
    keys = ("served_model", "served_provider", "finish_reason")
    assert {k: streamed.get(k) for k in keys} == {k: non_streamed.get(k) for k in keys}


@pytest.mark.parametrize("routing_name", sorted(ROUTING_PROVIDERS))
def test_routing_provider_alone_is_unpriceable(routing_name):
    """A routing name with an unmappable model must yield None (skip the emit)."""
    assert (
        resolve_billable_provider(
            served_provider=None,
            cost_provider=None,
            model_provider=routing_name,
            served_model=None,
            requested_model="some-unknown-alias",
        )
        is None
    )


def test_routing_provider_rederived_from_served_model():
    """A routing name is rescued once served_model maps to a real upstream."""
    assert (
        resolve_billable_provider(
            served_provider=None,
            cost_provider=None,
            model_provider="inference_router",
            served_model="deepseek-v4-pro",
            requested_model="tier-medium",
        )
        == "deepseek"
    )


def test_served_provider_wins_over_config():
    """What IR says it actually used beats anything resolved at config time."""
    assert (
        resolve_billable_provider(
            served_provider="deepseek",
            cost_provider="pgx-local",
            model_provider="inference_router",
            served_model="deepseek-v4-pro",
            requested_model="tier-medium",
        )
        == "deepseek"
    )


def test_cost_provider_used_when_ir_is_silent():
    """Old IR (no served_provider): fall back to the tenant-resolved provider."""
    assert (
        resolve_billable_provider(
            served_provider=None,
            cost_provider="pgx-local",
            model_provider="inference_router",
            served_model=None,
            requested_model="local-llm",
        )
        == "pgx-local"
    )


def test_external_byo_provider_passes_through():
    """Direct external BYO has no routing hop — model_provider is billable."""
    assert (
        resolve_billable_provider(
            served_provider=None,
            cost_provider=None,
            model_provider="anthropic",
            served_model=None,
            requested_model="claude-opus-4-8",
        )
        == "anthropic"
    )


def test_no_provider_at_all_is_unpriceable():
    assert (
        resolve_billable_provider(
            served_provider=None,
            cost_provider=None,
            model_provider=None,
            served_model=None,
            requested_model="x",
        )
        is None
    )


# ===========================================================================
# served_model / served_provider reaching the CALLER, not just the cost event
# ===========================================================================
#
# The extraction above was correct and its answer was thrown away. It ran only
# inside the BYO-cost branch of ``PydanticAIAgent.run`` — guarded by
# ``cost_source == BYO_COST_SOURCE`` — so every ordinary run answered the API
# caller with ``served_model: null`` even though inference-router had told us,
# in the response body, exactly which model served the request. These tests
# exercise the NON-BYO path first, because that is where the bug lived and
# where a regression would be invisible to the billing tests above.


def _ir_run_result(provider_details: dict | None, *, output: str = "ok") -> MagicMock:
    """A pydantic-ai run result shaped like one an inference-router run yields."""
    message = MagicMock()
    message.provider_details = provider_details

    usage = MagicMock()
    usage.has_values.return_value = True
    usage.input_tokens = 11
    usage.output_tokens = 22
    usage.total_tokens = 33

    result = MagicMock()
    result.output = output
    result.all_messages.return_value = [message]
    result.usage = usage
    return result


def _agent(
    cost_source: str | None = None, model_provider: str = "inference_router"
) -> PydanticAIAgent:
    """An agent whose single model call is fully mocked out."""
    config = PydanticAIAgentConfig(
        name="served_routing_agent",
        model_provider=model_provider,
        model="tier-medium",
        db_path="",
        base_url="http://inference-router.invalid:8150/api",
        cost_source=cost_source,
    )
    with (
        patch(f"{AGENT_MODULE}.PydanticAIAgent._initialize_model", return_value=MagicMock()),
        patch(f"{AGENT_MODULE}.Agent"),
    ):
        return PydanticAIAgent(config=config)


async def _run(agent: PydanticAIAgent, run_result: MagicMock):
    agent.agent.run = AsyncMock(return_value=run_result)
    with patch(f"{AGENT_MODULE}.get_metrics_client") as metrics:
        metrics.return_value = MagicMock()
        return await agent.run("hello")


@pytest.mark.asyncio
async def test_served_routing_lands_on_usage_metrics_without_byo_cost():
    """⭐ THE BUG. cost_source is None — the overwhelmingly common case."""
    result = await _run(
        _agent(cost_source=None),
        _ir_run_result({"served_model": "deepseek-v4-pro", "served_provider": "deepseek"}),
    )
    assert result.usage_metrics is not None
    assert result.usage_metrics.served_model == "deepseek-v4-pro"
    assert result.usage_metrics.served_provider == "deepseek"
    # The token fields must be untouched by the addition.
    assert result.usage_metrics.prompt_tokens == 11
    assert result.usage_metrics.completion_tokens == 22


@pytest.mark.asyncio
async def test_served_routing_is_none_when_the_provider_omits_it():
    """A provider that does not report these yields None — never an exception,
    and never a fabricated value standing in for "we did not ask"."""
    result = await _run(
        _agent(cost_source=None),
        _ir_run_result({"finish_reason": "stop"}),
    )
    assert result.usage_metrics is not None
    assert result.usage_metrics.served_model is None
    assert result.usage_metrics.served_provider is None


@pytest.mark.asyncio
async def test_no_provider_details_at_all_still_completes_the_run():
    """An older IR, or any non-IR provider, carries no provider_details."""
    result = await _run(_agent(cost_source=None), _ir_run_result(None))
    assert result.output == "ok"
    assert result.usage_metrics.served_model is None


@pytest.mark.asyncio
async def test_a_broken_message_history_cannot_fail_the_run():
    """Defensive by contract: a routing annotation may never break a run."""
    run_result = _ir_run_result(None)
    run_result.all_messages.side_effect = RuntimeError("message history exploded")
    result = await _run(_agent(cost_source=None), run_result)
    assert result.output == "ok"
    assert result.usage_metrics.served_model is None


@pytest.mark.asyncio
async def test_byo_cost_path_still_prices_on_the_served_model():
    """The hoist must not have cost billing its input.

    The cost event and the caller's usage_metrics now come from ONE traversal,
    so this also pins that they cannot disagree about which model served the run.
    """
    from src.worker.factory.llm_config_resolver import BYO_COST_SOURCE

    with patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit:
        result = await _run(
            # An EXTERNAL provider: only those are metered by WM (IR meters the rest).
            _agent(cost_source=BYO_COST_SOURCE, model_provider="deepseek"),
            _ir_run_result({"served_model": "deepseek-v4-pro", "served_provider": "deepseek"}),
        )

    emit.assert_called_once()
    assert emit.call_args.kwargs["served_model"] == "deepseek-v4-pro"
    assert emit.call_args.kwargs["provider"] == "deepseek"
    assert result.usage_metrics.served_model == emit.call_args.kwargs["served_model"]


def test_attach_served_routing_tolerates_absent_metrics():
    """Token usage can legitimately be unavailable; stamping must not require it."""
    pair = PydanticAIAgent._attach_served_routing(
        None,
        _ir_run_result({"served_model": "m", "served_provider": "p"}),
    )
    assert pair == ("m", "p")


@pytest.mark.asyncio
async def test_inference_router_run_emits_nothing_router_is_the_single_meter():
    """Even with a stray cost_source, a platform-routed run is IR's to meter."""
    from src.worker.factory.llm_config_resolver import BYO_COST_SOURCE

    with patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit:
        await _run(
            _agent(cost_source=BYO_COST_SOURCE),
            _ir_run_result({"served_model": "deepseek-v4-pro", "served_provider": "deepseek"}),
        )
    emit.assert_not_called()
