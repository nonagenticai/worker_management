"""Tests for PydanticAI agent — model creation, output extraction, provider config,
run lifecycle, MCP recovery, tool handling, MLX wrappers, and ToolTemplate."""

import asyncio
import contextlib
import json
import os
import time
from contextlib import AsyncExitStack
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx2 as httpx
import pytest
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.exceptions import UserError

from src.worker.provider_registry import EXTERNAL_PROVIDERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "src.worker.agents.pydantic_ai_agent"


@contextlib.contextmanager
def _external_llm_allowed():
    """Admit external providers for construction tests only: a PROVEN platform tenant.

    External/paid providers are refused unless the tenant is BYO or a gitops
    commercial-platform tenant that the request PROVED (owner ruling 2026-10-03;
    the deployment-wide ``ALLOW_EXTERNAL_LLM_PROVIDERS`` flag was removed). The
    tests below assert the SDK wiring of those providers, so they take that
    ground explicitly. ⛔ Never widen this into an autouse fixture: it would
    blind the guard in exactly the tests that are supposed to prove it fires.
    """
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context():
        yield


def _make_agent(
    model_provider="mlx",
    model="test-model",
    base_url="http://test:8080",
    output_schema=None,
    output_validator_code=None,
    mcp_servers=None,
    system_prompt="You are a helpful assistant.",
    instruction=None,
    api_key=None,
    temperature=0.0,
    tools=None,
    max_tokens=None,
    output_retries=None,
    enable_reasoning=None,
):
    """Create a PydanticAIAgent with mocked model initialization."""
    with (
        patch(f"{MODULE}.MLXOpenAIProvider"),
        patch(f"{MODULE}.InferenceRouterOpenAIProvider"),
        patch(f"{MODULE}.OpenAIChatModel"),
        patch(f"{MODULE}.OpenAIProvider"),
        patch(f"{MODULE}.DeepSeekProvider"),
        patch(f"{MODULE}.GroqProvider"),
        patch(f"{MODULE}.GroqModel"),
        patch(f"{MODULE}.CohereProvider"),
        patch(f"{MODULE}.CohereModel"),
        patch(f"{MODULE}.AnthropicProvider"),
        patch(f"{MODULE}.AnthropicModel"),
        patch(f"{MODULE}.XaiProvider"),
        patch(f"{MODULE}.PinnedContextXaiModel"),
        patch(f"{MODULE}.Agent") as mock_agent_cls,
        patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
    ):
        mock_agent_instance = MagicMock()
        mock_agent_instance.output_validator = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        from src.worker.agents.pydantic_ai_agent import (
            PydanticAIAgent,
            PydanticAIAgentConfig,
        )

        # Provide env vars for providers that need API keys
        env = {}
        if model_provider == "deepseek":
            env["DEEPSEEK_API_KEY"] = api_key or "test-key"
        elif model_provider == "groq":
            env["GROQ_API_KEY"] = api_key or "test-key"
        elif model_provider == "openai":
            env["OPENAI_API_KEY"] = api_key or "test-key"
        elif model_provider == "cohere":
            env["COHERE_API_KEY"] = api_key or "test-key"
        elif model_provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = api_key or "test-key"
        elif model_provider == "xai":
            env["XAI_API_KEY"] = api_key or "test-key"

        # An external provider is refused by default (local-only policy). Every
        # caller below that passes one is asserting the CONSTRUCTION path, so the
        # opt-in is taken here rather than duplicated at ~20 call sites. Derived
        # from the registry, never a second hand-written list of the seven.
        external = model_provider in EXTERNAL_PROVIDERS

        with (
            patch.dict(os.environ, env, clear=False),
            _external_llm_allowed() if external else contextlib.nullcontext(),
        ):
            config = PydanticAIAgentConfig(
                name="test_agent",
                model_provider=model_provider,
                model=model,
                db_path="",
                base_url=base_url,
                output_schema=output_schema,
                output_validator_code=output_validator_code,
                mcp_servers=mcp_servers or [],
                system_prompt=system_prompt,
                instruction=instruction,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_reasoning=enable_reasoning,
                output_retries=output_retries,
                tools=tools or [],
            )
            agent = PydanticAIAgent(config=config)
            return agent, mock_agent_cls, mock_agent_instance


# ===================================================================
# Config validation
# ===================================================================


class TestPydanticAIAgentConfig:
    def test_valid_config(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        config = PydanticAIAgentConfig(
            name="myagent",
            model_provider="openai",
            model="gpt-4o",
            db_path="/tmp/test.db",
        )
        assert config.name == "myagent"
        assert config.temperature == 0.0
        assert config.system_prompt == "You are a helpful assistant."

    def test_invalid_empty_name(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(Exception):
            PydanticAIAgentConfig(name="", model_provider="openai", model="gpt-4o", db_path="")

    def test_invalid_temperature(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(Exception):
            PydanticAIAgentConfig(
                name="test", model_provider="openai", model="gpt-4o", db_path="", temperature=5.0
            )

    def test_tools_validator_rejects_non_tool(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(Exception, match="Tool instances"):
            PydanticAIAgentConfig(
                name="test",
                model_provider="openai",
                model="gpt-4o",
                db_path="",
                tools=["not a tool"],
            )

    def test_tools_validator_accepts_tool_instances(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        def dummy(ctx):
            return "ok"

        tool = Tool(dummy, name="dummy", description="A dummy tool")
        config = PydanticAIAgentConfig(
            name="test",
            model_provider="openai",
            model="gpt-4o",
            db_path="",
            tools=[tool],
        )
        assert len(config.tools) == 1

    def test_all_valid_providers(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        for provider in ("openai", "deepseek", "groq", "cohere", "mlx", "inference_router", "xai"):
            config = PydanticAIAgentConfig(
                name="test", model_provider=provider, model="test-model", db_path=""
            )
            assert config.model_provider == provider

    def test_max_tokens_upper_bound_rejected(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(Exception):
            PydanticAIAgentConfig(
                name="test",
                model_provider="inference_router",
                model="local-llm",
                db_path="",
                max_tokens=16385,
            )

    def test_max_tokens_at_ceiling_accepted(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        config = PydanticAIAgentConfig(
            name="test",
            model_provider="inference_router",
            model="local-llm",
            db_path="",
            max_tokens=16384,
        )
        assert config.max_tokens == 16384


class TestResolveMaxTokens:
    """Layer-2 runtime cap — covers the 2026-05-13 vLLM crash defense."""

    def _config(self, *, provider: Any, max_tokens: int | None):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        return PydanticAIAgentConfig(
            name="t",
            model_provider=provider,
            model="m",
            db_path="",
            max_tokens=max_tokens,
        )

    def test_self_hosted_unset_uses_the_self_hosted_default(self):
        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            SELF_HOSTED_PROVIDERS,
            _resolve_max_tokens,
        )

        for provider in SELF_HOSTED_PROVIDERS:
            cfg = self._config(provider=provider, max_tokens=None)
            assert _resolve_max_tokens(cfg) == DEFAULT_SELF_HOSTED_MAX_TOKENS

    def test_self_hosted_default_clears_the_observed_truncation_ceiling(self):
        """The default must exceed the budget that failed in production.

        Regression guard for 2026-08-09: at 4000, three workers carrying no
        explicit max_tokens (ks-relationship-extractor, footage-critic,
        footage-director) died with "Model token limit (4000) exceeded before
        any response was generated". The default is also the reason
        redesign-html and course-outline each needed a hand-written per-worker
        override in the w2v seed.

        Upper bound is the inference-router 300s CHAIN DEADLINE -- not the
        DEGRADED breaker, which is a post-hoc classifier running in a `finally`
        after the call already returned and therefore cannot bound anything.
        Measured against WALL-CLOCK throughput (37.05 tok/s =
        generation_tokens_sum / e2e_request_latency_seconds_sum), NOT the
        62.0 tok/s inter-token rate -- that one excludes prefill and queue wait
        and overstates the headroom by ~1.67x. At 37.05 tok/s a 6000-token
        decode is 162s, while 8000 is 216s and leaves little room for prefill
        and queue on a busy server.
        """
        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            SAFE_MAX_TOKENS,
        )

        assert DEFAULT_SELF_HOSTED_MAX_TOKENS > 4000
        assert DEFAULT_SELF_HOSTED_MAX_TOKENS <= 6000
        assert DEFAULT_SELF_HOSTED_MAX_TOKENS < SAFE_MAX_TOKENS

    def test_self_hosted_under_both_ceilings_passes_through(self):
        """8192 is under BOTH ceilings and must survive untouched.

        ⚠️ This test used to assert 12000 passes through, and that assertion was
        correct for a single-ceiling world and is wrong now: 12000 is under
        SAFE_MAX_TOKENS but ABOVE the time budget, and it is the exact value
        that decoded to nothing on 59 of the 83 lost generations of
        2026-08-24. The pass-through case moved down to a value that genuinely
        fits, rather than the assertion being deleted.
        """
        from src.worker.agents.pydantic_ai_agent import (
            SELF_HOSTED_PROVIDERS,
            _resolve_max_tokens,
        )

        for provider in SELF_HOSTED_PROVIDERS:
            cfg = self._config(provider=provider, max_tokens=8192)
            assert _resolve_max_tokens(cfg) == 8192

    def test_safe_max_tokens_still_binds_when_it_is_the_smaller_ceiling(self):
        """The kernel-crash guard must not be swallowed by the time budget.

        The two clamps exist for unrelated reasons and only the SMALLER may
        apply. Widen the time budget past SAFE_MAX_TOKENS and the FP4-MoE
        guard must still be the thing that binds — otherwise raising a deadline
        silently re-arms the 2026-05-13 crash.
        """
        from src.worker.agents.pydantic_ai_agent import (
            SAFE_MAX_TOKENS,
            SELF_HOSTED_PROVIDERS,
            _resolve_max_tokens,
        )

        with _wide_time_budget():
            for provider in SELF_HOSTED_PROVIDERS:
                cfg = self._config(provider=provider, max_tokens=SAFE_MAX_TOKENS)
                assert _resolve_max_tokens(cfg) == SAFE_MAX_TOKENS

    def test_self_hosted_above_ceiling_is_clamped(self):
        """Defensive: an in-process override or future bypassing of the field
        validator must still get clamped at runtime. Use model_construct to
        sidestep the pydantic le=16384 bound and verify the helper clamps.

        Scoped to a wide time budget as of 2026-08-25: there are now TWO
        ceilings and this one is about the context window, not the clock."""
        from src.worker.agents.pydantic_ai_agent import (
            SAFE_MAX_TOKENS,
            SELF_HOSTED_PROVIDERS,
            PydanticAIAgentConfig,
            _resolve_max_tokens,
        )

        # Widen the time budget so SAFE_MAX_TOKENS is the SMALLER of the two
        # ceilings — otherwise this asserts the time budget by accident and the
        # kernel-crash guard it was written for goes untested.
        with _wide_time_budget():
            for provider in SELF_HOSTED_PROVIDERS:
                cfg = PydanticAIAgentConfig.model_construct(
                    name="t",
                    model_provider=provider,
                    model="m",
                    db_path="",
                    max_tokens=130000,
                )
                assert _resolve_max_tokens(cfg) == SAFE_MAX_TOKENS

    def test_external_provider_max_tokens_unchanged_when_set(self):
        # External providers are NOT clamped. Use a value within the
        # PydanticAIAgentConfig le=16384 bound; the no-clamp behaviour is the
        # point of the test, not the field constraint.
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        for provider in ("openai", "anthropic", "deepseek", "groq", "cohere"):
            cfg = self._config(provider=provider, max_tokens=16000)
            assert _resolve_max_tokens(cfg) == 16000

    def test_external_provider_unset_returns_none(self):
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        for provider in ("openai", "anthropic", "deepseek", "groq", "cohere"):
            cfg = self._config(provider=provider, max_tokens=None)
            assert _resolve_max_tokens(cfg) is None


# ===================================================================
# Output extraction
# ===================================================================


class TestOutputExtraction:
    def test_extract_direct_output(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "Hello world"
        assert agent._extract_text_output(mock_response) == "Hello world"

    def test_extract_none_output_falls_back_to_messages(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_part = MagicMock()
        mock_part.content = "Extracted from messages"
        mock_msg = MagicMock()
        mock_msg.parts = [mock_part]
        mock_response.all_messages.return_value = [mock_msg]
        assert agent._extract_text_output(mock_response) == "Extracted from messages"

    def test_extract_none_output_no_messages(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        assert agent._extract_text_output(mock_response) is None

    def test_extract_dict_output(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = {"key": "value"}
        assert agent._extract_text_output(mock_response) == {"key": "value"}

    def test_extract_empty_string_output(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = ""
        assert agent._extract_text_output(mock_response) == ""

    def test_extract_from_json_messages(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        messages_json = json.dumps(
            [
                {
                    "role": "model-text-response",
                    "parts": [{"part_kind": "text", "content": "From JSON"}],
                }
            ]
        )
        mock_response.all_messages_json.return_value = messages_json
        assert agent._extract_text_output(mock_response) == "From JSON"

    def test_extract_from_json_assistant_role(self):
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        messages_json = json.dumps([{"role": "assistant", "content": "Direct assistant content"}])
        mock_response.all_messages_json.return_value = messages_json
        assert agent._extract_text_output(mock_response) == "Direct assistant content"

    def test_extract_fallback_direct_content_on_msg(self):
        """Test fallback to msg.content when msg.parts has no content."""
        agent, _, _ = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_msg = MagicMock(spec=[])
        mock_msg.content = "direct content"
        # msg has no parts attribute
        mock_response.all_messages.return_value = [mock_msg]
        result = agent._extract_text_output(mock_response)
        assert result == "direct content"


# ===================================================================
# Token usage extraction
# ===================================================================


class TestTokenUsageExtraction:
    def test_extract_from_run_result_usage(self):
        agent, _, _ = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = True
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_result.usage = mock_usage
        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics is not None
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20
        assert metrics.total_tokens == 30

    def test_extract_no_usage_returns_none(self):
        agent, _, _ = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_result.usage = mock_usage
        # Remove model usage path
        agent.agent = MagicMock()
        agent.agent.model = MagicMock()
        agent.agent.model.usage = None
        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics is None

    def test_extract_from_legacy_model_usage(self):
        agent, _, _ = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_result.usage = mock_usage

        # Set up legacy model usage
        model_usage = MagicMock()
        model_usage.input_tokens = None
        model_usage.prompt_tokens = 5
        model_usage.output_tokens = None
        model_usage.completion_tokens = 15
        model_usage.total_tokens = 20
        agent.agent = MagicMock()
        agent.agent.model.usage = model_usage

        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics is not None
        assert metrics.prompt_tokens == 5
        assert metrics.completion_tokens == 15
        assert metrics.total_tokens == 20

    def test_extract_none_run_result(self):
        agent, _, _ = _make_agent()
        agent.agent = MagicMock()
        agent.agent.model.usage = None
        metrics = agent._extract_token_usage("test", None)
        assert metrics is None

    def test_extract_exception_returns_none(self):
        agent, _, _ = _make_agent()
        mock_result = MagicMock()
        type(mock_result).usage = PropertyMock(side_effect=RuntimeError("broken"))
        agent.agent = MagicMock()
        agent.agent.model = MagicMock(spec=[])  # no usage attribute
        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics is None


# ===================================================================
# max_tokens cap-hit detection (truncated completions)
# ===================================================================


def _recording_metrics_client():
    """Build a real AgentMetricsClient wired to an in-memory OTLP reader.

    Bypasses ``__init__`` (which probes the network and no-ops without an OTLP
    endpoint) so the instruments are real and their values are readable. The
    counter is process-global per meter provider, so tests read the value before
    AND after rather than assuming it starts at zero.

    ``OTEL_SDK_DISABLED`` is cleared for the duration of the MeterProvider
    construction: ``tests/test_agent_metrics.py:13`` sets it to "true" at module
    import and never restores it, so in a full-suite run every SDK instrument
    created afterwards is a no-op and this counter would silently read 0. The
    SDK latches the flag when the provider is built, hence the narrow scope.
    """
    import os
    from unittest.mock import patch as _patch

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from src.telemetry.agent_metrics import AgentMetricsClient

    reader = InMemoryMetricReader()
    env_without_flag = {k: v for k, v in os.environ.items() if k != "OTEL_SDK_DISABLED"}
    with _patch.dict(os.environ, env_without_flag, clear=True):
        provider = MeterProvider(metric_readers=[reader])

    client = AgentMetricsClient.__new__(AgentMetricsClient)
    client.enabled = True
    client.meter = provider.get_meter("test-max-tokens-cap")
    client._counters = {}
    client._histograms = {}
    client._up_down_counters = {}
    client._setup_metrics()
    return client, reader


def _counter_value(reader, name="agent_max_tokens_cap_hit_total"):
    """Sum every data point of ``name`` currently held by the in-memory reader."""
    data = reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == name:
                    total += sum(point.value for point in metric.data.data_points)
    return total


def _run_result(completion_tokens, finish_reason=None):
    """Mock an AgentRunResult reporting ``completion_tokens`` output tokens.

    ``finish_reason=None`` mimics a provider that reported none, which is what
    forces the token-equality fallback.
    """
    result = MagicMock()
    usage = MagicMock()
    usage.has_values.return_value = True
    usage.input_tokens = 100
    usage.output_tokens = completion_tokens
    usage.total_tokens = 100 + completion_tokens
    result.usage = usage

    response_message = MagicMock()
    response_message.kind = "response"
    response_message.finish_reason = finish_reason
    result.all_messages.return_value = [response_message]
    return result


class TestMaxTokensCapHit:
    def test_completion_at_cap_warns_and_increments_counter(self, caplog):
        """completion == the cap that was actually sent => cap hit."""
        import logging

        from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS

        agent, _, _ = _make_agent(model_provider="mlx")
        assert agent._max_tokens_cap == DEFAULT_SELF_HOSTED_MAX_TOKENS

        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        with (
            patch(f"{MODULE}.get_metrics_client", return_value=client),
            caplog.at_level(logging.WARNING),
        ):
            metrics = agent._extract_token_usage(
                "test_agent", _run_result(DEFAULT_SELF_HOSTED_MAX_TOKENS)
            )

        assert metrics is not None
        assert metrics.completion_tokens == DEFAULT_SELF_HOSTED_MAX_TOKENS
        assert _counter_value(reader) == before + 1

        warnings = [
            r.getMessage() for r in caplog.records if "max_tokens cap hit" in r.getMessage()
        ]
        assert len(warnings) == 1
        cap_records = [r for r in caplog.records if "max_tokens cap hit" in r.getMessage()]
        assert all(r.levelno == logging.WARNING for r in cap_records)
        assert "test_agent" in warnings[0]
        assert "mlx" in warnings[0]
        assert str(DEFAULT_SELF_HOSTED_MAX_TOKENS) in warnings[0]

    def test_finish_reason_length_counts_even_below_cap(self, caplog):
        """finish_reason='length' is authoritative and beats the token proxy."""
        import logging

        agent, _, _ = _make_agent(model_provider="mlx")
        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        with (
            patch(f"{MODULE}.get_metrics_client", return_value=client),
            caplog.at_level(logging.WARNING),
        ):
            agent._extract_token_usage("test_agent", _run_result(120, finish_reason="length"))

        assert _counter_value(reader) == before + 1
        warnings = [r for r in caplog.records if "max_tokens cap hit" in r.getMessage()]
        assert len(warnings) == 1
        assert "finish_reason=length" in warnings[0].getMessage()

    def test_finish_reason_stop_at_cap_is_not_a_cap_hit(self, caplog):
        """A model that stopped on its own but landed on the budget is not truncated."""
        import logging

        from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS

        agent, _, _ = _make_agent(model_provider="mlx")
        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        with (
            patch(f"{MODULE}.get_metrics_client", return_value=client),
            caplog.at_level(logging.WARNING),
        ):
            agent._extract_token_usage(
                "test_agent",
                _run_result(DEFAULT_SELF_HOSTED_MAX_TOKENS, finish_reason="stop"),
            )

        assert _counter_value(reader) == before
        assert not [r for r in caplog.records if "max_tokens cap hit" in r.getMessage()]

    def test_completion_well_under_cap_does_not_warn_or_increment(self, caplog):
        import logging

        agent, _, _ = _make_agent(model_provider="mlx")
        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        with (
            patch(f"{MODULE}.get_metrics_client", return_value=client),
            caplog.at_level(logging.WARNING),
        ):
            metrics = agent._extract_token_usage("test_agent", _run_result(143))

        assert metrics is not None
        assert metrics.completion_tokens == 143
        assert _counter_value(reader) == before
        assert not [r for r in caplog.records if "max_tokens cap hit" in r.getMessage()]

    def test_no_cap_configured_never_counts(self, caplog):
        """External providers may send no max_tokens at all — nothing to compare."""
        import logging

        agent, _, _ = _make_agent(model_provider="openai")
        assert agent._max_tokens_cap is None

        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        with (
            patch(f"{MODULE}.get_metrics_client", return_value=client),
            caplog.at_level(logging.WARNING),
        ):
            agent._extract_token_usage("test_agent", _run_result(9999, finish_reason="length"))

        assert _counter_value(reader) == before
        assert not [r for r in caplog.records if "max_tokens cap hit" in r.getMessage()]

    def test_legacy_model_usage_branch_also_detects_cap_hit(self):
        """The legacy model-usage branch must count too, not just the primary one."""
        from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS

        agent, _, _ = _make_agent(model_provider="mlx")
        client, reader = _recording_metrics_client()
        before = _counter_value(reader)

        run_result = MagicMock()
        usage = MagicMock()
        usage.has_values.return_value = False
        run_result.usage = usage
        run_result.all_messages.return_value = []  # no finish_reason available

        model_usage = MagicMock()
        model_usage.input_tokens = 100
        model_usage.output_tokens = DEFAULT_SELF_HOSTED_MAX_TOKENS
        model_usage.total_tokens = 100 + DEFAULT_SELF_HOSTED_MAX_TOKENS
        agent.agent = MagicMock()
        agent.agent.model.usage = model_usage

        with patch(f"{MODULE}.get_metrics_client", return_value=client):
            metrics = agent._extract_token_usage("test_agent", run_result)

        assert metrics is not None
        assert _counter_value(reader) == before + 1


# ===================================================================
# Model creation per provider
# ===================================================================


class TestMLXModelCreation:
    def test_mlx_without_base_url_raises(self):
        env_backup = os.environ.pop("OPENAI_API_BASE", None)
        try:
            with patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test", model_provider="mlx", model="test-model", db_path="", base_url=None
                )
                with pytest.raises(UserError, match="No MLX instance URL configured"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["OPENAI_API_BASE"] = env_backup

    def test_mlx_with_base_url_succeeds(self):
        agent, _, _ = _make_agent(model_provider="mlx", base_url="http://mlx:8080")
        assert agent is not None

    def test_mlx_with_env_var_succeeds(self):
        with patch.dict(os.environ, {"OPENAI_API_BASE": "http://mini1.example.com/v1"}):
            agent, _, _ = _make_agent(model_provider="mlx", base_url=None)
            assert agent is not None


class TestDeepSeekModelCreation:
    def test_deepseek_creation_with_key(self):
        agent, _, _ = _make_agent(model_provider="deepseek", model="deepseek-chat")
        assert agent.config.model_provider == "deepseek"

    def test_deepseek_without_key_raises(self):
        env_backup = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="deepseek",
                    model="deepseek-chat",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for DeepSeek"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["DEEPSEEK_API_KEY"] = env_backup

    def test_deepseek_with_custom_base_url(self):
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            patch(f"{MODULE}.DeepSeekProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
            patch(f"{MODULE}.AsyncOpenAI") as mock_openai,
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="deepseek",
                model="deepseek-chat",
                db_path="",
                base_url="http://custom:9090",
            )
            PydanticAIAgent(config=config)
            # Should use custom client
            mock_openai.assert_called_once()


class TestXaiModelCreation:
    def test_xai_creation_with_key(self):
        agent, _, _ = _make_agent(model_provider="xai", model="grok-4.5")
        assert agent.config.model_provider == "xai"

    def test_xai_without_key_raises(self):
        env_backup = os.environ.pop("XAI_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="xai",
                    model="grok-4.5",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for xAI"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["XAI_API_KEY"] = env_backup

    def test_xai_uses_xaiprovider_and_xaimodel(self):
        """Grok routes through the native xAI SDK classes, not OpenAIChatModel."""
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"XAI_API_KEY": "test-key"}),
            patch(f"{MODULE}.XaiProvider") as mock_provider,
            patch(f"{MODULE}.PinnedContextXaiModel") as mock_model,
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="xai",
                model="grok-4.5",
                db_path="",
            )
            PydanticAIAgent(config=config)
            mock_provider.assert_called_once_with(api_key="test-key")
            mock_model.assert_called_once()
            # model name is passed positionally as the first arg
            assert mock_model.call_args.args[0] == "grok-4.5"


class TestGroqModelCreation:
    def test_groq_creation_with_key(self):
        agent, _, _ = _make_agent(model_provider="groq", model="llama-3.3-70b-versatile")
        assert agent.config.model_provider == "groq"

    def test_groq_without_key_raises(self):
        env_backup = os.environ.pop("GROQ_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="groq",
                    model="llama-3.3",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for Groq"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["GROQ_API_KEY"] = env_backup

    def test_groq_with_custom_base_url(self):
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch(f"{MODULE}.GroqProvider"),
            patch(f"{MODULE}.GroqModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
            patch(f"{MODULE}.AsyncGroq") as mock_groq,
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="groq",
                model="llama-3.3",
                db_path="",
                base_url="http://custom:9090",
            )
            PydanticAIAgent(config=config)
            mock_groq.assert_called_once()


class TestOpenAIModelCreation:
    def test_openai_creation_with_key(self):
        agent, _, _ = _make_agent(model_provider="openai", model="gpt-4o")
        assert agent.config.model_provider == "openai"

    def test_openai_without_key_raises(self):
        env_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="openai",
                    model="gpt-4o",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for OpenAI"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["OPENAI_API_KEY"] = env_backup

    def test_openai_with_custom_base_url(self):
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(f"{MODULE}.OpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
            patch(f"{MODULE}.AsyncOpenAI") as mock_openai,
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="openai",
                model="gpt-4o",
                db_path="",
                base_url="http://custom:9090",
            )
            PydanticAIAgent(config=config)
            mock_openai.assert_called_once()


class TestCohereModelCreation:
    def test_cohere_creation_with_key(self):
        agent, _, _ = _make_agent(model_provider="cohere", model="command-r-plus")
        assert agent.config.model_provider == "cohere"

    def test_cohere_without_key_raises(self):
        env_backup = os.environ.pop("COHERE_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="cohere",
                    model="command-r",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for Cohere"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["COHERE_API_KEY"] = env_backup

    def test_cohere_with_custom_base_url(self):
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}),
            patch(f"{MODULE}.CohereProvider"),
            patch(f"{MODULE}.CohereModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
            patch(f"{MODULE}.AsyncCohereClient") as mock_cohere,
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="cohere",
                model="command-r",
                db_path="",
                base_url="http://custom:9090",
            )
            PydanticAIAgent(config=config)
            mock_cohere.assert_called_once()


class TestInferenceRouterProvider:
    def test_inference_router_is_valid_provider(self):
        agent, _, _ = _make_agent(
            model_provider="inference_router",
            model="entity-spotter",
            base_url="http://inference-router:8150",
        )
        assert agent.config.model_provider == "inference_router"

    def test_inference_router_without_base_url_raises(self):
        env_backup = os.environ.pop("INFERENCE_ROUTER_URL", None)
        try:
            with patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="inference_router",
                    model="test-model",
                    db_path="",
                    base_url=None,
                )
                with pytest.raises(UserError, match="No inference-router URL configured"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["INFERENCE_ROUTER_URL"] = env_backup

    def test_inference_router_with_env_var_succeeds(self):
        with patch.dict(os.environ, {"INFERENCE_ROUTER_URL": "http://ir:8150/v1"}):
            agent, _, _ = _make_agent(
                model_provider="inference_router", model="test-model", base_url=None
            )
            assert agent is not None

    def test_inference_router_does_not_pass_schema_id(self):
        """Schema-id-based routing was removed; inference-router self-derives from body fingerprint."""
        with (
            patch(f"{MODULE}.InferenceRouterOpenAIProvider") as mock_ir,
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test_ir",
                model_provider="inference_router",
                model="local-llm",
                db_path="",
                base_url="http://ir:8150",
            )
            PydanticAIAgent(config=config)
            call_kwargs = mock_ir.call_args[1]
            assert "schema_id" not in call_kwargs

    def test_inference_router_appends_api(self):
        with (
            patch(f"{MODULE}.InferenceRouterOpenAIProvider") as mock_ir,
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test_ir",
                model_provider="inference_router",
                model="test-model",
                db_path="",
                base_url="http://ir:8150",
            )
            PydanticAIAgent(config=config)
            assert mock_ir.call_args[1]["base_url"] == "http://ir:8150/api"

    def test_inference_router_preserves_existing_api(self):
        with (
            patch(f"{MODULE}.InferenceRouterOpenAIProvider") as mock_ir,
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test_ir",
                model_provider="inference_router",
                model="test-model",
                db_path="",
                base_url="http://ir:8150/api",
            )
            PydanticAIAgent(config=config)
            assert mock_ir.call_args[1]["base_url"] == "http://ir:8150/api"

    def test_inference_router_passes_model_name(self):
        with (
            patch(f"{MODULE}.InferenceRouterOpenAIProvider"),
            # The inference-router path builds InferenceRouterOpenAIChatModel, the
            # subclass that surfaces IR's served_model/served_provider extras — not
            # the bare OpenAIChatModel. Patching the base class here would no longer
            # intercept the construction.
            patch(f"{MODULE}.InferenceRouterOpenAIChatModel") as mock_model,
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test_ir",
                model_provider="inference_router",
                model="entity-spotter-v3",
                db_path="",
                base_url="http://ir:8150",
            )
            PydanticAIAgent(config=config)
            assert mock_model.call_args[1]["model_name"] == "entity-spotter-v3"


class TestAnthropicModelCreation:
    def test_anthropic_creation_with_key(self):
        agent, _, _ = _make_agent(
            model_provider="anthropic", model="claude-sonnet-4-6", base_url=None
        )
        assert agent.config.model_provider == "anthropic"

    def test_anthropic_without_key_raises(self):
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with _external_llm_allowed(), patch(f"{MODULE}.Agent"):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="test",
                    model_provider="anthropic",
                    model="claude-sonnet-4-6",
                    db_path="",
                )
                with pytest.raises(UserError, match="API key required for Anthropic"):
                    PydanticAIAgent(config=config)
        finally:
            if env_backup:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

    def test_anthropic_passes_key_to_provider(self):
        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}),
            patch(f"{MODULE}.AnthropicProvider") as mock_provider,
            patch(f"{MODULE}.AnthropicModel") as mock_model,
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="anthropic",
                model="claude-sonnet-4-6",
                db_path="",
            )
            PydanticAIAgent(config=config)
            assert mock_provider.call_args[1]["api_key"] == "sk-ant-test"
            assert mock_model.call_args[1]["model_name"] == "claude-sonnet-4-6"


class TestVllmModelCreation:
    def test_vllm_creation_uses_default_url(self):
        agent, _, _ = _make_agent(model_provider="vllm", model="local-llm", base_url=None)
        assert agent.config.model_provider == "vllm"

    def test_vllm_default_url_applied(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(f"{MODULE}.OpenAIProvider") as mock_provider,
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            os.environ.pop("VLLM_URL", None)
            from src.worker.agents.pydantic_ai_agent import (
                VLLM_DEFAULT_URL,
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="vllm",
                model="local-llm",
                db_path="",
                base_url=None,
            )
            PydanticAIAgent(config=config)
            assert mock_provider.call_args[1]["base_url"] == VLLM_DEFAULT_URL + "/v1"
            # Same placeholder as the inference_router path, never vLLM's "EMPTY":
            # VLLM_URL points at inference-router (gitops #2999).
            from src.worker.agents.pydantic_ai_agent import IR_CLIENT_API_KEY

            assert mock_provider.call_args[1]["api_key"] == IR_CLIENT_API_KEY != "EMPTY"

    def test_vllm_env_var_overrides_default(self):
        with (
            patch.dict(os.environ, {"VLLM_URL": "http://custom-vllm:9999"}),
            patch(f"{MODULE}.OpenAIProvider") as mock_provider,
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="vllm",
                model="local-llm",
                db_path="",
                base_url=None,
            )
            PydanticAIAgent(config=config)
            assert mock_provider.call_args[1]["base_url"] == "http://custom-vllm:9999/v1"


class TestVlmLocalModelCreation:
    def test_vlm_local_creation_uses_default_url(self, monkeypatch):
        # vlm_local is served via inference-router (2026-10-07).
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://ir:8150")
        agent, _, _ = _make_agent(
            model_provider="vlm_local",
            model="mlx-community/Qwen2-VL-7B-Instruct-4bit",
            base_url=None,
        )
        assert agent.config.model_provider == "vlm_local"

    def test_vlm_local_routes_via_inference_router(self):
        with (
            patch.dict(os.environ, {"INFERENCE_ROUTER_URL": "http://ir:8150"}),
            patch(f"{MODULE}.InferenceRouterOpenAIProvider") as mock_provider,
            patch(f"{MODULE}.InferenceRouterOpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="test",
                model_provider="vlm_local",
                model="mlx-community/Qwen2-VL-7B-Instruct-4bit",
                db_path="",
                base_url="http://config-vlm:8100/api/vlm",
            )
            PydanticAIAgent(config=config)
            kwargs = mock_provider.call_args[1]
            # A raw backend URL on the row is never dialled.
            assert kwargs["base_url"] == "http://ir:8150/api"
            assert kwargs["pinned_provider"] == "vlm_local"


class TestProviderMetadata:
    def test_metadata_covers_all_enum_values(self):
        import typing

        from src.worker.agents.pydantic_ai_agent import PROVIDER_METADATA, ModelProvider

        enum_values = set(typing.get_args(ModelProvider))
        assert set(PROVIDER_METADATA.keys()) == enum_values, (
            "PROVIDER_METADATA drift: missing or extra keys relative to ModelProvider"
        )

    def test_metadata_shape(self):
        from src.worker.agents.pydantic_ai_agent import PROVIDER_METADATA

        for name, meta in PROVIDER_METADATA.items():
            assert "label" in meta, name
            assert "description" in meta, name
            assert "requires_api_key" in meta, name
            assert isinstance(meta["requires_api_key"], bool), name


# ===================================================================
# Retry configuration
# ===================================================================


class TestRetryConfiguration:
    def test_mlx_gets_10_retries(self):
        _, mock_agent_cls, _ = _make_agent(model_provider="mlx", base_url="http://test:8080")
        assert mock_agent_cls.call_args[1]["retries"] == 10

    def test_inference_router_gets_0_retries(self):
        """inference_router uses NativeOutput + vLLM guided-decoding so
        validation failures on the FIRST attempt are vanishingly rare.
        With pgx-local at max_num_seqs=1, any retry budget here amplifies
        a slow but correct first response into a 3× duplicate-request
        storm against the queue (3 × 120s read = 360s ≈ the 5-min cycle
        seen on trust.pdf 2026-05-19). Transport-level robustness comes
        from AsyncHTTPTransport(retries=ir_http_transport_retries)."""
        _, mock_agent_cls, _ = _make_agent(
            model_provider="inference_router", base_url="http://ir:8150"
        )
        assert mock_agent_cls.call_args[1]["retries"] == 0

    def test_non_mlx_gets_3_retries(self):
        _, mock_agent_cls, _ = _make_agent(model_provider="deepseek", model="deepseek-chat")
        assert mock_agent_cls.call_args[1]["retries"] == 3

    def test_output_retries_override_on_inference_router(self):
        """Per-worker output_retries overrides the inference_router=0 default.
        This is the cai-architecture-classifier path: short-prompt structured
        output where a single guided-decoding hiccup must not fail the call."""
        _, mock_agent_cls, _ = _make_agent(
            model_provider="inference_router",
            base_url="http://ir:8150",
            output_retries=2,
        )
        assert mock_agent_cls.call_args[1]["retries"] == 2

    def test_output_retries_override_on_mlx(self):
        """Override wins even when the provider default is already high (mlx=10)."""
        _, mock_agent_cls, _ = _make_agent(
            model_provider="mlx",
            base_url="http://test:8080",
            output_retries=1,
        )
        assert mock_agent_cls.call_args[1]["retries"] == 1

    def test_output_retries_override_on_other(self):
        _, mock_agent_cls, _ = _make_agent(
            model_provider="deepseek",
            model="deepseek-chat",
            output_retries=5,
        )
        assert mock_agent_cls.call_args[1]["retries"] == 5

    def test_output_retries_none_preserves_provider_defaults(self):
        """Regression guard: None must preserve the exact provider defaults so
        the trust.pdf-storm invariant on inference_router (=0) is never silently
        flipped globally. See pydantic_ai_agent.py:1205-1214."""
        _, mock_mlx, _ = _make_agent(
            model_provider="mlx", base_url="http://test:8080", output_retries=None
        )
        assert mock_mlx.call_args[1]["retries"] == 10

        _, mock_ir, _ = _make_agent(
            model_provider="inference_router",
            base_url="http://ir:8150",
            output_retries=None,
        )
        assert mock_ir.call_args[1]["retries"] == 0

        _, mock_other, _ = _make_agent(
            model_provider="deepseek", model="deepseek-chat", output_retries=None
        )
        assert mock_other.call_args[1]["retries"] == 3

    def test_output_retries_zero_is_explicit_override(self):
        """output_retries=0 is a meaningful explicit value (not falsy-as-None).
        Catches the `if config.output_retries:` truthy-bug regression."""
        _, mock_agent_cls, _ = _make_agent(
            model_provider="mlx", base_url="http://test:8080", output_retries=0
        )
        assert mock_agent_cls.call_args[1]["retries"] == 0

    def test_output_retries_bounds_rejected(self):
        """Pydantic rejects out-of-bounds values at config construction."""
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(Exception):
            PydanticAIAgentConfig(
                name="x", model_provider="mlx", model="m", db_path="", output_retries=-1
            )
        with pytest.raises(Exception):
            PydanticAIAgentConfig(
                name="x", model_provider="mlx", model="m", db_path="", output_retries=11
            )


# ===================================================================
# Output type selection
# ===================================================================


class TestOutputTypeSelection:
    def test_mlx_uses_prompted_output(self):
        with patch(f"{MODULE}.PromptedOutput") as mock_prompted:
            _make_agent(
                model_provider="mlx",
                base_url="http://test:8080",
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
            mock_prompted.assert_called_once()

    def test_inference_router_uses_native_output(self):
        """inference_router routes to vLLM (Nemotron) + deepseek + anthropic;
        all three honor OpenAI's response_format=json_schema. NativeOutput
        sends the schema in response_format so the provider returns
        schema-conformant JSON on the first try (vLLM via xgrammar guided-
        decoding by default in v0.7+). Replaces the prior PromptedOutput
        mode that was driving a malformed-JSON retry storm."""
        with (
            patch(f"{MODULE}.NativeOutput") as mock_native,
            patch(f"{MODULE}.PromptedOutput") as mock_prompted,
        ):
            _make_agent(
                model_provider="inference_router",
                base_url="http://ir:8150",
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
            mock_native.assert_called_once()
            mock_prompted.assert_not_called()

    def test_non_mlx_does_not_use_prompted_output(self):
        with patch(f"{MODULE}.PromptedOutput") as mock_prompted:
            _make_agent(
                model_provider="deepseek",
                model="deepseek-chat",
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
            mock_prompted.assert_not_called()

    def test_no_schema_uses_str(self):
        _, mock_agent_cls, _ = _make_agent(model_provider="mlx", base_url="http://test:8080")
        assert mock_agent_cls.call_args[1]["output_type"] is str

    # ------------------------------------------------------------------
    # _output_type_from_schema — scalar / list / object roots
    #
    # pydantic-ai's StructuredDict only accepts object-typed JSON schemas
    # ("Schema must be an object"). Scalar roots (boolean / number / string)
    # and array roots need to be handed to pydantic-ai as bare Python types
    # so it builds the appropriate response_format. These tests pin that
    # mapping and the per-provider mode wrap.
    # ------------------------------------------------------------------

    def test_helper_none_maps_to_str(self):
        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        assert _output_type_from_schema(None, "vllm") is str
        assert _output_type_from_schema({}, "vllm") is str

    def test_helper_boolean_maps_to_bool_bare(self):
        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        # Bare providers (no NativeOutput / PromptedOutput wrap)
        assert _output_type_from_schema({"type": "boolean"}, "vllm") is bool
        assert _output_type_from_schema({"type": "boolean"}, "deepseek") is bool

    def test_helper_boolean_wraps_for_inference_router(self):
        from pydantic_ai import NativeOutput

        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        out = _output_type_from_schema({"type": "boolean"}, "inference_router")
        assert isinstance(out, NativeOutput)

    def test_helper_boolean_wraps_for_mlx(self):
        from pydantic_ai import PromptedOutput

        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        out = _output_type_from_schema({"type": "boolean"}, "mlx")
        assert isinstance(out, PromptedOutput)

    def test_helper_scalar_types_round_trip(self):
        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        assert _output_type_from_schema({"type": "integer"}, "vllm") is int
        assert _output_type_from_schema({"type": "number"}, "vllm") is float
        assert _output_type_from_schema({"type": "string"}, "vllm") is str

    def test_helper_array_root_becomes_list(self):
        from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

        out = _output_type_from_schema({"type": "array", "items": {"type": "string"}}, "vllm")
        # list[str] — pydantic-ai accepts parameterised generics directly.
        assert getattr(out, "__origin__", None) is list

    def test_boolean_schema_does_not_crash_setup_agent(self):
        """Regression: previously raised 'Schema must be an object' for any
        non-object root, falling back to a non-agent worker that ignored the
        schema and returned free-form text."""
        _, mock_agent_cls, _ = _make_agent(
            model_provider="inference_router",
            base_url="http://ir:8150",
            output_schema={"type": "boolean"},
        )
        # output_type must be present on the Agent kwargs (a NativeOutput),
        # not silently dropped.
        kwargs = mock_agent_cls.call_args[1]
        assert "output_type" in kwargs
        assert kwargs["output_type"] is not str


# ===================================================================
# Model settings
# ===================================================================


class TestModelSettings:
    def test_openai_model_settings(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(model_provider="openai", temperature=0.5)
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["temperature"] == 0.5

    def test_mlx_model_settings_has_max_tokens(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            PydanticAIAgent,
        )

        agent, _, _ = _make_agent(model_provider="mlx", temperature=0.3, base_url="http://t:8080")
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS
        assert settings["temperature"] == 0.3

    def test_groq_model_settings(self):
        from pydantic_ai.models.groq import GroqModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(model_provider="groq", temperature=0.8)
        config = agent.config
        mock_model = MagicMock(spec=GroqModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["temperature"] == 0.8

    def test_generic_model_settings(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(model_provider="cohere", temperature=0.6)
        config = agent.config
        # Use a mock that is NOT an instance of OpenAIChatModel or GroqModel
        mock_model = MagicMock(spec=[])
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["temperature"] == 0.6

    def test_get_model_settings_uses_config_max_tokens_when_set(self):
        """When config.max_tokens is set, it overrides the provider-default hardcode."""
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(
            model_provider="inference_router",
            temperature=0.1,
            base_url="http://ir:8150",
            max_tokens=6000,
        )
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["max_tokens"] == 6000

    def test_get_model_settings_falls_back_to_default_for_inference_router(self):
        """inference_router without config.max_tokens gets the self-hosted default.

        Asserted against the constant, not a literal: this test previously
        hardcoded 4000, so it pinned the number rather than the behaviour and
        would have had to be edited by anyone changing the default -- exactly
        the kind of edit that invites changing the test to match a mistake.
        """
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            PydanticAIAgent,
        )

        agent, _, _ = _make_agent(
            model_provider="inference_router",
            temperature=0.1,
            base_url="http://ir:8150",
        )
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS

    def test_get_model_settings_falls_back_to_default_for_mlx(self):
        """mlx without config.max_tokens gets the self-hosted default."""
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            PydanticAIAgent,
        )

        agent, _, _ = _make_agent(model_provider="mlx", temperature=0.1, base_url="http://mlx:8080")
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS

    def test_get_model_settings_omits_max_tokens_for_openai_when_none(self):
        """OpenAI (and other non-mlx/non-inference_router) providers omit max_tokens
        when config.max_tokens is None, preserving the pre-existing behavior."""
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(model_provider="openai", temperature=0.2, api_key="test-key")
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        # pydantic-ai ModelSettings is a TypedDict backed by a dict — missing
        # keys indicate the field was not set.
        assert "max_tokens" not in settings

    def test_get_model_settings_applies_config_max_tokens_to_openai(self):
        """OpenAI honours config.max_tokens when set, even though it has no
        hardcoded default."""
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(
            model_provider="openai",
            temperature=0.2,
            api_key="test-key",
            max_tokens=2048,
        )
        config = agent.config
        mock_model = MagicMock(spec=OpenAIChatModel)
        settings = PydanticAIAgent._get_model_settings(agent, config, mock_model)
        assert settings is not None
        assert settings["max_tokens"] == 2048


# ===================================================================
# enable_reasoning -> extra_body (per-provider wire shape)
# ===================================================================


class TestReasoningExtraBody:
    """The chain-of-thought switch, whose wire shape DIFFERS BY PROVIDER.

    Sending the wrong shape is a silent no-op — accepted, forwarded, never read —
    so each provider family is asserted on its exact payload, not on "some
    extra_body is present".
    """

    @staticmethod
    def _settings(**kwargs):
        from pydantic_ai.models.openai import OpenAIChatModel

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        agent, _, _ = _make_agent(**kwargs)
        mock_model = MagicMock(spec=OpenAIChatModel)
        return PydanticAIAgent._get_model_settings(agent, agent.config, mock_model)

    def test_inference_router_uses_flat_enable_reasoning(self):
        """IR declares ``enable_reasoning`` itself and rewrites it to
        chat_template_kwargs.enable_thinking; the flat key is the shape it reads."""
        settings = self._settings(
            model_provider="inference_router",
            temperature=0.1,
            base_url="http://ir:8150",
            enable_reasoning=False,
        )
        assert settings is not None
        assert settings["extra_body"] == {"enable_reasoning": False}

    def test_inference_router_true_is_forwarded_too(self):
        """The switch is not a one-way opt-out: True must reach IR unchanged."""
        settings = self._settings(
            model_provider="inference_router",
            temperature=0.1,
            base_url="http://ir:8150",
            enable_reasoning=True,
        )
        assert settings is not None
        assert settings["extra_body"] == {"enable_reasoning": True}

    def test_vllm_uses_chat_template_kwargs(self):
        """Direct vLLM has nothing in front of it to translate, so it needs vLLM's
        own kwarg. This is the path every auto-created worker actually takes."""
        settings = self._settings(
            model_provider="vllm",
            temperature=0.1,
            base_url="http://vllm:8000",
            enable_reasoning=False,
        )
        assert settings is not None
        assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_mlx_uses_chat_template_kwargs(self):
        settings = self._settings(
            model_provider="mlx",
            temperature=0.1,
            base_url="http://mlx:8080",
            enable_reasoning=False,
        )
        assert settings is not None
        assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_none_emits_no_extra_body_key_at_all(self):
        """The default-preserving case: absent, not False, not an empty dict.

        ModelSettings is a TypedDict backed by a plain dict, so a missing key is
        the only proof the outbound payload is byte-for-byte unchanged.

        ``vllm`` is deliberately NOT in this list — see
        ``test_vllm_none_defaults_thinking_off``. Sending nothing to pgx is not a
        neutral default, so it gets its own case rather than sharing this one.
        """
        for provider, base_url in (
            ("inference_router", "http://ir:8150"),
            ("mlx", "http://mlx:8080"),
        ):
            settings = self._settings(
                model_provider=provider,
                temperature=0.1,
                base_url=base_url,
                enable_reasoning=None,
            )
            assert settings is not None
            assert "extra_body" not in settings, provider

    def test_vllm_none_defaults_thinking_off(self):
        """An UNSET switch must still turn thinking off on the direct vLLM path.

        pgx serves Qwen3.8, whose chat template defaults chain-of-thought ON. The
        model it replaced defaulted OFF, so "send nothing" silently flipped it on
        for every caller that does not pass the flag: measured 2026-08-24, the same
        prompt cost 420 completion tokens / 49.5 s unswitched vs 28 / 3.5 s with
        the switch, for the same answer. 960 of 1009 vllm agents built in a week
        carried no switch.

        Asserted on the exact payload, not on "some extra_body is present": the
        wrong wire shape is accepted, forwarded and never read.
        """
        settings = self._settings(
            model_provider="vllm",
            temperature=0.1,
            base_url="http://vllm:8000",
            enable_reasoning=None,
        )
        assert settings is not None
        assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_vllm_explicit_true_still_wins_over_the_default(self):
        """The default is a default, not a lock. Without this the change would be
        indistinguishable from hardcoding thinking off, and the one caller that
        wants chain-of-thought would have no way back."""
        settings = self._settings(
            model_provider="vllm",
            temperature=0.1,
            base_url="http://vllm:8000",
            enable_reasoning=True,
        )
        assert settings is not None
        assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_vllm_default_does_not_leak_to_other_self_hosted_providers(self, monkeypatch):
        """SELF_HOSTED_PROVIDERS also covers mlx and vlm_local, which are different
        servers serving different models. Neither has been measured, so an unset
        switch must stay unset for them — the same refusal-to-guess that keeps
        external providers out of this function."""
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://ir:8150")  # vlm_local -> IR
        for provider, base_url in (("mlx", "http://mlx:8080"), ("vlm_local", "http://vlm:8090")):
            settings = self._settings(
                model_provider=provider,
                temperature=0.1,
                base_url=base_url,
                enable_reasoning=None,
            )
            assert settings is not None
            assert "extra_body" not in settings, provider

    def test_external_provider_gets_no_extra_body(self):
        """openai is not self-hosted and its reasoning parameter is unverified —
        we emit NOTHING rather than guess a shape that would silently no-op."""
        settings = self._settings(
            model_provider="openai",
            temperature=0.2,
            api_key="test-key",
            enable_reasoning=False,
        )
        assert settings is not None
        assert "extra_body" not in settings

    def test_reasoning_does_not_disturb_max_tokens(self):
        """extra_body is applied as a post-step; the max_tokens resolution that
        shares the function must survive it untouched."""
        from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS

        settings = self._settings(
            model_provider="vllm",
            temperature=0.1,
            base_url="http://vllm:8000",
            enable_reasoning=False,
        )
        assert settings is not None
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS
        assert settings["temperature"] == 0.1


# ===================================================================
# Prompt building with graph context
# ===================================================================


class TestPromptBuilding:
    def test_build_context_aware_prompt_basic(self):
        agent, _, _ = _make_agent()
        result = agent._build_context_aware_prompt("Do the task", {}, {})
        assert "Do the task" in result

    def test_build_context_with_graph_info(self):
        agent, _, _ = _make_agent()
        graph_ctx = {
            "session_id": "sess-123",
            "graph_id": "graph-456",
            "node_id": "node-1",
            "completed_nodes": ["node-0"],
        }
        result = agent._build_context_aware_prompt("Task", graph_ctx, {})
        assert "sess-123" in result
        assert "graph-456" in result
        assert "node-1" in result
        assert "node-0" in result

    def test_build_context_with_previous_outputs(self):
        agent, _, _ = _make_agent()
        graph_ctx = {
            "previous_outputs": {"node-0": "some output"},
        }
        result = agent._build_context_aware_prompt("Task", graph_ctx, {})
        assert "node-0" in result
        assert "some output" in result

    def test_build_context_truncates_long_outputs(self):
        agent, _, _ = _make_agent()
        long_output = "x" * 500
        graph_ctx = {"previous_outputs": {"node-0": long_output}}
        result = agent._build_context_aware_prompt("Task", graph_ctx, {})
        assert "..." in result

    def test_build_context_with_variables(self):
        agent, _, _ = _make_agent()
        graph_ctx = {"variables": {"var1": "value1", "var2": "value2"}}
        result = agent._build_context_aware_prompt("Task", graph_ctx, {})
        assert "var1=value1" in result

    def test_build_context_with_node_config(self):
        agent, _, _ = _make_agent()
        result = agent._build_context_aware_prompt("Task", {}, {"timeout": 30})
        assert "timeout" in result

    def test_build_graph_context_info_empty(self):
        agent, _, _ = _make_agent()
        info = agent._build_graph_context_info({})
        assert info == []


# ===================================================================
# system_prompt + instruction -> Agent(instructions=...)
# ===================================================================


class TestCombinedInstructions:
    """The terminal hop of a per-message prompt override.

    ``POST /workers/{name}/message`` folds ``system_prompt`` / ``instruction`` into
    the worker config, ``_globals`` hands both to ``create_worker``, and they land
    here -- joined into a SINGLE ``instructions=`` string so strict chat templates
    still see exactly one system message at index 0. Baked once at construction,
    which is why the endpoint fingerprints both fields into the cache key.
    """

    def test_both_are_joined_into_one_instructions_string(self):
        _, agent_cls, _ = _make_agent(
            system_prompt="You are a data analyst.", instruction="Be terse."
        )
        assert agent_cls.call_args.kwargs["instructions"] == "You are a data analyst.\n\nBe terse."

    def test_system_prompt_alone_is_passed_through(self):
        _, agent_cls, _ = _make_agent(system_prompt="You are a data analyst.")
        assert agent_cls.call_args.kwargs["instructions"] == "You are a data analyst."

    def test_instruction_alone_is_passed_through(self):
        """The join must not depend on system_prompt being present."""
        _, agent_cls, _ = _make_agent(system_prompt=None, instruction="Be terse.")
        assert agent_cls.call_args.kwargs["instructions"] == "Be terse."

    def test_neither_yields_none_not_empty_string(self):
        """``instructions=""`` would emit an empty system message; None omits it."""
        _, agent_cls, _ = _make_agent(system_prompt=None, instruction=None)
        assert agent_cls.call_args.kwargs["instructions"] is None

    def test_system_prompt_is_not_also_passed_as_system_prompt_kwarg(self):
        """Passing ``system_prompt=`` and ``instructions=`` together produces two
        consecutive system messages, which Qwen3.6's template 400s on."""
        _, agent_cls, _ = _make_agent(system_prompt="You are a data analyst.")
        assert "system_prompt" not in agent_cls.call_args.kwargs


# ===================================================================
# run() lifecycle
# ===================================================================


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_run_success(self):
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "result text"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = True
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 10
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")

        assert result.output == "result text"
        assert result.usage_metrics is not None
        assert result.usage_metrics.prompt_tokens == 5
        mock_metrics.record_agent_start.assert_called_once()
        mock_metrics.record_agent_end.assert_called_once()
        mock_metrics.record_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_none_response_raises(self):
        agent, _, mock_agent_inst = _make_agent()
        mock_agent_inst.run = AsyncMock(return_value=None)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            with pytest.raises(ValueError, match="unexpected None response"):
                await agent.run("hello")

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        agent, _, mock_agent_inst = _make_agent()

        async def slow_run(prompt):
            raise TimeoutError("timed out")

        mock_agent_inst.run = slow_run

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(TimeoutError, match="timed out"):
                await agent.run("hello")
            mock_metrics.record_agent_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_general_exception(self):
        agent, _, mock_agent_inst = _make_agent()

        async def failing_run(prompt):
            raise RuntimeError("agent failed")

        mock_agent_inst.run = failing_run

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="agent failed"):
                await agent.run("hello")

    @pytest.mark.asyncio
    async def test_run_with_none_output_returns_empty_string(self):
        """When output is None but tokens were used, return empty string to prevent None propagation."""
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = True
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 10
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            result = await agent.run("hello")
        assert result.output == ""
        assert result.usage_metrics.completion_tokens == 10


# ===================================================================
# run_with_context
# ===================================================================


class TestRunWithContext:
    @pytest.mark.asyncio
    async def test_run_with_context_success(self):
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "context result"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_response.usage = mock_usage
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        # Mock model.usage as None to avoid legacy path
        agent.agent = mock_agent_inst
        mock_agent_inst.model = MagicMock()
        mock_agent_inst.model.usage = None

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            result = await agent.run_with_context(
                {
                    "task": "analyze data",
                    "graph_context": {"graph_id": "g1", "node_id": "n1"},
                    "node_config": {"timeout": 30},
                }
            )
        assert result.output == "context result"

    @pytest.mark.asyncio
    async def test_run_with_context_exception(self):
        agent, _, mock_agent_inst = _make_agent()

        async def failing_run(prompt):
            raise RuntimeError("context run failed")

        mock_agent_inst.run = failing_run

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="context run failed"):
                await agent.run_with_context({"task": "fail"})


# ===================================================================
# MCP error recovery
# ===================================================================


class TestMCPErrorRecovery:
    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_no_mcp_keyword(self):
        agent, _, _ = _make_agent()
        result = await agent._handle_mcp_error_recovery(
            RuntimeError("something else"), "prompt", "test"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_no_servers(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = []
        result = await agent._handle_mcp_error_recovery(
            RuntimeError("MCP failed"), "prompt", "test"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_reinit_fails(self):
        agent, _, _mock_agent_inst = _make_agent()
        agent.config.mcp_servers = [MagicMock()]

        with patch.object(agent, "_initialize_mcp_servers", new_callable=AsyncMock):
            # mcp_servers_ready stays False after init
            agent.mcp_servers_ready = False
            result = await agent._handle_mcp_error_recovery(
                RuntimeError("MCP failed"), "prompt", "test"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_success(self):
        agent, _, mock_agent_inst = _make_agent()
        agent.config.mcp_servers = [MagicMock()]

        mock_response = MagicMock()
        mock_response.output = "recovered"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_response.usage = mock_usage
        mock_agent_inst.run = AsyncMock(return_value=mock_response)
        agent.agent = mock_agent_inst
        mock_agent_inst.model = MagicMock()
        mock_agent_inst.model.usage = None

        async def mock_init():
            agent.mcp_servers_ready = True

        with patch.object(agent, "_initialize_mcp_servers", side_effect=mock_init):
            result = await agent._handle_mcp_error_recovery(
                RuntimeError("MCP connection error"), "prompt", "test"
            )
        assert result is not None
        assert result.output == "recovered"

    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_retry_timeout(self):
        agent, _, mock_agent_inst = _make_agent()
        agent.config.mcp_servers = [MagicMock()]

        async def mock_init():
            agent.mcp_servers_ready = True

        async def timeout_run(prompt):
            raise TimeoutError("retry timed out")

        mock_agent_inst.run = timeout_run
        agent.agent = mock_agent_inst

        with patch.object(agent, "_initialize_mcp_servers", side_effect=mock_init):
            with pytest.raises(TimeoutError):
                await agent._handle_mcp_error_recovery(RuntimeError("MCP error"), "prompt", "test")

    @pytest.mark.asyncio
    async def test_handle_mcp_error_recovery_retry_fails(self):
        agent, _, mock_agent_inst = _make_agent()
        agent.config.mcp_servers = [MagicMock()]

        async def mock_init():
            agent.mcp_servers_ready = True

        async def failing_run(prompt):
            raise RuntimeError("retry failed")

        mock_agent_inst.run = failing_run
        agent.agent = mock_agent_inst

        with patch.object(agent, "_initialize_mcp_servers", side_effect=mock_init):
            result = await agent._handle_mcp_error_recovery(
                RuntimeError("MCP error"), "prompt", "test"
            )
        assert result is None


# ===================================================================
# MCP server lifecycle
# ===================================================================


class TestMCPServerLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_mcp_no_servers(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = []
        await agent._initialize_mcp_servers()
        assert agent.mcp_servers_ready is True

    @pytest.mark.asyncio
    async def test_initialize_mcp_success(self):
        agent, _, _ = _make_agent()
        mock_server = AsyncMock()
        agent.config.mcp_servers = [mock_server]
        await agent._initialize_mcp_servers()
        assert agent.mcp_servers_ready is True
        assert agent._mcp_exit_stack is not None

    @pytest.mark.asyncio
    async def test_initialize_mcp_cancelled(self):
        agent, _, _ = _make_agent()
        mock_server = AsyncMock()
        agent.config.mcp_servers = [mock_server]

        with patch.object(agent, "_start_mcp_servers", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await agent._initialize_mcp_servers()
        assert agent.mcp_servers_ready is False

    @pytest.mark.asyncio
    async def test_initialize_mcp_failure(self):
        agent, _, _ = _make_agent()
        mock_server = AsyncMock()
        agent.config.mcp_servers = [mock_server]

        with patch.object(agent, "_start_mcp_servers", side_effect=RuntimeError("mcp fail")):
            await agent._initialize_mcp_servers()
        assert agent.mcp_servers_ready is False

    @pytest.mark.asyncio
    async def test_start_mcp_servers_cancelled(self):
        """Cancellation during server startup propagates."""
        agent, _, _ = _make_agent()
        agent._mcp_exit_stack = AsyncExitStack()
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        agent.config.mcp_servers = [mock_server]

        with (
            patch.object(
                agent._mcp_exit_stack, "enter_async_context", side_effect=asyncio.CancelledError
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await agent._start_mcp_servers()

    @pytest.mark.asyncio
    async def test_start_mcp_servers_partial_failure_continues(self):
        agent, _, _ = _make_agent()
        agent._mcp_exit_stack = AsyncExitStack()

        mock_server_ok = MagicMock()
        mock_server_fail = MagicMock()
        agent.config.mcp_servers = [mock_server_fail, mock_server_ok]

        call_count = 0

        async def mock_enter(ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("server 1 failed")
            return ctx

        with patch.object(agent._mcp_exit_stack, "enter_async_context", side_effect=mock_enter):
            await agent._start_mcp_servers()
        # Should have attempted both servers
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_mcp_on_error(self):
        agent, _, _ = _make_agent()
        agent._mcp_exit_stack = AsyncExitStack()
        agent.mcp_servers_ready = True
        await agent._cleanup_mcp_on_error()
        assert agent.mcp_servers_ready is False
        assert agent._mcp_exit_stack is None

    @pytest.mark.asyncio
    async def test_cleanup_mcp_on_error_cancelled(self):
        agent, _, _ = _make_agent()
        mock_stack = AsyncMock()
        mock_stack.aclose = AsyncMock(side_effect=asyncio.CancelledError)
        agent._mcp_exit_stack = mock_stack
        with pytest.raises(asyncio.CancelledError):
            await agent._cleanup_mcp_on_error()

    @pytest.mark.asyncio
    async def test_cleanup_mcp_on_error_exception(self):
        agent, _, _ = _make_agent()
        mock_stack = AsyncMock()
        mock_stack.aclose = AsyncMock(side_effect=RuntimeError("cleanup fail"))
        agent._mcp_exit_stack = mock_stack
        await agent._cleanup_mcp_on_error()
        assert agent._mcp_exit_stack is None

    @pytest.mark.asyncio
    async def test_ensure_mcp_servers_ready_already_ready(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = [MagicMock()]
        agent.mcp_servers_ready = True
        # Should be a no-op
        await agent._ensure_mcp_servers_ready()
        assert agent.mcp_servers_ready is True

    @pytest.mark.asyncio
    async def test_ensure_mcp_servers_ready_init_cancelled(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = [MagicMock()]
        agent.mcp_servers_ready = False

        with patch.object(agent, "_initialize_mcp_servers", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await agent._ensure_mcp_servers_ready()
        assert agent.mcp_servers_ready is False

    @pytest.mark.asyncio
    async def test_ensure_mcp_servers_ready_init_fails(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = [MagicMock()]
        agent.mcp_servers_ready = False

        with patch.object(agent, "_initialize_mcp_servers", side_effect=RuntimeError("init fail")):
            await agent._ensure_mcp_servers_ready()
        assert agent.mcp_servers_ready is False

    @pytest.mark.asyncio
    async def test_ensure_mcp_no_servers(self):
        agent, _, _ = _make_agent()
        agent.config.mcp_servers = []
        agent.mcp_servers_ready = False
        await agent._ensure_mcp_servers_ready()
        # no-op when no servers configured


# ===================================================================
# Health check and cleanup
# ===================================================================


class TestHealthAndCleanup:
    @pytest.mark.asyncio
    async def test_check_health_returns_true(self):
        agent, _, _ = _make_agent()
        assert await agent.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_no_agent_returns_false(self):
        agent, _, _ = _make_agent()
        agent.agent = None
        assert await agent.check_health() is False

    @pytest.mark.asyncio
    async def test_cleanup(self):
        agent, _, _ = _make_agent()
        await agent.cleanup()
        # Should not raise

    @pytest.mark.asyncio
    async def test_is_mcp_server_healthy_with_is_running(self):
        agent, _, _ = _make_agent()
        server = MagicMock()
        server.is_running = True
        assert await agent._is_mcp_server_healthy(server) is True

    @pytest.mark.asyncio
    async def test_is_mcp_server_healthy_with_command(self):
        agent, _, _ = _make_agent()
        server = MagicMock(spec=["command"])
        assert await agent._is_mcp_server_healthy(server) is True

    @pytest.mark.asyncio
    async def test_is_mcp_server_healthy_none(self):
        agent, _, _ = _make_agent()
        result = await agent._is_mcp_server_healthy(None)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_mcp_server_healthy_exception(self):
        agent, _, _ = _make_agent()
        server = MagicMock()
        # Make hasattr blow up
        type(server).is_running = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))  # noqa: ARG005
        result = await agent._is_mcp_server_healthy(server)
        assert result is False


# ===================================================================
# Tool management
# ===================================================================


class TestToolManagement:
    def test_get_tool_exists(self):
        agent, _, _ = _make_agent()
        mock_tool = MagicMock()
        agent._tool_registry["my_tool"] = mock_tool
        assert agent.get_tool("my_tool") is mock_tool

    def test_get_tool_not_exists(self):
        agent, _, _ = _make_agent()
        assert agent.get_tool("nonexistent") is None

    def test_add_tools_from_json(self):
        agent, _, _ = _make_agent()
        json_tools = [
            {
                "name": "add_numbers",
                "description": "Add two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "integer"},
                    },
                },
                "implementation": "async def impl(ctx, **kwargs):\n    return {'result': kwargs['a'] + kwargs['b']}",
            }
        ]
        agent.add_tools_from_json(json_tools)
        assert "add_numbers" in agent._tool_registry

    def test_add_tools_from_json_overwrite(self):
        agent, _, _ = _make_agent()
        tool_def = {
            "name": "my_tool",
            "description": "A tool",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            "output_schema": {"type": "object", "properties": {"y": {"type": "string"}}},
            "implementation": "async def impl(ctx, **kwargs):\n    return {'y': kwargs.get('x', '')}",
        }
        agent.add_tools_from_json([tool_def])
        assert "my_tool" in agent._tool_registry
        # Add again — should overwrite with warning
        agent.add_tools_from_json([tool_def])
        assert "my_tool" in agent._tool_registry

    def test_add_tools_from_json_invalid_raises(self):
        agent, _, _ = _make_agent()
        with pytest.raises(Exception):
            agent.add_tools_from_json([{"invalid": "data"}])

    def test_add_tools_from_json_python_code_field(self):
        """Test python_code field gets wrapped via _wrap_function_as_impl."""
        agent, _, _ = _make_agent()
        tool_def = {
            "name": "greet",
            "description": "Greet someone",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
            },
            "python_code": "def greet(name):\n    return {'greeting': f'Hello {name}'}",
        }
        agent.add_tools_from_json([tool_def])
        assert "greet" in agent._tool_registry


# ===================================================================
# Output validators
# ===================================================================


class TestOutputValidators:
    # The exec-based validator path is DELETED (request-supplied Python ran with
    # full builtins). Any non-empty code now fails closed at agent build time,
    # whatever it contains, and is never executed.
    @pytest.mark.parametrize(
        "code",
        [
            "def validate(ctx, output):\n    return output",
            "def something_else(ctx, output):\n    return output",
            "def validate(:\n    bad syntax",
        ],
    )
    def test_output_validator_code_fails_closed(self, code):
        from src.security.request_overrides import OutputValidatorCodeNotSupportedError

        with pytest.raises(OutputValidatorCodeNotSupportedError):
            _make_agent(output_validator_code=code)

    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_empty_output_validator_code_is_absent(self, code):
        _agent, _, mock_agent_inst = _make_agent(output_validator_code=code)
        mock_agent_inst.output_validator.assert_not_called()

    def test_register_default_schema_validator_object(self):
        _agent, _, mock_agent_inst = _make_agent(
            model_provider="openai",
            output_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        mock_agent_inst.output_validator.assert_called()

    def test_register_default_schema_validator_with_properties(self):
        _agent, _, mock_agent_inst = _make_agent(
            model_provider="openai",
            output_schema={
                "type": "object",
                "properties": {"items": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["items"],
            },
        )
        mock_agent_inst.output_validator.assert_called()


# ===================================================================
# Default schema validation feedback
# ===================================================================


class TestSchemaValidationFeedback:
    """Test the create_validation_feedback and validate functions.

    We use object-type schemas when going through agent creation (because StructuredDict
    requires object schemas), and test non-object schema validators by calling
    _register_default_schema_validator directly.
    """

    def _get_validator_via_agent(self, schema):
        """Extract the registered validator function by capturing it from agent creation."""
        _agent, _, mock_agent_inst = _make_agent(model_provider="openai", output_schema=schema)
        call_args = mock_agent_inst.output_validator.call_args
        validator_func = call_args[0][0]
        return validator_func

    def _get_validator_direct(self, schema):
        """Register a validator directly without StructuredDict, for non-object schemas."""
        agent, _, mock_agent_inst = _make_agent(model_provider="openai")
        # Call _register_default_schema_validator directly to bypass StructuredDict
        agent._register_default_schema_validator(schema)
        call_args = mock_agent_inst.output_validator.call_args
        validator_func = call_args[0][0]
        return validator_func

    def test_object_schema_valid(self):
        validator = self._get_validator_via_agent(
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        )
        ctx = MagicMock()
        result = validator(ctx, {"x": "hello"})
        assert result == {"x": "hello"}

    def test_object_schema_missing_field_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        )
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, {"y": "wrong"})

    def test_object_schema_not_dict_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(
            {"type": "object", "properties": {"x": {"type": "string"}}}
        )
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "not a dict")

    def test_array_schema_valid(self):
        validator = self._get_validator_direct({"type": "array", "items": {"type": "string"}})
        ctx = MagicMock()
        result = validator(ctx, ["a", "b"])
        assert result == ["a", "b"]

    def test_array_schema_not_list_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "array", "items": {"type": "string"}})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "not a list")

    def test_string_schema_valid(self):
        validator = self._get_validator_direct({"type": "string"})
        ctx = MagicMock()
        assert validator(ctx, "hello") == "hello"

    def test_string_schema_not_string_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "string"})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, 42)

    def test_string_enum_valid(self):
        validator = self._get_validator_direct({"type": "string", "enum": ["yes", "no"]})
        ctx = MagicMock()
        assert validator(ctx, "yes") == "yes"

    def test_string_enum_invalid_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "string", "enum": ["yes", "no"]})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "maybe")

    def test_number_schema_valid(self):
        validator = self._get_validator_direct({"type": "number"})
        ctx = MagicMock()
        assert validator(ctx, 3.14) == 3.14

    def test_number_schema_invalid_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "number"})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "not a number")

    def test_integer_schema_valid(self):
        validator = self._get_validator_direct({"type": "integer"})
        ctx = MagicMock()
        assert validator(ctx, 42) == 42

    def test_integer_schema_invalid_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "integer"})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, 3.14)

    def test_boolean_schema_valid(self):
        validator = self._get_validator_direct({"type": "boolean"})
        ctx = MagicMock()
        assert validator(ctx, True) is True

    def test_boolean_schema_invalid_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "boolean"})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "true")

    def test_unknown_schema_type_passes(self):
        validator = self._get_validator_direct({"type": "unknown"})
        ctx = MagicMock()
        assert validator(ctx, "anything") == "anything"

    def test_string_format_feedback(self):
        """Test feedback for string with format."""
        validator = self._get_validator_direct({"type": "string", "format": "date-time"})
        ctx = MagicMock()
        # Valid string should pass
        assert validator(ctx, "2025-01-01T00:00:00") == "2025-01-01T00:00:00"

    # -------------------------------------------------------------
    # Discriminated-union (fr-agent) schema — the regression case
    # behind 2026-05-07-plan3-r3. The previous shallow validator
    # passed `{"kind":"kaizen.log_candidate","text":"..."}` because
    # it only checked top-level type+required; with the jsonschema
    # validator it must be rejected.
    # -------------------------------------------------------------

    FR_AGENT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "final_answer"},
                    "text": {"type": "string", "minLength": 1},
                },
                "required": ["kind", "text"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "tool_calls"},
                    "calls": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "pattern": r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
                                },
                                "input": {},
                            },
                            "required": ["action", "input"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["kind", "calls"],
                "additionalProperties": False,
            },
        ],
    }

    def test_fr_agent_kaizen_as_kind_raises(self):
        """The exact malformed shape that caused wm_failed in prod."""
        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, {"kind": "kaizen.log_candidate", "text": "hi"})

    def test_fr_agent_final_answer_passes(self):
        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        out = {"kind": "final_answer", "text": "hello"}
        assert validator(ctx, out) == out

    def test_fr_agent_tool_calls_with_kaizen_action_passes(self):
        """kaizen.log_candidate is valid as an action, just not as a kind."""
        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        out = {
            "kind": "tool_calls",
            "calls": [{"action": "kaizen.log_candidate", "input": {"intent": "x"}}],
        }
        assert validator(ctx, out) == out

    def test_fr_agent_empty_calls_warns_not_fatal(self, caplog):
        """Empty ``calls`` violates ``minItems: 1`` — a grammar-stripped bound.

        The decoder grammar never carried ``minItems`` (xgrammar can't compile
        it), so the model was never told to emit >=1 call. Enforcing it here used
        to fatally 500 the run (inference_router, retries=0) with no way for the
        model to recover. It is now a non-fatal advisory: the structurally-valid
        ``tool_calls`` shape passes, and a WARNING records the stripped-bound
        violation. "at least one call" is the consumer's responsibility on
        receipt (an empty tool_calls list is a normal "no action" state).
        """
        import logging

        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        out = {"kind": "tool_calls", "calls": []}
        with caplog.at_level(logging.WARNING):
            assert validator(ctx, out) == out
        stripped = [r.message for r in caplog.records if "grammar-stripped constraint" in r.message]
        assert len(stripped) == 1
        # The oneOf warning must drill into the intended branch and name the REAL
        # constraint (calls too short), not a sibling branch's discriminator miss.
        assert "calls" in stripped[0]
        assert "text" not in stripped[0]

    def test_fr_agent_final_answer_missing_text_raises(self):
        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, {"kind": "final_answer"})

    def test_fr_agent_extra_field_raises(self):
        """additionalProperties: false must be enforced."""
        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(self.FR_AGENT_SCHEMA)
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, {"kind": "final_answer", "text": "hi", "extra": 1})

    # -------------------------------------------------------------
    # Grammar-stripped bound keywords must be ADVISORY, not fatal.
    # xgrammar can't compile minimum/maximum/minItems/pattern/…, so
    # the decoder is never given them; a reply that violates one must
    # warn-and-pass, never raise (which on inference_router/retries=0
    # was a fatal 500 that took down 100% of prod demo_video).
    # -------------------------------------------------------------

    def test_stripped_numeric_bound_violation_warns_not_fatal(self, caplog):
        import logging

        validator = self._get_validator_via_agent(
            {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        )
        ctx = MagicMock()
        out = {"score": 7}  # exceeds maximum:5 — the exact prod-killing shape
        with caplog.at_level(logging.WARNING):
            assert validator(ctx, out) == out
        warnings = [r.message for r in caplog.records if "grammar-stripped constraint" in r.message]
        assert len(warnings) == 1
        assert "score" in warnings[0]

    def test_stripped_minitems_violation_warns_not_fatal(self, caplog):
        import logging

        validator = self._get_validator_via_agent(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                },
                "required": ["tags"],
            }
        )
        ctx = MagicMock()
        out = {"tags": []}
        with caplog.at_level(logging.WARNING):
            assert validator(ctx, out) == out
        assert any("grammar-stripped constraint" in r.message for r in caplog.records)

    def test_stripped_bound_satisfied_emits_no_warning(self, caplog):
        """An in-range value must be clean — no false-positive dead-signal noise."""
        import logging

        validator = self._get_validator_via_agent(
            {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        )
        ctx = MagicMock()
        with caplog.at_level(logging.WARNING):
            assert validator(ctx, {"score": 3}) == {"score": 3}
        assert not any("grammar-stripped constraint" in r.message for r in caplog.records)

    def test_structural_violation_still_raises_and_no_advisory(self, caplog):
        """Missing a required field is structural (survives the strip) → fatal, no advisory."""
        import logging

        from pydantic_ai import ModelRetry

        validator = self._get_validator_via_agent(
            {
                "type": "object",
                "properties": {"score": {"type": "integer", "maximum": 5}},
                "required": ["score"],
            }
        )
        ctx = MagicMock()
        with caplog.at_level(logging.WARNING), pytest.raises(ModelRetry):
            validator(ctx, {})  # 'score' missing
        # Short-circuit: the advisory pass must never run when enforced fails.
        assert not any("grammar-stripped constraint" in r.message for r in caplog.records)

    def test_enum_violation_still_raises(self):
        """enum survives the strip, so it stays fatally enforced."""
        from pydantic_ai import ModelRetry

        validator = self._get_validator_direct({"type": "string", "enum": ["yes", "no"]})
        ctx = MagicMock()
        with pytest.raises(ModelRetry):
            validator(ctx, "maybe")


# ===================================================================
# run_agent_with_timeout
# ===================================================================


class TestRunAgentWithTimeout:
    @pytest.mark.asyncio
    async def test_timeout_with_mcp_servers(self):
        agent, _, mock_agent_inst = _make_agent()
        agent.config.mcp_servers = [MagicMock()]
        mock_agent_inst.run = AsyncMock(return_value="ok")
        result = await agent._run_agent_with_timeout("test", "agent_name")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_timeout_without_mcp_servers(self):
        agent, _, mock_agent_inst = _make_agent()
        agent.config.mcp_servers = []
        mock_agent_inst.run = AsyncMock(return_value="ok")
        result = await agent._run_agent_with_timeout("test", "agent_name")
        assert result == "ok"


# ===================================================================
# MLX wrappers
# ===================================================================


class TestMLXWrappers:
    @pytest.mark.asyncio
    async def test_mlx_completions_wrapper_strips_think_tags(self):
        from src.worker.agents.pydantic_ai_agent import MLXCompletionsWrapper

        mock_completions = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "<think>thinking here</think>\nHello world"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completions.create.return_value = mock_response

        wrapper = MLXCompletionsWrapper(mock_completions)
        result = await wrapper.create(model="test")
        assert "<think>" not in result.choices[0].message.content
        assert "Hello world" in result.choices[0].message.content

    @pytest.mark.asyncio
    async def test_mlx_completions_wrapper_no_think_tags(self):
        from src.worker.agents.pydantic_ai_agent import MLXCompletionsWrapper

        mock_completions = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "No thinking here"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completions.create.return_value = mock_response

        wrapper = MLXCompletionsWrapper(mock_completions)
        result = await wrapper.create(model="test")
        assert result.choices[0].message.content == "No thinking here"

    @pytest.mark.asyncio
    async def test_mlx_completions_wrapper_no_choices(self):
        from src.worker.agents.pydantic_ai_agent import MLXCompletionsWrapper

        mock_completions = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_completions.create.return_value = mock_response

        wrapper = MLXCompletionsWrapper(mock_completions)
        result = await wrapper.create(model="test")
        assert result.choices == []

    @pytest.mark.asyncio
    async def test_mlx_completions_wrapper_no_content(self):
        from src.worker.agents.pydantic_ai_agent import MLXCompletionsWrapper

        mock_completions = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completions.create.return_value = mock_response

        wrapper = MLXCompletionsWrapper(mock_completions)
        result = await wrapper.create(model="test")
        assert result.choices[0].message.content is None

    def test_mlx_chat_wrapper(self):
        from src.worker.agents.pydantic_ai_agent import MLXChatWrapper, MLXCompletionsWrapper

        mock_chat = MagicMock()
        wrapper = MLXChatWrapper(mock_chat)
        assert isinstance(wrapper.completions, MLXCompletionsWrapper)

    def test_mlx_async_openai(self):
        from src.worker.agents.pydantic_ai_agent import MLXAsyncOpenAI, MLXChatWrapper

        with patch("src.worker.agents.pydantic_ai_agent.AsyncOpenAI.__init__", return_value=None):
            client = MLXAsyncOpenAI.__new__(MLXAsyncOpenAI)
            mock_chat = MagicMock()
            mock_chat.completions = MagicMock()
            client.chat = mock_chat
            # Re-wrap
            client.chat = MLXChatWrapper(client.chat)
            assert isinstance(client.chat, MLXChatWrapper)

    def test_mlx_openai_provider(self):
        from src.worker.agents.pydantic_ai_agent import MLXOpenAIProvider

        with (
            patch("src.worker.agents.pydantic_ai_agent.OpenAIProvider.__init__", return_value=None),
            patch("src.worker.agents.pydantic_ai_agent.MLXAsyncOpenAI") as mock_mlx_client,
        ):
            provider = MLXOpenAIProvider.__new__(MLXOpenAIProvider)
            provider.__init__(base_url="http://test:8080", api_key="mlx-local")
            # max_retries=0: mlx is self-hosted, and the OpenAI SDK retries read
            # timeouts (_base_client.py:1653), so its default of 2 would turn one
            # timed-out decode into three on hardware we own. See
            # tests/test_ir_client_transport_hardening.py for the assertion against
            # the real client object rather than this mock.
            mock_mlx_client.assert_called_once_with(
                base_url="http://test:8080", api_key="mlx-local", max_retries=0, http_client=None
            )

    def test_inference_router_provider(self):
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        with (
            patch("src.worker.agents.pydantic_ai_agent.OpenAIProvider.__init__", return_value=None),
            patch("src.worker.agents.pydantic_ai_agent.AsyncOpenAI") as mock_openai,
        ):
            provider = InferenceRouterOpenAIProvider.__new__(InferenceRouterOpenAIProvider)
            provider.__init__(base_url="http://ir:8150/v1", api_key="internal")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            # X-Schema-Id is no longer injected — inference-router self-derives
            # schema_id from the request's response_format.json_schema fingerprint.
            assert "X-Schema-Id" not in call_kwargs["default_headers"]
            # Identity headers are present ONLY when they carry a value.
            # Constructed with no tenant, X-Tenant-Id must be ABSENT: the old
            # unconditional form sent it as "" , which IR reads as the nil
            # sentinel rather than as "unknown". See
            # tests/test_outbound_tenant_is_real.py for the full contract.
            assert "X-Tenant-Id" not in call_kwargs["default_headers"]


# ===================================================================
# Factory function
# ===================================================================


class TestCreatePydanticAIAgent:
    def test_create_agent_basic(self):
        with (
            patch(f"{MODULE}.MLXOpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

            agent = create_pydantic_ai_agent(
                name="factory_agent",
                description="Test factory agent",
                model_provider="mlx",
                model="test-model",
                base_url="http://test:8080",
            )
            assert agent.name == "factory_agent"

    def test_create_agent_with_mcp_servers(self):
        with (
            patch(f"{MODULE}.MLXOpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

            mcp = MagicMock()
            agent = create_pydantic_ai_agent(
                name="mcp_agent",
                description="Agent with MCP",
                model_provider="mlx",
                model="test-model",
                base_url="http://test:8080",
                mcp_servers_list=[mcp],
            )
            assert len(agent.config.mcp_servers) == 1

    def test_create_agent_all_params(self):
        with (
            patch(f"{MODULE}.MLXOpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

            agent = create_pydantic_ai_agent(
                name="full_agent",
                description="Full config",
                model_provider="mlx",
                model="test-model",
                temperature=0.5,
                db_path="/tmp/db",
                system_prompt="Be concise.",
                instruction="Follow rules.",
                user_prompt="Start here",
                tools=None,
                deps=["dep1"],
                base_url="http://test:8080",
                api_key="key123",
                tools_dir="/tools",
                output_schema=None,
                output_validator_code=None,
            )
            assert agent.config.temperature == 0.5
            assert agent.config.instruction == "Follow rules."
            assert agent.config.deps == ["dep1"]


# ===================================================================
# Agent registration
# ===================================================================


class TestAgentRegistration:
    def test_register_pydantic_ai_agent(self):
        with patch(f"{MODULE}.register_agent", create=True):
            import src.worker.agents.pydantic_ai_agent as mod

            # Check the function exists and doesn't crash
            mod._register_pydantic_ai_agent()

    def test_register_pydantic_ai_agent_failure(self):
        """Registration failure should not raise."""
        from src.worker.agents.pydantic_ai_agent import _register_pydantic_ai_agent

        with patch(
            "src.worker.agents.pydantic_ai_agent.register_agent",
            side_effect=ImportError("no registry"),
            create=True,
        ):
            # Should not raise
            _register_pydantic_ai_agent()


# ===================================================================
# _get_api_key
# ===================================================================


class TestGetApiKey:
    """Key-lookup semantics only.

    ⚠️ ``get_api_key`` is now also the second local-only barrier, and it is only
    ever reached for an EXTERNAL provider in production — so these cases keep
    using ``openai`` (the realistic caller) and take the opt-in explicitly. The
    refusal itself is asserted in test_local_only_llm_guard.py; weakening these
    configs to a local provider would have moved them off the real path.
    """

    def test_from_config(self):
        agent, _, _ = _make_agent(api_key="config-key")
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        config = PydanticAIAgentConfig(
            name="test",
            model_provider="openai",
            model="gpt-4o",
            db_path="",
            api_key="config-key",
        )
        with _external_llm_allowed():
            result = agent._get_api_key(config, "OPENAI_API_KEY")
        assert result == "config-key"

    def test_from_env(self):
        agent, _, _ = _make_agent()
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        config = PydanticAIAgentConfig(
            name="test", model_provider="openai", model="gpt-4o", db_path=""
        )
        with _external_llm_allowed(), patch.dict(os.environ, {"MY_KEY": "env-key"}):
            result = agent._get_api_key(config, "MY_KEY")
        assert result == "env-key"

    def test_returns_none_when_missing(self):
        agent, _, _ = _make_agent()
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        config = PydanticAIAgentConfig(
            name="test", model_provider="openai", model="gpt-4o", db_path=""
        )
        os.environ.pop("NONEXISTENT_KEY", None)
        with _external_llm_allowed():
            result = agent._get_api_key(config, "NONEXISTENT_KEY")
        assert result is None


# ===================================================================
# ToolTemplate
# ===================================================================

TOOL_TEMPLATE_MODULE = "src.worker.agents.tool_template"


class TestToolTemplateFromJson:
    def test_basic_from_json(self):
        from src.worker.agents.tool_template import ToolTemplate

        data = {
            "name": "add",
            "description": "Add numbers",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "integer"}},
            },
            "implementation": "async def impl(ctx, **kwargs):\n    return {'result': kwargs['a'] + kwargs['b']}",
        }
        template = ToolTemplate.from_json(data)
        assert template.name == "add"
        assert template.description == "Add numbers"

    def test_from_json_python_code_field(self):
        from src.worker.agents.tool_template import ToolTemplate

        data = {
            "name": "greet",
            "description": "Greet user",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
            },
            "python_code": "def greet(name):\n    return {'greeting': f'Hello {name}'}",
        }
        template = ToolTemplate.from_json(data)
        assert template.name == "greet"

    def test_create_tool(self):
        from src.worker.agents.tool_template import ToolTemplate

        data = {
            "name": "echo",
            "description": "Echo back",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {"echo": {"type": "string"}},
            },
            "implementation": "async def impl(ctx, **kwargs):\n    return {'echo': kwargs.get('text', '')}",
        }
        template = ToolTemplate.from_json(data)
        tool = template.create_tool()
        assert tool.name == "echo"

    def test_create_tool_no_implementation_raises(self):
        from src.worker.agents.tool_template import ToolTemplate

        # Create a ToolTemplate with implementation, then set it to None post-init
        template = ToolTemplate(
            name="broken",
            description="Broken tool",
            input_schema=type("M", (BaseModel,), {}),
            output_schema=type("M", (BaseModel,), {}),
            implementation=lambda ctx, **kwargs: None,  # noqa: ARG005
        )
        # Force implementation to None to test the guard
        object.__setattr__(template, "implementation", None)
        with pytest.raises(ValueError, match="no implementation"):
            template.create_tool()


class TestToolTemplateFieldTypes:
    def test_get_field_type_string(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="string")
        assert ToolTemplate._get_field_type(field, "Test") is str

    def test_get_field_type_integer(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="integer")
        assert ToolTemplate._get_field_type(field, "Test") is int

    def test_get_field_type_number(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="number")
        assert ToolTemplate._get_field_type(field, "Test") is float

    def test_get_field_type_boolean(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="boolean")
        assert ToolTemplate._get_field_type(field, "Test") is bool

    def test_get_field_type_array_with_items(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="array", items={"type": "string"})
        result = ToolTemplate._get_field_type(field, "Test")
        assert result == list[Any]

    def test_get_field_type_array_no_items(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="array")
        result = ToolTemplate._get_field_type(field, "Test")
        assert result == list[Any]

    def test_get_field_type_object_with_properties(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(
            type="object",
            properties={"name": {"type": "string"}},
            required=["name"],
        )
        result = ToolTemplate._get_field_type(field, "Test")
        assert issubclass(result, BaseModel)

    def test_get_field_type_object_no_properties(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="object")
        result = ToolTemplate._get_field_type(field, "Test")
        assert result == dict[str, Any]

    def test_get_field_type_unknown(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        field = SchemaField(type="custom_type")
        result = ToolTemplate._get_field_type(field, "Test")
        assert result is Any


class TestToolTemplateSecurityValidation:
    def test_validate_safe_code(self):
        from src.worker.agents.tool_template import ToolTemplate

        code = "async def impl(ctx, **kwargs):\n    return {'result': len(kwargs)}"
        ToolTemplate._validate_code_security(code)  # Should not raise

    def test_validate_dangerous_import_os(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous import"):
            ToolTemplate._validate_code_security("import os\nasync def impl(ctx): pass")

    def test_validate_dangerous_import_subprocess(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous import"):
            ToolTemplate._validate_code_security("import subprocess\nasync def impl(ctx): pass")

    def test_validate_dangerous_call_eval(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous function call"):
            ToolTemplate._validate_code_security("x = eval('1+1')\nasync def impl(ctx): pass")

    def test_validate_dangerous_call_exec(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous function call"):
            ToolTemplate._validate_code_security("exec('print(1)')\nasync def impl(ctx): pass")

    def test_validate_dangerous_attribute(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous attribute"):
            ToolTemplate._validate_code_security("x = obj.__globals__\nasync def impl(ctx): pass")

    def test_validate_syntax_error(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Invalid Python syntax"):
            ToolTemplate._validate_code_security("def invalid(:\n    pass")

    def test_validate_import_from_dangerous(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="Dangerous import"):
            ToolTemplate._validate_code_security(
                "from importlib import import_module\nasync def impl(ctx): pass"
            )


class TestToolTemplateSandbox:
    def test_create_sandboxed_namespace(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        assert "str" in ns
        assert "int" in ns
        assert "len" in ns
        assert "math" in ns
        assert "json" in ns
        assert "re" in ns
        assert "datetime" in ns
        assert "RunContext" in ns

    def test_execute_in_sandbox_success(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        code = "async def impl(ctx, **kwargs):\n    return {'result': 42}"
        impl = ToolTemplate._execute_in_sandbox(code, ns)
        assert callable(impl)

    def test_execute_in_sandbox_no_impl_raises(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        code = "x = 1"
        with pytest.raises(ValueError, match="must define an 'impl' function"):
            ToolTemplate._execute_in_sandbox(code, ns)

    def test_execute_in_sandbox_impl_not_callable_raises(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        code = "impl = 42"
        with pytest.raises(ValueError, match="must be callable"):
            ToolTemplate._execute_in_sandbox(code, ns)


class TestToolTemplateFailureTracking:
    def setup_method(self):
        """Reset global state before each test."""
        import src.worker.agents.tool_template as tt

        with tt._failure_lock:
            tt._failure_counts.clear()
        with tt._stats_lock:
            tt._execution_stats.clear()

    def test_record_tool_failure(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_tool_failure("test_tool")
        assert not ToolTemplate._is_tool_failed("test_tool")

    def test_tool_exceeds_failure_threshold(self):
        from src.worker.agents.tool_template import MAX_FAILURES_PER_TOOL, ToolTemplate

        for _ in range(MAX_FAILURES_PER_TOOL):
            ToolTemplate._record_tool_failure("bad_tool")
        assert ToolTemplate._is_tool_failed("bad_tool") is True

    def test_record_execution_stats_success(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_execution_stats("my_tool", 1.5, True)
        stats = ToolTemplate.get_tool_stats("my_tool")
        assert stats is not None
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 0
        assert stats["total_time"] == 1.5

    def test_record_execution_stats_failure(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_execution_stats("my_tool", 0.5, False)
        stats = ToolTemplate.get_tool_stats("my_tool")
        assert stats is not None
        assert stats["failed_executions"] == 1

    def test_record_execution_stats_multiple(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_execution_stats("my_tool", 1.0, True)
        ToolTemplate._record_execution_stats("my_tool", 3.0, True)
        stats = ToolTemplate.get_tool_stats("my_tool")
        assert stats is not None
        assert stats["total_executions"] == 2
        assert stats["min_time"] == 1.0
        assert stats["max_time"] == 3.0
        assert stats["avg_time"] == 2.0

    def test_get_tool_stats_nonexistent(self):
        from src.worker.agents.tool_template import ToolTemplate

        assert ToolTemplate.get_tool_stats("nonexistent") is None

    def test_get_all_stats(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_execution_stats("tool_a", 1.0, True)
        ToolTemplate._record_execution_stats("tool_b", 2.0, False)
        all_stats = ToolTemplate.get_all_stats()
        assert "tool_a" in all_stats
        assert "tool_b" in all_stats

    def test_reset_failure_counts_on_timeout(self):
        import src.worker.agents.tool_template as tt
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_tool_failure("tool_x")
        # Simulate time passing
        tt._last_failure_reset = time.time() - tt.FAILURE_RESET_TIME - 10
        ToolTemplate._reset_failure_counts_if_needed()
        # Failure counts should have been cleared
        assert not ToolTemplate._is_tool_failed("tool_x")


class TestToolTemplateCaching:
    def setup_method(self):
        import src.worker.agents.tool_template as tt

        with tt._cache_lock:
            tt._compilation_cache.clear()

    def test_cache_key_generation(self):
        from src.worker.agents.tool_template import ToolTemplate

        key1 = ToolTemplate._get_cache_key("code1", "Input", "Output")
        key2 = ToolTemplate._get_cache_key("code2", "Input", "Output")
        assert key1 != key2
        assert len(key1) == 16

    def test_cache_hit(self):
        from src.worker.agents.tool_template import ToolTemplate

        code = "async def impl(ctx, **kwargs):\n    return {'result': 1}"
        input_model = type("TestInput", (BaseModel,), {})
        output_model = type("TestOutput", (BaseModel,), {})

        # First call — cache miss
        impl1 = ToolTemplate._create_implementation(code, input_model, output_model)
        # Second call — cache hit
        impl2 = ToolTemplate._create_implementation(code, input_model, output_model)
        assert callable(impl1)
        assert callable(impl2)

    def test_evict_cache_if_needed(self):
        import src.worker.agents.tool_template as tt
        from src.worker.agents.tool_template import ToolTemplate

        with tt._cache_lock:
            for i in range(tt.MAX_CACHE_SIZE + 10):
                tt._compilation_cache[f"key_{i}"] = "value"
        ToolTemplate._evict_cache_if_needed()
        assert len(tt._compilation_cache) < tt.MAX_CACHE_SIZE + 10


class TestToolTemplateExecution:
    def setup_method(self):
        import src.worker.agents.tool_template as tt

        with tt._failure_lock:
            tt._failure_counts.clear()
        with tt._stats_lock:
            tt._execution_stats.clear()

    @pytest.mark.asyncio
    async def test_execute_implementation_async(self):
        from src.worker.agents.tool_template import ToolTemplate

        input_model = type("In", (BaseModel,), {"__annotations__": {"x": int}})
        output_model = type("Out", (BaseModel,), {"__annotations__": {"result": int}})

        async def impl(ctx, **kwargs):
            return {"result": kwargs["x"] * 2}

        ctx = MagicMock()
        ctx.tool_name = "test_tool"
        result = await ToolTemplate._execute_implementation(
            impl, ctx, {"x": 5}, input_model, output_model
        )
        assert result == {"result": 10}

    @pytest.mark.asyncio
    async def test_execute_implementation_sync(self):
        from src.worker.agents.tool_template import ToolTemplate

        input_model = type("In", (BaseModel,), {"__annotations__": {"x": int}})
        output_model = type("Out", (BaseModel,), {"__annotations__": {"result": int}})

        def impl(ctx, **kwargs):
            return {"result": kwargs["x"] * 3}

        ctx = MagicMock()
        ctx.tool_name = "sync_tool"
        result = await ToolTemplate._execute_implementation(
            impl, ctx, {"x": 4}, input_model, output_model
        )
        assert result == {"result": 12}

    @pytest.mark.asyncio
    async def test_execute_implementation_tool_disabled(self):
        from src.worker.agents.tool_template import MAX_FAILURES_PER_TOOL, ToolTemplate

        input_model = type("In", (BaseModel,), {"__annotations__": {"x": int}})
        output_model = type("Out", (BaseModel,), {"__annotations__": {"result": int}})

        # Exceed failure threshold
        for _ in range(MAX_FAILURES_PER_TOOL):
            ToolTemplate._record_tool_failure("disabled_tool")

        async def impl(ctx, **kwargs):
            return {"result": 0}

        ctx = MagicMock()
        ctx.tool_name = "disabled_tool"
        with pytest.raises(ValueError, match="execution failed"):
            await ToolTemplate._execute_implementation(
                impl, ctx, {"x": 1}, input_model, output_model
            )

    @pytest.mark.asyncio
    async def test_execute_implementation_validation_error(self):
        from src.worker.agents.tool_template import ToolTemplate

        input_model = type("In", (BaseModel,), {"__annotations__": {"x": int}})
        output_model = type("Out", (BaseModel,), {"__annotations__": {"result": int}})

        async def impl(ctx, **kwargs):
            return {"result": "not_an_int"}  # Wrong type

        ctx = MagicMock()
        ctx.tool_name = "val_err_tool"
        with pytest.raises(ValueError, match="execution failed"):
            await ToolTemplate._execute_implementation(
                impl, ctx, {"x": 1}, input_model, output_model
            )


class TestToolTemplateWrapFunction:
    def test_wrap_function_as_impl(self):
        from src.worker.agents.tool_template import ToolTemplate

        code = "def add(a, b):\n    return a + b"
        wrapped = ToolTemplate._wrap_function_as_impl(code)
        assert "async def impl" in wrapped
        assert "add(" in wrapped

    def test_wrap_function_no_func_raises(self):
        from src.worker.agents.tool_template import ToolTemplate

        with pytest.raises(ValueError, match="No function definition"):
            ToolTemplate._wrap_function_as_impl("x = 1")


class TestToolTemplateCreatePydanticModel:
    def test_create_model_with_required_and_optional(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        props = {
            "name": SchemaField(type="string"),
            "age": SchemaField(type="integer"),
        }
        model = ToolTemplate._create_pydantic_model("TestModel", props, ["name"])
        # Dynamically-built pydantic model; ty sees BaseModel only.
        instance = cast("Any", model(name="Alice"))
        assert instance.name == "Alice"
        assert instance.age is None

    def test_create_model_all_optional(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        props = {"x": SchemaField(type="string")}
        model = ToolTemplate._create_pydantic_model("OptModel", props)
        instance = cast("Any", model())
        assert instance.x is None


class TestCleanupClosesHttpClient:
    @pytest.mark.asyncio
    async def test_cleanup_closes_model_http_client(self):
        """cleanup() must close the model's AsyncOpenAI (httpx connection pool) so an
        evicted worker does not leak a pool on the long-lived gateway. Built with a
        REAL openai model so the assertion is against the real client's pool state,
        not a mock that could hide the wrong attribute path."""
        from src.worker.agents.pydantic_ai_agent import (
            PydanticAIAgent,
            PydanticAIAgentConfig,
        )

        with (
            _external_llm_allowed(),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            config = PydanticAIAgentConfig(
                name="cleanup_test",
                model_provider="openai",
                model="gpt-4o",
                db_path="",
                base_url="https://api.openai.com/v1",
                system_prompt="You are a helpful assistant.",
                mcp_servers=[],
                tools=[],
            )
            agent = PydanticAIAgent(config=config)

        client = agent.agent.model.client  # real AsyncOpenAI
        assert client.is_closed() is False  # pool open before cleanup
        await agent.cleanup()
        assert client.is_closed() is True  # cleanup released the connection pool


# ---------------------------------------------------------------------------
# Prompt logging must not shred the container log
# ---------------------------------------------------------------------------


class TestPromptLoggingIsBounded:
    """A logged prompt must stay ONE bounded log record.

    Measured on the live pod 2026-08-13: 82.2% of the container log's bytes
    (8.63 MB of 10.49 MB, over one 9-hour rotation) were unprefixed continuation
    lines from a single INFO that interpolated the whole prompt. Python's logging
    prefixes only the first line of a multi-line message, so a 30 KB prompt lands
    as one prefixed line plus thousands of anonymous ones. That forced a 10 Mi
    rotation every ~9 h and cost two sessions their log history.

    These assert on the RENDERED record, not on the call. Asserting that
    ``logger.info`` was called would pass either way -- the call was never the
    problem, its rendered size was.
    """

    def test_collapses_newlines_into_a_single_record(self):
        """Truncation alone is not enough; embedded newlines are the actual defect."""
        from src.worker.agents.pydantic_ai_agent import _summarize_for_log

        prompt = "Procedure:\n" + "\n".join(f'{{"id": "p{i}"}}' for i in range(200))
        out = _summarize_for_log(prompt)
        assert "\n" not in out, "a multi-line summary still emits unprefixed lines"

    def test_truncates_and_reports_the_true_size(self):
        """The reader must be able to tell the prompt was cut, and by how much."""
        from src.worker.agents.pydantic_ai_agent import (
            _MAX_LOGGED_PROMPT_CHARS,
            _summarize_for_log,
        )

        prompt = "x" * (_MAX_LOGGED_PROMPT_CHARS * 10)
        out = _summarize_for_log(prompt)
        assert len(out) < len(prompt) / 5, "the dominant term must actually be cut"
        assert str(len(prompt)) in out, "the true length must survive truncation"

    def test_short_prompts_are_left_intact(self):
        """The bound must not cost readability on the ordinary case."""
        from src.worker.agents.pydantic_ai_agent import _summarize_for_log

        assert _summarize_for_log("summarise this document") == "summarise this document"

    def test_rendered_record_is_one_line_and_bounded(self):
        """End to end through a real Formatter, the shape that actually reaches stdout."""
        import io
        import logging as _logging

        from src.worker.agents.pydantic_ai_agent import _summarize_for_log

        stream = io.StringIO()
        handler = _logging.StreamHandler(stream)
        handler.setFormatter(
            _logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        test_logger = _logging.getLogger("prompt_logging_shape_test")
        test_logger.addHandler(handler)
        test_logger.setLevel(_logging.INFO)
        try:
            prompt = "Principles (100):\n" + "\n".join(f'{{"id": "p{i}"}}' for i in range(5000))
            test_logger.info(
                "Running agent %s (ID: %s) with prompt: %s",
                "principle_consistency",
                "abc",
                _summarize_for_log(prompt),
            )
        finally:
            test_logger.removeHandler(handler)

        rendered = stream.getvalue()
        assert rendered.count("\n") == 1, (
            f"one log call must emit one line; emitted {rendered.count(chr(10))}"
        )
        assert len(rendered) < 4000, f"record is {len(rendered)} bytes -- still log-flooding"
        assert str(len(prompt)) in rendered, "the truncation note must report the true size"

    def test_accepts_a_multimodal_list_prompt(self):
        """`run()` takes `str | list[Any]`, and the list form is the LONGER one.

        A signature of `str` type-checked clean locally under `ty check .` but
        failed CI's `ty check src` — the call site passes the union. Excluding
        the list branch would have skipped bounding the very shape that renders
        longest.
        """
        from src.worker.agents.pydantic_ai_agent import _summarize_for_log

        parts = [{"type": "text", "text": "x" * 500} for _ in range(50)]
        out = _summarize_for_log(parts)
        assert isinstance(out, str)
        assert "\n" not in out
        assert len(out) < 4000, "a list prompt must be bounded too"


class TestLogFormatIsHonoured:
    """LOG_FORMAT must actually change the output, not just exist.

    The deployment set `LOG_FORMAT=json` while NOTHING in src/ read it, so this
    service emitted plain text while w2v_orchestrator and requirements both
    emitted JSON (verified on the live pods 2026-08-13). A knob that silently
    does nothing is the same defect class as a metric exported under a name
    nothing can query -- assert on the RENDERED record, not on the setting.
    """

    def test_json_formatter_emits_one_parseable_object(self):
        import json as _json
        import logging as _logging

        from src.api.app_config import JsonLogFormatter

        rec = _logging.LogRecord(
            name="src.some.module",
            level=_logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="prompt was %s",
            args=("truncated",),
            exc_info=None,
        )
        line = JsonLogFormatter().format(rec)
        assert "\n" not in line, "a log record must stay one line"
        parsed = _json.loads(line)
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "src.some.module"
        assert parsed["message"] == "prompt was truncated"
        assert parsed["timestamp"].endswith("+00:00")

    def test_exception_does_not_become_unprefixed_lines(self):
        """A traceback must ride INSIDE the JSON, not trail after it."""
        import json as _json
        import logging as _logging

        from src.api.app_config import JsonLogFormatter

        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            rec = _logging.LogRecord(
                name="x",
                level=_logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        line = JsonLogFormatter().format(rec)
        assert "\n" not in line
        assert "ValueError: boom" in _json.loads(line)["exc_info"]


class TestResponsePayloadIsBounded:
    """The RESPONSE side needs the same bound as the prompt side.

    After #175 bounded the prompt, `Response from agent (object): {...}` became
    the largest remaining single term in the log (8.2% of bytes, avg 794 B,
    longest line 4,890 chars — measured on the live pod 2026-08-13). Bounding
    only one side just moves the flood to the other, which is why both call
    sites now share src/utils/log_bounds.py.
    """

    def test_dict_response_is_collapsed_and_bounded(self):
        from src.utils.log_bounds import MAX_LOGGED_PAYLOAD_CHARS, summarize_for_log

        payload = {
            "conflicts": [{"principle_id": f"p{i}", "reasoning": "x" * 200} for i in range(60)]
        }
        out = summarize_for_log(payload)
        assert "\n" not in out
        assert len(out) < MAX_LOGGED_PAYLOAD_CHARS + 100
        assert str(len(str(payload))) in out, "the true size must survive truncation"

    def test_small_response_is_left_readable(self):
        from src.utils.log_bounds import summarize_for_log

        assert summarize_for_log({"ok": True}) == "{'ok': True}"

    def test_both_call_sites_share_one_bound(self):
        """Guard against the two bounds drifting apart again."""
        from src.utils.log_bounds import MAX_LOGGED_PAYLOAD_CHARS
        from src.worker.agents.pydantic_ai_agent import _MAX_LOGGED_PROMPT_CHARS

        assert _MAX_LOGGED_PROMPT_CHARS == MAX_LOGGED_PAYLOAD_CHARS


class TestJsonHandlerIsInstalledByTheRealEntrypoint:
    """The formatter must be installed by the module the CONTAINER starts.

    #176 wired LOG_FORMAT into `api/app_config.configure_logging()`. The
    Dockerfile runs `uvicorn src.api.worker_management_api:app`, and nothing on
    that path imports app_config — so the deployed pod kept emitting plain text
    while `settings.log_format == "json"`. Verified on the pod: the setting was
    read correctly; the handler was installed in a module that never loads.

    A test asserting `JsonLogFormatter().format(...)` returns JSON passed the
    whole time, because the formatter was always fine. What was broken was WHERE
    it got installed — so this test reloads the real entrypoint with the env var
    set and asserts on the ROOT LOGGER's handlers.
    """

    def test_log_format_json_installs_the_handler_on_the_root_logger(self, monkeypatch):
        import importlib
        import logging as _logging

        from src.utils.log_format import JsonLogFormatter

        root = _logging.getLogger()
        saved = list(root.handlers)
        try:
            monkeypatch.setenv("LOG_FORMAT", "json")
            module = importlib.import_module("src.api.worker_management_api")
            importlib.reload(module)
            assert any(isinstance(h.formatter, JsonLogFormatter) for h in root.handlers), (
                "the entrypoint did not install the JSON formatter; handlers="
                f"{[type(h.formatter).__name__ for h in root.handlers]}"
            )
            assert len(root.handlers) == 1, (
                "the JSON handler must REPLACE basicConfig's text handler, not join "
                f"it — {len(root.handlers)} handlers means every record is emitted twice"
            )
        finally:
            root.handlers = saved

    def test_default_stays_plain_text(self, monkeypatch):
        import importlib
        import logging as _logging

        from src.utils.log_format import JsonLogFormatter

        root = _logging.getLogger()
        saved = list(root.handlers)
        try:
            monkeypatch.setenv("LOG_FORMAT", "text")
            module = importlib.import_module("src.api.worker_management_api")
            importlib.reload(module)
            assert not any(isinstance(h.formatter, JsonLogFormatter) for h in root.handlers)
        finally:
            root.handlers = saved


# ===================================================================
# The vllm client's identity header, and the lost-answer status.
#
# Both defects were found by measuring inference-router's own logs over
# 2026-08-24T15:30Z -> 08-25T06:00Z, not by reading code:
#   * `worker_name` was NULL on all 95 `upstream_empty_content` rows and
#     POPULATED on every `upstream_empty_result` row of the same period,
#     same provider, same model. Two caller paths, one anonymous.
#   * 59 of those 95 rows carried completion_tokens=12000 with content_len=0
#     and latency up to 1,050,715 ms. Every one was booked status="success".
# ===================================================================


class TestVllmClientIdentity:
    """gitops #2999 repointed VLLM_URL at the inference-router, so this client
    stopped talking to something that ignores headers and started talking to
    something that LOGS them. It had never sent one.
    """

    @staticmethod
    def _cfg(name: str = "ks-relationship-extractor"):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        return PydanticAIAgentConfig(
            name=name,
            model_provider="vllm",
            model="local-llm",
            db_path="",
            base_url="http://inference-router.inference-router.svc.cluster.local:8150",
        )

    def test_vllm_provider_sends_worker_name(self):
        from src.worker.agents.pydantic_ai_agent import ModelFactory

        model = ModelFactory._create_vllm_model(self._cfg())
        headers = model._provider.client._client.headers
        assert headers.get("X-Worker-Name") == "ks-relationship-extractor"

    def test_vllm_provider_keeps_the_pydantic_ai_user_agent(self):
        """⚠️ THE REGRESSION THIS PINS is "swap in a bare httpx.AsyncClient to
        add one header". That silently drops the OpenAI-matched timeouts and
        the pydantic-ai User-Agent that create_async_http_client() supplies.
        Adding a header must not cost the client's defaults.
        """
        from src.worker.agents.pydantic_ai_agent import ModelFactory

        model = ModelFactory._create_vllm_model(self._cfg())
        ua = model._provider.client._client.headers.get("user-agent", "")
        assert ua.startswith("pydantic-ai/"), f"lost the pydantic-ai UA: {ua!r}"

    def test_worker_name_is_empty_string_never_the_literal_none(self):
        """A literal "None" reaching IR's log field reads as a real worker
        called None. Empty string is the contract the IR provider already uses.
        """
        cfg = self._cfg()
        cfg.name = "x"  # min_length=1 forbids "" at construction
        from src.worker.agents.pydantic_ai_agent import ModelFactory

        model = ModelFactory._create_vllm_model(cfg)
        assert model._provider.client._client.headers.get("X-Worker-Name") == "x"

    @pytest.mark.asyncio
    async def test_worker_name_is_actually_on_the_wire(self):
        """⛔ ASSERTS THE WIRE, NOT THE OBJECT. A header set on a client
        attribute that never reaches an outbound request is exactly the shape
        of the bug being fixed — the old code's identity was "correct" in every
        sense except that nothing was sent. This drives a real request through
        the constructed object graph and reads the headers off the captured
        httpx.Request.
        """
        seen: dict[str, str] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local-llm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        from src.worker.agents.pydantic_ai_agent import ModelFactory

        model = ModelFactory._create_vllm_model(self._cfg("principle_consistency"))
        openai_client = model._provider.client
        openai_client._client._transport = httpx.MockTransport(_capture)

        await openai_client.chat.completions.create(
            model="local-llm",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert seen.get("x-worker-name") == "principle_consistency"


class TestLostAnswerIsNotASuccess:
    """A run that BILLED completion tokens and produced no output was recorded
    status="success" — the metric said the agent worked while the caller got "".
    """

    @staticmethod
    def _usage(output_tokens: int = 12000):
        u = MagicMock()
        u.has_values.return_value = True
        u.input_tokens = 5
        u.output_tokens = output_tokens
        u.total_tokens = 5 + output_tokens
        return u

    @pytest.mark.asyncio
    async def test_lost_answer_is_booked_empty_output(self):
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        mock_response.usage = self._usage()
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")

        assert result.output == ""
        kwargs = mock_metrics.record_aa_prompt_execution.call_args.kwargs
        assert kwargs["status"] == "empty_output"
        # ⚠️ record_execution stays success=True ON PURPOSE: it means "the run
        # completed without raising", and flipping it would redefine
        # agent_executions_failed (today: the exception path) under every
        # dashboard built on it. If someone "tidies" that, this line reds.
        assert mock_metrics.record_execution.call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_a_real_answer_is_still_a_success(self):
        """CONTROL. Without this, changing the status to "empty_output"
        unconditionally would pass the test above.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "a real answer"
        mock_response.usage = self._usage(10)
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")

        assert result.output == "a real answer"
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "success"

    @pytest.mark.asyncio
    async def test_no_output_and_no_tokens_is_not_a_lost_answer(self):
        """⚠️ THE BOUNDARY. "The model was never asked" and "the model answered
        and we lost it" are different failures; only the second is this metric's
        subject. completion_tokens == 0 with no output must stay `success`, or
        the new status becomes a catch-all and stops meaning anything.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        mock_response.usage = self._usage(0)
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            await agent.run("hello")

        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "success"


class TestVacuousAnswerIsNotASuccess:
    """A schema that DECLARED required fields, answered with none of them, was
    booked `success` — a well-formed WRONG answer recorded as healthy.

    ⛔ THE WHOLE DESIGN LIVES IN test_no_required_means_empty_is_a_real_answer.
    Every other case here also passes if the predicate simply returns True for
    any empty-ish body, and that mutant would misclassify three healthy
    extractors (ks-relationship-extractor, ks-entity-spotter,
    principle_consistency) whose `{}` is a correct "found nothing" — 253 of 375
    empty bodies over 24h, measured 2026-09-02. That control is the test.
    """

    # ClassVar because RUF012 blocks a bare mutable class attribute — these are
    # read-only fixtures, never mutated by a case.
    REQUIRED_SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"decision": {"type": "string"}, "rationale": {"type": "string"}},
        "required": ["decision", "rationale"],
    }
    NO_REQUIRED_SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"entities": {"type": "array"}},
    }

    @staticmethod
    def _usage(output_tokens: int = 120):
        u = MagicMock()
        u.has_values.return_value = True
        u.input_tokens = 5
        u.output_tokens = output_tokens
        u.total_tokens = 5 + output_tokens
        return u

    async def _status_for(self, output, schema):
        agent, _, mock_agent_inst = _make_agent(output_schema=schema)
        mock_response = MagicMock()
        mock_response.output = output
        mock_response.usage = self._usage()
        mock_agent_inst.run = AsyncMock(return_value=mock_response)
        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            await agent.run("hello")
        return mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"]

    @pytest.mark.asyncio
    async def test_required_fields_all_absent_is_vacuous(self):
        assert await self._status_for("{}", self.REQUIRED_SCHEMA) == "vacuous_output"

    @pytest.mark.asyncio
    async def test_no_required_means_empty_is_a_real_answer(self):
        """⛔ THE CONTROL, and the reason this is a new status rather than a
        widening of empty_output. A schema with no `required` returning `{}` is
        a correct "found nothing" and MUST stay success.
        """
        assert await self._status_for("{}", self.NO_REQUIRED_SCHEMA) == "success"

    @pytest.mark.asyncio
    async def test_required_present_and_populated_is_success(self):
        body = '{"decision": "ship it", "rationale": "measured"}'
        assert await self._status_for(body, self.REQUIRED_SCHEMA) == "success"

    @pytest.mark.asyncio
    async def test_partial_answer_is_NOT_vacuous(self):
        """⚠️ Some required keys present, others missing, is a VALIDATION
        failure — counted on aa_prompt_validation_failures_total, which can fire
        several times inside one run. A per-run status cannot express that, so
        only a TOTAL absence is booked here.
        """
        assert await self._status_for('{"decision": "ship it"}', self.REQUIRED_SCHEMA) == "success"

    @pytest.mark.asyncio
    async def test_required_keys_present_but_null_or_blank_are_absent(self):
        body = '{"decision": null, "rationale": ""}'
        assert await self._status_for(body, self.REQUIRED_SCHEMA) == "vacuous_output"

    @pytest.mark.asyncio
    async def test_unparseable_output_is_not_claimed_here(self):
        """Fails CLOSED. A parse failure is a different defect with its own
        signal; booking it as vacuous would hide it behind this one.
        """
        assert await self._status_for("not json at all", self.REQUIRED_SCHEMA) == "success"

    @pytest.mark.asyncio
    async def test_json_list_is_not_claimed_here(self):
        """`required` only has meaning for an object."""
        assert await self._status_for("[1, 2, 3]", self.REQUIRED_SCHEMA) == "success"

    @pytest.mark.asyncio
    async def test_no_schema_at_all_is_success(self):
        assert await self._status_for("{}", None) == "success"

    @pytest.mark.asyncio
    async def test_lost_answer_still_wins_over_vacuous(self):
        """⚠️ ORDERING. A lost answer is also, trivially, an answer with no
        required fields. Booking one event under two statuses would corrupt
        both counts, so empty_output must be checked first — this pins it.
        """
        agent, _, mock_agent_inst = _make_agent(output_schema=self.REQUIRED_SCHEMA)
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        mock_response.usage = self._usage(12000)
        mock_agent_inst.run = AsyncMock(return_value=mock_response)
        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            await agent.run("hello")
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "empty_output"


# ---------------------------------------------------------------------------
# 2026-08-25 — the time-budget ceiling and the validation-failure counter.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _time_budget(*, provider_timeout_s: float, decode: float = 11.55, headroom: float = 0.95):
    """Swap the Settings object _resolve_max_tokens reads, for one block.

    ⛔ Deliberately NOT env vars + reset_settings_cache(). ``get_settings`` is
    ``lru_cache``d and shared process-wide, so a test that mutates it leaks into
    every later test in the worker and the damage surfaces somewhere else. This
    rebinds one module global instead and restores it, which cannot leak.
    """
    from src.settings import Settings
    from src.worker.agents import pydantic_ai_agent as mod

    real = mod._get_settings
    fake = Settings(
        self_hosted_provider_timeout_s=provider_timeout_s,
        self_hosted_decode_tokens_per_s=decode,
        self_hosted_token_budget_headroom=headroom,
    )
    mod._get_settings = lambda: fake
    try:
        yield fake
    finally:
        mod._get_settings = real


def _wide_time_budget():
    """A budget large enough that SAFE_MAX_TOKENS is the smaller ceiling."""
    return _time_budget(provider_timeout_s=5000.0)


@contextlib.contextmanager
def _captured_validation_failures():
    """Collect record_aa_prompt_validation_failure calls made during the block."""
    from unittest.mock import MagicMock

    from src.worker.agents import pydantic_ai_agent as mod

    calls: list[dict] = []
    client = MagicMock()
    client.record_aa_prompt_validation_failure.side_effect = lambda **kw: calls.append(kw)
    real = mod.get_metrics_client
    mod.get_metrics_client = lambda: client
    try:
        yield calls
    finally:
        mod.get_metrics_client = real


class TestSelfHostedTimeBudgetCeiling:
    """A max_tokens that cannot finish inside the provider timeout is a defect.

    Root cause of the 2026-08-24 runaway generations. inference-router logged 83
    ``upstream_empty_content`` rows on pgx-local over 16:10Z -> 04:01Z:
    ``finish_reason "length"`` on all 83, ``content_len 0``, ``completion_tokens``
    12000 on 59 of them, ``latency_ms`` up to 1,050,715 (17.5 minutes returning
    nothing). Seven registered workers sat at 10000-16000 and every one of them
    passed the only clamp that existed, because that clamp guards the CONTEXT
    WINDOW and says nothing about whether the request can complete.
    """

    def _config(self, *, provider: str = "inference_router", max_tokens: int | None):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        return PydanticAIAgentConfig(
            name="t", model_provider=provider, model="m", db_path="", max_tokens=max_tokens
        )

    def test_the_SHIPPED_defaults_permit_12000_and_block_15000(self):
        """⛔ Pins the three Settings DEFAULTS, with no overrides at all.

        Every other test in this class injects its own provider_timeout /
        decode / headroom through ``_time_budget``, so none of them touches the
        values that actually ship. Two mutants proved it: flipping
        ``self_hosted_token_budget_headroom`` from 0.95 to 0.75 (which blocks
        the live 12000 override) and to 1.0 (which loosens the bound) both left
        the whole class GREEN. A default nothing asserts is a default anyone can
        change silently — which is the failure mode this ceiling exists to
        prevent one layer down.
        """
        from src.settings import Settings

        s = Settings()
        assert s.self_hosted_provider_timeout_s == 1100.0
        assert s.self_hosted_decode_tokens_per_s == 11.55
        assert s.self_hosted_token_budget_headroom == 0.95
        assert s.self_hosted_max_tokens_ceiling == 12069
        # The two facts the number has to satisfy, stated independently of it.
        assert s.self_hosted_max_tokens_ceiling > 12000
        assert s.self_hosted_max_tokens_ceiling < 15000

    def test_the_ceiling_permits_what_is_affordable_and_blocks_what_is_not(self):
        """floor(1100 x 11.55 x 0.95) = 12069.75 -> 12069, the line between the two.

        ⛔ The number that matters is NOT round, and the two boundary asserts
        below are the whole test. 12000 must survive: the 2026-08-24 incident
        rows show a full 12000-token decode completing END-TO-END in ~1042s,
        inside the 1100s rung, and requirements_management sized that value
        against a measured p99 output tail of 5,540 (cap 6000 = 1.08x the tail
        failed 31% in production; cap 12000 = 0 failures in n=62). 15000 must
        not: ~1299s exceeds the rung.

        An earlier revision used headroom 0.75 -> 9528, which blocked 12000.
        That was a guess overriding another team's measurement.
        """
        with _time_budget(provider_timeout_s=1100.0) as s:
            ceiling = s.self_hosted_max_tokens_ceiling
            assert ceiling == 12069
            assert ceiling > 12000, "12000 is measured-affordable and must not be clamped"
            assert ceiling < 15000, "15000 cannot decode inside the 1100s rung"

    @pytest.mark.parametrize(
        ("configured", "expected_clamp"),
        [
            (16000, True),  # redesign-html — ~1385s, 26% over the 1100s rung
            (15000, True),  # ks-conflict-detector, ks-merge-executor — ~1299s
            # ⛔ 12000 must NOT clamp. It is the live per-message override used
            # by requirements_management's principle_consistency agent, it
            # decodes end-to-end in ~1042s inside the rung, and its owner sized
            # it against a measured p99 tail of 5,540.
            (12000, False),
            (10000, False),  # ~866s
            (8192, False),  # worker-* family
            (8000, False),  # ks-* family
            (6000, False),  # DEFAULT_SELF_HOSTED_MAX_TOKENS
        ],
    )
    def test_the_configured_values_seen_in_production(self, configured, expected_clamp):
        """Every value measured on the live worker_management DB, 2026-08-25.

        ⚠️ Renamed from test_the_seven_live_worker_values: six of those "seven"
        have 0.0 executions in 7 days, so calling them live was wrong.
        """
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        with _time_budget(provider_timeout_s=1100.0) as s:
            got = _resolve_max_tokens(self._config(max_tokens=configured))
            if expected_clamp:
                assert got == s.self_hosted_max_tokens_ceiling
                assert got < configured
            else:
                assert got == configured

    def test_external_providers_are_untouched(self):
        """The ceiling is a property of the self-hosted backend, not a policy."""
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        with _time_budget(provider_timeout_s=1100.0):
            for provider in ("openai", "anthropic", "deepseek", "groq"):
                cfg = self._config(provider=provider, max_tokens=16000)
                assert _resolve_max_tokens(cfg) == 16000

    def test_unset_still_returns_the_self_hosted_default(self):
        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            _resolve_max_tokens,
        )

        with _time_budget(provider_timeout_s=1100.0):
            assert _resolve_max_tokens(self._config(max_tokens=None)) == (
                DEFAULT_SELF_HOSTED_MAX_TOKENS
            )

    def test_a_clamp_is_counted_once_and_names_the_binding_ceiling(self):
        """A clamp that only logs is the defect one layer up."""
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        with _time_budget(provider_timeout_s=1100.0), _captured_validation_failures() as calls:
            _resolve_max_tokens(self._config(max_tokens=16000))

        assert len(calls) == 1
        assert calls[0]["error_type"] == "max_tokens_exceeds_time_budget"
        assert calls[0]["worker_name"] == "t"

    def test_the_kernel_crash_clamp_is_counted_under_its_own_error_type(self):
        """The two ceilings must stay distinguishable in the metric, not just in the log."""
        from src.worker.agents.pydantic_ai_agent import (
            SAFE_MAX_TOKENS,
            PydanticAIAgentConfig,
            _resolve_max_tokens,
        )

        cfg = PydanticAIAgentConfig.model_construct(
            name="t",
            model_provider="inference_router",
            model="m",
            db_path="",
            max_tokens=SAFE_MAX_TOKENS + 5000,
        )
        with _wide_time_budget(), _captured_validation_failures() as calls:
            assert _resolve_max_tokens(cfg) == SAFE_MAX_TOKENS

        assert len(calls) == 1
        assert calls[0]["error_type"] == "max_tokens_exceeds_safe_ceiling"

    def test_a_value_that_fits_records_nothing(self):
        """Negative control: the counter must not fire on the healthy path."""
        from src.worker.agents.pydantic_ai_agent import _resolve_max_tokens

        with _time_budget(provider_timeout_s=1100.0), _captured_validation_failures() as calls:
            _resolve_max_tokens(self._config(max_tokens=8192))
            _resolve_max_tokens(self._config(max_tokens=12000))  # the live override
            _resolve_max_tokens(self._config(max_tokens=None))

        assert calls == []

    def test_broken_telemetry_never_breaks_the_clamp(self):
        """The clamp is the point; the counter is a side effect."""
        from src.worker.agents import pydantic_ai_agent as mod

        def boom():
            raise RuntimeError("otlp exporter is down")

        real = mod.get_metrics_client
        mod.get_metrics_client = boom
        try:
            with _time_budget(provider_timeout_s=1100.0) as s:
                assert mod._resolve_max_tokens(self._config(max_tokens=16000)) == (
                    s.self_hosted_max_tokens_ceiling
                )
        finally:
            mod.get_metrics_client = real


class TestValidationFailuresAreCounted:
    """`record_aa_prompt_validation_failure` was dead; these four paths feed it.

    Before 2026-08-25 the recorder had ZERO call sites in ``src/`` and
    ``aa_prompt_validation_failures_total`` had zero series in Prometheus, while
    every one of the sites below carried its signal as a ``logger.warning`` —
    one of which the code itself already labelled "a greppable dead-signal
    metric".

    ⛔ These CANNOT be folded into ``aa_prompt_execution_total``'s ``status``
    dimension the way #192 folded ``empty_output``. pydantic-ai's retry loop
    runs inside a single ``await agent.run()``, so a failure a later retry
    recovers from is booked ``status="success"`` and vanishes. ``status`` is
    per-RUN; these are per-ATTEMPT.
    """

    def _validator(self, output_schema, grammar_schema=None):
        agent, _, mock_agent_inst = _make_agent(model_provider="openai")
        agent._register_default_schema_validator(output_schema, grammar_schema)
        return agent, mock_agent_inst

    def test_a_fatal_schema_failure_is_counted(self):
        from pydantic_ai import ModelRetry

        _agent, inst = self._validator(
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        )
        validator = inst.output_validator.call_args[0][0]

        with _captured_validation_failures() as calls:
            with pytest.raises(ModelRetry):
                validator(MagicMock(), {"y": "wrong"})

        assert [c["error_type"] for c in calls] == ["schema_invalid"]

    def test_a_valid_output_is_not_counted(self):
        """Negative control — without this the test above passes on a stuck counter."""
        _agent, inst = self._validator(
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        )
        validator = inst.output_validator.call_args[0][0]

        with _captured_validation_failures() as calls:
            assert validator(MagicMock(), {"x": "ok"}) == {"x": "ok"}

        assert calls == []

    def test_a_soft_violation_is_counted_and_does_not_raise(self):
        """The grammar-stripped constraint case: advisory, but no longer invisible."""
        full = {
            "type": "object",
            "properties": {"xs": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
            "required": ["xs"],
        }
        grammar = {
            "type": "object",
            "properties": {"xs": {"type": "array", "items": {"type": "string"}}},
            "required": ["xs"],
        }
        _agent, inst = self._validator(full, grammar)
        validator = inst.output_validator.call_args[0][0]

        with _captured_validation_failures() as calls:
            # Satisfies the decoder-enforced grammar, violates minItems.
            assert validator(MagicMock(), {"xs": ["only-one"]}) == {"xs": ["only-one"]}

        assert [c["error_type"] for c in calls] == ["schema_soft_violation"]

    def test_a_validator_that_cannot_compile_is_counted(self, monkeypatch):
        """⛔ The silent state: NO output validation registered, log-only until now.

        The method's own docstring blames exactly this for making the
        2026-07-20 outage unreproducible.
        """
        import jsonschema

        def boom(_schema):
            raise ValueError("unusable schema")

        monkeypatch.setattr(jsonschema, "Draft202012Validator", boom)

        with _captured_validation_failures() as calls:
            agent, _, inst = _make_agent(model_provider="openai")
            agent._register_default_schema_validator({"type": "object"})

        assert [c["error_type"] for c in calls] == ["validator_compile_failed"]
        # And it really did bail out before registering anything.
        inst.output_validator.assert_not_called()

    def test_an_advisory_validator_that_cannot_compile_is_counted_separately(self, monkeypatch):
        """Enforcement survives; only the soft-violation signal is lost. Say which."""
        import jsonschema

        real = jsonschema.Draft202012Validator
        calls_seen = {"n": 0}

        def flaky(schema):
            calls_seen["n"] += 1
            if calls_seen["n"] == 2:  # the advisory one
                raise ValueError("unusable advisory schema")
            return real(schema)

        monkeypatch.setattr(jsonschema, "Draft202012Validator", flaky)

        with _captured_validation_failures() as calls:
            agent, _, inst = _make_agent(model_provider="openai")
            agent._register_default_schema_validator({"type": "object"})

        assert [c["error_type"] for c in calls] == ["advisory_validator_compile_failed"]
        # Enforcement is unaffected — a validator WAS registered.
        inst.output_validator.assert_called()

    def test_broken_telemetry_never_breaks_validation(self):
        from pydantic_ai import ModelRetry

        from src.worker.agents import pydantic_ai_agent as mod

        _agent, inst = self._validator(
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        )
        validator = inst.output_validator.call_args[0][0]

        def boom():
            raise RuntimeError("otlp exporter is down")

        real = mod.get_metrics_client
        mod.get_metrics_client = boom
        try:
            with pytest.raises(ModelRetry):
                validator(MagicMock(), {"y": "wrong"})
        finally:
            mod.get_metrics_client = real


# ---------------------------------------------------------------------------
# 2026-09-04 — three holes in run()'s telemetry, none of them #216's subject.
# ---------------------------------------------------------------------------


class TestNoneNeverReachesTheCallerUnderSuccess:
    """The `output = ""` coercion was trapped inside `if lost_answer:`, and
    `lost_answer` additionally requires TRUTHY completion_tokens. So a run that
    returned None having billed ZERO completion tokens skipped the coercion,
    was booked status="success", and handed the caller a None.

    ⛔ THE FIXTURE IS FIVE-VALUED ON PURPOSE, AND THAT WIDTH IS THE TEST.
    A single-valued fixture (None only) cannot kill a hardcode of that value:
    coercing `output = ""` unconditionally passes a None-only test while
    destroying every other answer shape. The four non-None values are therefore
    not padding — they are the control group that makes the None case mean
    something.

    ⚠️ And they encode the BOUNDARY the file already fought for: `{}`, `[]` and
    `""` are correct answers, not lost ones. Widening the coercion to
    `if not output:` would still pass the None case and would still be wrong —
    these three rows are what says so.
    """

    # (response.output, what run() must return, the status it must book)
    CASES: ClassVar[list[tuple[Any, Any, str]]] = [
        # THE DEFECT: None with no tokens billed. Coerced, still `success`,
        # because no answer was lost — nothing was ever generated.
        (None, "", "success"),
        # A schema-valid "found nothing" from a no-`required` extractor. Must
        # survive UNCHANGED; this is the row a blanket coercion kills.
        ({}, {}, "success"),
        ([], [], "success"),
        # Genuinely empty text is an answer the caller asked for, not a None.
        ("", "", "success"),
        ("a real answer", "a real answer", "success"),
    ]

    @staticmethod
    def _usage():
        """Zero completion tokens — the regime `lost_answer` cannot see."""
        u = MagicMock()
        u.has_values.return_value = True
        u.input_tokens = 5
        u.output_tokens = 0
        u.total_tokens = 5
        return u

    async def _run(self, response_output):
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = response_output
        # Only consulted when output is None; makes strategy 2/3 find nothing.
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        mock_response.usage = self._usage()
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")
        return result, mock_metrics

    @pytest.mark.parametrize(("sent", "expected_output", "expected_status"), CASES)
    @pytest.mark.asyncio
    async def test_output_is_never_none_and_is_otherwise_untouched(
        self, sent, expected_output, expected_status
    ):
        result, mock_metrics = await self._run(sent)

        # The leak itself.
        assert result.output is not None
        # ⛔ And the width: every non-None shape must come back BYTE-IDENTICAL.
        # `is not None` alone would pass under a blanket `output = ""`.
        assert result.output == expected_output
        assert type(result.output) is type(expected_output)
        status = mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"]
        assert status == expected_status

    @pytest.mark.asyncio
    async def test_the_defect_row_specifically(self):
        """The row that was broken, asserted on its own so a failure names it
        rather than surfacing as one parametrised id among five.
        """
        result, mock_metrics = await self._run(None)
        assert result.output == ""
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "success"

    @pytest.mark.asyncio
    async def test_lost_answer_classification_is_unchanged(self):
        """⚠️ CONTROL ON THE FIX ITSELF. Widening the COERCION must not widen
        the CLASSIFICATION: with tokens billed, None is still `empty_output`.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_response = MagicMock()
        mock_response.output = None
        mock_response.all_messages.return_value = []
        mock_response.all_messages_json.side_effect = Exception("no json")
        u = MagicMock()
        u.has_values.return_value = True
        u.input_tokens = 5
        u.output_tokens = 12000
        u.total_tokens = 12005
        mock_response.usage = u
        mock_agent_inst.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")

        assert result.output == ""
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "empty_output"


class TestCancellationIsRecordedAndReraised:
    """asyncio.CancelledError is a BaseException in py3, so it matched neither
    `except TimeoutError` nor `except Exception`, and the `finally` records no
    outcome. A cancelled run emitted a start and an end and NOTHING in between.
    """

    @staticmethod
    async def _cancel(_prompt):
        raise asyncio.CancelledError

    @pytest.mark.asyncio
    async def test_cancellation_books_status_cancelled(self):
        agent, _, mock_agent_inst = _make_agent()
        mock_agent_inst.run = self._cancel

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(asyncio.CancelledError):
                await agent.run("hello")

        kwargs = mock_metrics.record_aa_prompt_execution.call_args.kwargs
        assert kwargs["status"] == "cancelled"
        assert kwargs["latency_seconds"] is not None
        # Exactly one record — not one per handler.
        assert mock_metrics.record_aa_prompt_execution.call_count == 1

    @pytest.mark.asyncio
    async def test_cancellation_is_reraised_not_swallowed(self):
        """⛔ THE NON-NEGOTIABLE HALF. Swallowing a cancellation tells the
        awaiting task the cancel took effect while this coroutine returns
        normally; asyncio.timeout(), TaskGroup and every cancel scope above
        depend on it propagating. A fix that recorded the metric and returned
        would satisfy the test above and break cooperative cancellation.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_agent_inst.run = self._cancel

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_mc.return_value = MagicMock()
            with pytest.raises(asyncio.CancelledError):
                await agent.run("hello")

    @pytest.mark.asyncio
    async def test_cancellation_is_not_booked_as_a_generic_failure(self):
        """It must be distinguishable from an upstream fault, or a wave of
        deliberate shutdowns reads as a wave of agent defects.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_agent_inst.run = self._cancel

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(asyncio.CancelledError):
                await agent.run("hello")

        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] != "failed"
        exec_kwargs = mock_metrics.record_execution.call_args.kwargs
        assert exec_kwargs["success"] is False
        assert exec_kwargs["error_classification"] == "cancelled"

    @pytest.mark.asyncio
    async def test_the_run_lifecycle_still_closes(self):
        """The `finally` must still run — a cancelled run that leaks its
        transport reference pins the httpx pool open forever.
        """
        agent, _, mock_agent_inst = _make_agent()
        mock_agent_inst.run = self._cancel

        with patch(f"{MODULE}.get_metrics_client") as mock_mc:
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(asyncio.CancelledError):
                await agent.run("hello")

        mock_metrics.record_agent_end.assert_called_once()
        assert agent._active_runs == 0


class TestMcpRecoveredRunIsNotBookedFailed:
    """status="failed" was emitted BEFORE _handle_mcp_error_recovery ran, and a
    run the recovery then RESCUED returned its answer with no compensating
    record. Every MCP-recovered run was booked 100% failed, so the recovery path
    had a 0% success rate by construction rather than by measurement.
    """

    @staticmethod
    def _agent_that_fails():
        agent, _, mock_agent_inst = _make_agent()

        async def failing_run(_prompt):
            raise RuntimeError("MCP connection error")

        mock_agent_inst.run = failing_run
        agent.config.mcp_servers = [MagicMock()]
        return agent

    @pytest.mark.asyncio
    async def test_a_recovered_run_is_booked_recovered(self):
        agent = self._agent_that_fails()
        rescued = MagicMock()
        rescued.output = "recovered answer"

        with (
            patch(f"{MODULE}.get_metrics_client") as mock_mc,
            patch.object(agent, "_handle_mcp_error_recovery", new=AsyncMock(return_value=rescued)),
        ):
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            result = await agent.run("hello")

        assert result is rescued
        kwargs = mock_metrics.record_aa_prompt_execution.call_args.kwargs
        assert kwargs["status"] == "recovered"

    @pytest.mark.asyncio
    async def test_recovery_books_exactly_one_record(self):
        """⛔ NO DOUBLE COUNT. Emitting a compensating `recovered` on top of the
        original `failed` would inflate the denominator and make the recovery
        rate unreadable in the other direction. One run, one record.
        """
        agent = self._agent_that_fails()
        rescued = MagicMock()
        rescued.output = "recovered answer"

        with (
            patch(f"{MODULE}.get_metrics_client") as mock_mc,
            patch.object(agent, "_handle_mcp_error_recovery", new=AsyncMock(return_value=rescued)),
        ):
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            await agent.run("hello")

        assert mock_metrics.record_aa_prompt_execution.call_count == 1
        statuses = [
            c.kwargs["status"] for c in mock_metrics.record_aa_prompt_execution.call_args_list
        ]
        assert "failed" not in statuses

    @pytest.mark.asyncio
    async def test_an_unrecovered_run_is_still_booked_failed(self):
        """⚠️ THE CONTROL. Without it, booking every exception `recovered`
        passes the test above and erases the failure signal entirely.
        """
        agent = self._agent_that_fails()

        with (
            patch(f"{MODULE}.get_metrics_client") as mock_mc,
            patch.object(agent, "_handle_mcp_error_recovery", new=AsyncMock(return_value=None)),
        ):
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(RuntimeError, match="MCP connection error"):
                await agent.run("hello")

        assert mock_metrics.record_aa_prompt_execution.call_count == 1
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_a_recovery_that_itself_raises_still_books_an_outcome(self):
        """Moving the record below the recovery call creates a path where it is
        never reached: _handle_mcp_error_recovery re-raises TimeoutError. The
        `finally` is what stops the fix from opening a new blind spot.
        """
        agent = self._agent_that_fails()

        with (
            patch(f"{MODULE}.get_metrics_client") as mock_mc,
            patch.object(
                agent,
                "_handle_mcp_error_recovery",
                new=AsyncMock(side_effect=TimeoutError("retry timed out")),
            ),
        ):
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            with pytest.raises(TimeoutError):
                await agent.run("hello")

        assert mock_metrics.record_aa_prompt_execution.call_count == 1
        assert mock_metrics.record_aa_prompt_execution.call_args.kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_recovered_latency_covers_the_retry(self):
        """The caller waited through the retry too, so the pre-recovery elapsed
        is not the latency anyone experienced.
        """
        agent = self._agent_that_fails()
        rescued = MagicMock()
        rescued.output = "recovered answer"

        async def slow_recovery(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            return rescued

        with (
            patch(f"{MODULE}.get_metrics_client") as mock_mc,
            patch.object(agent, "_handle_mcp_error_recovery", new=slow_recovery),
        ):
            mock_metrics = MagicMock()
            mock_mc.return_value = mock_metrics
            await agent.run("hello")

        kwargs = mock_metrics.record_aa_prompt_execution.call_args.kwargs
        assert kwargs["status"] == "recovered"
        assert kwargs["latency_seconds"] >= 0.05


# ---------------------------------------------------------------------------
# Per-tenant IR bundle token on the outbound hook (plan §2.4, WM_IR_BUNDLE_TOKENS)
# ---------------------------------------------------------------------------

_BT_IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150"
_BT_TENANT = "019edc1e-0000-7000-8000-0000000000c1"
_BT_TOKEN = "lt_tenant-bundle-token"  # pragma: allowlist secret


@pytest.mark.asyncio
class TestIrBundleTokenOnIdentityHook:
    """Authorization: Bearer <tenant token> goes ONLY to the IR host, only when on."""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        from src.api.identity_context import set_identity_context

        monkeypatch.setenv("INFERENCE_ROUTER_URL", _BT_IR_URL)
        # A licence-PROVEN tenant: only a proven tenant is ever minted for
        # (see tests/test_ir_bundle_mint_proven_tenant.py for the refusals).
        set_identity_context(_BT_TENANT, "lic-1", "sub-1", tenant_provenance="license")
        self.minted: list[str] = []

        async def _fake(tenant_id: str):
            self.minted.append(tenant_id)
            return _BT_TOKEN

        from src.services import ir_bundle_tokens

        monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _fake)
        yield
        set_identity_context(None, None, None)

    @staticmethod
    def _hook():
        from src.settings import reset_settings_cache
        from src.worker.agents.pydantic_ai_agent import make_identity_event_hook

        # Each test sets its env AFTER the autouse cache reset; the first import
        # of the agent module caches Settings, so standalone runs read a stale
        # flag (measured: test_flag_on_ir_host_gets_tenant_bearer red alone).
        reset_settings_cache()
        return make_identity_event_hook(
            worker_name="w",
            worker_version="1",
            fallback_tenant_id="",
            fallback_license_id="",
            fallback_subject_id="",
        )

    @staticmethod
    def _req(url: str) -> httpx.Request:
        return httpx.Request("POST", url, headers={"Authorization": "Bearer internal"})

    async def test_flag_on_ir_host_gets_tenant_bearer(self, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        await self._hook()(req)
        assert req.headers["Authorization"] == f"Bearer {_BT_TOKEN}"
        assert self.minted == [_BT_TENANT]
        # X-Tenant-Id stays the raw uuid — never the Bearer form.
        assert req.headers["X-Tenant-Id"] == _BT_TENANT
        assert not req.headers["X-Tenant-Id"].lower().startswith("bearer")

    async def test_flag_on_other_host_never_gets_the_tenant_token(self, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        for url in (
            "http://vllm.example.com:8000/v1/chat/completions",
            "https://api.deepseek.com/v1/chat/completions",
            "http://inference-router.evil.example:8150/v1/chat/completions",
        ):
            req = self._req(url)
            await self._hook()(req)
            assert req.headers["Authorization"] == "Bearer internal", url
        assert self.minted == []

    async def test_flag_off_sends_no_bearer_to_ir(self, monkeypatch):
        # Flag off ⇒ no token, and the SDK placeholder is dropped, never sent.
        monkeypatch.delenv("WM_IR_BUNDLE_TOKENS", raising=False)
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        await self._hook()(req)
        assert "Authorization" not in req.headers
        assert self.minted == []

    async def test_no_token_minted_drops_the_placeholder(self, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        from src.services import ir_bundle_tokens

        async def _none(tenant_id: str):
            return None

        monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _none)
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        await self._hook()(req)
        assert "Authorization" not in req.headers

    async def test_unresolved_tenant_is_refused_and_mints_nothing(self, monkeypatch):
        """No end tenant ⇒ the completion is refused (IRTenantRequiredError),
        never sent tokenless or as a WM/platform tenant."""
        from src.api.identity_context import set_identity_context
        from src.services.ir_bundle_tokens import IRTenantRequiredError

        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        set_identity_context(None, None, None)
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        with pytest.raises(IRTenantRequiredError):
            await self._hook()(req)
        assert self.minted == []

    async def test_proven_platform_tenant_gets_its_own_token(self, monkeypatch):
        # Owner ruling 2026-10-06: platform tenants are licensed like any other.
        from src.api.identity_context import set_identity_context
        from src.api.tenant_constants import default_tenant_uuid

        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        set_identity_context(default_tenant_uuid(), "lic-1", "sub-1", tenant_provenance="license")
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        await self._hook()(req)
        assert req.headers["Authorization"] == f"Bearer {_BT_TOKEN}"
        assert self.minted == [default_tenant_uuid()]

    async def test_same_host_different_scheme_or_port_never_gets_the_token(self, monkeypatch):
        # Origin match, not hostname match: another listener on the IR host
        # (other port, or https vs http) must not receive the tenant token.
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        host = "inference-router.inference-router.svc.cluster.local"
        for url in (
            f"https://{host}:9999/v1/chat/completions",
            f"http://{host}:9999/v1/chat/completions",
            f"https://{host}:8150/v1/chat/completions",
            f"http://{host}/v1/chat/completions",
        ):
            req = self._req(url)
            await self._hook()(req)
            assert req.headers["Authorization"] == "Bearer internal", url
        assert self.minted == []

    async def test_default_port_matches_when_ir_url_omits_it(self, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "https://ir.example")
        req = self._req("https://ir.example:443/v1/chat/completions")
        await self._hook()(req)
        assert req.headers["Authorization"] == f"Bearer {_BT_TOKEN}"

    async def test_token_step_error_never_fails_the_call(self, monkeypatch, caplog):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        from src.services import ir_bundle_tokens

        secret = "lt_leaked-in-exception-message"  # pragma: allowlist secret

        async def _boom(tenant_id: str):
            raise RuntimeError(f"lock bound to another loop {secret}")

        monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _boom)
        counter = ir_bundle_tokens.BUNDLE_TOKEN_TOTAL.labels(outcome="error")
        before = counter._value.get()
        req = self._req(f"{_BT_IR_URL}/v1/chat/completions")
        with caplog.at_level("DEBUG"):
            await self._hook()(req)  # must not raise
        assert "Authorization" not in req.headers
        assert counter._value.get() == before + 1
        assert secret not in caplog.text
