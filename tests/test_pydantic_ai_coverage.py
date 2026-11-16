import json
import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel

from tests.test_pydantic_ai_agent import _external_llm_allowed

MODULE = "src.worker.agents.pydantic_ai_agent"


def _make_agent(
    model_provider="mlx",
    model="test-model",
    base_url="http://test:8080",
    system_prompt="You are helpful.",
    instruction=None,
    tools=None,
    mcp_servers=None,
    output_schema=None,
    output_validator_code=None,
    api_key=None,
    temperature=0.5,
    mock_model_class=None,
):
    if mock_model_class is None:
        mock_model_class = OpenAIChatModel

    mock_model_instance = MagicMock(spec=mock_model_class)

    from src.worker.agents.pydantic_ai_agent import PydanticAIAgent, PydanticAIAgentConfig

    config = PydanticAIAgentConfig(
        name="test-agent",
        description="test",
        model_provider=model_provider,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        instruction=instruction,
        tools=tools or [],
        db_path="",
        base_url=base_url,
        api_key=api_key,
        mcp_servers=mcp_servers or [],
        output_schema=output_schema,
        output_validator_code=output_validator_code,
    )

    with (
        patch(f"{MODULE}.PydanticAIAgent._initialize_model", return_value=mock_model_instance),
        patch(f"{MODULE}.Agent"),
    ):
        agent = PydanticAIAgent(config=config)
    return agent


class TestPydanticAIAgentConfig:
    def test_valid_config_defaults(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        cfg = PydanticAIAgentConfig(name="a", model_provider="mlx", model="m", db_path="/tmp")
        assert cfg.name == "a"
        assert cfg.temperature == 0.0
        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.output_schema is None

    def test_validation_rejects_invalid_inputs(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgentConfig

        with pytest.raises(ValidationError):
            PydanticAIAgentConfig(name="", model_provider="mlx", model="m", db_path="")
        with pytest.raises(ValidationError):
            PydanticAIAgentConfig(name="x" * 101, model_provider="mlx", model="m", db_path="")
        with pytest.raises(ValidationError):
            # Intentional bad provider — verifies pydantic Literal rejection.
            PydanticAIAgentConfig(
                name="a", model_provider=cast("Any", "invalid"), model="m", db_path=""
            )
        with pytest.raises(ValidationError):
            PydanticAIAgentConfig(
                name="a", model_provider="mlx", model="m", db_path="", temperature=-0.1
            )
        with pytest.raises(ValidationError):
            PydanticAIAgentConfig(
                name="a", model_provider="mlx", model="m", db_path="", temperature=2.1
            )
        with pytest.raises(ValidationError, match="Tool instances"):
            PydanticAIAgentConfig(
                name="a", model_provider="mlx", model="m", db_path="", tools=["not-a-tool"]
            )


class TestModelSettings:
    def test_openai_model_settings_per_provider(self):
        from src.worker.agents.pydantic_ai_agent import DEFAULT_SELF_HOSTED_MAX_TOKENS

        agent = _make_agent(model_provider="mlx", temperature=0.3)
        settings = agent._get_model_settings(agent.config, MagicMock(spec=OpenAIChatModel))
        assert settings["temperature"] == 0.3
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS

        groq_settings = agent._get_model_settings(agent.config, MagicMock(spec=GroqModel))
        assert groq_settings["temperature"] == 0.3

        generic_settings = agent._get_model_settings(agent.config, MagicMock())
        assert generic_settings["temperature"] == 0.3

    def test_inference_router_gets_max_tokens(self):
        from src.worker.agents.pydantic_ai_agent import (
            DEFAULT_SELF_HOSTED_MAX_TOKENS,
            PydanticAIAgentConfig,
        )

        config = PydanticAIAgentConfig(
            name="ir",
            model_provider="inference_router",
            model="m",
            db_path="",
            base_url="http://ir:8150",
            temperature=0.2,
        )
        agent = _make_agent(model_provider="inference_router", base_url="http://ir:8150")
        settings = agent._get_model_settings(config, MagicMock(spec=OpenAIChatModel))
        assert settings["max_tokens"] == DEFAULT_SELF_HOSTED_MAX_TOKENS
        assert settings["temperature"] == 0.2


class TestOutputSchemaSetup:
    def test_mlx_uses_prompted_output_and_deepseek_does_not(self):
        for provider, should_use_prompted in [("mlx", True), ("deepseek", False)]:
            with (
                patch(
                    f"{MODULE}.PydanticAIAgent._initialize_model",
                    return_value=MagicMock(spec=OpenAIChatModel),
                ),
                patch(f"{MODULE}.Agent"),
                patch(f"{MODULE}.PromptedOutput") as mock_prompted,
            ):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="t",
                    model_provider=cast("Any", provider),
                    model="m",
                    db_path="",
                    base_url="http://test:8080",
                    api_key="k",
                    output_schema={"type": "object", "properties": {"a": {"type": "string"}}},
                )
                PydanticAIAgent(config=config)
                assert mock_prompted.called == should_use_prompted, f"Failed for {provider}"

    def test_output_validator_code_is_refused_not_registered(self):
        from src.security.request_overrides import OutputValidatorCodeNotSupportedError

        validator_code = "def validate(ctx, output):\n    return output\n"
        with pytest.raises(OutputValidatorCodeNotSupportedError):
            _make_agent(output_validator_code=validator_code)

    def test_default_schema_validators(self):
        agent = _make_agent(
            output_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        )
        agent.agent.output_validator.assert_called_once()

        agent2 = _make_agent()
        agent2._register_default_schema_validator({"type": "array", "items": {"type": "string"}})
        agent2._register_default_schema_validator({"type": "string"})
        assert agent2.agent.output_validator.call_count >= 2


class TestContextAwarePrompt:
    def test_full_context_included_in_prompt(self):
        agent = _make_agent()
        prompt = agent._build_context_aware_prompt(
            task="Analyze data",
            graph_context={
                "session_id": "s1",
                "graph_id": "g1",
                "node_id": "n1",
                "completed_nodes": ["n0"],
                "previous_outputs": {"n0": "result_0"},
                "variables": {"key": "value"},
            },
            node_config={"param": "val"},
        )
        for expected in [
            "Analyze data",
            "Session ID: s1",
            "Graph ID: g1",
            "Current Node: n1",
            "Completed nodes: n0",
            "n0: result_0",
            "key=value",
            "param",
        ]:
            assert expected in prompt

    def test_empty_context_returns_task_only(self):
        agent = _make_agent()
        assert agent._build_context_aware_prompt("Do X", {}, {}) == "Do X"

    def test_previous_outputs_truncated_and_empty_ignored(self):
        agent = _make_agent()
        info = []
        agent._add_previous_outputs_context({"previous_outputs": {"a": "x" * 300}}, info)
        assert "..." in info[0]

        info2 = []
        agent._add_previous_outputs_context({"previous_outputs": {}}, info2)
        assert info2 == []

    def test_variables_context(self):
        agent = _make_agent()
        info = []
        agent._add_variables_context({"variables": {"a": 1, "b": 2}}, info)
        assert "a=1" in info[0] and "b=2" in info[0]


class TestTokenUsageExtraction:
    def test_from_run_result_usage(self):
        agent = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = True
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_result.usage.return_value = mock_usage

        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20
        assert metrics.total_tokens == 30

    def test_fallback_to_model_usage_and_legacy_names(self):
        agent = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_result.usage.return_value = mock_usage

        model_usage = MagicMock(spec=[])
        model_usage.input_tokens = None
        model_usage.prompt_tokens = 42
        model_usage.output_tokens = None
        model_usage.completion_tokens = 18
        model_usage.total_tokens = 60
        agent.agent.model = MagicMock()
        agent.agent.model.usage = model_usage

        metrics = agent._extract_token_usage("test", mock_result)
        assert metrics.prompt_tokens == 42
        assert metrics.completion_tokens == 18

    def test_returns_none_when_no_usage_or_exception(self):
        agent = _make_agent()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_result.usage.return_value = mock_usage
        agent.agent.model = MagicMock(spec=[])
        assert agent._extract_token_usage("test", mock_result) is None

        mock_result2 = MagicMock()
        mock_result2.usage.side_effect = RuntimeError("oops")
        assert agent._extract_token_usage("test", mock_result2) is None


class TestExtractTextOutput:
    def test_direct_and_dict_and_empty_string(self):
        agent = _make_agent()
        for val in ["hello", {"k": "v"}, ""]:
            resp = MagicMock()
            resp.output = val
            assert agent._extract_text_output(resp) == val

    def test_none_output_extracts_from_messages(self):
        agent = _make_agent()
        resp = MagicMock()
        resp.output = None
        mock_part = MagicMock()
        mock_part.content = "from messages"
        mock_msg = MagicMock()
        mock_msg.parts = [mock_part]
        resp.all_messages.return_value = [mock_msg]
        assert agent._extract_text_output(resp) == "from messages"

    def test_none_output_extracts_from_json_or_returns_none(self):
        agent = _make_agent()
        resp = MagicMock()
        resp.output = None
        resp.all_messages.return_value = []
        resp.all_messages_json.return_value = json.dumps(
            [
                {
                    "role": "model-text-response",
                    "parts": [{"part_kind": "text", "content": "json_out"}],
                }
            ]
        )
        assert agent._extract_text_output(resp) == "json_out"

        resp2 = MagicMock()
        resp2.output = None
        resp2.all_messages.return_value = []
        resp2.all_messages_json.side_effect = Exception("fail")
        assert agent._extract_text_output(resp2) is None

    def test_none_output_from_msg_content_attr(self):
        agent = _make_agent()
        resp = MagicMock()
        resp.output = None
        msg = MagicMock(spec=[])
        msg.content = "direct content"
        resp.all_messages.return_value = [msg]
        assert agent._extract_text_output(resp) == "direct content"


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_success(self):
        agent = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "result text"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = True
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 10
        mock_usage.total_tokens = 15
        mock_response.usage.return_value = mock_usage
        agent.agent.run = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            result = await agent.run("test prompt")
        assert result.output == "result text"
        assert result.usage_metrics.total_tokens == 15

    @pytest.mark.asyncio
    async def test_run_none_response_and_timeout_and_exception(self):
        agent = _make_agent()
        agent.agent.run = AsyncMock(return_value=None)
        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            with pytest.raises(ValueError, match="unexpected None"):
                await agent.run("p")

        async def timeout_run(prompt):
            raise TimeoutError("timeout")

        agent.agent.run = timeout_run
        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            with pytest.raises(TimeoutError, match="timed out"):
                await agent.run("p")

        agent.agent.run = AsyncMock(side_effect=RuntimeError("model error"))
        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="model error"):
                await agent.run("p")


class TestRunWithContext:
    @pytest.mark.asyncio
    async def test_builds_prompt_and_propagates_errors(self):
        agent = _make_agent()
        mock_response = MagicMock()
        mock_response.output = "ctx result"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_response.usage.return_value = mock_usage
        agent.agent.run = AsyncMock(return_value=mock_response)
        agent.agent.model = MagicMock(spec=[])

        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            result = await agent.run_with_context(
                {
                    "task": "process data",
                    "graph_context": {"graph_id": "g1"},
                    "node_config": {"key": "val"},
                }
            )
        assert result.output == "ctx result"
        assert "process data" in agent.agent.run.call_args[0][0]

        agent.agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(f"{MODULE}.get_metrics_client") as m:
            m.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="boom"):
                await agent.run_with_context({"task": "x"})


class TestMCPErrorRecovery:
    @pytest.mark.asyncio
    async def test_non_mcp_and_no_servers_return_none(self):
        agent = _make_agent()
        assert await agent._handle_mcp_error_recovery(RuntimeError("normal"), "p", "t") is None

        agent2 = _make_agent(mcp_servers=[])
        assert await agent2._handle_mcp_error_recovery(RuntimeError("MCP failed"), "p", "t") is None

    @pytest.mark.asyncio
    async def test_mcp_recovery_success(self):
        agent = _make_agent(mcp_servers=[MagicMock()])
        mock_response = MagicMock()
        mock_response.output = "recovered"
        mock_usage = MagicMock()
        mock_usage.has_values.return_value = False
        mock_response.usage.return_value = mock_usage
        agent.agent.run = AsyncMock(return_value=mock_response)
        agent.agent.model = MagicMock(spec=[])

        with patch.object(agent, "_initialize_mcp_servers", new_callable=AsyncMock) as mi:

            async def mark_ready():
                agent.mcp_servers_ready = True

            mi.side_effect = mark_ready
            result = await agent._handle_mcp_error_recovery(RuntimeError("MCP failed"), "p", "t")
        assert result is not None and result.output == "recovered"

    @pytest.mark.asyncio
    async def test_mcp_recovery_reinit_fails(self):
        agent = _make_agent(mcp_servers=[MagicMock()])
        with patch.object(agent, "_initialize_mcp_servers", new_callable=AsyncMock):
            agent.mcp_servers_ready = False
            assert await agent._handle_mcp_error_recovery(RuntimeError("MCP"), "p", "t") is None


class TestHealthCleanupTools:
    @pytest.mark.asyncio
    async def test_health_check(self):
        agent = _make_agent()
        assert await agent.check_health() is True
        agent.agent = None
        assert await agent.check_health() is False

    @pytest.mark.asyncio
    async def test_cleanup_and_get_tool(self):
        agent = _make_agent()
        await agent.cleanup()
        assert agent.get_tool("nonexistent") is None

    def test_add_tools_from_json(self):
        agent = _make_agent()
        tool_json = {
            "name": "add_numbers",
            "description": "Add two numbers",
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
        agent.add_tools_from_json([tool_json])
        assert "add_numbers" in agent._tool_registry

        agent.add_tools_from_json([tool_json])

        with pytest.raises(Exception):
            agent.add_tools_from_json([{"name": "bad"}])


class TestFactoryFunction:
    def test_create_pydantic_ai_agent(self):
        with (
            patch(f"{MODULE}.MLXOpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

            agent = create_pydantic_ai_agent(
                name="factory-agent",
                description="test",
                model_provider="mlx",
                model="test-model",
                base_url="http://test:8080",
                db_path="",
            )
            assert agent.name == "factory-agent"

            agent2 = create_pydantic_ai_agent(
                name="mcp-agent",
                description="test",
                model_provider="mlx",
                model="m",
                base_url="http://test:8080",
                db_path="",
                mcp_servers_list=[MagicMock()],
            )
            assert len(agent2.config.mcp_servers) == 1


class TestMLXWrappers:
    @pytest.mark.asyncio
    async def test_completions_wrapper_strips_think_tags(self):
        from src.worker.agents.pydantic_ai_agent import MLXChatWrapper, MLXCompletionsWrapper

        mock_completions = AsyncMock()
        choice = MagicMock()
        choice.message.content = "<think>reasoning here</think>\nActual answer"
        response = MagicMock()
        response.choices = [choice]
        mock_completions.create.return_value = response

        wrapper = MLXCompletionsWrapper(mock_completions)
        result = await wrapper.create(prompt="test")
        assert "<think>" not in result.choices[0].message.content
        assert "Actual answer" in result.choices[0].message.content

        mock_chat = MagicMock()
        chat_wrapper = MLXChatWrapper(mock_chat)
        assert hasattr(chat_wrapper, "completions")

    def test_inference_router_provider_no_schema_id_header(self):
        """X-Schema-Id is no longer injected — inference-router self-derives from body."""
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(base_url="http://ir:8150/v1")
        assert "X-Schema-Id" not in provider._client.default_headers
        # ⛔ INVERTED 2026-08-27. Phase 5a's rule was "identity headers always
        # present, default to empty string when unset". That is the defect: an
        # empty header is a PRESENT header, so inference-router's
        # Header(default=...) never fired and IR coerced "" to the nil tenant
        # sentinel. 1,400 of 13,893 routed calls in 24h dead-lettered in billing
        # because of it. Unset identity is now ABSENT, which is the only shape
        # that can mean "unknown" rather than "nobody".
        assert "X-Tenant-Id" not in provider._client.default_headers
        assert "X-License-Id" not in provider._client.default_headers
        assert "X-Subject-Id" not in provider._client.default_headers

    def test_inference_router_provider_forwards_identity_headers(self):
        """Phase 5a: tenant_id / license_id / subject_id flow through to default_headers."""
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(
            base_url="http://ir:8150/v1",
            tenant_id="aaaaaaaa-0000-4000-8000-000000000001",
            license_id="lic-uuid-1",
            subject_id="subj-uuid-2",
        )
        headers = provider._client.default_headers
        assert "X-Schema-Id" not in headers
        assert headers.get("X-Tenant-Id") == "aaaaaaaa-0000-4000-8000-000000000001"
        assert headers.get("X-License-Id") == "lic-uuid-1"
        assert headers.get("X-Subject-Id") == "subj-uuid-2"

    def test_inference_router_provider_none_subject_becomes_empty_string(self):
        """A None subject must never reach the wire as the literal "None".

        ⛔ REWRITTEN 2026-08-27. The original assertion was `== ""`, i.e. the
        header is present and empty. The real hazard this test guards is the
        string "None" leaking from an f-string; absence satisfies that hazard
        strictly better than "" does, and "" is what caused the nil-tenant
        dead-letter defect on the sibling header.
        """
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(
            base_url="http://ir:8150/v1",
            tenant_id="aaaaaaaa-0000-4000-8000-000000000001",
            subject_id=None,
        )
        assert "X-Subject-Id" not in provider._client.default_headers
        assert provider._client.default_headers.get("X-Subject-Id") != "None"

    def test_create_inference_router_model_passes_identity_to_provider(self):
        """End-to-end: PydanticAIAgentConfig identity fields land in provider headers."""
        from src.worker.agents.pydantic_ai_agent import (
            PydanticAIAgent,
            PydanticAIAgentConfig,
        )

        config = PydanticAIAgentConfig(
            name="ir-agent",
            model_provider="inference_router",
            model="my-schema",
            db_path="",
            base_url="http://ir:8150",
            tenant_id="aaaaaaaa-0000-4000-8000-000000000001",
            license_id="lic-1",
            subject_id="subj-2",
        )
        agent = PydanticAIAgent.__new__(PydanticAIAgent)
        with patch(
            "src.worker.agents.pydantic_ai_agent.InferenceRouterOpenAIProvider"
        ) as mock_provider:
            agent._create_inference_router_model(config)

        call_kwargs = mock_provider.call_args.kwargs
        assert call_kwargs["tenant_id"] == "aaaaaaaa-0000-4000-8000-000000000001"
        assert call_kwargs["license_id"] == "lic-1"
        assert call_kwargs["subject_id"] == "subj-2"
        assert "schema_id" not in call_kwargs


class TestInferenceRouterIdentityContextOverride:
    """Provider's httpx event hook reads identity from ContextVars at request-send time.

    This is what makes per-user attribution work for the cached
    `WorkerInstanceCache` path: the same `AsyncOpenAI` client is reused
    across requests from different end-users, and identity is sourced from
    the caller's async context — never from the cached construction args.
    """

    @pytest.mark.asyncio
    async def test_event_hook_overrides_headers_from_contextvar(self):
        """ContextVar set at request time overrides constructor fallbacks."""
        import httpx

        from src.api.identity_context import set_identity_context
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(
            base_url="http://ir:8150/v1",
            tenant_id="fallback-tenant",
            license_id="fallback-license",
            subject_id="fallback-subject",
        )

        # Simulate two concurrent users sharing the cached provider.
        async def issue_with_identity(tenant: str, license_: str, subject: str) -> httpx.Request:
            set_identity_context(tenant, license_, subject)
            captured: dict[str, httpx.Request] = {}

            async def capture(request: httpx.Request) -> None:
                captured["req"] = request

            req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
            # Run the provider's event hooks against the synthetic request.
            for hook in provider._client._client.event_hooks["request"]:
                await hook(req)
            await capture(req)
            return captured["req"]

        req_a = await issue_with_identity("tenant-a", "lic-a", "user-a")
        assert req_a.headers["X-Tenant-Id"] == "tenant-a"
        assert req_a.headers["X-License-Id"] == "lic-a"
        assert req_a.headers["X-Subject-Id"] == "user-a"

        req_b = await issue_with_identity("tenant-b", "lic-b", "user-b")
        assert req_b.headers["X-Tenant-Id"] == "tenant-b"
        assert req_b.headers["X-License-Id"] == "lic-b"
        assert req_b.headers["X-Subject-Id"] == "user-b"

    @pytest.mark.asyncio
    async def test_event_hook_falls_back_to_constructor_when_contextvar_unset(self):
        """When no middleware has set the ContextVar, constructor args win."""
        import httpx

        from src.api.identity_context import set_identity_context
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(
            base_url="http://ir:8150/v1",
            tenant_id="ctor-tenant",
            license_id="ctor-license",
            subject_id="ctor-subject",
        )

        # Explicitly clear the ContextVars in this task.
        set_identity_context(None, None, None)

        req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
        for hook in provider._client._client.event_hooks["request"]:
            await hook(req)

        assert req.headers["X-Tenant-Id"] == "ctor-tenant"
        assert req.headers["X-License-Id"] == "ctor-license"
        assert req.headers["X-Subject-Id"] == "ctor-subject"

    @pytest.mark.asyncio
    async def test_event_hook_concurrent_tasks_see_their_own_identity(self):
        """ContextVars are task-local: concurrent requests don't bleed identity."""
        import asyncio

        import httpx

        from src.api.identity_context import set_identity_context
        from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

        provider = InferenceRouterOpenAIProvider(base_url="http://ir:8150/v1")

        async def one_user(subject: str, sleep_s: float) -> str:
            set_identity_context(f"tenant-{subject}", f"lic-{subject}", subject)
            await asyncio.sleep(sleep_s)  # interleave with the other task
            req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
            for hook in provider._client._client.event_hooks["request"]:
                await hook(req)
            return req.headers["X-Subject-Id"]

        results = await asyncio.gather(
            one_user("alice", 0.01),
            one_user("bob", 0.005),
            one_user("carol", 0.0),
        )
        assert sorted(results) == ["alice", "bob", "carol"]


class TestMCPLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self):
        agent = _make_agent()
        await agent._initialize_mcp_servers()
        assert agent.mcp_servers_ready is True

        agent._mcp_exit_stack = AsyncMock()
        await agent._cleanup_mcp_on_error()
        assert agent.mcp_servers_ready is False
        assert agent._mcp_exit_stack is None

    @pytest.mark.asyncio
    async def test_mcp_server_health_checks(self):
        agent = _make_agent()

        server_running = MagicMock()
        server_running.is_running = True
        assert await agent._is_mcp_server_healthy(server_running) is True

        server_cmd = MagicMock(spec=["command"])
        server_cmd.command = "npx"
        assert await agent._is_mcp_server_healthy(server_cmd) is True

        assert await agent._is_mcp_server_healthy(None) is False

    @pytest.mark.asyncio
    async def test_ensure_mcp_servers_ready(self):
        agent = _make_agent()
        agent.config.mcp_servers = []
        await agent._ensure_mcp_servers_ready()

        agent2 = _make_agent(mcp_servers=[MagicMock()])
        agent2.mcp_servers_ready = True
        await agent2._ensure_mcp_servers_ready()


class TestRunAgentWithTimeout:
    @pytest.mark.asyncio
    async def test_timeout_values(self):
        agent = _make_agent()
        mock_response = MagicMock()
        agent.agent.run = AsyncMock(return_value=mock_response)
        assert await agent._run_agent_with_timeout("p", "t") is mock_response

        agent2 = _make_agent(mcp_servers=[MagicMock()])
        agent2.agent.run = AsyncMock(return_value=mock_response)
        assert await agent2._run_agent_with_timeout("p", "t") is mock_response


class TestBaseAgentModels:
    def test_agent_run_result(self):
        from src.worker.agents.base_agent import AgentRunResult

        result = AgentRunResult(
            output="test",
            session_id="s1",
            cost_usd=0.05,
            duration_ms=1200,
            is_error=False,
        )
        assert result.output == "test"
        assert result.cost_usd == 0.05

        minimal = AgentRunResult(output="x")
        assert minimal.usage_metrics is None and minimal.session_id is None

    def test_token_usage_metrics(self):
        from src.worker.agents.base_agent import TokenUsageMetrics

        m = TokenUsageMetrics(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert m.model_dump()["total_tokens"] == 30

        empty = TokenUsageMetrics()
        assert empty.prompt_tokens is None

    def test_base_agent_get_agent_type_and_defaults(self):
        from src.worker.agents.base_agent import BaseAgent

        class MyAgent(BaseAgent):
            async def run(self, prompt):
                pass

            async def run_with_context(self, enhanced_context):
                pass

            async def check_health(self):
                return True

            async def cleanup(self):
                pass

        agent = MyAgent()
        assert agent.get_agent_type() == "my"
        assert agent.name == "MyAgent"
        assert agent.db_path == ""


class TestAgentRegistry:
    def test_register_create_unregister(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        factory = MagicMock(return_value=MagicMock())
        registry.register("test_type", factory)
        assert registry.is_registered("test_type")

        registry.create("test_type", name="n")
        factory.assert_called_once_with(name="n")

        registry.register("test_type", MagicMock())

        registry.unregister("test_type")
        assert not registry.is_registered("test_type")

        with pytest.raises(KeyError):
            registry.unregister("nope")

    def test_create_nonexistent_and_factory_failure(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        with pytest.raises(ValueError, match="Unknown agent type"):
            registry.create("nope")

        registry.register("bad", MagicMock(side_effect=RuntimeError("factory fail")))
        with pytest.raises(RuntimeError, match="factory fail"):
            registry.create("bad")

    def test_list_and_get_class(self):
        from src.worker.agents.agent_registry import AgentRegistry
        from src.worker.agents.base_agent import BaseAgent

        registry = AgentRegistry()

        class FakeAgent(BaseAgent):
            async def run(self, prompt):
                pass

            async def run_with_context(self, enhanced_context):
                pass

            async def check_health(self):
                return True

            async def cleanup(self):
                pass

        registry.register("a", MagicMock())
        registry.register("b", MagicMock(), agent_class=FakeAgent)
        assert sorted(registry.list_agent_types()) == ["a", "b"]
        assert registry.get_agent_class("b") is FakeAgent
        assert registry.get_agent_class("missing") is None

    def test_global_registry(self):
        from src.worker.agents.agent_registry import get_agent_registry, list_available_agent_types

        reg = get_agent_registry()
        assert reg is not None
        assert isinstance(list_available_agent_types(), list)


class TestToolTemplate:
    def test_from_json_and_create_tool(self):
        from src.worker.agents.tool_template import ToolTemplate

        data = {
            "name": "greet",
            "description": "Greet someone",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            "implementation": "async def impl(ctx, **kwargs):\n    return {'message': f'Hello {kwargs[\"name\"]}'}",
        }
        template = ToolTemplate.from_json(data)
        assert template.name == "greet"
        assert template.create_tool().name == "greet"

    def test_from_json_with_python_code_field(self):
        from src.worker.agents.tool_template import ToolTemplate

        data = {
            "name": "add",
            "description": "Add",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "integer"}},
            },
            "python_code": "def add_numbers(a, b):\n    return {'result': a + b}",
        }
        template = ToolTemplate.from_json(data)
        assert template.name == "add"

    def test_wrap_function(self):
        from src.worker.agents.tool_template import ToolTemplate

        wrapped = ToolTemplate._wrap_function_as_impl("def my_func(x):\n    return x * 2")
        assert "async def impl" in wrapped
        with pytest.raises(ValueError, match="No function definition"):
            ToolTemplate._wrap_function_as_impl("x = 1")

    def test_security_validation(self):
        from src.worker.agents.tool_template import ToolTemplate

        for code, match in [
            ("import os", "Dangerous import"),
            ("import subprocess", "Dangerous import"),
            ("eval('1+1')", "Dangerous function"),
            ("exec('print(1)')", "Dangerous function"),
            ("x.__globals__", "Dangerous attribute"),
            ("def (", "Invalid Python syntax"),
        ]:
            with pytest.raises(ValueError, match=match):
                ToolTemplate._validate_code_security(code)

        ToolTemplate._validate_code_security("x = len([1, 2, 3])\ny = str(x)")

    def test_sandboxed_namespace(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        for key in ["len", "str", "math", "json", "re"]:
            assert key in ns

    def test_field_types(self):
        from src.worker.agents.schemas import SchemaField
        from src.worker.agents.tool_template import ToolTemplate

        assert ToolTemplate._get_field_type(SchemaField(type="string"), "P") is str
        assert ToolTemplate._get_field_type(SchemaField(type="integer"), "P") is int
        assert ToolTemplate._get_field_type(SchemaField(type="number"), "P") is float
        assert ToolTemplate._get_field_type(SchemaField(type="boolean"), "P") is bool
        assert (
            ToolTemplate._get_field_type(SchemaField(type="array", items={"type": "string"}), "P")
            == list[Any]
        )
        assert ToolTemplate._get_field_type(SchemaField(type="object"), "P") == dict[str, Any]
        assert ToolTemplate._get_field_type(SchemaField(type="foobar"), "P") is Any

        ft = ToolTemplate._get_field_type(
            SchemaField(type="object", properties={"x": {"type": "string"}}), "P"
        )
        assert issubclass(ft, BaseModel)

    def test_failure_tracking_and_stats(self):
        from src.worker.agents.tool_template import ToolTemplate

        ToolTemplate._record_tool_failure("test_ft")
        assert not ToolTemplate._is_tool_failed("test_ft")
        for _ in range(10):
            ToolTemplate._record_tool_failure("test_ft")
        assert ToolTemplate._is_tool_failed("test_ft")

        ToolTemplate._record_execution_stats("stat_tool", 0.5, True)
        ToolTemplate._record_execution_stats("stat_tool", 1.5, False)
        stats = ToolTemplate.get_tool_stats("stat_tool")
        assert stats is not None
        assert stats["total_executions"] == 2
        assert stats["min_time"] == 0.5

        ToolTemplate._record_execution_stats("all_s", 1.0, True)
        assert "all_s" in ToolTemplate.get_all_stats()

    def test_cache_operations(self):
        from src.worker.agents.tool_template import MAX_CACHE_SIZE, ToolTemplate, _compilation_cache

        k1 = ToolTemplate._get_cache_key("code", "In", "Out")
        k2 = ToolTemplate._get_cache_key("code", "In", "Out")
        assert k1 == k2
        assert k1 != ToolTemplate._get_cache_key("code2", "In", "Out")

        original = dict(_compilation_cache)
        for i in range(MAX_CACHE_SIZE + 5):
            _compilation_cache[f"evict_{i}"] = f"v_{i}"
        ToolTemplate._evict_cache_if_needed()
        assert len(_compilation_cache) < MAX_CACHE_SIZE + 5
        _compilation_cache.clear()
        _compilation_cache.update(original)

    def test_create_tool_no_implementation(self):
        from src.worker.agents.tool_template import ToolTemplate

        template = ToolTemplate(
            name="no_impl",
            description="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            implementation=lambda: None,
        )
        # Intentionally clear implementation to drive the error path.
        cast("Any", template).implementation = None
        with pytest.raises(ValueError, match="no implementation"):
            template.create_tool()

    def test_execute_in_sandbox_no_impl(self):
        from src.worker.agents.tool_template import ToolTemplate

        ns = ToolTemplate._create_sandboxed_namespace()
        with pytest.raises(ValueError, match="must define"):
            ToolTemplate._execute_in_sandbox("x = 1", ns)


class TestSchemas:
    def test_schema_field_and_tool_schema(self):
        from src.worker.agents.schemas import SchemaField, ToolSchema

        f = SchemaField(type="string", description="a field")
        assert f.type == "string"

        with pytest.raises(ValidationError):
            # Intentional bad type — verifies pydantic rejection.
            SchemaField(type=cast("Any", 123))
        with pytest.raises(ValidationError, match="dictionary"):
            SchemaField.model_validate([1, 2, 3])

        ts = ToolSchema(
            name="t",
            description="d",
            input_schema={"properties": {}},
            output_schema={"properties": {}},
            implementation="x = 1",
        )
        assert ts.name == "t"

        with pytest.raises(ValidationError, match="Invalid Python"):
            ToolSchema(
                name="t", description="d", input_schema={}, output_schema={}, implementation="def ("
            )


class TestPydanticAIAgentWrapper:
    def test_availability_and_mock_agent(self):
        from src.worker.agents.pydantic_ai_agent_wrapper import (
            MockPydanticAIAgent,
            is_pydantic_ai_available,
        )

        assert is_pydantic_ai_available() is True

        agent = MockPydanticAIAgent(name="mock-test", db_path="/tmp")
        assert agent.name == "mock-test"
        assert agent.get_tool("x") is None
        agent.add_tools_from_json([{"name": "t"}])

    @pytest.mark.asyncio
    async def test_mock_agent_async_methods(self):
        from src.worker.agents.pydantic_ai_agent_wrapper import MockPydanticAIAgent

        agent = MockPydanticAIAgent(name="m")
        result = await agent.run("test")
        assert "error" in result and "not available" in result["output"]

        ctx_result = await agent.run_with_context({"task": "t"})
        assert "error" in ctx_result

        assert await agent.check_health() is False
        await agent.cleanup()

    def test_safe_create_variants(self):
        from src.worker.agents.pydantic_ai_agent_wrapper import MockPydanticAIAgent

        with (
            patch(f"{MODULE}.MLXOpenAIProvider"),
            patch(f"{MODULE}.OpenAIChatModel"),
            patch(f"{MODULE}.Agent"),
            patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            from src.worker.agents.pydantic_ai_agent_wrapper import safe_create_pydantic_ai_agent

            agent = safe_create_pydantic_ai_agent(
                name="safe-agent",
                description="test",
                model_provider="mlx",
                model="m",
                base_url="http://test:8080",
            )
            assert agent.name == "safe-agent"

        with patch(
            "src.worker.agents.pydantic_ai_agent_wrapper.create_pydantic_ai_agent",
            side_effect=RuntimeError("kaboom"),
        ):
            from src.worker.agents.pydantic_ai_agent_wrapper import safe_create_pydantic_ai_agent

            agent = safe_create_pydantic_ai_agent(
                name="fail", description="t", model_provider="mlx", model="m"
            )
            assert isinstance(agent, MockPydanticAIAgent)

        with patch("src.worker.agents.pydantic_ai_agent_wrapper.PYDANTIC_AI_AVAILABLE", False):
            from src.worker.agents.pydantic_ai_agent_wrapper import safe_create_pydantic_ai_agent

            agent = safe_create_pydantic_ai_agent(
                name="u", description="t", model_provider="mlx", model="m"
            )
            assert isinstance(agent, MockPydanticAIAgent)


class TestModelInitialization:
    def test_openai_and_cohere_require_api_key(self):
        from pydantic_ai.exceptions import UserError

        for provider, env_var, model_name in [
            ("openai", "OPENAI_API_KEY", "gpt-4"),
            ("cohere", "COHERE_API_KEY", "command"),
        ]:
            with patch(f"{MODULE}.Agent"):
                os.environ.pop(env_var, None)
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="t", model_provider=cast("Any", provider), model=model_name, db_path=""
                )
                # Opted in: the fail-closed local-only guard refuses an external
                # provider BEFORE the key is looked up, so without this the test
                # would assert the wrong refusal and still be green.
                with _external_llm_allowed(), pytest.raises(UserError, match="API key required"):
                    PydanticAIAgent(config=config)

    def test_providers_with_custom_base_url(self):
        # Every provider below is EXTERNAL, and their custom-base-url wiring only
        # exists on the explicit opt-in now that the local-only guard is fail-closed.
        provider_configs = [
            ("openai", "OpenAIProvider", "OpenAIChatModel", OpenAIChatModel, {"api_key": "k"}),
            ("deepseek", "DeepSeekProvider", "OpenAIChatModel", OpenAIChatModel, {"api_key": "k"}),
            ("groq", "GroqProvider", "GroqModel", GroqModel, {"api_key": "k"}),
            ("cohere", "CohereProvider", "CohereModel", None, {"api_key": "k"}),
        ]
        for provider, prov_cls, model_cls, spec, extra in provider_configs:
            patches = [
                patch(f"{MODULE}.Agent"),
                patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
                patch(
                    f"{MODULE}.{model_cls}",
                    return_value=MagicMock(spec=spec) if spec else MagicMock(),
                ),
                patch(f"{MODULE}.{prov_cls}"),
            ]
            if provider in ("openai", "deepseek"):
                patches.append(patch(f"{MODULE}.AsyncOpenAI"))
            if provider == "groq":
                patches.append(patch(f"{MODULE}.AsyncGroq"))
            if provider == "cohere":
                patches.append(patch(f"{MODULE}.AsyncCohereClient"))

            with _external_llm_allowed(), contextlib_exitstack(patches):
                from src.worker.agents.pydantic_ai_agent import (
                    PydanticAIAgent,
                    PydanticAIAgentConfig,
                )

                config = PydanticAIAgentConfig(
                    name="t",
                    model_provider=cast("Any", provider),
                    model="m",
                    db_path="",
                    base_url="http://custom:9000",
                    **cast("dict[str, Any]", extra),
                )
                agent = PydanticAIAgent(config=config)
                assert agent.name == "t"

    def test_get_api_key_priority(self):
        agent = _make_agent(api_key="config-key")
        with patch.dict(os.environ, {"TEST_KEY": "env-key"}):
            assert agent._get_api_key(agent.config, "TEST_KEY") == "config-key"

        agent2 = _make_agent()
        with patch.dict(os.environ, {"MY_KEY": "from-env"}):
            assert agent2._get_api_key(agent2.config, "MY_KEY") == "from-env"


def contextlib_exitstack(patches):
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        stack = contextlib.ExitStack()
        with stack:
            for p in patches:
                stack.enter_context(p)
            yield

    return _ctx()
