"""Tests for RemoteCodexAgent — config, run, error handling, health check, factory."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.worker.agents.remote_cli_agent import (
    RemoteCLIAgent as RemoteCodexAgent,
)
from src.worker.agents.remote_cli_agent import (
    RemoteCLIAgentConfig as RemoteCodexAgentConfig,
)
from src.worker.agents.remote_cli_agent import (
    create_remote_codex_agent,
)
from src.worker.agents.remote_container_executor import AgentExecutionResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> RemoteCodexAgentConfig:
    """Return a minimal valid config, merging any overrides."""
    defaults: dict[str, Any] = {
        "cli": "codex",
        "name": "test-codex",
        "db_path": "test.db",
        "container_id": "container-abc123",
    }
    defaults.update(overrides)
    return RemoteCodexAgentConfig(**defaults)


def _patch_executor():
    """Patch RemoteContainerExecutor so __init__ doesn't hit the network."""
    return patch(
        "src.worker.agents.remote_cli_agent.RemoteContainerExecutor",
        autospec=True,
    )


def _patch_timer():
    """Patch AgentExecutionTimer so metrics don't require a real backend."""
    mock_timer = MagicMock()
    mock_timer.__enter__ = MagicMock(return_value=mock_timer)
    mock_timer.__exit__ = MagicMock(return_value=False)
    mock_timer.record_tokens = MagicMock()
    return patch(
        "src.worker.agents.remote_cli_agent.AgentExecutionTimer",
        return_value=mock_timer,
    )


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentConfig
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentConfig:
    def test_defaults(self):
        config = _make_config()
        assert config.container_id == "container-abc123"
        assert config.workspace == "/workspace"
        assert config.timeout == 300
        assert config.model is None
        assert config.system_prompt == "You are a helpful coding assistant."
        assert config.instruction is None
        assert config.description is None

    def test_container_id_required(self):
        with pytest.raises(Exception):
            RemoteCodexAgentConfig(cli="codex", name="test", db_path="test.db")

    def test_custom_values(self):
        config = _make_config(
            workspace="/custom",
            timeout=600,
            model="gpt-5-codex",
            system_prompt="Custom prompt",
        )
        assert config.workspace == "/custom"
        assert config.timeout == 600
        assert config.model == "gpt-5-codex"
        assert config.system_prompt == "Custom prompt"


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentInit
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentInit:
    def test_init_sets_attributes(self):
        with _patch_executor():
            # Make the mock instance return a valid endpoint
            # RemoteExecutionConfig.get_executor_endpoint is called on the config, not the executor
            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            assert agent.name == "test-codex"
            assert agent.db_path == "test.db"
            assert agent.config is config
            assert agent.agent_id is not None

    def test_init_creates_executor(self):
        with _patch_executor() as mock_exec_cls:
            config = _make_config(executor_endpoint="https://coding.example.com")
            RemoteCodexAgent(config=config)
            mock_exec_cls.assert_called_once()

    def test_init_logs_correctly(self, caplog):
        with _patch_executor():
            config = _make_config(executor_endpoint="https://coding.example.com")
            import logging

            with caplog.at_level(logging.INFO):
                RemoteCodexAgent(config=config)
            assert "RemoteCLIAgent[codex]" in caplog.text
            assert "test-codex" in caplog.text
            assert "container-abc123" in caplog.text


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentRun
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentRun:
    @pytest.mark.asyncio
    async def test_run_dispatches_to_executor_with_codex_cli(self):
        """run() should call executor.execute_agent with cli='codex'."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(
                    success=True,
                    output="Task completed successfully",
                    exit_code=0,
                    execution_time=1.5,
                    container_id="container-abc123",
                )
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = await agent.run("Write tests for foo.py")

            mock_executor.execute_agent.assert_called_once_with(
                cli="codex",
                prompt="Write tests for foo.py",
                system_prompt="You are a helpful coding assistant.",
                model=None,
                # Origin attribution: the remote hop is told WHOSE WORK it is,
                # so coding-gateway's downstream spend is attributable to this
                # worker instead of to worker_management's service account.
                worker_name="test-codex",
            )
            assert result.output == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_run_passes_model_when_set(self):
        """run() should forward the model from config."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(success=True, output="done", exit_code=0)
            )

            config = _make_config(
                executor_endpoint="https://coding.example.com",
                model="gpt-5-codex",
            )
            agent = RemoteCodexAgent(config=config)
            await agent.run("test prompt")

            call_kwargs = mock_executor.execute_agent.call_args
            assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "gpt-5-codex"

    @pytest.mark.asyncio
    async def test_run_returns_parsed_json_output(self):
        """When executor returns JSON string, _parse_output should parse it."""
        json_output = json.dumps({"answer": "42", "status": "ok"})
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(success=True, output=json_output, exit_code=0)
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)
            result = await agent.run("prompt")

            assert result.output == {"answer": "42", "status": "ok"}
            assert result.usage_metrics is None


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentRunFailure
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentRunFailure:
    @pytest.mark.asyncio
    async def test_run_raises_on_executor_failure(self):
        """run() should raise RuntimeError when executor returns success=False."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(
                    success=False,
                    output="",
                    error="Container crashed",
                    exit_code=1,
                )
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            with pytest.raises(RuntimeError, match="execution failed"):
                await agent.run("will fail")

    @pytest.mark.asyncio
    async def test_run_raises_on_executor_failure_unknown_error(self):
        """When error is None, the message should say 'Unknown remote execution error'."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(
                    success=False,
                    output="",
                    error=None,
                    exit_code=1,
                )
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            with pytest.raises(RuntimeError, match="Unknown remote execution error"):
                await agent.run("will fail")

    @pytest.mark.asyncio
    async def test_run_wraps_executor_exception(self):
        """Exceptions from the executor should be wrapped in RuntimeError."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                side_effect=ConnectionError("network unreachable")
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            with pytest.raises(RuntimeError, match="execution failed"):
                await agent.run("prompt")


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentRunWithContext
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentRunWithContext:
    @pytest.mark.asyncio
    async def test_run_with_context_builds_enhanced_prompt(self):
        """run_with_context should build a prompt from task + graph context and call run()."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(
                    success=True, output="context result", exit_code=0
                )
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            enhanced_context = {
                "task": "Analyze the codebase",
                "graph_context": {
                    "graph_id": "g-1",
                    "node_id": "n-1",
                    "session_id": "sess-abc",
                    "completed_nodes": ["n-0"],
                    "variables": {"key": "value"},
                },
                "node_config": {"max_retries": 3},
            }

            result = await agent.run_with_context(enhanced_context)

            # Verify the executor was called
            call_args = mock_executor.execute_agent.call_args
            prompt_sent = (
                call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][1]
                if len(call_args[0]) > 1
                else call_args.kwargs["prompt"]
            )

            assert "Analyze the codebase" in prompt_sent
            assert "sess-abc" in prompt_sent
            assert "n-0" in prompt_sent
            assert result.output == "context result"

    @pytest.mark.asyncio
    async def test_run_with_context_empty_context(self):
        """run_with_context should work with minimal/empty context."""
        with _patch_executor() as mock_exec_cls, _patch_timer():
            mock_executor = mock_exec_cls.return_value
            mock_executor.execute_agent = AsyncMock(
                return_value=AgentExecutionResponse(success=True, output="ok", exit_code=0)
            )

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = await agent.run_with_context({"task": "Do something"})
            assert result.output == "ok"


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentParseOutput
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentParseOutput:
    def test_parse_json_string(self):
        with _patch_executor():
            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = agent._parse_output(json.dumps({"key": "value"}))
            assert result.output == {"key": "value"}
            assert result.usage_metrics is None

    def test_parse_plain_text(self):
        with _patch_executor():
            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = agent._parse_output("Plain text response")
            assert result.output == "Plain text response"
            assert result.usage_metrics is None

    def test_parse_json_array(self):
        with _patch_executor():
            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = agent._parse_output(json.dumps([1, 2, 3]))
            assert result.output == [1, 2, 3]

    def test_parse_invalid_json_falls_back_to_text(self):
        with _patch_executor():
            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = agent._parse_output("{not valid json")
            assert result.output == "{not valid json"
            assert result.usage_metrics is None


# ---------------------------------------------------------------------------
# TestRemoteCodexAgentHealthCheck
# ---------------------------------------------------------------------------


class TestRemoteCodexAgentHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_delegates_to_executor(self):
        with _patch_executor() as mock_exec_cls:
            mock_executor = mock_exec_cls.return_value
            mock_executor.check_health = AsyncMock(return_value=True)

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = await agent.check_health()
            assert result is True
            mock_executor.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_failure(self):
        with _patch_executor() as mock_exec_cls:
            mock_executor = mock_exec_cls.return_value
            mock_executor.check_health = AsyncMock(return_value=False)

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = await agent.check_health()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_exception(self):
        with _patch_executor() as mock_exec_cls:
            mock_executor = mock_exec_cls.return_value
            mock_executor.check_health = AsyncMock(side_effect=ConnectionError("down"))

            config = _make_config(executor_endpoint="https://coding.example.com")
            agent = RemoteCodexAgent(config=config)

            result = await agent.check_health()
            assert result is False


# ---------------------------------------------------------------------------
# TestCreateRemoteCodexAgent
# ---------------------------------------------------------------------------


class TestCreateRemoteCodexAgent:
    def test_factory_creates_agent(self):
        with _patch_executor():
            agent = create_remote_codex_agent(
                name="factory-codex",
                description="A test codex agent",
                db_path="test.db",
                container_id="ctr-999",
                executor_endpoint="https://coding.example.com",
            )
            assert isinstance(agent, RemoteCodexAgent)
            assert agent.name == "factory-codex"
            assert agent.config.container_id == "ctr-999"
            assert agent.config.description == "A test codex agent"

    def test_factory_combines_system_prompt_and_instruction(self):
        with _patch_executor():
            agent = create_remote_codex_agent(
                name="factory-codex",
                description="desc",
                db_path="test.db",
                container_id="ctr-1",
                executor_endpoint="https://coding.example.com",
                system_prompt="Be precise.",
                instruction="Focus on tests.",
            )
            assert agent.config.system_prompt is not None
            assert "Be precise." in agent.config.system_prompt
            assert "Focus on tests." in agent.config.system_prompt

    def test_factory_default_system_prompt(self):
        with _patch_executor():
            agent = create_remote_codex_agent(
                name="factory-codex",
                description="desc",
                db_path="test.db",
                container_id="ctr-1",
                executor_endpoint="https://coding.example.com",
            )
            assert agent.config.system_prompt == "You are a helpful coding assistant."

    def test_factory_forwards_model_and_timeout(self):
        with _patch_executor():
            agent = create_remote_codex_agent(
                name="factory-codex",
                description="desc",
                db_path="test.db",
                container_id="ctr-1",
                executor_endpoint="https://coding.example.com",
                model="gpt-5-codex",
                timeout=600,
            )
            assert agent.config.model == "gpt-5-codex"
            assert agent.config.timeout == 600


# ---------------------------------------------------------------------------
# TestCreateRemoteCodexAgentKwargs
# ---------------------------------------------------------------------------


class TestCreateRemoteCodexAgentKwargs:
    def test_extra_kwargs_are_ignored(self):
        """Factory should accept and silently ignore extra keyword arguments."""
        with _patch_executor():
            agent = create_remote_codex_agent(
                name="factory-codex",
                description="desc",
                db_path="test.db",
                container_id="ctr-1",
                executor_endpoint="https://coding.example.com",
                model_provider="openai",
                tools=[],
                base_url="http://test",
                random_kwarg="should be ignored",
            )
            assert isinstance(agent, RemoteCodexAgent)
            assert agent.name == "factory-codex"
