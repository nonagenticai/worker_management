import json
import os
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# WorkerCreationConfig / WorkerConfig / AgentConfig / MCPConfig models
# ---------------------------------------------------------------------------


class TestPydanticConfigs:
    def test_worker_creation_config_minimal(self):
        from src.worker.worker_factory import WorkerCreationConfig

        cfg = WorkerCreationConfig(worker_name="w1", tool_names=["t1"])
        assert cfg.worker_name == "w1"
        assert cfg.tool_names == ["t1"]
        assert cfg.agent_type is None
        assert cfg.temperature is None

    def test_worker_creation_config_full(self):
        from src.worker.worker_factory import WorkerCreationConfig

        cfg = WorkerCreationConfig(
            worker_name="w",
            tool_names=[],
            agent_role="coder",
            agent_type="claude_code",
            model_provider="anthropic",
            model="sonnet",
            temperature=0.5,
            instruction="do stuff",
            system_prompt="sys",
            user_prompt="usr",
            description="desc",
            deps=["dep1"],
            mcp_config={"mcpServers": {}},
            mcp_config_preset="github",
            mcp_servers=["s1"],
            base_url_override="http://x",
            api_key_override="key",
            tools_dir_override="/tools",
        )
        assert cfg.model_provider == "anthropic"
        assert cfg.mcp_servers == ["s1"]

    def test_worker_creation_config_ignores_extra_fields(self):
        from src.worker.worker_factory import WorkerCreationConfig

        cfg = WorkerCreationConfig(
            worker_name="w",
            tool_names=[],
            **cast("dict[str, Any]", {"bogus_field": "nope"}),
        )
        assert cfg.worker_name == "w"
        # Extra fields should not become model fields
        assert "bogus_field" not in type(cfg).model_fields

    def test_worker_config_with_agent_config(self):
        from src.worker.worker_factory import AgentConfig, WorkerConfig

        ac = AgentConfig(agent_role="coder", agent_type="codex", model="gpt-4o")
        wc = WorkerConfig(worker_name="w", tool_names=["t"], agent_config=ac)
        assert wc.agent_config is not None
        assert wc.agent_config.agent_type == "codex"

    def test_mcp_config_model(self):
        from src.worker.worker_factory import MCPConfig

        mc = MCPConfig(mcp_config={"mcpServers": {}}, mcp_config_preset="gh", mcp_servers=["s"])
        assert mc.mcp_config_preset == "gh"

    def test_agent_config_defaults(self):
        from src.worker.worker_factory import AgentConfig

        ac = AgentConfig()
        assert ac.agent_role is None
        assert ac.agent_type is None
        assert ac.temperature is None


# ---------------------------------------------------------------------------
# _setup_worker_database
# ---------------------------------------------------------------------------


class TestSetupWorkerDatabase:
    def test_creates_db_file(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _setup_worker_database

        monkeypatch.chdir(tmp_path)
        path = _setup_worker_database("myworker")
        assert path == "./myworker_state.sqlite"
        assert (tmp_path / "myworker_state.sqlite").exists()

    def test_skips_existing_file(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _setup_worker_database

        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "myworker_state.sqlite"
        existing.write_text("existing")
        path = _setup_worker_database("myworker")
        assert path == "./myworker_state.sqlite"
        assert existing.read_text() == "existing"

    def test_disabled_returns_empty(self):
        from src.worker.worker_factory import _setup_worker_database

        assert _setup_worker_database("x", disable_state_storage=True) == ""

    def test_os_error_logs_but_returns_path(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _setup_worker_database

        monkeypatch.chdir(tmp_path)
        with patch("builtins.open", side_effect=OSError("disk full")):
            with patch("os.path.exists", return_value=False):
                path = _setup_worker_database("fail")
                assert path == "./fail_state.sqlite"


# ---------------------------------------------------------------------------
# _create_worker_configs
# ---------------------------------------------------------------------------


class TestCreateWorkerConfigs:
    def test_returns_correct_types(self):
        from src.worker.worker_factory import AgentConfig, WorkerConfig, _create_worker_configs

        wc, ac = _create_worker_configs(
            worker_name="w",
            tool_names=["t"],
            agent_role="coder",
            agent_type="claude_code",
            model_provider="anthropic",
            model="sonnet",
            temperature=0.3,
            instruction="inst",
            system_prompt="sys",
            user_prompt="usr",
            description="d",
            deps=["dep"],
            base_url_override="http://x",
            api_key_override="key",
            tools_dir_override="/t",
        )
        assert isinstance(wc, WorkerConfig)
        assert isinstance(ac, AgentConfig)
        assert wc.worker_name == "w"
        assert ac.agent_type == "claude_code"
        assert ac.temperature == 0.3
        assert wc.base_url_override == "http://x"

    def test_defaults_agent_type_to_pydantic_ai(self):
        from src.worker.worker_factory import _create_worker_configs

        _, ac = _create_worker_configs(
            worker_name="w",
            tool_names=[],
            agent_role="r",
            agent_type=None,
            model_provider=None,
            model=None,
            temperature=None,
            instruction=None,
            system_prompt=None,
            user_prompt=None,
            description=None,
            deps=None,
            base_url_override=None,
            api_key_override=None,
            tools_dir_override=None,
        )
        assert ac.agent_type == "pydantic_ai"


# ---------------------------------------------------------------------------
# resolve_mcp_config
# ---------------------------------------------------------------------------


class TestResolveMcpConfig:
    @pytest.mark.asyncio
    async def test_direct_dict_config_takes_priority(self):
        from src.worker.worker_factory import resolve_mcp_config

        cfg = {
            "mcpServers": {
                "s1": {"url": "http://inference-router.inference-router.svc.cluster.local:8150/sse"}
            }
        }
        result = await resolve_mcp_config(mcp_config=cfg, mcp_config_preset="gh", mcp_servers=["s"])
        assert result == cfg

    @pytest.mark.asyncio
    async def test_direct_string_config_takes_priority(self):
        from src.security.request_overrides import McpConfigNotAllowedError
        from src.worker.worker_factory import resolve_mcp_config

        # A path is never opened: file-path MCP configs are refused (fail-closed).
        with pytest.raises(McpConfigNotAllowedError):
            await resolve_mcp_config(mcp_config="/path/to/config.json")

    @pytest.mark.asyncio
    async def test_preset_resolves_when_file_exists(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import resolve_mcp_config

        monkeypatch.chdir(tmp_path)
        mcp_dir = tmp_path / "mcp_configs"
        mcp_dir.mkdir()
        cfg_file = mcp_dir / "test_agent_config.json"
        cfg_file.write_text("{}")
        result = await resolve_mcp_config(mcp_config_preset="test", worker_name="w")
        assert result == "./mcp_configs/test_agent_config.json"

    @pytest.mark.asyncio
    async def test_preset_returns_none_when_missing(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import resolve_mcp_config

        monkeypatch.chdir(tmp_path)
        result = await resolve_mcp_config(mcp_config_preset="nonexistent", worker_name="w")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_dict_falls_through(self):
        from src.worker.worker_factory import resolve_mcp_config

        result = await resolve_mcp_config(mcp_config={})
        assert result is None

    @pytest.mark.asyncio
    async def test_all_none_returns_none(self):
        from src.worker.worker_factory import resolve_mcp_config

        result = await resolve_mcp_config()
        assert result is None


# ---------------------------------------------------------------------------
# _handle_server_filtering
# ---------------------------------------------------------------------------


class TestHandleServerFiltering:
    def test_filters_existing_servers(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _handle_server_filtering

        monkeypatch.chdir(tmp_path)
        mcp_dir = tmp_path / "mcp_configs"
        mcp_dir.mkdir()
        cfg = {"mcpServers": {"s1": {"url": "http://a"}, "s2": {"url": "http://b"}}}
        (mcp_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        result = _handle_server_filtering(["s1"], "w")
        assert result is not None
        assert "s1" in result["mcpServers"]
        assert "s2" not in result["mcpServers"]

    def test_returns_none_when_no_config_files(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _handle_server_filtering

        monkeypatch.chdir(tmp_path)
        result = _handle_server_filtering(["s1"], "w")
        assert result is None

    def test_returns_none_when_no_servers_found(self, tmp_path, monkeypatch):
        from src.worker.worker_factory import _handle_server_filtering

        monkeypatch.chdir(tmp_path)
        mcp_dir = tmp_path / "mcp_configs"
        mcp_dir.mkdir()
        cfg = {"mcpServers": {"s1": {"url": "http://a"}}}
        (mcp_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        result = _handle_server_filtering(["nonexistent"], "w")
        assert result is None


# ---------------------------------------------------------------------------
# _normalize_mcp_config_for_environment
# ---------------------------------------------------------------------------


class TestNormalizeMcpConfig:
    def test_converts_kanban_to_http(self, monkeypatch):
        from src.worker.worker_factory import _normalize_mcp_config_for_environment

        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:8080/sse")
        cfg = {
            "mcpServers": {
                "kanban": {
                    "command": "node",
                    "args": ["/path/to/kanban-mcp/index.js"],
                }
            }
        }
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["url"] == "http://kanban:8080/sse"
        assert "command" not in result["mcpServers"]["kanban"]

    def test_skips_without_env_var(self):
        from src.worker.worker_factory import _normalize_mcp_config_for_environment

        cfg = {"mcpServers": {"s": {"command": "node", "args": ["kanban-mcp"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["s"]["command"] == "node"

    def test_skips_url_based_configs(self, monkeypatch):
        from src.worker.worker_factory import _normalize_mcp_config_for_environment

        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:8080")
        cfg = {"mcpServers": {"s": {"url": "http://existing"}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["s"]["url"] == "http://existing"

    def test_handles_non_dict_servers(self):
        from src.worker.worker_factory import _normalize_mcp_config_for_environment

        result = _normalize_mcp_config_for_environment(
            {"mcpServers": "bad"}, running_in_docker=True
        )
        assert result == {"mcpServers": "bad"}

    def test_invalid_timeout_uses_default(self, monkeypatch):
        from src.worker.worker_factory import _normalize_mcp_config_for_environment

        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:8080")
        monkeypatch.setenv("KANBAN_MCP_TIMEOUT", "invalid")
        cfg = {"mcpServers": {"k": {"command": "node", "args": ["kanban-mcp"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["k"]["timeout"] == 30.0


# ---------------------------------------------------------------------------
# ClaudeCodeAgent
# ---------------------------------------------------------------------------


class TestClaudeCodeAgent:
    def _make_agent(self, **overrides):
        from src.worker.agents.claude_code_agent import ClaudeCodeAgent, ClaudeCodeAgentConfig

        defaults: dict[str, Any] = {"name": "test_claude", "db_path": "/tmp/test.db"}
        defaults.update(overrides)
        config = ClaudeCodeAgentConfig(**defaults)
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0")):
            return ClaudeCodeAgent(config=config)

    def test_build_command_basic(self):
        agent = self._make_agent()
        cmd = agent._build_command("hello")
        assert cmd[0] == agent.config.claude_code_path
        assert "-p" in cmd
        assert "hello" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_build_command_with_model_and_system_prompt(self):
        agent = self._make_agent(model="opus", system_prompt="be helpful")
        cmd = agent._build_command("task")
        assert "--model" in cmd
        assert "opus" in cmd
        assert "--system-prompt" in cmd
        assert "be helpful" in cmd

    def test_parse_output_json(self):
        agent = self._make_agent()
        output = json.dumps(
            {
                "type": "result",
                "result": "done",
                "session_id": "abc",
                "total_cost_usd": 0.05,
                "duration_ms": 1000,
                "is_error": False,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        result = agent._parse_output(output)
        assert result.output == "done"
        assert result.session_id == "abc"
        assert result.cost_usd == 0.05
        assert result.usage_metrics.prompt_tokens == 100
        assert result.usage_metrics.total_tokens == 150

    def test_parse_output_invalid_json_returns_raw(self):
        agent = self._make_agent()
        result = agent._parse_output("not json at all")
        assert result.output == "not json at all"
        assert result.usage_metrics is None

    def test_build_context_aware_prompt(self):
        agent = self._make_agent()
        prompt = agent._build_context_aware_prompt(
            "do thing",
            {"session_id": "s1", "completed_nodes": ["n1", "n2"], "variables": {"x": 1}},
            {"retry": True},
        )
        assert "do thing" in prompt
        assert "s1" in prompt
        assert "n1, n2" in prompt

    def test_proxy_mode_skips_cli_verification(self, monkeypatch):
        from src.worker.agents.claude_code_agent import ClaudeCodeAgent, ClaudeCodeAgentConfig

        monkeypatch.setenv("CLI_PROXY_ENABLED", "true")
        monkeypatch.setenv("CLAUDE_CODE_PROXY_URL", "http://proxy:9000")
        config = ClaudeCodeAgentConfig(name="proxy_test", db_path="/tmp/t.db")
        agent = ClaudeCodeAgent(config=config)
        assert agent.config.use_proxy is True
        assert agent.config.proxy_url == "http://proxy:9000"

    def test_k8s_mode_from_env(self, monkeypatch):
        from src.worker.agents.claude_code_agent import ClaudeCodeAgent, ClaudeCodeAgentConfig

        monkeypatch.setenv("USE_K8S_EXECUTOR", "true")
        monkeypatch.setenv("EXECUTOR_NAMESPACE", "test-ns")
        config = ClaudeCodeAgentConfig(name="k8s_test", db_path="/tmp/t.db")
        agent = ClaudeCodeAgent(config=config)
        assert agent.config.use_k8s is True
        assert agent.config.k8s_namespace == "test-ns"

    def test_agent_host_mode_from_env(self, monkeypatch):
        from src.worker.agents.claude_code_agent import ClaudeCodeAgent, ClaudeCodeAgentConfig

        monkeypatch.setenv("USE_AGENT_HOST", "true")
        config = ClaudeCodeAgentConfig(name="ah_test", db_path="/tmp/t.db")
        agent = ClaudeCodeAgent(config=config)
        assert agent.config.use_agent_host is True

    def test_verify_cli_not_found_raises(self):
        from src.worker.agents.claude_code_agent import ClaudeCodeAgent, ClaudeCodeAgentConfig

        with patch("shutil.which", return_value=None):
            with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
                config = ClaudeCodeAgentConfig(name="test", db_path="/tmp/t.db")
                with pytest.raises(FileNotFoundError, match="not found"):
                    ClaudeCodeAgent(config=config)


# ---------------------------------------------------------------------------
# CodexAgent
# ---------------------------------------------------------------------------


class TestCodexAgent:
    def _make_agent(self, **overrides):
        from src.worker.agents.codex_agent import CodexAgent, CodexAgentConfig

        defaults: dict[str, Any] = {"name": "test_codex", "db_path": "/tmp/test.db"}
        defaults.update(overrides)
        config = CodexAgentConfig(**defaults)
        with patch("shutil.which", return_value="/usr/bin/codex"):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0")):
                return CodexAgent(config=config)

    def test_build_command_basic(self):
        agent = self._make_agent()
        cmd = agent._build_command("hello")
        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "hello" in cmd[-1]

    def test_build_command_prepends_system_prompt(self):
        agent = self._make_agent(system_prompt="be careful")
        cmd = agent._build_command("task")
        assert "be careful" in cmd[-1]
        assert "task" in cmd[-1]

    def test_build_command_with_model(self):
        agent = self._make_agent(model="gpt-4o")
        cmd = agent._build_command("x")
        assert "--model" in cmd
        assert "gpt-4o" in cmd

    def test_parse_output_json(self):
        agent = self._make_agent()
        result = agent._parse_output('{"key": "value"}')
        assert result.output == {"key": "value"}

    def test_parse_output_plain_text(self):
        agent = self._make_agent()
        result = agent._parse_output("plain text output")
        assert result.output == "plain text output"

    def test_proxy_mode_from_env(self, monkeypatch):
        from src.worker.agents.codex_agent import CodexAgent, CodexAgentConfig

        monkeypatch.setenv("CLI_PROXY_ENABLED", "true")
        monkeypatch.setenv("CLI_PROXY_URL", "http://proxy:9000")
        config = CodexAgentConfig(name="proxy_test", db_path="/tmp/t.db")
        agent = CodexAgent(config=config)
        assert agent.config.use_proxy is True
        assert agent.config.proxy_url == "http://proxy:9000"

    def test_k8s_mode_from_env(self, monkeypatch):
        from src.worker.agents.codex_agent import CodexAgent, CodexAgentConfig

        monkeypatch.setenv("USE_K8S_EXECUTOR", "true")
        monkeypatch.setenv("EXECUTOR_NAMESPACE", "codex-ns")
        config = CodexAgentConfig(name="k8s_test", db_path="/tmp/t.db")
        agent = CodexAgent(config=config)
        assert agent.config.use_k8s is True

    def test_build_context_aware_prompt(self):
        agent = self._make_agent()
        prompt = agent._build_context_aware_prompt(
            "analyze",
            {"session_id": "s2", "completed_nodes": ["a"]},
            {},
        )
        assert "analyze" in prompt
        assert "s2" in prompt

    def test_verify_codex_not_found_raises(self):
        from src.worker.agents.codex_agent import CodexAgent, CodexAgentConfig

        with patch("shutil.which", return_value=None):
            with patch("subprocess.run", side_effect=FileNotFoundError("no codex")):
                config = CodexAgentConfig(name="test", db_path="/tmp/t.db")
                with pytest.raises(FileNotFoundError, match="not found"):
                    CodexAgent(config=config)


# ---------------------------------------------------------------------------
# create_claude_code_agent / create_codex_agent factory functions
# ---------------------------------------------------------------------------


class TestAgentFactoryFunctions:
    def test_create_claude_code_agent_combines_prompts(self):
        from src.worker.agents.claude_code_agent import create_claude_code_agent

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0")):
                agent = create_claude_code_agent(
                    name="test",
                    description="test agent",
                    db_path="/tmp/t.db",
                    system_prompt="base prompt",
                    instruction="extra instruction",
                )
        assert agent.config.system_prompt is not None
        assert "base prompt" in agent.config.system_prompt
        assert "extra instruction" in agent.config.system_prompt

    def test_create_codex_agent_combines_prompts(self):
        from src.worker.agents.codex_agent import create_codex_agent

        with patch("shutil.which", return_value="/usr/bin/codex"):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0")):
                agent = create_codex_agent(
                    name="test",
                    description="test agent",
                    db_path="/tmp/t.db",
                    system_prompt="sys",
                    instruction="inst",
                )
        assert agent.config.system_prompt is not None
        assert "sys" in agent.config.system_prompt
        assert "inst" in agent.config.system_prompt

    def test_create_claude_code_agent_default_prompt(self):
        from src.worker.agents.claude_code_agent import create_claude_code_agent

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="1.0")):
                agent = create_claude_code_agent(
                    name="test",
                    description="test agent",
                    db_path="/tmp/t.db",
                )
        assert agent.config.system_prompt is not None
        assert "helpful" in agent.config.system_prompt


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_agent_run_result_model(self):
        from src.worker.agents.base_agent import AgentRunResult, TokenUsageMetrics

        usage = TokenUsageMetrics(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = AgentRunResult(output="ok", usage_metrics=usage, session_id="s1")
        assert result.output == "ok"
        assert result.usage_metrics is not None
        assert result.usage_metrics.total_tokens == 30

    def test_base_agent_config_validation(self):
        from src.worker.agents.base_agent import BaseAgentConfig

        cfg = BaseAgentConfig(name="test", db_path="/tmp/t.db")
        assert cfg.system_prompt == "You are a helpful assistant."

        with pytest.raises(ValidationError):
            BaseAgentConfig(name="", db_path="/tmp/t.db")

    def test_get_agent_type(self):
        from src.worker.agents.base_agent import BaseAgent

        class MyTestAgent(BaseAgent):
            async def run(self, prompt):
                pass

            async def run_with_context(self, enhanced_context):
                pass

            async def check_health(self):
                return True

            async def cleanup(self):
                pass

        agent = MyTestAgent()
        assert agent.get_agent_type() == "mytest"
        assert isinstance(agent.agent_id, UUID)


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_create(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        factory = MagicMock(return_value="agent_instance")
        registry.register("test_type", factory)
        assert registry.is_registered("test_type")
        result = registry.create("test_type", name="x")
        assert result == "agent_instance"

    def test_create_unknown_type_raises(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        with pytest.raises(ValueError, match="Unknown agent type"):
            registry.create("nonexistent")

    def test_unregister(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        registry.register("t", MagicMock())
        registry.unregister("t")
        assert not registry.is_registered("t")

    def test_unregister_nonexistent_raises(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        with pytest.raises(KeyError):
            registry.unregister("nope")

    def test_list_agent_types(self):
        from src.worker.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        registry.register("a", MagicMock())
        registry.register("b", MagicMock())
        assert sorted(registry.list_agent_types()) == ["a", "b"]

    def test_get_agent_class(self):
        from src.worker.agents.agent_registry import AgentRegistry
        from src.worker.agents.base_agent import BaseAgent

        registry = AgentRegistry()
        registry.register("t", MagicMock(), agent_class=BaseAgent)
        assert registry.get_agent_class("t") is BaseAgent
        assert registry.get_agent_class("missing") is None


# ---------------------------------------------------------------------------
# tool_conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    def _make_tool_json(self):
        return {
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
                "required": ["message"],
            },
            "implementation": "async def impl(ctx, **kwargs):\n    return {'message': f'Hi {kwargs[\"name\"]}'}",
        }

    def test_parse_json_to_tools(self):
        from src.worker.agents.tool_conversion import parse_json_to_tools

        tools = parse_json_to_tools([self._make_tool_json()])
        assert len(tools) == 1
        assert tools[0].name == "greet"

    def test_parse_json_to_tools_invalid_schema(self):
        from src.worker.agents.tool_conversion import parse_json_to_tools

        bad = {"name": "t", "description": "d"}
        with pytest.raises(ValidationError):
            parse_json_to_tools([bad])

    def test_load_local_tool_nonexistent(self):
        from src.worker.agents.tool_conversion import load_local_tool

        result = load_local_tool("/nonexistent/path.py")
        assert result is None

    def test_load_tools_from_nonexistent_directory(self):
        from src.worker.agents.tool_conversion import load_tools_from_directory

        result = load_tools_from_directory("/nonexistent/dir")
        assert result == []


# ---------------------------------------------------------------------------
# ToolSchema / SchemaField
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_schema_field_valid(self):
        from src.worker.agents.schemas import SchemaField

        field = SchemaField(type="string", description="a name")
        assert field.type == "string"

    def test_schema_field_rejects_non_dict(self):
        from src.worker.agents.schemas import SchemaField

        with pytest.raises(ValidationError):
            SchemaField.model_validate([1, 2, 3])

    def test_schema_field_requires_string_type(self):
        from src.worker.agents.schemas import SchemaField

        with pytest.raises(ValidationError):
            SchemaField.model_validate({"type": 123})

    def test_tool_schema_validates_implementation(self):
        from src.worker.agents.schemas import ToolSchema

        ts = ToolSchema(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            implementation="x = 1",
        )
        assert ts.implementation == "x = 1"

    def test_tool_schema_rejects_invalid_code(self):
        from src.worker.agents.schemas import ToolSchema

        with pytest.raises(ValidationError, match="Invalid Python"):
            ToolSchema(
                name="t",
                description="d",
                input_schema={},
                output_schema={},
                implementation="def (broken",
            )


# ---------------------------------------------------------------------------
# K8sJobExecutor
# ---------------------------------------------------------------------------


class TestK8sJobExecutor:
    def test_build_job_manifest_structure(self):
        executor = self._make_executor()
        manifest = executor._build_job_manifest(
            agent_name="test-agent",
            prompt="do something",
            execution_id="abc12345-1234-1234-1234-123456789abc",
            config={"ttl_seconds": 600},
        )
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["kind"] == "Job"
        assert manifest["metadata"]["name"] == "claude-agent-abc12345"
        assert manifest["metadata"]["namespace"] == "test-ns"
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "claude-executor"
        env_names = [e["name"] for e in container["env"]]
        assert "AGENT_PROMPT" in env_names
        assert "EXECUTION_ID" in env_names

    def test_build_job_manifest_debug_mode(self):
        executor = self._make_executor()
        manifest = executor._build_job_manifest(
            agent_name="test",
            prompt="debug",
            execution_id="abc12345",
            config={"debug": True},
        )
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_build_job_manifest_image_override(self):
        executor = self._make_executor()
        manifest = executor._build_job_manifest(
            agent_name="test",
            prompt="x",
            execution_id="abc12345",
            config={},
            image_override="custom:latest",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "custom:latest"

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self):
        executor = self._make_executor()
        assert executor.batch_v1 is None
        with pytest.raises(RuntimeError, match="not initialized"):
            await executor._wait_for_completion("j", 1)

    def _make_executor(self):
        from src.k8s_job_executor import K8sJobExecutor

        return K8sJobExecutor(namespace="test-ns")


# ---------------------------------------------------------------------------
# db_config
# ---------------------------------------------------------------------------


class TestDbConfig:
    def test_get_postgres_config_from_url(self, monkeypatch):
        from src.db.db_config import get_postgres_config

        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5433/mydb")
        cfg = get_postgres_config()
        assert cfg["host"] == "host"
        assert cfg["port"] == 5433
        assert cfg["database"] == "mydb"
        assert cfg["user"] == "user"
        assert cfg["password"] == "pass"

    def test_get_postgres_config_missing_url(self, monkeypatch):
        from src.db.db_config import get_postgres_config

        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL"):
            get_postgres_config()

    def test_get_postgres_config_defaults(self, monkeypatch):
        from src.db.db_config import get_postgres_config

        monkeypatch.setenv("DATABASE_URL", "postgresql:///")
        cfg = get_postgres_config()
        assert cfg["host"] == "localhost"
        assert cfg["port"] == 5432

    def test_get_pg_pool_returns_none_initially(self):
        from src.db import db_config

        original = db_config._pg_pool
        db_config._pg_pool = None
        try:
            assert db_config.get_pg_pool() is None
        finally:
            db_config._pg_pool = original


# ---------------------------------------------------------------------------
# AgentMetrics
# ---------------------------------------------------------------------------


class TestAgentMetrics:
    def test_disabled_client_records_nothing(self):
        from src.telemetry.agent_metrics import AgentMetricsClient

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
        assert not client.enabled
        client.record_execution("a", "t", success=True)
        client.record_agent_start("a", "t")
        client.record_agent_end("a", "t")

    def test_execution_timer_context_manager(self):
        from src.telemetry.agent_metrics import AgentExecutionTimer

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            with AgentExecutionTimer("agent", "type", "provider", "model") as timer:
                assert timer.start_time is not None


# ---------------------------------------------------------------------------
# _load_worker_tools
# ---------------------------------------------------------------------------


class TestLoadWorkerTools:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        from src.worker.worker_factory import _load_worker_tools

        result = await _load_worker_tools([], "w")
        assert result == []


# ---------------------------------------------------------------------------
# agents/__init__ - verify imports don't crash
# ---------------------------------------------------------------------------


class TestAgentsInit:
    def test_base_types_importable(self):
        from src.worker.agents import AgentRunResult, BaseAgent, TokenUsageMetrics

        assert BaseAgent is not None
        assert AgentRunResult is not None
        assert TokenUsageMetrics is not None

    def test_registry_functions_importable(self):
        from src.worker.agents import (
            create_agent,
            list_available_agent_types,
            register_agent,
        )

        assert callable(create_agent)
        assert callable(register_agent)
        types = list_available_agent_types()
        assert isinstance(types, list)

    def test_claude_code_registered(self):
        from src.worker.agents import list_available_agent_types

        types = list_available_agent_types()
        assert "claude_code" in types

    def test_codex_registered(self):
        from src.worker.agents import list_available_agent_types

        types = list_available_agent_types()
        assert "codex" in types
