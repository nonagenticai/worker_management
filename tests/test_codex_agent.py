"""Tests for Codex agent — proxy, direct, k8s modes."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.worker.agents.codex_agent import (
    CodexAgent,
    CodexAgentConfig,
    create_codex_agent,
)


class TestCodexAgentConfig:
    def test_default_config(self):
        config = CodexAgentConfig(name="test", db_path="test.db")
        assert config.use_proxy is False
        assert config.use_k8s is False
        assert config.k8s_namespace == "agent-jobs"
        assert config.timeout == 300

    def test_k8s_config(self):
        config = CodexAgentConfig(
            name="test",
            db_path="test.db",
            use_k8s=True,
            k8s_namespace="custom-ns",
            k8s_executor_image="registry.example.com:5000/codex-executor:latest",
        )
        assert config.use_k8s is True
        assert config.k8s_namespace == "custom-ns"
        assert config.k8s_executor_image == "registry.example.com:5000/codex-executor:latest"

    def test_proxy_config(self):
        config = CodexAgentConfig(
            name="test",
            db_path="test.db",
            use_proxy=True,
            proxy_url="http://proxy:9000",
        )
        assert config.use_proxy is True
        assert config.proxy_url == "http://proxy:9000"


class TestK8sModeInit:
    def test_k8s_mode_from_env(self):
        with patch.dict(
            os.environ,
            {
                "USE_K8S_EXECUTOR": "true",
                "EXECUTOR_NAMESPACE": "my-namespace",
                "K8S_CODEX_EXECUTOR_IMAGE": "registry.example.com:5000/codex-executor:v2",
            },
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_k8s is True
            assert agent.config.k8s_namespace == "my-namespace"
            assert agent.config.k8s_executor_image == "registry.example.com:5000/codex-executor:v2"

    def test_k8s_mode_overrides_proxy(self):
        """K8s mode should take precedence over proxy mode."""
        with patch.dict(
            os.environ,
            {
                "USE_K8S_EXECUTOR": "true",
                "CLI_PROXY_ENABLED": "true",
                "CLI_PROXY_URL": "http://proxy:9000",
            },
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_k8s is True
            assert agent.config.use_proxy is False

    def test_k8s_mode_skips_cli_verification(self):
        with patch.dict(os.environ, {"USE_K8S_EXECUTOR": "true"}):
            with patch.object(CodexAgent, "_verify_cli_available") as mock_verify:
                config = CodexAgentConfig(name="test", db_path="test.db")
                CodexAgent(config=config)
                mock_verify.assert_not_called()


class TestProxyModeInit:
    def test_proxy_mode_from_env(self):
        with patch.dict(
            os.environ,
            {
                "CLI_PROXY_ENABLED": "true",
                "CLI_PROXY_URL": "http://proxy:9000",
            },
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_proxy is True
            assert agent.config.proxy_url == "http://proxy:9000"

    def test_proxy_mode_skips_cli_verification(self):
        with (
            patch.dict(
                os.environ,
                {
                    "CLI_PROXY_ENABLED": "true",
                    "CLI_PROXY_URL": "http://proxy:9000",
                },
            ),
            patch.object(CodexAgent, "_verify_cli_available") as mock_verify,
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            CodexAgent(config=config)
            mock_verify.assert_not_called()


class TestDirectModeInit:
    def test_direct_mode_verifies_cli(self):
        with patch.object(CodexAgent, "_verify_cli_available") as mock_verify:
            config = CodexAgentConfig(name="test", db_path="test.db")
            CodexAgent(config=config)
            mock_verify.assert_called_once()


class TestBuildCommand:
    def test_basic_command(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            cmd = agent._build_command("Hello")
            assert cmd[0] == "codex"
            assert "exec" in cmd
            # System prompt prepended to prompt
            assert any("Hello" in arg for arg in cmd)

    def test_command_with_model(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db", model="gpt-4o")
            agent = CodexAgent(config=config)
            cmd = agent._build_command("Hello")
            assert "--model" in cmd
            assert "gpt-4o" in cmd

    def test_system_prompt_prepended(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db", system_prompt="Be concise")
            agent = CodexAgent(config=config)
            cmd = agent._build_command("What is 2+2?")
            # Last arg should have both system prompt and user prompt
            combined = cmd[-1]
            assert "Be concise" in combined
            assert "What is 2+2?" in combined


class TestRunModes:
    @pytest.mark.asyncio
    async def test_run_k8s_mode(self):
        with patch.dict(os.environ, {"USE_K8S_EXECUTOR": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch.object(agent, "_execute_via_k8s", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = json.dumps({"result": "K8s output"})
                result = await agent.run("test prompt")
                mock_exec.assert_called_once_with("test prompt")
                assert result.output == {"result": "K8s output"}

    @pytest.mark.asyncio
    async def test_run_proxy_mode(self):
        with patch.dict(
            os.environ,
            {
                "CLI_PROXY_ENABLED": "true",
                "CLI_PROXY_URL": "http://proxy:9000",
            },
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch.object(agent, "_execute_via_proxy", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = "Proxy output"
                result = await agent.run("test prompt")
                mock_exec.assert_called_once_with("test prompt")
                assert result.output == "Proxy output"

    @pytest.mark.asyncio
    async def test_run_direct_mode(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch.object(agent, "_execute_direct", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = "Direct output"
                result = await agent.run("test prompt")
                mock_exec.assert_called_once()
                assert result.output == "Direct output"


class TestParseOutput:
    def test_parse_json_output(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            output = json.dumps({"result": "Hello", "status": "ok"})
            result = agent._parse_output(output)
            assert result.output == {"result": "Hello", "status": "ok"}

    def test_parse_non_json_output(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            result = agent._parse_output("Plain text output")
            assert result.output == "Plain text output"
            assert result.usage_metrics is None


class TestK8sExecution:
    @pytest.mark.asyncio
    async def test_execute_via_k8s_dispatches(self):
        """K8s mode should dispatch run() to _execute_via_k8s."""
        with patch.dict(os.environ, {"USE_K8S_EXECUTOR": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch.object(agent, "_execute_via_k8s", new_callable=AsyncMock) as mock_k8s:
                mock_k8s.return_value = json.dumps({"type": "result", "result": "Answer is 4"})
                await agent.run("What is 2+2?")
                mock_k8s.assert_called_once_with("What is 2+2?")

    @pytest.mark.asyncio
    async def test_execute_via_k8s_import_error(self):
        """Missing kubernetes package should raise RuntimeError."""
        with patch.dict(os.environ, {"USE_K8S_EXECUTOR": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch.dict("sys.modules", {"src.k8s_job_executor": None}):
                with pytest.raises(RuntimeError, match="execution failed"):
                    await agent.run("test")


class TestCreateCodexAgent:
    def test_factory_function(self):
        with patch.object(CodexAgent, "_verify_cli_available"):
            agent = create_codex_agent(
                name="test",
                description="Test agent",
                db_path="test.db",
                system_prompt="Be helpful",
                instruction="Additional instructions",
            )
            assert agent.name == "test"
            assert agent.config.system_prompt is not None
            assert "Be helpful" in agent.config.system_prompt
            assert "Additional instructions" in agent.config.system_prompt

    def test_factory_with_kwargs(self):
        """Extra kwargs should be ignored gracefully."""
        with patch.object(CodexAgent, "_verify_cli_available"):
            agent = create_codex_agent(
                name="test",
                description="Test",
                db_path="test.db",
                model_provider="openai",
                tools=[],
                base_url="http://test",
            )
            assert agent.name == "test"


# ==============================================================================
# Agent-Host Mode Tests
# ==============================================================================


class TestAgentHostModeConfig:
    def test_config_fields_exist(self):
        config = CodexAgentConfig(name="test", db_path="test.db")
        assert config.use_agent_host is False
        assert config.agent_host_container_id is None
        assert config.agent_host_url is None

    def test_from_env_var(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_agent_host is True


class TestAgentHostModeInitCodex:
    def test_skips_cli_verification(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            with patch.object(CodexAgent, "_verify_cli_available") as mock_verify:
                config = CodexAgentConfig(name="test", db_path="test.db")
                CodexAgent(config=config)
                mock_verify.assert_not_called()

    def test_precedence_over_k8s(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true", "USE_K8S_EXECUTOR": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_agent_host is True
            assert agent.config.use_k8s is False

    def test_precedence_over_proxy(self):
        with patch.dict(
            os.environ,
            {
                "USE_AGENT_HOST": "true",
                "CLI_PROXY_ENABLED": "true",
                "CLI_PROXY_URL": "http://proxy:9000",
            },
        ):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            assert agent.config.use_agent_host is True
            assert agent.config.use_proxy is False


class TestAgentHostExecutionCodex:
    @pytest.mark.asyncio
    async def test_run_dispatches_to_agent_host(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            with patch.object(
                agent, "_execute_via_agent_host", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = json.dumps({"result": "agent-host output"})
                await agent.run("test prompt")
                mock_exec.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_creates_container(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            mock_instance = AsyncMock()
            mock_instance.url = "http://agent-host:9100"
            mock_instance.name = "mini1"

            create_resp = AsyncMock()
            create_resp.status_code = 201
            create_resp.json = MagicMock(
                return_value={
                    "container_id": "codex-ctr",
                    "container_name": "codex-test",
                }
            )

            exec_resp = AsyncMock()
            exec_resp.status_code = 200
            exec_resp.json = MagicMock(
                return_value={
                    "success": True,
                    "exit_code": 0,
                    "stdout": "codex output",
                    "stderr": "",
                }
            )

            mock_client = AsyncMock()
            mock_client.post.side_effect = [create_resp, exec_resp]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with (
                patch("src.services.agent_host_registry.agent_host_registry") as mock_registry,
                patch("httpx.AsyncClient", return_value=mock_client),
            ):
                mock_registry.get_best_instance = AsyncMock(return_value=mock_instance)
                await agent._execute_via_agent_host("hello")

            assert agent.config.agent_host_container_id == "codex-ctr"
            # Verify container_type is "codex"
            create_call = mock_client.post.call_args_list[0]
            assert create_call[1]["json"]["container_type"] == "codex"

    @pytest.mark.asyncio
    async def test_success_returns_stdout(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "success": True,
                    "exit_code": 0,
                    "stdout": "codex out",
                    "stderr": "",
                }
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await agent._execute_via_agent_host("test")
            assert result == "codex out"

    @pytest.mark.asyncio
    async def test_failure_includes_stdout_and_stderr(self):
        """T14 regression: error must include both stdout and stderr."""
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "success": False,
                    "exit_code": 1,
                    "stdout": '{"error":"rate_limited"}',
                    "stderr": "rate limit exceeded",
                }
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(RuntimeError) as exc_info:
                    await agent._execute_via_agent_host("test")
            error_msg = str(exc_info.value)
            assert "stdout=" in error_msg
            assert "stderr=" in error_msg
            assert "rate_limited" in error_msg

    @pytest.mark.asyncio
    async def test_failure_stdout_truncated(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "success": False,
                    "exit_code": 1,
                    "stdout": "x" * 1000,
                    "stderr": "",
                }
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(RuntimeError) as exc_info:
                    await agent._execute_via_agent_host("test")
            assert len(str(exc_info.value)) < 700

    @pytest.mark.asyncio
    async def test_no_healthy_instance(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)

            with patch("src.services.agent_host_registry.agent_host_registry") as mock_registry:
                mock_registry.get_best_instance = AsyncMock(return_value=None)
                with pytest.raises(RuntimeError, match="No healthy"):
                    await agent._execute_via_agent_host("test")

    @pytest.mark.asyncio
    async def test_builds_codex_command(self):
        """Verify codex command format: codex exec [--model X] combined_prompt."""
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "success": True,
                    "exit_code": 0,
                    "stdout": "out",
                    "stderr": "",
                }
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                await agent._execute_via_agent_host("What is 2+2?")

            call_json = mock_client.post.call_args[1]["json"]
            cmd = call_json["command"]
            assert cmd[0] == "codex"
            assert cmd[1] == "exec"


class TestAgentHostCleanupCodex:
    @pytest.mark.asyncio
    async def test_destroys_container(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200

            mock_client = AsyncMock()
            mock_client.delete.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                await agent.cleanup()
            mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_clears_container_id(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            agent.config.agent_host_container_id = "ctr-123"
            agent.config.agent_host_url = "http://agent-host:9100"

            resp = AsyncMock()
            resp.status_code = 200

            mock_client = AsyncMock()
            mock_client.delete.return_value = resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                await agent.cleanup()
            assert agent.config.agent_host_container_id is None

    @pytest.mark.asyncio
    async def test_noop_when_no_container(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true"}):
            config = CodexAgentConfig(name="test", db_path="test.db")
            agent = CodexAgent(config=config)
            with patch("httpx.AsyncClient") as mock_client_cls:
                await agent.cleanup()
                mock_client_cls.assert_not_called()
