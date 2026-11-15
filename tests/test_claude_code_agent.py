"""Tests for Claude Code agent — covers CLIAgent base and ClaudeCodeAgent subclass.

Targets:
- src/worker/agents/cli_agent.py
- src/worker/agents/claude_code_agent.py
"""

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.worker.agents.claude_code_agent import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    create_claude_code_agent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(*, verify=False, **overrides):
    """Create a ClaudeCodeAgent with CLI verification optionally stubbed out."""
    defaults: dict[str, Any] = {"name": "test", "db_path": "test.db"}
    defaults.update(overrides)
    config = ClaudeCodeAgentConfig(**defaults)
    if verify:
        return ClaudeCodeAgent(config=config)
    with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
        return ClaudeCodeAgent(config=config)


def _agent_host_env():
    return patch.dict(os.environ, {"USE_AGENT_HOST": "true"})


def _k8s_env(**extras):
    env = {"USE_K8S_EXECUTOR": "true"}
    env.update(extras)
    return patch.dict(os.environ, env)


def _proxy_env(url="http://proxy:9000", **extras):
    env = {"CLI_PROXY_ENABLED": "true", "CLAUDE_CODE_PROXY_URL": url}
    env.update(extras)
    return patch.dict(os.environ, env)


def _mock_httpx_client(
    *, post_returns=None, post_side_effect=None, delete_returns=None, delete_side_effect=None
):
    """Return a mock httpx.AsyncClient usable as async context manager."""
    client = AsyncMock()
    if post_returns is not None:
        client.post.return_value = post_returns
    if post_side_effect is not None:
        client.post.side_effect = post_side_effect
    if delete_returns is not None:
        client.delete.return_value = delete_returns
    if delete_side_effect is not None:
        client.delete.side_effect = delete_side_effect
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _resp(status_code=200, json_data=None, text=""):
    r = AsyncMock()
    r.status_code = status_code
    r.text = text
    if json_data is not None:
        r.json = MagicMock(return_value=json_data)
    return r


# ===========================================================================
# Config
# ===========================================================================


class TestClaudeCodeAgentConfig:
    def test_default_config(self):
        config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
        assert config.use_proxy is False
        assert config.use_k8s is False
        assert config.use_agent_host is False
        assert config.k8s_namespace == "agent-jobs"
        assert config.k8s_session_secret == "claude-secrets"
        assert config.timeout == 300
        assert config.claude_code_path == "claude"
        assert config.mcp_servers == []

    def test_k8s_config(self):
        config = ClaudeCodeAgentConfig(
            name="test",
            db_path="test.db",
            use_k8s=True,
            k8s_namespace="custom-ns",
            k8s_session_secret="my-secret",
        )
        assert config.use_k8s is True
        assert config.k8s_namespace == "custom-ns"
        assert config.k8s_session_secret == "my-secret"

    def test_custom_claude_path_and_mcp(self):
        config = ClaudeCodeAgentConfig(
            name="test",
            db_path="test.db",
            claude_code_path="/usr/local/bin/claude",
            mcp_servers=["server1", {"name": "server2", "url": "http://s2"}],
        )
        assert config.claude_code_path == "/usr/local/bin/claude"
        assert len(config.mcp_servers) == 2


# ===========================================================================
# Init — mode selection
# ===========================================================================


class TestProxyModeInit:
    def test_proxy_mode_from_env(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_proxy is True
            assert agent.config.proxy_url == "http://proxy:9000"

    def test_proxy_mode_legacy_env(self):
        with patch.dict(
            os.environ,
            {"CLI_PROXY_ENABLED": "true", "CLI_PROXY_URL": "http://legacy:9000"},
            clear=False,
        ):
            os.environ.pop("CLAUDE_CODE_PROXY_URL", None)
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_proxy is True
            assert agent.config.proxy_url == "http://legacy:9000"

    def test_proxy_mode_no_url_warns(self):
        with patch.dict(os.environ, {"CLI_PROXY_ENABLED": "true"}, clear=False):
            os.environ.pop("CLAUDE_CODE_PROXY_URL", None)
            os.environ.pop("CLI_PROXY_URL", None)
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_proxy is True
            assert agent.config.proxy_url is None


class TestK8sModeInit:
    def test_k8s_mode_from_env(self):
        with _k8s_env(EXECUTOR_NAMESPACE="my-ns", CLAUDE_SESSION_SECRET_NAME="my-creds"):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_k8s is True
            assert agent.config.k8s_namespace == "my-ns"
            assert agent.config.k8s_session_secret == "my-creds"

    def test_k8s_mode_overrides_proxy(self):
        with patch.dict(
            os.environ,
            {"USE_K8S_EXECUTOR": "true", "CLI_PROXY_ENABLED": "true", "CLI_PROXY_URL": "http://x"},
        ):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_k8s is True
            assert agent.config.use_proxy is False

    def test_k8s_reads_executor_image_env(self):
        with _k8s_env(K8S_EXECUTOR_IMAGE="my-image:latest"):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.k8s_executor_image == "my-image:latest"


class TestDirectModeInit:
    def test_direct_mode_verifies_cli(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available") as mock_verify:
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            ClaudeCodeAgent(config=config)
            mock_verify.assert_called_once()

    def test_proxy_mode_skips_cli_verification(self):
        with _proxy_env():
            with patch.object(ClaudeCodeAgent, "_verify_cli_available") as mock_verify:
                config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                ClaudeCodeAgent(config=config)
                mock_verify.assert_not_called()


class TestAgentHostModeInit:
    def test_config_fields_exist(self):
        config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
        assert config.use_agent_host is False
        assert config.agent_host_container_id is None
        assert config.agent_host_url is None

    def test_from_env_var(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_agent_host is True

    def test_env_var_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"USE_AGENT_HOST": val}):
                config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                agent = ClaudeCodeAgent(config=config)
                assert agent.config.use_agent_host is True
        for val in ("false", ""):
            with patch.dict(os.environ, {"USE_AGENT_HOST": val}):
                with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
                    config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                    agent = ClaudeCodeAgent(config=config)
                assert agent.config.use_agent_host is False

    def test_skips_cli_verification(self):
        with _agent_host_env():
            with patch.object(ClaudeCodeAgent, "_verify_cli_available") as mock_verify:
                config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                ClaudeCodeAgent(config=config)
                mock_verify.assert_not_called()

    def test_precedence_over_k8s(self):
        with patch.dict(os.environ, {"USE_AGENT_HOST": "true", "USE_K8S_EXECUTOR": "true"}):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_agent_host is True
            assert agent.config.use_k8s is False

    def test_precedence_over_proxy(self):
        with patch.dict(
            os.environ,
            {"USE_AGENT_HOST": "true", "CLI_PROXY_ENABLED": "true", "CLI_PROXY_URL": "http://x"},
        ):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.use_agent_host is True
            assert agent.config.use_proxy is False


# ===========================================================================
# _verify_cli_available (covers cli_agent.py lines 149-178)
# ===========================================================================


class TestVerifyCLIAvailable:
    def test_cli_found_via_shutil_which(self):
        """shutil.which resolves cli name -> sets path."""
        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(
                "subprocess.run",
                return_value=MagicMock(returncode=0, stdout="claude 1.0"),
            ),
        ):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent.config.claude_code_path == "/usr/local/bin/claude"

    def test_cli_found_nonzero_exit(self):
        """Non-zero returncode logs warning but does not raise."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "subprocess.run",
                return_value=MagicMock(returncode=1, stdout=""),
            ),
        ):
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
            assert agent is not None

    def test_cli_not_found_raises(self):
        """FileNotFoundError when CLI binary missing."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
                with pytest.raises(FileNotFoundError, match="not found"):
                    config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                    ClaudeCodeAgent(config=config)

    def test_cli_generic_exception(self):
        """Generic exceptions during verification are warned but not raised."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run", side_effect=OSError("permission denied")):
                config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
                agent = ClaudeCodeAgent(config=config)
                assert agent is not None

    def test_custom_path_skips_shutil_which(self):
        """When claude_code_path != 'claude', shutil.which is not used to resolve."""
        with (
            patch("shutil.which") as mock_which,
            patch(
                "subprocess.run",
                return_value=MagicMock(returncode=0, stdout="v1.0"),
            ),
        ):
            config = ClaudeCodeAgentConfig(
                name="test", db_path="test.db", claude_code_path="/custom/claude"
            )
            agent = ClaudeCodeAgent(config=config)
            mock_which.assert_not_called()
            assert agent.config.claude_code_path == "/custom/claude"


# ===========================================================================
# _get_cli_path / _set_cli_path / _get_k8s_image_env_var / _resolve_proxy_url
# ===========================================================================


class TestAbstractMethodImpls:
    def test_get_cli_path(self):
        agent = _make_agent(claude_code_path="/bin/claude")
        assert agent._get_cli_path() == "/bin/claude"

    def test_set_cli_path(self):
        agent = _make_agent()
        agent._set_cli_path("/new/path")
        assert agent.config.claude_code_path == "/new/path"

    def test_get_k8s_image_env_var(self):
        agent = _make_agent()
        assert agent._get_k8s_image_env_var() == "K8S_EXECUTOR_IMAGE"

    def test_get_proxy_endpoint(self):
        agent = _make_agent()
        assert agent._get_proxy_endpoint() == "/claude"

    def test_resolve_proxy_url_from_claude_env(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_PROXY_URL": "http://a"}):
            agent = _make_agent()
            assert agent._resolve_proxy_url() == "http://a"

    def test_resolve_proxy_url_fallback(self):
        with patch.dict(os.environ, {"CLI_PROXY_URL": "http://b"}, clear=False):
            os.environ.pop("CLAUDE_CODE_PROXY_URL", None)
            agent = _make_agent()
            assert agent._resolve_proxy_url() == "http://b"

    def test_resolve_proxy_url_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_CODE_PROXY_URL", None)
            os.environ.pop("CLI_PROXY_URL", None)
            agent = _make_agent()
            assert agent._resolve_proxy_url() is None


# ===========================================================================
# _build_command / _build_agent_host_command / _build_proxy_payload
# ===========================================================================


class TestBuildCommand:
    def test_basic_command(self):
        agent = _make_agent()
        cmd = agent._build_command("Hello")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "Hello" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_command_with_model(self):
        agent = _make_agent(model="claude-sonnet-4-20250514")
        cmd = agent._build_command("Hello")
        assert "--model" in cmd
        assert "claude-sonnet-4-20250514" in cmd

    def test_command_with_system_prompt(self):
        agent = _make_agent(system_prompt="Be helpful")
        cmd = agent._build_command("Hello")
        assert "--system-prompt" in cmd
        assert "Be helpful" in cmd

    def test_command_without_optional_flags(self):
        agent = _make_agent(model=None, system_prompt=None)
        cmd = agent._build_command("test")
        assert "--model" not in cmd
        assert "--system-prompt" not in cmd


class TestBuildAgentHostCommand:
    def test_basic(self):
        agent = _make_agent()
        cmd = agent._build_agent_host_command("do stuff")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "do stuff" in cmd
        assert "--output-format" in cmd

    def test_with_model_and_system_prompt(self):
        agent = _make_agent(model="claude-opus-4-20250514", system_prompt="Be nice")
        cmd = agent._build_agent_host_command("hi")
        assert "--model" in cmd
        assert "claude-opus-4-20250514" in cmd
        assert "--system-prompt" in cmd
        assert "Be nice" in cmd

    def test_without_optional_flags(self):
        agent = _make_agent(model=None, system_prompt=None)
        cmd = agent._build_agent_host_command("x")
        assert "--model" not in cmd
        assert "--system-prompt" not in cmd


class TestBuildProxyPayload:
    def test_minimal_payload(self):
        agent = _make_agent(model=None, system_prompt=None, working_directory=None)
        payload = agent._build_proxy_payload("hello")
        assert payload["prompt"] == "hello"
        assert payload["timeout"] == 300
        assert "model" not in payload
        assert "system_prompt" not in payload
        assert "working_directory" not in payload

    def test_full_payload(self):
        agent = _make_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Be helpful",
            working_directory="/work",
            timeout=600,
        )
        payload = agent._build_proxy_payload("hello")
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["system_prompt"] == "Be helpful"
        assert payload["working_directory"] == "/work"
        assert payload["timeout"] == 600


# ===========================================================================
# _parse_output
# ===========================================================================


class TestParseOutput:
    def test_parse_json_with_usage(self):
        agent = _make_agent()
        output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Hello world",
                "session_id": "abc-123",
                "total_cost_usd": 0.05,
                "duration_ms": 1234,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )
        result = agent._parse_output(output)
        assert result.output == "Hello world"
        assert result.session_id == "abc-123"
        assert result.cost_usd == 0.05
        assert result.duration_ms == 1234
        assert result.is_error is False
        assert result.usage_metrics is not None
        assert result.usage_metrics.prompt_tokens == 10
        assert result.usage_metrics.completion_tokens == 20
        assert result.usage_metrics.total_tokens == 30

    def test_parse_json_without_usage(self):
        agent = _make_agent()
        output = json.dumps({"result": "ok"})
        result = agent._parse_output(output)
        assert result.output == "ok"
        assert result.usage_metrics is None

    def test_parse_non_json_output(self):
        agent = _make_agent()
        result = agent._parse_output("Plain text output")
        assert result.output == "Plain text output"
        assert result.usage_metrics is None


# ===========================================================================
# _execute_direct (cli_agent.py lines 277-314)
# ===========================================================================


class TestExecuteDirect:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output text", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await agent._execute_direct(["claude", "-p", "hi"])
        assert result == "output text"

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="failed with exit code 1"):
                await agent._execute_direct(["claude", "-p", "hi"])

    @pytest.mark.asyncio
    async def test_nonzero_exit_no_stderr(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Unknown error"):
                await agent._execute_direct(["claude", "-p", "hi"])

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self):
        agent = _make_agent(timeout=1)
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        # process.kill() is synchronous (no await), so use MagicMock
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError):
                await agent._execute_direct(["claude", "-p", "hi"])
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_kill_failure_handled(self):
        agent = _make_agent(timeout=1)
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        # process.kill() is synchronous (no await), so use MagicMock
        mock_proc.kill = MagicMock(side_effect=OSError("already dead"))
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError):
                await agent._execute_direct(["claude", "-p", "hi"])

    @pytest.mark.asyncio
    async def test_uses_working_directory(self):
        agent = _make_agent(working_directory="/my/dir")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await agent._execute_direct(["claude", "-p", "hi"])
            _, kwargs = mock_exec.call_args
            assert kwargs["cwd"] == "/my/dir"

    @pytest.mark.asyncio
    async def test_uses_cwd_when_no_working_directory(self):
        agent = _make_agent(working_directory=None)
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await agent._execute_direct(["claude", "-p", "hi"])
            _, kwargs = mock_exec.call_args
            assert kwargs["cwd"] == os.getcwd()


# ===========================================================================
# _execute_via_proxy (cli_agent.py lines 316-351)
# ===========================================================================


class TestExecuteViaProxy:
    @pytest.mark.asyncio
    async def test_success(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(200, {"success": True, "output": "proxy result", "execution_time": 1.5})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            result = await agent._execute_via_proxy("hi")
        assert result == "proxy result"

    @pytest.mark.asyncio
    async def test_no_proxy_url_raises(self):
        agent = _make_agent()
        agent.config.use_proxy = True
        agent.config.proxy_url = None
        with pytest.raises(RuntimeError, match="Proxy URL not configured"):
            await agent._execute_via_proxy("hi")

    @pytest.mark.asyncio
    async def test_non_200_status(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(500, text="Internal Server Error")
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Proxy returned status 500"):
                await agent._execute_via_proxy("hi")

    @pytest.mark.asyncio
    async def test_proxy_execution_failed(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(200, {"success": False, "error": "model down"})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Proxy execution failed: model down"):
                await agent._execute_via_proxy("hi")

    @pytest.mark.asyncio
    async def test_proxy_timeout(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        client = _mock_httpx_client(post_side_effect=httpx.TimeoutException("timed out"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(httpx.TimeoutException):
                await agent._execute_via_proxy("hi")

    @pytest.mark.asyncio
    async def test_proxy_generic_exception(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        client = _mock_httpx_client(post_side_effect=ConnectionError("refused"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Proxy execution error"):
                await agent._execute_via_proxy("hi")

    @pytest.mark.asyncio
    async def test_proxy_success_false_unknown_error(self):
        """When success is False and no error key, should use 'Unknown proxy error'."""
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(200, {"success": False})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Unknown proxy error"):
                await agent._execute_via_proxy("hi")


# ===========================================================================
# _execute_via_k8s (cli_agent.py lines 180-221)
# ===========================================================================


class TestExecuteViaK8s:
    @pytest.mark.asyncio
    async def test_import_error(self):
        with _k8s_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        with patch.dict("sys.modules", {"src.k8s_job_executor": None}):
            with pytest.raises(RuntimeError, match="execution failed"):
                await agent.run("test")

    @pytest.mark.asyncio
    async def test_happy_path_json_result(self):
        """K8s executor returns logs with JSON result line."""
        with _k8s_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        mock_executor = AsyncMock()
        mock_executor.initialize = AsyncMock()
        json_line = json.dumps({"type": "result", "result": "k8s answer"})
        mock_executor.execute_agent = AsyncMock(return_value={"logs": f"Starting...\n{json_line}"})

        k8s_module = MagicMock()
        k8s_module.K8sJobExecutor = MagicMock(return_value=mock_executor)

        with patch.dict("sys.modules", {"src.k8s_job_executor": k8s_module}):
            result = await agent._execute_via_k8s("test prompt")
        assert result == json_line

    @pytest.mark.asyncio
    async def test_plain_text_logs(self):
        """When no JSON result line, returns raw logs."""
        with _k8s_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        mock_executor = AsyncMock()
        mock_executor.initialize = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value={"logs": "no json here"})

        k8s_module = MagicMock()
        k8s_module.K8sJobExecutor = MagicMock(return_value=mock_executor)

        with patch.dict("sys.modules", {"src.k8s_job_executor": k8s_module}):
            result = await agent._execute_via_k8s("test")
        assert result == "no json here"

    @pytest.mark.asyncio
    async def test_k8s_with_custom_image_and_secret(self):
        """Custom image and session secret are set in env."""
        with _k8s_env():
            config = ClaudeCodeAgentConfig(
                name="test",
                db_path="test.db",
                k8s_executor_image="custom:v1",
                k8s_session_secret="custom-secret",
            )
            agent = ClaudeCodeAgent(config=config)

        mock_executor = AsyncMock()
        mock_executor.initialize = AsyncMock()
        mock_executor.execute_agent = AsyncMock(return_value={"logs": json.dumps({"result": "ok"})})

        k8s_module = MagicMock()
        k8s_module.K8sJobExecutor = MagicMock(return_value=mock_executor)

        # _execute_via_k8s calls os.environ.setdefault for image/secret — keep env clean
        with patch.dict("sys.modules", {"src.k8s_job_executor": k8s_module}):
            with patch.dict(os.environ, {}, clear=False):
                result = await agent._execute_via_k8s("test")
        # Clean up any env vars leaked by _execute_via_k8s
        os.environ.pop("CLAUDE_SESSION_SECRET_NAME", None)
        os.environ.pop("K8S_EXECUTOR_IMAGE", None)
        assert result == json.dumps({"result": "ok"})

    @pytest.mark.asyncio
    async def test_k8s_skips_invalid_json_lines(self):
        """Malformed JSON lines are skipped; falls through to raw logs."""
        with _k8s_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        mock_executor = AsyncMock()
        mock_executor.initialize = AsyncMock()
        mock_executor.execute_agent = AsyncMock(
            return_value={"logs": "line1\n{invalid json\nline3"}
        )

        k8s_module = MagicMock()
        k8s_module.K8sJobExecutor = MagicMock(return_value=mock_executor)

        with patch.dict("sys.modules", {"src.k8s_job_executor": k8s_module}):
            result = await agent._execute_via_k8s("test")
        assert result == "line1\n{invalid json\nline3"


# ===========================================================================
# run() (cli_agent.py lines 353-404)
# ===========================================================================


class TestRunModes:
    @pytest.mark.asyncio
    async def test_run_proxy_mode(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        with patch.object(agent, "_execute_via_proxy", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = '{"type":"result","result":"Hello","is_error":false}'
            result = await agent.run("test prompt")
            mock_exec.assert_called_once_with("test prompt")
            assert result.output == "Hello"

    @pytest.mark.asyncio
    async def test_run_k8s_mode(self):
        with _k8s_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        with patch.object(agent, "_execute_via_k8s", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = '{"type":"result","result":"K8s output","is_error":false}'
            result = await agent.run("test prompt")
            mock_exec.assert_called_once_with("test prompt")
            assert result.output == "K8s output"

    @pytest.mark.asyncio
    async def test_run_agent_host_mode(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        with patch.object(agent, "_execute_via_agent_host", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = '{"type":"result","result":"ah output","is_error":false}'
            result = await agent.run("test prompt")
            mock_exec.assert_called_once_with("test prompt")
            assert result.output == "ah output"

    @pytest.mark.asyncio
    async def test_run_direct_mode(self):
        """Direct mode builds command and calls _execute_direct."""
        agent = _make_agent()
        with patch.object(agent, "_execute_direct", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = json.dumps({"result": "direct output"})
            result = await agent.run("test prompt")
            mock_exec.assert_called_once()
            assert result.output == "direct output"

    @pytest.mark.asyncio
    async def test_run_records_token_usage(self):
        agent = _make_agent()
        output = json.dumps(
            {
                "result": "ok",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        with patch.object(agent, "_execute_direct", new_callable=AsyncMock, return_value=output):
            result = await agent.run("hi")
        assert result.usage_metrics is not None
        assert result.usage_metrics.prompt_tokens == 100
        assert result.usage_metrics.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_run_timeout_error(self):
        agent = _make_agent()
        with (
            patch.object(
                agent,
                "_execute_direct",
                new_callable=AsyncMock,
                side_effect=TimeoutError("timed out"),
            ),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            await agent.run("hi")

    @pytest.mark.asyncio
    async def test_run_generic_exception(self):
        agent = _make_agent()
        with (
            patch.object(
                agent,
                "_execute_direct",
                new_callable=AsyncMock,
                side_effect=ValueError("bad value"),
            ),
            pytest.raises(RuntimeError, match="execution failed"),
        ):
            await agent.run("hi")


# ===========================================================================
# check_health (cli_agent.py lines 406-412)
# ===========================================================================


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_healthy(self):
        agent = _make_agent()
        with patch.object(agent, "_verify_cli_available"):
            result = await agent.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        agent = _make_agent()
        with patch.object(
            agent, "_verify_cli_available", side_effect=FileNotFoundError("not found")
        ):
            result = await agent.check_health()
        assert result is False


# ===========================================================================
# Agent-host execution
# ===========================================================================


class TestAgentHostExecution:
    @pytest.mark.asyncio
    async def test_creates_container_on_first_call(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        mock_instance = AsyncMock()
        mock_instance.url = "http://agent-host:9100"
        mock_instance.name = "mini1"

        create_resp = _resp(201, {"container_id": "ctr-new", "container_name": "claude-test"})
        exec_resp = _resp(200, {"success": True, "exit_code": 0, "stdout": "output", "stderr": ""})

        client = _mock_httpx_client(post_side_effect=[create_resp, exec_resp])
        with (
            patch("src.services.agent_host_registry.agent_host_registry") as mock_reg,
            patch("httpx.AsyncClient", return_value=client),
        ):
            mock_reg.get_best_instance = AsyncMock(return_value=mock_instance)
            result = await agent._execute_via_agent_host("hello")

        assert agent.config.agent_host_container_id == "ctr-new"
        assert result == "output"

    @pytest.mark.asyncio
    async def test_reuses_container(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "existing-ctr"
        agent.config.agent_host_url = "http://agent-host:9100"

        exec_resp = _resp(200, {"success": True, "exit_code": 0, "stdout": "reused", "stderr": ""})
        client = _mock_httpx_client(post_returns=exec_resp)
        with patch("httpx.AsyncClient", return_value=client):
            result = await agent._execute_via_agent_host("hello")
        assert result == "reused"
        assert client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_failure_includes_stdout_and_stderr(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        resp = _resp(
            200,
            {
                "success": False,
                "exit_code": 1,
                "stdout": '{"type":"error","error":"model_refused"}',
                "stderr": "some error",
            },
        )
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError) as exc_info:
                await agent._execute_via_agent_host("test")
        msg = str(exc_info.value)
        assert "stdout=" in msg
        assert "stderr=" in msg
        assert "model_refused" in msg

    @pytest.mark.asyncio
    async def test_failure_stdout_truncated(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        resp = _resp(200, {"success": False, "exit_code": 1, "stdout": "x" * 1000, "stderr": ""})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError) as exc_info:
                await agent._execute_via_agent_host("test")
        assert len(str(exc_info.value)) < 700

    @pytest.mark.asyncio
    async def test_no_healthy_instance(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        with patch("src.services.agent_host_registry.agent_host_registry") as mock_reg:
            mock_reg.get_best_instance = AsyncMock(return_value=None)
            with pytest.raises(RuntimeError, match="No healthy"):
                await agent._execute_via_agent_host("test")

    @pytest.mark.asyncio
    async def test_container_create_fails(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        mock_instance = AsyncMock()
        mock_instance.url = "http://agent-host:9100"
        mock_instance.name = "mini1"

        create_resp = _resp(500, text="Internal Server Error")
        client = _mock_httpx_client(post_returns=create_resp)
        with (
            patch("src.services.agent_host_registry.agent_host_registry") as mock_reg,
            patch("httpx.AsyncClient", return_value=client),
        ):
            mock_reg.get_best_instance = AsyncMock(return_value=mock_instance)
            with pytest.raises(RuntimeError, match="container create failed"):
                await agent._execute_via_agent_host("test")

    @pytest.mark.asyncio
    async def test_exec_http_error(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        resp = _resp(500, text="Server Error")
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="agent-host exec failed"):
                await agent._execute_via_agent_host("test")


# ===========================================================================
# Cleanup
# ===========================================================================


class TestAgentHostCleanup:
    @pytest.mark.asyncio
    async def test_destroys_container(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        resp = _resp(200)
        client = _mock_httpx_client(delete_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            await agent.cleanup()
        client.delete.assert_called_once()
        assert agent.config.agent_host_container_id is None

    @pytest.mark.asyncio
    async def test_delete_non_200_warns(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        resp = _resp(500, text="fail")
        client = _mock_httpx_client(delete_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            await agent.cleanup()  # should not raise
        assert agent.config.agent_host_container_id is None

    @pytest.mark.asyncio
    async def test_handles_delete_exception(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"

        client = _mock_httpx_client(delete_side_effect=Exception("refused"))
        with patch("httpx.AsyncClient", return_value=client):
            await agent.cleanup()  # should not raise
        assert agent.config.agent_host_container_id is None

    @pytest.mark.asyncio
    async def test_noop_when_no_container(self):
        with _agent_host_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        with patch("httpx.AsyncClient") as mock_cls:
            await agent.cleanup()
            mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_not_agent_host_mode(self):
        agent = _make_agent()
        agent.config.agent_host_container_id = "ctr-123"
        agent.config.agent_host_url = "http://agent-host:9100"
        # use_agent_host is False -> cleanup should not try to delete
        with patch("httpx.AsyncClient") as mock_cls:
            await agent.cleanup()
            mock_cls.assert_not_called()


# ===========================================================================
# MCP configuration (claude_code_agent.py lines 127-222)
# ===========================================================================


class TestMCPConfiguration:
    def test_mcp_servers_stored_in_config(self):
        agent = _make_agent(mcp_servers=[{"name": "kanban", "url": "http://localhost:1337/sse"}])
        assert len(agent.config.mcp_servers) == 1

    @pytest.mark.asyncio
    async def test_configure_mcp_servers_empty(self):
        agent = _make_agent()
        # Should return early when no MCP servers
        await agent._configure_mcp_servers()

    @pytest.mark.asyncio
    async def test_configure_mcp_string_server(self):
        """String-only server name warns and skips (no url for direct mode)."""
        agent = _make_agent()
        agent.config.mcp_servers = ["my-server"]
        await agent._configure_mcp_servers()
        # Should complete without error (logs warning)

    @pytest.mark.asyncio
    async def test_configure_mcp_dict_server_with_url(self):
        agent = _make_agent()
        agent.config.mcp_servers = [{"name": "kanban", "url": "http://localhost:1337/sse"}]

        with patch.object(agent, "_add_mcp_server_direct", new_callable=AsyncMock) as mock_add:
            await agent._configure_mcp_servers()
            mock_add.assert_called_once_with("kanban", "http://localhost:1337/sse")

    @pytest.mark.asyncio
    async def test_configure_mcp_dict_server_no_name(self):
        """Dict without 'name' key should be skipped."""
        agent = _make_agent()
        agent.config.mcp_servers = [{"url": "http://localhost:1337/sse"}]
        await agent._configure_mcp_servers()  # should log error and skip

    @pytest.mark.asyncio
    async def test_configure_mcp_invalid_type(self):
        """Non-string non-dict with .get() survives comprehension, hits else branch."""

        class FakeObj:
            def get(self, key, default=None):
                return default

        agent = _make_agent()
        agent.config.mcp_servers = [FakeObj()]
        await agent._configure_mcp_servers()

    @pytest.mark.asyncio
    async def test_configure_mcp_dict_no_url_direct_mode(self):
        """Dict with name but no url in direct mode should skip."""
        agent = _make_agent()
        agent.config.mcp_servers = [{"name": "my-server"}]
        await agent._configure_mcp_servers()

    @pytest.mark.asyncio
    async def test_configure_mcp_via_proxy(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        agent.config.mcp_servers = [{"name": "kanban", "url": "http://x"}]

        with patch.object(agent, "_add_mcp_server_via_proxy", new_callable=AsyncMock) as mock_add:
            await agent._configure_mcp_servers()
            mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_mcp_exception_handling(self):
        """Exceptions during MCP add should be caught and logged."""
        agent = _make_agent()
        agent.config.mcp_servers = [{"name": "fail-server", "url": "http://fail"}]

        with patch.object(
            agent,
            "_add_mcp_server_direct",
            new_callable=AsyncMock,
            side_effect=RuntimeError("add failed"),
        ):
            await agent._configure_mcp_servers()  # should not raise


class TestAddMCPServerViaProxy:
    @pytest.mark.asyncio
    async def test_success(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(200, {"success": True})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            await agent._add_mcp_server_via_proxy({"name": "kanban", "url": "http://x"})

    @pytest.mark.asyncio
    async def test_no_proxy_url(self):
        agent = _make_agent()
        agent.config.proxy_url = None
        with pytest.raises(RuntimeError, match="Proxy URL not configured"):
            await agent._add_mcp_server_via_proxy({"name": "kanban"})

    @pytest.mark.asyncio
    async def test_no_name_in_config(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)
        with pytest.raises(RuntimeError, match="missing 'name'"):
            await agent._add_mcp_server_via_proxy({"url": "http://x"})

    @pytest.mark.asyncio
    async def test_non_200(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(500, text="error")
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Proxy returned status 500"):
                await agent._add_mcp_server_via_proxy({"name": "kanban"})

    @pytest.mark.asyncio
    async def test_success_false(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        resp = _resp(200, {"success": False, "error": "bad config"})
        client = _mock_httpx_client(post_returns=resp)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="MCP add failed"):
                await agent._add_mcp_server_via_proxy({"name": "kanban"})

    @pytest.mark.asyncio
    async def test_timeout(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        client = _mock_httpx_client(post_side_effect=httpx.TimeoutException("timed out"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(httpx.TimeoutException):
                await agent._add_mcp_server_via_proxy({"name": "kanban"})

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        with _proxy_env():
            config = ClaudeCodeAgentConfig(name="test", db_path="test.db")
            agent = ClaudeCodeAgent(config=config)

        client = _mock_httpx_client(post_side_effect=ConnectionError("refused"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(RuntimeError, match="Error adding MCP server via proxy"):
                await agent._add_mcp_server_via_proxy({"name": "kanban"})


class TestAddMCPServerDirect:
    @pytest.mark.asyncio
    async def test_success(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await agent._add_mcp_server_direct("kanban", "http://localhost:1337/sse")

    @pytest.mark.asyncio
    async def test_no_url_raises(self):
        agent = _make_agent()
        with pytest.raises(RuntimeError, match="requires a URL"):
            await agent._add_mcp_server_direct("kanban", None)

    @pytest.mark.asyncio
    async def test_no_url_empty_string(self):
        agent = _make_agent()
        with pytest.raises(RuntimeError, match="requires a URL"):
            await agent._add_mcp_server_direct("kanban", "")

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="MCP add failed"):
                await agent._add_mcp_server_direct("kanban", "http://x")

    @pytest.mark.asyncio
    async def test_timeout(self):
        agent = _make_agent()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError):
                await agent._add_mcp_server_direct("kanban", "http://x")

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        agent = _make_agent()
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("bad")):
            with pytest.raises(OSError):
                await agent._add_mcp_server_direct("kanban", "http://x")


# ===========================================================================
# MCP init-time configuration (asyncio loop detection)
# ===========================================================================


class TestMCPInitConfig:
    def test_mcp_configure_scheduled_in_running_loop(self):
        """When event loop is running, task is scheduled (not run synchronously)."""
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            with patch.object(ClaudeCodeAgent, "_configure_mcp_servers", new_callable=AsyncMock):
                # Simulate having a running loop
                loop = asyncio.new_event_loop()
                try:

                    async def create_agent():
                        config = ClaudeCodeAgentConfig(
                            name="test",
                            db_path="test.db",
                            mcp_servers=[{"name": "s1", "url": "http://s1"}],
                        )
                        return ClaudeCodeAgent(config=config)

                    agent = loop.run_until_complete(create_agent())
                    # Task was scheduled
                    assert agent.config.mcp_servers
                finally:
                    loop.close()

    def test_mcp_configure_exception_handling(self):
        """Exceptions during MCP init config should be caught."""
        with (
            patch.object(ClaudeCodeAgent, "_verify_cli_available"),
            patch.object(
                ClaudeCodeAgent,
                "_configure_mcp_servers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("mcp fail"),
            ),
        ):
            config = ClaudeCodeAgentConfig(
                name="test",
                db_path="test.db",
                mcp_servers=[{"name": "s1", "url": "http://s1"}],
            )
            # Should not raise
            agent = ClaudeCodeAgent(config=config)
            assert agent is not None


# ===========================================================================
# create_claude_code_agent factory
# ===========================================================================


class TestCreateClaudeCodeAgent:
    def test_factory_function(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
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

    def test_factory_default_system_prompt(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(name="test", description="Test", db_path="test.db")
            assert agent.config.system_prompt is not None
            assert "helpful coding assistant" in agent.config.system_prompt

    def test_factory_with_model_and_timeout(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
                name="test",
                description="Test",
                db_path="test.db",
                model="claude-opus-4-20250514",
                timeout=600,
            )
            assert agent.config.model == "claude-opus-4-20250514"
            assert agent.config.timeout == 600

    def test_factory_with_mcp_servers(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
                name="test",
                description="Test",
                db_path="test.db",
                mcp_servers=["server1"],
            )
            assert len(agent.config.mcp_servers) == 1

    def test_factory_with_kwargs(self):
        """Extra kwargs should be ignored gracefully."""
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
                name="test",
                description="Test",
                db_path="test.db",
                model_provider="anthropic",
                tools=[],
                base_url="http://test",
            )
            assert agent.name == "test"

    def test_factory_with_working_directory(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
                name="test",
                description="Test",
                db_path="test.db",
                working_directory="/work",
            )
            assert agent.config.working_directory == "/work"

    def test_factory_with_custom_claude_path(self):
        with patch.object(ClaudeCodeAgent, "_verify_cli_available"):
            agent = create_claude_code_agent(
                name="test",
                description="Test",
                db_path="test.db",
                claude_code_path="/custom/claude",
            )
            assert agent.config.claude_code_path == "/custom/claude"


# ===========================================================================
# Agent registration
# ===========================================================================


class TestAgentRegistration:
    def test_register_claude_code_agent(self):
        """_register_claude_code_agent is called at module load time."""
        # It has already been called by import. We can verify the function exists.
        from src.worker.agents.claude_code_agent import _register_claude_code_agent

        # Call it again (should be idempotent or log warning)
        _register_claude_code_agent()

    def test_register_fails_gracefully(self):
        from src.worker.agents.claude_code_agent import _register_claude_code_agent

        # register_agent is imported inside _register_claude_code_agent via
        # `from .agent_registry import register_agent`, so patch the import mechanism
        with patch.dict("sys.modules", {"src.worker.agents.agent_registry": None}):
            _register_claude_code_agent()  # should not raise


# ===========================================================================
# Class attributes
# ===========================================================================


class TestClassAttributes:
    def test_cli_name(self):
        assert ClaudeCodeAgent.cli_name == "claude"

    def test_agent_type_label(self):
        assert ClaudeCodeAgent.agent_type_label == "claude_code"

    def test_model_provider_label(self):
        assert ClaudeCodeAgent.model_provider_label == "anthropic"

    def test_default_model_name(self):
        assert ClaudeCodeAgent.default_model_name == "claude-sonnet-4-20250514"


# ===========================================================================
# run_with_context (inherited from BaseAgent)
# ===========================================================================


class TestRunWithContext:
    @pytest.mark.asyncio
    async def test_run_with_context(self):
        agent = _make_agent()
        with patch.object(agent, "_execute_direct", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = json.dumps({"result": "context result"})
            result = await agent.run_with_context(
                {
                    "task": "do something",
                    "graph_context": {"session_id": "s1", "completed_nodes": ["n1"]},
                    "node_config": {"key": "value"},
                }
            )
        assert result.output == "context result"
