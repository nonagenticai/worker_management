"""Tests for mcp_utils, docker_mcp_manager, and mcp_config_resolver.

Targets uncovered lines across all three modules. Avoids duplicating the
basic validation/creation tests already in test_mcp_utils_unit.py.
"""

import json
import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

from src.worker.docker_mcp_manager import (
    ContainerNotFoundError,
    DockerCommandError,
    DockerMCPManager,
)
from src.worker.factory.mcp_config_resolver import (
    _handle_preset_config,
    _handle_server_filtering,
    _normalize_mcp_config_for_environment,
    _resolve_mcp_servers_for_cli_agents,
    resolve_mcp_config,
)
from src.worker.mcp_utils import (
    MCPServerManager,
    create_http_server,
    create_http_server_sync,
    create_mcp_servers_from_config,
    create_stdio_server,
    expand_environment_variables,
    get_auth_token,
    load_mcp_servers,
    resolve_working_directory,
    validate_mcp_config,
    validate_server_config,
)

# ============================================================================
# validate_server_config — edge cases not covered elsewhere
# ============================================================================


class TestValidateServerConfigEdgeCases:
    def test_float_timeout_accepted(self):
        assert validate_server_config("s", {"url": "http://x", "timeout": 0.5}) is True

    def test_none_timeout_treated_as_non_numeric(self):
        assert validate_server_config("s", {"url": "http://x", "timeout": None}) is False

    def test_both_url_and_command_still_valid(self):
        """Logs a warning but returns True."""
        assert validate_server_config("s", {"url": "http://x", "command": "y"}) is True


# ============================================================================
# validate_mcp_config — structural checks
# ============================================================================


class TestValidateMCPConfigStructure:
    def test_rejects_non_dict(self):
        # Intentional bad type — verifies runtime guard.
        assert validate_mcp_config(cast("Any", 42)) is False

    def test_rejects_list_mcpServers(self):
        assert validate_mcp_config({"mcpServers": ["a"]}) is False

    def test_valid_with_multiple_servers(self):
        cfg = {
            "mcpServers": {
                "a": {"url": "http://a"},
                "b": {"command": "node"},
            }
        }
        assert validate_mcp_config(cfg) is True

    def test_one_bad_server_fails_whole_config(self):
        cfg = {
            "mcpServers": {
                "good": {"url": "http://a"},
                "bad": {},
            }
        }
        assert validate_mcp_config(cfg) is False


# ============================================================================
# expand_environment_variables
# ============================================================================


class TestExpandEnvVarsExtra:
    def test_multiple_vars_in_single_arg(self, monkeypatch):
        monkeypatch.setenv("X", "hello")
        monkeypatch.setenv("Y", "world")
        result = expand_environment_variables(["${X}_${Y}"])
        assert result == ["hello_world"]

    def test_partial_expansion_with_missing(self, monkeypatch):
        monkeypatch.setenv("PRESENT", "yes")
        monkeypatch.delenv("ABSENT_XYZ_123", raising=False)
        result = expand_environment_variables(["${PRESENT}-${ABSENT_XYZ_123}"])
        assert result == ["yes-"]

    def test_no_vars_passthrough(self):
        result = expand_environment_variables(["--flag", "value"])
        assert result == ["--flag", "value"]


# ============================================================================
# resolve_working_directory
# ============================================================================


class TestResolveWorkingDirectoryExtra:
    def test_nonexistent_path_still_resolves(self):
        """Non-existent path returns a resolved string (with warning)."""
        result = resolve_working_directory("/tmp/_nonexistent_dir_xyz_9999", None)
        assert result is not None
        assert "nonexistent" in result

    def test_relative_without_base_dir(self):
        """Relative path with no base_dir resolves against cwd."""
        result = resolve_working_directory("relative_child", None)
        assert result is not None
        assert os.path.isabs(result)


# ============================================================================
# create_http_server_sync
# ============================================================================


class TestCreateHttpServerSyncExtra:
    def test_creates_server_with_url(self):
        server = create_http_server_sync("s", {"url": "http://example.com/sse"})
        assert isinstance(server, MCPServerSSE)
        assert server.url == "http://example.com/sse"

    def test_empty_url_returns_none(self):
        assert create_http_server_sync("s", {"url": ""}) is None

    def test_missing_url_returns_none(self):
        assert create_http_server_sync("s", {}) is None


# ============================================================================
# create_http_server (async) — deep coverage
# ============================================================================


class TestCreateHttpServerAsync:
    @pytest.mark.asyncio
    async def test_basic_sse_creation(self):
        config = {"url": "http://localhost:9090/sse"}
        server = await create_http_server("test", config)
        assert isinstance(server, MCPServerSSE)
        assert server.url == "http://localhost:9090/sse"
        assert server.timeout == 15.0  # default

    @pytest.mark.asyncio
    async def test_custom_timeout_and_read_timeout(self):
        config = {"url": "http://x/sse", "timeout": 45, "read_timeout": 120}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.timeout == 45.0
        assert server.read_timeout == 120.0

    @pytest.mark.asyncio
    async def test_sse_read_timeout_fallback(self):
        config = {"url": "http://x/sse", "sse_read_timeout": 99}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.read_timeout == 99.0

    @pytest.mark.asyncio
    async def test_explicit_read_timeout_takes_precedence(self):
        config = {"url": "http://x/sse", "read_timeout": 50, "sse_read_timeout": 99}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.read_timeout == 50.0

    @pytest.mark.asyncio
    async def test_headers_passed_through(self):
        config = {"url": "http://x/sse", "headers": {"X-Key": "val"}}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.headers is not None
        assert server.headers["X-Key"] == "val"

    @pytest.mark.asyncio
    async def test_tool_prefix_applied(self):
        config = {"url": "http://x/sse", "tool_prefix": "github"}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.tool_prefix == "github"

    @pytest.mark.asyncio
    async def test_empty_tool_prefix_ignored(self):
        config = {"url": "http://x/sse", "tool_prefix": "  "}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.tool_prefix is None

    @pytest.mark.asyncio
    async def test_valid_log_level(self):
        config = {"url": "http://x/sse", "log_level": "WARNING"}
        server = await create_http_server("s", config)
        assert server is not None
        assert server.log_level == "WARNING"

    @pytest.mark.asyncio
    async def test_invalid_log_level_ignored(self):
        config = {"url": "http://x/sse", "log_level": "BOGUS"}
        server = await create_http_server("s", config)
        assert server is not None
        # log_level should not be set — stays default (None)
        assert server.log_level is None

    @pytest.mark.asyncio
    async def test_auth_enabled_injects_token(self):
        config = {
            "url": "http://x/sse",
            "auth": {
                "enabled": True,
                "token_url": "http://auth/token",
                "client_id": "cid",
                "client_secret": "csec",
                "grant_type": "client_credentials",
            },
        }
        with patch(
            "src.worker.mcp_utils.get_auth_token",
            new_callable=AsyncMock,
            return_value="my-token",
        ):
            server = await create_http_server("s", config)
        assert server is not None
        assert server.headers is not None
        assert server.headers["Authorization"] == "Bearer my-token"

    @pytest.mark.asyncio
    async def test_auth_custom_header_name_and_scheme(self):
        config = {
            "url": "http://x/sse",
            "auth": {
                "enabled": True,
                "token_url": "http://auth/token",
                "auth_header_name": "X-Api-Key",
                "auth_header": "Token",
            },
        }
        with patch(
            "src.worker.mcp_utils.get_auth_token",
            new_callable=AsyncMock,
            return_value="tok",
        ):
            server = await create_http_server("s", config)
        assert server is not None
        assert server.headers is not None
        assert server.headers["X-Api-Key"] == "Token tok"

    @pytest.mark.asyncio
    async def test_auth_failure_proceeds_without_token(self):
        config = {
            "url": "http://x/sse",
            "auth": {"enabled": True, "token_url": "http://auth/token"},
        }
        with patch(
            "src.worker.mcp_utils.get_auth_token",
            new_callable=AsyncMock,
            return_value=None,
        ):
            server = await create_http_server("s", config)
        assert isinstance(server, MCPServerSSE)
        assert "Authorization" not in (server.headers or {})

    @pytest.mark.asyncio
    async def test_http_client_config_creates_custom_client(self):
        config = {
            "url": "http://x/sse",
            "headers": {"X-Key": "val"},
            "http_client": {
                "timeout": 60,
                "verify": False,
            },
        }
        server = await create_http_server("s", config)
        assert isinstance(server, MCPServerSSE)
        # When http_client is set, headers are moved into the client
        assert server.http_client is not None

    @pytest.mark.asyncio
    async def test_http_client_empty_config_no_custom_client(self):
        config = {"url": "http://x/sse", "http_client": {}}
        server = await create_http_server("s", config)
        assert isinstance(server, MCPServerSSE)
        # Empty http_client dict should not create a custom client
        assert server.http_client is None


# ============================================================================
# get_auth_token — async token fetching
# ============================================================================


class TestGetAuthToken:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        assert await get_auth_token({"enabled": False}) is None

    @pytest.mark.asyncio
    async def test_missing_token_url_returns_none(self):
        assert await get_auth_token({"enabled": True}) is None

    @pytest.mark.asyncio
    async def test_successful_token_fetch(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "fetched-token"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                    "client_id": "myid",
                    "client_secret": "mysecret",
                    "grant_type": "client_credentials",
                }
            )
        assert token == "fetched-token"

    @pytest.mark.asyncio
    async def test_custom_token_property(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"api_key": "custom-key"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                    "token_property": "api_key",
                }
            )
        assert token == "custom-key"

    @pytest.mark.asyncio
    async def test_missing_token_in_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"other_field": "value"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_http_status_error(self):
        mock_request = MagicMock()
        mock_request.url = "http://auth/token"
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("401", request=mock_request, response=mock_response)
        )

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_connect_error(self):
        mock_request = MagicMock()
        mock_request.url = "http://auth/token"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused", request=mock_request)
        )

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_request_error(self):
        mock_request = MagicMock()
        mock_request.url = "http://auth/token"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("Timeout", request=mock_request)
        )

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_response.text = "not json"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_unexpected_exception(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                }
            )
        assert token is None

    @pytest.mark.asyncio
    async def test_content_type_header(self):
        """Custom content_type in auth_config is sent as header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "t"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.worker.mcp_utils.httpx.AsyncClient", return_value=mock_client):
            token = await get_auth_token(
                {
                    "enabled": True,
                    "token_url": "http://auth/token",
                    "content_type": "application/x-www-form-urlencoded",
                }
            )
        assert token == "t"
        # Verify the custom content type was passed
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


# ============================================================================
# create_stdio_server — managed docker and regular
# ============================================================================


class TestCreateStdioServerManagedDocker:
    def test_docker_run_creates_managed_server(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/test/mcp-server"],
            "env": {},
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        # _mcp_managed / _mcp_image are dynamic attrs added by mcp_utils.
        assert cast("Any", server)._mcp_managed is True
        assert cast("Any", server)._mcp_image == "ghcr.io/test/mcp-server"

    def test_docker_run_with_tool_prefix(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/test/img"],
            "env": {},
            "tool_prefix": "myprefix",
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        assert server.tool_prefix == "myprefix"

    def test_docker_run_with_timeout(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/test/img"],
            "env": {},
            "timeout": 60,
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        assert server.timeout == 60.0

    def test_docker_run_invalid_timeout(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/test/img"],
            "env": {},
            "timeout": "bad",
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        # Falls back to default

    def test_docker_run_no_image_detected(self):
        """Args that don't look like a Docker image should return None."""
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm"],
            "env": {},
        }
        result = create_stdio_server("test", config, None)
        assert result is None

    def test_docker_github_server_missing_token(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
            "env": {},
        }
        result = create_stdio_server("test", config, None)
        assert result is None

    def test_docker_github_server_empty_token(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        }
        result = create_stdio_server("test", config, None)
        assert result is None

    def test_docker_github_server_valid_token(self):
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_token123"},
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        assert cast("Any", server)._mcp_managed is True

    def test_docker_image_detected_by_mcp_prefix(self):
        """Fallback: image with 'mcp/' pattern is detected."""
        config = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "-e", "X=Y", "mcp/some-server"],
            "env": {},
        }
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        assert cast("Any", server)._mcp_image == "mcp/some-server"

    def test_docker_image_detected_by_server_suffix(self):
        """Fallback: image ending with '-server' is detected."""
        # This relies on '-server' suffix fallback, but the pattern also requires
        # 'ghcr.io/' or 'mcp/' — let's test with ghcr.io
        config2 = {
            "command": "docker",
            "args": ["run", "-i", "--rm", "-e", "X=Y", "ghcr.io/org/custom-server"],
            "env": {},
        }
        server = create_stdio_server("test", config2, None)
        assert isinstance(server, MCPServerStdio)
        assert cast("Any", server)._mcp_image == "ghcr.io/org/custom-server"


class TestCreateRegularStdioServer:
    def test_basic_creation(self):
        config = {"command": "node", "args": ["server.js"]}
        server = create_stdio_server("test", config, None)
        assert isinstance(server, MCPServerStdio)
        assert server.command == "node"
        assert server.args is not None
        assert list(server.args) == ["server.js"]

    def test_env_merges_with_os_environ(self, monkeypatch):
        monkeypatch.setenv("INHERITED", "yes")
        config = {
            "command": "node",
            "args": [],
            "env": {"CUSTOM": "val"},
        }
        server = create_stdio_server("test", config, None)
        assert server is not None
        assert server.env is not None
        assert server.env["CUSTOM"] == "val"
        assert server.env["INHERITED"] == "yes"

    def test_cwd_resolution(self, tmp_path):
        subdir = tmp_path / "workdir"
        subdir.mkdir()
        config = {"command": "node", "args": [], "cwd": "workdir"}
        server = create_stdio_server("test", config, tmp_path)
        assert server is not None
        assert server.cwd == str(subdir)

    def test_valid_log_level(self):
        config = {"command": "node", "args": [], "log_level": "ERROR"}
        server = create_stdio_server("test", config, None)
        assert server is not None
        assert server.log_level == "ERROR"

    def test_invalid_log_level_ignored(self):
        config = {"command": "node", "args": [], "log_level": "NONSENSE"}
        server = create_stdio_server("test", config, None)
        assert server is not None
        assert server.log_level is None

    def test_tool_prefix_whitespace_ignored(self):
        config = {"command": "node", "args": [], "tool_prefix": "   "}
        server = create_stdio_server("test", config, None)
        assert server is not None
        assert server.tool_prefix is None

    def test_env_expansion_in_args(self, monkeypatch):
        monkeypatch.setenv("MY_PORT", "3000")
        config = {"command": "node", "args": ["--port", "${MY_PORT}"]}
        server = create_stdio_server("test", config, None)
        assert server is not None
        assert server.args is not None
        assert list(server.args) == ["--port", "3000"]


# ============================================================================
# create_mcp_servers_from_config — more edge cases
# ============================================================================


class TestCreateMCPServersFromConfigEdges:
    def test_integer_input_returns_empty(self):
        # Intentional bad input — verifies runtime guard.
        assert create_mcp_servers_from_config(cast("Any", 12345)) == []

    def test_list_input_returns_empty(self):
        assert create_mcp_servers_from_config(cast("Any", ["item"])) == []

    def test_mcpServers_not_dict_returns_empty(self):
        assert create_mcp_servers_from_config({"mcpServers": "string"}) == []

    def test_file_with_mixed_servers(self, tmp_path):
        cfg = {
            "mcpServers": {
                "sse1": {"url": "http://a/sse"},
                "stdio1": {"command": "node", "args": ["x.js"]},
                "bad": {"timeout": 5},
            }
        }
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(cfg))
        servers = create_mcp_servers_from_config(str(f))
        assert len(servers) == 2

    def test_base_dir_from_file_path(self, tmp_path):
        """When loading from file, base_dir defaults to file's parent."""
        subdir = tmp_path / "workdir"
        subdir.mkdir()
        cfg = {
            "mcpServers": {
                "s": {"command": "node", "args": [], "cwd": "workdir"},
            }
        }
        f = tmp_path / "config.json"
        f.write_text(json.dumps(cfg))
        servers = create_mcp_servers_from_config(str(f))
        assert len(servers) == 1
        srv = servers[0]
        assert isinstance(srv, MCPServerStdio)
        assert srv.cwd == str(subdir)


# ============================================================================
# load_mcp_servers (async) — covers the async variant
# ============================================================================


class TestLoadMCPServers:
    @pytest.mark.asyncio
    async def test_dict_with_url_server(self):
        config = {"mcpServers": {"s": {"url": "http://localhost/sse"}}}
        servers = await load_mcp_servers(config)
        assert len(servers) == 1
        assert isinstance(servers[0], MCPServerSSE)

    @pytest.mark.asyncio
    async def test_dict_with_stdio_server(self):
        config = {"mcpServers": {"s": {"command": "node", "args": ["x.js"]}}}
        servers = await load_mcp_servers(config)
        assert len(servers) == 1
        assert isinstance(servers[0], MCPServerStdio)

    @pytest.mark.asyncio
    async def test_file_input(self, tmp_path):
        cfg = {"mcpServers": {"s": {"url": "http://localhost/sse"}}}
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps(cfg))
        servers = await load_mcp_servers(str(f))
        assert len(servers) == 1

    @pytest.mark.asyncio
    async def test_nonexistent_file(self):
        assert await load_mcp_servers("/tmp/_nonexistent_mcp_config_xyz.json") == []

    @pytest.mark.asyncio
    async def test_invalid_json_file(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{bad json")
        assert await load_mcp_servers(str(f)) == []

    @pytest.mark.asyncio
    async def test_invalid_type_returns_empty(self):
        assert await load_mcp_servers(cast("Any", 42)) == []

    @pytest.mark.asyncio
    async def test_no_mcp_servers_key(self):
        assert await load_mcp_servers({"other": 1}) == []

    @pytest.mark.asyncio
    async def test_mcp_servers_not_dict(self):
        assert await load_mcp_servers({"mcpServers": "bad"}) == []

    @pytest.mark.asyncio
    async def test_invalid_server_skipped(self):
        config = {
            "mcpServers": {
                "good": {"url": "http://x/sse"},
                "bad": {},
            }
        }
        servers = await load_mcp_servers(config)
        assert len(servers) == 1

    @pytest.mark.asyncio
    async def test_mixed_types(self):
        config = {
            "mcpServers": {
                "sse": {"url": "http://x/sse"},
                "stdio": {"command": "node", "args": ["s.js"]},
            }
        }
        servers = await load_mcp_servers(config)
        assert len(servers) == 2


# ============================================================================
# MCPServerManager — lifecycle management
# ============================================================================


class TestMCPServerManager:
    def test_init(self):
        mgr = MCPServerManager()
        assert mgr.servers == {}
        assert mgr.server_locks == {}

    @pytest.mark.asyncio
    async def test_start_creates_and_caches(self):
        mgr = MCPServerManager()
        fake = MagicMock(command="test")
        with (
            patch.object(mgr, "_create_server", return_value=fake),
            patch.object(mgr, "_wait_for_server_ready", return_value=None),
            patch.object(mgr, "_is_server_healthy", return_value=True),
        ):
            s1 = await mgr.start_mcp_server({"url": "http://x", "name": "n"})
            s2 = await mgr.start_mcp_server({"url": "http://x", "name": "n"})
        assert s1 is s2
        assert len(mgr.servers) == 1

    @pytest.mark.asyncio
    async def test_start_recreates_unhealthy_server(self):
        mgr = MCPServerManager()
        old = MagicMock(command="old")
        new = MagicMock(command="new")
        with (
            patch.object(mgr, "_create_server", side_effect=[old, new]),
            patch.object(mgr, "_wait_for_server_ready", return_value=None),
        ):
            s1 = await mgr.start_mcp_server({"url": "http://x", "name": "n"})
            with patch.object(mgr, "_is_server_healthy", return_value=False):
                s2 = await mgr.start_mcp_server({"url": "http://x", "name": "n"})
        assert s1 is old
        assert s2 is new

    @pytest.mark.asyncio
    async def test_start_managed_docker_server(self):
        mgr = MCPServerManager()
        fake = MagicMock(command="docker")
        fake._mcp_managed = True
        fake._mcp_server_name = "test"
        fake._mcp_image = "ghcr.io/test"
        fake._mcp_env_vars = {}
        fake.args = []
        with (
            patch.object(mgr, "_create_server", return_value=fake),
            patch.object(mgr, "_wait_for_server_ready", return_value=None),
        ):
            server = await mgr.start_mcp_server({"command": "docker", "name": "test"})
        assert server.args == ["exec", "-i", "mcp-test-managed", "/entrypoint"]

    @pytest.mark.asyncio
    async def test_create_server_with_url(self):
        mgr = MCPServerManager()
        server = await mgr._create_server({"url": "http://x/sse", "name": "test"}, "")
        assert isinstance(server, MCPServerSSE)

    @pytest.mark.asyncio
    async def test_create_server_with_command(self):
        mgr = MCPServerManager()
        server = await mgr._create_server({"command": "node", "name": "test", "args": []}, "")
        assert isinstance(server, MCPServerStdio)

    @pytest.mark.asyncio
    async def test_create_server_invalid_config(self):
        mgr = MCPServerManager()
        with pytest.raises(Exception, match="Invalid server config"):
            await mgr._create_server({"name": "test"}, "")

    @pytest.mark.asyncio
    async def test_ensure_container_running_updates_args(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        srv._mcp_managed = True
        srv._mcp_server_name = "myserver"
        srv._mcp_image = "ghcr.io/test/img"
        srv._mcp_env_vars = {}
        srv.args = []
        await mgr._ensure_container_running(srv)
        assert srv.args == ["exec", "-i", "mcp-myserver-managed", "/entrypoint"]

    @pytest.mark.asyncio
    async def test_ensure_container_running_no_image(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        srv._mcp_managed = True
        srv._mcp_server_name = "s"
        srv._mcp_image = None
        srv._mcp_env_vars = {}
        # Should log error but not crash
        await mgr._ensure_container_running(srv)

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_timeout(self):
        mgr = MCPServerManager()
        with patch.object(mgr, "_is_server_healthy", return_value=False):
            with pytest.raises(Exception, match="failed to become ready"):
                await mgr._wait_for_server_ready(MagicMock(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_exception_during_check(self):
        mgr = MCPServerManager()
        with patch.object(mgr, "_is_server_healthy", side_effect=RuntimeError("check failed")):
            with pytest.raises(Exception, match="failed to become ready"):
                await mgr._wait_for_server_ready(MagicMock(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_is_server_healthy_none(self):
        mgr = MCPServerManager()
        assert await mgr._is_server_healthy(None) is False

    @pytest.mark.asyncio
    async def test_is_server_healthy_with_is_running(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        del srv._mcp_managed  # Remove _mcp_managed attribute
        srv.is_running = AsyncMock(return_value=True)
        assert await mgr._is_server_healthy(srv) is True

    @pytest.mark.asyncio
    async def test_is_server_healthy_with_command_attr(self):
        mgr = MCPServerManager()
        srv = MagicMock(spec=[])  # Empty spec — no is_running
        srv.command = "node"
        assert await mgr._is_server_healthy(srv) is True

    @pytest.mark.asyncio
    async def test_is_server_healthy_managed_server(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        srv._mcp_managed = True
        srv._mcp_server_name = "s"
        srv.is_running = AsyncMock(return_value=True)
        assert await mgr._is_server_healthy(srv) is True

    @pytest.mark.asyncio
    async def test_is_server_healthy_exception(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        srv.is_running = AsyncMock(side_effect=RuntimeError("fail"))
        assert await mgr._is_server_healthy(srv) is False

    @pytest.mark.asyncio
    async def test_cleanup_managed_server(self):
        mgr = MCPServerManager()
        srv = MagicMock()
        srv._mcp_managed = True
        mgr.servers["k1"] = srv
        await mgr._cleanup_server("k1")
        assert "k1" not in mgr.servers

    @pytest.mark.asyncio
    async def test_cleanup_server_with_stop(self):
        mgr = MCPServerManager()
        srv = MagicMock(spec=[])
        srv.stop = AsyncMock()
        mgr.servers["k"] = srv
        await mgr._cleanup_server("k")
        srv.stop.assert_awaited_once()
        assert "k" not in mgr.servers

    @pytest.mark.asyncio
    async def test_cleanup_server_with_cleanup_method(self):
        mgr = MCPServerManager()
        srv = MagicMock(spec=[])
        srv.cleanup = AsyncMock()
        mgr.servers["k"] = srv
        await mgr._cleanup_server("k")
        srv.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_key(self):
        mgr = MCPServerManager()
        await mgr._cleanup_server("no-such-key")  # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_error_still_removes(self):
        mgr = MCPServerManager()
        srv = MagicMock(spec=[])
        srv.stop = AsyncMock(side_effect=RuntimeError("fail"))
        mgr.servers["k"] = srv
        await mgr._cleanup_server("k")
        assert "k" not in mgr.servers


# ============================================================================
# DockerMCPManager
# ============================================================================


class TestDockerMCPManager:
    def test_init(self):
        mgr = DockerMCPManager()
        assert mgr.container_prefix == "mcp-"
        assert mgr.managed_label == "mcp.managed=true"

    def test_parse_docker_output_empty(self):
        mgr = DockerMCPManager()
        assert mgr._parse_docker_output("") == []

    def test_parse_docker_output_valid_json(self):
        mgr = DockerMCPManager()
        line = json.dumps(
            {
                "ID": "abc123",
                "Names": "mcp-test",
                "Image": "ghcr.io/test",
                "Status": "Up 2 hours",
                "State": "running",
                "CreatedAt": "2024-01-01",
                "Ports": "8080/tcp",
                "Mounts": "",
                "Networks": "bridge",
                "Labels": "mcp.managed=true",
            }
        )
        result = mgr._parse_docker_output(line)
        assert len(result) == 1
        assert result[0]["id"] == "abc123"
        assert result[0]["name"] == "mcp-test"
        assert result[0]["managed"] is True

    def test_parse_docker_output_multiple_lines(self):
        mgr = DockerMCPManager()
        lines = "\n".join(
            [
                json.dumps({"ID": "a", "Names": "mcp-1"}),
                json.dumps({"ID": "b", "Names": "mcp-2"}),
            ]
        )
        result = mgr._parse_docker_output(lines)
        assert len(result) == 2

    def test_parse_docker_output_invalid_json_skipped(self):
        mgr = DockerMCPManager()
        mixed = "not json\n" + json.dumps({"ID": "ok", "Names": "mcp-ok"})
        result = mgr._parse_docker_output(mixed)
        assert len(result) == 1
        assert result[0]["id"] == "ok"

    def test_parse_container_line_non_dict(self):
        mgr = DockerMCPManager()
        assert mgr._parse_container_line('"just a string"') is None

    def test_parse_container_line_valid(self):
        mgr = DockerMCPManager()
        line = json.dumps({"ID": "x", "Names": "mcp-x", "Image": "img"})
        result = mgr._parse_container_line(line)
        assert result is not None
        assert result["id"] == "x"
        assert result["image"] == "img"

    @pytest.mark.asyncio
    async def test_list_containers_docker_not_found(self):
        mgr = DockerMCPManager()
        with patch.object(mgr, "_execute_docker_list_command", side_effect=FileNotFoundError):
            with pytest.raises(DockerCommandError, match="Docker not found"):
                await mgr.list_managed_containers()

    @pytest.mark.asyncio
    async def test_list_containers_os_error(self):
        mgr = DockerMCPManager()
        with patch.object(mgr, "_execute_docker_list_command", side_effect=OSError("fail")):
            with pytest.raises(DockerCommandError, match="Docker execution failed"):
                await mgr.list_managed_containers()

    @pytest.mark.asyncio
    async def test_get_container_logs_empty_name(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="non-empty string"):
            await mgr.get_container_logs("")

    @pytest.mark.asyncio
    async def test_get_container_logs_none_name(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="non-empty string"):
            # Intentional bad type — verifies runtime guard.
            await mgr.get_container_logs(cast("Any", None))

    @pytest.mark.asyncio
    async def test_get_container_logs_zero_lines(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="positive integer"):
            await mgr.get_container_logs("mcp-test", lines=0)

    @pytest.mark.asyncio
    async def test_get_container_logs_negative_lines(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="positive integer"):
            await mgr.get_container_logs("mcp-test", lines=-5)

    @pytest.mark.asyncio
    async def test_get_container_logs_docker_not_found(self):
        mgr = DockerMCPManager()
        with patch.object(mgr, "_execute_docker_logs_command", side_effect=FileNotFoundError):
            with pytest.raises(DockerCommandError, match="Docker not found"):
                await mgr.get_container_logs("mcp-test")

    @pytest.mark.asyncio
    async def test_get_container_logs_os_error(self):
        mgr = DockerMCPManager()
        with patch.object(mgr, "_execute_docker_logs_command", side_effect=OSError("fail")):
            with pytest.raises(DockerCommandError, match="Docker logs execution failed"):
                await mgr.get_container_logs("mcp-test")

    @pytest.mark.asyncio
    async def test_execute_docker_list_command_success(self):
        mgr = DockerMCPManager()
        container_json = json.dumps({"ID": "abc", "Names": "mcp-test"})
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(container_json.encode(), b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await mgr._execute_docker_list_command()
        assert len(result) == 1
        assert result[0]["id"] == "abc"

    @pytest.mark.asyncio
    async def test_execute_docker_list_command_failure(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"error message"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(DockerCommandError, match="Docker ps failed"):
                await mgr._execute_docker_list_command()

    @pytest.mark.asyncio
    async def test_execute_docker_logs_command_success(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"log line 1\nlog line 2\n", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await mgr._execute_docker_logs_command("mcp-test", 50)
        assert "log line 1" in result

    @pytest.mark.asyncio
    async def test_execute_docker_logs_container_not_found(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"No such container: mcp-test"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(ContainerNotFoundError, match="not found"):
                await mgr._execute_docker_logs_command("mcp-test", 50)

    @pytest.mark.asyncio
    async def test_execute_docker_logs_other_error(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"some docker error"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(DockerCommandError, match="Docker logs failed"):
                await mgr._execute_docker_logs_command("mcp-test", 50)


# ============================================================================
# mcp_config_resolver — _handle_preset_config
# ============================================================================


class TestHandlePresetConfig:
    def test_existing_preset(self, tmp_path, monkeypatch):
        preset_dir = tmp_path / "mcp_configs"
        preset_dir.mkdir()
        (preset_dir / "github_agent_config.json").write_text("{}")
        monkeypatch.chdir(tmp_path)
        result = _handle_preset_config("github", "worker1")
        assert result is not None
        assert "github_agent_config.json" in result

    def test_missing_preset(self):
        result = _handle_preset_config("nonexistent_xyz_abc", "worker1")
        assert result is None


# ============================================================================
# mcp_config_resolver — _handle_server_filtering
# ============================================================================


class TestHandleServerFiltering:
    def test_filters_requested_servers(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        cfg = {
            "mcpServers": {
                "github": {"url": "http://gh/sse"},
                "kanban": {"url": "http://kb/sse"},
                "slack": {"url": "http://sl/sse"},
            }
        }
        (config_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        monkeypatch.chdir(tmp_path)

        result = _handle_server_filtering(["github", "kanban"], "w")
        assert result is not None
        assert "github" in result["mcpServers"]
        assert "kanban" in result["mcpServers"]
        assert "slack" not in result["mcpServers"]

    def test_nonexistent_servers_return_none(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        cfg = {"mcpServers": {"github": {"url": "http://gh/sse"}}}
        (config_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        monkeypatch.chdir(tmp_path)
        assert _handle_server_filtering(["nonexistent"], "w") is None

    def test_partial_match(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        cfg = {
            "mcpServers": {
                "github": {"url": "http://gh/sse"},
                "kanban": {"url": "http://kb/sse"},
            }
        }
        (config_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        monkeypatch.chdir(tmp_path)
        result = _handle_server_filtering(["github", "nonexistent"], "w")
        assert result is not None
        assert "github" in result["mcpServers"]
        assert len(result["mcpServers"]) == 1

    def test_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _handle_server_filtering(["github"], "w") is None

    def test_invalid_json_config_file(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        (config_dir / "github_agent_config.json").write_text("{bad json")
        monkeypatch.chdir(tmp_path)
        assert _handle_server_filtering(["github"], "w") is None

    def test_falls_through_config_paths(self, tmp_path, monkeypatch):
        """When first config path has bad JSON, falls to next available."""
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        (config_dir / "github_agent_config.json").write_text("{bad json")
        cfg = {"mcpServers": {"github": {"url": "http://gh/sse"}}}
        (config_dir / "default_mcp_config.json").write_text(json.dumps(cfg))
        monkeypatch.chdir(tmp_path)
        result = _handle_server_filtering(["github"], "w")
        assert result is not None
        assert "github" in result["mcpServers"]


# ============================================================================
# mcp_config_resolver — _normalize_mcp_config_for_environment
# ============================================================================


class TestNormalizeMCPConfigForEnvironment:
    def test_no_kanban_url_leaves_config_unchanged(self, monkeypatch):
        monkeypatch.delenv("KANBAN_MCP_URL", raising=False)
        cfg = {"mcpServers": {"kanban": {"command": "node", "args": ["kanban-mcp"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=False)
        assert result["mcpServers"]["kanban"]["command"] == "node"

    def test_kanban_command_converted_to_http(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban-mcp:1337/sse")
        cfg = {
            "mcpServers": {
                "kanban": {
                    "command": "node",
                    "args": ["/path/to/kanban-mcp/index.js"],
                    "log_level": "DEBUG",
                }
            }
        }
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        kanban = result["mcpServers"]["kanban"]
        assert kanban["url"] == "http://kanban-mcp:1337/sse"
        assert "command" not in kanban
        assert kanban["log_level"] == "DEBUG"
        assert kanban["timeout"] == 30.0

    def test_custom_timeout(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        monkeypatch.setenv("KANBAN_MCP_TIMEOUT", "60")
        cfg = {"mcpServers": {"kanban": {"command": "node", "args": ["kanban-mcp/x.js"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["timeout"] == 60.0

    def test_invalid_timeout_falls_back(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        monkeypatch.setenv("KANBAN_MCP_TIMEOUT", "not-a-number")
        cfg = {"mcpServers": {"kanban": {"command": "node", "args": ["kanban-mcp/x.js"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["timeout"] == 30.0

    def test_url_based_config_not_modified(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {"mcpServers": {"kanban": {"url": "http://already-http/sse"}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["url"] == "http://already-http/sse"

    def test_non_kanban_command_not_modified(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {"mcpServers": {"github": {"command": "node", "args": ["github-mcp/index.js"]}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["github"]["command"] == "node"

    def test_non_dict_mcpServers(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {"mcpServers": "not-a-dict"}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result == {"mcpServers": "not-a-dict"}

    def test_string_args(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {"mcpServers": {"kanban": {"command": "node", "args": "kanban-mcp/index.js"}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["url"] == "http://kanban:1337/sse"

    def test_non_dict_server_cfg_entry(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {
            "mcpServers": {
                "bad": "not-a-dict",
                "kanban": {"command": "node", "args": ["kanban-mcp"]},
            }
        }
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        # bad entry should remain as-is; kanban should be converted
        assert result["mcpServers"]["bad"] == "not-a-dict"
        assert result["mcpServers"]["kanban"]["url"] == "http://kanban:1337/sse"

    def test_tuple_args(self, monkeypatch):
        monkeypatch.setenv("KANBAN_MCP_URL", "http://kanban:1337/sse")
        cfg = {"mcpServers": {"kanban": {"command": "node", "args": ("kanban-mcp", "index.js")}}}
        result = _normalize_mcp_config_for_environment(cfg, running_in_docker=True)
        assert result["mcpServers"]["kanban"]["url"] == "http://kanban:1337/sse"


# ============================================================================
# mcp_config_resolver — resolve_mcp_config
# ============================================================================


class TestResolveMCPConfig:
    @pytest.mark.asyncio
    async def test_direct_config_dict_has_priority(self):
        cfg = {
            "mcpServers": {
                "s": {"url": "http://inference-router.inference-router.svc.cluster.local:8150/sse"}
            }
        }
        result = await resolve_mcp_config(mcp_config=cfg, worker_name="w")
        assert result is cfg

    @pytest.mark.asyncio
    async def test_direct_config_str_has_priority(self):
        from src.security.request_overrides import McpConfigNotAllowedError

        with pytest.raises(McpConfigNotAllowedError):
            await resolve_mcp_config(mcp_config="/path/to/config.json")

    @pytest.mark.asyncio
    async def test_empty_dict_returns_none(self):
        result = await resolve_mcp_config(mcp_config={}, worker_name="w")
        assert result is None

    @pytest.mark.asyncio
    async def test_preset_fallback(self, tmp_path, monkeypatch):
        preset_dir = tmp_path / "mcp_configs"
        preset_dir.mkdir()
        (preset_dir / "github_agent_config.json").write_text("{}")
        monkeypatch.chdir(tmp_path)
        result = await resolve_mcp_config(mcp_config_preset="github", worker_name="w")
        assert result is not None
        assert "github_agent_config.json" in result

    @pytest.mark.asyncio
    async def test_preset_not_found_falls_to_none(self):
        result = await resolve_mcp_config(mcp_config_preset="nonexistent_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_server_filtering_fallback(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()
        cfg = {"mcpServers": {"github": {"url": "http://gh/sse"}}}
        (config_dir / "github_agent_config.json").write_text(json.dumps(cfg))
        monkeypatch.chdir(tmp_path)
        result = await resolve_mcp_config(mcp_servers=["github"], worker_name="w")
        assert isinstance(result, dict)
        assert "github" in result["mcpServers"]

    @pytest.mark.asyncio
    async def test_all_none_returns_none(self):
        result = await resolve_mcp_config(worker_name="w")
        assert result is None


# ============================================================================
# mcp_config_resolver — _resolve_mcp_servers_for_cli_agents
# ============================================================================


class TestResolveMCPServersForCLIAgents:
    @pytest.mark.asyncio
    async def test_none_returns_none(self):
        assert await _resolve_mcp_servers_for_cli_agents(mcp_servers=None) is None

    @pytest.mark.asyncio
    async def test_empty_list_returns_none(self):
        assert await _resolve_mcp_servers_for_cli_agents(mcp_servers=[]) is None

    @pytest.mark.asyncio
    async def test_url_server_resolved(self):
        filtered = {"mcpServers": {"github": {"url": "http://gh:8080/sse"}}}
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["github"], worker_name="w"
            )
        assert result is not None
        assert len(result) == 1
        assert result[0]["url"] == "http://gh:8080/sse"

    @pytest.mark.asyncio
    async def test_command_server_resolved(self):
        filtered = {
            "mcpServers": {
                "local": {"command": "node", "args": ["srv.js"], "env": {"K": "V"}},
            }
        }
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["local"], worker_name="w"
            )
        assert result is not None
        assert result[0]["command"] == "node"
        assert result[0]["args"] == ["srv.js"]
        assert result[0]["env"] == {"K": "V"}

    @pytest.mark.asyncio
    async def test_kanban_docker_url_rewritten(self):
        filtered = {"mcpServers": {"kanban": {"url": "http://kanban-mcp:1337/sse"}}}
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["kanban"], worker_name="w"
            )
        assert result is not None
        assert result[0]["url"] == "http://localhost:1337/sse"

    @pytest.mark.asyncio
    async def test_no_config_returns_none(self):
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["github"], worker_name="w"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_config_is_string_returns_none(self):
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value="/path/to/config.json",
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["github"], worker_name="w"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_server_with_no_url_or_command_skipped(self):
        filtered = {"mcpServers": {"broken": {"env": {"K": "V"}}}}
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["broken"], worker_name="w"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_server_not_in_config_skipped(self):
        filtered = {"mcpServers": {"github": {"url": "http://gh/sse"}}}
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["nonexistent"], worker_name="w"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_mcp_servers_dict_returns_none(self):
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value={"mcpServers": {}},
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["github"], worker_name="w"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_mixed_url_and_command_servers(self):
        filtered = {
            "mcpServers": {
                "github": {"url": "http://gh:8080/sse"},
                "local": {"command": "node", "args": ["srv.js"], "env": {}},
            }
        }
        with patch(
            "src.worker.factory.mcp_config_resolver.resolve_mcp_config",
            new_callable=AsyncMock,
            return_value=filtered,
        ):
            result = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=["github", "local"], worker_name="w"
            )
        assert result is not None
        assert len(result) == 2
