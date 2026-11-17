"""Updated MCP utilities with Docker container reuse and management."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx2 as httpx
from pydantic_ai.mcp import FastMCPClient, MCPToolset
from pydantic_ai.toolsets import WrapperToolset

from src.security.redact import (
    is_sensitive_name,
    redact_args,
    redact_auth_header_content,
    redact_url_userinfo,
)

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages MCP server lifecycle with Docker container management."""

    def __init__(self) -> None:
        """Initialize the MCP server manager."""
        self.servers = {}
        self.server_locks = {}

    async def start_mcp_server(self, server_config: dict[str, Any], tool_prefix: str = "") -> Any:
        """Start an MCP server and ensure it's ready."""
        server_key = f"{server_config.get('command', '')}_{tool_prefix}"

        if server_key not in self.server_locks:
            self.server_locks[server_key] = asyncio.Lock()

        async with self.server_locks[server_key]:
            if server_key in self.servers:
                server = self.servers[server_key]
                if await self._is_server_healthy(server):
                    return server
                else:
                    await self._cleanup_server(server_key)

            # Create and start new server
            server = await self._create_server(server_config, tool_prefix)

            # If it's a managed Docker server, ensure container is running
            if hasattr(server, "_mcp_managed") and server._mcp_managed:
                await self._ensure_container_running(server)

            await self._wait_for_server_ready(server)
            self.servers[server_key] = server
            return server

    async def _ensure_container_running(self, server: Any) -> None:
        """Ensure the managed Docker container is running and update server args."""
        try:
            server_name = getattr(server, "_mcp_server_name", "unknown")
            image = getattr(server, "_mcp_image", None)
            getattr(server, "_mcp_env_vars", {})

            if not image:
                logger.error(f"No image specified for managed server {server_name}")
                return

            logger.info(
                f"Ensuring container is running for server {server_name} with image {image}"
            )

            # Update the server's command to use the managed container
            container_name = f"mcp-{server_name}-managed"
            logger.info(f"Container {container_name} ready for MCP server {server_name}")
            server.args = ["exec", "-i", container_name, "/entrypoint"]
            logger.info(f"Updated server args for {server_name}: {server.args}")

        except Exception:
            logger.exception(
                f"Error ensuring container for server {getattr(server, '_mcp_server_name', 'unknown')}"
            )
            raise

    async def _create_server(self, server_config: dict[str, Any], tool_prefix: str) -> Any:  # noqa: ARG002, RUF100
        """Create MCP server instance from configuration."""
        # Your existing server creation logic here
        # But with proper error handling and logging
        # NEVER log server_config raw -- it can carry a url with userinfo
        # creds, an Authorization/Bearer header, or an env/args secret. Log
        # only identifying, non-secret shape.
        if "url" in server_config:
            transport = "http"
        elif "command" in server_config:
            transport = "stdio"
        else:
            transport = "invalid"
        logger.info(
            "Creating MCP server '%s': transport=%s keys=%s",
            server_config.get("name", "unnamed"),
            transport,
            sorted(server_config.keys()),
        )

        if "url" in server_config:
            return create_http_server_sync(server_config.get("name", "unnamed"), server_config)
        elif "command" in server_config:
            return create_stdio_server(server_config.get("name", "unnamed"), server_config, None)
        else:
            # Same rule as the log line above: never fold the raw config into
            # a message string -- exception text is exactly as loggable as a
            # logger call (str(exc), exc_info, an upstream handler's
            # logger.exception(...)), so this is the same leak vector.
            raise Exception(
                f"Invalid server config for '{server_config.get('name', 'unnamed')}': "
                f"keys={sorted(server_config.keys())} (must have 'url' or 'command')"
            )

    async def _wait_for_server_ready(self, server: Any, timeout: float = 30) -> None:
        """Wait for MCP server to be ready with health checks."""
        start_time = asyncio.get_running_loop().time()

        while asyncio.get_running_loop().time() - start_time < timeout:
            try:
                # Try to ping the server or check if it's responsive
                if await self._is_server_healthy(server):
                    logger.info(f"MCP server ready: {server}")
                    return
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Waiting for MCP server to be ready: {e}")
                await asyncio.sleep(1)

        raise Exception(f"MCP server failed to become ready within {timeout} seconds")

    async def _is_server_healthy(self, server: Any) -> bool:
        """Check if MCP server is healthy."""
        try:
            # For managed Docker servers, check container health
            if hasattr(server, "_mcp_managed") and server._mcp_managed:
                server_name = getattr(server, "_mcp_server_name", "unknown")
                container_name = f"mcp-{server_name}-managed"
                logger.debug(f"Checking health for managed container: {container_name}")

            # For regular servers
            if hasattr(server, "is_running"):
                return bool(await server.is_running())

            # Basic check - see if server object exists and has expected attributes
            return server is not None and hasattr(server, "command")

        except Exception as e:
            logger.warning(f"MCP server health check failed: {e}")
            return False

    async def _cleanup_server(self, server_key: str) -> None:
        """Clean up an MCP server and optionally its Docker container."""
        if server_key in self.servers:
            server = self.servers[server_key]
            try:
                # If it's a managed Docker server, optionally keep container running
                # or clean it up based on policy
                if hasattr(server, "_mcp_managed") and server._mcp_managed:
                    # For now, keep containers running for reuse
                    # Could add cleanup logic here if needed
                    pass
                elif hasattr(server, "stop"):
                    await server.stop()
                elif hasattr(server, "cleanup"):
                    await server.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up MCP server {server_key}: {e}")
            finally:
                del self.servers[server_key]


class McpServerToolset(WrapperToolset[Any]):
    """An MCP toolset that still carries the server entry it was built from.

    pydantic-ai 2 removed `MCPServerSSE`/`MCPServerStdio`, which were the public
    objects callers (and the tests) read `url` / `command` / `args` / `env` /
    `tool_prefix` / `timeout` off. This wrapper keeps that surface while
    delegating tool discovery and calls to the real toolset, and re-exposes
    `is_running` because `WrapperToolset` drops it.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        name: str,
        server_config: dict[str, Any],
        tool_prefix: str | None = None,
        timeout: float | None = None,
        managed: bool = False,
        log_level: str | None = None,
        image: str | None = None,
        read_timeout: float | None = None,
        inner: Any | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.name = name
        self.server_config = server_config
        self.tool_prefix = tool_prefix
        self.timeout = timeout
        self.managed = managed
        self.log_level = log_level
        self.image = image
        self.read_timeout = read_timeout
        self._inner = inner if inner is not None else wrapped

    @property
    def _mcp_managed(self) -> bool:
        """Whether this server runs in a container this service manages.

        Kept as an underscore-prefixed property because `MCPServerManager` and the
        docker lifecycle code branch on it, and that predates this port.
        """
        return self.managed

    @property
    def url(self) -> str | None:
        """The configured URL, for HTTP/SSE servers."""
        value = self.server_config.get("url")
        return value if isinstance(value, str) else None

    @property
    def command(self) -> str | None:
        """The configured command, for stdio servers."""
        value = self.server_config.get("command")
        return value if isinstance(value, str) else None

    @property
    def args(self) -> list[str]:
        """The configured command arguments, for stdio servers."""
        value = self.server_config.get("args", [])
        return list(value) if isinstance(value, list) else []

    @property
    def _mcp_image(self) -> str | None:
        """The container image, for managed docker servers (legacy attribute name)."""
        return self.image

    @property
    def cwd(self) -> str | None:
        """The configured working directory, for stdio servers."""
        value = self.server_config.get("cwd")
        return value if isinstance(value, str) else None

    @property
    def headers(self) -> dict[str, str]:
        """The configured HTTP headers, for HTTP/SSE servers."""
        value = self.server_config.get("headers", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def env(self) -> dict[str, str]:
        """The configured environment, for stdio servers."""
        value = self.server_config.get("env", {})
        return dict(value) if isinstance(value, dict) else {}

    async def is_running(self) -> bool:
        """Whether the underlying MCP session is open."""
        checker = getattr(self._inner, "is_running", None)
        return bool(await checker()) if checker is not None else False


def _build_toolset(
    name: str,
    server_config: dict[str, Any],
    *,
    tool_prefix: str | None = None,
    timeout: float | None = None,
    client_timeout: float | None = None,
    managed: bool = False,
    log_level: str | None = None,
    image: str | None = None,
    read_timeout: float | None = None,
) -> McpServerToolset:
    """Build a pydantic-ai 2 toolset for one MCP server entry.

    A server is an `MCPToolset` wrapping a fastmcp `Client` built from an
    `mcpServers` config fragment; tool prefixing is a wrapper rather than a
    field. Returns `McpServerToolset` so the address/transport details stay
    reachable to callers and the lifecycle manager.
    """
    client_kwargs: dict[str, Any] = {}
    effective = client_timeout if client_timeout is not None else timeout
    if effective is not None:
        client_kwargs["timeout"] = effective
    client = FastMCPClient({"mcpServers": {name: server_config}}, **client_kwargs)
    toolset_kwargs: dict[str, Any] = {}
    if log_level is not None:
        toolset_kwargs["log_level"] = log_level
    toolset: Any = MCPToolset(client=client, **toolset_kwargs)
    wrapped = toolset.prefixed(tool_prefix) if tool_prefix else toolset
    return McpServerToolset(
        wrapped,
        name=name,
        server_config=server_config,
        tool_prefix=tool_prefix,
        timeout=timeout,
        managed=managed,
        log_level=log_level,
        image=image,
        read_timeout=read_timeout,
        inner=toolset,
    )


def create_mcp_servers_from_config(  # noqa: PLR0912 -- dispatches over MCP transport variants (stdio/sse/http) × config-source shape
    config_source: str | dict[str, Any],
    base_dir_for_resolution: Path | None = None,
) -> list[Any]:
    """Create MCP servers synchronously from config file or dictionary.

    This function replaces the async load_mcp_servers for use in agent initialization.

    Args:
        config_source: Path to MCP configuration file (str) or the configuration itself (dict).
        base_dir_for_resolution: Base directory to resolve relative paths for 'cwd' in Stdio servers.

    Returns:
        List of MCP toolsets
    """
    config: dict[str, Any] | None = None

    # Validate input type early
    if isinstance(config_source, list):
        logger.error(
            f"Invalid config_source type: received list {config_source}. "
            "Expected file path (str) or configuration dictionary. "
            "This often happens when mcp_servers list is passed directly instead of a proper config."
        )
        return []

    # Load configuration
    if isinstance(config_source, str):
        config_file_path = Path(config_source).resolve()
        if not base_dir_for_resolution:
            base_dir_for_resolution = config_file_path.parent
        try:
            with open(config_file_path) as f:
                config = json.load(f)
            logger.info(f"Loaded MCP configuration from {config_file_path}")
        except FileNotFoundError:
            logger.exception(f"MCP configuration file not found: {config_file_path}")
            return []
        except json.JSONDecodeError:
            logger.exception(f"Error decoding MCP configuration file {config_file_path}")
            return []
        except Exception:
            logger.exception(f"Failed to load MCP config from {config_file_path}")
            return []
    elif isinstance(config_source, dict):
        config = config_source
        if not base_dir_for_resolution:
            base_dir_for_resolution = Path.cwd()
        logger.info("Loading MCP configuration from provided dictionary.")
    else:
        logger.error(
            f"Invalid config_source type for MCP servers: {type(config_source)}. "
            "Must be path (str) or dict."
        )
        return []

    if not config or "mcpServers" not in config:
        logger.warning("No 'mcpServers' key found in the configuration.")
        return []

    # Validate that mcpServers is a dictionary
    mcp_servers_config = config.get("mcpServers", {})
    if not isinstance(mcp_servers_config, dict):
        logger.error(f"'mcpServers' must be a dictionary, got {type(mcp_servers_config)}")
        return []

    servers: list[Any] = []
    successful_servers = 0
    failed_servers = 0

    for name, server_config in mcp_servers_config.items():
        try:
            # Validate server configuration
            if not validate_server_config(name, server_config):
                failed_servers += 1
                continue

            # Create appropriate server type
            server: Any | None = None
            if "url" in server_config:
                # HTTP-based servers
                server = create_http_server_sync(name, server_config)
                if server:
                    servers.append(server)
                    successful_servers += 1
                else:
                    failed_servers += 1
            elif "command" in server_config:
                # Command-based (Stdio) servers
                server = create_stdio_server(name, server_config, base_dir_for_resolution)
                if server:
                    servers.append(server)
                    successful_servers += 1
                else:
                    failed_servers += 1

        except Exception as e:
            logger.error(
                f"Unexpected error processing MCP server '{name}': {e!s}",
                exc_info=True,
            )
            failed_servers += 1

    logger.info(
        f"MCP server creation completed: {successful_servers} successful, {failed_servers} failed"
    )
    if failed_servers > 0:
        logger.warning(f"{failed_servers} MCP servers failed to create - check logs for details")

    return servers


def create_http_server_sync(name: str, server_config: dict[str, Any]) -> Any | None:
    """Create an HTTP MCP server synchronously (without auth for now)."""
    try:
        url = server_config.get("url")
        if not url:
            logger.error(f"HTTP MCP server '{name}' missing required 'url' field")
            return None

        logger.info(f"Creating HTTP MCP server '{name}' with URL: {redact_url_userinfo(url)}")
        return _build_toolset(name, {"url": url})

    except Exception:
        logger.exception(f"Failed to create HTTP MCP server '{name}'")
        return None


async def get_auth_token(auth_config: dict[str, Any]) -> str | None:
    """Get authentication token for MCP server with improved error handling.

    Args:
        auth_config: Authentication configuration dict

    Returns:
        Authentication token string or None if failed
    """
    if not auth_config.get("enabled", False):
        return None

    try:
        timeout_config = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            token_url = auth_config.get("token_url")
            if not token_url:
                logger.error("Auth enabled but token_url is missing.")
                return None

            payload = {
                "grant_type": auth_config.get("grant_type"),
                "client_id": auth_config.get("client_id"),
                "client_secret": auth_config.get("client_secret"),
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            response = await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": auth_config.get("content_type", "application/json")},
            )
            response.raise_for_status()
            data = response.json()
            token = data.get(auth_config.get("token_property", "access_token"))
            if not token:
                # NEVER log the response body: a token endpoint's JSON can
                # carry access_token/refresh_token/id_token under a key other
                # than the configured token_property. Key names only.
                data_keys = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
                logger.error(
                    f"Token property '{auth_config.get('token_property', 'access_token')}' "
                    f"not found in auth response from {redact_url_userinfo(str(token_url))}. "
                    f"Response keys: {data_keys}"
                )
                return None
            logger.info(
                f"Successfully obtained auth token from {redact_url_userinfo(str(token_url))}"
            )
            return str(token)
    except httpx.HTTPStatusError as e:
        # NEVER log e.response.text: some token endpoints echo request
        # params (including client_secret) back in error bodies.
        logger.exception(
            f"HTTP Status Error while getting auth token from "
            f"{redact_url_userinfo(str(e.request.url))}: {e.response.status_code} "
            f"(body omitted, {len(e.response.content)} bytes)"
        )
        return None
    except httpx.ConnectError as e:
        logger.exception(
            f"Connection Error: Failed to connect to auth token URL "
            f"{redact_url_userinfo(str(e.request.url))}. Error"
        )
        return None
    except httpx.RequestError as e:
        logger.exception(
            f"Request Error while getting auth token from {redact_url_userinfo(str(e.request.url))}"
        )
        return None
    except json.JSONDecodeError:
        # NEVER log the raw response text -- same rationale as above.
        response_len = len(response.text) if "response" in locals() else None
        logger.exception(
            f"JSON Decode Error while parsing auth token response from "
            f"{redact_url_userinfo(str(auth_config.get('token_url')))}. "
            f"Response length: {response_len if response_len is not None else 'N/A'}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error getting auth token from "
            f"{redact_url_userinfo(str(auth_config.get('token_url', 'N/A')))}: {e!s}",
            exc_info=True,
        )
        return None


def validate_server_config(name: str, server_config: dict[str, Any]) -> bool:
    """Validate MCP server configuration.

    Args:
        name: Server name
        server_config: Server configuration dict

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(server_config, dict):
        logger.error(f"MCP server '{name}' config must be a dictionary")
        return False

    has_url = "url" in server_config
    has_command = "command" in server_config

    if not has_url and not has_command:
        logger.error(f"MCP server '{name}' must have either 'url' or 'command' defined")
        return False

    if has_url and has_command:
        logger.warning(f"MCP server '{name}' has both 'url' and 'command' defined, using 'url'")

    # Validate timeout if present
    if "timeout" in server_config:
        try:
            timeout = float(server_config["timeout"])
            if timeout <= 0:
                logger.error(f"MCP server '{name}' timeout must be positive")
                return False
        except (ValueError, TypeError):  # fmt: skip
            logger.exception(f"MCP server '{name}' timeout must be a number")
            return False

    return True


async def create_http_server(name: str, server_config: dict[str, Any]) -> Any | None:
    """Create an HTTP MCP toolset with proper error handling.

    Args:
        name: Server name
        server_config: Server configuration dict

    Returns:
        Toolset instance or None if failed
    """
    try:
        headers = server_config.get("headers", {})
        auth_config = server_config.get("auth", {})

        # Handle authentication if enabled
        if auth_config.get("enabled", False):
            token = await get_auth_token(auth_config)
            if token:
                auth_header_name = auth_config.get("auth_header_name", "Authorization")
                auth_scheme = auth_config.get("auth_header", "Bearer")
                headers[auth_header_name] = f"{auth_scheme} {token}"
            else:
                logger.warning(f"Proceeding without auth token for MCP server {name}")

        timeout_seconds = float(server_config.get("timeout", 15))
        read_timeout_config = server_config.get("read_timeout")
        if read_timeout_config is None:
            read_timeout_config = server_config.get("sse_read_timeout", 15)
        read_timeout_seconds = (
            float(read_timeout_config) if read_timeout_config is not None else None
        )
        # fastmcp's client exposes one `timeout`, not a separate read timeout. Take
        # the longer of the two so a slow SSE stream keeps the budget it asked for.
        client_timeout = max(timeout_seconds, read_timeout_seconds or 0.0)

        # Build the fastmcp `mcpServers` entry. pydantic-ai 2 takes headers and
        # timeouts on the client config, not on the server object.
        server_entry: dict[str, Any] = {"url": server_config["url"]}
        if headers:
            server_entry["headers"] = headers

        tool_prefix: str | None = None
        if "tool_prefix" in server_config:
            prefix = server_config["tool_prefix"]
            if isinstance(prefix, str) and prefix.strip():
                tool_prefix = prefix.strip()
            else:
                logger.warning(f"Invalid tool_prefix for server {name}, ignoring")

        if server_config.get("http_client"):
            # pydantic-ai 2 builds the HTTP transport inside fastmcp's Client and
            # gives no seam for our own httpx.AsyncClient (proxy/TLS tuning).
            # No shipped preset uses it; surfaced rather than silently dropped.
            logger.warning(
                "MCP server %s declares http_client options; pydantic-ai 2 cannot "
                "accept a custom httpx client, using the default transport",
                name,
            )

        log_level: str | None = None
        if "log_level" in server_config:
            candidate = server_config["log_level"]
            if isinstance(candidate, str) and candidate.upper() in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]:
                log_level = candidate
            else:
                logger.warning(f"Invalid log_level for server {name}, ignoring")

        http_server = _build_toolset(
            name,
            server_entry,
            tool_prefix=tool_prefix,
            timeout=timeout_seconds,
            client_timeout=client_timeout,
            log_level=log_level,
            read_timeout=read_timeout_seconds,
        )
        prefix_info = f"with tool_prefix='{tool_prefix}'" if tool_prefix else "without tool_prefix"
        logger.info(
            f"Created HTTP MCP server: {name} at "
            f"{redact_url_userinfo(server_config['url'])} {prefix_info}"
        )
        return http_server

    except Exception as e:
        logger.error(f"Failed to create HTTP MCP server '{name}': {e!s}", exc_info=True)
        return None


def expand_environment_variables(args: list[Any]) -> list[str]:
    """Expand environment variables in command arguments.

    Args:
        args: List of command arguments

    Returns:
        List of expanded arguments as strings
    """
    expanded_args = []
    for arg in args:
        if isinstance(arg, str):
            # Handle environment variable expansion: ${VAR}
            expanded_arg = arg
            for match in re.finditer(r"\$\{([^}]+)\}", arg):
                var_name = match.group(1)
                var_value = os.environ.get(var_name, "")
                expanded_arg = expanded_arg.replace(f"${{{var_name}}}", var_value)
            expanded_args.append(expanded_arg)
        else:
            expanded_args.append(str(arg))  # Convert to string for consistency
    return expanded_args


def resolve_working_directory(cwd_str: str | None, base_dir: Path | None) -> str | None:
    """Resolve working directory path.

    Args:
        cwd_str: Working directory string from config
        base_dir: Base directory for relative path resolution

    Returns:
        Resolved absolute path string or None
    """
    if not cwd_str:
        return None

    try:
        path_obj = Path(cwd_str)
        if path_obj.is_absolute():
            resolved_path = path_obj.resolve()
        elif base_dir:
            resolved_path = (base_dir / path_obj).resolve()
        else:
            resolved_path = Path(cwd_str).resolve()

        # Verify the path exists
        if not resolved_path.exists():
            logger.warning(f"Working directory does not exist: {resolved_path}")

        return str(resolved_path)
    except Exception:
        logger.exception(f"Error resolving working directory '{cwd_str}'")
        return None


def create_stdio_server(
    name: str, server_config: dict[str, Any], base_dir: Path | None
) -> Any | None:
    """Create a stdio MCP toolset with Docker container reuse."""
    try:
        # Check if this is a Docker-based server that we can manage
        command = server_config["command"]
        args = server_config.get("args", [])

        if command == "docker" and args and args[0] == "run":
            # This is a docker run command - use managed containers
            return _create_managed_docker_server(name, server_config)
        else:
            # Regular command - use original logic
            return _create_regular_stdio_server(name, server_config, base_dir)

    except Exception as e:
        logger.error(f"Failed to create stdio MCP server '{name}': {e!s}", exc_info=True)
        return None


def _create_managed_docker_server(name: str, server_config: dict[str, Any]) -> Any | None:
    """Create a Docker-based MCP server using managed containers with container reuse."""
    try:
        args = server_config.get("args", [])
        env_vars = server_config.get("env", {})

        # Log the environment variables being processed (mask sensitive data)
        logger.info(f"Processing server config for {name}")
        logger.info(f"Args: {redact_args(args)}")
        if env_vars:
            masked_env = {}
            for k, v in env_vars.items():
                if is_sensitive_name(k):
                    masked_env[k] = "***masked***" if v else "***empty***"
                elif isinstance(v, str):
                    # Non-sensitive name doesn't mean non-sensitive value --
                    # e.g. DATABASE_URL can carry userinfo credentials, and an
                    # EXTRA_HEADERS-style env var can carry a literal
                    # Authorization/Bearer credential.
                    masked_env[k] = redact_auth_header_content(redact_url_userinfo(v))
                else:
                    masked_env[k] = v
            logger.info(f"Environment variables: {masked_env}")
        else:
            logger.warning(f"No environment variables found in server config for {name}")

        # Extract image name from args
        # Typical format: ["run", "-i", "--rm", "-e", "VAR", "image:tag"]
        image = None
        for _i, arg in enumerate(args):
            if not arg.startswith("-") and arg != "run" and "=" not in arg and "/" in arg:
                # This is likely the image name (contains / for registry path)
                image = arg
                break

        # Fallback: look for common image patterns
        if not image:
            for arg in args:
                if "ghcr.io/" in arg or "mcp/" in arg or arg.endswith("-server"):
                    image = arg
                    break

        if not image:
            logger.error(
                f"Could not determine Docker image for server {name}. Args: {redact_args(args)}"
            )
            return None

        # A GitHub MCP container without its token starts and then fails every
        # tool call; refuse at build time with a message naming the variable.
        if image == "ghcr.io/github/github-mcp-server" and not env_vars.get(
            "GITHUB_PERSONAL_ACCESS_TOKEN"
        ):
            logger.error(
                "GITHUB_PERSONAL_ACCESS_TOKEN is required but not found in env_vars for server %s",
                name,
            )
            logger.error("Available env vars: %s", list(env_vars.keys()))
            raise ValueError(
                f"GITHUB_PERSONAL_ACCESS_TOKEN is required for GitHub MCP server {name}"
            )

        # The message carries catalogue identifiers, not credentials: the taint CodeQL
        # follows is the adjacent env_vars dict, which is never included here.
        # lgtm[py/clear-text-logging-sensitive-data]
        logger.info(f"Creating managed Docker MCP server: {name} with image {image}")

        # pydantic-ai 2: one MCPToolset over a fastmcp stdio client. The
        # container is still managed here (name, reuse) — the toolset only
        # carries the `docker exec` transport.
        tool_prefix = server_config.get("tool_prefix")
        if isinstance(tool_prefix, str) and not tool_prefix.strip():
            logger.warning(f"Invalid tool_prefix for server {name}, ignoring")
            tool_prefix = None
        timeout_seconds: float | None = None
        if "timeout" in server_config:
            try:
                timeout_seconds = float(server_config["timeout"])
            except (ValueError, TypeError):  # fmt: skip
                logger.warning(f"Invalid timeout for server {name}, using default")

        stdio_server = _build_toolset(
            name,
            {
                "command": "docker",
                "args": ["exec", "-i", f"mcp-{name}-managed", "./github-mcp-server"],
                "env": env_vars,
            },
            tool_prefix=tool_prefix.strip() if isinstance(tool_prefix, str) else None,
            timeout=timeout_seconds,
            managed=True,
            image=image,
        )

        prefix_info = (
            f"with tool_prefix='{server_config.get('tool_prefix')}'"
            if server_config.get("tool_prefix")
            else "without tool_prefix"
        )
        logger.info(f"Created managed Docker MCP server: {name} {prefix_info}")
        return stdio_server

    except Exception as e:
        logger.error(
            f"Failed to create managed Docker MCP server '{name}': {e!s}",
            exc_info=True,
        )
        return None


def _create_regular_stdio_server(
    name: str, server_config: dict[str, Any], base_dir: Path | None
) -> Any | None:
    """Create a regular (non-Docker) stdio-based MCP server."""
    try:
        # Prepare environment variables
        env = {**os.environ}
        if "env" in server_config and isinstance(server_config["env"], dict):
            env.update(server_config["env"])

        # Expand environment variables in arguments
        args = expand_environment_variables(server_config.get("args", []))

        # Resolve working directory
        resolved_cwd = resolve_working_directory(server_config.get("cwd"), base_dir)

        # Build the fastmcp stdio entry. `cwd` and `env` live in the transport
        # config in pydantic-ai 2; the toolset itself carries no transport knobs.
        stdio_entry: dict[str, Any] = {
            "command": server_config["command"],
            "args": args,
            "env": env,
        }
        if resolved_cwd:
            stdio_entry["cwd"] = resolved_cwd

        tool_prefix: str | None = None
        if "tool_prefix" in server_config:
            prefix = server_config["tool_prefix"]
            if isinstance(prefix, str) and prefix.strip():
                tool_prefix = prefix.strip()
            else:
                logger.warning(f"Invalid tool_prefix for server {name}, ignoring")

        timeout_seconds: float | None = None
        if "timeout" in server_config:
            try:
                timeout_seconds = float(server_config["timeout"])
            except (ValueError, TypeError):  # fmt: skip
                logger.warning(f"Invalid timeout for server {name}, using default")

        log_level: str | None = None
        if "log_level" in server_config:
            candidate = server_config["log_level"]
            if isinstance(candidate, str) and candidate.upper() in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]:
                log_level = candidate
            else:
                logger.warning(f"Invalid log_level for server {name}, ignoring")

        stdio_server = _build_toolset(
            name,
            stdio_entry,
            tool_prefix=tool_prefix,
            timeout=timeout_seconds,
            log_level=log_level,
        )
        prefix_info = (
            f"with tool_prefix='{server_config.get('tool_prefix')}'"
            if server_config.get("tool_prefix")
            else "without tool_prefix"
        )
        logger.info(f"Created managed Docker MCP server: {name} {prefix_info}")
        return stdio_server

    except Exception as e:
        logger.error(
            f"Failed to create managed Docker MCP server '{name}': {e!s}",
            exc_info=True,
        )
        return None


async def load_mcp_servers(
    config_source: str | dict[str, Any],
    base_dir_for_resolution: Path | None = None,
) -> list[Any]:
    """Load MCP servers from config file or dictionary using new pydantic-ai dataclass format.

    This is the updated version with improved error handling and validation.

    Args:
        config_source: Path to MCP configuration file (str) or the configuration itself (dict).
        base_dir_for_resolution: Base directory to resolve relative paths for 'cwd' in Stdio servers.

    Returns:
        List of MCP toolsets
    """
    config: dict[str, Any] | None = None

    # Load configuration
    if isinstance(config_source, str):
        config_file_path = Path(config_source).resolve()
        if not base_dir_for_resolution:
            base_dir_for_resolution = config_file_path.parent
        try:
            with open(config_file_path) as f:
                config = json.load(f)
            logger.info(f"Loaded MCP configuration from {config_file_path}")
        except FileNotFoundError:
            logger.exception(f"MCP configuration file not found: {config_file_path}")
            return []
        except json.JSONDecodeError:
            logger.exception(f"Error decoding MCP configuration file {config_file_path}")
            return []
        except Exception:
            logger.exception(f"Failed to load MCP config from {config_file_path}")
            return []
    elif isinstance(config_source, dict):
        config = config_source
        if not base_dir_for_resolution:
            base_dir_for_resolution = Path.cwd()
        logger.info("Loading MCP configuration from provided dictionary.")
    else:
        logger.error("Invalid config_source type for MCP servers. Must be path (str) or dict.")
        return []

    if not config or "mcpServers" not in config:
        logger.warning("No 'mcpServers' key found in the configuration.")
        return []

    # Validate that mcpServers is a dictionary
    mcp_servers_config = config.get("mcpServers", {})
    if not isinstance(mcp_servers_config, dict):
        logger.error("'mcpServers' must be a dictionary")
        return []

    servers: list[Any] = []
    successful_servers = 0
    failed_servers = 0

    for name, server_config in mcp_servers_config.items():
        try:
            # Validate server configuration
            if not validate_server_config(name, server_config):
                failed_servers += 1
                continue

            # Create appropriate server type
            server: Any | None = None
            if "url" in server_config:
                # HTTP-based servers
                server = await create_http_server(name, server_config)
                if server:
                    servers.append(server)
                    successful_servers += 1
                else:
                    failed_servers += 1
            elif "command" in server_config:
                # Command-based (Stdio) servers
                server = create_stdio_server(name, server_config, base_dir_for_resolution)
                if server:
                    servers.append(server)
                    successful_servers += 1
                else:
                    failed_servers += 1

        except Exception as e:
            logger.error(
                f"Unexpected error processing MCP server '{name}': {e!s}",
                exc_info=True,
            )
            failed_servers += 1

    logger.info(
        f"MCP server loading completed: {successful_servers} successful, {failed_servers} failed"
    )
    if failed_servers > 0:
        logger.warning(f"{failed_servers} MCP servers failed to load - check logs for details")

    return servers


def validate_mcp_config(config: dict[str, Any]) -> bool:
    """Validate MCP configuration structure.

    Args:
        config: MCP configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(config, dict):
        logger.error("MCP configuration must be a dictionary")
        return False

    if "mcpServers" not in config:
        logger.error("MCP configuration must contain 'mcpServers' key")
        return False

    mcp_servers = config["mcpServers"]
    if not isinstance(mcp_servers, dict):
        logger.error("'mcpServers' must be a dictionary")
        return False

    if not mcp_servers:
        logger.warning("No MCP servers configured")
        return True

    # Validate individual server configurations
    for name, server_config in mcp_servers.items():
        if not validate_server_config(name, server_config):
            return False

    return True
