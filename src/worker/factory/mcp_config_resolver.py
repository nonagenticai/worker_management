"""MCP configuration resolution logic.

Handles resolving MCP preset configs, server filtering,
environment normalization, and loading MCP config from files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from src.security.request_overrides import assert_mcp_config_allowed

logger = logging.getLogger(__name__)


def _handle_preset_config(preset: str, worker_name: str) -> str | None:
    """Handle MCP preset configuration resolution.

    Args:
        preset: The preset name to resolve
        worker_name: Name of the worker for logging

    Returns:
        Path to the preset configuration file if found, None otherwise
    """
    preset_path = f"./mcp_configs/{preset}_agent_config.json"
    if Path(preset_path).exists():
        logger.info("Using MCP preset '%s' for worker %s", preset, worker_name)
        return preset_path

    logger.warning("MCP preset '%s' not found at %s", preset, preset_path)
    return None


def _handle_server_filtering(servers: list[str], worker_name: str) -> dict[str, Any] | None:
    """Handle MCP server filtering from default configuration.

    Args:
        servers: List of server names to filter for
        worker_name: Name of the worker for logging

    Returns:
        Filtered configuration dictionary with only specified servers,
        or None if configuration cannot be loaded
    """
    # Try multiple possible config file locations
    config_paths = [
        "./mcp_configs/github_agent_config.json",
        "./mcp_configs/default_mcp_config.json",
        "./mcp_configs/agent_config.json",
    ]

    config = None
    config_path_used = None

    for config_path in config_paths:
        if Path(config_path).exists():
            try:
                with open(config_path, encoding="utf-8") as config_file:
                    config = json.load(config_file)
                    config_path_used = config_path
                    logger.debug(f"Loaded MCP config from {config_path}")
                    break
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                continue

    if not config:
        logger.warning(
            f"No MCP configuration found for server filtering. "
            f"Tried paths: {config_paths}. "
            f"Cannot filter servers: {servers}"
        )
        return None

    try:
        filtered_config: dict[str, Any] = {"mcpServers": {}}
        found_servers = []
        not_found_servers = []

        for server_name in servers:
            if server_name in config.get("mcpServers", {}):
                filtered_config["mcpServers"][server_name] = config["mcpServers"][server_name]
                found_servers.append(server_name)
            else:
                not_found_servers.append(server_name)

        if not_found_servers:
            logger.warning(
                f"Requested MCP servers not found in config {config_path_used}: {not_found_servers}. "
                f"Available servers: {list(config.get('mcpServers', {}).keys())}"
            )

        if found_servers:
            logger.info(f"Using filtered MCP servers {found_servers} for worker {worker_name}")
            return filtered_config
        else:
            logger.error(
                f"None of the requested MCP servers {servers} were found in the configuration"
            )
            return None

    except KeyError:
        logger.exception("Missing expected key in config")
        return None


async def resolve_mcp_config(
    mcp_config: str | dict[str, Any] | None = None,
    mcp_config_preset: str | None = None,
    mcp_servers: list[str] | None = None,
    worker_name: str = "unknown",
) -> str | dict[str, Any] | None:
    """Resolve MCP configuration from various input options.

    Priority order:
    1. mcp_config (explicit config file or dict)
    2. mcp_config_preset (preset config file)
    3. mcp_servers (specific server selection from default config)

    Args:
        mcp_config: Direct configuration as file path or dict
        mcp_config_preset: Preset configuration name
        mcp_servers: List of specific servers to use
        worker_name: Name of the worker for logging

    Returns:
        Resolved configuration as file path or dict, or None if none found
    """
    # Fail closed on a stored row or direct caller carrying what the request
    # validators refuse: a stdio ``command`` (spawns a process with this pod's
    # full environment), a file path, or a non-allowlisted URL. Raises
    # McpConfigNotAllowedError; nothing is spawned or opened.
    assert_mcp_config_allowed(mcp_config, mcp_config_preset)

    # Priority 1: Direct config
    if mcp_config and (
        isinstance(mcp_config, str) or (isinstance(mcp_config, dict) and len(mcp_config) > 0)
    ):
        return mcp_config

    # Priority 2: Preset config
    if mcp_config_preset:
        result = _handle_preset_config(mcp_config_preset, worker_name)
        if result:
            return result

    # Priority 3: Server filtering
    if mcp_servers:
        return _handle_server_filtering(mcp_servers, worker_name)

    return None


async def _resolve_mcp_servers_for_cli_agents(
    mcp_servers: list[str] | None = None,
    worker_name: str = "unknown",
) -> list[dict[str, str]] | None:
    """Resolve MCP server names to name/URL dicts for CLI agents (Claude Code, Codex).

    This function takes a list of MCP server names and resolves them to dictionaries
    with 'name' and 'url' keys by looking them up in the MCP configuration files.

    It also handles Docker-to-host URL conversion when necessary (e.g., converting
    'http://kanban-mcp:1337/sse' to 'http://localhost:1337/sse' for host CLI access).

    Args:
        mcp_servers: List of MCP server names (e.g., ["kanban", "github"])
        worker_name: Name of the worker for logging

    Returns:
        List of dicts with 'name' and 'url' keys, or None if resolution fails

    Example:
        Input: ["kanban"]
        Output: [{"name": "kanban", "url": "http://localhost:1337/sse"}]
    """
    if not mcp_servers:
        return None

    logger.info(f"Resolving MCP servers for CLI agent '{worker_name}': {mcp_servers}")

    # Load MCP configuration to get URLs
    resolved_config = await resolve_mcp_config(
        mcp_servers=mcp_servers,
        worker_name=worker_name,
    )

    if not resolved_config or not isinstance(resolved_config, dict):
        logger.warning(
            f"Could not resolve MCP configuration for CLI agent '{worker_name}'. "
            f"MCP servers will not be configured."
        )
        return None

    mcp_servers_dict = resolved_config.get("mcpServers", {})
    if not mcp_servers_dict:
        logger.warning(f"No mcpServers found in configuration for '{worker_name}'")
        return None

    result = []
    for server_name in mcp_servers:
        server_config = mcp_servers_dict.get(server_name)
        if not server_config:
            logger.warning(
                f"MCP server '{server_name}' not found in configuration for '{worker_name}'"
            )
            continue

        # Check if it's a URL-based or command-based MCP server
        url = server_config.get("url")
        command = server_config.get("command")

        if url:
            # URL-based MCP server (SSE/HTTP transport)
            # Convert Docker-internal URLs to host-accessible URLs
            if "kanban-mcp:" in url:
                url = url.replace("kanban-mcp:", "localhost:")
                logger.info(f"Converted Docker URL to host URL for '{server_name}': {url}")

            result.append({"name": server_name, "url": url})
            logger.info(f"Resolved MCP server '{server_name}' to URL: {url}")
        elif command:
            # Command-based MCP server (stdio transport)
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            result.append({"name": server_name, "command": command, "args": args, "env": env})
            logger.info(
                f"Resolved MCP server '{server_name}' to command: {command} {' '.join(args)}"
            )
        else:
            logger.warning(
                f"MCP server '{server_name}' has neither URL nor command in configuration for '{worker_name}'"
            )
            continue

    if not result:
        logger.warning(f"No MCP servers could be resolved for CLI agent '{worker_name}'")
        return None

    logger.info(f"Successfully resolved {len(result)} MCP server(s) for CLI agent '{worker_name}'")
    return result


def _normalize_mcp_config_for_environment(
    resolved_config: dict[str, Any],
    *,
    running_in_docker: bool,  # noqa: ARG001
) -> dict[str, Any]:
    """Convert command-based MCP configs to HTTP when running alongside external servers."""
    servers = resolved_config.get("mcpServers")
    if not isinstance(servers, dict):
        return resolved_config

    kanban_url = os.getenv("KANBAN_MCP_URL")
    if not kanban_url:
        # No configured Kanban MCP endpoint; leave config untouched
        return resolved_config

    try:
        timeout = float(os.getenv("KANBAN_MCP_TIMEOUT", "30"))
    except ValueError:
        timeout = 30.0

    for server_name, server_cfg in list(servers.items()):
        if not isinstance(server_cfg, dict):
            continue

        cfg = cast("dict[str, Any]", server_cfg)

        # Skip configs already using HTTP transport
        if "url" in cfg:
            continue

        command = cfg.get("command")
        args = cfg.get("args")

        args_str = ""
        if isinstance(args, (list, tuple)):
            args_str = " ".join(str(arg) for arg in args)
        elif isinstance(args, str):
            args_str = args

        if command == "node" and "kanban-mcp" in args_str:
            log_level = cfg.get("log_level", "INFO")
            servers[server_name] = {
                "url": kanban_url,
                "timeout": timeout,
                "log_level": log_level,
            }
            logger.info(
                "Converted Kanban MCP server '%s' to HTTP endpoint '%s' with timeout %s",
                server_name,
                kanban_url,
                timeout,
            )

    return resolved_config
