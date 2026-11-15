"""Backward-compatible re-export. Real implementation in src.worker.factory."""

from src.worker.factory.agent_factory import (
    AgentConfig,
    MCPConfig,
    WorkerConfig,
    _create_agent_worker,
    _create_basic_worker,
    _create_worker_configs,
)
from src.worker.factory.mcp_config_resolver import (
    _handle_preset_config,
    _handle_server_filtering,
    _normalize_mcp_config_for_environment,
    _resolve_mcp_servers_for_cli_agents,
    resolve_mcp_config,
)
from src.worker.factory.tool_loader import (
    MockToolTemplate,
    _fetch_single_tool,
    _load_worker_tools,
)
from src.worker.factory.worker_factory import (
    McpServerCreator,
    WorkerCreationConfig,
    _setup_mcp_servers_for_worker,
    _setup_worker_database,
    create_worker,
)

__all__ = [
    "AgentConfig",
    "MCPConfig",
    "McpServerCreator",
    "MockToolTemplate",
    "WorkerConfig",
    "WorkerCreationConfig",
    "_create_agent_worker",
    "_create_basic_worker",
    "_create_worker_configs",
    "_fetch_single_tool",
    "_handle_preset_config",
    "_handle_server_filtering",
    "_load_worker_tools",
    "_normalize_mcp_config_for_environment",
    "_resolve_mcp_servers_for_cli_agents",
    "_setup_mcp_servers_for_worker",
    "_setup_worker_database",
    "create_worker",
    "resolve_mcp_config",
]
