"""Agent package for the Worker Management System.

This package contains agent implementations using PydanticAI and other
AI frameworks, tool conversion utilities, and agent templates.
"""

# Import base components
from .agent_registry import (
    create_agent,
    get_agent_registry,
    list_available_agent_types,
    register_agent,
)
from .base_agent import AgentRunResult, BaseAgent, TokenUsageMetrics

# Import all agents to trigger auto-registration
# This ensures all agent types are registered when the package is imported
try:  # noqa: RUF100, SIM105
    from .pydantic_ai_agent import PydanticAIAgent, create_pydantic_ai_agent
except ImportError:
    pass  # PydanticAI may not be available

try:  # noqa: RUF100, SIM105
    from .claude_code_agent import ClaudeCodeAgent, create_claude_code_agent
except ImportError:
    pass  # Claude Code may not be installed

try:  # noqa: RUF100, SIM105
    from .codex_agent import CodexAgent, create_codex_agent
except ImportError:
    pass  # Codex may not be installed

try:
    from .remote_cli_agent import (
        RemoteCLIAgent,
        create_remote_claude_agent,
        create_remote_codex_agent,
    )

    # Backward-compatible aliases
    RemoteClaudeAgent = RemoteCLIAgent
    RemoteCodexAgent = RemoteCLIAgent
except ImportError:
    pass  # httpx may not be available

__all__ = [
    "AgentRunResult",
    "BaseAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "PydanticAIAgent",
    "RemoteClaudeAgent",
    "RemoteCodexAgent",
    "TokenUsageMetrics",
    "create_agent",
    "create_claude_code_agent",
    "create_codex_agent",
    "create_pydantic_ai_agent",
    "create_remote_claude_agent",
    "create_remote_codex_agent",
    "get_agent_registry",
    "list_available_agent_types",
    "register_agent",
]
