"""Wrapper module for PydanticAI agent with conditional imports.

This module provides a safe interface to PydanticAI functionality with graceful
degradation when dependencies are not available.
"""

import logging
from typing import Any

# Dependency-free registry (see src/worker/provider_registry.py) — safe to
# import unconditionally even when pydantic_ai itself is unavailable, which is
# the whole point of this wrapper module. Deliberately a runtime import, not a
# TYPE_CHECKING one (TC001): the registry is stdlib-only so it costs nothing,
# and the drift guard in tests/test_provider_registry.py resolves this
# signature with typing.get_type_hints(), which needs the name at runtime.
from src.settings import get_settings
from src.worker.provider_registry import ModelProvider  # noqa: TC001

logger = logging.getLogger(__name__)

# Try to import PydanticAI dependencies
PYDANTIC_AI_AVAILABLE = False
create_pydantic_ai_agent = None

try:
    from .pydantic_ai_agent import create_pydantic_ai_agent

    PYDANTIC_AI_AVAILABLE = True
    logger.info("PydanticAI agent module loaded successfully")
except ImportError as e:
    logger.warning(
        f"PydanticAI agent module not available: {e}. Agent functionality will be disabled."
    )
except Exception:
    logger.exception(
        "Unexpected error loading PydanticAI agent module. Agent functionality will be disabled."
    )


class MockPydanticAIAgent:
    """Mock agent for when PydanticAI is not available.

    This class provides the same interface as PydanticAIAgent but returns
    error responses to indicate that PydanticAI dependencies are not available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002, RUF100
        """Initialize mock agent with minimal configuration.

        Args:
            *args: Variable positional arguments (ignored)
            **kwargs: Keyword arguments, 'name' and 'db_path' will be used if provided
        """
        self.name = kwargs.get("name", "mock-agent")
        self.db_path = kwargs.get("db_path", "mock_db_path")
        logger.warning(f"Created mock agent '{self.name}' because PydanticAI is not available")

    async def run(self, prompt: str | list[Any]) -> dict[str, Any]:  # noqa: ARG002, RUF100
        """Mock run method that returns an error.

        Args:
            prompt: The prompt to process (ignored — string OR multimodal
                content-block list, both are rejected uniformly here)

        Returns:
            Dictionary with error message and None metrics
        """
        error_msg = f"PydanticAI not available for agent '{self.name}'"
        logger.error(error_msg)
        return {
            "output": error_msg,
            "usage_metrics": None,
            "error": "PydanticAI dependencies not available",
        }

    async def run_with_context(self, enhanced_context: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002, RUF100
        """Mock run_with_context method that returns an error.

        Args:
            enhanced_context: Graph execution context (ignored)

        Returns:
            Dictionary with error message and None metrics
        """
        error_msg = f"PydanticAI not available for agent '{self.name}'"
        logger.error(error_msg)
        return {
            "output": error_msg,
            "usage_metrics": None,
            "error": "PydanticAI dependencies not available",
        }

    async def check_health(self) -> bool:
        """Mock health check that always returns False.

        Returns:
            Always False since mock agent is not functional
        """
        return False

    async def cleanup(self) -> None:
        """Mock cleanup method.

        No actual cleanup needed for mock agent.
        """

    def get_tool(self, tool_name: str) -> Any | None:  # noqa: ARG002, RUF100
        """Mock get_tool method.

        Args:
            tool_name: Name of tool to retrieve (ignored)

        Returns:
            Always None since mock agent has no tools
        """
        return None

    def add_tools_from_json(self, json_tools: list[dict[str, Any]]) -> None:  # noqa: ARG002, RUF100
        """Mock add_tools_from_json method.

        Args:
            json_tools: List of tool definitions (ignored)
        """
        logger.warning(f"Cannot add tools to mock agent '{self.name}' - PydanticAI not available")


def safe_create_pydantic_ai_agent(  # noqa: PLR0913 -- mirror of create_pydantic_ai_agent's signature for the safe wrapper
    name: str,
    description: str,
    # Reuses the ModelProvider Literal itself instead of re-declaring it; the
    # old inline copy silently rejected xai/google/vlm_local. Still a static
    # Literal for mypy because ModelProvider *is* the Literal alias.
    model_provider: ModelProvider = "vllm",
    model: str | None = None,
    temperature: float = 0.7,
    db_path: str = "",
    system_prompt: str | None = None,
    instruction: str | None = None,
    user_prompt: str = "",
    tools: list[Any] | None = None,
    deps: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    tools_dir: str | None = None,
    mcp_servers_list: list[Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    output_validator_code: str | None = None,
) -> Any | MockPydanticAIAgent:
    """Safely create a PydanticAI agent with fallback to mock agent.

    Returns the real agent if PydanticAI is available, otherwise returns a mock agent.

    Args:
        name: Agent name
        description: Agent description
        model_provider: LLM provider to use
        model: Model name; ``None`` means the platform model
            (``Settings.local_llm_model``).
        temperature: Sampling temperature
        db_path: Database path for persistence
        system_prompt: System prompt for the agent
        instruction: Instruction for the agent
        user_prompt: Initial user prompt
        tools: List of tools available to the agent
        deps: Dependencies for the agent
        base_url: Custom base URL for the provider
        api_key: API key for the provider
        tools_dir: Directory containing tool definitions
        mcp_servers_list: List of MCP servers to use

    Returns:
        PydanticAIAgent instance if available, MockPydanticAIAgent otherwise
    """
    if model is None:
        model = get_settings().local_llm_model
    if PYDANTIC_AI_AVAILABLE and create_pydantic_ai_agent is not None:
        try:
            return create_pydantic_ai_agent(
                name=name,
                description=description,
                model_provider=model_provider,
                model=model,
                temperature=temperature,
                db_path=db_path,
                system_prompt=system_prompt,
                instruction=instruction,
                user_prompt=user_prompt,
                tools=tools,
                deps=deps,
                base_url=base_url,
                api_key=api_key,
                tools_dir=tools_dir,
                mcp_servers_list=mcp_servers_list,
                output_schema=output_schema,
                output_validator_code=output_validator_code,
            )
        except Exception:
            logger.exception("Failed to create PydanticAI agent. Creating mock agent instead.")
            return MockPydanticAIAgent(
                name=name,
                description=description,
                model_provider=model_provider,
                model=model,
                temperature=temperature,
                db_path=db_path,
            )
    else:
        logger.warning(f"PydanticAI not available. Creating mock agent for '{name}'")
        return MockPydanticAIAgent(
            name=name,
            description=description,
            model_provider=model_provider,
            model=model,
            temperature=temperature,
            db_path=db_path,
        )


def is_pydantic_ai_available() -> bool:
    """Check if PydanticAI dependencies are available.

    Returns:
        True if PydanticAI is available, False otherwise
    """
    return PYDANTIC_AI_AVAILABLE
