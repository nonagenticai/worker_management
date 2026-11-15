"""Agent registry for managing different agent type implementations.

This module provides a registry pattern for dynamically registering and
creating different types of agents (pydantic_ai, claude_code, codex, etc.).
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for managing agent type implementations.

    This class maintains a mapping of agent type strings to their factory
    functions, allowing for dynamic agent creation and extensibility.
    """

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agents: dict[str, Callable[..., BaseAgent]] = {}
        self._agent_classes: dict[str, type[BaseAgent]] = {}

    def register(
        self,
        agent_type: str,
        factory_func: Callable[..., BaseAgent],
        agent_class: type[BaseAgent] | None = None,
    ) -> None:
        """Register an agent type with its factory function.

        Args:
            agent_type: String identifier for the agent type (e.g., 'pydantic_ai')
            factory_func: Callable that creates an instance of the agent
            agent_class: Optional agent class for type checking

        Raises:
            ValueError: If agent_type is already registered
        """
        if agent_type in self._agents:
            logger.warning(f"Agent type '{agent_type}' is already registered. Overwriting.")

        self._agents[agent_type] = factory_func
        if agent_class:
            self._agent_classes[agent_type] = agent_class

        logger.info(f"Registered agent type: {agent_type}")

    def unregister(self, agent_type: str) -> None:
        """Unregister an agent type.

        Args:
            agent_type: String identifier for the agent type to remove

        Raises:
            KeyError: If agent_type is not registered
        """
        if agent_type not in self._agents:
            raise KeyError(f"Agent type '{agent_type}' is not registered")

        del self._agents[agent_type]
        self._agent_classes.pop(agent_type, None)
        logger.info(f"Unregistered agent type: {agent_type}")

    def create(self, agent_type: str, **kwargs: Any) -> BaseAgent:
        """Create an agent instance of the specified type.

        Args:
            agent_type: String identifier for the agent type
            **kwargs: Configuration parameters to pass to the agent factory

        Returns:
            Instance of the requested agent type

        Raises:
            ValueError: If agent_type is not registered
            Exception: If agent creation fails
        """
        if agent_type not in self._agents:
            available = ", ".join(self.list_agent_types())
            raise ValueError(f"Unknown agent type: '{agent_type}'. Available types: {available}")

        try:
            factory = self._agents[agent_type]
            agent = factory(**kwargs)
            logger.info(
                f"Created agent of type '{agent_type}' with name '{kwargs.get('name', 'unknown')}'"
            )
            return agent
        except Exception:
            logger.exception(f"Failed to create agent of type '{agent_type}'")
            raise

    def is_registered(self, agent_type: str) -> bool:
        """Check if an agent type is registered.

        Args:
            agent_type: String identifier for the agent type

        Returns:
            True if registered, False otherwise
        """
        return agent_type in self._agents

    def list_agent_types(self) -> list[str]:
        """List all registered agent types.

        Returns:
            List of registered agent type identifiers
        """
        return list(self._agents.keys())

    def get_agent_class(self, agent_type: str) -> type[BaseAgent] | None:
        """Get the agent class for a given type.

        Args:
            agent_type: String identifier for the agent type

        Returns:
            Agent class if available, None otherwise
        """
        return self._agent_classes.get(agent_type)


# Global agent registry instance
_global_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance.

    Returns:
        The global AgentRegistry instance
    """
    global _global_registry  # noqa: PLW0603 -- lazy-init singleton
    if _global_registry is None:
        _global_registry = AgentRegistry()
        logger.info("Initialized global agent registry")
    return _global_registry


def register_agent(
    agent_type: str,
    factory_func: Callable[..., BaseAgent],
    agent_class: type[BaseAgent] | None = None,
) -> None:
    """Register an agent type with the global registry.

    Convenience function for registering agents with the global registry.

    Args:
        agent_type: String identifier for the agent type
        factory_func: Callable that creates an instance of the agent
        agent_class: Optional agent class for type checking
    """
    registry = get_agent_registry()
    registry.register(agent_type, factory_func, agent_class)


def create_agent(agent_type: str, **kwargs: Any) -> BaseAgent:
    """Create an agent instance using the global registry.

    Convenience function for creating agents from the global registry.

    Args:
        agent_type: String identifier for the agent type
        **kwargs: Configuration parameters to pass to the agent factory

    Returns:
        Instance of the requested agent type

    Raises:
        ValueError: If agent_type is not registered
    """
    registry = get_agent_registry()
    return registry.create(agent_type, **kwargs)


def list_available_agent_types() -> list[str]:
    """List all available agent types from the global registry.

    Returns:
        List of registered agent type identifiers
    """
    registry = get_agent_registry()
    return registry.list_agent_types()
