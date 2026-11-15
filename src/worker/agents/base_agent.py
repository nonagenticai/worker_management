"""Base agent abstraction for worker management system.

This module provides the abstract base class that all agent types must implement,
ensuring a consistent interface across different agent implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid7

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Attribute a FAILED run's exception carries: the ``TokenUsageMetrics`` dump of
#: what the run SENT (see ``PydanticAIAgent._stamp_sent_usage_metrics``). The
#: message endpoint renders it as ``error.details.usage_metrics``.
SENT_USAGE_METRICS_ATTR = "wm_sent_usage_metrics"


class TokenUsageMetrics(BaseModel):
    """Token usage metrics for agent execution.

    ``served_model`` / ``served_provider`` are the ROUTING answer, not usage, and
    they live here for one practical reason: this object is what
    ``MessageResponse.usage_metrics`` is built from
    (``usage_metrics=response.usage_metrics.model_dump()``), so it is the only
    channel that already reaches an API caller. Every other agent type leaves
    them None, which is the correct reading: nothing resolved an alias.
    """

    prompt_tokens: int | None = Field(default=None, description="Number of tokens in the prompt.")
    completion_tokens: int | None = Field(
        default=None, description="Number of tokens in the completion."
    )
    total_tokens: int | None = Field(default=None, description="Total number of tokens used.")
    served_model: str | None = Field(
        default=None,
        description=(
            "Model the upstream ACTUALLY served, as reported by inference-router. "
            "Differs from the requested model whenever a routing alias (local-llm, "
            "tier-medium) was sent. None when the provider does not report it."
        ),
    )
    served_provider: str | None = Field(
        default=None,
        description=(
            "Provider that actually served the call, as reported by "
            "inference-router. None when the provider does not report it."
        ),
    )
    served_model_root: str | None = Field(
        default=None,
        description=(
            "Model ROOT (the weights behind the served name) that answered, as "
            "reported by inference-router. None when the provider does not report it."
        ),
    )
    # Sampling provenance: what THIS worker put on the wire, read back from the
    # exact model_settings object handed to pydantic-ai (never re-derived from the
    # stored config, which is defaulted twice on the way down). None means the
    # value was not sent — never a guess.
    requested_model: str | None = Field(
        default=None,
        description=(
            "Model id this worker REQUESTED (the request body's `model`). Compare "
            "with served_model: they differ whenever a routing alias was sent."
        ),
    )
    temperature: float | None = Field(
        default=None,
        description="Effective temperature sent on the request, after defaulting. None ⇒ not sent.",
    )
    temperature_not_sent: str | None = Field(
        default=None,
        description=(
            "Set when a temperature was configured or requested but the model refuses "
            "sampling settings, so it was stripped before the request (temperature is "
            "then None). The reason names the model. None ⇒ nothing was withheld."
        ),
    )
    seed: int | None = Field(
        default=None,
        description="Sampling seed sent on the request. None ⇒ no seed key was sent.",
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Effective max_tokens sent on the request, after clamping/defaulting "
            "(``_resolve_max_tokens``). None ⇒ no max_tokens key was sent."
        ),
    )
    repetition_penalty: float | None = Field(
        default=None,
        description=(
            "vLLM ``repetition_penalty`` sent on the request (as a top-level body key "
            "via extra_body). None ⇒ not sent."
        ),
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="OpenAI ``frequency_penalty`` sent on the request. None ⇒ not sent.",
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "Provider-reported finish reason of the last model response, normalised by "
            "pydantic-ai (stop / length / content_filter / tool_call / error). "
            "``length`` means the answer was truncated by max_tokens. None ⇒ not reported."
        ),
    )
    correlation_id: str | None = Field(
        default=None,
        description=(
            "``X-Correlation-Id`` forwarded to inference-router for this call — the join "
            "key between this row and IR's inference_call ledger. None ⇒ none was sent."
        ),
    )
    grammar_stripped_keywords: list[str] | None = Field(
        default=None,
        description=(
            "JSON-Schema keywords the output schema declares that were stripped from the "
            "guided-decoding grammar (xgrammar cannot compile them). None ⇒ nothing stripped."
        ),
    )
    schema_bounds_enforced: bool | None = Field(
        default=None,
        description=(
            "Whether count/length bounds (minItems/maxItems/minLength/maxLength), which "
            "the grammar cannot carry, were enforced post-hoc for THIS run "
            "(enforce_schema_bounds). False ⇒ advisory only. None ⇒ the schema declares none."
        ),
    )
    bound_violations: list[str] | None = Field(
        default=None,
        description=(
            "JSON paths in the output that violated a count/length bound, checked after "
            "decoding against the original schema. Populated whether or not enforcement "
            "was on (when on, a violation fails the run instead). [] ⇒ in bounds; "
            "None ⇒ the schema declares no such bound."
        ),
    )


class AgentRunResult(BaseModel):
    """Result from running an agent, including output and usage metrics."""

    output: Any = Field(..., description="The output from the agent's execution.")
    usage_metrics: TokenUsageMetrics | None = Field(
        default=None, description="Token usage metrics for the agent run."
    )
    # Claude Code CLI specific fields (from --output-format json)
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity."
    )
    cost_usd: float | None = Field(
        default=None, description="Total cost in USD for this execution."
    )
    duration_ms: int | None = Field(default=None, description="Execution duration in milliseconds.")
    is_error: bool | None = Field(
        default=None, description="Whether the execution resulted in an error."
    )


class BaseAgent(ABC):
    """Abstract base class for all agent types.

    All agent implementations must inherit from this class and implement
    the required abstract methods to ensure compatibility with the worker
    management system.

    Attributes:
        agent_id: Unique identifier for the agent instance
        name: Human-readable name for the agent
        db_path: Path to the agent's database file
    """

    def __init__(self, agent_id: UUID | None = None) -> None:
        """Initialize base agent.

        Args:
            agent_id: Optional unique identifier for the agent
        """
        # Provide safe defaults so downstream code can rely on attributes existing
        # Subclasses are expected to set meaningful values as appropriate
        self.agent_id: UUID = agent_id or uuid7()
        self.name: str = self.__class__.__name__
        self.db_path: str = ""

    @abstractmethod
    async def run(self, prompt: str | list[Any]) -> AgentRunResult:
        """Run the agent with a given prompt.

        Args:
            prompt: Either a plain text prompt (legacy) OR a list of typed
                content blocks (multimodal). Multimodal lists are only
                supported by ``pydantic_ai`` agents configured against a
                vision-capable model. CLI / Codex / Claude-Code agents must
                raise ``NotImplementedError`` when handed a list.

        Returns:
            AgentRunResult containing output and optional usage metrics

        Raises:
            NotImplementedError: If the agent type doesn't support
                multimodal input and ``prompt`` is a list.
            Exception: If agent execution fails.
        """

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the agent is healthy and ready to execute tasks.

        Returns:
            True if agent is healthy, False otherwise
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources.

        This method should be called when the agent is no longer needed
        to ensure proper resource cleanup (e.g., closing connections,
        releasing file handles, etc.).

        Raises:
            Exception: If cleanup fails
        """

    def _build_context_aware_prompt(
        self, task: str, graph_context: dict[str, Any], node_config: dict[str, Any]
    ) -> str:
        """Build a context-aware prompt that includes graph execution information."""
        context_parts = [task]

        if graph_context:
            context_parts.append("\n## Context Information")
            if graph_context.get("session_id"):
                context_parts.append(f"Session ID: {graph_context['session_id']}")
            if graph_context.get("completed_nodes"):
                context_parts.append(
                    f"Completed nodes: {', '.join(graph_context['completed_nodes'])}"
                )
            if graph_context.get("variables"):
                context_parts.append(f"Variables: {graph_context['variables']}")

        if node_config:
            context_parts.append(f"\n## Node Configuration\n{node_config}")

        return "\n".join(context_parts)

    async def run_with_context(self, enhanced_context: dict[str, Any]) -> AgentRunResult:
        """Run the agent with enhanced graph execution context.

        Default implementation builds an enhanced prompt and delegates to run().
        Subclasses can override for custom context handling.
        """
        task = enhanced_context.get("task", "")
        graph_context = enhanced_context.get("graph_context", {})
        node_config = enhanced_context.get("node_config", {})
        enhanced_prompt = self._build_context_aware_prompt(task, graph_context, node_config)
        return await self.run(enhanced_prompt)

    def get_agent_type(self) -> str:
        """Get the type of this agent.

        Returns:
            String identifier for the agent type (e.g., 'pydantic_ai', 'claude_code')
        """
        return self.__class__.__name__.lower().replace("agent", "")


class BaseAgentConfig(BaseModel):
    """Base configuration for all agent types."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    agent_role: str | None = None
    system_prompt: str | None = Field(default="You are a helpful assistant.")
    instruction: str | None = None
    db_path: str
