"""Agent creation dispatch logic.

Handles agent creation via the registry, agent config construction,
and fallback to basic workers.
"""

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from src.settings import get_settings
from src.worker.factory.mcp_config_resolver import _resolve_mcp_servers_for_cli_agents
from src.worker.worker_template import WorkerTemplate

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


class MCPConfig(BaseModel):
    """Configuration for MCP (Model Control Protocol) settings."""

    mcp_config: str | dict[str, Any] | None = None
    mcp_config_preset: str | None = None
    mcp_servers: list[str] | None = None


class AgentConfig(BaseModel):
    """Configuration for agent-specific settings."""

    model_config = ConfigDict(protected_namespaces=())

    agent_role: str | None = None
    agent_type: str | None = None
    model_provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # Tri-state chain-of-thought switch. None ⇒ never forwarded, so the outbound
    # payload is byte-for-byte what it is today.
    enable_reasoning: bool | None = None
    # Sampling seed. None ⇒ never forwarded, so no seed key reaches the wire.
    seed: int | None = None
    # inference-router provider pin (X-Provider). None ⇒ no header.
    ir_provider: str | None = None
    # Repetition controls. None ⇒ never forwarded (no key on the wire).
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    system_prompt: str | None = None
    instruction: str | None = None
    user_prompt: str | None = None


class WorkerConfig(BaseModel):
    """Configuration for worker creation parameters."""

    worker_name: str
    tool_names: list[str]
    description: str | None = None
    deps: list[str] | None = None
    base_url_override: str | None = None
    api_key_override: str | None = None
    # True only when api_key_override is the tenant's own resolved BYO key.
    tenant_byo_key: bool = False
    tools_dir_override: str | None = None
    agent_config: AgentConfig | None = None
    mcp_config: MCPConfig | None = None


def _create_worker_configs(  # noqa: PLR0913 -- builder takes one arg per configurable axis
    worker_name: str,
    tool_names: list[str],
    agent_role: str,
    agent_type: str | None,
    model_provider: str | None,
    model: str | None,
    temperature: float | None,
    instruction: str | None,
    system_prompt: str | None,
    user_prompt: str | None,
    description: str | None,
    deps: list[str] | None,
    base_url_override: str | None,
    api_key_override: str | None,
    tools_dir_override: str | None,
    max_tokens: int | None = None,
    enable_reasoning: bool | None = None,  # noqa: FBT001 -- keyword-only at every call site
    seed: int | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    ir_provider: str | None = None,
    tenant_byo_key: bool = False,  # noqa: FBT001, FBT002 -- keyword-only at every call site
) -> tuple[WorkerConfig, AgentConfig]:
    """Create worker and agent configuration objects.

    Args:
        worker_name: Name of the worker
        tool_names: List of tool names
        agent_role: Role of the agent
        agent_type: Type of agent (defaults to "pydantic_ai")
        model_provider: AI model provider
        model: AI model name
        temperature: Model temperature setting
        instruction: Agent instruction
        system_prompt: System prompt for the agent
        user_prompt: User prompt for the agent
        description: Worker description
        deps: Worker dependencies
        base_url_override: Override for base URL
        api_key_override: Override for API key
        tools_dir_override: Override for tools directory
        max_tokens: Generation budget (None ⇒ provider default)
        enable_reasoning: Tri-state chain-of-thought switch (None ⇒ send nothing)
        seed: Sampling seed (None ⇒ send nothing)
        repetition_penalty: vLLM repetition penalty (None ⇒ send nothing)
        frequency_penalty: OpenAI frequency_penalty (None ⇒ send nothing)

    Returns:
        Tuple of (WorkerConfig, AgentConfig)
    """
    agent_config = AgentConfig(
        agent_role=agent_role,
        agent_type=agent_type or "pydantic_ai",
        model_provider=model_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_reasoning=enable_reasoning,
        seed=seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        ir_provider=ir_provider,
        system_prompt=system_prompt,
        instruction=instruction,
        user_prompt=user_prompt,
    )

    worker_config = WorkerConfig(
        worker_name=worker_name,
        tool_names=tool_names,
        description=description,
        deps=deps,
        base_url_override=base_url_override,
        api_key_override=api_key_override,
        tenant_byo_key=tenant_byo_key and bool(api_key_override),
        tools_dir_override=tools_dir_override,
        agent_config=agent_config,
    )

    return worker_config, agent_config


async def _create_agent_worker(  # noqa: PLR0913 -- internal factory; arg count matches the config shape it materializes
    worker_config: WorkerConfig,
    agent_config: AgentConfig,
    tools: list[Any],
    db_path_str: str,
    mcp_servers_list: list[Any],
    mcp_servers: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
    output_validator_code: str | None = None,
    container_id: str | None = None,
    executor_endpoint: str | None = None,
    workspace: str | None = None,
    output_retries: int | None = None,
    worker_version: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """Create an agent worker instance using the agent registry.

    Args:
        worker_config: Worker configuration
        agent_config: Agent-specific configuration
        tools: List of tools for the agent
        db_path_str: Path to the worker's database
        mcp_servers_list: List of MCP server instances (for PydanticAI)
        mcp_servers: List of MCP server names (for CLI agents like Claude Code/Codex)
        output_schema: JSON Schema for structured output
        output_validator_code: Python code for custom output validation
        container_id: Container ID for remote execution (remote_claude, remote_codex)
        executor_endpoint: Base URL of coding service for remote execution
        workspace: Working directory in remote container

    Returns:
        Agent worker instance
    """
    agent_type = agent_config.agent_type or "pydantic_ai"

    try:
        # Import registry and create agent dynamically
        from src.worker.agents.agent_registry import create_agent

        logger.info(
            f"Creating agent of type '{agent_type}' for worker '{worker_config.worker_name}'"
        )

        # Build agent creation kwargs
        agent_kwargs: dict[str, Any] = {
            "name": worker_config.worker_name,
            "description": worker_config.description
            or f"{agent_config.agent_role} using {agent_type}",
            "db_path": db_path_str,
            "system_prompt": agent_config.system_prompt,
            "instruction": agent_config.instruction,
            "user_prompt": agent_config.user_prompt or "",
            "tools": tools,
            "deps": worker_config.deps or [],
            "base_url": worker_config.base_url_override,
            "api_key": worker_config.api_key_override,
            "tools_dir": worker_config.tools_dir_override,
            "mcp_servers_list": mcp_servers_list,
            "output_schema": output_schema,
            "output_validator_code": output_validator_code,
        }

        # Add agent-type specific parameters with defaults for required fields
        # No model on the worker => the platform model (the one pgx served-model-name,
        # Settings.local_llm_model) on the IR-routed ``vllm`` provider -- the same pair
        # as the send_message auto-create path (message_endpoints). The old "mlx"
        # model name is no IR alias, so it rode default_chain under a false name.
        agent_kwargs["model_provider"] = agent_config.model_provider or "vllm"
        agent_kwargs["model"] = agent_config.model or get_settings().local_llm_model
        agent_kwargs["temperature"] = (
            agent_config.temperature if agent_config.temperature is not None else 0.7
        )
        # Only forward max_tokens when explicitly set so we don't override the
        # create_pydantic_ai_agent default (None) and clobber provider hardcodes.
        if agent_config.max_tokens is not None and agent_type == "pydantic_ai":
            agent_kwargs["max_tokens"] = agent_config.max_tokens

        # Same "only when explicitly set" rule for the chain-of-thought switch:
        # leaving it out keeps create_pydantic_ai_agent's default of None, which
        # emits no extra_body at all, so today's wire payload is unchanged.
        if agent_config.enable_reasoning is not None and agent_type == "pydantic_ai":
            agent_kwargs["enable_reasoning"] = agent_config.enable_reasoning

        # Sampling seed: same "only when explicitly set" rule, so a worker with no
        # seed builds an agent whose request body carries no seed key at all.
        if agent_config.seed is not None and agent_type == "pydantic_ai":
            agent_kwargs["seed"] = agent_config.seed
        if agent_config.ir_provider and agent_type == "pydantic_ai":
            agent_kwargs["ir_provider"] = agent_config.ir_provider
        # Repetition controls: the same rule, one key each.
        if agent_config.repetition_penalty is not None and agent_type == "pydantic_ai":
            agent_kwargs["repetition_penalty"] = agent_config.repetition_penalty
        if agent_config.frequency_penalty is not None and agent_type == "pydantic_ai":
            agent_kwargs["frequency_penalty"] = agent_config.frequency_penalty

        # Per-worker pydantic-ai output-validation retry override (mirrors the
        # output_schema/output_validator_code passthrough). Only forwarded for
        # pydantic_ai; other agent types ignore it.
        if output_retries is not None and agent_type == "pydantic_ai":
            agent_kwargs["output_retries"] = output_retries

        # WS-1 prompt-replay: forward the active worker version so the pydantic-ai
        # provider can stamp X-Worker-Version on outbound IR calls under replay
        # mode. Only pydantic_ai builds an InferenceRouterOpenAIProvider; other
        # agent factories don't accept the kwarg.
        if worker_version is not None and agent_type == "pydantic_ai":
            agent_kwargs["worker_version"] = worker_version

        # Identity: bake the resolved tenant into the agent so the pydantic-ai
        # provider sends a REAL X-Tenant-Id. Before this, nothing on the factory
        # chain carried tenant_id at all, so the provider fell back to "" and
        # inference-router metered the call against the nil sentinel -- 1,400 of
        # 13,893 routed calls in the 24h to 2026-08-27. Same "only when
        # explicitly set" rule as worker_version so non-pydantic_ai factories,
        # which do not accept the kwarg, are unaffected.
        if tenant_id is not None and agent_type == "pydantic_ai":
            agent_kwargs["tenant_id"] = tenant_id
        # Tenant-resolved BYO key marker; only forwarded when set so other agent
        # factories (which do not accept the kwarg) are unaffected.
        if worker_config.tenant_byo_key and agent_type == "pydantic_ai":
            agent_kwargs["tenant_byo_key"] = True

        # ⚠️ A `license_token` branch stood here, threading an outbound LICENCE
        # credential down to the pydantic_ai agent. RETIRED 2026-09-11: the
        # value was dropped by `_tenant_header_value` and reached no wire, so
        # it was a live bearer credential with no consumer. The licence that
        # inference-router reads is `license_id` on `X-License-Id`, which is
        # resolved per-request from the ContextVar layer, not baked here.

        # For remote agent types, pass container_id and remote execution params
        if agent_type in ["remote_claude", "remote_codex"]:
            agent_kwargs["container_id"] = container_id or ""
            if executor_endpoint:
                agent_kwargs["executor_endpoint"] = executor_endpoint
            agent_kwargs["workspace"] = workspace or "/workspace"

        # For CLI agents (Claude Code, Codex), resolve MCP servers to name/URL dicts
        if agent_type in ["claude_code", "codex"] and mcp_servers:
            logger.info(f"Resolving MCP servers for CLI agent '{agent_type}': {mcp_servers}")
            resolved_mcp_servers = await _resolve_mcp_servers_for_cli_agents(
                mcp_servers=mcp_servers,
                worker_name=worker_config.worker_name,
            )
            if resolved_mcp_servers:
                agent_kwargs["mcp_servers"] = resolved_mcp_servers
                logger.info(
                    f"Passing {len(resolved_mcp_servers)} resolved MCP server(s) to '{agent_type}' agent"
                )
            else:
                logger.warning(
                    f"Could not resolve MCP servers for CLI agent '{agent_type}'. "
                    "Agent will be created without MCP configuration."
                )

        # Create agent using registry
        return create_agent(agent_type, **agent_kwargs)

    except ValueError:
        logger.exception(
            f"Failed to create agent of type '{agent_type}'. Creating basic worker instead."
        )
    except Exception:
        logger.exception(
            f"Unexpected error creating agent of type '{agent_type}'. Creating basic worker instead."
        )

    # Fallback to basic worker if agent creation fails
    return WorkerTemplate(
        name=worker_config.worker_name,
        description=worker_config.description
        or f"Basic worker (Failed to create {agent_type} agent)",
        tools=tools,
        output_type=None,
        retries=3,
        fastapi_app=None,
        db_path=db_path_str,
    )


def _create_basic_worker(
    worker_name: str,
    description: str | None,
    tools: list[Any],
    worker_app: FastAPI,
    db_path_str: str,
) -> WorkerTemplate:
    """Create a basic worker template without agent capabilities.

    Args:
        worker_name: Name of the worker
        description: Worker description
        tools: List of tools
        worker_app: FastAPI application instance
        db_path_str: Path to database file

    Returns:
        Basic WorkerTemplate instance
    """
    return WorkerTemplate(
        name=worker_name,
        description=description or "Standard worker without agent capabilities",
        tools=tools,
        output_type=None,
        retries=3,
        fastapi_app=worker_app,
        db_path=db_path_str,
    )
