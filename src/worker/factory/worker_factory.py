"""Worker factory orchestrator.

Top-level create_worker() function that coordinates sub-modules for
MCP config resolution, tool loading, agent creation, and database setup.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.services.mlx_model_registry import mlx_model_registry
from src.worker.agents.pydantic_ai_agent_wrapper import PYDANTIC_AI_AVAILABLE
from src.worker.factory.agent_factory import (
    _create_agent_worker,
    _create_basic_worker,
    _create_worker_configs,
)
from src.worker.factory.mcp_config_resolver import (
    _normalize_mcp_config_for_environment,
    resolve_mcp_config,
)
from src.worker.factory.tool_loader import _load_worker_tools

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# Type definition for MCP server creation function
McpServerCreator = Optional[Callable[[str | dict[str, Any], Path | None], list[Any]]]

# Import MCP utilities with fallback
create_mcp_servers_from_config: McpServerCreator = None

try:
    from src.worker.mcp_utils import create_mcp_servers_from_config

    logger.info("MCP dependencies loaded successfully")
except ImportError as import_error:
    logger.warning(
        "MCP dependencies not available: %s. MCP server functionality will be disabled.",
        import_error,
    )
except Exception:
    logger.exception(
        "Unexpected error loading MCP dependencies. MCP server functionality will be disabled.",
    )


class WorkerCreationConfig(BaseModel):
    """Complete configuration for worker creation including all parameters."""

    worker_name: str
    tool_names: list[str]
    agent_role: str | None = None
    agent_type: str | None = None
    model_provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    instruction: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    description: str | None = None
    deps: list[str] | None = None
    mcp_config: str | dict[str, Any] | None = None
    mcp_config_preset: str | None = None
    mcp_servers: list[str] | None = None
    base_url_override: str | None = None
    api_key_override: str | None = None
    tools_dir_override: str | None = None


def _setup_worker_database(worker_name: str, disable_state_storage: bool = False) -> str:  # noqa: FBT001, FBT002, RUF100
    """Set up SQLite database for a worker.

    Args:
        worker_name: Name of the worker
        disable_state_storage: If True, skip database file creation

    Returns:
        Path to the database file (empty string if disabled)
    """
    if disable_state_storage:
        logger.info(
            "State storage disabled for worker %s, skipping database creation",
            worker_name,
        )
        return ""

    db_path_str = f"./{worker_name}_state.sqlite"
    if not os.path.exists(db_path_str):
        try:
            with open(db_path_str, "a", encoding="utf-8"):
                pass  # File creation handled by context manager
            logger.info("Created database stub for worker %s at %s", worker_name, db_path_str)
        except OSError:
            logger.exception("Failed to create database file %s", db_path_str)
    return db_path_str


async def _setup_mcp_servers_for_worker(  # noqa: PLR0911 -- early-return dispatch over MCP config shape variants
    mcp_config: str | dict[str, Any] | None,
    mcp_config_preset: str | None,
    mcp_servers: list[str] | None,
    worker_name: str,
    db_path_str: str,
) -> list[Any]:
    """Set up MCP servers for a worker.

    Args:
        mcp_config: MCP configuration as file path or dict
        mcp_config_preset: MCP preset name
        mcp_servers: List of specific MCP servers
        worker_name: Name of the worker
        db_path_str: Path to the worker's database

    Returns:
        List of MCP server instances
    """
    if not PYDANTIC_AI_AVAILABLE:
        if any([mcp_config, mcp_config_preset, mcp_servers]):
            logger.warning(
                "MCP server configuration requested but PydanticAI is not available. "
                "Worker %s will be created without MCP servers.",
                worker_name,
            )
        return []

    if create_mcp_servers_from_config is None:
        logger.error("MCP server creation function not available")
        return []

    resolved_mcp_config = await resolve_mcp_config(
        mcp_config=mcp_config,
        mcp_config_preset=mcp_config_preset,
        mcp_servers=mcp_servers,
        worker_name=worker_name,
    )

    if isinstance(resolved_mcp_config, dict):
        running_in_docker = os.getenv("RUNNING_IN_DOCKER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        resolved_mcp_config = _normalize_mcp_config_for_environment(
            resolved_mcp_config,
            running_in_docker=running_in_docker,
        )

    if not resolved_mcp_config:
        return []

    # Debug logging - only when MCP config is actually provided
    logger.info(f"Original mcp_servers parameter: {mcp_servers} (type: {type(mcp_servers)})")
    logger.info(f"Resolved MCP config: {resolved_mcp_config} (type: {type(resolved_mcp_config)})")
    logger.info("Loading MCP server configuration for worker %s...", worker_name)
    logger.debug("Resolved MCP config type: %s", type(resolved_mcp_config))
    logger.debug("Resolved MCP config content: %s", resolved_mcp_config)

    base_dir_for_mcp_resolution: Path | None = None

    if isinstance(resolved_mcp_config, str):
        config_file_path = Path(resolved_mcp_config).resolve()
        if config_file_path.exists():
            base_dir_for_mcp_resolution = config_file_path.parent
        else:
            logger.warning(
                "MCP config file path specified but not found: %s. Trying relative to DB path.",
                config_file_path,
            )
            potential_path = Path(db_path_str).parent / resolved_mcp_config
            if potential_path.exists():
                resolved_mcp_config = str(potential_path.resolve())
                base_dir_for_mcp_resolution = potential_path.parent
                logger.info("Resolved MCP config path to: %s", resolved_mcp_config)
            else:
                logger.error(
                    "MCP config file %s not found at specified path or relative to DB path.",
                    resolved_mcp_config,
                )
                return []
    elif isinstance(resolved_mcp_config, dict):
        base_dir_for_mcp_resolution = Path(db_path_str).parent.resolve()
    else:
        # Handle unexpected types
        logger.error(
            "Resolved MCP config has unexpected type %s: %s",
            type(resolved_mcp_config),
            resolved_mcp_config,
        )
        return []

    try:
        # Validate that resolved_mcp_config is not a list before passing to create_mcp_servers_from_config
        if isinstance(resolved_mcp_config, list):
            logger.error(
                "Resolved MCP config is a list instead of dict for worker %s: %s. "
                "This indicates a bug in resolve_mcp_config function.",
                worker_name,
                resolved_mcp_config,
            )
            return []

        mcp_servers_list = create_mcp_servers_from_config(
            resolved_mcp_config, base_dir_for_mcp_resolution
        )
        logger.info(
            "Created %d MCP servers for worker %s.",
            len(mcp_servers_list),
            worker_name,
        )
        return mcp_servers_list
    except AttributeError:
        logger.exception(
            "Attribute error creating MCP servers for worker %s. This often means a list was passed instead of a dict.",
            worker_name,
        )
        return []
    except ValueError:
        logger.exception(
            "Configuration error creating MCP servers for worker %s",
            worker_name,
        )
        return []
    except OSError:
        logger.exception(
            "File system error creating MCP servers for worker %s",
            worker_name,
        )
        return []
    except KeyError:
        logger.exception(
            "Missing configuration key for MCP servers for worker %s",
            worker_name,
        )
        return []


async def create_worker(  # noqa: PLR0913 -- worker factory with one arg per configurable axis
    worker_name: str,
    tool_names: list[str],
    agent_role: str | None = None,
    agent_type: str | None = None,
    model_provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    instruction: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    description: str | None = None,
    deps: list[str] | None = None,
    mcp_config: str | dict[str, Any] | None = None,
    mcp_config_preset: str | None = None,
    mcp_servers: list[str] | None = None,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
    tools_dir_override: str | None = None,
    output_schema: dict[str, Any] | None = None,
    output_validator_code: str | None = None,
    output_retries: int | None = None,
    disable_state_storage: bool = False,  # noqa: FBT001, FBT002, RUF100
    container_id: str | None = None,
    executor_endpoint: str | None = None,
    workspace: str | None = None,
    max_tokens: int | None = None,
    enable_reasoning: bool | None = None,  # noqa: FBT001 -- keyword-only at every call site
    seed: int | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    ir_provider: str | None = None,
    worker_version: str | None = None,
    # Identity. Threaded so the pydantic-ai provider can stamp a REAL
    # X-Tenant-Id on outbound inference-router calls instead of "". Safe to
    # bake because WorkerInstanceCache is keyed per tenant; see the
    # InferenceRouterOpenAIProvider docstring for why license/subject are not.
    tenant_id: str | None = None,
    # True only when api_key_override is the tenant's own BYO key, resolved by
    # llm_config_resolver. Admits a non-platform tenant to an external provider
    # and forbids the owner env-key fallback (ModelFactory.get_api_key).
    tenant_byo_key: bool = False,  # noqa: FBT001, FBT002 -- keyword-only at every call site
) -> Any:
    """Create a worker instance using the legacy function signature.

    This function maintains backward compatibility with the original create_worker
    function signature while delegating to the new configuration-based implementation.

    Args:
        worker_name: Name of the worker
        tool_names: List of tool names to load for the worker
        agent_role: Role of the agent (if None, creates basic worker)
        agent_type: Type of agent implementation (defaults to "pydantic_ai")
        model_provider: AI model provider (e.g., "mlx", "openai", "deepseek")
        model: AI model name
        temperature: Model temperature setting (0.0-1.0)
        instruction: Instruction text for the agent
        system_prompt: System prompt for the agent
        user_prompt: User prompt for the agent
        description: Description of the worker
        deps: List of dependencies
        mcp_config: MCP configuration as file path or dict
        mcp_config_preset: MCP preset configuration name
        mcp_servers: List of specific MCP servers to use
        base_url_override: Override for API base URL
        api_key_override: Override for API key
        tools_dir_override: Override for tools directory
        output_schema: JSON Schema for structured output
        output_validator_code: Python code for custom output validation
        disable_state_storage: If True, skip ALL database operations
        max_tokens: Per-worker generation budget (None ⇒ provider default)
        enable_reasoning: Tri-state chain-of-thought switch. None ⇒ emit nothing
            downstream (unchanged behaviour); True/False are forwarded to
            self-hosted backends only.
        seed: Sampling seed. None ⇒ no seed key is sent (unchanged behaviour).
        repetition_penalty: vLLM repetition penalty. None ⇒ nothing is sent.
        frequency_penalty: OpenAI frequency_penalty. None ⇒ nothing is sent.

    Returns:
        Worker instance (either agent-based or basic template)
    """
    # Load tools and set up basic infrastructure
    tools = await _load_worker_tools(tool_names, worker_name)
    worker_app = FastAPI()
    db_path_str = _setup_worker_database(worker_name, disable_state_storage)

    logger.info("Creating worker %s with database at %s", worker_name, db_path_str)

    # Set up MCP servers if needed
    mcp_servers_list = await _setup_mcp_servers_for_worker(
        mcp_config, mcp_config_preset, mcp_servers, worker_name, db_path_str
    )

    # Create agent-based worker if agent_role is specified
    if agent_role:
        # For MLX provider, always resolve the correct instance URL
        resolved_base_url = base_url_override
        if model_provider == "mlx" and model:
            resolved_url = await mlx_model_registry.resolve_base_url(
                model, fallback_url=base_url_override
            )
            if resolved_url:
                logger.info(f"MLX model '{model}' resolved to instance: {resolved_url}")
                resolved_base_url = resolved_url
        elif model_provider == "inference_router":
            from src.services.inference_router_registry import inference_router_registry

            ir_url = inference_router_registry.get_base_url()
            resolved_base_url = base_url_override or ir_url
            if model:
                ir_model = await inference_router_registry.get_model(model, tenant_id)
                if ir_model:
                    logger.info(
                        f"Inference-router schema '{model}' validated, chain: {ir_model.provider_chain}"
                    )
                else:
                    logger.warning(
                        f"Schema '{model}' not found in inference-router registry, proceeding anyway"
                    )

        worker_config, agent_config = _create_worker_configs(
            worker_name,
            tool_names,
            agent_role,
            agent_type,
            model_provider,
            model,
            temperature,
            instruction,
            system_prompt,
            user_prompt,
            description,
            deps,
            resolved_base_url,  # Use resolved URL instead of original
            api_key_override,
            tools_dir_override,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            seed=seed,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            ir_provider=ir_provider,
            tenant_byo_key=tenant_byo_key,
        )

        return await _create_agent_worker(
            worker_config,
            agent_config,
            tools,
            db_path_str,
            mcp_servers_list,
            mcp_servers,
            output_schema,
            output_validator_code,
            container_id=container_id,
            executor_endpoint=executor_endpoint,
            workspace=workspace,
            output_retries=output_retries,
            worker_version=worker_version,
            tenant_id=tenant_id,
        )

    # Create basic worker without agent capabilities
    return _create_basic_worker(worker_name, description, tools, worker_app, db_path_str)
