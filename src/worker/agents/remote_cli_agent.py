"""Unified remote CLI agent for code_graph_helper containers.

Replaces separate remote_claude_agent.py and remote_codex_agent.py with
a single parameterized implementation. The only difference between them
was the `cli` parameter passed to the executor.
"""

import json
import logging
from typing import Any, Literal
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field

from src.telemetry.agent_metrics import AgentExecutionTimer

from .base_agent import AgentRunResult, BaseAgent
from .remote_container_executor import RemoteContainerExecutor, RemoteExecutionConfig

logger = logging.getLogger(__name__)

# Mapping from CLI name to metrics labels
_CLI_METRICS = {
    "claude": {
        "agent_type": "remote_claude",
        "model_provider": "anthropic",
        "default_model": "claude-sonnet-4-20250514",
    },
    "codex": {
        "agent_type": "remote_codex",
        "model_provider": "openai",
        "default_model": "gpt-5-codex",
    },
}


class RemoteCLIAgentConfig(BaseModel):
    """Configuration for a remote CLI agent (Claude or Codex)."""

    model_config = ConfigDict(protected_namespaces=())

    cli: Literal["claude", "codex"] = Field(..., description="Which CLI to delegate to")
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    system_prompt: str | None = Field(default="You are a helpful coding assistant.")
    instruction: str | None = None
    db_path: str
    executor_endpoint: str | None = Field(default=None)
    container_id: str = Field(..., description="Container ID where agent should execute")
    workspace: str = Field(default="/workspace")
    model: str | None = Field(default=None)
    timeout: int = Field(default=300)


class RemoteCLIAgent(BaseAgent):
    """Remote CLI agent that executes in code_graph_helper containers.

    Works for both Claude and Codex — the `cli` config field determines
    which CLI tool is invoked in the remote container.
    """

    def __init__(self, config: RemoteCLIAgentConfig, agent_id: UUID | None = None) -> None:
        super().__init__(agent_id)
        self.agent_id = agent_id or uuid7()
        self.config = config
        self.name = config.name
        self.db_path = config.db_path

        executor_config = RemoteExecutionConfig(
            executor_endpoint=config.executor_endpoint or "",
            container_id=config.container_id,
            workspace=config.workspace,
            timeout=config.timeout,
        )
        self.executor = RemoteContainerExecutor(executor_config)

        metrics = _CLI_METRICS[config.cli]
        self._agent_type = metrics["agent_type"]
        self._model_provider = metrics["model_provider"]
        self._default_model = metrics["default_model"]

        endpoint = executor_config.raw_executor_endpoint()
        logger.info(
            f"RemoteCLIAgent[{config.cli}] '{self.name}' (ID: {self.agent_id}) initialized. "
            f"Executor: {endpoint}, Container: {config.container_id}"
        )

    async def run(self, prompt: str | list[Any]) -> AgentRunResult:
        """Run the prompt against the remote-container CLI agent and return parsed output."""
        if isinstance(prompt, list):
            raise NotImplementedError(
                "Remote CLI agents do not support multimodal (list) input — "
                "use agent_type='pydantic_ai' against a vision-capable model "
                "(claude-sonnet-4-5, gpt-4o, gemini-2-pro, gemma-4) instead."
            )
        logger.info(
            f"Running RemoteCLIAgent[{self.config.cli}] '{self.name}' "
            f"in remote container {self.config.container_id}"
        )

        with AgentExecutionTimer(
            agent_name=self.name,
            agent_type=self._agent_type,
            model_provider=self._model_provider,
            model_name=self.config.model or self._default_model,
        ) as timer:
            try:
                result = await self.executor.execute_agent(
                    cli=self.config.cli,
                    prompt=prompt,
                    system_prompt=self.config.system_prompt,
                    model=self.config.model,
                    # WHOSE WORK this remote execution is. Without it the
                    # identity headers name the tenant but not the unit of work
                    # inside this service, which is the half that lets one
                    # `service_key` be split across the many workers that fan
                    # into it.
                    worker_name=self.name,
                )

                if not result.success:
                    error_msg = result.error or "Unknown remote execution error"
                    logger.error(f"Remote execution failed: {error_msg}")
                    raise RuntimeError(f"Remote execution failed: {error_msg}")

                parsed_result = self._parse_output(result.output)

                if parsed_result.usage_metrics:
                    timer.record_tokens(
                        {
                            "prompt_tokens": parsed_result.usage_metrics.prompt_tokens or 0,
                            "completion_tokens": parsed_result.usage_metrics.completion_tokens or 0,
                            "total_tokens": parsed_result.usage_metrics.total_tokens or 0,
                        }
                    )
                return parsed_result

            except Exception as e:
                error_msg = (
                    f"RemoteCLIAgent[{self.config.cli}] '{self.name}' execution failed: {e!s}"
                )
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from e

    def _parse_output(self, output: str) -> AgentRunResult:
        try:
            parsed = json.loads(output)
            return AgentRunResult(output=parsed, usage_metrics=None)
        except json.JSONDecodeError:
            return AgentRunResult(output=output, usage_metrics=None)

    async def check_health(self) -> bool:
        """Return True when the remote executor reports the container is reachable."""
        try:
            return await self.executor.check_health()
        except Exception as e:
            logger.warning(
                f"Health check failed for RemoteCLIAgent[{self.config.cli}] '{self.name}': {e}"
            )
            return False

    async def cleanup(self) -> None:
        """Tear down any remote-container resources held by this agent."""
        logger.info(f"Cleaning up RemoteCLIAgent[{self.config.cli}] '{self.name}'")


def _create_remote_agent(  # noqa: PLR0913 -- remote-agent factory with one arg per configurable axis
    cli: Literal["claude", "codex"],
    name: str,
    description: str,
    db_path: str,
    container_id: str,
    executor_endpoint: str | None = None,
    system_prompt: str | None = None,
    instruction: str | None = None,
    workspace: str = "/workspace",
    model: str | None = None,
    timeout: int = 300,
    **kwargs: Any,  # noqa: ARG001
) -> RemoteCLIAgent:
    final_system_prompt = system_prompt or "You are a helpful coding assistant."
    if instruction:
        final_system_prompt = f"{final_system_prompt}\n\n{instruction}"

    config = RemoteCLIAgentConfig(
        cli=cli,
        name=name,
        description=description,
        system_prompt=final_system_prompt,
        instruction=instruction,
        db_path=db_path,
        executor_endpoint=executor_endpoint,
        container_id=container_id,
        workspace=workspace,
        model=model,
        timeout=timeout,
    )

    logger.info(f"Creating RemoteCLIAgent[{cli}] '{name}' targeting container {container_id}")
    return RemoteCLIAgent(config=config)


def create_remote_claude_agent(**kwargs: Any) -> RemoteCLIAgent:
    """Construct a RemoteCLIAgent targeting the `claude` CLI in a remote container."""
    return _create_remote_agent(cli="claude", **kwargs)


def create_remote_codex_agent(**kwargs: Any) -> RemoteCLIAgent:
    """Construct a RemoteCLIAgent targeting the `codex` CLI in a remote container."""
    return _create_remote_agent(cli="codex", **kwargs)


def _register_remote_agents() -> None:
    try:
        from .agent_registry import register_agent

        register_agent("remote_claude", create_remote_claude_agent, RemoteCLIAgent)
        register_agent("remote_codex", create_remote_codex_agent, RemoteCLIAgent)
        logger.info("Registered remote_claude and remote_codex agent types")
    except Exception as e:
        logger.warning(f"Failed to register remote agents: {e}")


_register_remote_agents()
