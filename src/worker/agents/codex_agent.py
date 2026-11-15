"""Codex agent implementation for worker management system.

This module provides a wrapper for the Codex CLI tool, allowing it to be
used as an agent type within the worker management system.
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .base_agent import AgentRunResult
from .cli_agent import CLIAgent, CLIAgentConfig

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)


class CodexAgentConfig(CLIAgentConfig):
    """Configuration for a Codex agent."""

    codex_path: str = Field(default="codex", description="Path to codex executable")
    mcp_servers: list[str] = Field(
        default_factory=list, description="List of MCP servers to enable"
    )


class CodexAgent(CLIAgent):
    """Agent wrapper for Codex CLI tool."""

    cli_name = "codex"
    agent_type_label = "codex"
    model_provider_label = "openai"
    default_model_name = "codex-default"
    config: CodexAgentConfig

    def __init__(self, config: CodexAgentConfig, agent_id: UUID | None = None) -> None:
        super().__init__(config, agent_id)

    # --- CLIAgent abstract method implementations ---

    def _get_cli_path(self) -> str:
        return self.config.codex_path

    def _set_cli_path(self, path: str) -> None:
        self.config.codex_path = path

    def _get_k8s_image_env_var(self) -> str:
        return "K8S_CODEX_EXECUTOR_IMAGE"

    def _resolve_proxy_url(self) -> str | None:
        return os.environ.get("CLI_PROXY_URL", "http://host.docker.internal:9000")

    def _get_proxy_endpoint(self) -> str:
        return "/codex"

    def _build_proxy_payload(self, prompt: str) -> dict[str, Any]:
        combined_prompt = prompt
        if self.config.system_prompt:
            combined_prompt = f"{self.config.system_prompt}\n\n{prompt}"

        payload: dict[str, Any] = {"prompt": combined_prompt, "timeout": self.config.timeout}
        if self.config.model:
            payload["model"] = self.config.model
        if self.config.working_directory:
            payload["working_directory"] = self.config.working_directory
        return payload

    def _build_command(self, prompt: str) -> list[str]:
        cmd = [self.config.codex_path, "exec"]
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        combined_prompt = prompt
        if self.config.system_prompt:
            combined_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        cmd.append(combined_prompt)

        logger.debug(f"Built codex command: {' '.join(cmd[:4])}...")
        return cmd

    def _build_agent_host_command(self, prompt: str) -> list[str]:
        combined_prompt = prompt
        if self.config.system_prompt:
            combined_prompt = f"{self.config.system_prompt}\n\n{prompt}"

        cmd = ["codex", "exec"]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        cmd.append(combined_prompt)
        return cmd

    def _parse_output(self, output: str) -> AgentRunResult:
        try:
            parsed = json.loads(output)
            return AgentRunResult(output=parsed, usage_metrics=None)
        except json.JSONDecodeError:
            return AgentRunResult(output=output, usage_metrics=None)


def create_codex_agent(
    name: str,
    description: str,
    db_path: str,
    system_prompt: str | None = None,
    instruction: str | None = None,
    working_directory: str | None = None,
    codex_path: str = "codex",
    model: str | None = None,
    timeout: int = 300,
    mcp_servers: list[str] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> CodexAgent:
    """Construct a CodexAgent backed by the local `codex` CLI."""
    final_system_prompt = system_prompt or "You are a helpful coding assistant."
    if instruction:
        final_system_prompt = f"{final_system_prompt}\n\n{instruction}"

    config = CodexAgentConfig(
        name=name,
        description=description,
        system_prompt=final_system_prompt,
        instruction=instruction,
        db_path=db_path,
        working_directory=working_directory,
        codex_path=codex_path,
        model=model,
        timeout=timeout,
        mcp_servers=mcp_servers or [],
    )

    logger.info(f"Creating CodexAgent '{name}' with model {model or 'default'}")
    return CodexAgent(config=config)


def _register_codex_agent() -> None:
    try:
        from .agent_registry import register_agent

        register_agent("codex", create_codex_agent, CodexAgent)
        logger.info("Registered codex agent type with global registry")
    except Exception as e:
        logger.warning(f"Failed to register codex agent: {e}")


_register_codex_agent()
