"""Claude Code agent implementation for worker management system.

This module provides a wrapper for the Claude Code CLI tool, allowing it to be
used as an agent type within the worker management system.
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx2 as httpx
from pydantic import Field

from .base_agent import AgentRunResult, TokenUsageMetrics
from .cli_agent import CLIAgent, CLIAgentConfig

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)


class ClaudeCodeAgentConfig(CLIAgentConfig):
    """Configuration for a Claude Code agent."""

    claude_code_path: str = Field(default="claude", description="Path to claude executable")
    mcp_servers: list[str | dict[str, Any]] = Field(
        default_factory=list,
        description="List of MCP servers. Can be strings (name), dicts with name/url, "
        "or dicts with command/args/env for stdio transport",
    )


class ClaudeCodeAgent(CLIAgent):
    """Agent wrapper for Claude Code CLI tool."""

    cli_name = "claude"
    agent_type_label = "claude_code"
    model_provider_label = "anthropic"
    default_model_name = "claude-sonnet-4-20250514"
    config: ClaudeCodeAgentConfig

    def __init__(self, config: ClaudeCodeAgentConfig, agent_id: UUID | None = None) -> None:
        super().__init__(config, agent_id)

        # Configure MCP servers if any are specified
        if self.config.mcp_servers:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    self._mcp_setup_task = loop.create_task(self._configure_mcp_servers())
                    logger.info(f"MCP server configuration scheduled for agent '{self.name}'")
                except RuntimeError:
                    asyncio.run(self._configure_mcp_servers())
            except Exception as e:
                logger.warning(f"Failed to configure MCP servers for agent '{self.name}': {e}")

    # --- CLIAgent abstract method implementations ---

    def _get_cli_path(self) -> str:
        return self.config.claude_code_path

    def _set_cli_path(self, path: str) -> None:
        self.config.claude_code_path = path

    def _get_k8s_image_env_var(self) -> str:
        return "K8S_EXECUTOR_IMAGE"

    def _resolve_proxy_url(self) -> str | None:
        return os.environ.get("CLAUDE_CODE_PROXY_URL", os.environ.get("CLI_PROXY_URL"))

    def _get_proxy_endpoint(self) -> str:
        return "/claude"

    def _build_proxy_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt, "timeout": self.config.timeout}
        if self.config.model:
            payload["model"] = self.config.model
        if self.config.system_prompt:
            payload["system_prompt"] = self.config.system_prompt
        if self.config.working_directory:
            payload["working_directory"] = self.config.working_directory
        return payload

    def _build_command(self, prompt: str) -> list[str]:
        cmd = [self.config.claude_code_path, "-p", prompt, "--output-format", "json"]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if self.config.system_prompt:
            cmd.extend(["--system-prompt", self.config.system_prompt])
        logger.debug(f"Built claude-code command: {' '.join(cmd)}")
        return cmd

    def _build_agent_host_command(self, prompt: str) -> list[str]:
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if self.config.system_prompt:
            cmd.extend(["--system-prompt", self.config.system_prompt])
        return cmd

    def _parse_output(self, output: str) -> AgentRunResult:
        try:
            parsed = json.loads(output)
            usage_metrics = None
            if "usage" in parsed:
                usage = parsed["usage"]
                usage_metrics = TokenUsageMetrics(
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                )
            return AgentRunResult(
                output=parsed.get("result", output),
                usage_metrics=usage_metrics,
                session_id=parsed.get("session_id"),
                cost_usd=parsed.get("total_cost_usd"),
                duration_ms=parsed.get("duration_ms"),
                is_error=parsed.get("is_error", False),
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse claude-code output as JSON, returning raw output")
            return AgentRunResult(output=output, usage_metrics=None)

    # --- MCP server configuration (Claude-specific) ---

    async def _configure_mcp_servers(self) -> None:
        if not self.config.mcp_servers:
            return

        server_names = [
            s if isinstance(s, str) else s.get("name", "unknown") for s in self.config.mcp_servers
        ]
        logger.info(
            f"Configuring {len(self.config.mcp_servers)} MCP server(s) for agent '{self.name}': "
            f"{', '.join(server_names)}"
        )

        for server in self.config.mcp_servers:
            if isinstance(server, str):
                server_name = server
                server_config = {"name": server_name}
                logger.warning(
                    f"MCP server '{server_name}' specified without URL/command. "
                    "This may fail. Use dict format with url or command."
                )
            elif isinstance(server, dict):
                server_name = str(server.get("name", ""))
                server_config = server
                if not server_name:
                    logger.error(f"MCP server config missing 'name': {server}. Skipping.")
                    continue
            else:
                logger.error(f"Invalid MCP server type: {type(server)}. Skipping.")
                continue

            try:
                if self.config.use_proxy:
                    await self._add_mcp_server_via_proxy(server_config)
                else:
                    server_url = server_config.get("url")
                    if server_url:
                        await self._add_mcp_server_direct(server_name, server_url)
                    else:
                        logger.warning(
                            f"Direct mode only supports URL-based MCPs. "
                            f"Skipping command-based MCP '{server_name}'. Use proxy mode instead."
                        )
                        continue
                logger.info(
                    f"Successfully configured MCP server '{server_name}' for agent '{self.name}'"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to configure MCP server '{server_name}' for agent '{self.name}': {e}"
                )

    async def _add_mcp_server_via_proxy(self, server_config: dict[str, Any]) -> None:
        if not self.config.proxy_url:
            raise RuntimeError("Proxy URL not configured")
        server_name = server_config.get("name")
        if not server_name:
            raise RuntimeError("MCP server config missing 'name'")

        proxy_endpoint = f"{self.config.proxy_url}/mcp/add"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    proxy_endpoint, json=server_config, headers=self._identity_headers()
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Proxy returned status {response.status_code}: {response.text}"
                    )
                result = response.json()
                if not result.get("success", False):
                    raise RuntimeError(f"MCP add failed: {result.get('error', 'Unknown error')}")
        except httpx.TimeoutException:
            logger.exception(f"Proxy request timed out while adding MCP server '{server_name}'")
            raise
        except Exception as e:
            logger.exception(f"Error adding MCP server '{server_name}' via proxy")
            raise RuntimeError(f"Error adding MCP server via proxy: {e!s}") from e

    async def _add_mcp_server_direct(self, name: str, url: str | None = None) -> None:
        if not url:
            raise RuntimeError(f"MCP server '{name}' requires a URL/command")
        try:
            cmd = [self.config.claude_code_path, "mcp", "add", name, url]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(
                    f"MCP add failed with exit code {process.returncode}: {error_msg}"
                )
        except TimeoutError:
            logger.exception(f"MCP add timed out for server '{name}'")
            raise
        except Exception:
            logger.exception(f"Error adding MCP server '{name}'")
            raise


def create_claude_code_agent(
    name: str,
    description: str,
    db_path: str,
    system_prompt: str | None = None,
    instruction: str | None = None,
    working_directory: str | None = None,
    claude_code_path: str = "claude",
    model: str | None = None,
    timeout: int = 300,
    mcp_servers: list[str] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> ClaudeCodeAgent:
    """Construct a ClaudeCodeAgent backed by the local `claude` CLI."""
    final_system_prompt = system_prompt or "You are a helpful coding assistant."
    if instruction:
        final_system_prompt = f"{final_system_prompt}\n\n{instruction}"

    config = ClaudeCodeAgentConfig(
        name=name,
        description=description,
        system_prompt=final_system_prompt,
        instruction=instruction,
        db_path=db_path,
        working_directory=working_directory,
        claude_code_path=claude_code_path,
        model=model,
        timeout=timeout,
        mcp_servers=list(mcp_servers) if mcp_servers else [],
    )

    logger.info(f"Creating ClaudeCodeAgent '{name}' with model {model or 'default'}")
    return ClaudeCodeAgent(config=config)


def _register_claude_code_agent() -> None:
    try:
        from .agent_registry import register_agent

        register_agent("claude_code", create_claude_code_agent, ClaudeCodeAgent)
        logger.info("Registered claude_code agent type with global registry")
    except Exception as e:
        logger.warning(f"Failed to register claude_code agent: {e}")


_register_claude_code_agent()
