"""Base class for CLI-based agents (Claude Code, Codex).

Extracts shared execution infrastructure: K8s jobs, agent-host containers,
proxy delegation, and direct CLI execution.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess  # nosec B404 -- only used with list-form argv + shell=False
from abc import abstractmethod
from typing import Any
from uuid import UUID, uuid7

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.api.identity_context import get_identity_headers
from src.telemetry.agent_metrics import AgentExecutionTimer

from .base_agent import AgentRunResult, BaseAgent

logger = logging.getLogger(__name__)


class CLIAgentConfig(BaseModel):
    """Base configuration for CLI-based agents."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    system_prompt: str | None = Field(default="You are a helpful coding assistant.")
    instruction: str | None = None
    db_path: str
    working_directory: str | None = Field(default=None)
    model: str | None = Field(default=None)
    timeout: int = Field(default=300)
    use_proxy: bool = Field(default=False)
    proxy_url: str | None = Field(default=None)
    use_k8s: bool = Field(default=False)
    k8s_namespace: str = Field(default="agent-jobs")
    k8s_session_secret: str = Field(default="claude-secrets")
    k8s_executor_image: str | None = Field(default=None)
    use_agent_host: bool = Field(default=False)
    agent_host_container_id: str | None = Field(default=None)
    agent_host_url: str | None = Field(default=None)


class CLIAgent(BaseAgent):
    """Base class for agents that wrap CLI tools.

    Provides shared infrastructure for K8s job execution, agent-host
    containers, proxy delegation, and direct CLI execution.
    Subclasses implement CLI-specific command building and output parsing.
    """

    cli_name: str = ""
    agent_type_label: str = ""
    model_provider_label: str = ""
    default_model_name: str = ""

    def __init__(self, config: CLIAgentConfig, agent_id: UUID | None = None) -> None:
        super().__init__(agent_id)
        self.agent_id = agent_id or uuid7()
        self.config = config
        self.name = config.name
        self.db_path = config.db_path

        if os.environ.get("USE_AGENT_HOST", "").lower() in ("true", "1", "yes"):
            self.config.use_agent_host = True
            logger.info(f"{self.__class__.__name__} '{self.name}' will use agent-host execution")
        elif os.environ.get("USE_K8S_EXECUTOR", "").lower() in ("true", "1", "yes"):
            self.config.use_k8s = True
            self.config.k8s_namespace = os.environ.get(
                "EXECUTOR_NAMESPACE", self.config.k8s_namespace
            )
            self.config.k8s_session_secret = os.environ.get(
                "CLAUDE_SESSION_SECRET_NAME", self.config.k8s_session_secret
            )
            self.config.k8s_executor_image = os.environ.get(
                self._get_k8s_image_env_var(), self.config.k8s_executor_image
            )
            logger.info(
                f"{self.__class__.__name__} '{self.name}' will use k8s executor in namespace "
                f"'{self.config.k8s_namespace}'"
            )
        elif os.environ.get("CLI_PROXY_ENABLED", "").lower() in ("true", "1", "yes"):
            self.config.use_proxy = True
            proxy_url = self._resolve_proxy_url()
            if not proxy_url:
                logger.warning(
                    f"{self.__class__.__name__} '{self.name}': CLI_PROXY_ENABLED=true but no proxy URL set."
                )
            else:
                self.config.proxy_url = proxy_url
                logger.info(
                    f"{self.__class__.__name__} '{self.name}' will use proxy at {proxy_url}"
                )

        if not self.config.use_proxy and not self.config.use_k8s and not self.config.use_agent_host:
            self._verify_cli_available()

        mode = (
            "agent-host"
            if self.config.use_agent_host
            else (
                "k8s" if self.config.use_k8s else "proxy" if self.config.use_proxy else "direct CLI"
            )
        )
        logger.info(
            f"{self.__class__.__name__} '{self.name}' (ID: {self.agent_id}) initialized with "
            f"working directory: {config.working_directory or 'current'}, mode: {mode}"
        )

    # --- Abstract methods subclasses must implement ---

    @abstractmethod
    def _get_cli_path(self) -> str: ...

    @abstractmethod
    def _set_cli_path(self, path: str) -> None: ...

    @abstractmethod
    def _get_k8s_image_env_var(self) -> str: ...

    @abstractmethod
    def _resolve_proxy_url(self) -> str | None: ...

    @abstractmethod
    def _get_proxy_endpoint(self) -> str: ...

    @abstractmethod
    def _build_proxy_payload(self, prompt: str) -> dict[str, Any]: ...

    @abstractmethod
    def _build_command(self, prompt: str) -> list[str]: ...

    @abstractmethod
    def _build_agent_host_command(self, prompt: str) -> list[str]: ...

    @abstractmethod
    def _parse_output(self, output: str) -> AgentRunResult: ...

    # --- Shared implementations ---

    def _identity_headers(self) -> dict[str, str]:
        """Identity headers for every outbound hop this agent makes.

        ⛔ WHY THIS EXISTS -- AND WHY IT IS NOT A COST EVENT. The CLI-agent
        fleet's spend IS already metered, downstream, by inference-router: of
        worker_management's ~14,840 historical cost events, 14,833 had an exact
        inference-router twin. Emitting again here would DOUBLE-BILL, and
        ``emit_byo_cost_event_task`` refuses a platform-routed provider for
        precisely that reason.

        What was actually missing is ORIGIN. Every httpx call on these paths
        went out bare -- no tenant, no subject, no app, no worker -- so IR
        recorded the spend against worker_management's own service account and
        could not say which app or worker it came from. These headers are the
        whole fix: same spend, now attributable.

        The dict comes from the ONE canonical builder
        (``identity_context.get_identity_headers``), which is also what the
        pydantic-ai httpx hook's contract mirrors. Four copies of a header dict
        is how ``_create_vllm_model`` ended up sending the worker name without
        the identity triple; there is one builder so there cannot be a fifth
        divergence.
        """
        return get_identity_headers(worker_name=self.name)

    def _verify_cli_available(self) -> None:
        cli_path = self._get_cli_path()
        if cli_path == self.cli_name:
            resolved_path = shutil.which(cli_path)
            if resolved_path:
                self._set_cli_path(resolved_path)
                cli_path = resolved_path
                logger.debug("Resolved %s CLI path to %s via shutil.which", self.cli_name, cli_path)

        try:
            result = subprocess.run(  # nosec B603 -- list-form argv, no shell; cli_path resolved via shutil.which  # noqa: S603
                [cli_path, "--version"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode == 0:
                logger.info(
                    f"{self.cli_name} CLI found: {cli_path} (version info: {result.stdout.strip()})"
                )
            else:
                logger.warning(
                    f"{self.cli_name} CLI responded with non-zero exit code: {result.returncode}"
                )
        except FileNotFoundError:
            error_msg = (
                f"{self.cli_name} CLI not found at: {cli_path}. "
                f"Please ensure {self.cli_name} is installed and in PATH."
            )
            logger.exception(error_msg)
            raise FileNotFoundError(error_msg) from None
        except Exception as e:
            logger.warning(f"Could not verify {self.cli_name} availability: {e}")

    async def _execute_via_k8s(self, prompt: str) -> str:
        try:
            from src.k8s_job_executor import K8sJobExecutor
        except ImportError:
            raise RuntimeError(
                "K8sJobExecutor not available. Ensure kubernetes package is installed "
                "and running inside a cluster."
            ) from None

        executor = K8sJobExecutor(namespace=self.config.k8s_namespace)
        await executor.initialize()

        from uuid import uuid7

        execution_id = uuid7()
        config = {"ttl_seconds": 300, "debug": False}

        if self.config.k8s_executor_image:
            os.environ.setdefault(self._get_k8s_image_env_var(), self.config.k8s_executor_image)

        if self.config.k8s_session_secret != "claude-secrets":  # nosec B105 -- default K8s Secret name constant, not a credential value  # noqa: S105
            os.environ.setdefault("CLAUDE_SESSION_SECRET_NAME", self.config.k8s_session_secret)

        result = await executor.execute_agent(
            agent_name=self.name,
            prompt=prompt,
            execution_id=execution_id,
            config=config,
            image_override=self.config.k8s_executor_image,
        )

        logs = result.get("logs", "")
        for line in reversed(str(logs).split("\n")):
            line = line.strip()  # noqa: PLW2901 -- intentional inline normalize
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    if "result" in parsed or "type" in parsed:
                        return line
                except json.JSONDecodeError:
                    continue
        return str(logs)

    async def _execute_via_agent_host(self, prompt: str) -> str:
        from src.services.agent_host_registry import agent_host_registry

        if not self.config.agent_host_container_id:
            instance = await agent_host_registry.get_best_instance()
            if not instance:
                raise RuntimeError("No healthy agent-host instances available")

            self.config.agent_host_url = instance.url
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{instance.url}/containers",
                    json={
                        "container_type": self.cli_name,
                        "worker_name": self.name,
                        "worker_id": str(self.agent_id),
                        "timeout": self.config.timeout * 4,
                    },
                    headers=self._identity_headers(),
                )
                if resp.status_code != 201:
                    raise RuntimeError(
                        f"agent-host container create failed: {resp.status_code} {resp.text}"
                    )
                data = resp.json()
                self.config.agent_host_container_id = data["container_id"]
                logger.info(
                    "Created agent-host container %s on %s",
                    data["container_name"],
                    instance.name,
                )

        cmd = self._build_agent_host_command(prompt)
        async with httpx.AsyncClient(timeout=self.config.timeout + 30) as client:
            resp = await client.post(
                f"{self.config.agent_host_url}/containers/{self.config.agent_host_container_id}/exec",
                json={
                    "command": cmd,
                    "workdir": "/workspace",
                    "timeout": self.config.timeout,
                    "user": "node",
                },
                headers=self._identity_headers(),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"agent-host exec failed: {resp.status_code} {resp.text}")
            data = resp.json()
            if not data.get("success"):
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                raise RuntimeError(
                    f"agent-host exec failed (exit={data.get('exit_code')}): "
                    f"stderr={stderr}, stdout={stdout[:500]}"
                )
            return data.get("stdout", "")

    async def _execute_direct(self, cmd: list[str]) -> str:
        working_dir = self.config.working_directory or os.getcwd()
        logger.debug(f"Executing {self.cli_name} in directory: {working_dir}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.config.timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(
                    f"{self.cli_name} execution failed with code {process.returncode}: {error_msg}"
                )
                raise RuntimeError(
                    f"{self.cli_name} failed with exit code {process.returncode}: {error_msg}"
                )

            output = stdout.decode()
            logger.debug(f"{self.cli_name} output length: {len(output)} characters")
            return output

        except TimeoutError:
            logger.exception(f"{self.cli_name} execution timed out after {self.config.timeout}s")
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.warning(f"Failed to kill timed-out process: {e}")
            raise

    async def _execute_via_proxy(self, prompt: str) -> str:
        if not self.config.proxy_url:
            raise RuntimeError("Proxy URL not configured")

        proxy_endpoint = f"{self.config.proxy_url}{self._get_proxy_endpoint()}"
        logger.debug(f"Sending request to proxy: {proxy_endpoint}")

        payload = self._build_proxy_payload(prompt)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout + 10) as client:
                response = await client.post(
                    proxy_endpoint, json=payload, headers=self._identity_headers()
                )

                if response.status_code != 200:
                    error_msg = f"Proxy returned status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                result = response.json()

                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown proxy error")
                    logger.error(f"Proxy execution failed: {error_msg}")
                    raise RuntimeError(f"Proxy execution failed: {error_msg}")

                output = result.get("output", "")
                execution_time = result.get("execution_time", 0)
                logger.info(f"Proxy execution completed in {execution_time:.2f}s")
                return str(output)

        except httpx.TimeoutException:
            logger.exception(f"Proxy request timed out after {self.config.timeout}s")
            raise
        except Exception as e:
            logger.exception("Proxy execution error")
            raise RuntimeError(f"Proxy execution error: {e!s}") from e

    async def run(self, prompt: str | list[Any]) -> AgentRunResult:
        """Run the CLI agent against the given text prompt and return its parsed result."""
        if isinstance(prompt, list):
            raise NotImplementedError(
                "CLI agents do not support multimodal (list) input — "
                "use agent_type='pydantic_ai' against a vision-capable model "
                "(claude-sonnet-4-5, gpt-4o, gemini-2-pro, gemma-4) instead."
            )
        mode = (
            "agent-host"
            if self.config.use_agent_host
            else (
                "k8s" if self.config.use_k8s else "proxy" if self.config.use_proxy else "direct CLI"
            )
        )
        logger.info(
            f"Running {self.__class__.__name__} '{self.name}' (ID: {self.agent_id}) via {mode}"
        )

        with AgentExecutionTimer(
            agent_name=self.name,
            agent_type=self.agent_type_label,
            model_provider=self.model_provider_label,
            model_name=self.config.model or self.default_model_name,
        ) as timer:
            try:
                if self.config.use_agent_host:
                    output = await self._execute_via_agent_host(prompt)
                elif self.config.use_k8s:
                    output = await self._execute_via_k8s(prompt)
                elif self.config.use_proxy:
                    output = await self._execute_via_proxy(prompt)
                else:
                    cmd = self._build_command(prompt)
                    output = await self._execute_direct(cmd)

                result = self._parse_output(output)

                if result.usage_metrics:
                    timer.record_tokens(
                        {
                            "prompt_tokens": result.usage_metrics.prompt_tokens or 0,
                            "completion_tokens": result.usage_metrics.completion_tokens or 0,
                            "total_tokens": result.usage_metrics.total_tokens or 0,
                        }
                    )
                return result

            except TimeoutError:
                error_msg = (
                    f"{self.__class__.__name__} '{self.name}' execution timed out "
                    f"after {self.config.timeout}s"
                )
                logger.exception(error_msg)
                raise TimeoutError(error_msg) from None
            except Exception as e:
                error_msg = f"{self.__class__.__name__} '{self.name}' execution failed: {e!s}"
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from e

    async def check_health(self) -> bool:
        """Return True when the configured CLI binary is reachable."""
        try:
            self._verify_cli_available()
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.__class__.__name__} '{self.name}': {e}")
            return False

    async def cleanup(self) -> None:
        """Release CLI-side resources (proxy session, temp working dir, etc.)."""
        logger.info(f"Cleaning up {self.__class__.__name__} '{self.name}'")
        if (
            self.config.use_agent_host
            and self.config.agent_host_container_id
            and self.config.agent_host_url
        ):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.delete(
                        f"{self.config.agent_host_url}/containers/"
                        f"{self.config.agent_host_container_id}",
                        headers=self._identity_headers(),
                    )
                    if resp.status_code == 200:
                        logger.info(
                            "Destroyed agent-host container %s",
                            self.config.agent_host_container_id,
                        )
                    else:
                        logger.warning("Failed to destroy agent-host container: %s", resp.text)
            except Exception as e:
                logger.warning("Error destroying agent-host container: %s", e)
            finally:
                self.config.agent_host_container_id = None
