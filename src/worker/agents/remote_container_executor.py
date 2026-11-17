"""Remote container executor for delegated agent execution.

This module enables worker_management to delegate agent execution to
coding-gateway. Service base URL comes from Settings (``CODING_SERVICE_URL``);
production fail-fast lives in ``Settings.model_post_init``.
"""

import logging
import os
from typing import Any, cast

import httpx2 as httpx
from pydantic import BaseModel, Field

from src.api.identity_context import get_identity_headers
from src.security.request_overrides import check_executor_endpoint, tls_verify_for
from src.settings import get_settings
from src.worker.service_authenticator import get_service_authenticator

logger = logging.getLogger(__name__)


class RemoteExecutionConfig(BaseModel):
    """Configuration for remote container execution."""

    executor_endpoint: str = Field(
        default="",
        description="Base URL of coding service (resolved from CODING_SERVICE_URL env when empty)",
    )
    container_id: str = Field(..., description="Container ID where agent should execute")
    workspace: str = Field(default="/workspace", description="Working directory in the container")
    timeout: int = Field(default=300, description="Timeout for execution in seconds")

    def get_executor_endpoint(self) -> str:
        """Get executor endpoint from config, env, or Settings (in that order).

        Always allowlist-checked: every caller sends this service's OAuth2 bearer
        to the result, so a stored or constructed value that is not listed raises
        :class:`ExecutorEndpointNotAllowedError` BEFORE any token is fetched.
        """
        return check_executor_endpoint(self.raw_executor_endpoint())

    def raw_executor_endpoint(self) -> str:
        """The UNCHECKED resolved endpoint — for logging only, never for a request."""
        return (
            self.executor_endpoint
            or os.getenv("CODING_SERVICE_URL")
            or get_settings().coding_service_url
        )


class AgentExecutionRequest(BaseModel):
    """Request to execute an agent in a remote container."""

    cli: str = Field(..., description="CLI to use: 'claude' or 'codex'")
    prompt: str = Field(..., description="Task prompt for the agent")
    system_prompt: str | None = Field(default=None, description="System prompt for the agent")
    model: str | None = Field(default=None, description="Model to use (e.g., 'sonnet', 'opus')")
    workspace: str = Field(default="/workspace", description="Working directory in container")
    timeout: int = Field(default=300, description="Execution timeout in seconds")


class AgentExecutionResponse(BaseModel):
    """Response from remote agent execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0
    execution_time: float = 0.0
    container_id: str | None = None


class RemoteContainerExecutor:
    """Executor that delegates agent execution to coding-gateway.

    This executor communicates with coding-gateway (in-cluster svc.cluster.local) to
    execute Claude/Codex in repository containers.
    """

    def __init__(self, config: RemoteExecutionConfig) -> None:
        """Initialize remote container executor.

        Args:
            config: Configuration for remote execution
        """
        self.config = config

        endpoint = config.raw_executor_endpoint()
        logger.info(
            f"RemoteContainerExecutor initialized. "
            f"Endpoint: {endpoint}, "
            f"Container: {config.container_id}"
        )

    async def execute_agent(
        self,
        cli: str,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        worker_name: str = "",
    ) -> AgentExecutionResponse:
        """Execute agent in remote container via API.

        Args:
            cli: CLI to use ('claude' or 'codex')
            prompt: Task prompt
            system_prompt: Optional system prompt
            model: Optional model specification
            worker_name: Originating worker, forwarded as ``X-Worker-Name`` so
                the spend coding-gateway triggers downstream is attributable to
                the worker that asked for it rather than to WM's service
                account. Empty ⇒ header omitted, never guessed.

        Returns:
            AgentExecutionResponse with execution results

        Raises:
            RuntimeError: If remote execution fails
            httpx.TimeoutException: If request times out
        """
        executor_endpoint = self.config.get_executor_endpoint()
        endpoint = f"{executor_endpoint}/api/containers/{self.config.container_id}/execute-agent"

        logger.info(
            f"Executing {cli} agent in remote container {self.config.container_id} "
            f"via {executor_endpoint}"
        )

        # Build request
        request = AgentExecutionRequest(
            cli=cli,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            workspace=self.config.workspace,
            timeout=self.config.timeout,
        )

        # Get OAuth2 token for service-to-service auth
        auth = get_service_authenticator()
        token = await auth.get_token()

        # ⛔ TWO DIFFERENT CREDENTIALS, AND THEY DO NOT MIX. `Authorization`
        # carries OUR service-to-service OAuth2 bearer -- it says which SERVICE
        # is calling, and it is the only thing coding-gateway authenticates on.
        # The identity headers say on WHOSE BEHALF, and they are what makes the
        # downstream spend attributable to a real tenant/subject/app/worker
        # instead of to this service account.
        #
        # ⛔ Never move a licence token onto `Authorization`: the estate
        # convention is `X-Tenant-Id: Bearer <token>`, and conflating the two is
        # exactly what caused the 2026-04-26 production 401s (see
        # auth_middleware._extract_license_token, which reads the INBOUND form
        # and is now the only place that rule is recorded -- outbound_license.py
        # was retired 2026-09-11 with the dead outbound credential it described).
        # `get_identity_headers` deliberately returns no
        # credential at all, so merging it here cannot clobber the bearer --
        # but it is spread FIRST so a future identity key can never silently
        # win over `Authorization` either.
        headers = {
            **get_identity_headers(worker_name=worker_name),
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout + 10,
                # Unverified only for in-cluster Kubernetes Service hostnames (ambient-mesh
                # mTLS); every other allowlisted host is TLS-verified.
                verify=tls_verify_for(endpoint),
            ) as client:
                response = await client.post(
                    endpoint,
                    json=request.model_dump(exclude_none=True),
                    headers=headers,
                )

                if response.status_code != 200:
                    error_msg = (
                        f"Remote executor returned status {response.status_code}: {response.text}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                result = response.json()
                logger.info(
                    f"Remote execution completed. "
                    f"Success: {result.get('success')}, "
                    f"Exit code: {result.get('exit_code')}"
                )

                return AgentExecutionResponse(**result)

        except httpx.TimeoutException:
            error_msg = (
                f"Remote execution timed out after {self.config.timeout}s "
                f"(container: {self.config.container_id})"
            )
            logger.exception(error_msg)
            raise
        except httpx.HTTPError as e:
            error_msg = f"HTTP error during remote execution: {e!s}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during remote execution: {e!s}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    async def check_health(self) -> bool:
        """Check if remote executor is available and healthy.

        Returns:
            True if executor is healthy, False otherwise
        """
        try:
            executor_endpoint = self.config.get_executor_endpoint()
            endpoint = f"{executor_endpoint}/health"

            # Get OAuth2 token for service-to-service auth
            auth = get_service_authenticator()
            token = await auth.get_token()

            async with httpx.AsyncClient(timeout=10, verify=tls_verify_for(endpoint)) as client:
                response = await client.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if response.status_code == 200:
                    logger.info(f"Remote executor health check passed: {executor_endpoint}")
                    return True
                else:
                    logger.warning(f"Remote executor health check failed: {response.status_code}")
                    return False

        except Exception as e:
            logger.warning(f"Remote executor health check error: {e}")
            return False

    async def get_container_info(self) -> dict[str, Any]:
        """Get information about the remote container.

        Returns:
            Dictionary with container information

        Raises:
            RuntimeError: If request fails
        """
        executor_endpoint = self.config.get_executor_endpoint()
        endpoint = f"{executor_endpoint}/api/containers/{self.config.container_id}"

        # Get OAuth2 token for service-to-service auth
        auth = get_service_authenticator()
        token = await auth.get_token()

        try:
            async with httpx.AsyncClient(timeout=10, verify=tls_verify_for(endpoint)) as client:
                response = await client.get(
                    endpoint,
                    headers={
                        **get_identity_headers(),
                        "Authorization": f"Bearer {token}",
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Failed to get container info: {response.status_code}")

                return cast("dict[str, Any]", response.json())

        except Exception as e:
            logger.exception("Error getting container info")
            raise RuntimeError(f"Failed to get container info: {e!s}") from e
