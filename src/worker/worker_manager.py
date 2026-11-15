"""WorkerManager - Bridge between graph execution and worker API system."""

import json
import logging
import warnings
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manager class for sending messages to workers from graph nodes."""

    def __init__(self, base_url: str = "http://localhost:8887") -> None:
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=180.0)
        return self.client

    async def send_message_to_worker(
        self,
        worker_name: str,
        message: str,
        context: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to a worker via the worker API.

        Args:
            worker_name: Name of the worker to send message to
            message: Message content
            context: Optional context data
            tenant_id: The run's tenant. When set, forwarded as a **raw** (non-Bearer)
                ``X-Tenant-Id`` header so the receiving ``/workers/{name}/message``
                endpoint's auth middleware takes its trusted-network branch and
                re-resolves the tenant's BYO posture from licenses — enforcing
                ``byo_required`` on graph-executed LLM stages instead of falling back
                to ``DEFAULT_TENANT_ID`` (``local_allowed``) and running on platform
                GPU. Absent (None) ⇒ no header ⇒ byte-identical to the prior behavior.
            subject_id: The originating end-user (for per-user cost attribution).
                Forwarded as ``X-Subject-Id`` when present. Note: the receiver's
                trusted-network auth branch currently sets ``subject_id=None`` and
                does not yet honor this header — forwarded for forward-compat only.

        Returns:
            Dict containing the response from the worker
        """
        logger.info("Sending message to worker %s via API", worker_name)

        try:
            client = await self._get_client()

            # Prepare request payload
            payload = {"message": message, "context": context or {}}

            # Forward run identity on the loopback so BYO posture is enforced (see
            # tenant_id/subject_id docstring). Raw tenant_id (NOT "Bearer <token>")
            # keeps the receiver on the tokenless trusted-network path.
            headers: dict[str, str] = {}
            if tenant_id:
                headers["X-Tenant-Id"] = tenant_id
                if subject_id:
                    headers["X-Subject-Id"] = subject_id

            # Send request to worker API
            url = f"{self.base_url}/workers/{worker_name}/message"
            response = await client.post(url, json=payload, headers=headers or None)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response from worker {worker_name}: {e!s}"
                    logger.exception(error_msg)
                    return {
                        "response": f"Error: {error_msg}",
                        "error": True,
                        "exception": str(e),
                        "context": context,
                    }

                logger.info("Successfully received response from worker %s", worker_name)
                # The /workers/{name}/message endpoint returns the agent output
                # under "output" (MessageResponse), NOT "response". Carry a text
                # form under "response" (for routing/End in the graph WorkerNode)
                # AND the raw structured value under "output" (so the WorkerNode
                # can lift a numeric quality_score into the inference bag → it is
                # persisted onto graph_execution_sessions.quality_score). A bare
                # "response" key is tolerated as a legacy fallback.
                agent_output = response_data.get("output")
                if agent_output is None:
                    response_text = response_data.get("response", "")
                elif isinstance(agent_output, str):
                    response_text = agent_output
                else:
                    response_text = json.dumps(agent_output, default=str)
                return {
                    "response": response_text,
                    "output": agent_output,
                    "output_type": response_data.get("output_type"),
                    "metadata": response_data.get("metadata", {}),
                    "usage_metrics": response_data.get("usage_metrics", {}),
                    "context": context,
                }

            error_msg = f"Worker API returned status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "error": True,
                "status_code": response.status_code,
                "context": context,
            }

        except httpx.RequestError as e:
            error_msg = f"Request error when contacting worker {worker_name}: {e!s}"
            logger.exception(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "error": True,
                "exception": str(e),
                "context": context,
            }
        except (ValueError, KeyError) as e:
            error_msg = f"Data processing error when contacting worker {worker_name}: {e!s}"
            logger.exception(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "error": True,
                "exception": str(e),
                "context": context,
            }
        except Exception as e:
            error_msg = f"Unexpected error when contacting worker {worker_name}: {e!s}"
            logger.exception(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "error": True,
                "exception": str(e),
                "context": context,
            }

    async def cleanup(self) -> None:
        """Cleanup HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup even if context manager not used."""
        if hasattr(self, "client") and self.client:
            warnings.warn(
                "WorkerManager client not properly closed. Use async context manager or call cleanup().",
                ResourceWarning,
                stacklevel=2,
            )
            # Don't attempt async cleanup in destructor as it's unreliable
            # Just log the warning to encourage proper usage

    async def __aenter__(self) -> WorkerManager:
        """Enter the async context, returning the manager itself."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit the async context, running cleanup on managed workers."""
        await self.cleanup()
