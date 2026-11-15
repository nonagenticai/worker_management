"""
Reference implementation for code_graph_helper agent execution endpoint.

Copy this file to code_graph_helper/api/agent_execution.py

This provides the minimal API contract needed for worker_management to execute
agents in code_graph_helper's repo containers.
"""

import asyncio
import logging
import os
import time

import aiodocker
from aiodocker.exceptions import DockerError
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["agent-execution"])


# ============================================================================
# Request/Response Models
# ============================================================================


class AgentExecutionRequest(BaseModel):
    """Request to execute an agent in a container."""

    cli: str = Field(..., description="CLI to use: 'claude' or 'codex'")
    prompt: str = Field(..., description="Task prompt for the agent")
    system_prompt: str | None = Field(default=None, description="System prompt for the agent")
    model: str | None = Field(
        default=None, description="Model to use (e.g., 'sonnet', 'opus' for Claude)"
    )
    workspace: str = Field(default="/workspace", description="Working directory in container")
    timeout: int = Field(default=300, description="Execution timeout in seconds")


class AgentExecutionResponse(BaseModel):
    """Response from agent execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int
    execution_time: float
    container_id: str


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/containers/{container_id}/execute-agent",
    response_model=AgentExecutionResponse,
    summary="Execute CLI agent in container",
    description=(
        "Execute Claude Code or Codex CLI in a specific container. "
        "This endpoint is called by worker_management to run agents "
        "in repo containers."
    ),
)
async def execute_agent(
    container_id: str,
    request: AgentExecutionRequest,
    authorization: str | None = Header(None),
) -> AgentExecutionResponse:
    """Execute CLI agent in a repo container.

    This is the main endpoint that worker_management calls to execute
    agents in containers managed by code_graph_helper.

    Args:
        container_id: Docker container ID where agent should execute
        request: Agent execution request with prompt, CLI, etc.
        authorization: Optional Bearer token for authentication

    Returns:
        AgentExecutionResponse with execution results

    Raises:
        HTTPException: If authentication fails, container not found, etc.
    """

    # 1. Authenticate request
    if authorization:
        api_key = authorization.replace("Bearer ", "")
        if not _validate_api_key(api_key):
            logger.warning(f"Invalid API key attempt for container {container_id}")
            raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info(
        f"Agent execution request: container={container_id}, "
        f"cli={request.cli}, workspace={request.workspace}"
    )

    # 2. Get Docker client and verify container
    docker = aiodocker.Docker()

    try:
        container = await docker.containers.get(container_id)
        container_info = await container.show()

        # Verify container is running
        status = container_info["State"]["Status"]
        if status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"Container {container_id} is not running (status: {status})",
            )

        # Optional: Verify container belongs to this service
        # (implement your own ownership logic here)
        # _verify_container_ownership(container_info)

    except DockerError as e:
        await docker.close()
        logger.error(f"Container {container_id} not found: {e}")
        raise HTTPException(status_code=404, detail=f"Container {container_id} not found")

    # 3. Build CLI command
    try:
        command = _build_cli_command(
            cli=request.cli,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            model=request.model,
        )
    except ValueError as e:
        await docker.close()
        raise HTTPException(status_code=400, detail=str(e))

    # 4. Execute in container
    start_time = time.time()

    try:
        result = await _execute_in_container(
            container=container,
            command=command,
            workspace=request.workspace,
            timeout=request.timeout,
        )

        execution_time = time.time() - start_time

        logger.info(
            f"Agent execution completed: container={container_id}, "
            f"exit_code={result['exit_code']}, "
            f"duration={execution_time:.2f}s"
        )

        await docker.close()

        return AgentExecutionResponse(
            success=result["exit_code"] == 0,
            output=result["output"],
            error=result.get("error"),
            exit_code=result["exit_code"],
            execution_time=execution_time,
            container_id=container_id,
        )

    except TimeoutError:
        execution_time = time.time() - start_time
        logger.error(
            f"Agent execution timed out: container={container_id}, timeout={request.timeout}s"
        )

        await docker.close()

        return AgentExecutionResponse(
            success=False,
            output="",
            error=f"Execution timed out after {request.timeout}s",
            exit_code=-1,
            execution_time=execution_time,
            container_id=container_id,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(f"Agent execution error: {e}")

        await docker.close()

        return AgentExecutionResponse(
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            execution_time=execution_time,
            container_id=container_id,
        )


@router.get(
    "/containers/{container_id}",
    summary="Get container information",
    description="Get details about a specific container",
)
async def get_container_info(container_id: str, authorization: str | None = Header(None)):
    """Get information about a container.

    Args:
        container_id: Docker container ID
        authorization: Optional Bearer token

    Returns:
        Container information dict
    """

    # Authenticate if authorization provided
    if authorization:
        api_key = authorization.replace("Bearer ", "")
        if not _validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        docker = aiodocker.Docker()
        container = await docker.containers.get(container_id)
        info = await container.show()
        await docker.close()

        return {
            "id": container_id,
            "name": info["Name"].lstrip("/"),
            "status": info["State"]["Status"],
            "workspace": "/workspace/repo",  # Adjust based on your setup
            "created_at": info["Created"],
            "image": info["Config"]["Image"],
        }

    except DockerError:
        raise HTTPException(404, f"Container {container_id} not found")


@router.get(
    "/health",
    summary="Health check",
    description="Check if service and Docker are available",
)
async def health_check():
    """Health check endpoint.

    Returns:
        Health status dict
    """
    try:
        docker = aiodocker.Docker()
        await docker.version()  # Test Docker connection
        await docker.close()
        docker_available = True
    except Exception:
        docker_available = False

    return {
        "status": "healthy" if docker_available else "degraded",
        "service": "code_graph_helper",
        "docker_available": docker_available,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_api_key(api_key: str) -> bool:
    """Validate API key against environment configuration.

    In production, use proper key management (database, vault, etc.)

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    valid_keys = os.getenv("API_KEYS", "").split(",")
    valid_keys = [k.strip() for k in valid_keys if k.strip()]

    if not valid_keys:
        logger.warning("No API keys configured (API_KEYS env var)")
        return True  # Allow if no keys configured (development mode)

    return api_key in valid_keys


def _build_cli_command(
    cli: str,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
) -> list[str]:
    """Build CLI command array for docker exec.

    Args:
        cli: CLI name ('claude' or 'codex')
        prompt: Task prompt
        system_prompt: Optional system prompt
        model: Optional model specification

    Returns:
        Command array for docker exec

    Raises:
        ValueError: If CLI is unknown
    """
    if cli == "claude":
        cmd = ["claude", "--print"]

        if model:
            cmd.extend(["--model", model])

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        cmd.append(prompt)

    elif cli == "codex":
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]

        if model:
            cmd.extend(["--model", model])

        # Codex doesn't have --system-prompt, prepend to prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        cmd.append(full_prompt)

    else:
        raise ValueError(f"Unknown CLI: {cli}. Expected 'claude' or 'codex'")

    logger.debug(f"Built CLI command: {cmd[:3]}... (prompt truncated)")
    return cmd


async def _execute_in_container(
    container, command: list[str], workspace: str, timeout: int
) -> dict:
    """Execute command in Docker container using docker exec.

    Args:
        container: aiodocker Container object
        command: Command array to execute
        workspace: Working directory
        timeout: Timeout in seconds

    Returns:
        Dict with output, exit_code, error

    Raises:
        asyncio.TimeoutError: If execution times out
    """
    # Create exec instance
    exec_config = {
        "Cmd": command,
        "AttachStdout": True,
        "AttachStderr": True,
        "WorkingDir": workspace,
    }

    exec_instance = await container.exec(exec_config)

    # Start execution
    exec_stream = exec_instance.start(detach=False)

    # Collect output with timeout
    output_chunks = []

    async def collect_output():
        """Collect output from exec stream."""
        async for chunk in exec_stream:
            output_chunks.append(chunk)

    # Run with timeout
    await asyncio.wait_for(collect_output(), timeout=timeout)

    # Decode output
    output = b"".join(output_chunks).decode("utf-8", errors="replace")

    # Get exit code
    inspect = await exec_instance.inspect()
    exit_code = inspect.get("ExitCode", 0)

    return {
        "output": output,
        "exit_code": exit_code,
        "error": None if exit_code == 0 else "CLI execution failed",
    }


# ============================================================================
# Optional: Container Ownership Verification
# ============================================================================


def _verify_container_ownership(container_info: dict):
    """Verify container belongs to this service.

    Implement your own logic here based on labels, naming, etc.

    Args:
        container_info: Container info from docker.containers.show()

    Raises:
        HTTPException: If container doesn't belong to this service
    """
    # Example: Check for specific label
    labels = container_info.get("Config", {}).get("Labels", {})
    if labels.get("managed_by") != "code_graph_helper":
        raise HTTPException(status_code=403, detail="Container not managed by this service")


# ============================================================================
# Integration Instructions
# ============================================================================

"""
TO INTEGRATE INTO code_graph_helper:

1. Copy this file to: code_graph_helper/api/agent_execution.py

2. In code_graph_helper/api/main.py:

   from .agent_execution import router as agent_router

   app.include_router(agent_router)

3. Add dependency to requirements/pyproject.toml:

   aiodocker>=0.21.0

4. Set environment variable:

   API_KEYS=secret-key-1,secret-key-2

5. Ensure repo containers have CLIs installed:

   # In Dockerfile for repo containers:
   RUN curl -fsSL https://claude.ai/install.sh | sh
   RUN npm install -g @openai/codex-cli

6. Mount auth credentials when creating containers:

   volumes:
     - ~/.claude:/home/executor/.claude:ro
     - ~/.codex:/home/executor/.codex:ro

7. Test the endpoint:

   curl -X POST http://localhost:8889/api/v1/containers/{id}/execute-agent \
     -H "Authorization: Bearer secret-key-1" \
     -d '{"cli":"claude","prompt":"Hello!"}'

8. Use from worker_management:

   agent = create_remote_claude_agent(
       executor_endpoint="http://localhost:8889/api/v1",
       api_key="secret-key-1",
       container_id="container-id-here"
   )
"""
