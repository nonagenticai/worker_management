# code_graph_helper Integration Guide

## What code_graph_helper Needs to Implement

To enable remote agent execution, code_graph_helper needs to:

1. **Execute agents in repo containers** (docker exec)
2. **Expose API endpoint** for worker_management to call
3. **Ensure CLIs are installed** in repo containers
4. **Mount auth credentials** into repo containers

## Minimal Implementation

### 1. Add Agent Execution Endpoint

**File: `code_graph_helper/api/agent_execution.py`**

```python
"""Agent execution endpoint for code_graph_helper.

This endpoint allows worker_management to execute Claude/Codex agents
in repo containers managed by code_graph_helper.
"""

import asyncio
import logging
import time
from typing import Optional

import aiodocker
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentExecutionRequest(BaseModel):
    """Request to execute an agent in a container."""

    cli: str = Field(..., description="CLI to use: 'claude' or 'codex'")
    prompt: str = Field(..., description="Task prompt")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    model: Optional[str] = Field(None, description="Model to use")
    workspace: str = Field(default="/workspace", description="Working directory")
    timeout: int = Field(default=300, description="Timeout in seconds")


class AgentExecutionResponse(BaseModel):
    """Response from agent execution."""

    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int
    execution_time: float
    container_id: str


@router.post("/containers/{container_id}/execute-agent", response_model=AgentExecutionResponse)
async def execute_agent(
    container_id: str, request: AgentExecutionRequest, authorization: Optional[str] = Header(None)
) -> AgentExecutionResponse:
    """Execute CLI agent in a repo container.

    This endpoint receives agent execution requests from worker_management
    and runs the CLI in the specified container using docker exec.
    """

    # 1. Authenticate request
    if authorization:
        api_key = authorization.replace("Bearer ", "")
        if not validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info(f"Agent execution request for container {container_id}")
    logger.info(f"CLI: {request.cli}, Workspace: {request.workspace}")

    # 2. Verify container exists and is running
    try:
        docker = aiodocker.Docker()
        container = await docker.containers.get(container_id)
        container_info = await container.show()

        if container_info["State"]["Status"] != "running":
            raise HTTPException(status_code=400, detail=f"Container {container_id} is not running")

    except aiodocker.exceptions.DockerError as e:
        await docker.close()
        raise HTTPException(status_code=404, detail=f"Container {container_id} not found")

    # 3. Build CLI command
    command = build_cli_command(
        cli=request.cli,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        model=request.model,
    )

    # 4. Execute in container
    start_time = time.time()

    try:
        # Create exec instance
        exec_config = {
            "Cmd": command,
            "AttachStdout": True,
            "AttachStderr": True,
            "WorkingDir": request.workspace,
        }

        exec_instance = await container.exec(exec_config)

        # Start execution with timeout
        exec_stream = exec_instance.start(detach=False)

        output_chunks = []
        async for chunk in exec_stream:
            output_chunks.append(chunk)

        # Wait with timeout
        await asyncio.wait_for(
            asyncio.sleep(0.1),  # Small delay for exec to finish
            timeout=request.timeout,
        )

        # Get output
        output = b"".join(output_chunks).decode("utf-8", errors="replace")

        # Get exit code
        inspect = await exec_instance.inspect()
        exit_code = inspect.get("ExitCode", 0)

        execution_time = time.time() - start_time

        logger.info(f"Agent execution completed in {execution_time:.2f}s, exit_code: {exit_code}")

        await docker.close()

        return AgentExecutionResponse(
            success=exit_code == 0,
            output=output,
            error=None if exit_code == 0 else "CLI execution failed",
            exit_code=exit_code,
            execution_time=execution_time,
            container_id=container_id,
        )

    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        logger.error(f"Agent execution timed out after {request.timeout}s")

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


def build_cli_command(
    cli: str, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None
) -> list[str]:
    """Build CLI command array for docker exec.

    Args:
        cli: CLI name ('claude' or 'codex')
        prompt: Task prompt
        system_prompt: Optional system prompt
        model: Optional model specification

    Returns:
        Command array for docker exec
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

        # Codex: prepend system prompt to prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        cmd.append(full_prompt)

    else:
        raise ValueError(f"Unknown CLI: {cli}")

    return cmd


def validate_api_key(api_key: str) -> bool:
    """Validate API key.

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    import os

    # Simple validation against environment variable
    # In production, use proper key management
    valid_keys = os.getenv("API_KEYS", "").split(",")
    return api_key in valid_keys


@router.get("/containers/{container_id}")
async def get_container_info(container_id: str):
    """Get information about a container."""
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
        }

    except aiodocker.exceptions.DockerError:
        raise HTTPException(404, f"Container {container_id} not found")
```

### 2. Update Repo Container Creation

Ensure repo containers have CLIs installed:

**File: `code_graph_helper/docker/repo-container.Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | sh

# Install Codex CLI
RUN npm install -g @openai/codex-cli

# Create workspace
RUN mkdir -p /workspace/repo && chmod 755 /workspace/repo

# Create user
RUN useradd -m -u 1000 executor
RUN chown -R executor:executor /workspace

USER executor
WORKDIR /workspace/repo

# Verify CLIs are installed
RUN claude --version && codex --version

CMD ["/bin/bash"]
```

### 3. Mount Auth Credentials

**File: `code_graph_helper/docker-compose.yml` or container creation code**

```yaml
services:
  repo-container:
    build: ./docker/repo-container.Dockerfile
    volumes:
      # Repo code
      - ./repos/${REPO_ID}:/workspace/repo:rw
      # CLI auth (read-only)
      - ~/.claude:/home/executor/.claude:ro
      - ~/.codex:/home/executor/.codex:ro
```

Or in Python container creation:

```python
# When provisioning repo container
container = await docker.containers.create(
    config={
        "Image": "repo-container:latest",
        "HostConfig": {
            "Binds": [
                f"{repo_path}:/workspace/repo:rw",
                f"{Path.home()}/.claude:/home/executor/.claude:ro",
                f"{Path.home()}/.codex:/home/executor/.codex:ro",
            ]
        },
    }
)
```

### 4. Register Endpoint in Main App

**File: `code_graph_helper/api/main.py`**

```python
from fastapi import FastAPI
from .agent_execution import router as agent_router

app = FastAPI()

# Register agent execution endpoint
app.include_router(agent_router, prefix="/api/v1", tags=["agent-execution"])

# Add to existing endpoints
# ...
```

## Testing

### 1. Test Endpoint Directly

```bash
# Test from command line
curl -X POST https://mini1.local:8889/containers/repo-abc123/execute-agent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret-key" \
  -d '{
    "cli": "claude",
    "prompt": "List files in the current directory",
    "workspace": "/workspace/repo",
    "timeout": 60
  }'

# Expected response:
{
  "success": true,
  "output": "Here are the files in the current directory:\n...",
  "error": null,
  "exit_code": 0,
  "execution_time": 3.2,
  "container_id": "repo-abc123"
}
```

### 2. Test from worker_management

```python
# In Python
from worker.agents.remote_claude_agent import create_remote_claude_agent

agent = create_remote_claude_agent(
    name="test-agent",
    description="Test remote execution",
    db_path="./test.db",
    executor_endpoint="https://mini1.local:8889",
    api_key="secret-key",
    container_id="repo-abc123",
    workspace="/workspace/repo",
)

result = await agent.run("What files are in this repository?")
print(result.output)
```

### 3. Test Health Check

```bash
curl https://mini1.local:8889/health

# Expected response:
{
  "status": "healthy",
  "service": "code_graph_helper",
  "docker_available": true
}
```

## Migration Path

### Phase 1: Add Endpoint (code_graph_helper)
- Implement `/containers/{id}/execute-agent` endpoint
- Test with curl/Postman

### Phase 2: Update Container Creation (code_graph_helper)
- Install CLIs in repo containers
- Mount auth credentials

### Phase 3: Create Remote Workers (worker_management)
- Already implemented! Just use `remote_claude` agent type

### Phase 4: Update Orchestrator
- Use remote workers for repo tasks
- Use local sandboxed workers for non-repo tasks

## Example: Complete Flow

```python
# orchestrator/workflows/analyze_repo.py


async def analyze_github_repo(github_url: str):
    """Complete workflow for analyzing a GitHub repo."""

    # 1. Provision repo container (code_graph_helper)
    repo = await code_graph_helper_client.post(
        "/repos/provision",
        json={
            "github_url": github_url,
            "include_cli": True,  # Ensure CLIs are installed
        },
    )
    container_id = repo["container_id"]

    try:
        # 2. Create remote worker (worker_management)
        worker = await worker_management_client.post(
            "/workers",
            json={
                "name": f"analyzer-{container_id[:8]}",
                "agent_type": "remote_claude",
                "config": {
                    "executor_endpoint": os.getenv("CODE_GRAPH_HELPER_ENDPOINT"),
                    "api_key": os.getenv("CODE_GRAPH_HELPER_API_KEY"),
                    "container_id": container_id,
                    "workspace": "/workspace/repo",
                    "model": "sonnet",
                    "system_prompt": "You are an expert code analyst.",
                },
            },
        )
        worker_id = worker["id"]

        # 3. Execute analysis tasks
        tasks = [
            "Analyze the overall architecture",
            "Identify security vulnerabilities",
            "Document the API surface",
        ]

        results = []
        for task in tasks:
            result = await worker_management_client.post(
                f"/workers/{worker_id}/execute", json={"task": task}
            )
            results.append(result)

        return {"repo": github_url, "analysis": results}

    finally:
        # 4. Cleanup
        await worker_management_client.delete(f"/workers/{worker_id}")
        await code_graph_helper_client.delete(f"/containers/{container_id}")
```

## Security Checklist

- [ ] API key authentication implemented
- [ ] HTTPS configured (not HTTP)
- [ ] Container ownership validation (prevent cross-tenant access)
- [ ] Rate limiting on execution endpoint
- [ ] Auth credentials mounted read-only
- [ ] Network policies (firewall rules)
- [ ] Audit logging for all executions
- [ ] Resource limits on containers

## Reference Implementation

See `examples/code_graph_helper_reference.py` for a complete reference implementation you can copy to code_graph_helper.
