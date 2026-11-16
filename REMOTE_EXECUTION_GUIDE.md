# Remote Container Execution Guide

## Overview

This guide explains how to execute Claude Code and Codex agents in **remote Docker containers** managed by other services (like `code_graph_helper`), while keeping all worker logic centralized in `worker_management`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Orchestrator                                                 │
│  - Coordinates workflows across all services                 │
└───────┬─────────────────────────────────────────┬───────────┘
        │                                         │
        ↓                                         ↓
┌─────────────────────┐              ┌─────────────────────────┐
│ worker_management   │   HTTPS API  │ code_graph_helper       │
│  - Worker state     │◄────────────►│  - Repo containers      │
│  - Message routing  │              │  - Code analysis        │
│  - Execution logic  │              │  - Execution env        │
└─────────────────────┘              └──────────┬──────────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     ↓                     ↓
                              ┌─────────────┐      ┌─────────────┐
                              │ Container A │      │ Container B │
                              │ repo-abc123 │      │ repo-xyz789 │
                              │ - Cloned    │      │ - Cloned    │
                              │   repo A    │      │   repo B    │
                              │ - Claude    │      │ - Codex     │
                              │   CLI       │      │   CLI       │
                              └─────────────┘      └─────────────┘
```

## Use Cases

### Repo Work (Remote Container Execution)
```python
# For tasks that need access to specific repo containers
agent = create_remote_claude_agent(
    name="repo-analyzer",
    executor_endpoint="https://mini1.local:8889",
    container_id="repo-abc123",  # Managed by code_graph_helper
)
# Executes in remote repo container
```

## Complete Workflow

### 1. Orchestrator Workflow

```python
# Example orchestrator workflow for repo analysis


async def analyze_repository_workflow(github_url: str):
    """Analyze a GitHub repository using remote execution."""

    # Step 1: Provision repo in code_graph_helper
    repo_response = await code_graph_helper.provision_repo({"github_url": github_url})
    container_id = repo_response["container_id"]

    # Step 2: Create worker in worker_management for this repo
    worker_response = await worker_management.create_worker(
        {
            "name": f"analyzer-{container_id[:8]}",
            "agent_type": "remote_claude",
            "config": {
                "executor_endpoint": "https://mini1.local:8889",
                "api_key": os.environ["CODE_GRAPH_HELPER_API_KEY"],
                "container_id": container_id,
                "workspace": "/workspace/repo",
                "model": "sonnet",
                "system_prompt": "You are a code analysis expert.",
            },
        }
    )
    worker_id = worker_response["id"]

    # Step 3: Execute analysis tasks
    results = []
    tasks = [
        "Analyze the authentication flow",
        "Identify potential security vulnerabilities",
        "Document the API endpoints",
    ]

    for task in tasks:
        result = await worker_management.execute_worker({"worker_id": worker_id, "task": task})
        results.append(result)

    # Step 4: Cleanup (optional)
    await worker_management.delete_worker(worker_id)
    await code_graph_helper.delete_container(container_id)

    return results
```

### 2. Worker Management API

**Create Remote Worker:**
```bash
POST /workers
{
  "name": "repo-analyzer",
  "agent_type": "remote_claude",
  "config": {
    "executor_endpoint": "https://mini1.local:8889",
    "api_key": "secret-api-key",
    "container_id": "repo-abc123",
    "workspace": "/workspace/repo",
    "model": "sonnet",
    "timeout": 300
  }
}

Response:
{
  "id": "worker-uuid",
  "name": "repo-analyzer",
  "agent_type": "remote_claude",
  "status": "idle"
}
```

**Execute Task:**
```bash
POST /workers/{worker_id}/execute
{
  "task": "Analyze the authentication system in this codebase"
}

# worker_management internally calls:
# POST https://mini1.local:8889/containers/repo-abc123/execute-agent

Response:
{
  "success": true,
  "output": "Authentication analysis:\n1. Uses JWT tokens...",
  "execution_time": 12.5
}
```

### 3. Code Graph Helper API Contract

**IMPORTANT**: code_graph_helper must implement these endpoints:

#### POST /containers/{container_id}/execute-agent

Execute a CLI agent (Claude/Codex) inside a specific container.

**Request:**
```json
{
  "cli": "claude",  // or "codex"
  "prompt": "Analyze the authentication flow",
  "system_prompt": "You are a code analysis expert.",
  "model": "sonnet",  // optional
  "workspace": "/workspace/repo",
  "timeout": 300
}
```

**Response:**
```json
{
  "success": true,
  "output": "Analysis results here...",
  "error": null,
  "exit_code": 0,
  "execution_time": 12.5,
  "container_id": "repo-abc123"
}
```

**Implementation Example (FastAPI):**
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

router = APIRouter()


class AgentExecutionRequest(BaseModel):
    cli: str  # "claude" or "codex"
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    workspace: str = "/workspace"
    timeout: int = 300


class AgentExecutionResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int
    execution_time: float
    container_id: str


@router.post("/containers/{container_id}/execute-agent")
async def execute_agent(
    container_id: str, request: AgentExecutionRequest
) -> AgentExecutionResponse:
    """Execute CLI agent in a container."""

    # 1. Verify container exists
    container = await get_container(container_id)
    if not container:
        raise HTTPException(404, "Container not found")

    # 2. Build CLI command
    cmd = [request.cli]

    if request.cli == "claude":
        cmd.append("--print")
        if request.model:
            cmd.extend(["--model", request.model])
        if request.system_prompt:
            cmd.extend(["--system-prompt", request.system_prompt])
        cmd.append(request.prompt)

    elif request.cli == "codex":
        cmd.append("exec")
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if request.model:
            cmd.extend(["--model", request.model])
        # Codex: prepend system prompt to prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        cmd.append(full_prompt)

    # 3. Execute in container using docker exec
    import time

    start_time = time.time()

    try:
        result = await docker_exec(
            container_id=container_id,
            command=cmd,
            workdir=request.workspace,
            timeout=request.timeout,
        )

        execution_time = time.time() - start_time

        return AgentExecutionResponse(
            success=result.exit_code == 0,
            output=result.stdout,
            error=result.stderr if result.exit_code != 0 else None,
            exit_code=result.exit_code,
            execution_time=execution_time,
            container_id=container_id,
        )

    except asyncio.TimeoutError:
        return AgentExecutionResponse(
            success=False,
            output="",
            error=f"Execution timed out after {request.timeout}s",
            exit_code=-1,
            execution_time=request.timeout,
            container_id=container_id,
        )
    except Exception as e:
        return AgentExecutionResponse(
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            execution_time=time.time() - start_time,
            container_id=container_id,
        )


async def docker_exec(
    container_id: str, command: list[str], workdir: str, timeout: int
) -> ExecResult:
    """Execute command in Docker container."""
    import aiodocker

    docker = aiodocker.Docker()
    container = await docker.containers.get(container_id)

    # Create exec instance
    exec_instance = await container.exec(
        cmd=command,
        stdout=True,
        stderr=True,
        workdir=workdir,
    )

    # Start and wait with timeout
    result = await asyncio.wait_for(exec_instance.start(), timeout=timeout)

    # Get output
    output = result.output.decode() if hasattr(result, "output") else ""

    # Inspect to get exit code
    inspect = await exec_instance.inspect()
    exit_code = inspect.get("ExitCode", 0)

    await docker.close()

    return ExecResult(stdout=output, stderr="", exit_code=exit_code)
```

#### GET /containers/{container_id}

Get container information.

**Response:**
```json
{
  "id": "repo-abc123",
  "name": "code-graph-repo-owner-reponame",
  "status": "running",
  "workspace": "/workspace/repo",
  "github_url": "https://github.com/owner/reponame",
  "created_at": "2025-01-15T10:30:00Z"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "code_graph_helper",
  "docker_available": true
}
```

## Security

### Authentication

**API Key Authentication:**
```python
# In worker_management config
agent = create_remote_claude_agent(
    executor_endpoint="https://mini1.local:8889",
    api_key=os.environ["CODE_GRAPH_HELPER_API_KEY"],
    container_id="repo-abc123",
)


# code_graph_helper validates
@router.post("/containers/{container_id}/execute-agent")
async def execute_agent(
    container_id: str,
    request: AgentExecutionRequest,
    api_key: str = Header(None, alias="Authorization"),
):
    if not validate_api_key(api_key):
        raise HTTPException(401, "Invalid API key")
    # ...
```

**Or Mutual TLS:**
```python
# worker_management uses client certificates
agent = create_remote_claude_agent(
    executor_endpoint="https://mini1.local:8889",
    verify_ssl=True,
    # Client cert configured in httpx
)

# code_graph_helper validates client certificates
# via reverse proxy (nginx, traefik) or FastAPI
```

### Network Security

**1. VPN/Private Network:**
```
worker_management ←→ VPN ←→ code_graph_helper
```

**2. Firewall Rules:**
```bash
# On code_graph_helper machine
# Only allow connections from worker_management
iptables -A INPUT -p tcp --dport 8889 -s <worker_management_ip> -j ACCEPT
iptables -A INPUT -p tcp --dport 8889 -j DROP
```

**3. HTTPS with TLS:**
```python
# Always use HTTPS in production
executor_endpoint = "https://mini1.local:8889"  # ✅
executor_endpoint = "http://mini1.local:8889"  # ❌
```

## Deployment Scenarios

### Scenario 1: Same Machine

```
worker_management (localhost:8887)
code_graph_helper (localhost:8889)

executor_endpoint="http://localhost:8889"
```

### Scenario 2: Different Machines (Local Network)

```
worker_management (current.local:8887)
code_graph_helper (mini1.local:8889)

executor_endpoint="http://mini1.local:8889"
# Or with HTTPS:
executor_endpoint="https://mini1.local:8889"
```

### Scenario 3: Remote Machines (VPN)

```
worker_management (10.0.1.5:8887)
code_graph_helper (10.0.1.10:8889)

executor_endpoint="https://10.0.1.10:8889"
```

### Scenario 4: Cloud Deployment

```
worker_management (worker.example.com)
code_graph_helper (codegraph.example.com)

executor_endpoint="https://codegraph.example.com:8889"
```

## Configuration Management

### Environment Variables

**worker_management `.env`:**
```bash
# Code Graph Helper Configuration
CODE_GRAPH_HELPER_ENDPOINT=https://mini1.local:8889
CODE_GRAPH_HELPER_API_KEY=secret-key-here
CODE_GRAPH_HELPER_VERIFY_SSL=true
```

**code_graph_helper `.env`:**
```bash
# API Configuration
API_PORT=8889
API_KEYS=secret-key-here,another-key

# Docker Configuration
DOCKER_SOCKET=/var/run/docker.sock

# CLI Configuration (auth for CLIs)
CLAUDE_AUTH_PATH=/home/user/.claude
CODEX_AUTH_PATH=/home/user/.codex
```

### Dynamic Configuration

For relocatable deployments:

**Service Discovery:**
```python
# worker_management discovers code_graph_helper endpoint
async def get_code_graph_helper_endpoint():
    # Option 1: Environment variable
    endpoint = os.getenv("CODE_GRAPH_HELPER_ENDPOINT")

    # Option 2: Service registry (Consul, etcd)
    # endpoint = await service_registry.get("code_graph_helper")

    # Option 3: DNS
    # endpoint = f"https://{resolve_service_dns('code_graph_helper')}:8889"

    return endpoint
```

## Monitoring

### Logging

**worker_management logs:**
```
INFO: Creating RemoteClaudeAgent 'repo-analyzer' targeting container repo-abc123
INFO: Executing claude agent in remote container repo-abc123
INFO: Remote execution completed. Success: True, Exit code: 0
```

**code_graph_helper logs:**
```
INFO: Received agent execution request for container repo-abc123
INFO: Executing: ['claude', '--print', 'Analyze authentication...']
INFO: Execution completed in 12.5s, exit code: 0
```

### Metrics

Track:
- Remote execution count
- Remote execution latency
- Remote execution errors
- Container utilization

```python
# worker_management
from opentelemetry import metrics

meter = metrics.get_meter(__name__)
remote_exec_counter = meter.create_counter("remote_executions_total")
remote_exec_duration = meter.create_histogram("remote_execution_duration_seconds")

# In RemoteContainerExecutor
remote_exec_counter.add(1, {"executor": self.config.executor_endpoint})
remote_exec_duration.record(execution_time, {"success": str(result.success)})
```

## Troubleshooting

### Connection Refused

**Problem:** `Connection refused to https://mini1.local:8889`

**Solutions:**
1. Verify code_graph_helper is running: `curl https://mini1.local:8889/health`
2. Check firewall rules
3. Verify endpoint URL is correct
4. Check network connectivity: `ping mini1.local`

### Authentication Failed

**Problem:** `401 Unauthorized`

**Solutions:**
1. Verify API key is correct
2. Check API key is being sent: `Authorization: Bearer {key}`
3. Verify code_graph_helper API key validation

### Container Not Found

**Problem:** `404 Container repo-abc123 not found`

**Solutions:**
1. Verify container exists: `docker ps -a | grep repo-abc123`
2. Check container ID is correct
3. Ensure container hasn't been deleted

### Execution Timeout

**Problem:** `Execution timed out after 300s`

**Solutions:**
1. Increase timeout in worker config
2. Check if task is too complex
3. Verify container has sufficient resources
4. Check if CLI is hanging (need input, etc.)

## Best Practices

### 1. Separate Concerns

✅ **DO**: Keep worker logic in worker_management
```python
# worker_management handles routing, state, retries
agent = create_remote_claude_agent(...)
result = await agent.run(task)
```

✅ **DO**: Keep execution in code_graph_helper
```python
# code_graph_helper only executes, no business logic
result = docker_exec(container_id, command)
```

### 2. Use API Keys

✅ **DO**: Authenticate all requests
```python
api_key = os.environ["CODE_GRAPH_HELPER_API_KEY"]
```

❌ **DON'T**: Expose endpoints without auth

### 3. Validate Container Access

✅ **DO**: Verify container belongs to requesting user/tenant
```python
container = await get_container(container_id)
if container.owner_id != request.user.id:
    raise HTTPException(403, "Forbidden")
```

### 4. Use HTTPS

✅ **DO**: Use HTTPS in production
```python
executor_endpoint = "https://mini1.local:8889"
```

❌ **DON'T**: Use HTTP (except localhost development)

### 5. Implement Health Checks

✅ **DO**: Check executor health before creating workers
```python
if not await executor.check_health():
    raise RuntimeError("Executor unavailable")
```

### 6. Handle Failures Gracefully

✅ **DO**: Implement retries and fallbacks
```python
for attempt in range(3):
    try:
        result = await agent.run(task)
        break
    except RuntimeError:
        if attempt == 2:
            raise
        await asyncio.sleep(2**attempt)
```

## Summary

**Key Points:**

1. **Separation of Concerns**: worker_management = logic, code_graph_helper = execution
2. **API-Based Communication**: HTTPS API, not Docker socket exposure
3. **Security**: API keys, TLS, network isolation
4. **Flexibility**: Services can be on any machine, location-agnostic
5. **Simplicity**: Minimal changes to both services

**Agent Types:**

- `claude_code`: Claude Code in dedicated K8s Job container
- `codex`: Codex in dedicated K8s Job container
- `remote_claude`: Claude Code in existing code_graph_helper container
- `remote_codex`: Codex in existing code_graph_helper container

**Next Steps:**

1. Implement `/containers/{id}/execute-agent` endpoint in code_graph_helper
2. Test with orchestrator workflows
3. Set up authentication (API keys or mTLS)
4. Configure service discovery for dynamic endpoints
5. Monitor and tune performance
