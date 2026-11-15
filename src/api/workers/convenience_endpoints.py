"""Type-specific convenience endpoints for worker creation."""

import logging
from typing import Annotated

from fastapi import Depends

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.workers._router import router
from src.api.workers.crud_endpoints import create_worker_endpoint
from src.api.workers.models import WorkerCreateRequest, WorkerResponse
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_WORKERS_CREATE_CLAUDE_CODE,
    ROUTE_WORKERS_CREATE_CODEX,
    ROUTE_WORKERS_CREATE_PYDANTIC_AI,
    ROUTE_WORKERS_CREATE_REMOTE_CLAUDE,
    ROUTE_WORKERS_CREATE_REMOTE_CODEX,
    tokenless_tenant_gate,
)

logger = logging.getLogger(__name__)


@router.post(
    "/pydantic-ai",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_pydantic_ai_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE_PYDANTIC_AI))],
)
async def create_pydantic_ai_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a PydanticAI agent worker. Convenience endpoint that sets agent_type='pydantic_ai'."""
    request.agent_type = "pydantic_ai"
    return await create_worker_endpoint(ctx, request)


@router.post(
    "/claude-code",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_claude_code_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE_CLAUDE_CODE))],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "worker_name": "code-reviewer",
                        "system_prompt": "You are a code reviewer.",
                        "model": "sonnet",
                    }
                }
            }
        }
    },
)
async def create_claude_code_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a Claude Code agent worker that runs in a dedicated K8s Job container."""
    request.agent_type = "claude_code"
    if not request.agent_role:
        request.agent_role = f"Claude Code agent for {request.worker_name}"
    return await create_worker_endpoint(ctx, request)


@router.post(
    "/codex",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_codex_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE_CODEX))],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "worker_name": "code-writer",
                        "system_prompt": "You are a code writer.",
                        "model": "gpt-5-codex",
                    }
                }
            }
        }
    },
)
async def create_codex_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a Codex agent worker that runs in a dedicated K8s Job container."""
    request.agent_type = "codex"
    if not request.agent_role:
        request.agent_role = f"Codex agent for {request.worker_name}"
    return await create_worker_endpoint(ctx, request)


@router.post(
    "/remote-claude",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_remote_claude_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE_REMOTE_CLAUDE))],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "worker_name": "repo-analyzer",
                        "container_id": "abc123",
                        "system_prompt": "You are a code analyst.",
                    }
                }
            }
        }
    },
)
async def create_remote_claude_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a remote Claude agent worker that executes in a code_graph_helper container."""
    request.agent_type = "remote_claude"
    if not request.agent_role:
        request.agent_role = f"Remote Claude agent for {request.worker_name}"
    return await create_worker_endpoint(ctx, request)


@router.post(
    "/remote-codex",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_remote_codex_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE_REMOTE_CODEX))],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "worker_name": "repo-coder",
                        "container_id": "abc123",
                        "system_prompt": "You are a code writer.",
                    }
                }
            }
        }
    },
)
async def create_remote_codex_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a remote Codex agent worker that executes in a code_graph_helper container."""
    request.agent_type = "remote_codex"
    if not request.agent_role:
        request.agent_role = f"Remote Codex agent for {request.worker_name}"
    return await create_worker_endpoint(ctx, request)
