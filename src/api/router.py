"""Main API router for the Worker Management System."""

from fastapi import APIRouter

from src.api.graph_execution_api import router as execution_router
from src.api.graph_management_api import router as graph_router
from src.api.graph_sessions_correlation_api import router as sessions_correlation_router
from src.api.prompts import router as prompts_router
from src.api.providers import chat_router, providers_router
from src.api.tenant_config import modality_router as tenant_config_modality_router
from src.api.tenant_config import router as tenant_config_router
from src.api.worker_states import router as worker_states_router

# Import all sub-routers
from src.api.workers import router as workers_router
from src.api.workflow_routing import router as workflow_routing_router

# Create main API router
api_router: APIRouter = APIRouter()

# Include all sub-routers
api_router.include_router(workers_router)
api_router.include_router(worker_states_router)
api_router.include_router(graph_router)
api_router.include_router(execution_router)
api_router.include_router(sessions_correlation_router)
api_router.include_router(providers_router)
api_router.include_router(chat_router)
api_router.include_router(tenant_config_router)
api_router.include_router(tenant_config_modality_router)
api_router.include_router(workflow_routing_router)
api_router.include_router(prompts_router)
# Note: tool_registry is a separate microservice on port 8889

# Export the main router
__all__ = ["api_router"]
