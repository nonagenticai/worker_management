"""Workers API package – backward-compatible re-exports.

All public symbols that were previously available from the monolithic
``src.api.workers`` module are re-exported here so that existing imports
such as ``from src.api.workers import router`` or
``from src.api.workers import worker_cache`` continue to work.
"""

from src.api.workers._globals import (
    WorkerInstanceCache,
    state_manager,
    worker_cache,
    worker_registry,
)
from src.api.workers._helpers import _is_connection_error
from src.api.workers._router import _register_endpoints, router
from src.api.workers.models import (
    AgentStateCreate,
    AgentStateResponse,
    MessageRequest,
    MessageResponse,
    WorkerCreateRequest,
    WorkerResponse,
    WorkerStateBase,
)
from src.db.workers_state_db import StateDatabase  # noqa: F401, RUF100

# Re-export service singletons that were available on the old monolithic module.
# Tests patch methods on these objects via ``src.api.workers.<singleton>.<method>``.
from src.services.inference_router_registry import (  # noqa: F401, RUF100
    inference_router_registry,
)
from src.services.mlx_model_registry import mlx_model_registry  # noqa: F401, RUF100

# Wire up all endpoint modules (order-sensitive – see _router.py).
_register_endpoints()

# Re-export endpoint functions for tests that import them directly.
from src.api.workers.crud_endpoints import create_worker_endpoint
from src.api.workers.message_endpoints import send_message

__all__ = [
    "AgentStateCreate",
    "AgentStateResponse",
    "MessageRequest",
    "MessageResponse",
    # Models
    "WorkerCreateRequest",
    # Cache / globals
    "WorkerInstanceCache",
    "WorkerResponse",
    "WorkerStateBase",
    # Helpers
    "_is_connection_error",
    "create_worker_endpoint",
    # Router
    "router",
    # Endpoint functions (for direct test imports)
    "send_message",
    "state_manager",
    "worker_cache",
    "worker_registry",
]
