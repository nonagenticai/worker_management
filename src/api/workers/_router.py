"""Central router for the Workers API package.

All endpoint modules import ``router`` from this file and attach their
handlers with ``@router.get`` / ``@router.post`` / etc.

The import order below matters because FastAPI matches routes in
registration order: static segments (``/mlx/models``, ``/states``,
``/types``) must be registered **before** parameterised segments
(``/{worker_name}``).
"""

from fastapi import APIRouter

# Create the single workers router shared by all endpoint modules.
router = APIRouter(prefix="/workers", tags=["Workers"])


def _register_endpoints() -> None:
    """Import every endpoint module so their decorators execute.

    This is called once from ``__init__.py`` after the router object
    already exists.  Splitting this into a helper keeps the module-level
    namespace clean and avoids circular-import issues.
    """
    # Order matters: static paths first, then parameterised paths.
    import src.api.workers.state_endpoints  # noqa: I001, RUF100
    import src.api.workers.convenience_endpoints
    import src.api.workers.crud_endpoints
    import src.api.workers.message_endpoints  # noqa: F401, RUF100
