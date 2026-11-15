"""Install on a test app EXACTLY the exception handlers production installs.

Production serves ``src.api.worker_management_api:app`` (Dockerfile CMD), which
registers ``http_exception_handler`` for ``HTTPException`` (FastAPI and
Starlette), ``validation_exception_handler``, ``api_error_handler`` and
``general_exception_handler``. ``http_exception_handler`` re-wraps every
``HTTPException`` as ``{"error": {"code", "message", "status"}}`` — so a bare
``FastAPI()`` test app, which renders ``{"detail": ...}``, pins a body no client
ever receives.

There is no registration function to call, so the handlers are copied off the
production app object itself: nothing to keep in sync by hand, and a handler
added to (or removed from) production reaches these tests automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI


def production_exception_handlers() -> dict[Any, Any]:
    from src.api.worker_management_api import app as production_app

    return dict(production_app.exception_handlers)


def install_production_exception_handlers(app: FastAPI) -> FastAPI:
    for exc_class_or_status, handler in production_exception_handlers().items():
        app.add_exception_handler(exc_class_or_status, handler)
    return app


def wire_error(status: int, message: Any, code: str) -> dict[str, Any]:
    """The body ``http_exception_handler`` renders for ``HTTPException(status, message)``
    (no ``X-Request-ID`` sent, no ``error_details``/``raw_body`` on the exception)."""
    return {"error": {"code": code, "message": message, "status": status}}
