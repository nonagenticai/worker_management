"""Unit tests for the inference-telemetry projection performed by
``PydanticGraphExecutionService._persist_session``.

These tests exercise the projection logic deterministically, without touching
a real database. They install a recording fake on the service's ``session_db``
attribute and verify that ``upsert_finalized_session`` is called with the
correct kwargs for each kind of inference outcome:

1. Provider A wins on first attempt (fallback_count == 0).
2. Provider A fails then B wins (fallback_count == 1).
3. All providers fail (chosen_provider is None, tokens are None).
4. Non-inference graph (no ``inference`` key in framework_state).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from src.db.graph_execution_db import (
    ExecutionOutputs,
    InferenceTelemetry,
    ScopeContext,
)
from src.worker.graphs.pydantic_graph_execution_service import (
    PydanticGraphExecutionService,
)


class _RecordingSessionDB:
    """Captures arguments to ``upsert_finalized_session`` for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def upsert_finalized_session(
        self,
        scope: ScopeContext,
        graph_id: str,
        status: str,
        *,
        outputs: ExecutionOutputs | None = None,
        telemetry: InferenceTelemetry | None = None,
        error_message: str | None = None,
        request_payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        out = outputs if outputs is not None else ExecutionOutputs()
        tel = telemetry if telemetry is not None else InferenceTelemetry()
        self.calls.append(
            {
                "scope": scope,
                "graph_id": graph_id,
                "status": status,
                "outputs": out,
                "telemetry": tel,
                "error_message": error_message,
                "request_payload": request_payload,
                "correlation_id": correlation_id,
            }
        )
        # Mirror the real return shape; tests only assert on ``calls``.
        return {"session_id": scope.session_id}


def _make_service() -> tuple[PydanticGraphExecutionService, _RecordingSessionDB]:
    """Build a service with ``execution_db`` patched out and a recording session_db."""
    with patch(
        "src.worker.graphs.pydantic_graph_execution_service.GraphExecutionDatabase",
        None,
    ):
        svc = PydanticGraphExecutionService(execution_db=None)
    rec = _RecordingSessionDB()
    svc.session_db = rec
    return svc, rec
