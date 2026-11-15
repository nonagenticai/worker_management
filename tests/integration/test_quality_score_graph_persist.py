"""End-to-end integration test for the qa_validator quality_score producer.

These tests prove the full round-trip that the metering exec_quality ETL →
governor-effectiveness ``governed_avg_quality`` depends on:

1. A graph with a single ``worker`` node is created via ``POST /graphs``.
2. The graph is executed via ``POST /execution/graphs/{id}/execute``.
3. The worker node (graph_factory's inline ``WorkerNode`` — the class this
   engine actually runs) calls ``WorkerManager.send_message_to_worker``; we mock
   ONLY that call to return a structured worker output carrying a numeric
   ``quality_score`` (exactly what a real qa_validator with an object
   ``output_schema`` returns under the ``output`` field).
4. ``record_quality_score`` lifts the score into the inference bag, and
   ``PydanticGraphExecutionService._persist_session`` projects it onto
   ``graph_execution_sessions.quality_score`` (NUMERIC(5,4), migration 018).

Everything below the mocked worker call (graph factory, pydantic-graph FSM run,
inference bag, DB layer, HTTP API) is real. The DB is real Postgres and the
tests skip gracefully when ``DATABASE_URL`` is unset — same pattern as
``tests/integration/test_inference_graph_end_to_end.py``.

This guards against the regression where the producer lift lived in
``nodes/worker_node.py`` (a class the /execution engine does NOT run) and the
``WorkerManager`` dropped the worker output by reading the wrong response key.

Run locally with ``DATABASE_URL`` exported to a real Postgres (same as
``tests/integration/test_inference_graph_end_to_end.py``)::

    POSTGRES_SSLMODE=disable \
    uv run pytest tests/integration/test_quality_score_graph_persist.py \
        -v --no-cov -m pg
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.worker.graphs.pydantic_graph_execution_service as exec_service_module
from src.auth import get_tenant_id
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/integration/test_inference_graph_end_to_end.py)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def pg_pool() -> AsyncIterator[None]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


@pytest.fixture
def reset_execution_service() -> Iterator[None]:
    exec_service_module._EXECUTION_SERVICE = None
    try:
        yield
    finally:
        exec_service_module._EXECUTION_SERVICE = None


@pytest_asyncio.fixture
async def tenant_id(pg_pool) -> AsyncIterator[str]:
    """Unique tenant per test; DELETE graphs cascades to session rows."""
    cid = tenant_uuid(f"int-qs-{uuid.uuid4().hex[:12]}")
    try:
        yield cid
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM graphs WHERE tenant_id = $1", cid)


@pytest.fixture
def client(tenant_id: str, reset_execution_service) -> Iterator[TestClient]:
    from src.api.graph_execution_api import router as execution_router
    from src.api.graph_management_api import router as management_router

    app = FastAPI()
    app.dependency_overrides[get_tenant_id] = lambda: tenant_id
    app.include_router(management_router)
    app.include_router(execution_router)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worker_node(worker_name: str, task_template: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": f"qa-{uuid.uuid4().hex[:8]}",
        "node_type": "worker",
        "worker_name": worker_name,
        "task_template": task_template,
    }


def _create_graph(client: TestClient, nodes: list[dict[str, Any]]) -> str:
    resp = client.post(
        "/graphs/",
        json={
            "name": f"qs-{uuid.uuid4().hex[:12]}",
            "description": "quality_score producer e2e test",
            "nodes": nodes,
            "edges": [],
        },
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def _worker_envelope(output: Any) -> dict[str, Any]:
    """Shape WorkerManager.send_message_to_worker returns on success."""
    import json

    response_text = output if isinstance(output, str) else json.dumps(output)
    return {"response": response_text, "output": output, "metadata": {}, "usage_metrics": {}}


async def _fetch_quality_score(tenant_id: str, session_id: str) -> Any:
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT quality_score
            FROM graph_execution_sessions
            WHERE tenant_id = $1 AND session_id = $2
            """,
            tenant_id,
            session_id,
        )
    return None if row is None else row["quality_score"]


def _run(client: TestClient, graph_id: str) -> str:
    exec_resp = client.post(
        f"/execution/graphs/{graph_id}/execute",
        json={"variables": {"deliverable": "the thing to validate"}},
    )
    assert exec_resp.status_code == 200, exec_resp.text
    return str(exec_resp.json()["session_id"])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestQualityScoreProducerEndToEnd:
    async def test_numeric_score_persists(self, client: TestClient, tenant_id: str) -> None:
        """qa_validator emits quality_score 0.87 → row.quality_score == 0.8700."""
        node = _worker_node("qa_validator", "Validate: {deliverable}")
        graph_id = _create_graph(client, [node])

        envelope = _worker_envelope(
            {"validation_passed": True, "quality_score": 0.87, "decision": "pass"}
        )
        with patch(
            "src.worker.worker_manager.WorkerManager.send_message_to_worker",
            new=AsyncMock(return_value=envelope),
        ):
            session_id = _run(client, graph_id)

        score = await _fetch_quality_score(tenant_id, session_id)
        assert score is not None, "quality_score was not persisted"
        assert float(score) == pytest.approx(0.87, abs=1e-4)

    async def test_percentage_score_is_rescaled(self, client: TestClient, tenant_id: str) -> None:
        """A 0–100 scale score (87) is normalised to 0.8700 end-to-end."""
        node = _worker_node("qa_validator", "Validate: {deliverable}")
        graph_id = _create_graph(client, [node])

        envelope = _worker_envelope({"validation_passed": True, "quality_score": 87})
        with patch(
            "src.worker.worker_manager.WorkerManager.send_message_to_worker",
            new=AsyncMock(return_value=envelope),
        ):
            session_id = _run(client, graph_id)

        score = await _fetch_quality_score(tenant_id, session_id)
        assert score is not None
        assert float(score) == pytest.approx(0.87, abs=1e-4)

    async def test_string_output_leaves_score_null(
        self, client: TestClient, tenant_id: str
    ) -> None:
        """A non-validator worker returning plain text → quality_score stays NULL."""
        node = _worker_node("execution_agent", "Do: {deliverable}")
        graph_id = _create_graph(client, [node])

        envelope = _worker_envelope("just a free-text deliverable, no score")
        with patch(
            "src.worker.worker_manager.WorkerManager.send_message_to_worker",
            new=AsyncMock(return_value=envelope),
        ):
            session_id = _run(client, graph_id)

        score = await _fetch_quality_score(tenant_id, session_id)
        assert score is None
