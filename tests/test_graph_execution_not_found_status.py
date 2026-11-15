"""A graph that does not exist must not be reported as a successful execution.

Issue #211. `execute_graph` raised `ValueError("Graph %s not found")` and the
broad `except (ValueError, TypeError, RuntimeError)` two hundred lines below
caught its own raise, persisted a `graph_execution_sessions` row with
`status="failed"`, and returned a normal `ExecutionResponse`. uvicorn logged

    "POST /execution/graphs/019ebfd3-.../execute HTTP/1.1" 200 OK

in the same millisecond as the raise. The routers' `except ValueError ->
NotFoundError` branch was dead code on this path.

Measured: 1,501 failures reported as HTTP 200 from 2026-08-27 onward, against a
caller (n8n workflow "AA Activity - Action", node "Exec-quality probe") running
with `onError=continueRegularOutput` -- so the AA conveyor ran an exec-quality
probe that had not executed once and never said so.

## Why this file draws the line where it does

Not every failure in that `except` is the same animal, and this suite pins both
sides of the split:

* the graph does not exist -> **404**. Nothing ran. There are no node outputs and
  the session id would point at a graph that is not in `graphs`.
* the graph exists and failed while running -> **200 + status="failed"**, kept
  deliberately. The body is the report: `node_outputs`, `completed_nodes` and
  `failed_nodes` carry the partial result a caller needs, and an `HTTPException`
  would throw all of it away. `TestExecutionFailuresStillReturn200` is the
  negative control for that, and it is the test that fails if someone "fixes"
  #211 by turning the whole `except` into an error status.

`tests/test_api_endpoints_coverage.py::test_execute_graph_not_found` already
asserted 404 and passed throughout -- because it replaces the service with a
`MagicMock` whose `execute_graph` raises. It proved the router's branch works
when reached, never that production reaches it. The endpoint tests here drive the
REAL `PydanticGraphExecutionService` and stub only the database read.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from fastapi.testclient import TestClient

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from src.worker.graphs.graph_models import ExecutionRequest
from src.worker.graphs.pydantic_graph_execution_service import (
    PydanticGraphExecutionService,
)

# The raise assertions below use `ValueError`, not the new `GraphNotFoundError`,
# ON PURPOSE. The invariant under test is "a missing graph must not come back as a
# response object", which is expressible against the code as it stands on
# `origin/main` -- so this whole module COLLECTS on unfixed code and fails with a
# real behavioural assertion instead of an ImportError at collection time. An
# ImportError proves a symbol is absent; it does not prove the endpoint answered
# 200. `test_the_error_type_is_the_typed_one` pins the concrete type separately.

GRAPH_ID = "019ebfd3-e11f-702b-9a4e-549162e931a7"  # the id from issue #211

NODE_ID = "00000000-0000-0000-0000-000000000001"


def _make_svc(execution_db=None) -> PydanticGraphExecutionService:
    """A real service with no DB pools attached."""
    with (
        patch(
            "src.worker.graphs.pydantic_graph_execution_service.GraphExecutionDatabase",
            None,
        ),
        patch(
            "src.worker.graphs.pydantic_graph_execution_service.GraphExecutionSessionDB",
            None,
        ),
    ):
        return PydanticGraphExecutionService(execution_db=execution_db)


def _valid_graph() -> dict:
    """A graph that exists and builds -- the negative control's subject."""
    return {
        "id": GRAPH_ID,
        "tenant_id": "creator-1",
        "subject_id": "creator-1",
        "visibility": "tenant",
        "name": "probe",
        "nodes": [
            {
                "id": NODE_ID,
                "name": "worker_node_1",
                "node_type": "worker",
                "worker_name": "test_worker",
                "task_template": "do stuff",
                "dependencies": [],
            }
        ],
        "edges": [],
        "variables": {},
    }


def _unbuildable_graph() -> dict:
    """A graph that EXISTS but cannot be built -- a genuine execution failure."""
    graph = _valid_graph()
    graph["id"] = "graph-bad"
    graph["nodes"] = [{"id": NODE_ID, "node_type": "worker"}]  # name missing
    return graph


# ===========================================================================
# Service level
# ===========================================================================


class TestServiceRaisesOnMissingGraph:
    @pytest.mark.asyncio
    async def test_missing_graph_raises_instead_of_returning_a_response(self):
        """RED before the fix: this returned ExecutionResponse(status="failed")."""
        svc = _make_svc()
        with (
            patch.object(
                svc, "_get_graph_from_database", new_callable=AsyncMock, return_value=None
            ),
            pytest.raises(ValueError) as exc_info,
        ):
            await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id=GRAPH_ID,
                request=ExecutionRequest(variables={}),
            )

        # The id must survive in the message: the routers turn it into the 404
        # detail.
        assert GRAPH_ID in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_session_row_is_persisted_for_a_graph_that_does_not_exist(self):
        """1,501 FAILED rows pointed at a graph_id absent from `graphs`.

        Nothing executed, so persisting a session fabricates an execution record
        and pollutes `graph_execution_sessions`.
        """
        mock_db = AsyncMock()
        mock_db.add_graph_state = AsyncMock()
        svc = _make_svc(execution_db=mock_db)

        with (
            patch.object(
                svc, "_get_graph_from_database", new_callable=AsyncMock, return_value=None
            ),
            pytest.raises(ValueError),
        ):
            await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id=GRAPH_ID,
                request=ExecutionRequest(variables={}),
            )

        mock_db.add_graph_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_stepwise_also_raises(self):
        """`execute_graph_stepwise` delegates, so it must not diverge."""
        svc = _make_svc()
        with (
            patch.object(
                svc, "_get_graph_from_database", new_callable=AsyncMock, return_value=None
            ),
            pytest.raises(ValueError),
        ):
            await svc.execute_graph_stepwise(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id=GRAPH_ID,
            )

    def test_the_error_type_is_the_typed_one(self):
        """Pin the concrete type, and that it stays a `ValueError`.

        Subclassing matters: every pre-existing `except ValueError` on this path
        -- including the routers' older substring branch, which is the fallback
        if the typed branch is ever removed -- must keep behaving as it did.
        """
        from src.worker.graphs.pydantic_graph_execution_service import GraphNotFoundError

        assert issubclass(GraphNotFoundError, ValueError)
        err = GraphNotFoundError(GRAPH_ID)
        assert err.graph_id == GRAPH_ID
        assert "not found" in str(err).lower()


# ===========================================================================
# Negative control -- the classes that must KEEP returning 200
# ===========================================================================


class TestExecutionFailuresStillReturn200:
    """A graph that exists and fails while running is an execution OUTCOME.

    If these flip to an error status, the fix has over-reached: the caller loses
    the partial result body and the session id it needs to investigate.
    """

    @pytest.mark.asyncio
    async def test_unbuildable_graph_still_returns_a_failed_response(self):
        svc = _make_svc()
        with patch.object(
            svc,
            "_get_graph_from_database",
            new_callable=AsyncMock,
            return_value=_unbuildable_graph(),
        ):
            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-bad",
                request=ExecutionRequest(variables={}),
            )

        assert response.status == "failed"
        assert response.error_message is not None
        assert response.session_id

    @pytest.mark.asyncio
    async def test_a_failed_execution_still_persists_its_session(self):
        mock_db = AsyncMock()
        mock_db.add_graph_state = AsyncMock()
        svc = _make_svc(execution_db=mock_db)

        with patch.object(
            svc,
            "_get_graph_from_database",
            new_callable=AsyncMock,
            return_value=_unbuildable_graph(),
        ):
            await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-bad",
                request=ExecutionRequest(variables={}),
            )

        mock_db.add_graph_state.assert_called_once()


# ===========================================================================
# Endpoint level -- the REAL service behind the route
# ===========================================================================


@pytest.fixture
def client_with_real_service():
    """TestClient whose route is backed by a real service, DB read stubbed.

    Yields (client, set_graph). `set_graph(None)` makes the lookup miss;
    `set_graph(dict)` makes it hit. Only `_get_graph_from_database` is replaced,
    so the swallow/re-raise decision under test runs for real.
    """
    svc = _make_svc()
    lookup = AsyncMock(return_value=None)

    def set_graph(graph_data):
        lookup.return_value = graph_data

    with (
        patch.object(svc, "_get_graph_from_database", lookup),
        patch("src.api.graph_execution_api.get_execution_service", return_value=svc),
        patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
    ):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = AsyncMock(
            return_value={"response": "done", "status": "success"}
        )
        MockWM.return_value = mock_wm

        app.dependency_overrides[get_tenant_id] = lambda: "creator-1"

        @asynccontextmanager
        async def _fake_lifespan(_a):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _fake_lifespan
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, set_graph
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original


class TestEndpointStatusForMissingGraph:
    def test_execute_returns_404_not_200(self, client_with_real_service):
        """RED before the fix: 200 with a body saying status="failed"."""
        client, set_graph = client_with_real_service
        set_graph(None)

        resp = client.post(f"/execution/graphs/{GRAPH_ID}/execute", json={"variables": {}})

        # The bug in one assertion: a failure must not be reported as a success.
        assert resp.status_code != 200, resp.text
        assert resp.status_code == 404, resp.text

    def test_the_404_body_is_not_an_execution_response(self, client_with_real_service):
        """A 200 body carried session_id + status="failed"; a 404 must carry neither.

        This is the assertion that a caller keying on the BODY (rather than the
        status) would have needed: `onError=continueRegularOutput` sees only the
        body, and `status="failed"` was buried in a shape that otherwise looks
        exactly like a successful run.
        """
        client, set_graph = client_with_real_service
        set_graph(None)

        resp = client.post(f"/execution/graphs/{GRAPH_ID}/execute", json={"variables": {}})
        body = resp.json()

        assert "session_id" not in body
        assert body["error"]["status"] == 404
        assert body["error"]["code"] == "NOT_FOUND"
        assert GRAPH_ID in body["error"]["message"]

    def test_stepwise_endpoint_returns_404_too(self, client_with_real_service):
        client, set_graph = client_with_real_service
        set_graph(None)

        resp = client.post(f"/execution/graphs/{GRAPH_ID}/execute/stepwise", json={"variables": {}})

        assert resp.status_code == 404, resp.text

    def test_a_graph_that_exists_still_executes_over_200(self, client_with_real_service):
        """The negative control at the endpoint: the happy path is untouched."""
        client, set_graph = client_with_real_service
        set_graph(_valid_graph())

        resp = client.post(f"/execution/graphs/{GRAPH_ID}/execute", json={"variables": {}})

        assert resp.status_code == 200, resp.text
        assert resp.json()["session_id"]

    def test_the_404_does_not_depend_on_the_words_in_the_message(self):
        """Why the routers catch the TYPE and not the substring "not found".

        The pre-existing branch is `if "not found" in str(e).lower()`. Reword the
        exception -- "graph unavailable", "no such graph", a translation -- and it
        silently falls through to `BadRequestError`, turning a 404 into a 400 with
        no test noticing. The typed branch is what makes that impossible, and this
        is the test that fails if someone deletes it as redundant.
        """
        from src.worker.graphs.pydantic_graph_execution_service import GraphNotFoundError

        err = GraphNotFoundError(GRAPH_ID)
        err.args = ("graph unavailable",)  # a future rewording
        assert "not found" not in str(err).lower()

        svc = _make_svc()
        with (
            patch.object(svc, "_get_graph_from_database", new_callable=AsyncMock, side_effect=err),
            patch("src.api.graph_execution_api.get_execution_service", return_value=svc),
        ):
            app.dependency_overrides[get_tenant_id] = lambda: "creator-1"

            @asynccontextmanager
            async def _fake_lifespan(_a):
                yield

            original = app.router.lifespan_context
            app.router.lifespan_context = _fake_lifespan
            try:
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.post(f"/execution/graphs/{GRAPH_ID}/execute", json={"variables": {}})
            finally:
                app.dependency_overrides.clear()
                app.router.lifespan_context = original

        assert resp.status_code == 404, resp.text
