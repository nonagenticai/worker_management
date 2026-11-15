"""Tests for PydanticGraphExecutionService.

Tests cover:
1. Cache behavior (key generation, TTL expiry, LRU eviction)
2. Real graph execution (single node, two-node chain, worker failure, graph not found)
3. Task template interpolation
4. DB persistence via _persist_session
5. Session management (status, clear, stats)
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.worker.graphs.graph_models import ExecutionRequest, NodeSchema
from src.worker.graphs.pydantic_graph_execution_service import (
    GraphNotFoundError,
    PydanticGraphExecutionService,
)

# ---------------------------------------------------------------------------
# Helpers — graph data dicts as returned by GraphStateDatabase.get_graph
# ---------------------------------------------------------------------------

NODE_ID_1 = "00000000-0000-0000-0000-000000000001"
NODE_ID_2 = "00000000-0000-0000-0000-000000000002"


def _single_node_graph(
    node_id: str = NODE_ID_1,
    worker_name: str = "test_worker",
    task_template: str = "do stuff",
) -> dict:
    return {
        "id": "graph-1",
        "tenant_id": "creator-1",
        "subject_id": "creator-1",
        "visibility": "tenant",
        "name": "test_graph",
        "nodes": [
            {
                "id": node_id,
                "name": "worker_node_1",
                "node_type": "worker",
                "worker_name": worker_name,
                "task_template": task_template,
                "dependencies": [],
            }
        ],
        "edges": [],
        "variables": {},
    }


def _two_node_chain() -> dict:
    """Two worker nodes: A -> B."""
    return {
        "id": "graph-chain",
        "tenant_id": "creator-1",
        "subject_id": "creator-1",
        "visibility": "tenant",
        "name": "chain_graph",
        "nodes": [
            {
                "id": NODE_ID_1,
                "name": "node_A",
                "node_type": "worker",
                "worker_name": "worker_a",
                "task_template": "task A",
                "dependencies": [],
            },
            {
                "id": NODE_ID_2,
                "name": "node_B",
                "node_type": "worker",
                "worker_name": "worker_b",
                "task_template": "task B",
                "dependencies": [],
            },
        ],
        "edges": [
            {"source": NODE_ID_1, "target": NODE_ID_2},
        ],
        "variables": {},
    }


def _template_graph() -> dict:
    """Single node with a {{variable}} in its task template."""
    return {
        "id": "graph-tpl",
        "tenant_id": "creator-1",
        "subject_id": "creator-1",
        "visibility": "tenant",
        "name": "template_graph",
        "nodes": [
            {
                "id": NODE_ID_1,
                "name": "templated_node",
                "node_type": "worker",
                "worker_name": "tpl_worker",
                "task_template": "Process {input_file}",
                "dependencies": [],
            }
        ],
        "edges": [],
        "variables": {},
    }


def _unbuildable_graph() -> dict:
    """A graph that EXISTS but fails to build (node is missing its name).

    The counterpart to a missing graph: a real execution failure, which keeps
    returning ``ExecutionResponse(status="failed")`` over HTTP 200 because the
    body is the report the caller needs.
    """
    return {
        "id": "graph-bad",
        "tenant_id": "creator-1",
        "subject_id": "creator-1",
        "visibility": "tenant",
        "name": "bad",
        "nodes": [{"id": NODE_ID_1, "node_type": "worker"}],  # missing name
        "edges": [],
        "variables": {},
    }


def _make_svc(execution_db=None) -> PydanticGraphExecutionService:
    """Create a service with no auto-created execution_db and no session_db.

    The finalize projection added in Task 5 instantiates a
    ``GraphExecutionSessionDB``; patching it to ``None`` keeps these
    service-level tests from hitting a real PostgreSQL pool.
    """
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


async def _run_graph(svc, graph_data, worker_side_effect=None, variables=None):
    """Helper: patch DB + WorkerManager, execute graph, return response."""
    if worker_side_effect is None:
        worker_response = {"response": "done", "status": "success"}
        worker_side_effect = AsyncMock(return_value=worker_response)
    elif isinstance(worker_side_effect, Exception):
        exc = worker_side_effect
        worker_side_effect = AsyncMock(side_effect=exc)
    elif isinstance(worker_side_effect, dict):
        resp = worker_side_effect
        worker_side_effect = AsyncMock(return_value=resp)

    with (
        patch.object(
            svc,
            "_get_graph_from_database",
            new_callable=AsyncMock,
            return_value=graph_data,
        ),
        patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
    ):
        mock_wm = MagicMock()
        mock_wm.send_message_to_worker = worker_side_effect
        MockWM.return_value = mock_wm
        response = await svc.execute_graph(
            tenant_id="creator-1",
            subject_id="creator-1",
            graph_id=graph_data["id"] if graph_data else "missing",
            request=ExecutionRequest(variables=variables or {}),
        )
    return response, mock_wm


# ===================================================================
# 1. Cache behavior
# ===================================================================


class TestCacheKeyGeneration:
    """Cache key determinism and uniqueness."""

    def test_same_input_produces_same_key(self):
        svc = _make_svc()
        node = {"id": NODE_ID_1, "name": "n1", "node_type": "worker"}
        assert svc._get_cache_key([node], []) == svc._get_cache_key([node], [])

    def test_different_nodes_produce_different_keys(self):
        svc = _make_svc()
        a = {"id": NODE_ID_1, "name": "alpha", "node_type": "worker"}
        b = {"id": NODE_ID_2, "name": "beta", "node_type": "worker"}
        assert svc._get_cache_key([a], []) != svc._get_cache_key([b], [])

    def test_edges_affect_cache_key(self):
        svc = _make_svc()
        node = {"id": NODE_ID_1, "name": "n", "node_type": "worker"}
        edge = {"source": NODE_ID_1, "target": NODE_ID_2}
        key_no_edge = svc._get_cache_key([node], [])
        key_with_edge = svc._get_cache_key([node], [edge])
        assert key_no_edge != key_with_edge

    def test_cache_key_with_pydantic_models(self):
        """_get_cache_key handles NodeSchema objects via model_dump()."""
        from uuid import UUID

        svc = _make_svc()
        node = NodeSchema(id=UUID(NODE_ID_1), name="n1", node_type="worker", worker_name="w1")
        # Should not raise and should return a string
        key = svc._get_cache_key([node], [])
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest


class TestCacheExpiry:
    """TTL-based cache expiry."""

    def test_expired_entries_are_removed(self):
        svc = _make_svc()
        svc._cache_ttl = 60

        svc._graph_cache["old"] = "g_old"
        svc._cache_timestamps["old"] = datetime.now(UTC) - timedelta(seconds=120)
        svc._graph_cache["fresh"] = "g_fresh"
        svc._cache_timestamps["fresh"] = datetime.now(UTC) - timedelta(seconds=10)

        svc._cleanup_expired_cache_entries()

        assert "old" not in svc._graph_cache
        assert "old" not in svc._cache_timestamps
        assert "fresh" in svc._graph_cache

    def test_nothing_removed_when_all_fresh(self):
        svc = _make_svc()
        svc._cache_ttl = 300
        svc._graph_cache["a"] = "ga"
        svc._cache_timestamps["a"] = datetime.now(UTC)

        svc._cleanup_expired_cache_entries()
        assert "a" in svc._graph_cache


class TestCacheSizeLimit:
    """LRU eviction when cache exceeds max size."""

    def test_oldest_evicted_when_over_limit(self):
        svc = _make_svc()
        svc._cache_max_size = 2

        base = datetime.now(UTC)
        for i in range(4):
            svc._graph_cache[f"k{i}"] = f"g{i}"
            svc._cache_timestamps[f"k{i}"] = base + timedelta(seconds=i)

        svc._enforce_cache_size_limit()

        assert len(svc._graph_cache) == 2
        # Newest two survive
        assert "k2" in svc._graph_cache
        assert "k3" in svc._graph_cache
        assert "k0" not in svc._graph_cache
        assert "k1" not in svc._graph_cache

    def test_no_eviction_when_under_limit(self):
        svc = _make_svc()
        svc._cache_max_size = 10
        svc._graph_cache["only"] = "graph"
        svc._cache_timestamps["only"] = datetime.now(UTC)

        svc._enforce_cache_size_limit()
        assert len(svc._graph_cache) == 1


# ===================================================================
# 2. Real graph execution tests
# ===================================================================


class TestSingleNodeExecution:
    """Execute a single-node worker graph, mocking only the DB + worker call."""

    @pytest.mark.asyncio
    async def test_completes_with_worker_response(self):
        svc = _make_svc()
        worker_resp = {"response": "task done", "status": "success"}
        response, mock_wm = await _run_graph(svc, _single_node_graph(), worker_resp)

        assert response.status == "completed"
        assert response.session_id  # non-empty
        assert response.error_message is None
        assert response.final_output is not None
        # Worker was actually called with the right worker name and template text
        mock_wm.send_message_to_worker.assert_called_once()
        call_kwargs = mock_wm.send_message_to_worker.call_args
        assert (
            call_kwargs.kwargs.get("worker_name") == "test_worker"
            or call_kwargs[1].get("worker_name") == "test_worker"
            or (len(call_kwargs[0]) > 0 and call_kwargs[0][0] == "test_worker")
        )

    @pytest.mark.asyncio
    async def test_run_identity_is_forwarded_to_the_worker_loopback(self):
        # The run's (tenant_id, subject_id) must flow execute_graph -> WorkflowState
        # -> OFFICIAL WorkerNode -> send_message_to_worker, so the loopback POST can
        # carry a raw X-Tenant-Id and the receiver enforces byo_required (rather than
        # falling back to DEFAULT_TENANT_ID / local_allowed and riding platform GPU).
        svc = _make_svc()
        response, mock_wm = await _run_graph(svc, _single_node_graph())

        assert response.status == "completed"
        mock_wm.send_message_to_worker.assert_called_once()
        kwargs = mock_wm.send_message_to_worker.call_args.kwargs
        # _run_graph invokes execute_graph with tenant_id=subject_id="creator-1".
        assert kwargs.get("tenant_id") == "creator-1"
        assert kwargs.get("subject_id") == "creator-1"

    @pytest.mark.asyncio
    async def test_node_output_captured(self):
        svc = _make_svc()
        worker_resp = {"response": "hello world", "status": "ok"}
        response, _ = await _run_graph(svc, _single_node_graph(), worker_resp)

        assert response.status == "completed"
        # The node output should be stored in node_outputs under the node ID
        assert NODE_ID_1 in response.node_outputs
        assert response.node_outputs[NODE_ID_1] == worker_resp

    @pytest.mark.asyncio
    async def test_variables_passed_through(self):
        svc = _make_svc()
        worker_resp = {"response": "processed", "status": "ok"}
        response, _ = await _run_graph(svc, _single_node_graph(), worker_resp, variables={"x": 42})
        assert response.status == "completed"


class TestTwoNodeChainExecution:
    """Two worker nodes in sequence: A -> B."""

    @pytest.mark.asyncio
    async def test_both_nodes_execute_in_order(self):
        svc = _make_svc()
        call_order = []

        async def side_effect(worker_name, message, context=None, **kwargs):
            call_order.append(worker_name)
            return {"response": f"{worker_name} done", "status": "ok"}

        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_two_node_chain(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(side_effect=side_effect)
            MockWM.return_value = mock_wm

            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-chain",
                request=ExecutionRequest(variables={}),
            )

        assert response.status == "completed"
        # Both workers called in order
        assert call_order == ["worker_a", "worker_b"]
        # Both node IDs have outputs
        assert NODE_ID_1 in response.node_outputs
        assert NODE_ID_2 in response.node_outputs

    @pytest.mark.asyncio
    async def test_second_node_sees_first_node_output(self):
        """Node B should have access to node A's output via state."""
        svc = _make_svc()
        captured_messages = []

        async def side_effect(worker_name, message, context=None, **kwargs):
            captured_messages.append((worker_name, message))
            return {"response": f"result from {worker_name}", "status": "ok"}

        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_two_node_chain(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(side_effect=side_effect)
            MockWM.return_value = mock_wm

            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-chain",
                request=ExecutionRequest(variables={}),
            )

        assert response.status == "completed"
        assert len(captured_messages) == 2
        # After A completes, B executes — that confirms ordering
        assert captured_messages[0][0] == "worker_a"
        assert captured_messages[1][0] == "worker_b"


class TestWorkerFailure:
    """Worker raises an exception during execution."""

    @pytest.mark.asyncio
    async def test_worker_exception_captured_in_output(self):
        """WorkerNode.run() catches exceptions and returns End('Error: ...')."""
        svc = _make_svc()
        response, _ = await _run_graph(svc, _single_node_graph(), RuntimeError("Worker crashed"))
        # The graph itself completes because WorkerNode wraps errors with End()
        assert response.status == "completed"
        assert response.final_output is not None
        assert "Worker crashed" in str(response.final_output)

    @pytest.mark.asyncio
    async def test_worker_exception_in_chain_stops_propagation(self):
        """In A->B, if A's worker fails, the error is in final output."""
        svc = _make_svc()

        async def failing_a(worker_name, message, context=None, **kwargs):
            if worker_name == "worker_a":
                raise RuntimeError("A exploded")
            return {"response": "ok", "status": "ok"}

        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_two_node_chain(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(side_effect=failing_a)
            MockWM.return_value = mock_wm

            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-chain",
                request=ExecutionRequest(variables={}),
            )

        # WorkerNode A catches exception, returns End("Error: A exploded")
        # so the graph completes but with error text as final output
        assert response.final_output is not None
        assert "A exploded" in str(response.final_output)


class TestGraphNotFound:
    """Database returns None — the graph does not exist under this scope.

    ⚠️ Behaviour CHANGED in issue #211, and the two tests that used to live here
    are the reason the defect survived: they asserted
    ``response.status == "failed"``, i.e. they encoded the swallow as the
    contract. `execute_graph` caught its own ``ValueError`` and returned a normal
    ``ExecutionResponse``, so the endpoint answered HTTP 200 and 1,501
    consecutive failures were reported to the AA conveyor as successes.

    A graph that does not exist is a REQUEST error: nothing ran, so there is no
    execution to describe. It now raises, and the routers answer 404. The full
    treatment — endpoint status, the untouched 200-on-execution-failure control,
    and the no-session-row invariant — is in
    ``tests/test_graph_execution_not_found_status.py``.
    """

    @pytest.mark.asyncio
    async def test_raises_instead_of_returning_a_failed_response(self):
        svc = _make_svc()
        with pytest.raises(GraphNotFoundError) as exc_info:
            await _run_graph(svc, None)

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.graph_id == "missing"

    @pytest.mark.asyncio
    async def test_the_raise_is_still_a_valueerror(self):
        """Pre-existing ``except ValueError`` handlers must keep catching it."""
        svc = _make_svc()
        with pytest.raises(ValueError):
            await _run_graph(svc, None)


class TestTaskTemplateInterpolation:
    """Task template variables get resolved before sending to worker."""

    @pytest.mark.asyncio
    async def test_template_with_variable(self):
        svc = _make_svc()
        captured = {}

        async def capture(worker_name, message, context=None, **kwargs):
            captured["message"] = message
            return {"response": "ok", "status": "ok"}

        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_template_graph(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(side_effect=capture)
            MockWM.return_value = mock_wm

            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-tpl",
                variables={"input_file": "data.csv"},
            )

        assert response.status == "completed"
        # The template "Process {input_file}" should resolve to "Process data.csv"
        assert captured["message"] == "Process data.csv"

    @pytest.mark.asyncio
    async def test_template_without_matching_variable_uses_original(self):
        """When a template variable is not in the context, format() raises KeyError
        and the code falls back to the original template string."""
        svc = _make_svc()
        captured = {}

        async def capture(worker_name, message, context=None, **kwargs):
            captured["message"] = message
            return {"response": "ok", "status": "ok"}

        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_template_graph(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(side_effect=capture)
            MockWM.return_value = mock_wm

            response = await svc.execute_graph(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-tpl",
                variables={},  # No input_file variable
            )

        assert response.status == "completed"
        # Falls back to original template when variable missing
        assert captured["message"] == "Process {input_file}"


class TestInvalidGraphSchema:
    """Bad graph data should produce a failed response."""

    @pytest.mark.asyncio
    async def test_missing_node_name_fails(self):
        bad_data = {
            "id": "graph-bad",
            "tenant_id": "creator-1",
            "subject_id": "creator-1",
            "visibility": "tenant",
            "name": "bad",
            "nodes": [{"id": NODE_ID_1, "node_type": "worker"}],  # missing name
            "edges": [],
            "variables": {},
        }
        svc = _make_svc()
        response, _ = await _run_graph(svc, bad_data)
        assert response.status == "failed"
        assert response.error_message is not None

    @pytest.mark.asyncio
    async def test_edge_referencing_nonexistent_node_fails(self):
        bad_data = {
            "id": "graph-bad-edge",
            "tenant_id": "creator-1",
            "subject_id": "creator-1",
            "visibility": "tenant",
            "name": "bad_edge",
            "nodes": [
                {
                    "id": NODE_ID_1,
                    "name": "n1",
                    "node_type": "worker",
                    "worker_name": "w1",
                    "dependencies": [],
                }
            ],
            "edges": [{"source": NODE_ID_1, "target": "00000000-0000-0000-0000-000000000099"}],
            "variables": {},
        }
        svc = _make_svc()
        response, _ = await _run_graph(svc, bad_data)
        assert response.status == "failed"
        assert response.error_message is not None


# ===================================================================
# 3. DB persistence via _persist_session
# ===================================================================


class TestPersistSession:
    """Verify _persist_session calls execution_db.add_graph_state correctly."""

    @pytest.mark.asyncio
    async def test_persist_called_on_successful_execution(self):
        mock_db = AsyncMock()
        mock_db.add_graph_state = AsyncMock()
        svc = _make_svc(execution_db=mock_db)

        response, _ = await _run_graph(
            svc,
            _single_node_graph(),
            {"response": "ok", "status": "ok"},
        )

        assert response.status == "completed"
        mock_db.add_graph_state.assert_called_once()
        call_kwargs = mock_db.add_graph_state.call_args.kwargs
        assert call_kwargs["tenant_id"] == "creator-1"
        assert call_kwargs["graph_id"] == "graph-1"
        payload = call_kwargs["state_payload"]
        assert payload["status"] == "completed"
        assert payload["event_type"] == "execution_complete"
        assert payload["session_id"] == response.session_id

    @pytest.mark.asyncio
    async def test_persist_called_on_failed_execution(self):
        mock_db = AsyncMock()
        mock_db.add_graph_state = AsyncMock()
        svc = _make_svc(execution_db=mock_db)

        # A graph that EXISTS but cannot be built. This used to use
        # `_run_graph(svc, None)` (graph not found), which no longer produces a
        # response at all — and it never should have been the vehicle here: what
        # this test wants is a genuine execution failure, which is exactly the
        # class that still persists a session and still answers 200.
        response, _ = await _run_graph(svc, _unbuildable_graph())

        assert response.status == "failed"
        mock_db.add_graph_state.assert_called_once()
        payload = mock_db.add_graph_state.call_args.kwargs["state_payload"]
        assert payload["status"] == "failed"
        assert payload["error_message"] is not None

    @pytest.mark.asyncio
    async def test_persist_db_error_does_not_propagate(self):
        """If execution_db.add_graph_state raises, execution still returns ok."""
        mock_db = AsyncMock()
        mock_db.add_graph_state = AsyncMock(side_effect=RuntimeError("db down"))
        svc = _make_svc(execution_db=mock_db)

        response, _ = await _run_graph(
            svc,
            _single_node_graph(),
            {"response": "ok", "status": "ok"},
        )
        # Execution should still complete even though persist failed
        assert response.status == "completed"

    @pytest.mark.asyncio
    async def test_no_persist_when_no_execution_db(self):
        """When execution_db is None, _persist_session is a no-op."""
        svc = _make_svc(execution_db=None)
        assert svc.execution_db is None
        # Should not raise
        await svc._persist_session(
            tenant_id="c",
            subject_id="c",
            graph_id="g",
            session_id="s",
            status="completed",
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
            error_message=None,
            final_output=None,
        )


# ===================================================================
# 4. Session management
# ===================================================================


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_get_session_status_existing_without_db_row_is_not_completed(self):
        """A session with no finalized DB row must NOT be reported as completed.

        The fallback branch has no evidence the graph finished — ``final_output``
        is None and no terminal row exists — so claiming ``completed`` tells a
        poller the graph finished and produced nothing. The honest answer is
        ``running``.
        """
        svc = _make_svc()
        # Force DB-less path so the in-memory fallback branch runs.
        svc.session_db = None
        svc.active_sessions["sess-1"] = MagicMock()

        result = await svc.get_session_status(
            tenant_id="creator-1", subject_id="creator-1", session_id="sess-1"
        )
        assert result is not None
        assert result["session_id"] == "sess-1"
        assert result["status"] == "running"
        assert result["status"] != "completed"
        assert result["final_output"] is None

    @pytest.mark.asyncio
    async def test_get_session_status_finished_session_still_reports_completed(self):
        """A genuinely finished session (DB row present) still reports completed."""
        svc = _make_svc()
        now = datetime.now(UTC)
        svc.session_db = MagicMock()
        svc.session_db.get_session = AsyncMock(
            return_value={
                "session_id": "sess-done",
                "status": "completed",
                "completed_nodes": ["n1"],
                "failed_nodes": [],
                "node_outputs": {"n1": "ok"},
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        # Present in active_sessions too: the DB row must win over the fallback.
        svc.active_sessions["sess-done"] = MagicMock()

        result = await svc.get_session_status(
            tenant_id="creator-1", subject_id="creator-1", session_id="sess-done"
        )
        assert result is not None
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_session_status_nonexistent(self):
        svc = _make_svc()
        svc.session_db = None
        result = await svc.get_session_status(
            tenant_id="creator-1", subject_id="creator-1", session_id="no-such-session"
        )
        assert result is None

    def test_clear_cache_removes_everything(self):
        svc = _make_svc()
        svc._graph_cache["k1"] = "v1"
        svc._cache_timestamps["k1"] = datetime.now(UTC)
        svc.active_sessions["s1"] = MagicMock()

        svc.clear_cache()

        assert svc._graph_cache == {}
        assert svc._cache_timestamps == {}
        assert svc.active_sessions == {}

    def test_get_stats_returns_correct_counts(self):
        svc = _make_svc()
        svc._graph_cache["a"] = "ga"
        svc._graph_cache["b"] = "gb"
        svc.active_sessions["s1"] = MagicMock()

        stats = svc.get_stats()
        assert stats["active_sessions"] == 1
        pg = stats["pydantic_graph_stats"]
        assert pg["cached_graphs"] == 2
        assert pg["active_sessions"] == 1

    def test_get_available_engines(self):
        svc = _make_svc()
        engines = svc.get_available_engines()
        assert "simple" in engines
        assert "pydantic_graph" in engines

    @pytest.mark.asyncio
    async def test_cleanup_session_removes_active_session(self):
        svc = _make_svc()
        svc.active_sessions["sess-1"] = MagicMock()
        svc._session_timestamps["sess-1"] = time.time()

        await svc.cleanup_session("sess-1")

        assert "sess-1" not in svc.active_sessions
        assert "sess-1" not in svc._session_timestamps

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session_is_noop(self):
        svc = _make_svc()
        await svc.cleanup_session("ghost")
        # No exception = pass

    def test_build_execution_metadata(self):
        from src.worker.graphs.pydantic_graph_state import WorkflowState

        svc = _make_svc()
        state = WorkflowState(
            session_id="s1",
            graph_id="g1",
            variables={},
            node_outputs={},
            completed_nodes=["n1", "n2"],
            failed_nodes=["n3"],
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        meta = svc._build_execution_metadata(state, "s1")
        assert meta["session_id"] == "s1"
        assert meta["nodes_executed"] == 2
        assert meta["failed_nodes"] == 1
        assert meta["total_processed"] == 3
        assert "execution_start" in meta
        assert "execution_end" in meta


class TestExecuteGraphStepwise:
    """execute_graph_stepwise delegates to execute_graph."""

    @pytest.mark.asyncio
    async def test_stepwise_delegates(self):
        svc = _make_svc()
        with (
            patch.object(
                svc,
                "_get_graph_from_database",
                new_callable=AsyncMock,
                return_value=_single_node_graph(),
            ),
            patch("src.worker.graphs.graph_factory.WorkerManager") as MockWM,
        ):
            mock_wm = MagicMock()
            mock_wm.send_message_to_worker = AsyncMock(
                return_value={"response": "ok", "status": "ok"}
            )
            MockWM.return_value = mock_wm

            response = await svc.execute_graph_stepwise(
                tenant_id="creator-1",
                subject_id="creator-1",
                graph_id="graph-1",
                variables={"a": 1},
            )

        assert response.status == "completed"
