"""Unit tests for LoadAdapterNode (Task 4, Phase 1 inference-graph).

These tests cover:

1. Happy path — POST 200 is recorded as ``status="ok"`` and the chain
   continues.
2. Connection error + ``require_adapter=False`` — chain still continues,
   failure recorded.
3. Connection error + ``require_adapter=True`` — ``End(...)`` short-circuit.
4. Non-2xx HTTP response — treated as failure.
5. Timeout — treated as failure.

Only the outbound httpx call is mocked (via ``patch.object`` on
``httpx.AsyncClient``'s ``post``). Graph state, factory, and registry
are real.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest
from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import EdgeSchema, NodeSchema
from src.worker.graphs.pydantic_graph_state import WorkflowState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_node(
    adapter_id: str = "lora-abc",
    require_adapter: bool = False,
    adapter_manager_url: str = "http://adapter-manager.test:8001",
) -> NodeSchema:
    return NodeSchema(
        name=f"load_adapter_{adapter_id}",
        node_type="load_adapter",
        framework_config={
            "adapter_id": adapter_id,
            "adapter_manager_url": adapter_manager_url,
            "require_adapter": require_adapter,
            "timeout": 5,
        },
    )


def _make_trailing_function_node() -> NodeSchema:
    """Build a trivial successor node the adapter node can hand off to."""
    return NodeSchema(
        name="after_adapter",
        node_type="function",
        function_name="noop",
        data={},
    )


def _make_state() -> WorkflowState:
    return WorkflowState(
        session_id="sess-1",
        graph_id="graph-1",
        variables={"prompt": "hi"},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )


class _FakeCtx:
    def __init__(self, state: WorkflowState) -> None:
        self.state = state


def _make_ctx(state: WorkflowState) -> GraphRunContext[WorkflowState]:
    return cast("GraphRunContext[WorkflowState]", cast("Any", _FakeCtx(state)))


def _build_registry(
    nodes: list[NodeSchema],
    edges: list[EdgeSchema] | None = None,
) -> dict[str, type[BaseNode[WorkflowState, None, Any]]]:
    factory = GraphFactory(worker_manager=None)
    node_map = {str(n.id): n for n in nodes}
    edge_map = factory._build_edge_map(edges or [])
    classes = factory._create_node_classes(nodes, node_map, edge_map)
    return {cast("Any", cls())._node_schema_id: cls for cls in classes}


def _mock_post_response(status_code: int) -> MagicMock:
    """Build a MagicMock that looks like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if 200 <= status_code < 300:
        resp.raise_for_status = MagicMock(return_value=None)
    else:
        req = httpx.Request("POST", "http://adapter-manager.test:8001/adapters/load")
        http_resp = httpx.Response(status_code=status_code, request=req)
        err = httpx.HTTPStatusError(
            f"{status_code} error",
            request=req,
            response=http_resp,
        )
        resp.raise_for_status = MagicMock(side_effect=err)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadAdapterNodeHappyPath:
    @pytest.mark.asyncio
    async def test_200_returns_next_node_and_records_ok(self) -> None:
        adapter_node = _make_adapter_node("lora-happy")
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(return_value=_mock_post_response(200)),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert not isinstance(result, End)
        assert isinstance(result, cls_next)

        loads = state.framework_state["adapter_loads"]
        assert len(loads) == 1
        assert loads[0]["adapter_id"] == "lora-happy"
        assert loads[0]["status"] == "ok"
        assert isinstance(loads[0]["latency_ms"], int)
        assert loads[0]["latency_ms"] >= 0

        assert state.node_outputs[str(adapter_node.id)] == {
            "adapter_id": "lora-happy",
            "loaded": True,
        }

    @pytest.mark.asyncio
    async def test_200_no_successor_ends_cleanly(self) -> None:
        adapter_node = _make_adapter_node("lora-solo")
        registry = _build_registry([adapter_node], [])
        cls_adapter = registry[str(adapter_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(return_value=_mock_post_response(200)),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        assert result.data == {"adapter_id": "lora-solo", "loaded": True}


class TestLoadAdapterNodeNonRequiredFailures:
    @pytest.mark.asyncio
    async def test_connect_error_require_false_continues(self) -> None:
        adapter_node = _make_adapter_node("lora-X", require_adapter=False)
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert not isinstance(result, End)
        assert isinstance(result, cls_next)

        loads = state.framework_state["adapter_loads"]
        assert len(loads) == 1
        assert loads[0]["adapter_id"] == "lora-X"
        assert loads[0]["status"] == "error"
        assert "refused" in loads[0]["error"]

        assert str(adapter_node.id) in state.failed_nodes
        assert state.node_status[str(adapter_node.id)] == "failed"

    @pytest.mark.asyncio
    async def test_connect_error_no_successor_ends_with_failure(self) -> None:
        adapter_node = _make_adapter_node("lora-X", require_adapter=False)
        registry = _build_registry([adapter_node], [])
        cls_adapter = registry[str(adapter_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        assert result.data == {
            "status": "failed",
            "error": "adapter load failed",
            "adapter_id": "lora-X",
        }

    @pytest.mark.asyncio
    async def test_non_2xx_treated_as_failure(self) -> None:
        adapter_node = _make_adapter_node("lora-500", require_adapter=False)
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(return_value=_mock_post_response(500)),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert isinstance(result, cls_next)
        loads = state.framework_state["adapter_loads"]
        assert loads[0]["status"] == "error"
        assert "500" in loads[0]["error"]
        assert str(adapter_node.id) in state.failed_nodes

    @pytest.mark.asyncio
    async def test_timeout_treated_as_failure(self) -> None:
        adapter_node = _make_adapter_node("lora-T", require_adapter=False)
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.TimeoutException("slow")),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert isinstance(result, cls_next)
        loads = state.framework_state["adapter_loads"]
        assert loads[0]["status"] == "error"
        assert "slow" in loads[0]["error"]

    @pytest.mark.asyncio
    async def test_stdlib_timeout_error_treated_as_failure(self) -> None:
        """Cover the ``TimeoutError`` arm of ``(TimeoutError, httpx.HTTPError)``.

        ``httpx.TimeoutException`` is an ``httpx.HTTPError`` subclass and
        therefore only exercises the second tuple element. This test
        raises the stdlib ``TimeoutError`` directly so the first arm is
        exercised too.
        """
        adapter_node = _make_adapter_node("lora-stdlib-T", require_adapter=False)
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(side_effect=TimeoutError("stdlib timeout")),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        # Chain continues because require_adapter=False.
        assert not isinstance(result, End)
        assert isinstance(result, cls_next)

        loads = state.framework_state["adapter_loads"]
        assert len(loads) == 1
        assert loads[0]["adapter_id"] == "lora-stdlib-T"
        assert loads[0]["status"] == "error"
        assert "stdlib timeout" in loads[0]["error"]

        assert str(adapter_node.id) in state.failed_nodes
        assert state.node_status[str(adapter_node.id)] == "failed"


class TestLoadAdapterNodeRequiredFailures:
    @pytest.mark.asyncio
    async def test_connect_error_require_true_short_circuits(self) -> None:
        adapter_node = _make_adapter_node("lora-REQ", require_adapter=True)
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]

        with patch(
            "httpx2.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            state = _make_state()
            instance = cls_adapter()
            result = await instance.run(_make_ctx(state))

        assert isinstance(result, End)
        assert result.data == {
            "status": "failed",
            "error": "adapter load failed",
            "adapter_id": "lora-REQ",
        }
        loads = state.framework_state["adapter_loads"]
        assert loads[0]["status"] == "error"


class TestLoadAdapterNodeConfigValidation:
    def test_missing_adapter_id_raises_at_model_validation(self) -> None:
        # NodeSchema-level validation must reject missing adapter_id
        # with a specific error message.
        with pytest.raises(ValueError, match="adapter_id"):
            NodeSchema(
                name="load_bad",
                node_type="load_adapter",
                framework_config={},
            )


class TestLoadAdapterNodeNotConfigured:
    """No adapter-manager URL → typed not-configured failure, zero HTTP."""

    @pytest.fixture(autouse=True)
    def _unset_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.worker.graphs.nodes import load_adapter_node as mod

        monkeypatch.delenv("WM_ADAPTER_MANAGER_URL", raising=False)
        monkeypatch.setattr(mod, "_default_adapter_manager_url", lambda: "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("require_adapter", [False, True])
    async def test_unset_makes_no_http_call(self, require_adapter: bool) -> None:
        adapter_node = _make_adapter_node(
            "lora-nc", require_adapter=require_adapter, adapter_manager_url=""
        )
        next_node = _make_trailing_function_node()
        registry = _build_registry(
            [adapter_node, next_node],
            [EdgeSchema(source=adapter_node.id, target=next_node.id)],
        )
        cls_adapter = registry[str(adapter_node.id)]
        cls_next = registry[str(next_node.id)]

        post = AsyncMock(return_value=_mock_post_response(200))
        with (
            patch("httpx2.AsyncClient.post", new=post),
            patch(
                "src.worker.graphs.nodes.load_adapter_node.httpx.AsyncClient",
                wraps=httpx.AsyncClient,
            ) as client_ctor,
        ):
            state = _make_state()
            result = await cls_adapter().run(_make_ctx(state))

        assert post.await_count == 0
        assert client_ctor.call_count == 0

        loads = state.framework_state["adapter_loads"]
        assert loads == [
            {
                "adapter_id": "lora-nc",
                "status": "not_configured",
                "error_code": "ADAPTER_MANAGER_NOT_CONFIGURED",
                "latency_ms": 0,
                "error": loads[0]["error"],
            }
        ]
        assert "not configured" in loads[0]["error"]
        assert str(adapter_node.id) in state.failed_nodes

        if require_adapter:
            assert isinstance(result, End)
            assert result.data == {
                "status": "failed",
                "error": "adapter manager not configured",
                "error_code": "ADAPTER_MANAGER_NOT_CONFIGURED",
                "adapter_id": "lora-nc",
            }
        else:
            assert isinstance(result, cls_next)

    @pytest.mark.asyncio
    async def test_unset_no_successor_ends_failed(self) -> None:
        adapter_node = _make_adapter_node("lora-nc2", adapter_manager_url="")
        registry = _build_registry([adapter_node])
        post = AsyncMock(return_value=_mock_post_response(200))
        with patch("httpx2.AsyncClient.post", new=post):
            state = _make_state()
            result = await registry[str(adapter_node.id)]().run(_make_ctx(state))
        assert post.await_count == 0
        assert isinstance(result, End)
        assert cast("dict[str, Any]", result.data)["error_code"] == (
            "ADAPTER_MANAGER_NOT_CONFIGURED"
        )

    def test_direct_vllm_override_refused_at_build(self) -> None:
        from src.worker.graphs.nodes.load_adapter_node import _get_adapter_config

        schema = _make_adapter_node(adapter_manager_url="http://vllm.example.com:8001")
        with pytest.raises(ValueError, match="inference-router"):
            _get_adapter_config(schema)
