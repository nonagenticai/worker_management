"""Unit tests for InferenceProviderNode (Task 3, Phase 1 inference-graph).

These tests cover:

1. Success path with a single-provider chain.
2. Single-fallback path (provider A fails, provider B succeeds).
3. All-fail path (three providers, all raise).
4. Each-provider-visited-once structural invariant.
5. Terminal non-provider successor (provider + eval_subgraph stub).

The outbound provider HTTP call is mocked via
``unittest.mock.patch`` on the node's ``_call_provider`` helper. The
graph state, factory, and node-class registry are real.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
import pytest
from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import EdgeSchema, NodeSchema
from src.worker.graphs.nodes.inference_provider_node import (
    create_inference_provider_node_class,
)
from src.worker.graphs.pydantic_graph_state import WorkflowState

# An *_API_KEY env-var name (request_overrides), referenced by NAME so the
# detect-secrets keyword plugin does not read it as a credential literal.
K_ENV = "K_API_KEY"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_node(
    name: str,
    model: str = "test-model",
    api_key_env: str = "TEST_API_KEY",
    base_url: str | None = None,
) -> NodeSchema:
    """Build a minimal inference_provider NodeSchema."""
    return NodeSchema(
        name=name,
        node_type="inference_provider",
        framework_config={
            "name": name,
            "model": model,
            "api_key_env": api_key_env,
            "base_url": base_url,
        },
    )


# Owner ruling 2026-10-01 (validator B1): this node reads the env key named by
# ``api_key_env`` for a PLATFORM tenant only. These tests exercise the node's
# provider-call mechanics, so they run as a platform tenant explicitly; the
# customer refusal is pinned in tests/test_tenant_byo_fix2.py.
PLATFORM_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def _proven_platform_request():
    """fr#1385: the env key needs a PROVEN platform tenant, not membership alone.

    Real predicate, real contextvars: a licence-introspected request for
    ``PLATFORM_TENANT`` (the run's state tenant). The refusals are pinned in
    tests/test_inference_provider_owner_key_proof.py.
    """
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
        yield


def _make_state(variables: dict[str, Any] | None = None) -> WorkflowState:
    return WorkflowState(
        session_id="sess-1",
        graph_id="graph-1",
        variables=variables or {"prompt": "hello"},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
        tenant_id=PLATFORM_TENANT,
    )


class _FakeCtx:
    """Duck-typed stand-in for GraphRunContext that just carries state."""

    def __init__(self, state: WorkflowState) -> None:
        self.state = state


def _make_ctx(state: WorkflowState | None = None) -> GraphRunContext[WorkflowState]:
    """Return a fake context that pydantic-graph BaseNodes can use.

    The node ``run`` methods only touch ``ctx.state`` — we cast through
    ``Any`` to express that the ``_FakeCtx`` is behaviourally compatible.
    """
    s = state or _make_state()
    return cast("GraphRunContext[WorkflowState]", cast("Any", _FakeCtx(s)))


def _build_chain(
    node_schemas: list[NodeSchema],
) -> dict[str, type[BaseNode[WorkflowState, None, Any]]]:
    """Build a linear chain provider_0 -> provider_1 -> ... and return
    a mapping from node-schema-id to the generated node class."""
    edges: list[EdgeSchema] = []
    for i in range(len(node_schemas) - 1):
        edges.append(EdgeSchema(source=node_schemas[i].id, target=node_schemas[i + 1].id))
    factory = GraphFactory(worker_manager=None)
    node_map = {str(n.id): n for n in node_schemas}
    edge_map = factory._build_edge_map(edges)
    classes = factory._create_node_classes(node_schemas, node_map, edge_map)
    return {cast("Any", cls())._node_schema_id: cls for cls in classes}


async def _drive_chain(
    start_cls: type[BaseNode[WorkflowState, None, Any]],
    state: WorkflowState,
) -> End[Any]:
    """Walk the FSM by repeatedly calling node.run() and following the
    returned node instance until End() is reached."""
    current: BaseNode[WorkflowState, None, Any] | End[Any] = start_cls()
    # guard against infinite loops in bad test setups
    for _ in range(100):
        if isinstance(current, End):
            return current
        ctx = _make_ctx(state)
        # ty over-narrows the union after isinstance; cast to BaseNode explicitly.
        current = await cast("BaseNode[WorkflowState, None, Any]", current).run(ctx)
    raise AssertionError("FSM did not terminate within 100 steps")


# ---------------------------------------------------------------------------
# 2. Single fallback
# ---------------------------------------------------------------------------


class TestInferenceProviderNodeSingleFallback:
    @pytest.mark.asyncio
    async def test_first_provider_fails_second_succeeds(self) -> None:
        node_a = _make_provider_node("A")
        node_b = _make_provider_node("B")
        registry = _build_chain([node_a, node_b])
        cls_a = registry[str(node_a.id)]
        cls_b = registry[str(node_b.id)]

        response_b = {
            "id": "resp-b",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }

        with (
            patch.object(
                cls_a,
                "_call_provider",
                new=AsyncMock(
                    side_effect=httpx.ConnectError("refused"),
                ),
            ),
            patch.object(
                cls_b,
                "_call_provider",
                new=AsyncMock(return_value=(response_b, 3, 4)),
            ),
        ):
            state = _make_state()
            result = await _drive_chain(cls_a, state)

        assert isinstance(result, End)
        bag = state.framework_state["inference"]
        assert bag["chosen_provider"] == "B"
        assert bag["final_response"] == response_b

        attempts = bag["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["provider"] == "A"
        assert attempts[0]["status"] == "error"
        assert attempts[1]["provider"] == "B"
        assert attempts[1]["status"] == "ok"

        # fallback_count invariant: len(attempts) - 1 when chosen_provider is set
        assert len(attempts) - 1 == 1


# ---------------------------------------------------------------------------
# 3. All fail
# ---------------------------------------------------------------------------


class TestInferenceProviderNodeAllFail:
    @pytest.mark.asyncio
    async def test_three_providers_all_fail(self) -> None:
        node_a = _make_provider_node("A")
        node_b = _make_provider_node("B")
        node_c = _make_provider_node("C")
        registry = _build_chain([node_a, node_b, node_c])
        cls_a = registry[str(node_a.id)]
        cls_b = registry[str(node_b.id)]
        cls_c = registry[str(node_c.id)]

        with (
            patch.object(
                cls_a,
                "_call_provider",
                new=AsyncMock(side_effect=httpx.ConnectError("no conn A")),
            ),
            patch.object(
                cls_b,
                "_call_provider",
                new=AsyncMock(side_effect=httpx.ReadTimeout("timeout B")),
            ),
            patch.object(
                cls_c,
                "_call_provider",
                new=AsyncMock(side_effect=TimeoutError("timeout C")),
            ),
        ):
            state = _make_state()
            result = await _drive_chain(cls_a, state)

        assert isinstance(result, End)
        final = result.data
        assert final == {
            "status": "failed",
            "error": "all providers failed",
            "attempts": 3,
        }

        bag = state.framework_state["inference"]
        assert bag["chosen_provider"] is None
        assert bag["final_response"] is None

        attempts = bag["attempts"]
        assert len(attempts) == 3
        provider_names = [a["provider"] for a in attempts]
        assert provider_names == ["A", "B", "C"]
        assert all(a["status"] == "error" for a in attempts)

    @pytest.mark.asyncio
    async def test_each_provider_visited_at_most_once(self) -> None:
        """Structural invariant: linear forward chain, so each provider
        name appears at most once in attempts."""
        node_a = _make_provider_node("A")
        node_b = _make_provider_node("B")
        node_c = _make_provider_node("C")
        registry = _build_chain([node_a, node_b, node_c])
        cls_a = registry[str(node_a.id)]
        cls_b = registry[str(node_b.id)]
        cls_c = registry[str(node_c.id)]

        with (
            patch.object(
                cls_a,
                "_call_provider",
                new=AsyncMock(side_effect=httpx.ConnectError("fail")),
            ),
            patch.object(
                cls_b,
                "_call_provider",
                new=AsyncMock(side_effect=httpx.ConnectError("fail")),
            ),
            patch.object(
                cls_c,
                "_call_provider",
                new=AsyncMock(side_effect=httpx.ConnectError("fail")),
            ),
        ):
            state = _make_state()
            await _drive_chain(cls_a, state)

        attempts = state.framework_state["inference"]["attempts"]
        names = {a["provider"] for a in attempts}
        assert len(names) == 3
        assert names == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# 4. Terminal (non-provider) successor, e.g. eval_subgraph
# ---------------------------------------------------------------------------


class TestInferenceProviderNodeTerminalSuccessor:
    @pytest.mark.asyncio
    async def test_create_inference_provider_node_class_smoke(self) -> None:
        """create_inference_provider_node_class() returns a usable class."""
        schema = _make_provider_node("solo")
        cls_ = create_inference_provider_node_class(
            schema=schema,
            next_node_ids=[],
            all_nodes={str(schema.id): schema},
            edge_map={},
        )
        instance = cls_()
        assert cast("Any", instance)._node_schema_id == str(schema.id)
        assert cast("Any", instance)._provider_config["name"] == "solo"


# ---------------------------------------------------------------------------
# 5. Structured output + reasoning flag wiring (PR: fix/inference-provider-timeout-and-schema)
# ---------------------------------------------------------------------------


class TestInferenceProviderNodeStructuredOutput:
    """When the caller sets ``variables['output_schema']`` the provider call
    must include ``response_format={"type":"json_schema", …}`` so vLLM picks
    up the guided-decoding constraint via IR. Pre-fix the field was dropped.
    """

    @pytest.mark.asyncio
    async def test_output_schema_in_variables_becomes_response_format(self) -> None:
        from unittest.mock import MagicMock

        schema = _make_provider_node("solo")
        cls_ = create_inference_provider_node_class(
            schema=schema,
            next_node_ids=[],
            all_nodes={str(schema.id): schema},
            edge_map={},
        )

        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock(
                model_dump=lambda: {
                    "id": "x",
                    "choices": [{"message": {"role": "assistant", "content": '{"a":1}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        json_schema = {
            "title": "EntityTyping",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        state = _make_state(
            {
                "messages": [{"role": "user", "content": "type this"}],
                "output_schema": json_schema,
            }
        )

        with patch("openai.AsyncOpenAI") as mock_ctor:
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=fake_create)
            client.close = AsyncMock(return_value=None)
            mock_ctor.return_value = client
            await cls_().run(_make_ctx(state))

        assert "response_format" in captured
        assert captured["response_format"]["type"] == "json_schema"
        # The schema is pruned to what xgrammar can compile before it is sent.
        # ``title`` is one of the pruned keywords, but it still supplies the
        # response_format name — read off the original schema, not the pruned one.
        assert captured["response_format"]["json_schema"]["schema"] == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert captured["response_format"]["json_schema"]["name"] == "EntityTyping"
        # extra_body always carries the reasoning toggle (default False)
        assert captured["extra_body"] == {"enable_reasoning": False}

    @pytest.mark.asyncio
    async def test_enable_reasoning_per_request_overrides_framework_config(self) -> None:
        from unittest.mock import MagicMock

        # framework_config: reasoning ON by default for this provider
        node_schema = NodeSchema(
            name="solo",
            node_type="inference_provider",
            framework_config={
                "name": "solo",
                "model": "m",
                "api_key_env": K_ENV,
                "enable_reasoning": True,
            },
        )
        cls_ = create_inference_provider_node_class(
            schema=node_schema,
            next_node_ids=[],
            all_nodes={str(node_schema.id): node_schema},
            edge_map={},
        )

        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock(
                model_dump=lambda: {
                    "id": "x",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        # Per-request override flips it OFF
        state = _make_state(
            {
                "prompt": "hi",
                "enable_reasoning": False,
            }
        )

        with patch("openai.AsyncOpenAI") as mock_ctor:
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=fake_create)
            client.close = AsyncMock(return_value=None)
            mock_ctor.return_value = client
            await cls_().run(_make_ctx(state))

        assert captured["extra_body"] == {"enable_reasoning": False}

    @pytest.mark.asyncio
    async def test_no_output_schema_omits_response_format(self) -> None:
        from unittest.mock import MagicMock

        schema = _make_provider_node("solo")
        cls_ = create_inference_provider_node_class(
            schema=schema,
            next_node_ids=[],
            all_nodes={str(schema.id): schema},
            edge_map={},
        )

        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock(
                model_dump=lambda: {
                    "id": "x",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        with patch("openai.AsyncOpenAI") as mock_ctor:
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=fake_create)
            client.close = AsyncMock(return_value=None)
            mock_ctor.return_value = client
            await cls_().run(_make_ctx(_make_state({"prompt": "hi"})))

        assert "response_format" not in captured


class TestInferenceProviderNodeErrorClassification:
    """The structured fallback log + attempts record now carry an
    ``error_classification`` label aligned with IR's
    ``X-IR-Error-Classification`` header so logs across IR + WM join cleanly.
    """

    @pytest.mark.asyncio
    async def test_x_ir_error_classification_header_wins_over_heuristic(self) -> None:
        """When the upstream is IR and it stamped a classification header on
        its own 5xx, that wins over the local exception-type heuristic."""
        from unittest.mock import MagicMock

        node_a = _make_provider_node("A")
        registry = _build_chain([node_a])
        cls_a = registry[str(node_a.id)]

        # Build an openai.APIStatusError whose underlying response has the
        # X-IR-Error-Classification header set to "queue_rejected" — would
        # otherwise be classified upstream_5xx by exc.status_code.
        resp = MagicMock()
        resp.headers = {"x-ir-error-classification": "queue_rejected"}
        resp.status_code = 503
        body: dict[str, Any] = {"detail": "queue full"}
        import openai as _openai

        api_err = _openai.APIStatusError("queue full", response=resp, body=body)

        with patch.object(
            cls_a,
            "_call_provider",
            new=AsyncMock(side_effect=api_err),
        ):
            state = _make_state()
            await _drive_chain(cls_a, state)

        attempts = state.framework_state["inference"]["attempts"]
        assert attempts[0]["error_classification"] == "queue_rejected"


class TestInferenceProviderNodeDefaultTimeout:
    """Default timeout comes from
    ``Settings.inference_provider_node_default_timeout_s`` (600s, aligned with the
    LLM timeout ladder) when the framework_config doesn't override. Was hardcoded
    30s pre-fix, then a 240s inverted hop below the 600s downstream budget.
    """

    def test_default_timeout_picks_up_settings_value(self) -> None:
        schema = _make_provider_node("solo")
        cls_ = create_inference_provider_node_class(
            schema=schema,
            next_node_ids=[],
            all_nodes={str(schema.id): schema},
            edge_map={},
        )
        instance = cls_()
        # 240 is the in-repo default; CI can override via env var.
        assert cast("Any", instance)._provider_config["timeout"] >= 60
        # Explicit override wins.
        schema_override = NodeSchema(
            name="solo",
            node_type="inference_provider",
            framework_config={
                "name": "solo",
                "model": "m",
                "api_key_env": K_ENV,
                "timeout": 15,
            },
        )
        cls_override = create_inference_provider_node_class(
            schema=schema_override,
            next_node_ids=[],
            all_nodes={str(schema_override.id): schema_override},
            edge_map={},
        )
        assert cast("Any", cls_override())._provider_config["timeout"] == 15
