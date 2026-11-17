"""Count/length bounds: always MEASURED post-hoc, FATAL only when a run opts in.

`maxItems`/`maxLength`/`minItems`/`minLength` are stripped from the xgrammar
grammar, and the output validator only ever reported them as
`schema_soft_violation`, so a bound did nothing. Judge schemas are about to carry
`maxItems`/`maxLength` to stop T=0 repetition loops.

Enforcement is OPT-IN (`enforce_schema_bounds`, per message or per worker
record; the message value wins). Default OFF because requirements_management's
`principle_consistency` has a written contract that WM does not reject an
oversized reply: RM clamps on receipt, and a default-on 422 would quarantine its
most-conflicted procedures.

The contract these tests pin:
  - default: an oversized output returns 200, unchanged, with the violating paths
    in `usage_metrics.bound_violations`;
  - opt-in (#358, George 2026-10-10 "retry then fail"): each violation is a
    `SchemaBoundRetry` (a ModelRetry whose feedback names the path and the bound)
    within the worker's output retry budget; if the LAST attempt still violates,
    the run fails with `SchemaBoundViolationError` → HTTP 422
    `SCHEMA_BOUND_VIOLATION`. Never truncated;
  - the #144 advisory behaviour (tests/test_schema_validator_grammar_parity.py)
    is untouched when not opted in.

No test here makes a network call: agents run on an in-process
`httpx.MockTransport`.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest
from pydantic_ai import ModelRetry

from src.utils.xgrammar_schema import (
    _XGRAMMAR_UNSUPPORTED_KEYS,
    BOUND_KEYWORDS,
    SchemaBoundViolationError,
    strip_xgrammar_unsupported,
)
from src.worker.agents import pydantic_ai_agent as agent_mod
from src.worker.agents.pydantic_ai_agent import SchemaBoundRetry
from tests.test_schema_validator_grammar_parity import _validator_for

if TYPE_CHECKING:
    from collections.abc import Iterator

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

MODULE = "src.worker.agents.pydantic_ai_agent"

# The shape of worker-qa-verify-applicability's bounded `off_target` (the field
# the T=0 loop repeated in).
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "off_target": {
            "type": "array",
            "maxItems": 10,
            "items": {"type": "string", "maxLength": 256},
        },
        "rationale": {"type": "string", "maxLength": 850},
    },
    "required": ["off_target"],
}


def _twelve() -> dict[str, Any]:
    return {"off_target": [f"Human-in-the-Loop Gate {i}" for i in range(12)], "rationale": "r"}


def _in_bounds() -> dict[str, Any]:
    return {"off_target": ["a", "b", "c"], "rationale": "short"}


@contextmanager
def _run_scope(*, enforce: bool) -> Iterator[list[str]]:
    """What `PydanticAIAgent.run` sets around a run, for validator-level tests."""
    recorded: list[str] = []
    t1 = agent_mod._ENFORCE_SCHEMA_BOUNDS.set(enforce)
    t2 = agent_mod._BOUND_VIOLATIONS.set(recorded)
    try:
        yield recorded
    finally:
        agent_mod._BOUND_VIOLATIONS.reset(t2)
        agent_mod._ENFORCE_SCHEMA_BOUNDS.reset(t1)


# ------------------------------------------------------------- validator level


class TestDefaultIsAdvisoryButMeasured:
    def test_oversized_output_passes_unchanged_and_its_path_is_recorded(self):
        validator = _validator_for(JUDGE_SCHEMA)
        payload = _twelve()
        with _run_scope(enforce=False) as recorded:
            assert validator(object(), payload) == payload
        assert recorded == ["/off_target"]
        assert len(payload["off_target"]) == 12  # never truncated

    def test_a_retry_that_lands_in_bounds_clears_the_record(self):
        validator = _validator_for(JUDGE_SCHEMA)
        with _run_scope(enforce=False) as recorded:
            validator(object(), _twelve())
            validator(object(), _in_bounds())
        assert recorded == []


class TestOptInRetriesWithBoundFeedback:
    def test_twelve_items_against_max_items_ten_is_a_bound_retry(self):
        """MUTATION: build the post-hoc validator from the STRIPPED schema
        (`Draft202012Validator(effective_grammar)` for the advisory one) ⇒ the
        stripped schema has no maxItems ⇒ nothing raises ⇒ red."""
        validator = _validator_for(JUDGE_SCHEMA)
        payload = _twelve()
        with _run_scope(enforce=True) as recorded, pytest.raises(SchemaBoundRetry) as info:
            validator(object(), payload)
        err = info.value
        assert isinstance(err, ModelRetry)
        assert [(v["path"], v["keyword"], v["limit"]) for v in err.violations] == [
            ("/off_target", "maxItems", 10)
        ]
        assert "off_target must have at most 10 items" in err.message
        assert "/off_target" in err.message
        assert recorded == ["/off_target"]
        assert len(payload["off_target"]) == 12

    def test_in_bounds_output_passes(self):
        validator = _validator_for(JUDGE_SCHEMA)
        with _run_scope(enforce=True) as recorded:
            assert validator(object(), _in_bounds()) == _in_bounds()
        assert recorded == []

    def test_nested_max_length_names_the_item_path(self):
        validator = _validator_for(JUDGE_SCHEMA)
        with _run_scope(enforce=True), pytest.raises(SchemaBoundRetry) as info:
            validator(object(), {"off_target": ["ok", "x" * 300]})
        assert info.value.violations[0]["path"] == "/off_target/1"
        assert info.value.violations[0]["keyword"] == "maxLength"
        assert "off_target.1 must have at most 256 characters" in info.value.message

    def test_non_bound_stripped_keyword_stays_advisory_even_when_opted_in(self):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "maximum": 5}},
            "required": ["n"],
        }
        validator = _validator_for(schema)
        with _run_scope(enforce=True):
            assert validator(object(), {"n": 12}) == {"n": 12}

    def test_the_terminal_error_is_not_a_model_retry(self):
        assert not issubclass(SchemaBoundViolationError, ModelRetry)
        assert issubclass(SchemaBoundRetry, ModelRetry)

    def test_bound_keywords_are_all_stripped_from_the_grammar(self):
        assert BOUND_KEYWORDS <= _XGRAMMAR_UNSUPPORTED_KEYS
        assert "maxItems" not in json.dumps(strip_xgrammar_unsupported(JUDGE_SCHEMA))


# ------------------------------------------------------------------ real agent


def _ir_json_response(content: dict[str, Any]):
    def _respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-bounds",
                "object": "chat.completion",
                "created": 0,
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )

    return _respond


def _real_agent(content: dict[str, Any]) -> Any:
    """A REAL PydanticAIAgent (real pydantic-ai Agent + OpenAI SDK), stub transport."""
    from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

    agent = create_pydantic_ai_agent(
        name="bounds_probe",
        description="d",
        model_provider="inference_router",
        model="stub-model",
        base_url="http://inference-router.invalid",
        output_schema=JUDGE_SCHEMA,
    )
    agent.agent.model.client._client._transport = httpx.MockTransport(_ir_json_response(content))
    return agent


@pytest.mark.asyncio
async def test_real_agent_default_returns_oversized_output_with_bound_violations():
    agent = _real_agent(_twelve())
    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        result = await agent.run("judge this")
    assert result.output == _twelve()
    um = result.usage_metrics
    assert um.grammar_stripped_keywords == ["maxItems", "maxLength"]
    assert um.schema_bounds_enforced is False
    assert um.bound_violations == ["/off_target"]


@pytest.mark.asyncio
async def test_real_agent_in_bounds_reports_an_empty_violation_list():
    """MUTATION: drop the stamp in `_attach_sampling_provenance` ⇒ None ⇒ red."""
    agent = _real_agent(_in_bounds())
    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        result = await agent.run("judge this")
    assert result.usage_metrics.bound_violations == []
    assert result.usage_metrics.grammar_stripped_keywords == ["maxItems", "maxLength"]


@pytest.mark.asyncio
async def test_real_agent_opt_in_raises_the_typed_error():
    agent = _real_agent(_twelve())
    with (
        patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
        pytest.raises(SchemaBoundViolationError) as info,
    ):
        await agent.run("judge this", enforce_schema_bounds=True)
    assert info.value.violations[0]["keyword"] == "maxItems"


@pytest.mark.asyncio
async def test_the_flag_does_not_leak_past_its_run():
    agent = _real_agent(_twelve())
    with (
        patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
        pytest.raises(SchemaBoundViolationError),
    ):
        await agent.run("judge this", enforce_schema_bounds=True)
    assert agent_mod._ENFORCE_SCHEMA_BOUNDS.get() is False
    assert agent_mod._BOUND_VIOLATIONS.get() is None


# ------------------------------------------------ retry then fail (#358)

# infer.offer_bug_verdict's shape: a defect verdict needs >=1 fix_outline step.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["defect", "false_positive"]},
        "fix_outline": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    },
    "required": ["verdict", "fix_outline"],
}
EMPTY_OUTLINE: dict[str, Any] = {"verdict": "defect", "fix_outline": []}
GOOD_OUTLINE: dict[str, Any] = {"verdict": "defect", "fix_outline": ["guard the None"]}
STRUCTURALLY_INVALID: dict[str, Any] = {"verdict": "defect"}  # required key missing


class _Script:
    """Stub transport: answers request N with ``outputs[N]`` (last one repeats).

    ``tool_mode`` answers with a tool call to whichever output tool the request
    offered (pydantic-ai's tool/function-calling output mode, used by every
    provider except inference_router/mlx); otherwise as message content
    (NativeOutput, inference_router).
    """

    def __init__(self, outputs: list[dict[str, Any]], *, tool_mode: bool = False) -> None:
        self.outputs = outputs
        self.tool_mode = tool_mode
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        content = self.outputs[min(len(self.requests) - 1, len(self.outputs) - 1)]
        if self.tool_mode:
            tool_name = body["tools"][0]["function"]["name"]
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(self.requests)}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(content)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": json.dumps(content)}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "stub-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )

    def retry_prompts(self) -> list[str]:
        """Every message the agent sent AFTER the first request (the feedback)."""
        return [
            json.dumps(r["messages"][len(self.requests[0]["messages"]) :])
            for r in self.requests[1:]
        ]


def _scripted_agent(
    outputs: list[dict[str, Any]], *, output_retries: int | None, provider: str = "inference_router"
) -> tuple[Any, _Script]:
    from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

    script = _Script(outputs, tool_mode=provider != "inference_router")
    agent = create_pydantic_ai_agent(
        name="verdict_probe",
        description="d",
        model_provider=provider,
        model="stub-model",
        base_url="http://inference-router.invalid/v1",
        output_schema=VERDICT_SCHEMA,
        output_retries=output_retries,
    )
    agent.agent.model.client._client._transport = httpx.MockTransport(script)
    return agent, script


def _counted(metrics: MagicMock) -> list[str]:
    return [
        c.kwargs["error_type"] for c in metrics.record_aa_prompt_validation_failure.call_args_list
    ]


# vllm: a local provider on tool/function-calling output mode (a different
# pydantic-ai retry path from inference_router's NativeOutput).
PROVIDERS = ["inference_router", "vllm"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_default_still_returns_the_violating_output_without_retrying(provider):
    """(a) Not opted in ⇒ advisory exactly as before: no retry, output returned."""
    agent, script = _scripted_agent(
        [EMPTY_OUTLINE, GOOD_OUTLINE], output_retries=2, provider=provider
    )
    metrics = MagicMock()
    with patch(f"{MODULE}.get_metrics_client", return_value=metrics):
        result = await agent.run("verdict?")
    assert result.output == EMPTY_OUTLINE
    assert len(script.requests) == 1
    assert result.usage_metrics.bound_violations == ["/fix_outline"]
    assert "schema_bound_retry" not in _counted(metrics)
    assert "schema_soft_violation" in _counted(metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_opt_in_retry_that_satisfies_the_bound_succeeds(provider):
    """(b) MUTATION: raise SchemaBoundViolationError straight away (the old
    opt-in) ⇒ the run fails on attempt 1 ⇒ red."""
    agent, script = _scripted_agent(
        [EMPTY_OUTLINE, GOOD_OUTLINE], output_retries=2, provider=provider
    )
    metrics = MagicMock()
    with patch(f"{MODULE}.get_metrics_client", return_value=metrics):
        result = await agent.run("verdict?", enforce_schema_bounds=True)
    assert result.output == GOOD_OUTLINE
    assert len(script.requests) == 2
    assert _counted(metrics).count("schema_bound_retry") == 1
    assert "schema_bound_violation" not in _counted(metrics)
    # The last attempt was in bounds, so nothing is reported as violating.
    assert result.usage_metrics.bound_violations == []
    assert result.usage_metrics.schema_bounds_enforced is True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("output_retries", [0, 1, 2])
async def test_opt_in_every_output_violates_ends_in_the_typed_error(provider, output_retries):
    """(c) Exactly ``output_retries`` retries, then SchemaBoundViolationError
    with the path — never pydantic-ai's generic UnexpectedModelBehavior.

    MUTATION: drop the ``except UnexpectedModelBehavior`` mapping in
    ``_run_agent_with_timeout`` ⇒ UnexpectedModelBehavior escapes ⇒ red."""
    agent, script = _scripted_agent(
        [EMPTY_OUTLINE], output_retries=output_retries, provider=provider
    )
    metrics = MagicMock()
    with (
        patch(f"{MODULE}.get_metrics_client", return_value=metrics),
        pytest.raises(SchemaBoundViolationError) as info,
    ):
        await agent.run("verdict?", enforce_schema_bounds=True)
    assert len(script.requests) == output_retries + 1
    assert [(v["path"], v["keyword"]) for v in info.value.violations] == [
        ("/fix_outline", "minItems")
    ]
    assert info.value.reason == "schema_bound_violation"
    assert _counted(metrics).count("schema_bound_retry") == output_retries + 1
    assert _counted(metrics).count("schema_bound_violation") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_structural_schema_invalid_exhaustion_is_unchanged(provider):
    """(d) A structural miss is still a schema_invalid ModelRetry, and running
    out of retries on it is still pydantic-ai's UnexpectedModelBehavior — the
    bound mapping must not swallow it, opted in or not."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    for enforce in (False, True):
        agent, script = _scripted_agent([STRUCTURALLY_INVALID], output_retries=1, provider=provider)
        metrics = MagicMock()
        with (
            patch(f"{MODULE}.get_metrics_client", return_value=metrics),
            pytest.raises(UnexpectedModelBehavior),
        ):
            await agent.run("verdict?", enforce_schema_bounds=enforce)
        assert len(script.requests) == 2
        assert _counted(metrics).count("schema_invalid") == 2
        assert "schema_bound_retry" not in _counted(metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_exhaustion_is_blamed_on_the_last_failure_not_an_earlier_bound(provider):
    """A bound retry followed by a STRUCTURAL failure that spends the budget is
    a structural failure: the earlier bound violation must not be reported.

    Guards against a sticky "a bound was violated at some point" flag: only
    the ``__cause__`` of the failure that spent the budget decides the mapping.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    agent, script = _scripted_agent(
        [EMPTY_OUTLINE, STRUCTURALLY_INVALID], output_retries=1, provider=provider
    )
    with (
        patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
        pytest.raises(UnexpectedModelBehavior) as info,
    ):
        await agent.run("verdict?", enforce_schema_bounds=True)
    assert not isinstance(info.value, SchemaBoundViolationError)
    assert len(script.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_retry_feedback_names_the_path_and_the_bound(provider):
    """(e) The model is told WHAT to fix: the path and the bound from the
    ORIGINAL schema (the grammar it decoded under has no minItems at all)."""
    agent, script = _scripted_agent(
        [EMPTY_OUTLINE, GOOD_OUTLINE], output_retries=1, provider=provider
    )
    with patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()):
        await agent.run("verdict?", enforce_schema_bounds=True)
    [feedback] = script.retry_prompts()
    assert "fix_outline must have at least 1 item" in feedback
    assert "/fix_outline" in feedback
    assert "minItems=1" in feedback
    # And the decoder really was never given the bound.
    assert "minItems" not in json.dumps(script.requests[0].get("response_format", {}))
    assert "minItems" not in json.dumps(script.requests[0].get("tools", []))


# -------------------------------------------------------------------- endpoint


@pytest.fixture
def endpoint_client():
    """TestClient whose worker cache hands back a caller-supplied worker.

    Yields (client, set_worker, set_config, add_state).
    """
    from fastapi.testclient import TestClient

    from src.api.worker_management_api import app
    from src.auth import get_tenant_id

    mp = "src.api.workers.message_endpoints"
    worker_record: dict[str, Any] = {"worker_id": "wid", "worker_name": "judge", "is_agent": True}
    with (
        patch(f"{mp}.worker_registry") as mock_registry,
        patch(f"{mp}.worker_cache") as mock_cache,
        patch(f"{mp}.StateDatabase") as mock_state_db,
        patch(f"{mp}._lookup_tenant_byo_mode", new=AsyncMock(return_value=None)),
        patch(f"{mp}._backoff_delay", return_value=0.0),
        patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
    ):
        mock_cache.remove_instance = AsyncMock(return_value=None)
        mock_registry.get_worker = AsyncMock(side_effect=lambda *_a, **_k: dict(worker_record))
        add_state = AsyncMock(return_value=None)
        mock_state_db.return_value.add_worker_state = add_state

        def set_worker(worker: Any) -> None:
            mock_cache.get_or_create_worker = AsyncMock(return_value=worker)

        def set_config(config: dict[str, Any]) -> None:
            worker_record["config"] = config

        app.dependency_overrides[get_tenant_id] = lambda: "test-tenant"

        @asynccontextmanager
        async def _fake_lifespan(_a):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _fake_lifespan
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, set_worker, set_config, add_state
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original


def test_rm_contract_default_oversized_reply_is_200_with_bound_violations(endpoint_client):
    """RM principle_consistency's contract: WM does not reject an oversized reply.

    MUTATION: default enforcement ON (`_ENFORCE_SCHEMA_BOUNDS` default True, or
    `_effective_enforce_schema_bounds` returning True when unset) ⇒ 422 ⇒ red.
    """
    client, set_worker, _set_config, _ = endpoint_client
    set_worker(_real_agent(_twelve()))

    resp = client.post("/workers/judge/message", json={"message": "judge this"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output"] == _twelve()
    assert body["usage_metrics"]["bound_violations"] == ["/off_target"]
    assert body["usage_metrics"]["schema_bounds_enforced"] is False


def test_opt_in_oversized_reply_is_422_schema_bound_violation(endpoint_client):
    client, set_worker, _set_config, add_state = endpoint_client
    set_worker(_real_agent(_twelve()))

    resp = client.post(
        "/workers/judge/message",
        json={"message": "judge this", "enforce_schema_bounds": True},
    )

    assert resp.status_code == 422, resp.text
    assert "SCHEMA_BOUND_VIOLATION" in resp.text
    assert "/off_target" in resp.text
    error_rows = [
        c.kwargs["worker_state"]
        for c in add_state.call_args_list
        if c.kwargs.get("state_type") == "error"
    ]
    assert len(error_rows) == 1
    assert error_rows[0]["error_type"] == "schema_bound_violation"
    assert error_rows[0]["violations"][0]["path"] == "/off_target"


def test_worker_config_opt_in_applies_when_the_message_is_silent(endpoint_client):
    client, set_worker, set_config, _ = endpoint_client
    set_worker(_real_agent(_twelve()))
    set_config({"enforce_schema_bounds": True})

    resp = client.post("/workers/judge/message", json={"message": "judge this"})

    assert resp.status_code == 422, resp.text


def test_message_value_wins_over_worker_config(endpoint_client):
    client, set_worker, set_config, _ = endpoint_client
    set_worker(_real_agent(_twelve()))
    set_config({"enforce_schema_bounds": True})

    resp = client.post(
        "/workers/judge/message",
        json={"message": "judge this", "enforce_schema_bounds": False},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["usage_metrics"]["bound_violations"] == ["/off_target"]


def test_opt_in_retry_then_422_through_the_endpoint(endpoint_client):
    """End to end: retries spent on a bound ⇒ the same 422 + JSON path as before."""
    client, set_worker, _set_config, add_state = endpoint_client
    agent, script = _scripted_agent([EMPTY_OUTLINE], output_retries=2)
    set_worker(agent)

    resp = client.post(
        "/workers/judge/message",
        json={"message": "verdict?", "enforce_schema_bounds": True},
    )

    assert resp.status_code == 422, resp.text
    assert "SCHEMA_BOUND_VIOLATION" in resp.text
    assert "/fix_outline" in resp.text
    assert len(script.requests) == 3
    error_rows = [
        c.kwargs["worker_state"]
        for c in add_state.call_args_list
        if c.kwargs.get("state_type") == "error"
    ]
    assert len(error_rows) == 1
    assert error_rows[0]["error_type"] == "schema_bound_violation"
    assert error_rows[0]["violations"][0]["path"] == "/fix_outline"


def test_opt_in_retry_that_lands_in_bounds_is_200_through_the_endpoint(endpoint_client):
    client, set_worker, _set_config, _ = endpoint_client
    agent, script = _scripted_agent([EMPTY_OUTLINE, GOOD_OUTLINE], output_retries=2)
    set_worker(agent)

    resp = client.post(
        "/workers/judge/message",
        json={"message": "verdict?", "enforce_schema_bounds": True},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["output"] == GOOD_OUTLINE
    assert len(script.requests) == 2


@pytest.mark.asyncio
async def test_mcp_recovery_rerun_spending_its_budget_on_a_bound_is_the_typed_error():
    """The MCP-recovery re-run shares the main path's mapping: a re-run that
    spends its output-retry budget on a bound violation surfaces the typed
    SchemaBoundViolationError (→ 422), not pydantic-ai's generic error swallowed
    into ``None``.

    MUTATION: drop the ``except UnexpectedModelBehavior`` arm in
    ``_handle_mcp_error_recovery`` ⇒ the generic arm returns None ⇒ red."""
    from src.worker.agents import pydantic_ai_agent as mod

    agent, script = _scripted_agent([EMPTY_OUTLINE], output_retries=0)
    agent.config.mcp_servers = [{"name": "stub"}]

    async def _ready() -> None:
        agent.mcp_servers_ready = True

    token = mod._ENFORCE_SCHEMA_BOUNDS.set(True)
    try:
        with (
            patch.object(agent, "_initialize_mcp_servers", _ready),
            patch(f"{MODULE}.get_metrics_client", return_value=MagicMock()),
            pytest.raises(SchemaBoundViolationError) as info,
        ):
            await agent._handle_mcp_error_recovery(
                RuntimeError("MCP server went away"), "verdict?", "verdict_probe"
            )
    finally:
        mod._ENFORCE_SCHEMA_BOUNDS.reset(token)
    assert len(script.requests) == 1
    assert [(v["path"], v["keyword"]) for v in info.value.violations] == [
        ("/fix_outline", "minItems")
    ]
