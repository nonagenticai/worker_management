"""Unit tests for the WorkerNode quality_score producer.

A ``qa_validator`` (or any worker) that returns a numeric ``quality_score`` in
its output dict must have that score lifted into the inference bag, so the
orchestrator finalize path persists it onto
``graph_execution_sessions.quality_score`` (consumed downstream by the metering
exec_quality ETL → governor-effectiveness ``governed_avg_quality``).

These tests cover:

1. ``_normalise_quality_score`` — the pure scale-normalisation helper.
2. ``WorkerNode._maybe_record_quality`` — the guarded lift into the bag,
   including the regression that workers WITHOUT a quality_score are untouched.

No mocks of the score path — real ``WorkflowState`` instances throughout.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.worker.graphs.nodes.worker_node import WorkerNode, _normalise_quality_score
from src.worker.graphs.pydantic_graph_state import WorkflowState

INFERENCE_KEY = WorkflowState.INFERENCE_KEY


def _blank_state() -> WorkflowState:
    return WorkflowState(
        session_id="sess-1",
        graph_id="graph-1",
        variables={},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )


def _node() -> WorkerNode:
    return WorkerNode(node_id="qa-1", worker_name="qa_validator", task_template="validate")


def _record(output: Any) -> WorkflowState:
    """Run the lift against a fresh state and return the state."""
    state = _blank_state()
    ctx = SimpleNamespace(state=state)
    _node()._maybe_record_quality(ctx, output)  # type: ignore[arg-type]
    return state


# ---------------------------------------------------------------------------
# _normalise_quality_score (pure)
# ---------------------------------------------------------------------------


class TestNormaliseQualityScore:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.82, 0.82),  # in-range float
            (0.0, 0.0),  # lower bound
            (1.0, 1.0),  # upper bound
            (0, 0.0),  # int in range
            (1, 1.0),  # int upper bound
            ("0.7", 0.7),  # numeric string
            (95, 0.95),  # percentage rescue (0-100 scale)
            (50.0, 0.5),  # percentage rescue
            (250, 1.0),  # > 100 → clamp high
            (-3, 0.0),  # < 0 → clamp low
        ],
    )
    def test_returns_normalised_score(self, raw: Any, expected: float) -> None:
        assert _normalise_quality_score(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            True,  # bool is not a score (even though bool subclasses int)
            False,
            None,
            "high",  # non-numeric string
            "",
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
            10**400,  # int too large for float() → OverflowError, must not escape
        ],
    )
    def test_returns_none_for_unusable(self, raw: Any) -> None:
        assert _normalise_quality_score(raw) is None


# ---------------------------------------------------------------------------
# WorkerNode._maybe_record_quality (lift into bag)
# ---------------------------------------------------------------------------


class TestMaybeRecordQuality:
    def test_lifts_numeric_score_into_bag(self) -> None:
        state = _record({"validation_passed": True, "quality_score": 0.82, "issues": []})
        assert state.framework_state[INFERENCE_KEY]["quality_score"] == 0.82

    def test_normalises_percentage_score(self) -> None:
        state = _record({"quality_score": 90})
        assert state.framework_state[INFERENCE_KEY]["quality_score"] == 0.9

    def test_output_without_quality_score_is_noop(self) -> None:
        # Regression: non-validator workers (procedure_selector, …) return dicts
        # without quality_score and must leave the bag completely untouched.
        state = _record({"procedure_ids": [1, 2], "rationale": "ok"})
        assert INFERENCE_KEY not in state.framework_state

    def test_non_dict_output_is_noop(self) -> None:
        state = _record("just a string deliverable")
        assert INFERENCE_KEY not in state.framework_state

    def test_non_numeric_quality_score_is_noop(self) -> None:
        state = _record({"quality_score": "excellent"})
        assert INFERENCE_KEY not in state.framework_state

    def test_bool_quality_score_is_noop(self) -> None:
        state = _record({"quality_score": True})
        assert INFERENCE_KEY not in state.framework_state
