"""Unit tests for ``ExecutionResponse`` inference-telemetry fields and
``ExecutionRequest.validate_variables`` env-configurable cap.

Phase 1 inference-graph adds 5 optional fields to ``ExecutionResponse`` and
raises the ``variables`` cap from a hard-coded 10 KB to an env-configurable
limit defaulting to 1 MiB (``WM_GRAPH_VARIABLES_MAX_BYTES``).

The env var is read via a module-level helper (``_variables_max_bytes``)
that re-reads ``os.environ`` on every call, so tests can monkeypatch without
having to reload the module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import ValidationError

from src.worker.graphs.graph_models import ExecutionRequest, ExecutionResponse

# ---------------------------------------------------------------------------
# ExecutionResponse — new inference telemetry fields
# ---------------------------------------------------------------------------


class TestExecutionResponseInferenceFields:
    def test_backward_compat_no_inference_fields(self):
        """Existing callers that omit the 5 new fields must still instantiate.

        The fields default to ``None`` so a legacy payload round-trips
        unchanged.
        """
        now = datetime.now(UTC)
        resp = ExecutionResponse(
            session_id="s-1",
            status="completed",
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
            started_at=now,
            updated_at=now,
        )
        assert resp.chosen_provider is None
        assert resp.fallback_count is None
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None
        assert resp.validation_passed is None

    def test_accepts_all_new_fields_and_serializes(self):
        now = datetime.now(UTC)
        resp = ExecutionResponse(
            session_id="s-1",
            status="completed",
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
            started_at=now,
            updated_at=now,
            chosen_provider="Qwen3-4B-AWQ",
            fallback_count=2,
            prompt_tokens=42,
            completion_tokens=17,
            validation_passed=True,
        )
        dumped = resp.model_dump()
        assert dumped["chosen_provider"] == "Qwen3-4B-AWQ"
        assert dumped["fallback_count"] == 2
        assert dumped["prompt_tokens"] == 42
        assert dumped["completion_tokens"] == 17
        assert dumped["validation_passed"] is True

    def test_validation_passed_false_preserved(self):
        """False must round-trip distinctly from None (not coerced to None)."""
        now = datetime.now(UTC)
        resp = ExecutionResponse(
            session_id="s-1",
            status="failed",
            node_outputs={},
            completed_nodes=[],
            failed_nodes=[],
            started_at=now,
            updated_at=now,
            validation_passed=False,
        )
        assert resp.validation_passed is False

    def test_offline_only_fields_are_not_accepted(self):
        """``request_payload`` and ``final_response`` are offline-only.

        Adding them to ``ExecutionResponse`` must be rejected (``extra="forbid"``).
        """
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ExecutionResponse(
                session_id="s-1",
                status="completed",
                node_outputs={},
                completed_nodes=[],
                failed_nodes=[],
                started_at=now,
                updated_at=now,
                **cast("dict[str, Any]", {"request_payload": {"model": "x"}}),
            )
        with pytest.raises(ValidationError):
            ExecutionResponse(
                session_id="s-1",
                status="completed",
                node_outputs={},
                completed_nodes=[],
                failed_nodes=[],
                started_at=now,
                updated_at=now,
                **cast("dict[str, Any]", {"final_response": {"id": "r-1"}}),
            )


# ---------------------------------------------------------------------------
# ExecutionRequest.validate_variables — env-configurable cap
# ---------------------------------------------------------------------------


class TestExecutionRequestVariablesCap:
    def test_default_cap_accepts_500kb_payload(self, monkeypatch):
        """Default 1 MiB cap easily accommodates a 500 KB payload.

        We clear the env var to exercise the default branch of
        ``_variables_max_bytes``.
        """
        monkeypatch.delenv("WM_GRAPH_VARIABLES_MAX_BYTES", raising=False)

        # Build a ~500 KB payload. ``str({"k": "x" * N})`` is a few bytes
        # longer than ``N``; 500_000 copies is comfortably under 1 MiB.
        big = {"payload": "x" * 500_000}
        req = ExecutionRequest(variables=big)
        assert req.variables == big

    def test_default_cap_rejects_2mb_payload(self, monkeypatch):
        """2 MB exceeds the 1 MiB default; validator must reject."""
        monkeypatch.delenv("WM_GRAPH_VARIABLES_MAX_BYTES", raising=False)

        too_big = {"payload": "x" * (2 * 1024 * 1024)}
        with pytest.raises(ValidationError) as exc_info:
            ExecutionRequest(variables=too_big)
        assert "too large" in str(exc_info.value).lower()

    def test_custom_env_cap_5kb_rejects_10kb_payload(self, monkeypatch):
        """Lowering ``WM_GRAPH_VARIABLES_MAX_BYTES=5000`` rejects a 10 KB payload.

        NOTE: the env var is read by ``_variables_max_bytes()`` on every
        validator call; no module reload is needed for this monkeypatch to
        take effect.
        """
        monkeypatch.setenv("WM_GRAPH_VARIABLES_MAX_BYTES", "5000")

        payload = {"payload": "x" * 10_000}
        with pytest.raises(ValidationError) as exc_info:
            ExecutionRequest(variables=payload)
        assert "5000" in str(exc_info.value) or "too large" in str(exc_info.value).lower()

    def test_invalid_env_var_falls_back_to_default(self, monkeypatch):
        """Garbage env value must fall back to the 1 MiB default."""
        monkeypatch.setenv("WM_GRAPH_VARIABLES_MAX_BYTES", "not-a-number")

        # 500 KB must still be accepted under the fallback 1 MiB limit.
        big = {"payload": "x" * 500_000}
        req = ExecutionRequest(variables=big)
        assert req.variables == big
