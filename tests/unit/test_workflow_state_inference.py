"""Unit tests for WorkflowState inference telemetry accessors.

These tests exercise the helpers added in Task 1 of the Phase 1
inference-graph plan (``record_inference_attempt``,
``mark_inference_success``, ``set_inference_request``,
``set_validation_result``) and the ``to_dict`` / ``from_dict``
round-trip through the ``framework_state["inference"]`` bag.

All tests use real ``WorkflowState`` instances — no mocks.
"""

from src.worker.graphs.pydantic_graph_state import WorkflowState


def _blank_state() -> WorkflowState:
    """Return a minimal ``WorkflowState`` suitable for unit tests."""
    return WorkflowState(
        session_id="sess-1",
        graph_id="graph-1",
        variables={},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )


class TestInferenceKeyConstant:
    def test_constant_value(self) -> None:
        assert WorkflowState.INFERENCE_KEY == "inference"


class TestRecordInferenceAttempt:
    def test_first_call_initialises_bag_with_defaults(self) -> None:
        state = _blank_state()
        assert "inference" not in state.framework_state

        state.record_inference_attempt(
            provider="Qwen3-4B-AWQ",
            status="ok",
            latency_ms=120,
            tokens_prompt=50,
            tokens_completion=10,
        )

        bag = state.framework_state["inference"]
        assert bag["chosen_provider"] is None
        assert bag["final_response"] is None
        assert bag["validation_passed"] is None
        assert bag["request_payload"] is None
        assert bag["attempts"] == [
            {
                "provider": "Qwen3-4B-AWQ",
                "status": "ok",
                "latency_ms": 120,
                "tokens_prompt": 50,
                "tokens_completion": 10,
                "error": None,
                "error_classification": None,
                "served_model": None,
            }
        ]

    def test_error_attempt_records_error_message(self) -> None:
        state = _blank_state()

        state.record_inference_attempt(
            provider="claude",
            status="error",
            latency_ms=5000,
            error="timeout waiting for response",
        )

        attempts = state.framework_state["inference"]["attempts"]
        assert len(attempts) == 1
        assert attempts[0] == {
            "provider": "claude",
            "status": "error",
            "latency_ms": 5000,
            "tokens_prompt": None,
            "tokens_completion": None,
            "error": "timeout waiting for response",
            "error_classification": None,
            "served_model": None,
        }

    def test_multiple_attempts_accumulate_in_order(self) -> None:
        state = _blank_state()

        state.record_inference_attempt(
            provider="providerA",
            status="error",
            latency_ms=200,
            error="429 rate-limited",
        )
        state.record_inference_attempt(
            provider="providerB",
            status="error",
            latency_ms=800,
            error="500 internal server error",
        )
        state.record_inference_attempt(
            provider="providerC",
            status="ok",
            latency_ms=450,
            tokens_prompt=7,
            tokens_completion=3,
        )

        attempts = state.framework_state["inference"]["attempts"]
        assert [a["provider"] for a in attempts] == [
            "providerA",
            "providerB",
            "providerC",
        ]
        assert [a["status"] for a in attempts] == ["error", "error", "ok"]
        assert attempts[0]["error"] == "429 rate-limited"
        assert attempts[1]["error"] == "500 internal server error"
        assert attempts[2]["tokens_prompt"] == 7
        assert attempts[2]["tokens_completion"] == 3


class TestMarkInferenceSuccess:
    def test_sets_chosen_provider_and_final_response(self) -> None:
        state = _blank_state()

        state.mark_inference_success(
            provider="providerB",
            response={"x": 1},
            tokens_prompt=10,
            tokens_completion=20,
        )

        bag = state.framework_state["inference"]
        assert bag["chosen_provider"] == "providerB"
        assert bag["final_response"] == {"x": 1}
        # Other fields remain at their lazy-init defaults.
        assert bag["attempts"] == []
        assert bag["validation_passed"] is None
        assert bag["request_payload"] is None

    def test_interleaves_cleanly_with_record_attempts(self) -> None:
        state = _blank_state()
        state.record_inference_attempt(
            provider="providerA",
            status="error",
            latency_ms=100,
            error="boom",
        )
        state.record_inference_attempt(
            provider="providerB",
            status="ok",
            latency_ms=150,
            tokens_prompt=4,
            tokens_completion=6,
        )
        state.mark_inference_success(
            provider="providerB",
            response={"text": "hello"},
            tokens_prompt=4,
            tokens_completion=6,
        )

        bag = state.framework_state["inference"]
        assert bag["chosen_provider"] == "providerB"
        assert bag["final_response"] == {"text": "hello"}
        assert len(bag["attempts"]) == 2
        assert bag["attempts"][-1]["provider"] == "providerB"


class TestSetInferenceRequest:
    def test_sets_request_payload(self) -> None:
        state = _blank_state()

        state.set_inference_request({"foo": "bar"})

        bag = state.framework_state["inference"]
        assert bag["request_payload"] == {"foo": "bar"}
        assert bag["attempts"] == []
        assert bag["chosen_provider"] is None
        assert bag["final_response"] is None
        assert bag["validation_passed"] is None

    def test_overwrites_prior_payload(self) -> None:
        state = _blank_state()
        state.set_inference_request({"first": True})
        state.set_inference_request({"second": True})

        assert state.framework_state["inference"]["request_payload"] == {"second": True}


class TestSetValidationResult:
    def test_true(self) -> None:
        state = _blank_state()
        state.set_validation_result(True)
        assert state.framework_state["inference"]["validation_passed"] is True

    def test_false(self) -> None:
        state = _blank_state()
        state.set_validation_result(False)
        assert state.framework_state["inference"]["validation_passed"] is False


class TestSetQualityScore:
    def test_sets_score_in_bag(self) -> None:
        state = _blank_state()
        state.set_quality_score(0.82)
        assert state.framework_state["inference"]["quality_score"] == 0.82

    def test_overwrites_prior_score(self) -> None:
        state = _blank_state()
        state.set_quality_score(0.5)
        state.set_quality_score(0.91)
        assert state.framework_state["inference"]["quality_score"] == 0.91

    def test_round_trips_through_dict(self) -> None:
        state = _blank_state()
        state.set_quality_score(0.73)
        restored = WorkflowState.from_dict(state.to_dict())
        assert restored.framework_state["inference"]["quality_score"] == 0.73


class TestInferenceBagRoundTrip:
    def test_round_trip_preserves_full_bag(self) -> None:
        state = _blank_state()

        request_payload = {
            "model": "local-llm",
            "messages": [{"role": "user", "content": "hi"}],
        }
        state.set_inference_request(request_payload)
        state.record_inference_attempt(
            provider="Qwen3-4B-AWQ",
            status="error",
            latency_ms=300,
            error="connection refused",
        )
        state.record_inference_attempt(
            provider="vllm",
            status="ok",
            latency_ms=420,
            tokens_prompt=12,
            tokens_completion=8,
        )
        winning_response = {
            "id": "resp-42",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        }
        state.mark_inference_success(
            provider="vllm",
            response=winning_response,
            tokens_prompt=12,
            tokens_completion=8,
        )
        state.set_validation_result(True)

        original_bag = state.framework_state["inference"]

        serialised = state.to_dict()
        restored = WorkflowState.from_dict(serialised)
        restored_bag = restored.framework_state["inference"]

        # Exact-equality invariant: the reconstructed bag must match the
        # original byte-for-byte (for JSON-serialisable values).
        assert restored_bag == original_bag

        # Spot-check each field individually to protect against the
        # dict-equality check silently comparing two dicts that both mutated
        # in lockstep.
        assert restored_bag["request_payload"] == request_payload
        assert restored_bag["chosen_provider"] == "vllm"
        assert restored_bag["final_response"] == winning_response
        assert restored_bag["validation_passed"] is True
        assert len(restored_bag["attempts"]) == 2
        assert restored_bag["attempts"][0] == {
            "provider": "Qwen3-4B-AWQ",
            "status": "error",
            "latency_ms": 300,
            "tokens_prompt": None,
            "tokens_completion": None,
            "error": "connection refused",
            "error_classification": None,
            "served_model": None,
        }
        assert restored_bag["attempts"][1] == {
            "provider": "vllm",
            "status": "ok",
            "latency_ms": 420,
            "tokens_prompt": 12,
            "tokens_completion": 8,
            "error": None,
            "error_classification": None,
            "served_model": None,
        }

    def test_round_trip_when_bag_never_initialised(self) -> None:
        """State without any inference calls must round-trip without adding the key."""
        state = _blank_state()

        serialised = state.to_dict()
        restored = WorkflowState.from_dict(serialised)

        assert "inference" not in restored.framework_state
