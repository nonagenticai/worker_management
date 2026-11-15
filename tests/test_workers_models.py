"""Unit tests for src.api.workers.models response models."""

from __future__ import annotations

from typing import Any, cast


def test_worker_response_has_roundtrip_fields():
    from src.api.workers.models import WorkerResponse

    response = WorkerResponse(
        id=1,
        name="test",
        status="ready",
        db_path=None,
        is_agent=True,
        agent_type="pydantic_ai",
        model_provider="openai",
        model="gpt-4",
        system_prompt="You are a helper.",
        instruction=None,
        temperature=0.2,
        tool_names=["calc", "search"],
        output_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        output_validator=None,
        mcp_config=None,
        base_url_override="http://vllm.llm-inference.svc.cluster.local:8000/v1",
        api_key_override=None,
    )
    assert response.temperature == 0.2
    assert response.tool_names == ["calc", "search"]
    assert response.output_schema == {"type": "object", "properties": {"x": {"type": "string"}}}
    assert response.base_url_override is not None
    assert response.base_url_override.startswith("http://vllm")


def test_worker_response_includes_max_tokens():
    from src.api.workers.models import WorkerResponse

    resp = WorkerResponse(
        name="test_worker",
        status="ready",
        is_agent=False,
        max_tokens=6000,
    )
    assert resp.max_tokens == 6000


def test_worker_response_max_tokens_defaults_to_none():
    from src.api.workers.models import WorkerResponse

    resp = WorkerResponse(
        name="test_worker",
        status="ready",
        is_agent=False,
    )
    assert resp.max_tokens is None


def test_worker_create_request_accepts_output_retries():
    from src.api.workers.models import WorkerCreateRequest

    req = WorkerCreateRequest(worker_name="w", output_retries=2)
    assert req.output_retries == 2


def test_worker_create_request_output_retries_defaults_to_none():
    from src.api.workers.models import WorkerCreateRequest

    req = WorkerCreateRequest(worker_name="w")
    assert req.output_retries is None


def test_worker_create_request_output_retries_zero_is_valid():
    from src.api.workers.models import WorkerCreateRequest

    req = WorkerCreateRequest(worker_name="w", output_retries=0)
    assert req.output_retries == 0


def test_worker_create_request_rejects_negative_output_retries():
    import pytest

    from src.api.workers.models import WorkerCreateRequest

    with pytest.raises(Exception):
        WorkerCreateRequest(worker_name="w", output_retries=-1)


def test_worker_create_request_rejects_output_retries_above_10():
    import pytest

    from src.api.workers.models import WorkerCreateRequest

    with pytest.raises(Exception):
        WorkerCreateRequest(worker_name="w", output_retries=11)


def test_worker_response_includes_output_retries():
    from src.api.workers.models import WorkerResponse

    resp = WorkerResponse(
        name="test_worker",
        status="ready",
        is_agent=True,
        output_retries=3,
    )
    assert resp.output_retries == 3


def test_message_request_rejects_unknown_fields():
    """Unknown fields should raise ValidationError, not be silently dropped."""
    import pytest
    from pydantic import ValidationError

    from src.api.workers.models import MessageRequest

    with pytest.raises(ValidationError) as exc_info:
        MessageRequest(
            message="hello",
            # obsolete field from karma's old body
            **cast("dict[str, Any]", {"disable_security_validation": True}),
        )
    # The error should mention either the field name or "Extra inputs".
    error_text = str(exc_info.value)
    assert (
        "disable_security_validation" in error_text
        or "Extra" in error_text
        or "extra" in error_text
    )


def test_message_request_accepts_known_fields_only():
    """Sanity: known fields still validate successfully."""
    from src.api.workers.models import MessageRequest

    req = MessageRequest(
        message="hello",
        system_prompt="you are helpful",
    )
    assert req.message == "hello"
    assert req.system_prompt == "you are helpful"


def test_message_request_rejects_stale_karma_fields():
    """Regression guard: disable_security_validation is still rejected.
    (max_tokens and temperature are now supported per-message fields — see
    test_message_request_accepts_max_tokens and the per-message temperature tests
    in test_ir_passthrough_sampling_correlation — so they are no longer here.
    temperature=1 is a valid in-bounds value; 1.5 is still refused, by bounds.)"""
    import pytest
    from pydantic import ValidationError

    from src.api.workers.models import MessageRequest

    with pytest.raises(ValidationError):
        MessageRequest(message="hello", temperature=1.5)
    for stale_field in ("disable_security_validation",):
        with pytest.raises(ValidationError):
            # Intentional bad fields — verifies pydantic extra="forbid".
            MessageRequest(message="hello", **cast("dict[str, Any]", {stale_field: 1}))


def test_message_request_accepts_max_tokens():
    """max_tokens is a per-message override (folded into worker config at message
    time), so it must be accepted despite extra="forbid", and bounded 1..16384."""
    import pytest
    from pydantic import ValidationError

    from src.api.workers.models import MessageRequest

    assert MessageRequest(message="hi", max_tokens=4000).max_tokens == 4000
    # Omitted ⇒ None (keep the worker record's value / provider default).
    assert MessageRequest(message="hi").max_tokens is None
    for bad in (0, 16385):
        with pytest.raises(ValidationError):
            MessageRequest(message="hi", max_tokens=bad)
