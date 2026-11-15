"""Tests for the allow_prompt_constructs flag on validate_string_security.

The flag opts a single field out of detect_command_injection (only) so that
legitimate LLM prompts with inline backticks and ${...} references aren't
false-flagged. All other detectors stay active.
"""

from __future__ import annotations

import pytest

from src.api.security_validation import (
    SecurityValidationError,
    validate_string_security,
)

# --- existing default behaviour stays strict --------------------------


def test_default_rejects_backtick_wrapped_text():
    """Without the flag, single-line backticks are still rejected — the
    pre-existing behaviour is unchanged for non-prompt fields."""
    with pytest.raises(SecurityValidationError, match="command injection"):
        validate_string_security(
            "Pass the `worker_name` arg",
            field_name="user_input",
        )


def test_default_rejects_dollar_paren():
    with pytest.raises(SecurityValidationError, match="command injection"):
        validate_string_security("foo $(rm -rf /) bar", field_name="user_input")


# --- new flag carves out prompt fields --------------------------------


def test_prompt_constructs_allow_inline_backticks():
    """Prompt-class fields should accept inline backticks for code spans."""
    text = (
        "Each module is an object with `title`, `description`, and a "
        "`lessons` array; each lesson is an object with `title`, `topic`, "
        "`description`, `video_type`, and `target_duration_seconds`."
    )
    result = validate_string_security(
        text,
        field_name="system_prompt",
        max_length=5000,
        allow_prompt_constructs=True,
    )
    assert result == text


def test_prompt_constructs_still_block_sql_injection():
    """The opt-out is narrowly scoped to detect_command_injection; SQL
    injection patterns must still be rejected even on prompt fields."""
    with pytest.raises(SecurityValidationError, match="SQL injection"):
        validate_string_security(
            "Output: ' OR '1'='1' --",
            field_name="system_prompt",
            allow_prompt_constructs=True,
        )


def test_prompt_constructs_still_block_xss():
    with pytest.raises(SecurityValidationError, match="XSS"):
        validate_string_security(
            "Reply with <script>alert(1)</script>",
            field_name="system_prompt",
            allow_prompt_constructs=True,
        )


def test_prompt_constructs_still_enforce_max_length():
    with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
        validate_string_security(
            "x" * 6000,
            field_name="system_prompt",
            max_length=5000,
            allow_prompt_constructs=True,
        )


def test_prompt_constructs_still_block_null_bytes():
    with pytest.raises(SecurityValidationError, match="null bytes"):
        validate_string_security(
            "Hello\x00world",
            field_name="system_prompt",
            allow_prompt_constructs=True,
        )


# --- model-level regression: course-curriculum prompt -----------------


def test_course_curriculum_prompt_with_backticks_validates():
    """Regression test for the 2026-04-30 PUT-worker run: the original
    course-curriculum-generator prompt uses backticks around field names
    and previously raised 422. With allow_prompt_constructs=True on the
    system_prompt field, it must validate cleanly through
    WorkerCreateRequest."""
    from src.api.workers.models import WorkerCreateRequest

    payload = {
        "worker_name": "test-curriculum",
        "agent_type": "pydantic_ai",
        "agent_role": "Curriculum architect",
        "model_provider": "inference_router",
        "model": "deepseek",
        "temperature": 0.1,
        "system_prompt": (
            "You are a course curriculum architect. Given a website analysis, "
            "you produce a structured course as a single JSON object with keys "
            '"title" (string), "description" (string), and "modules" (array). '
            "Each module is an object with `title`, `description`, and a "
            "`lessons` array; each lesson is an object with `title`, `topic`, "
            "`description`, `video_type`, and `target_duration_seconds`. "
            "Output ONLY the JSON object — no prose, no markdown fences."
        ),
    }
    # Should not raise
    req = WorkerCreateRequest.model_validate(payload)
    assert req.system_prompt is not None
    assert req.system_prompt.startswith("You are a course curriculum")
