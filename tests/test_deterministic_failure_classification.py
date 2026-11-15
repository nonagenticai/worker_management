"""Unit tests for the terminal-failure retryability classifier.

Context: callers retry on 5xx and not on 4xx (the n8n Workers node gates on
`statusCode >= 500`). Before this classifier existed, `send_message` returned 500
for EVERY terminal failure, so a request that could never succeed was re-run up to
MAX_RETRIES times, multiplying billed GPU decodes for a guaranteed failure.

The strings asserted below are verbatim production values taken from
`worker_states` where `state_type='error'` over a 30-day window (n=3645). The
counts in the docstrings are that measurement, not an estimate.
"""

from __future__ import annotations

import pytest

from src.api.workers._helpers import _is_deterministic_failure

# Verbatim production error strings that MUST be classified non-retryable.
# Together these were 3347 of 3645 terminal errors (91.8%).
DETERMINISTIC_PRODUCTION_ERRORS = [
    # n=1909
    "Model token limit (2500) exceeded before any response was generated. "
    "Increase the `max_tokens` model setting.",
    # n=1081
    "Model token limit (4000) exceeded before any response was generated. "
    "Increase the `max_tokens` model setting.",
    # n=69 / 34 / 32 / 6 / 2 / 2 / 1 across these limits
    "Model token limit (2000) exceeded before any response was generated.",
    "Model token limit (6000) exceeded before any response was generated.",
    "Model token limit (1024) exceeded before any response was generated.",
    "Model token limit (16384) exceeded before any response was generated.",
    "Model token limit (12000) exceeded before any response was generated.",
    "Model token limit (8192) exceeded before any response was generated.",
    # n=207
    "Exceeded maximum output retries (0)",
    # n=4
    "Exceeded maximum output retries (3)",
]

# Verbatim production error strings that MUST stay retryable (5xx). Together
# these were 298 of 3645 terminal errors (8.2%) and a retry genuinely can rescue
# them, so misclassifying any of these would turn a recoverable blip into a
# hard user-visible failure.
TRANSIENT_PRODUCTION_ERRORS = [
    "Connection error.",  # n=275
    "status_code: 503, model_name: local-llm, body: {'detail': \"All providers "
    'failed. Last error: pgx-local is DEGRADED"}',  # n=15
    "status_code: 503, model_name: local-llm, body: {'detail': 'All providers "
    "failed. Last error: Schema validation failed'}",  # n=3
    "Request timed out.",  # n=3
    "Agent 'worker-expand' timed out",  # n=2
]


@pytest.mark.parametrize("message", DETERMINISTIC_PRODUCTION_ERRORS)
def test_production_deterministic_errors_are_not_retryable(message: str) -> None:
    assert _is_deterministic_failure(Exception(message)) is True


@pytest.mark.parametrize("message", TRANSIENT_PRODUCTION_ERRORS)
def test_production_transient_errors_stay_retryable(message: str) -> None:
    assert _is_deterministic_failure(Exception(message)) is False


def test_unrecognised_error_defaults_to_retryable() -> None:
    """Fail-safe: an error we have never seen keeps the old 500 behaviour.

    Under-classifying costs a wasted retry; over-classifying reports a transient
    fault as permanent. The default must therefore be False.
    """
    assert _is_deterministic_failure(Exception("some brand new failure mode")) is False
    assert _is_deterministic_failure(Exception("")) is False


def test_classification_is_case_insensitive() -> None:
    """Backends vary the casing of these messages; the marker match must not."""
    assert _is_deterministic_failure(Exception("EXCEEDED MAXIMUM OUTPUT RETRIES (0)")) is True
    assert (
        _is_deterministic_failure(
            Exception("Model Token Limit (10) Exceeded Before Any Response Was Generated")
        )
        is True
    )


def test_marker_does_not_match_a_merely_similar_message() -> None:
    """Guard the narrowness of the markers.

    A response that WAS generated and then hit a limit is a different failure and
    may well succeed on retry, so it must not be swept up by the token-limit marker.
    """
    assert (
        _is_deterministic_failure(Exception("Model token limit exceeded after a partial response"))
        is False
    )
    assert _is_deterministic_failure(Exception("output retries are configured")) is False
