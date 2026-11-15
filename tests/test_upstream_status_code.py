"""Unit tests for `_upstream_status_code` — reading a status off the exception.

Issue #212. `send_message`'s error path had no 503 branch, and
`_is_connection_error` matches five literal substrings, none of which appears in
a `ModelHTTPError` repr. So a clean upstream 503 — a capacity signal a caller can
back off on — fell through to `HTTPException(500)`, which means "this service is
broken". In one hour of the 2026-08-31 incident, 2,538 of 2,544 failures were
that single class, and downstream `w2v-orchestrator` raised
`TransientUpstreamError: upstream 500` and red-lined four E2E shards pointing at
worker-management rather than at the inference path that was actually down.

The fix reads a structured `int` off the exception instead of adding a sixth
substring. These tests use the REAL exception classes (`pydantic_ai`, `httpx`,
`openai`) rather than stand-ins with a hand-set attribute: the defect WAS an
assumption about where the status lives on the real object, so a fake that is
given a `.status_code` cannot detect it. `test_the_real_modelhttperror_repr...`
pins the fact that made this invisible.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from src.api.workers._helpers import (
    _CAUSE_CHAIN_DEPTH,
    _PASS_THROUGH_UPSTREAM_STATUSES,
    _is_connection_error,
    _is_deterministic_failure,
    _upstream_status_code,
)

# Verbatim production body (`worker_states` where state_type='error', n=15; the
# same string is already pinned in `test_deterministic_failure_classification.py`).
PROD_503_BODY = {"detail": "All providers failed. Last error: pgx-local is DEGRADED"}

# NOT from production: a constructed worst case in which the 503's body happens to
# quote a provider-level connection failure. It is legal for IR to emit this, and it
# is the case where the substring classifier and the structured status DISAGREE.
ADVERSARIAL_503_BODY = {"detail": "All providers failed. Last error: pgx-local: Connection error."}


def _model_http_error(status: int = 503, body: dict | None = None) -> ModelHTTPError:
    return ModelHTTPError(status_code=status, model_name="local-llm", body=body or PROD_503_BODY)


# ===========================================================================
# The gap this closes
# ===========================================================================


def test_the_real_modelhttperror_carries_its_status_as_a_plain_attribute():
    """`ModelHTTPError` has `.status_code` and NO `.response`.

    This is the whole mechanism. The repo's existing `extract_error_detail`
    (`src/worker/errors/classification.py`) reads only `exc.response.status_code`
    and therefore reports `status_code=None` for precisely the class that caused
    the incident.
    """
    exc = _model_http_error()
    assert exc.status_code == 503
    assert getattr(exc, "response", None) is None
    assert _upstream_status_code(exc) == 503


def test_the_existing_substring_classifiers_cannot_see_the_production_503():
    """This is the defect, in two assertions.

    Neither classifier matches, so the production 503 fell through to
    `HTTPException(500)` — 2,538 times in one hour.
    """
    exc = _model_http_error()
    assert _is_deterministic_failure(exc) is False
    assert _is_connection_error(exc) is False
    # ...while the status was on the object the whole time.
    assert _upstream_status_code(exc) == 503


def test_a_503_whose_body_quotes_a_connection_error_is_still_a_503():
    """Where the substring classifier and the structured status DISAGREE.

    `_is_connection_error` matches on `str(exc)`, and `str(ModelHTTPError)`
    embeds the body — so a 503 whose body quotes a provider-level "Connection
    error." trips it and would be answered 502. The upstream said 503. A status
    the upstream actually sent must outrank a substring guess about the message,
    which is why the structured branch is checked FIRST in `message_endpoints`.
    `test_upstream_503_beats_the_connection_error_substring` pins that ordering
    end to end.
    """
    exc = _model_http_error(body=ADVERSARIAL_503_BODY)
    assert _is_connection_error(exc) is True
    assert _upstream_status_code(exc) == 503


# ===========================================================================
# Shapes
# ===========================================================================


def test_reads_status_off_an_openai_api_status_error():
    request = httpx.Request("POST", "https://vllm.example.com:8001/v1/chat/completions")
    response = httpx.Response(503, request=request, json={"detail": "overloaded"})
    exc = openai.APIStatusError("overloaded", response=response, body=None)
    assert _upstream_status_code(exc) == 503


def test_reads_status_off_an_httpx_http_status_error():
    """`httpx.HTTPStatusError` keeps the status on `.response`, not on itself."""
    request = httpx.Request("GET", "https://vllm.example.com:8001/health")
    response = httpx.Response(503, request=request)
    exc = httpx.HTTPStatusError("503", request=request, response=response)
    assert getattr(exc, "status_code", None) is None
    assert _upstream_status_code(exc) == 503


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 400, 404])
def test_returns_whatever_status_the_upstream_sent(status: int):
    """Reading is separate from acting. The helper reports; the endpoint decides."""
    assert _upstream_status_code(_model_http_error(status)) == status


# ===========================================================================
# Absence, and things that must NOT be mistaken for a status
# ===========================================================================


def test_an_exception_with_no_status_returns_none():
    """None keeps every pre-existing branch in charge — the fail-safe direction."""
    assert _upstream_status_code(RuntimeError("Connection error.")) is None
    assert _upstream_status_code(ValueError("")) is None


def test_a_bool_is_not_a_status():
    """`bool` is an `int` subclass; `True` must not read as HTTP 1."""

    class Weird(Exception):
        status_code = True

    assert _upstream_status_code(Weird()) is None


@pytest.mark.parametrize("bogus", ["503", None, 0, 99, 600, 1000, -1, 5.03])
def test_out_of_range_and_non_int_values_are_rejected(bogus):
    """A stringy "503" or an out-of-band number is not a status code.

    Coercing here would let an unrelated attribute named `status_code` steer our
    HTTP response.
    """

    class Weird(Exception):
        pass

    exc = Weird()
    exc.status_code = bogus
    assert _upstream_status_code(exc) is None


def test_a_mock_like_attribute_is_not_a_status():
    """Test doubles routinely carry auto-created attributes; none is an int."""
    from unittest.mock import MagicMock

    exc = RuntimeError("boom")
    exc.status_code = MagicMock()
    assert _upstream_status_code(exc) is None


# ===========================================================================
# The cause-chain walk
# ===========================================================================


def test_finds_a_status_one_link_down_the_cause_chain():
    """The retry ladder re-raises, so the status bearer is not always outermost."""
    top = RuntimeError("agent run failed")
    top.__cause__ = _model_http_error()
    assert _upstream_status_code(top) == 503


def test_falls_back_to_context_when_there_is_no_explicit_cause():
    """A bare `raise` inside an `except` sets __context__, not __cause__."""
    top = RuntimeError("agent run failed")
    top.__context__ = _model_http_error()
    assert _upstream_status_code(top) == 503


def test_the_outermost_status_wins_over_a_deeper_one():
    """Nearest to the caller is the most specific verdict."""
    top = _model_http_error(502)
    top.__cause__ = _model_http_error(503)
    assert _upstream_status_code(top) == 502


def test_the_walk_is_capped():
    """An unbounded walk on a long chain is a latency risk on the error path."""
    deep = _model_http_error(503)
    cursor: BaseException = deep
    for _ in range(_CAUSE_CHAIN_DEPTH + 2):
        nxt = RuntimeError("wrapper")
        nxt.__cause__ = cursor
        cursor = nxt
    assert _upstream_status_code(cursor) is None


def test_a_cycle_terminates():
    """A self-referential chain must not hang the error path."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _upstream_status_code(a) is None


# ===========================================================================
# The pass-through set
# ===========================================================================


def test_pass_through_set_is_deliberately_only_503():
    """Widening hands the upstream direct control of our status code.

    If this list grows, the growth must come with the measurement that justified
    it — the same discipline as `_DETERMINISTIC_FAILURE_MARKERS`.
    """
    assert _PASS_THROUGH_UPSTREAM_STATUSES == (503,)
