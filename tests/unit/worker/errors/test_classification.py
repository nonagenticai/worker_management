"""Unit tests for the shared error classification helpers.

These cover the inputs that matter operationally: an IR-tagged
``openai.APIStatusError`` (the hot path), an ``httpx.HTTPStatusError``
without IR header (status-code-based fallback), a network timeout, and
a plain ``ValueError`` (the ``unknown`` floor).
"""

from __future__ import annotations

import httpx
import openai

from src.worker.errors import classify_call_error, extract_error_detail


def _make_response(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> httpx.Response:
    """Build a real httpx.Response so .text / .headers behave like prod."""
    request = httpx.Request("POST", "https://ir.example.com/v1/chat/completions")
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        content=(body or "").encode("utf-8"),
        request=request,
    )


def test_classify_prefers_ir_header_over_status_heuristic():
    response = _make_response(
        500,
        headers={"x-ir-error-classification": "provider_marked_degraded"},
    )
    exc = openai.APIStatusError("ir 500", response=response, body=None)
    # Header wins even though 500 would heuristically map to upstream_5xx.
    assert classify_call_error(exc) == "provider_marked_degraded"


def test_classify_httpx_5xx_without_header_falls_back_to_upstream_5xx():
    response = _make_response(502)
    exc = httpx.HTTPStatusError("bad gateway", request=response.request, response=response)
    assert classify_call_error(exc) == "upstream_5xx"


def test_classify_bare_timeout_error_maps_to_upstream_timeout():
    assert classify_call_error(TimeoutError("timed out")) == "upstream_timeout"


def test_classify_unknown_for_generic_value_error():
    assert classify_call_error(ValueError("boom")) == "unknown"


def test_extract_error_detail_full_fields_from_ir_response():
    response = _make_response(
        502,
        headers={
            "x-ir-error-classification": "upstream_5xx",
            "x-ir-request-id": "req-abc-123",
            "x-ir-upstream-provider": "vllm-pgx",
            "content-type": "application/json",
        },
        body='{"detail":"upstream vllm read timeout","code":"E502"}',
    )
    exc = openai.APIStatusError("ir 502", response=response, body=None)
    detail = extract_error_detail(exc)

    assert detail["error_classification"] == "upstream_5xx"
    assert detail["status_code"] == 502
    assert detail["ir_request_id"] == "req-abc-123"
    assert detail["upstream_provider"] == "vllm-pgx"
    assert detail["exc_type"].endswith("APIStatusError")
    # JSON body should be compact-encoded so OTel attrs stay scalar-friendly.
    assert detail["response_body"] is not None
    assert "upstream vllm read timeout" in detail["response_body"]
    assert "E502" in detail["response_body"]


def test_extract_error_detail_no_response_returns_none_fields():
    detail = extract_error_detail(TimeoutError("slow"))
    assert detail["error_classification"] == "upstream_timeout"
    assert detail["status_code"] is None
    assert detail["ir_request_id"] is None
    assert detail["upstream_provider"] is None
    assert detail["response_body"] is None


def test_extract_error_detail_generic_value_error_is_unknown():
    detail = extract_error_detail(ValueError("boom"))
    assert detail["error_classification"] == "unknown"
    assert detail["status_code"] is None
    assert detail["response_body"] is None


def test_extract_error_detail_truncates_long_response_body():
    big_body = "x" * 5000
    response = _make_response(500, body=big_body)
    exc = httpx.HTTPStatusError("oops", request=response.request, response=response)
    detail = extract_error_detail(exc)
    assert detail["response_body"] is not None
    assert len(detail["response_body"]) <= 2048
