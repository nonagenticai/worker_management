"""Shared exception classification for the worker_management code paths.

The pydantic-graph FSM node and the PydanticAIAgent both surface errors
from the inference router. IR PR #41 tags responses with the
``X-IR-Error-Classification`` header so downstream services can pivot
logs/metrics on the cause. This module is the single place that knows
how to map an exception object to that vocabulary, plus how to extract
the rest of the structured detail IR ships alongside it.

Kept dependency-light on purpose so any worker module can import it
without dragging in graph/agent code.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import openai

_MAX_BODY_CHARS = 2048
# Links below the top-level exception searched for IR's classification header.
_CAUSE_CHAIN_DEPTH = 3


def classify_call_error(exc: BaseException) -> str:
    """Label a provider-call exception with IR's ``X-IR-Error-Classification`` vocabulary.

    An explicit header on ``exc.response`` (set by IR on its own 5xx)
    wins over local heuristics so the cause IR already determined is
    preserved end-to-end.

    pydantic-ai's ``ModelHTTPError`` carries no ``.response``; IR's header sits
    on the ``openai.APIStatusError`` it was raised ``from``, so the
    ``__cause__``/``__context__`` chain is walked (bounded, cycle-guarded).
    """
    seen: set[int] = set()
    cursor: BaseException | None = exc
    depth = 0
    while cursor is not None and depth <= _CAUSE_CHAIN_DEPTH and id(cursor) not in seen:
        seen.add(id(cursor))
        response = getattr(cursor, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                label = headers.get("x-ir-error-classification")
                if isinstance(label, str) and label:
                    return label
        cursor = cursor.__cause__ or cursor.__context__
        depth += 1
    if isinstance(exc, openai.APITimeoutError | httpx.ReadTimeout | httpx.PoolTimeout):
        return "upstream_timeout"
    if isinstance(
        exc,
        openai.APIConnectionError
        | httpx.ConnectError
        | httpx.RemoteProtocolError
        | httpx.ReadError
        | httpx.WriteError,
    ):
        return "transient_network"
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 500 <= status < 600:
            return "upstream_5xx"
        if isinstance(status, int) and 400 <= status < 500:
            return "upstream_4xx"
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int) and 500 <= status < 600:
            return "upstream_5xx"
        if isinstance(status, int) and 400 <= status < 500:
            return "upstream_4xx"
    if isinstance(exc, TimeoutError):
        return "upstream_timeout"
    return "unknown"


def _coerce_status_code(response: Any) -> int | None:
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _coerce_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return value if isinstance(value, str) and value else None


def _coerce_body(response: Any) -> str | None:
    """Pull a truncated response body off ``response`` without raising.

    httpx and openai's ``APIStatusError.response`` both expose ``.text``
    once the body has been read. If it has not (or is binary), we fall
    back to ``str(response)``. We never re-issue a read on the response
    here — IR has already had its chance to surface the body before this
    exception was thrown.
    """
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text:
        return None
    body = text[:_MAX_BODY_CHARS]
    content_type = ""
    headers = getattr(response, "headers", None)
    if headers is not None:
        ct = headers.get("content-type")
        if isinstance(ct, str):
            content_type = ct
    if "json" in content_type.lower():
        try:
            parsed = json.loads(body)
        except ValueError, TypeError:
            return body
        # Keep dicts/lists json-encoded so downstream log renderers can decide
        # whether to keep them as nested fields. Compact form keeps under the cap.
        return json.dumps(parsed, separators=(",", ":"))[:_MAX_BODY_CHARS]
    return body


def extract_error_detail(exc: BaseException) -> dict[str, Any]:
    """Return a flat dict of structured error fields safe for logging + OTel attrs.

    All values are JSON-serialisable scalars (``str`` / ``int`` / ``None``)
    so the dict can be splatted into both ``logger.error(..., extra=...)``
    and OTel metric attribute dicts without further massaging.
    """
    response = getattr(exc, "response", None)
    return {
        "exc_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "error_classification": classify_call_error(exc),
        "status_code": _coerce_status_code(response),
        "ir_request_id": _coerce_header(response, "x-ir-request-id"),
        "upstream_provider": _coerce_header(response, "x-ir-upstream-provider"),
        "response_body": _coerce_body(response),
    }
