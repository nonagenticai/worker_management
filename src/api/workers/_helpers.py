"""Shared utility functions for the Workers API."""

import random

from src.security.redact import redact_secrets_in_text


def _is_connection_error(exc: Exception) -> bool:
    """Check if exception is a connection/DNS error that warrants worker recreation."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in [
            "connection error",
            "name or service not known",
            "connect call failed",
            "no route to host",
            "connection refused",
        ]
    )


#: Substrings that identify a terminal agent failure which CANNOT succeed if the
#: identical request is re-issued. Deliberately narrow: every entry was observed
#: in production (``worker_states`` where ``state_type='error'``, 30-day window,
#: n=3645) and is deterministic in the request, not in the environment. Under-
#: classifying is safe (the caller retries as before); over-classifying is not
#: (a transient fault would be reported as permanent), so add only measured
#: strings, never speculative ones.
_DETERMINISTIC_FAILURE_MARKERS = (
    # "Model token limit (2500) exceeded before any response was generated."
    # The prompt leaves no room for output under this model's max_tokens; the
    # next identical attempt truncates at exactly the same point. n=3136.
    "exceeded before any response was generated",
    # "Exceeded maximum output retries (0)" — the agent already exhausted its own
    # structured-output retry budget. Re-running restarts that same budget and
    # ends the same way. n=211.
    "exceeded maximum output retries",
)


def _is_deterministic_failure(exc: Exception) -> bool:
    """Check whether re-issuing the identical request is provably futile.

    A caller that retries on 5xx (the n8n Workers node does) will otherwise
    re-run the full agent for a failure that cannot change, multiplying billed
    GPU decodes and wall clock by the retry count for a guaranteed failure.

    Args:
        exc: The terminal exception raised by the agent run.

    Returns:
        True if the failure is a property of the request itself and an
        identical retry cannot succeed; False for anything else, including
        every unrecognised error (fail-safe: keep the retryable status).
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _DETERMINISTIC_FAILURE_MARKERS)


def _backoff_delay(
    attempt: int,
    *,
    base_delay_s: float,
    max_delay_s: float,
    jitter_ratio: float = 0.25,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff (seconds) with bounded jitter for LLM-stage retries.

    ``attempt`` is 1-based (first retry = 1). Delay =
    min(base_delay_s * 2**(attempt-1), max_delay_s), then +/- ``jitter_ratio``
    multiplicative jitter, clamped to [0.0, max_delay_s]. Pass a seeded ``rng``
    (or ``jitter_ratio=0.0``) for deterministic tests.

    Args:
        attempt: 1-based retry number (first retry is 1).
        base_delay_s: Base delay before exponential growth.
        max_delay_s: Ceiling for a single sleep after growth and jitter.
        jitter_ratio: Symmetric jitter fraction; 0.0 disables jitter.
        rng: Optional seeded Random for deterministic tests.

    Returns:
        The delay in seconds, clamped to [0.0, max_delay_s].
    """
    capped = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
    rand = random.random if rng is None else rng.random
    jitter = capped * jitter_ratio * (2.0 * rand() - 1.0)
    return max(0.0, min(capped + jitter, max_delay_s))


#: Depth cap for the ``__cause__``/``__context__` walk in
#: :func:`_describe_connection_failure`. Three links reaches the httpx primitive
#: behind an OpenAI-SDK wrapper without risking an unbounded detail string on a
#: deeply chained failure.
_CAUSE_CHAIN_DEPTH = 3


def _describe_connection_failure(exc: BaseException, *, worker_name: str, attempts: int) -> str:
    """Name the hop and the underlying cause of an upstream connection failure.

    ``str(openai.APIConnectionError(...))`` is the fixed literal ``"Connection
    error."`` — no host, no port, no cause. That string reached the n8n caller
    verbatim inside a ``500 INTERNAL_ERROR`` envelope, and it is why the
    2026-08-30 14:49Z outage ran 40 hours unattributed: ``worker-select-asset``
    failed on 60-85% of calls while worker-management, inference-router and the
    vLLM upstream were all healthy with zero restarts, and pod logs retain only
    2-4 hours so the onset was already unreachable.

    The distinguishing information is one link down the ``__cause__`` chain.
    ``ConnectTimeout``/``ConnectError`` mean the upstream is unreachable;
    ``PoolTimeout`` means *our own* client pool is exhausted and the upstream is
    fine; ``ReadError``/``RemoteProtocolError`` mean it accepted and then hung up
    mid-response. Those are opposite fixes, and today's message cannot tell them
    apart.

    Args:
        exc: The terminal exception, already classified by
            :func:`_is_connection_error`.
        worker_name: The worker whose run failed.
        attempts: How many agent runs were burned before giving up — the retry
            ladder can spend seven, so a caller seeing one failure has no idea
            how much work it actually cost.

    Returns:
        A single-line detail string. Always includes the original message, so no
        information is lost relative to the bare ``str(exc)`` it replaces.
    """

    def _render(e: BaseException) -> str:
        text = str(e).strip()
        return f"{type(e).__name__}: {text}" if text else type(e).__name__

    # Seed the dedupe with the TOP-LEVEL rendering, not an empty list: the first
    # __context__ link is usually the previous attempt's identical exception, so
    # comparing only against the previous CHAIN entry lets one twin through.
    chain: list[str] = []
    previous = _render(exc)
    seen: set[int] = {id(exc)}
    cursor: BaseException | None = exc.__cause__ or exc.__context__
    while cursor is not None and len(chain) < _CAUSE_CHAIN_DEPTH:
        if id(cursor) in seen:
            break
        seen.add(id(cursor))
        link = _render(cursor)
        # The retry ladder re-raises: each attempt's APIConnectionError carries the
        # previous attempt's IDENTICAL exception as its __context__, so an undeduped
        # walk renders "Connection error. <- Connection error. <- Connection error."
        # and pushes the one informative link off the end of the depth cap.
        if link != previous:
            chain.append(link)
            previous = link
        cursor = cursor.__cause__ or cursor.__context__

    detail = _render(exc)
    if chain:
        detail += " <- " + " <- ".join(chain)
    # Every link is raw upstream text (an httpx error echoes its request URL, and a
    # URL can carry ``?key=``; issue #327) and this string reaches the caller.
    detail = redact_secrets_in_text(detail)
    return f"{detail} [worker={worker_name}, attempts={attempts}]"


#: Upstream HTTP statuses that ``send_message`` re-emits AS THEMSELVES instead of
#: collapsing into a generic 500.
#:
#: Deliberately a single entry. 503 is the one upstream status with a caller
#: semantic that 500 actively destroys: "I am at capacity, come back later" versus
#: "this service is broken". In one hour of the 2026-08-31 incident the
#: ``/workers/*/message`` access log was 200:1,074 / 500:2,538 / everything
#: else:0, and 2,538 of those 2,544 failures were a single class --
#: ``ModelHTTPError: status_code: 503, model_name: local-llm, body: {'detail':
#: "All providers failed. ..."}``. Downstream, ``w2v-orchestrator`` raised
#: ``TransientUpstreamError: upstream 500`` and four E2E shards went red pointing
#: at worker-management rather than at the inference path that was actually down.
#:
#: Widen only on measurement, the same discipline as
#: ``_DETERMINISTIC_FAILURE_MARKERS``. Pass-through is not free: it hands the
#: upstream direct control of our status code, so each entry must be a status we
#: are willing to mean verbatim.
_PASS_THROUGH_UPSTREAM_STATUSES: tuple[int, ...] = (503,)

_MIN_HTTP_STATUS = 100
_MAX_HTTP_STATUS = 599


def _status_code_on(exc: BaseException) -> int | None:
    """Read a structured HTTP status off one exception object, or None.

    Two shapes cover every provider exception this path sees:

    * ``exc.status_code`` -- ``pydantic_ai.exceptions.ModelHTTPError`` (which
      stores it as a plain ``int`` attribute and has no ``.response`` at all) and
      ``openai.APIStatusError``.
    * ``exc.response.status_code`` -- ``httpx.HTTPStatusError``.

    Checking ``.response`` alone is what made this invisible: the repo's existing
    ``extract_error_detail`` does exactly that and therefore reports
    ``status_code=None`` for the very class that caused the incident.
    """
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        # `bool` is an `int` subclass, and a MagicMock attribute is neither --
        # both must fail this test rather than be coerced.
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and _MIN_HTTP_STATUS <= candidate <= _MAX_HTTP_STATUS
        ):
            return candidate
    return None


def _upstream_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status the UPSTREAM actually returned, or None.

    Reads a structured attribute off the exception rather than pattern-matching
    its text. ``_is_connection_error``'s five literal substrings are the
    cautionary tale: none of them appears in a ``ModelHTTPError`` repr, so a clean
    upstream 503 matched nothing and fell through to 500. A sixth substring would
    have fixed that one repr and broken again on the next one; the status is
    already on the object as an ``int``.

    Walks ``__cause__``/``__context__`` to ``_CAUSE_CHAIN_DEPTH`` because the
    retry ladder re-raises, so the status-bearing exception is not always the
    outermost one. The walk is cycle-guarded for the same reason as
    :func:`_describe_connection_failure`: a self-referential chain must not hang
    the error path.

    Args:
        exc: The terminal exception from the agent run.

    Returns:
        The upstream status as an ``int``, or ``None`` when the exception carries
        no structured status (which keeps every pre-existing branch in charge).
    """
    direct = _status_code_on(exc)
    if direct is not None:
        return direct

    seen: set[int] = {id(exc)}
    cursor: BaseException | None = exc.__cause__ or exc.__context__
    depth = 0
    while cursor is not None and depth < _CAUSE_CHAIN_DEPTH:
        if id(cursor) in seen:
            break
        seen.add(id(cursor))
        found = _status_code_on(cursor)
        if found is not None:
            return found
        cursor = cursor.__cause__ or cursor.__context__
        depth += 1
    return None


#: ``X-IR-Error-Classification`` value inference-router sets when no provider
#: slot is free. The request is fine and the upstream is healthy -- it is busy.
IR_QUEUE_REJECTED = "queue_rejected"


def _ir_meta_on(exc: BaseException) -> tuple[str | None, str | None]:
    """Read (x-ir-error-classification, retry-after) off one exception's response."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None, None
    # httpx.Headers is case-insensitive; a plain dict (tests, other clients) is not.
    lowered = (
        {str(k).lower(): v for k, v in headers.items()} if isinstance(headers, dict) else headers
    )
    label = lowered.get("x-ir-error-classification")
    retry_after = lowered.get("retry-after")
    return (
        label if isinstance(label, str) and label else None,
        retry_after if isinstance(retry_after, str) and retry_after else None,
    )


def _ir_error_meta(exc: BaseException) -> tuple[str | None, str | None]:
    """Return IR's (error classification, Retry-After) for a failed call, or Nones.

    pydantic-ai's ``ModelHTTPError`` has no ``.response``: IR's headers live one
    link down, on the ``openai.APIStatusError`` it was raised ``from``. Walks
    ``__cause__``/``__context__`` exactly like :func:`_upstream_status_code`
    (same depth cap, same cycle guard) and returns the first exception that
    carries a classification header.
    """
    label, retry_after = _ir_meta_on(exc)
    if label is not None:
        return label, retry_after

    seen: set[int] = {id(exc)}
    cursor: BaseException | None = exc.__cause__ or exc.__context__
    depth = 0
    while cursor is not None and depth < _CAUSE_CHAIN_DEPTH:
        if id(cursor) in seen:
            break
        seen.add(id(cursor))
        label, retry_after = _ir_meta_on(cursor)
        if label is not None:
            return label, retry_after
        cursor = cursor.__cause__ or cursor.__context__
        depth += 1
    return None, None


def _is_queue_rejected(exc: BaseException) -> bool:
    """True when inference-router refused the call because its queue is full."""
    return _ir_error_meta(exc)[0] == IR_QUEUE_REJECTED
