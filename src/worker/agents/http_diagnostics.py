"""Outbound-HTTP failure diagnostics for the inference-router client.

⛔ THIS EXISTS BECAUSE THE ERROR LINE HAD NOTHING TO JOIN ON.

The 2026-08 "Connection error" investigation reconstructed 13,689 Running→Error
pairs and could describe the failure precisely -- death inside
``httpcore/_async/http11.py:177 _receive_response_headers``, i.e. BEFORE one
response byte, a median 1.64s after the request was issued, 83.8% under 5s, and
no mass whatsoever at 30s/60s/300s/600s/1200s -- and still could not name a
cause. Timeout was ruled out (the client's read budget is
``ir_http_read_timeout_s`` = 600s, and a 1.64s death is three orders of
magnitude short of it). The mesh was ruled out (ztunnel logged 10,692
``connection complete``, 0 failed, 0 error, in the worst burst hour). The router
was ruled out (0 ``client disconnect`` in 30 days). Resources were ruled out
(0 OOM, CPU < 1%).

What stopped the investigation was not a missing theory. It was that
``PydanticAIAgent.run``'s error line carried only ``timestamp, level, logger,
message, exc_info`` -- no elapsed, no target, no request id, and above all no
statement of whether the socket had been opened for this request or taken back
out of the keep-alive pool. Every candidate explanation predicts the same
five-field log line, so no field in it could discriminate between them.

This transport emits the fields that discriminate:

``elapsed_s``
    The decisive one. A stale-socket EOF lands ~1.6s after issue; a timeout
    lands on the configured budget. These are not confusable once the number
    is written down.
``connection_state``
    ``new`` | ``reused`` | ``unknown`` -- see below. The leading hypothesis is
    that httpx's default ``max_keepalive_connections=20`` / ``keepalive_expiry=5.0``
    (the client carried no explicit ``limits=``) sits underneath a measured
    17-24 in-flight requests per agent name, so sockets are evicted and
    re-offered constantly, and ztunnel counted ~2 L4 connections per routed
    request. A server-closed socket handed back out of the pool produces
    EXACTLY an EOF-before-headers. ``connection_state=reused`` on the failures
    confirms that; ``connection_state=new`` refutes it outright.
``request_id`` / ``request_id_source``
    Join key to inference-router's own logs. ``X-Correlation-Id`` is already
    stamped on every outbound call by ``make_identity_event_hook`` and is
    already read by IR, so it is the join key that exists rather than a new
    one -- ⛔ this module adds NO header to the wire. When the ContextVar is
    unset there is nothing to join on, and the field says so
    (``request_id_source=local``) instead of implying a correlation that the
    router never saw.
``response_headers_received``
    Distinguishes "died before the response line" from "died mid-response".
    The whole investigation rests on the former; nothing was logging it.

    ⚠️ Measured on the ``.complete`` trace event, NOT ``.started``. httpcore
    emits ``receive_response_headers.started`` when it ENTERS the read, before
    a single byte has arrived, so a socket that hangs up with no status line
    at all still emits it. Keying this field on ``.started`` reports True on
    precisely the failure it exists to identify -- verified: the stale-socket
    reproduction logs ``.started`` and never ``.complete``, and dies as
    ``RemoteProtocolError: Server disconnected without sending a response``.

HOW CONNECTION REUSE IS OBTAINED
--------------------------------
httpcore exposes a per-request ``trace`` extension (``httpcore/_trace.py``):
a callback invoked as ``await cb(f"{module}.{event}.started", info)``. It is
measured, not assumed -- verified against httpx 0.28.1 / httpcore 1.0.9 with a
local HTTP/1.1 keep-alive server:

    req0: connection.connect_tcp.started present   -> new
    req1: connection.connect_tcp.started ABSENT    -> reused
    req2: connection.connect_tcp.started ABSENT    -> reused

and a positive control in the other direction -- against an HTTP/1.0 server
(no keep-alive) all three requests emit ``connect_tcp.started``, so the signal
tracks reality and is not just "the event stopped firing".

⚠️ The state is deliberately THREE-valued, not a bool. ``connect_tcp.started``
being absent only means "reused" if the request actually reached the wire. A
failure at pool-acquire (``ir_http_pool_timeout_s``) emits no trace event at
all, and reporting that as ``reused`` would manufacture evidence for the very
hypothesis this module exists to test. Absent-and-never-started is ``unknown``.

The callback MUST be async: httpcore raises ``TypeError`` on a sync callback
under the async interface.
"""

from __future__ import annotations

import logging
import time
import typing
import uuid

import httpx2 as httpx

logger = logging.getLogger(__name__)

# httpcore Trace names. The prefix is the emitting module's logger name
# (``logger.name.split(".")[-1]``), so these are "connection" and "http11"
# rather than anything we choose. Pinned by test, because a httpcore rename
# would silently turn every request into ``connection_state=unknown``.
CONNECT_TRACE_EVENT = "connection.connect_tcp.started"
REQUEST_SENT_TRACE_EVENT = "http11.send_request_headers.started"
RESPONSE_HEADERS_TRACE_EVENT = "http11.receive_response_headers.complete"

#: Greppable prefix for the failure line. One token, so a burst can be counted
#: out of the log stream without a regex over free text.
FAILURE_EVENT = "ir_http_request_failed"

CONNECTION_STATE_NEW = "new"
CONNECTION_STATE_REUSED = "reused"
CONNECTION_STATE_UNKNOWN = "unknown"


class _RequestTrace:
    """Records which httpcore lifecycle events fired for a single request."""

    __slots__ = ("connected", "request_sent", "response_headers_received")

    def __init__(self) -> None:
        self.connected = False
        self.request_sent = False
        self.response_headers_received = False

    @property
    def connection_state(self) -> str:
        if self.connected:
            return CONNECTION_STATE_NEW
        if self.request_sent:
            # Bytes went out and no TCP connect preceded them: the socket came
            # back out of the keep-alive pool.
            return CONNECTION_STATE_REUSED
        # Nothing reached the wire -- pool-acquire failure, or a connect that
        # died before httpcore traced it. Saying "reused" here would be an
        # invention, not a measurement.
        return CONNECTION_STATE_UNKNOWN


def resolve_request_id(request: httpx.Request) -> tuple[str, str]:
    """Return ``(request_id, source)`` for *request*.

    Prefers the ``X-Correlation-Id`` already on the wire, because that is the
    value inference-router logs and therefore the only one a cross-service join
    can use. Falls back to a locally-minted id that is honestly labelled as
    such -- a local id correlates this line to nothing downstream, and
    pretending otherwise sends the next investigator hunting in IR's logs for
    a value IR never received.
    """
    correlation = request.headers.get("X-Correlation-Id")
    if correlation:
        return correlation, "correlation-id"
    return uuid.uuid4().hex, "local"


class DiagnosticAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """``AsyncHTTPTransport`` that logs a joinable line when a request fails.

    Behaviour on the success path is unchanged: the response is returned
    untouched and nothing is logged. This is strictly an error-path
    instrument, so it cannot add log volume proportional to traffic.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send *request*, logging a joinable diagnostic line if it fails."""
        trace = _RequestTrace()
        # Chain rather than clobber: httpx/openai may already have installed a
        # trace extension, and eating it would be a second observability
        # regression introduced by an observability fix.
        existing = request.extensions.get("trace")

        async def _trace_hook(name: str, info: typing.Any) -> None:
            if name == CONNECT_TRACE_EVENT:
                trace.connected = True
            elif name == REQUEST_SENT_TRACE_EVENT:
                trace.request_sent = True
            elif name == RESPONSE_HEADERS_TRACE_EVENT:
                trace.response_headers_received = True
            if existing is not None:
                await existing(name, info)

        request.extensions["trace"] = _trace_hook

        # monotonic, not wall clock: this number is compared against timeout
        # budgets, and an NTP step must not be able to fabricate a 1.6s death.
        started = time.monotonic()
        try:
            return await super().handle_async_request(request)
        except Exception as exc:
            elapsed_s = time.monotonic() - started
            request_id, request_id_source = resolve_request_id(request)
            fields = {
                "request_id": request_id,
                "request_id_source": request_id_source,
                "elapsed_s": round(elapsed_s, 4),
                "target_url": str(request.url),
                "http_method": request.method,
                "connection_state": trace.connection_state,
                "response_headers_received": trace.response_headers_received,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            # logger.exception, not logger.error: this is the ONLY place the
            # real httpcore frames are still on the exception. By the time it
            # surfaces in PydanticAIAgent.run the OpenAI SDK has rewrapped it
            # as APIConnectionError("Connection error."), which is the string
            # the investigation had and the reason it had to guess at the
            # layer below.
            logger.exception(
                "%s request_id=%s request_id_source=%s elapsed_s=%.4f "
                "target_url=%s http_method=%s connection_state=%s "
                "response_headers_received=%s error_type=%s: %s",
                FAILURE_EVENT,
                fields["request_id"],
                fields["request_id_source"],
                elapsed_s,
                fields["target_url"],
                fields["http_method"],
                fields["connection_state"],
                fields["response_headers_received"],
                fields["error_type"],
                fields["error_message"],
                extra=fields,
            )
            raise
