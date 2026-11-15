"""Behavioural regression: send_message must not advertise futile work as retryable.

This is the wiring test for `_is_deterministic_failure`. The unit tests in
`test_deterministic_failure_classification.py` prove the classifier's verdicts;
they cannot prove the endpoint acts on them. Asserting on `inspect.getsource` (as
some older tests in this repo do) would prove even less -- it checks that a line
was written, not that a request produces the status.

Why the status matters: the n8n Workers node gates its retry on
`statusCode >= 500` (`helpers/LicenseAuthenticator.ts`). While every terminal
failure returned 500, a request that could never succeed was re-issued up to
MAX_RETRIES times, and each re-issue restarts the full agent run inside
worker_management -- real GPU decodes billed for a guaranteed failure. Over a
30-day production window, 3347 of 3645 terminal errors (91.8%) were in the
deterministic class.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.exceptions import ModelHTTPError

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id

_MP = "src.api.workers.message_endpoints"

# Verbatim production strings, with their 30-day counts.
DETERMINISTIC_ERROR = (
    "Model token limit (2500) exceeded before any response was generated. "
    "Increase the `max_tokens` model setting."
)  # n=1909
OUTPUT_RETRY_ERROR = "Exceeded maximum output retries (0)"  # n=207
TRANSIENT_ERROR = "Connection error."  # n=275


@pytest.fixture
def failing_worker_client():
    """TestClient whose agent run raises a caller-supplied exception.

    Yields (client, set_failure). `set_failure(exc)` installs the exception that
    `worker.run` will raise, so each test drives the identical endpoint path and
    differs only in the terminal error -- which is the variable under test.
    """
    # ⚠️ The retry ladder must be able to RUN, or this fixture silently stops testing
    # what it claims to. Before these three patches, `worker_cache.remove_instance` was
    # a bare MagicMock: the ladder awaited it, raised "'MagicMock' object can't be
    # awaited", and THAT became the terminal error. So the transient case never reached
    # the connection-error branch at all -- it asserted 500 and passed for the wrong
    # reason, a vacuous negative control. `_backoff_delay` is neutralised so the two
    # recreate-retries do not put ~6s of real sleeps in the suite.
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.StateDatabase") as mock_state_db,
        patch(f"{_MP}._lookup_tenant_byo_mode", new=AsyncMock(return_value=None)),
        patch(f"{_MP}._backoff_delay", return_value=0.0),
    ):
        mock_cache.remove_instance = AsyncMock(return_value=None)
        mock_registry.get_worker = AsyncMock(
            return_value={"worker_id": "wid", "worker_name": "expander", "is_agent": True}
        )
        # Worker-state persistence is not under test; keep it inert so a failure
        # there cannot be mistaken for the status-code behaviour.
        mock_state_db.return_value.add_worker_state = AsyncMock(return_value=None)

        worker = MagicMock()
        mock_cache.get_or_create_worker = AsyncMock(return_value=worker)

        def set_failure(exc: Exception) -> None:
            worker.run = AsyncMock(side_effect=exc)

        app.dependency_overrides[get_tenant_id] = lambda: "test-tenant"

        @asynccontextmanager
        async def _fake_lifespan(_a):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _fake_lifespan
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, set_failure
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original


@pytest.mark.parametrize("error_message", [DETERMINISTIC_ERROR, OUTPUT_RETRY_ERROR])
def test_deterministic_failure_returns_non_retryable_status(failing_worker_client, error_message):
    """A failure that cannot change must come back as 4xx, so the caller stops."""
    client, set_failure = failing_worker_client
    set_failure(RuntimeError(error_message))

    resp = client.post("/workers/expander/message", json={"message": "expand this"})

    assert resp.status_code == 422, resp.text
    # The caller's retry gate is `>= 500`; anything in that range re-runs the agent.
    assert resp.status_code < 500
    # The operator still needs the cause, so the detail must survive the remap.
    assert error_message[:30] in resp.text


def test_transient_failure_stays_retryable(failing_worker_client):
    """The negative control.

    If this test ever flips to 4xx, the classifier has grown too broad and a
    recoverable blip is being reported to the caller as permanent -- the failure
    mode that is strictly worse than the bug this change fixes.

    The invariant is the CALLER'S GATE, not the exact number: the n8n Workers node
    retries on `statusCode >= 500`. An unreachable upstream now comes back 502
    (BAD_GATEWAY) rather than 500 (INTERNAL_ERROR), which retries identically and
    stops calling someone else's outage our internal error.
    """
    client, set_failure = failing_worker_client
    set_failure(RuntimeError(TRANSIENT_ERROR))

    resp = client.post("/workers/expander/message", json={"message": "expand this"})

    assert resp.status_code >= 500, resp.text
    assert resp.status_code == 502, resp.text


def test_transient_failure_detail_names_the_hop(failing_worker_client):
    """`str(APIConnectionError(...))` is the bare literal "Connection error.".

    That string reached the caller verbatim and is why the 2026-08-30 14:49Z
    outage ran 40 hours unattributed. RED before this change: the detail was
    exactly TRANSIENT_ERROR and named neither the worker nor how many agent runs
    the retry ladder had already burned.
    """
    client, set_failure = failing_worker_client
    set_failure(RuntimeError(TRANSIENT_ERROR))

    resp = client.post("/workers/expander/message", json={"message": "expand this"})

    # The original message must survive -- enriching may not lose information.
    assert TRANSIENT_ERROR in resp.text
    # ...but it may no longer be the WHOLE message. This is the mutation guard: a
    # revert to `detail=str(e)` passes every assertion above and fails this one.
    # 3 = 1 primary + 2 recreate-retries (llm_retry_max_attempts default). Pinning the
    # number is the point: the ladder's cost was invisible in the old message.
    assert "[worker=expander, attempts=3]" in resp.text
    assert "RuntimeError" in resp.text


def test_transient_failure_detail_carries_the_underlying_cause(failing_worker_client):
    """The cause chain is the whole diagnostic value.

    ConnectTimeout means the upstream is unreachable; PoolTimeout means our own
    client pool is exhausted and the upstream is fine. Opposite fixes, and the
    bare literal cannot tell them apart.
    """
    client, set_failure = failing_worker_client

    try:
        raise TimeoutError("pool timeout waiting for a free connection")
    except TimeoutError as root:
        exc = RuntimeError(TRANSIENT_ERROR)
        exc.__cause__ = root
    set_failure(exc)

    resp = client.post("/workers/expander/message", json={"message": "expand this"})

    assert resp.status_code == 502, resp.text
    assert "TimeoutError" in resp.text
    assert "pool timeout waiting for a free connection" in resp.text


def test_unrecognised_failure_stays_retryable(failing_worker_client):
    """Fail-safe: an unseen error keeps the pre-existing 500 behaviour."""
    client, set_failure = failing_worker_client
    set_failure(RuntimeError("a failure mode nobody has catalogued yet"))

    resp = client.post("/workers/expander/message", json={"message": "expand this"})

    assert resp.status_code == 500, resp.text


# ===========================================================================
# Issue #212 -- an upstream 503 must not be re-badged as a 500, and the ERROR
# line must carry the status.
# ===========================================================================

# Verbatim production shape: 2,538 of the 2,544 failures in one hour of the
# 2026-08-31 incident were this exact class.
UPSTREAM_503_BODY = {"detail": "All providers failed. Last error: pgx-local is DEGRADED"}

# A 503 whose body happens to quote a provider-level connection failure. Not from
# production -- constructed to force the substring classifier and the structured
# status to disagree, because `_is_connection_error` matches on `str(exc)` and
# `str(ModelHTTPError)` embeds the body.
UPSTREAM_503_QUOTING_A_CONNECTION_ERROR = {
    "detail": "All providers failed. Last error: pgx-local: Connection error."
}


def _model_http_error(status: int = 503, body: dict | None = None) -> ModelHTTPError:
    """The REAL pydantic_ai exception, not a stand-in.

    The defect was an assumption about where the status lives on the real object
    (`ModelHTTPError` has `.status_code` and no `.response` at all), so a fake
    handed a `.status_code` would test the assumption rather than the object.
    """
    return ModelHTTPError(
        status_code=status, model_name="local-llm", body=body or UPSTREAM_503_BODY
    )


def _error_records(caplog):
    """ERROR-level records from the send-message path, as an operator would grep.

    Uses `record.getMessage()`, i.e. the message AFTER %-interpolation. Asserting
    on `record.msg` would pass against the unformatted template and prove
    nothing about what is written to the log.
    """
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "src.api.workers.message_endpoints"
    ]


class TestUpstreamStatusIsNotRebadged:
    def test_upstream_503_is_not_rebadged_as_500(self, failing_worker_client):
        """RED before the fix: 500.

        A 503 is a capacity signal the caller can back off on; a 500 says "this
        service is broken". Downstream, `w2v-orchestrator` raised
        `TransientUpstreamError: upstream 500` and four E2E shards went red
        pointing at worker-management instead of the inference path.
        """
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503))

        resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code != 500, resp.text
        assert resp.status_code == 503, resp.text
        # Still >= 500, so the n8n Workers node retries exactly as it did.
        assert resp.status_code >= 500
        # The upstream detail must survive the pass-through.
        assert "pgx-local is DEGRADED" in resp.text

    def test_upstream_503_beats_the_connection_error_substring(self, failing_worker_client):
        """Precedence: a structured status outranks a substring guess.

        RED before the fix in a different way -- this one returned 502, because
        `_is_connection_error` matched the body text. The upstream said 503.
        """
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503, body=UPSTREAM_503_QUOTING_A_CONNECTION_ERROR))

        resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 503, resp.text

    def test_a_503_found_down_the_cause_chain_still_passes_through(self, failing_worker_client):
        """The retry ladder re-raises, so the status bearer is not always outermost."""
        client, set_failure = failing_worker_client
        wrapper = RuntimeError("agent run failed")
        wrapper.__cause__ = _model_http_error(503)
        set_failure(wrapper)

        resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 503, resp.text

    def test_other_upstream_5xx_are_not_passed_through(self, failing_worker_client):
        """The pass-through set is deliberately just 503.

        Widening hands the upstream direct control of our status code. An upstream
        500 keeps arriving as our 500 until someone measures a reason to change it.
        """
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(500))

        resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 500, resp.text

    def test_a_deterministic_failure_still_wins_over_the_status_branch(self, failing_worker_client):
        """Ordering guard: 422 is a property of the REQUEST and outranks everything.

        Without this, adding the status branch above the 422 branch would silently
        restore the retry storm that `_is_deterministic_failure` exists to stop.
        """
        client, set_failure = failing_worker_client
        exc = ModelHTTPError(
            status_code=503,
            model_name="local-llm",
            body={"detail": DETERMINISTIC_ERROR},
        )
        set_failure(exc)

        resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 422, resp.text


class TestTheErrorLineCarriesTheStatus:
    """An operator grepping ERROR for "500" during the incident found NOTHING.

    The ERROR text was `"Error sending message to worker promo-review"` -- the
    status lived only inside `exc_info`. The sole line carrying the number was
    `logger.info("HTTPException handler: 500 ...")` in `src/api/error_handlers.py`,
    at INFO. Counted properly, that INFO line and this ERROR line were 2,544 each.
    """

    def test_error_line_names_the_status_it_returns(self, failing_worker_client, caplog):
        """RED before the fix: the ERROR message contained no status at all."""
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503))

        with caplog.at_level(logging.ERROR, logger="src.api.workers.message_endpoints"):
            resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 503
        messages = _error_records(caplog)
        assert messages, "no ERROR record was emitted by the send-message path"
        # The grep an operator actually runs.
        assert any("503" in m for m in messages), messages

    def test_error_line_names_a_500_when_it_returns_500(self, failing_worker_client, caplog):
        """The case from the incident: 2,544 ERROR lines, none containing "500"."""
        client, set_failure = failing_worker_client
        set_failure(RuntimeError("a failure mode nobody has catalogued yet"))

        with caplog.at_level(logging.ERROR, logger="src.api.workers.message_endpoints"):
            resp = client.post("/workers/expander/message", json={"message": "expand this"})

        assert resp.status_code == 500
        messages = _error_records(caplog)
        assert any("500" in m for m in messages), messages

    def test_error_line_distinguishes_our_status_from_the_upstream_status(
        self, failing_worker_client, caplog
    ):
        """Both numbers, labelled.

        The whole misdirection was that our 500 was read as OUR fault. The line
        has to say which status came from where.
        """
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503))

        with caplog.at_level(logging.ERROR, logger="src.api.workers.message_endpoints"):
            client.post("/workers/expander/message", json={"message": "expand this"})

        messages = _error_records(caplog)
        assert any("upstream_status=503" in m for m in messages), messages

    def test_error_line_keeps_its_existing_prefix(self, failing_worker_client, caplog):
        """Do not break the greps and dashboards that already match this text.

        "Error sending message to worker" matched 2,544 lines in the incident
        hour; the status is appended to that line, not substituted for it.
        """
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503))

        with caplog.at_level(logging.ERROR, logger="src.api.workers.message_endpoints"):
            client.post("/workers/expander/message", json={"message": "expand this"})

        messages = _error_records(caplog)
        assert any(m.startswith("Error sending message to worker expander") for m in messages), (
            messages
        )

    def test_the_traceback_is_still_attached(self, failing_worker_client, caplog):
        """Moving from logger.exception() to logger.error() must not drop exc_info."""
        client, set_failure = failing_worker_client
        set_failure(_model_http_error(503))

        with caplog.at_level(logging.ERROR, logger="src.api.workers.message_endpoints"):
            client.post("/workers/expander/message", json={"message": "expand this"})

        records = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and r.name == "src.api.workers.message_endpoints"
        ]
        assert records
        assert any(r.exc_info is not None for r in records), "exc_info was lost"
