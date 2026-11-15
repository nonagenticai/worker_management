"""WM#324 step 1: the no-header tokenless branch logs its peer. LOG ONLY.

A tokenless caller that sends no X-Tenant-Id lands on DEFAULT_TENANT_ID and
increments ``identity_omitted_total{reason="default"}``, but until this change
it logged nothing, so the caller could not be attributed. ztunnel is L4 and
preserves the source pod IP, so ``request.client.host`` is the attribution.

Pinned here:
  * a no-header request logs ``peer=<client host>``;
  * its response is byte-for-byte what it was (no behaviour change);
  * the bare-header (asserted) path is unchanged: its own WARNING still names
    the peer, and the new line does not fire for it;
  * the new line is deduplicated like its sibling, and the peer never becomes a
    metric label.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.api.auth_middleware as auth_middleware_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.settings import reset_settings_cache

PEER = "10.42.1.7"
OTHER_PEER = "10.42.2.9"
ASSERTED_TENANT = "0195f48b-d000-7000-8000-000000000003"
LOGGER_NAME = auth_middleware_module.logger.name
NO_HEADER_MARKER = "identity omitted: a tokenless caller"
ASSERTED_MARKER = "unauthenticated tenant assertion"


def _clear_caches() -> None:
    auth_middleware_module._assertion_log_seen.clear()
    seen = getattr(auth_middleware_module, "_omission_log_seen", None)
    if seen is not None:
        seen.clear()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    _clear_caches()
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT", raising=False)
    monkeypatch.delenv("REQUIRE_AUTHENTICATED_TENANT", raising=False)
    reset_settings_cache()
    yield
    _clear_caches()
    reset_settings_cache()


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, object]:
        return {
            "tenant_id": request.state.tenant_id,
            "license_id": request.state.license_id,
            "subject_id": request.state.subject_id,
        }

    return application


def _call(app: FastAPI, headers: dict[str, str], peer: str = PEER):
    with TestClient(app, client=(peer, 54321), raise_server_exceptions=False) as client:
        return client.get("/api/probe", headers=headers)


def _lines(caplog: pytest.LogCaptureFixture, marker: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER_NAME and marker in r.getMessage()]


def test_a_no_header_request_logs_its_peer_ip(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """⛔ RED on origin/main: this branch logged no peer at all."""
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        resp = _call(app, {})

    lines = _lines(caplog, NO_HEADER_MARKER)
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    message = lines[0].getMessage()
    assert f"peer={PEER} " in message
    assert "route=/api/probe " in message
    assert lines[0].levelno >= logging.INFO, "must survive the production INFO level"

    # The response is exactly what the branch served before this change.
    assert resp.status_code == 200
    assert resp.json() == {
        "tenant_id": auth_middleware_module.DEFAULT_TENANT_ID,
        "license_id": None,
        "subject_id": None,
    }


def test_the_no_header_response_is_unchanged_when_logging_is_suppressed(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """Same response with and without the line: logging cannot alter the outcome."""
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        caplog.set_level(logging.CRITICAL, logger=LOGGER_NAME)
        quiet = _call(app, {})
        _clear_caches()
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        loud = _call(app, {})
    assert (quiet.status_code, quiet.json()) == (loud.status_code, loud.json())


def test_a_broken_metrics_backend_does_not_silence_the_peer_line(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    metrics = MagicMock()
    metrics.record_identity_omission.side_effect = RuntimeError("otlp exploded")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        resp = _call(app, {})
    assert resp.status_code == 200
    assert len(_lines(caplog, NO_HEADER_MARKER)) == 1


def test_the_bare_header_path_is_unchanged(app: FastAPI, caplog: pytest.LogCaptureFixture) -> None:
    """The asserted branch keeps its own WARNING (with peer) and gets no new line."""
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        resp = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert resp.status_code == 200
    assert resp.json() == {"tenant_id": ASSERTED_TENANT, "license_id": None, "subject_id": None}
    asserted = _lines(caplog, ASSERTED_MARKER)
    assert len(asserted) == 1
    assert asserted[0].levelno == logging.WARNING
    assert f"peer={PEER} " in asserted[0].getMessage()
    assert _lines(caplog, NO_HEADER_MARKER) == []


def test_the_peer_line_is_deduplicated_per_peer(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        for _ in range(3):
            _call(app, {})
        _call(app, {}, peer=OTHER_PEER)

    peers = [r.getMessage() for r in _lines(caplog, NO_HEADER_MARKER)]
    assert len(peers) == 2
    assert f"peer={PEER} " in peers[0]
    assert f"peer={OTHER_PEER} " in peers[1]


def test_the_peer_is_never_a_metric_label(app: FastAPI) -> None:
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        _call(app, {})
    assert metrics.record_identity_omission.call_count == 2
    for call in metrics.record_identity_omission.call_args_list:
        assert set(call.kwargs) == {"reason", "caller", "path"}
        assert PEER not in repr(call)
