"""worker_management#338 — the workflowAutoRollback webhook is authenticated and fail-visible.

The REAL ``routes._emit_rollback_webhook`` runs; only the network (an
``httpx.MockTransport`` standing in for fr) is replaced.

Receiver contract: fr ``POST /api/internal/workflow-rollback`` (fr#1427) —
header ``X-Internal-Token`` == fr ``FR_WM_ROLLBACK_TOKEN``; 503 unset, 401 bad
token, 400 bad payload, 200 ``{"accepted": true}``.

MUTATIONS (each turns a test here red):
* drop the header ⇒ ``test_200_is_ok_and_sends_the_token_to_the_configured_url``;
* send with only a url (old behaviour) ⇒ ``test_unconfigured_token_sends_nothing``;
* count a 401 as ok ⇒ ``test_401_is_http_error``;
* log ``%s`` of the exception ⇒ ``test_token_is_never_logged``;
* drop the import-time pre-initialisation ⇒ ``test_all_five_series_exist_at_import_with_value_zero``;
* drop the ``(httpx.InvalidURL, UnicodeEncodeError)`` except ⇒
  ``test_malformed_config_is_unconfigured_and_never_raises``;
* drop the final ``except Exception`` catch-all ⇒
  ``test_unenumerated_bad_url_is_error_and_never_raises`` and
  ``test_catch_all_never_logs_the_token``;
* log ``%r``/``%s`` of the exception in the catch-all ⇒
  ``test_catch_all_never_logs_the_token``;
* widen the catch-all to ``BaseException`` ⇒
  ``test_cancellation_still_propagates``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2 as httpx
import pytest
from prometheus_client import REGISTRY

from src.api.workflow_routing import routes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

URL = "http://fr-backend.fr.svc.cluster.local:8888/api/internal/workflow-rollback"
TOKEN = "cd" * 32  # pragma: allowlist secret
METRIC = "wm_workflow_rollback_webhook_total"
PAYLOAD = {"event": "workflowAutoRollback", "asset_key": "cell:research"}


def _value(outcome: str) -> float | None:
    return REGISTRY.get_sample_value(METRIC, {"outcome": outcome})


def _counts() -> dict[str, float]:
    return {o: _value(o) or 0.0 for o in routes.ROLLBACK_WEBHOOK_OUTCOMES}


class FakeFr:
    def __init__(self, answer: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.answer = answer

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.answer(request)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    from src.settings import reset_settings_cache

    monkeypatch.setenv("FR_WORKFLOW_ROLLBACK_URL", URL)
    monkeypatch.setenv("FR_WM_ROLLBACK_TOKEN", TOKEN)
    reset_settings_cache()
    yield monkeypatch
    monkeypatch.undo()
    reset_settings_cache()


def _install(answer: Callable[[httpx.Request], httpx.Response]) -> FakeFr:
    fake = FakeFr(answer)
    routes.set_rollback_webhook_transport_for_tests(httpx.MockTransport(fake.handle))
    return fake


@pytest.fixture(autouse=True)
def _restore_transport() -> Iterator[None]:
    yield
    routes.set_rollback_webhook_transport_for_tests(None)


def _emit() -> None:
    asyncio.run(routes._emit_rollback_webhook(dict(PAYLOAD)))


def _delta(before: dict[str, float]) -> dict[str, float]:
    after = _counts()
    return {o: after[o] - before[o] for o in after if after[o] != before[o]}


_FRESH_IMPORT = (
    """
import json
from prometheus_client import REGISTRY
import src.api.workflow_routing.routes  # noqa: F401
outcomes = ("ok", "http_error", "transport_error", "unconfigured", "error")
print(json.dumps({o: REGISTRY.get_sample_value("%s", {"outcome": o}) for o in outcomes}))
"""
    % METRIC
)


def test_all_five_series_exist_at_import_with_value_zero() -> None:
    # A FRESH process: in this one other tests may already have incremented a
    # series, which would hide a missing pre-initialisation.
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[1],
    )
    values = json.loads(out.stdout.strip().splitlines()[-1])
    assert values == {
        "ok": 0.0,
        "http_error": 0.0,
        "transport_error": 0.0,
        "unconfigured": 0.0,
        "error": 0.0,
    }
    assert len(values) == 5
    assert set(routes.ROLLBACK_WEBHOOK_OUTCOMES) == set(values)
    for outcome in values:  # and registered on the shared registry here too
        assert _value(outcome) is not None


@pytest.mark.parametrize("unset", ["FR_WORKFLOW_ROLLBACK_URL", "FR_WM_ROLLBACK_TOKEN"])
def test_unconfigured_sends_nothing(
    env: pytest.MonkeyPatch, unset: str, caplog: pytest.LogCaptureFixture
) -> None:
    from src.settings import reset_settings_cache

    env.setenv(unset, "")
    reset_settings_cache()
    fake = _install(lambda _r: httpx.Response(200, json={"accepted": True}))
    before = _counts()
    with caplog.at_level(logging.WARNING, logger=routes.logger.name):
        _emit()
    assert fake.requests == []
    assert _delta(before) == {"unconfigured": 1.0}
    assert any("not sent" in r.getMessage() for r in caplog.records)


def test_unconfigured_token_sends_nothing(env: pytest.MonkeyPatch) -> None:
    from src.settings import reset_settings_cache

    env.delenv("FR_WM_ROLLBACK_TOKEN")
    reset_settings_cache()
    fake = _install(lambda _r: httpx.Response(200, json={"accepted": True}))
    before = _counts()
    _emit()
    assert fake.requests == []
    assert _delta(before) == {"unconfigured": 1.0}


def test_200_is_ok_and_sends_the_token_to_the_configured_url(env: pytest.MonkeyPatch) -> None:
    fake = _install(lambda _r: httpx.Response(200, json={"accepted": True}))
    before = _counts()
    _emit()
    assert len(fake.requests) == 1
    req = fake.requests[0]
    assert str(req.url) == URL
    assert req.method == "POST"
    assert req.headers["X-Internal-Token"] == TOKEN
    assert req.read() and b"workflowAutoRollback" in req.content
    assert _delta(before) == {"ok": 1.0}


@pytest.mark.parametrize("status", [400, 401, 404, 503])
def test_non_2xx_is_http_error(env: pytest.MonkeyPatch, status: int) -> None:
    _install(lambda _r: httpx.Response(status, json={"error": "nope"}))
    before = _counts()
    _emit()
    assert _delta(before) == {"http_error": 1.0}


def test_401_is_http_error(env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _install(lambda _r: httpx.Response(401, text="BODY-MUST-NOT-BE-LOGGED"))
    before = _counts()
    with caplog.at_level(logging.DEBUG):
        _emit()
    assert _delta(before) == {"http_error": 1.0}
    assert "401" in caplog.text
    assert "BODY-MUST-NOT-BE-LOGGED" not in caplog.text


def test_connect_error_is_transport_error_and_never_raises(env: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    fake = _install(refuse)
    before = _counts()
    _emit()  # must not raise into the rollback request
    assert len(fake.requests) == 1
    assert _delta(before) == {"transport_error": 1.0}


def test_oserror_is_transport_error(env: pytest.MonkeyPatch) -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise OSError("network unreachable")

    _install(boom)
    before = _counts()
    _emit()
    assert _delta(before) == {"transport_error": 1.0}


def test_token_is_never_logged(env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def leaky(request: httpx.Request) -> httpx.Response:
        # An error whose str() carries the token, as a careless %s would log it.
        raise httpx.ConnectError(
            f"failed with {request.headers['X-Internal-Token']}", request=request
        )

    with caplog.at_level(logging.DEBUG):
        for answer in (
            leaky,
            lambda r: httpx.Response(401, text=r.headers["X-Internal-Token"]),
            lambda _r: httpx.Response(200, json={"accepted": True}),
        ):
            _install(answer)
            _emit()
    assert caplog.records, "expected at least one log record"
    assert TOKEN not in caplog.text


@pytest.mark.parametrize(
    ("var", "value", "raises"),
    [
        # httpx 0.28: an unclosed IPv6 host raises httpx.InvalidURL, whose MRO is
        # (InvalidURL, Exception) — NOT an httpx.HTTPError.
        ("FR_WORKFLOW_ROLLBACK_URL", "http://[::1/x", httpx.InvalidURL),
        # A non-ASCII token fails encoding the header: UnicodeEncodeError.
        ("FR_WM_ROLLBACK_TOKEN", "t\u00f6ken", UnicodeEncodeError),
    ],
)
def test_malformed_config_is_unconfigured_and_never_raises(
    env: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    var: str,
    value: str,
    raises: type[Exception],
) -> None:
    from src.settings import reset_settings_cache

    assert not issubclass(raises, httpx.HTTPError)  # the premise of the except entry
    env.setenv(var, value)
    reset_settings_cache()
    fake = _install(lambda _r: httpx.Response(200, json={"accepted": True}))
    before = _counts()
    with caplog.at_level(logging.DEBUG):
        _emit()  # must not raise into the rollback request
    assert fake.requests == []
    assert _delta(before) == {"unconfigured": 1.0}
    assert raises.__name__ in caplog.text
    assert value not in caplog.text


@pytest.mark.parametrize(
    ("url", "raises", "logged"),
    [
        # Port out of range: anyio raises ExceptionGroup[OverflowError] from the
        # connect — neither httpx.HTTPError nor OSError. Fails before any socket
        # is opened, so the real (un-mocked) network path is safe here.
        ("http://127.0.0.1:99999/", ExceptionGroup, "ExceptionGroup"),
    ],
)
def test_unenumerated_bad_url_is_error_and_never_raises(
    env: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    url: str,
    raises: type[Exception],
    logged: str,
) -> None:
    from src.settings import reset_settings_cache

    # Premise: the specific excepts do NOT cover this class, so only the
    # catch-all stands between it and the committed rollback request.
    assert not issubclass(raises, (httpx.HTTPError, OSError, httpx.InvalidURL, UnicodeEncodeError))
    env.setenv("FR_WORKFLOW_ROLLBACK_URL", url)
    reset_settings_cache()
    routes.set_rollback_webhook_transport_for_tests(None)  # the real httpcore path raises these
    before = _counts()
    with caplog.at_level(logging.DEBUG):
        _emit()  # must not raise into the rollback request
    assert _delta(before) == {"error": 1.0}
    assert f"(unexpected): {logged}" in caplog.text
    assert TOKEN not in caplog.text


def test_idna_invalid_host_is_a_transport_error_and_never_raises(
    env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """httpx2 defers IDNA: ``http://xn--/`` is not rejected while building the URL.

    httpx 0.x raised ``idna.IDNAError`` before any socket; httpx2 hands the host
    to getaddrinfo, so it surfaces as a ConnectError (an HTTPError) and lands in
    the enumerated transport branch. The point this test defends is unchanged:
    the rollback request never raises and the token is never logged.
    """
    from src.settings import reset_settings_cache

    env.setenv("FR_WORKFLOW_ROLLBACK_URL", "http://xn--/")
    reset_settings_cache()
    routes.set_rollback_webhook_transport_for_tests(None)
    before = _counts()
    with caplog.at_level(logging.DEBUG):
        _emit()
    assert _delta(before) == {"transport_error": 1.0}
    assert TOKEN not in caplog.text


class _TokenCarryingError(Exception):
    """Not an httpx/OS/Unicode error: reaches the catch-all, message carries the token."""


def test_catch_all_never_logs_the_token(
    env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def leaky(request: httpx.Request) -> httpx.Response:
        raise _TokenCarryingError(f"boom {request.headers['X-Internal-Token']}")

    _install(leaky)
    before = _counts()
    with caplog.at_level(logging.DEBUG):
        _emit()
    assert _delta(before) == {"error": 1.0}
    assert "_TokenCarryingError" in caplog.text
    assert TOKEN not in caplog.text


def test_cancellation_still_propagates(env: pytest.MonkeyPatch) -> None:
    assert not issubclass(asyncio.CancelledError, Exception)  # premise (3.8+)

    def cancel(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    _install(cancel)
    before = _counts()
    with pytest.raises(asyncio.CancelledError):
        _emit()
    assert _delta(before) == {}
