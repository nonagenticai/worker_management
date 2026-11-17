"""Issue #327: the Google API key must never reach a URL, a 502 detail or a log.

Two independent layers, each with its own test:

1. ROOT: ``_fetch_google_models`` sends the key in the ``x-goog-api-key``
   header, not as ``?key=`` (httpx errors echo the request URL).
   MUTATION: put the key back in the URL ⇒ ``test_google_key_is_sent_in_the_header_not_the_url`` red.
2. DEFENCE IN DEPTH: upstream error text is scrubbed before it reaches the 502
   detail, the BYO relay or a logger.
   MUTATION: drop the scrub in ``_run_fetcher`` / ``_scrub`` ⇒ the
   ``*_echoing_the_key_*`` tests red.
3. The scrub is LINEAR on adversarial input (it runs on the event loop over
   unbounded upstream text). MUTATION: restore the unbounded URL scheme
   ``[A-Za-z][A-Za-z0-9+.-]*://`` ⇒ ``test_redact_is_linear_on_adversarial_input`` red.

Every key here is an obvious fake; never paste a real one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx2 as httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.providers import _router
from src.api.worker_management_api import app
from src.api.workers._helpers import _describe_connection_failure
from src.auth import get_tenant_id
from src.security import redact as _redact
from src.security.redact import redact_secrets_in_text
from src.worker.provider_registry import PROVIDER_API_KEY_ENV_VARS
from tests._platform_tenant import PLATFORM_TENANT, caller_tenant
from tests.test_pydantic_ai_agent import _external_llm_allowed

# Google key SHAPE (AIza + 35 chars) but unmistakably fake.
FAKE_KEY = "AIza" + "FAKE" + "a" * 31  # pragma: allowlist secret
# A second key-shaped value this process does NOT know (not the env key, not the
# row key): only the shape/query-param scrub can catch it.
OTHER_FAKE_KEY = "AIza" + "FAKE" + "b" * 31  # pragma: allowlist secret
GOOGLE_ENV = PROVIDER_API_KEY_ENV_VARS["google"]


def _run(coro):
    return asyncio.run(coro)


async def _fake_tenant_id(request=None):
    return "test-tenant"


@pytest.fixture
def client():
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        caller_tenant(PLATFORM_TENANT),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


def _mock_transport_client(handler):
    """Patch ``httpx.AsyncClient`` in the router so the REAL ``_httpx_json_get`` runs."""
    real = httpx.AsyncClient

    def factory(*_a, **_k):
        return real(transport=httpx.MockTransport(handler))

    return patch.object(_router.httpx, "AsyncClient", side_effect=factory)


def _no_key_in_logs(caplog) -> None:
    for record in caplog.records:
        blob = repr(record.__dict__) + record.getMessage()
        assert FAKE_KEY not in blob, record
        assert OTHER_FAKE_KEY not in blob, record


# ------------------------------------------------------------------ layer 1


def test_google_key_is_sent_in_the_header_not_the_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"models": [{"name": "models/gemini-x"}]})

    with _mock_transport_client(handler):
        ids = _run(_router._fetch_google_models(api_key=FAKE_KEY))

    assert ids == ["gemini-x"]
    assert len(seen) == 1
    req = seen[0]
    assert req.headers.get("x-goog-api-key") == FAKE_KEY
    assert FAKE_KEY not in str(req.url)
    assert "key=" not in str(req.url)
    assert str(req.url) == "https://generativelanguage.googleapis.com/v1beta/models"


def test_real_upstream_400_through_the_route_carries_no_key(client, monkeypatch, caplog):
    """End to end on the platform path with the REAL httpx error string."""
    monkeypatch.setenv(GOOGLE_ENV, FAKE_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    caplog.set_level(logging.DEBUG)
    with _external_llm_allowed(), _mock_transport_client(handler):
        resp = client.get("/providers/google/models")

    assert resp.status_code == 502, resp.text
    assert "Upstream unreachable for provider 'google'" in resp.text
    assert "400 Bad Request" in resp.text  # the useful part survives
    assert FAKE_KEY not in resp.text
    assert any(r.getMessage() == "provider_models_fetch_failed" for r in caplog.records)
    _no_key_in_logs(caplog)


# ------------------------------------------------------------------ layer 2


def test_upstream_error_echoing_the_key_is_scrubbed_in_502_and_log(client, monkeypatch, caplog):
    """Even if an upstream error still echoes a key, neither 502 nor log carry it."""
    monkeypatch.setenv(GOOGLE_ENV, FAKE_KEY)
    boom = RuntimeError(
        "Client error '400 Bad Request' for url "
        f"'https://generativelanguage.googleapis.com/v1beta/models?key={FAKE_KEY}&alt=json' "
        f"(also saw {OTHER_FAKE_KEY})"
    )
    caplog.set_level(logging.DEBUG)
    with _external_llm_allowed(), patch.object(_router, "_httpx_json_get", side_effect=boom):
        resp = client.get("/providers/google/models")

    assert resp.status_code == 502, resp.text
    assert FAKE_KEY not in resp.text
    assert OTHER_FAKE_KEY not in resp.text
    assert "key=***" in resp.text
    assert "alt=json" in resp.text  # non-secret params stay readable
    assert any(r.getMessage() == "provider_models_fetch_failed" for r in caplog.records)
    _no_key_in_logs(caplog)


def test_byo_live_check_error_echoing_the_key_is_scrubbed(caplog):
    """The BYO relay path (live_row_models -> ByoLiveCheckError -> byo_call_failed)."""
    boom = RuntimeError(
        f"401 for url 'https://generativelanguage.googleapis.com/v1beta/models?key={FAKE_KEY}' "
        f"echo {OTHER_FAKE_KEY}"
    )
    entry = {"api_key": FAKE_KEY}
    caplog.set_level(logging.DEBUG)
    with patch.object(_router, "_httpx_json_get", side_effect=boom):
        with pytest.raises(_router.ByoLiveCheckError) as info:
            _run(_router.live_row_models("t-327", "google", entry, fresh=True))
    msg = str(info.value)
    assert "401" in msg
    assert FAKE_KEY not in msg
    assert OTHER_FAKE_KEY not in msg
    detail = str(_router.byo_call_failed("t-327", "google", msg).detail)
    assert FAKE_KEY not in detail
    assert OTHER_FAKE_KEY not in detail
    _no_key_in_logs(caplog)


# ------------------------------------------------------------------ helper unit


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        (f"https://h/v1?key={FAKE_KEY}", FAKE_KEY),
        ("https://h/v1?alt=json&api_key=plainfakevalue", "plainfakevalue"),
        ("https://h/v1?access_token=plainfaketoken#frag", "plainfaketoken"),
        (f"bare {OTHER_FAKE_KEY} in prose", OTHER_FAKE_KEY),
        ("postgres://user:fakepassword@db/x", "fakepassword"),  # pragma: allowlist secret
        ("Authorization: Bearer fakebearervalue-0123", "fakebearervalue-0123"),
        ("Authorization: Basic ZmFrZXVzZXI6ZmFrZXBhc3M=", "ZmFrZXVzZXI6ZmFrZXBhc3M="),
        ("Proxy-Authorization: Bearer fake.jwt.tokenvalue", "fake.jwt.tokenvalue"),
        ("relayed: bearer fakebearervalue-0123 end", "fakebearervalue-0123"),
    ],
)
def test_redact_secrets_in_text_masks(text, leaked):
    assert leaked not in redact_secrets_in_text(text)


def test_redact_secrets_in_text_masks_an_exact_secret_of_any_shape():
    assert "s3cr3t-fake" not in redact_secrets_in_text("boom s3cr3t-fake boom", "s3cr3t-fake")


def test_redact_secrets_in_text_leaves_ordinary_text_alone():
    """Positive control: a scrub that blanks everything would pass the tests above."""
    text = "Client error '404 Not Found' for url 'https://h/v1/models?alt=json&page=2'"
    assert redact_secrets_in_text(text, None, "") == text


def test_redact_secrets_in_text_leaves_prose_auth_words_alone():
    """Over-redaction guard: auth words followed by non-token prose stay readable."""
    for text in (
        "upgrade to the Basic tier to use this model",
        "Authorization: denied for this project",
        "authorization: missing",
        "Bearer token expired",
    ):
        assert redact_secrets_in_text(text) == text


# ------------------------------------------------------------------ linear time


_ADVERSARIAL_LEN = 200 * 1024


@pytest.mark.parametrize(
    "text",
    [
        "a" * _ADVERSARIAL_LEN,
        "a" * _ADVERSARIAL_LEN + "://",
        ("a" * 40 + "://") * (_ADVERSARIAL_LEN // 43),
        "?key=" * (_ADVERSARIAL_LEN // 5),
        "?key" * (_ADVERSARIAL_LEN // 4),
        ("Authorization" + " " * 40) * (_ADVERSARIAL_LEN // 53),
        ("bearer" + " " * 40) * (_ADVERSARIAL_LEN // 46),
        ("a://" + "x" * 60 + ":") * (_ADVERSARIAL_LEN // 65),
        " " * _ADVERSARIAL_LEN,
    ],
    ids=[
        "letters",
        "letters-then-scheme",
        "many-schemes",
        "many-key-params",
        "many-key-no-equals",
        "authorization-whitespace",
        "bearer-whitespace",
        "userinfo-no-at",
        "spaces",
    ],
)
def test_redact_is_linear_on_adversarial_input(text):
    started = time.perf_counter()
    redact_secrets_in_text(text, "unrelated-fake-secret")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"{elapsed:.3f}s on {len(text)} chars"


def test_userinfo_password_still_masked_with_bounded_scheme():
    out = redact_secrets_in_text("connect failed: postgres://user:fakepw-327@db:5432/x")
    assert "fakepw-327" not in out
    assert "postgres://user:***@db:5432/x" in out


def test_oversized_input_is_capped_and_a_cut_secret_leaves_no_prefix():
    """Cap -> scrub -> drop a guard tail: a secret split by the cut leaks nothing."""
    exact = "s3cr3t-fake-" + "z" * 30
    head = f"?key={FAKE_KEY} " + "x" * (_redact._MAX_SCRUB_CHARS - 200)
    cut_at = _redact._MAX_SCRUB_CHARS - len(head)
    text = head + "y" * (cut_at - 8) + exact + " " + OTHER_FAKE_KEY + " " + "t" * 200_000
    out = redact_secrets_in_text(text, exact)
    assert len(out) < _redact._MAX_SCRUB_CHARS
    assert out.endswith("[truncated]")
    assert FAKE_KEY not in out and "key=***" in out  # scrubbed region still scrubbed
    assert exact[:8] not in out  # the half the cut left behind is gone too
    assert "AIza" + "FAKE" + "b" not in out


# ------------------------------------------------------------------ other sinks


def test_run_fetcher_502_carries_no_unscrubbed_exception_in_its_chain():
    async def boom():
        raise RuntimeError(f"400 for url 'https://h/v1/models?key={FAKE_KEY}'")

    with pytest.raises(HTTPException) as info:
        _run(_router._run_fetcher("google", boom()))
    assert FAKE_KEY not in str(info.value.detail)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


def test_connection_failure_chain_text_is_scrubbed():
    try:
        try:
            raise httpx.ConnectError(f"refused for 'https://h/v1?key={FAKE_KEY}'")
        except httpx.ConnectError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as outer:
        detail = _describe_connection_failure(outer, worker_name="w", attempts=1)
    assert FAKE_KEY not in detail
    assert "key=***" in detail


def test_chat_stream_error_frame_masks_the_key_and_keeps_provider(client):
    """_chat_router's generic stream-error frame (not BYO, not queue-rejected)."""

    async def fake_stream_text(delta=True):
        yield "partial"
        raise RuntimeError(f"400 for url 'https://h/v1beta/models/m:stream?key={FAKE_KEY}'")

    fake_stream = MagicMock()
    fake_stream.stream_text = lambda delta=True: fake_stream_text(delta=delta)
    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_agent = MagicMock()
    fake_agent.run_stream = MagicMock(return_value=fake_stream_ctx)

    with patch("src.api.providers._chat_router._build_agent", return_value=fake_agent):
        resp = client.post(
            "/chat/completions",
            json={
                "provider": "vllm",
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert FAKE_KEY not in resp.text
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    errors = [f for f in frames if isinstance(f, dict) and "error" in f]
    assert len(errors) == 1, frames
    assert "key=***" in errors[0]["error"]
    assert errors[0]["provider"] == "vllm"
