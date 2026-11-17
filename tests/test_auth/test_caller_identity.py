"""Tests for src/auth/caller_identity.py.

Token cache mint/refresh/single-flight, httpx hook host filtering, and
require_role acceptance/rejection. Uses httpx.MockTransport (project
convention; respx is not a dep) and direct ContextVar manipulation for the
require_role tests.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import httpx2 as httpx
import pytest
from fastapi import HTTPException, Request

from src.api.identity_context import set_caller_context
from src.auth.caller_identity import (
    CallerTokenCache,
    CallerTokenError,
    make_caller_request_hook,
    require_role,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# require_role's inner _dep takes a Request but our tests don't need a real
# one — the dep reads its data from a ContextVar. cast a sentinel to keep
# ty happy without baking a fixture client into pure unit tests.
_FAKE_REQUEST = cast("Request", cast("Any", None))

TOKEN_URL = "http://keycloak.test/realms/secure-apps/protocol/openid-connect/token"

asyncio_test = pytest.mark.asyncio


def _token_resp(access_token: str = "tok-1", expires_in: int = 300) -> dict:  # noqa: S107  # test fixture, not a real credential
    return {"access_token": access_token, "expires_in": expires_in, "token_type": "Bearer"}


def _make_token_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_outbound_client(
    handler: Callable[[httpx.Request], httpx.Response],
    hook: Callable,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [hook]},
    )


# -- CallerTokenCache --------------------------------------------------------


@asyncio_test
async def test_get_token_mints_and_caches():
    calls = {"n": 0}

    def _handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_token_resp())

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_handler))
    try:
        t1 = await cache.get_token("svc", "secret")
        t2 = await cache.get_token("svc", "secret")
    finally:
        await cache.aclose()
    assert t1 == "tok-1"
    assert t2 == "tok-1"
    assert calls["n"] == 1


@asyncio_test
async def test_get_token_refreshes_after_expiry():
    responses = iter(
        [
            httpx.Response(200, json=_token_resp("tok-old", expires_in=31)),
            httpx.Response(200, json=_token_resp("tok-new", expires_in=300)),
        ],
    )

    def _handler(_req: httpx.Request) -> httpx.Response:
        return next(responses)

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_handler))
    try:
        t1 = await cache.get_token("svc", "secret")
        with patch("src.auth.caller_identity.time.monotonic", return_value=time.monotonic() + 60):
            t2 = await cache.get_token("svc", "secret")
    finally:
        await cache.aclose()
    assert t1 == "tok-old"
    assert t2 == "tok-new"


@asyncio_test
async def test_concurrent_refresh_single_flights():
    """10 concurrent get_token() calls → exactly one Keycloak POST."""
    calls = {"n": 0}

    async def _handler_async(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # Sleep so concurrent tasks pile up on the lock — proves single-flight.
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=_token_resp())

    cache = CallerTokenCache(
        token_url=TOKEN_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler_async)),
    )
    try:
        results = await asyncio.gather(*(cache.get_token("svc", "secret") for _ in range(10)))
    finally:
        await cache.aclose()
    assert all(r == "tok-1" for r in results)
    assert calls["n"] == 1


@asyncio_test
async def test_keycloak_5xx_raises():
    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_handler))
    try:
        with pytest.raises(CallerTokenError):
            await cache.get_token("svc", "secret")
    finally:
        await cache.aclose()


@asyncio_test
async def test_keycloak_response_missing_access_token_raises():
    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"expires_in": 300})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_handler))
    try:
        with pytest.raises(CallerTokenError):
            await cache.get_token("svc", "secret")
    finally:
        await cache.aclose()


def test_empty_token_url_raises():
    with pytest.raises(ValueError, match="non-empty token_url"):
        CallerTokenCache(token_url="")


# -- httpx event hook --------------------------------------------------------


@asyncio_test
async def test_hook_injects_header_for_internal_host():
    def _token_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_token_resp("inner-tok"))

    captured: dict[str, str] = {}

    def _outbound_handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("X-Caller-Authorization", "")
        return httpx.Response(200, json={})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_token_handler))
    hook = make_caller_request_hook(cache, "svc", "secret")
    try:
        async with _make_outbound_client(_outbound_handler, hook) as client:
            await client.get("http://service.internal/api/x")
    finally:
        await cache.aclose()
    assert captured["auth"] == "Bearer inner-tok"


@asyncio_test
async def test_hook_skips_external_host():
    """SA tokens must never leak to external services."""

    def _token_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_token_resp())

    captured: dict[str, str] = {}

    def _outbound_handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("X-Caller-Authorization", "")
        return httpx.Response(200, json={})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_token_handler))
    hook = make_caller_request_hook(cache, "svc", "secret")
    try:
        async with _make_outbound_client(_outbound_handler, hook) as client:
            await client.get("https://api.example.com/x")
    finally:
        await cache.aclose()
    assert captured["auth"] == ""


@asyncio_test
async def test_hook_no_op_when_credentials_empty():
    """Empty client_id / client_secret = feature disabled."""

    def _token_handler(_req: httpx.Request) -> httpx.Response:
        # Should NOT be called when credentials are empty.
        return httpx.Response(200, json=_token_resp())

    captured: dict[str, str] = {}

    def _outbound_handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("X-Caller-Authorization", "")
        return httpx.Response(200, json={})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_token_handler))
    hook = make_caller_request_hook(cache, "", "")
    try:
        async with _make_outbound_client(_outbound_handler, hook) as client:
            await client.get("http://service.internal/x")
    finally:
        await cache.aclose()
    assert captured["auth"] == ""


@asyncio_test
async def test_hook_failsoft_on_token_mint_error():
    """Token mint failure must NOT 5xx the user request — proceed without header."""

    def _token_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    captured: dict[str, str] = {}

    def _outbound_handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("X-Caller-Authorization", "")
        return httpx.Response(200, json={})

    cache = CallerTokenCache(token_url=TOKEN_URL, http_client=_make_token_client(_token_handler))
    hook = make_caller_request_hook(cache, "svc", "secret")
    try:
        async with _make_outbound_client(_outbound_handler, hook) as client:
            resp = await client.get("http://service.internal/x")
    finally:
        await cache.aclose()
    assert resp.status_code == 200
    assert captured["auth"] == ""


# -- require_role ------------------------------------------------------------


def test_require_role_passes_when_role_present():
    set_caller_context("some-sa", ("secrets-reader", "secrets-writer"))
    dep = require_role("secrets-writer")
    dep(_FAKE_REQUEST)


def test_require_role_raises_when_role_absent():
    set_caller_context("some-sa", ("secrets-reader",))
    dep = require_role("secrets-writer")
    with pytest.raises(HTTPException) as exc:
        dep(_FAKE_REQUEST)
    assert exc.value.status_code == 403
    assert exc.value.detail == "requires_role:secrets-writer"


def test_require_role_raises_when_no_caller_context():
    set_caller_context("", ())
    dep = require_role("secrets-writer")
    with pytest.raises(HTTPException) as exc:
        dep(_FAKE_REQUEST)
    assert exc.value.status_code == 403
