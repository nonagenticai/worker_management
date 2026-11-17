"""WM-6: tenant endpoints dial the address that passed the owner-infra check.

The request hook (``refuse_owner_infra_request``) resolved the name and httpx
then resolved it AGAIN at connect, so a name re-pointed between the two (DNS
rebinding) reached owner infrastructure after passing the check. The pinned
transport resolves once, validates every address, and connects to that address
with the original ``Host`` and TLS SNI.

MUTATION: build the BYO clients without ``pinned_tenant_transport`` (the code
before WM-6) ⇒ the inner transport is handed the HOSTNAME, so the rebinding
cases dial it and every case below goes red.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any

import httpx2 as httpx
import pytest

from src.security import request_overrides as ro
from src.security.request_overrides import TenantBaseUrlNotAllowedError

PUBLIC = "93.184.215.14"
PRIVATE = "10.42.0.17"
HOST = "api.vendor-byo.example"
VENDOR = f"https://{HOST}/v1"
TEST_VENDOR_KEY = "sk-t"  # pragma: allowlist secret


@pytest.fixture
def dialled(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Capture what the real (inner) transport would have connected to."""
    seen: list[httpx.Request] = []

    async def fake_send(self: Any, request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "m"}]}, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_send)
    return seen


def _answers(monkeypatch: pytest.MonkeyPatch, *answers: tuple[str, ...]) -> list[str]:
    """Resolver returning ``answers`` in turn (the last one repeats)."""
    calls: list[str] = []

    def _resolve(host: str) -> tuple[str, ...]:
        calls.append(host)
        return answers[min(len(calls), len(answers)) - 1]

    monkeypatch.setattr(ro, "_resolve_host_ips", _resolve)
    ro._dns_cache.clear()
    return calls


def _guarded_client() -> httpx.AsyncClient:
    from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

    cfg = PydanticAIAgentConfig(
        name="t",
        model_provider="openai",
        model="gpt-4o",
        base_url=VENDOR,
        api_key="sk-junk",  # pragma: allowlist secret
        tenant_byo_key=True,
        db_path="",
    )
    client = ModelFactory._byo_guarded_http_client(cfg)
    assert client is not None
    return client


class TestGuardedAgentClient:
    @pytest.mark.asyncio
    async def test_dials_the_validated_ip_with_original_host_and_sni(self, monkeypatch, dialled):
        _answers(monkeypatch, (PUBLIC,))
        client = _guarded_client()
        resp = await client.post(VENDOR + "/chat/completions", json={})
        await client.aclose()
        assert resp.status_code == 200
        assert len(dialled) == 1
        sent = dialled[0]
        assert sent.url.host == PUBLIC
        assert sent.headers["host"] == HOST
        assert sent.extensions.get("sni_hostname") == HOST
        # The caller's request (which httpx builds redirects from) is untouched.
        assert resp.request.url.host == HOST

    @pytest.mark.asyncio
    async def test_rebinding_after_the_hook_check_dials_nothing(self, monkeypatch, dialled):
        """The hook's lookup says public; the next lookup says private. The
        connection must never be made to the name (which the OS would now
        resolve to the private address)."""
        _answers(monkeypatch, (PUBLIC,), (PRIVATE,))
        client = _guarded_client()
        with pytest.raises(TenantBaseUrlNotAllowedError):
            await client.post(VENDOR + "/chat/completions", json={})
        await client.aclose()
        assert dialled == []

    @pytest.mark.asyncio
    async def test_mixed_answer_is_refused(self, monkeypatch, dialled):
        _answers(monkeypatch, (PUBLIC, PRIVATE))
        client = _guarded_client()
        with pytest.raises(TenantBaseUrlNotAllowedError):
            await client.post(VENDOR + "/chat/completions", json={})
        await client.aclose()
        assert dialled == []


class TestModelListFetch:
    """``_httpx_json_get`` for a tenant ``base_url`` (the /providers live check
    and POST /tenant-config/llm/test)."""

    @pytest.mark.asyncio
    async def test_tenant_base_url_fetch_dials_the_validated_ip(self, monkeypatch, dialled):
        from src.api.providers import _router

        _answers(monkeypatch, (PUBLIC,))
        models = await _router._fetch_openai_models(api_key=TEST_VENDOR_KEY, base_url=VENDOR)
        assert models == ["m"]
        assert [r.url.host for r in dialled] == [PUBLIC]
        assert dialled[0].headers["host"] == HOST

    @pytest.mark.asyncio
    async def test_tenant_base_url_resolving_private_is_refused(self, monkeypatch, dialled):
        from src.api.providers import _router

        _answers(monkeypatch, (PRIVATE,))
        with pytest.raises(TenantBaseUrlNotAllowedError):
            await _router._fetch_openai_models(api_key=TEST_VENDOR_KEY, base_url=VENDOR)
        assert dialled == []

    @pytest.mark.asyncio
    async def test_platform_default_url_is_unchanged(self, monkeypatch, dialled):
        """No base_url ⇒ the vendor default, not pinned (call shape unchanged)."""
        from src.api.providers import _router

        calls = _answers(monkeypatch, (PRIVATE,))
        await _router._fetch_openai_models(api_key=TEST_VENDOR_KEY)
        assert [r.url.host for r in dialled] == ["api.openai.com"]
        assert calls == []


class TestPinnedIp:
    def test_literal_rules_still_apply(self):
        with pytest.raises(TenantBaseUrlNotAllowedError):
            ro.pinned_tenant_endpoint_ip("10.0.0.1")
        with pytest.raises(TenantBaseUrlNotAllowedError):
            ro.pinned_tenant_endpoint_ip("inference-router.inference-router.svc.cluster.local")
        assert ro.pinned_tenant_endpoint_ip(PUBLIC) == PUBLIC

    def test_unresolvable_is_refused(self, monkeypatch):
        _answers(monkeypatch, ())
        with pytest.raises(TenantBaseUrlNotAllowedError):
            ro.pinned_tenant_endpoint_ip(HOST)

    def test_prefers_ipv4(self, monkeypatch):
        _answers(monkeypatch, ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", PUBLIC))
        assert ro.pinned_tenant_endpoint_ip(HOST) == PUBLIC


@pytest.mark.asyncio
async def test_real_socket_connects_to_the_pinned_address(monkeypatch):
    """End to end over a real socket: the name does not exist in real DNS, so
    the request can only succeed by dialling the pinned address, and the server
    must see the ORIGINAL Host. Loopback is admitted for this test only."""
    seen: dict[str, str] = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        for line in head.split("\r\n")[1:]:
            if line.lower().startswith("host:"):
                seen["host"] = line.split(":", 1)[1].strip()
        body = b'{"data":[{"id":"m"}]}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    real_reason = ro._ip_owner_infra_reason

    def _allow_loopback(ip: Any, denied: Any = frozenset()) -> str | None:
        return None if ip == ipaddress.ip_address("127.0.0.1") else real_reason(ip, denied)

    monkeypatch.setattr(ro, "_ip_owner_infra_reason", _allow_loopback)
    _answers(monkeypatch, ("127.0.0.1",))
    name = "pinned-only.invalid-tld-wm6.example"
    try:
        async with httpx.AsyncClient(transport=ro.pinned_tenant_transport()) as client:
            resp = await client.get(f"http://{name}:{port}/v1/models")
    finally:
        server.close()
        await server.wait_closed()
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "m"}]}
    assert seen["host"] == f"{name}:{port}"
