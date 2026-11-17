"""Short-lived per-tenant inference-router bundle tokens (plan §2.4, ruling 1.0.3).

Owner ruling: *use short-lived tokens, no pasting*. worker_management mints a
≤60-min token per tenant from licenses
``POST /internal/tenants/{t}/bundle-token?app=inference-router&ttl_minutes=60``
(realm role ``bundle-token-minter``, SA client ``worker-management-bundle-minter``),
holds it IN MEMORY ONLY until ``expires_at - 5 min``, single-flights the mint per
tenant, and never logs or persists it. A refusal (404/409/5xx) means "no header",
never an error on the agent hot path.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2 as httpx
import pytest

from src.services import ir_bundle_tokens as bt

pytestmark = pytest.mark.asyncio

TENANT_A = "019edc1e-0000-7000-8000-00000000000a"
TENANT_B = "019edc1e-0000-7000-8000-00000000000b"
LICENSES = "http://licenses.licenses.svc.cluster.local:8544"
TOKEN_URL = "https://kc.example/realms/secure-apps/protocol/openid-connect/token"
SECRET_TOKEN = "lt_SECRET-tenant-token-never-logged"  # pragma: allowlist secret
SA_JWT = "eyJ.SA-JWT-must-never-be-logged.sig"  # pragma: allowlist secret


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC).timestamp()

    def __call__(self) -> float:
        return self.now


class _Licenses:
    """A fake licenses + Keycloak pair behind one httpx.MockTransport."""

    def __init__(self, clock: _Clock, *, status: int = 201, ttl_s: int = 3600) -> None:
        self.clock = clock
        self.status = status
        self.ttl_s = ttl_s
        self.mints: list[httpx.Request] = []
        self.sa_mints = 0
        self.delay = 0.0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.sa_mints += 1
            return httpx.Response(200, json={"access_token": SA_JWT, "expires_in": 300})
        self.mints.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status != 201:
            return httpx.Response(self.status, json={"detail": {"error": "no_bundle"}})
        exp = datetime.fromtimestamp(self.clock.now + self.ttl_s, tz=UTC)
        tenant = request.url.path.split("/")[3]
        return httpx.Response(
            201,
            json={
                "token": f"{SECRET_TOKEN}-{tenant}-{len(self.mints)}",
                "expires_at": exp.isoformat(),
                "tenant_id": tenant,
                "bundle": "non-agentic-ai",
                "app": "inference-router",
                "license_id": "lic-1",
                "token_id": f"tok-{len(self.mints)}",
            },
        )


def _cache(lic: _Licenses, clock: _Clock) -> bt.BundleTokenCache:
    return bt.BundleTokenCache(
        licenses_url=LICENSES,
        token_url=TOKEN_URL,
        client_id="worker-management-bundle-minter",
        client_secret="s3cret",  # pragma: allowlist secret
        transport=httpx.MockTransport(lic.handler),
        clock=clock,
    )


def _outcome(outcome: str) -> float:
    return bt.BUNDLE_TOKEN_TOTAL.labels(outcome=outcome)._value.get()


async def test_mint_request_shape_matches_the_licenses_route() -> None:
    clock = _Clock()
    lic = _Licenses(clock)
    token = await _cache(lic, clock).get(TENANT_A)
    assert token is not None and token.startswith(SECRET_TOKEN)
    (req,) = lic.mints
    assert req.method == "POST"
    assert req.url.path == f"/internal/tenants/{TENANT_A}/bundle-token"
    assert dict(req.url.params) == {"app": "inference-router", "ttl_minutes": "60"}
    # The SA JWT from the dedicated minter client, as a plain bearer.
    assert req.headers["Authorization"] == f"Bearer {SA_JWT}"


async def test_token_is_reused_until_expiry_minus_five_minutes() -> None:
    clock = _Clock()
    lic = _Licenses(clock, ttl_s=3600)
    cache = _cache(lic, clock)
    first = await cache.get(TENANT_A)
    clock.now += 3600 - 301  # 5 min + 1 s before expiry: still reused
    assert await cache.get(TENANT_A) == first
    assert len(lic.mints) == 1
    clock.now += 2  # now inside the last 5 min: re-minted
    second = await cache.get(TENANT_A)
    assert second != first
    assert len(lic.mints) == 2


async def test_tokens_are_per_tenant() -> None:
    clock = _Clock()
    lic = _Licenses(clock)
    cache = _cache(lic, clock)
    a = await cache.get(TENANT_A)
    b = await cache.get(TENANT_B)
    assert a != b
    assert TENANT_A in str(a) and TENANT_B in str(b)
    assert len(lic.mints) == 2


async def test_concurrent_callers_single_flight_one_mint_per_tenant() -> None:
    clock = _Clock()
    lic = _Licenses(clock)
    lic.delay = 0.05
    cache = _cache(lic, clock)
    results = await asyncio.gather(*(cache.get(TENANT_A) for _ in range(20)))
    assert len(set(results)) == 1
    assert len(lic.mints) == 1


async def test_single_flight_does_not_serialise_different_tenants() -> None:
    clock = _Clock()
    lic = _Licenses(clock)
    lic.delay = 0.05
    cache = _cache(lic, clock)
    await asyncio.gather(*(cache.get(t) for t in (TENANT_A, TENANT_B) for _ in range(5)))
    assert len(lic.mints) == 2


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(404, "not_found"), (409, "conflict"), (500, "server_error"), (503, "server_error")],
)
async def test_refusal_returns_none_and_counts_outcome(status: int, outcome: str) -> None:
    clock = _Clock()
    lic = _Licenses(clock, status=status)
    before = _outcome(outcome)
    assert await _cache(lic, clock).get(TENANT_A) is None
    assert _outcome(outcome) == before + 1


async def test_refusal_is_negative_cached_briefly_then_retried() -> None:
    """A tenant with no bundle must not turn every IR call into a licenses call."""
    clock = _Clock()
    lic = _Licenses(clock, status=404)
    cache = _cache(lic, clock)
    assert await cache.get(TENANT_A) is None
    assert await cache.get(TENANT_A) is None
    assert len(lic.mints) == 1
    clock.now += bt.NEGATIVE_TTL_S + 1
    lic.status = 201
    assert await cache.get(TENANT_A) is not None
    assert len(lic.mints) == 2


async def test_transport_failure_is_no_token_not_an_exception() -> None:
    async def boom(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": SA_JWT, "expires_in": 300})
        raise httpx.ConnectError("refused", request=request)

    clock = _Clock()
    cache = bt.BundleTokenCache(
        licenses_url=LICENSES,
        token_url=TOKEN_URL,
        client_id="c",
        client_secret="s",  # pragma: allowlist secret
        transport=httpx.MockTransport(boom),
        clock=clock,
    )
    before = _outcome("transport_error")
    assert await cache.get(TENANT_A) is None
    assert _outcome("transport_error") == before + 1


async def test_sa_mint_failure_is_no_token() -> None:
    async def kc_down(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(401, json={"error": "unauthorized_client"})
        raise AssertionError("licenses must not be called without an SA token")

    clock = _Clock()
    cache = bt.BundleTokenCache(
        licenses_url=LICENSES,
        token_url=TOKEN_URL,
        client_id="c",
        client_secret="s",  # pragma: allowlist secret
        transport=httpx.MockTransport(kc_down),
        clock=clock,
    )
    before = _outcome("sa_error")
    assert await cache.get(TENANT_A) is None
    assert _outcome("sa_error") == before + 1


async def test_ok_mint_is_counted() -> None:
    clock = _Clock()
    before = _outcome("ok")
    await _cache(_Licenses(clock), clock).get(TENANT_A)
    assert _outcome("ok") == before + 1


async def test_caplog_never_contains_a_token(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    clock = _Clock()
    lic = _Licenses(clock)
    cache = _cache(lic, clock)
    await cache.get(TENANT_A)
    await cache.get(TENANT_A)
    lic.status = 404
    await cache.get(TENANT_B)
    text = caplog.text + " ".join(str(r.args) for r in caplog.records)
    assert "wm_ir_bundle_token" in text  # it does log — just never the secret
    assert SECRET_TOKEN not in text
    assert SA_JWT not in text
    assert "s3cret" not in text


async def test_token_shorter_than_the_lead_is_used_but_not_cached() -> None:
    """licenses caps expiry at the licence's own expiry, which can be < 5 min away."""
    clock = _Clock()
    lic = _Licenses(clock, ttl_s=120)
    cache = _cache(lic, clock)
    assert await cache.get(TENANT_A) is not None
    assert await cache.get(TENANT_A) is not None
    assert len(lic.mints) == 2


async def test_not_persisted_anywhere_but_the_instance() -> None:
    """Two caches share nothing: the token lives in process memory only."""
    clock = _Clock()
    lic = _Licenses(clock)
    await _cache(lic, clock).get(TENANT_A)
    await _cache(lic, clock).get(TENANT_A)
    assert len(lic.mints) == 2


async def test_tenant_bearer_token_is_none_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WM_IR_BUNDLE_TOKENS", raising=False)
    monkeypatch.setenv("LICENSES_INTERNAL_URL", LICENSES)
    monkeypatch.setenv("KEYCLOAK_TOKEN_URL", TOKEN_URL)
    monkeypatch.setenv("BUNDLE_MINTER_CLIENT_ID", "c")
    monkeypatch.setenv("BUNDLE_MINTER_CLIENT_SECRET", "s")
    bt.reset_bundle_token_cache()

    called: list[Any] = []

    async def _never(self: Any, tenant_id: str) -> str:
        called.append(tenant_id)
        return "x"

    monkeypatch.setattr(bt.BundleTokenCache, "get", _never)
    assert await bt.tenant_bearer_token(TENANT_A) is None
    assert called == []


async def test_tenant_bearer_token_is_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
    monkeypatch.delenv("LICENSES_INTERNAL_URL", raising=False)
    bt.reset_bundle_token_cache()
    before = _outcome("not_configured")
    assert await bt.tenant_bearer_token(TENANT_A) is None
    assert _outcome("not_configured") == before + 1


async def test_settings_flags_default_off() -> None:
    from src.settings import Settings

    s = Settings()
    assert s.wm_ir_bundle_tokens is False
    assert s.wm_tier_alias_passthrough is False


def test_expires_at_naive_is_read_as_utc() -> None:
    aware = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)
    assert bt._parse_expires_at(naive.isoformat()) == aware.timestamp()
    assert bt._parse_expires_at((aware + timedelta(hours=1)).isoformat()) == (
        aware.timestamp() + 3600
    )


# --------------------------------------------------------------------------
# 409 ambiguous_bundle: disambiguate by the calling app's bundle, never guess.
# --------------------------------------------------------------------------


class _AmbiguousLicenses(_Licenses):
    """licences answering 409 ambiguous_bundle unless ``&bundle=`` names a held one."""

    def __init__(self, clock: _Clock, bundles: list[str]) -> None:
        super().__init__(clock)
        self.bundles = bundles

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return await super().handler(request)
        bundle = request.url.params.get("bundle")
        if bundle is None:
            self.mints.append(request)
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": "ambiguous_bundle",
                        "reason": "app_in_several_held_bundles",
                        "bundles": list(self.bundles),
                    }
                },
            )
        if bundle not in self.bundles:
            self.mints.append(request)
            return httpx.Response(404, json={"detail": {"error": "no_bundle"}})
        return await super().handler(request)


def _ambiguous_cache(
    lic: _Licenses, clock: _Clock, by_app: dict[str, str] | None = None
) -> bt.BundleTokenCache:
    return bt.BundleTokenCache(
        licenses_url=LICENSES,
        token_url=TOKEN_URL,
        client_id="worker-management-bundle-minter",
        client_secret="s3cret",  # pragma: allowlist secret
        transport=httpx.MockTransport(lic.handler),
        clock=clock,
        bundle_by_app=by_app,
    )


async def test_unambiguous_mint_sends_no_bundle_param_even_with_an_app() -> None:
    clock = _Clock()
    lic = _Licenses(clock)
    cache = _ambiguous_cache(lic, clock, {"w2v": "conveyor"})
    token = await cache.get(TENANT_A, "w2v")
    assert token is not None
    (req,) = lic.mints
    assert dict(req.url.params) == {"app": "inference-router", "ttl_minutes": "60"}
    # One tenant entry serves every app, exactly as before.
    assert await cache.get(TENANT_A, "kag") == token
    assert len(lic.mints) == 1


async def test_409_with_configured_app_bundle_retries_once_with_bundle_param() -> None:
    clock = _Clock()
    lic = _AmbiguousLicenses(clock, ["conveyor", "inference-router"])
    cache = _ambiguous_cache(lic, clock, {"w2v": "conveyor"})
    before = _outcome("ok_disambiguated")
    token = await cache.get(TENANT_A, "w2v")
    assert token is not None and token.startswith(SECRET_TOKEN)
    assert [dict(r.url.params) for r in lic.mints] == [
        {"app": "inference-router", "ttl_minutes": "60"},
        {"app": "inference-router", "ttl_minutes": "60", "bundle": "conveyor"},
    ]
    assert _outcome("ok_disambiguated") == before + 1
    # Cached like a success -- also past the 60 s ambiguity record.
    clock.now += bt.NEGATIVE_TTL_S + 1
    assert await cache.get(TENANT_A, "w2v") == token
    assert len(lic.mints) == 2


async def test_409_mapped_bundle_beats_an_app_key_that_is_also_a_held_bundle() -> None:
    """X-App-Id ``w2v`` is ALSO a bundle key, and the w2v bundle reaches
    inference-router through app_requires. The map must win: before licences
    #282/#283 WM's IR mints for this tenant resolved ``bundle=conveyor``, so
    falling back to the ``w2v`` bundle would re-attribute the spend.
    """
    clock = _Clock()
    lic = _AmbiguousLicenses(clock, ["conveyor", "w2v", "inference-router"])
    cache = _ambiguous_cache(lic, clock, {"w2v": "conveyor", "kag": "conveyor"})
    assert await cache.get(TENANT_A, "w2v") is not None
    assert lic.mints[-1].url.params["bundle"] == "conveyor"
    # kag (ks-relationship-extractor's caller) pins the same entry: one mint.
    assert await cache.get(TENANT_A, "kag") is not None
    assert [r.url.params.get("bundle") for r in lic.mints] == [None, "conveyor"]


async def test_409_exact_app_key_match_when_unmapped() -> None:
    clock = _Clock()
    lic = _AmbiguousLicenses(clock, ["conveyor", "inference-router"])
    cache = _ambiguous_cache(lic, clock)
    assert await cache.get(TENANT_A, "conveyor") is not None
    assert lic.mints[-1].url.params["bundle"] == "conveyor"


async def test_409_without_match_is_none_and_warning_names_the_bundles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    lic = _AmbiguousLicenses(clock, ["conveyor", "inference-router"])
    cache = _ambiguous_cache(lic, clock, {"kag": "kag-research"})
    before = _outcome("ambiguous_unresolved")
    with caplog.at_level(logging.WARNING, logger=bt.__name__):
        assert await cache.get(TENANT_A, "w2v") is None
        assert await cache.get(TENANT_A, "kag") is None  # mapped, but not held
        assert await cache.get(TENANT_A, "") is None  # no X-App-Id at all
    # One mint, no bundle retry: the 409 is negative-cached like any refusal.
    assert len(lic.mints) == 1
    assert "bundle" not in lic.mints[0].url.params
    assert _outcome("ambiguous_unresolved") == before + 1
    text = caplog.text
    assert "outcome=conflict" in text and "error=ambiguous_bundle" in text
    assert "bundles=conveyor,inference-router" in text
    assert SECRET_TOKEN not in text and SA_JWT not in text


async def test_409_other_conflict_never_retries() -> None:
    """ambiguous_app_licence (or a 409 without bundles) is not ours to resolve."""
    clock = _Clock()
    lic = _Licenses(clock, status=409)
    cache = _ambiguous_cache(lic, clock, {"w2v": "conveyor"})
    assert await cache.get(TENANT_A, "w2v") is None
    assert len(lic.mints) == 1


async def test_409_bundle_listed_twice_is_not_a_match() -> None:
    clock = _Clock()
    lic = _AmbiguousLicenses(clock, ["conveyor", "conveyor"])
    cache = _ambiguous_cache(lic, clock, {"w2v": "conveyor"})
    assert await cache.get(TENANT_A, "w2v") is None
    assert len(lic.mints) == 1


async def test_tenant_bearer_token_passes_the_inbound_app_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.identity_context import set_app_id_context

    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
    monkeypatch.setenv("LICENSES_INTERNAL_URL", LICENSES)
    monkeypatch.setenv("KEYCLOAK_TOKEN_URL", TOKEN_URL)
    monkeypatch.setenv("BUNDLE_MINTER_CLIENT_ID", "c")
    monkeypatch.setenv("BUNDLE_MINTER_CLIENT_SECRET", "s")
    monkeypatch.setenv("WM_IR_BUNDLE_BY_APP", '{"w2v": "conveyor"}')
    from src.settings import reset_settings_cache

    reset_settings_cache()
    bt.reset_bundle_token_cache()
    seen: list[tuple[str, str]] = []

    async def _get(self: Any, tenant_id: str, app_key: str = "") -> str:
        seen.append((tenant_id, app_key))
        assert self._bundle_by_app == {"w2v": "conveyor"}
        return "tok"

    monkeypatch.setattr(bt.BundleTokenCache, "get", _get)
    set_app_id_context("w2v")
    try:
        assert await bt.tenant_bearer_token(TENANT_A) == "tok"
    finally:
        set_app_id_context("")
        bt.reset_bundle_token_cache()
        reset_settings_cache()
    assert seen == [(TENANT_A, "w2v")]


def test_bundle_by_app_setting_parse() -> None:
    from src.settings import Settings, parse_ir_bundle_by_app

    assert parse_ir_bundle_by_app("") == {}
    assert parse_ir_bundle_by_app("  ") == {}
    assert parse_ir_bundle_by_app('{"w2v": "conveyor", "kag": "kag-research"}') == {
        "w2v": "conveyor",
        "kag": "kag-research",
    }
    for bad in ("not json", '["w2v"]', '{"w2v": 1}', '{"W2V": "conveyor"}', '{"w2v": ""}'):
        with pytest.raises(ValueError):
            parse_ir_bundle_by_app(bad)
    assert Settings().wm_ir_bundle_by_app == ""
    with pytest.raises(RuntimeError, match="WM_IR_BUNDLE_BY_APP"):
        Settings(wm_ir_bundle_by_app="{nope")
