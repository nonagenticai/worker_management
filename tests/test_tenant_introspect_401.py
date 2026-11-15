"""Regression tests for the tenant.py dependency-path introspect resolver.

Closes the latent "growth_bundle 401 class" #94 flagged but left in
``src.auth.tenant``: a *transient* license-introspection failure must NOT
hard-401 a valid token (which would fail a long run's tail LLM stage). Instead:
serve stale last-known-good when warm, else surface a retryable 503.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.tenant import (
    _introspect_license,
    _license_cache,
    _license_stale_cache,
    _license_token_not_found_since,
    get_tenant_id,
)


def _mock_client(*, status=200, json_body=None, side_effect=None):
    """Build an async-context-manager httpx client mock."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    if side_effect is not None:
        client.post = AsyncMock(side_effect=side_effect)
    else:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = json_body or {}
        client.post = AsyncMock(return_value=resp)
    return client


def _fake_request(token: str | None):
    """Request with empty state (forces Priority-2 introspection) + Bearer token."""
    headers = {"X-Tenant-Id": f"Bearer {token}"} if token is not None else {}
    return SimpleNamespace(state=SimpleNamespace(), headers=headers)


def _clear():
    _license_cache.clear()
    _license_stale_cache.clear()
    _license_token_not_found_since.clear()


# ---------------------------------------------------------------------------
# _introspect_license — stale-while-error semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_active_result_populates_stale_cache():
    _clear()
    client = _mock_client(status=200, json_body={"active": True, "subject_id": "t-1"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=client):
        result = await _introspect_license("good-token")
    assert result == {"active": True, "subject_id": "t-1"}
    # stale cache now warm with the last-known-good ACTIVE verdict
    assert any(v.get("active") for v in _license_stale_cache.values())


@pytest.mark.asyncio(loop_scope="session")
async def test_transient_failure_serves_stale_not_none():
    _clear()
    # 1) prime caches with an active verdict
    ok = _mock_client(status=200, json_body={"active": True, "subject_id": "t-2"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=ok):
        await _introspect_license("tok")
    # 2) positive cache expired but stale warm -> transient blip serves stale
    _license_cache.clear()
    boom = _mock_client(side_effect=Exception("connection refused"))
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=boom):
        result = await _introspect_license("tok")
    assert result == {"active": True, "subject_id": "t-2"}


@pytest.mark.asyncio(loop_scope="session")
async def test_transient_failure_cold_cache_returns_none():
    _clear()
    boom = _mock_client(side_effect=Exception("timeout"))
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=boom):
        result = await _introspect_license("never-seen")
    assert result is None


@pytest.mark.asyncio(loop_scope="session")
async def test_non_200_cold_cache_returns_none():
    _clear()
    client = _mock_client(status=503)
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=client):
        result = await _introspect_license("blip")
    assert result is None


# ---------------------------------------------------------------------------
# get_tenant_id — verdict mapping (the actual 401-vs-503 fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_transient_unavailable_raises_503_not_401():
    _clear()
    boom = _mock_client(side_effect=Exception("licenses unreachable"))
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=boom):
        with pytest.raises(HTTPException) as exc:
            await get_tenant_id(_fake_request("valid-but-unverifiable"))
    assert exc.value.status_code == 503  # retryable, NOT a spurious 401


@pytest.mark.asyncio(loop_scope="session")
async def test_genuinely_inactive_token_still_401():
    _clear()
    client = _mock_client(status=200, json_body={"active": False})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=client):
        with pytest.raises(HTTPException) as exc:
            await get_tenant_id(_fake_request("revoked"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_active_token_returns_tenant():
    _clear()
    client = _mock_client(
        status=200, json_body={"active": True, "subject_id": "subj-1", "tenant_id": "tenant-xyz"}
    )
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=client):
        tenant = await get_tenant_id(_fake_request("good"))
    assert tenant == "tenant-xyz"


@pytest.mark.asyncio(loop_scope="session")
async def test_transient_blip_with_warm_stale_returns_tenant():
    _clear()
    ok = _mock_client(
        status=200, json_body={"active": True, "subject_id": "subj-1", "tenant_id": "warm-tenant"}
    )
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=ok):
        await get_tenant_id(_fake_request("tok2"))
    _license_cache.clear()  # positive cache gone, stale still warm
    boom = _mock_client(side_effect=Exception("blip"))
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=boom):
        tenant = await get_tenant_id(_fake_request("tok2"))
    assert tenant == "warm-tenant"


# ---------------------------------------------------------------------------
# token_not_found rotation-race grace (200 active:false, reason token_not_found)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_token_not_found_warm_stale_serves_last_known_good():
    """A rotation-race token_not_found on a token seen active moments ago serves
    the warm stale ACTIVE verdict — the run continues instead of 401ing."""
    _clear()
    ok = _mock_client(status=200, json_body={"active": True, "subject_id": "rot-tenant"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=ok):
        await _introspect_license("rot-tok")
    _license_cache.clear()  # positive cache gone, stale still warm
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
        result = await _introspect_license("rot-tok")
    assert result == {"active": True, "subject_id": "rot-tenant"}


@pytest.mark.asyncio(loop_scope="session")
async def test_token_not_found_cold_replica_returns_none_503_not_401():
    """A cold replica (no warm stale) hitting a rotation-race token_not_found
    returns None -> the caller gets a retryable 503, NOT an instant 401."""
    _clear()
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
        result = await _introspect_license("cold-tok")
    assert result is None
    # And through the dependency: 503, not 401.
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
        with pytest.raises(HTTPException) as exc:
            await get_tenant_id(_fake_request("cold-tok"))
    assert exc.value.status_code == 503


@pytest.mark.asyncio(loop_scope="session")
async def test_token_not_found_past_window_is_authoritative_401():
    """Once the grace window elapses from the FIRST sighting, a persistent
    token_not_found resolves to the authoritative inactive verdict -> 401."""
    _clear()
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    # First sighting at t=0 (within window -> None), second at t>grace (expired).
    with patch("src.auth.tenant.time.monotonic", side_effect=[0.0, 10_000.0]):
        with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
            first = await _introspect_license("gone-tok")
            second = await _introspect_license("gone-tok")
    assert first is None  # within window: transient
    assert second == {"active": False, "reason": "token_not_found"}  # expired: authoritative
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
        with patch("src.auth.tenant.time.monotonic", return_value=10_001.0):
            with pytest.raises(HTTPException) as exc:
                await get_tenant_id(_fake_request("gone-tok"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_token_invalid_is_not_graced_immediate_401():
    """The grace is reason-SPECIFIC: a revoked/expired token (token_invalid) or an
    inactive license is NOT graced even with a warm stale entry — instant 401."""
    _clear()
    ok = _mock_client(status=200, json_body={"active": True, "subject_id": "was-good"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=ok):
        await _introspect_license("revk-tok")
    _license_cache.clear()  # stale still warm
    revoked = _mock_client(status=200, json_body={"active": False, "reason": "token_invalid"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=revoked):
        result = await _introspect_license("revk-tok")
    assert result == {"active": False, "reason": "token_invalid"}  # NOT stale-served
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=revoked):
        with pytest.raises(HTTPException) as exc:
            await get_tenant_id(_fake_request("revk-tok"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_recovery_resets_token_not_found_window():
    """Recovery to active:true clears the since-entry, so a later flip gets a
    fresh full window rather than an already-elapsed one."""
    _clear()
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    with patch("src.auth.tenant.time.monotonic", side_effect=[0.0]):
        with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
            await _introspect_license("flap-tok")  # records first-seen=0
    assert "lic:" in next(iter(_license_token_not_found_since), "lic:")  # entry present
    ok = _mock_client(status=200, json_body={"active": True, "subject_id": "recovered"})
    with patch("src.auth.tenant.httpx.AsyncClient", return_value=ok):
        await _introspect_license("flap-tok")  # recovery pops the since-entry
    assert len(_license_token_not_found_since) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_persistent_token_not_found_does_not_reopen_window():
    """Regression: a permanently-missing token must SETTLE to authoritative (401)
    after the window, not re-open a fresh grace window. The expired branch re-pins
    the ORIGINAL first-sighting, so the stored value stays at t0 across calls."""
    _clear()
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    with patch("src.auth.tenant.time.monotonic", side_effect=[0.0, 200.0, 400.0]):
        with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
            r1 = await _introspect_license("perma-gone")  # t=0, within window
            r2 = await _introspect_license("perma-gone")  # t=200, expired
            r3 = await _introspect_license("perma-gone")  # t=400, still expired
    assert r1 is None
    assert r2 == {"active": False, "reason": "token_not_found"}
    assert r3 == {"active": False, "reason": "token_not_found"}
    # First-sighting stays pinned at t0 (re-pinned, not reset to 200/400).
    assert list(_license_token_not_found_since.values()) == [0.0]


@pytest.mark.asyncio(loop_scope="session")
async def test_grace_zero_disables_token_not_found_transient():
    """LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS=0 restores the instant-401 behavior."""
    _clear()
    tnf = _mock_client(status=200, json_body={"active": False, "reason": "token_not_found"})
    with patch("src.auth.tenant._TOKEN_NOT_FOUND_GRACE", 0.0):
        with patch("src.auth.tenant.httpx.AsyncClient", return_value=tnf):
            result = await _introspect_license("cold-tok")
    assert result == {"active": False, "reason": "token_not_found"}  # authoritative -> 401
