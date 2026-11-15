"""Unit tests for the trusted-network byo_mode lookup (Fix A).

WM's trusted-network (no-token) S2S path resolves a tenant's byo_mode by
tenant_id via the licenses service so byo_required is enforced for worker-app
runs (requirements/kag/coding → WM) that forward only a raw X-Tenant-Id.
"""

from contextlib import asynccontextmanager

import pytest

from src.api import auth_middleware


class _FakeResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _patch_client(monkeypatch, *, status_code=200, payload=None, raise_exc=None, captured=None):
    @asynccontextmanager
    async def _fake_client(*_a, **_k):
        class _C:
            async def get(self, _url, params=None):
                if captured is not None:
                    captured["url"] = _url
                    captured["params"] = params
                if raise_exc:
                    raise raise_exc
                return _FakeResp(status_code, payload or {})

        yield _C()

    monkeypatch.setattr(auth_middleware.httpx, "AsyncClient", _fake_client)


@pytest.fixture(autouse=True)
def _clear_cache():
    """⛔ ALL THREE caches, not just the positive one.

    ``_tenant_byo_stale_cache`` outlives the positive cache by design (15 min vs
    5), and ``_tenant_byo_unresolved_since`` anchors the grace window. Clearing
    only the first leaves a previous test's last-known-good — or a previous
    test's open failure window — deciding this one's unresolved arm, which is a
    test that passes for the wrong reason in whichever order pytest happens to
    run.
    """
    for cache in (
        auth_middleware._tenant_byo_cache,
        auth_middleware._tenant_byo_stale_cache,
        auth_middleware._tenant_byo_unresolved_since,
    ):
        cache.clear()
    yield
    for cache in (
        auth_middleware._tenant_byo_cache,
        auth_middleware._tenant_byo_stale_cache,
        auth_middleware._tenant_byo_unresolved_since,
    ):
        cache.clear()


@pytest.mark.asyncio
async def test_returns_byo_required_and_caches(monkeypatch):
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"})
    mode = await auth_middleware._lookup_tenant_byo_mode("tenant-1")
    assert mode == "byo_required"
    # Second call is served from cache even if the client now errors.
    _patch_client(monkeypatch, raise_exc=RuntimeError("should not be called"))
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-1") == "byo_required"


@pytest.mark.asyncio
async def test_empty_tenant_is_local_allowed(monkeypatch):
    _patch_client(monkeypatch, raise_exc=RuntimeError("must not call"))
    assert await auth_middleware._lookup_tenant_byo_mode("") == "local_allowed"


@pytest.mark.asyncio
async def test_a_blip_on_an_unknown_tenant_still_proceeds(monkeypatch):
    """⚠️ RENAMED from ``test_fail_open_on_error``; the name was the whole claim.

    The fail-open is no longer unconditional — it is now the grace-window arm of
    ``_unresolved_byo_mode``, reachable only for a tenant this pod holds no
    last-known-good for. That is exactly the case here (the cache fixture just
    cleared), so a licences blip must NOT deny. The arms that changed are pinned
    in ``test_tenant_byo_unresolved_is_not_permission.py``.
    """
    _patch_client(monkeypatch, raise_exc=RuntimeError("licenses down"))
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-x") == "local_allowed"


@pytest.mark.asyncio
async def test_a_non_200_on_an_unknown_tenant_still_proceeds(monkeypatch):
    """⚠️ RENAMED from ``test_fail_open_on_non_200``. Same reason as above."""
    _patch_client(monkeypatch, status_code=404, payload={"detail": "Tenant not found"})
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-missing") == "local_allowed"


@pytest.mark.asyncio
async def test_no_product_sends_no_app_param(monkeypatch):
    captured: dict = {}
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"}, captured=captured)
    await auth_middleware._lookup_tenant_byo_mode("tenant-1")
    # Tenant-wide call carries no ?app param (backward compatible).
    assert captured["params"] is None


@pytest.mark.asyncio
async def test_product_forwards_app_param(monkeypatch):
    captured: dict = {}
    _patch_client(monkeypatch, payload={"byo_mode": "platform_only"}, captured=captured)
    mode = await auth_middleware._lookup_tenant_byo_mode("tenant-1", "code_graph_helper")
    assert mode == "platform_only"
    assert captured["params"] == {"app": "code_graph_helper"}


@pytest.mark.asyncio
async def test_product_and_tenant_wide_cached_separately(monkeypatch):
    # Per-product resolve...
    _patch_client(monkeypatch, payload={"byo_mode": "platform_only"})
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-1", "prod-a") == "platform_only"
    # ...and a tenant-wide resolve for the SAME tenant must not collide in cache.
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"})
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-1") == "byo_required"
    # The per-product value is still cached under its own (tenant, product) key.
    _patch_client(monkeypatch, raise_exc=RuntimeError("should be cached"))
    assert await auth_middleware._lookup_tenant_byo_mode("tenant-1", "prod-a") == "platform_only"


# ---------------------------------------------------------------------------
# A non-200 must be OBSERVABLE, not merely survivable.
#
# `test_fail_open_on_non_200` above already pinned the RETURN VALUE, and it has
# always passed. What nothing pinned is that the decision leaves a trace — so
# the branch that silently disables byo_required enforcement was covered by a
# test and still invisible in production.
#
# Measured on the deployed pods 2026-08-19: 8 of 79 byo-mode lookups answered
# 404 (~10% of S2S runs skipped enforcement). The licences service logged the
# refusal at its end; this service, which ACTS on the answer, logged nothing.
# ---------------------------------------------------------------------------

import logging


@pytest.mark.asyncio
async def test_a_non_200_is_logged_not_just_survived(monkeypatch, caplog):
    _patch_client(monkeypatch, status_code=404, payload={"detail": "Tenant not found"})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert await auth_middleware._lookup_tenant_byo_mode("tenant-missing") == "local_allowed"
    messages = [r.getMessage() for r in caplog.records]
    assert any("byo_mode_lookup_non_200" in m for m in messages), (
        f"a 404 disabled byo_required enforcement and logged NOTHING; records={messages}"
    )


@pytest.mark.asyncio
async def test_the_log_line_names_the_tenant_and_the_status(monkeypatch, caplog):
    """A count cannot answer 'which id failed'. That id is the whole investigation.

    The production occurrences were `1769980520185202330` (not a UUID at all) and
    a UUIDv4 where tenant ids are UUIDv7 — neither is a registered tenant. Without
    the id in the line there is nothing to trace back to a caller.
    """
    _patch_client(monkeypatch, status_code=404, payload={})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        await auth_middleware._lookup_tenant_byo_mode("1769980520185202330", "coding")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "1769980520185202330" in joined
    assert "404" in joined
    assert "coding" in joined


@pytest.mark.asyncio
async def test_a_5xx_is_logged_too(monkeypatch, caplog):
    """404 is not the only silent shape — any non-200 falls through the same way."""
    _patch_client(monkeypatch, status_code=503, payload={})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert await auth_middleware._lookup_tenant_byo_mode("tenant-x") == "local_allowed"
    assert any("byo_mode_lookup_non_200" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_success_stays_silent(monkeypatch, caplog):
    """The discriminating half. A warning on every lookup is not a signal.

    71 of the 79 production lookups succeeded; if those logged too, the 8 that
    matter would be buried in the noise they were supposed to stand out from.
    """
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert await auth_middleware._lookup_tenant_byo_mode("tenant-ok") == "byo_required"
    assert not [r for r in caplog.records if "byo_mode_lookup_non_200" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_single_blip_never_denies_whatever_the_status(monkeypatch):
    """⚠️ RENAMED from ``test_the_fail_open_decision_is_unchanged``.

    The decision IS changed — that is the point of the fix — so a test asserting
    it was unchanged would now be asserting the defect. What must still hold,
    and what this pins, is the property the old test was really protecting: a
    SINGLE non-200, of any shape, must not take out a live worker run.

    ⛔ Note what makes this test honest: every iteration clears all three caches,
    so each status is a genuinely first failure. Reusing one warm window across
    the loop would let iteration 7 pass on iteration 1's timer.
    """
    for status in (400, 401, 403, 404, 500, 502, 503):
        auth_middleware._tenant_byo_cache.clear()
        auth_middleware._tenant_byo_stale_cache.clear()
        auth_middleware._tenant_byo_unresolved_since.clear()
        _patch_client(monkeypatch, status_code=status, payload={})
        assert await auth_middleware._lookup_tenant_byo_mode("t") == "local_allowed"
