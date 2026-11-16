"""Verify a prompt release's ``actor_card_id`` with fr (worker_management#336).

#335 made every move INTO CANARY/PUBLISHED name a Planka card (19 digits,
format only). This module asks fr — the owner of card -> board -> tenant —
whether that card EXISTS and is owned by a PLATFORM tenant, and refuses the
transition otherwise. It runs after the role gates and the format check
(``routes._require_release_card``) and BEFORE any write.

THE PRODUCER CONTRACT (read from fr's code, not from the design doc)
--------------------------------------------------------------------
fr ``backend/src/routes/internal-card-platform-owner.ts`` at fr origin/main
``214cac74b3935f77d7b952ec24ee5fdcb1a9be7d``::

    GET /api/internal/cards/:cardId/platform-owner   X-Internal-Token: <FR_WM_CARD_LOOKUP_TOKEN>
    :131-133  env unset            -> 503 {"error":"card_lookup_unconfigured"}
    :139      bad/absent token     -> 404 {"error":"card_not_found"}   (NOT 401)
    :141      cardId !~ ^[0-9]{19}$ -> 404 card_not_found
    :157-159  missing / unattributable / non-platform card -> 404 card_not_found
    :153-155  Planka / DB error    -> 503 {"error":"card_lookup_unavailable"}
    :161      platform card        -> 200 {"card_id", "tenant_id", "platform": true}

Every 404 is the same body (``NOT_FOUND_BODY``, :60), so WM keys on the
STATUS only. ⚠️ fr answers a WRONG token with 404, so a mis-provisioned token
reads here as ``prompt_release_card_unknown`` (422), not as unverifiable. That
still refuses — fail-closed holds — but the error names the card, not the
credential; check the token first when every release reads "unknown".

WM's mapping (all refusals in WM's ``{"detail": {"error": …}}`` envelope)::

    200, platform:true, tenant_id in WM PLATFORM_TENANT_IDS -> proceed
    200, tenant_id NOT in WM PLATFORM_TENANT_IDS            -> 422 prompt_release_card_not_platform
    404 (any body)                                          -> 422 prompt_release_card_unknown
    5xx / 401 / 403 / any other status / timeout / connect
      error / malformed body / URL or token env unset       -> 503 prompt_release_card_unverifiable

The tenant re-check uses WM's OWN ``PLATFORM_TENANT_IDS`` (never
``COMMERCIAL_PLATFORM_TENANT_IDS``): fr and WM keep two literals of the set,
so a drift becomes a loud 422 instead of a silent grant.

CACHE — the documented 60 s window
----------------------------------
A POSITIVE answer is cached in-process for :data:`CACHE_TTL_SECONDS` (60 s),
keyed by card id. Nothing negative and no error is ever cached. Consequence:
a card moved off a platform board (fr#1418 shows cross-tenant moves exist) is
still accepted by THIS pod for up to 60 s after the move; after that the next
release re-asks fr and is refused. The clock is injectable for the test that
pins this expiry.

The token is never logged; only the card id and the outcome are.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException
from prometheus_client import Counter

from src.api.tenant_constants import platform_tenant_ids
from src.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_TTL_SECONDS",
    "CARD_LOOKUP_PATH",
    "VERIFY_TOTAL",
    "clear_cache",
    "set_clock_for_tests",
    "set_transport_for_tests",
    "verify_release_card",
]

# fr internal-card-platform-owner.ts:3 (route) — appended to FR_CARD_LOOKUP_URL.
CARD_LOOKUP_PATH = "/api/internal/cards/{card_id}/platform-owner"
# fr :135 reads exactly this header.
TOKEN_HEADER = "X-Internal-Token"  # noqa: S105 - a header NAME, not a secret
# 3 s total, 1 s connect, NO retry: the caller retries the whole transition.
TIMEOUT = httpx.Timeout(3.0, connect=1.0)
CACHE_TTL_SECONDS = 60.0
_CACHE_MAX = 1024

NOT_PLATFORM = {"error": "prompt_release_card_not_platform"}
UNKNOWN = {"error": "prompt_release_card_unknown"}
UNVERIFIABLE = {"error": "prompt_release_card_unverifiable"}
_RETRY_AFTER = {"Retry-After": "30"}

VERIFY_TOTAL = Counter(
    "wm_prompt_release_card_verify_total",
    "Prompt-release actor_card_id verifications against fr, by outcome "
    "(ok | unknown | not_platform | unverifiable | cache_hit).",
    ["outcome"],
)


# --- test seams -------------------------------------------------------------
class _Seams:
    """The clock the cache reads and the transport the fr call uses."""

    clock: Callable[[], float] = time.monotonic
    transport: httpx.AsyncBaseTransport | None = None


# card_id -> (tenant_id, expires_at on _Seams.clock)
_cache: dict[str, tuple[str, float]] = {}


def set_clock_for_tests(clock: Callable[[], float] | None) -> None:
    """Swap the cache clock (``None`` restores ``time.monotonic``)."""
    _Seams.clock = clock or time.monotonic


def set_transport_for_tests(transport: httpx.AsyncBaseTransport | None) -> None:
    """Route the fr call through ``transport`` (``None`` restores the network)."""
    _Seams.transport = transport


def clear_cache() -> None:
    """Drop every cached positive answer."""
    _cache.clear()


def _cache_get(card_id: str) -> str | None:
    hit = _cache.get(card_id)
    if hit is None:
        return None
    tenant_id, expires_at = hit
    if _Seams.clock() >= expires_at:
        _cache.pop(card_id, None)
        return None
    return tenant_id


def _cache_put(card_id: str, tenant_id: str) -> None:
    if len(_cache) >= _CACHE_MAX:
        now = _Seams.clock()
        for key in [k for k, (_, exp) in _cache.items() if now >= exp]:
            _cache.pop(key, None)
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)), None)
    _cache[card_id] = (tenant_id, _Seams.clock() + CACHE_TTL_SECONDS)


def _refuse(
    card_id: str, outcome: str, status: int, detail: dict[str, str], why: str
) -> HTTPException:
    VERIFY_TOTAL.labels(outcome=outcome).inc()
    logger.warning("prompt release card %s: %s (%s)", card_id, outcome, why)
    headers = _RETRY_AFTER if status == 503 else None
    return HTTPException(status_code=status, detail=detail, headers=headers)


def _platform_tenant(body: Any, card_id: str) -> str | None:
    """The tenant of a well-formed fr 200 body (fr :161), else ``None``."""
    if not isinstance(body, dict):
        return None
    tenant_id = body.get("tenant_id")
    if body.get("platform") is not True or body.get("card_id") != card_id:
        return None
    if not isinstance(tenant_id, str) or not tenant_id:
        return None
    return tenant_id


async def verify_release_card(card_id: str) -> None:
    """Return when fr confirms ``card_id`` is a platform card; raise otherwise.

    ``card_id`` has already passed the 19-digit format check.
    """
    cached = _cache_get(card_id)
    if cached is not None:
        VERIFY_TOTAL.labels(outcome="cache_hit").inc()
        logger.info("prompt release card %s: cache_hit", card_id)
        return

    settings = get_settings()
    base = settings.fr_card_lookup_url.strip().rstrip("/")
    token = settings.fr_wm_card_lookup_token.strip()
    if not base or not token:
        raise _refuse(card_id, "unverifiable", 503, UNVERIFIABLE, "lookup url/token unset")

    url = base + CARD_LOOKUP_PATH.format(card_id=card_id)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=_Seams.transport) as client:
            resp = await client.get(url, headers={TOKEN_HEADER: token})
    except httpx.HTTPError as exc:
        # Class name only: an httpx error string can carry the request.
        raise _refuse(card_id, "unverifiable", 503, UNVERIFIABLE, type(exc).__name__) from None

    if resp.status_code == 404:
        raise _refuse(card_id, "unknown", 422, UNKNOWN, "fr 404")
    if resp.status_code != 200:
        raise _refuse(card_id, "unverifiable", 503, UNVERIFIABLE, f"fr {resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        raise _refuse(card_id, "unverifiable", 503, UNVERIFIABLE, "fr body not json") from None
    tenant_id = _platform_tenant(body, card_id)
    if tenant_id is None:
        raise _refuse(card_id, "unverifiable", 503, UNVERIFIABLE, "fr body malformed")
    # WM's OWN platform set — a drift from fr's literal is a refusal, not a grant.
    if tenant_id.strip().lower() not in platform_tenant_ids():
        logger.error(
            "prompt release card %s: fr says platform, WM set disagrees (list drift)", card_id
        )
        raise _refuse(card_id, "not_platform", 422, NOT_PLATFORM, "tenant not in WM platform set")

    _cache_put(card_id, tenant_id)
    VERIFY_TOTAL.labels(outcome="ok").inc()
    logger.info("prompt release card %s: ok", card_id)
