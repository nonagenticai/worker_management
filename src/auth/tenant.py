"""Tenant + per-user isolation for worker management service.

Uses license token (`X-Tenant-Id: Bearer <token>`) as the ONLY auth method.
"""

import hashlib
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

import httpx
from cachetools import TTLCache
from fastapi import HTTPException, Request

from src.repositories.visibility import can_mutate
from src.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# License introspection caches. Positive (5 min) + stale-while-error (15 min):
# the stale cache records the last-known-good ACTIVE result so a transient
# licenses blip (timeout / 5xx) serves last-known-good instead of a spurious
# 401. Mirrors AuthenticationMiddleware (#94); this dependency-path resolver is
# the twin #94 flagged but left for follow-up.
_license_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_license_stale_cache: TTLCache = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_STALE_TTL", "900"))
)
# The licenses `/license/introspect` endpoint returns 200 {active:false} for a
# genuinely bad token (never a 4xx), so any timeout/connection-error/non-200
# here is an INFRA failure, not a token verdict. 2s->10s (a 2s timeout under
# load was conflated with an invalid token — the growth_bundle 401 class).
_INTROSPECT_TIMEOUT = float(os.getenv("LICENSE_INTROSPECT_TIMEOUT", "10.0"))

# A 200 {active:false, reason:"token_not_found"} during an ADDITIVE token
# rotation is a propagation race (new token row not yet committed/replicated, or
# a stale Secret value), NOT a rejection. Only "token_not_found" is graced —
# revoked/deactivated tokens return "token_invalid" / "license_inactive" and
# still 401 immediately. Served from warm stale if present, else None (→ 503).
# Bounded from the first sighting; 0 disables. Mirrors AuthenticationMiddleware.
_TOKEN_NOT_FOUND_GRACE = float(os.getenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "120"))
_license_token_not_found_since: TTLCache = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_STALE_TTL", "900"))
)


async def _introspect_license(token: str) -> dict[str, Any] | None:
    """Introspect a license token to get tenant context.

    Args:
        token: The license token (without Bearer prefix).

    Returns:
        The authoritative introspection dict (``active`` True or False) on a 200
        response; the stale last-known-good ACTIVE dict on a transient failure
        with a warm stale cache; or ``None`` when introspection is transiently
        UNAVAILABLE and no stale entry exists. ``None`` therefore means "could
        not reach a verdict" (licenses slow/unreachable) — NEVER "token
        rejected" (a bad token comes back as a 200 ``active:false`` dict). The
        caller maps ``None`` to a retryable 503, not a 401.
    """
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    cached = _license_cache.get(cache_key)
    if cached is not None:
        return cast("dict[str, Any]", cached)

    try:
        async with httpx.AsyncClient(timeout=_INTROSPECT_TIMEOUT) as client:
            resp = await client.post(get_settings().license_introspect_url, json={"token": token})
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                if data.get("active"):
                    _license_cache[cache_key] = data
                    # Record the good result for stale-while-error fallback.
                    _license_stale_cache[cache_key] = data
                    # Recovery: a later token_not_found flip gets a fresh window.
                    _license_token_not_found_since.pop(cache_key, None)
                    return data
                # 200 active:false is authoritative for a genuinely bad token.
                # EXCEPT "token_not_found" during an additive rotation is a
                # propagation race — ride it out as transient, bounded from the
                # first sighting.
                if data.get("reason") == "token_not_found" and _TOKEN_NOT_FOUND_GRACE > 0:
                    now = time.monotonic()
                    first = _license_token_not_found_since.get(cache_key)
                    if first is None:
                        _license_token_not_found_since[cache_key] = first = now
                    if now - first < _TOKEN_NOT_FOUND_GRACE:
                        logger.warning(
                            "AUDIT introspect token_not_found treated as transient "
                            "(rotation race) cache_key=%s elapsed=%.0fs window=%.0fs",
                            cache_key,
                            now - first,
                            _TOKEN_NOT_FOUND_GRACE,
                        )
                        stale = _license_stale_cache.get(cache_key)
                        if stale is not None:
                            # Warm replica: serve last-known-good.
                            return cast("dict[str, Any]", stale)
                        # Cold replica (no warm stale): None -> caller retryable 503.
                        return None
                    # Window elapsed: re-pin the ORIGINAL first-sighting so its
                    # TTL refreshes under continuous traffic. Otherwise the
                    # since-entry evicts at its TTL and the next token_not_found
                    # re-opens a fresh window — a permanently-missing token would
                    # flap 503<->401 forever instead of settling to 401.
                    _license_token_not_found_since[cache_key] = first
                    logger.warning(
                        "introspect token_not_found grace expired cache_key=%s "
                        "window=%.0fs; returning authoritative inactive",
                        cache_key,
                        _TOKEN_NOT_FOUND_GRACE,
                    )
                return data
            logger.warning("License introspection returned status %d", resp.status_code)
    except Exception as e:
        logger.warning(f"License introspection failed: {e}")

    # Transient failure (timeout / connection error / non-200 — the endpoint
    # returns 200 for every token verdict, so this is always infra, never a
    # rejection). Serve last-known-good if we have one so a briefly slow /
    # unreachable licenses service does not spuriously 401 a token that was
    # valid moments ago; else return None (caller -> retryable 503).
    stale = _license_stale_cache.get(cache_key)
    if stale is not None:
        logger.warning(
            "License introspection unavailable; serving stale last-known-good (active=%s)",
            cast("dict[str, Any]", stale).get("active"),
        )
        return cast("dict[str, Any]", stale)
    return None


#: Refusal detail for a tenant that resolved to the nil uuid. 401, not 400: this
#: dependency has no 400 vocabulary (its only refusal is ``401 Missing tenant
#: context``), and on every path that reaches here the value was bound by the
#: middleware or returned by licence introspection — never typed by the caller
#: into a field of this request. Inventing a second status here would also break
#: the estate rule that a nil rejection reuses the seam's EXISTING shape.
_NIL_TENANT_DETAIL = (
    "The nil uuid is not a tenant. It is a removed sentinel, registered in "
    "licenses.tenants as 'default-anon'; no request may be scoped to it."
)


def _refuse_nil_tenant(tenant_id: object, *, source: str) -> str:
    """Return ``tenant_id`` as a string unless it is the nil uuid, which 401s.

    ⛔ WHY THIS EXISTS SEPARATELY FROM ``tenant_constants.resolve_tenant_id``.
    ``resolve_tenant_id`` is the funnel for ``auth_middleware``'s three inbound
    sites, and the first version of this fix claimed it was THE funnel. It is
    not. ``get_tenant_id`` below is a SECOND resolver, reached through
    ``Depends(get_tenant_context)`` from ~50 route sites across 12 modules, and
    it never calls ``resolve_tenant_id`` at all.

    It is normally shadowed by the middleware's fast path — but middleware
    registration is FAIL-OPEN: ``worker_management_api.py`` wraps
    ``add_middleware`` in ``try/except Exception: logger.exception(...)``, so a
    registration failure downgrades every route to this unguarded resolver
    while the process keeps serving. That is exactly the condition under which a
    guard has to still hold, so the check is duplicated here deliberately rather
    than assumed away.

    Scope: this rejects the nil uuid and nothing else. A value that is not a UUID
    at all is passed through untouched — tightening THAT is a separate change
    with its own blast radius across those ~50 sites, and bundling it here would
    hide this guard inside a behaviour change.

    The introspection path below reads ``tenant_id`` ONLY. It used to prefer
    ``subject_id`` -- the conflation ``auth_middleware.py`` fixed on its own path
    on 2026-04-27 (a Subject UUID stamped as the tenant). That change is
    separate from this guard; see ``get_tenant_id``.
    """
    candidate = str(tenant_id)
    try:
        parsed = uuid.UUID(candidate.strip())
    except ValueError:
        return candidate
    if parsed.int == 0:
        logger.warning("refusing the nil uuid as a tenant (source=%s)", source)
        raise HTTPException(401, _NIL_TENANT_DETAIL)
    return candidate


async def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from X-Tenant-Id Bearer token.

    Priority:
        1. User state set by AuthenticationMiddleware (fast path)
        2. Direct license introspection from X-Tenant-Id header

    Returns:
        Tenant ID string

    Raises:
        HTTPException 401: If no tenant context available
    """
    # Priority 1: User context set by authentication middleware
    if hasattr(request.state, "tenant_id") and request.state.tenant_id:
        return _refuse_nil_tenant(request.state.tenant_id, source="request.state.tenant_id")

    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        # Check various tenant ID fields
        if isinstance(user, dict):
            tenant_id = user.get("tenant_id") or user.get("user_id") or user.get("sub")
            if tenant_id:
                return _refuse_nil_tenant(tenant_id, source="request.state.user")
        else:
            tenant_id = getattr(user, "tenant_id", None) or getattr(user, "sub", None)
            if tenant_id:
                return _refuse_nil_tenant(tenant_id, source="request.state.user")

    # Priority 2: Extract from X-Tenant-Id header and introspect
    x_tenant_id = request.headers.get("X-Tenant-Id")
    if x_tenant_id and x_tenant_id.lower().startswith("bearer "):
        token = x_tenant_id[7:].strip()
        license_data = await _introspect_license(token)
        # None => introspection could not reach a verdict (licenses slow /
        # unreachable, cold stale cache) => retryable 503, NOT a spurious 401.
        # A hard 401 here would fail a VALID token on a long run's tail LLM
        # stage (the growth_bundle 401 class); #94 fixed the middleware, this is
        # its dependency-path twin. 503 lands in w2v_common's retryable set, so
        # stage_call retries with backoff instead of failing the run.
        if license_data is None:
            raise HTTPException(
                status_code=503,
                detail="License introspection temporarily unavailable",
            )
        if license_data.get("active"):
            # ⛔ tenant_id ONLY -- never subject_id, not even as a fallback. A
            # licences subject is a USER, and one subject holds licences in many
            # tenants (measured 2026-10-14: subject 8382f0b3-... owns workers in
            # 12 tenants). Preferring it stamped the user's uuid as a tenant that
            # does not exist and merged every tenant that user belongs to into
            # one; that is how 73 workers / 38 graphs came to be keyed by a
            # subject (archived by migration 040). Falling back to it when
            # tenant_id is absent would be the same defect for a malformed
            # response, and licences cannot legitimately omit it
            # (``license.tenant_id`` is uuid NOT NULL). Same rule as
            # auth_middleware.py: no tenant_id => no tenant => 401 below.
            tenant_id = license_data.get("tenant_id")
            if tenant_id:
                return _refuse_nil_tenant(tenant_id, source="license introspection")

    raise HTTPException(401, "Missing tenant context")


async def get_resource_with_tenant_check(
    resource_id: str,
    tenant_id: str,
    db_func: Callable[[str], Awaitable[Any]],
    resource_type: str = "Resource",
    *,
    subject_id: str = "",
    require_mutate: bool = False,
) -> Any:
    """BOLA prevention: verify resource belongs to (tenant_id, subject_id).

    Under the per-user-within-tenant model:
    - tenant-visible rows (visibility='tenant'): any tenant member may read
      and mutate.
    - user-visible rows (visibility='user'): only the originating subject_id
      may read or mutate; any other caller (including other members of the
      same tenant) gets 404.

    Args:
        resource_id: ID of the resource to fetch.
        tenant_id: Tenant ID from ``get_tenant_id``.
        db_func: Async function to fetch resource by ID.
        resource_type: Type name for error messages.
        subject_id: Caller's subject_id (from ``get_subject_id``); empty
            string treated as "tenant-only caller".
        require_mutate: When True, also reject user-visible rows the caller
            does not own. When False (default) the caller may read tenant
            rows freely.

    Returns:
        Resource dict/object if it belongs to (tenant_id, subject_id) under
        the visibility rules above.

    Raises:
        HTTPException 404: If resource not found, belongs to a different
            tenant, or is user-visible to a different subject.
    """
    resource = await db_func(resource_id)
    if not resource:
        raise HTTPException(404, f"{resource_type} not found")

    resource_tenant = (
        resource.get("tenant_id")
        if isinstance(resource, dict)
        else getattr(resource, "tenant_id", None)
    )

    if resource_tenant != tenant_id:
        logger.warning(
            "Cross-tenant access attempt: tenant=%s tried to access %s id=%s",
            tenant_id,
            resource_type,
            resource_id,
        )
        # 404 (not 403): don't leak existence across tenants.
        raise HTTPException(404, f"{resource_type} not found")

    # Visibility check: user-visible rows are only readable/mutable by their
    # creating subject. We always enforce this on mutating callers; readers
    # also see only their own user-visible rows by passing through scope_clause
    # at the DB layer. This branch is the belt-and-suspenders check for
    # endpoints that fetch by id directly.
    if not can_mutate(resource, subject_id=subject_id) and require_mutate:
        logger.warning(
            "Cross-subject mutation attempt: subject=%s tried to mutate %s id=%s",
            subject_id,
            resource_type,
            resource_id,
        )
        raise HTTPException(404, f"{resource_type} not found")

    return resource
