"""Caller-identity primitives for outbound SA-token injection + RBAC.

Three pieces:

1. ``CallerTokenCache`` — fetches Keycloak SA tokens via client_credentials
   grant against the secure-apps realm and caches them per (client_id,
   client_secret) pair. Refreshes ~30s before expiry. Single-flight refresh
   via asyncio.Lock so concurrent callers don't stampede Keycloak.

2. ``caller_identity_request_hook`` — httpx event hook that injects
   ``X-Caller-Authorization: Bearer <sa-jwt>`` on outbound requests where
   the host ends in the private internal DNS zone. Skips other hosts entirely.

3. ``require_role(role)`` — FastAPI Depends factory that asserts the
   request's caller (set into ContextVars by ``AuthenticationMiddleware``
   from the x-caller-roles header license-authz injected) carries ``role``.
   Raises 403 otherwise.

Template lives in worker_management; subsequent Group A/B repos copy these
files verbatim. M3 of caller-identity-rollout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx2 as httpx
from fastapi import HTTPException, Request, status

from src.api.identity_context import get_caller_roles_from_context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


# Refresh tokens this many seconds before expiry to avoid issuing requests
# with about-to-expire tokens that downstream services may reject.
_REFRESH_LEAD_SECONDS = 30
# Default expiry assumed when Keycloak's response omits expires_in (~5 min).
_DEFAULT_EXPIRY_SECONDS = 300


@dataclass
class _CachedToken:
    """One cached access token + the absolute monotonic deadline to refresh."""

    access_token: str
    refresh_at_monotonic: float


class CallerTokenCache:
    """Keycloak SA token cache with single-flight refresh.

    One instance per process; caches multiple (client_id, client_secret)
    tuples to keep the same shape callers will need when one service mints
    multiple SA tokens (rare but legitimate — e.g., separate cred sets per
    downstream tenant).
    """

    def __init__(self, token_url: str, http_client: httpx.AsyncClient | None = None) -> None:
        if not token_url:
            msg = "CallerTokenCache requires a non-empty token_url"
            raise ValueError(msg)
        self._token_url = token_url
        # Caller may share an httpx client; otherwise we own one for outbound
        # token mints (closed by close()).
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=5.0)
        self._cache: dict[tuple[str, str], _CachedToken] = {}
        # One lock per (client_id, client_secret) so concurrent mints for the
        # SAME credential pair single-flight, but different pairs don't block
        # each other.
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def get_token(self, client_id: str, client_secret: str) -> str:
        """Return a valid access token, minting or refreshing as needed."""
        key = (client_id, client_secret)
        cached = self._cache.get(key)
        if cached and time.monotonic() < cached.refresh_at_monotonic:
            return cached.access_token
        return await self._refresh(key)

    async def _refresh(self, key: tuple[str, str]) -> str:
        """Single-flight refresh for the given credential pair."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check inside the lock — another coroutine may have already
            # refreshed while we were waiting.
            cached = self._cache.get(key)
            if cached and time.monotonic() < cached.refresh_at_monotonic:
                return cached.access_token

            client_id, client_secret = key
            try:
                resp = await self._client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                )
            except httpx.HTTPError as e:
                logger.warning("caller-identity token mint network error: %s", e)
                msg = "caller-identity token mint failed"
                raise CallerTokenError(msg) from e

            if resp.status_code != httpx.codes.OK:
                logger.warning(
                    "caller-identity token mint returned %s for client_id=%s",
                    resp.status_code,
                    client_id,
                )
                msg = f"caller-identity token mint failed: {resp.status_code}"
                raise CallerTokenError(msg)

            payload = resp.json()
            access_token = payload.get("access_token")
            if not access_token:
                msg = "caller-identity token response missing access_token"
                raise CallerTokenError(msg)
            expires_in = int(payload.get("expires_in") or _DEFAULT_EXPIRY_SECONDS)
            refresh_at = time.monotonic() + max(expires_in - _REFRESH_LEAD_SECONDS, 1)
            self._cache[key] = _CachedToken(
                access_token=access_token, refresh_at_monotonic=refresh_at
            )
            return access_token


class CallerTokenError(Exception):
    """Raised when minting a caller SA token fails."""


def make_caller_request_hook(
    cache: CallerTokenCache,
    client_id: str,
    client_secret: str,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Build an httpx event hook that attaches X-Caller-Authorization on internal hosts.

    Usage:
        client = httpx.AsyncClient(event_hooks={"request": [make_caller_request_hook(...)]})

    Skips if the request host doesn't end in the private internal DNS zone (we don't want to
    leak SA tokens to external services). Also skips if the credentials are
    empty (opt-out via empty config).
    """

    async def _hook(request: httpx.Request) -> None:
        if not client_id or not client_secret:
            return
        host = request.url.host
        if not host or not host.endswith(".internal"):
            return
        try:
            token = await cache.get_token(client_id, client_secret)
        except CallerTokenError:
            # Fail-soft: outbound call proceeds without caller header rather
            # than 5xx-ing the user-facing request. license-authz on the
            # downstream side will see absent header → opt-in semantics.
            return
        request.headers["X-Caller-Authorization"] = f"Bearer {token}"

    return _hook


def require_role(role: str) -> Callable[[Request], None]:
    """FastAPI Depends factory: assert the calling SA carries ``role``.

    Reads from the per-request caller-roles ContextVar that
    ``AuthenticationMiddleware`` populates from the x-caller-roles header
    license-authz injected after verifying the inbound caller JWT. Raises
    403 if the role is absent.

    Note the Depends factory takes a Request param so FastAPI calls it once
    per route invocation (rather than caching the empty-roles result that
    would happen with a parameter-less Depends).
    """

    def _dep(_request: Request) -> None:
        roles = get_caller_roles_from_context()
        if role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires_role:{role}",
            )

    return _dep


def require_any_role(*accepted: str) -> Callable[[Request], None]:
    """Like :func:`require_role`, but ANY one of ``accepted`` suffices.

    Same verified source (the caller-roles ContextVar the middleware fills
    ONLY from a signature-verified, azp-allowlisted SA JWT), same 403. The
    detail names every accepted role, first one first, so
    ``requires_role:<first>`` stays a prefix of it.
    """
    if not accepted:
        msg = "require_any_role needs at least one role"
        raise ValueError(msg)
    wanted = frozenset(accepted)
    detail = "requires_role:" + "|".join(accepted)

    def _dep(_request: Request) -> None:
        roles = get_caller_roles_from_context()
        if wanted.isdisjoint(roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    return _dep
