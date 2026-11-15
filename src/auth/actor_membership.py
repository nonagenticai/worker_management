"""Which tenant does a Keycloak user belong to? (actor trust, owner decision D4).

A non-delegating licence caller may assert ``X-Actor-Id`` for a HUMAN only when
that human is a member of the caller's own tenant. Membership is read from
licences' ``GET /internal/subjects/{sub}`` (``subject.tenant_id``, migration
029), the registry's own answer to "which tenant is this subject a member of".

* **Cached.** A definite answer (a tenant, ``null`` = undecided, or 404) is
  cached for :data:`POSITIVE_TTL_S`; a failure (no credential, 401/403, 5xx,
  transport) for :data:`NEGATIVE_TTL_S`, so a licences outage cannot turn every
  request into a licences call.
* **Fails closed.** Anything but a definite same-tenant answer drops the actor.
* **WM's own credential.** A client_credentials SA token from
  ``ACTOR_LOOKUP_CLIENT_ID`` / ``ACTOR_LOOKUP_CLIENT_SECRET`` (falls back to the
  caller-identity client). licences admits ``service-internal`` /
  ``onboarding-internal`` / ``license-admin`` on that route. Never logged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx
from cachetools import TTLCache

from src.settings import get_settings

logger = logging.getLogger(__name__)

POSITIVE_TTL_S = 300
NEGATIVE_TTL_S = 60
_HTTP_TIMEOUT_S = 2.0

#: Verdicts. ``same_tenant`` is the only one that admits the actor.
SAME_TENANT = "same_tenant"
CROSS_TENANT = "cross_tenant"
UNKNOWN_SUBJECT = "unknown_subject"
TENANT_UNDECIDED = "tenant_undecided"
LOOKUP_UNAVAILABLE = "lookup_unavailable"

# sub -> member tenant ("" = undecided) | None (404). Failures are cached
# separately so their shorter TTL applies.
_member_cache: TTLCache[str, str | None] = TTLCache[str, str | None](
    maxsize=4096, ttl=POSITIVE_TTL_S
)
_failure_cache: TTLCache[str, bool] = TTLCache(maxsize=4096, ttl=NEGATIVE_TTL_S)


@dataclass
class _SAToken:
    token: str
    valid_until: float


_sa_state: dict[str, _SAToken] = {}


def reset_actor_membership_cache() -> None:
    """Drop every cached answer and SA token (tests; settings changes)."""
    _member_cache.clear()
    _failure_cache.clear()
    _sa_state.clear()


def _credentials() -> tuple[str, str, str, str] | None:
    s = get_settings()
    client_id = s.actor_lookup_client_id or s.keycloak_caller_client_id
    secret = s.actor_lookup_client_secret or s.keycloak_caller_client_secret
    if not (s.licenses_internal_url and s.keycloak_token_url and client_id and secret):
        return None
    return s.licenses_internal_url.rstrip("/"), s.keycloak_token_url, client_id, secret


async def _sa_token(client: httpx.AsyncClient, token_url: str, cid: str, secret: str) -> str | None:
    cached = _sa_state.get(cid)
    now = time.time()
    if cached is not None and now < cached.valid_until:
        return cached.token
    try:
        resp = await client.post(
            token_url,
            data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret},
        )
    except httpx.HTTPError as e:
        logger.warning("actor_membership sa_mint_failed error=%s", type(e).__name__)
        return None
    if resp.status_code != httpx.codes.OK:
        logger.warning("actor_membership sa_mint_failed status=%s", resp.status_code)
        return None
    try:
        payload = resp.json()
        token = str(payload["access_token"])
        expires_in = int(payload.get("expires_in") or 300)
    except ValueError, KeyError, TypeError:
        return None
    _sa_state[cid] = _SAToken(token=token, valid_until=now + max(expires_in - 30, 1))
    return token


async def _fetch_member_tenant(sub: str) -> tuple[bool, str | None]:
    """Return ``(ok, tenant)``: ok=False on any failure; tenant None on 404."""
    creds = _credentials()
    if creds is None:
        return False, None
    licenses_url, token_url, cid, secret = creds
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        token = await _sa_token(client, token_url, cid, secret)
        if token is None:
            return False, None
        try:
            resp = await client.get(
                f"{licenses_url}/internal/subjects/{sub}",
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as e:
            logger.warning("actor_membership lookup_failed error=%s", type(e).__name__)
            return False, None
    if resp.status_code == httpx.codes.NOT_FOUND:
        return True, None
    if resp.status_code != httpx.codes.OK:
        if resp.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            _sa_state.pop(cid, None)
        logger.warning("actor_membership lookup_failed status=%s", resp.status_code)
        return False, None
    try:
        tenant = resp.json().get("tenant_id")
    except ValueError, AttributeError:
        return False, None
    return True, (str(tenant) if tenant else "")


async def actor_tenant_verdict(actor_sub: str, caller_tenant: str) -> str:
    """Is ``actor_sub`` a member of ``caller_tenant``? One of the verdicts above."""
    if actor_sub in _member_cache:
        member = _member_cache[actor_sub]
    elif actor_sub in _failure_cache:
        return LOOKUP_UNAVAILABLE
    else:
        ok, member = await _fetch_member_tenant(actor_sub)
        if not ok:
            _failure_cache[actor_sub] = True
            return LOOKUP_UNAVAILABLE
        _member_cache[actor_sub] = member
    if member is None:
        return UNKNOWN_SUBJECT
    if not member:
        return TENANT_UNDECIDED
    if member.lower() == (caller_tenant or "").strip().lower():
        return SAME_TENANT
    return CROSS_TENANT
