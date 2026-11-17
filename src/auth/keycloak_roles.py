"""Verified caller identity from the ``X-Caller-Authorization`` SA JWT.

Why this module exists
----------------------
``AuthenticationMiddleware`` used to populate the caller-identity ContextVars
straight from the ``x-caller-roles`` / ``x-caller-client-id`` **request
headers**, on the stated assumption that license-authz had already verified an
upstream JWT and injected them. For worker_management that assumption never
held:

* the ingress EnvoyFilter (``ext-authz-enforce.yaml``) strips the client's
  tenant headers but has never stripped ``x-caller-roles``, and its STEP-2
  block copies only the tenant/license/app/plan family out of the license-authz
  response — the verified caller headers are discarded;
* the ``workers`` namespace is ambient with **no waypoint**, so no in-mesh
  ext_authz runs in front of this service either.

So the header was attacker-supplied end to end, and
``graph_sessions_correlation_api`` traded it for an unscoped cross-tenant read.

This module replaces that trust with a signature check on the channel the
codebase actually designed for it: ``caller_identity.caller_identity_request_hook``
already *emits* ``X-Caller-Authorization: Bearer <sa-jwt>`` on outbound
``private-DNS`` calls — worker_management simply never consumed it inbound.

Two gates, both required
------------------------
1. the JWT must verify against the ``secure-apps`` JWKS, and
2. its ``azp`` must be in :data:`ALLOWED_CALLER_CLIENTS`.

The ``azp`` allowlist is not belt-and-braces. ``secure-apps`` is **not** an
SA-only realm — it also authenticates human dashboard users, who carry
``license-admin`` (see ``billing_service/src/auth.py``: "Dashboard
authentication — Keycloak JWT validation requiring license-admin role"). Since
``license-admin`` is one of ``graph_sessions_correlation_api``'s three internal
roles, a roles-only check would hand every billing-dashboard user an unscoped
cross-tenant read. license-authz guards the same seam the same way, via
``SERVICE_PRINCIPAL_ALLOWED_CLIENTS``.

Failure policy: fail closed on the **privilege**, open on the **request** —
with ONE exception, the presented-but-invalid token.

* header ABSENT, caller-identity UNCONFIGURED, Keycloak/JWKS UNREACHABLE (or
  serving an unusable key set), ``azp`` missing, ``azp`` not allowlisted:
  return ``("", ())`` and the request proceeds as an ordinary tenant-scoped
  caller. A 503 would take down every public ``/workers/`` call the moment
  Keycloak blinked — strictly worse than an enrichment pass being skipped. The
  not-allowlisted case is a VALID token (marketing-service-account sends one a
  few times a day) and must keep working.
* header PRESENT and the token itself FAILS validation (bad signature, expired,
  wrong issuer/realm, missing exp/iss, unknown kid after one refresh,
  undecodable, or a non-Bearer value): raise :class:`CallerTokenInvalidError`,
  which the middleware turns into a 401. The keys were fetched fine, so this is
  a verdict on the token, not an outage — a caller presenting a forged or stale
  identity is told so rather than silently served on its other credentials.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx2 as httpx
import jwt

from src.settings import get_settings

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

# Refreshed on TTL expiry or on a `kid` miss. Without the TTL the cache never
# reflects Keycloak signing-key rotation and caller identity silently breaks
# every rotation until pods restart (the failure billing_service hit).
_JWKS_TTL_SECONDS = 600

_jwks_cache: dict | None = None
_jwks_cache_ts: float = 0.0

# Empty means DENY-ALL here, deliberately unlike license-authz's
# SERVICE_PRINCIPAL_ALLOWED_CLIENTS (where empty means allow-any). See the
# module docstring: `secure-apps` carries human accounts, so allow-any would be
# a privilege grant, not a compatibility default.
_DEFAULT_ALLOWED_CALLER_CLIENTS = ("improvement-loop-service",)


_CALLER_TOKEN_LEEWAY_S = 10


class _UnknownKidError(Exception):
    """Raised internally when a token's kid is absent from the cached JWKS."""


class CallerTokenInvalidError(Exception):
    """A PRESENT ``X-Caller-Authorization`` token failed validation → 401.

    ``failure_class`` names the failure (e.g. ``InvalidSignatureError``,
    ``ExpiredSignatureError``, ``MalformedHeader``) and is safe to return to the
    caller. The token itself is never stored on the exception.
    """

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


def allowed_caller_clients() -> frozenset[str]:
    """Return the `azp` values permitted to assert internal-service standing.

    Read per-call (not import-time) so tests and a config reload see changes.
    """
    raw = get_settings().caller_allowed_clients
    if not raw.strip():
        return frozenset(_DEFAULT_ALLOWED_CALLER_CLIENTS)
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


def _jwks_url() -> str:
    """Return the secure-apps JWKS endpoint, or "" when caller-identity is unconfigured.

    Prefers the explicit setting. Falls back to deriving it from the already
    configured token endpoint — same realm, standard OIDC layout — so this
    works on today's config without a second manifest change. Per
    config-externalization-policy this derives from configured values and never
    hardcodes a host.
    """
    settings = get_settings()
    explicit = settings.keycloak_jwks_url.strip()
    if explicit:
        return explicit
    token_url = settings.keycloak_token_url.strip()
    if token_url.endswith("/token"):
        return token_url[: -len("/token")] + "/certs"
    return ""


async def _fetch_jwks(url: str) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=5.0) as client:  # nosec B501
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _get_jwks(url: str, *, force_refresh: bool = False) -> dict:
    global _jwks_cache, _jwks_cache_ts  # noqa: PLW0603
    now = time.monotonic()
    stale = (now - _jwks_cache_ts) > _JWKS_TTL_SECONDS
    if force_refresh or _jwks_cache is None or stale:
        _jwks_cache = await _fetch_jwks(url)
        _jwks_cache_ts = now
    return _jwks_cache


def _expected_realm(jwks_url: str) -> str:
    """The realm name embedded in ``jwks_url``, or "" if the URL has no /realms/ segment.

    DERIVED from the configured JWKS URL rather than configured separately, so it
    cannot drift out of sync with the key source and needs no new manifest env var.
    """
    parts = urlsplit(jwks_url).path.strip("/").split("/")
    if "realms" in parts:
        i = parts.index("realms")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _verify_with_jwks(token: str, jwks: dict, *, expected_realm: str = "") -> dict:
    """Verify ``token`` against ``jwks``. Raises _UnknownKidError if the kid is absent."""
    jwk_set = jwt.PyJWKSet.from_dict(jwks)
    kid = jwt.get_unverified_header(token).get("kid")
    signing_key = next((k for k in jwk_set.keys if k.key_id == kid), None)
    if signing_key is None:
        raise _UnknownKidError(kid)
    # verify_aud is off for the same reason license-authz and billing_service
    # disable it: SA tokens carry an `aud` of "account", not this service.
    # `require: ["exp"]` is not redundant with verify_exp — PyJWT only checks
    # exp when the claim is PRESENT, so a token minted without one would never
    # expire. Real Keycloak tokens always carry exp; demanding it costs nothing
    # and removes the immortal-token case. `iss` is required for the realm check
    # below — without it a token omitting the claim would skip that check entirely.
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        options={"verify_aud": False, "require": ["exp", "iss"]},
        # A present-but-invalid caller token is now a 401, so a few seconds of
        # Keycloak/pod clock skew must not reject a fresh token (iat/nbf/exp).
        leeway=_CALLER_TOKEN_LEEWAY_S,
    )

    # Realm pinning, not full issuer pinning — and the difference is load-bearing.
    #
    # ⚠️ Do NOT "improve" this into `jwt.decode(..., issuer=<the JWKS URL's origin>)`.
    # Keycloak's `iss` is its FRONTEND url, not the URL we fetched keys from. Measured
    # live 2026-08-02 on a real improvement-loop-service SA token:
    #     iss      = https://auth.nonagentic.ai/realms/secure-apps
    #     JWKS URL = http://keycloak.keycloak.example.com:8080/realms/secure-apps/...
    # Pinning the derived origin would reject every genuine token in production while
    # passing every test that mints its own.
    #
    # What this DOES close: a token minted by a different realm on the same Keycloak.
    # A token from an unrelated Keycloak already fails at the signature, so this is
    # defense in depth rather than the primary control (which remains the `azp`
    # allowlist).
    if expected_realm:
        iss = str(payload.get("iss") or "")
        if not iss.endswith(f"/realms/{expected_realm}"):
            raise jwt.InvalidIssuerError(f"token issuer {iss!r} is not realm {expected_realm!r}")
    return payload


def _roles_from_payload(payload: dict) -> tuple[str, ...]:
    """Union realm roles with top-level roles, mirroring improvement-loop's require_internal."""
    realm = payload.get("realm_access") or {}
    realm_roles = realm.get("roles") or []
    top_roles = payload.get("roles") or []
    merged = {r for r in [*realm_roles, *top_roles] if isinstance(r, str) and r}
    return tuple(sorted(merged))


def _bearer_token(request: Request) -> str:
    raw = request.headers.get("x-caller-authorization", "")
    if not raw.lower().startswith("bearer "):
        return ""
    return raw[7:].strip()


async def verified_caller_identity(request: Request) -> tuple[str, tuple[str, ...]]:
    """Return ``(caller_client_id, caller_roles)`` from a verified SA JWT.

    Returns ``("", ())`` when the header is absent, caller-identity is
    unconfigured, Keycloak/JWKS is unreachable, ``azp`` is missing, or ``azp``
    is not allowlisted.

    Raises :class:`CallerTokenInvalidError` when the header is PRESENT and the
    token fails validation (malformed, bad signature, expired, wrong
    issuer/realm, missing required claim, unknown kid after one refresh).

    ``caller_client_id`` is taken from the **verified** ``azp``, so the
    cross-tenant audit line in ``graph_sessions_correlation_api`` records an
    actor the caller cannot choose. Previously that field was a plain request
    header, letting an attacker stamp an innocent service as the actor of their
    own cross-tenant read.
    """
    if not request.headers.get("x-caller-authorization", "").strip():
        return "", ()
    token = _bearer_token(request)
    if not token:
        # Present but not `Bearer <token>`: malformed, not absent.
        logger.warning("caller_identity_token_rejected: MalformedHeader")
        raise CallerTokenInvalidError("MalformedHeader")

    url = _jwks_url()
    if not url:
        logger.warning("caller_identity_unconfigured: no JWKS or token URL configured")
        return "", ()

    realm = _expected_realm(url)
    try:
        try:
            payload = _verify_with_jwks(token, await _get_jwks(url), expected_realm=realm)
        except _UnknownKidError:
            # Signing key rotated: refetch once before giving up.
            payload = _verify_with_jwks(
                token, await _get_jwks(url, force_refresh=True), expected_realm=realm
            )
    except (httpx.HTTPError, OSError, jwt.PyJWKClientConnectionError, jwt.PyJWKSetError) as exc:
        # UNREACHABLE, not invalid: Keycloak could not be reached, or served a
        # key set with no usable keys. No privilege, but the request continues.
        # These are listed BEFORE the PyJWTError clause below because
        # PyJWKClientConnectionError and PyJWKSetError are PyJWTError
        # subclasses — the order is what keeps a Keycloak blip from becoming
        # a 401 on every caller that sends a token.
        logger.warning("caller_identity_jwks_unavailable: %s", type(exc).__name__)
        return "", ()
    except (_UnknownKidError, jwt.PyJWTError) as exc:
        # INVALID: the keys were fetched and the token failed against them
        # (InvalidSignatureError, ExpiredSignatureError, DecodeError,
        # InvalidIssuerError, InvalidAudienceError, MissingRequiredClaimError,
        # or a kid absent even after one refresh). The token is never logged.
        logger.warning("caller_identity_token_rejected: %s", type(exc).__name__)
        failure = "UnknownKidError" if isinstance(exc, _UnknownKidError) else type(exc).__name__
        raise CallerTokenInvalidError(failure) from None

    azp = payload.get("azp")
    if not isinstance(azp, str) or not azp:
        logger.warning("caller_identity_missing_azp")
        return "", ()

    allowed = allowed_caller_clients()
    if azp not in allowed:
        logger.warning("caller_identity_client_not_allowlisted: azp=%s", azp)
        return "", ()

    return azp, _roles_from_payload(payload)
