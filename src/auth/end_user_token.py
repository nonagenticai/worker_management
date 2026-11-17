"""End-user proof for DELEGATED requests: the forwarded portal-realm JWT.

WHY THIS EXISTS (owner ruling 2026-10-01)
-----------------------------------------
fr-backend calls this service as ``svc-fr-backend`` and names the run's tenant in
``X-Effective-Tenant-Id``. That header is a CLAIM: the middleware checks it for
spelling only (tenant provenance ``delegated``), so worker_management#245 refuses
to decrypt the tenant's stored BYO key for it. The result was that a customer's
own key could never be used from fr at all.

The proof that closes the gap without trusting the service account for every
tenant: fr forwards the SIGNED-IN USER's own portal-realm access token in
``X-End-User-Token``. This module verifies it and, only when ALL hold, records
the tenant as end-user-proven:

1. the request is delegated, i.e. the caller's introspected subject is in
   ``AUTH_DELEGATING_SUBJECTS`` (decided by the middleware before this runs, and
   re-checked by ``tenant_key_entitled`` at use);
2. the token's RS256 signature verifies against the PORTAL realm JWKS (reusing
   ``keycloak_roles``'s fetch/verify, with its own cache);
3. ``exp`` and ``iss`` are present, the token is unexpired, and ``iss`` equals
   ``Settings.end_user_token_issuer`` exactly
   (``https://auth.nonagentic.ai/realms/portal`` — the value fr pins);
4. it is an ACCESS token (``typ`` = ``Bearer``; ID / refresh tokens refused)
   issued to an allowed portal client (``azp`` in
   ``Settings.end_user_token_allowed_clients``; empty setting = deny-all);
5. its ``tenant_id`` claim (the raw claim fr reads in
   ``backend/src/services/keycloak.ts``; NO ``sub`` fallback) is present and
   the effective tenant equals it OR is listed in the admin-only
   ``member_tenant_ids`` claim (fr#1385 Option 1, :func:`token_proves_tenant`;
   exact match).

FAILURE POLICY: fail closed on the PRIVILEGE, open on the REQUEST. Nothing here
raises; an absent, invalid, expired, mismatched or unverifiable token simply
leaves the tenant unproven, and the reason is recorded so the BYO-key error the
caller then receives can say what to fix. A delegated request that needs no BYO
key (``local-llm``) is unaffected. The token is never logged or stored.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING

import httpx2 as httpx
import jwt

from src.api.identity_context import set_end_user_identity, set_end_user_proof
from src.auth import keycloak_roles
from src.settings import get_settings

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

END_USER_TOKEN_HEADER = "X-End-User-Token"  # noqa: S105 -- a header NAME
END_USER_REALM = "portal"
# The portal realm role that marks the platform owner. Same name and same claim
# path (``realm_access.roles`` ONLY, no top-level ``roles`` fallback) as fr's
# ``SUPER_ADMIN_REALM_ROLE`` / ``rolesFromClaims``
# (backend/src/lib/super-admin-gate.ts, middleware/request-context.ts).
SUPER_ADMIN_REALM_ROLE = "super_admin"

PROOF_OK = "ok"
PROOF_ABSENT = "absent"
PROOF_UNCONFIGURED = "unconfigured"
PROOF_ISSUER_UNCONFIGURED = "issuer_unconfigured"
PROOF_JWKS_UNAVAILABLE = "jwks_unavailable"
PROOF_NO_TENANT_CLAIM = "no_tenant_claim"
PROOF_TENANT_MISMATCH = "tenant_mismatch"
PROOF_ISSUER_MISMATCH = "invalid:InvalidIssuerError"
PROOF_NOT_ACCESS_TOKEN = "not_access_token"  # noqa: S105 -- a reason code
PROOF_CLIENT_NOT_ALLOWED = "client_not_allowed"

# Keycloak stamps ``typ`` on every token it mints: ``Bearer`` for an access
# token, ``ID`` for an ID token, ``Refresh`` / ``Offline`` for refresh tokens.
_ACCESS_TOKEN_TYP = "bearer"  # noqa: S105 -- a token TYPE name

_JWKS_TTL_SECONDS = 600
# url -> (fetched_at_monotonic, jwks). Separate from keycloak_roles' single-slot
# secure-apps cache: two realms, two key sets.
_jwks_cache: dict[str, tuple[float, dict]] = {}

_REALM_SEGMENT = re.compile(r"/realms/[^/]+/")


def delegating_subjects() -> frozenset[str]:
    """``AUTH_DELEGATING_SUBJECTS`` read at call time (same source as the middleware)."""
    raw = os.getenv("AUTH_DELEGATING_SUBJECTS", "") or ""
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def end_user_jwks_url() -> str:
    """JWKS URL of the FIRST accepted realm — kept for existing callers/tests.

    The verifier itself iterates :func:`end_user_sources`, which carries one URL
    per realm; this helper must not be used to decide what is trusted.
    """
    sources = end_user_sources()
    return sources[0][2] if sources else ""


def end_user_realms() -> tuple[str, ...]:
    """The realms whose access tokens may prove an end user's tenant.

    ``Settings.end_user_token_realms`` (CSV). Empty falls back to the single
    historical realm so a misconfiguration can never widen the trust set.
    """
    raw = (get_settings().end_user_token_realms or "").strip()
    realms = tuple(r.strip() for r in raw.split(",") if r.strip())
    return realms or (END_USER_REALM,)


def _issuer_origin() -> str:
    """Frontend origin for ``iss``, taken from the configured issuer.

    Keycloak's ``iss`` is its FRONTEND url, not the in-cluster JWKS origin we
    fetch keys from, so the per-realm issuer cannot be derived from the JWKS
    URL. ``end_user_token_issuer`` supplies the origin; the realm is swapped.
    """
    configured = get_settings().end_user_token_issuer.strip()
    if "/realms/" in configured:
        return configured.split("/realms/", 1)[0]
    return configured


def end_user_sources() -> tuple[tuple[str, str, str], ...]:
    """``(realm, issuer, jwks_url)`` accepted for ``X-End-User-Token``, in order.

    One entry per realm in ``Settings.end_user_token_realms``. Each realm gets
    ITS OWN JWKS: the token must verify by signature against that realm's keys
    AND carry exactly that realm's ``iss``, so accepting two realms is not a
    weakening — a realm-``portal`` token still cannot pass as realm ``fr``.
    """
    origin = _issuer_origin()
    explicit_jwks = get_settings().end_user_jwks_url.strip()
    base = keycloak_roles._jwks_url()
    sources: list[tuple[str, str, str]] = []
    for realm in end_user_realms():
        issuer = f"{origin}/realms/{realm}" if origin else ""
        if explicit_jwks and realm == END_USER_REALM:
            # Backward compatibility: an explicit URL stays the portal key set.
            jwks = explicit_jwks
        elif base and _REALM_SEGMENT.search(base):
            jwks = _REALM_SEGMENT.sub(f"/realms/{realm}/", base, count=1)
        else:
            jwks = ""
        sources.append((realm, issuer, jwks))
    return tuple(sources)


async def _get_jwks(url: str, *, force_refresh: bool = False) -> dict:
    now = time.monotonic()
    hit = _jwks_cache.get(url)
    if force_refresh or hit is None or (now - hit[0]) > _JWKS_TTL_SECONDS:
        jwks = await keycloak_roles._fetch_jwks(url)
        _jwks_cache[url] = (now, jwks)
        return jwks
    return hit[1]


def _verify(token: str, jwks: dict, *, expected_realm: str, expected_issuer: str) -> dict:
    payload = keycloak_roles._verify_with_jwks(token, jwks, expected_realm=expected_realm)
    # An empty issuer is NOT "skip the pin" (validator N-f): no configured
    # issuer means no token can be proven.
    if not expected_issuer or str(payload.get("iss") or "") != expected_issuer:
        raise jwt.InvalidIssuerError("issuer mismatch")
    return payload


def allowed_clients() -> frozenset[str]:
    """``Settings.end_user_token_allowed_clients`` as a set; empty = deny-all."""
    raw = get_settings().end_user_token_allowed_clients or ""
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


def _token_shape(payload: dict) -> str:
    """``""`` when ``payload`` is an access token from an allowed client, else the reason."""
    typ = payload.get("typ")
    if not isinstance(typ, str) or typ.strip().lower() != _ACCESS_TOKEN_TYP:
        return PROOF_NOT_ACCESS_TOKEN
    azp = payload.get("azp")
    if not isinstance(azp, str) or azp.strip() not in allowed_clients():
        return PROOF_CLIENT_NOT_ALLOWED
    return ""


def _bearer(raw: str) -> str:
    raw = raw.strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def _realm_roles(payload: dict) -> frozenset[str]:
    """``realm_access.roles`` of a VERIFIED payload (fr's ``rolesFromClaims``)."""
    realm = payload.get("realm_access")
    roles = realm.get("roles") if isinstance(realm, dict) else None
    if not isinstance(roles, list):
        return frozenset()
    return frozenset(r for r in roles if isinstance(r, str) and r)


async def _verified_payload(raw_header: str | None) -> tuple[dict | None, str]:
    """Verify signature (portal JWKS), issuer and expiry. Never raises.

    Returns ``(payload, "")`` on success, else ``(None, reason)``.
    """
    token = _bearer(raw_header or "")
    if not token:
        return None, PROOF_ABSENT
    sources = tuple(s for s in end_user_sources() if s[2])
    if not sources:
        return None, PROOF_UNCONFIGURED
    if not any(issuer for _, issuer, _ in sources):
        # END_USER_TOKEN_ISSUER set to "" used to switch the issuer pin off
        # silently, leaving only the realm-suffix check. Refuse the proof
        # instead, loudly; the request itself still proceeds without the key.
        logger.error("end_user_token_issuer_unconfigured: END_USER_TOKEN_ISSUER is empty")
        return None, PROOF_ISSUER_UNCONFIGURED

    # Try each accepted realm against ITS OWN key set. A token verifies on at
    # most one of them (the signature is realm-specific), so this does not
    # weaken the check — it only stops one realm's token being unprovable
    # because a different realm was pinned.
    last_reason = PROOF_JWKS_UNAVAILABLE
    for realm, issuer, url in sources:
        if not issuer:
            last_reason = PROOF_ISSUER_UNCONFIGURED
            continue
        try:
            try:
                payload = _verify(
                    token,
                    await _get_jwks(url),
                    expected_realm=realm,
                    expected_issuer=issuer,
                )
            except keycloak_roles._UnknownKidError:
                payload = _verify(
                    token,
                    await _get_jwks(url, force_refresh=True),
                    expected_realm=realm,
                    expected_issuer=issuer,
                )
        except (httpx.HTTPError, OSError, jwt.PyJWKClientConnectionError, jwt.PyJWKSetError) as exc:
            logger.warning(
                "end_user_token_jwks_unavailable: realm=%s %s", realm, type(exc).__name__
            )
            last_reason = PROOF_JWKS_UNAVAILABLE
            continue
        except (keycloak_roles._UnknownKidError, jwt.PyJWTError) as exc:
            name = (
                "UnknownKidError"
                if isinstance(exc, keycloak_roles._UnknownKidError)
                else type(exc).__name__
            )
            last_reason = f"invalid:{name}"
            continue
        except Exception as exc:
            last_reason = f"invalid:{type(exc).__name__}"
            continue
        shape = _token_shape(payload)
        if shape:
            return None, shape
        return payload, ""
    return None, last_reason


def token_proves_tenant(claims: dict, effective: object) -> bool:
    """Does a VERIFIED token's claim set prove ``effective``? fr#1385 Option 1.

    The ONE tenant-match predicate, shared byte-for-byte in behaviour with fr via
    ``tests/fixtures/tenant_membership_vectors.json``. True iff ``tenant_id`` is a
    non-empty string AND ``effective`` equals it or is in ``member_tenant_ids``
    (a Keycloak-set, admin-only multivalued claim: a string or a list; non-string
    entries are ignored). Matching is EXACT: no case-folding, no trimming. A
    missing or empty ``effective`` is never proven. Pure; call it only on a
    payload whose signature, issuer, ``typ`` and ``azp`` already verified.
    """
    if not isinstance(effective, str) or not effective:
        return False
    home = claims.get("tenant_id")
    if not isinstance(home, str) or not home:
        return False
    if effective == home:
        return True
    members = claims.get("member_tenant_ids")
    if isinstance(members, str):
        members = [members]
    if not isinstance(members, list):
        return False
    return any(isinstance(m, str) and m == effective for m in members)


def _tenant_proof(payload: dict, effective_tenant: str) -> tuple[str, str]:
    claim = payload.get("tenant_id")
    if not isinstance(claim, str) or not claim.strip():
        return "", PROOF_NO_TENANT_CLAIM
    if not token_proves_tenant(payload, effective_tenant):
        return "", PROOF_TENANT_MISMATCH
    return str(effective_tenant).strip().lower(), PROOF_OK


async def verify_end_user_tenant(raw_header: str | None, effective_tenant: str) -> tuple[str, str]:
    """Return ``(proven_tenant_or_"", reason)`` for a forwarded end-user token.

    ``proven_tenant`` is non-empty only when the reason is :data:`PROOF_OK`.
    Never raises.
    """
    payload, reason = await _verified_payload(raw_header)
    if payload is None:
        return "", reason
    return _tenant_proof(payload, effective_tenant)


async def stamp_end_user_proof(request: Request, effective_tenant: str, delegator: str) -> None:
    """Verify ``X-End-User-Token`` on a delegated request and record the outcome.

    Called by ``AuthenticationMiddleware`` on the delegated branch only, AFTER
    ``set_identity_context`` (which clears any earlier proof). ``delegator`` is
    the introspected subject the middleware already found in
    ``AUTH_DELEGATING_SUBJECTS``.
    """
    payload, reason = await _verified_payload(request.headers.get(END_USER_TOKEN_HEADER))
    tenant = ""
    if payload is not None:
        tenant, reason = _tenant_proof(payload, effective_tenant)
        # The verified token's own facts, recorded EVEN on a tenant mismatch: a
        # portal ``super_admin`` acting for a customer tenant carries a JWT of
        # its own tenant. Read only by ``owner_key_test_entitled`` (key TEST,
        # nothing else) and the owner audit log. Never a tenant proof.
        sub = payload.get("sub")
        claim = payload.get("tenant_id")
        set_end_user_identity(
            sub=sub if isinstance(sub, str) else "",
            token_tenant=claim.strip().lower() if isinstance(claim, str) else "",
            super_admin=SUPER_ADMIN_REALM_ROLE in _realm_roles(payload),
            delegator=delegator,
        )
    if reason != PROOF_ABSENT:
        logger.info(
            "end_user_token tenant=%s delegator=%s outcome=%s",
            effective_tenant,
            delegator,
            reason,
        )
    set_end_user_proof(
        tenant_id=tenant if reason == PROOF_OK else "",
        delegator=delegator if reason == PROOF_OK else "",
        reason=reason,
    )
    request.state.end_user_tenant_proven = reason == PROOF_OK


# ----------------------------------------------------------------------------
# LICENCE BRANCH (George 2026-10-15: licences are tenant×app only)
# ----------------------------------------------------------------------------
# A caller presenting a TENANT ``workers`` licence (not a delegating service
# account) may name the signed-in user with the SAME ``X-End-User-Token``. The
# licence proves the tenant; the token proves WHICH member of it acted. Same
# verification as the delegated branch (``_verified_payload``: portal JWKS
# signature, pinned ``iss``, ``exp``, ``typ``=Bearer, ``azp`` allowlist) and the
# same membership predicate (``_tenant_proof`` -> ``token_proves_tenant``),
# here against the LICENCE's tenant. Unlike the delegated branch, the result is
# used as the request's subject, so the middleware REFUSES a present token that
# fails (never a silent fall back to the licence's service subject).

PROOF_NO_SUB = "no_sub"

# Reasons that mean "this server could not verify", not "the token is bad".
LICENCE_END_USER_UNVERIFIABLE: frozenset[str] = frozenset(
    {PROOF_UNCONFIGURED, PROOF_ISSUER_UNCONFIGURED, PROOF_JWKS_UNAVAILABLE}
)
# Reasons that mean "a valid user, but not a member of the licence's tenant".
LICENCE_END_USER_FOREIGN_TENANT: frozenset[str] = frozenset(
    {PROOF_TENANT_MISMATCH, PROOF_NO_TENANT_CLAIM}
)


async def verify_licence_end_user(
    raw_header: str | None, licence_tenant: str
) -> tuple[dict | None, str]:
    """Verify ``X-End-User-Token`` against the LICENCE's tenant. Never raises.

    Returns ``(payload, PROOF_OK)`` when the token verifies, carries a ``sub``
    and its user is a member of ``licence_tenant``; ``(None, PROOF_ABSENT)``
    when no token was sent; else ``(None, reason)``.
    """
    payload, reason = await _verified_payload(raw_header)
    if payload is None:
        return None, reason
    _tenant, reason = _tenant_proof(payload, licence_tenant)
    if reason != PROOF_OK:
        return None, reason
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        return None, PROOF_NO_SUB
    return payload, PROOF_OK


def stamp_licence_end_user(payload: dict, licence_tenant: str) -> None:
    """Record a verified licence-branch end user (after ``set_identity_context``).

    ``delegator`` stays EMPTY: nothing was delegated, so every delegator-gated
    predicate (``owner_key_test_entitled``, the delegated BYO proof) stays
    closed. The identity feeds the tenant-config audit (``_actor_kind``).
    """
    sub = payload.get("sub")
    claim = payload.get("tenant_id")
    set_end_user_identity(
        sub=sub if isinstance(sub, str) else "",
        token_tenant=claim.strip().lower() if isinstance(claim, str) else "",
        super_admin=SUPER_ADMIN_REALM_ROLE in _realm_roles(payload),
        delegator="",
    )
    set_end_user_proof(tenant_id=str(licence_tenant).strip().lower(), delegator="", reason=PROOF_OK)
