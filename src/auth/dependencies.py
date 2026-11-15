"""FastAPI dependencies for tenant + per-user isolation.

Uses license token authentication only. No Keycloak/JWT.

Tenant + subject scoping:
- get_tenant_id: organization the caller represents (X-Tenant-Id introspected
    from a license token, or the X-Tenant-Id raw header on trusted networks)
- get_subject_id: the user within the tenant (X-Subject-Id header injected by
    license-authz from the licenses DB Subject UUID; empty string when the
    caller has not yet upgraded — treated as "tenant-only caller")
- get_visibility: 'tenant' (default) or 'user' visibility for newly-created rows

Note: no SQLAlchemy ORM User class exists in this repo (raw-SQL psycopg3
data layer). The ``effective_subject_id`` pattern is implemented inline in
every DB-write method as ``subject_id or tenant_id``. A before_insert
SQLAlchemy listener is not applicable here.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request

from src.auth.tenant import get_resource_with_tenant_check, get_tenant_id

# Canonical visibility value set per canonical standard 2026-04-27.
VISIBILITY_VALUES: frozenset[str] = frozenset({"tenant", "user"})


async def get_tenant_name(request: Request) -> str:
    """Get tenant name from request state (stashed by AuthenticationMiddleware).

    Falls back to empty string — never raises.  Call sites treat "" as
    "caller did not supply a tenant name".
    """
    return str(getattr(request.state, "tenant_name", "") or "")


async def get_subject_id(request: Request) -> str:
    """Get subject (user) ID from request state.

    Falls back to empty string when the caller has not provided an
    X-Subject-Id (Plan 2 not yet rolled out, or trusted-network call without
    a header). Empty string is treated as "tenant-only caller" downstream:
    - list queries return only visibility='tenant' rows
    - mutations land with subject_id := tenant_id, preserving the legacy
      backfilled semantics for any tenant member.

    Returning a string (never raising) keeps this dependency optional from
    the caller's perspective and matches the plan's transition strategy.
    """
    return str(getattr(request.state, "subject_id", "") or "")


async def get_byo_mode(request: Request) -> str:
    """Get the tenant's BYO LLM entitlement from request state.

    Set by AuthenticationMiddleware from the license introspection response.
    Falls back to 'local_allowed' (never raises) so internal/trusted-network
    callers keep today's behavior; only tenants the licenses service marks
    'byo_required' are forced to bring their own provider.
    """
    return str(getattr(request.state, "byo_mode", "") or "local_allowed")


async def get_identity_proven(request: Request) -> bool:
    """True only when this request presented a licence token that was verified.

    ⛔ THIS IS A TWO-VALUED ANSWER TO A QUESTION THE REST OF THIS MODULE DOES
    NOT ASK. ``get_tenant_id`` answers "which tenant is this request scoped
    to"; it is satisfied on BOTH auth branches, because
    ``AuthenticationMiddleware`` resolves a tenant on both. This one answers
    "did anybody PROVE it", and only the token branch can say yes:

    * token branch — ``_introspect_license`` returned ``active: true`` and
      ``_stamp_identity`` set ``request.state.identity_proven = True``.
    * tokenless trusted-network branch — the caller was admitted on
      ``TRUSTED_CIDRS`` membership (the whole pod network by default) and
      NAMED ITS OWN TENANT in a bare ``X-Tenant-Id`` header. Stamped False.
    * bypass / auth disabled / public endpoint / no middleware at all — the
      attribute is absent.

    ⭐ FAIL CLOSED BY OMISSION, NOT BY REMEMBERING. The default is False, so a
    branch that never stamps anything (a future one included) is unproven
    without having to know this function exists. Do NOT "fix" that default to
    True to make a local run or a test convenient: the only consumer is a
    handler deciding whether to hand back a decrypted tenant credential, and
    an unproven caller getting one is the whole defect this exists to close.

    ⛔ Do NOT re-derive this from the peer CIDR, from ``license_id`` being
    None, or from any request header. The CIDR is exactly what failed to
    establish identity here, and a header on this path is caller-supplied.
    """
    return bool(getattr(request.state, "identity_proven", False))


async def get_tenant_proven(request: Request) -> bool:
    """True only when the caller's OWN credential established ``tenant_id``.

    ⛔ THIS IS THE SECOND HALF OF A QUESTION ``get_identity_proven`` ONLY
    ANSWERS HALF OF, AND CONFLATING THE TWO IS A CROSS-TENANT KEY HANDOUT.
    ``identity_proven`` answers "did this caller prove WHO IT IS". A delegating
    service account (subject ∈ ``AUTH_DELEGATING_SUBJECTS``) genuinely did: it
    presented a licence token and ``_introspect_license`` returned
    ``active: true``. But that same request may then be re-scoped to a
    DIFFERENT tenant by the ``X-Effective-Tenant-Id`` header, which
    ``_effective_tenant`` validates on SPELLING alone — exactly the check the
    tokenless branch is indicted for. Proving you are ``svc-requirements`` is
    not proving you may hold tenant X's provider credential.

    So this answers the other question: "was the tenant this request is scoped
    to established by the caller's own credential, or ADOPTED from a header?"

    * token branch, no delegation — the tenant came out of the introspected
      licence. Proven. True.
    * token branch, delegated — ``tenant_id`` was replaced by the
      ``X-Effective-Tenant-Id`` value. The caller is who it says it is; the
      tenant is still its own assertion. **False.**
    * tokenless trusted-network branch — the tenant is a bare header. False
      (``identity_proven`` is False there too; both say no, for the same
      reason).
    * bypass / auth disabled / public endpoint / no middleware — absent.

    ⭐ FAIL CLOSED BY OMISSION, in the same direction as ``get_identity_proven``
    and deliberately not in the direction the name "adopted" would have taken
    it. A flag spelled ``tenant_adopted`` would default to False = "not
    adopted" = hand the key over, so a future middleware branch that adopts a
    tenant and forgets to stamp would fail OPEN. Spelled as a proof, the
    default False means "nobody proved it" and a forgetful branch withholds.

    ⛔ Do NOT re-derive this from the presence of the header, from
    ``subject_id``, or from ``license_id`` being None. The header is
    caller-supplied, and on the delegating branch ``license_id`` is
    deliberately dropped to None for billing reasons that have nothing to do
    with entitlement — reading it here would couple a security gate to an
    attribution decision.
    """
    return bool(getattr(request.state, "tenant_proven", False))


async def get_visibility(
    x_visibility: Annotated[str, Header(alias="X-Visibility")] = "tenant",
) -> str:
    """Get desired visibility for newly-created rows from the X-Visibility header.

    Valid values: 'tenant' (default — every tenant member sees + may mutate)
    or 'user' (only the creating subject sees + may mutate). Anything else
    coerces to 'tenant' (the safer accidental default — exposes rather than
    over-restricts; legacy clients that omit the header are not broken).

    This is read-path-only: list/get/update/delete ignore it. Only create
    paths consume it to set the row's visibility column.

    Declared via ``Header`` so the parameter shows up on the OpenAPI surface
    (orval / docs) instead of being hidden inside ``Request``.
    """
    raw = (x_visibility or "tenant").strip().lower()
    return raw if raw in VISIBILITY_VALUES else "tenant"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Resolved per-request tenant + per-user scoping.

    Bundles the four scoping values that every tenant-scoped endpoint needs so
    handlers can declare a single ``ctx: TenantContext = Depends(get_tenant_context)``
    parameter instead of repeating the same four ``Depends(...)`` lines.
    """

    tenant_id: str
    subject_id: str
    tenant_name: str
    visibility: str
    byo_mode: str = "local_allowed"
    # Whether the tenant above was PROVEN (verified licence token) or merely
    # ASSERTED by a caller on the trusted network. See ``get_identity_proven``.
    # Defaults False so a context built by hand — in a test, a script, or a
    # future call site — is unproven unless it says otherwise.
    identity_proven: bool = False
    # Whether the tenant above was established by the caller's OWN credential
    # or ADOPTED from the X-Effective-Tenant-Id delegation header. See
    # ``get_tenant_proven``: a delegating service account has
    # ``identity_proven=True`` and ``tenant_proven=False`` at the same time,
    # and that combination is exactly the one that must not be handed a key.
    # Defaults False for the same reason as the field above.
    tenant_proven: bool = False


async def get_tenant_context(
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    subject_id: Annotated[str, Depends(get_subject_id)],
    tenant_name: Annotated[str, Depends(get_tenant_name)],
    visibility: Annotated[str, Depends(get_visibility)],
    byo_mode: Annotated[str, Depends(get_byo_mode)],
    identity_proven: Annotated[bool, Depends(get_identity_proven)],
    tenant_proven: Annotated[bool, Depends(get_tenant_proven)],
) -> TenantContext:
    """Compose the canonical scoping dependencies into one object.

    Reuses the existing dependency functions verbatim (same extraction,
    same 401 behaviour from ``get_tenant_id``); this only changes the call
    surface, not the resolution logic.
    """
    return TenantContext(
        tenant_id=tenant_id,
        subject_id=subject_id,
        tenant_name=tenant_name,
        visibility=visibility,
        byo_mode=byo_mode,
        identity_proven=identity_proven,
        tenant_proven=tenant_proven,
    )


__all__ = [
    "VISIBILITY_VALUES",
    "TenantContext",
    "get_byo_mode",
    "get_identity_proven",
    "get_resource_with_tenant_check",
    "get_subject_id",
    "get_tenant_context",
    "get_tenant_id",
    "get_tenant_name",
    "get_tenant_proven",
    "get_visibility",
]
