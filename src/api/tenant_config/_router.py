"""BYO LLM tenant-config endpoints.

``GET/PUT/DELETE /tenant-config/llm`` + ``GET /tenant-config/llm/catalog`` +
``GET /tenant-config/llm/audit`` (the customer-visible history of every PUT,
DELETE and key TEST on the tenant's config, owner ruling 2026-10-02).

Every query is scoped to ``ctx.tenant_id``, so a tenant can only ever
read/write its OWN config. Role gating (tenant_admin) lives in fr, which
proxies here. The API key is encrypted at rest (AES-256-GCM) and NEVER
returned: GET exposes only ``hasKey``.

PUT key semantics: ``apiKey`` absent = preserve existing, ``apiKey:""`` = clear,
any other value = set.

⛔ THIS IS *NOT* "BOLA-SAFE BY CONSTRUCTION", AND THIS DOCSTRING USED TO SAY IT
WAS. The retracted sentence read "Every query is scoped to ``ctx.tenant_id``
(from the license token), so a tenant can only ever read/write its OWN config —
BOLA-safe by construction." The parenthetical is the load-bearing half and it is
FALSE on one of the two branches that reach these handlers. ``get_tenant_context``
is a SCOPING dependency, not an authentication check: ``get_tenant_id``
(``src/auth/tenant.py:205-206``) reads back ``request.state.tenant_id``, and on the
tokenless trusted-network branch that value was stamped at
``src/api/auth_middleware.py:668`` from the caller's own bare ``X-Tenant-Id``
header, accepted on SYNTAX alone (any canonical non-nil uuid; ``TENANT_UUIDS`` is
a name→uuid convenience map, not an allowlist). So any pod inside
``TRUSTED_CIDRS`` (default ⊇ 10.0.0.0/8 ⊇ the k3s pod CIDR) can name any tenant
and read its rows. BOLA-safety holds on the TOKEN branch only.

── PER-ROUTE TOKENLESS VERDICT, recorded for nonagenticai/worker_management#233 ──
Measured 2026-09-11 against the running Prometheus + Loki, after
``WorkerManagementTokenlessTenantSurfaceGrew`` (gitops-production
``clusters/strato/services/prometheus/rules-c.yaml``, floor ``> 13``) reached 15.

``GET /tenant-config/llm/all``  — ⛔ AUTH GAP. NOT intended, NOT admitted.
    ``_db.list_configs(ctx.tenant_id)`` is ``WHERE tenant_id = $1``
    (``src/db/tenant_llm_config_db.py:51-62``) executed under a tenant the caller
    merely asserted. That is a genuine cross-tenant read primitive: it returns
    provider, model, base_url, temperature, max_tokens, params, hasKey and
    ownerProvided for every configured tier. The key itself stays masked
    (``_to_view`` → ``has_key=bool(...)``), so this is NOT the credential-bearing
    class that ``/tenant-config/modality/{tts,image,embedding}/resolve`` was — but
    "masked" is not "scoped".

    ⭐ UPDATED 2026-09-12: that credential-bearing class is NARROWED — ⛔ NOT
    CLOSED, and this paragraph said "CLOSED" for one day. Retracted, because an
    overstated verdict in the ledger is worse than the gap it overstates: the
    next reader stops looking. The modality ``/resolve`` handler now returns
    ``api_key`` only to a caller that proved BOTH its identity (verified licence
    token) and its entitlement to the tenant it is scoped to, and reports
    ``hasKey`` to everyone else (``_modality_router.resolve_modality_config``).

    ⛔ RESIDUAL PATH, NAMED. The first version of that gate read
    ``identity_proven`` alone, and a DELEGATING service account (subject ∈
    ``AUTH_DELEGATING_SUBJECTS``, ``auth_middleware.py`` token branch) satisfies
    it while holding a tenant it merely ASSERTED: ``X-Effective-Tenant-Id`` is
    validated on spelling alone, exactly like the tokenless branch's
    ``X-Tenant-Id``. Probed against ``e3bcc30``, that path yielded
    ``{'tenant_id': <victim>, 'identity_proven': True}`` and therefore the
    victim's decrypted key. It is closed HERE by a second flag
    (``TenantContext.tenant_proven``, stamped ``not delegated``); the class is
    called narrowed rather than closed because the SHAPE of the defect — one
    boolean standing in for a question it does not answer — is a property of
    this auth stack, not of this one route, and ``/llm/all`` below is still a
    cross-tenant read primitive under an asserted tenant.

    ⚠️ WHY THIS WAS SAFE IN THE MEANTIME, measured rather than assumed:
    ``tenant_modality_config`` on the store the pods actually use holds
    **1 row, 0 of them with an api_key** (independently re-derived
    2026-09-12). There was no key on that table for the residual path to hand
    to anybody. That is a statement about TODAY'S DATA, not about the code —
    the first tenant to configure a modality BYO key would have made it live.

    ⛔ Read all of the above as fields on one route, NOT as this docstring's
    verdict being discharged: the three modality routes are still
    tokenless-readable, ``/llm/all`` is still a cross-tenant read primitive, and
    the fix below — the CALLER presenting a token — is still the fix. It was
    always the credential, not the ratchet, that was urgent; the credential
    exposure is PRE-EXISTING and sits among the original 13 shapes, so it is not
    what moved the surface count to 15.

``GET /tenant-config/llm/catalog`` — ⛔ AUTH GAP, lesser. NOT intended.
    Returns the static ``filter_for_mode(ctx.byo_mode)`` catalog, no DB read. The
    leak is the FILTER, not the rows: ``byo_mode`` is resolved for the asserted
    tenant at ``auth_middleware.py:709``, so the shape of the response is a
    per-tenant commercial-posture oracle (byo_required / platform_only /
    local_allowed). Note ``_lookup_tenant_byo_mode`` fail-opens to
    ``local_allowed`` on a licenses non-200 (``auth_middleware.py:411-420``), so a
    404 there yields the UNFILTERED catalog.

``GET /tenant-config/llm/audit`` — ✅ REFUSED TOKENLESS (401). NOT on the surface.
    Added 2026-10-02 (WM #320) and measured live after merge 7fafcfd3 to serve a
    tokenless trusted-CIDR caller naming any tenant in a bare ``X-Tenant-Id``
    (provenance ``asserted``): a cross-tenant read of who set / tested / removed
    whose key. Unlike the routes above it had NO tokenless caller to break — its
    only intended caller is fr, which calls it DELEGATED (svc-fr-backend licence
    + ``X-Effective-Tenant-Id``). So it was refused at birth rather than joining
    the surface: ``_require_licence_or_delegated_caller`` admits provenance
    ``license``, and ``delegated`` only from a subject in
    ``AUTH_DELEGATING_SUBJECTS``; ``asserted`` / ``default`` / ``bypass`` /
    ``unknown`` get 401 regardless of ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``.
    It carries no ``tokenless_tenant_gate`` and has no label in
    ``TOKENLESS_TENANT_REQUEST_ROUTES``: a refused route is not tokenless
    surface, and the gate (a route-level dependency, so it runs first) would
    count every refused request as ``served``. ⚠️ ``delegated`` still means the
    effective tenant is ASSERTED by a trusted service account, not proven — the
    same residual as every delegated route; this closes the tokenless half only.

⚠️ INTENT IS UNRECORDED, NOT INTENDED. Nothing in src/, tests/, docs/, or the
alert text ever stated that tokenless read of these routes was deliberate. The
only statements that touched the question assumed a token (the retracted sentence
above) or said the opposite (``auth_middleware._audit_unauthenticated_tenant_assertion``
docstring: "observe now, refuse later"). So this is an absence of a decision, not
a decision — do not read the historical silence as consent.

⛔ AND YET DO NOT REFUSE THESE ROUTES TODAY, AND DO NOT RAISE THE RATCHET FLOOR.
Both were considered and both are wrong right now; the measurements are the
reason, so they are recorded here rather than left for the next reader to redo.
 (1) BLAST RADIUS OF REFUSING. The tokenless caller is ONE workload:
     fr-backend (Loki ztunnel ``src.workload=fr-*``, ns ``fr``, UA ``Bun/1.3.13``,
     peers 10.42.6.141 on 2026-09-09 and 10.42.6.112 on 2026-09-10). It is
     tokenless BY EXPLICIT DESIGN — ``fr/backend/src/integrations/workers.ts``
     constructs its worker_management client with ``sendAuthorization: false``.
     It therefore has NO token to present, so a 401 here is not a migration
     prompt, it is an outage of fr's LLM-provider settings page. ⚠️ The low
     request count (3 + 3 over 30d on these two shapes) is a measure of how new
     the fr feature is, NOT of how few users depend on it — do not read it as
     permission to break the route.
 (2) THE FLOOR MUST NOT MOVE TO 15 — and it must not be moved DOWN on this
     change either. The original reason 15 was not a ceiling has since been
     removed at the source: ``auth_middleware`` used to derive the ``path`` label
     from a character heuristic (``_metric_path``) that preserved alphabetic
     segments, so ``/providers/{name}/models`` emitted a DISTINCT shape per
     provider name — 11 possible from that one route (``PROVIDER_FETCHERS``) —
     and any alphabetic path parameter anywhere could mint a new one. The label
     is now the MATCHED ROUTE TEMPLATE (``auth_middleware._route_shape``, read
     from ``scope["route"]``), so that route contributes exactly ONE shape and
     the whole label set is bounded by the route table. ⛔ Do NOT re-baseline the
     floor to whatever the smaller number turns out to be until it has been
     MEASURED after this ships — the shapes stop growing, but a previously
     unobserved route can still appear, and a floor set to an observation is the
     same mistake in a smaller costume. The residue is still the fr caller, not
     the threshold.
 (3) THE REAL FIX IS THE CALLER. fr must present its own licence token in
     ``X-Tenant-Id: Bearer <token>``, exactly as ``auth_middleware`` asks. ⛔ NOT by
     forwarding ``X-License-Id`` — that lets any admitted peer assert a LICENCE as
     well as a tenant, a strictly larger hole (kag refuses this in as many words,
     ``kag/src/auth/tenant.py``). Once fr holds a token, arming
     ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` closes this without a
     per-route special case.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Optional

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.api.error_handlers import APIError
from src.api.identity_context import (
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    get_end_user_identity,
    get_tenant_provenance_from_context,
)
from src.api.tenant_constants import is_commercial_platform_tenant
from src.auth import TenantContext, get_tenant_context
from src.auth.end_user_token import delegating_subjects
from src.auth.tokenless_gate import (
    ROUTE_TENANT_CONFIG_LLM_ALL,
    ROUTE_TENANT_CONFIG_LLM_CATALOG,
    ROUTE_TENANT_CONFIG_LLM_DELETE,
    ROUTE_TENANT_CONFIG_LLM_GET,
    ROUTE_TENANT_CONFIG_LLM_PUT,
    ROUTE_TENANT_CONFIG_LLM_TEST,
    tokenless_tenant_gate,
)
from src.db.tenant_llm_config_db import (
    AUDIT_LIMIT_DEFAULT,
    AUDIT_LIMIT_MAX,
    DEFAULT_ROLE,
    TenantLLMConfigDB,
    key_fingerprint,
)
from src.security.managed_byo import assert_managed_row_change_allowed
from src.security.request_overrides import (
    TENANT_BASE_URL_NOT_ALLOWED,
    TenantBaseUrlNotAllowedError,
    check_tenant_base_url,
    prewarm_owner_infra_dns,
)
from src.security.secret_box import (
    SecretKeyMissingError,
    encrypt_secret,
    has_secret_key,
)
from src.settings import get_settings
from src.worker.agents.provider_catalog import (
    PaidUpstreamForLocalPresetError,
    assert_local_preset_stays_on_hardware,
    filter_for_mode,
    get_preset,
)
from src.worker.factory.llm_config_resolver import (
    WORK_TIERS,
    evaluate_tenant_row,
    owner_key_test_entitled,
    tenant_key_entitled,
    tier_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenant-config", tags=["tenant-config"])

_db = TenantLLMConfigDB()

# OpenAPI annotation for the optional tier selector shared by the CRUD routes.
TierQuery = Query(
    description=(
        "Work-size tier to address (small/medium/high). Omit to address the "
        "tenant default row (Phase-1 behavior)."
    ),
)


def _role_for_tier(tier: Optional[str]) -> str:
    """Map an optional ``?tier=`` to a DB role, rejecting unknown values.

    Absent ⇒ ``__default__``. A non-empty value must be a known work tier.
    """
    if tier is None:
        return DEFAULT_ROLE
    if tier not in WORK_TIERS:
        raise HTTPException(422, f"Unknown tier '{tier}' (expected one of {sorted(WORK_TIERS)})")
    return tier_role(tier)


class LLMConfigView(BaseModel):
    """Masked view returned by GET — never includes the API key."""

    configured: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = Field(default=None, serialization_alias="baseUrl")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, serialization_alias="maxTokens")
    params: dict[str, Any] = Field(default_factory=dict)
    has_key: bool = Field(default=False, serialization_alias="hasKey")
    owner_provided: bool = Field(default=False, serialization_alias="ownerProvided")
    updated_at: Optional[str] = Field(default=None, serialization_alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class LLMConfigAuditEntry(BaseModel):
    """One customer-visible audit entry. No key, no key hash, no actor ``sub``."""

    at: str
    tier: Optional[str] = None
    action: str
    actor: str
    provider: Optional[str] = None
    key_id: str = Field(serialization_alias="keyId")

    model_config = ConfigDict(populate_by_name=True)


class LLMConfigAuditView(BaseModel):
    """``GET /tenant-config/llm/audit`` response: newest first."""

    entries: list[LLMConfigAuditEntry]


class LLMConfigUpdate(BaseModel):
    """PUT body. ``api_key`` (alias apiKey) absence vs presence is significant.

    ``owner_provided`` (alias ownerProvided) marks a managed-BYO key: absence
    preserves the existing value, mirroring ``api_key``'s preserve semantics.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: Optional[str] = Field(default=None, alias="baseUrl")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens", ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[str] = Field(default=None, alias="apiKey")
    owner_provided: Optional[bool] = Field(default=None, alias="ownerProvided")

    model_config = ConfigDict(populate_by_name=True)


def _acting_for_another_tenant(tenant_id: str) -> bool:
    """A verified end-user token is present and names a different tenant."""
    sub, token_tenant, _super_admin, _delegator = get_end_user_identity()
    return bool(sub) and token_tenant != str(tenant_id).strip().lower()


def _key_id(encrypted: Optional[str | bytes]) -> str:
    """Non-reversible KEY ID for the owner_action line AND the audit row.

    12 hex of SHA-256 of the STORED (encrypted) value — never of the plaintext
    key. ``none`` when no key is involved. One helper for both, so the operator
    log and the customer's audit entry name the same key the same way.
    """
    return key_fingerprint(encrypted)


def _actor_kind(tenant_id: str) -> str:
    """Who is acting, for the CUSTOMER-visible audit row. Never the ``sub``.

    * ``platform_operator`` — a verified end-user JWT carrying portal
      ``super_admin`` acting for ANOTHER tenant (the same "acting for another
      tenant" test as the operator ``owner_action`` line).
    * ``workspace_admin`` — a verified end-user JWT whose tenant IS the
      effective tenant (the customer's own user, delegated through fr).
    * ``service`` — everything else: no verified end-user JWT (licence token,
      tokenless in-mesh caller), or a foreign non-super_admin user.
    """
    sub, token_tenant, super_admin, _delegator = get_end_user_identity()
    if _acting_for_another_tenant(tenant_id) and super_admin:
        return "platform_operator"
    if sub and token_tenant == str(tenant_id).strip().lower():
        return "workspace_admin"
    return "service"


def _put_action(*, key_provided: bool, api_key: Optional[str]) -> str:
    """``set_key`` / ``clear_key`` / ``update`` — mirrors PUT's key semantics."""
    if not key_provided:
        return "update"
    return "clear_key" if api_key == "" else "set_key"


def _tier_of(role: str) -> Optional[str]:
    """``tier:medium`` → ``medium``; the default row → ``None`` (as ``/llm/all``)."""
    return role[len("tier:") :] if role.startswith("tier:") else None


def _audit_owner_action(
    action: str,
    tenant_id: str,
    tier: Optional[str],
    provider: Optional[str],
    *,
    key_id: str = "none",
    **extra: object,
) -> None:
    """INFO audit line when someone acts on ANOTHER tenant's LLM config.

    Owner requirement 2026-10-02. Fires when the request carried a VERIFIED
    end-user portal JWT (delegated branch, ``end_user_token``) whose
    ``tenant_id`` differs from the effective tenant being written/tested — i.e.
    the owner (or anyone) acting FOR a customer tenant. Records tenant, tier,
    provider, the actor's ``sub`` and ``key_id`` (see ``_key_id``: a hash of the
    ENCRYPTED stored value, ``none`` when no key); NEVER the key, a hash of the
    plaintext key, or the token.
    """
    if not _acting_for_another_tenant(tenant_id):
        return
    sub, token_tenant, super_admin, _delegator = get_end_user_identity()
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        "tenant_llm_config owner_action action=%s tenant=%s tier=%s provider=%s "
        "actor_sub=%s actor_tenant=%s actor_super_admin=%s key_id=%s%s",
        action,
        tenant_id,
        tier or "default",
        provider or "-",
        sub,
        token_tenant or "-",
        super_admin,
        key_id,
        tail,
    )


def _to_view(row: Optional[dict[str, Any]]) -> LLMConfigView:
    if not row:
        return LLMConfigView(configured=False)
    updated = row.get("updated_at")
    return LLMConfigView(
        configured=True,
        provider=row.get("provider"),
        model=row.get("model"),
        base_url=row.get("base_url"),
        temperature=row.get("temperature"),
        max_tokens=row.get("max_tokens"),
        params=row.get("params") or {},
        has_key=bool(row.get("api_key_encrypted")),
        owner_provided=bool(row.get("owner_provided")),
        updated_at=updated.isoformat() if updated else None,
    )


@router.get(
    "/llm",
    response_model=LLMConfigView,
    response_model_by_alias=True,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_GET))],
)
async def get_llm_config(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> LLMConfigView:
    """Return the tenant's LLM config for ``tier`` (default row when omitted, masked)."""
    row = await _db.get_config(ctx.tenant_id, _role_for_tier(tier))
    return _to_view(row)


@router.get(
    "/llm/all",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_ALL))],
)
async def list_llm_configs(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, Any]:
    """Return every configured row for the tenant (masked), one per role/tier.

    Each entry is the masked :class:`LLMConfigView` plus its ``role`` and the
    derived ``tier`` (``None`` for the default row) — the shape the multi-tier
    settings UI consumes.
    """
    rows = await _db.list_configs(ctx.tenant_id)
    configs: list[dict[str, Any]] = []
    for row in rows:
        role = row.get("role") or DEFAULT_ROLE
        entry = _to_view(row).model_dump(by_alias=True)
        entry["role"] = role
        entry["tier"] = role[len("tier:") :] if role.startswith("tier:") else None
        configs.append(entry)
    return {"configs": configs}


@router.get(
    "/llm/catalog",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_CATALOG))],
)
async def get_llm_catalog(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, Any]:
    """Return the BYO provider catalog for the settings UI, gated by byo_mode.

    A commercial (``byo_required``) tenant never sees the platform ``local``
    preset; a ``platform_only`` tenant sees only ``local``.
    """
    return {"providers": filter_for_mode(ctx.byo_mode)}


AUDIT_TOKENLESS_REFUSED = (
    "GET /tenant-config/llm/audit requires a verified licence token, or a delegated "
    "call from a configured delegating service account; a bare X-Tenant-Id is not accepted."
)


async def _require_licence_or_delegated_caller(request: Request) -> None:
    """401 unless the tenant was established by a verified licence token.

    Admits provenance ``license``, and ``delegated`` only when the introspected
    DELEGATOR (``request.state.license['subject_id']`` — not the forwarded
    ``X-Subject-Id``, which replaces ``subject_id`` on that branch) is in
    ``AUTH_DELEGATING_SUBJECTS``. Everything else — ``asserted`` (tokenless,
    bare ``X-Tenant-Id``), ``default`` (tokenless, no header), ``bypass`` and
    ``unknown`` — is refused, independent of
    ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``. Fail closed by omission:
    a future auth branch that stamps no provenance reads ``unknown`` and is
    refused. See the module docstring's verdict for this route.
    """
    provenance = get_tenant_provenance_from_context()
    proven = bool(getattr(request.state, "identity_proven", False))
    if proven and provenance == TENANT_PROVENANCE_LICENSE:
        return
    if proven and provenance == TENANT_PROVENANCE_DELEGATED:
        introspected = getattr(request.state, "license", None) or {}
        delegator = str(introspected.get("subject_id") or "")
        if delegator and delegator in delegating_subjects():
            return
    logger.warning("tenant_llm_config audit refused provenance=%s", provenance)
    raise HTTPException(status_code=401, detail=AUDIT_TOKENLESS_REFUSED)


@router.get(
    "/llm/audit",
    response_model=LLMConfigAuditView,
    response_model_by_alias=True,
    # ⛔ NOT ``tokenless_tenant_gate``: this route is refused tokenless, so it
    # is not on that surface (see the module docstring's verdict).
    dependencies=[Depends(_require_licence_or_delegated_caller)],
)
async def get_llm_config_audit(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    limit: Annotated[
        int,
        Query(ge=1, description=f"Max entries (capped at {AUDIT_LIMIT_MAX})."),
    ] = AUDIT_LIMIT_DEFAULT,
) -> LLMConfigAuditView:
    """The tenant's BYO-config history, newest first (owner ruling 2026-10-02).

    Every PUT / DELETE / key TEST on this tenant's LLM config, by anyone —
    including the platform operator acting for the tenant. Scoped to
    ``ctx.tenant_id`` only. ``limit`` above ``AUDIT_LIMIT_MAX`` is clamped, not
    refused. Each entry names WHAT kind of actor acted (``actor``), never who.
    """
    rows = await _db.list_audit(ctx.tenant_id, min(limit, AUDIT_LIMIT_MAX))
    return LLMConfigAuditView(
        entries=[
            LLMConfigAuditEntry(
                at=row["created_at"].isoformat(),
                tier=_tier_of(row.get("role") or DEFAULT_ROLE),
                action=row["action"],
                actor=row["actor_kind"],
                provider=row.get("provider"),
                key_id=row["key_id"],
            )
            for row in rows
        ]
    )


@router.put(
    "/llm",
    response_model=LLMConfigView,
    response_model_by_alias=True,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_PUT))],
)
async def put_llm_config(
    body: LLMConfigUpdate,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> LLMConfigView:
    """Upsert the tenant's LLM config for ``tier`` (default row when omitted).

    Validates the selection against the catalog and encrypts the API key before
    storage. ``apiKey`` absent preserves the existing key; ``""`` clears it.
    """
    role = _role_for_tier(tier)
    preset = get_preset(body.provider)
    if preset is None:
        raise HTTPException(422, f"Unknown provider '{body.provider}'")

    # Enforce the tenant's byo_mode at the write path too (the catalog already
    # hides disallowed presets, but the API must not trust the UI): a commercial
    # tenant may not store the platform ``local`` preset; a platform_only tenant
    # may not store an external BYO provider.
    if ctx.byo_mode == "byo_required" and not preset["is_external"]:
        raise HTTPException(
            403, "This tenant must bring its own provider; platform inference is not permitted."
        )
    if ctx.byo_mode == "platform_only" and preset["is_external"]:
        raise HTTPException(
            403,
            "This tenant is locked to platform inference; external providers are not permitted.",
        )

    existing = await _db.get_config(ctx.tenant_id, role)
    existing_key = existing.get("api_key_encrypted") if existing else None

    # Resolve the final encrypted key: preserve / clear / set.
    key_provided = "api_key" in body.model_fields_set
    if not key_provided:
        final_key = existing_key
    elif body.api_key == "":
        final_key = None
    else:
        if not has_secret_key():
            raise HTTPException(503, "BYO key storage is not configured (WM_SECRET_KEY unset)")
        try:
            final_key = encrypt_secret(body.api_key or "")
        except SecretKeyMissingError as exc:
            raise HTTPException(503, "BYO key storage is not configured") from exc

    # owner_provided: absent in the body preserves the prior value (mirrors the
    # api_key preserve semantics) so a plain provider/model edit never silently
    # resets a managed-BYO key to tenant-provided.
    if "owner_provided" in body.model_fields_set and body.owner_provided is not None:
        final_owner_provided = body.owner_provided
    else:
        final_owner_provided = bool(existing.get("owner_provided")) if existing else False

    # A managed (owner_provided) row carries the OWNER's key: its provider,
    # base_url, key and mark are the owner's to change, never the tenant's
    # (403, src/security/managed_byo.py). Checked before anything is written.
    assert_managed_row_change_allowed(
        existing,
        provider=body.provider,
        base_url_in_body=body.base_url,
        key_in_body=key_provided,
        owner_provided_final=final_owner_provided,
    )

    # Requirement checks against the catalog.
    base_url = (
        body.base_url
        if body.base_url is not None
        else (existing.get("base_url") if existing else None)
    )
    if preset["requires_base_url"] and not base_url:
        raise HTTPException(422, f"Provider '{body.provider}' requires a base_url")
    # The endpoint this service will send the tenant's (or, owner_provided, the
    # platform's) key to must be allowlisted — globally or for this tenant.
    # Re-checked on every write, including a preserved stored value.
    try:
        await prewarm_owner_infra_dns(base_url)  # N4: DNS off the event loop
        base_url = check_tenant_base_url(str(ctx.tenant_id), base_url)
    except TenantBaseUrlNotAllowedError as exc:
        raise APIError(422, str(exc), error_code=TENANT_BASE_URL_NOT_ALLOWED) from exc
    if preset["requires_api_key"] and not final_key:
        raise HTTPException(422, f"Provider '{body.provider}' requires an API key")

    # A platform (non-external) preset must not be paired with a model the
    # router serves on a PAID upstream. Two such rows — provider='local',
    # model='tier-medium' — spent $140.31 over 46,102 requests on DeepSeek for a
    # tenant whose configuration said "platform inference", and nobody ever
    # named DeepSeek. See assert_local_preset_stays_on_hardware for the full
    # measurement and for why this refuses where the read-side rule coerces.
    try:
        assert_local_preset_stays_on_hardware(body.provider, body.model)
    except PaidUpstreamForLocalPresetError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Owner ruling 2026-10-01: a CUSTOMER's platform (non-external, `local` →
    # inference-router) row may name only the one customer model, local-llm.
    # Any other IR model (grok-4-fast, `default`, a routing-rule schema id) is
    # the owner's routing and, upstream, possibly the owner's commercial keys.
    # The worker path coerces such a row at read time too
    # (enforce_customer_routing); refusing it here stops it being stored at all.
    # Platform tenants keep the full IR model choice.
    local_llm = get_settings().local_llm_model
    if (
        not preset["is_external"]
        and not is_commercial_platform_tenant(str(ctx.tenant_id))
        and body.model != local_llm
    ):
        raise HTTPException(
            422,
            f"Model '{body.model}' is not available to this tenant on platform inference "
            f"(provider '{body.provider}'). Use model '{local_llm}', or select an external "
            "provider with your own API key.",
        )

    await _db.upsert_config(
        tenant_id=ctx.tenant_id,
        role=role,
        provider=body.provider,
        model=body.model,
        base_url=base_url,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        params=body.params,
        api_key_encrypted=final_key,
        owner_provided=final_owner_provided,
        # Same transaction as the change: no audit row ⇒ no change (owner
        # ruling 2026-10-02 — the customer must see every write to its key).
        audit_action=_put_action(key_provided=key_provided, api_key=body.api_key),
        audit_actor_kind=_actor_kind(str(ctx.tenant_id)),
    )
    logger.info(
        "tenant_llm_config upsert tenant=%s role=%s provider=%s model=%s has_key=%s",
        ctx.tenant_id,
        role,
        body.provider,
        body.model,
        bool(final_key),
    )
    _audit_owner_action(
        "put",
        str(ctx.tenant_id),
        tier,
        body.provider,
        key_id=_key_id(final_key),
        model=body.model,
        has_key=bool(final_key),
    )
    # Best-effort: drop cached agent instances so the next run rebuilds against
    # the new provider/key instead of reusing a stale one.
    try:
        from src.api.workers._globals import worker_cache

        await worker_cache.invalidate_tenant(ctx.tenant_id)
    except Exception:
        logger.warning("tenant_llm_config: cache invalidation failed for %s", ctx.tenant_id)

    row = await _db.get_config(ctx.tenant_id, role)
    return _to_view(row)


@router.delete(
    "/llm",
    status_code=204,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_DELETE))],
)
async def delete_llm_config(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> None:
    """Delete the tenant's LLM config for ``tier`` (default row when omitted)."""
    role = _role_for_tier(tier)
    if _acting_for_another_tenant(str(ctx.tenant_id)):
        existing = await _db.get_config(ctx.tenant_id, role)
        _audit_owner_action(
            "delete",
            str(ctx.tenant_id),
            tier,
            (existing or {}).get("provider"),
            key_id=_key_id((existing or {}).get("api_key_encrypted")),
        )
    # Same transaction as the delete: the ``remove`` audit row (provider and key
    # id of the row actually deleted) commits with it or not at all.
    await _db.delete_config(ctx.tenant_id, role, audit_actor_kind=_actor_kind(str(ctx.tenant_id)))
    try:
        from src.api.workers._globals import worker_cache

        await worker_cache.invalidate_tenant(ctx.tenant_id)
    except Exception:
        logger.warning("tenant_llm_config: cache invalidation failed for %s", ctx.tenant_id)


# POST /tenant-config/llm/test cooldown: the last live outcome per (tenant,
# provider, key fingerprint, endpoint) is reused for this many seconds, so a
# caller (including a tokenless in-mesh one on a keyless `local` row) cannot
# turn the endpoint into a vendor / inference-router request pump.
KEY_TEST_COOLDOWN_S = 10
_key_test_cooldown: TTLCache[tuple[str, str, str, str], tuple[bool, list[str], str, int]] = (
    TTLCache(maxsize=1024, ttl=KEY_TEST_COOLDOWN_S)
)
# Single-flight (validator 3 N2): concurrent tests of the same credential that
# all miss the cooldown share ONE live call instead of each reaching the vendor.
_key_test_inflight: dict[
    tuple[str, str, str, str], asyncio.Task[tuple[bool, list[str], str, int]]
] = {}


class LLMKeyTestRequest(BaseModel):
    """POST /tenant-config/llm/test body: ``{tier}`` OR ``{provider, model?}``.

    Neither ⇒ the tenant's ``__default__`` row.
    """

    tier: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class LLMKeyTestResult(BaseModel):
    """Outcome of one live call with a tenant row's credential. Never the key."""

    ok: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    # Set when the key works but the configured model is not in the provider's
    # model list for that key (an alias the vendor accepts but does not list
    # would otherwise read as a broken key).
    warning: Optional[str] = None


async def _row_for_test(tenant_id: str, body: LLMKeyTestRequest) -> Optional[dict[str, Any]]:
    """The tenant row the test addresses (same fallback as the resolver)."""
    if body.provider:
        for row in await _db.list_configs(tenant_id) or []:
            preset = get_preset(row.get("provider") or "")
            if preset and body.provider in (preset["provider"], preset["key"]):
                return dict(row)
        return None
    role = _role_for_tier(body.tier)
    row = await _db.get_config(tenant_id, role)
    if row is None and role != DEFAULT_ROLE:
        row = await _db.get_config(tenant_id, DEFAULT_ROLE)
    return dict(row) if row else None


@router.post(
    "/llm/test",
    response_model=LLMKeyTestResult,
    response_model_exclude_none=True,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_LLM_TEST))],
)
async def test_llm_config(
    body: LLMKeyTestRequest,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> LLMKeyTestResult:
    """Test a tenant LLM row's key with ONE minimal live call (owner ruling 2026-10-01).

    Works for ANY row the tenant has configured — an external BYO / managed-BYO
    key, or a platform (``local`` → inference-router) row. The call is the
    provider's MODEL LIST with that row's credential: free (no tokens billed),
    authenticated on every supported vendor (a bad key is a 401/403 there), and
    the same call ``GET /providers`` uses to decide which rows to list. A
    1-token completion would also prove quota/billing, but it costs money on
    every click and needs the full agent stack; it is deliberately not used, so
    a key with exhausted credit can still test ``ok`` here.

    ``ok`` is false (HTTP 200) for a missing row, a withheld/undecryptable key,
    a refused endpoint, or a vendor that rejects the key. The key is never
    returned or logged.
    """
    if body.tier is not None and body.provider is not None:
        raise HTTPException(422, "Send either {tier} or {provider, model}, not both.")
    tenant_id = str(ctx.tenant_id)
    row = await _row_for_test(tenant_id, body)
    if row is None:
        what = (
            f"provider '{body.provider}'" if body.provider else f"tier '{body.tier or 'default'}'"
        )
        return LLMKeyTestResult(
            ok=False,
            provider=body.provider,
            model=body.model,
            error=f"No LLM configuration for {what} on this tenant. Save one first.",
        )
    preset = get_preset(row.get("provider") or "")
    model = body.model or row.get("model")
    if preset is None:
        return LLMKeyTestResult(
            ok=False,
            provider=row.get("provider"),
            model=model,
            error=f"Provider '{row.get('provider')}' is not in the catalog.",
        )
    provider = preset["provider"]
    await prewarm_owner_infra_dns(row.get("base_url"))  # N4: DNS off the event loop
    # The tenant's own proven user, OR (owner requirement 2026-10-02) a portal
    # ``super_admin`` testing a customer tenant's key. The owner predicate is
    # used HERE ONLY: it never reaches the resolver, /providers or /chat.
    own = tenant_key_entitled(tenant_id)
    owner = (not own) and owner_key_test_entitled(tenant_id)
    verdict = evaluate_tenant_row(tenant_id, row, preset, entitled=own or owner)
    usable = verdict["status"] == "usable"
    # WM-4: the customer-visible row is written AFTER the verdict, so a test
    # refused before any live call (withheld / undecryptable key, refused
    # endpoint) is recorded as ``test_refused``, never as a ``test`` that did
    # not happen. Still BEFORE the live call and FAIL-CLOSED: a test the
    # customer could not see is refused (503) rather than run. A test changes
    # no config, so there is no config write to share a transaction with.
    action = "test" if usable else "test_refused"
    try:
        await _db.record_audit(
            tenant_id=tenant_id,
            role=row.get("role") or DEFAULT_ROLE,
            action=action,
            actor_kind=_actor_kind(tenant_id),
            provider=provider,
            key_id=_key_id(row.get("api_key_encrypted")),
        )
    except Exception as exc:
        logger.exception(
            "tenant_llm_config audit write failed tenant=%s action=%s", tenant_id, action
        )
        raise HTTPException(503, "Could not record the key test in the audit log.") from exc
    _audit_owner_action(
        action,
        tenant_id,
        body.tier,
        provider,
        key_id=_key_id(row.get("api_key_encrypted")),
        owner_key_test=owner,
    )
    if not usable:
        return LLMKeyTestResult(
            ok=False, provider=provider, model=model, error=verdict["error"].message
        )

    from src.api.providers._router import ByoLiveCheckError, _byo_cache_key, live_row_models

    # Per-tenant cooldown (validator N-d): within KEY_TEST_COOLDOWN_S the last
    # live outcome for this tenant's row credential is returned instead of a new
    # vendor / inference-router call. Keyed like the /providers live cache —
    # (tenant, provider, key fingerprint, endpoint) — so a saved key edit tests
    # at once, and one tenant never sees another's result.
    cooldown_key = _byo_cache_key(tenant_id, provider, verdict["entry"])
    cached = _key_test_cooldown.get(cooldown_key)
    if cached is None:
        task = _key_test_inflight.get(cooldown_key)
        if task is None:
            entry = verdict["entry"]

            async def _live_check() -> tuple[bool, list[str], str, int]:
                t0 = time.perf_counter()
                try:
                    ids = await live_row_models(tenant_id, provider, entry, fresh=True)
                    outcome = (True, ids, "", int((time.perf_counter() - t0) * 1000))
                except ByoLiveCheckError as exc:
                    outcome = (False, [], str(exc), int((time.perf_counter() - t0) * 1000))
                _key_test_cooldown[cooldown_key] = outcome
                return outcome

            task = asyncio.ensure_future(_live_check())
            _key_test_inflight[cooldown_key] = task
            task.add_done_callback(lambda _t, k=cooldown_key: _key_test_inflight.pop(k, None))
        # shield: one waiter's disconnect must not cancel the call the others share.
        cached = await asyncio.shield(task)
    live_ok, ids, live_error, latency_ms = cached
    if not live_ok:
        return LLMKeyTestResult(
            ok=False,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            error=f"{provider} rejected the call with this key: {live_error}",
        )
    logger.info(
        "tenant_llm_key_test tenant=%s provider=%s ok=True latency_ms=%d",
        tenant_id,
        provider,
        latency_ms,
    )
    warning = (
        None
        if not model or model in ids
        else (
            f"The key works, but model '{model}' is not in {provider}'s model list for "
            "this key; check the model name."
        )
    )
    return LLMKeyTestResult(
        ok=True, provider=provider, model=model, latency_ms=latency_ms, warning=warning
    )
