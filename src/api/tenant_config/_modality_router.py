"""BYO modality tenant-config endpoints (tts / asr / image / embedding).

``GET/PUT/DELETE /tenant-config/modality/{modality}`` +
``GET /tenant-config/modality/{modality}/{all,catalog}`` +
``GET /tenant-config/modality/{modality}/resolve``.

Mirrors the LLM tenant-config router (``_router.py``) with a ``modality`` path
segment. Every query is scoped to ``ctx.tenant_id`` (BOLA-safe); the API key is
AES-256-GCM encrypted at rest and NEVER returned by GET (only ``hasKey``).

The one NEW endpoint is ``/resolve``: the S2S seam the modality backends call
via ``w2v_common.clients.ModalityConfigClient``. WM is the ONLY holder of
``WM_SECRET_KEY``, so it decrypts the key here and returns the resolved
``{provider, model, base_url, api_key, cost_provider, cost_source, byo}`` — the
backends never decrypt (per-service-secrets invariant preserved). Resolution
precedence mirrors ``resolve_llm_config``: tier row → default row →
(byo_required ⇒ 409, else platform self-hosted default).

⛔ THAT SENTENCE USED TO END "over the trusted in-cluster network", AND THAT WAS
THE WHOLE BUG. "The trusted in-cluster network" is not a caller identity. This
is the only route in worker_management that returns a tenant's plaintext
provider credential in a response body, and ``AuthenticationMiddleware`` admits
a tokenless request from anywhere in ``TRUSTED_CIDRS`` (default ⊇ 10.0.0.0/8 ⊇
the k3s pod CIDR) and then lets it NAME ITS OWN TENANT in a bare ``X-Tenant-Id``
header, checked on spelling alone. The ``workers`` namespace is ambient L4 with
no waypoint, so there is no mTLS peer identity behind that either. Any pod in
the cluster — the ``n8n`` namespace, which runs user-authored workflows,
included — could therefore read any tenant's provider key by asserting that
tenant's uuid in a header.

``api_key`` is now populated ONLY for a caller that proved BOTH facts:
``ctx.identity_proven`` (it presented a verified licence token) AND
``ctx.tenant_proven`` (the tenant it is scoped to is the one that licence
established, not one it ADOPTED from ``X-Effective-Tenant-Id``). Any other
caller gets the same resolution with ``api_key=None`` and ``hasKey`` saying a
key exists that it was not handed, and the withholding is counted on
``worker_management_modality_key_withheld_total`` with a ``reason`` label. See
:func:`resolve_modality_config`.

⛔ ONE FLAG WAS ANSWERING TWO QUESTIONS, AND THAT WAS A RESIDUAL HANDOUT.
``identity_proven`` alone is not enough, and the first version of this gate
thought it was. A delegating service account (subject ∈
``AUTH_DELEGATING_SUBJECTS``) presents a real token — so ``identity_proven`` is
genuinely True — and may then re-scope the request to ANY tenant via
``X-Effective-Tenant-Id``, which ``_effective_tenant`` validates on spelling
alone, the very check this docstring indicts two paragraphs up. It got a
``tenant_proven=True`` stamp it had not earned and, with it, the victim tenant's
decrypted key. The second flag exists because "I proved who I am" and "I proved
I may act for this tenant" are different sentences. ⛔ This is NOT and must not become
``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``, which refuses fifteen route
shapes of live traffic and stays default-OFF: one field on one route is a
narrower change than a 401, and it did not have to wait for that flag.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.auth_middleware import _lookup_tenant_byo_mode
from src.api.error_handlers import APIError
from src.api.identity_context import get_caller_client_id_from_context
from src.api.tenant_constants import is_commercial_platform_tenant
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_TENANT_CONFIG_MODALITY_ALL,
    ROUTE_TENANT_CONFIG_MODALITY_CATALOG,
    ROUTE_TENANT_CONFIG_MODALITY_DELETE,
    ROUTE_TENANT_CONFIG_MODALITY_GET,
    ROUTE_TENANT_CONFIG_MODALITY_PUT,
    ROUTE_TENANT_CONFIG_MODALITY_RESOLVE,
    tokenless_tenant_gate,
)
from src.db.tenant_modality_config_db import DEFAULT_ROLE, MODALITIES, TenantModalityConfigDB
from src.security.managed_byo import assert_managed_row_change_allowed
from src.security.request_overrides import (
    TENANT_BASE_URL_NOT_ALLOWED,
    TenantBaseUrlNotAllowedError,
    check_tenant_base_url,
    prewarm_owner_infra_dns,
)
from src.security.secret_box import (
    SecretBoxError,
    decrypt_secret,
    encrypt_secret,
    has_secret_key,
)
from src.telemetry.agent_metrics import (
    MODALITY_KEY_WITHHELD_ADOPTED_TENANT,
    MODALITY_KEY_WITHHELD_INDETERMINATE,
    MODALITY_KEY_WITHHELD_OWNER_KEY,
    MODALITY_KEY_WITHHELD_UNPROVEN,
    get_metrics_client,
)
from src.worker.agents.modality_catalog import billing_provider_for, filter_for_mode, get_preset
from src.worker.factory.llm_config_resolver import WORK_TIERS, tier_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenant-config/modality", tags=["tenant-config"])

_db = TenantModalityConfigDB()

BYO_COST_SOURCE = "worker-management"

ModalityPath = Path(description="Modality: tts | asr | image | embedding")
TierQuery = Query(
    description="Work-size tier (small/medium/high). Omit to address the default row.",
)
ProductQuery = Query(
    description=(
        "Product/app key (licenses App.key). When given, BYO posture is resolved "
        "PER-PRODUCT (a tenant_app_byo_mode override wins over the tenant-wide "
        "byo_mode). Omit for tenant-wide posture (backward compatible)."
    ),
)


def _check_modality(modality: str) -> None:
    if modality not in MODALITIES:
        raise HTTPException(
            404, f"Unknown modality '{modality}' (expected one of {sorted(MODALITIES)})"
        )


# Error code a stored row with a refused base_url resolves to (409), the same
# reason string the LLM resolver uses for the same defect.
BASE_URL_NOT_ALLOWLISTED = "base_url_not_allowlisted"


def _checked_base_url(
    tenant_id: str, preset: dict[str, Any], base_url: Optional[str]
) -> Optional[str]:
    """Return a modality row's ``base_url`` normalised, or raise ``TenantBaseUrlNotAllowedError``.

    The endpoint a modality backend sends the tenant's key (or, for a managed
    ``owner_provided`` row, the OWNER's key) to gets the same validation as the
    LLM tenant config: on the global or that tenant's allowlist, and never owner
    infrastructure by name or by any address it resolves to
    (:func:`check_tenant_base_url`). Before this, ``PUT`` stored any URL and
    ``/resolve`` handed it out alongside the key.

    One narrow exception: a PLATFORM tenant may keep a preset's own
    ``default_base_url`` (the ``self_hosted`` presets name the owner's GPU
    services, e.g. ``http://tts.example.com:9202``). That is owner hardware on the
    owner's own tenant. A customer is refused it like any other internal URL,
    because a customer row pointing there runs on owner hardware billed as BYO.
    """
    if not base_url:
        return None
    default = preset.get("default_base_url")
    if (
        default
        and is_commercial_platform_tenant(str(tenant_id))
        and base_url.strip().rstrip("/") == str(default).rstrip("/")
    ):
        return base_url
    return check_tenant_base_url(str(tenant_id), base_url)


def _role_for_tier(tier: Optional[str]) -> str:
    if tier is None:
        return DEFAULT_ROLE
    if tier not in WORK_TIERS:
        raise HTTPException(422, f"Unknown tier '{tier}' (expected one of {sorted(WORK_TIERS)})")
    return tier_role(tier)


class ModalityConfigView(BaseModel):
    """Masked view returned by GET — never includes the API key."""

    configured: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = Field(default=None, serialization_alias="baseUrl")
    params: dict[str, Any] = Field(default_factory=dict)
    has_key: bool = Field(default=False, serialization_alias="hasKey")
    owner_provided: bool = Field(default=False, serialization_alias="ownerProvided")
    updated_at: Optional[str] = Field(default=None, serialization_alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class ModalityConfigUpdate(BaseModel):
    """PUT body. ``api_key`` (alias apiKey) absence vs presence is significant.

    ``owner_provided`` (alias ownerProvided) marks a managed-BYO key: absence
    preserves the existing value, mirroring ``api_key``'s preserve semantics.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: Optional[str] = Field(default=None, alias="baseUrl")
    params: dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[str] = Field(default=None, alias="apiKey")
    owner_provided: Optional[bool] = Field(default=None, alias="ownerProvided")

    model_config = ConfigDict(populate_by_name=True)


class ModalityResolveView(BaseModel):
    """S2S resolution result. ``api_key`` is the DECRYPTED key, or None.

    ⛔ ``api_key`` IS NOT UNCONDITIONAL, and it never was safe for it to be.
    It is populated only when the calling request proved BOTH that it is who it
    says (verified licence token) and that the tenant it is scoped to is the one
    that licence established rather than one adopted from
    ``X-Effective-Tenant-Id``. Every other caller — the tokenless
    trusted-network one (pod-network membership plus a bare ``X-Tenant-Id``
    header it chose itself) and the delegating service account alike — gets the
    whole resolution with ``api_key=None`` and ``has_key`` telling it a key
    exists that it was not given. See :func:`resolve_modality_config`.

    ``has_key`` mirrors the ``hasKey`` field the sibling GET view has always
    exposed, so an unproven caller can still distinguish "this tenant has no
    BYO key" from "this tenant has one and you were not handed it" without
    either case being a partial or an error. ⚠️ And note it therefore adds NO
    oracle: ``GET /tenant-config/modality/{modality}`` already returns
    ``hasKey`` to exactly the same callers under exactly the same tenant
    scoping. Nothing is disclosed here that was not already reachable one route
    over — considered and deliberately left as is, because removing it would
    only make the withholding indistinguishable from "no key configured".

    ⚠️ ``key_withheld`` REACHES NO CONSUMER TODAY, recorded rather than fixed
    here. ``w2v_common.clients.modality_config_client.ModalityConfigClient``
    builds its ``ModalityRoute`` from a fixed field list that does not include
    it, so a backend handed ``byo=True, api_key=None`` fails closed with no
    reason to log or surface. Surfacing it is a w2v_common change plus a bump in
    every modality backend, which is a different PR from this one and must not
    ride along with a security narrowing — the counter
    (``worker_management_modality_key_withheld_total``, now labelled by
    ``reason``) is the instrument that makes the gap visible in the meantime,
    and it is server-side, so it cannot be lost to a stale client.

    ``byo`` is True when an external tenant provider was resolved (the backend
    should build an external adapter + emit a byo cost_event); False means the
    backend falls open to its platform self-hosted default.
    """

    configured: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    has_key: bool = Field(default=False, serialization_alias="hasKey")
    key_withheld: bool = Field(default=False, serialization_alias="keyWithheld")
    cost_provider: Optional[str] = None
    cost_source: str = BYO_COST_SOURCE
    byo: bool = False
    owner_provided: bool = False
    role: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


def _to_view(row: Optional[dict[str, Any]]) -> ModalityConfigView:
    if not row:
        return ModalityConfigView(configured=False)
    updated = row.get("updated_at")
    return ModalityConfigView(
        configured=True,
        provider=row.get("provider"),
        model=row.get("model"),
        base_url=row.get("base_url"),
        params=row.get("params") or {},
        has_key=bool(row.get("api_key_encrypted")),
        owner_provided=bool(row.get("owner_provided")),
        updated_at=updated.isoformat() if updated else None,
    )


@router.get(
    "/{modality}",
    response_model=ModalityConfigView,
    response_model_by_alias=True,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_GET))],
)
async def get_modality_config(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> ModalityConfigView:
    """Return the tenant's config for (modality, tier) — masked."""
    _check_modality(modality)
    row = await _db.get_config(ctx.tenant_id, modality, _role_for_tier(tier))
    return _to_view(row)


@router.get(
    "/{modality}/all",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_ALL))],
)
async def list_modality_configs(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
) -> dict[str, Any]:
    """Return every configured row for the tenant + modality (masked)."""
    _check_modality(modality)
    rows = await _db.list_configs(ctx.tenant_id, modality)
    configs: list[dict[str, Any]] = []
    for row in rows:
        role = row.get("role") or DEFAULT_ROLE
        entry = _to_view(row).model_dump(by_alias=True)
        entry["role"] = role
        entry["tier"] = role[len("tier:") :] if role.startswith("tier:") else None
        configs.append(entry)
    return {"configs": configs}


@router.get(
    "/{modality}/catalog",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_CATALOG))],
)
async def get_modality_catalog(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
) -> dict[str, Any]:
    """Return the BYO provider catalog for the modality, gated by byo_mode."""
    _check_modality(modality)
    return {"providers": filter_for_mode(modality, ctx.byo_mode)}


@router.put(
    "/{modality}",
    response_model=ModalityConfigView,
    response_model_by_alias=True,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_PUT))],
)
async def put_modality_config(
    body: ModalityConfigUpdate,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> ModalityConfigView:
    """Upsert the tenant's config for (modality, tier). Encrypts the API key."""
    _check_modality(modality)
    role = _role_for_tier(tier)
    preset = get_preset(modality, body.provider)
    if preset is None:
        raise HTTPException(422, f"Unknown provider '{body.provider}' for modality '{modality}'")

    if ctx.byo_mode == "byo_required" and not preset["is_external"]:
        raise HTTPException(
            403, "This tenant must bring its own provider; platform inference is not permitted."
        )
    if ctx.byo_mode == "platform_only" and preset["is_external"]:
        raise HTTPException(
            403,
            "This tenant is locked to platform inference; external providers are not permitted.",
        )

    existing = await _db.get_config(ctx.tenant_id, modality, role)
    existing_key = existing.get("api_key_encrypted") if existing else None

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
        except SecretBoxError as exc:
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
    # (403). Checked before anything is validated or written.
    assert_managed_row_change_allowed(
        existing,
        provider=body.provider,
        base_url_in_body=body.base_url,
        key_in_body=key_provided,
        owner_provided_final=final_owner_provided,
    )

    base_url = (
        body.base_url
        if body.base_url is not None
        else (existing.get("base_url") if existing else None)
    )
    if preset["requires_base_url"] and not base_url:
        raise HTTPException(422, f"Provider '{body.provider}' requires a base_url")
    # Re-checked on every write, including a preserved stored value.
    try:
        await prewarm_owner_infra_dns(base_url)  # N4: DNS off the event loop
        base_url = _checked_base_url(ctx.tenant_id, preset, base_url)
    except TenantBaseUrlNotAllowedError as exc:
        raise APIError(422, str(exc), error_code=TENANT_BASE_URL_NOT_ALLOWED) from exc
    if preset["requires_api_key"] and not final_key:
        raise HTTPException(422, f"Provider '{body.provider}' requires an API key")

    await _db.upsert_config(
        tenant_id=ctx.tenant_id,
        modality=modality,
        role=role,
        provider=body.provider,
        model=body.model,
        base_url=base_url,
        params=body.params,
        api_key_encrypted=final_key,
        owner_provided=final_owner_provided,
    )
    logger.info(
        "tenant_modality_config upsert tenant=%s modality=%s role=%s provider=%s model=%s has_key=%s",
        ctx.tenant_id,
        modality,
        role,
        body.provider,
        body.model,
        bool(final_key),
    )
    row = await _db.get_config(ctx.tenant_id, modality, role)
    return _to_view(row)


@router.delete(
    "/{modality}",
    status_code=204,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_DELETE))],
)
async def delete_modality_config(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
    tier: Annotated[Optional[str], TierQuery] = None,
) -> None:
    """Delete the tenant's config for (modality, tier)."""
    _check_modality(modality)
    await _db.delete_config(ctx.tenant_id, modality, _role_for_tier(tier))


def _is_proven(ctx: TenantContext) -> bool:
    """Did the request that produced ``ctx`` prove its identity?

    ⭐ FAIL CLOSED. ``TenantContext.identity_proven`` already defaults to False,
    and this reads it through ``getattr`` with a False default as well, so a
    context object from an older shape, a hand-built one in a script, or
    anything that is not a ``TenantContext`` at all still comes out unproven.
    There is no input to this function that yields True by accident: only an
    explicit True, set by ``_stamp_identity`` after a licence token was
    introspected ``active``, gets a key out of this handler.

    ⚠️ SCOPE OF THAT DEFAULT, CORRECTED 2026-09-12 — this used to be sold as
    "fail closed TWICE OVER", implying the route depends on it. IT DOES NOT.
    ``_withhold_reason`` opens with a ``hasattr`` guard that returns
    ``indeterminate`` before this function is reached, so on the request path
    the default is UNREACHABLE and flipping it to True changes no endpoint
    behaviour — a mutant that flips it survives every endpoint test, which is
    how the overstatement was found. The default is the contract for DIRECT
    callers only (the resolve handler's own log line below, and any future
    importer), and it is pinned there by
    ``test_getattr_defaults_hold_for_a_direct_caller``, not by any route test.
    """
    return getattr(ctx, "identity_proven", False) is True


def _is_tenant_proven(ctx: TenantContext) -> bool:
    """Did the caller's OWN credential establish ``ctx.tenant_id``?

    The twin of :func:`_is_proven`, and the reason there are two. ``identity``
    and ``tenant`` are separate facts and a delegating service account splits
    them: it presented a real licence token (identity proven) and then re-scoped
    the request to somebody else's tenant with ``X-Effective-Tenant-Id``, a
    header the middleware validates on SPELLING alone (tenant NOT proven).

    Same double-default as :func:`_is_proven` — ``TenantContext.tenant_proven``
    defaults False and this ``getattr`` defaults False again, so an older
    context shape, a hand-built object, or a peer that rolls back to a release
    without the field all read as unproven — and the same ⚠️ correction applies
    to how far that reaches: ``_withhold_reason``'s ``hasattr`` guard fires
    first, so this default is a DIRECT-CALL guarantee, pinned by
    ``test_getattr_defaults_hold_for_a_direct_caller``, and not something any
    route test exercises.
    """
    return getattr(ctx, "tenant_proven", False) is True


def _withhold_reason(ctx: TenantContext) -> Optional[str]:
    """Why this caller may not hold the key, or ``None`` when it may.

    ⛔ THE ORDER OF THESE THREE TESTS IS THE LABEL CONTRACT, not style. Each
    population has a different owner and a different fix, and collapsing any
    two of them makes the smaller one unmeasurable inside the larger:

    1. A shape that does not carry BOTH flags cannot be interrogated at all →
       ``indeterminate``. Checked FIRST so "we could not tell" can never be
       reported as a verdict we reached. A context carrying ``identity_proven``
       but not ``tenant_proven`` — what a rollback looks like — lands here
       rather than being read as entitled on the strength of the half it has.
    2. Identity itself unproven → ``unproven_identity``. The tokenless
       trusted-network caller, unchanged. Its fix is "present a token".
    3. Identity proven, tenant adopted → ``adopted_tenant``. The delegating
       service account. Its fix is NOT "present a token" — it already has one —
       so it must not be counted with (2) or its owners will chase the wrong
       remediation.

    Returning ``None`` requires BOTH flags to be explicitly True. Every other
    combination, including any that does not exist yet, withholds.
    """
    if not hasattr(ctx, "identity_proven") or not hasattr(ctx, "tenant_proven"):
        return MODALITY_KEY_WITHHELD_INDETERMINATE
    if not _is_proven(ctx):
        return MODALITY_KEY_WITHHELD_UNPROVEN
    if not _is_tenant_proven(ctx):
        return MODALITY_KEY_WITHHELD_ADOPTED_TENANT
    return None


def _managed_key_caller_clients() -> frozenset[str]:
    """The verified ``azp`` values allowed to receive an owner_provided key (D5)."""
    from src.settings import get_settings

    raw = get_settings().managed_byo_key_caller_clients or ""
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


def _owner_key_withhold_reason() -> Optional[str]:
    """``None`` iff the caller is a VERIFIED, listed service (D5); else the reason.

    ⛔ An owner_provided (managed-BYO) row holds the OWNER's credential. The
    tenant's own licence token proves the tenant, and that is exactly the
    party that must NOT hold the owner's key in clear (it could spend it
    anywhere, unmetered). So the identity/tenant proofs that release a tenant's
    OWN key are irrelevant here: only the service that spends the key on the
    tenant's behalf may receive it. ``caller_client_id`` comes solely from a
    signature-verified, ``CALLER_ALLOWED_CLIENTS``-listed
    ``X-Caller-Authorization`` JWT (``AuthenticationMiddleware`` is its only
    setter), never from a request header.
    """
    caller = get_caller_client_id_from_context()
    if caller and caller in _managed_key_caller_clients():
        return None
    return MODALITY_KEY_WITHHELD_OWNER_KEY


def _key_for_caller(
    ctx: TenantContext,
    modality: str,
    usable_key: Optional[str],
    *,
    owner_provided: bool = False,
) -> Optional[str]:
    """The key to put in the response body — ``None`` unless the caller proved itself.

    ``owner_provided`` (managed BYO): released ONLY to a verified service caller
    (:func:`_owner_key_withhold_reason`), never on the tenant's own token. A
    tenant's own BYO key keeps the identity+tenant proof rule below.

    Returns ``usable_key`` only for a caller that proved BOTH who it is and its
    entitlement to THIS tenant, ``None`` otherwise, and counts every
    withholding on ``worker_management_modality_key_withheld_total`` with the
    reason, so the question "does a legitimate caller actually depend on this
    key path, and which one?" stays measurable instead of becoming an
    assumption.

    ⛔ ORDERING IS LOAD-BEARING: the decision is made and returned on its own,
    and the counter is best-effort AFTER it. A metrics backend that is down
    must never be able to hand out a secret, so there is no path where a
    telemetry failure changes the return value.
    """
    if usable_key is None:
        return None
    reason = _owner_key_withhold_reason() if owner_provided else _withhold_reason(ctx)
    if reason is None:
        return usable_key
    if owner_provided:
        logger.warning(
            "modality_resolve WITHHELD the decrypted owner_provided (managed-BYO) %s key "
            "(tenant=%s, reason=%s): that key is the OWNER's and is released only to a "
            "verified service caller listed in MANAGED_BYO_KEY_CALLER_CLIENTS, never on "
            "the tenant's own token.",
            modality,
            ctx.tenant_id,
            reason,
        )
    else:
        _log_tenant_key_withheld(ctx, modality, reason)
    try:
        get_metrics_client().record_modality_key_withheld(modality, reason=reason)
    except Exception:  # pragma: no cover - defensive; telemetry must not gate a secret
        logger.debug("modality key withheld counter failed", exc_info=True)
    return None


def _log_tenant_key_withheld(ctx: TenantContext, modality: str, reason: str) -> None:
    logger.warning(
        "modality_resolve WITHHELD the decrypted %s key (tenant=%s, reason=%s): the caller "
        "did not prove entitlement to this tenant. The caller received the resolution with "
        "api_key=None and hasKey=true. If this is a legitimate backend, give it a licence "
        "token FOR THIS TENANT — do not widen this route, and note that a delegating "
        "service account's own token does not answer this.",
        modality,
        ctx.tenant_id,
        reason,
    )


@router.get(
    "/{modality}/resolve",
    response_model=ModalityResolveView,
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_RESOLVE))],
)
async def resolve_modality_config(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    modality: Annotated[str, ModalityPath],
    tier: Annotated[Optional[str], TierQuery] = None,
    product: Annotated[Optional[str], ProductQuery] = None,
) -> ModalityResolveView:
    """S2S: resolve the effective provider for (tenant, modality, tier).

    Returns the DECRYPTED key — ⛔ BUT ONLY TO A CALLER WHOSE IDENTITY WAS
    PROVEN. This is the only route in worker_management that puts a tenant's
    plaintext BYO provider credential in a response body, and the branch that
    admits most of its callers proves nothing: ``AuthenticationMiddleware``
    accepts a tokenless request from anywhere in ``TRUSTED_CIDRS`` (the pod
    network) and lets it NAME ITS OWN TENANT in a bare ``X-Tenant-Id`` header,
    validated on spelling alone. The ``workers`` namespace is ambient L4 with
    no waypoint, so there is no mTLS peer identity to fall back on either.
    Unfixed, any pod in the cluster could read any tenant's provider key by
    asserting that tenant's uuid in a header.

    So ``api_key`` is populated only when ``ctx.identity_proven`` AND
    ``ctx.tenant_proven`` are both True — see ``src.auth.get_identity_proven``
    and ``src.auth.get_tenant_proven``. Everyone else gets the SAME resolution
    with ``api_key=None`` and ``has_key=True``, mirroring the ``hasKey`` shape
    the sibling GET has always used; it is not a 401, not a 500 and not a
    partial, because the posture/provider/model half of this answer was never
    secret and refusing it outright would break resolution for callers that do
    not need the key at all.

    ⛔ THE SECOND FLAG IS NOT REDUNDANT. A delegating service account satisfies
    the first and not the second: it holds a verified token (identity proven)
    and then names somebody else's tenant in ``X-Effective-Tenant-Id``, a
    header checked on spelling alone. Gating on ``identity_proven`` by itself
    left that path handing out the ADOPTED tenant's key — narrower than the
    tokenless hole, but the same hole. Delegation itself is untouched: the
    request still resolves for the adopted tenant, still bills it, still gets
    provider/model/base_url. Only the credential is withheld, and the reason is
    counted as ``adopted_tenant`` so that population is visible rather than
    hidden inside the tokenless one.

    ⭐ The gate keys off WHICH AUTH BRANCH RAN, never off the peer CIDR: the
    CIDR is precisely the thing that failed to establish identity. And it
    fails closed — an indeterminate answer withholds (see ``_withhold_reason``).

    ⛔ This is deliberately NOT
    ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``. That flag refuses
    fifteen route shapes of live traffic and stays default-OFF; this narrows
    one field on one route. Turning the credential off does not wait on the
    flag, and does not move it.

    Precedence: tier row → default row → platform default. ``byo_required`` with
    no usable external row raises 409 (the backend must not fall back to owner
    hardware). ``platform_only`` ignores any stored external row (byo=False).

    ``product`` (optional): resolve BYO posture PER-PRODUCT — a tenant may be
    e.g. ``byo_required`` for product A but ``platform_only`` for product B.
    Omit for the tenant-wide posture from the request context (unchanged).

    ⚠️ ROUND 16: this route now also carries
    ``dependencies=[Depends(tokenless_tenant_gate(ROUTE_TENANT_CONFIG_MODALITY_RESOLVE))]``
    for the ``worker_management_tokenless_tenant_request_total`` COUNT ONLY.
    That dependency runs before this function, never touches its return
    value, and (with ``require_authenticated_tenant`` staying False) never
    raises — so it changes nothing about the ``identity_proven``/
    ``tenant_proven`` withholding above. It exists purely so an operator can
    see this route's tokenless traffic volume alongside every other route on
    this surface; the credential decision above remains the sole gate on the
    response body.
    """
    _check_modality(modality)
    role = _role_for_tier(tier)

    # Per-product posture: when a product is named, re-resolve byo_mode against
    # the licenses per-(tenant, product) override; else use the tenant-wide value
    # already on the request context.
    byo_mode = ctx.byo_mode
    if product:
        byo_mode = await _lookup_tenant_byo_mode(ctx.tenant_id, product)

    row = await _db.get_config(ctx.tenant_id, modality, role)
    if row is None and role != DEFAULT_ROLE:
        row = await _db.get_config(ctx.tenant_id, modality, DEFAULT_ROLE)

    preset = None
    usable_key: Optional[str] = None
    if row is not None:
        preset = get_preset(modality, row["provider"])
        if preset is None:
            logger.warning(
                "tenant_modality_config for %s/%s names unknown provider %s; ignoring",
                ctx.tenant_id,
                modality,
                row["provider"],
            )
            row = None
        else:
            enc = row.get("api_key_encrypted")
            if enc:
                try:
                    usable_key = decrypt_secret(enc)
                except SecretBoxError:
                    logger.exception(
                        "tenant_modality_config key for %s/%s failed to decrypt; "
                        "falling back (fail-closed)",
                        ctx.tenant_id,
                        modality,
                    )
                    row = None
            elif preset["requires_api_key"]:
                row = None

    # platform_only: drop any stored external row (BYO disabled); a local row
    # is still honored as a platform-model selection.
    if byo_mode == "platform_only" and row and preset is not None and preset["is_external"]:
        logger.info(
            "modality_resolve tenant=%s modality=%s platform_only: ignoring external row",
            ctx.tenant_id,
            modality,
        )
        row = None

    # The endpoint the backend will send this row's key to (the tenant's, or the
    # OWNER's for a managed row) must pass the PUT validation again: a row
    # written before it existed, written directly, or whose name now resolves
    # somewhere internal. ⛔ No fallback: a refused row is a clear 409, never a
    # silent switch to the platform default.
    resolved_base_url: Optional[str] = None
    if row is not None and preset is not None:
        try:
            await prewarm_owner_infra_dns(row.get("base_url"))  # N4: DNS off the event loop
            resolved_base_url = _checked_base_url(ctx.tenant_id, preset, row.get("base_url"))
        except TenantBaseUrlNotAllowedError as exc:
            logger.warning(
                "modality_resolve tenant=%s modality=%s provider=%s: stored base_url refused (%s)",
                ctx.tenant_id,
                modality,
                row["provider"],
                exc,
            )
            raise APIError(
                409,
                f"This tenant's {modality} provider '{row['provider']}' cannot be used: {exc}. "
                f"Fix the endpoint in the tenant's {modality} settings.",
                error_code=BASE_URL_NOT_ALLOWLISTED,
            ) from exc

    if row is not None and preset is not None:
        cost_provider = billing_provider_for(
            modality, row["provider"], row["model"], preset["provider"]
        )
        has_key = usable_key is not None
        returned_key = _key_for_caller(
            ctx, modality, usable_key, owner_provided=bool(row.get("owner_provided"))
        )
        logger.info(
            "modality_resolve tenant=%s modality=%s role=%s provider=%s model=%s byo=%s "
            "has_key=%s key_returned=%s identity_proven=%s tenant_proven=%s",
            ctx.tenant_id,
            modality,
            row.get("role", role),
            preset["provider"],
            row["model"],
            preset["is_external"],
            has_key,
            returned_key is not None,
            _is_proven(ctx),
            _is_tenant_proven(ctx),
        )
        return ModalityResolveView(
            configured=True,
            provider=preset["provider"],
            model=row["model"],
            base_url=resolved_base_url,
            api_key=returned_key,
            has_key=has_key,
            key_withheld=has_key and returned_key is None,
            cost_provider=cost_provider,
            cost_source=row.get("cost_source") or BYO_COST_SOURCE,
            byo=bool(preset["is_external"]),
            owner_provided=bool(preset["is_external"]) and bool(row.get("owner_provided")),
            role=row.get("role", role),
        )

    # No usable tenant row.
    if byo_mode == "byo_required":
        raise HTTPException(
            409,
            f"This tenant must configure its own {modality} provider before use.",
        )
    return ModalityResolveView(configured=False, byo=False)
