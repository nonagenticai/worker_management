"""Run-time overlay of per-tenant BYO LLM config onto a worker's config.

Every run path funnels through ``WorkerInstanceCache._create_worker_instance``;
calling :func:`resolve_llm_config` there means fr / kag / coding-ai / karma all
inherit a tenant's BYO provider with no per-app change.

Precedence (highest first):

1. **Genuine per-worker override** — the worker config carries an explicit
   ``api_key_override`` OR a non-system ``model_provider`` (an admin already
   pinned a specific external provider for this worker). Left untouched.
2. **Tenant tier row** — a known ``work_tier`` (small/medium/high) selects the
   tenant's per-tier ``tenant_llm_config`` row (DB role ``tier:<work_tier>``).
   Used when present + usable.
3. **Tenant default** — the worker is system-routed (inference_router / mlx /
   vllm / vlm_local) with no explicit key, AND the tenant has a usable
   ``tenant_llm_config`` row at role ``__default__``. Also the fallback when a
   ``work_tier`` is given but that tier has no configured row. The row's
   provider/model/base_url/key (and temperature/max_tokens where the worker
   didn't set them) are overlaid, and the call goes DIRECT to that provider
   (bypassing inference-router).
4. **System default** — no usable tenant row. If ``byo_mode == 'byo_required'``
   (commercial tenant), raise :class:`BYORequiredError` (hard-fail — never ride
   owner GPU). Otherwise leave the worker on its seeded system routing.

``byo_mode == 'platform_only'`` (demo/test/trial — locked to platform inference)
short-circuits BYO: any stored *external* row is ignored and the worker stays on
its seeded system routing (inference-router). A stored ``local`` row is still
honored (it only selects a platform model), so per-tier platform selection works.

A ``work_tier`` is a property of the *task* (carried on the worker ``config``);
absent or unknown values resolve exactly as Phase 1 did (``__default__`` only),
so the change is fully backward compatible.

A decrypt failure is fail-closed: treated as "no usable key" (falls to case 3),
never surfaces plaintext or crashes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from src.api.identity_context import (
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    get_end_user_identity,
    get_end_user_proof,
    get_end_user_proof_reason,
    get_tenant_id_from_context,
    get_tenant_provenance_from_context,
)
from src.api.tenant_constants import is_commercial_platform_tenant
from src.auth.tenant_proof import owner_commercial_keys_refusal
from src.db.tenant_llm_config_db import DEFAULT_ROLE, TenantLLMConfigDB
from src.security.request_overrides import (
    TenantBaseUrlNotAllowedError,
    check_tenant_base_url,
    inference_router_base_for,
    is_owner_infra_url,
    prewarm_owner_infra_dns,
)
from src.security.secret_box import (
    SecretBoxError,
    decrypt_secret,
)
from src.settings import get_settings
from src.worker.agents.provider_catalog import (
    ON_HARDWARE_BILLING_PROVIDERS,
    billing_provider_for,
    get_preset,
    is_on_hardware_ir_pin,
)

logger = logging.getLogger(__name__)

# Providers that route to owner hardware and are therefore overridable by a
# tenant's BYO config. Mirrors SELF_HOSTED_PROVIDERS in pydantic_ai_agent.py
# (kept local to avoid importing that heavy module here).
SYSTEM_ROUTED: frozenset[str] = frozenset({"inference_router", "mlx", "vllm", "vlm_local"})

BYO_COST_SOURCE = "worker-management"

# Work-size tiers a caller may request. A tier maps to the DB role ``tier:<tier>``;
# the ``__default__`` row is always the fallback. Unknown/absent ⇒ default only.
WORK_TIERS: frozenset[str] = frozenset({"small", "medium", "high"})

#: The all-zeros UUID. Never a tenant, and specifically never a PRIVILEGED one.
_NIL_TENANT = "00000000-0000-0000-0000-000000000000"


# Reserved system/internal tenants: platform-owned runs (e2e/calibration, the
# ...0001 sentinel) route to seeded SYSTEM LLM config even when a
# transient/misclassified byo_mode says byo_required — they must never 402.
# Extra IDs come from the GitOps-set WM_SYSTEM_TENANT_IDS env (comma-separated).
#
# ⛔ THE NIL UUID WAS IN THIS SET AND HAS BEEN REMOVED. It did not merely fail to
# reject the nil tenant — it EXEMPTED it, granting the estate's anonymous
# sentinel the one privilege in this module (skip the byo_required hard-fail and
# fall through to seeded platform routing, i.e. spend on owner hardware that no
# customer can be charged for). `tenant_id` here is `ctx.tenant_id`, so it is
# reachable from any request whose tenant resolves to nil.
#
# The env is filtered for the same value: an allow-list that nothing validates is
# a one-ConfigMap-typo path back to the same exemption, which would make removing
# the literal above cosmetic.
def _system_tenants(env_raw: str) -> frozenset[str]:
    """Build the reserved-tenant set from ``WM_SYSTEM_TENANT_IDS``.

    A named pure function rather than an inline comprehension so the nil filter
    is testable WITHOUT ``importlib.reload`` — reloading this module rebinds
    module-level ``_db``, which silently detaches every other test's monkeypatch
    and turns one test's setup into another test's failure.
    """
    return frozenset(
        {
            "00000000-0000-0000-0000-000000000001",
            *(
                t.strip()
                for t in env_raw.split(",")
                if t.strip() and t.strip().lower() != _NIL_TENANT
            ),
        }
    )


_SYSTEM_TENANTS: frozenset[str] = _system_tenants(os.getenv("WM_SYSTEM_TENANT_IDS", ""))


def tier_role(work_tier: Optional[str]) -> str:
    """Map a work_tier to its ``tenant_llm_config`` role.

    Returns ``tier:<work_tier>`` for a known tier, else :data:`DEFAULT_ROLE`.
    """
    if work_tier and work_tier in WORK_TIERS:
        return f"tier:{work_tier}"
    return DEFAULT_ROLE


_db = TenantLLMConfigDB()


def _allowlisted_row(
    tenant_cfg: Optional[dict[str, Any]], tenant_id: Optional[str]
) -> Optional[dict[str, Any]]:
    """Return ``tenant_cfg`` with its ``base_url`` normalised, or None if refused.

    The endpoint must be allowlisted (globally or for this tenant) before this
    service sends the tenant's — or, ``owner_provided``, the platform's — key
    there. ``PUT /tenant-config/llm`` refuses one; this is the fail-closed twin
    for rows written before that check or directly in the DB. A refused row is
    treated as absent, so a ``byo_required`` tenant hard-fails instead of relaying.
    """
    if not tenant_cfg or not tenant_cfg.get("base_url"):
        return tenant_cfg
    try:
        return {
            **tenant_cfg,
            "base_url": check_tenant_base_url(tenant_id, tenant_cfg["base_url"]),
        }
    except TenantBaseUrlNotAllowedError:
        logger.exception(
            "tenant_llm_config base_url for %s is not allowlisted; ignoring row (fail-closed)",
            tenant_id,
        )
        return None


def tenant_key_entitled(tenant_id: Optional[str]) -> bool:
    """May THIS request cause ``tenant_id``'s stored provider key to be decrypted?

    worker_management#245. True only when the current request's tenant was
    PROVEN by the caller's own licence (``tenant_provenance == license`` — the
    ContextVar twin of ``request.state.tenant_proven``, stamped by
    ``AuthenticationMiddleware``) AND it is the very tenant being resolved.

    ⛔ Every other provenance is a CLAIM, not a proof, and gets no key:
    ``delegated`` (a service account re-scoped the request with
    ``X-Effective-Tenant-Id``, checked for spelling only), ``asserted`` /
    ``default`` (tokenless trusted-network ``X-Tenant-Id``), ``bypass`` (auth
    switched off) and ``unknown`` (no request context at all, e.g. a background
    task). Same predicate as ``_modality_router._withhold_reason``: both flags
    must be positively established, and anything unstated withholds.

    The tenant-equality conjunct is what makes "tenant A can never get tenant
    B's key" hold on paths that resolve a tenant from somewhere other than the
    request — e.g. the graph worker node prefers graph-state ``tenant_id`` over
    the request ContextVar. A licence proves ONE tenant; it proves nothing about
    a different one named in state.
    """
    if not tenant_id:
        return False
    tenant = str(tenant_id).strip().lower()
    if (get_tenant_id_from_context() or "").strip().lower() != tenant:
        return False
    provenance = get_tenant_provenance_from_context()
    if provenance == TENANT_PROVENANCE_LICENSE:
        return True
    if provenance == TENANT_PROVENANCE_DELEGATED:
        # Owner ruling 2026-10-01: a delegated request is entitled ONLY when the
        # middleware verified the forwarded END-USER portal JWT
        # (``X-End-User-Token``: signature, issuer, expiry, ``tenant_id`` claim
        # == the effective tenant) AND the subject that forwarded it is still a
        # configured delegator. The service account alone proves nothing.
        from src.auth.end_user_token import delegating_subjects

        proven_tenant, delegator = get_end_user_proof()
        return (
            bool(proven_tenant)
            and proven_tenant == tenant
            and bool(delegator)
            and delegator in delegating_subjects()
        )
    return False


def owner_key_test_entitled(tenant_id: Optional[str]) -> bool:
    """May THIS request TEST ``tenant_id``'s stored key on the owner's behalf?

    Owner requirement 2026-10-02: the platform owner (portal realm role
    ``super_admin``) configures a customer tenant's BYO key from fr's admin
    tenant page and must be able to test it green/red. fr delegates as
    ``svc-fr-backend`` with ``X-Effective-Tenant-Id`` = the customer tenant and
    forwards the owner's own portal JWT, whose ``tenant_id`` is the OWNER's
    tenant, so ``tenant_key_entitled`` (rightly) says no.

    True only when ALL hold: the request is DELEGATED, it is scoped to this very
    tenant, the forwarded end-user token VERIFIED (signature via the portal
    JWKS, issuer, expiry: ``end_user_token._verified_payload``), its
    ``realm_access.roles`` carry ``super_admin``, it has a ``sub``, and the
    subject that forwarded it is still in ``AUTH_DELEGATING_SUBJECTS``.

    ⛔ USE ONLY AT ``POST /tenant-config/llm/test``. This is deliberately a
    SEPARATE predicate and ``tenant_key_entitled`` is NOT widened: the owner must
    never run inference, /chat, or a provider listing on a customer's key. The
    test returns a boolean and a safe error, never the key.
    """
    if not tenant_id:
        return False
    tenant = str(tenant_id).strip().lower()
    if (get_tenant_id_from_context() or "").strip().lower() != tenant:
        return False
    if get_tenant_provenance_from_context() != TENANT_PROVENANCE_DELEGATED:
        return False
    from src.auth.end_user_token import delegating_subjects

    sub, _token_tenant, super_admin, delegator = get_end_user_identity()
    return (
        super_admin is True and bool(sub) and bool(delegator) and delegator in delegating_subjects()
    )


def key_withheld_reason(tenant_id: Optional[str]) -> str:
    """Human-readable reason this request may NOT decrypt ``tenant_id``'s key.

    Never contains key or token material. Used in the BYO-key-withheld error so
    the caller is told exactly what to send.
    """
    provenance = get_tenant_provenance_from_context()
    if (get_tenant_id_from_context() or "").strip().lower() != str(tenant_id or "").lower():
        return "the request is not for this tenant"
    if provenance == TENANT_PROVENANCE_DELEGATED:
        reason = get_end_user_proof_reason() or "absent"
        return (
            "the request was delegated by a service account and the signed-in user's "
            f"token did not prove the tenant (X-End-User-Token: {reason})"
        )
    return f"the tenant was not proven by the caller (tenant_provenance={provenance})"


def _unlock_row_key(
    tenant_cfg: dict[str, Any],
    tenant_id: Optional[str],
    *,
    requires_api_key: bool,
    key_entitled: bool,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return ``(row_or_None, plaintext_key_or_None)`` for a known-provider row.

    ⛔ worker_management#245: the stored key is decrypted ONLY when
    ``key_entitled``. For an unproven or adopted tenant the ciphertext is never
    touched and the row is treated as unusable, exactly like a decrypt failure
    (fail-closed). Only the provenance is logged, never the value.
    """
    enc = tenant_cfg.get("api_key_encrypted")
    if enc and not key_entitled:
        logger.warning(
            "tenant_llm_config key for %s WITHHELD (tenant_provenance=%s): the caller did "
            "not prove this tenant with its own licence; the stored key is not decrypted "
            "and the row is treated as unusable",
            tenant_id,
            get_tenant_provenance_from_context(),
        )
        return None, None
    if enc:
        try:
            return tenant_cfg, decrypt_secret(enc)
        except SecretBoxError:
            logger.exception(
                "tenant_llm_config key for %s failed to decrypt; falling back (fail-closed)",
                tenant_id,
            )
            return None, None
    if requires_api_key:
        # Stored row lost its key somehow; not usable.
        return None, None
    return tenant_cfg, None


class BYORequiredError(Exception):
    """A byo_required tenant has no usable BYO LLM config (hard-fail)."""


_UNUSABLE_REASON_TEXT: dict[str, str] = {
    "withheld": "its key was not decrypted for this request",
    "decrypt_failed": "the stored key could not be decrypted",
    "missing_key": "the row has no API key",
    "base_url_not_allowlisted": "its base_url is not on the endpoint allowlist",
    "base_url_owner_infra": (
        "its base_url points at platform infrastructure, which a tenant provider may never use"
    ),
    "unknown_provider": "the provider is not in the catalog",
}


class TenantLLMKeyUnusableError(Exception):
    """A tenant ``tenant_llm_config`` row exists but its key cannot be used.

    Owner ruling 2026-10-01: never silently fall back to platform or
    ``local-llm`` routing. ``status_code``/``error_code``/``details`` are the
    HTTP contract every route maps this to:

    * ``reason == "withheld"`` → **403** ``BYO_KEY_WITHHELD`` (the caller did not
      prove the tenant; the fix is on the CALLER: forward the end user's token);
    * anything else → **409** ``BYO_KEY_UNUSABLE`` (the stored row is broken;
      the fix is in the tenant's LLM settings).

    The message names the tenant's provider and tier, never a key, env var or
    owner credential.
    """

    def __init__(
        self,
        *,
        tenant_id: Optional[str],
        provider: str,
        tier: Optional[str],
        role: Optional[str],
        reason: str,
        withheld_detail: str = "",
    ) -> None:
        self.tenant_id = str(tenant_id or "")
        self.provider = provider
        self.tier = tier or "default"
        self.role = role or DEFAULT_ROLE
        self.reason = reason
        self.withheld = reason == "withheld"
        self.status_code = 403 if self.withheld else 409
        self.error_code = "BYO_KEY_WITHHELD" if self.withheld else "BYO_KEY_UNUSABLE"
        # The row that failed may be the __default__ fallback for a tier request:
        # point the fix at the ROW, not at the tier that was asked for.
        tier_q = f"?tier={self.role[5:]}" if self.role.startswith("tier:") else ""
        if self.withheld:
            self.fix = (
                "Forward the signed-in user's portal access token in the X-End-User-Token "
                "header (delegated calls), or call with the tenant's own licence token."
            )
        else:
            self.fix = (
                f"Re-enter the {provider} API key (and endpoint) in the tenant LLM settings "
                f"(PUT /tenant-config/llm{tier_q}), then check it with "
                "POST /tenant-config/llm/test."
            )
        why = _UNUSABLE_REASON_TEXT.get(reason, reason)
        if withheld_detail:
            why = f"{why}: {withheld_detail}"
        self.message = (
            f"Tenant {self.tenant_id} is configured to use its own {provider} key for the "
            f"'{self.tier}' tier, but {why}. The request was NOT run on a platform model "
            f"instead. Fix: {self.fix}"
        )
        super().__init__(self.message)

    def details(self) -> dict[str, Any]:
        """Structured, secret-free error details (``error.details`` in the envelope)."""
        return {
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "tier": self.tier,
            "reason": self.reason,
            "fix": self.fix,
        }


def _is_genuine_override(config: dict[str, Any]) -> bool:
    if config.get("api_key_override"):
        return True
    provider = config.get("model_provider")
    return bool(provider) and provider not in SYSTEM_ROUTED


# Billing providers that run on OWNER hardware. `_BILLING_PROVIDER_OVERRIDES`
# already records, per `local`-preset model, which upstream actually serves it —
# it has to, or the run prices at $0 — so it is the repo's existing source of
# truth for "does this alias leave the cluster", and it is kept current because a
# new alias that omits an entry bills nothing and gets noticed.
# Single source of truth, shared with the write-side guard
# (`assert_local_preset_stays_on_hardware`) so the two halves of the rule can
# never disagree about which upstreams are on owner hardware.
_ON_HARDWARE_BILLING_PROVIDERS: frozenset[str] = ON_HARDWARE_BILLING_PROVIDERS

# Opaque work-size aliases. `tier-medium` / `tier-flash` name a SIZE, not a
# vendor, so selecting one is not a choice to send data off-box — unlike
# `deepseek-v4-pro`, which says exactly where it goes and is left alone.
_TIER_ALIAS_PREFIX = "tier-"


def _local_safe_model(preset: dict[str, Any], stored_model: str, tenant_id: Any) -> str:
    """A non-external preset must not reach an external upstream via a TIER alias.

    ⭐ WHAT THIS COST, measured on prod 2026-08-16. The `local` preset is
    documented as "the only non-external preset (routes through inference-router
    -> pgx GPU)", yet it offers `tier-medium` / `tier-flash`, which the live
    routing configmap owns solely on the `deepseek` provider. The KAG tenant's
    rows were ``provider=local, model=tier-medium`` for tier:medium and tier:high;
    karma's extraction phase is tier `high`. Over 6h: **1,078 requests auto-pinned
    to https://api.deepseek.com/v1, 0 to pgx-local**, while pgx idled at
    ``num_requests_running=0``. 322 returned an empty body (a reasoning model
    spends the whole max_tokens budget on reasoning and returns ``content: ""``
    with ``finish_reason: "length"``), which pydantic-ai raises as "Model token
    limit exceeded before any response was generated" — the error behind KAG's
    89% job-failure rate. The same schema validates 480/524 with ZERO validation
    failures on pgx-local, in 1-10s.

    Two harms, so the rule is not merely a performance fix: tenant content left
    the cluster for 100% of KAG relationship extraction while the tenant's own
    configuration said `local`.

    THE RULE (George, 2026-08-16): *if the tenant is on local-llm, use the local
    LLM and ignore tiers.* Scoped deliberately:

    - **Only tier aliases.** ``tier-*`` names a work SIZE, not a vendor; a tenant
      picking one has not consented to egress. ``deepseek-v4-pro`` names its
      destination and is left untouched — coercing it would silently downgrade a
      deliberate choice.
    - **Only non-external presets.** A tenant on Anthropic/DeepSeek/OpenAI keeps
      exactly what it configured. "Keeps everything else honest and untouched."
    - **Only tenant rows.** The seeded platform workers
      (``scripts/seed_provisioning_workers.py``, now pinned to ``local-llm``)
      never come through here — they carry no ``tenant_llm_config`` row — so
      this does not re-route them.

    ⚠️ Coerce, never raise: this is the agent hot path, and a hard failure would
    turn a routing misconfiguration into an outage. The drift is loud (WARNING
    naming both models) rather than fatal. Enforcing at the point of USE rather
    than in the catalog is deliberate — the offending rows are already written in
    production, and no stored state can route around a check that happens here.
    """
    if preset.get("is_external", True):
        return stored_model
    if not stored_model.startswith(_TIER_ALIAS_PREFIX):
        return stored_model
    preset_key = preset.get("key") or ""
    # Empty fallback: an alias with NO override is unknown, not known-local, and
    # must not be coerced on a guess — `test_every_local_model_has_a_billing_provider`
    # keeps the catalog's own aliases out of that blind spot.
    upstream = billing_provider_for(preset_key, stored_model, "")
    if not upstream or upstream in _ON_HARDWARE_BILLING_PROVIDERS:
        return stored_model
    local_model = preset.get("model_classes", {}).get("flash") or get_settings().local_llm_model
    logger.warning(
        "byo_llm_route tenant=%s: non-external preset %r stored tier alias model=%r, "
        "which the router serves on %r (OFF owner hardware); coercing to %r. A tenant "
        "on the local preset must not egress via a work-size alias.",
        tenant_id,
        preset_key,
        stored_model,
        upstream,
        local_model,
    )
    return local_model


def _unlock_tier_row_or_raise(
    tenant_cfg: dict[str, Any],
    preset: dict[str, Any],
    tenant_id: Optional[str],
    *,
    tier: Optional[str],
    role: str,
    entitled: bool,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Unlock the tenant tier row's key, or raise :class:`TenantLLMKeyUnusableError`.

    ⛔ Owner ruling 2026-10-01: a tenant row whose key is withheld or unusable is
    a CLEAR ERROR naming the tenant's provider and the fix. It used to WARN and
    fall through to platform routing, which silently ran the tenant on owner
    hardware / the platform model instead of the provider it configured.
    Returns ``(row_or_None, key_or_None)``; ``None`` row only for a keyless
    platform (``local``) row on a refused endpoint, treated as absent exactly as
    before (nothing of the tenant's is dropped and the result IS platform
    routing).
    """
    has_key = bool(tenant_cfg.get("api_key_encrypted"))
    unlocked, usable_key = _unlock_row_key(
        tenant_cfg,
        tenant_id,
        requires_api_key=bool(preset["requires_api_key"]),
        key_entitled=entitled,
    )

    def _err(reason: str, detail: str = "") -> TenantLLMKeyUnusableError:
        return TenantLLMKeyUnusableError(
            tenant_id=tenant_id,
            provider=preset["provider"],
            tier=tier,
            role=role,
            reason=reason,
            withheld_detail=detail,
        )

    if unlocked is None:
        if has_key and not entitled:
            raise _err("withheld", key_withheld_reason(tenant_id))
        raise _err("decrypt_failed" if has_key else "missing_key")
    # A stored base_url must be allowlisted before this service sends a key
    # there; see _allowlisted_row.
    allowed = _allowlisted_row(unlocked, tenant_id)
    if allowed is None and (preset["is_external"] or has_key):
        raise _err("base_url_not_allowlisted")
    return allowed, usable_key if allowed is not None else None


async def resolve_llm_config(
    config: dict[str, Any],
    tenant_id: Optional[str],
    *,
    byo_mode: str = "local_allowed",
    work_tier: Optional[str] = None,
    key_entitled: Optional[bool] = None,
) -> dict[str, Any]:
    """Return ``config`` with the tenant's BYO LLM provider overlaid if applicable.

    ``work_tier`` (small/medium/high) selects the tenant's per-tier row
    (role ``tier:<work_tier>``), falling back to the ``__default__`` row when
    that tier is unconfigured. ``work_tier`` is read from ``config['work_tier']``
    when not passed explicitly. Absent/unknown ⇒ ``__default__`` only (Phase-1
    behavior).

    ``key_entitled`` says whether a stored ``api_key_encrypted`` may be
    decrypted for this tenant (worker_management#245). ``None`` derives it from
    the request context via :func:`tenant_key_entitled`; the worker cache passes
    the value it already folded into its cache key so the two cannot disagree.
    A keyed row for a caller that is not entitled is treated as unusable — the
    key is never decrypted — and resolution falls through exactly as for a
    tenant with no row (system routing, or ``BYORequiredError`` when
    ``byo_required``). A key-less row (e.g. ``local``) is unaffected.

    Mutates and returns the same dict (the caller passes a fresh per-request
    config dict). May raise :class:`BYORequiredError`.
    """
    if _is_genuine_override(config):
        # Exit path 1 — an admin already pinned this worker to a provider. The
        # overlay is skipped, but the cost_event still needs a *billable*
        # provider: without this, cost_provider stayed unset and the emit site
        # fell back to model_provider. That is fine for a real external provider
        # but writes a routing name (no provider_pricing row -> permanently NULL
        # cost_usd) whenever the override is key-only. So only stamp it when the
        # pinned provider is genuinely non-system-routed.
        #
        # `_is_genuine_override` also returns True on an api_key_override alone,
        # where model_provider may still be a routing name — hence the second
        # check. No `billing_provider_for` lookup here: that table is keyed on
        # *preset key* ("local"), not provider name, so a pinned external
        # provider IS its own billing provider.
        pinned = config.get("model_provider")
        if pinned and pinned not in SYSTEM_ROUTED:
            config["cost_provider"] = pinned
        return config

    if work_tier is None:
        work_tier = config.get("work_tier")
    role = tier_role(work_tier)

    tenant_cfg = None
    if tenant_id:
        tenant_cfg = await _db.get_config(tenant_id, role)
        # A requested tier with no configured row falls back to the default row.
        if tenant_cfg is None and role != DEFAULT_ROLE:
            tenant_cfg = await _db.get_config(tenant_id, DEFAULT_ROLE)

    usable_key: Optional[str] = None
    preset: Optional[dict[str, Any]] = None
    tier_label = work_tier if role != DEFAULT_ROLE else "default"
    if tenant_cfg:
        row_role = tenant_cfg.get("role") or role
        preset = get_preset(tenant_cfg["provider"])
        if preset is None:
            if tenant_cfg.get("api_key_encrypted"):
                # A keyed row the catalog cannot route is the tenant's own key
                # going nowhere: say so, never fall back to platform routing.
                raise TenantLLMKeyUnusableError(
                    tenant_id=tenant_id,
                    provider=str(tenant_cfg["provider"]),
                    tier=tier_label,
                    role=row_role,
                    reason="unknown_provider",
                )
            logger.warning(
                "tenant_llm_config for %s names unknown provider %s; ignoring",
                tenant_id,
                tenant_cfg["provider"],
            )
            tenant_cfg = None
        elif byo_mode == "platform_only" and preset["is_external"]:
            # platform_only: BYO (external) is disabled BY POLICY, not broken.
            # Drop the stored external row (its key is never touched) so the
            # worker stays on platform inference; a `local` row falls through
            # and is honored as a platform-model selection.
            logger.info(
                "byo_llm_route tenant=%s platform_only: ignoring external row provider=%s "
                "-> system routing",
                tenant_id,
                preset["provider"],
            )
            tenant_cfg = None
        else:
            # N4: resolve the row's endpoint off the event loop before the sync check.
            await prewarm_owner_infra_dns(tenant_cfg.get("base_url"))
            tenant_cfg, usable_key = _unlock_tier_row_or_raise(
                tenant_cfg,
                preset,
                tenant_id,
                tier=tier_label,
                role=row_role,
                entitled=(tenant_key_entitled(tenant_id) if key_entitled is None else key_entitled),
            )

    if tenant_cfg and preset is not None:
        config["model_provider"] = preset["provider"]
        config["model"] = _local_safe_model(preset, tenant_cfg["model"], tenant_id)
        config["base_url_override"] = tenant_cfg.get("base_url")
        config["api_key_override"] = usable_key
        # ⛔ The ONLY writer of this flag on the worker path. It tells ModelFactory
        # the key is THIS tenant's own (or managed for it), which is the one
        # ground on which a non-platform tenant may reach an external provider,
        # and it forbids the owner env-key fallback. Never set without a
        # decrypted tenant key in hand — and never for a NON-external preset
        # (`local` → inference-router): a key on that row is not a vendor key,
        # the call stays on owner hardware behind IR, and setting the flag there
        # made enforce_customer_routing skip its local-llm coercion, so a keyed
        # `local` row naming e.g. grok-4-fast reached IR as the customer with a
        # paid model (validator 2026-10-01, D1).
        config["tenant_byo_key"] = bool(usable_key) and bool(preset["is_external"])
        if config.get("temperature") is None and tenant_cfg.get("temperature") is not None:
            config["temperature"] = tenant_cfg["temperature"]
        if config.get("max_tokens") is None and tenant_cfg.get("max_tokens") is not None:
            config["max_tokens"] = tenant_cfg["max_tokens"]
        # ⚠️ Only a call that PHYSICALLY leaves the platform may carry a
        # cost_source, because cost_source is the sole trigger for
        # `emit_byo_cost_event_task` (pydantic_ai_agent.py). The `local` preset's
        # provider IS inference-router, which already meters every call it
        # serves — stamping cost_source there made worker_management emit a
        # SECOND cost_event for the same LLM call, and flagged it byo=true even
        # though the tokens burned on owner hardware (or on the platform's own
        # DeepSeek key). Measured 2026-08-26: 14,833 of 14,840 correlation-
        # bearing worker-management rows in billing_cost_events had an exact
        # inference-router twin.
        #
        # `preset["is_external"]` is the catalog's own declared, tested
        # ("test_local_is_only_non_external") answer to "does this run off owner
        # hardware", and it is what the modality sibling already keys its `byo`
        # flag on (_modality_router.py). A router-routed row keeps its
        # cost_provider (used for attribution/logging) but gets NO cost_source,
        # so the emit gate stays shut and inference-router remains the single
        # meter of record.
        if preset["is_external"]:
            config["cost_source"] = tenant_cfg.get("cost_source") or BYO_COST_SOURCE
        else:
            logger.info(
                "byo_cost_event_suppressed_router_routed tenant=%s preset=%s "
                "provider=%s model=%s — inference-router meters this call",
                tenant_id,
                tenant_cfg["provider"],
                preset["provider"],
                config["model"],
            )
        # Billing provider for the emitted cost_event. The `local` preset routes
        # through inference_router (model_provider), but the served model is
        # billed under its real upstream provider (pgx-local / deepseek) — which
        # is how metering's provider_pricing is keyed. Emitting model_provider
        # ("inference_router") here was a pricing_miss → $0 for every BYO-local
        # run. External presets have no billing_providers map, so cost_provider
        # falls back to the provider (e.g. "anthropic"), unchanged.
        #
        # ⚠️ Priced on the EFFECTIVE model (``config["model"]``), never on the
        # stored one. When ``_local_safe_model`` coerces a tier alias, the run
        # moves to pgx-local while the row still says ``tier-medium`` — billing
        # off the row would invoice owner-hardware inference at DeepSeek's rate.
        # The coercion and the price must read the same value or the fix above
        # creates a billing defect of its own.
        config["cost_provider"] = billing_provider_for(
            tenant_cfg["provider"], config["model"], preset["provider"]
        )
        # Managed BYO: the platform owner supplied this key, so the emitted
        # cost_event is a normal marked-up line (owner_provided, no byo flag)
        # rather than a force-zero'd byo line. Carried onto the agent config in
        # _globals alongside cost_source/cost_provider.
        config["owner_provided"] = bool(tenant_cfg.get("owner_provided"))
        # Per-call tenant-key audit: fingerprint of the stored key VERSION
        # (never the key) + the row role, stamped on every cost event.
        if config["tenant_byo_key"]:
            from src.telemetry.cost_events import key_fingerprint

            config["key_fingerprint"] = key_fingerprint(tenant_cfg.get("api_key_encrypted"))
            config["byo_role"] = str(tenant_cfg.get("role") or role)
        logger.info(
            "byo_llm_route tenant=%s tier=%s role=%s provider=%s model=%s "
            "cost_source=%s cost_provider=%s",
            tenant_id,
            work_tier or "default",
            tenant_cfg.get("role", role),
            preset["provider"],
            tenant_cfg["model"],
            # None ⇒ router-routed: deliberately unarmed, no second meter.
            config.get("cost_source"),
            config["cost_provider"],
        )
        _apply_tier_alias_passthrough(config, tenant_id, work_tier)
        return config

    # No usable tenant config.
    # System/internal tenants (zero-UUID sentinels + WM_SYSTEM_TENANT_IDS) are
    # exempt: a transient/misclassified byo_required must never 402 them — they
    # fall through to seeded system routing (exit path 2 below).
    if byo_mode == "byo_required" and str(tenant_id) not in _SYSTEM_TENANTS:
        raise BYORequiredError(
            "This tenant must configure its own LLM provider before running agents."
        )

    # Exit path 2 — the worker stays on its seeded system routing. There is no
    # tenant row to read a provider from, so the only signal available here is
    # the seeded model name, which for platform routing is a `local`-preset
    # alias. Map it when the billing table knows it; when it does not (an
    # unrecognised or dynamically-resolved alias such as `tier-medium` on a
    # worker with no tenant row) we deliberately leave cost_provider UNSET
    # rather than guessing — a wrong provider prices the run at someone else's
    # rate. The emit site then prefers inference-router's `served_provider` and,
    # if even that is missing, drops the event loudly via ROUTING_PROVIDERS.
    if _apply_tier_alias_passthrough(config, tenant_id, work_tier):
        return config
    seeded_model = config.get("model")
    if seeded_model and not config.get("cost_provider"):
        derived = billing_provider_for("local", seeded_model, "")
        if derived:
            config["cost_provider"] = derived
    return config


async def resolve_tenant_byo_providers(
    tenant_id: Optional[str],
    *,
    byo_mode: str = "local_allowed",
    key_entitled: Optional[bool] = None,
) -> dict[str, dict[str, Any]]:
    """The external providers THIS tenant can reach with ITS OWN key, by provider.

    Owner ruling 2026-10-01: a customer tenant may use a commercial provider only
    with a key from its own ``tenant_llm_config`` row — its own BYO, or managed
    BYO (``owner_provided``) the owner set for that one tenant. This is the
    resolver behind ``GET /providers``, ``GET /providers/{name}/models`` and
    ``POST /chat/completions``; the worker path uses :func:`resolve_llm_config`.

    Returns ``{model_provider: {...}}`` where each value carries ``api_key``
    (plaintext, never logged or returned to a caller), ``base_url``,
    ``preset_key``, ``model`` (the row's configured model), ``cost_source``,
    ``owner_provided`` and ``role``. A provider appears only when a row:

    - belongs to ``tenant_id`` (rows are read by tenant, so tenant A's managed
      key can never surface for tenant B),
    - names a known, EXTERNAL preset (the ``local`` preset is inference-router
      and needs no BYO),
    - has a key that decrypts AND the caller is entitled to decrypt it
      (:func:`tenant_key_entitled`, worker_management#245 — licence-proven
      tenant only),
    - has an allowlisted ``base_url`` (or none), and
    - is not disabled by ``byo_mode == 'platform_only'``.

    A keyless external row (the ``custom`` preset) is NOT usable here: with no
    tenant key the only credential left would be the owner's, which is exactly
    what the ruling forbids. Rows are visited ``__default__`` first, so when two
    roles name the same provider the default row's key wins.
    """
    status = await tenant_byo_status(tenant_id, byo_mode=byo_mode, key_entitled=key_entitled)
    return {name: st["entry"] for name, st in status.items() if st["status"] == "usable"}


async def tenant_byo_status(
    tenant_id: Optional[str],
    *,
    byo_mode: str = "local_allowed",
    key_entitled: Optional[bool] = None,
    include_local: bool = False,
) -> dict[str, dict[str, Any]]:
    """Per provider the tenant has a row for: usable, withheld or unusable.

    External presets always; with ``include_local`` also the non-external
    (``local`` → ``inference_router``) rows, so a caller can list/test EVERY
    configured row (a future local row whose key is a licence token is handled
    the same way: a stored key is decrypted only for an entitled caller).

    ``{provider: {"status": "usable"|"withheld"|"unusable", "reason": str,
    "error": TenantLLMKeyUnusableError|None, "entry": {...}|None}}``. ``entry``
    (with the plaintext ``api_key``) is present only for ``usable``. Same row
    rules as :func:`resolve_tenant_byo_providers`, which is this filtered to
    ``usable``; the extra states let a route answer "you have a key but it was
    withheld / is broken" instead of "add a key".
    """
    if not tenant_id:
        return {}
    entitled = tenant_key_entitled(tenant_id) if key_entitled is None else key_entitled
    rows = await _db.list_configs(tenant_id)
    # N4: the per-row endpoint checks below resolve DNS; do it off the loop first.
    await prewarm_owner_infra_dns(*(dict(r).get("base_url") for r in rows or []))
    out: dict[str, dict[str, Any]] = {}
    for raw in rows or []:
        row = dict(raw)
        if str(row.get("tenant_id", tenant_id)) != str(tenant_id):
            # Defence in depth: the query is tenant-scoped; never trust a row
            # that says otherwise.
            continue
        preset = get_preset(row.get("provider") or "")
        if preset is None or not (preset["is_external"] or include_local):
            continue
        if byo_mode == "platform_only" and preset["is_external"]:
            continue
        provider = preset["provider"]
        if out.get(provider, {}).get("status") == "usable":
            continue
        cand = evaluate_tenant_row(tenant_id, row, preset, entitled=entitled)
        # Rows are visited __default__ first; the first row for a provider wins
        # unless a later one is usable and the earlier one is not.
        if provider not in out or cand["status"] == "usable":
            out[provider] = cand
    return out


def _key_fp(api_key_encrypted: object) -> str:
    from src.telemetry.cost_events import key_fingerprint

    return key_fingerprint(api_key_encrypted)


def evaluate_tenant_row(
    tenant_id: Optional[str],
    row: dict[str, Any],
    preset: dict[str, Any],
    *,
    entitled: bool,
) -> dict[str, Any]:
    """Evaluate ONE tenant_llm_config row: usable / withheld / unusable.

    Shared by :func:`tenant_byo_status` and ``POST /tenant-config/llm/test`` so
    the listing, the chat gate and the key test can never disagree about a row.
    The stored key is decrypted only when ``entitled``.
    """
    provider = preset["provider"]
    role = row.get("role") or DEFAULT_ROLE
    tier = role[5:] if role.startswith("tier:") else "default"
    has_key = bool(row.get("api_key_encrypted"))

    def _bad(reason: str) -> dict[str, Any]:
        err = TenantLLMKeyUnusableError(
            tenant_id=tenant_id,
            provider=provider,
            tier=tier,
            role=role,
            reason=reason,
            withheld_detail=key_withheld_reason(tenant_id) if reason == "withheld" else "",
        )
        return {
            "status": "withheld" if reason == "withheld" else "unusable",
            "reason": reason,
            "error": err,
            "entry": None,
        }

    needs_key = bool(preset["is_external"] or preset["requires_api_key"])
    if not has_key and needs_key:
        # A keyless external row (``custom``) would need the owner's key.
        return _bad("missing_key")
    if has_key and not entitled:
        return _bad("withheld")
    unlocked, key = _unlock_row_key(row, tenant_id, requires_api_key=needs_key, key_entitled=True)
    if unlocked is None or (has_key and not key):
        return _bad("decrypt_failed")
    allowed = _allowlisted_row(unlocked, tenant_id)
    if allowed is None:
        return _bad("base_url_not_allowlisted")
    return {
        "status": "usable",
        "reason": "ok",
        "error": None,
        "entry": {
            "api_key": key,
            "base_url": allowed.get("base_url"),
            "preset_key": preset["key"],
            "is_external": bool(preset["is_external"]),
            "model": allowed.get("model"),
            # Only a call that leaves the platform is BYO-metered here;
            # inference-router meters local rows itself.
            "cost_source": (
                (allowed.get("cost_source") or BYO_COST_SOURCE) if preset["is_external"] else None
            ),
            "owner_provided": bool(allowed.get("owner_provided")),
            "role": allowed.get("role") or role,
            # Audit fingerprint of the stored key version (never the key).
            "key_fingerprint": _key_fp(row.get("api_key_encrypted")) if key else "",
        },
    }


def _pin_override_to_inference_router(config: dict[str, Any], tenant_id: Optional[str]) -> None:
    """A customer's IR ``base_url_override`` may only name THE inference-router.

    Validator 3 N1: ``model_override`` writes the MLX instance URL here, which
    would call owner hardware directly, past inference-router's per-tenant gate
    and meter — so any override on another host is DROPPED (the IR creator then
    falls back to ``INFERENCE_ROUTER_URL``). An override on the inference-router
    origin is KEPT, normalised to its ``/api`` base: that is the URL the 187 live
    customer rows already send, and dropping it made a worker unbuildable
    wherever ``INFERENCE_ROUTER_URL`` is unset (the pg CI job, 2026-10-01).
    """
    override = config.get("base_url_override")
    if not override:
        return
    ir_base = inference_router_base_for(str(override))
    if ir_base is None:
        logger.warning(
            "customer_routing tenant=%s: dropping non-inference-router base_url_override",
            tenant_id,
        )
    config["base_url_override"] = ir_base


# Plan §2.4 (WM_TIER_ALIAS_PASSTHROUGH): the work-size aliases inference-router
# resolves against a tenant's policy. ``tier-flash`` is IR's alias of ``tier-small``.
_TIER_PASSTHROUGH_RE = re.compile(r"^tier-(small|medium|high|flash)$")
_TIER_PASSTHROUGH_DEFAULT = "default"


def is_tier_passthrough_alias(model: Any) -> bool:
    """True for ``tier-small|medium|high|flash`` or ``default``."""
    return isinstance(model, str) and (
        model == _TIER_PASSTHROUGH_DEFAULT or _TIER_PASSTHROUGH_RE.match(model) is not None
    )


def _apply_tier_alias_passthrough(
    config: dict[str, Any], tenant_id: Optional[str], work_tier: Optional[str]
) -> bool:
    """With the flag on, map ``local-llm``/``default`` + ``work_tier`` to ``tier-<work_tier>``.

    Only for a CUSTOMER on ``inference_router``: platform tiers target
    ``local-llm`` only (ruling 1.0.7) — a platform ``tier-medium`` auto-pins to
    the owner's paid deepseek in IR's routing configmap. ``cost_provider`` is
    dropped rather than guessed: the provider that serves a tier alias is
    decided by the tenant's IR policy, so attribution comes from IR's
    ``served_provider`` (``billing_provider_for('local', 'tier-medium')`` would
    otherwise answer ``deepseek`` for a call IR may serve on pgx-local).
    Returns True when it rewrote the model.
    """
    if not get_settings().wm_tier_alias_passthrough:
        return False
    if is_commercial_platform_tenant(tenant_id):
        return False
    if config.get("model_provider") != "inference_router":
        return False
    model = config.get("model")
    if model not in (get_settings().local_llm_model, _TIER_PASSTHROUGH_DEFAULT):
        return False
    if not isinstance(work_tier, str) or work_tier not in WORK_TIERS:
        return False
    config["model"] = f"tier-{work_tier}"
    config.pop("cost_provider", None)
    logger.info(
        "tier_alias_passthrough tenant=%s work_tier=%s model %r -> %r",
        tenant_id,
        work_tier,
        model,
        config["model"],
    )
    return True


def _strip_commercial_reach_for_unproven_platform(
    config: dict[str, Any], tenant_id: Optional[str], refusal: str
) -> dict[str, Any]:
    """A platform-set tenant this request did NOT prove: owner hardware, no owner keys.

    ⛔ Membership is not proof (owner ruling 2026-10-03, fr#1385): a tokenless
    caller can assert a platform tenant, and an absent header defaults to one.
    Such a tenant keeps the platform's LOCAL paths untouched (``vllm`` / ``mlx``
    / ``vlm_local`` direct, ``inference_router`` ``local-llm``), but loses every
    route to the owner's commercial upstreams behind inference-router: a
    stored ``ir_provider`` pin is dropped unless it names an on-hardware
    upstream (:func:`is_on_hardware_ir_pin`, e.g. ``pgx-local``), and any other IR model (``deepseek-…``,
    ``grok-…``, ``default``, ``tier-*`` — a platform tier pins to the owner's
    paid deepseek in IR's routing configmap) is coerced to ``local-llm``. An
    EXTERNAL provider on the owner's key is not rewritten here: ModelFactory's
    ``enforce_local_only_provider`` refuses it at build with the same predicate.
    """
    pin = config.get("ir_provider")
    # A pin to an upstream on owner HARDWARE (pgx-local) reaches no commercial
    # key and is kept; any other / unknown pin is dropped (fail safe).
    if pin and not is_on_hardware_ir_pin(pin):
        logger.warning(
            "unproven_platform_routing tenant=%s reason=%s: dropping stored ir_provider pin %r",
            tenant_id,
            refusal,
            pin,
        )
        config["ir_provider"] = None
    if (config.get("agent_type") or "pydantic_ai") != "pydantic_ai":
        return config
    local_model = get_settings().local_llm_model
    if config.get("model_provider") == "inference_router" and config.get("model") != local_model:
        logger.warning(
            "unproven_platform_routing tenant=%s reason=%s: inference_router model %r -> %r",
            tenant_id,
            refusal,
            config.get("model"),
            local_model,
        )
        config["model"] = local_model
    return config


def enforce_customer_routing(config: dict[str, Any], tenant_id: Optional[str]) -> dict[str, Any]:
    """Worker-path routing rules for a NON-platform tenant (owner ruling 2026-10-01).

    Applied after :func:`resolve_llm_config` on every worker build. A platform
    tenant that THIS request PROVED (``owner_commercial_keys_refusal``) is
    returned untouched; a platform tenant that is merely asserted / defaulted
    keeps its local paths but loses the IR pin and every non-``local-llm`` IR
    model (:func:`_strip_commercial_reach_for_unproven_platform`). For a
    customer that is NOT running on its own
    resolved BYO key (``tenant_byo_key``):

    - a stored ``ir_provider`` pin is DROPPED — pinning inference-router to an
      upstream is how a commercial provider behind it would be reached on the
      owner's key (validator B2; the inbound ``X-Provider`` was already dropped);
    - ``inference_router`` with any model other than ``local-llm`` is coerced to
      ``local-llm`` (``deepseek-v4-pro``, ``tier-*``, a routing-rule schema id,
      or a ``local``-preset row naming a paid model; validator B2);
    - a local DIRECT provider (``vllm`` / ``mlx`` / ``vlm_local``, or no provider
      at all, which the agent factory defaults to ``vllm``) is moved to
      ``inference_router`` / ``local-llm`` — the one customer path to owner
      hardware, behind inference-router's own per-tenant gate and metering
      (validator N6; chat and /providers already refuse these for customers).

    Coerce, never raise: this is the agent hot path and every rewrite keeps the
    customer on the model it is entitled to. Each rewrite is logged at WARNING.
    A customer on its own BYO key still loses the IR pin (it is not going
    through IR at all).
    """
    if is_commercial_platform_tenant(tenant_id):
        refusal = owner_commercial_keys_refusal(tenant_id)
        if refusal is None:
            return config
        return _strip_commercial_reach_for_unproven_platform(config, tenant_id, refusal)
    pin = config.get("ir_provider")
    if pin:
        logger.warning(
            "customer_routing tenant=%s: dropping stored ir_provider pin %r", tenant_id, pin
        )
        config["ir_provider"] = None
    provider = config.get("model_provider")
    # Validator 3 D3 (belt): an EXTERNAL provider pointed at owner infrastructure
    # (the inference-router Service, any in-cluster, private-DNS or tailnet
    # host) would reach it with no identity attached and be booked as
    # the tenant's own BYO spend. Refused outright — no fallback to platform
    # routing (owner ruling 2026-10-01). The tenant allowlist already refuses
    # such a row at PUT and at resolve; this catches every other source of
    # ``base_url_override`` (a stored worker config, a future path).
    external_override = config.get("base_url_override")
    if (
        external_override
        and provider
        and provider != "inference_router"
        and provider not in _LOCAL_DIRECT_PROVIDERS
        and is_owner_infra_url(str(external_override))
    ):
        logger.warning(
            "customer_routing tenant=%s: refusing external provider %r on a platform-"
            "infrastructure base_url",
            tenant_id,
            provider,
        )
        work_tier = config.get("work_tier")
        raise TenantLLMKeyUnusableError(
            tenant_id=tenant_id,
            provider=str(provider),
            tier=work_tier if isinstance(work_tier, str) else None,
            role=tier_role(work_tier if isinstance(work_tier, str) else None),
            reason="base_url_owner_infra",
        )
    # A customer on its OWN external key is not going through owner hardware or
    # inference-router at all, so the coercion below does not apply. The early
    # return is NEVER taken for inference_router or a local-direct provider:
    # whatever key rides along, those calls land on owner hardware/IR as the
    # customer, so the local-llm rule must still run (validator D1).
    if (
        config.get("tenant_byo_key")
        and config.get("api_key_override")
        and provider
        and provider != "inference_router"
        and provider not in _LOCAL_DIRECT_PROVIDERS
    ):
        return config
    if (config.get("agent_type") or "pydantic_ai") != "pydantic_ai":
        # CLI agents (claude_code / codex / remote_*) do not dispatch through
        # ModelFactory; their `model` is not an LLM-router model name.
        return config
    local_model = get_settings().local_llm_model
    if provider in _LOCAL_DIRECT_PROVIDERS or not provider:
        logger.warning(
            "customer_routing tenant=%s: local direct provider %r -> inference_router/%s",
            tenant_id,
            provider,
            local_model,
        )
        config["model_provider"] = "inference_router"
        config["model"] = local_model
        _pin_override_to_inference_router(config, tenant_id)
        return config
    if provider == "inference_router":
        _pin_override_to_inference_router(config, tenant_id)
        if get_settings().wm_tier_alias_passthrough and is_tier_passthrough_alias(
            config.get("model")
        ):
            # Plan §2.4: IR resolves the alias against the tenant's own policy
            # (no policy ⇒ IR default-deny serves pgx-local). Concrete models
            # are still coerced below; the local-direct branch above is unchanged.
            return config
        if config.get("model") != local_model:
            logger.warning(
                "customer_routing tenant=%s: inference_router model %r -> %r",
                tenant_id,
                config.get("model"),
                local_model,
            )
            config["model"] = local_model
    return config


# Owner-hardware providers reached DIRECTLY (bypassing inference-router).
_LOCAL_DIRECT_PROVIDERS: frozenset[str] = frozenset({"vllm", "mlx", "vlm_local"})
