"""Unit tests for the BYO LLM resolution seam (precedence + fail-closed)."""

from __future__ import annotations

import secrets

import pytest

from src.api.identity_context import TENANT_PROVENANCE_LICENSE, set_identity_context
from src.security.secret_box import encrypt_secret
from src.worker.factory import llm_config_resolver as r
from src.worker.factory.llm_config_resolver import (
    BYORequiredError,
    TenantLLMKeyUnusableError,
    resolve_llm_config,
)

pytestmark = pytest.mark.asyncio

TENANT = "019edc1e-0000-7000-8000-000000000001"

# Aliases owned SOLELY by the external `deepseek` provider in
# gitops clusters/strato/services/inference-router/providers-configmap.yaml.
# inference-router's _auto_pin_by_alias sends any of these straight to
# https://api.deepseek.com/v1, bypassing default_chain [pgx-local, deepseek].
# Opaque work-SIZE aliases that the live routing configmap owns solely on the
# external `deepseek` provider. inference-router's _auto_pin_by_alias sends any of
# these straight to https://api.deepseek.com/v1. `deepseek-v4-*` is deliberately
# NOT here: it names its destination, so selecting it is a consented choice.
OFFBOX_TIER_ALIASES = frozenset({"tier-medium", "tier-flash"})


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))


@pytest.fixture(autouse=True)
def _proven_caller():
    """These precedence tests model a caller that PROVED ``TENANT`` with its own
    licence — the only caller entitled to a decrypted key (worker_management#245).
    The unproven/adopted/cross-tenant cases live in
    ``tests/test_byo_key_requires_proven_tenant.py``.
    """
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    yield
    set_identity_context(None, None, None)


class _FakeDB:
    def __init__(self, row):
        self._row = row

    async def get_config(self, tenant_id, role="__default__"):
        return self._row


def _install_db(monkeypatch, row):
    monkeypatch.setattr(r, "_db", _FakeDB(row))


def _row(**over):
    base = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "api_key_encrypted": encrypt_secret("sk-tenant-key"),
        "cost_source": "worker-management",
    }
    base.update(over)
    return base


async def test_tenant_overlay_applied_to_system_routed(monkeypatch):
    _install_db(monkeypatch, _row())
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT)
    assert out["model_provider"] == "anthropic"
    assert out["model"] == "claude-sonnet-4-6"
    assert out["api_key_override"] == "sk-tenant-key"
    assert out["cost_source"] == "worker-management"
    # External preset: cost_provider falls back to the provider (matches pricing).
    assert out["cost_provider"] == "anthropic"


async def test_genuine_external_override_untouched(monkeypatch):
    _install_db(monkeypatch, _row())
    cfg = {"model_provider": "openai", "model": "gpt-4o"}  # already external
    out = await resolve_llm_config(cfg, TENANT)
    assert out["model_provider"] == "openai"
    assert "api_key_override" not in out


async def test_explicit_worker_key_untouched(monkeypatch):
    _install_db(monkeypatch, _row())
    cfg = {"model_provider": "inference_router", "api_key_override": "worker-pinned"}
    out = await resolve_llm_config(cfg, TENANT)
    assert out["api_key_override"] == "worker-pinned"
    assert out["model_provider"] == "inference_router"


async def test_no_tenant_config_local_allowed_passthrough(monkeypatch):
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert out["model_provider"] == "inference_router"


async def test_no_tenant_config_byo_required_hard_fails(monkeypatch):
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(BYORequiredError):
        await resolve_llm_config(cfg, TENANT, byo_mode="byo_required")


# --- system/internal tenant byo_required exemption (never 402) ------------
# Platform-owned runs (zero-UUID sentinels + WM_SYSTEM_TENANT_IDS) must route to
# seeded system config even under a transient/misclassified byo_required.

SYSTEM_TENANT = "00000000-0000-0000-0000-000000000001"


async def test_system_tenant_byo_required_does_not_hard_fail(monkeypatch):
    # A reserved system tenant with no usable tenant row + byo_required must NOT
    # raise; it falls through to seeded system routing.
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, SYSTEM_TENANT, byo_mode="byo_required")
    assert out["model_provider"] == "inference_router"  # seeded routing kept


async def test_ordinary_tenant_byo_required_still_hard_fails(monkeypatch):
    # A non-system tenant is unaffected: byo_required still hard-fails.
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(BYORequiredError):
        await resolve_llm_config(cfg, TENANT, byo_mode="byo_required")


# ⛔ REWRITTEN 2026-09-06, not deleted. Both tests below used to assert the
# OPPOSITE: that the nil uuid is an exempt system tenant. They passed, and what
# they pinned was the defect — the estate's anonymous sentinel holding the one
# privilege this module grants (skip the byo_required hard-fail, fall through to
# seeded platform routing, i.e. spend owner GPU that no customer can be charged
# for). The nil uuid is not a customer and must never be privileged; ...0001 is a
# real reserved platform tenant and keeps its exemption.
NIL_TENANT = "00000000-0000-0000-0000-000000000000"


async def test_the_nil_uuid_is_NOT_exempt_and_still_hard_fails(monkeypatch):
    """The nil uuid must behave like any other unentitled tenant, not like a
    platform one. Previously it was exempt and returned seeded routing."""
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(BYORequiredError):
        await resolve_llm_config(cfg, NIL_TENANT, byo_mode="byo_required")


async def test_the_nil_uuid_is_not_in_the_system_tenant_set():
    assert NIL_TENANT not in r._SYSTEM_TENANTS


async def test_the_real_reserved_sentinel_is_still_exempt():
    """CONTROL: the removal is targeted at the nil uuid, not at the exemption."""
    assert SYSTEM_TENANT in r._SYSTEM_TENANTS


async def test_platform_only_ignores_external_row(monkeypatch):
    # A platform_only tenant with a stored EXTERNAL provider row must NOT use it —
    # the row is ignored and the worker stays on platform (inference_router).
    _install_db(monkeypatch, _row(provider="anthropic", model="claude-sonnet-4-6"))
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="platform_only")
    assert out["model_provider"] == "inference_router"
    assert out["model"] == "default"
    assert "api_key_override" not in out


async def test_platform_only_honors_local_row(monkeypatch):
    # A `local` (platform) row only selects a platform model — it is honored.
    _install_db(monkeypatch, _row(provider="local", model="default", api_key_encrypted=None))
    cfg = {"model_provider": "inference_router", "model": "seeded"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="platform_only")
    assert out["model_provider"] == "inference_router"
    assert out["model"] == "default"
    # cost_provider is the real pricing provider, NOT the inference_router router
    # name (which has no provider_pricing row → pricing_miss → $0).
    assert out["cost_provider"] == "pgx-local"


@pytest.mark.parametrize(
    "model,expected_cost_provider",
    [
        ("local-llm", "pgx-local"),
        ("default", "pgx-local"),
        ("deepseek-v4-flash", "deepseek"),
        ("deepseek-v4-pro", "deepseek"),
    ],
)
async def test_local_preset_cost_provider_mapping(monkeypatch, model, expected_cost_provider):
    # The `local` preset routes through inference_router but each served model is
    # billed under its real upstream provider so metering can price it.
    _install_db(monkeypatch, _row(provider="local", model=model, api_key_encrypted=None))
    cfg = {"model_provider": "inference_router", "model": "seeded"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert out["model_provider"] == "inference_router"  # routing unchanged
    assert out["model"] == model
    assert out["cost_provider"] == expected_cost_provider


# --- cost_provider on the exit paths that previously left it unset (#137) ---
# cost_provider used to be set ONLY inside the tenant-row branch, so both other
# exits emitted cost events under model_provider — a routing name with no
# provider_pricing row => cost_usd NULL forever.


async def test_genuine_override_gets_cost_provider(monkeypatch):
    # Exit path 1: an admin pinned this worker to a real external provider.
    _install_db(monkeypatch, _row())
    cfg = {"model_provider": "anthropic", "model": "claude-opus-4-8"}
    out = await resolve_llm_config(cfg, TENANT)
    assert out["model_provider"] == "anthropic"  # overlay skipped
    assert out["cost_provider"] == "anthropic"


async def test_key_only_override_leaves_cost_provider_unset(monkeypatch):
    # A key-only override has no billable provider to name — model_provider is
    # still the routing name, so we must NOT stamp it as a cost_provider.
    _install_db(monkeypatch, _row())
    cfg = {
        "model_provider": "inference_router",
        "model": "tier-medium",
        "api_key_override": "sk-worker-pinned",
    }
    out = await resolve_llm_config(cfg, TENANT)
    assert out.get("cost_provider") is None


async def test_no_tenant_row_derives_cost_provider_from_seeded_model(monkeypatch):
    # Exit path 2: no usable tenant row, worker stays on system routing. The
    # seeded model is a known `local` alias, so it maps to a billable upstream.
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "local-llm"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert out["model_provider"] == "inference_router"  # routing unchanged
    assert out["cost_provider"] == "pgx-local"


async def test_no_tenant_row_unknown_model_leaves_cost_provider_unset(monkeypatch):
    # Unknown alias: guessing a provider would bill the run at someone else's
    # rate, so leave it unset and let the emit site's ROUTING_PROVIDERS guard
    # drop the event loudly.
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "some-unknown-alias"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert out.get("cost_provider") is None


async def test_platform_only_no_row_passthrough(monkeypatch):
    # No usable row + platform_only never hard-fails (only byo_required does).
    _install_db(monkeypatch, None)
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="platform_only")
    assert out["model_provider"] == "inference_router"


async def test_decrypt_failure_fails_closed(monkeypatch):
    # Stored ciphertext that won't decrypt under the current key. Owner ruling
    # 2026-10-01: a clear 409, never a silent fall-back to platform routing.
    _install_db(monkeypatch, _row(api_key_encrypted="not-a-valid-blob"))
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(TenantLLMKeyUnusableError) as exc_info:
        await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert exc_info.value.reason == "decrypt_failed"
    assert exc_info.value.status_code == 409
    assert cfg["model_provider"] == "inference_router"  # nothing overlaid
    assert "api_key_override" not in cfg


async def test_decrypt_failure_byo_required_hard_fails(monkeypatch):
    _install_db(monkeypatch, _row(api_key_encrypted="not-a-valid-blob"))
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(TenantLLMKeyUnusableError):
        await resolve_llm_config(cfg, TENANT, byo_mode="byo_required")


async def test_unknown_provider_with_a_key_is_a_clear_error(monkeypatch):
    _install_db(monkeypatch, _row(provider="totally-made-up"))
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(TenantLLMKeyUnusableError) as exc_info:
        await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert exc_info.value.reason == "unknown_provider"


async def test_unknown_provider_without_a_key_is_ignored(monkeypatch):
    _install_db(monkeypatch, _row(provider="totally-made-up", api_key_encrypted=None))
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, byo_mode="local_allowed")
    assert out["model_provider"] == "inference_router"


async def test_temperature_only_filled_when_worker_unset(monkeypatch):
    _install_db(monkeypatch, _row(temperature=0.9))
    # Worker already set temperature -> tenant value must NOT override.
    cfg = {"model_provider": "mlx", "model": "x", "temperature": 0.1}
    out = await resolve_llm_config(cfg, TENANT)
    assert out["temperature"] == 0.1
    # Worker unset -> tenant value fills in.
    cfg2 = {"model_provider": "mlx", "model": "x"}
    out2 = await resolve_llm_config(cfg2, TENANT)
    assert out2["temperature"] == 0.9


async def test_no_tenant_id_passthrough(monkeypatch):
    _install_db(monkeypatch, _row())
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, None)
    assert out["model_provider"] == "inference_router"


# --- work_tier resolution --------------------------------------------------


class _RoleDB:
    """Fake DB returning a distinct row per role (tier rows + default)."""

    def __init__(self, rows_by_role):
        self._rows = rows_by_role

    async def get_config(self, tenant_id, role="__default__"):
        return self._rows.get(role)


def _install_role_db(monkeypatch, rows_by_role):
    monkeypatch.setattr(r, "_db", _RoleDB(rows_by_role))


async def test_tier_role_mapping():
    assert r.tier_role("small") == "tier:small"
    assert r.tier_role("high") == "tier:high"
    assert r.tier_role(None) == "__default__"
    assert r.tier_role("nonsense") == "__default__"


async def test_tier_row_selected(monkeypatch):
    _install_role_db(
        monkeypatch,
        {
            "__default__": _row(model="claude-sonnet-4-6"),
            "tier:high": _row(model="claude-opus-4-8"),
        },
    )
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, work_tier="high")
    assert out["model"] == "claude-opus-4-8"


async def test_tier_missing_falls_back_to_default(monkeypatch):
    # high tier unconfigured -> use the tenant __default__ row.
    _install_role_db(monkeypatch, {"__default__": _row(model="claude-sonnet-4-6")})
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, work_tier="high")
    assert out["model"] == "claude-sonnet-4-6"


async def test_work_tier_read_from_config(monkeypatch):
    _install_role_db(
        monkeypatch,
        {
            "__default__": _row(model="claude-sonnet-4-6"),
            "tier:small": _row(model="claude-haiku-4-5"),
        },
    )
    # No explicit work_tier arg: resolver reads it off the config dict.
    cfg = {"model_provider": "inference_router", "model": "default", "work_tier": "small"}
    out = await resolve_llm_config(cfg, TENANT)
    assert out["model"] == "claude-haiku-4-5"


async def test_unknown_tier_uses_default(monkeypatch):
    _install_role_db(
        monkeypatch,
        {
            "__default__": _row(model="claude-sonnet-4-6"),
            "tier:high": _row(model="claude-opus-4-8"),
        },
    )
    cfg = {"model_provider": "inference_router", "model": "default"}
    out = await resolve_llm_config(cfg, TENANT, work_tier="gigantic")
    assert out["model"] == "claude-sonnet-4-6"  # unknown tier -> default row


async def test_tier_with_no_rows_at_all_byo_required_hard_fails(monkeypatch):
    _install_role_db(monkeypatch, {})  # neither tier nor default configured
    cfg = {"model_provider": "inference_router", "model": "default"}
    with pytest.raises(BYORequiredError):
        await resolve_llm_config(cfg, TENANT, byo_mode="byo_required", work_tier="high")


# ---------------------------------------------------------------------------
# ⭐ A NON-EXTERNAL PRESET MUST NEVER RESOLVE TO AN OFF-HARDWARE MODEL.
# ---------------------------------------------------------------------------


class TestLocalPresetStaysLocal:
    """``local`` means owner hardware. A work-tier alias must not override that.

    ⭐ WHAT THIS COST, measured on prod 2026-08-16. The catalog docstring says
    ``local`` "is the only non-external preset (routes through inference-router ->
    pgx GPU)". Its curated ``models`` list nevertheless offered ``tier-flash``,
    ``tier-medium``, ``deepseek-v4-flash`` and ``deepseek-v4-pro`` — every one of
    which is owned SOLELY by the external ``deepseek`` provider, so
    inference-router's ``_auto_pin_by_alias`` sends them straight to
    ``https://api.deepseek.com/v1`` over the public internet.

    The KAG tenant's stored rows were ``provider=local`` with
    ``model=tier-medium`` for ``tier:medium`` and ``tier:high``. karma's extraction
    phase is tier ``high``. Result over 6 h: **1,078 requests auto-pinned to
    DeepSeek, 0 to pgx-local**, while pgx sat at ``num_requests_running=0``.
    322 of those returned an empty body (``deepseek-v4-pro`` is a reasoning model:
    it spends the whole ``max_tokens`` budget on reasoning and returns
    ``content: ""`` with ``finish_reason: "length"``), which pydantic-ai raises as
    "Model token limit (6000) exceeded before any response was generated" — the
    error that drove KAG's 89% job-failure rate for weeks.

    Two independent harms, so two independent reasons this invariant exists:
      1. correctness — the workload validates 480/524 with ZERO validation
         failures on pgx-local, in 1-10 s, versus a 30% empty-response rate at
         ~60 s on DeepSeek;
      2. **egress** — tenant content left the cluster for 100% of KAG relationship
         extraction, on a deployment being positioned for HIPAA / research use,
         while the tenant's own configuration said ``local``.

    The rule (George, 2026-08-16): *if the tenant is on local-llm, use the local
    LLM and ignore tiers.* Tiers keep working for genuinely external presets.
    """

    async def test_local_row_naming_an_external_alias_is_coerced_to_local(self, monkeypatch):
        """⭐ THE REGRESSION. ``provider=local`` + ``model=tier-medium`` must not leave pgx."""
        _install_db(
            monkeypatch, _row(provider="local", model="tier-medium", api_key_encrypted=None)
        )
        cfg = await resolve_llm_config({"model_provider": "vllm"}, TENANT, work_tier="high")

        assert cfg["model"] not in OFFBOX_TIER_ALIASES, (
            f"a local-preset tenant resolved to {cfg['model']!r}, which inference-router "
            "auto-pins to api.deepseek.com — tenant content leaves the cluster"
        )
        assert cfg["model"] == "local-llm"
        assert cfg["model_provider"] == "inference_router"

    async def test_local_row_already_naming_local_llm_is_untouched(self, monkeypatch):
        """Negative control: the coercion must not be a blanket rewrite."""
        _install_db(monkeypatch, _row(provider="local", model="local-llm", api_key_encrypted=None))
        cfg = await resolve_llm_config({"model_provider": "vllm"}, TENANT, work_tier="small")

        assert cfg["model"] == "local-llm"

    async def test_external_preset_keeps_its_tier_model(self, monkeypatch):
        """⭐ "keeps everything else honest and untouched".

        A tenant that genuinely selected DeepSeek/Anthropic still gets exactly the
        model it configured. Without this, the fix would silently downgrade every
        paying BYO tenant to Gemma — strictly worse than the bug.
        """
        _install_db(monkeypatch, _row(provider="deepseek", model="deepseek-v4-pro"))
        cfg = await resolve_llm_config({"model_provider": "vllm"}, TENANT, work_tier="high")

        assert cfg["model"] == "deepseek-v4-pro"
        assert cfg["model_provider"] == "deepseek"

    async def test_explicitly_named_deepseek_model_is_left_alone(self, monkeypatch):
        """⭐ SCOPE GUARD. `deepseek-v4-pro` names where it goes; a tier alias does not.

        Coercing this too would silently downgrade a deliberate selection to Gemma —
        the mirror-image defect of the one being fixed.
        """
        _install_db(
            monkeypatch, _row(provider="local", model="deepseek-v4-pro", api_key_encrypted=None)
        )
        cfg = await resolve_llm_config({"model_provider": "vllm"}, TENANT, work_tier="high")

        assert cfg["model"] == "deepseek-v4-pro"

    async def test_coerced_run_is_billed_as_pgx_local_not_deepseek(self, monkeypatch):
        """⭐ The coercion must move the PRICE with the traffic.

        `cost_provider` was derived from the STORED model, so a row saying
        `tier-medium` kept billing as `deepseek` after the run had been moved to
        pgx-local — invoicing owner-hardware inference at an external vendor's rate.
        Found while re-reading my own change; the coercion and the price have to read
        the same value.
        """
        _install_db(
            monkeypatch, _row(provider="local", model="tier-medium", api_key_encrypted=None)
        )
        cfg = await resolve_llm_config({"model_provider": "vllm"}, TENANT, work_tier="high")

        assert cfg["model"] == "local-llm"
        assert cfg["cost_provider"] == "pgx-local", (
            f"run moved to pgx-local but bills as {cfg['cost_provider']!r}"
        )

    async def test_every_local_model_has_a_billing_provider(self):
        """Closes the coercion's blind spot before it can open.

        The rule only fires when `billing_provider_for` knows the alias's real
        upstream; an alias with no entry is treated as unknown and left alone. That
        is the safe default for the router's long tail of per-agent schema ids, but
        it would silently exempt a NEW catalog alias. Every model the catalog itself
        offers under `local` must therefore be classified.
        """
        from src.worker.agents import provider_catalog as pc

        local = pc.get_preset("local")
        assert local is not None
        unclassified = [m for m in local["models"] if not pc.billing_provider_for("local", m, "")]
        assert not unclassified, (
            f"local-preset models with no billing provider: {unclassified} — a tier alias "
            "among these would bypass the local-stays-local coercion (and bill $0)"
        )


# ---------------------------------------------------------------------------
# Tier-alias passthrough (plan §2.4, WM_TIER_ALIAS_PASSTHROUGH, default off)
# ---------------------------------------------------------------------------

_PASSTHROUGH_ALIASES = ("tier-small", "tier-medium", "tier-high", "tier-flash", "default")


def _ir_cfg(model: str, **over):
    cfg = {"model_provider": "inference_router", "model": model, "agent_type": "pydantic_ai"}
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("alias", _PASSTHROUGH_ALIASES)
async def test_enforce_passthrough_on_keeps_tier_alias(monkeypatch, alias):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    out = r.enforce_customer_routing(_ir_cfg(alias), TENANT)
    assert out["model_provider"] == "inference_router"
    assert out["model"] == alias


@pytest.mark.parametrize("model", ["deepseek-v4-pro", "tier-huge", "grok-4.5", "tier-medium-x"])
async def test_enforce_passthrough_on_still_coerces_non_alias(monkeypatch, model):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    out = r.enforce_customer_routing(_ir_cfg(model), TENANT)
    assert out["model"] == "local-llm"


@pytest.mark.parametrize("alias", _PASSTHROUGH_ALIASES)
async def test_enforce_flag_off_is_today(monkeypatch, alias):
    monkeypatch.delenv("WM_TIER_ALIAS_PASSTHROUGH", raising=False)
    out = r.enforce_customer_routing(_ir_cfg(alias), TENANT)
    assert out["model"] == "local-llm"


@pytest.mark.parametrize("provider", ["vllm", "mlx", "vlm_local", None])
async def test_enforce_passthrough_on_local_direct_still_coerced(monkeypatch, provider):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    cfg = _ir_cfg("tier-medium", model_provider=provider)
    out = r.enforce_customer_routing(cfg, TENANT)
    assert out["model_provider"] == "inference_router"
    assert out["model"] == "local-llm"


@pytest.mark.parametrize("model", ["local-llm", "default"])
@pytest.mark.parametrize("tier", ["small", "medium", "high"])
async def test_resolve_passthrough_on_maps_work_tier(monkeypatch, model, tier):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg(model, work_tier=tier), TENANT)
    assert out["model"] == f"tier-{tier}"
    # Not guessed: IR's served_provider is the billing provider for a tier alias.
    assert out.get("cost_provider") in (None, "")


@pytest.mark.parametrize("model", ["local-llm", "default"])
async def test_resolve_flag_off_is_today(monkeypatch, model):
    monkeypatch.delenv("WM_TIER_ALIAS_PASSTHROUGH", raising=False)
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg(model, work_tier="medium"), TENANT)
    assert out["model"] == model


async def test_resolve_passthrough_on_without_work_tier_unchanged(monkeypatch):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg("local-llm"), TENANT)
    assert out["model"] == "local-llm"


async def test_resolve_passthrough_on_concrete_model_unchanged(monkeypatch):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg("deepseek-v4-pro", work_tier="high"), TENANT)
    assert out["model"] == "deepseek-v4-pro"


async def test_resolve_passthrough_on_platform_tenant_unchanged(monkeypatch):
    """Platform tiers target local-llm only (ruling 1.0.7): a platform tenant's
    ``tier-medium`` would auto-pin to the owner's paid deepseek in IR."""
    from src.api.tenant_constants import default_tenant_uuid

    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg("local-llm", work_tier="medium"), default_tenant_uuid())
    assert out["model"] == "local-llm"


async def test_resolve_passthrough_on_local_row_maps_work_tier(monkeypatch):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, _row(provider="local", model="default", api_key_encrypted=None))
    out = await resolve_llm_config(_ir_cfg("default", work_tier="high"), TENANT)
    assert out["model_provider"] == "inference_router"
    assert out["model"] == "tier-high"
    assert out.get("cost_provider") in (None, "")
    assert "cost_source" not in out


async def test_resolve_then_enforce_passthrough_end_to_end(monkeypatch):
    monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
    _install_db(monkeypatch, None)
    out = await resolve_llm_config(_ir_cfg("local-llm", work_tier="small"), TENANT)
    out = r.enforce_customer_routing(out, TENANT)
    assert out["model"] == "tier-small"
