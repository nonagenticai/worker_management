"""Unit tests for the BYO provider catalog."""

from __future__ import annotations

import pytest

from src.worker.agents import provider_catalog as pc


def test_catalog_keys_present():
    keys = {p["key"] for p in pc.provider_catalog()}
    assert {
        "local",
        "anthropic",
        "openai",
        "google-gemini",
        "deepseek",
        "groq",
        "cohere",
        "xai",
        "openai_compatible",
        "custom",
    } == keys


def test_every_preset_maps_to_known_modelfactory_provider():
    import typing

    from src.worker.agents.pydantic_ai_agent import ModelProvider

    valid = set(typing.get_args(ModelProvider))
    for p in pc.provider_catalog():
        assert p["provider"] in valid, p


def test_local_is_only_non_external():
    externals = {p["key"] for p in pc.provider_catalog() if p["is_external"]}
    assert "local" not in externals
    assert "anthropic" in externals and "openai_compatible" in externals


def test_compatible_requires_base_url():
    compat = pc.get_preset("openai_compatible")
    anthropic = pc.get_preset("anthropic")
    assert compat is not None and anthropic is not None
    assert compat["requires_base_url"] is True
    assert anthropic["requires_base_url"] is False


def test_get_preset_unknown_returns_none():
    assert pc.get_preset("nope") is None


def test_is_external_helper():
    assert pc.is_external("openai") is True
    assert pc.is_external("local") is False
    assert pc.is_external("nope") is False


def test_every_preset_has_model_classes():
    for p in pc.provider_catalog():
        assert "model_classes" in p, p["key"]
        mc = p["model_classes"]
        # Curated providers map all three classes to a model in their own list;
        # free-text providers leave it empty for the admin to fill.
        if p["models"]:
            assert set(mc) == {"flash", "reasoning", "frontier"}, p["key"]
            assert all(v in p["models"] for v in mc.values()), p["key"]
        else:
            assert mc == {}, p["key"]


def test_deepseek_models_are_the_real_upstream_ids():
    """Guard the catalog against re-drifting off what DeepSeek actually serves.

    Regression: the preset used to list `deepseek-chat` / `deepseek-reasoner`, which
    the API no longer serves. Nothing validated it — `PUT /tenant-config/llm` stores
    `model` free-form — so the fr dropdown offered two models that 400 at run time
    while the stored tenant row looked valid. Verified 2026-07-19 against
    GET https://api.deepseek.com/v1/models.
    """
    preset = pc.get_preset("deepseek")
    assert preset["models"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert preset["model_classes"] == {
        "flash": "deepseek-v4-flash",
        "reasoning": "deepseek-v4-pro",
        "frontier": "deepseek-v4-pro",
    }


def test_local_model_classes_all_name_the_one_platform_model():
    """Platform inference is ONE model (`local-llm`); every tier class maps to it.

    The class KEYS stay (the fr ProviderTierCard reads model_classes[classKey]); only
    the values collapsed. `reasoning`/`frontier` used to be `tier-medium`, a placeholder
    that billed deepseek and was refused on PUT under this very preset.
    """
    mc = pc.get_preset("local")["model_classes"]
    assert set(mc) == {"flash", "reasoning", "frontier"}
    assert set(mc.values()) == {"local-llm"}


def test_every_local_model_class_has_a_billing_provider():
    """Each class-backing model needs a _BILLING_PROVIDER_OVERRIDES row, else the emitted
    cost_event bills under "inference_router", which provider_pricing has no row for →
    pricing_miss → $0. Guards the whole class of "the tier looks free" bugs."""
    for model in pc.get_preset("local")["model_classes"].values():
        billed = pc.billing_provider_for("local", model, "inference_router")
        assert billed != "inference_router", f"{model} has no billing override"


def test_local_preset_offers_only_what_its_own_put_accepts():
    """Every model the `local` preset offers must survive the write-side guard.

    Before 2026-09-24 it offered tier-flash, tier-medium, deepseek-v4-flash and
    deepseek-v4-pro -- all billed `deepseek`, so PUT /tenant-config/llm refused each
    one with a 422: a dropdown of options that could not be saved.
    """
    local = pc.get_preset("local")
    assert local["models"] == ["local-llm"]
    for model in [*local["models"], *local["model_classes"].values()]:
        pc.assert_local_preset_stays_on_hardware("local", model)


def test_anthropic_model_classes_ordered_by_capability():
    mc = pc.get_preset("anthropic")["model_classes"]
    assert mc == {
        "flash": "claude-haiku-4-5",
        "reasoning": "claude-sonnet-4-6",
        "frontier": "claude-opus-4-8",
    }


def test_filter_for_mode_byo_required_hides_local():
    keys = {p["key"] for p in pc.filter_for_mode("byo_required")}
    assert "local" not in keys
    assert "anthropic" in keys and "openai_compatible" in keys


def test_filter_for_mode_platform_only_is_local_only():
    presets = pc.filter_for_mode("platform_only")
    assert [p["key"] for p in presets] == ["local"]


def test_filter_for_mode_local_allowed_returns_all():
    assert len(pc.filter_for_mode("local_allowed")) == len(pc.provider_catalog())


def test_filter_for_mode_unknown_returns_all():
    # Defensive: an unexpected byo_mode must not leak to deny-all/allow-none.
    assert len(pc.filter_for_mode("bogus")) == len(pc.provider_catalog())


def test_billing_provider_for_local_preset_maps_to_real_upstream():
    assert pc.billing_provider_for("local", "local-llm", "inference_router") == "pgx-local"
    assert pc.billing_provider_for("local", "default", "inference_router") == "pgx-local"
    assert pc.billing_provider_for("local", "deepseek-v4-pro", "inference_router") == "deepseek"
    assert pc.billing_provider_for("local", "deepseek-v4-flash", "inference_router") == "deepseek"


def test_billing_provider_for_unmapped_falls_back():
    # External presets / unknown models bill under the fallback (their provider).
    assert pc.billing_provider_for("anthropic", "claude-sonnet-4-6", "anthropic") == "anthropic"
    assert (
        pc.billing_provider_for("local", "some-future-model", "inference_router")
        == "inference_router"
    )


def test_billing_map_not_leaked_into_catalog():
    # The billing override is module-level, never in the tenant-facing catalog.
    for p in pc.provider_catalog():
        assert "billing_providers" not in p


def test_tier_flash_bills_under_deepseek():
    # A stored tier-flash row (the provisioning seeds used to pin it; they now pin
    # local-llm) maps to deepseek/deepseek-v4-flash whenever deepseek is enabled. Without the override it falls back to the routing
    # name "inference_router", which has no provider_pricing row → $0 forever.
    assert pc.billing_provider_for("local", "tier-flash", "inference_router") == "deepseek"


def test_every_seeded_worker_model_is_in_its_preset():
    # A seeded worker asking for a model the preset doesn't list is a silent drift
    # between scripts/ and the catalog; tier-flash was exactly that.
    import importlib.util
    from pathlib import Path

    seed_path = Path(__file__).resolve().parents[1] / "scripts" / "seed_provisioning_workers.py"
    spec = importlib.util.spec_from_file_location("_seed_provisioning_workers", seed_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    local_models = set(pc.get_preset("local")["models"])
    for worker in module.PROVISIONING_WORKERS:
        if worker.get("model_provider") != "inference_router":
            continue
        assert worker["model"] in local_models, (
            f"{worker['worker_name']} requests {worker['model']!r}, "
            "which is not in the local preset's models"
        )


# ── assert_local_preset_stays_on_hardware ────────────────────────────────────
# The write-side half of the rule whose read-side half is _local_safe_model.
# Two tenant_llm_config rows reading provider='local', model='tier-medium' spent
# $140.31 over 46,102 requests on paid DeepSeek for a tenant whose own config
# said "platform inference". migrations/025 removes those rows; this guard is
# what stops them being written again — deleting fixes today, refusing fixes the
# class.


def test_local_preset_with_tier_medium_is_refused():
    # THE EXACT SHAPE THAT COST $140.31. If this test ever goes green by being
    # deleted rather than by the guard, the trap is re-armed.
    with pytest.raises(pc.PaidUpstreamForLocalPresetError) as exc:
        pc.assert_local_preset_stays_on_hardware("local", "tier-medium")
    message = str(exc.value)
    # The message must NAME the upstream — a refusal that does not say who gets
    # paid leaves the operator guessing which provider to select instead.
    assert "deepseek" in message
    assert "tier-medium" in message


def test_local_preset_with_tier_flash_is_refused():
    # The flash-class twin. Same alias family, same paid upstream: the guard
    # must not be tier-medium-specific, or the next row is tier-flash.
    with pytest.raises(pc.PaidUpstreamForLocalPresetError):
        pc.assert_local_preset_stays_on_hardware("local", "tier-flash")


def test_local_preset_with_a_named_paid_model_is_also_refused():
    # WIDER THAN _local_safe_model ON PURPOSE. The read-side rule fires only on
    # `tier-*`, because coercing a stored `deepseek-v4-pro` would silently
    # downgrade a live run. At WRITE time there is no live run to protect and a
    # human is choosing, and "platform inference" paired with a paid API IS the
    # spend trap. The honest alternative exists and is one dropdown away: the
    # `deepseek` preset, which requires the tenant's own key.
    for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
        with pytest.raises(pc.PaidUpstreamForLocalPresetError):
            pc.assert_local_preset_stays_on_hardware("local", model)


def test_a_legitimate_local_row_still_persists():
    # THE OTHER HALF OF THE GATE. A guard that refuses everything is not a
    # guard, it is an outage — these are the models the local preset exists to
    # serve, and every one must still be storable.
    for model in ("local-llm", "default"):
        pc.assert_local_preset_stays_on_hardware("local", model)


def test_an_external_preset_keeps_exactly_what_it_configured():
    # A tenant that SELECTED DeepSeek/Anthropic is paying deliberately, with its
    # own key. The guard is about the mismatch between what the row claims and
    # where it bills — not about paid models per se.
    pc.assert_local_preset_stays_on_hardware("deepseek", "deepseek-v4-pro")
    pc.assert_local_preset_stays_on_hardware("anthropic", "claude-opus-4-8")
    pc.assert_local_preset_stays_on_hardware("xai", "grok-4.5")


def test_an_unknown_preset_is_not_the_guards_business():
    # get_preset() returns None and the router already 422s on it. The guard
    # must not raise a DIFFERENT error for the same input, or the API's
    # "Unknown provider" message becomes unreachable.
    pc.assert_local_preset_stays_on_hardware("no-such-preset", "tier-medium")


def test_an_unmapped_local_model_is_allowed_and_this_is_the_known_gap():
    # DOCUMENTS A DELIBERATE LIMITATION so nobody "fixes" it into an outage. A
    # model with no _BILLING_PROVIDER_OVERRIDES entry is treated as on-hardware.
    # Failing closed here would reject every legitimate new local model the
    # moment the router gained one. The gap is not silent: an unmapped alias
    # also prices at $0 in metering, which is how the existing entries came to
    # be added.
    assert pc.billing_provider_for("local", "some-future-local-model", "") == ""
    pc.assert_local_preset_stays_on_hardware("local", "some-future-local-model")


def test_the_two_halves_of_the_rule_share_one_source_of_truth():
    # _local_safe_model (read side) and this guard (write side) must never
    # disagree about which upstreams are on owner hardware. They are the same
    # frozenset object, not two copies that can drift.
    from src.worker.factory import llm_config_resolver as lcr

    assert lcr._ON_HARDWARE_BILLING_PROVIDERS is pc.ON_HARDWARE_BILLING_PROVIDERS
    assert "pgx-local" in pc.ON_HARDWARE_BILLING_PROVIDERS
    assert "deepseek" not in pc.ON_HARDWARE_BILLING_PROVIDERS
