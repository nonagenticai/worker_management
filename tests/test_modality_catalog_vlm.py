"""Platform vlm preset routes to the Gemma inference-router path, not dead mini2.

The `vlm` `local` preset used to be `mlx-vlm` / Qwen2-VL on the dormant mini2
backend. Platform vision is now the multimodal Gemma (`local-llm`) served via the
inference_router — the same model the platform LLM uses — so the modality-gateway
proxies platform vlm to IR. This pins that contract.
"""

from __future__ import annotations

from src.worker.agents import modality_catalog as mc


def test_vlm_local_preset_points_at_inference_router_gemma():
    preset = mc.get_preset("vlm", "local")
    assert preset is not None
    assert preset["provider"] == "inference_router"
    assert preset["models"] == ["local-llm"]
    assert preset["is_external"] is False  # platform default; platform_only tenants keep it
    assert preset["requires_api_key"] is False
    assert preset["requires_base_url"] is False


def test_vlm_local_bills_under_real_upstream_pgx_local():
    # Routes through inference_router, so it must bill under the real upstream
    # (pgx-local), not the router name — mirrors the LLM catalog's local preset.
    assert mc.billing_provider_for("vlm", "local", "local-llm", "inference_router") == "pgx-local"


def test_vlm_self_hosted_preset_keeps_dormant_mlx_vlm_alternate():
    # The mini2 mlx-vlm backend stays reachable as the self_hosted BYO option.
    preset = mc.get_preset("vlm", "self_hosted")
    assert preset is not None
    assert preset["default_base_url"] == "http://vlm.example.com:9203"
    assert preset["is_external"] is True


def test_vlm_platform_only_tenant_sees_the_gemma_local_preset():
    # platform_only gating keys on is_external/key, so the repointed local preset
    # is still the single platform option a platform_only tenant may select.
    presets = mc.filter_for_mode("vlm", "platform_only")
    keys = {p["key"] for p in presets}
    assert keys == {"local"}
    assert presets[0]["provider"] == "inference_router"
