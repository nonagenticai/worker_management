"""BYO modality provider catalog — presets for tts / asr / image / embedding / vlm.

The non-LLM modality backends (tts_backend, asr_backend, image_gen_backend,
mlx_backend) each run a self-hosted Apple-silicon model today. This catalog adds
the per-tenant BYO choice: keep the platform ``local`` preset, or bring an
external provider key. It mirrors :mod:`src.worker.agents.provider_catalog`
(same preset keys + gating semantics) but is indexed by ``modality`` and drops
the LLM-only tier ``model_classes``.

``is_external`` marks providers that run OFF owner hardware — the set a
``byo_required`` tenant may select. Each modality has exactly one non-external
``local`` preset (the platform self-hosted default) which a ``platform_only``
tenant is limited to.

``provider`` doubles as the billing provider (metering ``provider_pricing`` key)
AND the adapter-selection key the modality backend switches on — so for the
tts/asr/image/embedding ``local`` presets it is the concrete self-hosted service
name (kokoro / mlx-whisper / mflux / mlx-embed), not a router alias. The ``vlm``
``local`` preset is the exception: platform vision is the multimodal Gemma served
via the ``inference_router`` (``local-llm``), so its ``provider`` is the router and
it bills under the real upstream ``pgx-local`` via ``_BILLING_PROVIDER_OVERRIDES``.
"""

from __future__ import annotations

from typing import Any

from src.settings import LOCAL_LLM_MODEL

MODALITIES: frozenset[str] = frozenset({"tts", "asr", "image", "embedding", "vlm"})

# modality -> ordered list of presets (UI display order).
_CATALOG: dict[str, list[dict[str, Any]]] = {
    "tts": [
        {
            "key": "local",
            "label": "Platform (kokoro / kitten)",
            "provider": "kokoro",
            "default_base_url": None,
            "requires_api_key": False,
            "requires_base_url": False,
            "is_external": False,
            "models": ["kokoro", "kitten"],
            "required_params": [],
        },
        {
            "key": "openai",
            "label": "OpenAI TTS",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
            "required_params": [],
        },
        {
            "key": "elevenlabs",
            "label": "ElevenLabs",
            "provider": "elevenlabs",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5"],
            "required_params": [],
        },
        {
            # Self-hosted / local endpoint modelled as a BYO target: the owner's
            # own GPU service (e.g. tts.example.com) or a rent-GPU box. is_external=True so
            # a byo_required tenant may select it; the gateway proxies to the
            # configured base_url instead of its hardcoded discovery fallback.
            "key": "self_hosted",
            "label": "Self-hosted / local endpoint (owner GPU or rent-GPU)",
            "provider": "self_hosted",
            "default_base_url": "http://tts.example.com:9202",
            "requires_api_key": False,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
    ],
    "asr": [
        {
            "key": "local",
            "label": "Platform (mlx-whisper)",
            "provider": "mlx-whisper",
            "default_base_url": None,
            "requires_api_key": False,
            "requires_base_url": False,
            "is_external": False,
            "models": ["whisper-large-v3", "whisper-medium"],
            "required_params": [],
        },
        {
            "key": "openai",
            "label": "OpenAI Whisper",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
            "required_params": [],
        },
        {
            "key": "deepgram",
            "label": "Deepgram",
            "provider": "deepgram",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["nova-3", "nova-2"],
            "required_params": [],
        },
        {
            # Self-hosted / local ASR endpoint modelled as a BYO target (owner GPU
            # asr.example.com or a rent-GPU box). See the tts self_hosted preset.
            "key": "self_hosted",
            "label": "Self-hosted / local endpoint (owner GPU or rent-GPU)",
            "provider": "self_hosted",
            "default_base_url": "http://asr.example.com:9201",
            "requires_api_key": False,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
    ],
    "image": [
        {
            "key": "local",
            "label": "Platform (mflux)",
            "provider": "mflux",
            "default_base_url": None,
            "requires_api_key": False,
            "requires_base_url": False,
            "is_external": False,
            "models": ["schnell", "dev"],
            "required_params": [],
        },
        {
            "key": "openai",
            "label": "OpenAI Images (DALL·E / gpt-image)",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["gpt-image-1", "dall-e-3"],
            "required_params": [],
        },
        {
            # Self-hosted / local image endpoint modelled as a BYO target (owner
            # GPU image-gen.example.com or a rent-GPU box). See the tts self_hosted preset.
            "key": "self_hosted",
            "label": "Self-hosted / local endpoint (owner GPU or rent-GPU)",
            "provider": "self_hosted",
            "default_base_url": "http://image-gen.example.com:9204",
            "requires_api_key": False,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
    ],
    "embedding": [
        {
            "key": "local",
            "label": "Platform (mlx embeddings)",
            "provider": "mlx-embed",
            "default_base_url": None,
            "requires_api_key": False,
            "requires_base_url": False,
            "is_external": False,
            "models": ["bge-large", "nomic-embed-text"],
            "required_params": [],
        },
        {
            "key": "openai",
            "label": "OpenAI Embeddings",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["text-embedding-3-small", "text-embedding-3-large"],
            "required_params": [],
        },
        {
            "key": "openai_compatible",
            "label": "OpenAI-compatible embeddings (vLLM / Together / …)",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
        {
            # Self-hosted / local embedding endpoint modelled as a BYO target (owner
            # GPU mlx-ingestion-backend.example.com or a rent-GPU box). See tts self_hosted.
            "key": "self_hosted",
            "label": "Self-hosted / local endpoint (owner GPU or rent-GPU)",
            "provider": "self_hosted",
            "default_base_url": "http://mlx-ingestion-backend.example.com:9120",
            "requires_api_key": False,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
    ],
    "vlm": [
        {
            # Platform vision = the multimodal Gemma (`local-llm`) on the pgx vLLM,
            # served via the inference-router — the SAME model the platform LLM
            # uses. A tenant's own LLM may not be multimodal, so vlm stays a
            # distinct modality; only this platform preset shares the LLM's Gemma.
            # The modality-gateway forces model=local-llm and proxies to IR (it
            # bills under the real upstream pgx-local, see _BILLING_PROVIDER_OVERRIDES).
            # The dormant mlx-vlm (Qwen2-VL) on mini2 remains available via the
            # `self_hosted` preset below (default_base_url http://vlm.example.com:9203).
            "key": "local",
            "label": "Platform (local vision)",
            "provider": "inference_router",
            "default_base_url": None,
            "requires_api_key": False,
            "requires_base_url": False,
            "is_external": False,
            "models": [LOCAL_LLM_MODEL],
            "required_params": [],
        },
        {
            "key": "openai",
            "label": "OpenAI Vision (gpt-4o)",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["gpt-4o", "gpt-4o-mini"],
            "required_params": [],
        },
        {
            # Anthropic Claude vision via the Messages API. The gateway VLM route
            # translates the OpenAI-vision request ⇄ Anthropic Messages, so the
            # tenant speaks OpenAI while billing under provider "anthropic".
            "key": "anthropic",
            "label": "Anthropic Claude Vision (Messages API)",
            "provider": "anthropic",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": False,
            "is_external": True,
            "models": ["claude-opus-4-8", "claude-sonnet-5"],
            "required_params": [],
        },
        {
            "key": "openai_compatible",
            "label": "OpenAI-compatible vision (vLLM / OpenRouter / …)",
            "provider": "openai",
            "default_base_url": None,
            "requires_api_key": True,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
        {
            # Self-hosted / local VLM endpoint modelled as a BYO target (owner GPU
            # vlm.example.com or a rent-GPU box). See the tts self_hosted preset.
            "key": "self_hosted",
            "label": "Self-hosted / local endpoint (owner GPU or rent-GPU)",
            "provider": "self_hosted",
            "default_base_url": "http://vlm.example.com:9203",
            "requires_api_key": False,
            "requires_base_url": True,
            "is_external": True,
            "models": [],  # free-text
            "required_params": [],
        },
    ],
}

_BY_KEY: dict[str, dict[str, dict[str, Any]]] = {
    modality: {p["key"]: p for p in presets} for modality, presets in _CATALOG.items()
}


def modality_catalog(modality: str) -> list[dict[str, Any]]:
    """Return the full catalog for a modality (unfiltered), or [] if unknown."""
    return [dict(p) for p in _CATALOG.get(modality, [])]


def filter_for_mode(modality: str, byo_mode: str) -> list[dict[str, Any]]:
    """Return the presets a tenant in ``byo_mode`` may select for ``modality``.

    Same gating as the LLM catalog: ``byo_required`` → external only,
    ``platform_only`` → ``local`` only, else all.
    """
    presets = _CATALOG.get(modality, [])
    if byo_mode == "byo_required":
        return [dict(p) for p in presets if p["is_external"]]
    if byo_mode == "platform_only":
        return [dict(p) for p in presets if not p["is_external"]]
    return [dict(p) for p in presets]


def get_preset(modality: str, key: str) -> dict[str, Any] | None:
    """Return the preset for (modality, key), or None if unknown."""
    preset = _BY_KEY.get(modality, {}).get(key)
    return dict(preset) if preset else None


def is_external(modality: str, key: str) -> bool:
    """True if the (modality, key) preset runs off owner hardware."""
    preset = _BY_KEY.get(modality, {}).get(key)
    return bool(preset and preset["is_external"])


# Per-(modality, key, model) billing-provider overrides. The tts/asr/image/
# embedding ``local`` presets carry the real upstream directly in ``provider``
# (the concrete self-hosted service name), so they need no remap. The ``vlm``
# ``local`` preset is the exception: it routes through ``inference_router`` to the
# Gemma ``local-llm``, and metering's provider_pricing keys on the real upstream
# (``pgx-local``), not the router name — so it bills under ``pgx-local``, mirroring
# the LLM catalog's ``local`` preset (provider_catalog._BILLING_PROVIDER_OVERRIDES).
_BILLING_PROVIDER_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {
    "vlm": {"local": {LOCAL_LLM_MODEL: "pgx-local"}},
}


def billing_provider_for(modality: str, key: str, model: str, fallback: str) -> str:
    """Provider the emitted cost_event is billed under for (modality, key, model).

    Returns a per-model override when one exists, else the preset's own provider,
    else ``fallback`` (unknown preset).
    """
    override = _BILLING_PROVIDER_OVERRIDES.get(modality, {}).get(key, {}).get(model)
    if override:
        return override
    preset = _BY_KEY.get(modality, {}).get(key)
    return str(preset["provider"]) if preset else fallback
