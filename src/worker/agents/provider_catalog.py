"""BYO LLM provider catalog — presets shown in the tenant settings UI.

Each preset maps a customer-facing catalog id (``key``) to a concrete
:data:`~src.worker.agents.pydantic_ai_agent.ModelProvider` adapter, plus the
metadata the fr ``/settings/llm-provider`` UI needs (label, curated models,
whether a key / base_url is required). The PUT ``/tenant-config/llm`` endpoint
validates a tenant's selection against this catalog.

``is_external`` marks providers that run OFF owner hardware — the set a
``byo_required`` (commercial) tenant is allowed to select. ``local`` is the
only non-external preset (routes through inference-router -> pgx GPU); it
exists for internal tenants and as the explicit "use platform inference"
choice.
"""

from __future__ import annotations

from typing import Any

from src.settings import LOCAL_LLM_MODEL

# Catalog id -> preset. Order is the UI display order.
_CATALOG: list[dict[str, Any]] = [
    {
        "key": "local",
        "label": "Platform (local inference)",
        "provider": "inference_router",
        "default_base_url": None,
        "requires_api_key": False,
        "requires_base_url": False,
        "is_external": False,
        # ONE model: pgx serves a single vLLM model under the served-model-name
        # `local-llm` (gitops scripts/host-configs/pgx/systemd/vllm.service), and
        # every consumer names that alias. Platform inference therefore offers
        # exactly one model, and every tier class maps to it -- the class KEYS stay
        # because the fr /settings/llm-provider ProviderTierCard reads
        # model_classes[flash|reasoning|frontier].
        #
        # Removed 2026-09-24: `default`, `tier-flash`, `tier-medium`,
        # `deepseek-v4-flash`, `deepseek-v4-pro`. `default` and `tier-*` are
        # placeholders, not models. And every one of the last four bills `deepseek`
        # in _BILLING_PROVIDER_OVERRIDES, so assert_local_preset_stays_on_hardware
        # already refused it on PUT: the UI offered four options every save 422'd
        # on, and backed `reasoning`/`frontier` with one of them. Tenants that want
        # DeepSeek select the `deepseek` preset with their own key.
        # Rows ALREADY stored with those names still resolve and bill correctly via
        # _BILLING_PROVIDER_OVERRIDES / _local_safe_model; that is observed state,
        # not an offer.
        "models": [LOCAL_LLM_MODEL],
        "model_classes": {
            "flash": LOCAL_LLM_MODEL,
            "reasoning": LOCAL_LLM_MODEL,
            "frontier": LOCAL_LLM_MODEL,
        },
        "required_params": [],
    },
    {
        "key": "anthropic",
        "label": "Anthropic Claude",
        "provider": "anthropic",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": [
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-fable-5",
        ],
        "model_classes": {
            "flash": "claude-haiku-4-5",
            "reasoning": "claude-sonnet-4-6",
            "frontier": "claude-opus-4-8",
        },
        "required_params": [],
    },
    {
        "key": "openai",
        "label": "OpenAI",
        "provider": "openai",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": ["gpt-4o", "gpt-4o-mini", "o3", "o3-mini"],
        "model_classes": {"flash": "gpt-4o-mini", "reasoning": "o3-mini", "frontier": "o3"},
        "required_params": [],
    },
    {
        "key": "google-gemini",
        "label": "Google Gemini",
        "provider": "google",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "model_classes": {
            "flash": "gemini-2.5-flash",
            "reasoning": "gemini-2.5-pro",
            "frontier": "gemini-2.5-pro",
        },
        "required_params": [],
    },
    {
        "key": "deepseek",
        "label": "DeepSeek",
        "provider": "deepseek",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        # The upstream ids DeepSeek actually serves. Verified 2026-07-19 against
        # the live API with the in-cluster key: GET https://api.deepseek.com/v1/models
        # -> exactly ["deepseek-v4-flash", "deepseek-v4-pro"].
        # These were `deepseek-chat` / `deepseek-reasoner`, which DeepSeek no longer
        # serves — so the fr /settings/llm-provider dropdown offered two models that
        # every run would 400 on, and the tenant's stored row looked perfectly valid.
        # The `local` preset above already carried the correct ids (see
        # _BILLING_PROVIDER_OVERRIDES), so the catalog was self-inconsistent.
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "model_classes": {
            "flash": "deepseek-v4-flash",
            "reasoning": "deepseek-v4-pro",
            "frontier": "deepseek-v4-pro",
        },
        "required_params": [],
    },
    {
        "key": "groq",
        "label": "Groq",
        "provider": "groq",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        "model_classes": {
            "flash": "llama-3.3-70b-versatile",
            "reasoning": "openai/gpt-oss-120b",
            "frontier": "openai/gpt-oss-120b",
        },
        "required_params": [],
    },
    {
        "key": "cohere",
        "label": "Cohere",
        "provider": "cohere",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": ["command-a-03-2025", "command-r-plus"],
        "model_classes": {
            "flash": "command-r-plus",
            "reasoning": "command-a-03-2025",
            "frontier": "command-a-03-2025",
        },
        "required_params": [],
    },
    {
        "key": "xai",
        "label": "xAI Grok",
        "provider": "xai",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": False,
        "is_external": True,
        "models": [
            "grok-4.5",
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
        ],
        "model_classes": {
            "flash": "grok-4.20-0309-non-reasoning",
            "reasoning": "grok-4.20-0309-reasoning",
            "frontier": "grok-4.5",
        },
        "required_params": [],
    },
    {
        "key": "openai_compatible",
        "label": "OpenAI-compatible (Together / Fireworks / OpenRouter / vLLM)",
        "provider": "openai",
        "default_base_url": None,
        "requires_api_key": True,
        "requires_base_url": True,
        "is_external": True,
        "models": [],  # free-text
        "model_classes": {},  # free-text: admin types the model per tier
        "required_params": [],
    },
    {
        "key": "custom",
        "label": "Custom endpoint",
        "provider": "openai",
        "default_base_url": None,
        "requires_api_key": False,
        "requires_base_url": True,
        "is_external": True,
        "models": [],  # free-text
        "model_classes": {},  # free-text: admin types the model per tier
        "required_params": [],
    },
]

_BY_KEY: dict[str, dict[str, Any]] = {p["key"]: p for p in _CATALOG}


def provider_catalog() -> list[dict[str, Any]]:
    """Return the full catalog (every preset, unfiltered)."""
    return [dict(p) for p in _CATALOG]


def filter_for_mode(byo_mode: str) -> list[dict[str, Any]]:
    """Return the catalog a tenant in ``byo_mode`` may select from.

    Platform inference (``local`` → inference-router → owner GPU) is a privileged
    grant, so the catalog is gated by the tenant's licenses ``byo_mode``:

    * ``byo_required``  — commercial: external presets only (``local`` hidden).
    * ``platform_only`` — demo/test/trial: ``local`` only (BYO disabled).
    * ``local_allowed`` (and any unknown value) — every preset.
    """
    if byo_mode == "byo_required":
        return [dict(p) for p in _CATALOG if p["is_external"]]
    if byo_mode == "platform_only":
        return [dict(p) for p in _CATALOG if not p["is_external"]]
    return [dict(p) for p in _CATALOG]


def get_preset(key: str) -> dict[str, Any] | None:
    """Return the preset for ``key`` or ``None`` if unknown."""
    preset = _BY_KEY.get(key)
    return dict(preset) if preset else None


def is_external(key: str) -> bool:
    """True if the preset runs off owner hardware (selectable by commercial tenants)."""
    preset = _BY_KEY.get(key)
    return bool(preset and preset["is_external"])


# Real upstream provider each served model is BILLED under, per preset. Metering's
# provider_pricing is keyed on provider (NOT the inference_router router name), so
# the `local` preset — which routes through inference_router — must bill each model
# under its real upstream: local-llm/default -> pgx-local ($10/$25), deepseek-* ->
# deepseek list rates. Kept OUT of the preset dicts so it never leaks into the
# tenant-facing /tenant-config/llm/catalog response. External presets have no
# override and bill under their own provider. See resolve_llm_config.
_BILLING_PROVIDER_OVERRIDES: dict[str, dict[str, str]] = {
    "local": {
        LOCAL_LLM_MODEL: "pgx-local",
        "default": "pgx-local",
        # `tier-medium` auto-pins to the deepseek provider, so it must bill under
        # deepseek. Without this entry the fallback is the preset's own provider
        # ("inference_router"), which metering's provider_pricing has no row for →
        # pricing_miss → every tier-medium run bills $0. That is the exact bug the
        # deepseek-v4-* entries below were added to fix; a new alias needs the same
        # treatment or the tier looks free.
        "tier-medium": "deepseek",
        # `tier-flash` (pinned by the provisioning seeds until they moved to
        # local-llm) maps to deepseek/deepseek-v4-flash whenever deepseek is enabled. Without this entry it fell back to
        # "inference_router" — unpriceable, same $0 bug as tier-medium.
        "tier-flash": "deepseek",
        "deepseek-v4-flash": "deepseek",
        "deepseek-v4-pro": "deepseek",
    },
}


# Provider names that identify a *router*, not a billable upstream. metering's
# provider_pricing has no row for these, so a cost_event carrying one prices to
# NULL and stays unbilled forever (billing computes cost_usd exactly once, at
# flush time — there is no retro-pricing pass). They must therefore never reach
# a cost_event: the emit site re-derives the real provider from the served model
# and, failing that, drops the event loudly rather than writing a NULL row.
ROUTING_PROVIDERS: frozenset[str] = frozenset({"inference_router"})


# Billing providers that run on OWNER hardware, i.e. a request billed under one
# of these never left the cluster. Everything else in
# `_BILLING_PROVIDER_OVERRIDES` is a PAID third-party upstream.
ON_HARDWARE_BILLING_PROVIDERS: frozenset[str] = frozenset({"pgx-local"})


def is_on_hardware_ir_pin(pin: object) -> bool:
    """True when an inference-router provider pin names an upstream on OWNER hardware.

    Used where an X-Provider / stored ``ir_provider`` pin is withheld from a
    tenant that may not reach the owner's commercial upstreams (fr#1385): a pin
    to owner hardware reaches no commercial key, so it is kept. A POSITIVE
    allowlist (:data:`ON_HARDWARE_BILLING_PROVIDERS`), exact match: an unknown or
    unregistered upstream name counts as commercial and is dropped (fail safe).
    """
    return isinstance(pin, str) and pin in ON_HARDWARE_BILLING_PROVIDERS


class PaidUpstreamForLocalPresetError(ValueError):
    """A non-external preset was paired with a model served by a PAID upstream."""


def billing_provider_for(preset_key: str, model: str, fallback: str) -> str:
    """Provider the emitted cost_event should be billed under for (preset, model).

    Returns the per-model upstream override when one exists (the `local` preset),
    else ``fallback`` (the preset's own provider — correct for external BYO).
    """
    return _BILLING_PROVIDER_OVERRIDES.get(preset_key, {}).get(model, fallback)


def paid_upstream_for(preset_key: str, model: str) -> str | None:
    """Paid upstream that serves ``model`` under ``preset_key``, or ``None``.

    ``None`` means "not known to leave owner hardware": either the preset/model
    pair has no billing override at all (an unmapped alias — see the caveat in
    :func:`assert_local_preset_stays_on_hardware`), or its upstream is in
    :data:`ON_HARDWARE_BILLING_PROVIDERS`.
    """
    upstream = billing_provider_for(preset_key, model, "")
    if not upstream or upstream in ON_HARDWARE_BILLING_PROVIDERS:
        return None
    return upstream


def assert_local_preset_stays_on_hardware(preset_key: str, model: str) -> None:
    """Refuse a stored config that claims platform inference but bills a paid API.

    ⭐ WHAT THIS COST, measured on prod 2026-08-26. Two ``tenant_llm_config``
    rows for tenant ``aaaaaaaa-0000-4000-8000-000000000001`` read
    ``provider='local', model='tier-medium'`` for roles ``tier:medium`` and
    ``tier:high``. ``tier-medium`` is owned solely by the paid ``deepseek``
    provider in the live routing configmap, so those rows spent **$140.31 over
    46,102 requests** on api.deepseek.com while the tenant's own configuration
    said "platform (local inference)". Nobody ever named DeepSeek.

    ``_local_safe_model`` (src/worker/factory/llm_config_resolver.py) already
    neutralises this at READ time, and gitops #2989 gave ``pgx-local`` a
    ``tier-medium`` alias. Both are live, and reverting either one silently
    re-arms the spend. This is the write-side half: the row stops being
    creatable at all, so the trap does not depend on a guard elsewhere staying
    in place.

    Scope, and how it differs from ``_local_safe_model`` on purpose:

    * **Wider on the model.** The read-side rule fires only on ``tier-*``
      aliases, because coercing a stored ``deepseek-v4-pro`` would silently
      downgrade a live run. At write time there is no live run to protect and a
      human is choosing, so ANY model that bills a paid upstream is refused
      under a non-external preset — including ``deepseek-v4-pro``. Pairing
      "platform inference" with a paid API is the spend trap itself, and the
      remedy is honest and already available: select the ``deepseek`` preset,
      which requires the tenant's own key.
    * **Raise, never coerce.** The read side must not turn a misconfiguration
      into an outage. A ``PUT /tenant-config/llm`` is not a hot path, so it can
      say no.
    * **Non-external presets only.** A tenant that selected Anthropic/DeepSeek/
      OpenAI directly keeps exactly what it configured.

    ⚠️ Known limitation: a model with NO entry in
    ``_BILLING_PROVIDER_OVERRIDES`` is treated as on-hardware and allowed
    through. Failing closed there would reject every legitimate new local model
    the moment the router gained one, turning catalog lag into a settings-page
    outage. That gap is not silent — an unmapped alias also prices at $0 in
    metering, which is how the existing entries came to be added.

    Raises:
        PaidUpstreamForLocalPresetError: the pair would egress to a paid API.
    """
    preset = _BY_KEY.get(preset_key)
    if preset is None or preset["is_external"]:
        return
    upstream = paid_upstream_for(preset_key, model)
    if upstream is None:
        return
    raise PaidUpstreamForLocalPresetError(
        f"Provider '{preset_key}' is platform inference (owner hardware), but model "
        f"'{model}' is served by the paid upstream '{upstream}' — storing this pair "
        f"would bill '{upstream}' for traffic the configuration calls local. "
        f"Select the '{upstream}' provider with your own API key, or choose a model "
        f"that runs on platform hardware."
    )
