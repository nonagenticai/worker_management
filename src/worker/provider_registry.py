"""Single source of truth for the LLM provider registry.

Everything that needs to know *which* providers exist, *which* of them need a
credential, and *which* environment variable carries that credential reads it
from here. Historically this knowledge was copy-pasted into six places
(``WorkerCreateRequest`` twice, ``AgentTemplate``'s ``Literal`` and a runtime
``valid_providers`` list next to it, ``safe_create_pydantic_ai_agent``'s
``Literal``, and the providers router) and drifted: ``cohere``/``google``
skipped API-key validation entirely, and ``xai``/``google``/``vlm_local``
were rejected by ``AgentTemplate`` despite being fully supported by
``ModelFactory``.

This module is deliberately **dependency-free** (stdlib only), and it lives in
``src.worker`` rather than ``src.worker.agents`` on purpose: ``src/worker/
agents/__init__.py`` eagerly imports every agent implementation, so anything
under it costs the full ``pydantic_ai`` / ``cohere`` / ``groq`` / ``openai``
import chain. ``src/worker/__init__.py`` is docstring-only, so importing this
module costs nothing. That is what lets the lightweight consumers — the API
request schemas, ``worker_template``, and the optional-dependency wrapper
``pydantic_ai_agent_wrapper`` — read the registry cheaply.
:mod:`src.worker.agents.pydantic_ai_agent` re-exports ``ModelProvider`` and
``PROVIDER_METADATA`` so existing call sites and tests keep working unchanged.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

# Ordering rationale: cluster-local providers first (no API key required, most
# likely to resolve models on page-load), then the inference router, then
# hosted providers in usage-frequency order. fr's /chat auto-selects the first
# entry on page load — keep a known-good default at the top.
ModelProvider = Literal[
    "vllm",
    "inference_router",
    "mlx",
    "vlm_local",
    "anthropic",
    "deepseek",
    "groq",
    "openai",
    "cohere",
    "google",
    "xai",
]

# Metadata describing each provider for the GET /providers endpoint AND for
# every API-key validation decision in the codebase.
#
# The ``# pragma: allowlist secret`` markers below are NOT hiding credentials.
# These values are env var *names* ("OPENAI_API_KEY"), never key material —
# detect-secrets' KeywordDetector matches the ``"..._key...": "LITERAL"`` shape
# and cannot tell a variable name from a value. They are allowlisted at the line
# rather than recorded in .secrets.baseline, because putting them in the
# baseline would assert "known secret, reviewed" about something that is not a
# secret at all.
#
# ``requires_api_key`` and ``api_key_env_var`` are two faces of one fact and are
# checked for agreement at import time below — a provider that needs a key must
# name the env var that carries it, and a keyless provider must not name one.
# ``api_key_env_var`` values must match what the corresponding
# ``ModelFactory._create_*_model`` passes to ``ModelFactory.get_api_key``
# (note ``google`` reads GEMINI_API_KEY, not GOOGLE_API_KEY).
#
# ⛔ ``external`` IS A SEPARATE FACT FROM ``requires_api_key``, and conflating
# them is the failure this field exists to prevent. It answers exactly one
# question: does a call to this provider leave hardware we own? Today the two
# happen to partition the registry identically, which is precisely why deriving
# "external" from "needs a credential" would look correct and rot silently —
# a self-hosted backend put behind an auth token would become "external", and a
# free-tier hosted API with no key would become "local". The estate's local-only
# policy (``ModelFactory.enforce_local_only_provider``) is enforced against THIS
# field, so it is declared per row and checked at import time: a provider added
# without it is an ImportError, not a silently-permitted egress path.
#
# Keys match ModelProvider literal values. Same ordering as the Literal so
# /providers returns rows in the UI-preferred order.
PROVIDER_METADATA: dict[str, dict[str, Any]] = {
    "vllm": {
        "label": "vLLM (GPU)",
        "description": "pgx vLLM model (local-llm), routed via inference-router (VLLM_URL).",
        "requires_api_key": False,
        "api_key_env_var": None,
        "external": False,
    },
    "inference_router": {
        "label": "Inference Router",
        "description": "Failover + metering router (pgx vLLM primary, DeepSeek paid fallback).",
        "requires_api_key": False,
        "api_key_env_var": None,
        # NOT external: worker_management's hop terminates at inference-router,
        # in-cluster. IR may itself hold paid upstreams in its chain, but that
        # is IR's policy to enforce (its paid providers are disabled), and this
        # flag describes OUR egress, not the estate's whole topology.
        "external": False,
    },
    "mlx": {
        "label": "MLX Inference",
        "description": "Native MLX on Apple Silicon dev boxes (mini1/mini2) via MLX registry.",
        "requires_api_key": False,
        "api_key_env_var": None,
        "external": False,
    },
    "vlm_local": {
        "label": "VLM Local (mini2)",
        "description": "MLX-VLM on mini2 via vlm.example.com:9203 (vision-only).",
        "requires_api_key": False,
        "api_key_env_var": None,
        "external": False,
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "description": "Hosted Anthropic API (claude-sonnet, claude-opus, claude-haiku).",
        "requires_api_key": True,
        "api_key_env_var": "ANTHROPIC_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "description": "Hosted DeepSeek API (deepseek-chat, deepseek-reasoner).",
        "requires_api_key": True,
        "api_key_env_var": "DEEPSEEK_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "groq": {
        "label": "Groq",
        "description": "Hosted Groq API (mixtral, llama, …).",
        "requires_api_key": True,
        "api_key_env_var": "GROQ_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "openai": {
        "label": "OpenAI",
        "description": "Hosted OpenAI API (gpt-4, gpt-4o, …).",
        "requires_api_key": True,
        "api_key_env_var": "OPENAI_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "cohere": {
        "label": "Cohere",
        "description": "Hosted Cohere API (command-r, …).",
        "requires_api_key": True,
        "api_key_env_var": "COHERE_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "google": {
        "label": "Google Gemini",
        "description": "Hosted Google Gemini API (gemini-2.5-pro, gemini-2.5-flash).",
        "requires_api_key": True,
        "api_key_env_var": "GEMINI_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
    "xai": {
        "label": "xAI Grok",
        "description": "Hosted xAI API (grok-4.5, grok-4.3, grok-4.20-* …).",
        "requires_api_key": True,
        # ⚠️ The native xAI SDK talks to api.x.ai directly — it does NOT go
        # through inference-router, so disabling IR's paid providers does
        # nothing for this path. That gap is what the local-only guard closes.
        "api_key_env_var": "XAI_API_KEY",  # pragma: allowlist secret
        "external": True,
    },
}


# Every provider name, in declaration order. get_args() on a Literal returns the
# literal values in the order they were written.
PROVIDER_NAMES: tuple[str, ...] = get_args(ModelProvider)

# Providers that need a server-side credential — derived, never hand-listed.
API_KEY_PROVIDERS: frozenset[str] = frozenset(
    name for name, meta in PROVIDER_METADATA.items() if meta["requires_api_key"]
)

# provider -> environment variable carrying its credential. Derived from the
# same rows as API_KEY_PROVIDERS, so the two can never disagree.
PROVIDER_API_KEY_ENV_VARS: dict[str, str] = {
    name: meta["api_key_env_var"]
    for name, meta in PROVIDER_METADATA.items()
    if meta["api_key_env_var"]
}

# Providers whose client leaves our hardware — derived, never hand-listed, so
# adding a row to PROVIDER_METADATA extends these automatically.
EXTERNAL_PROVIDERS: frozenset[str] = frozenset(
    name for name, meta in PROVIDER_METADATA.items() if meta["external"]
)

# ⭐ THE ALLOWLIST THE LOCAL-ONLY GUARD IS BUILT ON. It is deliberately the
# POSITIVE set: the guard admits membership here and refuses everything else,
# including a provider name that reaches the factory without a registry row at
# all. Deriving the refusal from EXTERNAL_PROVIDERS instead would fail OPEN for
# exactly the case nobody anticipates — an unregistered name.
LOCAL_PROVIDERS: frozenset[str] = frozenset(
    name for name, meta in PROVIDER_METADATA.items() if not meta["external"]
)


def _validate_registry() -> None:
    """Fail loudly at import time if the registry rows are internally inconsistent.

    Cheap enough to run unconditionally, and it turns "a provider silently skips
    API-key validation" (the 2026 cohere/google bug) into an ImportError that no
    CI run or pod start can miss.

    Raises:
        RuntimeError: if metadata keys drift from ModelProvider, or if
            ``requires_api_key`` and ``api_key_env_var`` disagree for any row.
    """
    literal_names = set(PROVIDER_NAMES)
    metadata_names = set(PROVIDER_METADATA)
    if literal_names != metadata_names:
        raise RuntimeError(
            "provider registry drift: PROVIDER_METADATA keys "
            f"{sorted(metadata_names)} != ModelProvider {sorted(literal_names)}"
        )

    mismatched = sorted(
        name
        for name, meta in PROVIDER_METADATA.items()
        if bool(meta["requires_api_key"]) is not bool(meta["api_key_env_var"])
    )
    if mismatched:
        raise RuntimeError(
            "provider registry drift: requires_api_key and api_key_env_var "
            f"disagree for {mismatched}"
        )

    # ⛔ ``external`` must be DECLARED, and declared as a bool. A row that omits
    # it, or sets it to something truthy-but-not-boolean, is a provider whose
    # egress nobody classified — and the local-only guard would then be deciding
    # policy from an accident. Fail at import rather than at the first call.
    undeclared = sorted(
        name
        for name, meta in PROVIDER_METADATA.items()
        if not isinstance(meta.get("external"), bool)
    )
    if undeclared:
        raise RuntimeError(
            "provider registry drift: every provider must declare `external` as a "
            f"bool (does the call leave our hardware?); missing/invalid for {undeclared}"
        )


_validate_registry()
