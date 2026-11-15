"""Provider catalog + per-provider live model discovery.

The `GET /providers` endpoint enumerates everything worker_management's
pydantic-ai factory can dispatch to — the list is derived from the
`ModelProvider` literal, so a new provider lands here automatically
once its enum entry + factory + metadata row exist.

`GET /providers/{name}/models` live-queries the upstream backend for the
current model list. On upstream failure the caller gets a 502 with the
underlying error string — no static fallback, because a lying dropdown
is worse than a visible failure.

── TOKENLESS VERDICT, recorded for nonagenticai/worker_management#233 ──────────
Measured 2026-09-11, after `WorkerManagementTokenlessTenantSurfaceGrew`
(gitops-production `clusters/strato/services/prometheus/rules-c.yaml`, floor
`> 13`) reached 15 on the back of `/providers`, `/providers/vllm/models` and
`/providers/inference_router/models`.

⚠️ NEITHER HANDLER IN THIS FILE IS TENANT-SCOPED, AND THAT IS THE POINT — it is
also why this file is NOT the auth gap its sibling `../tenant_config/_router.py`
is. Both handlers take no auth dependency at all (`list_providers()` and
`list_provider_models(name)` have no `Depends`), read no tenant, and touch no DB:
`provider_is_available` is `bool(os.environ.get(...))` and every entry in
`PROVIDER_FETCHERS` reads only platform-global config. A tokenless caller
therefore learns NOTHING about any tenant here.

⛔ BUT "NOT A TENANT LEAK" IS NOT "INTENDED PUBLIC", AND NOTHING IN THIS REPO HAS
EVER SAID THESE ROUTES ARE PUBLIC. They are absent from `PUBLIC_ENDPOINTS`
(`src/api/auth_middleware.py:102-112`) and match none of its prefixes, so their
tokenless service is a side-effect of the trusted-network branch, not a decision
anyone recorded. What an unauthenticated pod on `10.0.0.0/8` actually gets:
  - a PLATFORM CREDENTIAL-PRESENCE ORACLE — `available` is one bit per provider
    saying whether this deployment holds that provider's API key;
  - the ability to SPEND those keys: `/{name}/models` issues a real outbound
    request to OpenAI/Anthropic/Google/... on every call;
  - internal env-var names and raw upstream error text, echoed into the 502 body
    (see `list_provider_models`, and the docstring on `list_providers` recording
    that this already reached an end user once).
None of that is tenant data; all of it is infrastructure surface. Treat it as a
lower-severity residue of the same missing decision, not as a settled exemption.

⛔ DO NOT "FIX" THIS BY ADDING A TOKEN REQUIREMENT HERE ALONE. Measured caller:
fr-backend, which constructs its client with `sendAuthorization: false`
(`fr/backend/src/integrations/workers.ts`) and so has no token to present —
a 401 here breaks fr's worker/provider picker rather than prompting a migration.
The fix belongs in the caller; see the long note in `../tenant_config/_router.py`
for the full measurement and the reason the ratchet floor must NOT be moved to 15.

── TENANT-SCOPED SINCE 2026-10-01 (owner ruling) ────────────────────────────────
⚠️ The "NEITHER HANDLER IS TENANT-SCOPED" verdict above is HISTORICAL. Both
handlers now read the request tenant (``request.state.tenant_id``) and split:

- PLATFORM tenants (``tenant_constants.commercial_platform_tenant_ids()``, the
  gitops ``COMMERCIAL_PLATFORM_TENANT_IDS``): the full catalog and any
  inference-router model. An EXTERNAL row is ``available`` only when its key is
  registered AND the request PROVED the tenant (owner ruling 2026-10-03,
  ``src.auth.tenant_proof``); otherwise ``/{name}/models`` is 403
  ``PROVIDER_NOT_ENABLED_FOR_TENANT``.
- Every OTHER tenant (a customer, or an unknown/absent tenant — fail closed):
  ``GET /providers`` lists EXACTLY the tenant's configured ``tenant_llm_config``
  rows whose key passes a live test (a model-list call with that row's own
  credential, cached ~60 s per tenant+provider+key fingerprint) — no catalog,
  nothing it has not configured, nothing that would fail on first use
  (scope ruling 2026-10-01: local-llm becomes an ordinary per-tenant row too).
  ``/providers/{name}/models`` for an external row is the live list fetched with
  the tenant's key; for ``inference_router`` it is filtered to ``local-llm``.
  A row whose key is withheld (403 ``BYO_KEY_WITHHELD``) or broken (409
  ``BYO_KEY_UNUSABLE``) says so; no row is 403 ``BYO_PROVIDER_NOT_CONFIGURED``;
  a vendor rejection is 502 ``BYO_PROVIDER_CALL_FAILED``. The owner's env keys
  are never read for a customer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import typing
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.error_handlers import NotFoundError
from src.api.tenant_constants import is_commercial_platform_tenant
from src.auth.tenant_proof import owner_commercial_keys_permitted, owner_commercial_keys_refusal
from src.auth.tokenless_gate import (
    ROUTE_PROVIDERS_LIST,
    tokenless_tenant_gate,
    tokenless_tenant_gate_provider_models,
)
from src.security.redact import redact_secrets_in_text
from src.security.request_overrides import pinned_tenant_transport
from src.services.inference_router_registry import inference_router_registry
from src.services.mlx_model_registry import mlx_model_registry
from src.settings import get_settings
from src.worker.factory.llm_config_resolver import (
    TenantLLMKeyUnusableError,
    resolve_tenant_byo_providers,
    tenant_byo_status,
)
from src.worker.provider_registry import (
    EXTERNAL_PROVIDERS,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_METADATA,
    PROVIDER_NAMES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers", tags=["Providers"])


# PROVIDER_NAMES is re-exported from the registry (where it is derived from the
# ModelProvider literal) so a new provider shows up here as soon as its enum
# entry + metadata row exist. Kept as a module attribute because tests and other
# call sites import it from this module.


class ProviderDescriptor(BaseModel):
    """One row of GET /providers."""

    name: str = Field(
        description="Provider identifier; use as model_provider when creating workers"
    )
    label: str = Field(description="Human-readable label")
    description: str = Field(description="One-line explanation")
    requires_api_key: bool = Field(
        description="True if a credential must be available server-side to reach this provider"
    )
    available: bool = Field(
        description=(
            "True if this provider can be dispatched to right now. A provider that "
            "requires a key is available only when that key is present server-side. "
            "A row with requires_api_key=true and available=false is NOT unusable — a "
            "caller supplying api_key_override (BYO) can still reach it — but selecting "
            "it with no key WILL 502. For a non-platform tenant only rows it can "
            "actually use are returned, so this is always true there."
        )
    )
    source: str = Field(
        default="platform",
        description=(
            "Whose credential serves this provider for the calling tenant: "
            "'platform' (platform routing / platform key), 'byo' (the tenant's own "
            "key) or 'managed_byo' (a key the owner set for this tenant)."
        ),
    )


class ProviderListResponse(BaseModel):
    """Response body for GET /providers."""

    object: str = "list"
    data: list[ProviderDescriptor]


class ProviderModel(BaseModel):
    """One row in the per-provider model list."""

    id: str
    object: str = "model"


class ProviderModelListResponse(BaseModel):
    """Response body for GET /providers/{name}/models."""

    object: str = "list"
    provider: str
    data: list[ProviderModel]


def provider_is_available(name: str, tenant_id: str | None) -> bool:
    """True if ``tenant_id`` can dispatch to ``name`` with the credentials present now.

    A keyless provider (vllm, inference_router, mlx, vlm_local) is always
    "available" in the credential sense — whether its *backend* is reachable is a
    different question, and the honest place to answer it is the live
    ``GET /providers/{name}/models`` call, not a cached catalog flag.

    An EXTERNAL provider (the owner's commercial keys) is available only when
    BOTH hold (owner ruling 2026-10-03, fr#1385):

    * its key is REGISTERED — mounted into this pod from gitops secrets (today
      only ``XAI_API_KEY``; ``anthropic``/``deepseek`` keys live in
      inference-router, not here, so those are reached via ``inference_router``);
    * ``tenant_id`` may use the owner's keys —
      :func:`src.auth.tenant_proof.owner_commercial_keys_permitted`: a gitops
      commercial-platform tenant that THIS request proved. An asserted header
      or the default tenant never qualifies.

    This MUST agree with the 403 ``PROVIDER_NOT_ENABLED_FOR_TENANT`` in
    :func:`list_provider_models` and the chat router: measured fr#1385
    (2026-10-02), the catalog once said ``xai`` was available while the very next
    call was refused, and fr's picker offered a provider that could never work.

    Read from ``os.environ`` on every call rather than snapshotting at import.
    """
    env_var = PROVIDER_API_KEY_ENV_VARS.get(name)
    if name in EXTERNAL_PROVIDERS and not owner_commercial_keys_permitted(tenant_id):
        return False
    if env_var is None:
        return True
    return bool(os.environ.get(env_var))


# Providers a customer tenant is never offered, even though they are local: they
# are direct paths to owner GPU hosts that bypass inference-router (and with it
# the router's per-tenant gate and metering). Customers reach owner hardware
# only via ``inference_router`` / ``local-llm``.
_CUSTOMER_PLATFORM_PROVIDER = "inference_router"


def caller_tenant_id(request: Request) -> str | None:
    """The tenant the auth middleware resolved for this request, or None."""
    tenant_id = getattr(request.state, "tenant_id", None)
    return str(tenant_id) if tenant_id else None


def _caller_byo_mode(request: Request) -> str:
    return str(getattr(request.state, "byo_mode", None) or "local_allowed")


async def tenant_byo_providers(request: Request) -> dict[str, dict[str, Any]]:
    """The caller tenant's usable BYO providers (see resolve_tenant_byo_providers)."""
    return await resolve_tenant_byo_providers(
        caller_tenant_id(request), byo_mode=_caller_byo_mode(request)
    )


async def tenant_byo_provider_status(
    request: Request, *, include_local: bool = False
) -> dict[str, dict[str, Any]]:
    """Per-provider usable/withheld/unusable status (see tenant_byo_status)."""
    return await tenant_byo_status(
        caller_tenant_id(request),
        byo_mode=_caller_byo_mode(request),
        include_local=include_local,
    )


class TenantLLMHTTPException(HTTPException):
    """A 4xx/5xx with a machine-readable ``error.code`` and ``error.details``.

    Rendered by ``error_handlers.http_exception_handler`` as
    ``{"error": {"code", "message", "status", "details"}}``.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        if details:
            self.error_details = details


def key_error_http(err: TenantLLMKeyUnusableError) -> TenantLLMHTTPException:
    """The HTTP form of a withheld (403) / unusable (409) tenant key."""
    return TenantLLMHTTPException(err.status_code, err.error_code, err.message, err.details())


def byo_call_failed(
    tenant_id: str | None, provider: str, error: str, *, model: str | None = None
) -> TenantLLMHTTPException:
    """502 for a vendor call on the TENANT'S OWN key that failed.

    Never retried on a platform model.
    """
    on_model = ""
    if model:
        on_model = f" (model {model!r})"
    return TenantLLMHTTPException(
        502,
        "BYO_PROVIDER_CALL_FAILED",
        (
            f"The call to tenant {tenant_id}'s own {provider} key{on_model} failed: {error}. "
            "Nothing was run on a platform model instead. Fix: check the API key, its "
            "quota and the model name in the tenant LLM settings "
            "(POST /tenant-config/llm/test tests the key)."
        ),
        {"tenant_id": tenant_id, "provider": provider, "model": model},
    )


def byo_not_configured(tenant_id: str | None, provider: str) -> TenantLLMHTTPException:
    """403 for a provider the tenant has no own key for.

    Names no key, env var or owner.
    """
    return TenantLLMHTTPException(
        403,
        "BYO_PROVIDER_NOT_CONFIGURED",
        (
            f"Provider '{provider}' is not available to this tenant. To use it, add your "
            "own API key for it in the tenant LLM settings (bring your own key)."
        ),
        {"tenant_id": tenant_id, "provider": provider},
    )


def provider_not_enabled_for_tenant(
    tenant_id: str | None, provider: str, reason: str | None = None
) -> TenantLLMHTTPException:
    """403 ``PROVIDER_NOT_ENABLED_FOR_TENANT``: a platform provider this tenant may not use.

    The typed refusal for the OWNER'S commercial keys (fr#1385): the tenant is
    not a proven commercial-platform workspace. fr matches on this CODE, never
    on a bare 403. Names no key, env var or owner. ``reason`` is the closed
    refusal reason from :func:`owner_commercial_keys_refusal` (log + details).
    """
    logger.info(
        "provider_refused_not_enabled_for_tenant",
        extra={"provider": provider, "tenant_id": tenant_id, "reason": reason},
    )
    return TenantLLMHTTPException(
        403,
        "PROVIDER_NOT_ENABLED_FOR_TENANT",
        (
            f"Provider '{provider}' is not enabled for this tenant. The platform's own "
            "provider keys serve only the owner's platform workspaces, and only to a "
            "caller that proves the workspace (a licence, or a verified end-user token). "
            "Use inference_router/local-llm, or add your own API key for this provider "
            "in the tenant LLM settings."
        ),
        {"tenant_id": tenant_id, "provider": provider, "reason": reason},
    )


# ── Live key verification (owner ruling 2026-10-01, item 5) ─────────────────────
# A customer's GET /providers lists a BYO provider only when a LIVE model-list
# fetch with the tenant's key succeeds. Results are cached per (tenant,
# provider, key fingerprint, endpoint) so a key edit is seen at once (new
# fingerprint), and so the catalog does not hit the vendor on every page load.
# A success is remembered for BYO_LIVE_OK_TTL_S, a failure for a shorter
# BYO_LIVE_FAIL_TTL_S. The key itself is never a cache key: only a truncated
# SHA-256 of it.
BYO_LIVE_OK_TTL_S = 60
BYO_LIVE_FAIL_TTL_S = 15
_byo_live_ok: TTLCache[tuple[str, str, str, str], list[str]] = TTLCache(
    maxsize=1024, ttl=BYO_LIVE_OK_TTL_S
)
_byo_live_fail: TTLCache[tuple[str, str, str, str], str] = TTLCache(
    maxsize=1024, ttl=BYO_LIVE_FAIL_TTL_S
)


class ByoLiveCheckError(RuntimeError):
    """The live model fetch with the tenant's key failed (message is secret-free)."""


def _byo_cache_key(
    tenant_id: str, provider: str, entry: dict[str, Any]
) -> tuple[str, str, str, str]:
    fp = hashlib.sha256(str(entry.get("api_key") or "").encode("utf-8")).hexdigest()[:16]
    return (tenant_id, provider, fp, str(entry.get("base_url") or ""))


def _scrub(text: str, secret: str | None) -> str:
    """Remove credentials from an upstream error string (issue #327).

    The exact key plus any credential-bearing URL parameter or known key shape
    (:func:`src.security.redact.redact_secrets_in_text`).
    """
    return redact_secrets_in_text(text, secret)


async def _fetch_row_models(tenant_id: str, provider: str, entry: dict[str, Any]) -> list[str]:
    """One live model-list call for a tenant row, with THAT row's credential.

    External presets: the vendor's model list with the tenant key (cheapest
    reliable authenticated call — free, no tokens, and every listed vendor
    rejects a bad key on it with 401/403). ``inference_router`` rows (the
    ``local`` preset): inference-router's ``/api/models`` for this tenant —
    forced fresh, so an outage is a red test, not a cached list.
    """
    if provider == _CUSTOMER_PLATFORM_PROVIDER:
        models = await inference_router_registry.refresh_model_cache(tenant_id)
        if not models:
            raise RuntimeError("inference-router returned no models for this tenant")
        return [m.id for m in models]
    fetcher = PROVIDER_FETCHERS.get(provider)
    if fetcher is None or provider not in EXTERNAL_PROVIDERS:
        raise RuntimeError(f"no model-list backend for provider '{provider}'")
    return await fetcher(api_key=entry.get("api_key") or "", base_url=entry.get("base_url"))


async def live_row_models(
    tenant_id: str, provider: str, entry: dict[str, Any], *, fresh: bool = False
) -> list[str]:
    """Model ids a tenant row's credential can see; raises ByoLiveCheckError.

    ``fresh=True`` bypasses (and refreshes) the cache — the model-list route and
    the key-test route always want the live answer.
    """
    key = _byo_cache_key(tenant_id, provider, entry)
    if not fresh:
        if key in _byo_live_ok:
            return _byo_live_ok[key]
        if key in _byo_live_fail:
            raise ByoLiveCheckError(_byo_live_fail[key])
    try:
        ids = await _fetch_row_models(tenant_id, provider, entry)
    except Exception as exc:
        msg = _scrub(str(exc), entry.get("api_key")) or type(exc).__name__
        _byo_live_fail[key] = msg
        _byo_live_ok.pop(key, None)
        raise ByoLiveCheckError(msg) from None
    _byo_live_fail.pop(key, None)
    _byo_live_ok[key] = ids
    return ids


def customer_platform_model() -> str:
    """The one inference-router model a customer tenant may use (``local-llm``)."""
    return get_settings().local_llm_model


def _descriptor(name: str, *, available: bool, source: str = "platform") -> ProviderDescriptor:
    return ProviderDescriptor(
        name=name,
        label=PROVIDER_METADATA[name]["label"],
        description=PROVIDER_METADATA[name]["description"],
        requires_api_key=PROVIDER_METADATA[name]["requires_api_key"],
        available=available,
        source=source,
    )


@router.get(
    "",
    response_model=ProviderListResponse,
    operation_id="list_providers",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_PROVIDERS_LIST))],
)
async def list_providers(request: Request) -> ProviderListResponse:
    """Enumerate the providers the CALLING TENANT can route to.

    Platform tenants (gitops ``COMMERCIAL_PLATFORM_TENANT_IDS``) get every
    provider. ``available`` there is per tenant (owner ruling 2026-10-03,
    fr#1385): a keyless provider is available; an EXTERNAL provider (the owner's
    commercial keys) is available only when its key is registered (mounted from
    gitops secrets) AND this request PROVED the tenant — a licence, or a
    verified end-user token. An asserted ``X-Tenant-Id`` / delegation header, or
    the default tenant, sees every external row unavailable, exactly as
    ``/{name}/models`` and ``/chat/completions`` then refuse it (403
    ``PROVIDER_NOT_ENABLED_FOR_TENANT``).

    Every other tenant (a customer, unknown or absent) gets exactly its
    configured ``tenant_llm_config`` rows whose key passes a live test — its own
    BYO keys and the ``local`` inference-router row — and never the owner's
    keys or a local direct provider.

    ⚠️ ``available`` is a CREDENTIAL fact, not a reachability one. Two keyless
    platform rows are reported available but can still fail: ``vlm_local`` (its
    mini2 backend is deliberately dormant; ``/{name}/models`` 502s) and ``mlx``
    (can return ``200 {"data": []}``). The honest place to answer reachability
    is the live ``/{name}/models`` call.
    """
    tenant_id = caller_tenant_id(request)
    if is_commercial_platform_tenant(tenant_id):
        rows = [
            _descriptor(name, available=provider_is_available(name, tenant_id))
            for name in PROVIDER_NAMES
        ]
        return ProviderListResponse(data=rows)

    # A customer sees EXACTLY its configured tenant_llm_config rows whose key
    # tests OK (a live model-list call with that row's credential; see
    # live_row_models) — nothing it has not configured, nothing that would fail
    # on first use. /chat is the tenant admin's key-test tool.
    status = await tenant_byo_provider_status(request, include_local=True)
    candidates = [n for n in PROVIDER_NAMES if status.get(n, {}).get("status") == "usable"]

    async def _works(name: str) -> bool:
        try:
            await live_row_models(str(tenant_id), name, status[name]["entry"])
        except ByoLiveCheckError as exc:
            logger.info(
                "providers_row_key_check_failed",
                extra={"tenant_id": tenant_id, "provider": name, "error": str(exc)},
            )
            return False
        return True

    verdicts = await asyncio.gather(*(_works(n) for n in candidates))
    rows = [
        _descriptor(name, available=True, source=_row_source(status[name]["entry"]))
        for name, ok in zip(candidates, verdicts, strict=True)
        if ok
    ]
    return ProviderListResponse(data=rows)


def _row_source(entry: dict[str, Any]) -> str:
    """The row's credential source.

    ``managed_byo`` (owner set the key), ``byo`` (tenant's own key) or
    ``platform`` (a keyless platform row, i.e. the ``local`` preset).
    """
    if entry.get("owner_provided"):
        return "managed_byo"
    if entry.get("api_key"):
        return "byo"
    return "platform"


async def _httpx_json_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    tenant_endpoint: bool = False,
) -> dict[str, Any]:
    """Thin wrapper around httpx.AsyncClient.get so tests can patch one symbol.

    ``tenant_endpoint=True`` (a tenant row's own ``base_url``) dials through
    :func:`pinned_tenant_transport` (WM-6): the address that passed the
    owner-infra check is the address connected to, with no second lookup.
    """
    transport = pinned_tenant_transport() if tenant_endpoint else None
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await client.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


def _tenant_endpoint_kw(base_url: str | None) -> dict[str, Any]:
    """``{"tenant_endpoint": True}`` for a tenant ``base_url`` fetch, else ``{}``.

    Empty for the vendor-default URL so the platform path's call shape is
    unchanged.
    """
    return {"tenant_endpoint": True} if base_url else {}


def _models_url(base_url: str | None, default: str) -> str:
    """``{base_url}/models`` for an OpenAI-compatible BYO endpoint, else ``default``.

    ``base_url`` is the tenant row's endpoint, already allowlisted by the BYO
    resolver before it is handed here.
    """
    if base_url:
        return f"{base_url.rstrip('/')}/models"
    return default


def _require_env(name: str) -> str:
    """Return env var value or raise RuntimeError with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _key_or_env(api_key: str | None, provider: str) -> str:
    """The credential for a model-list fetch.

    ``api_key is None`` ⇒ the PLATFORM path (the caller passed no key): the
    owner's env key, as before. A passed key — the tenant's BYO key — is used
    AS-IS and an EMPTY one is refused: it must never be topped up with the
    owner's env key (validator N4; ``api_key or env`` did exactly that).
    """
    if api_key is None:
        return _require_env(PROVIDER_API_KEY_ENV_VARS[provider])
    if not api_key:
        raise RuntimeError(f"No API key for provider '{provider}' (empty tenant key)")
    return api_key


async def _fetch_openai_models(
    api_key: str | None = None, base_url: str | None = None
) -> list[str]:
    key = _key_or_env(api_key, "openai")
    data = await _httpx_json_get(
        _models_url(base_url, "https://api.openai.com/v1/models"),
        headers={"Authorization": f"Bearer {key}"},
        **_tenant_endpoint_kw(base_url),
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_deepseek_models(
    api_key: str | None = None, base_url: str | None = None
) -> list[str]:
    key = _key_or_env(api_key, "deepseek")
    data = await _httpx_json_get(
        _models_url(base_url, "https://api.deepseek.com/v1/models"),
        headers={"Authorization": f"Bearer {key}"},
        **_tenant_endpoint_kw(base_url),
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_groq_models(api_key: str | None = None, base_url: str | None = None) -> list[str]:
    key = _key_or_env(api_key, "groq")
    data = await _httpx_json_get(
        _models_url(base_url, "https://api.groq.com/openai/v1/models"),
        headers={"Authorization": f"Bearer {key}"},
        **_tenant_endpoint_kw(base_url),
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_xai_models(api_key: str | None = None, base_url: str | None = None) -> list[str]:
    key = _key_or_env(api_key, "xai")
    data = await _httpx_json_get(
        _models_url(base_url, "https://api.x.ai/v1/models"),
        headers={"Authorization": f"Bearer {key}"},
        **_tenant_endpoint_kw(base_url),
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_cohere_models(
    api_key: str | None = None,
    base_url: str | None = None,  # noqa: ARG001 -- uniform BYO fetcher signature
) -> list[str]:
    key = _key_or_env(api_key, "cohere")
    data = await _httpx_json_get(
        "https://api.cohere.ai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    return [m["name"] for m in data.get("models", [])]


async def _fetch_google_models(
    api_key: str | None = None,
    base_url: str | None = None,  # noqa: ARG001 -- uniform BYO fetcher signature
) -> list[str]:
    key = _key_or_env(api_key, "google")
    # The key goes in the ``x-goog-api-key`` header, NEVER the ``?key=`` query
    # string: httpx errors echo the full request URL, which carried the key into
    # the 502 detail, the BYO relay and the logs (issue #327). The Gemini API
    # documents this header for its REST endpoints.
    data = await _httpx_json_get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
    )
    return [m["name"].removeprefix("models/") for m in data.get("models", []) if "name" in m]


async def _fetch_anthropic_models(
    api_key: str | None = None,
    base_url: str | None = None,  # noqa: ARG001 -- uniform BYO fetcher signature
) -> list[str]:
    key = _key_or_env(api_key, "anthropic")
    data = await _httpx_json_get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_vllm_models(tenant_id: str | None = None) -> list[str]:
    base = os.environ.get("VLLM_URL") or get_settings().vllm_url
    if not base:
        raise RuntimeError("VLLM_URL is not configured")
    base = base.rstrip("/")
    url = f"{base}/v1/models"
    # VLLM_URL IS inference-router in prod (gitops #2999): this is a router
    # call and carries the caller tenant's licence token like any other. A bare
    # vLLM (any other origin) gets no identity headers, exactly as before.
    from src.services.ir_bundle_tokens import ir_listing_headers, is_ir_origin

    headers = await ir_listing_headers(url, tenant_id) if is_ir_origin(url) else None
    data = await _httpx_json_get(url, headers=headers)
    return [m["id"] for m in data.get("data", [])]


async def _fetch_vlm_local_models() -> list[str]:
    base = os.environ.get("VLM_LOCAL_URL") or get_settings().vlm_local_url
    if not base:
        raise RuntimeError("VLM_LOCAL_URL is not configured")
    base = base.rstrip("/")
    data = await _httpx_json_get(f"{base}/models")
    return [m["id"] for m in data.get("data", [])]


async def _fetch_mlx_models() -> list[str]:
    payload = await mlx_model_registry.get_models_for_api()
    return [m["id"] for m in payload.get("data", [])]


async def _fetch_inference_router_models(tenant_id: str | None = None) -> list[str]:
    payload = await inference_router_registry.get_models_for_api(tenant_id)
    return [m["id"] for m in payload.get("data", [])]


PROVIDER_FETCHERS: dict[str, typing.Callable[..., typing.Awaitable[list[str]]]] = {
    "openai": _fetch_openai_models,
    "deepseek": _fetch_deepseek_models,
    "groq": _fetch_groq_models,
    "cohere": _fetch_cohere_models,
    "mlx": _fetch_mlx_models,
    "inference_router": _fetch_inference_router_models,
    "anthropic": _fetch_anthropic_models,
    "google": _fetch_google_models,
    "vllm": _fetch_vllm_models,
    "vlm_local": _fetch_vlm_local_models,
    "xai": _fetch_xai_models,
}


# Import-time drift guard. Every provider the API advertises must have a way to
# list its models; a missing row used to surface as an unhandled KeyError (500)
# on the first request for that provider. Failing at import turns it into a
# CI/pod-start failure instead of a production 500.
_UNROUTED_PROVIDERS = sorted(set(PROVIDER_NAMES) - set(PROVIDER_FETCHERS))
if _UNROUTED_PROVIDERS:
    raise RuntimeError(
        f"PROVIDER_FETCHERS is missing a model fetcher for {_UNROUTED_PROVIDERS}; "
        "every ModelProvider must be routable by GET /providers/{name}/models"
    )


async def _run_fetcher(name: str, fetch: typing.Awaitable[list[str]]) -> list[str]:
    from src.services.ir_bundle_tokens import IRTenantNotProvenError, IRTenantRequiredError

    try:
        return await fetch
    except IRTenantRequiredError, IRTenantNotProvenError:
        # A deliberate, typed refusal (no / unproven end tenant) -- not an
        # upstream outage.
        # Already counted + logged at ERROR; surface its own 403, never a 502.
        raise
    except Exception as exc:
        # Upstream error text can carry the credential (an httpx error echoes the
        # request URL; issue #327). Scrub credential query params and known key
        # shapes before the text reaches the log line or the 502 detail. The owner
        # env key is deliberately NOT read here to match it exactly: that would be
        # one more computed-name env reader (tests/test_owner_key_proof_sites.py),
        # and every fetcher now sends its key in a header, which httpx never echoes.
        error = redact_secrets_in_text(str(exc))
        logger.warning("provider_models_fetch_failed", extra={"provider": name, "error": error})
        # ``str(exc)`` is EMPTY for some httpx errors (a bare ConnectError /
        # ReadTimeout), which left "...for provider 'vlm_local': " with no
        # reason at all. Name the exception type instead of nothing.
        detail = f"Upstream unreachable for provider '{name}': {error or type(exc).__name__}"
    # Raised OUTSIDE the ``except`` block on purpose: ``raise ... from None`` only
    # hides the traceback, the unscrubbed exception would still sit on
    # ``__context__`` for any chain walker to render. Out here there is no
    # handled exception, so both ``__cause__`` and ``__context__`` are None.
    raise HTTPException(status_code=502, detail=detail)


def _model_list(name: str, ids: list[str]) -> ProviderModelListResponse:
    return ProviderModelListResponse(provider=name, data=[ProviderModel(id=mid) for mid in ids])


async def _list_customer_provider_models(
    name: str, request: Request, tenant_id: str | None
) -> ProviderModelListResponse:
    """Model list for a NON-platform tenant. Never reads an owner env key."""
    if name == _CUSTOMER_PLATFORM_PROVIDER:
        # Forward the tenant so inference-router's own per-tenant gate applies,
        # and filter here as well: until that gate is live IR returns its full
        # list, and a customer must still see only the platform model.
        allowed = customer_platform_model()
        ids = await _run_fetcher(name, _fetch_inference_router_models(tenant_id))
        return _model_list(name, [mid for mid in ids if mid == allowed])

    status = (await tenant_byo_provider_status(request)) if name in EXTERNAL_PROVIDERS else {}
    st = status.get(name)
    if st is None or name not in PROVIDER_FETCHERS:
        logger.info(
            "provider_models_refused_not_tenant_usable",
            extra={"provider": name, "tenant_id": tenant_id},
        )
        raise byo_not_configured(tenant_id, name)
    if st["status"] != "usable":
        # The tenant HAS a row for this provider; say why it cannot be used.
        raise key_error_http(st["error"])
    try:
        # The LATEST list, live, with the tenant's key (refreshes the cache the
        # /providers catalog reads).
        ids = await live_row_models(str(tenant_id), name, st["entry"], fresh=True)
    except ByoLiveCheckError as exc:
        raise byo_call_failed(tenant_id, name, str(exc)) from None
    return _model_list(name, ids)


@router.get(
    "/{name}/models",
    response_model=ProviderModelListResponse,
    operation_id="list_provider_models",
    dependencies=[Depends(tokenless_tenant_gate_provider_models)],
)
async def list_provider_models(name: str, request: Request) -> ProviderModelListResponse:
    """Live-query the upstream for the provider's current model list, for this tenant."""
    if name not in PROVIDER_NAMES:
        raise NotFoundError(resource="Provider", identifier=name)

    tenant_id = caller_tenant_id(request)
    if not is_commercial_platform_tenant(tenant_id):
        return await _list_customer_provider_models(name, request, tenant_id)

    # ⛔ The owner-key gate applies HERE TOO, and this is the one external call
    # that does NOT go through ModelFactory: the fetchers below hit api.x.ai and
    # friends directly with httpx and a mounted credential. Listing models is
    # unbilled, but it is still this estate reaching a paid vendor with the
    # OWNER'S key, so it follows the same rule as a completion: a proven
    # commercial-platform tenant only (owner ruling 2026-10-03, fr#1385). Same
    # predicate as ``provider_is_available``, so the catalog and this call agree.
    # 403 with a TYPED code, not 502: the upstream is reachable, we refuse on
    # purpose, and fr renders the code as a state.
    if name in EXTERNAL_PROVIDERS:
        refusal = owner_commercial_keys_refusal(tenant_id)
        if refusal is not None:
            raise provider_not_enabled_for_tenant(tenant_id, name, refusal)

    # Belt-and-braces: the import-time guard above makes this unreachable for a
    # correctly-wired build, but a bare PROVIDER_FETCHERS[name] would turn any
    # future gap into an unhandled KeyError -> 500. A known provider we cannot
    # reach is an upstream/config problem, not "no such provider", so 502.
    fetcher = PROVIDER_FETCHERS.get(name)
    if fetcher is None:
        logger.error("provider_fetcher_missing", extra={"provider": name})
        raise HTTPException(
            status_code=502,
            detail=f"No model-discovery backend is configured for provider '{name}'",
        )

    # inference-router is always told WHO is asking (its per-tenant gate).
    # vllm is inference-router in prod, so it is told too (and licensed).
    fetch = fetcher(tenant_id) if name in (_CUSTOMER_PLATFORM_PROVIDER, "vllm") else fetcher()
    return _model_list(name, await _run_fetcher(name, fetch))
