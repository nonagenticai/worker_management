"""Inference Router Registry — model discovery via inference-router /api/models."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx
from cachetools import TTLCache

from src.settings import get_settings

logger = logging.getLogger(__name__)


def _settings_inference_router_url() -> str:
    return get_settings().inference_router_url


def _settings_inference_router_cache_ttl() -> int:
    return get_settings().inference_router_cache_ttl


@dataclass
class InferenceRouterModel:
    """A model (schema_id) available via the inference-router."""

    id: str  # schema_id (e.g. "ks-entity-spotter-agent") or "default"
    provider_chain: list[str] = field(default_factory=list)
    has_grammar: bool = False
    has_json_schema: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this provider registration."""
        return {
            "id": self.id,
            "provider_chain": self.provider_chain,
            "has_grammar": self.has_grammar,
            "has_json_schema": self.has_json_schema,
        }


# Cache key for a call that carries no tenant (worker-factory model validation,
# startup). Never a tenant id, so it cannot collide with one.
_NO_TENANT_KEY = ""


# How long a FAILED fetch is remembered (see InferenceRouterRegistry.__init__).
_FAILURE_CACHE_TTL_S = 5


def _cache_key(tenant_id: str | None) -> str:
    return tenant_id or _NO_TENANT_KEY


class InferenceRouterRegistry:
    """Registry that discovers models from the inference-router /api/models endpoint.

    ⛔ PER-TENANT. inference-router filters ``/api/models`` by the caller's
    ``X-Tenant-Id`` (a customer sees ``local-llm`` only; owner ruling
    2026-10-01), so the list is a property of the TENANT, not of the router. The
    cache is therefore keyed by tenant id: one ``"models"`` slot shared by every
    caller would hand tenant A whatever list tenant B fetched last.
    """

    def __init__(self) -> None:
        self._model_cache: TTLCache[str, list[InferenceRouterModel]] = TTLCache(
            maxsize=256, ttl=_settings_inference_router_cache_ttl()
        )
        # A FAILED fetch is remembered only briefly (validator N5): long enough
        # that an inference-router outage does not turn every request into a
        # 10 s timeout, short enough that recovery is seen within seconds
        # rather than after the full model-cache TTL.
        self._failure_cache: TTLCache[str, bool] = TTLCache(maxsize=256, ttl=_FAILURE_CACHE_TTL_S)
        # Single-flight PER TENANT (validator N5): concurrent cold misses for
        # the SAME tenant share one fetch, and different tenants never wait on
        # each other. One global lock held across the 10 s HTTP call used to
        # serialise every tenant's cold miss behind any other's. Entries live
        # only while their fetch is in flight, so the dict is bounded by
        # concurrency, not by the number of tenants ever seen.
        self._inflight: dict[str, asyncio.Future[list[InferenceRouterModel]]] = {}

    async def fetch_models(self, tenant_id: str | None = None) -> list[InferenceRouterModel]:
        """Fetch the models inference-router offers ``tenant_id`` (``[]`` on failure)."""
        return await self._fetch_models_or_none(tenant_id) or []

    async def _fetch_models_or_none(
        self, tenant_id: str | None = None
    ) -> list[InferenceRouterModel] | None:
        """Fetch the models inference-router offers ``tenant_id``; None on failure.

        Sends ``X-Tenant-Id`` (and ``X-License-Id`` when the request has one) so
        inference-router's per-tenant gate filters the list, plus the tenant's
        bundle token as ``Authorization`` when ``WM_IR_BUNDLE_TOKENS`` is on. ⛔ No
        tenant ⇒ ``IRTenantRequiredError`` (raised by ``ir_listing_headers``, not
        swallowed into ``None``): WM never lists on its own behalf, there is no
        service/platform tenant fallback. ``X-Worker-Name`` is always
        ``worker-management``. ``None`` (not ``[]``) distinguishes a failed
        fetch from a genuinely empty list, so a failure is never cached as the
        tenant's model list for the full TTL.
        """
        from src.services.ir_bundle_tokens import ir_listing_headers

        base_url = self.get_base_url()
        models_url = f"{base_url}/api/models"
        # Same per-tenant bundle token as the completion calls (WM-9): once IR
        # introspection is enforced, an untokened listing is a 401. The shared
        # builder owns flag-off/proof/exact-origin/fail-soft and REFUSES a
        # tenantless listing; flag off ⇒ no Authorization.
        headers = await ir_listing_headers(models_url, tenant_id)
        if tenant_id:
            from src.api.identity_context import get_license_id_from_context

            license_id = get_license_id_from_context()
            if license_id:
                headers["X-License-Id"] = license_id
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    raw_models: list[dict[str, Any]] = data.get("data", [])
                    models = [
                        InferenceRouterModel(
                            id=m["id"],
                            provider_chain=m.get("provider_chain") or [],
                            has_grammar=m.get("has_grammar", False),
                            has_json_schema=m.get("has_json_schema", False),
                        )
                        for m in raw_models
                    ]
                    logger.info(f"Fetched {len(models)} models from inference-router")
                    return models
                else:
                    logger.warning(
                        f"Failed to fetch models from inference-router: HTTP {response.status_code}"
                    )
        except httpx.TimeoutException:
            logger.warning("Timeout fetching models from inference-router")
        except httpx.ConnectError:
            logger.warning(f"Connection error to inference-router at {base_url}")
        except Exception:
            logger.exception("Error fetching models from inference-router")

        return None

    async def get_all_models(self, tenant_id: str | None = None) -> list[InferenceRouterModel]:
        """Get the models available to ``tenant_id``, using its cache slot if valid."""
        key = _cache_key(tenant_id)
        if key in self._model_cache:
            return self._model_cache[key]
        if key in self._failure_cache:
            return []
        return await self.refresh_model_cache(tenant_id)

    async def get_model(
        self, model_name: str, tenant_id: str | None = None
    ) -> InferenceRouterModel | None:
        """Get a specific model by schema_id."""
        models = await self.get_all_models(tenant_id)
        for m in models:
            if m.id == model_name:
                return m
        return None

    async def refresh_model_cache(self, tenant_id: str | None = None) -> list[InferenceRouterModel]:
        """Force refresh ``tenant_id``'s cache slot (single-flight per tenant).

        A concurrent refresh for the SAME tenant joins the in-flight fetch; no
        lock is held across the HTTP call, so other tenants are never blocked.
        A failed fetch returns ``[]`` and is remembered only for
        ``_FAILURE_CACHE_TTL_S``; the model cache keeps no entry for it.
        """
        key = _cache_key(tenant_id)
        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[InferenceRouterModel]] = loop.create_future()
        self._inflight[key] = future
        try:
            fetched = await self._fetch_models_or_none(tenant_id)
            if fetched is None:
                self._failure_cache[key] = True
                self._model_cache.pop(key, None)
                models: list[InferenceRouterModel] = []
            else:
                self._failure_cache.pop(key, None)
                self._model_cache[key] = fetched
                models = fetched
            logger.info(
                "Inference-router model cache refreshed: %d models (tenant=%s, ok=%s)",
                len(models),
                tenant_id or "<none>",
                fetched is not None,
            )
            future.set_result(models)
            return models
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
                # Mark retrieved so a waiter-less failure never logs
                # "Future exception was never retrieved".
                future.exception()
            raise
        finally:
            self._inflight.pop(key, None)

    async def get_models_for_api(self, tenant_id: str | None = None) -> dict[str, Any]:
        """Format ``tenant_id``'s models for API response."""
        models = await self.get_all_models(tenant_id)
        return {
            "object": "list",
            "data": [m.to_dict() for m in models],
            "base_url": self.get_base_url(),
            "cache_ttl_seconds": _settings_inference_router_cache_ttl(),
        }

    def get_base_url(self) -> str:
        """Return the inference-router URL (resolved from Settings each call)."""
        return _settings_inference_router_url()


# Global singleton
inference_router_registry = InferenceRouterRegistry()
