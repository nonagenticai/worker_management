"""MLX Model Registry Service for dynamic model discovery and routing.

DEPRECATED 2026-05-08: The original mlx-inference-mini1 / mlx-inference
hostnames were retired. They have been replaced by the inference-router
service on PGX. Each ``MLX_INSTANCE_{RING,MINI1,MINI2}_URL`` env var
should be set on the Deployment (typically all three to the same
inference-router URL until/unless someone wires up distinct upstreams).
The ``mlx`` provider abstraction (used as a label in graph_execution rows,
providers API responses, and worker configs) keeps working through this
indirection.

This service discovers models from the configured inference instance(s)
and provides intelligent routing based on model availability.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx
from cachetools import TTLCache

from src.settings import get_settings

logger = logging.getLogger(__name__)


def _settings_mlx_cache_ttl() -> int:
    return get_settings().mlx_model_cache_ttl


def _build_mlx_instances() -> dict[str, str]:
    """Build the {label: url} routing dict from Settings each call.

    Empty values are filtered so that an unconfigured environment yields an
    empty dict rather than silently routing to a placeholder.
    Order is preserved (``ring`` first = preferred).
    """
    s = get_settings()
    candidates: list[tuple[str, str]] = [
        ("ring", s.mlx_instance_ring_url),
        ("mini1", s.mlx_instance_mini1_url),
        ("mini2", s.mlx_instance_mini2_url),
    ]
    return {name: url for name, url in candidates if url}


@dataclass
class MLXModel:
    """Represents an MLX model available on one or more instances."""

    id: str  # e.g., "mlx-community/Qwen3-1.7B-8bit"
    instances: list[str] = field(default_factory=list)  # ["mini1", "mini2"]
    instance_urls: dict[str, str] = field(default_factory=dict)  # {"mini1": "https://..."}
    aliases: list[str] = field(default_factory=list)  # ["qwen3-1.7b"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "instances": self.instances,
            "instance_urls": self.instance_urls,
            "aliases": self.aliases,
        }


class MLXModelRegistry:
    """Registry for MLX models across multiple inference instances.

    Provides:
    - Dynamic model discovery from /model/info endpoint
    - TTL-based caching (default 5 minutes)
    - Round-robin routing for models on multiple instances
    - Automatic fallback to available instances
    """

    def __init__(self) -> None:
        self._model_cache: TTLCache[str, dict[str, MLXModel]] = TTLCache(
            maxsize=1, ttl=_settings_mlx_cache_ttl()
        )
        self._round_robin_state: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def fetch_models_from_instance(
        self, instance_name: str, base_url: str
    ) -> list[dict[str, Any]]:
        """Fetch models from a single MLX instance.

        Args:
            instance_name: Name of the instance (e.g., "mini1")
            base_url: Base URL of the instance

        Returns:
            List of model dictionaries from the /model/info endpoint
        """
        try:
            async with httpx.AsyncClient(timeout=10.0, verify=False) as client:  # nosec B501 — internal service, self-signed certs
                response = await client.get(f"{base_url}/model/info")
                if response.status_code == 200:
                    data = response.json()
                    models: list[dict[str, Any]] = data.get("models", [])
                    logger.info(f"Fetched {len(models)} models from MLX instance '{instance_name}'")
                    return models
                else:
                    logger.warning(
                        f"Failed to fetch models from {instance_name}: HTTP {response.status_code}"
                    )
        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching models from MLX instance '{instance_name}'")
        except httpx.ConnectError:
            logger.warning(f"Connection error to MLX instance '{instance_name}' at {base_url}")
        except Exception:
            logger.exception(f"Error fetching models from {instance_name}")

        return []

    async def refresh_model_cache(self) -> dict[str, MLXModel]:
        """Refresh the model cache by fetching from all instances concurrently.

        Returns:
            Dictionary mapping model IDs to MLXModel objects
        """
        async with self._lock:
            instances = _build_mlx_instances()
            # Fetch from all instances concurrently
            tasks = [self.fetch_models_from_instance(name, url) for name, url in instances.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate models
            models: dict[str, MLXModel] = {}

            for (instance_name, base_url), result in zip(instances.items(), results, strict=False):
                if isinstance(result, BaseException):
                    logger.warning(f"Error from instance {instance_name}: {result}")
                    continue

                # result is List[Dict[str, Any]] at this point
                model_list: list[dict[str, Any]] = result
                for model_data in model_list:
                    model_id = model_data.get("id", "")
                    if not model_id:
                        continue

                    if model_id not in models:
                        # Create new model entry
                        models[model_id] = MLXModel(
                            id=model_id,
                            instances=[instance_name],
                            instance_urls={instance_name: base_url},
                            aliases=self._generate_aliases(model_id),
                        )
                    else:
                        # Model exists on another instance too
                        models[model_id].instances.append(instance_name)
                        models[model_id].instance_urls[instance_name] = base_url

            self._model_cache["models"] = models
            logger.info(f"Model cache refreshed: {len(models)} unique models")
            return models

    def _generate_aliases(self, model_id: str) -> list[str]:
        """Generate user-friendly aliases for a model ID.

        Examples:
        - "mlx-community/Qwen3-1.7B-8bit" -> ["qwen3-1.7b", "qwen3-1.7b-8bit"]
        - "mlx-community/Llama-3.2-1B-Instruct-4bit" -> ["llama-3.2-1b", "llama-3.2-1b-instruct"]
        """
        aliases = []

        # Extract model name from path (after last /)
        model_name = model_id.rsplit("/", maxsplit=1)[-1].lower()
        aliases.append(model_name)

        # Create shortened version without quantization suffix
        for suffix in ["-8bit", "-4bit", "-3bit", "-2bit", "-6bit"]:
            if model_name.endswith(suffix):
                short_name = model_name[: -len(suffix)]
                aliases.append(short_name)
                break

        # Add version without "-instruct" suffix
        for suffix in ["-instruct", "-chat", "-base"]:
            base = model_name.replace(suffix, "")
            if base != model_name and base not in aliases:
                aliases.append(base)

        return list(set(aliases))

    async def get_all_models(self) -> dict[str, MLXModel]:
        """Get all available models, using cache if valid.

        Returns:
            Dictionary mapping model IDs to MLXModel objects
        """
        if "models" in self._model_cache:
            return self._model_cache["models"]
        return await self.refresh_model_cache()

    async def get_model(self, model_name: str) -> MLXModel | None:
        """Get a specific model by ID or alias.

        Args:
            model_name: Model ID or alias

        Returns:
            MLXModel if found, None otherwise
        """
        models = await self.get_all_models()

        # Direct lookup by ID
        if model_name in models:
            return models[model_name]

        # Search by alias
        model_name_lower = model_name.lower()
        for model in models.values():
            if model_name_lower in model.aliases or model_name_lower == model.id.lower():
                return model

        return None

    async def get_instance_for_model(self, model_name: str) -> str | None:
        """Get the best instance URL for a model using round-robin.

        If model is on multiple instances, alternates between them.
        If model is on one instance, returns that instance.

        Args:
            model_name: Model ID or alias

        Returns:
            Instance base URL if model found, None otherwise
        """
        model = await self.get_model(model_name)
        if not model or not model.instances:
            return None

        if len(model.instances) == 1:
            # Single instance, just return it
            instance = model.instances[0]
            return model.instance_urls.get(instance)

        # Round-robin for multiple instances
        async with self._lock:
            current_idx = self._round_robin_state.get(model.id, 0)
            instance = model.instances[current_idx % len(model.instances)]
            self._round_robin_state[model.id] = current_idx + 1

        logger.debug(
            f"Round-robin routing model '{model_name}' to instance '{instance}' (idx={current_idx})"
        )
        return model.instance_urls.get(instance)

    async def get_default_instance(self) -> str | None:
        """Get the URL of the first available MLX instance.

        Used as fallback when model is generic "mlx" (no specific model requested).
        Tries each configured instance and returns the first one that has models.

        Returns:
            Instance base URL if any instance is available, None otherwise
        """
        models = await self.get_all_models()
        if models:
            # Return the URL of the first instance that has any model
            first_model = next(iter(models.values()))
            if first_model.instances:
                instance = first_model.instances[0]
                url = first_model.instance_urls.get(instance)
                if url:
                    logger.info(f"Default MLX instance resolved to '{instance}': {url}")
                    return url

        # No models in cache, try instances directly
        for instance_name, instance_url in _build_mlx_instances().items():
            try:
                async with httpx.AsyncClient(timeout=5.0, verify=False) as client:  # nosec B501 — internal service, self-signed certs
                    response = await client.get(f"{instance_url}/model/info")
                    if response.status_code == 200:
                        logger.info(
                            f"Default MLX instance (direct probe): {instance_name} at {instance_url}"
                        )
                        return instance_url
            except (httpx.HTTPError, OSError) as exc:
                logger.debug(
                    "mlx_instance_probe_failed",
                    extra={"instance": instance_name, "error": str(exc)},
                )
                continue

        logger.warning("No MLX instances available")
        return None

    async def get_any_available_model(self) -> str | None:
        """Get the first available model name from any instance.

        Returns:
            Model ID string if any model is available, None otherwise
        """
        models = await self.get_all_models()
        if models:
            return next(iter(models.keys()))
        return None

    async def resolve_base_url(
        self, model_name: str | None, fallback_url: str | None = None
    ) -> str | None:
        """Resolve the base URL for a model, with optional fallback.

        This is the main entry point for worker_factory to get the correct
        base URL for a model.

        Args:
            model_name: Model ID or alias to resolve
            fallback_url: URL to use if model is not found in registry

        Returns:
            Base URL for the model's instance, or fallback if not found
        """
        if not model_name:
            return fallback_url

        # For generic "mlx" model name, resolve to default instance
        if model_name.lower() == "mlx":
            default_url = await self.get_default_instance()
            if default_url:
                return default_url
            return fallback_url

        instance_url = await self.get_instance_for_model(model_name)
        if instance_url:
            logger.info(f"Resolved model '{model_name}' to instance: {instance_url}")
            return instance_url

        logger.debug(f"Model '{model_name}' not found in registry, using fallback: {fallback_url}")
        return fallback_url

    async def get_models_for_api(self) -> dict[str, Any]:
        """Get models formatted for the API response.

        Returns:
            Dictionary with models and instance metadata for API response
        """
        models = await self.get_all_models()

        model_list = []
        for model_id, model in models.items():
            # Use round-robin to pick primary instance for display
            primary_instance = model.instances[0] if model.instances else None
            primary_url = model.instance_urls.get(primary_instance, "") if primary_instance else ""

            model_list.append(
                {
                    "id": model_id,
                    "instance_name": primary_instance,
                    "instance_url": primary_url,
                    "all_instances": model.instances,
                    "aliases": model.aliases,
                }
            )

        return {
            "object": "list",
            "data": model_list,
            "instances": _build_mlx_instances(),
            "cache_ttl_seconds": _settings_mlx_cache_ttl(),
        }


# Global singleton instance
mlx_model_registry = MLXModelRegistry()
