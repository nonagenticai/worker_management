"""Tests for MLX Model Registry — model discovery, routing, and caching."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Settings load with empty MLX_INSTANCE_*_URL by default in tests; provide
# dummy values so _build_mlx_instances() returns the expected 3-entry dict.
os.environ.setdefault("MLX_INSTANCE_MINI1_URL", "http://test-mlx-mini1:8150")
os.environ.setdefault("MLX_INSTANCE_MINI2_URL", "http://test-mlx-mini2:8150")
os.environ.setdefault("MLX_INSTANCE_RING_URL", "http://test-mlx-ring:8150")

from src.services.mlx_model_registry import (
    MLXModelRegistry,
    _build_mlx_instances,
)
from src.settings import reset_settings_cache

reset_settings_cache()


@pytest.fixture
def registry():
    """Fresh MLXModelRegistry instance for each test."""
    return MLXModelRegistry()


# Backwards-compat alias used by existing assertions.
MLX_INSTANCES = _build_mlx_instances()


@pytest.fixture
def sample_models():
    """Sample model data as returned by /model/info endpoint."""
    return [
        {"id": "mlx-community/Qwen3-1.7B-8bit", "object": "model"},
        {"id": "mlx-community/Llama-3.2-1B-Instruct-4bit", "object": "model"},
    ]


@pytest.fixture
def sample_models_mini2():
    """Different models on mini2."""
    return [
        {"id": "mlx-community/Qwen3-1.7B-8bit", "object": "model"},  # Same as mini1
        {"id": "mlx-community/Phi-4-mini-instruct-8bit", "object": "model"},
    ]


class TestAliasGeneration:
    def test_basic_alias(self, registry):
        aliases = registry._generate_aliases("mlx-community/Qwen3-1.7B-8bit")
        assert "qwen3-1.7b-8bit" in aliases
        assert "qwen3-1.7b" in aliases

    def test_instruct_alias(self, registry):
        aliases = registry._generate_aliases("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert "llama-3.2-1b-instruct-4bit" in aliases
        assert "llama-3.2-1b-4bit" in aliases or "llama-3.2-1b" in aliases

    def test_no_quantization_suffix(self, registry):
        aliases = registry._generate_aliases("mlx-community/SomeModel")
        assert "somemodel" in aliases

    def test_simple_model_name(self, registry):
        aliases = registry._generate_aliases("my-model")
        assert "my-model" in aliases


class TestModelDiscovery:
    @pytest.mark.asyncio
    async def test_fetch_models_success(self, registry, sample_models):
        with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": sample_models}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await registry.fetch_models_from_instance("mini1", "https://mini1.example.com")
            assert len(result) == 2
            assert result[0]["id"] == "mlx-community/Qwen3-1.7B-8bit"

    @pytest.mark.asyncio
    async def test_fetch_models_timeout(self, registry):
        import httpx

        with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await registry.fetch_models_from_instance("mini1", "https://mini1.example.com")
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_models_connection_error(self, registry):
        import httpx

        with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await registry.fetch_models_from_instance("mini1", "https://mini1.example.com")
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_models_http_error(self, registry):
        with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await registry.fetch_models_from_instance("mini1", "https://mini1.example.com")
            assert result == []


class TestCacheAndRefresh:
    @pytest.mark.asyncio
    async def test_refresh_populates_cache(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance") as mock_fetch:
            mock_fetch.return_value = sample_models

            models = await registry.refresh_model_cache()
            assert "mlx-community/Qwen3-1.7B-8bit" in models
            assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in models
            assert len(models) == 2

    @pytest.mark.asyncio
    async def test_get_all_models_uses_cache(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance") as mock_fetch:
            mock_fetch.return_value = sample_models

            # First call populates cache
            models1 = await registry.get_all_models()
            # Second call should use cache (no additional fetch)
            models2 = await registry.get_all_models()

            assert models1 == models2
            # fetch_models_from_instance is called once per instance during refresh
            # Second get_all_models call should use cache (no additional fetches)
            assert mock_fetch.call_count == len(MLX_INSTANCES)

    @pytest.mark.asyncio
    async def test_multi_instance_aggregation(self, registry, sample_models, sample_models_mini2):
        call_count = 0

        async def mock_fetch(instance_name, base_url):
            nonlocal call_count
            call_count += 1
            if instance_name == "mini1":
                return sample_models
            elif instance_name == "mini2":
                return sample_models_mini2
            # ring and any other instances return empty
            return []

        with patch.object(registry, "fetch_models_from_instance", side_effect=mock_fetch):
            models = await registry.refresh_model_cache()

            # Qwen3 should be on both mini1 and mini2
            qwen = models["mlx-community/Qwen3-1.7B-8bit"]
            assert "mini1" in qwen.instances
            assert "mini2" in qwen.instances

            # Phi should only be on mini2
            phi = models["mlx-community/Phi-4-mini-instruct-8bit"]
            assert phi.instances == ["mini2"]

            # Llama should only be on mini1
            llama = models["mlx-community/Llama-3.2-1B-Instruct-4bit"]
            assert llama.instances == ["mini1"]


class TestModelLookup:
    @pytest.mark.asyncio
    async def test_get_model_by_id(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            model = await registry.get_model("mlx-community/Qwen3-1.7B-8bit")
            assert model is not None
            assert model.id == "mlx-community/Qwen3-1.7B-8bit"

    @pytest.mark.asyncio
    async def test_get_model_by_alias(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            model = await registry.get_model("qwen3-1.7b")
            assert model is not None
            assert model.id == "mlx-community/Qwen3-1.7B-8bit"

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            model = await registry.get_model("nonexistent-model")
            assert model is None

    @pytest.mark.asyncio
    async def test_get_model_case_insensitive(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            model = await registry.get_model("MLX-COMMUNITY/QWEN3-1.7B-8BIT")
            assert model is not None


class TestRouting:
    @pytest.mark.asyncio
    async def test_single_instance_routing(self, registry):
        with patch.object(registry, "fetch_models_from_instance") as mock_fetch:
            # Only mini1 has the model
            async def side_effect(name, url):
                if name == "mini1":
                    return [{"id": "test-model"}]
                return []

            mock_fetch.side_effect = side_effect
            url = await registry.get_instance_for_model("test-model")
            assert url is not None

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, registry, sample_models):
        async def mock_fetch(name, url):
            # All instances have the model
            return [{"id": "mlx-community/Qwen3-1.7B-8bit"}]

        with patch.object(registry, "fetch_models_from_instance", side_effect=mock_fetch):
            await registry.refresh_model_cache()

            for _ in range(len(MLX_INSTANCES) * 2):
                url = await registry.get_instance_for_model("mlx-community/Qwen3-1.7B-8bit")
                assert url is not None

            # Round-robin counter must have advanced once per call across all
            # configured instances. URL set size is no longer asserted because
            # all MLX_INSTANCES default to the same inference-router URL after
            # the 2026-05-08 mlx-host retirement; what matters is that the
            # rotation logic ran for every instance entry.
            model_id = "mlx-community/Qwen3-1.7B-8bit"
            assert registry._round_robin_state[model_id] == len(MLX_INSTANCES) * 2

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, registry):
        with patch.object(registry, "fetch_models_from_instance", return_value=[]):
            url = await registry.get_instance_for_model("nonexistent")
            assert url is None


class TestDefaultInstance:
    @pytest.mark.asyncio
    async def test_get_default_instance_from_cache(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            await registry.refresh_model_cache()
            url = await registry.get_default_instance()
            assert url is not None

    @pytest.mark.asyncio
    async def test_get_default_instance_empty_cache(self, registry):
        with patch.object(registry, "fetch_models_from_instance", return_value=[]):
            with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_cls.return_value = mock_client

                url = await registry.get_default_instance()
                assert url is not None

    @pytest.mark.asyncio
    async def test_get_default_instance_all_down(self, registry):
        import httpx

        with patch.object(registry, "fetch_models_from_instance", return_value=[]):
            with patch("src.services.mlx_model_registry.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.ConnectError("down"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_cls.return_value = mock_client

                url = await registry.get_default_instance()
                assert url is None


class TestResolveBaseUrl:
    @pytest.mark.asyncio
    async def test_resolve_specific_model(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            url = await registry.resolve_base_url("qwen3-1.7b")
            assert url is not None

    @pytest.mark.asyncio
    async def test_resolve_generic_mlx(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            await registry.refresh_model_cache()
            url = await registry.resolve_base_url("mlx")
            assert url is not None  # Should resolve to default instance

    @pytest.mark.asyncio
    async def test_resolve_empty_name(self, registry):
        url = await registry.resolve_base_url("", fallback_url="http://fallback")
        assert url == "http://fallback"

    @pytest.mark.asyncio
    async def test_resolve_unknown_model_uses_fallback(self, registry):
        with patch.object(registry, "fetch_models_from_instance", return_value=[]):
            url = await registry.resolve_base_url("unknown-model", fallback_url="http://fallback")
            assert url == "http://fallback"

    @pytest.mark.asyncio
    async def test_resolve_none_name(self, registry):
        url = await registry.resolve_base_url(None, fallback_url="http://fallback")
        assert url == "http://fallback"


class TestGetAnyAvailableModel:
    @pytest.mark.asyncio
    async def test_returns_first_model(self, registry, sample_models):
        with patch.object(registry, "fetch_models_from_instance", return_value=sample_models):
            model_name = await registry.get_any_available_model()
            assert model_name is not None
            assert "mlx-community" in model_name

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self, registry):
        with patch.object(registry, "fetch_models_from_instance", return_value=[]):
            model_name = await registry.get_any_available_model()
            assert model_name is None
