"""Tests for Inference Router Registry — model discovery and caching."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Provide a dummy URL so the Settings-backed registry returns something
# matching the legacy test assertion ("inference-router" substring).
os.environ.setdefault("INFERENCE_ROUTER_URL", "http://test-inference-router:8150")

from src.services.inference_router_registry import InferenceRouterRegistry
from src.settings import reset_settings_cache

reset_settings_cache()

# WM never lists inference-router models on its own behalf: every listing is
# for a caller's END tenant (no tenant ⇒ IRTenantRequiredError, see
# TestFetchModelsBundleToken). The bundle-token flag is off by default here,
# so these exercise the transport/caching behaviour only.
_TENANT = "019edc1e-0000-7000-8000-0000000000c8"


@pytest.fixture
def registry():
    """Fresh InferenceRouterRegistry instance for each test."""
    return InferenceRouterRegistry()


@pytest.fixture
def sample_models_response():
    """Sample /api/models response from inference-router."""
    return {
        "object": "list",
        "data": [
            {
                "id": "default",
                "object": "model",
                "owned_by": "inference-router",
                "provider_chain": ["pgx-local", "vllm", "deepseek", "claude"],
                "has_grammar": False,
                "has_json_schema": False,
            },
            {
                "id": "ks-entity-spotter-agent",
                "object": "model",
                "owned_by": "inference-router",
                "provider_chain": ["pgx-local", "vllm", "deepseek", "claude"],
                "has_grammar": True,
                "has_json_schema": True,
            },
        ],
    }


class TestFetchModels:
    @pytest.mark.asyncio
    async def test_fetch_models_success(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            models = await registry.fetch_models(_TENANT)

            assert len(models) == 2
            assert models[0].id == "default"
            assert models[1].id == "ks-entity-spotter-agent"
            assert models[1].has_grammar is True
            assert models[1].has_json_schema is True

    @pytest.mark.asyncio
    async def test_fetch_models_timeout(self, registry):
        import httpx

        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            models = await registry.fetch_models(_TENANT)
            assert models == []

    @pytest.mark.asyncio
    async def test_fetch_models_connection_error(self, registry):
        import httpx

        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            models = await registry.fetch_models(_TENANT)
            assert models == []

    @pytest.mark.asyncio
    async def test_fetch_models_http_error(self, registry):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            models = await registry.fetch_models(_TENANT)
            assert models == []


class TestCacheAndRefresh:
    @pytest.mark.asyncio
    async def test_get_all_models_populates_cache(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            models = await registry.get_all_models(_TENANT)
            assert len(models) == 2

            # Second call should use cache (no new HTTP call)
            models2 = await registry.get_all_models(_TENANT)
            assert len(models2) == 2
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_refresh_forces_new_fetch(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await registry.get_all_models(_TENANT)
            await registry.refresh_model_cache(_TENANT)
            assert mock_client.get.call_count == 2


class TestModelLookup:
    @pytest.mark.asyncio
    async def test_get_model_found(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            model = await registry.get_model("ks-entity-spotter-agent", _TENANT)
            assert model is not None
            assert model.id == "ks-entity-spotter-agent"
            assert model.has_grammar is True

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            model = await registry.get_model("nonexistent-schema", _TENANT)
            assert model is None


class TestApiResponse:
    @pytest.mark.asyncio
    async def test_get_models_for_api(self, registry, sample_models_response):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_models_response
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await registry.get_models_for_api(_TENANT)
            assert result["object"] == "list"
            assert len(result["data"]) == 2
            assert "base_url" in result
            assert "cache_ttl_seconds" in result


class TestBaseUrl:
    def test_get_base_url_returns_env_default(self, registry):
        url = registry.get_base_url()
        assert "inference-router" in url


_WM9_IR_URL = "http://test-inference-router:8150"
_WM9_TENANT = "019edc1e-0000-7000-8000-0000000000c9"
_WM9_TOKEN = "lt_wm9-tenant-bundle-token"  # pragma: allowlist secret


class TestFetchModelsBundleToken:
    """WM-9: the /api/models listing carries the tenant's IR bundle token.

    Once IR introspection is enforced an untokened listing is a 401, so the
    listing must take the same per-tenant token as the completion calls —
    and, like them, only on the exact INFERENCE_ROUTER_URL origin.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        from src.services import ir_bundle_tokens

        monkeypatch.setenv("INFERENCE_ROUTER_URL", _WM9_IR_URL)
        self.minted: list[str] = []

        async def _fake_token(tenant_id: str) -> str:
            self.minted.append(tenant_id)
            return _WM9_TOKEN

        monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _fake_token)
        # Pin the licence ContextVar: an earlier test in the same process can
        # leave one set, which would make the exact-headers assertion order-
        # dependent (measured in the full suite: X-License-Id 'lic-x' leaked).
        monkeypatch.setattr("src.api.identity_context.get_license_id_from_context", lambda: None)
        # The request's tenant, PROVEN by a licence: a bundle token is minted
        # only for a proven tenant (tests/test_ir_bundle_mint_proven_tenant.py).
        from src.api.identity_context import set_identity_context

        set_identity_context(_WM9_TENANT, "lic-1", "sub-1", tenant_provenance="license")
        reset_settings_cache()
        yield
        set_identity_context(None, None, None)
        reset_settings_cache()

    @staticmethod
    async def _sent_headers(registry, tenant_id, base_url=None):
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "local-llm"}]}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            if base_url is not None:
                registry.get_base_url = lambda: base_url
            models = await registry.fetch_models(tenant_id)
            assert [m.id for m in models] == ["local-llm"]
            return mock_client.get.call_args.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_flag_on_listing_carries_tenant_bearer(self, registry, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        reset_settings_cache()
        headers = await self._sent_headers(registry, _WM9_TENANT)
        assert headers["Authorization"] == f"Bearer {_WM9_TOKEN}"
        assert headers["X-Tenant-Id"] == _WM9_TENANT
        assert self.minted == [_WM9_TENANT]

    @pytest.mark.asyncio
    async def test_flag_off_listing_has_no_authorization(self, registry, monkeypatch):
        monkeypatch.delenv("WM_IR_BUNDLE_TOKENS", raising=False)
        reset_settings_cache()
        headers = await self._sent_headers(registry, _WM9_TENANT)
        assert headers == {
            "X-Tenant-Id": _WM9_TENANT,
            "X-Worker-Name": "worker-management",
            "X-Job-Id": "worker-management",
            "X-App-Id": "workers",
        }
        assert self.minted == []

    @pytest.mark.asyncio
    async def test_token_never_sent_to_a_non_ir_origin(self, registry, monkeypatch):
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        reset_settings_cache()
        for other in (
            "http://evil.example:8150",
            "https://test-inference-router:8150",
            "http://test-inference-router:9999",
        ):
            headers = await self._sent_headers(registry, _WM9_TENANT, base_url=other)
            assert "Authorization" not in headers, other
        assert self.minted == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tenant", [None, "", "00000000-0000-0000-0000-000000000000"])
    async def test_tenantless_listing_is_refused_and_never_sent(
        self, registry, monkeypatch, tenant
    ):
        """⛔ No end tenant ⇒ IRTenantRequiredError, counted, and NO GET is made.

        There is no WM/platform service tenant to list as (the retired
        WM_IR_SERVICE_TENANT_ID fallback). Holds with the flag on or off.
        """
        from src.services.ir_bundle_tokens import (
            TENANTLESS_IR_CALL_REFUSED_TOTAL,
            IRTenantRequiredError,
        )

        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        reset_settings_cache()
        counter = TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(
            call="listing", worker="worker-management"
        )
        before = counter._value.get()
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(IRTenantRequiredError):
                await registry.fetch_models(tenant)
            mock_client.get.assert_not_called()
        assert counter._value.get() == before + 1
        assert self.minted == []

    @pytest.mark.asyncio
    async def test_unproven_platform_tenant_listing_is_refused(self, registry, monkeypatch):
        """A tenant this request did not prove (here: not the request tenant) ⇒
        IRTenantNotProvenError, no token minted and no GET made."""
        from src.api.tenant_constants import commercial_platform_tenant_ids
        from src.services.ir_bundle_tokens import IRTenantNotProvenError

        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        reset_settings_cache()
        platform = sorted(commercial_platform_tenant_ids())[0]
        with patch("src.services.inference_router_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(IRTenantNotProvenError):
                await registry.fetch_models(platform)
            mock_client.get.assert_not_called()
        assert self.minted == []

    @pytest.mark.asyncio
    async def test_token_error_never_fails_the_listing(self, registry, monkeypatch):
        from src.services import ir_bundle_tokens

        async def _boom(tenant_id: str):
            raise RuntimeError("boom")

        monkeypatch.setattr(ir_bundle_tokens, "tenant_bearer_token", _boom)
        monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "true")
        reset_settings_cache()
        headers = await self._sent_headers(registry, _WM9_TENANT)
        assert "Authorization" not in headers
