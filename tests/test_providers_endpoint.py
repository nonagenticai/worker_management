"""Tests for GET /providers and GET /providers/{name}/models."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

# Environment must be set before importing app
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pytest
from fastapi.testclient import TestClient

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from src.worker.provider_registry import (
    EXTERNAL_PROVIDERS,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_NAMES,
)
from tests._platform_tenant import PLATFORM_TENANT, caller_tenant
from tests.test_pydantic_ai_agent import _external_llm_allowed


async def _fake_tenant_id(request=None):
    return "test-tenant"


@pytest.fixture
def client():
    """TestClient with auth overridden and pg pool init skipped."""
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    # These tests pin the PLATFORM-tenant behaviour (unchanged by the 2026-10-01
    # ruling); the customer rules live in tests/test_tenant_scoped_providers.py.
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        caller_tenant(PLATFORM_TENANT),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


class TestGetProviders:
    def test_returns_all_enum_derived_providers(self, client):
        resp = client.get("/providers")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names = [row["name"] for row in body["data"]]
        expected = {
            "openai",
            "deepseek",
            "groq",
            "cohere",
            "mlx",
            "inference_router",
            "anthropic",
            "google",
            "vllm",
            "vlm_local",
            "xai",
        }
        assert set(names) == expected

    def test_each_row_has_required_fields(self, client):
        resp = client.get("/providers")
        body = resp.json()
        for row in body["data"]:
            assert {"name", "label", "description", "requires_api_key"} <= row.keys(), row

    def test_anthropic_requires_api_key(self, client):
        resp = client.get("/providers")
        anthropic = next(r for r in resp.json()["data"] if r["name"] == "anthropic")
        assert anthropic["requires_api_key"] is True

    def test_vllm_does_not_require_api_key(self, client):
        resp = client.get("/providers")
        vllm = next(r for r in resp.json()["data"] if r["name"] == "vllm")
        assert vllm["requires_api_key"] is False


class TestGetProviderModels:
    def test_unknown_provider_returns_404(self, client):
        resp = client.get("/providers/not-a-real-provider/models")
        assert resp.status_code == 404

    def test_vllm_proxies_to_v1_models(self, client, monkeypatch):
        monkeypatch.setenv("VLLM_URL", "http://test-vllm:8000")
        payload = {"data": [{"id": "local-llm"}, {"id": "another-model"}]}
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value=payload),
        ) as gmock:
            resp = client.get("/providers/vllm/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "vllm"
        assert [m["id"] for m in body["data"]] == ["local-llm", "another-model"]
        called_url = gmock.call_args.args[0]
        assert called_url.endswith("/v1/models")

    def test_anthropic_sends_xapikey_header(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        payload = {"data": [{"id": "claude-sonnet-4-6"}]}
        with (
            _external_llm_allowed(),
            patch(
                "src.api.providers._router._httpx_json_get",
                new=AsyncMock(return_value=payload),
            ) as gmock,
        ):
            resp = client.get("/providers/anthropic/models")
        assert resp.status_code == 200
        headers = gmock.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_deepseek_sends_bearer_auth(self, client, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
        payload = {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}
        with (
            _external_llm_allowed(),
            patch(
                "src.api.providers._router._httpx_json_get",
                new=AsyncMock(return_value=payload),
            ) as gmock,
        ):
            resp = client.get("/providers/deepseek/models")
        assert resp.status_code == 200
        headers = gmock.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-ds-test"

    def test_upstream_failure_returns_502(self, client, monkeypatch):
        monkeypatch.setenv("VLLM_URL", "http://test-vllm:8000")
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(side_effect=RuntimeError("upstream down")),
        ):
            resp = client.get("/providers/vllm/models")
        assert resp.status_code == 502
        body = resp.json()
        assert "Upstream unreachable" in body["error"]["message"]

    def test_missing_api_key_returns_502(self, client, monkeypatch):
        # Opted in: without it the local-only guard refuses at 403 BEFORE the key is
        # ever looked up, so this would assert the wrong refusal.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with _external_llm_allowed():
            resp = client.get("/providers/anthropic/models")
        assert resp.status_code == 502
        body = resp.json()
        assert "ANTHROPIC_API_KEY" in body["error"]["message"]

    def test_mlx_delegates_to_registry(self, client):
        payload = {"data": [{"id": "qwen-mini1"}, {"id": "phi-mini2"}]}
        with patch(
            "src.api.providers._router.mlx_model_registry.get_models_for_api",
            new=AsyncMock(return_value=payload),
        ):
            resp = client.get("/providers/mlx/models")
        assert resp.status_code == 200
        assert [m["id"] for m in resp.json()["data"]] == ["qwen-mini1", "phi-mini2"]

    def test_vlm_local_proxies_to_models(self, client, monkeypatch):
        monkeypatch.setenv("VLM_LOCAL_URL", "http://test-vlm:9203/api/vlm")
        payload = {"data": [{"id": "mlx-community/Qwen2-VL-7B-Instruct-4bit"}]}
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value=payload),
        ) as gmock:
            resp = client.get("/providers/vlm_local/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "vlm_local"
        assert [m["id"] for m in body["data"]] == ["mlx-community/Qwen2-VL-7B-Instruct-4bit"]
        called_url = gmock.call_args.args[0]
        assert called_url.endswith("/api/vlm/models")

    def test_vlm_local_url_override_wins(self, client, monkeypatch):
        monkeypatch.setenv("VLM_LOCAL_URL", "http://override-vlm:9999/api/vlm")
        payload = {"data": []}
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value=payload),
        ) as gmock:
            resp = client.get("/providers/vlm_local/models")
        assert resp.status_code == 200
        called_url = gmock.call_args.args[0]
        assert called_url == "http://override-vlm:9999/api/vlm/models"

    def test_inference_router_delegates_to_registry(self, client):
        payload = {"data": [{"id": "entity-spotter"}, {"id": "default"}]}
        with patch(
            "src.api.providers._router.inference_router_registry.get_models_for_api",
            new=AsyncMock(return_value=payload),
        ):
            resp = client.get("/providers/inference_router/models")
        assert resp.status_code == 200
        assert [m["id"] for m in resp.json()["data"]] == ["entity-spotter", "default"]

    def test_provider_fetchers_cover_every_enum_value(self):
        """Drift guard: adding a ModelProvider literal without wiring its fetcher
        would cause a silent KeyError at runtime. This test fails loudly instead."""
        import typing

        from src.api.providers._router import PROVIDER_FETCHERS, PROVIDER_NAMES
        from src.worker.agents.pydantic_ai_agent import ModelProvider

        enum_values = set(typing.get_args(ModelProvider))
        assert set(PROVIDER_NAMES) == enum_values
        assert set(PROVIDER_FETCHERS.keys()) == enum_values, (
            "PROVIDER_FETCHERS drift: every ModelProvider must have a fetcher"
        )

    def test_vllm_url_env_override(self, client, monkeypatch):
        monkeypatch.setenv("VLLM_URL", "http://custom:9000")
        payload = {"data": [{"id": "x"}]}
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value=payload),
        ) as gmock:
            resp = client.get("/providers/vllm/models")
        assert resp.status_code == 200
        called_url = gmock.call_args.args[0]
        assert called_url == "http://custom:9000/v1/models"


class TestProviderAvailability:
    """`available` must discriminate, in BOTH directions.

    The defect this guards: /providers advertised all 11 providers
    unconditionally, so 6 of them 502'd the moment a caller selected one and the
    frontend painted the raw upstream error (including the internal env-var
    name) at the end user.

    Each test below is written so that hardcoding `available` to a constant
    fails it — a flag that is always True is exactly the bug, and a flag that is
    always False would hide every working provider.
    """

    def test_keyless_providers_are_always_available(self, client, monkeypatch):
        # Strip every provider key from the environment. Keyless providers must
        # be unaffected -- their availability does not depend on a credential.
        for env_var in PROVIDER_API_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)

        rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
        keyless = [n for n in PROVIDER_NAMES if n not in PROVIDER_API_KEY_ENV_VARS]

        assert keyless, "fixture is vacuous: no keyless provider in the registry"
        for name in keyless:
            assert rows[name]["available"] is True, f"{name} is keyless, must stay available"

    def test_key_provider_is_unavailable_when_its_key_is_absent(self, client, monkeypatch):
        for env_var in PROVIDER_API_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)

        rows = {r["name"]: r for r in client.get("/providers").json()["data"]}

        assert PROVIDER_API_KEY_ENV_VARS, "fixture is vacuous: no key-requiring provider"
        for name in PROVIDER_API_KEY_ENV_VARS:
            assert rows[name]["available"] is False, (
                f"{name} has no key in env and must not be advertised as available"
            )

    def test_key_provider_becomes_available_when_its_key_is_set(self, client, monkeypatch):
        # The other direction. Without this, `available = False` for every
        # key-requiring provider would pass the test above.
        for env_var in PROVIDER_API_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)

        target = sorted(PROVIDER_API_KEY_ENV_VARS)[0]
        monkeypatch.setenv(PROVIDER_API_KEY_ENV_VARS[target], "sk-test-not-a-real-key")

        # The client runs as a PROVEN platform tenant, the one caller a registered
        # owner key is offered to (see TestCatalogAgreesWithTheOwnerKeyGate).
        rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
        assert rows[target]["available"] is True, f"{target} has a key set and must be available"

        others = [n for n in PROVIDER_API_KEY_ENV_VARS if n != target]
        assert others, "fixture is vacuous: only one key-requiring provider"
        for name in others:
            assert rows[name]["available"] is False, (
                f"setting {target}'s key must not make {name} available -- "
                "availability is per-provider, not 'any key is set'"
            )

    def test_an_empty_string_key_is_not_available(self, client, monkeypatch):
        # SOPS does not encrypt an empty string, and empty-not-absent is the
        # exact shape of the fleet's silent-failure class: a var that is present
        # but carries nothing must not read as configured.
        target = sorted(PROVIDER_API_KEY_ENV_VARS)[0]
        monkeypatch.setenv(PROVIDER_API_KEY_ENV_VARS[target], "")

        rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
        assert rows[target]["available"] is False

    def test_every_row_carries_the_field(self, client):
        rows = client.get("/providers").json()["data"]
        assert rows, "fixture is vacuous: /providers returned no rows"
        for row in rows:
            assert isinstance(row["available"], bool), f"{row['name']} missing available"


class TestCatalogAgreesWithTheOwnerKeyGate:
    """fr#1385: the catalog must not advertise what /models will refuse.

    Prod mounted ``XAI_API_KEY`` while the deployment refused external providers,
    so ``GET /providers`` said ``xai`` was ``available``, fr's /chat picker
    offered it (fr hides only ``available: false``), and
    ``GET /providers/xai/models`` answered 403. Since the owner ruling of
    2026-10-03 the gate is PER TENANT — a proven commercial-platform workspace
    only — and the catalog applies the very same predicate, so the two cannot
    disagree for any caller.
    """

    @pytest.mark.parametrize(
        "provider", sorted(set(EXTERNAL_PROVIDERS) & set(PROVIDER_API_KEY_ENV_VARS))
    )
    def test_external_provider_with_key_is_unavailable_to_an_unproven_platform_tenant(
        self, client, monkeypatch, provider
    ):
        monkeypatch.setenv(PROVIDER_API_KEY_ENV_VARS[provider], "sk-test-not-a-real-key")
        with caller_tenant(PLATFORM_TENANT, proven=False):
            rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
            refused = client.get(f"/providers/{provider}/models")
        # The row is still LISTED (a platform tenant sees the provider exists) ...
        assert provider in rows
        # ... but not offered, because /models refuses it.
        assert rows[provider]["available"] is False, (
            f"{provider} has a key but the tenant is not proven; advertising it "
            "available makes fr offer a provider that 403s"
        )
        assert refused.status_code == 403, refused.text
        assert refused.json()["error"]["code"] == "PROVIDER_NOT_ENABLED_FOR_TENANT"

    @pytest.mark.parametrize(
        "provider", sorted(set(EXTERNAL_PROVIDERS) & set(PROVIDER_API_KEY_ENV_VARS))
    )
    def test_external_provider_with_key_is_available_to_a_proven_platform_tenant(
        self, client, monkeypatch, provider
    ):
        """The other direction: a constant ``False`` would pass the test above."""
        monkeypatch.setenv(PROVIDER_API_KEY_ENV_VARS[provider], "sk-test-not-a-real-key")
        rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
        assert rows[provider]["available"] is True

    def test_keyless_local_providers_unaffected_by_the_gate(self, client):
        with caller_tenant(PLATFORM_TENANT, proven=False):
            rows = {r["name"]: r for r in client.get("/providers").json()["data"]}
        assert rows["inference_router"]["available"] is True
        assert rows["vlm_local"]["available"] is True


class TestUnreachableBackendNamesAReason:
    """fr#1385: vlm_local's 502 carried an EMPTY reason (bare httpx error)."""

    def test_empty_exception_text_is_replaced_by_its_type(self, client, monkeypatch):
        import httpx2 as httpx

        monkeypatch.setenv("VLM_LOCAL_URL", "http://vlm.test:9203/api/vlm")

        async def boom(*_a, **_kw):
            raise httpx.ConnectError("")

        with patch("src.api.providers._router._httpx_json_get", side_effect=boom):
            resp = client.get("/providers/vlm_local/models")
        assert resp.status_code == 502, resp.text
        assert "ConnectError" in resp.text, resp.text
        assert not resp.text.rstrip('"}').endswith("'vlm_local': "), resp.text


class TestGetProviderModelsOwnerKeyGate:
    """The model-discovery endpoint is the one external call that does NOT go
    through ModelFactory, so the owner-key gate has to be enforced here too."""

    @pytest.mark.parametrize("provider", sorted(EXTERNAL_PROVIDERS))
    def test_external_provider_models_refused_to_an_unproven_platform_tenant(
        self, client, provider
    ):
        """403 and NO outbound call: refused before httpx is reached, not after."""
        with (
            caller_tenant(PLATFORM_TENANT, proven=False),
            patch(
                "src.api.providers._router._httpx_json_get",
                new=AsyncMock(return_value={"data": []}),
            ) as gmock,
        ):
            resp = client.get(f"/providers/{provider}/models")
        assert resp.status_code == 403
        assert gmock.await_count == 0, "refusal must precede the vendor call, not follow it"
        assert resp.json()["error"]["code"] == "PROVIDER_NOT_ENABLED_FOR_TENANT"
        assert provider in resp.json()["error"]["message"]

    @pytest.mark.parametrize("provider", ["vllm", "vlm_local"])
    def test_local_providers_are_untouched_by_the_guard(self, client, monkeypatch, provider):
        """Negative control: the guard must not be a blanket refusal."""
        monkeypatch.setenv("VLLM_URL", "http://test-vllm:8000")
        monkeypatch.setenv("VLM_LOCAL_URL", "http://test-vlm:8000")
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value={"data": [{"id": "local-llm"}]}),
        ):
            resp = client.get(f"/providers/{provider}/models")
        assert resp.status_code == 200
