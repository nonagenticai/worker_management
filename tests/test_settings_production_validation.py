"""Settings refuse to start in production with empty infra placeholders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from src.settings import Settings, get_settings, reset_settings_cache

if TYPE_CHECKING:
    from collections.abc import Iterator


def _mk(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "database_url": "postgresql+psycopg://u:p@h:5432/db",  # pragma: allowlist secret
        "license_introspect_url": "http://lic:8544/license/introspect",
        "minio_endpoint": "minio:9000",
        "inference_router_url": "http://ir:8150",
        "mlx_instance_mini1_url": "http://mlx-mini1:8150",
        "mlx_instance_mini2_url": "http://mlx-mini2:8150",
        "mlx_instance_ring_url": "http://mlx-ring:8150",
        "vllm_url": "http://vllm:8000",
        "vlm_local_url": "http://vlm:9203/api/vlm",
        "wm_adapter_manager_url": "http://adapter:8001",
        "coding_service_url": "http://coding-gateway:8687",
        "agent_host_mini1_url": "http://agent-host-1:9100",
        "agent_host_mini2_url": "http://agent-host-2:9100",
        "otlp_metrics_endpoint": "http://otel:4318/v1/metrics",
        "environment": "production",
    }
    base.update(overrides)
    # Use model_validate — dict-kwarg unpack is rejected by ty CI.
    return Settings.model_validate(base)


@pytest.fixture(autouse=True)
def _clear_env_and_cache() -> Iterator[None]:
    """Tests mutate env vars; keep them isolated and clear the settings cache."""
    saved = {
        k: os.environ.get(k)
        for k in (
            "DATABASE_URL",
            "LICENSE_INTROSPECT_URL",
            "MINIO_ENDPOINT",
            "ENVIRONMENT",
            "LOG_LEVEL",
            "OTEL_SERVICE_NAME",
            "INFERENCE_ROUTER_URL",
            "MLX_INSTANCE_MINI1_URL",
            "MLX_INSTANCE_MINI2_URL",
            "MLX_INSTANCE_RING_URL",
            "VLLM_URL",
            "VLM_LOCAL_URL",
            "WM_ADAPTER_MANAGER_URL",
            "CODING_SERVICE_URL",
            "AGENT_HOST_MINI1_URL",
            "AGENT_HOST_MINI2_URL",
            "OTLP_METRICS_ENDPOINT",
        )
    }
    for k in saved:
        os.environ.pop(k, None)
    reset_settings_cache()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_settings_cache()


class TestProductionValidation:
    def test_complete_config_allows_init(self) -> None:
        settings = _mk()
        assert settings.environment == "production"
        assert settings.database_url

    @pytest.mark.parametrize(
        "empty_field",
        [
            "database_url",
            "license_introspect_url",
            "minio_endpoint",
        ],
    )
    def test_empty_required_raises(self, empty_field: str) -> None:
        with pytest.raises(RuntimeError, match=empty_field):
            _mk(**{empty_field: ""})

    def test_test_environment_allows_empty(self) -> None:
        Settings.model_validate({"environment": "test"})  # must not raise

    def test_development_default_environment_allows_empty(self) -> None:
        Settings.model_validate({})  # default environment == development


class TestEnvVarBinding:
    def test_unprefixed_env_vars_bind(self) -> None:
        os.environ["DATABASE_URL"] = (
            "postgresql+psycopg://u:p@h:5432/db"  # pragma: allowlist secret
        )
        os.environ["LICENSE_INTROSPECT_URL"] = "http://lic:8544/license/introspect"
        os.environ["MINIO_ENDPOINT"] = "minio:9000"
        os.environ["INFERENCE_ROUTER_URL"] = "http://ir:8150"
        os.environ["MLX_INSTANCE_MINI1_URL"] = "http://mlx-mini1:8150"
        os.environ["MLX_INSTANCE_MINI2_URL"] = "http://mlx-mini2:8150"
        os.environ["MLX_INSTANCE_RING_URL"] = "http://mlx-ring:8150"
        os.environ["VLLM_URL"] = "http://vllm:8000"
        os.environ["VLM_LOCAL_URL"] = "http://vlm:9203/api/vlm"
        os.environ["WM_ADAPTER_MANAGER_URL"] = "http://adapter:8001"
        os.environ["CODING_SERVICE_URL"] = "http://coding-gateway:8687"
        os.environ["AGENT_HOST_MINI1_URL"] = "http://agent-host-1:9100"
        os.environ["AGENT_HOST_MINI2_URL"] = "http://agent-host-2:9100"
        os.environ["OTLP_METRICS_ENDPOINT"] = "http://otel:4318/v1/metrics"
        os.environ["ENVIRONMENT"] = "production"
        os.environ["OTEL_SERVICE_NAME"] = "worker-management-service"
        reset_settings_cache()
        settings = get_settings()
        assert settings.database_url.startswith("postgresql+psycopg://")
        assert settings.license_introspect_url.endswith("/license/introspect")
        assert settings.minio_endpoint == "minio:9000"
        assert settings.environment == "production"
        assert settings.otel_service_name == "worker-management-service"
        assert settings.inference_router_url == "http://ir:8150"
        assert settings.vllm_url == "http://vllm:8000"

    def test_pool_defaults(self) -> None:
        settings = Settings.model_validate({"environment": "test"})
        assert settings.postgres_pool_min == 10
        assert settings.postgres_pool_max == 20
        assert settings.postgres_connect_timeout == 10


class TestAdapterManagerOptionalAndNoDirectGpu:
    """WM_ADAPTER_MANAGER_URL is optional; a direct-vLLM URL fails boot."""

    def test_production_boots_with_adapter_manager_unset(self) -> None:
        settings = _mk(wm_adapter_manager_url="")
        assert settings.environment == "production"
        assert settings.wm_adapter_manager_url == ""

    def test_production_boots_with_adapter_manager_env_absent(self) -> None:
        for k, v in {
            "DATABASE_URL": "postgresql+psycopg://u:p@h:5432/db",  # pragma: allowlist secret
            "LICENSE_INTROSPECT_URL": "http://lic:8544/license/introspect",
            "MINIO_ENDPOINT": "minio:9000",
            "INFERENCE_ROUTER_URL": "http://ir:8150",
            "MLX_INSTANCE_MINI1_URL": "http://mlx-mini1:8150",
            "MLX_INSTANCE_MINI2_URL": "http://mlx-mini2:8150",
            "MLX_INSTANCE_RING_URL": "http://mlx-ring:8150",
            "VLLM_URL": "http://vllm:8000",
            "VLM_LOCAL_URL": "http://vlm:9203/api/vlm",
            "CODING_SERVICE_URL": "http://coding-gateway:8687",
            "AGENT_HOST_MINI1_URL": "http://agent-host-1:9100",
            "AGENT_HOST_MINI2_URL": "http://agent-host-2:9100",
            "OTLP_METRICS_ENDPOINT": "http://otel:4318/v1/metrics",
            "ENVIRONMENT": "production",
        }.items():
            os.environ[k] = v
        reset_settings_cache()
        assert get_settings().wm_adapter_manager_url == ""

    @pytest.mark.parametrize("environment", ["production", "development", "test"])
    @pytest.mark.parametrize(
        "url",
        [
            "http://vllm.example.com:8001",
            "https://VLLM.example.com/",
            "vllm.example.com:8001",
        ],
    )
    def test_direct_vllm_url_fails_boot(self, environment: str, url: str) -> None:
        with pytest.raises(RuntimeError, match="inference-router") as exc:
            _mk(wm_adapter_manager_url=url, environment=environment)
        assert "vllm.example.com" in str(exc.value)

    def test_non_gpu_adapter_manager_url_allowed(self) -> None:
        settings = _mk(wm_adapter_manager_url="http://adapter-manager.example.com:8001")
        assert settings.wm_adapter_manager_url.startswith("http://adapter-manager")
