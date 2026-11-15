"""Validator 3 (2026-10-01): D3, N1, N2, N3. (D4 is pinned in test_tenant_byo_fix2.)

Each class names the defect it pins and the mutation that turns it red.

- D3  a tenant BYO endpoint may never be owner infrastructure — not at PUT, not at
      resolve, not on the worker path (enforce_customer_routing), not on a
      redirect hop. Repro: ``{provider: openai_compatible, model: grok-4-fast,
      baseUrl: <inference-router Service>/v1, apiKey: junk}``.
- N1  ``model_override`` leaves an MLX ``base_url_override`` on a customer IR config.
- N2  key-test cooldown single-flight.
- N3  key-test cooldown is per tenant, for keyless ``local`` rows.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx
import pytest
from fastapi.testclient import TestClient

from src.security import request_overrides as ro
from src.security.request_overrides import (
    TENANT_BASE_URL_NOT_ALLOWED,
    TenantBaseUrlNotAllowedError,
    allowed_tenant_base_url_origins,
    check_tenant_base_url,
    owner_infra_reason,
    refuse_owner_infra_request,
)
from src.security.secret_box import encrypt_secret
from src.settings import reset_settings_cache
from src.worker.factory import llm_config_resolver as r
from tests._platform_tenant import CUSTOMER_TENANT_A, CUSTOMER_TENANT_B, PLATFORM_TENANT
from tests._tenant_llm_audit_fake import AuditFakeMixin

IR = "http://inference-router.inference-router.svc.cluster.local:8150"
IR_V1 = IR + "/v1"
JUNK = "junk"  # the repro's apiKey; not a credential
VENDOR = "https://api.vendor-byo.example/v1"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.delenv("TENANT_BASE_URL_ALLOWLIST", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def tenant_allowlist(monkeypatch: pytest.MonkeyPatch):
    def _set(value: str) -> None:
        monkeypatch.setenv("TENANT_BASE_URL_ALLOWLIST", value)
        reset_settings_cache()

    return _set


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch):
    """Pin what each name resolves to; unknown names do not resolve."""
    table: dict[str, tuple[str, ...]] = {}
    monkeypatch.setattr(ro, "_resolve_host_ips", lambda host: table.get(host, ()))
    ro._dns_cache.clear()
    return table


class _OneRowDB(AuditFakeMixin):
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.upserts = 0

    async def get_config(self, tenant_id, role="__default__"):
        return self.rows.get((str(tenant_id), role))

    async def list_configs(self, tenant_id):
        return [dict(v) for (t, _), v in self.rows.items() if t == str(tenant_id)]

    async def upsert_config(self, *, tenant_id, role="__default__", **cols):
        self.take_audit(tenant_id, role, cols)
        self.upserts += 1
        self.rows[(str(tenant_id), role)] = {"tenant_id": str(tenant_id), "role": role, **cols}


def _ctx(tenant: str):
    from src.auth import TenantContext

    return TenantContext(
        tenant_id=tenant,
        subject_id=tenant,
        tenant_name="t",
        visibility="tenant",
        byo_mode="local_allowed",
    )


@pytest.fixture
def put_client(monkeypatch: pytest.MonkeyPatch):
    from src.api.tenant_config import _router as tc
    from src.api.worker_management_api import app
    from src.auth import get_tenant_context

    db = _OneRowDB()
    monkeypatch.setattr(tc, "_db", db)
    app.dependency_overrides[get_tenant_context] = lambda: _ctx(CUSTOMER_TENANT_A)
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c, db
    app.dependency_overrides.clear()


def _repro_body(base_url: str = IR_V1) -> dict[str, Any]:
    return {
        "provider": "openai_compatible",
        "model": "grok-4-fast",
        "baseUrl": base_url,
        "apiKey": JUNK,
    }


# Every owner-infra shape, each LISTED by the operator for the tenant so the
# refusal provably comes from the owner-infra rule, not from "not allowlisted".
OWNER_INFRA_URLS = [
    IR_V1,
    "http://inference-router:8150/v1",  # in-cluster short name
    "http://vllm.local:8000/v1",
    "http://mini1.local:9204/v1",
    "http://node.tail000000.ts.net:8000/v1",
    "http://100.100.100.100:9204/v1",  # a tailnet IP
    "http://100.100.100.101:8000/v1",  # a tailnet IP
    "http://10.43.12.34:8150/v1",  # ClusterIP
    "http://192.168.1.111:9204/v1",
    "http://127.0.0.1:8150/v1",
    "http://localhost:8150/v1",
    "http://[::1]:8150/v1",
    "https://api.nonagentic.ai/v1",
    "https://203.0.113.11/v1",  # stand-in public IP
]


def _origin(url: str) -> str:
    from urllib.parse import urlsplit

    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}"


class TestD3TenantAllowlistHasNoOwnerInfra:
    """MUTATION: ``allowed_tenant_base_url_origins`` returns ``allowed_base_url_origins()``
    unfiltered and ``check_tenant_base_url`` skips the owner-infra check ⇒ the IR
    repro is accepted ⇒ red."""

    def test_inherited_worker_allowlist_contributes_no_owner_infra(self):
        hosts = {key[1] for key in allowed_tenant_base_url_origins(CUSTOMER_TENANT_A)}
        assert "inference-router.inference-router.svc.cluster.local" not in hosts
        assert all(owner_infra_reason(h, resolve=False) is None for h in hosts)

    def test_exact_repro_url_is_refused(self):
        with pytest.raises(TenantBaseUrlNotAllowedError, match="platform infrastructure"):
            check_tenant_base_url(CUSTOMER_TENANT_A, IR_V1)

    @pytest.mark.parametrize("url", OWNER_INFRA_URLS)
    def test_owner_infra_refused_even_when_operator_listed(self, url, tenant_allowlist):
        tenant_allowlist(f"{CUSTOMER_TENANT_A}={_origin(url)}")
        with pytest.raises(TenantBaseUrlNotAllowedError):
            check_tenant_base_url(CUSTOMER_TENANT_A, url)

    def test_name_resolving_to_cluster_ip_is_refused(self, tenant_allowlist, dns):
        """A public-looking name an attacker points at a ClusterIP / tailnet IP."""
        tenant_allowlist(f"{CUSTOMER_TENANT_A}=https://evil.example")
        dns["evil.example"] = ("10.43.12.34",)
        with pytest.raises(TenantBaseUrlNotAllowedError, match="resolves to"):
            check_tenant_base_url(CUSTOMER_TENANT_A, "https://evil.example/v1")
        ro._dns_cache.clear()
        dns["evil.example"] = ("93.184.215.14", "100.100.100.100")
        with pytest.raises(TenantBaseUrlNotAllowedError, match="resolves to"):
            check_tenant_base_url(CUSTOMER_TENANT_A, "https://evil.example/v1")

    def test_unresolvable_name_is_refused(self, tenant_allowlist, dns):
        tenant_allowlist(f"{CUSTOMER_TENANT_A}=https://nx.example")
        with pytest.raises(TenantBaseUrlNotAllowedError, match="does not resolve"):
            check_tenant_base_url(CUSTOMER_TENANT_A, "https://nx.example/v1")

    def test_listed_public_vendor_still_passes(self, tenant_allowlist, dns):
        """The other half: the feature still works for a real external endpoint."""
        tenant_allowlist(f"{CUSTOMER_TENANT_A}={_origin(VENDOR)}")
        dns["api.vendor-byo.example"] = ("93.184.215.14",)
        assert check_tenant_base_url(CUSTOMER_TENANT_A, VENDOR) == VENDOR


class TestD3PutRefusesOwnerInfra:
    """MUTATION: drop the owner-infra filter/check ⇒ PUT stores the IR row (200) ⇒ red."""

    def test_exact_repro_put_is_422_and_not_written(self, put_client):
        c, db = put_client
        resp = c.put("/tenant-config/llm", json=_repro_body())
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == TENANT_BASE_URL_NOT_ALLOWED
        assert "platform infrastructure" in resp.text
        assert db.upserts == 0

    @pytest.mark.parametrize("url", OWNER_INFRA_URLS)
    def test_operator_listed_owner_infra_put_is_422(self, put_client, tenant_allowlist, url):
        c, db = put_client
        tenant_allowlist(f"{CUSTOMER_TENANT_A}={_origin(url)}")
        resp = c.put("/tenant-config/llm", json=_repro_body(url))
        assert resp.status_code == 422, resp.text
        assert db.upserts == 0


class TestD3ResolveAndChatRefuseStoredOwnerInfraRow:
    """A row written before the fix (or directly in the DB) never reaches IR.
    MUTATION: drop the filter/check ⇒ resolve returns base_url_override=IR with
    tenant_byo_key ⇒ red."""

    @staticmethod
    def _row() -> dict[str, Any]:
        return {
            "tenant_id": CUSTOMER_TENANT_A,
            "role": "__default__",
            "provider": "openai_compatible",
            "model": "grok-4-fast",
            "base_url": IR_V1,
            "temperature": None,
            "max_tokens": None,
            "params": {},
            "api_key_encrypted": encrypt_secret(JUNK),
            "cost_source": "worker-management",
            "owner_provided": False,
        }

    @pytest.mark.asyncio
    async def test_resolve_raises_instead_of_pointing_the_byo_client_at_ir(self, monkeypatch):
        db = _OneRowDB()
        db.rows[(CUSTOMER_TENANT_A, "__default__")] = self._row()
        monkeypatch.setattr(r, "_db", db)
        with pytest.raises(r.TenantLLMKeyUnusableError) as exc_info:
            await r.resolve_llm_config(
                {"model_provider": "inference_router", "model": "local-llm"},
                CUSTOMER_TENANT_A,
                key_entitled=True,
            )
        assert exc_info.value.reason == "base_url_not_allowlisted"
        assert exc_info.value.status_code == 409


class TestD3EnforceCustomerRoutingBelt:
    """MUTATION: remove the owner-infra refusal in enforce_customer_routing ⇒ the
    config passes through the BYO early return pointed at IR ⇒ red."""

    @staticmethod
    def _cfg(base_url: str, provider: str = "openai") -> dict[str, Any]:
        return {
            "agent_role": "r",
            "model_provider": provider,
            "model": "grok-4-fast",
            "base_url_override": base_url,
            "api_key_override": JUNK,
            "tenant_byo_key": True,
            "work_tier": "medium",
        }

    @pytest.mark.parametrize("url", [IR_V1, "http://100.100.100.100:9204/v1", "http://mini2.local"])
    @pytest.mark.parametrize("provider", ["openai", "deepseek", "anthropic", "groq"])
    def test_external_provider_on_owner_infra_is_refused(self, url, provider):
        with pytest.raises(r.TenantLLMKeyUnusableError) as exc_info:
            r.enforce_customer_routing(self._cfg(url, provider), CUSTOMER_TENANT_A)
        exc = exc_info.value
        assert exc.reason == "base_url_owner_infra"
        assert exc.status_code == 409
        assert exc.error_code == "BYO_KEY_UNUSABLE"
        assert "platform infrastructure" in exc.message
        assert JUNK not in exc.message

    def test_refused_even_without_the_byo_flag(self):
        cfg = self._cfg(IR_V1)
        cfg["tenant_byo_key"] = False
        with pytest.raises(r.TenantLLMKeyUnusableError):
            r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)

    def test_name_resolving_to_owner_infra_is_refused(self, dns):
        dns["sneaky.example"] = ("10.43.0.10",)
        with pytest.raises(r.TenantLLMKeyUnusableError):
            r.enforce_customer_routing(self._cfg("https://sneaky.example/v1"), CUSTOMER_TENANT_A)

    def test_public_byo_endpoint_passes(self, dns):
        dns["api.vendor-byo.example"] = ("93.184.215.14",)
        out = r.enforce_customer_routing(self._cfg(VENDOR), CUSTOMER_TENANT_A)
        assert out["base_url_override"] == VENDOR
        assert out["model_provider"] == "openai"

    def test_platform_tenant_untouched(self):
        cfg = self._cfg(IR_V1)
        assert r.enforce_customer_routing(dict(cfg), PLATFORM_TENANT) == cfg


class TestD3RedirectHopIsRefused:
    """The SDKs follow redirects; a public BYO endpoint that 307s to the IR
    Service must not deliver the request. MUTATION: drop the request hook
    (``_byo_guarded_http_client`` returns None) ⇒ IR receives the POST ⇒ red."""

    def test_byo_openai_client_carries_the_guard(self, dns):
        from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

        cfg = PydanticAIAgentConfig(
            name="t",
            model_provider="openai",
            model="gpt-4o",
            base_url=VENDOR,
            api_key=JUNK,
            tenant_byo_key=True,
            db_path="",
        )
        client = ModelFactory._byo_guarded_http_client(cfg)
        assert client is not None
        assert refuse_owner_infra_request in client.event_hooks["request"]
        assert client.follow_redirects is True
        # Not a tenant endpoint ⇒ SDK default client.
        cfg2 = cfg.model_copy(update={"tenant_byo_key": False})
        assert ModelFactory._byo_guarded_http_client(cfg2) is None

    @pytest.mark.asyncio
    async def test_redirect_to_ir_never_reaches_ir(self, dns):
        from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

        dns["api.vendor-byo.example"] = ("93.184.215.14",)
        hits: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hits.append(str(request.url))
            if request.url.host == "api.vendor-byo.example":
                return httpx.Response(307, headers={"location": IR_V1 + "/chat/completions"})
            return httpx.Response(200, json={"ok": True})

        cfg = PydanticAIAgentConfig(
            name="t",
            model_provider="openai",
            model="gpt-4o",
            base_url=VENDOR,
            api_key=JUNK,
            tenant_byo_key=True,
            db_path="",
        )
        client = ModelFactory._byo_guarded_http_client(cfg)
        assert client is not None
        client._transport = httpx.MockTransport(handler)
        with pytest.raises(TenantBaseUrlNotAllowedError):
            await client.post(VENDOR + "/chat/completions", json={})
        assert hits == [VENDOR + "/chat/completions"]
        await client.aclose()


class TestN1ModelOverrideBaseUrlCleared:
    """MUTATION: drop the base_url_override clear in the IR branch ⇒ the MLX URL
    survives on the customer config ⇒ red."""

    def test_customer_ir_config_loses_the_mlx_url(self):
        cfg = {
            "agent_role": "r",
            "model_provider": "inference_router",
            "model": "mlx-community/some-model",
            "base_url_override": "http://mlx-inference.example.com:8080",
        }
        out = r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)
        assert out["base_url_override"] is None
        assert (out["model_provider"], out["model"]) == ("inference_router", "local-llm")

    def test_customer_ir_on_local_llm_also_loses_it(self):
        cfg = {
            "model_provider": "inference_router",
            "model": "local-llm",
            "base_url_override": "http://mlx-inference.example.com:8080",
        }
        assert r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)["base_url_override"] is None

    def test_platform_keeps_it(self):
        cfg = {
            "model_provider": "inference_router",
            "model": "mlx-community/some-model",
            "base_url_override": "http://mlx-inference.example.com:8080",
        }
        out = r.enforce_customer_routing(dict(cfg), PLATFORM_TENANT)
        assert out["base_url_override"] == cfg["base_url_override"]


# ------------------------------------------------------------- N2 / N3 key test


def _local_row(tenant: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "role": "__default__",
        "provider": "local",
        "model": "local-llm",
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "params": {},
        "api_key_encrypted": None,
        "cost_source": None,
        "owner_provided": False,
    }


@pytest.fixture
def key_test_env(monkeypatch: pytest.MonkeyPatch):
    from src.api.providers import _router as pr
    from src.api.tenant_config import _router as tc

    db = _OneRowDB()
    db.rows[(CUSTOMER_TENANT_A, "__default__")] = _local_row(CUSTOMER_TENANT_A)
    db.rows[(CUSTOMER_TENANT_B, "__default__")] = _local_row(CUSTOMER_TENANT_B)
    monkeypatch.setattr(tc, "_db", db)
    monkeypatch.setattr(r, "_db", db)
    monkeypatch.setattr(tc, "tenant_key_entitled", lambda _t: True)
    pr._byo_live_ok.clear()
    pr._byo_live_fail.clear()
    tc._key_test_cooldown.clear()
    tc._key_test_inflight.clear()
    yield tc
    pr._byo_live_ok.clear()
    pr._byo_live_fail.clear()
    tc._key_test_cooldown.clear()
    tc._key_test_inflight.clear()


class TestN2KeyTestSingleFlight:
    """MUTATION: drop the in-flight map (each miss makes its own call) ⇒ 5
    concurrent tests make 5 IR calls ⇒ red."""

    @pytest.mark.asyncio
    async def test_concurrent_tests_share_one_live_call(self, key_test_env):
        tc = key_test_env
        calls: list[str] = []
        gate = asyncio.Event()

        async def slow_refresh(tenant_id):
            calls.append(tenant_id)
            await gate.wait()
            return [SimpleNamespace(id="local-llm")]

        with patch(
            "src.api.providers._router.inference_router_registry.refresh_model_cache",
            side_effect=slow_refresh,
        ):
            tasks = [
                asyncio.ensure_future(
                    tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_A))
                )
                for _ in range(5)
            ]
            for _ in range(5):
                await asyncio.sleep(0)
            gate.set()
            results = await asyncio.gather(*tasks)
        assert calls == [CUSTOMER_TENANT_A]
        assert all(res.ok for res in results)
        assert tc._key_test_inflight == {}

    @pytest.mark.asyncio
    async def test_one_waiter_cancelled_does_not_cancel_the_shared_call(self, key_test_env):
        tc = key_test_env
        gate = asyncio.Event()

        async def slow_refresh(_tenant_id):
            await gate.wait()
            return [SimpleNamespace(id="local-llm")]

        with patch(
            "src.api.providers._router.inference_router_registry.refresh_model_cache",
            side_effect=slow_refresh,
        ):
            first = asyncio.ensure_future(
                tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_A))
            )
            second = asyncio.ensure_future(
                tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_A))
            )
            for _ in range(5):
                await asyncio.sleep(0)
            first.cancel()
            gate.set()
            res = await second
        assert res.ok is True


class TestN3KeyTestCooldownIsPerTenant:
    """Keyless ``local`` rows share the key fingerprint sha(""); only the tenant
    in the cooldown key separates them. MUTATION: build the cooldown key without
    the tenant ⇒ tenant B gets tenant A's cached failure and IR is called once ⇒ red."""

    @pytest.mark.asyncio
    async def test_two_tenants_get_their_own_live_result(self, key_test_env):
        tc = key_test_env
        calls: list[str] = []

        async def refresh(tenant_id):
            calls.append(tenant_id)
            if tenant_id == CUSTOMER_TENANT_A:
                return []  # ⇒ "inference-router returned no models" ⇒ ok=False
            return [SimpleNamespace(id="local-llm")]

        with patch(
            "src.api.providers._router.inference_router_registry.refresh_model_cache",
            side_effect=refresh,
        ):
            a = await tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_A))
            b = await tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_B))
            a2 = await tc.test_llm_config(tc.LLMKeyTestRequest(), _ctx(CUSTOMER_TENANT_A))
        assert calls == [CUSTOMER_TENANT_A, CUSTOMER_TENANT_B]
        assert a.ok is False and a2.ok is False
        assert b.ok is True
