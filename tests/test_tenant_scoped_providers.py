"""Tenant-scoped LLM provider access (owner ruling 2026-10-01).

- No customer tenant may use the owner's commercial keys (env XAI_API_KEY, ...).
- A customer may use a commercial provider with ITS OWN BYO key, or a managed
  BYO key (``owner_provided``) the owner set for that ONE tenant.
- In /chat a customer sees inference_router with ``local-llm`` only, plus every
  provider it has a usable BYO config for, listed with that provider's live
  models fetched with the tenant's key.
- Platform tenants keep the previous behaviour.

Each test names the mutation that turns it red.
"""

from __future__ import annotations

import os
import secrets
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.exceptions import UserError

from src.api.identity_context import (
    TENANT_PROVENANCE_LICENSE,
    _tenant_id_var,
    _tenant_provenance_var,
)
from src.api.worker_management_api import app
from src.security.secret_box import encrypt_secret
from src.services.inference_router_registry import InferenceRouterRegistry
from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig
from src.worker.factory import llm_config_resolver
from tests._platform_tenant import (
    CUSTOMER_TENANT_A,
    CUSTOMER_TENANT_B,
    PLATFORM_TENANT,
    caller_tenant,
)
from tests.test_ir_passthrough_sampling_correlation import _IRStub

ROUTER = "src.api.providers._router"
CHAT = "src.api.providers._chat_router"
OWNER_XAI_KEY = "xai-OWNER-KEY-must-never-be-used"  # pragma: allowlist secret
OWNER_OPENAI_KEY = "sk-OWNER-openai-must-never-be-used"  # pragma: allowlist secret
TENANT_A_OPENAI_KEY = "sk-tenant-a-own-key"  # pragma: allowlist secret
MANAGED_XAI_KEY_FOR_A = "xai-managed-for-tenant-a"  # pragma: allowlist secret


class _FakeLLMConfigDB:
    """tenant_llm_config, keyed by tenant: the resolver reads by tenant only."""

    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows
        self.list_calls: list[str] = []

    async def list_configs(self, tenant_id: str) -> list[dict[str, Any]]:
        self.list_calls.append(tenant_id)
        return [dict(r) for r in self.rows.get(tenant_id, [])]

    async def get_config(self, tenant_id: str, role: str = "__default__") -> dict[str, Any] | None:
        for r in self.rows.get(tenant_id, []):
            if r["role"] == role:
                return dict(r)
        return None


def _row(
    tenant_id: str,
    provider: str,
    model: str,
    key: str,
    *,
    role: str = "__default__",
    owner_provided: bool = False,
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "role": role,
        "provider": provider,
        "model": model,
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "params": {},
        "api_key_encrypted": encrypt_secret(key),
        "cost_source": "worker-management",
        "owner_provided": owner_provided,
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    """Owner keys ARE mounted (as on the live pod); nothing is proven by default."""
    from src.settings import reset_settings_cache

    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.setenv("XAI_API_KEY", OWNER_XAI_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OWNER_OPENAI_KEY)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _fresh_live_key_cache():
    """The live key-test cache is module-level; never let one test's verdict leak."""
    from src.api.providers import _router

    _router._byo_live_ok.clear()
    _router._byo_live_fail.clear()
    yield
    _router._byo_live_ok.clear()
    _router._byo_live_fail.clear()


@pytest.fixture
def byo_db(monkeypatch: pytest.MonkeyPatch) -> _FakeLLMConfigDB:
    """Tenant A: own openai key + a MANAGED xai key. Tenant B: nothing."""
    db = _FakeLLMConfigDB(
        {
            CUSTOMER_TENANT_A: [
                _row(CUSTOMER_TENANT_A, "openai", "gpt-4o", TENANT_A_OPENAI_KEY),
                _row(
                    CUSTOMER_TENANT_A,
                    "xai",
                    "grok-4",
                    MANAGED_XAI_KEY_FOR_A,
                    role="tier:high",
                    owner_provided=True,
                ),
            ],
        }
    )
    monkeypatch.setattr(llm_config_resolver, "_db", db)
    return db


@pytest.fixture
def entitled_as_caller(monkeypatch: pytest.MonkeyPatch):
    """The caller proved its tenant with its own licence (worker_management#245).

    TestClient's auth bypass stamps `bypass` provenance, so the licence proof is
    modelled by making `tenant_key_entitled` answer for the patched caller.
    """
    state: dict[str, str | None] = {"tenant": None}

    def _entitled(tenant_id: str | None) -> bool:
        return bool(tenant_id) and tenant_id == state["tenant"]

    monkeypatch.setattr(llm_config_resolver, "tenant_key_entitled", _entitled)
    return state


@pytest.fixture
def client():
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c


def _as(entitled_as_caller: dict[str, str | None], tenant_id: str):
    entitled_as_caller["tenant"] = tenant_id
    return caller_tenant(tenant_id)


def _owner_env_spy():
    """Spy on the ONLY two places an owner env key can be read."""
    model_factory_reads: list[str] = []
    router_reads: list[str] = []
    real_owner = ModelFactory._owner_env_key

    def owner(env_var: str) -> str | None:
        model_factory_reads.append(env_var)
        return real_owner(env_var)

    from src.api.providers import _router

    real_require = _router._require_env

    def require(name: str) -> str:
        router_reads.append(name)
        return real_require(name)

    return model_factory_reads, router_reads, owner, require


# --------------------------------------------------------------------------- /providers


class TestCustomerProviderList:
    """Scope ruling 2026-10-01: a customer's GET /providers is EXACTLY its
    configured rows whose key passes a live test — no automatic catalog rows."""

    @staticmethod
    def _vendor(ok_keys: set[str]):
        async def fake_get(url: str, *, headers=None, timeout=10.0):
            auth = (headers or {}).get("Authorization", "")
            if any(auth == f"Bearer {k}" for k in ok_keys):
                return {"data": [{"id": "gpt-4o"}, {"id": "grok-4"}]}
            raise httpx.HTTPStatusError(
                "401 Unauthorized",
                request=httpx.Request("GET", url),
                response=httpx.Response(401),
            )

        return fake_get

    def test_customer_sees_only_rows_whose_key_tests_ok(self, client, byo_db, entitled_as_caller):
        """MUTATION: list usable rows without the live key test ⇒ xai (bad key)
        listed ⇒ red. MUTATION: return the platform catalog ⇒ red."""
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch(f"{ROUTER}._httpx_json_get", side_effect=self._vendor({TENANT_A_OPENAI_KEY})),
        ):
            resp = client.get("/providers")
        assert resp.status_code == 200, resp.text
        rows = {r["name"]: r for r in resp.json()["data"]}
        assert set(rows) == {"openai"}
        assert rows["openai"]["source"] == "byo"
        assert rows["openai"]["available"] is True

    def test_both_keys_ok_lists_both_with_their_source(self, client, byo_db, entitled_as_caller):
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch(
                f"{ROUTER}._httpx_json_get",
                side_effect=self._vendor({TENANT_A_OPENAI_KEY, MANAGED_XAI_KEY_FOR_A}),
            ),
        ):
            resp = client.get("/providers")
        rows = {r["name"]: r for r in resp.json()["data"]}
        assert set(rows) == {"openai", "xai"}
        assert rows["xai"]["source"] == "managed_byo"

    def test_customer_without_rows_sees_nothing(self, client, byo_db, entitled_as_caller):
        """No catalog, no env-key providers, no local direct providers, even
        though XAI_API_KEY / OPENAI_API_KEY are mounted."""
        with _as(entitled_as_caller, CUSTOMER_TENANT_B):
            resp = client.get("/providers")
        assert resp.json()["data"] == []

    def test_unknown_or_absent_tenant_fails_closed(self, client, byo_db, entitled_as_caller):
        with _as(entitled_as_caller, None):  # type: ignore[arg-type]
            resp = client.get("/providers")
        assert resp.json()["data"] == []

    def test_unproven_tenant_gets_no_byo_rows(self, client, byo_db, entitled_as_caller):
        """#245: a caller that did not prove tenant A cannot decrypt A's key, so
        A's BYO rows are not usable — and not listed (the vendor is never hit)."""
        entitled_as_caller["tenant"] = None
        with (
            caller_tenant(CUSTOMER_TENANT_A),
            patch(f"{ROUTER}._httpx_json_get", new_callable=AsyncMock) as http_get,
        ):
            resp = client.get("/providers")
        assert resp.json()["data"] == []
        http_get.assert_not_called()

    def test_a_configured_local_row_is_listed_when_inference_router_answers(
        self, client, byo_db, entitled_as_caller
    ):
        """A `local` row (→ inference_router) is an ordinary configured row: listed
        when its live test (IR's model list for the tenant) passes."""
        byo_db.rows[CUSTOMER_TENANT_B] = [
            {
                "tenant_id": CUSTOMER_TENANT_B,
                "role": "__default__",
                "provider": "local",
                "model": "local-llm",
                "base_url": None,
                "api_key_encrypted": None,
                "owner_provided": False,
            }
        ]
        from src.services.inference_router_registry import InferenceRouterModel

        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_B),
            patch(
                f"{ROUTER}.inference_router_registry.refresh_model_cache",
                new_callable=AsyncMock,
                return_value=[InferenceRouterModel(id="local-llm")],
            ) as refresh,
        ):
            resp = client.get("/providers")
        rows = {r["name"]: r for r in resp.json()["data"]}
        assert set(rows) == {"inference_router"}
        assert rows["inference_router"]["source"] == "platform"
        refresh.assert_awaited_with(CUSTOMER_TENANT_B)

    def test_platform_tenant_is_unchanged(self, client, byo_db, entitled_as_caller):
        """MUTATION: apply the customer filter to platform tenants ⇒ red."""
        # A PROVEN platform tenant (``_as`` → caller_tenant, proven by default):
        # env key presence decides `available` (owner ruling 2026-10-03).
        with _as(entitled_as_caller, PLATFORM_TENANT):
            resp = client.get("/providers")
        rows = {r["name"]: r for r in resp.json()["data"]}
        assert {"vllm", "mlx", "vlm_local", "anthropic", "xai", "openai"} <= set(rows)
        assert rows["xai"]["available"] is True  # env key presence, as before
        assert rows["anthropic"]["available"] is False


# ---------------------------------------------------------------- /providers/{name}/models


class TestCustomerModelLists:
    def test_customer_inference_router_models_are_local_llm_only(
        self, client, byo_db, entitled_as_caller
    ):
        """IR's per-tenant filter may not be live yet and return everything; WM
        still shows a customer only `local-llm`. MUTATION: drop the filter ⇒ red."""
        seen: list[str | None] = []

        async def fake_for_api(tenant_id=None):
            seen.append(tenant_id)
            return {"data": [{"id": "local-llm"}, {"id": "tier-medium"}, {"id": "grok-4"}]}

        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch(
                f"{ROUTER}.inference_router_registry.get_models_for_api", side_effect=fake_for_api
            ),
        ):
            resp = client.get("/providers/inference_router/models")
        assert resp.status_code == 200, resp.text
        assert [m["id"] for m in resp.json()["data"]] == ["local-llm"]
        assert seen == [CUSTOMER_TENANT_A]  # tenant forwarded to IR

    def test_platform_inference_router_models_unfiltered_and_tenant_forwarded(
        self, client, byo_db, entitled_as_caller
    ):
        seen: list[str | None] = []

        async def fake_for_api(tenant_id=None):
            seen.append(tenant_id)
            return {"data": [{"id": "local-llm"}, {"id": "tier-medium"}]}

        with (
            _as(entitled_as_caller, PLATFORM_TENANT),
            patch(
                f"{ROUTER}.inference_router_registry.get_models_for_api", side_effect=fake_for_api
            ),
        ):
            resp = client.get("/providers/inference_router/models")
        assert [m["id"] for m in resp.json()["data"]] == ["local-llm", "tier-medium"]
        assert seen == [PLATFORM_TENANT]

    def test_customer_xai_without_byo_is_403_and_env_key_never_read(
        self, client, byo_db, entitled_as_caller
    ):
        """Tenant B has no xai config. MUTATION: fall back to the env key ⇒ red
        (200 or 403-with-env-read)."""
        mf_reads, router_reads, owner, require = _owner_env_spy()
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_B),
            patch.object(ModelFactory, "_owner_env_key", staticmethod(owner)),
            patch(f"{ROUTER}._require_env", side_effect=require),
            patch(f"{ROUTER}._httpx_json_get", new_callable=AsyncMock) as http_get,
        ):
            resp = client.get("/providers/xai/models")
        assert resp.status_code == 403, resp.text
        detail = resp.json().get("detail") or resp.text
        assert "XAI_API_KEY" not in detail and "ALLOW_EXTERNAL" not in detail
        assert router_reads == [] and mf_reads == []
        http_get.assert_not_called()

    def test_customer_local_direct_provider_is_403(self, client, byo_db, entitled_as_caller):
        with _as(entitled_as_caller, CUSTOMER_TENANT_A):
            resp = client.get("/providers/vllm/models")
        assert resp.status_code == 403

    def test_customer_byo_openai_models_fetched_with_tenant_key(
        self, client, byo_db, entitled_as_caller
    ):
        """MUTATION: fetch with the env key ⇒ the Authorization header is the
        owner's and this goes red."""
        _, router_reads, _, require = _owner_env_spy()
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch(f"{ROUTER}._require_env", side_effect=require),
            patch(
                f"{ROUTER}._httpx_json_get",
                new_callable=AsyncMock,
                return_value={"data": [{"id": "gpt-5"}, {"id": "gpt-4o"}]},
            ) as http_get,
        ):
            resp = client.get("/providers/openai/models")
        assert resp.status_code == 200, resp.text
        assert [m["id"] for m in resp.json()["data"]] == ["gpt-5", "gpt-4o"]
        url = http_get.call_args.args[0]
        headers = http_get.call_args.kwargs["headers"]
        assert url == "https://api.openai.com/v1/models"
        assert headers["Authorization"] == f"Bearer {TENANT_A_OPENAI_KEY}"
        assert router_reads == []

    def test_managed_byo_for_a_is_not_usable_by_b(self, client, byo_db, entitled_as_caller):
        """MUTATION: resolve BYO rows without the tenant key (or across tenants) ⇒
        B lists/uses A's managed xai key ⇒ red."""
        with _as(entitled_as_caller, CUSTOMER_TENANT_B):
            listed = client.get("/providers").json()["data"]
            models = client.get("/providers/xai/models")
            chat = client.post(
                "/chat/completions",
                json={
                    "provider": "xai",
                    "model": "grok-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert "xai" not in {r["name"] for r in listed}
        assert models.status_code == 403
        assert chat.status_code == 403
        assert CUSTOMER_TENANT_A not in byo_db.list_calls


# --------------------------------------------------------------------- /chat/completions


def _chat(
    client: TestClient,
    ir: _IRStub,
    *,
    provider: str,
    model: str,
    headers: dict[str, str] | None = None,
) -> tuple[httpx.Response, list[Any]]:
    from src.api.providers import _chat_router

    real_build = _chat_router._build_agent
    built: list[Any] = []

    async def build_then_stub(req: Any, request: Any, byo: Any = None) -> Any:
        agent = await real_build(req, request, byo)
        built.append(agent)
        agent.model.client._client._transport = httpx.MockTransport(ir)
        return agent

    with patch(f"{CHAT}._build_agent", side_effect=build_then_stub):
        resp = client.post(
            "/chat/completions",
            json={
                "provider": provider,
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=headers or {},
        )
    return resp, built


class TestCustomerChat:
    def test_customer_xai_without_byo_is_403_and_env_key_never_read(
        self, client, byo_db, entitled_as_caller
    ):
        mf_reads, _, owner, _ = _owner_env_spy()
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_B),
            patch.object(ModelFactory, "_owner_env_key", staticmethod(owner)),
            patch(f"{CHAT}._build_agent") as build,
        ):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "xai",
                    "model": "grok-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 403, resp.text
        assert "XAI_API_KEY" not in resp.text
        build.assert_not_called()
        assert mf_reads == []

    def test_customer_inference_router_other_model_is_403(self, client, byo_db, entitled_as_caller):
        with _as(entitled_as_caller, CUSTOMER_TENANT_A), patch(f"{CHAT}._build_agent") as b:
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "inference_router",
                    "model": "tier-medium",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 403
        b.assert_not_called()

    def test_customer_local_llm_works_and_inbound_x_provider_is_dropped(
        self, client, byo_db, entitled_as_caller, monkeypatch
    ):
        """MUTATION: forward a customer's X-Provider ⇒ IR sees `x-provider: xai` ⇒ red."""
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://inference-router.invalid")
        ir = _IRStub()
        with _as(entitled_as_caller, CUSTOMER_TENANT_A):
            resp, _ = _chat(
                client,
                ir,
                provider="inference_router",
                model="local-llm",
                headers={"X-Provider": "xai"},
            )
        assert resp.status_code == 200, resp.text
        assert ir.calls, "inference-router was never called"
        _, headers = ir.calls[-1]
        assert "x-provider" not in headers
        assert headers.get("x-tenant-id") == CUSTOMER_TENANT_A

    def test_platform_inbound_x_provider_still_forwarded(
        self, client, byo_db, entitled_as_caller, monkeypatch
    ):
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://inference-router.invalid")
        ir = _IRStub()
        with _as(entitled_as_caller, PLATFORM_TENANT):
            resp, _ = _chat(
                client,
                ir,
                provider="inference_router",
                model="tier-medium",
                headers={"X-Provider": "pgx-local"},
            )
        assert resp.status_code == 200, resp.text
        assert ir.calls[-1][1].get("x-provider") == "pgx-local"

    def test_customer_byo_openai_chat_uses_tenant_key_and_emits_byo_cost_event(
        self, client, byo_db, entitled_as_caller
    ):
        """MUTATION (any one ⇒ red): build without the tenant key (env fallback ⇒
        owner bearer on the wire); drop tenant_byo_key (⇒ refused, 502); skip the
        cost emit (⇒ no event)."""
        mf_reads, _, owner, _ = _owner_env_spy()
        ir = _IRStub(served={})
        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch.object(ModelFactory, "_owner_env_key", staticmethod(owner)),
            patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit,
        ):
            resp, _built = _chat(client, ir, provider="openai", model="gpt-4o")
        assert resp.status_code == 200, resp.text
        _, headers = ir.calls[-1]
        assert headers["authorization"] == f"Bearer {TENANT_A_OPENAI_KEY}"
        assert mf_reads == []
        emit.assert_called_once()
        kwargs = emit.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["model_provider"] == "openai"
        assert kwargs["owner_provided"] is False
        assert kwargs["input_tokens"] == 3 and kwargs["output_tokens"] == 1

    def test_managed_byo_chat_emits_owner_provided(self, client, byo_db, entitled_as_caller):
        """Managed BYO bills as a marked-up owner_provided line, with A's managed key."""
        from src.worker.agents import pydantic_ai_agent

        captured: dict[str, Any] = {}

        def fake_xai(config: PydanticAIAgentConfig):
            captured["key"] = ModelFactory.get_api_key(config, "XAI_API_KEY")
            captured["byo"] = config.tenant_byo_key
            raise RuntimeError("stop before the network")

        with (
            _as(entitled_as_caller, CUSTOMER_TENANT_A),
            patch.object(
                pydantic_ai_agent.ModelFactory, "_create_xai_model", staticmethod(fake_xai)
            ),
        ):
            resp = client.post(
                "/chat/completions",
                json={
                    "provider": "xai",
                    "model": "grok-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 502  # our deliberate stop, after key resolution
        assert captured == {"key": MANAGED_XAI_KEY_FOR_A, "byo": True}


# ------------------------------------------------------------------------- ModelFactory


class TestModelFactoryGate:
    def _cfg(self, **kw: Any) -> PydanticAIAgentConfig:
        return PydanticAIAgentConfig(
            name="t", model_provider="xai", model="grok-4", db_path="", **kw
        )

    def test_byo_flag_admits_external_and_never_reads_env(self):
        with patch.object(ModelFactory, "_owner_env_key") as owner:
            key = ModelFactory.get_api_key(
                self._cfg(
                    api_key=TENANT_A_OPENAI_KEY, tenant_byo_key=True, tenant_id=CUSTOMER_TENANT_A
                ),
                "XAI_API_KEY",
            )
            empty = ModelFactory.get_api_key(
                self._cfg(api_key=None, tenant_byo_key=True, tenant_id=CUSTOMER_TENANT_A),
                "XAI_API_KEY",
            )
        assert key == TENANT_A_OPENAI_KEY
        assert empty is None  # never topped up from the owner's env key
        owner.assert_not_called()

    def test_proven_customer_is_refused_and_env_never_read(self):
        """MUTATION: drop the membership check from ground 2 ⇒ customer gets owner key."""
        from tests._platform_tenant import platform_tenant_context

        with patch.object(ModelFactory, "_owner_env_key") as owner:
            # Even a LICENCE-PROVEN customer: proof is not membership.
            with platform_tenant_context(CUSTOMER_TENANT_A):
                with pytest.raises(UserError, match="not_a_commercial_platform_tenant"):
                    ModelFactory.get_api_key(self._cfg(tenant_id=CUSTOMER_TENANT_A), "XAI_API_KEY")
            with pytest.raises(UserError):
                ModelFactory.get_api_key(self._cfg(), "XAI_API_KEY")  # no tenant at all
        owner.assert_not_called()

    def test_proven_platform_uses_env_key(self):
        """Owner ruling 2026-10-03: a PROVEN platform workspace uses the registered key."""
        from tests._platform_tenant import platform_tenant_context

        with platform_tenant_context(PLATFORM_TENANT):
            assert (
                ModelFactory.get_api_key(self._cfg(tenant_id=PLATFORM_TENANT), "XAI_API_KEY")
                == OWNER_XAI_KEY
            )

    def test_asserted_platform_is_refused_and_env_never_read(self):
        """MUTATION: drop the proof check from ground 2 ⇒ a header names 0001, gets the key."""
        from tests._platform_tenant import platform_tenant_context

        with patch.object(ModelFactory, "_owner_env_key") as owner:
            with platform_tenant_context(PLATFORM_TENANT, provenance="asserted"):
                with pytest.raises(UserError, match="provenance_asserted"):
                    ModelFactory.get_api_key(self._cfg(tenant_id=PLATFORM_TENANT), "XAI_API_KEY")
        owner.assert_not_called()

    def test_worker_path_resolver_sets_the_flag_only_with_a_tenant_key(self, byo_db):
        """resolve_llm_config is the worker path's only writer of tenant_byo_key."""
        import asyncio

        cfg = asyncio.run(
            llm_config_resolver.resolve_llm_config(
                {"model_provider": "inference_router", "model": "local-llm"},
                CUSTOMER_TENANT_A,
                key_entitled=True,
            )
        )
        assert cfg["model_provider"] == "openai"
        assert cfg["api_key_override"] == TENANT_A_OPENAI_KEY
        assert cfg["tenant_byo_key"] is True
        # Unentitled: a clear withheld error (owner ruling 2026-10-01), never a
        # config carrying the flag and never a silent platform fallback.
        with pytest.raises(llm_config_resolver.TenantLLMKeyUnusableError):
            asyncio.run(
                llm_config_resolver.resolve_llm_config(
                    {"model_provider": "inference_router", "model": "local-llm"},
                    CUSTOMER_TENANT_A,
                    key_entitled=False,
                )
            )


# ------------------------------------------------------------- resolver (real contextvars)


class TestResolverEntitlementWithRealContext:
    @pytest.mark.asyncio
    async def test_licence_proven_tenant_a_resolves_its_rows_and_b_gets_none(self, byo_db):
        t1 = _tenant_id_var.set(CUSTOMER_TENANT_A)
        t2 = _tenant_provenance_var.set(TENANT_PROVENANCE_LICENSE)
        try:
            a = await llm_config_resolver.resolve_tenant_byo_providers(CUSTOMER_TENANT_A)
            # Same proven request asking for B's rows: not entitled, and B has none.
            b = await llm_config_resolver.resolve_tenant_byo_providers(CUSTOMER_TENANT_B)
        finally:
            _tenant_provenance_var.reset(t2)
            _tenant_id_var.reset(t1)
        assert a["openai"]["api_key"] == TENANT_A_OPENAI_KEY
        assert a["xai"]["owner_provided"] is True
        assert b == {}

    @pytest.mark.asyncio
    async def test_platform_only_byo_mode_disables_byo(self, byo_db):
        out = await llm_config_resolver.resolve_tenant_byo_providers(
            CUSTOMER_TENANT_A, byo_mode="platform_only", key_entitled=True
        )
        assert out == {}


# --------------------------------------------------------------- IR registry cache


class TestInferenceRouterRegistryPerTenant:
    @pytest.mark.asyncio
    async def test_cache_isolated_per_tenant_and_tenant_header_sent(self, monkeypatch):
        """MUTATION: one shared 'models' slot ⇒ B is served A's cached list ⇒ red."""
        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://inference-router.invalid")
        from src.settings import reset_settings_cache

        reset_settings_cache()
        sent: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            tenant = request.headers.get("x-tenant-id")
            sent.append(tenant)
            ids = ["local-llm"] if tenant == CUSTOMER_TENANT_A else ["local-llm", "tier-medium"]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})

        real_client = httpx.AsyncClient

        def client_factory(*a: Any, **k: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(handler), **k)

        registry = InferenceRouterRegistry()
        with patch(
            "src.services.inference_router_registry.httpx.AsyncClient", side_effect=client_factory
        ):
            a1 = [m.id for m in await registry.get_all_models(CUSTOMER_TENANT_A)]
            p1 = [m.id for m in await registry.get_all_models(PLATFORM_TENANT)]
            a2 = [m.id for m in await registry.get_all_models(CUSTOMER_TENANT_A)]
        assert a1 == a2 == ["local-llm"]
        assert p1 == ["local-llm", "tier-medium"]
        assert sent == [CUSTOMER_TENANT_A, PLATFORM_TENANT]  # a2 from A's own slot
