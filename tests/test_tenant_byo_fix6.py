"""Fix round 6 (validator 4, 2026-10-01): G1, D5, N2-N5, N1 live-impact."""

from __future__ import annotations

import pytest

from tests import test_tenant_byo_fix5 as _fix5

# Fixtures and helpers re-used from fix5 (the modality resolve harness).
OWNER_KEY_PLACEHOLDER = _fix5.OWNER_KEY_PLACEHOLDER
VENDOR = _fix5.VENDOR
_seed = _fix5._seed
_env = _fix5._env  # autouse fixture
app_client = _fix5.app_client

SYSTEM_TENANT = "00000000-0000-0000-0000-000000000001"


# ===================================================================== G1
class TestG1TwoPlatformSets:
    """Prompt-write platform (fr mirror) != commercial-LLM platform (IR mirror)."""

    def test_system_tenant_is_commercial_platform_but_not_prompt_write_platform(self):
        from src.api.tenant_constants import (
            PLATFORM_TENANT_IDS,
            commercial_platform_tenant_ids,
            is_commercial_platform_tenant,
            is_platform_tenant,
        )

        assert is_commercial_platform_tenant(SYSTEM_TENANT) is True
        assert is_platform_tenant(SYSTEM_TENANT) is False
        assert SYSTEM_TENANT not in PLATFORM_TENANT_IDS
        # The commercial set (gitops value, see tests/conftest.py) is a strict
        # superset by exactly the system tenant.
        assert {SYSTEM_TENANT} == commercial_platform_tenant_ids() - PLATFORM_TENANT_IDS

    def test_commercial_set_fails_closed(self):
        from src.api.tenant_constants import is_commercial_platform_tenant

        for value in (None, "", 42, "n8n-dev", "00000000-0000-0000-0000-000000000000"):
            assert is_commercial_platform_tenant(value) is False
        assert is_commercial_platform_tenant("0195f48b-d000-7000-8000-000000000100") is False

    def test_commercial_set_comes_from_gitops_env_and_drops_names(self, monkeypatch):
        """fr#1385: the set is the env value (gitops), never a code default; a name,
        the nil uuid or a non-canonical spelling in it is dropped, never resolved."""
        from src.api.tenant_constants import (
            commercial_platform_tenant_ids,
            is_commercial_platform_tenant,
        )
        from src.settings import reset_settings_cache

        monkeypatch.delenv("COMMERCIAL_PLATFORM_TENANT_IDS", raising=False)
        reset_settings_cache()
        assert commercial_platform_tenant_ids() == frozenset()
        assert is_commercial_platform_tenant("0195f48b-d000-7000-8000-000000000001") is False

        monkeypatch.setenv(
            "COMMERCIAL_PLATFORM_TENANT_IDS",
            " 0195F48B-d000-7000-8000-000000000002 ,n8n-dev,"
            "00000000-0000-0000-0000-000000000000,{0195f48b-d000-7000-8000-000000000003},",
        )
        reset_settings_cache()
        # Case-folded like every tenant uuid; the name, nil and braced forms dropped.
        assert commercial_platform_tenant_ids() == {"0195f48b-d000-7000-8000-000000000002"}
        assert is_commercial_platform_tenant("0195f48b-d000-7000-8000-000000000003") is False

    def test_routing_uses_the_commercial_set(self):
        """The LLM resolver treats the system tenant as platform (IR parity)."""
        from src.worker.factory import llm_config_resolver

        assert llm_config_resolver.is_commercial_platform_tenant(SYSTEM_TENANT) is True

    @staticmethod
    def _paid_ir_config() -> dict:
        return {
            "agent_role": "r",
            "model_provider": "inference_router",
            "model": "deepseek-v4-pro",
            "ir_provider": "deepseek",
        }

    def test_routing_call_site_passes_the_system_tenant_through_untouched(self):
        """Behavioural pin at the real call site (validator 5 T1, scenario A).

        MUTATION: ``enforce_customer_routing`` gating on ``is_platform_tenant``
        (the fr prompt-write mirror, which does NOT contain the system tenant)
        ⇒ the owner's own WM/IR/AA calls lose the IR pin and are coerced to
        ``local-llm`` ⇒ red.
        """
        from src.worker.factory import llm_config_resolver
        from tests._platform_tenant import proven_tenant

        # fr#1385: untouched only when the request PROVED the system tenant.
        with proven_tenant(SYSTEM_TENANT):
            out = llm_config_resolver.enforce_customer_routing(
                self._paid_ir_config(), SYSTEM_TENANT
            )
        assert out == self._paid_ir_config()

    def test_routing_call_site_still_coerces_a_customer(self):
        """Control: the same config IS rewritten for a customer, so the test
        above cannot pass because routing is a no-op."""
        from src.worker.factory import llm_config_resolver

        out = llm_config_resolver.enforce_customer_routing(self._paid_ir_config(), CUSTOMER)
        assert (out["model_provider"], out["model"]) == ("inference_router", "local-llm")
        assert out["ir_provider"] is None


# ===================================================================== D5
# The managed-BYO owner key on GET /tenant-config/modality/{m}/resolve.


def _caller(monkeypatch, azp: str) -> None:
    """Stand in for the VERIFIED X-Caller-Authorization azp (middleware-only setter)."""
    from src.api.tenant_config import _modality_router as mr

    monkeypatch.setattr(mr, "get_caller_client_id_from_context", lambda: azp)


class TestD5OwnerKeyOnlyToVerifiedServiceCaller:
    """MUTATION: drop ``owner_provided=`` at the ``_key_for_caller`` call (or the
    owner branch inside it) ⇒ the tenant's own proven licence token receives the
    OWNER's decrypted key ⇒ red."""

    def test_tenant_own_token_never_gets_the_owner_key(self, app_client, monkeypatch):
        c, mdb, _, _ = app_client  # ctx: identity_proven AND tenant_proven (own licence)
        _caller(monkeypatch, "")
        _seed(mdb, url=VENDOR, owner_provided=True)
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["api_key"] is None
        assert body["hasKey"] is True
        assert body["keyWithheld"] is True
        assert body["owner_provided"] is True
        assert OWNER_KEY_PLACEHOLDER not in resp.text

    def test_unlisted_verified_service_does_not_get_the_owner_key(self, app_client, monkeypatch):
        c, mdb, _, _ = app_client
        _caller(monkeypatch, "improvement-loop-service")
        _seed(mdb, url=VENDOR, owner_provided=True)
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.json()["api_key"] is None
        assert OWNER_KEY_PLACEHOLDER not in resp.text

    def test_verified_modality_gateway_gets_the_owner_key(self, app_client, monkeypatch):
        c, mdb, _, _ = app_client
        _caller(monkeypatch, "modality-gateway")
        _seed(mdb, url=VENDOR, owner_provided=True)
        body = c.get("/tenant-config/modality/embedding/resolve").json()
        assert body["api_key"] == OWNER_KEY_PLACEHOLDER

    def test_empty_setting_is_deny_all(self, app_client, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("MANAGED_BYO_KEY_CALLER_CLIENTS", "")
        reset_settings_cache()
        c, mdb, _, _ = app_client
        _caller(monkeypatch, "modality-gateway")
        _seed(mdb, url=VENDOR, owner_provided=True)
        assert c.get("/tenant-config/modality/embedding/resolve").json()["api_key"] is None

    def test_tenant_byo_key_keeps_current_behaviour(self, app_client, monkeypatch):
        """Control: the tenant's OWN (not owner_provided) key still reaches its own
        proven licence token, with no service caller."""
        c, mdb, _, _ = app_client
        _caller(monkeypatch, "")
        _seed(mdb, url=VENDOR, owner_provided=False)
        body = c.get("/tenant-config/modality/embedding/resolve").json()
        assert body["api_key"] == OWNER_KEY_PLACEHOLDER
        assert body["owner_provided"] is False

    def test_withheld_reason_is_in_the_closed_metric_set(self):
        from src.telemetry.agent_metrics import (
            MODALITY_KEY_WITHHELD_OWNER_KEY,
            MODALITY_KEY_WITHHELD_REASONS,
        )

        assert MODALITY_KEY_WITHHELD_OWNER_KEY in MODALITY_KEY_WITHHELD_REASONS


# ===================================================================== N2
class TestN2EmbeddedIpv4IsUnwrapped:
    """MUTATION: drop ``_embedded_ipv4`` / the denied-IP set from
    ``_ip_owner_infra_reason`` ⇒ these literals pass as global ⇒ red."""

    @pytest.mark.parametrize(
        "literal",
        [
            "::ffff:203.0.113.11",  # IPv4-mapped public address
            "[::ffff:c0a8:016f]",  # same, hex spelling, bracketed
            "::ffff:203.0.113.15",  # a second public address
            "64:ff9b::a00:1",  # NAT64 of 10.0.0.1
            "64:ff9b::c0a8:016f",  # NAT64 of 192.168.1.111
            "::a00:1",  # IPv4-compatible 10.0.0.1
            "::a00:2",  # IPv4-compatible 10.0.0.2
        ],
    )
    def test_literal_is_owner_infra(self, literal):
        from src.security.request_overrides import owner_infra_reason

        assert owner_infra_reason(literal, resolve=False) is not None

    @pytest.mark.parametrize(
        "resolved", ["203.0.113.11", "::ffff:203.0.113.11", "64:ff9b::a00:1", "::a00:1"]
    )
    def test_name_resolving_to_an_embedded_or_estate_ip_is_refused(self, monkeypatch, resolved):
        from src.security import request_overrides as ro

        monkeypatch.setattr(ro, "_resolve_host_ips", lambda _host: (resolved,))
        ro._dns_cache.clear()
        assert ro.owner_infra_reason("api.sneaky.example", use_cache=False) is not None

    def test_public_ip_still_passes(self, monkeypatch):
        """Control: a genuinely public vendor address (v4 and its NAT64 form) passes."""
        from src.security import request_overrides as ro

        assert ro.owner_infra_reason("93.184.215.14", resolve=False) is None
        monkeypatch.setattr(ro, "_resolve_host_ips", lambda _host: ("93.184.215.14",))
        ro._dns_cache.clear()
        assert ro.owner_infra_reason("api.vendor.example", use_cache=False) is None


# ===================================================================== N3
class TestN3RedirectHookResolvesUncachedEveryHop:
    """MUTATION: the hook calls ``owner_infra_reason(..., use_cache=True)`` ⇒ the
    cached public answer wins over live DNS ⇒ red."""

    def test_hook_ignores_a_stale_public_cache_entry(self, monkeypatch):
        import asyncio
        import time

        import httpx

        from src.security import request_overrides as ro

        live = {"ips": ("10.0.0.7",)}
        calls: list[str] = []

        def _live(host):
            calls.append(host)
            return live["ips"]

        monkeypatch.setattr(ro, "_resolve_host_ips", _live)
        ro._dns_cache.clear()
        ro._dns_cache["api.rebind.example"] = (time.monotonic(), ("93.184.215.14",))
        req = httpx.Request("GET", "https://api.rebind.example/v1/chat")
        with pytest.raises(ro.TenantBaseUrlNotAllowedError):
            asyncio.run(ro.refuse_owner_infra_request(req))
        # Every hop resolves again: flip live DNS to public, two more hops.
        live["ips"] = ("93.184.215.14",)
        asyncio.run(ro.refuse_owner_infra_request(req))
        asyncio.run(ro.refuse_owner_infra_request(req))
        assert calls == ["api.rebind.example"] * 3


# ===================================================================== N4
def _dns_recorder(monkeypatch, answer: tuple[str, ...] = ("93.184.215.14",)) -> list[bool]:
    """Stub DNS; record, per lookup, whether it ran ON an event-loop thread."""
    import asyncio

    from src.security import request_overrides as ro

    on_loop: list[bool] = []

    def _resolve(host):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return answer

    monkeypatch.setattr(ro, "_resolve_host_ips", _resolve)
    ro._dns_cache.clear()
    return on_loop


class TestN4DnsRunsOffTheEventLoop:
    """MUTATION: drop the ``await prewarm_owner_infra_dns(...)`` at a site ⇒ the
    sync check's ``getaddrinfo`` runs on the loop thread ⇒ red."""

    def test_llm_put(self, app_client, monkeypatch):
        c, _, _, _ = app_client
        on_loop = _dns_recorder(monkeypatch)
        resp = c.put(
            "/tenant-config/llm",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": VENDOR, "apiKey": "k"},
        )
        assert resp.status_code == 200, resp.text
        assert on_loop and not any(on_loop), on_loop

    def test_modality_put(self, app_client, monkeypatch):
        c, _, _, _ = app_client
        on_loop = _dns_recorder(monkeypatch)
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": VENDOR, "apiKey": "k"},
        )
        assert resp.status_code == 200, resp.text
        assert on_loop and not any(on_loop), on_loop

    def test_modality_resolve(self, app_client, monkeypatch):
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=False)
        on_loop = _dns_recorder(monkeypatch)
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.status_code == 200, resp.text
        assert on_loop and not any(on_loop), on_loop

    def test_llm_resolve(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from src.security.secret_box import encrypt_secret
        from src.worker.factory import llm_config_resolver as lcr

        row = {
            "tenant_id": CUSTOMER,
            "role": "__default__",
            "provider": "openai_compatible",
            "model": "m",
            "base_url": VENDOR,
            "temperature": None,
            "max_tokens": None,
            "params": {},
            "api_key_encrypted": encrypt_secret("k"),
            "cost_source": "worker-management",
            "owner_provided": False,
        }
        db = type("DB", (), {})()
        db.get_config = AsyncMock(return_value=row)
        db.list_configs = AsyncMock(return_value=[row])
        monkeypatch.setattr(lcr, "_db", db)
        on_loop = _dns_recorder(monkeypatch)
        out = asyncio.run(
            lcr.resolve_llm_config(
                {"model_provider": "inference_router", "model": "local-llm"},
                CUSTOMER,
                byo_mode="local_allowed",
                key_entitled=True,
            )
        )
        assert out.get("base_url_override") == VENDOR
        assert on_loop and not any(on_loop), on_loop

    def test_worker_routing_belt(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from src.api.workers._globals import WorkerInstanceCache

        on_loop = _dns_recorder(monkeypatch)
        cfg = {
            "agent_role": "r",
            "agent_type": "pydantic_ai",
            "model_provider": "openai",
            "model": "gpt-x",
            "base_url_override": VENDOR,
        }
        with (
            patch("src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())),
            patch(
                "src.api.workers._globals.resolve_llm_config",
                new=AsyncMock(side_effect=lambda c, _tid, **_kw: c),
            ),
        ):
            asyncio.run(
                WorkerInstanceCache()._create_worker_instance(
                    "w", {"worker_name": "w", "config": dict(cfg)}, tenant_id=CUSTOMER
                )
            )
        assert on_loop and not any(on_loop), on_loop

    def test_llm_key_test(self, app_client, monkeypatch):
        """POST /tenant-config/llm/test (validator 5 T2)."""
        from unittest.mock import AsyncMock, patch

        from src.api.tenant_config import _router as tc
        from src.security.secret_box import encrypt_secret

        c, _, ldb, _ = app_client
        ldb.rows[(CUSTOMER, "__default__")] = {
            "tenant_id": CUSTOMER,
            "role": "__default__",
            "provider": "openai_compatible",
            "model": "m",
            "base_url": VENDOR,
            "temperature": None,
            "max_tokens": None,
            "params": {},
            "api_key_encrypted": encrypt_secret("k"),
            "owner_provided": False,
        }
        monkeypatch.setattr(tc, "tenant_key_entitled", lambda _t: True)
        on_loop = _dns_recorder(monkeypatch)
        with patch(
            "src.api.providers._router._httpx_json_get",
            new=AsyncMock(return_value={"data": [{"id": "m"}]}),
        ):
            resp = c.post("/tenant-config/llm/test", json={})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True, resp.text
        assert on_loop and not any(on_loop), on_loop

    def test_byo_status(self, monkeypatch):
        """``tenant_byo_status`` (GET /providers, the chat router; validator 5 T2)."""
        import asyncio
        from unittest.mock import AsyncMock

        from src.security.secret_box import encrypt_secret
        from src.worker.factory import llm_config_resolver as lcr

        row = {
            "tenant_id": CUSTOMER,
            "role": "__default__",
            "provider": "openai_compatible",
            "model": "m",
            "base_url": VENDOR,
            "temperature": None,
            "max_tokens": None,
            "params": {},
            "api_key_encrypted": encrypt_secret("k"),
            "owner_provided": False,
        }
        db = type("DB", (), {})()
        db.list_configs = AsyncMock(return_value=[row])
        monkeypatch.setattr(lcr, "_db", db)
        on_loop = _dns_recorder(monkeypatch)
        out = asyncio.run(lcr.tenant_byo_status(CUSTOMER, key_entitled=True))
        assert [st["status"] for st in out.values()] == ["usable"], out
        assert on_loop and not any(on_loop), on_loop


CUSTOMER = _fix5.CUSTOMER_TENANT_A


# ===================================================================== N5
class TestN5RefusalNeverEchoesAResolvedInternalIp:
    """MUTATION: put the resolved address back into the refusal text ⇒ red."""

    def test_check_tenant_base_url_text(self, monkeypatch, caplog):
        import logging

        from src.security import request_overrides as ro

        monkeypatch.setattr(ro, "_resolve_host_ips", lambda _host: ("10.43.17.9",))
        ro._dns_cache.clear()
        with (
            caplog.at_level(logging.WARNING, logger=ro.__name__),
            pytest.raises(ro.TenantBaseUrlNotAllowedError) as exc,
        ):
            ro.check_tenant_base_url(CUSTOMER, VENDOR)
        assert "10.43.17.9" not in str(exc.value)
        assert "private" in str(exc.value)
        assert "10.43.17.9" in caplog.text  # the operator still sees it

    def test_modality_put_response_body(self, app_client, monkeypatch):
        from src.security import request_overrides as ro

        c, _, _, _ = app_client
        monkeypatch.setattr(ro, "_resolve_host_ips", lambda _host: ("::ffff:10.43.17.9",))
        ro._dns_cache.clear()
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": VENDOR, "apiKey": "k"},
        )
        assert resp.status_code == 422, resp.text
        assert "10.43.17.9" not in resp.text


# ===================================================================== N1
IR_ENV = (
    "http://inference-router.inference-router.svc.cluster.local:8150"  # the in-cluster IR Service
)


class TestN1ClearedIrOverrideHitsTheSameEndpoint:
    """Live: 187 customer worker rows carry ``base_url_override`` = IR ``…:8150/api``
    or ``…:8150`` (35). The branch normalises it to the IR ``/api`` base for
    customers (fix7; it used to clear it and lean on ``INFERENCE_ROUTER_URL``).
    Pin that the OUTGOING inference-router base URL is byte-identical to what
    those workers hit with the override in place.

    MUTATION: make ``enforce_customer_routing`` keep the override, or break the
    ``/api`` normalisation/fallback ⇒ the URLs differ ⇒ red."""

    @staticmethod
    async def _final_ir_url(base_url_override, tenant_id) -> str:
        from unittest.mock import AsyncMock, patch

        from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig
        from src.worker.factory import worker_factory as wf

        captured: dict[str, object] = {}

        class _Stop(Exception):
            pass

        def _capture(*args, **kwargs):
            captured["base_url"] = args[11]  # resolved_base_url (positional, see worker_factory)
            raise _Stop

        with (
            patch.object(wf, "_load_worker_tools", new=AsyncMock(return_value=[])),
            patch.object(wf, "_setup_worker_database", return_value=":memory:"),
            patch.object(wf, "_setup_mcp_servers_for_worker", new=AsyncMock(return_value=[])),
            patch.object(wf, "_create_worker_configs", side_effect=_capture),
            patch(
                "src.services.inference_router_registry.inference_router_registry.get_model",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(_Stop):
                await wf.create_worker(
                    "w",
                    [],
                    agent_role="r",
                    agent_type="pydantic_ai",
                    model_provider="inference_router",
                    model="local-llm",
                    base_url_override=base_url_override,
                    tenant_id=tenant_id,
                )
        model = ModelFactory._create_inference_router_model(
            PydanticAIAgentConfig(
                name="w",
                model_provider="inference_router",
                model="local-llm",
                base_url=captured["base_url"],
                tenant_id=tenant_id,
                db_path=":memory:",
            )
        )
        return str(model.base_url)

    @pytest.mark.parametrize("shape", [f"{IR_ENV}/api", IR_ENV, f"{IR_ENV}/api/"])
    def test_cleared_override_url_equals_todays_url(self, monkeypatch, shape):
        import asyncio

        from src.settings import reset_settings_cache
        from src.worker.factory.llm_config_resolver import enforce_customer_routing

        monkeypatch.setenv("INFERENCE_ROUTER_URL", IR_ENV)
        reset_settings_cache()
        try:
            today = asyncio.run(self._final_ir_url(shape, CUSTOMER))
            cfg = enforce_customer_routing(
                {
                    "agent_type": "pydantic_ai",
                    "model_provider": "inference_router",
                    "model": "local-llm",
                    "base_url_override": shape,
                },
                CUSTOMER,
            )
            # fix7: an IR-origin override is KEPT, normalised to the /api base.
            assert cfg["base_url_override"] == f"{IR_ENV}/api"
            after = asyncio.run(self._final_ir_url(cfg["base_url_override"], CUSTOMER))
        finally:
            reset_settings_cache()
        assert after == today
        assert after.rstrip("/") == f"{IR_ENV}/api"


# ===================================================================== fix7
class TestFix7CustomerIrOverrideKeptOnlyOnTheIrOrigin:
    """pg CI (no ``INFERENCE_ROUTER_URL``) could not build ANY customer IR worker
    once the override was cleared unconditionally. An override on the
    inference-router origin is now kept (normalised to ``/api``); every other
    host is still dropped, so a customer never reaches a non-IR host through it.

    MUTATIONS: clear unconditionally ⇒ ``test_ir_origin_kept_without_env`` red;
    keep unconditionally / match on host only / accept the operator allowlist
    ⇒ the drop cases red.
    """

    @staticmethod
    def _route(override, provider="inference_router", tenant=CUSTOMER):
        from src.worker.factory.llm_config_resolver import enforce_customer_routing

        return enforce_customer_routing(
            {
                "agent_type": "pydantic_ai",
                "model_provider": provider,
                "model": "local-llm",
                "base_url_override": override,
            },
            tenant,
        )["base_url_override"]

    @pytest.fixture(autouse=True)
    def _no_ir_env(self, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.delenv("INFERENCE_ROUTER_URL", raising=False)
        reset_settings_cache()
        yield
        reset_settings_cache()

    @pytest.mark.parametrize("shape", [IR_ENV, f"{IR_ENV}/api", f"{IR_ENV}/api/", f"{IR_ENV}/v1"])
    def test_ir_origin_kept_without_env(self, shape):
        assert self._route(shape) == f"{IR_ENV}/api"

    def test_local_direct_provider_moved_to_ir_keeps_an_ir_origin(self):
        assert self._route(f"{IR_ENV}/v1", provider="vllm") == f"{IR_ENV}/api"

    @pytest.mark.parametrize(
        "bad",
        [
            "http://mlx-inference.example.com:8080",  # the model_override leftover
            "http://inference-router.invalid",
            "http://inference-router.inference-router.svc.cluster.local:8151/api",  # port
            "http://inference-router.inference-router.svc.cluster.local:8150@evil.example/api",
            "http://evil.example@inference-router.inference-router.svc.cluster.local:8150/api",
            "http://inference-router.inference-router.svc.cluster.local:8150/api?x=1",
            "http://inference-router.inference-router.svc.cluster.local.evil.example:8150",
            "https://api.openai.com/v1",
        ],
    )
    def test_any_other_host_is_dropped(self, bad):
        assert self._route(bad) is None
        assert self._route(bad, provider="vllm") is None

    def test_operator_allowlist_does_not_widen_the_customer_rule(self, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.setenv(
            "WORKER_MANAGEMENT_BASE_URL_OVERRIDE_ALLOWLIST", "mlx-inference.example.com:8080"
        )
        reset_settings_cache()
        assert self._route("http://mlx-inference.example.com:8080") is None

    def test_configured_inference_router_url_origin_is_ir(self, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("INFERENCE_ROUTER_URL", "https://ir.example.test:9443")
        reset_settings_cache()
        assert self._route("https://ir.example.test:9443/api") == "https://ir.example.test:9443/api"

    def test_platform_override_untouched(self):
        url = "http://mlx-inference.example.com:8080"
        assert self._route(url, tenant=SYSTEM_TENANT) == url
