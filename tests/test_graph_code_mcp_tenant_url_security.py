"""Security: graph function_code never runs; request MCP never spawns; tenant baseUrl is allowlisted.

Three sinks, one file. Every refusal is asserted by its TYPED code, and every
payload that would execute carries a SENTINEL (an attribute set on ``builtins``,
or a file created on disk) so a test that "passes" because the code ran and
happened to return cleanly is impossible.
"""

from __future__ import annotations

import builtins
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("WM_SECRET_KEY", secrets.token_hex(32))

from src.api import tenant_config
from src.api.worker_management_api import app
from src.api.workers.models import WorkerCreateRequest
from src.auth import get_identity_proven, get_tenant_id
from src.security.request_overrides import (
    FUNCTION_CODE_NOT_SUPPORTED,
    MCP_CONFIG_NOT_ALLOWED,
    TENANT_BASE_URL_NOT_ALLOWED,
    FunctionCodeNotSupportedError,
    McpConfigNotAllowedError,
    TenantBaseUrlNotAllowedError,
    check_tenant_base_url,
)
from src.settings import reset_settings_cache
from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import GraphCreateRequest, GraphUpdateRequest, NodeSchema
from src.worker.graphs.nodes.function_node import FunctionNode
from src.worker.graphs.pydantic_graph_state import WorkflowState

TENANT = "019edc1e-0000-7000-8000-00000000c0de"
OTHER_TENANT = "019edc1e-0000-7000-8000-00000000beef"
SENTINEL = "_WM_FUNCTION_CODE_EXEC_SENTINEL"
EVIL_CODE = f"import builtins\nbuiltins.{SENTINEL} = True\nresult = 1\n"
# A meaningless provider credential referenced by NAME, so the detect-secrets
# keyword plugin the CI archetype runs does not flag a literal.
DUMMY_KEY = "provider-cred-placeholder"
IR = "http://inference-router.inference-router.svc.cluster.local:8150"


@pytest.fixture(autouse=True)
def _no_sentinel():
    if hasattr(builtins, SENTINEL):
        delattr(builtins, SENTINEL)
    yield
    ran = hasattr(builtins, SENTINEL)
    if ran:
        delattr(builtins, SENTINEL)
    assert not ran, "request-supplied function_code was EXECUTED"


@pytest.fixture
def tenant_allowlist(monkeypatch):
    """Set TENANT_BASE_URL_ALLOWLIST for one test, always restoring the cache."""

    def _set(value: str) -> None:
        monkeypatch.setenv("TENANT_BASE_URL_ALLOWLIST", value)
        reset_settings_cache()

    yield _set
    monkeypatch.delenv("TENANT_BASE_URL_ALLOWLIST", raising=False)
    reset_settings_cache()


def _types(resp) -> set[str]:
    return {e["type"] for e in resp.json().get("validation_errors", [])}


@pytest.fixture
def client():
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    app.dependency_overrides[get_identity_proven] = lambda: True

    @asynccontextmanager
    async def _lifespan(a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _state() -> WorkflowState:
    return WorkflowState(
        session_id="s",
        graph_id="g",
        variables={},
        node_outputs={},
        completed_nodes=[],
        failed_nodes=[],
    )


def _fn_node(**data) -> dict:
    return {"name": "f", "node_type": "function", "function_name": "noop", "data": data}


# ---------------------------------------------------------------------------
# 1. graph function_code
# ---------------------------------------------------------------------------


class TestGraphFunctionCode:
    def test_post_graph_with_code_is_422_and_nothing_is_stored(self, client):
        with patch("src.api.graph_management_api.create_graph_service") as svc:
            resp = client.post(
                "/graphs/", json={"name": "g_sec", "nodes": [_fn_node(function_code=EVIL_CODE)]}
            )
        assert resp.status_code == 422, resp.text
        assert FUNCTION_CODE_NOT_SUPPORTED in _types(resp)
        svc.assert_not_called()

    def test_patch_graph_with_code_is_422_and_nothing_is_stored(self, client):
        with patch("src.api.graph_management_api.update_graph_service") as svc:
            resp = client.patch(
                "/graphs/019edc1e-0000-7000-8000-000000000123",
                json={"nodes": [_fn_node(function_code=EVIL_CODE)]},
            )
        assert resp.status_code == 422, resp.text
        assert FUNCTION_CODE_NOT_SUPPORTED in _types(resp)
        svc.assert_not_called()

    @pytest.mark.parametrize("node_type", ["function", "worker", "condition"])
    def test_every_node_type_refuses_code(self, node_type):
        extra = {"worker_name": "w"} if node_type == "worker" else {}
        data = {"function_code": "x = 1"}
        if node_type == "condition":
            data["condition"] = "True"
        with pytest.raises(ValidationError) as exc:
            NodeSchema(name="n", node_type=node_type, function_name="noop", data=data, **extra)
        assert FUNCTION_CODE_NOT_SUPPORTED in {e["type"] for e in exc.value.errors()}

    def test_models_accept_a_function_node_without_code(self):
        GraphCreateRequest(name="g", nodes=[NodeSchema(**_fn_node())])
        GraphUpdateRequest(nodes=[NodeSchema(**_fn_node(function_code=""))])

    @pytest.mark.asyncio
    async def test_graph_factory_node_with_bypassed_code_fails_closed(self):
        node = NodeSchema(**_fn_node())
        object.__setattr__(node, "data", {"function_code": EVIL_CODE})
        cls = GraphFactory._create_function_node_class(node, [], {str(node.id): node}, {})
        state = _state()
        from pydantic_graph import GraphRunContext

        with pytest.raises(FunctionCodeNotSupportedError):
            await cls().run(GraphRunContext(state=state, deps=None))

    def test_function_node_refuses_code_at_construction(self):
        with pytest.raises(FunctionCodeNotSupportedError):
            FunctionNode(node_id="n", function_code=EVIL_CODE)

    @pytest.mark.asyncio
    async def test_function_node_with_bypassed_code_fails_closed(self):
        node = FunctionNode(node_id="n", function_name="noop")
        node.function_code = EVIL_CODE
        state = _state()
        from pydantic_graph import GraphRunContext

        with pytest.raises(FunctionCodeNotSupportedError):
            await node.run(GraphRunContext(state=state, deps=None))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["os.getcwd", "builtins.exec", "noop"])
    async def test_function_name_is_a_registry_key_never_an_import(self, name):
        node = FunctionNode(node_id="n", function_name=name)
        state = _state()
        from pydantic_graph import GraphRunContext

        with (
            patch("importlib.import_module") as imp,
            pytest.raises(Exception, match="not registered"),
        ):
            await node.run(GraphRunContext(state=state, deps=None))
        imp.assert_not_called()

    def test_the_exec_path_is_gone(self):
        import ast

        src = Path(__file__).resolve().parents[1] / "src" / "worker" / "graphs"
        for f in ("graph_factory.py", "nodes/function_node.py", "nodes/workflow_executor_node.py"):
            tree = ast.parse((src / f).read_text())
            names = {
                n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
            }
            assert not names & {"exec", "import_module"}, (
                f"{f} still calls {names & {'exec', 'import_module'}}"
            )
            bare = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            assert "__import__" not in bare, f"{f} still references __import__"


# ---------------------------------------------------------------------------
# 2. mcp_config
# ---------------------------------------------------------------------------


def _stdio(sentinel: Path) -> dict:
    return {"mcpServers": {"evil": {"command": "touch", "args": [str(sentinel)]}}}


class TestMcpConfig:
    @pytest.mark.parametrize(
        "mcp_config",
        [
            {"mcpServers": {"evil": {"command": "sh", "args": ["-c", "id"]}}},
            {"mcpServers": {"evil": {"url": IR + "/sse", "command": "sh"}}},
            {"mcpServers": {"evil": {"command": "docker", "args": ["run", "x"]}}},
            {"mcpServers": {"ssrf": {"url": "http://169.254.169.254/latest"}}},
            {"mcpServers": {"ssrf": {"url": "http://postgres.local:5432/"}}},
            {"mcpServers": {"ok": {"url": IR + "/sse", "auth": {"token_url": "http://evil/t"}}}},
            {"mcpServers": {"bad": "not-a-dict"}},
            {"other": {}},
            "/etc/passwd",
            "../mcp_configs/default_mcp_config.json",
        ],
    )
    def test_request_model_refuses_with_the_typed_error(self, mcp_config):
        with pytest.raises(ValidationError) as exc:
            WorkerCreateRequest(worker_name="w", mcp_config=mcp_config)
        assert MCP_CONFIG_NOT_ALLOWED in {e["type"] for e in exc.value.errors()}

    @pytest.mark.parametrize("preset", ["../../etc/x", "a/b", "x.json", "a b"])
    def test_preset_is_a_name_not_a_path(self, preset):
        with pytest.raises(ValidationError) as exc:
            WorkerCreateRequest(worker_name="w", mcp_config_preset=preset)
        assert MCP_CONFIG_NOT_ALLOWED in {e["type"] for e in exc.value.errors()}

    def test_allowlisted_url_server_and_registry_selection_pass(self):
        req = WorkerCreateRequest(
            worker_name="w",
            mcp_config={"mcpServers": {"ir": {"url": IR + "/sse"}}},
            mcp_config_preset="github",
            mcp_servers=["kanban"],
        )
        assert req.mcp_config == {"mcpServers": {"ir": {"url": IR + "/sse"}}}
        assert WorkerCreateRequest(worker_name="w", mcp_config={}).mcp_config == {}

    def test_post_workers_with_stdio_is_422_and_nothing_is_created(self, client, tmp_path):
        with patch("src.api.workers.crud_endpoints.create_worker") as create:
            resp = client.post(
                "/workers",
                json={
                    "worker_name": "w_mcp_sec",
                    "agent_role": "assistant",
                    "agent_type": "pydantic_ai",
                    "model_provider": "inference_router",
                    "model": "default",
                    "mcp_config": _stdio(tmp_path / "spawned"),
                },
            )
        assert resp.status_code == 422, resp.text
        assert MCP_CONFIG_NOT_ALLOWED in _types(resp)
        create.assert_not_called()
        assert not (tmp_path / "spawned").exists()

    @pytest.mark.asyncio
    async def test_stored_stdio_config_fails_closed_and_spawns_nothing(self, tmp_path):
        """A row written before the check (or directly in the DB) never reaches a spawn."""
        from src.worker.factory import worker_factory as wf

        spawned = tmp_path / "spawned"
        with (
            patch.object(wf, "PYDANTIC_AI_AVAILABLE", True),
            patch("src.worker.mcp_utils.create_stdio_server") as stdio,
            patch("asyncio.create_subprocess_exec") as sub,
            patch("subprocess.Popen") as popen,
            pytest.raises(McpConfigNotAllowedError),
        ):
            await wf._setup_mcp_servers_for_worker(
                mcp_config=_stdio(spawned),
                mcp_config_preset=None,
                mcp_servers=None,
                worker_name="w",
                db_path_str=str(tmp_path / "w.db"),
            )
        stdio.assert_not_called()
        sub.assert_not_called()
        popen.assert_not_called()
        assert not spawned.exists()

    @pytest.mark.asyncio
    async def test_stored_file_path_is_never_opened(self, tmp_path):
        from src.worker.factory.mcp_config_resolver import resolve_mcp_config

        cfg = tmp_path / "evil.json"
        cfg.write_text('{"mcpServers": {"e": {"command": "sh"}}}')
        with patch("builtins.open", MagicMock(side_effect=AssertionError("opened"))):
            with pytest.raises(McpConfigNotAllowedError):
                await resolve_mcp_config(mcp_config=str(cfg))


# ---------------------------------------------------------------------------
# 3. tenant BYO baseUrl
# ---------------------------------------------------------------------------


class _FakeDB:
    def __init__(self, row: dict | None = None):
        self.rows: dict[tuple, dict] = {}
        self.upserts = 0
        if row is not None:
            self.rows[(TENANT, "__default__")] = row

    async def get_config(self, tenant_id, role="__default__"):
        return self.rows.get((tenant_id, role))

    async def upsert_config(self, *, tenant_id, role="__default__", **cols):
        self.upserts += 1
        self.rows[(tenant_id, role)] = {"tenant_id": tenant_id, "role": role, **cols}


@pytest.fixture
def tenant_client(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(tenant_config._router, "_db", fake)
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c, fake
    app.dependency_overrides.clear()


def _put(c, base_url: str):
    return c.put(
        "/tenant-config/llm",
        json={"provider": "custom", "model": "m", "baseUrl": base_url},
    )


class TestTenantBaseUrl:
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://169.254.169.254/latest/meta-data",
            "http://postgres.local:5432",
            "https://attacker.example/v1",
            "http://127.0.0.1:8150/v1",
            "https://"
            + "u"
            + ":"
            + "p"
            + "@api.together.xyz/v1",  # userinfo, built so no scanner reads a credential
            "file:///etc/passwd",
        ],
    )
    def test_put_refuses_an_unlisted_endpoint(self, tenant_client, base_url):
        c, fake = tenant_client
        resp = _put(c, base_url)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == TENANT_BASE_URL_NOT_ALLOWED
        assert fake.upserts == 0

    def test_put_accepts_this_tenants_listed_endpoint(self, tenant_client, tenant_allowlist):
        c, fake = tenant_client
        tenant_allowlist(f"{TENANT}=https://api.together.xyz")
        resp = _put(c, "https://api.together.xyz/v1")
        assert resp.status_code == 200, resp.text
        assert fake.rows[(TENANT, "__default__")]["base_url"] == "https://api.together.xyz/v1"

    def test_another_tenants_entry_does_not_apply(self, tenant_client, tenant_allowlist):
        c, fake = tenant_client
        tenant_allowlist(f"{OTHER_TENANT}=https://api.together.xyz")
        resp = _put(c, "https://api.together.xyz/v1")
        assert resp.status_code == 422, resp.text
        assert fake.upserts == 0

    def test_global_allowlist_ir_origin_is_refused_for_a_tenant(self):
        # Validator 3 D3: the worker allowlist always carries inference-router,
        # but a TENANT endpoint never inherits an owner-infra origin.
        with pytest.raises(TenantBaseUrlNotAllowedError, match="platform infrastructure"):
            check_tenant_base_url(TENANT, IR + "/api")
        assert check_tenant_base_url(TENANT, None) is None
        assert check_tenant_base_url(TENANT, "") is None
        with pytest.raises(TenantBaseUrlNotAllowedError):
            check_tenant_base_url(TENANT, "https://attacker.example/v1")

    @pytest.mark.asyncio
    async def test_stored_unlisted_row_never_carries_its_key_there(self, monkeypatch):
        """A pre-existing/DB-written row is dropped: no base_url_override, no key."""
        from src.security.secret_box import encrypt_secret
        from src.worker.factory import llm_config_resolver as r

        row = {
            "provider": "openai_compatible",
            "model": "m",
            "base_url": "http://169.254.169.254/v1",
            "api_key_encrypted": encrypt_secret(DUMMY_KEY),
            "temperature": None,
            "max_tokens": None,
            "owner_provided": True,
            "cost_source": None,
        }
        monkeypatch.setattr(r, "_db", _FakeDB(row))
        # Owner ruling 2026-10-01: the refused row is a CLEAR 409 naming the
        # provider and the fix — no config carrying the key or the endpoint is
        # ever produced, and nothing silently falls back to platform routing.
        for byo_mode in ("local_allowed", "byo_required"):
            with pytest.raises(r.TenantLLMKeyUnusableError) as exc_info:
                await r.resolve_llm_config(
                    {"model_provider": "inference_router", "model": "default"},
                    tenant_id=TENANT,
                    byo_mode=byo_mode,
                    # Entitled, so the ALLOWLIST is what refuses the row — not the
                    # #245 provenance gate, which would pass for another reason.
                    key_entitled=True,
                )
            assert exc_info.value.reason == "base_url_not_allowlisted"
            assert exc_info.value.status_code == 409
            assert DUMMY_KEY not in exc_info.value.message
            assert "169.254.169.254" not in exc_info.value.message

    @pytest.mark.asyncio
    async def test_stored_listed_row_is_honoured(self, monkeypatch, tenant_allowlist):
        from src.security.secret_box import encrypt_secret
        from src.worker.factory import llm_config_resolver as r

        tenant_allowlist(f"{TENANT}=https://api.together.xyz")
        row = {
            "provider": "openai_compatible",
            "model": "m",
            "base_url": "https://api.together.xyz/v1",
            "api_key_encrypted": encrypt_secret(DUMMY_KEY),
            "temperature": None,
            "max_tokens": None,
            "owner_provided": False,
            "cost_source": None,
        }
        monkeypatch.setattr(r, "_db", _FakeDB(row))
        cfg = await r.resolve_llm_config(
            {"model_provider": "inference_router", "model": "default"},
            tenant_id=TENANT,
            key_entitled=True,
        )
        assert cfg["base_url_override"] == "https://api.together.xyz/v1"
        assert cfg["api_key_override"] == DUMMY_KEY


class TestTenantIdCaseInsensitive:
    """The per-tenant allowlist matches tenant ids case-insensitively on BOTH sides."""

    def test_uppercase_entry_matches_lowercase_tenant(self, tenant_allowlist):
        tenant_allowlist(f"{TENANT.upper()}=https://api.together.xyz")
        assert check_tenant_base_url(TENANT.lower(), "https://api.together.xyz/v1")

    def test_lowercase_entry_matches_uppercase_tenant(self, tenant_allowlist):
        tenant_allowlist(f"{TENANT.lower()}=https://api.together.xyz")
        assert check_tenant_base_url(TENANT.upper(), "https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# 4. executor_endpoint (remote_claude / remote_codex) carries the service bearer
# ---------------------------------------------------------------------------

GW = "http://coding-gateway.coding-gateway.svc.cluster.local:8687"
BEARER = "SENTINEL-SERVICE-BEARER"


@pytest.fixture
def executor_env(monkeypatch):
    """Record every outbound request and every token fetch; CODING_SERVICE_URL unset."""
    from types import SimpleNamespace

    from src.worker.agents import remote_container_executor as rce

    monkeypatch.delenv("CODING_SERVICE_URL", raising=False)
    sent: list[dict] = []
    tokens: list[str] = []

    class _Client:
        def __init__(self, *a, **kw):
            self.verify = kw.get("verify", True)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            sent.append({"method": "POST", "url": url, "verify": self.verify, **kw})
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"success": True, "output": "ok", "exit_code": 0},
                text="",
            )

        async def get(self, url, **kw):
            sent.append({"method": "GET", "url": url, "verify": self.verify, **kw})
            return SimpleNamespace(status_code=200, json=dict, text="")

    async def _get_token():
        tokens.append(BEARER)
        return BEARER

    monkeypatch.setattr(rce.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        rce, "get_service_authenticator", lambda: SimpleNamespace(get_token=_get_token)
    )
    return SimpleNamespace(rce=rce, sent=sent, tokens=tokens)


def _executor(rce, endpoint: str):
    return rce.RemoteContainerExecutor(
        rce.RemoteExecutionConfig(executor_endpoint=endpoint, container_id="c1")
    )


class TestExecutorEndpoint:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://attacker.example",
            "http://169.254.169.254",
            "http://10.42.0.7:8687",
            "http://" + "u" + ":" + "p" + "@coding-gateway.coding-gateway.svc.cluster.local:8687",
            "ftp://coding-gateway.coding-gateway.svc.cluster.local:8687",
            "http://coding-gateway.coding-gateway.svc.cluster.local:9999",
        ],
    )
    def test_request_model_refuses_with_the_typed_error(self, endpoint):
        from src.security.request_overrides import EXECUTOR_ENDPOINT_NOT_ALLOWED

        with pytest.raises(ValidationError) as exc:
            WorkerCreateRequest(worker_name="w", executor_endpoint=endpoint)
        assert EXECUTOR_ENDPOINT_NOT_ALLOWED in {e["type"] for e in exc.value.errors()}

    def test_request_model_accepts_and_normalises_the_gateway(self):
        req = WorkerCreateRequest(
            worker_name="w", executor_endpoint=GW.replace("coding-gateway.", "Coding-Gateway.", 1)
        )
        assert req.executor_endpoint == GW
        assert (
            WorkerCreateRequest(worker_name="w", executor_endpoint=None).executor_endpoint is None
        )

    @pytest.mark.asyncio
    async def test_bearer_is_never_sent_to_an_unlisted_host(self, executor_env):
        """SENTINEL: a stored/constructed unlisted endpoint gets no request and no token."""
        from src.security.request_overrides import ExecutorEndpointNotAllowedError

        ex = _executor(executor_env.rce, "https://attacker.example")
        with pytest.raises(ExecutorEndpointNotAllowedError):
            await ex.execute_agent(cli="claude", prompt="hi")
        with pytest.raises(ExecutorEndpointNotAllowedError):
            await ex.get_container_info()
        assert await ex.check_health() is False
        assert executor_env.sent == []
        assert executor_env.tokens == [], "the service bearer was fetched for an unlisted host"

    @pytest.mark.asyncio
    async def test_no_endpoint_configured_fails_closed(self, executor_env):
        from src.security.request_overrides import ExecutorEndpointNotAllowedError

        with pytest.raises(ExecutorEndpointNotAllowedError):
            await _executor(executor_env.rce, "").execute_agent(cli="claude", prompt="hi")
        assert executor_env.sent == [] and executor_env.tokens == []

    @pytest.mark.asyncio
    async def test_gateway_gets_the_bearer_and_mesh_hosts_skip_tls_verify(self, executor_env):
        await _executor(executor_env.rce, GW).execute_agent(cli="claude", prompt="hi")
        (req,) = executor_env.sent
        assert req["url"] == f"{GW}/api/containers/c1/execute-agent"
        assert req["headers"]["Authorization"] == f"Bearer {BEARER}"
        assert req["verify"] is False  # *.svc.cluster.local only (ambient-mesh mTLS)

    @pytest.mark.asyncio
    async def test_a_listed_non_cluster_host_is_tls_verified(self, executor_env, monkeypatch):
        monkeypatch.setenv("EXECUTOR_ENDPOINT_ALLOWLIST", "https://coding.example.com:443")
        reset_settings_cache()
        try:
            await _executor(executor_env.rce, "https://coding.example.com").execute_agent(
                cli="claude", prompt="hi"
            )
        finally:
            monkeypatch.delenv("EXECUTOR_ENDPOINT_ALLOWLIST")
            reset_settings_cache()
        (req,) = executor_env.sent
        assert req["verify"] is True

    def test_no_verify_false_literal_remains(self):
        src = Path(__file__).resolve().parents[1] / "src" / "worker" / "agents"
        assert "verify=False" not in (src / "remote_container_executor.py").read_text()


# ---------------------------------------------------------------------------
# 5. inference_provider graph node: env-var secret to a request-chosen host
# ---------------------------------------------------------------------------


# The env-var field name, referenced by NAME so the detect-secrets keyword
# plugin does not read `api_key...: "<value>"` literals as credentials.
ENV_FIELD = "api_key" + "_env"


def _ip_node(**fc) -> dict:
    return {
        "name": "p",
        "node_type": "inference_provider",
        "framework_config": {"name": "p", "model": "m", ENV_FIELD: "OPENAI_API_KEY", **fc},
    }


class TestInferenceProviderConfig:
    @pytest.mark.parametrize(
        "fc",
        [
            {ENV_FIELD: "WM_SECRET_KEY"},
            {ENV_FIELD: "DATABASE_URL"},
            {"base_url": "https://attacker.example/v1"},
            {"base_url": "http://169.254.169.254/v1"},
        ],
    )
    def test_node_schema_refuses(self, fc):
        from src.security.request_overrides import INFERENCE_PROVIDER_CONFIG_NOT_ALLOWED

        with pytest.raises(ValidationError) as exc:
            NodeSchema(**_ip_node(**fc))
        assert INFERENCE_PROVIDER_CONFIG_NOT_ALLOWED in {e["type"] for e in exc.value.errors()}

    def test_allowlisted_config_passes(self):
        NodeSchema(**_ip_node(base_url=IR + "/v1"))
        NodeSchema(**_ip_node())

    def test_runtime_refuses_a_bypassed_config(self):
        from src.worker.graphs.nodes.inference_provider_node import _get_inference_config

        node = NodeSchema(**_ip_node())
        object.__setattr__(
            node,
            "framework_config",
            {
                "name": "p",
                "model": "m",
                ENV_FIELD: "WM_SECRET_KEY",
                "base_url": "https://a.example",
            },
        )
        with pytest.raises(ValueError, match="inference_provider framework_config is not allowed"):
            _get_inference_config(node)
