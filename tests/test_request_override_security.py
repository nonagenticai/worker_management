"""Security: request-supplied validator code never runs; base_url_override is allowlisted.

Every refusal here is asserted by its TYPED code, and every validator payload
carries a SENTINEL that sets an attribute on ``builtins`` if the code is ever
executed — so a test that "passes" because the code ran and happened to return
cleanly is impossible.
"""

from __future__ import annotations

import builtins
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.workers.models import MessageRequest, WorkerCreateRequest
from src.auth import get_identity_proven, get_tenant_id
from src.security.request_overrides import (
    BASE_URL_OVERRIDE_NOT_ALLOWED,
    OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
    OutputValidatorCodeNotSupportedError,
    normalize_base_url_override,
)

TENANT = "test-tenant-override-security"
SENTINEL = "_WM_VALIDATOR_EXEC_SENTINEL"
EVIL_VALIDATOR = (
    f"import builtins\nbuiltins.{SENTINEL} = True\ndef validate(ctx, output):\n    return output\n"
)
IR = "http://inference-router.inference-router.svc.cluster.local:8150"
_CRUD = "src.api.workers.crud_endpoints"
_MSG = "src.api.workers.message_endpoints"
_AGENT = "src.worker.agents.pydantic_ai_agent"


@pytest.fixture(autouse=True)
def _no_sentinel():
    if hasattr(builtins, SENTINEL):
        delattr(builtins, SENTINEL)
    yield
    ran = hasattr(builtins, SENTINEL)
    if ran:
        delattr(builtins, SENTINEL)
    assert not ran, "request-supplied validator code was EXECUTED"


def _types(resp) -> set[str]:
    return {e["type"] for e in resp.json().get("validation_errors", [])}


# ---------------------------------------------------------------------------
# Fixtures: a client over mocked registry/cache so nothing reaches a DB or LLM
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_ctx(*_a, **_kw):
    yield "lock"


@pytest.fixture
def deps():
    store: dict[str, dict] = {}

    async def _register(*, worker_name, config, **kw):
        store[worker_name] = {
            "worker_id": kw.get("worker_id"),
            "worker_name": worker_name,
            "is_agent": kw.get("is_agent", True),
            "status": "created",
            "config": dict(config),
            "output_validator": config.get("output_validator"),
            "db_path": kw.get("db_path"),
        }
        return {"worker_id": None}

    async def _get(*, worker_name, **_kw):
        return store.get(worker_name)

    with (
        patch(f"{_CRUD}.worker_registry") as crud_reg,
        patch(f"{_CRUD}.worker_cache") as crud_cache,
        patch(f"{_CRUD}.StateDatabase") as crud_db,
        patch(f"{_CRUD}.create_worker") as crud_create,
        patch(f"{_MSG}.worker_registry") as msg_reg,
        patch(f"{_MSG}.worker_cache") as msg_cache,
        patch(f"{_MSG}.state_manager") as msg_state,
        patch(f"{_MSG}.StateDatabase") as msg_db,
        patch(f"{_MSG}.create_worker") as msg_create,
    ):
        for reg in (crud_reg, msg_reg):
            reg.register_worker = AsyncMock(side_effect=_register)
            reg.get_worker = AsyncMock(side_effect=_get)
            reg.list_workers = AsyncMock(return_value=[])
        db = AsyncMock()
        db.add_worker_state = AsyncMock(return_value=1)
        db.get_latest_worker_state = AsyncMock(return_value=None)
        crud_db.return_value = db
        msg_db.return_value = db
        crud_cache.remove_worker = AsyncMock()
        msg_state.atomic_worker_operation = _noop_ctx
        msg_state.resolve_worker_id = AsyncMock(return_value=("wid", True))
        msg_state.validate_state_transition = MagicMock(return_value=True)
        run_resp = MagicMock(output="ok", data="ok", usage_metrics=None, session_id=None)
        run_resp.cost_usd = None
        run_resp.duration_ms = None
        worker = MagicMock()
        worker.run = AsyncMock(return_value=run_resp)
        msg_cache.get_or_create_worker = AsyncMock(return_value=worker)
        fake = MagicMock()
        fake.db_path = "/tmp/x.db"
        crud_create.return_value = fake
        msg_create.return_value = fake
        yield {
            "store": store,
            "crud_create": crud_create,
            "msg_create": msg_create,
            "msg_cache": msg_cache,
            "worker": worker,
        }


@pytest.fixture
def client(deps):
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


def _worker_body(name="w_sec", **extra):
    return {
        "worker_name": name,
        "agent_role": "assistant",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "default",
        **extra,
    }


# ---------------------------------------------------------------------------
# A. output_validator
# ---------------------------------------------------------------------------


class TestValidatorCodeIsRefusedAtTheApi:
    def test_post_workers_with_code_is_422_and_nothing_stored(self, client, deps):
        resp = client.post("/workers", json=_worker_body(output_validator=EVIL_VALIDATOR))
        assert resp.status_code == 422, resp.text
        assert OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED in _types(resp)
        assert deps["store"] == {}
        deps["crud_create"].assert_not_called()

    def test_put_workers_with_code_is_422_and_nothing_stored(self, client, deps):
        assert client.post("/workers", json=_worker_body()).status_code == 201
        before = dict(deps["store"]["w_sec"]["config"])
        resp = client.put("/workers/w_sec", json=_worker_body(output_validator=EVIL_VALIDATOR))
        assert resp.status_code == 422, resp.text
        assert OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED in _types(resp)
        assert deps["store"]["w_sec"]["config"] == before

    def test_message_with_code_is_422_and_no_agent_is_built(self, client, deps):
        resp = client.post(
            "/workers/w_msg/message",
            json={"message": "hi", "output_validator": EVIL_VALIDATOR},
        )
        assert resp.status_code == 422, resp.text
        assert OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED in _types(resp)
        deps["msg_cache"].get_or_create_worker.assert_not_awaited()
        deps["msg_create"].assert_not_called()

    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_validator_is_still_accepted(self, client, deps, empty):
        resp = client.post("/workers", json=_worker_body(output_validator=empty))
        assert resp.status_code == 201, resp.text
        assert deps["store"]["w_sec"]["config"]["output_validator"] is None
        msg = client.post(
            "/workers/w_sec/message", json={"message": "hi", "output_validator": empty}
        )
        assert msg.status_code == 200, msg.text

    @pytest.mark.parametrize("model", [WorkerCreateRequest, MessageRequest])
    @pytest.mark.parametrize(
        "code", [EVIL_VALIDATOR, "x = 1", "def validate(:", "  def validate(c, o): return o"]
    )
    def test_models_refuse_any_code_with_the_typed_error(self, model, code):
        kwargs = {"worker_name": "w"} if model is WorkerCreateRequest else {"message": "hi"}
        with pytest.raises(ValidationError) as ei:
            model(output_validator=code, **kwargs)
        assert [e["type"] for e in ei.value.errors()] == [OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED]


class TestStoredValidatorNeverRuns:
    @pytest.mark.parametrize("stored_in", ["column", "config"])
    def test_message_to_a_row_with_stored_code_fails_closed(self, client, deps, stored_in):
        # A row written before the refusal existed (0 of 793 live, 2026-09-22).
        deps["store"]["legacy"] = {
            "output_validator": EVIL_VALIDATOR if stored_in == "column" else None,
            "worker_id": "wid-legacy",
            "worker_name": "legacy",
            "is_agent": True,
            "status": "active",
            "db_path": "/tmp/l.db",
            "config": {
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
                "model": "default",
                "output_validator": EVIL_VALIDATOR if stored_in == "config" else None,
            },
        }
        resp = client.post("/workers/legacy/message", json={"message": "hi"})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED
        deps["msg_cache"].get_or_create_worker.assert_not_awaited()
        deps["worker"].run.assert_not_awaited()

    def test_agent_built_from_stored_code_raises_and_never_execs(self):
        """Backstop for every other path that builds an agent (cache, graphs)."""
        with (
            patch(f"{_AGENT}.InferenceRouterOpenAIProvider"),
            patch(f"{_AGENT}.MLXOpenAIProvider"),
            patch(f"{_AGENT}.OpenAIChatModel"),
            patch(f"{_AGENT}.OpenAIProvider"),
            patch(f"{_AGENT}.Agent") as agent_cls,
            patch(f"{_AGENT}.PydanticAIAgent._get_model_settings", return_value=None),
        ):
            agent_cls.return_value = MagicMock()
            from src.worker.agents.pydantic_ai_agent import (
                PydanticAIAgent,
                PydanticAIAgentConfig,
            )

            config = PydanticAIAgentConfig(
                name="legacy",
                model_provider="mlx",
                model="m",
                db_path="",
                base_url="http://test:8080",
                output_validator_code=EVIL_VALIDATOR,
                mcp_servers=[],
                tools=[],
            )
            with pytest.raises(OutputValidatorCodeNotSupportedError):
                PydanticAIAgent(config=config)
            agent_cls.return_value.output_validator.assert_not_called()

    def test_the_exec_path_is_gone_from_the_agent(self):
        import inspect

        import src.worker.agents.pydantic_ai_agent as mod

        assert not hasattr(mod.PydanticAIAgent, "_register_output_validator")
        assert "exec(" not in inspect.getsource(mod).replace("``exec()``", "")


# ---------------------------------------------------------------------------
# B. base_url_override
# ---------------------------------------------------------------------------


ALLOWED = [
    IR,
    f"{IR}/api",
    f"{IR}/",
    "https://inference-router.inference-router.svc.cluster.local:8150/api",
    # Host is lowercased BEFORE comparing, so a case variant of the allowed
    # host is the same origin and passes (and is stored lowercased).
    "HTTP://Inference-Router.INFERENCE-ROUTER.svc.cluster.local:8150/api",
]

REFUSED = [
    "http://evil.example.com:8150",
    "http://EVIL.example.com/v1",
    "http://inference-router.inference-router.svc.cluster.local:8151",  # wrong port
    "http://inference-router.inference-router.svc.cluster.local",  # default port 80
    "http://inference-router.inference-router.svc.cluster.local:8150@evil.example.com",
    "http://user:pw@inference-router.inference-router.svc.cluster.local:8150",  # pragma: allowlist secret
    "http://inference-router.inference-router.svc.cluster.local.evil.com:8150",
    "http://10.43.0.10:8150",
    "http://127.0.0.1:8150",
    "http://[::1]:8150",
    "http://169.254.169.254/latest/meta-data",
    "ftp://inference-router.inference-router.svc.cluster.local:8150",
    "file:///etc/passwd",
    "inference-router.inference-router.svc.cluster.local:8150",  # no scheme
    f"{IR}/api?token=x",
    f"{IR}#frag",
    f" {IR}",
    "http://inference-router.inference-router.svc.cluster.local:8150\\@evil.com",
    "http://evil.com#@inference-router.inference-router.svc.cluster.local:8150",
    "http:///nohost",
]


class TestBaseUrlOverrideAllowlist:
    @pytest.mark.parametrize("url", ALLOWED)
    def test_allowed_origin_passes(self, url):
        req = WorkerCreateRequest(worker_name="w", base_url_override=url)
        assert req.base_url_override is not None
        assert req.base_url_override.startswith(
            ("http://inference-router.", "https://inference-router.")
        )

    @pytest.mark.parametrize("url", REFUSED)
    def test_other_origins_are_refused_with_the_typed_error(self, url):
        with pytest.raises(ValidationError) as ei:
            WorkerCreateRequest(worker_name="w", base_url_override=url)
        assert [e["type"] for e in ei.value.errors()] == [BASE_URL_OVERRIDE_NOT_ALLOWED]

    def test_none_is_no_override(self):
        assert WorkerCreateRequest(worker_name="w").base_url_override is None

    def test_an_ip_literal_passes_only_when_listed(self, monkeypatch):
        from src.settings import reset_settings_cache

        with pytest.raises(ValidationError):
            WorkerCreateRequest(worker_name="w", base_url_override="http://10.0.0.5:9000/v1")
        monkeypatch.setenv("BASE_URL_OVERRIDE_ALLOWLIST", "10.0.0.5:9000")
        reset_settings_cache()
        try:
            assert (
                normalize_base_url_override("http://10.0.0.5:9000/v1") == "http://10.0.0.5:9000/v1"
            )
            # Listing the IP does not list the original default any more.
            with pytest.raises(ValidationError):
                WorkerCreateRequest(worker_name="w", base_url_override="http://10.0.0.6:9000/v1")
        finally:
            monkeypatch.delenv("BASE_URL_OVERRIDE_ALLOWLIST")
            reset_settings_cache()

    def test_post_and_put_refuse_a_foreign_host(self, client, deps):
        resp = client.post(
            "/workers", json=_worker_body(base_url_override="http://evil.example.com/v1")
        )
        assert resp.status_code == 422, resp.text
        assert _types(resp) == {BASE_URL_OVERRIDE_NOT_ALLOWED}
        assert deps["store"] == {}

        assert (
            client.post("/workers", json=_worker_body(base_url_override=f"{IR}/api")).status_code
            == 201
        )
        resp = client.put(
            "/workers/w_sec",
            json=_worker_body(base_url_override=f"{IR}@evil.example.com"),
        )
        assert resp.status_code == 422, resp.text
        assert _types(resp) == {BASE_URL_OVERRIDE_NOT_ALLOWED}
        assert deps["store"]["w_sec"]["config"]["base_url_override"] == f"{IR}/api"
