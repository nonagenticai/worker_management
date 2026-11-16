"""Project-id attribution — inbound X-Project-Id → ContextVar → both outbound paths.

`X-Project-Id` is the kanban project a run belongs to, so downstream spend can be
priced per project. It is orthogonal to `X-App-Id` (WHICH APP) and to
tenant/license/subject (WHO): one app serves many projects. worker_management is
a RELAY for it and nothing else — it reads the inbound header into a ContextVar
(`auth_middleware._stash_identity_context`) and re-stamps it on the two outbound
paths that reach inference-router:

  * `get_identity_headers()` — the CLI-agent / remote-container path.
  * `make_identity_event_hook`'s httpx request hook — the pydantic-ai /
    inference-router path, installed on BOTH `InferenceRouterOpenAIProvider` and
    `ModelFactory._create_vllm_model` (one shared hook implementation; see
    `test_the_identity_hook_is_defined_once_not_copied` in
    test_outbound_tenant_is_real.py).

These tests pin three properties, modelled on the X-App-Id analogue in
test_app_id_outbound.py:

  1. Set ⇒ relayed, on every path.
  2. ⛔ OMIT-WHEN-EMPTY. Unset/None/"" ⇒ the header is ABSENT, never present
     with an empty value. A present-but-empty header is a PRESENT header: the
     callee's `Header(default=...)` cannot fire, so downstream it reads as an
     asserted-but-empty project rather than "no project was named". An absent
     header is honest; an empty one is a claim that the project is nothing.
  3. ⛔ NOT VALIDATED HERE. A malformed value is relayed byte-for-byte.
     inference-router uuid-validates X-Project-Id at its own header-reading site
     and drops it when malformed; a second validator here would be a second
     place for the two services to disagree, and WM would then be silently
     dropping values IR would have accepted (or vice versa).
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.identity_context import (
    get_identity_headers,
    get_project_id_from_context,
    set_app_id_context,
    set_correlation_context,
    set_identity_context,
    set_project_id_context,
)
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIProvider,
    ModelFactory,
    PydanticAIAgentConfig,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)

PROJECT_ID = "aaaaaaaa-0000-4000-8000-0000000000a1"
TENANT = "aaaaaaaa-0000-4000-8000-000000000001"

# Deliberately not a uuid. This is the regression guard's input: WM must relay
# it unchanged rather than validating/dropping it here.
MALFORMED_PROJECT_ID = "not-a-uuid"


def _reset_context() -> None:
    set_identity_context("", "", "")
    set_correlation_context("", "")
    set_app_id_context("")
    set_project_id_context("")


@pytest.fixture(autouse=True)
def _clean_identity_context():
    """ContextVars are task-local, but a test that sets one would otherwise leak
    it into the next test running in the same context. Reset on both sides."""
    _reset_context()
    yield
    _reset_context()


def _ir_provider_hook(provider: InferenceRouterOpenAIProvider) -> object:
    """Pull the single outbound request event hook off the provider's client."""
    http_client = provider._client._client  # AsyncOpenAI -> httpx.AsyncClient
    hooks = http_client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    return hooks[0]


def _vllm_hooks(cfg: PydanticAIAgentConfig) -> list:
    model = ModelFactory._create_vllm_model(cfg)
    client = getattr(model._provider, "_http_client", None) or getattr(
        model._provider, "http_client", None
    )
    if client is None or not getattr(client, "event_hooks", {}).get("request"):
        client = model._provider.client._client
    hooks = client.event_hooks["request"]
    assert hooks, "vllm client should install the shared identity hook"
    return hooks


def _vllm_cfg() -> PydanticAIAgentConfig:
    return PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm",
        model="local-llm",
        db_path="",
        base_url="http://ir:8150/v1",
        tenant_id="tenant-1",
    )


# --------------------------------------- get_identity_headers (CLI-agent path)


def test_identity_headers_include_project_id_when_context_is_set() -> None:
    """A caller-stamped project id reaches the CLI-agent outbound header dict."""
    set_project_id_context(PROJECT_ID)
    assert get_identity_headers()["X-Project-Id"] == PROJECT_ID


@pytest.mark.parametrize("unset_value", ["", None], ids=["empty-string", "none"])
def test_identity_headers_omit_project_id_when_unset(unset_value: str | None) -> None:
    """⛔ THE LOAD-BEARING ASSERTION IN THIS FILE.

    No project in context ⇒ the key is ABSENT from the dict, not present with an
    empty value. `Header(default=...)` downstream only fires on absence, so an
    empty string would be read as "the project is nothing" instead of "no
    project was named" — strictly worse than sending nothing at all.
    """
    set_identity_context(TENANT, "lic-1", "subj-1")
    set_project_id_context(unset_value)
    headers = get_identity_headers()
    assert "X-Project-Id" not in headers
    # Not vacuous: the helper did run and did emit the neighbouring headers.
    assert headers["X-Tenant-Id"] == TENANT


def test_identity_headers_omit_project_id_when_the_contextvar_was_never_set() -> None:
    """The ContextVar's own default must be omit-worthy, not a placeholder."""
    assert get_project_id_from_context() == ""
    assert "X-Project-Id" not in get_identity_headers()


# --------------------------------------- inbound: middleware → ContextVar


def _probe_app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, str]:
        return {"ctx_project_id": get_project_id_from_context()}

    return application


def _probe(headers: dict[str, str]) -> str:
    client = TestClient(_probe_app(), raise_server_exceptions=False)
    with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
        resp = client.get("/api/probe", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["ctx_project_id"]


@pytest.mark.parametrize(
    "header_name",
    ["X-Project-Id", "x-project-id", "X-PROJECT-ID"],
    ids=["canonical", "lowercase", "uppercase"],
)
def test_middleware_reads_inbound_project_id_case_insensitively(header_name: str) -> None:
    """HTTP header names are case-insensitive, so the relay must be too — a
    caller spelling it `x-project-id` must not silently lose its attribution."""
    assert (
        _probe({"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", header_name: PROJECT_ID})
        == PROJECT_ID
    )


def test_middleware_leaves_project_id_empty_when_the_header_is_absent() -> None:
    """No inbound header ⇒ empty ContextVar. worker_management never invents a
    project id: it fans in from many upstream callers and a guessed value would
    collapse their distinct projects into one bucket on every cost event."""
    assert _probe({"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003"}) == ""


# --------------------------------------- outbound: the pydantic-ai httpx hook


@pytest.mark.asyncio
async def test_ir_provider_forwards_project_id_from_context() -> None:
    """A project id in context is stamped on the outbound inference-router request."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    set_project_id_context(PROJECT_ID)
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hook(req)
    assert req.headers["X-Project-Id"] == PROJECT_ID


@pytest.mark.asyncio
async def test_ir_provider_omits_project_id_when_context_empty() -> None:
    """⛔ Same omit-when-empty rule on the hook path: the key is not added at
    all. This hook learned the empty-header lesson the expensive way for
    X-Tenant-Id; X-Project-Id must not relearn it."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hook(req)
    assert "X-Project-Id" not in req.headers


def test_vllm_client_forwards_project_id_from_context() -> None:
    """Same shared hook, same contract, on the second installation site.

    This is the path `principle_consistency` actually uses (VLLM_URL is
    repointed at inference-router — see test_outbound_tenant_is_real.py). A
    second outbound path that dropped the project id would attribute the same
    run two different ways depending on which transport it took.
    """
    hooks = _vllm_hooks(_vllm_cfg())
    set_project_id_context(PROJECT_ID)
    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    for hook in hooks:
        asyncio.run(hook(req))
    assert req.headers.get("X-Project-Id") == PROJECT_ID


def test_vllm_client_omits_project_id_when_context_empty() -> None:
    hooks = _vllm_hooks(_vllm_cfg())
    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    for hook in hooks:
        asyncio.run(hook(req))
    assert "X-Project-Id" not in req.headers


# --------------------------------------- ⛔ regression guard: relayed, not validated


@pytest.mark.asyncio
async def test_a_malformed_project_id_is_relayed_verbatim_on_every_path() -> None:
    """⛔ DO NOT ADD A UUID VALIDATOR HERE, AND DO NOT LOOSEN THIS TEST.

    inference-router validates X-Project-Id at its own header-reading site and
    treats a malformed value as absent. A second validator in worker_management
    is a second place for the two services to disagree: the moment their rules
    drift, WM starts dropping values IR would have accepted (or laundering ones
    it would have rejected), and the disagreement is invisible because both
    sides look internally consistent. WM relays; IR decides.

    Asserted on the full chain — inbound header, both outbound paths — because a
    validator could be introduced at any one of the three.
    """
    # Inbound relay.
    assert _probe(
        {
            "X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003",
            "X-Project-Id": MALFORMED_PROJECT_ID,
        }
    ) == (MALFORMED_PROJECT_ID)

    # CLI-agent path.
    set_project_id_context(MALFORMED_PROJECT_ID)
    assert get_identity_headers()["X-Project-Id"] == MALFORMED_PROJECT_ID

    # pydantic-ai / inference-router hook path.
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hook(req)
    assert req.headers["X-Project-Id"] == MALFORMED_PROJECT_ID
