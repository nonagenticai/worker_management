"""worker_management never calls inference-router on its own behalf.

Owner ruling 2026-10-07 (binding):

1. Every WM -> inference-router call acts for a calling app with a real END
   tenant. No end tenant ⇒ the call is REFUSED before anything is sent (typed
   ``IRTenantRequiredError``, ERROR log, ``wm_ir_tenantless_call_refused_total``).
   There is no WM / platform service-tenant fallback any more.
2. WM never fabricates ``X-Project-Id`` (not the tenant id, not a value derived
   from it). An inbound project id is forwarded verbatim; a missing one is
   forwarded as absent and counted per inbound caller
   (``wm_ir_call_missing_project_total{caller=...}``).

Requests are sent through a real ``httpx.AsyncClient`` with a ``MockTransport``
standing in for the router, so "no router call" is counted at the wire.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest

from src.api.identity_context import (
    set_app_id_context,
    set_caller_context,
    set_identity_context,
    set_project_id_context,
)
from src.services.ir_bundle_tokens import (
    TENANTLESS_IR_CALL_REFUSED_TOTAL,
    IRTenantRequiredError,
)
from src.settings import reset_settings_cache
from src.worker.agents.pydantic_ai_agent import (
    IR_CALL_MISSING_PROJECT_TOTAL,
    make_identity_event_hook,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150"
COMPLETIONS = f"{IR_URL}/v1/chat/completions"
TENANT = "aaaaaaaa-0000-4000-8000-000000000100"
PROJECT = "aaaaaaaa-0000-4000-8000-0000000000a1"
NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_ROUTER_URL", IR_URL)
    # Flag off: these tests are about tenant + project, not the licence mint.
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "false")
    reset_settings_cache()
    set_identity_context(None, None, None)
    set_project_id_context("")
    set_app_id_context("")
    set_caller_context("", ())
    yield
    set_identity_context(None, None, None)
    set_project_id_context("")
    set_app_id_context("")
    set_caller_context("", ())
    reset_settings_cache()


def _hook(fallback_tenant: str = ""):
    return make_identity_event_hook(
        worker_name="wm-test-worker",
        worker_version="1",
        fallback_tenant_id=fallback_tenant,
        fallback_license_id="",
        fallback_subject_id="",
    )


async def _send(hook) -> list[httpx.Request]:
    reached: list[httpx.Request] = []

    def _router(request: httpx.Request) -> httpx.Response:
        reached.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_router), event_hooks={"request": [hook]}
    ) as client:
        await client.post(COMPLETIONS, json={})
    return reached


# ------------------------------------------------------------- (1) no tenant


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant", ["", NIL], ids=["no-tenant", "nil-sentinel"])
async def test_completion_without_an_end_tenant_is_refused_and_never_sent(tenant: str) -> None:
    set_identity_context(tenant, None, None)
    counter = TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(call="completion", worker="wm-test-worker")
    before = counter._value.get()
    reached: list[httpx.Request] = []
    with pytest.raises(IRTenantRequiredError) as exc:
        reached = await _send(_hook())
    assert reached == []
    assert counter._value.get() == before + 1
    assert exc.value.status_code == 403
    assert exc.value.error_code == "END_TENANT_REQUIRED"


@pytest.mark.asyncio
async def test_completion_with_an_end_tenant_is_sent_as_that_tenant() -> None:
    """Control for the refusal above: same hook, real tenant ⇒ one router call."""
    set_identity_context(TENANT, None, None)
    reached = await _send(_hook())
    assert len(reached) == 1
    assert reached[0].headers["X-Tenant-Id"] == TENANT


def test_every_api_create_worker_call_passes_the_end_tenant() -> None:
    """``create_worker`` validates the model against inference-router AS the
    tenant it is given. Every API call site must pass the caller's end tenant
    (crud POST/PUT/auto-create, message auto-create, the worker cache);
    a call without ``tenant_id=`` would now be a refused tenantless listing."""
    root = Path(__file__).resolve().parents[1] / "src" / "api"
    calls: list[tuple[str, int, bool]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "create_worker"
            ):
                has = any(kw.arg == "tenant_id" for kw in node.keywords)
                calls.append((path.name, node.lineno, has))
    assert len(calls) >= 5, calls  # not vacuous: the known call sites were found
    missing = [c for c in calls if not c[2]]
    assert missing == [], missing


# --------------------------------------------------------- (2) X-Project-Id


@pytest.mark.asyncio
async def test_inbound_project_is_forwarded_verbatim() -> None:
    set_identity_context(TENANT, None, None)
    set_project_id_context(PROJECT)
    reached = await _send(_hook())
    assert reached[0].headers["X-Project-Id"] == PROJECT


@pytest.mark.asyncio
async def test_no_project_is_ever_fabricated_and_the_caller_is_counted() -> None:
    """No inbound project ⇒ NO X-Project-Id (not the tenant, not a derived id),
    and the miss is counted against the inbound caller's verified client id."""
    set_identity_context(TENANT, None, None)
    set_caller_context("svc-kag-karma", ())
    set_app_id_context("kag")
    counter = IR_CALL_MISSING_PROJECT_TOTAL.labels(caller="svc-kag-karma")
    before = counter._value.get()
    reached = await _send(_hook())
    assert "X-Project-Id" not in reached[0].headers
    assert counter._value.get() == before + 1


@pytest.mark.asyncio
async def test_missing_project_falls_back_to_app_id_then_unknown() -> None:
    set_identity_context(TENANT, None, None)
    set_app_id_context("asset-evaluator")
    by_app = IR_CALL_MISSING_PROJECT_TOTAL.labels(caller="asset-evaluator")
    unknown = IR_CALL_MISSING_PROJECT_TOTAL.labels(caller="unknown")
    app_before, unknown_before = by_app._value.get(), unknown._value.get()
    await _send(_hook())
    set_app_id_context("")
    await _send(_hook())
    assert by_app._value.get() == app_before + 1
    assert unknown._value.get() == unknown_before + 1
