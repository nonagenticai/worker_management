"""Usage per project/user split by funding (D1/D3) and actor trust (D4).

Pinned here:
  * every EXTERNAL call emits a cost event — including the platform-paid ones
    (owner env key) that used to emit NOTHING (unmetered spend);
  * each event carries funding, project_id (validated like inference-router),
    job_id (inbound X-Job-Id, else the worker) and a key fingerprint — never
    the key;
  * platform-routed refusals are counted; mlx / vlm_local ride inference-router
    with the full identity hook;
  * a non-delegating licence may assert X-Actor-Id only for its own SA sub or a
    member of its own tenant; everything else is dropped and counted;
  * licence 401s are counted per reason.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from src.api import identity_context as ic
from src.api.auth_middleware import AuthenticationMiddleware
from src.auth import actor_membership
from src.settings import reset_settings_cache
from src.telemetry import attribution_metrics as am
from src.telemetry import cost_events
from src.worker.agents.pydantic_ai_agent import PydanticAIAgent, PydanticAIAgentConfig

AGENT_MODULE = "src.worker.agents.pydantic_ai_agent"
TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
OTHER_TENANT = "aaaaaaaa-0000-4000-8000-000000000002"
PROJECT = "aaaaaaaa-0000-4000-8000-0000000000a1"
OWN_SA_SUB = "aaaaaaaa-0000-4000-8000-0000000000f1"
MEMBER = "aaaaaaaa-0000-4000-8000-0000000000c3"
LICENCE_SUBJECT = "aaaaaaaa-0000-4000-8000-0000000000b2"


def _val(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    reset_settings_cache()
    actor_membership.reset_actor_membership_cache()
    ic.set_identity_context("", "", "")
    ic.set_project_id_context("")
    ic.set_job_id_context("")
    ic.set_actor_id_context("")
    yield
    ic.set_identity_context("", "", "")
    ic.set_project_id_context("")
    ic.set_job_id_context("")
    ic.set_actor_id_context("")
    actor_membership.reset_actor_membership_cache()


def _metering(monkeypatch) -> list[dict]:
    sink: list[dict] = []
    monkeypatch.setattr(
        cost_events,
        "get_settings",
        lambda: SimpleNamespace(
            metering_redis_url="redis://metering:6379/0",
            metering_costs_stream="improvement-loop:costs",
        ),
    )
    client = AsyncMock()

    async def _xadd(stream, fields, **kwargs):
        sink.append(dict(fields))

    client.xadd.side_effect = _xadd
    monkeypatch.setattr(cost_events, "_get_redis", lambda: client)
    return sink


async def _drain() -> None:
    await asyncio.gather(*list(cost_events._background_tasks))


# ------------------------------------------------------------------ funding


@pytest.mark.parametrize(
    ("provider", "tenant_key", "owner_provided", "expected"),
    [
        ("xai", True, False, "tenant_byo"),
        ("xai", True, True, "platform_external"),  # managed BYO: owner's key
        ("xai", False, False, "platform_external"),  # owner env key
        ("anthropic", False, True, "platform_external"),
        ("inference_router", False, False, "local_gpu"),
        ("mlx", False, False, "local_gpu"),
        ("vlm_local", True, False, "local_gpu"),
    ],
)
def test_funding_matrix(provider, tenant_key, owner_provided, expected) -> None:
    assert (
        cost_events.funding_for(
            model_provider=provider, tenant_byo_key=tenant_key, owner_provided=owner_provided
        )
        == expected
    )


def test_key_fingerprint_is_stable_and_never_the_key() -> None:
    blob = b"\x01\x02ciphertext-of-sk-live-123"
    fp = cost_events.key_fingerprint(blob)
    assert fp == cost_events.key_fingerprint(blob)
    assert len(fp) == 16
    assert "sk-live" not in fp
    assert cost_events.key_fingerprint(b"other-version") != fp
    assert cost_events.key_fingerprint(None) == ""


@pytest.mark.asyncio
async def test_event_carries_funding_project_job_and_key_fp(monkeypatch) -> None:
    sink = _metering(monkeypatch)
    ic.set_identity_context(TENANT, "lic-1", "sub-1")
    ic.set_project_id_context(PROJECT)
    ic.set_job_id_context("n8n-exec-42")
    before = _val(am.TENANT_KEY_USE_TOTAL, tenant=TENANT, role="tier:high")
    cost_events.emit_byo_cost_event_task(
        provider="xai",
        model="grok",
        input_tokens=3,
        output_tokens=4,
        latency_ms=5,
        model_provider="xai",
        worker_name="karma_extractor",
        funding="tenant_byo",
        key_fp="abcd1234abcd1234",
        key_role="tier:high",
    )
    await _drain()
    (ev,) = sink
    assert ev["funding"] == "tenant_byo"
    assert ev["project_id"] == PROJECT
    assert ev["job_id"] == "n8n-exec-42"
    ux = json.loads(ev["usage_extra"])
    assert ux["byo"] == "true"
    assert ux["funding"] == "tenant_byo"
    assert ux["key_fp"] == "abcd1234abcd1234"
    assert _val(am.TENANT_KEY_USE_TOTAL, tenant=TENANT, role="tier:high") == before + 1


@pytest.mark.asyncio
async def test_event_job_falls_back_to_worker_and_bad_project_is_absent(monkeypatch) -> None:
    sink = _metering(monkeypatch)
    ic.set_identity_context(TENANT, "lic-1", "sub-1")
    ic.set_project_id_context("not-a-uuid")  # IR's rule: malformed => absent
    ic.set_job_id_context("bad job id with spaces")  # dropped at the setter
    cost_events.emit_byo_cost_event_task(
        provider="xai",
        model="grok",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="xai",
        worker_name="karma_extractor",
        funding="platform_external",
    )
    await _drain()
    (ev,) = sink
    assert ev["project_id"] == ""
    assert ev["job_id"] == "karma_extractor"
    ux = json.loads(ev["usage_extra"])
    assert ux["owner_provided"] == "true"
    assert "byo" not in ux
    assert "key_fp" not in ux


def test_platform_routed_refusal_is_counted(monkeypatch) -> None:
    sink = _metering(monkeypatch)
    ic.set_identity_context(TENANT, "lic-1", "sub-1")
    before = _val(am.COST_EMIT_REFUSED_TOTAL, reason="platform_routed")
    cost_events.emit_byo_cost_event_task(
        provider="pgx-local",
        model="local-llm",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="mlx",
    )
    assert sink == []
    assert _val(am.COST_EMIT_REFUSED_TOTAL, reason="platform_routed") == before + 1


# ------------------------------------------- platform-paid external calls EMIT


def _run_result() -> MagicMock:
    message = MagicMock()
    message.provider_details = None
    usage = MagicMock()
    usage.has_values.return_value = True
    usage.input_tokens = 11
    usage.output_tokens = 22
    usage.total_tokens = 33
    result = MagicMock()
    result.output = "ok"
    result.all_messages.return_value = [message]
    result.usage.return_value = usage
    return result


async def _agent_run(**cfg) -> MagicMock:
    config = PydanticAIAgentConfig(name="ext_worker", model="grok-4", db_path="", **cfg)
    with (
        patch(f"{AGENT_MODULE}.PydanticAIAgent._initialize_model", return_value=MagicMock()),
        patch(f"{AGENT_MODULE}.Agent"),
    ):
        agent = PydanticAIAgent(config=config)
    agent.agent.run = AsyncMock(return_value=_run_result())
    with (
        patch(f"{AGENT_MODULE}.get_metrics_client", return_value=MagicMock()),
        patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit,
    ):
        await agent.run("hello")
    return emit


@pytest.mark.asyncio
async def test_owner_env_key_call_emits_platform_external() -> None:
    """⭐ The unmetered-spend bug: no cost_source (owner env key) => no event."""
    emit = await _agent_run(model_provider="xai", tenant_byo_key=False, cost_source=None)
    emit.assert_called_once()
    kw = emit.call_args.kwargs
    assert kw["funding"] == "platform_external"
    assert kw["provider"] == "xai"
    assert kw["key_fp"] == ""


@pytest.mark.asyncio
async def test_tenant_key_call_emits_tenant_byo_with_fingerprint() -> None:
    emit = await _agent_run(
        model_provider="xai",
        tenant_byo_key=True,
        cost_source="worker-management",
        key_fingerprint="ffff0000ffff0000",
        byo_role="__default__",
    )
    kw = emit.call_args.kwargs
    assert kw["funding"] == "tenant_byo"
    assert kw["key_fp"] == "ffff0000ffff0000"
    assert kw["key_role"] == "__default__"


def test_chat_router_platform_paid_external_call_emits(monkeypatch) -> None:
    from src.api.providers import _chat_router as cr

    req = SimpleNamespace(provider="xai", model="grok-4")
    usage = SimpleNamespace(input_tokens=5, output_tokens=6)
    monkeypatch.setattr(cr.ResponseExtractor, "extract_served_fields", staticmethod(lambda _r: {}))
    monkeypatch.setattr(cr, "_resolved_from_result", lambda _r: (None, None))
    with patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit:
        cr._emit_byo_cost(req, None, object(), usage, 0.1)
    emit.assert_called_once()
    assert emit.call_args.kwargs["funding"] == "platform_external"


def test_chat_router_platform_routed_call_emits_nothing(monkeypatch) -> None:
    from src.api.providers import _chat_router as cr

    req = SimpleNamespace(provider="inference_router", model="local-llm")
    usage = SimpleNamespace(input_tokens=5, output_tokens=6)
    with patch("src.telemetry.cost_events.emit_byo_cost_event_task") as emit:
        cr._emit_byo_cost(req, None, object(), usage, 0.1)
    emit.assert_not_called()


# ------------------------------------------------ mlx rides the IR identity hook


@pytest.mark.asyncio
async def test_mlx_client_sends_identity_headers(monkeypatch) -> None:
    from src.worker.agents.pydantic_ai_agent import ModelFactory

    cfg = PydanticAIAgentConfig(
        name="mlx_worker",
        model_provider="mlx",
        model="qwen",
        db_path="",
        base_url="http://mlx-dev.invalid:8150",
        tenant_id=TENANT,
    )
    model = ModelFactory._create_mlx_model(cfg)
    http_client = model._provider._client._client  # MLXAsyncOpenAI -> httpx client
    ic.set_identity_context(TENANT, "lic-1", "sub-1")
    ic.set_app_id_context("cdmsfirst")
    ic.set_project_id_context(PROJECT)
    ic.set_actor_id_context(MEMBER)
    ic.set_job_id_context("exec-7")
    req = httpx.Request("POST", "http://mlx-dev.invalid:8150/v1/chat/completions")
    for hook in http_client.event_hooks["request"]:
        await hook(req)
    assert req.headers["X-Tenant-Id"] == TENANT
    assert req.headers["X-License-Id"] == "lic-1"
    assert req.headers["X-App-Id"] == "cdmsfirst"
    assert req.headers["X-Project-Id"] == PROJECT
    assert req.headers["X-Actor-Id"] == MEMBER
    assert req.headers["X-Job-Id"] == "exec-7"
    assert http_client.headers["X-Worker-Name"] == "mlx_worker"


# ---------------------------------------------------------------- actor trust


def _app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(_request: Request) -> dict[str, str]:
        return {"actor": ic.get_actor_id_from_context()}

    return application


def _licence_probe(actor: str, *, own_sub: str | None = OWN_SA_SUB) -> str:
    client = TestClient(_app(), raise_server_exceptions=False)
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=False),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "tenant_id": TENANT,
                "subject_id": LICENCE_SUBJECT,
                "subject_keycloak_sub": own_sub,
                "byo_mode": "local_allowed",
            },
        ),
    ):
        resp = client.get("/api/probe", headers={"X-Tenant-Id": "Bearer tok", "X-Actor-Id": actor})
    assert resp.status_code == 200, resp.text
    return resp.json()["actor"]


def _membership(monkeypatch, answers: dict[str, tuple[bool, str | None]]) -> list[str]:
    calls: list[str] = []

    async def _fetch(sub: str):
        calls.append(sub)
        return answers.get(sub, (True, None))

    monkeypatch.setattr(actor_membership, "_fetch_member_tenant", _fetch)
    return calls


def test_a_licence_may_assert_its_own_service_account_sub(monkeypatch) -> None:
    calls = _membership(monkeypatch, {})
    assert _licence_probe(OWN_SA_SUB) == OWN_SA_SUB
    assert calls == []  # no lookup needed: introspection already proved it


def test_b_licence_may_assert_a_member_of_its_own_tenant_cached(monkeypatch) -> None:
    calls = _membership(monkeypatch, {MEMBER: (True, TENANT)})
    assert _licence_probe(MEMBER) == MEMBER
    assert _licence_probe(MEMBER) == MEMBER
    assert calls == [MEMBER]  # second request served from cache


def test_cross_tenant_actor_is_rejected_and_counted(monkeypatch) -> None:
    _membership(monkeypatch, {MEMBER: (True, OTHER_TENANT)})
    before = _val(am.ACTOR_REJECTED_TOTAL, reason="cross_tenant")
    assert _licence_probe(MEMBER) == ""
    assert _val(am.ACTOR_REJECTED_TOTAL, reason="cross_tenant") == before + 1


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ((True, None), "unknown_subject"),
        ((True, ""), "tenant_undecided"),
        ((False, None), "lookup_unavailable"),
    ],
)
def test_unprovable_membership_is_rejected(monkeypatch, answer, reason) -> None:
    _membership(monkeypatch, {MEMBER: answer})
    before = _val(am.ACTOR_REJECTED_TOTAL, reason=reason)
    assert _licence_probe(MEMBER) == ""
    assert _val(am.ACTOR_REJECTED_TOTAL, reason=reason) == before + 1


@pytest.mark.parametrize("bad", ["not-a-uuid", "00000000-0000-0000-0000-000000000000"])
def test_malformed_actor_is_rejected_and_counted(monkeypatch, bad) -> None:
    calls = _membership(monkeypatch, {})
    before = _val(am.ACTOR_REJECTED_TOTAL, reason="malformed")
    assert _licence_probe(bad) == ""
    assert _val(am.ACTOR_REJECTED_TOTAL, reason="malformed") == before + 1
    assert calls == []


def test_unconfigured_lookup_fails_closed_without_network() -> None:
    """No credential in the test env => lookup_unavailable, actor dropped."""
    before = _val(am.ACTOR_REJECTED_TOTAL, reason="lookup_unavailable")
    assert _licence_probe(MEMBER, own_sub=None) == ""
    assert _val(am.ACTOR_REJECTED_TOTAL, reason="lookup_unavailable") == before + 1


def test_rejection_is_logged_at_info(monkeypatch, caplog) -> None:
    _membership(monkeypatch, {MEMBER: (True, OTHER_TENANT)})
    with caplog.at_level("INFO", logger="src.api.auth_middleware"):
        _licence_probe(MEMBER)
    hits = [r for r in caplog.records if "X-Actor-Id rejected" in r.getMessage()]
    assert hits and hits[0].levelname == "INFO"
    assert "cross_tenant" in hits[0].getMessage()


# --------------------------------------------------------------- auth refusals


def test_missing_token_401_is_counted_and_logged_without_token(caplog) -> None:
    client = TestClient(_app(), raise_server_exceptions=False)
    before = _val(am.AUTH_REFUSED_TOTAL, reason="missing_token")
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=False),
        caplog.at_level("INFO", logger="src.api.auth_middleware"),
    ):
        resp = client.get("/api/probe")
    assert resp.status_code == 401
    assert _val(am.AUTH_REFUSED_TOTAL, reason="missing_token") == before + 1
    assert any("auth_refused reason=missing_token" in r.getMessage() for r in caplog.records)


def test_inactive_licence_401_is_counted_and_never_logs_the_token(caplog) -> None:
    client = TestClient(_app(), raise_server_exceptions=False)
    before = _val(am.AUTH_REFUSED_TOTAL, reason="license_inactive")
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=False),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={"active": False, "reason": "token_not_found"},
        ),
        caplog.at_level("INFO", logger="src.api.auth_middleware"),
    ):
        resp = client.get("/api/probe", headers={"X-Tenant-Id": "Bearer s3cret-tok"})
    assert resp.status_code == 401
    assert _val(am.AUTH_REFUSED_TOTAL, reason="license_inactive") == before + 1
    lines = [r.getMessage() for r in caplog.records if "auth_refused" in r.getMessage()]
    assert lines and "token_not_found" in lines[0]
    assert all("s3cret-tok" not in line for line in lines)


# ------------------------------------------------------------------ preinit


def test_new_metrics_are_preinitialised() -> None:
    am.preinit_attribution_metrics()
    text = generate_latest().decode()
    for series in (
        'wm_cost_emit_refused_total{reason="platform_routed"}',
        'wm_cost_event_funding_total{funding="platform_external"}',
        'worker_management_actor_rejected_total{reason="cross_tenant"}',
        'worker_management_auth_refused_total{reason="license_inactive"}',
        'wm_tenant_key_use_total{role="__default__",tenant="none"}',
    ):
        assert series in text, series
