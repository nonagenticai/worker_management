"""Per-member attribution — inbound X-Actor-Id → ContextVar → every outbound path.

`X-Actor-Id` is WHO ACTED (the Keycloak `sub` of the person who initiated the
work), distinct from `X-Subject-Id` (WHO PAYS). inference-router reads it at
`routers/completions.py` (`Header(alias="X-Actor-Id")`) and stamps it on the
cost event as `actor_id`; billing-service persists that field
(`services/cost_flusher.py`, `c.get("actor_id")`) into
`billing_cost_events.actor_id` and rolls it up per member in
`billing_cost_actor_daily`.

worker_management used to relay X-Tenant-Id / X-Project-Id / X-Spend-Scope but
dropped X-Actor-Id on the floor, so every LLM call it brokered landed on the
NULL actor and per-member billing could not attribute it.

Pinned here, on every path that already forwards X-Project-Id:
  1. inbound middleware relays the header (name case-insensitive) ONLY from a
     delegating subject or the trusted network; an end-user licence's is dropped;
  2. `get_identity_headers` (CLI-agent / remote-container path) forwards it;
  3. the pydantic-ai inference-router httpx hook forwards it;
  4. the BYO cost event this service emits itself carries `actor_id`;
  5. ⛔ NEVER DEFAULTED TO THE SUBJECT. An authenticated EXTERNAL request
     without a valid actor stays ACTOR-LESS downstream (owner decision
     2026-10-07: filling it would hide the caller's gap from enforcement).
     Only WM-originated work (no inbound request) uses WM's own
     ``SERVICE_ACTOR_ID``; with that unset too, the header is omitted.
  6. a trusted caller's value must be a canonical non-nil UUID — anything
     else is dropped and counted on worker_management_actor_missing_total.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import ACTOR_MISSING_TOTAL, AuthenticationMiddleware
from src.api.identity_context import (
    get_actor_id_from_context,
    get_effective_actor_id,
    get_identity_headers,
    set_actor_id_context,
    set_identity_context,
    set_project_id_context,
)
from src.settings import reset_settings_cache
from src.telemetry import cost_events
from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)

TENANT = "0195f48b-d000-7000-8000-000000000001"
SUBJECT = "0195f48b-d000-7000-8000-0000000000b2"
ACTOR = "0195f48b-d000-7000-8000-0000000000c3"
PROJECT_ID = "0195f48b-d000-7000-8000-0000000000a1"


SERVICE_ACTOR = "0195f48b-d000-7000-8000-0000000000d4"


def _reset() -> None:
    set_identity_context("", "", "")
    set_project_id_context("")
    set_actor_id_context("")


@pytest.fixture(autouse=True)
def _clean_identity_context(monkeypatch):
    # No SERVICE_ACTOR_ID unless a test sets one: the omit-when-empty legs below
    # must not depend on the developer's environment.
    monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    reset_settings_cache()
    _reset()
    yield
    _reset()
    reset_settings_cache()


@pytest.fixture
def service_actor(monkeypatch):
    monkeypatch.setenv("SERVICE_ACTOR_ID", SERVICE_ACTOR)
    reset_settings_cache()
    return SERVICE_ACTOR


# ------------------------------------------------ inbound (trust-gated)
#
# ⛔ X-Actor-Id is honoured ONLY from a caller the existing trust model already
# lets assert identity on someone else's behalf: a licence whose introspected
# subject is in AUTH_DELEGATING_SUBJECTS, or the tokenless TRUSTED_CIDRS branch.
# A non-delegating licence holder (an end user) cannot pick whose name their
# spend lands under — review 5331893773 on PR #300.

DELEGATOR = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
END_USER = "0195f48b-d000-7000-8000-0000000000e5"


def _app(sink: list | None = None) -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, object]:
        if sink is not None:
            # The REAL caller path through cost_events.emit_byo_cost_event_task,
            # not `_emit` — so the call site's own ContextVar read is what is
            # pinned (mutating it to `actor_id=""` must fail a test).
            cost_events.emit_byo_cost_event_task(
                provider="anthropic",
                model="m",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                model_provider="anthropic",
            )
            await asyncio.gather(*list(cost_events._background_tasks))
        return {
            "actor": get_actor_id_from_context(),
            "outbound": get_identity_headers().get("X-Actor-Id", ""),
            "effective": get_effective_actor_id(),
        }

    return application


def _probe_trusted_network(headers: dict[str, str]) -> str:
    return _probe_trusted_network_json(headers)["actor"]


def _probe_trusted_network_json(headers: dict[str, str]) -> dict:
    client = TestClient(_app(), raise_server_exceptions=False)
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=True),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        resp = client.get("/api/probe", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _probe_token(monkeypatch, subject: str, sink: list | None = None) -> str:
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", DELEGATOR)
    client = TestClient(_app(sink), raise_server_exceptions=False)
    with (
        # Not on a trusted network: the ONLY thing that can admit the header is
        # the licence subject being a configured delegator.
        patch("src.api.auth_middleware._is_trusted_network", return_value=False),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "tenant_id": TENANT,
                "subject_id": subject,
                "byo_mode": "local_allowed",
            },
        ),
    ):
        resp = client.get("/api/probe", headers={"X-Tenant-Id": "Bearer tok", "X-Actor-Id": ACTOR})
    assert resp.status_code == 200, resp.text
    return resp.json()["actor"]


@pytest.mark.parametrize("name", ["X-Actor-Id", "x-actor-id"])
def test_trusted_network_caller_actor_is_relayed(name: str) -> None:
    assert _probe_trusted_network({"X-Tenant-Id": TENANT, name: ACTOR}) == ACTOR


def test_middleware_leaves_actor_empty_when_absent() -> None:
    assert _probe_trusted_network({"X-Tenant-Id": TENANT}) == ""


def test_delegating_subject_actor_is_relayed(monkeypatch) -> None:
    assert _probe_token(monkeypatch, DELEGATOR) == ACTOR


def test_end_user_licence_actor_is_dropped_not_defaulted(monkeypatch) -> None:
    # Dropped, and ⛔ not replaced by the caller's own subject either.
    assert _probe_token(monkeypatch, END_USER) == ""


def test_untokened_untrusted_caller_never_reaches_a_relay() -> None:
    client = TestClient(_app(), raise_server_exceptions=False)
    with patch("src.api.auth_middleware._is_trusted_network", return_value=False):
        resp = client.get("/api/probe", headers={"X-Tenant-Id": TENANT, "X-Actor-Id": ACTOR})
    assert resp.status_code == 401


# ------------------------------------------------ get_identity_headers


def test_identity_headers_forward_actor_alongside_tenant_and_project() -> None:
    set_identity_context(TENANT, "", SUBJECT)
    set_project_id_context(PROJECT_ID)
    set_actor_id_context(ACTOR)
    headers = get_identity_headers(worker_name="w")
    assert headers["X-Tenant-Id"] == TENANT
    assert headers["X-Project-Id"] == PROJECT_ID
    assert headers["X-Actor-Id"] == ACTOR


def test_identity_headers_omit_actor_when_empty_and_never_default_to_subject() -> None:
    set_identity_context(TENANT, "", SUBJECT)
    headers = get_identity_headers()
    assert "X-Actor-Id" not in headers


# ------------------------------------------------ pydantic-ai IR hook


async def _hook_headers() -> httpx.Headers:
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hooks = provider._client._client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hooks[0](req)
    return req.headers


@pytest.mark.asyncio
async def test_ir_hook_forwards_actor_alongside_tenant_and_project() -> None:
    set_identity_context(TENANT, "", SUBJECT)
    set_project_id_context(PROJECT_ID)
    set_actor_id_context(ACTOR)
    headers = await _hook_headers()
    assert headers["X-Tenant-Id"] == TENANT
    assert headers["X-Project-Id"] == PROJECT_ID
    assert headers["X-Actor-Id"] == ACTOR


@pytest.mark.asyncio
async def test_ir_hook_omits_actor_when_empty() -> None:
    set_identity_context(TENANT, "", SUBJECT)
    headers = await _hook_headers()
    assert "X-Actor-Id" not in headers


# ------------------------------------------------ BYO cost event


def _metering(monkeypatch, sink: list) -> None:
    monkeypatch.setattr(
        cost_events,
        "get_settings",
        lambda: SimpleNamespace(
            metering_redis_url="redis://metering:6379/0",
            metering_costs_stream="improvement-loop:costs",
        ),
    )
    redis_client = AsyncMock()

    async def _xadd(stream, fields, **kwargs):
        sink.append(dict(fields))

    redis_client.xadd.side_effect = _xadd
    monkeypatch.setattr(cost_events, "_get_redis", lambda: redis_client)


def test_byo_cost_event_carries_delegated_actor_end_to_end(monkeypatch) -> None:
    """Inbound header -> middleware -> emit_byo_cost_event_task -> XADD fields."""
    sink: list = []
    _metering(monkeypatch, sink)
    assert _probe_token(monkeypatch, DELEGATOR, sink) == ACTOR
    assert len(sink) == 1
    assert sink[0]["actor_id"] == ACTOR


def test_byo_cost_event_omits_end_user_asserted_actor(monkeypatch) -> None:
    sink: list = []
    _metering(monkeypatch, sink)
    _probe_token(monkeypatch, END_USER, sink)
    assert len(sink) == 1
    assert sink[0]["actor_id"] == ""


# ------------------------------------------------ validation (malformed dropped)


def _missing(caller: str = "unknown") -> float:
    return ACTOR_MISSING_TOTAL.labels(caller=caller)._value.get()


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",  # nil sentinel is "nobody"
        "{0195f48b-d000-7000-8000-0000000000c3}",  # braces: uuid.UUID() accepts it
        "0195f48bd00070008000" + "0000000000c3",  # bare hex
        "urn:uuid:0195f48b-d000-7000-8000-0000000000c3",
        "0195f48b-d000-7000-8000-0000000000c3x",
    ],
)
def test_malformed_actor_from_trusted_caller_is_dropped_and_counted(bad: str) -> None:
    before = _missing()
    assert _probe_trusted_network({"X-Tenant-Id": TENANT, "X-Actor-Id": bad}) == ""
    assert _missing() == before + 1


def test_valid_actor_is_forwarded_unchanged_not_normalised() -> None:
    upper = ACTOR.upper()
    before = _missing()
    assert _probe_trusted_network({"X-Tenant-Id": TENANT, "X-Actor-Id": upper}) == upper
    assert _missing() == before  # an accepted actor is not a miss


def test_absent_actor_is_counted_by_caller_app_id() -> None:
    before = _missing("n8n")
    _probe_trusted_network({"X-Tenant-Id": TENANT, "X-App-Id": "n8n"})
    assert _missing("n8n") == before + 1


def test_untrusted_assertion_is_counted_as_missing(monkeypatch) -> None:
    before = _missing()
    assert _probe_token(monkeypatch, END_USER) == ""
    assert _missing() == before + 1


# ------------------------------------------------ fallback scope (owner decision)
#
# The SPLIT: an authenticated external request without a valid actor stays
# actor-less (counted, never filled); only WM-originated work gets
# SERVICE_ACTOR_ID.


@pytest.mark.parametrize("actor_header", [None, "garbage"])
def test_external_request_without_actor_stays_actorless_even_with_service_actor(
    service_actor, actor_header
) -> None:
    headers = {"X-Tenant-Id": TENANT}
    if actor_header is not None:
        headers["X-Actor-Id"] = actor_header
    before = _missing()
    body = _probe_trusted_network_json(headers)
    assert body == {"actor": "", "outbound": "", "effective": ""}
    assert _missing() == before + 1


def test_external_end_user_licence_stays_actorless_even_with_service_actor(
    monkeypatch, service_actor
) -> None:
    assert _probe_token(monkeypatch, END_USER) == ""


def test_external_request_with_actor_forwards_it_not_service_actor(service_actor) -> None:
    body = _probe_trusted_network_json({"X-Tenant-Id": TENANT, "X-Actor-Id": ACTOR})
    assert body == {"actor": ACTOR, "outbound": ACTOR, "effective": ACTOR}


def test_wm_originated_work_uses_service_actor_but_external_does_not(service_actor) -> None:
    """Same process, same settings: fresh context (WM-originated) → service
    actor; external request context → nothing."""
    set_identity_context(TENANT, "", SUBJECT)
    assert get_identity_headers()["X-Actor-Id"] == service_actor
    assert _probe_trusted_network_json({"X-Tenant-Id": TENANT})["outbound"] == ""


# ------------------------------------------------ fallback (WM-originated work)


def test_background_work_uses_service_actor_everywhere(service_actor) -> None:
    """No inbound request at all (fresh context): every outbound path carries
    WM's own service-account sub, never the subject."""
    set_identity_context(TENANT, "", SUBJECT)
    assert get_effective_actor_id() == service_actor
    assert get_identity_headers()["X-Actor-Id"] == service_actor


@pytest.mark.asyncio
async def test_ir_hook_background_uses_service_actor(service_actor) -> None:
    set_identity_context(TENANT, "", SUBJECT)
    headers = await _hook_headers()
    assert headers["X-Actor-Id"] == service_actor
    assert headers["X-Actor-Id"] != SUBJECT


@pytest.mark.asyncio
async def test_ir_hook_inbound_actor_wins_over_service_actor(service_actor) -> None:
    set_identity_context(TENANT, "", SUBJECT)
    set_actor_id_context(ACTOR)
    headers = await _hook_headers()
    assert headers["X-Actor-Id"] == ACTOR


@pytest.mark.asyncio
async def test_listing_headers_carry_effective_actor(service_actor) -> None:
    from src.services.ir_bundle_tokens import ir_listing_headers

    headers = await ir_listing_headers(f"{IR_URL}/api/models", TENANT)
    assert headers["X-Actor-Id"] == service_actor
    set_actor_id_context(ACTOR)
    headers = await ir_listing_headers(f"{IR_URL}/api/models", TENANT)
    assert headers["X-Actor-Id"] == ACTOR


@pytest.mark.asyncio
async def test_worker_run_task_inherits_the_triggering_callers_actor(service_actor) -> None:
    """A worker run spawned from a request (asyncio task) carries the CALLING
    service's actor, not WM's fallback."""
    set_identity_context(TENANT, "", SUBJECT)
    set_actor_id_context(ACTOR)

    async def run() -> str:
        return (await _hook_headers())["X-Actor-Id"]

    assert await asyncio.create_task(run()) == ACTOR


def test_malformed_service_actor_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", "wm-service-account")
    reset_settings_cache()
    set_identity_context(TENANT, "", SUBJECT)
    assert get_effective_actor_id() == ""
    assert "X-Actor-Id" not in get_identity_headers()


def test_byo_cost_event_external_without_actor_stays_actorless(monkeypatch, service_actor) -> None:
    sink: list = []
    _metering(monkeypatch, sink)
    _probe_token(monkeypatch, END_USER, sink)
    assert sink[0]["actor_id"] == ""


@pytest.mark.asyncio
async def test_byo_cost_event_wm_originated_carries_service_actor(
    monkeypatch, service_actor
) -> None:
    sink: list = []
    _metering(monkeypatch, sink)
    set_identity_context(TENANT, "", SUBJECT)  # tenant known; NO inbound request
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    await asyncio.gather(*list(cost_events._background_tasks))
    assert sink[0]["actor_id"] == service_actor
