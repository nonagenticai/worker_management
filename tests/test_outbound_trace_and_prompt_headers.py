"""AA-TRACE join keys + prompt attribution on the worker_management → inference-router hop.

Spec AA-TRACE §2/§3/§6.5-7: every WM → inference-router call must

  1. forward X-AA-Trace-Id, X-AA-Card-Id, X-Project-Id, X-AA-Parent-Exec and
     X-AA-Hop UNCHANGED (exact case, byte-for-byte) from the inbound request; and
  2. stamp X-AA-Prompt-Version: <prompt_id>@<n> and X-AA-Prompt-Sha256: <hex>
     so inference-router fills `inference_call.prompt_version_id` /
     `prompt_body_sha256` — and OMIT both when unknown, so an unversioned call
     lands as NULL and is counted rather than laundered.

⛔ worker_management OWNS this hop's forwarding test. The ship-tenancy lane only
covers n8n → service; nothing else ever looks at the WM → IR leg.

Inbound capture: `AuthenticationMiddleware._stash_identity_context`.
Outbound: `make_identity_event_hook` (pydantic-ai / IR path) and
`get_identity_headers` (CLI-agent path), both via
`identity_context.get_trace_and_prompt_headers`.
"""

from __future__ import annotations

import asyncio
import contextvars
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.identity_context import (
    get_identity_headers,
    set_aa_trace_context,
    set_identity_context,
    set_project_id_context,
    set_prompt_attribution,
)
from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
API_KEY = "test"  # pragma: allowlist secret  (dummy key for provider construction)
TENANT = "aaaaaaaa-0000-4000-8000-000000000003"

# Deliberately awkward values: not uuids, mixed case, padding-free punctuation —
# anything a "helpful" normaliser or validator would touch.
TRACE_HEADERS: dict[str, str] = {
    "X-AA-Trace-Id": "Tr4ce-aaaaaaaa/abc:DEF",
    "X-AA-Card-Id": "card_42.Mixed-Case",
    "X-Project-Id": "not-a-uuid-project",
    "X-AA-Parent-Exec": "n8n-exec-529894",
    "X-AA-Hop": "3",
}
PROMPT_HEADERS = ("X-AA-Prompt-Version", "X-AA-Prompt-Sha256")
SHA = "ab" * 32


def _reset() -> None:
    set_identity_context("", "", "")
    set_project_id_context("")
    set_aa_trace_context({})
    set_prompt_attribution(None, None)


@pytest.fixture(autouse=True)
def _clean_context():
    _reset()
    yield
    _reset()


def _ir_hook():
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hooks = provider._client._client._event_hooks.get("request", [])
    assert hooks, "request event hook should be installed"
    return hooks[0]


async def _outbound_headers() -> httpx.Headers:
    """Run the REAL inference-router hook in the current context."""
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await _ir_hook()(req)
    return req.headers


def _probe_app() -> FastAPI:
    """Inbound request → middleware → endpoint that performs the outbound hook.

    The endpoint optionally calls `set_prompt_attribution` (as the worker LLM
    path does) and returns what the real IR hook stamped, so the test exercises
    inbound capture and outbound stamping in the SAME request context.
    """
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request, pv: str | None = None, sha: str | None = None) -> dict:
        if pv is not None or sha is not None:
            set_prompt_attribution(pv, sha)
        out = await _outbound_headers()
        return {
            "hook": {k: v for k, v in out.items() if k.lower().startswith(("x-aa-", "x-project"))},
            "cli": {
                k: v
                for k, v in get_identity_headers().items()
                if k.lower().startswith(("x-aa-", "x-project"))
            },
        }

    return application


def _probe(headers: dict[str, str], **params: str) -> dict:
    client = TestClient(_probe_app(), raise_server_exceptions=False)
    with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
        resp = client.get("/api/probe", headers={"X-Tenant-Id": TENANT, **headers}, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _exact_case(raw: dict[str, str]) -> dict[str, str]:
    """httpx lower-cases on .items(); map back to canonical names for comparison."""
    canon = {k.lower(): k for k in (*TRACE_HEADERS, *PROMPT_HEADERS)}
    return {canon.get(k.lower(), k): v for k, v in raw.items()}


# --------------------------------------------------------------------- (a)


def test_all_five_inbound_trace_headers_reach_inference_router_unchanged() -> None:
    """(a) Inbound carries all 5 ⇒ the outbound IR request carries all 5, identical."""
    got = _probe(TRACE_HEADERS)
    assert _exact_case(got["hook"]) == TRACE_HEADERS
    # Same contract on the CLI-agent transport: one shared builder, no drift.
    assert _exact_case(got["cli"]) == TRACE_HEADERS


@pytest.mark.asyncio
async def test_trace_headers_are_stamped_in_exact_spec_case() -> None:
    """The wire name is the spec's exact case (httpx preserves the raw name)."""
    set_aa_trace_context(TRACE_HEADERS)
    set_project_id_context(TRACE_HEADERS["X-Project-Id"])
    out = await _outbound_headers()
    raw_names = {name.decode() for name, _ in out.raw}
    assert set(TRACE_HEADERS) <= raw_names


# --------------------------------------------------------------------- (b) (c)


@pytest.mark.asyncio
async def test_prompt_attribution_set_stamps_both_prompt_headers() -> None:
    """(b) set_prompt_attribution("worker-x@2", sha) ⇒ both headers present."""
    set_prompt_attribution("worker-x@2", SHA)
    out = await _outbound_headers()
    assert out["X-AA-Prompt-Version"] == "worker-x@2"
    assert out["X-AA-Prompt-Sha256"] == SHA
    assert get_identity_headers()["X-AA-Prompt-Version"] == "worker-x@2"


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", [None, ""], ids=["none", "empty-string"])
async def test_prompt_attribution_none_omits_both_prompt_headers(unset: str | None) -> None:
    """(c) None ⇒ NEITHER header present (not empty) — the call stores NULL."""
    set_identity_context(TENANT, "lic-1", "subj-1")
    set_prompt_attribution(unset, unset)
    out = await _outbound_headers()
    for name in PROMPT_HEADERS:
        assert name not in out
        assert name not in get_identity_headers()
    # Not vacuous: the hook ran and stamped its neighbours.
    assert out["X-Tenant-Id"] == TENANT


# --------------------------------------------------------------------- (d)


def test_no_leak_between_two_sequential_requests() -> None:
    """(d) Request 1 carries trace headers and sets a prompt; request 2 carries
    neither and sets nothing ⇒ request 2's outbound call carries none of them."""
    first = _probe(TRACE_HEADERS, pv="worker-x@2", sha=SHA)
    assert _exact_case(first["hook"]) == {
        **TRACE_HEADERS,
        "X-AA-Prompt-Version": "worker-x@2",
        "X-AA-Prompt-Sha256": SHA,
    }
    second = _probe({})
    assert second["hook"] == {}
    assert second["cli"] == {}


def test_no_leak_between_contexts_on_one_shared_hook() -> None:
    """(d) The cached provider is shared across requests; two contexts using the
    SAME hook instance each see only their own attribution."""
    hook = _ir_hook()

    async def call() -> httpx.Headers:
        req = httpx.Request("POST", f"{IR_URL}/chat/completions")
        await hook(req)
        return req.headers

    def in_ctx_a() -> httpx.Headers:
        set_aa_trace_context({"X-AA-Trace-Id": "trace-a"})
        set_prompt_attribution("worker-a@1", SHA)
        return asyncio.run(call())

    def in_ctx_b() -> httpx.Headers:
        return asyncio.run(call())

    a = contextvars.copy_context().run(in_ctx_a)
    b = contextvars.copy_context().run(in_ctx_b)
    assert a["X-AA-Trace-Id"] == "trace-a"
    assert a["X-AA-Prompt-Version"] == "worker-a@1"
    for name in ("X-AA-Trace-Id", *PROMPT_HEADERS):
        assert name not in b


# --------------------------------------------------------------------- (e)


@pytest.mark.parametrize("missing", list(TRACE_HEADERS))
def test_a_header_missing_inbound_is_absent_outbound(missing: str) -> None:
    """(e) Omit-when-absent: the dropped header is ABSENT outbound, never "";
    the other four still arrive unchanged."""
    inbound = {k: v for k, v in TRACE_HEADERS.items() if k != missing}
    got = _probe(inbound)
    hook = _exact_case(got["hook"])
    assert missing not in hook
    assert hook == inbound
    assert missing not in _exact_case(got["cli"])


def test_an_empty_inbound_header_is_absent_outbound() -> None:
    """(e) A present-but-empty inbound header is treated as absent, not relayed as ""."""
    got = _probe({**TRACE_HEADERS, "X-AA-Card-Id": ""})
    assert "X-AA-Card-Id" not in _exact_case(got["hook"])
