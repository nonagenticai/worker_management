"""Spend-scope attribution — inbound X-Spend-Scope → ContextVar → both outbound paths.

`X-Spend-Scope` is the REPORTING-ONLY sibling of `X-Project-Id`. Its single job
is to let billing tell two things apart that are indistinguishable today:

  * `project` — this spend belongs to a project. If no `X-Project-Id` arrives
    with it, that is a genuine attribution FAILURE worth chasing.
  * `tenant` — this spend is tenant-wide work BY DESIGN and there is nothing to
    attribute it to. A permanent, CORRECT residual.

Without the axis both land in one bucket, so the permanent residual reads as a
defect nobody can ever close. worker_management is a RELAY for it and nothing
else — it reads the inbound header into a ContextVar
(`auth_middleware._stash_identity_context`) and re-stamps it on the two outbound
paths that reach inference-router:

  * `get_identity_headers()` — the CLI-agent / remote-container path.
  * `make_identity_event_hook`'s httpx request hook — the pydantic-ai /
    inference-router path, installed on BOTH `InferenceRouterOpenAIProvider` and
    `ModelFactory._create_vllm_model`.

These tests pin four properties, modelled on the X-Project-Id analogue in
test_project_id_outbound.py:

  1. Both valid values relay end-to-end, on every path.
  2. ⛔ VALIDATED HERE, closed set of exactly two. Anything else — wrong case,
     a near-miss word, a uuid, whitespace — is DROPPED and logged, never
     forwarded and ⛔ never coerced to a default. This is the ONE deliberate
     divergence from X-Project-Id, which is relayed unvalidated because it is an
     opaque id whose only authority is inference-router's uuid check. A scope is
     a closed vocabulary reaching a report dimension; there is nothing for two
     services to disagree about, and coercion would manufacture
     "tenant-wide by design" out of "we do not know" — which is precisely the
     confusion this header was added to END.
  3. ⛔ OMIT-WHEN-EMPTY. Absent ⇒ absent downstream. Saying nothing about scope
     stays distinct from saying `tenant`.
  4. ⛔ ORTHOGONAL TO THE PROJECT ID. A request carrying `X-Spend-Scope: tenant`
     AND a valid `X-Project-Id` forwards BOTH. The scope is reporting-only and
     must never suppress, gate or otherwise move the id — a wrong or forged
     scope may at worst mislabel a denominator, never move money. That
     containment is the whole reason this design is safe, so it is asserted in
     both directions and on every path.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api import auth_middleware as auth_middleware_module
from src.api import identity_context as identity_context_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.identity_context import (
    SPEND_SCOPE_PROJECT,
    SPEND_SCOPE_TENANT,
    SPEND_SCOPES,
    get_identity_headers,
    get_project_id_from_context,
    get_spend_scope_from_context,
    set_app_id_context,
    set_correlation_context,
    set_identity_context,
    set_project_id_context,
    set_spend_scope_context,
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

# Every one of these must be DROPPED, not coerced. Chosen to cover the distinct
# ways a caller gets this wrong rather than to pad the count:
#   - case variants: the validator is exact and case-SENSITIVE on purpose, so a
#     normalising `.lower()` (the first step onto the coercion slope) fails here.
#   - `both` / `none` / `global`: plausible near-miss vocabulary an author would
#     reach for; each must be refused rather than absorbed into the closed set.
#   - a uuid: the shape of the SIBLING header, i.e. the exact value a caller
#     swapping the two headers would send. Must not be relayed as a scope.
#   - whitespace-padded `tenant`: no trimming, for the same reason as no
#     lowercasing — normalising is not this service's job and it hides a caller
#     bug that its own logs should show.
INVALID_SCOPES = [
    "TENANT",
    "Tenant",
    "PROJECT",
    "both",
    "none",
    "global",
    "aaaaaaaa-0000-4000-8000-0000000000a1",
    " tenant ",
    "tenant,project",
]


def _reset_context() -> None:
    set_identity_context("", "", "")
    set_correlation_context("", "")
    set_app_id_context("")
    set_project_id_context("")
    set_spend_scope_context("")


@pytest.fixture(autouse=True)
def _clear_spend_scope_log_dedup():
    """Clear the invalid-scope WARNING dedup cache around every test.

    ⛔ Without this the suite would pass for the WRONG REASON. The cache is
    module-global with a 300s TTL, so the second test to drop the same value in
    one run observes "no warning" and a test asserting the warning fires would
    go red, while a test asserting it does NOT fire would go green having proved
    nothing. Mirrors `auth_middleware._assertion_log_seen`'s own fixture in
    tests/test_unauthenticated_tenant_assertion.py.
    """
    identity_context_module._spend_scope_log_seen.clear()
    yield
    identity_context_module._spend_scope_log_seen.clear()


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


async def _hook_headers(scope: str | None, project_id: str = "") -> httpx.Headers:
    """Run the pydantic-ai outbound hook with this context and return its headers."""
    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, api_key=API_KEY)
    hook = _ir_provider_hook(provider)
    set_spend_scope_context(scope)
    set_project_id_context(project_id)
    req = httpx.Request("POST", f"{IR_URL}/chat/completions")
    await hook(req)
    return req.headers


# --------------------------------------- the closed set itself


def test_the_valid_scope_vocabulary_is_exactly_two_values() -> None:
    """⛔ `tenant` and `project` are the ENTIRE universe.

    Pinned as its own assertion because every other test in this file is
    parametrised over this set: widening it would silently widen them too and
    they would keep passing. A third value needs this line changed on purpose,
    with the reporting semantics of the new value written down.
    """
    assert {"tenant", "project"} == SPEND_SCOPES
    assert (SPEND_SCOPE_TENANT, SPEND_SCOPE_PROJECT) == ("tenant", "project")


# --------------------------------------- get_identity_headers (CLI-agent path)


@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
def test_identity_headers_include_a_valid_scope(scope: str) -> None:
    """Both valid values reach the CLI-agent outbound header dict, unchanged."""
    set_spend_scope_context(scope)
    assert get_identity_headers()["X-Spend-Scope"] == scope


@pytest.mark.parametrize("unset_value", ["", None], ids=["empty-string", "none"])
def test_identity_headers_omit_scope_when_unset(unset_value: str | None) -> None:
    """⛔ OMIT-WHEN-EMPTY, same rule as every sibling header.

    No scope in context ⇒ the key is ABSENT, not present with an empty value. An
    empty header is a PRESENT header: the callee's `Header(default=...)` cannot
    fire, so downstream it reads as an asserted-but-blank scope rather than
    "nobody stated one" — and a blank scope is not a reporting bucket anyone can
    act on. Omission must stay possible.
    """
    set_identity_context(TENANT, "lic-1", "subj-1")
    set_spend_scope_context(unset_value)
    headers = get_identity_headers()
    assert "X-Spend-Scope" not in headers
    # Not vacuous: the helper did run and did emit the neighbouring headers.
    assert headers["X-Tenant-Id"] == TENANT


def test_identity_headers_omit_scope_when_the_contextvar_was_never_set() -> None:
    """The ContextVar's own default must be omit-worthy, not a placeholder."""
    assert get_spend_scope_from_context() == ""
    assert "X-Spend-Scope" not in get_identity_headers()


# --------------------------------------- inbound: middleware → ContextVar


def _probe_app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, str]:
        return {
            "ctx_spend_scope": get_spend_scope_from_context(),
            "ctx_project_id": get_project_id_from_context(),
        }

    return application


def _probe(headers: dict[str, str]) -> dict[str, str]:
    client = TestClient(_probe_app(), raise_server_exceptions=False)
    with patch("src.api.auth_middleware._is_trusted_network", return_value=True):
        resp = client.get("/api/probe", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize(
    "header_name",
    ["X-Spend-Scope", "x-spend-scope", "X-SPEND-SCOPE"],
    ids=["canonical", "lowercase", "uppercase"],
)
def test_middleware_reads_the_inbound_header_name_case_insensitively(header_name: str) -> None:
    """HTTP header NAMES are case-insensitive, so the relay must be too.

    ⚠️ Deliberately NOT in tension with the case-SENSITIVE value check below:
    the name is an HTTP-level identifier the spec says is case-insensitive, the
    VALUE is a closed application vocabulary. Conflating the two rules is how a
    `.lower()` ends up on the value.
    """
    got = _probe(
        {"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", header_name: SPEND_SCOPE_TENANT}
    )
    assert got["ctx_spend_scope"] == SPEND_SCOPE_TENANT


def test_middleware_leaves_the_scope_empty_when_the_header_is_absent() -> None:
    """No inbound header ⇒ empty ContextVar. worker_management never invents a
    scope: a caller that says nothing has NOT said `tenant`."""
    assert _probe({"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003"})["ctx_spend_scope"] == ""


@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
def test_middleware_relays_both_valid_scopes_inbound(scope: str) -> None:
    got = _probe({"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", "X-Spend-Scope": scope})
    assert got["ctx_spend_scope"] == scope


# --------------------------------------- outbound: the pydantic-ai httpx hook


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
async def test_ir_provider_forwards_a_valid_scope(scope: str) -> None:
    """A valid scope in context is stamped on the outbound inference-router request."""
    assert (await _hook_headers(scope))["X-Spend-Scope"] == scope


@pytest.mark.asyncio
async def test_ir_provider_omits_the_scope_when_context_empty() -> None:
    """⛔ Same omit-when-empty rule on the hook path: the key is not added at all."""
    assert "X-Spend-Scope" not in await _hook_headers("")


@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
def test_vllm_client_forwards_a_valid_scope(scope: str) -> None:
    """Same shared hook, same contract, on the SECOND installation site.

    A second outbound path that dropped the scope would label the same run two
    different ways depending on which transport it happened to take — which is
    the failure mode the whole axis exists to remove.
    """
    hooks = _vllm_hooks(_vllm_cfg())
    set_spend_scope_context(scope)
    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    for hook in hooks:
        asyncio.run(hook(req))
    assert req.headers.get("X-Spend-Scope") == scope


def test_vllm_client_omits_the_scope_when_context_empty() -> None:
    hooks = _vllm_hooks(_vllm_cfg())
    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    for hook in hooks:
        asyncio.run(hook(req))
    assert "X-Spend-Scope" not in req.headers


# --------------------------------------- ⛔ invalid ⇒ dropped, logged, NOT defaulted


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", INVALID_SCOPES)
async def test_an_invalid_scope_is_dropped_on_every_path(bad: str) -> None:
    """⛔ DROPPED — not forwarded, not coerced — on the full chain.

    Asserted inbound AND on both outbound paths because a coercion could be
    introduced at any one of the three, and a test covering only the setter
    would not see a hook that defaulted the value on its way out.

    The assertion is deliberately `not in headers` rather than
    `!= bad`: `!= bad` would also pass if the value had been silently coerced to
    `tenant`, which is the specific failure this test exists to catch. Dropping
    means the call lands in the honest unknown bucket, visible and
    reconcilable — coercion would mint "tenant-wide by design" out of "we do not
    know" and make the residual permanently unclosable again.
    """
    # Inbound relay.
    assert (
        _probe({"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", "X-Spend-Scope": bad})[
            "ctx_spend_scope"
        ]
        == ""
    )

    # CLI-agent path.
    set_spend_scope_context(bad)
    assert get_spend_scope_from_context() == ""
    assert "X-Spend-Scope" not in get_identity_headers()

    # pydantic-ai / inference-router hook path.
    assert "X-Spend-Scope" not in await _hook_headers(bad)


@pytest.mark.parametrize("bad", INVALID_SCOPES)
def test_an_invalid_scope_is_logged_with_its_value(bad: str, caplog) -> None:
    """Dropping SILENTLY would be a second bug: the caller's request succeeds
    either way (the header is reporting-only), so the log line is the ONLY
    signal that a caller is misconfigured. Pin that it fires, that it is a
    WARNING, and that it carries the offending value — a log saying only
    "invalid scope" names nothing to fix.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context(bad)
    records = [r for r in caplog.records if "X-Spend-Scope" in r.getMessage()]
    assert records, f"dropping {bad!r} must be logged"
    assert records[0].levelno == logging.WARNING
    assert repr(bad) in records[0].getMessage()


def _scope_warnings(caplog) -> list:
    """Every record this module emitted about the header, in emission order."""
    return [r for r in caplog.records if "X-Spend-Scope" in r.getMessage()]


def test_a_long_invalid_scope_is_truncated_in_the_log(caplog) -> None:
    """⛔ THE BOUND THAT MATTERS IS VOLUME, NOT CARDINALITY.

    The value is CALLER-CONTROLLED and there is no length limit on an HTTP
    header worth relying on, so an untruncated `%r` writes as many bytes as the
    caller chose into the single rotating file `kubectl logs` serves — evicting
    every other workload's logs in the pod. One misconfigured caller is enough.

    Asserted as "the full value is ABSENT and the 64-char prefix is PRESENT"
    rather than a length check on the whole message: the message also carries
    the fixed explanatory prose, so a length assertion would drift with any
    wording change while saying nothing about the caller-controlled part.
    """
    bad = "z" * 4000
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context(bad)
    records = _scope_warnings(caplog)
    assert records, "dropping an over-long scope must still be logged"
    message = records[0].getMessage()
    limit = identity_context_module._SPEND_SCOPE_LOG_MAX_CHARS
    assert repr(bad) not in message, "the untruncated caller value reached the log"
    assert repr(bad[:limit]) in message
    # Not vacuous about WHERE it was cut: 64 chars in, not at some other bound.
    assert repr(bad[: limit + 1]) not in message
    # And the drop itself is unaffected by the truncation.
    assert get_spend_scope_from_context() == ""


def test_the_same_invalid_scope_logs_only_once_inside_the_ttl_window(caplog) -> None:
    """⛔ RATE, the second half of the bound.

    This fires once per REQUEST, so truncation alone caps the line width and
    leaves the line COUNT unbounded — a caller looping on a bad header still
    floods the file. Deduping through the TTL cache is what turns "per request"
    into "once per distinct value per 300s", which is the same trade
    `auth_middleware._assertion_log_seen` already makes for the same hazard.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        for _ in range(5):
            set_spend_scope_context("both")
    assert len(_scope_warnings(caplog)) == 1
    # Suppressing the LOG must never suppress the DROP: every one of the five
    # calls still refused the value.
    assert get_spend_scope_from_context() == ""


def test_dedup_is_per_value_not_a_global_mute(caplog) -> None:
    """⛔ NEGATIVE CONTROL for the test above, and the one that fails if the
    dedup is ever widened into a global "warned recently, stay quiet" flag.

    A second, DIFFERENT misconfigured caller must still be named. A throttle
    that hid it would leave the first bad value permanently masking every other
    one for 300s at a time — the log would become a report on whichever caller
    happened to break first, which is worse than no log at all because it reads
    as complete.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context("both")
        set_spend_scope_context("global")
    messages = [r.getMessage() for r in _scope_warnings(caplog)]
    assert len(messages) == 2
    assert repr("both") in messages[0]
    assert repr("global") in messages[1]


def test_the_dedup_key_is_the_truncated_value(caplog) -> None:
    """The cache is keyed on what was LOGGED, not on what the caller sent.

    Deliberate, and the reason is the same bound: keying on the raw value would
    let one caller pin up to `maxsize` arbitrary-length caller-controlled
    strings in the process's memory, moving the unbounded-bytes problem out of
    the log file and into the heap. The visible consequence — two distinct
    values sharing a 64-char prefix are reported once — is acceptable because
    the prefix alone already names the caller's bug.
    """
    limit = identity_context_module._SPEND_SCOPE_LOG_MAX_CHARS
    shared = "q" * limit
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context(shared + "-first")
        set_spend_scope_context(shared + "-second")
    assert len(_scope_warnings(caplog)) == 1


def test_the_log_bound_reuses_the_auth_middleware_numbers() -> None:
    """The source comment says these are `_assertion_log_seen`'s numbers ON
    PURPOSE, so the two sites cannot drift into different answers to the same
    question. A comment cannot hold that; this can.
    """
    ours = identity_context_module._spend_scope_log_seen
    theirs = auth_middleware_module._assertion_log_seen
    assert (ours.maxsize, ours.ttl) == (theirs.maxsize, theirs.ttl)


def test_a_valid_scope_is_not_logged_as_invalid(caplog) -> None:
    """Negative control for the test directly above.

    Without it, a setter that logged the warning UNCONDITIONALLY would pass
    every drop-is-logged assertion while being pure noise.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context(SPEND_SCOPE_TENANT)
        set_spend_scope_context(SPEND_SCOPE_PROJECT)
    assert [r for r in caplog.records if "X-Spend-Scope" in r.getMessage()] == []


def test_an_absent_scope_is_not_logged_as_invalid(caplog) -> None:
    """Omission is a legitimate answer, not a caller bug. Logging it would make
    the warning fire on the majority of fleet traffic and bury the real ones."""
    with caplog.at_level(logging.WARNING, logger="src.api.identity_context"):
        set_spend_scope_context("")
        set_spend_scope_context(None)
    assert [r for r in caplog.records if "X-Spend-Scope" in r.getMessage()] == []


# --------------------------------------- ⛔ orthogonality: scope never moves the id


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
async def test_scope_and_project_id_are_forwarded_together(scope: str) -> None:
    """⛔ THE LOAD-BEARING ASSERTION IN THIS FILE.

    `X-Spend-Scope` is REPORTING-ONLY. A request carrying a scope AND a valid
    project id must forward BOTH, on every path — including `tenant`, the value
    most likely to be misread as "there is no project here, drop the id".

    If the scope were ever allowed to suppress the id, a header with no
    provenance story would be deciding what gets ATTRIBUTED, and a forged value
    would move money instead of merely mislabelling a denominator. That
    containment is the entire safety argument for shipping this header, so it is
    pinned here rather than left to the comments asserting it.
    """
    # Inbound: both survive the middleware, independently.
    got = _probe(
        {
            "X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003",
            "X-Spend-Scope": scope,
            "X-Project-Id": PROJECT_ID,
        }
    )
    assert got == {"ctx_spend_scope": scope, "ctx_project_id": PROJECT_ID}

    # CLI-agent path: both keys present, both values intact.
    set_spend_scope_context(scope)
    set_project_id_context(PROJECT_ID)
    headers = get_identity_headers()
    assert headers["X-Spend-Scope"] == scope
    assert headers["X-Project-Id"] == PROJECT_ID

    # pydantic-ai / inference-router hook path.
    hook_headers = await _hook_headers(scope, PROJECT_ID)
    assert hook_headers["X-Spend-Scope"] == scope
    assert hook_headers["X-Project-Id"] == PROJECT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["TENANT", "both", ""])
async def test_an_invalid_or_absent_scope_never_suppresses_the_project_id(bad: str) -> None:
    """The other direction of orthogonality, and the one a naive implementation
    breaks: dropping the scope must drop ONLY the scope.

    A validator written as "if the scope is unusable, this call is
    unattributable" would take the project id down with it and turn a reporting
    header into an attribution outage for every misconfigured caller.
    """
    assert _probe(
        {
            "X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003",
            "X-Spend-Scope": bad,
            "X-Project-Id": PROJECT_ID,
        }
    ) == {"ctx_spend_scope": "", "ctx_project_id": PROJECT_ID}

    hook_headers = await _hook_headers(bad, PROJECT_ID)
    assert "X-Spend-Scope" not in hook_headers
    assert hook_headers["X-Project-Id"] == PROJECT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", sorted(SPEND_SCOPES))
async def test_a_scope_alone_does_not_invent_a_project_id(scope: str) -> None:
    """The mirror case: a scope with NO project id must not conjure one.

    `X-Spend-Scope: project` in particular asserts that a project exists; it must
    not be read as licence to default, guess or back-fill the id. A `project`
    scope arriving with no id is exactly the attribution FAILURE this axis was
    built to make visible — laundering it into a value would re-hide it.
    """
    assert _probe(
        {"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", "X-Spend-Scope": scope}
    ) == {
        "ctx_spend_scope": scope,
        "ctx_project_id": "",
    }

    hook_headers = await _hook_headers(scope)
    assert hook_headers["X-Spend-Scope"] == scope
    assert "X-Project-Id" not in hook_headers
