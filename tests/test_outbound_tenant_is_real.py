"""An outbound inference-router call must carry a REAL tenant, or none at all.

WHY THIS FILE EXISTS
--------------------
Measured 2026-08-27 against the live estate: 1,400 of 13,893 inference-router
routed calls in 24h (10.1%) arrived with the nil tenant sentinel, 1,086 of them
from the worker ``principle_consistency``. Every one of those dead-lettered in
billing with ``reason=nil_tenant_id``.

The tenant was never missing. ``requirements_management`` holds a real
``tenant_id`` UUID on the row it is reconciling and forwards it
(``src/agents/base.py``). It died inside worker_management, in two places at
once:

1. Nothing on the factory chain carried ``tenant_id``. ``_globals.py`` resolved
   it (it is the worker cache key) and then called ``create_worker()`` without
   it, so ``PydanticAIAgentConfig.tenant_id`` -- a field that already existed
   and was already wired to the provider -- was always ``None``.
2. The provider put ``X-Tenant-Id`` into its client-level default headers
   *unconditionally*, so an unresolved tenant went out as the EMPTY STRING. An
   empty header is a present header, so inference-router's
   ``Header(default=...)`` could not fire; IR coerced ``""`` to the nil
   sentinel instead.

Both halves are asserted here. The second matters independently of the first:
without it, any future gap in the chain degrades silently again instead of
logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import patch

import httpx2 as httpx
import pytest

MODULE = "src.worker.agents.pydantic_ai_agent"
REAL_TENANT = "e148e8af-3751-49bf-a0fa-56ab46d31ddc"
NIL_TENANT = "00000000-0000-0000-0000-000000000000"


def _build_provider(**kwargs):
    """Construct the provider with its two external dependencies stubbed.

    Returns (provider, default_headers, event_hook).
    """
    from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

    with (
        patch(f"{MODULE}.OpenAIProvider.__init__", return_value=None),
        patch(f"{MODULE}.AsyncOpenAI") as mock_openai,
    ):
        provider = InferenceRouterOpenAIProvider.__new__(InferenceRouterOpenAIProvider)
        provider.__init__(
            base_url="http://ir:8150/api",
            api_key="internal",  # pragma: allowlist secret
            **kwargs,
        )
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs["default_headers"]
        hook = call_kwargs["http_client"].event_hooks["request"][0]
        return provider, headers, hook


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://ir:8150/api/chat/completions")


# ---------------------------------------------------------------- construction


def test_a_real_tenant_is_sent_as_a_default_header():
    _, headers, _ = _build_provider(tenant_id=REAL_TENANT, worker_name="principle_consistency")
    assert headers["X-Tenant-Id"] == REAL_TENANT


def test_an_unresolved_tenant_is_ABSENT_not_empty():
    """THE DEFECT. This is the assertion the old code could not pass.

    The previous shape wrote ``"X-Tenant-Id": self._fallback_tenant_id`` into
    the dict unconditionally, so this key was present with the value ``""``.
    """
    _, headers, _ = _build_provider(worker_name="principle_consistency")
    assert "X-Tenant-Id" not in headers, (
        f"an empty identity header is a claim that the tenant is nothing; got {headers!r}"
    )
    # X-Worker-Name is NOT identity and must stay unconditional -- it is what
    # let the live investigation name principle_consistency at all.
    assert headers["X-Worker-Name"] == "principle_consistency"


def test_license_and_subject_follow_the_same_rule():
    _, headers, _ = _build_provider(tenant_id=REAL_TENANT)
    assert "X-License-Id" not in headers
    assert "X-Subject-Id" not in headers


# ------------------------------------------------------------------ the hook


def _run_hook(hook, request):
    asyncio.run(hook(request))
    return request


def test_hook_forwards_the_constructor_tenant_when_no_contextvar():
    """The background/reconciler path: no request context, real baked tenant."""
    _, _, hook = _build_provider(tenant_id=REAL_TENANT, worker_name="principle_consistency")
    req = _request()
    req.headers["X-Tenant-Id"] = REAL_TENANT  # as the default headers would
    _run_hook(hook, req)
    assert req.headers["X-Tenant-Id"] == REAL_TENANT


def test_hook_strips_an_empty_tenant_header_and_logs_loudly(caplog):
    """Belt and braces: even if something upstream sets it empty, it is removed.

    The log assertion is deliberately on a stable token (``outbound_tenant_
    unresolved``) rather than prose, so rewording the message cannot silently
    void this test.
    """
    _, _, hook = _build_provider(worker_name="principle_consistency")
    req = _request()
    req.headers["X-Tenant-Id"] = ""
    with caplog.at_level(logging.ERROR, logger="src.worker.agents.pydantic_ai_agent"):
        _run_hook(hook, req)
    assert "X-Tenant-Id" not in req.headers
    assert any("outbound_tenant_unresolved" in r.getMessage() for r in caplog.records), (
        f"expected a loud ERROR naming the worker; got {[r.getMessage() for r in caplog.records]}"
    )
    assert any("principle_consistency" in r.getMessage() for r in caplog.records)


def test_hook_treats_the_nil_sentinel_as_unresolved():
    """The sentinel must never be forwarded as if it were a tenant."""
    _, _, hook = _build_provider(tenant_id=NIL_TENANT, worker_name="w")
    req = _request()
    req.headers["X-Tenant-Id"] = NIL_TENANT
    _run_hook(hook, req)
    assert "X-Tenant-Id" not in req.headers


# ------------------------------------------------------- the factory chain


def test_create_pydantic_ai_agent_threads_tenant_into_the_config():
    with (
        patch(f"{MODULE}.MLXOpenAIProvider"),
        patch(f"{MODULE}.OpenAIChatModel"),
        patch(f"{MODULE}.Agent"),
        patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
    ):
        from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

        agent = create_pydantic_ai_agent(
            name="principle_consistency",
            description="d",
            model_provider="mlx",
            model="mlx",
            base_url="http://mlx:8080/v1",
            tenant_id=REAL_TENANT,
        )
        assert agent.config.tenant_id == REAL_TENANT


def test_create_pydantic_ai_agent_defaults_to_none_not_empty_string():
    """``None`` and ``""`` are not the same: only ``None`` is honestly unknown."""
    with (
        patch(f"{MODULE}.MLXOpenAIProvider"),
        patch(f"{MODULE}.OpenAIChatModel"),
        patch(f"{MODULE}.Agent"),
        patch(f"{MODULE}.PydanticAIAgent._get_model_settings", return_value=None),
    ):
        from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

        agent = create_pydantic_ai_agent(
            name="w",
            description="d",
            model_provider="mlx",
            model="mlx",
            base_url="http://mlx:8080/v1",
        )
        assert agent.config.tenant_id is None


@pytest.mark.asyncio
async def test_agent_factory_forwards_tenant_only_for_pydantic_ai():
    """Other agent factories do not accept the kwarg -- forwarding would 500."""
    from src.worker.factory.agent_factory import AgentConfig, WorkerConfig, _create_agent_worker

    seen: dict[str, dict] = {}

    def _capture(agent_type, **kwargs):
        seen[agent_type] = kwargs
        return object()

    for agent_type, expect_tenant in (("pydantic_ai", True), ("claude_code", False)):
        with patch("src.worker.agents.agent_registry.create_agent", side_effect=_capture):
            await _create_agent_worker(
                WorkerConfig(
                    worker_name="principle_consistency",
                    tool_names=[],
                    description="d",
                    deps=[],
                    base_url_override=None,
                    api_key_override=None,
                    tools_dir_override=None,
                    agent_config=AgentConfig(agent_role="r", agent_type=agent_type),
                ),
                AgentConfig(agent_role="r", agent_type=agent_type),
                [],
                "/tmp/db",
                [],
                tenant_id=REAL_TENANT,
            )
        assert ("tenant_id" in seen[agent_type]) is expect_tenant, (
            f"{agent_type}: tenant_id forwarding is wrong"
        )
        if expect_tenant:
            assert seen[agent_type]["tenant_id"] == REAL_TENANT


# ------------------------------------------------- the two halves, composed


def _capture_server():
    """A real HTTP server that records the headers it was sent."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict[str, str] = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update({k: v for k, v in self.headers.items()})
            body = json.dumps(
                {
                    "id": "1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local-llm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            return

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, seen


def _post_through_provider(**kwargs) -> dict[str, str]:
    """Build a REAL provider against a REAL local server and send one request
    THROUGH THE OPENAI SDK, exactly as production does.

    ⛔ THE FIRST VERSION OF THIS HELPER PROVED LESS THAN IT LOOKED LIKE.
    It read `provider._client._custom_headers` and passed them to
    `httpx.post(headers=...)` by hand. That SIMULATES the default-headers
    mechanism instead of exercising it, and the mutation battery said so: with
    the headers supplied explicitly, "stop putting X-Tenant-Id in the default
    headers" and "stop setting it in the hook" BOTH survived — the hand-passed
    dict covered for each of them in turn.

    Going through `client.chat.completions.create` makes the SDK merge its own
    `default_headers` and then run the httpx event hook, which is the exact
    composition the defect lived in.
    """
    from src.worker.agents.pydantic_ai_agent import InferenceRouterOpenAIProvider

    srv, seen = _capture_server()
    try:
        port = srv.server_address[1]
        provider = InferenceRouterOpenAIProvider(
            base_url=f"http://127.0.0.1:{port}/api",
            api_key="internal",  # pragma: allowlist secret
            **kwargs,
        )

        async def _go():
            await provider._client.chat.completions.create(
                model="local-llm", messages=[{"role": "user", "content": "hi"}]
            )

        asyncio.run(_go())
        return {k.lower(): v for k, v in seen.items()}
    finally:
        srv.shutdown()
        srv.server_close()


def test_wire_a_real_tenant_arrives_at_the_server():
    seen = _post_through_provider(tenant_id=REAL_TENANT, worker_name="principle_consistency")
    assert seen.get("x-tenant-id") == REAL_TENANT
    assert seen.get("x-worker-name") == "principle_consistency"


def test_wire_an_unresolved_tenant_sends_NO_header_at_all():
    """THE DEFECT, at the only level that ever showed it.

    Measured against the pre-fix code restored in a scratch copy: the server
    really did receive `X-Tenant-Id: ''` — an empty header value goes on the
    wire, httpx does not drop it. That is why inference-router's
    `Header(default=...)` could not fire and it fell through to the nil
    sentinel instead.

    ⚠️ HOW TO MUTATION-TEST THIS ONE. The fix has TWO redundant halves — the
    constructor stops putting an empty value in `default_headers`, and the hook
    strips anything unresolved. Either half alone is enough to keep the wire
    clean, so mutating just one leaves this test GREEN and it looks like a
    coverage gap. It is not: it is redundancy. The mutant that this test must
    kill is the COMBINATION of both (which is exactly the original shipped
    shape), and it does. The single-point mutants are killed by the unit tests
    above, which observe each half directly.
    """
    seen = _post_through_provider(worker_name="principle_consistency")
    assert "x-tenant-id" not in seen, (
        f"an empty X-Tenant-Id reached the wire: {seen.get('x-tenant-id')!r}"
    )
    # Control: the request really did arrive, so the absence above is an absent
    # header and not an absent request.
    assert seen.get("x-worker-name") == "principle_consistency"


# ------------------------------------------- the graph-execution entry point


def test_worker_node_passes_the_effective_tenant_into_the_cache():
    """The graph path had `effective_tenant` in hand and did not pass it.

    `get_or_create_worker` falls back to `worker_data.get("tenant_id")`, so this
    looked covered. It is not the same value: `effective_tenant` is the tenant
    this execution is FOR (graph state, else the request ContextVar) and is
    proven non-empty a few lines earlier, whereas the registry row's tenant_id
    is whatever the row was stored with. This is a live LLM path — it ends in an
    outbound inference-router call — so an unresolved tenant here becomes a
    nil-sentinel cost event in billing.

    ⚠️ Asserted on the AST, not by executing the call. I tried the execution
    version first and could not reach the site: the identity helpers are
    imported INSIDE the function, so `patch(module.get_tenant_id_from_context)`
    raises AttributeError. A source-level assertion is the weaker of the two and
    is labelled as such — but it does bind (deleting the kwarg fails it), and a
    test that cannot run is worth less than a weaker one that does.
    """
    import ast
    import inspect

    from src.worker.graphs.nodes import worker_node

    tree = ast.parse(inspect.getsource(worker_node))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get_or_create_worker"
    ]
    assert calls, "no get_or_create_worker call found — this test has gone vacuous"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "tenant_id" in kw, (
            f"get_or_create_worker at line {c.lineno} does not pass tenant_id; "
            "the registry row's own tenant is not the tenant this run is for"
        )
        assert isinstance(kw["tenant_id"], ast.Name) and kw["tenant_id"].id == "effective_tenant", (
            f"line {c.lineno} passes tenant_id but not `effective_tenant`"
        )


# --------------------------------------- the vllm client is an IR client too


def test_vllm_client_sends_the_identity_triple_not_just_the_worker_name():
    """THE DEFECT #198 DID NOT FIX, and the reason it did not.

    `VLLM_URL` was repointed at inference-router (gitops #2999), so
    `_create_vllm_model` became an inference-router client wearing a vllm label.
    Someone noticed half of it and added `X-Worker-Name` — but not the identity
    triple. Measured live 2026-08-27 on `principle_consistency`:

        worker_name: principle_consistency      <- the half that was added
        tenant_id:   00000000-...-000000000000  <- nil
        license_id:  unknown
        app_id:      unknown

    A stripped-header case loses ONLY the tenant. Losing license and app too is
    the signature of a client that never sent identity at all — which is also
    why #198's `outbound_tenant_unresolved` guard never fired once: it lives in
    a class this agent never instantiates.
    """
    from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

    cfg = PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm",
        model="local-llm",
        db_path="",
        base_url="http://ir:8150/v1",
        tenant_id=REAL_TENANT,
        license_id="lic-1",
        subject_id="subj-1",
    )
    model = ModelFactory._create_vllm_model(cfg)
    client = model._provider.client._client if hasattr(model._provider, "client") else None
    hooks = getattr(client, "event_hooks", {}) if client is not None else {}
    if not hooks:
        # Reach the http_client the provider was built with, whatever the
        # pydantic-ai version calls it.
        client = getattr(model._provider, "_http_client", None) or getattr(
            model._provider, "http_client", None
        )
        hooks = getattr(client, "event_hooks", {}) if client is not None else {}

    assert hooks.get("request"), (
        "the vllm client installs NO request hook, so it can never stamp identity"
    )

    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    for h in hooks["request"]:
        asyncio.run(h(req))

    assert req.headers.get("X-Tenant-Id") == REAL_TENANT
    assert req.headers.get("X-License-Id") == "lic-1"
    assert req.headers.get("X-Subject-Id") == "subj-1"


def test_vllm_client_omits_an_unresolved_tenant_rather_than_sending_nil():
    """Same contract as the IR provider: absent, never the sentinel."""
    from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

    cfg = PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm",
        model="local-llm",
        db_path="",
        base_url="http://ir:8150/v1",
        tenant_id=NIL_TENANT,
    )
    model = ModelFactory._create_vllm_model(cfg)
    client = getattr(model._provider, "_http_client", None) or getattr(
        model._provider, "http_client", None
    )
    if client is None or not getattr(client, "event_hooks", {}).get("request"):
        client = model._provider.client._client
    req = httpx.Request("POST", "http://ir:8150/v1/chat/completions")
    req.headers["X-Tenant-Id"] = NIL_TENANT
    for h in client.event_hooks["request"]:
        asyncio.run(h(req))
    assert "X-Tenant-Id" not in req.headers


def test_the_identity_hook_is_defined_once_not_copied():
    """The divergence WAS the bug, so assert there is a single definition.

    Two clients with two copies is how `X-Worker-Name` got added to one and the
    identity triple to neither. A third copy would reopen it.
    """
    import inspect

    from src.worker.agents import pydantic_ai_agent as m

    src = inspect.getsource(m)
    assert src.count("def make_identity_event_hook(") == 1

    # History: a THIRD client (VLMLocalOpenAIProvider, lowercase vlm_backend
    # contract) once had its own hook. It was removed 2026-10-07 — vlm_local
    # now routes through inference-router on the shared hook.
    #
    # So the invariant is about the IR-contract hook specifically: exactly one
    # definition, and every IR-bound client takes it from the factory.
    assert src.count('request.headers["X-Tenant-Id"] = ') == 1, (
        "more than one place writes the IR tenant header — the hook has been "
        "copied again, which is exactly how the vllm client came to send "
        "X-Worker-Name and no identity at all"
    )


def test_vlm_local_is_an_inference_router_client_pinned_to_vlm_local(monkeypatch):
    """The third client is gone: vlm_local now rides the ONE IR identity hook.

    It used to dial the raw vlm_backend (``vlm.example.com:9203``) with only
    lowercase ``x-tenant-id`` / ``x-user-id`` — no licence token, no app /
    project / actor / job — and WM's emitter refuses ``vlm_local``, so vision
    calls were unmetered. It is now built by ``_create_inference_router_model``
    pinned (``X-Provider``) to IR's ``vlm_local`` provider, and a stored raw
    backend URL is never dialled.
    """
    import src.worker.agents.pydantic_ai_agent as m

    monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://ir:8150")
    cfg = m.PydanticAIAgentConfig(
        name="vision-worker",
        model_provider="vlm_local",
        model="qwen-vl",
        db_path="",
        base_url="http://vlm.example.com:9203/api/vlm",
    )
    with patch.object(m.ModelFactory, "_create_inference_router_model") as ir:
        m.ModelFactory._create_vlm_local_model(cfg)
    routed = ir.call_args[0][0]
    assert routed.base_url == "http://ir:8150"
    assert routed.ir_provider == "vlm_local"
    assert routed.model == "qwen-vl"


# ----------------------------------------------------- the SILENT licence drop
#
# The tenant branch above shouts when it cannot resolve. Until this block, the
# adjacent licence branch dropped `X-License-Id` and said NOTHING -- so an
# outbound GPU call that no licence could be invoiced for left no trace naming
# the worker or the tenant that caused it. Measured on `principle_consistency`:
# 830 licensed / 2,672 unlicensed calls in 30h, invisible in this service.
#
# These tests pin the OBSERVABILITY, not a substitution. There is deliberately
# no test asserting that a licence gets manufactured from the tenant, because
# doing that is the documented worse failure (see the hook's comment).


@pytest.fixture
def _clean_identity_context():
    """Reset the identity/correlation ContextVars around a test.

    `asyncio.run` copies the ambient context into its task, so values set by a
    previous test would otherwise leak into this one's hook invocation and make
    these assertions depend on test ORDER.
    """
    from src.api.identity_context import set_correlation_context, set_identity_context

    set_identity_context(None, None, None)
    set_correlation_context(None, None)
    yield
    set_identity_context(None, None, None)
    set_correlation_context(None, None)


def test_a_dropped_license_is_logged_at_warning_naming_worker_and_tenant(
    caplog, _clean_identity_context
):
    """THE DEFECT. Pre-fix this branch emitted no record of any kind.

    The assertion is on a stable grep token (``outbound_license_unresolved``)
    rather than prose, matching the tenant test above, so rewording the message
    cannot silently void this test.

    ⚠️ The level floor is real: this service runs at ``LOG_LEVEL=INFO``, so a
    DEBUG line here would be deletion, not visibility. Capturing at WARNING
    means a demotion to INFO or DEBUG fails this test.
    """
    _, _, hook = _build_provider(tenant_id=REAL_TENANT, worker_name="principle_consistency")
    req = _request()
    req.headers["X-Tenant-Id"] = REAL_TENANT
    req.headers["X-License-Id"] = ""  # as an unconditional default header would

    with caplog.at_level(logging.WARNING, logger=MODULE):
        _run_hook(hook, req)

    # The drop itself still happens -- an empty licence header is a claim.
    assert "X-License-Id" not in req.headers
    # ...and it is now on the record.
    records = [r for r in caplog.records if "outbound_license_unresolved" in r.getMessage()]
    assert records, (
        f"the licence drop must not be silent; got {[r.getMessage() for r in caplog.records]}"
    )
    assert len(records) == 1, "one call, one line -- not a per-header storm"
    msg = records[0].getMessage()
    assert records[0].levelno >= logging.WARNING, (
        f"LOG_LEVEL=INFO here, so {records[0].levelname} would be deletion, not visibility"
    )
    # Enough to find the caller that must be fixed.
    assert "principle_consistency" in msg, msg
    assert REAL_TENANT in msg, msg


def test_the_dropped_license_warning_carries_correlation_for_tracing(
    _clean_identity_context, caplog
):
    """`correlation=` is what joins this line to the run that paid for it."""
    from src.api.identity_context import set_correlation_context, set_identity_context

    set_identity_context(REAL_TENANT, None, None)  # tenant resolved, licence NOT
    set_correlation_context("run-abc-123", "stage-7")

    _, _, hook = _build_provider(worker_name="principle_consistency")
    req = _request()
    with caplog.at_level(logging.WARNING, logger=MODULE):
        _run_hook(hook, req)

    msg = next(
        r.getMessage() for r in caplog.records if "outbound_license_unresolved" in r.getMessage()
    )
    assert "run-abc-123" in msg, msg
    # subject was also absent here; it is reported on this line rather than
    # getting a duplicate warning of its own.
    assert "subject_present=False" in msg, msg


def test_a_resolved_license_is_forwarded_and_logs_nothing(_clean_identity_context, caplog):
    """The control. Without it, an unconditional `logger.warning` would pass.

    This is the assertion that makes the test above meaningful: it proves the
    warning is a consequence of the DROP and not merely a line that always runs.
    """
    from src.api.identity_context import set_identity_context

    real_license = "3f2a1c58-9d44-4e0b-8f77-2b6c1e05aa91"
    set_identity_context(REAL_TENANT, real_license, None)

    _, _, hook = _build_provider(worker_name="principle_consistency")
    req = _request()
    with caplog.at_level(logging.WARNING, logger=MODULE):
        _run_hook(hook, req)

    assert req.headers["X-License-Id"] == real_license
    assert not [r for r in caplog.records if "outbound_license_unresolved" in r.getMessage()], (
        "a licence that RESOLVED must not warn"
    )


def test_an_unresolved_license_is_never_substituted(_clean_identity_context):
    """⛔ The rule the warning exists to protect.

    A visible gap is the deliverable. Filling `X-License-Id` with a tenant-
    derived or default value would make the row look attributed and bill a real
    customer for spend nobody proved was theirs -- strictly worse than missing.
    """
    from src.api.identity_context import set_identity_context

    set_identity_context(REAL_TENANT, None, None)

    _, _, hook = _build_provider(worker_name="principle_consistency")
    req = _request()
    _run_hook(hook, req)

    assert req.headers["X-Tenant-Id"] == REAL_TENANT  # tenant DID resolve
    assert "X-License-Id" not in req.headers, (
        f"no substitution, ever; got {req.headers.get('X-License-Id')!r}"
    )
