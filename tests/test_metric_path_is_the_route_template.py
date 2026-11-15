"""The ``path`` label on the auth counters is a ROUTE TEMPLATE, not a heuristic.

⛔ WHAT THIS PINS

``auth_middleware`` labels two counters with a "route shape" derived from the
request path:

  * ``worker_management_unauthenticated_tenant_assertion_total{path,enforced}``
  * ``worker_management_identity_omitted_total{field,reason,caller,path}``  (4 labels,
    so the cardinality blow-up is multiplicative)

Until this file existed, that shape came from a CHARACTER HEURISTIC: a segment
collapsed to ``{id}`` only when it was not alphabetic-after-stripping ``-`` and
``_``. The live route ``POST /workers/{worker_name}/message`` **auto-creates
whatever worker name it is handed**, so every alphabetic name a caller invents
passed through RAW into a Prometheus label. Only digit-bearing names collapsed.

That is the exact thing the sibling module forbids in bold
(``src/api/workers/system_worker_contracts.py``): "⛔ The raw ``worker_name``
must NEVER reach a label. It is a path parameter ... and AUTO-CREATES whatever
it is handed". Prod already carries 472 distinct worker names across 60
tenants, and the TSDB already holds a permanent
``path="/workers/zz-probe-nonexistent-worker/message"`` series minted by a
one-off probe — a gitops ratchet alert counts these shapes, so one tokenless
call to a new alphabetic name pins a SECURITY alert for seven days.

The fix reads the MATCHED ROUTE TEMPLATE out of ``scope["route"]`` instead. The
template set is bounded by the route table, which is what the ratchet alert
already (wrongly) assumed.

⚠️ WHY EVERY TEST HERE DRIVES THE REAL ROUTER

``app.routes`` CANNOT resolve a template from middleware on FastAPI >= 0.137:
each ``include_router`` contributes one opaque ``_IncludedRouter`` whose
``matches()`` returns ``Match.FULL`` with an EMPTY child scope and no ``path``
attribute. A fix built on iterating ``app.routes`` labels 100% of real traffic
``<unmatched>`` while every hand-rolled unit test passes. So these tests mount
the REAL ``src.api.workers`` router through ``include_router`` and drive it
with ``TestClient``; none of them fabricates ``scope["route"]``, because a test
that fabricates the thing under test proves nothing.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.auth_middleware as auth_middleware_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.settings import reset_settings_cache

# The live route whose path parameter is unbounded and caller-chosen.
MESSAGE_TEMPLATE = "/workers/{worker_name}/message"
WORKER_TEMPLATE = "/workers/{worker_name}"

# A canonical uuid that is spelled correctly and proves nothing.
ASSERTED_TENANT = "0195f48b-d000-7000-8000-000000000003"

# Deliberately ALPHABETIC-after-stripping: this is precisely the class the old
# character heuristic let through raw. It also matches the worker-name pattern
# on the route (``^[a-zA-Z0-9_-]+$``), so the request really does route.
CALLER_CHOSEN_NAME = "zz-probe-nonexistent-worker"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    """Clear the module-global WARNING dedup cache + the settings cache."""
    auth_middleware_module._assertion_log_seen.clear()
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT", raising=False)
    monkeypatch.delenv("REQUIRE_AUTHENTICATED_TENANT", raising=False)
    reset_settings_cache()
    yield
    auth_middleware_module._assertion_log_seen.clear()
    reset_settings_cache()


@pytest.fixture
def app() -> FastAPI:
    """The REAL workers router, mounted the way production mounts it.

    ``include_router`` is the load-bearing detail: it is what produces the
    opaque ``_IncludedRouter`` in ``app.routes`` that makes the naive fix
    silently wrong.
    """
    from src.api.workers import router as workers_router

    application = FastAPI()
    application.include_router(workers_router)
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    return application


def test_the_router_is_opaque_from_app_routes() -> None:
    """⛔ POSITIVE CONTROL for the trap this file exists to avoid.

    If this ever fails, FastAPI has changed and an ``app.routes`` walk might
    work again — but until then, any fix that iterates ``app.routes`` labels
    every real request ``<unmatched>``.
    """
    from src.api.workers import router as workers_router

    application = FastAPI()
    application.include_router(workers_router)
    included = [r for r in application.routes if type(r).__name__ == "_IncludedRouter"]
    assert included, "expected include_router to contribute an opaque _IncludedRouter"
    assert all(getattr(r, "path", None) is None for r in included)
    assert not any(getattr(r, "path", "") == MESSAGE_TEMPLATE for r in application.routes)


def _call(app: FastAPI, method: str, path: str, **kwargs) -> tuple[MagicMock, int]:
    """Tokenless request from a REAL peer inside TRUSTED_CIDRS.

    Returns the metrics mock and the status code. ``raise_server_exceptions``
    is off because several of these routes reach a database that is not
    initialised in a unit test — irrelevant here, since ``scope["route"]`` is
    populated by the router BEFORE the endpoint runs, so the label is decided
    whatever the handler then does.
    """
    metrics = MagicMock()
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        with TestClient(app, client=("10.42.1.7", 54321), raise_server_exceptions=False) as client:
            resp = client.request(method, path, headers={"X-Tenant-Id": ASSERTED_TENANT}, **kwargs)
    return metrics, resp.status_code


def _assertion_label(metrics: MagicMock) -> str:
    metrics.record_unauthenticated_tenant_assertion.assert_called_once()
    args, _kwargs = metrics.record_unauthenticated_tenant_assertion.call_args
    return args[0]


def _omission_labels(metrics: MagicMock) -> set[str]:
    calls = metrics.record_identity_omission.call_args_list
    assert calls, "the tokenless branch recorded no identity omission at all"
    return {c.kwargs["path"] for c in calls}


def _rendered(metrics: MagicMock) -> str:
    """Everything that reached a metric label, as one string."""
    return repr(metrics.record_unauthenticated_tenant_assertion.call_args_list) + repr(
        metrics.record_identity_omission.call_args_list
    )


# ---------------------------------------------------------------------------
# THE DEFECT.
# ---------------------------------------------------------------------------


def test_a_caller_chosen_worker_name_never_reaches_a_metric_label() -> None:
    """⛔ THE LOAD-BEARING TEST.

    ``POST /workers/{worker_name}/message`` auto-creates whatever it is handed.
    One tokenless call with an invented alphabetic name used to mint a
    permanent time series named after that string, on BOTH counters.
    """
    from src.api.workers import router as workers_router

    application = FastAPI()
    application.include_router(workers_router)
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    metrics, status = _call(application, "POST", f"/workers/{CALLER_CHOSEN_NAME}/message", json={})

    # 422: the body is invalid, which is deliberate — the route MATCHED (that
    # is what populates scope["route"]) and the endpoint's own auto-create side
    # effect is not needed to decide the label. A 404 here would mean the
    # request never routed and the assertion below would be vacuous.
    assert status == 422, "the request must actually route for this test to mean anything"

    assert _assertion_label(metrics) == MESSAGE_TEMPLATE
    assert _omission_labels(metrics) == {MESSAGE_TEMPLATE}
    assert CALLER_CHOSEN_NAME not in _rendered(metrics)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL (a): the fix must not collapse everything into one bucket.
# ---------------------------------------------------------------------------


def test_a_different_real_route_keeps_its_own_distinct_template(app: FastAPI) -> None:
    """A label that is always the same value is not a fix, it is a deletion."""
    static_metrics, static_status = _call(app, "GET", "/workers/types")
    assert static_status == 200
    assert _assertion_label(static_metrics) == "/workers/types"

    auth_middleware_module._assertion_log_seen.clear()
    param_metrics, _ = _call(app, "POST", f"/workers/{CALLER_CHOSEN_NAME}/message", json={})

    assert _assertion_label(param_metrics) == MESSAGE_TEMPLATE
    assert _assertion_label(static_metrics) != _assertion_label(param_metrics)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL (b): the unmatched case is bounded and carries nothing.
# ---------------------------------------------------------------------------


def test_an_unmatched_path_gets_a_bounded_label_with_no_caller_string(app: FastAPI) -> None:
    """A 404 never routes, so there is no template — and no caller string either.

    The probe path is ALPHABETIC on purpose: under the old heuristic it was
    emitted verbatim, so a 404-scanner alone could mint unbounded series.
    """
    probe = "zzunroutablecallerchosenpath"
    metrics, status = _call(app, "GET", f"/{probe}")

    assert status == 404
    assert _assertion_label(metrics) == auth_middleware_module._UNMATCHED_PATH
    assert probe not in _rendered(metrics)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL (c): the cases that ALREADY collapsed still collapse.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "segment",
    ["0195f48b-d000-7000-8000-000000000003", "12345"],
    ids=["uuid", "numeric"],
)
def test_id_bearing_segments_still_collapse(app: FastAPI, segment: str) -> None:
    """These were the only cases the old heuristic got right; keep them right.

    The first assertion is the TRUE control — it holds on both sides of the
    change, so a regression here means the fix broke a case that already
    worked. The second is new: the collapsed value is now the route's own
    parameter name rather than the heuristic's ``{id}`` placeholder.
    """
    metrics, _status = _call(app, "GET", f"/workers/{segment}")

    assert segment not in _rendered(metrics)
    assert _assertion_label(metrics) == WORKER_TEMPLATE


# ---------------------------------------------------------------------------
# The template set is bounded BY CONSTRUCTION, which is the whole claim.
# ---------------------------------------------------------------------------


def test_every_emitted_label_is_a_declared_template_or_the_unmatched_literal(
    app: FastAPI,
) -> None:
    """⛔ The property the gitops ratchet alert already assumes.

    Six requests with six DIFFERENT caller-chosen strings must produce at most
    the templates those requests route to, plus the unmatched literal — never
    six new series.
    """
    from src.api.workers import router as workers_router

    declared = {r.path for r in workers_router.routes} | {auth_middleware_module._UNMATCHED_PATH}

    emitted: set[str] = set()
    for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"):
        auth_middleware_module._assertion_log_seen.clear()
        metrics, _ = _call(app, "POST", f"/workers/{name}/message", json={})
        emitted.add(_assertion_label(metrics))

    assert emitted == {MESSAGE_TEMPLATE}, "six names must not be six series"
    assert emitted <= declared


# ---------------------------------------------------------------------------
# ⛔ THE TWO GUARANTEES THE ``dispatch`` RESTRUCTURE EXISTS TO OBTAIN.
#
# Both were surviving mutants: 50/50 tests passed with the ``finally`` removed,
# and 50/50 passed with the per-request queue replaced by a module-global. They
# are the properties that make deferral safe at all, so they are pinned here
# rather than left to the reviewer who happens to think of them.
# ---------------------------------------------------------------------------


def _asgi_app_with_named_routes() -> tuple[FastAPI, asyncio.Event]:
    """Three DISTINCT templates plus one route that hangs until cancelled."""
    started = asyncio.Event()
    application = FastAPI()

    @application.get("/alpha/{a}")
    async def _alpha(a: str) -> dict[str, str]:
        await asyncio.sleep(0)  # force a real suspension point
        return {"a": a}

    @application.get("/bravo/{b}/leaf")
    async def _bravo(b: str) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"b": b}

    @application.get("/charlie")
    async def _charlie() -> dict[str, bool]:
        return {"ok": True}

    @application.get("/hang")
    async def _hang() -> dict[str, bool]:
        started.set()
        await asyncio.Event().wait()  # never returns; the client aborts
        return {"ok": True}  # pragma: no cover - unreachable

    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    return application, started


def _async_client(application: FastAPI) -> httpx.AsyncClient:
    """A real ASGI client whose peer address is inside TRUSTED_CIDRS."""
    transport = httpx.ASGITransport(
        app=application, client=("10.42.1.7", 54321), raise_app_exceptions=False
    )
    return httpx.AsyncClient(transport=transport, base_url="http://worker-management.test")


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_cross_label_each_other() -> None:
    """⛔ The queue is PER-REQUEST, and a module-global would pass every other test.

    Deferral means an emitter outlives the auth decision that created it. If the
    queue were shared, request B's ``finally`` would drain request A's emitter
    against B's route template — silently mislabelling, never failing. Sixty
    concurrent requests over three distinct templates must come back 20/20/20.
    """
    application, _started = _asgi_app_with_named_routes()
    paths = ["/alpha/x{i}", "/bravo/y{i}/leaf", "/charlie"]

    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        async with _async_client(application) as client:
            responses = await asyncio.gather(
                *[
                    client.get(paths[i % 3].format(i=i), headers={"X-Tenant-Id": ASSERTED_TENANT})
                    for i in range(60)
                ]
            )

    assert {r.status_code for r in responses} == {200}, "all 60 must actually route"
    labels = Counter(
        call.args[0] for call in metrics.record_unauthenticated_tenant_assertion.call_args_list
    )
    assert labels == Counter({"/alpha/{a}": 20, "/bravo/{b}/leaf": 20, "/charlie": 20})
    # And no caller-chosen suffix leaked into any of them.
    assert not any(char.isdigit() for char in "".join(labels))


@pytest.mark.asyncio
async def test_a_cancelled_request_is_still_counted() -> None:
    """⛔ Why the flush is a ``finally`` and not a line before ``return``.

    ``_authenticate`` has a blanket ``except Exception``, so a handler that
    RAISES is converted to a 500 and any placement of the flush would still run.
    ``asyncio.CancelledError`` is a ``BaseException``: it sails straight past
    that handler. A client that hangs up mid-request is therefore the only shape
    that distinguishes a ``finally`` from a flush on the return path — and it is
    the shape a tokenless caller with a timeout produces in production, i.e.
    exactly the traffic these counters exist to measure. Dropping it would
    under-count the callers that must be migrated.
    """
    application, started = _asgi_app_with_named_routes()

    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        async with _async_client(application) as client:
            task = asyncio.create_task(
                client.get("/hang", headers={"X-Tenant-Id": ASSERTED_TENANT})
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Let the cancellation finish unwinding through the middleware.
            await asyncio.sleep(0.05)

    assert [
        call.args[0] for call in metrics.record_unauthenticated_tenant_assertion.call_args_list
    ] == ["/hang"], "a cancelled request recorded nothing — the flush is not in a finally"
