"""Silent auto-provisioning on ``POST /workers/{worker_name}/message``.

The endpoint auto-creates any worker name it is handed, with
``system_prompt=None`` and ``instruction=None``, and returns **200**. So a send
to a mistyped, renamed or never-seeded worker does not fail — it succeeds, from
a promptless generic worker, and the answer comes back as if nothing happened.

⛔ WHY NOTHING CAUGHT THIS FOR THE WHOLE OF ITS LIFE. The n8n caller has a
correct error-reader. It is *structurally* blind here: the auto-create SUCCEEDS,
so there is no error for it to read. A reader cannot report a failure that
arrives as a success, and no amount of hardening on the caller side changes
that — the signal has to be created on this side.

Evidence it is live (n8n_custom_nodes#1522): ``worker-doing-step`` is dispatched
by the medium-step DOING branch and has **no row at any timestamp**, while its
siblings ``worker-doing-turn`` / ``worker-doing-prose`` are upserted
continuously through the SAME path. Generalised: worker names are not a checked
vocabulary, so every typo and every rename degrades silently.

What this suite pins, and what it deliberately does NOT
-------------------------------------------------------
The auto-create STAYS THE DEFAULT. ``worker-doing-turn`` / ``-prose`` are live
on this path and a refusal by default would take them down — so the first test
here is a regression guard that the unknown-worker send still returns 200, and
it is as load-bearing as the detection tests. What changes is that the event is
now (a) a WARNING naming the worker, the tenant and the absent prompt, and (b) a
counter. A caller that owns a closed vocabulary opts out per message with
``require_existing=true``.

⭐ ``test_the_auto_created_worker_is_created_without_a_system_prompt`` pins the
HARM, not the plumbing: today the row is minted promptless. If anyone ever
starts supplying a prompt on this path, that test goes red and the change is
visible, instead of quietly making the rest of this file meaningless.

Fixtures mirror ``tests/test_bare_worker_contract_detection.py``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id

_MP = "src.api.workers.message_endpoints"
_LOGGER = "src.api.workers.message_endpoints"
TENANT = "test-tenant-autocreate"
WORKER_ID = "wid-autocreate"

#: The name from the live finding: dispatched by the medium-step DOING branch,
#: never provisioned, no row at any timestamp.
MISSING_WORKER = "worker-doing-step"
#: Its sibling on the same dispatch path, which IS provisioned. The negative
#: control — without it, the detection tests prove only "always fires".
EXISTING_WORKER = "worker-doing-turn"


def _config() -> dict[str, Any]:
    return {
        "agent_type": "pydantic_ai",
        "model_provider": "vllm",
        "model": "local-llm",
        "temperature": 0.1,
        "output_schema": None,
    }


def _worker_data(*, worker_name: str, status: str = "created") -> dict[str, Any]:
    return {
        "worker_name": worker_name,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        "output_schema": None,
        "config": _config(),
        "status": status,
    }


def _run_response(output: Any = "Hello!") -> MagicMock:
    resp = MagicMock()
    resp.output = output
    resp.data = output
    resp.usage_metrics = None
    resp.session_id = None
    resp.cost_usd = None
    resp.duration_ms = None
    return resp


@asynccontextmanager
async def _noop_ctx(*_a: Any, **_kw: Any):
    yield "fake-lock"


@pytest.fixture
def deps():
    """Registry MISSES by default — the unknown-worker case is the subject.

    ``get_worker`` is a side-effect list, not a constant: the endpoint looks the
    worker up, auto-creates, registers, and then looks it up AGAIN to pick up
    the row it just wrote. A constant ``None`` would make that re-fetch raise a
    500 and every test here would pass for the wrong reason.
    """
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.state_manager") as mock_state_mgr,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
        patch(f"{_MP}.mlx_model_registry") as mock_mlx,
    ):
        mock_state_mgr.atomic_worker_operation = _noop_ctx
        mock_state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        mock_state_mgr.validate_state_transition = MagicMock(return_value=True)

        mock_db = AsyncMock()
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.add_worker_state = AsyncMock(return_value=1)
        MockStateDB.return_value = mock_db

        worker_mock = MagicMock()
        worker_mock.run = AsyncMock(return_value=_run_response())
        mock_cache.get_or_create_worker = AsyncMock(return_value=worker_mock)
        mock_cache.remove_worker = AsyncMock()
        mock_cache.remove_instance = AsyncMock()

        # miss, then the row the auto-create just wrote
        mock_registry.get_worker = AsyncMock(
            side_effect=[
                None,
                _worker_data(worker_name=MISSING_WORKER, status="auto-created"),
            ]
        )
        mock_registry.register_worker = AsyncMock(return_value={"worker_id": WORKER_ID})

        fake_instance = MagicMock()
        fake_instance.db_path = "/tmp/auto.db"
        mock_create.return_value = fake_instance

        mock_mlx.get_model = AsyncMock(return_value=None)
        mock_mlx.get_instance_for_model = AsyncMock(return_value=None)

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "create": mock_create,
            "worker_mock": worker_mock,
        }


@pytest.fixture
def client(deps):
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(_a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _post(client, worker_name: str = MISSING_WORKER, **extra):
    return client.post(f"/workers/{worker_name}/message", json={"message": "Hello", **extra})


def _found(deps, worker_name: str = EXISTING_WORKER) -> None:
    """Flip the fixture so the worker EXISTS — no auto-create branch at all."""
    deps["registry"].get_worker = AsyncMock(return_value=_worker_data(worker_name=worker_name))


def _auto_create_warnings(caplog, worker_name: str) -> list[logging.LogRecord]:
    """WARNING records naming ``worker_name`` and the auto-create.

    Matched on the message TEXT and not on a patched symbol, so this suite can
    be run against the PRE-FIX code and shown to go red. Pre-fix the line exists
    but is at INFO, and ``levelno >= WARNING`` excludes it.

    ⚠️ Matches ``auto-creating`` (a row WAS minted), not ``auto-create`` — the
    ``require_existing`` refusal also warns and also says "auto-create", and the
    two must never be confused for one another.
    """
    return [
        rec
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and worker_name in rec.getMessage()
        and "auto-creating" in rec.getMessage().lower()
    ]


# ---------------------------------------------------------------------------
# THE REGRESSION GUARD — the live callers must keep working
# ---------------------------------------------------------------------------


def test_a_send_to_an_unknown_worker_still_auto_creates_and_returns_200(client, deps):
    """⛔ Auto-create is the DEFAULT and must stay it.

    ``worker-doing-turn`` and ``worker-doing-prose`` are upserted through this
    exact branch continuously. Making detection into a refusal would take them
    down. If this test ever goes red, the fix has become an outage.
    """
    resp = _post(client)

    assert resp.status_code == 200, resp.text
    deps["create"].assert_awaited_once()
    deps["registry"].register_worker.assert_awaited_once()


# ---------------------------------------------------------------------------
# ⭐ THE HARM — pinned, so a future change that fixes it is VISIBLE
# ---------------------------------------------------------------------------


def test_the_auto_created_worker_is_created_without_a_system_prompt(client, deps):
    """⭐ The actual defect: the minted worker is PROMPTLESS and still answers 200.

    This is a pin on today's behaviour, not an endorsement of it. If anyone
    later starts supplying a prompt on this path — a per-name default, a
    registered contract, an inherited template — this test goes red and says so,
    rather than passing while the rest of this file quietly loses its subject.
    """
    resp = _post(client)
    assert resp.status_code == 200, resp.text

    kwargs = deps["create"].await_args.kwargs
    assert kwargs["system_prompt"] is None, (
        "the auto-created worker now HAS a system prompt — the harm this suite "
        "describes has changed; re-read the module docstring before updating this"
    )
    assert kwargs["instruction"] is None
    assert kwargs["user_prompt"] is None
    # ...and the name really is the untrusted one the caller typed.
    assert kwargs["worker_name"] == MISSING_WORKER


# ---------------------------------------------------------------------------
# DETECTION — the log
# ---------------------------------------------------------------------------


def test_the_auto_create_emits_a_warning_naming_the_worker_tenant_and_missing_prompt(
    client, caplog
):
    """The headline. Pre-fix this line was ``logger.info`` and nothing saw it."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resp = _post(client)

    assert resp.status_code == 200
    warnings = _auto_create_warnings(caplog, MISSING_WORKER)
    assert warnings, (
        f"auto-creating '{MISSING_WORKER}' produced no WARNING. A send to a "
        f"nonexistent worker is invisible: it returns 200 from a promptless worker "
        f"and the caller's error-reader has nothing to read."
    )

    rec = warnings[0]
    text = rec.getMessage()
    assert "system prompt" in text.lower(), (
        f"the warning must say WHAT is wrong with the row it just minted; got: {text}"
    )

    # Structured fields, so this is greppable/queryable and not only prose.
    assert getattr(rec, "worker_name", None) == MISSING_WORKER
    assert getattr(rec, "tenant_id", None), "the warning must name the tenant"
    assert getattr(rec, "auto_created", None) is True
    assert "system_prompt" in rec.__dict__ and rec.__dict__["system_prompt"] is None


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — mandatory, or the two tests above prove only "always fires"
# ---------------------------------------------------------------------------


def test_an_existing_worker_produces_no_auto_create_warning_and_no_count(client, deps, caplog):
    """⛔ THE CONTROL. Same endpoint, same fixtures — the worker just EXISTS.

    ``worker-doing-turn`` is the live sibling of the missing one. If it also
    warns and counts, the detector is a constant and every assertion above is
    vacuous.
    """
    _found(deps)
    recorder = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        with patch(f"{_MP}.get_metrics_client", return_value=recorder):
            resp = _post(client, EXISTING_WORKER)

    assert resp.status_code == 200
    assert not _auto_create_warnings(caplog, EXISTING_WORKER), (
        "an EXISTING worker was reported as auto-created; the detector fires "
        "unconditionally and proves nothing"
    )
    recorder.record_worker_auto_created.assert_not_called()
    deps["create"].assert_not_awaited()


# ---------------------------------------------------------------------------
# DETECTION — the counter, at the wire
# ---------------------------------------------------------------------------


def test_the_endpoint_counts_the_auto_creation(client, deps):
    """Wire-level: the endpoint must emit the counter, not only log."""
    recorder = MagicMock()
    with patch(f"{_MP}.get_metrics_client", return_value=recorder):
        resp = _post(client)

    assert resp.status_code == 200
    recorder.record_worker_auto_created.assert_called_once_with(MISSING_WORKER)


# ---------------------------------------------------------------------------
# require_existing — the opt-out
# ---------------------------------------------------------------------------


def test_require_existing_defaults_to_false():
    """The default is the whole backwards-compatibility argument. Pin it."""
    from src.api.workers.models import MessageRequest

    assert MessageRequest(message="Hello").require_existing is False


def test_require_existing_true_on_an_unknown_worker_is_a_404_and_creates_nothing(client, deps):
    """⛔ Assert the NO-OP, not just the status code.

    A 404 that still minted the row would be strictly worse than today: the
    caller sees an error AND the promptless worker exists. So the assertion that
    matters is that ``create_worker`` and ``register_worker`` were never awaited.
    """
    resp = _post(client, require_existing=True)

    assert resp.status_code == 404, resp.text
    # The service wraps errors as {"error": {"code", "message", "status"}}; the
    # ``detail`` fallback matches tests/test_message_endpoints.py.
    body = resp.json()
    detail = str(body.get("error", {}).get("message", body.get("detail", "")))
    assert MISSING_WORKER in detail, "the 404 must name the worker the caller asked for"
    assert "auto-creat" in detail.lower(), (
        "the 404 must say the worker was NOT auto-created, or the caller cannot "
        f"tell it apart from any other 404; got: {detail}"
    )
    assert "require_existing" in detail, "the 404 must name the flag that caused it"

    deps["create"].assert_not_awaited()
    deps["registry"].register_worker.assert_not_awaited()


def test_require_existing_true_on_an_existing_worker_is_a_normal_200(client, deps):
    """The opt-out must cost a correct caller nothing."""
    _found(deps)

    resp = _post(client, EXISTING_WORKER, require_existing=True)

    assert resp.status_code == 200, resp.text
    assert resp.json()["output"] == "Hello!"
    deps["create"].assert_not_awaited()


def test_require_existing_does_not_join_the_worker_cache_key(client):
    """``require_existing`` changes no property of the agent that gets built.

    The override fingerprint is an explicit allowlist and this field must stay
    out of it: folding it in would split the cache slot in two and halve reuse
    for every caller that sets it, for no behavioural difference at all.
    """
    from src.api.workers.message_endpoints import _override_cache_fingerprint
    from src.api.workers.models import MessageRequest

    plain = MessageRequest(message="Hello")
    guarded = MessageRequest(message="Hello", require_existing=True)
    assert _override_cache_fingerprint(plain) == _override_cache_fingerprint(guarded)
    assert _override_cache_fingerprint(guarded) is None


# ---------------------------------------------------------------------------
# The metric — registration, bounded labels, and zero-before-anything
# ---------------------------------------------------------------------------


def _client_with_fake_meter():
    from src.telemetry.agent_metrics import AgentMetricsClient

    client = AgentMetricsClient.__new__(AgentMetricsClient)
    client.enabled = True
    client._counters = {}
    client._histograms = {}
    client._up_down_counters = {}
    client.meter = MagicMock()
    client.meter.create_counter.side_effect = lambda **_kw: MagicMock()
    client.meter.create_histogram.side_effect = lambda **_kw: MagicMock()
    client.meter.create_up_down_counter.side_effect = lambda **_kw: MagicMock()
    client._setup_metrics()
    return client


def test_the_counter_is_registered_under_a_stable_name():
    client = _client_with_fake_meter()
    names = {kw["name"] for _a, kw in client.meter.create_counter.call_args_list}
    assert "worker_management_worker_auto_created_total" in names


def test_the_metric_series_exist_at_zero_before_anything_is_counted():
    """⭐ An ABSENT series is indistinguishable from a broken exporter.

    Without pre-materialisation, "nobody is typo-ing a worker name" and
    "nothing is scraping this pod" render identically on a dashboard.
    """
    from src.api.workers.system_worker_contracts import contract_labels

    client = _client_with_fake_meter()
    counter = client._counters["worker_auto_created"]

    zeroed = {c.args[1]["contract"] for c in counter.add.call_args_list if c.args[0] == 0}
    assert zeroed == set(contract_labels()), (
        f"missing pre-materialised series: {sorted(set(contract_labels()) - zeroed)}"
    )
    # ...and nothing has been COUNTED yet: init must not fabricate a finding.
    assert not [c for c in counter.add.call_args_list if c.args[0] != 0]


def test_recording_collapses_an_unknown_name_to_a_bounded_label():
    """⛔ The cardinality bound. This endpoint mints whatever name it is handed.

    ``worker-doing-step`` is a tenant-side name this image does not register, so
    it books as ``<other>``. Labelling the raw value would hand an unbounded
    label to exactly the untrusted input this counter exists to characterise.
    """
    from src.api.workers.system_worker_contracts import CONTRACT_OTHER

    client = _client_with_fake_meter()
    counter = client._counters["worker_auto_created"]
    counter.add.reset_mock()

    client.record_worker_auto_created(MISSING_WORKER)
    client.record_worker_auto_created("improvement_workflow_development")

    assert [c.args for c in counter.add.call_args_list] == [
        (1, {"contract": CONTRACT_OTHER}),
        (1, {"contract": "improvement_workflow_development"}),
    ]


def test_recording_never_raises():
    """Instrumentation must not turn a degraded provision into a failed request."""
    client = _client_with_fake_meter()
    client._counters["worker_auto_created"].add.side_effect = RuntimeError("exporter down")
    client.record_worker_auto_created(MISSING_WORKER)  # must not raise
