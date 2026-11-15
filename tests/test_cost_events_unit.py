"""Unit tests for the BYO cost-event emitter (src/telemetry/cost_events.py).

Covers the no-op guards (unset Redis URL, no tenant in context) and the exact
field schema XADD'd to the metering stream — in particular the byo flag in
usage_extra and the worker-management cost_source that metering keys on.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.telemetry import cost_events


def _stub_settings(url: str = "redis://metering:6379/0", stream: str = "improvement-loop:costs"):
    return SimpleNamespace(metering_redis_url=url, metering_costs_stream=stream)


@pytest.mark.asyncio
async def test_emit_builds_expected_fields(monkeypatch):
    """_emit XADDs the metering schema with byo=true + worker-management source."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=1234,
    )

    fake_client.xadd.assert_awaited_once()
    args, kwargs = fake_client.xadd.call_args
    assert args[0] == "improvement-loop:costs"
    fields = args[1]
    assert fields["tenant_id"] == "tenant-1"
    assert fields["service_key"] == "worker-management"
    assert fields["cost_source"] == "worker-management"
    assert fields["provider"] == "anthropic"
    assert fields["served_model"] == "claude-opus-4-8"
    assert fields["input_tokens"] == "1000"
    assert fields["output_tokens"] == "500"
    assert fields["latency_ms"] == "1234"
    # metering's rollup keys cost_daily.byo off usage_extra["byo"] == "true"
    ux = json.loads(fields["usage_extra"])
    assert ux["byo"] == "true"
    # tenant-BYO (default): no owner_provided provenance marker.
    assert "owner_provided" not in ux
    # cost_usd is advisory (metering recomputes); must be a stringified number
    assert float(fields["cost_usd"]) == 0.0
    assert kwargs.get("approximate") is True
    # correlation defaults to empty when the caller did not stamp it.
    assert fields["correlation_id"] == ""
    assert fields["stage"] == ""


@pytest.mark.asyncio
async def test_emit_served_model_overrides_requested_alias(monkeypatch):
    """served_model carries the RESOLVED model; model keeps the requested alias.

    This is issue #137: WM asks inference-router for a routing alias
    ("tier-medium"), IR resolves it server-side, and billing prices on
    served_model. Emitting the alias there = no provider_pricing row =
    cost_usd NULL forever. The alias and the served name must differ here,
    otherwise the assertion cannot detect the copy-the-alias bug.
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="deepseek",
        model="tier-medium",
        served_model="deepseek-v4-pro",
        input_tokens=10,
        output_tokens=20,
        latency_ms=5,
    )
    fields = fake_client.xadd.call_args[0][1]
    # Requested alias retained for attribution...
    assert fields["model"] == "tier-medium"
    # ...but billing prices on the concrete resolved model.
    assert fields["served_model"] == "deepseek-v4-pro"
    assert isinstance(fields["served_model"], str)
    assert fields["provider"] == "deepseek"


@pytest.mark.asyncio
async def test_emit_served_model_falls_back_to_model(monkeypatch):
    """Omitted served_model (an IR that predates the field) => old behaviour."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )
    fields = fake_client.xadd.call_args[0][1]
    assert fields["served_model"] == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_task_forwards_served_model(monkeypatch):
    """emit_byo_cost_event_task threads served_model through to _emit."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")
    emit_mock = AsyncMock()
    monkeypatch.setattr(cost_events, "_emit", emit_mock)
    cost_events.emit_byo_cost_event_task(
        provider="deepseek",
        model="tier-medium",
        served_model="deepseek-v4-pro",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        # Required since the double-emit fix: the emitter REFUSES a
        # platform-routed model_provider (inference-router already metered it).
        model_provider="anthropic",
    )
    await asyncio.sleep(0)
    emit_mock.assert_awaited_once()
    assert emit_mock.await_args.kwargs["served_model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_emit_includes_correlation_and_stage(monkeypatch):
    """_emit forwards correlation_id + stage onto the metering event when set."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        correlation_id="run-abc",
        stage="script_generation",
    )
    fields = fake_client.xadd.call_args[0][1]
    assert fields["correlation_id"] == "run-abc"
    assert fields["stage"] == "script_generation"


@pytest.mark.asyncio
async def test_emit_owner_provided_omits_byo(monkeypatch):
    """Managed BYO: _emit stamps owner_provided="true" and NO byo flag."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        owner_provided=True,
    )
    ux = json.loads(fake_client.xadd.call_args[0][1]["usage_extra"])
    assert ux["owner_provided"] == "true"
    assert "byo" not in ux


@pytest.mark.asyncio
async def test_task_forwards_owner_provided(monkeypatch):
    """emit_byo_cost_event_task threads owner_provided through to _emit."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")
    emit_mock = AsyncMock()
    monkeypatch.setattr(cost_events, "_emit", emit_mock)
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        owner_provided=True,
        model_provider="anthropic",
    )
    await asyncio.sleep(0)  # let the fire-and-forget task run
    emit_mock.assert_awaited_once()
    assert emit_mock.await_args.kwargs["owner_provided"] is True


@pytest.mark.asyncio
async def test_task_forwards_correlation_from_context(monkeypatch):
    """The task wrapper reads correlation_id + stage from the identity context."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "run-xyz")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "narration")
    emit_mock = AsyncMock()
    monkeypatch.setattr(cost_events, "_emit", emit_mock)
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    # Let the fire-and-forget task scheduled via create_task() run.
    await asyncio.sleep(0)
    emit_mock.assert_awaited_once()
    kw = emit_mock.await_args.kwargs
    assert kw["correlation_id"] == "run-xyz"
    assert kw["stage"] == "narration"


@pytest.mark.asyncio
async def test_task_is_retained_until_complete_then_discarded(monkeypatch):
    """The scheduled emit task is held in _background_tasks (a strong ref, so it
    cannot be GC'd mid-XADD) and removed by its done-callback once it finishes."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")
    cost_events._background_tasks.clear()

    release = asyncio.Event()

    async def _slow_emit(**_):
        await release.wait()

    monkeypatch.setattr(cost_events, "_emit", _slow_emit)
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    # Strong ref retained while the emit is in flight.
    assert len(cost_events._background_tasks) == 1
    # Let it finish; the done-callback must remove it.
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(cost_events._background_tasks) == 0


@pytest.mark.asyncio
async def test_emit_noop_when_redis_unset(monkeypatch):
    """_emit returns silently (no xadd) when there is no metering client."""
    monkeypatch.setattr(cost_events, "_get_redis", lambda: None)
    # Should not raise.
    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="t",
        license_id="l",
        subject_id="s",
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )


def test_task_noop_when_redis_url_empty(monkeypatch):
    """The task wrapper no-ops (no coroutine scheduled) when the URL is unset."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings(url=""))
    called = {"emit": False}
    monkeypatch.setattr(cost_events, "_emit", lambda **_: called.__setitem__("emit", True))
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    assert called["emit"] is False


def test_task_noop_when_no_tenant(monkeypatch):
    """The task wrapper skips emission when no tenant is in context."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "")
    called = {"emit": False}
    monkeypatch.setattr(cost_events, "_emit", lambda **_: called.__setitem__("emit", True))
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    assert called["emit"] is False


@pytest.mark.asyncio
async def test_emit_includes_caller_client_id(monkeypatch):
    """The caller of THIS hop lands on the wire.

    Measured on prod 2026-09-04: no emitter put a caller on the wire at all, so
    every app-to-app cost event in the ledger was anonymous as to who called —
    which is why an app-to-app hop was billed to whichever service emitted the
    event rather than the one that asked.
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        caller_client_id="svc-requirements",
    )

    fields = fake_client.xadd.call_args[0][1]
    # CONTROL: an already-shipped field. If this fails the harness is broken and
    # the assertion below proves nothing.
    assert fields["service_key"] == "worker-management"
    assert fields["caller_client_id"] == "svc-requirements"


@pytest.mark.asyncio
async def test_emit_absent_caller_is_empty_not_invented(monkeypatch):
    """No verified caller must stay distinguishable from a known one.

    Most of today's traffic genuinely has none. Billing keeps an absent caller
    absent (its whitelist copies the key only when truthy), so a fabricated value
    here would be strictly worse than the empty string.
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )

    fields = fake_client.xadd.call_args[0][1]
    assert fields["service_key"] == "worker-management"  # CONTROL
    assert fields["caller_client_id"] == ""


@pytest.mark.asyncio
async def test_task_forwards_caller_client_id_from_context(monkeypatch):
    """The task wrapper reads caller_client_id from the identity context.

    It must be read at the CALL SITE, like every other identity field, not inside
    the background task — the ContextVar is live in the request context there.
    """
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")
    monkeypatch.setattr(
        cost_events, "get_caller_client_id_from_context", lambda: "svc-requirements"
    )
    emit_mock = AsyncMock()
    monkeypatch.setattr(cost_events, "_emit", emit_mock)

    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    await asyncio.sleep(0)

    emit_mock.assert_awaited_once()
    kw = emit_mock.await_args.kwargs
    assert kw["tenant_id"] == "tenant-1"  # CONTROL
    assert kw["caller_client_id"] == "svc-requirements"


@pytest.mark.asyncio
async def test_emit_includes_app_id(monkeypatch):
    """The calling application's id lands on the wire when the caller stamped one.

    app_id was hardcoded to "" here; app_id IS NULL is the estate's largest
    attribution gap ($4,384.57 of $5,081.86 over 30d, measured 2026-09-04).
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        app_id="requirements_management",
    )

    fields = fake_client.xadd.call_args[0][1]
    assert fields["service_key"] == "worker-management"  # CONTROL
    assert fields["app_id"] == "requirements_management"


@pytest.mark.asyncio
async def test_emit_absent_app_id_is_empty_not_invented(monkeypatch):
    """No inbound X-App-Id must stay distinguishable from a known one.

    worker_management fans in from many upstream apps. A guessed/defaulted
    app_id here would collapse all of their distinct origins into one bucket
    — strictly worse than the empty string this pins.
    """
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="lic-1",
        subject_id="sub-1",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )

    fields = fake_client.xadd.call_args[0][1]
    assert fields["service_key"] == "worker-management"  # CONTROL
    assert fields["app_id"] == ""


@pytest.mark.asyncio
async def test_task_forwards_app_id_from_context(monkeypatch):
    """The task wrapper reads app_id from the identity context, at the CALL
    SITE like every other identity field — not inside the background task."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "tenant-1")
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_caller_client_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_app_id_from_context", lambda: "requirements_management")
    emit_mock = AsyncMock()
    monkeypatch.setattr(cost_events, "_emit", emit_mock)

    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_provider="anthropic",
    )
    await asyncio.sleep(0)

    emit_mock.assert_awaited_once()
    kw = emit_mock.await_args.kwargs
    assert kw["tenant_id"] == "tenant-1"  # CONTROL
    assert kw["app_id"] == "requirements_management"
