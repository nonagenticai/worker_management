"""Every silent BYO cost-event drop is counted and said out loud.

``emit_byo_cost_event_task`` has three paths that return WITHOUT emitting.
Before this suite one of them logged nothing at all and two logged at DEBUG —
and these services run at ``LOG_LEVEL=INFO``, where DEBUG is deletion, not a
quieter log. So real BYO spend left the platform unmetered and nothing, in the
logs or in Prometheus, recorded that a line had gone missing.

Each test below drives ONE drop path and asserts both halves of the fix
together, because either alone still loses the event:

* the counter increments under that path's OWN ``reason`` label — they must not
  share a series, or the permanently-non-zero benign reason
  (``metering_unconfigured``) forces any alert on the real one
  (``no_tenant``) to be silenced;
* the message lands at WARNING. ``caplog`` is pinned to WARNING rather than
  read at whatever level it defaults to, so a regression back to
  ``logger.debug`` produces an EMPTY record list and fails, instead of being
  captured anyway and passing.

The reason strings are imported from ``agent_metrics`` rather than retyped, so
a rename that desynchronises the emitter from the pre-materialised label set
breaks here rather than silently creating a fourth, unmaterialised series.
"""

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Must precede the import of the module under test (see tests/test_agent_metrics.py).
os.environ["OTEL_SDK_DISABLED"] = "true"

from src.telemetry import cost_events
from src.telemetry.agent_metrics import (
    METERING_DROP_REASON_NO_EVENT_LOOP,
    METERING_DROP_REASON_NO_TENANT,
    METERING_DROP_REASON_UNCONFIGURED,
    METERING_DROP_REASONS,
)

TENANT = "11111111-1111-1111-1111-111111111111"


def _stub_settings(url: str = "redis://metering:6379/0"):
    return SimpleNamespace(metering_redis_url=url, metering_costs_stream="improvement-loop:costs")


@pytest.fixture
def metrics(monkeypatch):
    """Intercept the metrics client at the emitter's own call site."""
    client = MagicMock()
    monkeypatch.setattr(cost_events, "get_metrics_client", lambda: client)
    return client


def _emit_task(*, model_provider: str = "anthropic") -> None:
    """One representative BYO call. Explicit kwargs, not a **dict.

    A ``dict`` of mixed str/int values widens to ``str | int`` and the
    typechecker cannot then match it against the signature.
    """
    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="tier-medium",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=42,
        model_provider=model_provider,
        served_model="claude-opus-4-8",
    )


def _sole_drop_reason(metrics) -> str:
    """The one reason passed to the drop counter, or fail loudly."""
    calls = metrics.record_metering_event_dropped.call_args_list
    assert len(calls) == 1, f"expected exactly one drop, got {calls}"
    return calls[0].args[0]


# ---------------------------------------------------------------------------
# The closed label set
# ---------------------------------------------------------------------------


def test_reasons_are_the_closed_prematerialised_set():
    """CONTROL: the three reasons under test are exactly the materialised set.

    If this fails, every assertion below is testing a label that has no
    pre-materialised series — a zero would be indistinguishable from "this code
    path was never deployed".
    """
    assert set(METERING_DROP_REASONS) == {
        METERING_DROP_REASON_NO_TENANT,
        METERING_DROP_REASON_UNCONFIGURED,
        METERING_DROP_REASON_NO_EVENT_LOOP,
    }


# ---------------------------------------------------------------------------
# reason=metering_unconfigured
# ---------------------------------------------------------------------------


def test_unconfigured_drop_counts_and_warns(monkeypatch, caplog, metrics):
    """METERING_REDIS_URL unset: benign as config, still lost spend."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings(url=""))
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: TENANT)
    monkeypatch.setattr(cost_events, "_emit", lambda **_: pytest.fail("must not emit"))

    with caplog.at_level(logging.WARNING, logger=cost_events.logger.name):
        _emit_task()

    assert _sole_drop_reason(metrics) == METERING_DROP_REASON_UNCONFIGURED
    metrics.record_metering_event_emitted.assert_not_called()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert METERING_DROP_REASON_UNCONFIGURED in record.getMessage()


# ---------------------------------------------------------------------------
# reason=no_tenant -- the defect case
# ---------------------------------------------------------------------------


def test_no_tenant_drop_counts_and_warns(monkeypatch, caplog, metrics):
    """Metering LIVE, call real, no owner to bill. This is THE defect path."""
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "_emit", lambda **_: pytest.fail("must not emit"))

    with caplog.at_level(logging.WARNING, logger=cost_events.logger.name):
        _emit_task()

    assert _sole_drop_reason(metrics) == METERING_DROP_REASON_NO_TENANT
    metrics.record_metering_event_emitted.assert_not_called()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert METERING_DROP_REASON_NO_TENANT in message
    # The counter is deliberately unlabelled by provider/model; the operator
    # only learns WHICH call was lost from this line, so it must carry it.
    assert "anthropic" in message
    assert "claude-opus-4-8" in message


def test_no_tenant_drop_does_not_share_a_series_with_unconfigured(monkeypatch, caplog, metrics):
    """A no-tenant drop must never be attributed to the benign reason.

    Collapsing the two is the failure this label set exists to prevent:
    `metering_unconfigured` sits non-zero for the life of an unconfigured
    deployment, so a shared series makes the real signal unalertable.
    """
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: None)
    monkeypatch.setattr(cost_events, "_emit", lambda **_: pytest.fail("must not emit"))

    with caplog.at_level(logging.WARNING, logger=cost_events.logger.name):
        _emit_task()

    assert _sole_drop_reason(metrics) != METERING_DROP_REASON_UNCONFIGURED


# ---------------------------------------------------------------------------
# reason=no_event_loop
# ---------------------------------------------------------------------------


def test_no_event_loop_drop_counts_and_warns(monkeypatch, caplog, metrics):
    """Sync context: the emit coroutine is closed unrun, so the event is lost.

    Deliberately does NOT stub ``_emit`` -- the real coroutine object has to be
    built for ``coro.close()`` to be the thing under test. Nothing awaits it,
    so no Redis connection is ever opened.
    """
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: TENANT)

    # This test body is sync, so there is no running loop -- which IS the
    # precondition under test.
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    with caplog.at_level(logging.WARNING, logger=cost_events.logger.name):
        _emit_task()

    assert _sole_drop_reason(metrics) == METERING_DROP_REASON_NO_EVENT_LOOP
    metrics.record_metering_event_emitted.assert_not_called()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert METERING_DROP_REASON_NO_EVENT_LOOP in record.getMessage()


# ---------------------------------------------------------------------------
# The denominator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_increments_the_denominator(monkeypatch, metrics):
    """A scheduled emit counts as emitted and NOT as dropped.

    Without this the drop counter has no denominator and "40 drops" is
    unreadable: a rounding error against 4M emits, an outage against 40.
    """
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: TENANT)
    monkeypatch.setattr(cost_events, "get_tenant_provenance_from_context", lambda: "delegated")
    monkeypatch.setattr(cost_events, "_emit", AsyncMock())

    _emit_task()
    await asyncio.sleep(0)  # let the scheduled task run so it is not destroyed pending

    metrics.record_metering_event_emitted.assert_called_once_with()
    metrics.record_metering_event_dropped.assert_not_called()
    # ⛔ THE INSTRUMENT MUST BE PINNED TOO. This change exists to make an
    # unproven attribution VISIBLE, so a counter that can be deleted, silenced
    # or hardcoded with the suite still green would leave the fix unmeasurable
    # — the exact defect it was written to close, one level up. A mutation
    # battery proved this: without this line, three separate mutants of the
    # metric survived.
    metrics.record_metering_event_tenant_provenance.assert_called_once_with("delegated")


@pytest.mark.asyncio
async def test_platform_routed_refusal_is_not_counted_as_a_drop(monkeypatch, metrics):
    """The double-bill refusal is a REFUSAL, not lost spend.

    inference-router already metered that call, so nothing was lost and it must
    not inflate a counter whose every increment is meant to be unbilled money.
    """
    monkeypatch.setattr(cost_events, "get_settings", lambda: _stub_settings())
    monkeypatch.setattr(cost_events, "get_tenant_id_from_context", lambda: TENANT)
    monkeypatch.setattr(cost_events, "_emit", AsyncMock())

    routed = next(iter(cost_events.PLATFORM_ROUTED_PROVIDERS))
    _emit_task(model_provider=routed)

    metrics.record_metering_event_dropped.assert_not_called()
    metrics.record_metering_event_emitted.assert_not_called()
