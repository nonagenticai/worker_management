"""The metering cost-event stream default must name a stream that is READ.

``metering_costs_stream`` feeds ``_emit(stream=...)`` in
``src/telemetry/cost_events.py``, and Redis ``XADD`` creates the key on first
write. So a default naming a stream nothing consumes does not fail, does not
warn, and does not raise — it silently manufactures a brand-new stream and
drops every BYO cost event into it. The failure is indistinguishable from
success at the emitting end; only the absent bill shows it.

The default has been wrong twice. ``metering:costs`` was replaced by
``improvement-loop:costs`` in #92 (2026-06-29, "drop dead-stream default"), and
that replacement was itself dead: measured on the live billing Redis,
``improvement-loop:costs`` had ``EXISTS`` = 0 (never created) while
``billing:costs`` held 36,464 entries. A cluster-wide sweep of every Deployment
env found no workload pointing at the dead name.

This is INERT in production today: the running pod sets
``METERING_COSTS_STREAM=billing:costs`` explicitly (gitops
``clusters/strato/services/workers/deployment.yaml``), so nothing is currently
being lost and this test recovers no money. What it pins is the *unset* case —
a future pod, a local run, a new overlay that omits that env writes to a real
stream instead of a nonexistent one. That is a property no deployment manifest
can assert, which is why it is asserted here.

See issue #234.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from src.settings import Settings, get_settings, reset_settings_cache

if TYPE_CHECKING:
    from collections.abc import Iterator

# The live stream, and the two dead names that have shipped as the default.
_LIVE_STREAM = "billing:costs"
_DEAD_STREAMS = ("improvement-loop:costs", "metering:costs")

# case_sensitive=False, so the lower-case spelling binds too and would leak
# between tests if a sibling module left one behind.
_ENV_KEYS = ("METERING_COSTS_STREAM", "metering_costs_stream")


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """Every assertion here is about the UNSET case; guarantee it is unset."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    reset_settings_cache()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_settings_cache()


class TestMeteringStreamDefault:
    def test_default_with_env_unset_is_the_live_stream(self) -> None:
        """The literal ask of #234: construct with the env unset, get a real stream."""
        # model_validate, not dict-kwarg unpack — the latter is rejected by ty CI.
        assert Settings.model_validate({}).metering_costs_stream == _LIVE_STREAM

    def test_default_is_not_a_dead_stream(self) -> None:
        """Named tripwire. A revert to either dead value fails HERE, with the
        name in the assertion, rather than as an unexplained billing gap."""
        assert Settings.model_validate({}).metering_costs_stream not in _DEAD_STREAMS

    def test_the_emitter_reads_the_live_stream_by_default(self) -> None:
        """``cost_events`` resolves the stream through ``get_settings()`` at call
        time, so pin it there too: asserting the Field default alone would not
        catch a cached or overridden settings object reaching the emitter."""
        assert get_settings().metering_costs_stream == _LIVE_STREAM

    def test_env_still_overrides_the_default(self) -> None:
        """The default must not be hardcoded past the env: production sets
        METERING_COSTS_STREAM explicitly and that must keep winning."""
        os.environ["METERING_COSTS_STREAM"] = "some-other:costs"
        reset_settings_cache()
        assert Settings().metering_costs_stream == "some-other:costs"
        assert get_settings().metering_costs_stream == "some-other:costs"
