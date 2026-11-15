"""Tests for src/telemetry/agent_metrics.py.

All tests run with OTEL_SDK_DISABLED=true to avoid connecting to Grafana.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

# Ensure OTEL is disabled before importing the module under test
os.environ["OTEL_SDK_DISABLED"] = "true"

import src.telemetry.agent_metrics as agent_metrics_module
from src.telemetry.agent_metrics import (
    AgentExecutionTimer,
    AgentMetricsClient,
    get_metrics_client,
)

# ---------------------------------------------------------------------------
# AgentMetricsClient - disabled behaviour
# ---------------------------------------------------------------------------


class TestAgentMetricsClientDisabled:
    """Tests for AgentMetricsClient when OTEL_SDK_DISABLED=true."""

    def test_disabled_via_env_var(self):
        """When OTEL_SDK_DISABLED=true, the client should set enabled=False."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient(service_name="test-service")
            assert client.enabled is False
            assert client.meter is None

    def test_record_execution_is_noop_when_disabled(self):
        """record_execution should silently return without doing anything."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            # Should not raise
            client.record_execution(
                agent_name="test-agent",
                agent_type="pydantic_ai",
                model_provider="openai",
                model_name="gpt-4",
                duration_seconds=1.5,
                success=True,
                token_usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            )
            # Counters and histograms should be empty dicts (never populated)
            assert len(client._counters) == 0
            assert len(client._histograms) == 0

    def test_record_agent_start_is_noop_when_disabled(self):
        """record_agent_start should be a no-op when disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            # Should not raise
            client.record_agent_start("test-agent", "claude_code")
            assert len(client._up_down_counters) == 0

    def test_record_agent_end_is_noop_when_disabled(self):
        """record_agent_end should be a no-op when disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_agent_end("test-agent", "claude_code")
            assert len(client._up_down_counters) == 0

    def test_record_aa_prompt_execution_is_noop_when_disabled(self):
        """record_aa_prompt_execution should be a no-op when disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_aa_prompt_execution(
                prompt_id="p1",
                worker_name="w1",
                status="success",
                latency_seconds=0.5,
                input_tokens=10,
                output_tokens=20,
            )
            assert len(client._counters) == 0

    def test_record_aa_prompt_validation_failure_is_noop_when_disabled(self):
        """record_aa_prompt_validation_failure should be a no-op when disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_aa_prompt_validation_failure(
                prompt_id="p1",
                worker_name="w1",
                error_type="json_parse",
            )
            assert len(client._counters) == 0


# ---------------------------------------------------------------------------
# AgentMetricsClient - enabled behaviour (with mocked OTel)
# ---------------------------------------------------------------------------


class TestAgentMetricsClientEnabled:
    """Tests for AgentMetricsClient when enabled (mock OTel + requests)."""

    def _make_enabled_client(self):
        """Create an enabled client by mocking all OTel internals."""
        client = AgentMetricsClient.__new__(AgentMetricsClient)
        client.enabled = True
        client._counters = {}
        client._histograms = {}
        client._up_down_counters = {}

        # Build mock meter + instruments
        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_updown = MagicMock()

        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_up_down_counter.return_value = mock_updown

        client.meter = mock_meter
        client._setup_metrics()
        return client

    def test_record_execution_success_increments_counter(self):
        """Successful execution should increment executions_total."""
        client = self._make_enabled_client()
        client.record_execution(
            agent_name="a",
            agent_type="pydantic_ai",
            success=True,
            duration_seconds=2.0,
        )
        client._counters["executions_total"].add.assert_called()
        client._histograms["execution_duration"].record.assert_called()

    def test_record_metering_event_tenant_provenance_actually_increments(self):
        """⛔ THE SILENCE PIN. Proves the method BODY does work, not just that
        callers call it.

        A mutation battery caught this: the emit-site test asserts against a
        MOCKED metrics client, so turning this method into a no-op was
        invisible to it — the counter could be silenced fleet-wide with the
        suite fully green. An instrument that can be silenced without a test
        failing is not an instrument.
        """
        client = self._make_enabled_client()
        client.record_metering_event_tenant_provenance("delegated")
        client._counters["metering_event_tenant_provenance"].add.assert_called()
        args = client._counters["metering_event_tenant_provenance"].add.call_args
        assert args.args[0] == 1, args
        assert args.args[1] == {"provenance": "delegated"}, args

    def test_record_execution_failure_increments_failed_counter(self):
        """Failed execution should increment executions_failed."""
        client = self._make_enabled_client()
        client.record_execution(
            agent_name="a",
            agent_type="pydantic_ai",
            success=False,
        )
        client._counters["executions_failed"].add.assert_called()

    def test_record_execution_with_tokens(self):
        """Token usage should be forwarded to the token counters."""
        client = self._make_enabled_client()
        client.record_execution(
            agent_name="a",
            agent_type="pydantic_ai",
            success=True,
            token_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        client._counters["tokens_total"].add.assert_called()
        client._counters["tokens_prompt"].add.assert_called()
        client._counters["tokens_completion"].add.assert_called()

    def test_record_agent_start_increments_active(self):
        """record_agent_start should add 1 to active_agents."""
        client = self._make_enabled_client()
        client.record_agent_start("a", "claude_code")
        client._up_down_counters["active_agents"].add.assert_called_with(
            1, {"agent_name": "a", "agent_type": "claude_code"}
        )

    def test_record_agent_end_decrements_active(self):
        """record_agent_end should add -1 to active_agents."""
        client = self._make_enabled_client()
        client.record_agent_end("a", "claude_code")
        client._up_down_counters["active_agents"].add.assert_called_with(
            -1, {"agent_name": "a", "agent_type": "claude_code"}
        )

    def test_record_aa_prompt_execution_records_all_metrics(self):
        """AA prompt execution should record counter, latency, and tokens."""
        client = self._make_enabled_client()
        client.record_aa_prompt_execution(
            prompt_id="p1",
            worker_name="w1",
            status="success",
            latency_seconds=0.5,
            input_tokens=100,
            output_tokens=50,
            cost_cents=0.03,
            model="gpt-4",
        )
        client._counters["aa_prompt_execution"].add.assert_called()
        client._histograms["aa_prompt_latency"].record.assert_called()
        # All counters share a single mock, so just verify add was called multiple times
        # (execution counter + token input + token output + cost = at least 4 calls)
        assert client._counters["aa_prompt_tokens"].add.call_count >= 2

    def test_record_aa_prompt_validation_failure(self):
        """AA validation failure should increment the failure counter."""
        client = self._make_enabled_client()
        client.record_aa_prompt_validation_failure("p1", "w1", "schema_invalid")
        client._counters["aa_prompt_validation_failures"].add.assert_called_with(
            1,
            {"prompt_id": "p1", "worker_name": "w1", "error_type": "schema_invalid"},
        )

    def test_record_unauthenticated_tenant_assertion(self):
        """The tokenless-tenant-assertion counter takes only low-cardinality labels.

        Tenant and peer are caller-controlled on the branch this counts, so the
        signature deliberately has nowhere to put them. See
        src/api/auth_middleware._audit_unauthenticated_tenant_assertion.
        """
        client = self._make_enabled_client()
        client.record_unauthenticated_tenant_assertion("/api/workers", enforced=False)
        client._counters["unauthenticated_tenant_assertion"].add.assert_called_with(
            1,
            {"path": "/api/workers", "enforced": "false"},
        )
        client.record_unauthenticated_tenant_assertion("/api/workers", enforced=True)
        client._counters["unauthenticated_tenant_assertion"].add.assert_called_with(
            1,
            {"path": "/api/workers", "enforced": "true"},
        )

    def test_record_unauthenticated_tenant_assertion_is_noop_when_disabled(self):
        """Metrics being off must never turn the auth path into a 500."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_unauthenticated_tenant_assertion("/api/workers", enforced=True)
            assert len(client._counters) == 0

    def test_record_identity_omission(self):
        """The omission counter carries field, reason, caller and route shape.

        Every one of those four is bounded: two field names, a closed reason
        set, a VERIFIED (allowlisted) JWT azp, and a collapsed route shape.
        Tenant and peer are absent by construction — the signature has nowhere
        to put them. See src/api/auth_middleware._audit_identity_omission.
        """
        client = self._make_enabled_client()
        client.record_identity_omission(
            "license_id",
            reason="trusted_network_tokenless_asserted_tenant",
            caller="kag-service",
            path="/api/workers",
        )
        client._counters["identity_omitted"].add.assert_called_with(
            1,
            {
                "field": "license_id",
                "reason": "trusted_network_tokenless_asserted_tenant",
                "caller": "kag-service",
                "path": "/api/workers",
            },
        )

    def test_record_identity_omission_is_noop_when_disabled(self):
        """Metrics being off must never turn the auth path into a 500."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_identity_omission("subject_id", reason="r", caller="c", path="/p")
            assert len(client._counters) == 0

    def test_record_identity_omission_swallows_backend_errors(self):
        """⛔ Observe-only: a broken counter must not propagate into auth.

        This method is called from the middleware's tokenless branch on every
        request. If it could raise, adding an observation would have turned a
        served request into a 500 — the exact opposite of observe-only.
        """
        client = self._make_enabled_client()
        client._counters["identity_omitted"].add.side_effect = RuntimeError("otlp down")
        client.record_identity_omission(
            "license_id", reason="r", caller="c", path="/p"
        )  # must not raise


# ---------------------------------------------------------------------------
# AgentExecutionTimer
# ---------------------------------------------------------------------------


class TestAgentExecutionTimer:
    """Tests for the AgentExecutionTimer context manager."""

    def test_context_manager_records_timing_on_success(self):
        """__enter__/__exit__ should record start, execution, and end."""
        mock_client = MagicMock()
        mock_client.record_agent_start = MagicMock()
        mock_client.record_agent_end = MagicMock()
        mock_client.record_execution = MagicMock()

        with patch.object(agent_metrics_module, "_metrics_client", None):
            with patch.object(agent_metrics_module, "get_metrics_client", return_value=mock_client):
                timer = AgentExecutionTimer(
                    agent_name="test-agent",
                    agent_type="pydantic_ai",
                    model_provider="openai",
                    model_name="gpt-4",
                )
                timer.metrics_client = mock_client

                with timer:
                    # Simulate some work
                    time.sleep(0.01)

        # record_agent_start called with correct args
        mock_client.record_agent_start.assert_called_once_with("test-agent", "pydantic_ai")

        # record_execution called with success=True and a positive duration
        mock_client.record_execution.assert_called_once()
        call_kwargs = mock_client.record_execution.call_args
        assert call_kwargs.kwargs["success"] is True
        assert call_kwargs.kwargs["duration_seconds"] > 0
        assert call_kwargs.kwargs["agent_name"] == "test-agent"

        # record_agent_end called
        mock_client.record_agent_end.assert_called_once_with("test-agent", "pydantic_ai")

    def test_context_manager_records_failure_on_exception(self):
        """On exception, record_execution should be called with success=False."""
        mock_client = MagicMock()

        timer = AgentExecutionTimer(
            agent_name="failing-agent",
            agent_type="claude_code",
        )
        timer.metrics_client = mock_client

        with pytest.raises(ValueError, match="boom"), timer:
            raise ValueError("boom")

        # Should still have recorded execution with success=False
        mock_client.record_execution.assert_called_once()
        call_kwargs = mock_client.record_execution.call_args
        assert call_kwargs.kwargs["success"] is False

        # Agent end should still be called (always runs)
        mock_client.record_agent_end.assert_called_once()

    def test_record_tokens_delegates_to_metrics_client(self):
        """record_tokens should call metrics_client.record_execution with token data."""
        mock_client = MagicMock()

        timer = AgentExecutionTimer(
            agent_name="token-agent",
            agent_type="pydantic_ai",
            model_provider="mlx",
            model_name="llama-3",
        )
        timer.metrics_client = mock_client

        with timer:
            token_usage = {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
            timer.record_tokens(token_usage)

        # record_execution called from record_tokens (with token_usage) + from __exit__ (without)
        assert mock_client.record_execution.call_count == 2

        # The first call should be from record_tokens and include token_usage
        first_call_kwargs = mock_client.record_execution.call_args_list[0].kwargs
        assert first_call_kwargs["token_usage"] == token_usage
        assert first_call_kwargs["success"] is True

    def test_record_tokens_before_enter_does_nothing(self):
        """record_tokens called before __enter__ should not record (start_time is None)."""
        mock_client = MagicMock()

        timer = AgentExecutionTimer(
            agent_name="early-agent",
            agent_type="pydantic_ai",
        )
        timer.metrics_client = mock_client

        # Call record_tokens without entering context manager
        timer.record_tokens({"total_tokens": 50})

        # Should not have called record_execution because start_time is None
        mock_client.record_execution.assert_not_called()


# ---------------------------------------------------------------------------
# get_metrics_client singleton
# ---------------------------------------------------------------------------


class TestGetMetricsClient:
    """Tests for the get_metrics_client singleton factory."""

    def test_returns_same_instance(self):
        """get_metrics_client should return the same object on repeated calls."""
        # Reset the global singleton
        agent_metrics_module._metrics_client = None

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            first = get_metrics_client()
            second = get_metrics_client()
            assert first is second

        # Clean up
        agent_metrics_module._metrics_client = None

    def test_creates_instance_if_none(self):
        """get_metrics_client should create a new instance when _metrics_client is None."""
        agent_metrics_module._metrics_client = None

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = get_metrics_client()
            assert isinstance(client, AgentMetricsClient)

        agent_metrics_module._metrics_client = None


# ---------------------------------------------------------------------------
# Prometheus exported-name invariant
# ---------------------------------------------------------------------------


class TestPrometheusExportedNames:
    """The name Prometheus ends up holding must equal the name declared here.

    The OTLP -> Prometheus exporter appends an instrument's ``unit`` to its
    metric name unless the name already contains that unit token, and only then
    applies the ``_total`` suffix convention for monotonic counters. A
    descriptive unit therefore silently *renames* the metric.

    Measured live on 2026-08-13, three instruments in this file were shipping
    under names nothing could query:

        agent_max_tokens_cap_hit_total + "hits"       -> agent_max_tokens_cap_hit_hits_total
        agent_active_count             + "agents"     -> agent_active_count_agents
        aa_prompt_execution_total      + "executions" -> aa_prompt_execution_executions_total

    The first one is the cap-hit detector: every alert written against the name
    in the source matched an empty vector, silently. Asserting on the argument
    passed to ``create_counter`` would NOT have caught this -- that argument was
    always correct. The bug is in the mapping, so the test has to model the
    mapping.
    """

    # Units the exporter treats as dimensionless and never appends.
    DIMENSIONLESS = frozenset({"1", "", "s", "By"})

    @staticmethod
    def _declared_instruments():
        """Return [(kind, name, unit)] for every instrument _setup_metrics creates."""
        client = AgentMetricsClient.__new__(AgentMetricsClient)
        client.meter = MagicMock()
        client._counters = {}
        client._histograms = {}
        client._up_down_counters = {}
        client._setup_metrics()

        declared = []
        for kind in ("create_counter", "create_histogram", "create_up_down_counter"):
            for call in getattr(client.meter, kind).call_args_list:
                declared.append((kind, call.kwargs["name"], call.kwargs.get("unit", "")))
        return declared

    def test_setup_creates_instruments(self):
        """Guard the guard: if _setup_metrics stops registering, the loop below is vacuous."""
        declared = self._declared_instruments()
        assert len(declared) >= 10, f"expected the full instrument set, got {declared}"

    def test_no_instrument_unit_is_appended_to_its_name(self):
        """A unit token absent from the name gets appended, renaming the metric."""
        offenders = [
            (name, unit)
            for _kind, name, unit in self._declared_instruments()
            if unit not in self.DIMENSIONLESS and unit not in name
        ]
        assert offenders == [], (
            "These instruments will be exported under a different name than they "
            "declare, because the exporter appends the unit: "
            + ", ".join(f"{n!r} (unit={u!r}) -> {n}_{u}" for n, u in offenders)
            + ". Set unit='1'."
        )

    def test_cap_hit_counter_exports_under_its_declared_name(self):
        """Pin the specific metric that shipped broken, by name."""
        declared = {name: unit for _kind, name, unit in self._declared_instruments()}
        assert "agent_max_tokens_cap_hit_total" in declared
        assert declared["agent_max_tokens_cap_hit_total"] in self.DIMENSIONLESS


# ---------------------------------------------------------------------------
# Worker state-machine observability (observe-only counters)
# ---------------------------------------------------------------------------


class TestWorkerStateCounters:
    """The two counters that make the worker state machine's behaviour readable.

    ⛔ Both are OBSERVE-ONLY. The state machine fails open and three of four
    write paths never call the validator at all; making it refuse was measured
    at ~142,560 rejections / 30 days (≈55% of traffic). These tests pin the
    label sets and the pre-materialised zeros, not any rejection behaviour.
    """

    @staticmethod
    def _client_with_distinct_counters():
        """Enabled client where every counter is its OWN mock.

        ``TestAgentMetricsClientEnabled._make_enabled_client`` returns one shared
        mock for every ``create_counter`` call, so "this counter fired once"
        cannot be expressed against it — every other counter's
        pre-materialisation lands on the same object.
        """
        client = AgentMetricsClient.__new__(AgentMetricsClient)
        client.enabled = True
        client._counters = {}
        client._histograms = {}
        client._up_down_counters = {}

        mock_meter = MagicMock()
        mock_meter.create_counter.side_effect = lambda **_kw: MagicMock()
        mock_meter.create_histogram.side_effect = lambda **_kw: MagicMock()
        mock_meter.create_up_down_counter.side_effect = lambda **_kw: MagicMock()
        client.meter = mock_meter
        client._setup_metrics()
        return client

    # -- label sets are closed and match the enum ---------------------------

    def test_state_type_classes_match_the_enum(self):
        """The duplicated tuple must not drift from WorkerStateType.

        It is duplicated rather than imported because ``worker_state_manager``
        imports this module (cycle). This test is the thing that makes the
        duplication safe.
        """
        from src.services.worker_state_manager import WorkerStateType

        declared = set(agent_metrics_module.WORKER_STATE_TYPE_CLASSES)
        declared.discard(agent_metrics_module.WORKER_STATE_TYPE_UNKNOWN)
        assert declared == {m.value for m in WorkerStateType}

    def test_classify_known_member_returns_itself(self):
        for member in (
            "created",
            "processing",
            "completed",
            "error",
            "idle",
            "active",
            "default",
            "config-updated",
            "updated",
        ):
            assert agent_metrics_module.classify_worker_state_type(member) == member

    def test_classify_config_updated_is_itself_now_that_it_is_a_member(self):
        """⚠️ HYPHEN. INVERTED 2026-09-13: it used to assert ``<unknown>``.

        ``config-updated`` (153,087 live rows) became a ``WorkerStateType``
        member, so collapsing it here would be the drift
        ``test_state_type_classes_match_the_enum`` exists to catch.
        """
        assert agent_metrics_module.classify_worker_state_type("config-updated") == (
            "config-updated"
        )

    def test_classify_updated_is_itself_now_that_it_is_a_member(self):
        """⚠️ INVERTED 2026-09-17, the same way ``config-updated`` was on 09-13.

        ``updated`` (3,448 live rows) became a ``WorkerStateType`` member once
        its pair series went non-zero — ``(updated, processing)`` = 8/7d, 2/48h.
        Collapsing it here would be exactly the drift
        ``test_state_type_classes_match_the_enum`` exists to catch.
        """
        assert agent_metrics_module.classify_worker_state_type("updated") == "updated"

    def test_classify_other_real_non_members_are_unknown(self):
        # ⚠️ ``updated`` LEFT this list on 2026-09-17 (it is a member now), so
        # ``prompt-swapped`` is the last real live non-member standing. If it is
        # ever promoted, this test needs a new real exemplar — not deletion.
        for value in ("prompt-swapped", "", None, "<unknown>"):
            assert (
                agent_metrics_module.classify_worker_state_type(value)
                == agent_metrics_module.WORKER_STATE_TYPE_UNKNOWN
            )

    # -- the pair vocabulary, which is WIDER than the enum on purpose --------

    def test_pair_vocabulary_is_the_enum_with_no_candidate_outstanding(self):
        """⭐ The whole point: a census that cannot SPELL a candidate is useless.

        ``WORKER_STATE_TYPE_CLASSES`` collapses a non-member to ``"<unknown>"``,
        so a pair labelled with it cannot be joined against
        ``STATE_TRANSITIONS``. This vocabulary keeps a candidate as itself, and
        adds nothing else.

        ⚠️ ``WORKER_STATE_PAIR_CANDIDATES`` IS EMPTY AS OF 2026-09-17, and an
        empty tuple is the SUCCESS state, not a broken constant: the census has
        graded both values it was built to grade (``config-updated`` promoted
        2026-09-13, ``updated`` 2026-09-17), so the vocabulary is momentarily
        exactly the enum. It goes wider again the next time somebody proposes a
        promotion.

        ⛔ A promoted value must appear EXACTLY ONCE: left in both halves it
        would push ``WORKER_STATE_PAIR_VALUES`` to 12 and break the cardinality
        bound the next test pins.

        ⚠️ WHICH LINE IS DOING THE WORK CHANGED WHEN THE CANDIDATE LIST EMPTIED,
        and saying so is the point of this paragraph. ``WORKER_STATE_PAIR_KNOWN``
        is ``(*_WORKER_STATE_ENUM_VALUES, *WORKER_STATE_PAIR_CANDIDATES)``, so
        while ``..._CANDIDATES`` is empty the set-equality and the no-duplicates
        assertions below are TAUTOLOGIES — ``known`` is derived from the enum and
        cannot disagree with it. They are kept because they stop being
        tautological the instant a candidate is added back, which is exactly when
        the duplicate hazard returns.
        ⭐ THE LIVE ASSERTION TODAY IS ``... == ()``. It is the tripwire that
        fires if somebody adds a candidate without re-reading this test, and it
        is the only line here that can currently fail.
        """
        from src.services.worker_state_manager import WorkerStateType

        known = agent_metrics_module.WORKER_STATE_PAIR_KNOWN
        # ⚠️ Tautological while ..._CANDIDATES is empty; see the docstring. Kept
        # for the day it is not.
        assert set(known) == {m.value for m in WorkerStateType}
        assert len(known) == len(set(known)), f"duplicate label: {known}"
        # ⭐ THE LIVE ONE: the only assertion in this test that can fail today.
        assert agent_metrics_module.WORKER_STATE_PAIR_CANDIDATES == ()
        assert {"config-updated", "updated"} <= {m.value for m in WorkerStateType}
        # ⛔ Still a non-member, and the reason the rule is a rule: it has rows
        # (the 2026-09-09 census) and zero pair traffic, so it earns nothing.
        assert "prompt-swapped" not in known

    def test_pair_label_values_are_eleven_and_closed(self):
        """The cardinality bound, stated as a number: 11 per side, 121 ceiling."""
        values = agent_metrics_module.WORKER_STATE_PAIR_VALUES
        assert len(values) == 11
        assert len(set(values)) == 11
        assert set(values) == set(agent_metrics_module.WORKER_STATE_PAIR_KNOWN) | {"none", "other"}

    def test_classify_pair_side_passes_through_each_known_value(self):
        for value in agent_metrics_module.WORKER_STATE_PAIR_KNOWN:
            assert agent_metrics_module.classify_worker_state_pair_side(value) == value

    def test_classify_pair_side_keeps_config_updated_as_itself(self):
        """⚠️ HYPHEN, and ⛔ it must NOT collapse — it is the value being graded."""
        assert (
            agent_metrics_module.classify_worker_state_pair_side("config-updated")
            == "config-updated"
        )
        assert agent_metrics_module.classify_worker_state_pair_side("updated") == "updated"

    def test_classify_pair_side_maps_unknown_strings_to_other(self):
        """⛔ The cardinality bound: caller-supplied free text never reaches a label."""
        for value in ("prompt-swapped", "<unknown>", "Created", "9f3c-not-a-state", "none"):
            assert agent_metrics_module.classify_worker_state_pair_side(value) == "other", value

    def test_classify_pair_side_maps_a_missing_side_to_none(self):
        for value in (None, ""):
            assert agent_metrics_module.classify_worker_state_pair_side(value) == "none"

    def test_the_literal_string_none_is_other_not_none(self):
        """⚠️ ``none`` asserts A SIDE WAS MISSING; the word is a different finding."""
        assert agent_metrics_module.classify_worker_state_pair_side("none") == "other"

    def test_classify_pair_side_does_not_leak_the_enum_qualified_name(self):
        """⛔ THE ALLOW-LIST CHECK PASSES AND THE LABEL STILL ESCAPES IT.

        ``WorkerStateType`` is a ``(str, Enum)`` MIXIN. A member compares equal
        to its value, so ``state in WORKER_STATE_PAIR_KNOWN`` is True — but
        ``str()`` on it yields the QUALIFIED NAME, ``'WorkerStateType.CREATED'``.
        Shipped that way the bound reads as enforced while every enum-typed call
        site publishes a second, unbounded spelling of the same state: 11 labels
        per side become 18, and the pair ceiling 121 becomes 324.

        ⚠️ Every other test in this class passes a RAW string, which is exactly
        why none of them can fail for this. The mixin is the whole point of the
        case, so it must be driven with a real member, not with ``.value``.
        """
        from src.services.worker_state_manager import WorkerStateType

        for member in WorkerStateType:
            got = agent_metrics_module.classify_worker_state_pair_side(member)
            assert got == member.value, f"{member!r} -> {got!r}"
            assert "." not in got, f"qualified name leaked: {got!r}"
            assert type(got) is str, f"not a plain str: {type(got).__name__}"
            assert got in agent_metrics_module.WORKER_STATE_PAIR_KNOWN

    # -- pre-materialised zeros ---------------------------------------------

    def test_verdict_zeros_are_prematerialised(self):
        """A zero must be a MEASURED zero; an unobserved label has no series at all."""
        client = self._client_with_distinct_counters()
        zeroed = [
            call.args[1]["verdict"]
            for call in client._counters["worker_state_transition_verdict"].add.call_args_list
            if call.args[0] == 0
        ]
        assert sorted(zeroed) == sorted(agent_metrics_module.WORKER_STATE_VERDICTS)

    def test_metering_provenance_zeros_are_prematerialised(self):
        """The INSTRUMENT of the provenance fix must itself be pinned.

        ⛔ Mirrors ``test_verdict_zeros_are_prematerialised`` for the counter
        added to make an unproven billing attribution visible. Without this,
        a mutation battery showed the metric could be DELETED, SILENCED or
        HARDCODED and the whole suite stayed green — the fix would ship
        unmeasurable, which is the same defect it exists to close, one level
        up.

        The zeros matter specifically: ``prometheus_client`` emits HELP/TYPE
        with NO sample lines for a labelled counter nobody incremented, so
        "registered but never fired" and "absent" are indistinguishable at
        the query layer. Pre-materialising every value of the CLOSED set is
        what makes a later zero a MEASURED zero.
        """
        client = self._client_with_distinct_counters()
        zeroed = [
            call.args[1]["provenance"]
            for call in client._counters["metering_event_tenant_provenance"].add.call_args_list
            if call.args[0] == 0
        ]
        assert sorted(zeroed) == sorted(agent_metrics_module.TENANT_PROVENANCES)

    def test_write_zeros_are_prematerialised_for_all_30_combinations(self):
        client = self._client_with_distinct_counters()
        zeroed = {
            (call.args[1]["state_type_class"], call.args[1]["validated"])
            for call in client._counters["worker_state_write"].add.call_args_list
            if call.args[0] == 0
        }
        expected = {
            (state_class, validated)
            for state_class in agent_metrics_module.WORKER_STATE_TYPE_CLASSES
            for validated in agent_metrics_module.WORKER_STATE_WRITE_VALIDATED
        }
        assert zeroed == expected
        # 10 state_type_class labels (9 enum members + <unknown>) x 3 validated
        # values. Was 24, then 27 when ``config-updated`` became a member on
        # 2026-09-13, now 30 with ``updated`` on 2026-09-17.
        assert len(expected) == 30

    def test_unknown_pair_zeros_cover_exactly_the_40_reachable_pairs(self):
        """⛔ 40, not 121: both-sides-enum cannot reach the fail-open arm.

        Seeding the 81 enum x enum pairs would publish measured-looking zeros
        for a question this instrument never asks — they belong to the INVALID
        arm's own counter, and the next test pins them there.

        ⚠️ 40, not the 57 this asserted before 2026-09-17 nor the 72 before
        2026-09-13. ``config-updated`` became the 8th enum member and ``updated``
        the 9th, so 9x9=81 pairs are now unreachable instead of 7x7=49. The drop
        IS the change: those calls no longer arrive here.
        """
        client = self._client_with_distinct_counters()
        zeroed = {
            (call.args[1]["from_state"], call.args[1]["to_state"])
            for call in client._counters["worker_state_transition_unknown_pair"].add.call_args_list
            if call.args[0] == 0
        }
        assert zeroed == set(agent_metrics_module.WORKER_STATE_UNKNOWN_PAIRS)
        assert len(zeroed) == 40
        assert ("created", "processing") not in zeroed, "enum x enum is unreachable here"
        assert ("config-updated", "processing") not in zeroed, (
            "config-updated is an enum member now; this pair resolves as valid"
        )
        assert ("updated", "processing") not in zeroed, (
            "updated became an enum member on 2026-09-17; this pair left the fail-open arm"
        )
        assert ("other", "processing") in zeroed, "a non-member side is still reachable"
        assert ("none", "created") in zeroed

    def test_invalid_pair_zeros_cover_exactly_the_81_enum_pairs(self):
        """⛔ 81, and DISJOINT from the unknown arm's 40 — together exactly 121.

        The two pair counters partition the same 11x11 label space, and the
        partition IS the finding each one reports: a pair on the unknown counter
        means the state machine was BYPASSED (a side it cannot spell), a pair on
        this one means the state machine RAN and said no. Seeding either with the
        other's half would publish measured-looking zeros for a question that
        instrument never asks, and would make the two indistinguishable in
        PromQL.

        ⚠️ This set deliberately includes the ~26 pairs that are LEGAL under
        ``STATE_TRANSITIONS`` and therefore can never move off zero. Narrowing it
        would require importing ``STATE_TRANSITIONS`` here (a cycle:
        ``worker_state_manager`` imports this module) or re-declaring it (a
        second source of truth that drifts silently). A permanent zero next to a
        readable transition table is the cheaper wrong.
        """
        client = self._client_with_distinct_counters()
        zeroed = {
            (call.args[1]["from_state"], call.args[1]["to_state"])
            for call in client._counters["worker_state_transition_invalid_pair"].add.call_args_list
            if call.args[0] == 0
        }
        assert zeroed == set(agent_metrics_module.WORKER_STATE_INVALID_PAIRS)
        assert len(zeroed) == 81
        assert ("created", "completed") in zeroed, "enum x enum is the reachable set here"
        assert ("updated", "processing") in zeroed, "updated is the 9th enum member"
        assert ("other", "processing") not in zeroed, (
            "both sides are enum members on this arm; a sentinel is unreachable"
        )
        assert ("none", "created") not in zeroed
        unknown = set(agent_metrics_module.WORKER_STATE_UNKNOWN_PAIRS)
        assert zeroed & unknown == set(), "the two pair counters must partition, not overlap"
        assert len(zeroed | unknown) == 121, "and together they must be the whole label space"

    def test_record_invalid_pair_fires_once_with_classified_labels(self):
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_invalid_pair"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_invalid_pair("processing", "processing")
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"from_state": "processing", "to_state": "processing"})

    def test_record_invalid_pair_never_labels_the_raw_value(self):
        """⛔ The cardinality bound, on a counter that should never see a raw value.

        Both sides are enum members by construction at the real call site, so
        this can only happen if somebody calls it from the wrong arm — and the
        allow-list has to hold even then, because ``state_type`` is free text.
        """
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_invalid_pair"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_invalid_pair("prompt-swapped", None)
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"from_state": "other", "to_state": "none"})
        assert "prompt-swapped" not in str(new[0].args)

    def test_record_invalid_pair_swallows_instrument_failure(self):
        """Instrumentation must never be able to fail a state transition check."""
        client = self._client_with_distinct_counters()
        client._counters["worker_state_transition_invalid_pair"].add.side_effect = RuntimeError(
            "otel down"
        )
        client.record_worker_state_transition_invalid_pair("created", "completed")

    # -- the three-valued ``validated`` label --------------------------------

    def test_validated_label_set_is_exactly_three_values(self):
        """⛔ Two values cannot be read honestly; see the constant's comment.

        ``true`` is measured coverage, ``false`` is the exposure number, and
        ``by_construction`` is neither — a provably-valid transition nobody
        called the validator for. Collapsing it into either arm corrupts one of
        the two numbers this counter exists to produce.
        """
        assert agent_metrics_module.WORKER_STATE_WRITE_VALIDATED == (
            "true",
            "false",
            "by_construction",
        )

    def test_classify_validated_passes_through_each_member(self):
        for member in agent_metrics_module.WORKER_STATE_WRITE_VALIDATED:
            assert agent_metrics_module.classify_worker_state_validated(member) == member

    def test_classify_validated_falls_back_DOWNWARD_to_the_exposure_value(self):
        """⛔ Direction matters: a typo must never manufacture coverage.

        Anything unrecognised — including a bare bool, the shape this label used
        to be — collapses to ``"false"``, never to ``"true"`` or
        ``"by_construction"``. Over-reporting exposure is recoverable;
        over-reporting coverage is the failure being prevented.
        """
        for value in (None, "", "TRUE", "yes", "by-construction", True, False):
            assert agent_metrics_module.classify_worker_state_validated(value) == "false"

    def test_record_write_labels_by_construction_distinctly(self):
        """⭐ ``by_construction`` must reach the label as itself, not as "true"."""
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_write"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_write(
            "completed", validated=agent_metrics_module.WORKER_STATE_WRITE_VALIDATED_BY_CONSTRUCTION
        )
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"state_type_class": "completed", "validated": "by_construction"})

    # -- recording ----------------------------------------------------------

    def test_record_verdict_increments_once_with_its_label(self):
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_verdict"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_verdict(
            agent_metrics_module.WORKER_STATE_VERDICT_INVALID
        )
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"verdict": "invalid_transition"})

    def test_record_write_labels_known_state_and_validated(self):
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_write"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_write("processing", validated="true")
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"state_type_class": "processing", "validated": "true"})

    def test_record_write_never_labels_the_raw_value(self):
        """⛔ The cardinality bound: a free-text state_type must collapse to the sentinel."""
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_write"]
        before = len(counter.add.call_args_list)
        # ⚠️ ``prompt-swapped``, not ``config-updated``: the latter became an enum
        # member on 2026-09-13 and now labels as itself, which would have made
        # this collapse assertion pass for the wrong reason.
        client.record_worker_state_write("prompt-swapped", validated="false")
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"state_type_class": "<unknown>", "validated": "false"})
        assert "prompt-swapped" not in str(new[0].args)

    def test_record_write_swallows_instrument_failure(self):
        """Instrumentation must never be able to fail a persisted state write."""
        client = self._client_with_distinct_counters()
        client._counters["worker_state_write"].add.side_effect = RuntimeError("otel down")
        client.record_worker_state_write("processing", validated="true")  # must not raise

    def test_record_verdict_swallows_instrument_failure(self):
        client = self._client_with_distinct_counters()
        client._counters["worker_state_transition_verdict"].add.side_effect = RuntimeError("boom")
        client.record_worker_state_transition_verdict("valid")  # must not raise

    def test_record_unknown_pair_labels_both_sides(self):
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_unknown_pair"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_unknown_pair("config-updated", "processing")
        new = counter.add.call_args_list[before:]
        assert len(new) == 1
        assert new[0].args == (1, {"from_state": "config-updated", "to_state": "processing"})

    def test_record_unknown_pair_never_labels_a_raw_value(self):
        """⛔ The cardinality bound, at the recording seam."""
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_unknown_pair"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_unknown_pair("prompt-swapped", "7f2a-whatever")
        new = counter.add.call_args_list[before:]
        assert new[0].args == (1, {"from_state": "other", "to_state": "other"})
        assert "prompt-swapped" not in str(new[0].args)
        assert "7f2a-whatever" not in str(new[0].args)

    def test_record_unknown_pair_labels_a_missing_side_none(self):
        client = self._client_with_distinct_counters()
        counter = client._counters["worker_state_transition_unknown_pair"]
        before = len(counter.add.call_args_list)
        client.record_worker_state_transition_unknown_pair(None, "updated")
        new = counter.add.call_args_list[before:]
        assert new[0].args == (1, {"from_state": "none", "to_state": "updated"})

    def test_record_unknown_pair_swallows_instrument_failure(self):
        client = self._client_with_distinct_counters()
        client._counters["worker_state_transition_unknown_pair"].add.side_effect = RuntimeError(
            "boom"
        )
        client.record_worker_state_transition_unknown_pair("config-updated", "x")  # must not raise

    def test_both_counters_are_noops_when_disabled(self):
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = AgentMetricsClient()
            client.record_worker_state_transition_verdict("valid")
            client.record_worker_state_write("processing", validated="true")
            client.record_worker_state_transition_unknown_pair("config-updated", "processing")
            assert len(client._counters) == 0

    # -- exported names ------------------------------------------------------

    def test_counters_export_under_their_declared_names(self):
        """unit must be dimensionless or the exporter renames the metric."""
        declared = {
            name: unit for _kind, name, unit in TestPrometheusExportedNames._declared_instruments()
        }
        for name in (
            "worker_management_worker_state_transition_verdicts_total",
            "worker_management_worker_state_writes_total",
            "worker_management_worker_state_transition_unknown_pairs_total",
        ):
            assert name in declared
            assert declared[name] in TestPrometheusExportedNames.DIMENSIONLESS
