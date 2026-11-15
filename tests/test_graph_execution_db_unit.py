"""Unit tests for db/graph_execution_db.py with InMemoryDB.

Verifies real data flow: create sessions, add logs/events,
then read them back and check correctness.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.db.graph_execution_db import (
    ExecutionOutputs,
    GraphExecutionSessionDB,
    InferenceTelemetry,
    NodeExecutionRecord,
    ScopeContext,
)
from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_and_get_roundtrip(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            created = await db.create_session(
                tenant_id="c1",
                subject_id="c1",
                session_id="s-1",
                graph_id="g-1",
                execution_mode="automatic",
                status="initialized",
                variables={"x": 1},
            )

            fetched = await db.get_session(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert created["session_id"] == "s-1"
        assert created["graph_id"] == "g-1"
        assert created["tenant_id"] == "c1"
        assert created["execution_mode"] == "automatic"
        assert created["status"] == "initialized"

        assert fetched is not None
        assert fetched["session_id"] == "s-1"
        assert fetched["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            created = await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-2", graph_id="g-1"
            )

        assert created["execution_mode"] == "automatic"
        assert created["status"] == "initialized"


class TestGetSession:
    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            result = await db.get_session(tenant_id="c1", subject_id="c1", session_id="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_wrong_tenant_returns_none(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            result = await db.get_session(tenant_id="c2", subject_id="c2", session_id="s-1")

        assert result is None


# ---------------------------------------------------------------------------
# Update session status
# ---------------------------------------------------------------------------


class TestUpdateSessionStatus:
    @pytest.mark.asyncio
    async def test_update_to_running(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"), status="running"
            )

        assert updated is not None
        assert updated["status"] == "running"

    @pytest.mark.asyncio
    async def test_update_to_completed_sets_completed_at(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"), status="completed"
            )

        assert updated is not None
        assert updated["status"] == "completed"
        assert updated.get("completed_at") is not None

    @pytest.mark.asyncio
    async def test_update_with_optional_fields(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                outputs=ExecutionOutputs(
                    node_outputs={"n1": "out"},
                    completed_nodes=["n1"],
                    failed_nodes=[],
                    node_errors={},
                    worker_assignments={"n1": "w1"},
                ),
                error_message="some error",
            )

        assert updated is not None
        assert updated["node_outputs"] == {"n1": "out"}
        assert updated["completed_nodes"] == ["n1"]
        assert updated["error_message"] == "some error"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            result = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="nope"), status="failed"
            )

        assert result is None


# ---------------------------------------------------------------------------
# Inference telemetry columns (migration 009 / Phase 1 inference-graph)
# ---------------------------------------------------------------------------


class TestInferenceTelemetryColumns:
    """Round-trip write+read for the 7 inference telemetry columns added in
    migration 009. The columns default to NULL at session creation and are
    populated at session finalize via ``update_session_status``.
    """

    @pytest.mark.asyncio
    async def test_write_all_seven_columns_roundtrip(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        request_payload = {
            "model": "local-llm",
            "messages": [{"role": "user", "content": "hi"}],
        }
        final_response = {
            "id": "resp-42",
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}},
            ],
        }

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                outputs=ExecutionOutputs(final_response=final_response),
                telemetry=InferenceTelemetry(
                    chosen_provider="vllm",
                    fallback_count=1,
                    prompt_tokens=42,
                    completion_tokens=17,
                    validation_passed=True,
                ),
                request_payload=request_payload,
            )
            fetched = await db.get_session(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert updated is not None
        assert updated["chosen_provider"] == "vllm"
        assert updated["fallback_count"] == 1
        assert updated["prompt_tokens"] == 42
        assert updated["completion_tokens"] == 17
        assert updated["validation_passed"] is True
        assert updated["request_payload"] == request_payload
        assert updated["final_response"] == final_response

        assert fetched is not None
        assert fetched["chosen_provider"] == "vllm"
        assert fetched["fallback_count"] == 1
        assert fetched["prompt_tokens"] == 42
        assert fetched["completion_tokens"] == 17
        assert fetched["validation_passed"] is True
        assert fetched["request_payload"] == request_payload
        assert fetched["final_response"] == final_response

    @pytest.mark.asyncio
    async def test_partial_update_leaves_others_null(self):
        """Passing only ``chosen_provider`` must not touch the other 6 columns."""
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="running",
                telemetry=InferenceTelemetry(chosen_provider="Qwen3-4B-AWQ"),
            )

        assert updated is not None
        assert updated["chosen_provider"] == "Qwen3-4B-AWQ"
        # All other telemetry fields remain at their NULL default.
        assert updated.get("fallback_count") is None
        assert updated.get("prompt_tokens") is None
        assert updated.get("completion_tokens") is None
        assert updated.get("validation_passed") is None
        assert updated.get("request_payload") is None
        assert updated.get("final_response") is None

    @pytest.mark.asyncio
    async def test_jsonb_nested_roundtrip(self):
        """Nested dicts in ``request_payload`` and ``final_response`` must
        round-trip with exact dict equality, preserving shape and types.
        """
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        nested_payload = {
            "model": "local-llm",
            "metadata": {
                "tenant": "n8n-dev",
                "trace": {"id": "t-1", "sampled": True, "depth": 3},
            },
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.7,
        }
        nested_response = {
            "id": "resp-nested",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                outputs=ExecutionOutputs(final_response=nested_response),
                request_payload=nested_payload,
            )
            fetched = await db.get_session(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert fetched is not None
        assert fetched["request_payload"] == nested_payload
        assert fetched["final_response"] == nested_response
        # Protect against identity-vs-equality drift.
        assert fetched["request_payload"]["metadata"]["trace"]["depth"] == 3
        assert fetched["final_response"]["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_validation_passed_false_is_written(self):
        """``False`` is non-None and must be written (distinct from NULL)."""
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                telemetry=InferenceTelemetry(validation_passed=False),
            )

        assert updated is not None
        assert updated["validation_passed"] is False

    @pytest.mark.asyncio
    async def test_fallback_count_zero_is_written(self):
        """``0`` is non-None and must be written (distinct from NULL)."""
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            updated = await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                telemetry=InferenceTelemetry(fallback_count=0),
            )

        assert updated is not None
        assert updated["fallback_count"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_surfaces_inference_columns(self):
        """``list_sessions`` uses ``SELECT *`` so the new columns must appear."""
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                telemetry=InferenceTelemetry(
                    chosen_provider="claude",
                    fallback_count=2,
                    validation_passed=True,
                ),
            )
            rows = await db.list_sessions(tenant_id="c1", subject_id="c1")

        assert len(rows) == 1
        assert rows[0]["chosen_provider"] == "claude"
        assert rows[0]["fallback_count"] == 2
        assert rows[0]["validation_passed"] is True


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_existing(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            result = await db.delete_session(tenant_id="c1", subject_id="c1", session_id="s-1")
            after = await db.get_session(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert result is True
        assert after is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            result = await db.delete_session(tenant_id="c1", subject_id="c1", session_id="nope")

        assert result is False


# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_all_for_creator(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-2", graph_id="g-1"
            )
            await db.create_session(
                tenant_id="c2", subject_id="c2", session_id="s-3", graph_id="g-2"
            )

            result = await db.list_sessions(tenant_id="c1", subject_id="c1")

        assert len(result) == 2
        session_ids = {r["session_id"] for r in result}
        assert session_ids == {"s-1", "s-2"}

    @pytest.mark.asyncio
    async def test_list_with_graph_filter(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-2", graph_id="g-2"
            )

            result = await db.list_sessions(tenant_id="c1", subject_id="c1", graph_id="g-1")

        assert len(result) == 1
        assert result[0]["graph_id"] == "g-1"

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-2", graph_id="g-1"
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"), status="completed"
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-2"), status="failed"
            )

            result = await db.list_sessions(tenant_id="c1", subject_id="c1", status="completed")

        assert len(result) == 1
        assert result[0]["session_id"] == "s-1"
        assert result[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_with_validation_passed_filter(self):
        """Eval-set sampler relies on this filter to pull only validated traces."""
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-1", graph_id="g-1"
            )
            await db.create_session(
                tenant_id="c1", subject_id="c1", session_id="s-2", graph_id="g-1"
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                status="completed",
                telemetry=InferenceTelemetry(validation_passed=True),
            )
            await db.update_session_status(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-2"),
                status="completed",
                telemetry=InferenceTelemetry(validation_passed=False),
            )

            result = await db.list_sessions(tenant_id="c1", subject_id="c1", validation_passed=True)

        assert len(result) == 1
        assert result[0]["session_id"] == "s-1"
        assert result[0]["validation_passed"] is True


# ---------------------------------------------------------------------------
# Node execution logs
# ---------------------------------------------------------------------------


class TestNodeExecutionLogs:
    @pytest.mark.asyncio
    async def test_add_and_get_logs(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            log = await db.add_node_execution_log(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                NodeExecutionRecord(
                    node_id="n-1",
                    worker_name="w1",
                    task_template="do {{thing}}",
                    resolved_task="do work",
                    output="result here",
                ),
            )

            logs = await db.get_session_logs(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert log["node_id"] == "n-1"
        assert log["worker_name"] == "w1"
        assert log["output"] == "result here"
        assert len(logs) == 1
        assert logs[0]["node_id"] == "n-1"

    @pytest.mark.asyncio
    async def test_add_log_with_tz_aware_times(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            log = await db.add_node_execution_log(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                NodeExecutionRecord(
                    node_id="n-1",
                    worker_name="w1",
                    task_template="tmpl",
                    resolved_task="resolved",
                    start_time=datetime(2026, 1, 1, tzinfo=UTC),
                    end_time=datetime(2026, 1, 2, tzinfo=UTC),
                ),
            )

        # Tz-aware times should have tzinfo stripped
        assert log["start_time"].tzinfo is None
        assert log["end_time"].tzinfo is None

    @pytest.mark.asyncio
    async def test_multiple_logs_for_session(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.add_node_execution_log(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                NodeExecutionRecord(
                    node_id="n-1",
                    worker_name="w1",
                    task_template="t1",
                    resolved_task="r1",
                ),
            )
            await db.add_node_execution_log(
                ScopeContext(tenant_id="c1", subject_id="c1", session_id="s-1"),
                NodeExecutionRecord(
                    node_id="n-2",
                    worker_name="w2",
                    task_template="t2",
                    resolved_task="r2",
                ),
            )

            logs = await db.get_session_logs(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert len(logs) == 2
        node_ids = {entry["node_id"] for entry in logs}
        assert node_ids == {"n-1", "n-2"}


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------


class TestExecutionEvents:
    @pytest.mark.asyncio
    async def test_add_and_get_events(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            event = await db.add_execution_event(
                tenant_id="c1",
                subject_id="c1",
                session_id="s-1",
                event_type="node_started",
                node_id="n-1",
                event_data={"progress": 0},
            )

            events = await db.get_session_events(tenant_id="c1", subject_id="c1", session_id="s-1")

        assert event["event_type"] == "node_started"
        assert event["node_id"] == "n-1"
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_get_events_with_type_filter(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            await db.add_execution_event(
                tenant_id="c1",
                subject_id="c1",
                session_id="s-1",
                event_type="node_started",
                node_id="n-1",
            )
            await db.add_execution_event(
                tenant_id="c1",
                subject_id="c1",
                session_id="s-1",
                event_type="node_completed",
                node_id="n-1",
            )
            await db.add_execution_event(
                tenant_id="c1",
                subject_id="c1",
                session_id="s-1",
                event_type="node_started",
                node_id="n-2",
            )

            started = await db.get_session_events(
                tenant_id="c1", subject_id="c1", session_id="s-1", event_type="node_started"
            )
            completed = await db.get_session_events(
                tenant_id="c1", subject_id="c1", session_id="s-1", event_type="node_completed"
            )

        assert len(started) == 2
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_events_empty_when_no_session(self):
        mem = InMemoryDB()
        db = GraphExecutionSessionDB()

        with patch_get_pg_connection("src.db.graph_execution_db", mem):
            events = await db.get_session_events(
                tenant_id="c1", subject_id="c1", session_id="nonexistent"
            )

        assert events == []
