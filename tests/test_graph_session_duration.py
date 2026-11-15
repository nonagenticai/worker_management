"""Regression: a graph session's duration must not be structurally zero.

`upsert_finalized_session` is called ONCE, at finalize. It used to stamp
`created_at`, `updated_at` and `completed_at` from a single `datetime.now()`, so
every row it INSERTed satisfied `completed_at == created_at` and the session's
duration was always exactly zero -- not "usually small", but arithmetically
incapable of being anything else.

Measured on prod 2026-08-10: `graph_execution_sessions` held 830 rows, of which
830 had `completed_at = created_at`, 0 had `completed_at > created_at`, and 0 had
`completed_at IS NULL`. Any latency question asked of this table returned zero.

The caller knows the real start time; these tests pin that it reaches the row.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.graph_execution_db import (
    ExecutionOutputs,
    GraphExecutionSessionDB,
    ScopeContext,
)

# Column order in the INSERT's trailing timestamp triple.
CREATED_AT = -3
UPDATED_AT = -2
COMPLETED_AT = -1


@pytest.fixture
def captured_params():
    """Patch the connection so the INSERT's bound parameters can be inspected."""
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"session_id": "s1"})

    class _Ctx:
        async def __aenter__(self):
            return conn

        async def __aexit__(self, *_a):
            return False

    with patch("src.db.graph_execution_db.get_pg_connection", lambda: _Ctx()):
        yield conn


def _scope() -> ScopeContext:
    return ScopeContext(tenant_id="t1", subject_id="s1", session_id="sess-1", visibility="tenant")


def _outputs() -> ExecutionOutputs:
    return ExecutionOutputs(
        node_outputs={}, completed_nodes=[], failed_nodes=[], final_response=None
    )


@pytest.mark.asyncio
async def test_started_at_is_persisted_as_created_at(captured_params):
    """The caller's real start time must land in created_at, not the finalize time."""
    started = datetime.now(UTC) - timedelta(seconds=42)

    await GraphExecutionSessionDB().upsert_finalized_session(
        _scope(), "g1", "completed", outputs=_outputs(), started_at=started
    )

    params = captured_params.fetchrow.call_args[0]
    created_at, completed_at = params[CREATED_AT], params[COMPLETED_AT]

    # The row must carry the caller's start, to the second.
    assert created_at.replace(tzinfo=None) == started.replace(tzinfo=None)
    # And the duration it implies must be real, not zero.
    assert completed_at > created_at
    assert (completed_at - created_at).total_seconds() >= 41


@pytest.mark.asyncio
async def test_duration_is_not_structurally_zero(captured_params):
    """The defect stated directly: a terminal row must not have equal timestamps."""
    started = datetime.now(UTC) - timedelta(seconds=5)

    await GraphExecutionSessionDB().upsert_finalized_session(
        _scope(), "g1", "failed", outputs=_outputs(), started_at=started
    )

    params = captured_params.fetchrow.call_args[0]
    assert params[COMPLETED_AT] != params[CREATED_AT], (
        "completed_at == created_at means the duration is structurally zero -- "
        "this is the exact prod condition (830/830 rows) this fix removes"
    )


@pytest.mark.asyncio
async def test_omitting_started_at_preserves_previous_behaviour(captured_params):
    """Back-compat: with no start supplied, created_at falls back to the finalize time.

    This keeps the parameter additive for any caller that does not know its start,
    rather than making the timestamp None and pushing a NOT NULL violation into a
    path whose whole contract is "never raises".
    """
    await GraphExecutionSessionDB().upsert_finalized_session(
        _scope(), "g1", "completed", outputs=_outputs()
    )

    params = captured_params.fetchrow.call_args[0]
    assert params[CREATED_AT] is not None
    assert params[CREATED_AT] == params[UPDATED_AT]


@pytest.mark.asyncio
async def test_non_terminal_status_leaves_completed_at_null(captured_params):
    """Negative control: a still-running session has no completion time at all.

    Guards against a "fix" that stamps completed_at unconditionally, which would
    make every in-flight session look finished.
    """
    await GraphExecutionSessionDB().upsert_finalized_session(
        _scope(),
        "g1",
        "running",
        outputs=_outputs(),
        started_at=datetime.now(UTC) - timedelta(seconds=5),
    )

    params = captured_params.fetchrow.call_args[0]
    assert params[COMPLETED_AT] is None
