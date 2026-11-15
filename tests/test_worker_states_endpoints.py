"""Tests for the legacy /workers_states endpoints (src/api/worker_states.py).

Covers GET (with pagination) and POST (batch save) plus Pydantic model validation.
Uses correct mock targets and meaningful assertions.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.worker_states import (
    BulkWorkerStatesRequest,
    SaveWorkerStateRequest,
    WorkerStateBase,
    WorkerStateType,
)
from src.auth import get_identity_proven, get_tenant_id

TENANT = "test-tenant-ws"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with auth bypass and DB pool mocked out."""
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    # ENABLE_AUTH=false means no middleware runs at all, so
    # `request.state.identity_proven` is never set and `get_identity_proven`
    # falls back to its fail-closed default of False (see
    # src/auth/dependencies.py). This fixture is not exercising the tokenless
    # hole POST /workers_states now refuses (see
    # tests/test_worker_states_write_requires_proven_identity.py for that) —
    # it is simulating an ordinary authenticated caller, so state that
    # explicitly rather than let every existing POST test collide with the
    # default meant for a different, unproven population.
    app.dependency_overrides[get_identity_proven] = lambda: True

    @asynccontextmanager
    async def _fake_lifespan(a):
        yield

    original_router = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_router


def _mock_state_db(
    *,
    get_worker_states_return=None,
    add_worker_state_batch_return=None,
    side_effect=None,
):
    """Patch StateDatabase in worker_states module; return the mock instance."""
    mock_db = AsyncMock()
    if side_effect:
        mock_db.get_worker_states = AsyncMock(side_effect=side_effect)
        mock_db.add_worker_state_batch = AsyncMock(side_effect=side_effect)
    else:
        rows = get_worker_states_return if get_worker_states_return is not None else []

        async def _paged(*, tenant_id, subject_id, limit=100, offset=0):
            """Stand in for the DB: pagination happens in the QUERY, not the caller."""
            return rows[offset : offset + limit]

        mock_db.get_worker_states = AsyncMock(side_effect=_paged)
        mock_db.add_worker_state_batch = AsyncMock(
            return_value=add_worker_state_batch_return
            if add_worker_state_batch_return is not None
            else []
        )
    mock_cls = patch("src.api.worker_states.StateDatabase", return_value=mock_db)
    return mock_cls, mock_db


def _valid_state(
    *,
    worker_id="w1",
    worker_name="worker_one",
    state_type=cast("WorkerStateType", "processing"),  # noqa: B008
    worker_state=cast("Any", None),  # noqa: B008
):
    """Helper to build a valid state payload."""
    return {
        "worker_id": worker_id,
        "worker_name": worker_name,
        "state_type": state_type,
        "worker_state": worker_state or {"status": "ok"},
    }


SAMPLE_STATES = [
    {
        "id": 1,
        "worker_id": "w1",
        "worker_name": "alpha",
        "is_agent": True,
        "state_type": "completed",
        "timestamp": "2025-01-15T10:00:00",
        "metadata": {"task": "t1"},
        "worker_state": {"status": "completed", "result": 42},
    },
    {
        "id": 2,
        "worker_id": "w2",
        "worker_name": "beta",
        "is_agent": False,
        "state_type": "processing",
        "timestamp": "2025-01-15T11:00:00",
        "metadata": None,
        "worker_state": {"status": "processing"},
    },
    {
        "id": 3,
        "worker_id": "w1",
        "worker_name": "alpha",
        "is_agent": True,
        "state_type": "error",
        "timestamp": "2025-01-15T12:00:00",
        "metadata": None,
        "worker_state": {"status": "error", "error": "timeout"},
    },
]


# ===========================================================================
# GET /workers_states
# ===========================================================================


class TestGetWorkersStates:
    """GET /workers_states — retrieve worker states with pagination."""

    def test_empty_db(self, client):
        mock_cls, _ = _mock_state_db(get_worker_states_return=[])
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_data_with_correct_fields(self, client):
        mock_cls, mock_db = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        # Verify first record has all expected fields
        rec = data[0]
        assert rec["id"] == 1
        assert rec["worker_id"] == "w1"
        assert rec["worker_name"] == "alpha"
        assert rec["is_agent"] is True
        assert rec["state_type"] == "completed"
        assert rec["worker_state"]["result"] == 42
        assert rec["metadata"]["task"] == "t1"
        # Verify DB was called with correct tenant
        # Pagination is pushed INTO the query -- the endpoint must never read
        # the whole history and slice it in Python.
        mock_db.get_worker_states.assert_awaited_once_with(
            tenant_id=TENANT, subject_id="", limit=100, offset=0
        )

    def test_pagination_limit_1_offset_0(self, client):
        mock_cls, _ = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states?limit=1&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == 1

    def test_pagination_limit_1_offset_1(self, client):
        mock_cls, _ = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states?limit=1&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == 2

    def test_pagination_offset_beyond_data(self, client):
        mock_cls, _ = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states?limit=10&offset=100")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_default_pagination(self, client):
        """Default is limit=100, offset=0 — returns everything if < 100."""
        mock_cls, _ = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_no_args_caller_still_gets_valid_response(self, client):
        """A pre-existing caller passing nothing keeps working (first page)."""
        mock_cls, mock_db = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        assert resp.json()[0]["worker_id"] == "w1"
        kwargs = mock_db.get_worker_states.await_args.kwargs
        assert kwargs["limit"] == 100
        assert kwargs["offset"] == 0

    def test_limit_and_offset_are_forwarded_to_the_query(self, client):
        mock_cls, mock_db = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states?limit=2&offset=1")
        assert resp.status_code == 200
        kwargs = mock_db.get_worker_states.await_args.kwargs
        assert kwargs["limit"] == 2
        assert kwargs["offset"] == 1
        assert [r["id"] for r in resp.json()] == [2, 3]

    def test_limit_0_rejected(self, client):
        resp = client.get("/workers_states?limit=0")
        assert resp.status_code == 422

    def test_limit_1001_rejected(self, client):
        resp = client.get("/workers_states?limit=1001")
        assert resp.status_code == 422

    def test_negative_offset_rejected(self, client):
        resp = client.get("/workers_states?offset=-1")
        assert resp.status_code == 422

    def test_os_error_returns_500(self, client):
        mock_cls, _ = _mock_state_db(side_effect=OSError("disk full"))
        with mock_cls:
            resp = client.get("/workers_states")
        # create_error_response wraps body in {"error": {...}}
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert err["code"] == "FILE_SYSTEM_ERROR"
        assert "disk full" in err["message"]

    def test_runtime_error_returns_500(self, client):
        mock_cls, _ = _mock_state_db(side_effect=RuntimeError("bad state"))
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert err["code"] == "RUNTIME_ERROR"
        assert "bad state" in err["message"]

    def test_generic_exception_returns_500(self, client):
        mock_cls, _ = _mock_state_db(side_effect=ValueError("something weird"))
        with mock_cls:
            resp = client.get("/workers_states")
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert err["code"] == "INTERNAL_ERROR"

    def test_pagination_window(self, client):
        """limit=2, offset=1 returns items 1..2 (0-indexed)."""
        mock_cls, _ = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = client.get("/workers_states?limit=2&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == 2
        assert data[1]["id"] == 3


# ===========================================================================
# POST /workers_states
# ===========================================================================


class TestSaveWorkersStates:
    """POST /workers_states — batch save worker states."""

    def test_single_valid_state(self, client):
        mock_cls, mock_db = _mock_state_db(add_worker_state_batch_return=[101])
        with mock_cls:
            resp = client.post(
                "/workers_states",
                json={"worker_states": [_valid_state()]},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["message"] == "Workers' states saved successfully"
        assert len(body["results"]) == 1
        assert body["results"][0]["id"] == 101
        assert body["results"][0]["worker_id"] == "w1"
        # Verify DB was called with correct args
        mock_db.add_worker_state_batch.assert_awaited_once()
        call_kwargs = mock_db.add_worker_state_batch.call_args.kwargs
        assert call_kwargs["tenant_id"] == TENANT
        states_data = call_kwargs["states"]
        assert len(states_data) == 1
        assert states_data[0]["worker_id"] == "w1"
        assert states_data[0]["worker_name"] == "worker_one"
        assert states_data[0]["is_agent"] is False  # default
        assert states_data[0]["metadata"] is None  # default

    def test_batch_of_three(self, client):
        mock_cls, _mock_db = _mock_state_db(add_worker_state_batch_return=[1, 2, 3])
        states = [
            _valid_state(worker_id="a", worker_name="a1"),
            _valid_state(worker_id="b", worker_name="b1", state_type="completed"),
            _valid_state(worker_id="c", worker_name="c1", state_type="error"),
        ]
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": states})
        assert resp.status_code == 201
        body = resp.json()
        assert len(body["results"]) == 3
        assert [r["worker_id"] for r in body["results"]] == ["a", "b", "c"]

    def test_with_iso_timestamp(self, client):
        mock_cls, _ = _mock_state_db(add_worker_state_batch_return=[1])
        state = _valid_state()
        state["timestamp"] = "2025-03-18T12:00:00Z"
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 201

    def test_missing_worker_id(self, client):
        resp = client.post(
            "/workers_states",
            json={
                "worker_states": [{"worker_name": "x", "state_type": "idle", "worker_state": {}}]
            },
        )
        assert resp.status_code == 422

    def test_missing_worker_name(self, client):
        resp = client.post(
            "/workers_states",
            json={"worker_states": [{"worker_id": "x", "state_type": "idle", "worker_state": {}}]},
        )
        assert resp.status_code == 422

    def test_missing_worker_state(self, client):
        resp = client.post(
            "/workers_states",
            json={"worker_states": [{"worker_id": "x", "worker_name": "x", "state_type": "idle"}]},
        )
        assert resp.status_code == 422

    def test_invalid_state_type(self, client):
        state = _valid_state()
        state["state_type"] = "INVALID_TYPE"
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_empty_worker_states_list(self, client):
        resp = client.post("/workers_states", json={"worker_states": []})
        assert resp.status_code == 422

    def test_more_than_100_items(self, client):
        states = [_valid_state(worker_id=f"w{i}", worker_name=f"n{i}") for i in range(101)]
        resp = client.post("/workers_states", json={"worker_states": states})
        assert resp.status_code == 422

    def test_null_bytes_in_worker_id(self, client):
        state = _valid_state(worker_id="bad\x00id")
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_null_bytes_in_worker_name(self, client):
        state = _valid_state(worker_name="bad\x00name")
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_worker_state_too_large(self, client):
        huge = {"data": "x" * 60000}
        state = _valid_state(worker_state=huge)
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_null_bytes_in_worker_state_key(self, client):
        state = _valid_state(worker_state={"bad\x00key": "val"})
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_db_os_error(self, client):
        mock_cls, _ = _mock_state_db(side_effect=OSError("write failed"))
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": [_valid_state()]})
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "FILE_SYSTEM_ERROR"

    def test_db_runtime_error(self, client):
        mock_cls, _ = _mock_state_db(side_effect=RuntimeError("pool closed"))
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": [_valid_state()]})
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "RUNTIME_ERROR"

    def test_db_generic_exception(self, client):
        mock_cls, _ = _mock_state_db(side_effect=Exception("boom"))
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": [_valid_state()]})
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "INTERNAL_ERROR"

    def test_extra_fields_ignored(self, client):
        """extra='ignore' means unknown fields don't cause 422."""
        mock_cls, _ = _mock_state_db(add_worker_state_batch_return=[1])
        state = _valid_state()
        state["unknown_field"] = "should be ignored"
        with mock_cls:
            resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 201

    def test_all_valid_state_types(self, client):
        """Each WorkerStateType enum value is accepted."""
        for st in [
            "created",
            "processing",
            "completed",
            "error",
            "idle",
            "active",
            "default",
            "config-updated",
            "updated",
        ]:
            mock_cls, _ = _mock_state_db(add_worker_state_batch_return=[1])
            state = _valid_state(state_type=st)
            with mock_cls:
                resp = client.post("/workers_states", json={"worker_states": [state]})
            assert resp.status_code == 201, f"state_type={st} should be accepted"

    def test_whitespace_only_worker_id_rejected(self, client):
        state = _valid_state(worker_id="   ")
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422

    def test_whitespace_only_worker_name_rejected(self, client):
        state = _valid_state(worker_name="   ")
        resp = client.post("/workers_states", json={"worker_states": [state]})
        assert resp.status_code == 422


# ===========================================================================
# POST /workers_states requires a proven identity (GET stays tokenless)
# ===========================================================================


@pytest.fixture
def unproven_client():
    """Same stack as `client`, but leaves `get_identity_proven` at its real
    fail-closed default (False) — the tokenless trusted-network shape
    (header omitted -> DEFAULT_TENANT_ID, or header asserted and checked on
    spelling alone). Both stamp `identity_proven=False` in production; this
    fixture reproduces that by simply not overriding the dependency.
    """
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(a):
        yield

    original_router = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_router


class TestSaveWorkersStatesRequiresProvenIdentity:
    def test_unproven_caller_gets_401_and_nothing_is_persisted(self, unproven_client):
        mock_cls, mock_db = _mock_state_db(add_worker_state_batch_return=[999])
        with mock_cls:
            resp = unproven_client.post(
                "/workers_states",
                json={"worker_states": [_valid_state()]},
            )
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"
        mock_db.add_worker_state_batch.assert_not_awaited()

    def test_proven_caller_is_unaffected(self, client):
        """Control: the gate must not touch the path that already worked."""
        mock_cls, mock_db = _mock_state_db(add_worker_state_batch_return=[101])
        with mock_cls:
            resp = client.post(
                "/workers_states",
                json={"worker_states": [_valid_state()]},
            )
        assert resp.status_code == 201, resp.text
        mock_db.add_worker_state_batch.assert_awaited_once()

    def test_unproven_caller_can_still_read(self, unproven_client):
        """GET stays tokenless — only the write is gated."""
        mock_cls, _mock_db = _mock_state_db(get_worker_states_return=SAMPLE_STATES)
        with mock_cls:
            resp = unproven_client.get("/workers_states")
        assert resp.status_code == 200, resp.text


# ===========================================================================
# Pydantic model unit tests (no HTTP)
# ===========================================================================


class TestWorkerStateTypeEnum:
    def test_all_values(self):
        expected = {
            "created",
            "processing",
            "completed",
            "error",
            "idle",
            "active",
            "default",
            # ⛔ STRICT enum — an unlisted value is a 422. Every member here was
            # added on MEASURED pair traffic and nothing else: "config-updated"
            # on 2026-09-13 (4,615 of 4,616 unknown verdicts / 24h) and
            # "updated" on 2026-09-17 ((updated, processing) = 8/7d, 2/48h —
            # small, but non-zero on both windows).
            "config-updated",
            "updated",
        }
        actual = {e.value for e in WorkerStateType}
        assert actual == expected

    def test_string_enum(self):
        assert WorkerStateType.CREATED == "created"
        assert isinstance(WorkerStateType.COMPLETED, str)


class TestWorkerStateBaseValidation:
    def test_valid_construction(self):
        m = WorkerStateBase(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "processing"),
            worker_state={"status": "ok"},
        )
        assert m.worker_id == "w1"
        assert m.state_type == WorkerStateType.PROCESSING

    def test_empty_worker_id_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
            )

    def test_null_bytes_in_worker_id_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="bad\x00",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
            )

    def test_invalid_state_type_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "bogus"),
                worker_state={"a": 1},
            )

    def test_none_state_type_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", None),
                worker_state={"a": 1},
            )

    def test_metadata_too_large(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
                metadata={"big": "x" * 60000},
            )

    def test_metadata_non_dict_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
                metadata=cast("Any", "not-a-dict"),
            )

    def test_worker_state_null_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state=cast("Any", None),
            )

    def test_worker_state_too_large(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"big": "y" * 60000},
            )

    def test_null_bytes_in_worker_state_key(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"bad\x00key": "val"},
            )

    def test_null_bytes_in_worker_state_value(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"key": "bad\x00val"},
            )

    def test_control_chars_in_worker_id_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="bad\x01id",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
            )

    def test_worker_id_over_100_chars_rejected(self):
        with pytest.raises(Exception):
            WorkerStateBase(
                worker_id="x" * 101,
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
            )

    def test_extra_fields_ignored(self):
        """extra='ignore' allows unknown fields."""
        m = WorkerStateBase(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
            **cast("dict[str, Any]", {"unknown_extra": "ignored"}),
        )
        assert m.worker_id == "w1"

    def test_is_agent_defaults_false(self):
        m = WorkerStateBase(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
        )
        assert m.is_agent is False


class TestSaveWorkerStateRequestValidation:
    def test_valid_construction(self):
        m = SaveWorkerStateRequest(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "processing"),
            worker_state={"status": "ok"},
        )
        assert m.worker_id == "w1"

    def test_iso_timestamp(self):
        m = SaveWorkerStateRequest(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
            timestamp=cast("Any", "2025-03-18T12:00:00Z"),
        )
        assert m.timestamp is not None
        assert m.timestamp.year == 2025

    def test_unix_timestamp(self):
        m = SaveWorkerStateRequest(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
            timestamp=cast("Any", 1710763200),  # 2024-03-18T12:00:00Z
        )
        assert m.timestamp is not None
        assert m.timestamp.year == 2024

    def test_invalid_timestamp_format(self):
        with pytest.raises(Exception):
            SaveWorkerStateRequest(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
                timestamp=cast("Any", "not-a-date"),
            )

    def test_timestamp_year_too_old(self):
        with pytest.raises(Exception):
            SaveWorkerStateRequest(
                worker_id="w1",
                worker_name="test",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
                timestamp=cast("Any", "1700-01-01T00:00:00Z"),
            )

    def test_none_timestamp_ok(self):
        m = SaveWorkerStateRequest(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
            timestamp=None,
        )
        assert m.timestamp is None


class TestBulkWorkerStatesRequestValidation:
    def test_valid_single(self):
        state = SaveWorkerStateRequest(
            worker_id="w1",
            worker_name="test",
            state_type=cast("WorkerStateType", "idle"),
            worker_state={"a": 1},
        )
        req = BulkWorkerStatesRequest(worker_states=[state])
        assert len(req.worker_states) == 1

    def test_empty_list_rejected(self):
        with pytest.raises(Exception):
            BulkWorkerStatesRequest(worker_states=[])

    def test_over_100_rejected(self):
        states = [
            SaveWorkerStateRequest(
                worker_id=f"w{i}",
                worker_name=f"n{i}",
                state_type=cast("WorkerStateType", "idle"),
                worker_state={"a": 1},
            )
            for i in range(101)
        ]
        with pytest.raises(Exception):
            BulkWorkerStatesRequest(worker_states=states)


# ===========================================================================
# Real data flow test with InMemoryDB
# ===========================================================================


class TestRealDataFlow:
    """Use InMemoryDB to verify data actually round-trips through the DB layer."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_worker_states(self):
        """Insert states via DB layer, then read them back."""
        from src.db.workers_state_db import StateDatabase
        from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.workers_state_db", mem):
            db = StateDatabase()

            # Insert a state
            state_id = await db.add_worker_state(
                tenant_id=TENANT,
                subject_id=TENANT,
                worker_id="w-real-1",
                worker_name="real_worker",
                is_agent=True,
                state_type=cast("WorkerStateType", "processing"),
                worker_state={"status": "processing", "progress": 50},
                metadata={"task": "test-task"},
            )
            assert state_id is not None

            # Insert another
            state_id_2 = await db.add_worker_state(
                tenant_id=TENANT,
                subject_id=TENANT,
                worker_id="w-real-1",
                worker_name="real_worker",
                is_agent=True,
                state_type="completed",
                worker_state={"status": "completed", "result": "done"},
                metadata=None,
            )
            assert state_id_2 is not None

            # Read back all states for tenant
            states = await db.get_worker_states(tenant_id=TENANT, subject_id=TENANT)
            assert len(states) >= 2

            # Verify the data we inserted is there
            worker_states = [s for s in states if s.get("worker_id") == "w-real-1"]
            assert len(worker_states) == 2

    @pytest.mark.asyncio
    async def test_batch_insert_and_read_back(self):
        """Batch insert via add_worker_state_batch, then read back."""
        from src.db.workers_state_db import StateDatabase
        from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.workers_state_db", mem):
            db = StateDatabase()
            batch = [
                {
                    "worker_id": f"bw{i}",
                    "worker_name": f"batch_worker_{i}",
                    "is_agent": False,
                    "state_type": "idle",
                    "worker_state": {"index": i},
                    "metadata": None,
                }
                for i in range(5)
            ]
            ids = await db.add_worker_state_batch(tenant_id=TENANT, subject_id=TENANT, states=batch)
            assert len(ids) == 5

            # Read back
            all_states = await db.get_worker_states(tenant_id=TENANT, subject_id=TENANT)
            assert len(all_states) == 5

    @pytest.mark.asyncio
    async def test_latest_state(self):
        """get_latest_worker_state returns the most recent state."""
        from src.db.workers_state_db import StateDatabase
        from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

        mem = InMemoryDB()
        with patch_get_pg_connection("src.db.workers_state_db", mem):
            db = StateDatabase()
            for st in ["created", "processing", "completed"]:
                await db.add_worker_state(
                    tenant_id=TENANT,
                    subject_id=TENANT,
                    worker_id="wlatest",
                    worker_name="latest_test",
                    is_agent=True,
                    state_type=st,
                    worker_state={"status": st},
                    metadata=None,
                )

            latest = await db.get_latest_worker_state(
                tenant_id=TENANT, subject_id=TENANT, worker_id="wlatest"
            )
            assert latest is not None
            assert latest["state_type"] == "completed"
