"""Tests for ``src.api.workflow_routing`` (Lane D of Improvement Plan 4).

The repository talks to Postgres via ``src.db.db_config.get_pg_connection``.
Our CI lane runs against a real Postgres 18, but most of the WM unit suite
(see ``tests/conftest.py``) sets ``WORKER_MGMT_SKIP_MIGRATIONS=1`` and never
opens a pool — every test mocks the DB instead.

To keep this suite consistent with the rest of the repo (and runnable
locally without a Postgres), these tests substitute an in-memory fake
connection that mimics the asyncpg-style wrapper exposed by
``_PsycopgConnectionWrapper``: ``fetch``, ``fetchrow``, ``fetchval``,
``execute``. The fake re-implements just enough of the SQL semantics
(``$N`` placeholder substitution + dict storage keyed by ``asset_key``) to verify the route/repository contract.

This is the same approach used by ``tests/test_run_migrations.py``, where
``_FakePool`` and ``_FakeConn`` stand in for psycopg.
"""

from __future__ import annotations

import os
import secrets
import string
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests._production_error_handlers import install_production_exception_handlers, wire_error

# ---------------------------------------------------------------------------
# In-memory fake repository state (one row per asset_key, migration 036)
# ---------------------------------------------------------------------------


def _wid() -> str:
    """A fresh 16-character n8n-shaped workflow id (036's id shape)."""
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))


# The legacy endpoints are keyed by asset_key since 036; a bare ``list`` maps
# to ``cell:<list>`` (G-15 D1).
ASSET = "cell:READY"


class _FakeStore:
    """In-memory mirror of ``workflow_routing_pointer`` (036 shape).

    Exposes the small surface the legacy repository paths call, with the same
    SQL semantics:
    - ``promote`` upsert: shifts active → prior, sets new active, stamps
      ``promoted_at`` server-side; refused (no row) while a canary is armed.
    - ``rollback`` swap: only when ``prior_workflow_id IS NOT NULL`` and no
      canary is armed.
    - ``extend``: bumps a windowed live canary; a NULL window is a no-op.
    - ``clear_prior`` UPDATE: sets ``prior_workflow_id = NULL``.
    """

    def __init__(self) -> None:
        # key: asset_key → dict of column values
        self.rows: dict[str, dict[str, Any]] = {}

    def get(self, asset_key: str) -> dict[str, Any] | None:
        row = self.rows.get(asset_key)
        return dict(row) if row else None

    def list_observed(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.rows.values()
            if row.get("prior_workflow_id") is not None
            and row.get("observation_window_until") is not None
        ]

    def promote(
        self,
        asset_key: str,
        new_workflow_id: str,
        observation_window_minutes: int,
        rollback_thresholds: dict[str, Any],
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        existing = self.rows.get(asset_key)
        if existing and existing.get("canary_workflow_id") is not None:
            return None  # DO UPDATE … WHERE canary_workflow_id IS NULL
        prior = existing["active_workflow_id"] if existing else None
        row: dict[str, Any] = {
            **(existing or {}),
            "asset_key": asset_key,
            "active_workflow_id": new_workflow_id,
            "prior_workflow_id": prior,
            "promoted_at": now,
            "observation_window_until": now + timedelta(minutes=observation_window_minutes),
            "rollback_thresholds": rollback_thresholds or {},
        }
        self.rows[asset_key] = row
        return dict(row)

    def rollback(self, asset_key: str) -> dict[str, Any] | None:
        row = self.rows.get(asset_key)
        if (
            row is None
            or row.get("prior_workflow_id") is None
            or row.get("canary_workflow_id") is not None
        ):
            return None
        # Swap: active ← prior, prior ← NULL; the real statement also returns
        # the pre-swap active as ``rolled_back_from`` (FOR UPDATE sub-select).
        rolled_back_from = row["active_workflow_id"]
        row["active_workflow_id"] = row["prior_workflow_id"]
        row["prior_workflow_id"] = None
        return {**row, "rolled_back_from": rolled_back_from}

    def clear_prior(self, asset_key: str) -> dict[str, Any] | None:
        row = self.rows.get(asset_key)
        if row is None:
            return None
        row["prior_workflow_id"] = None
        return dict(row)

    def extend_observation_window(
        self, asset_key: str, additional_minutes: int
    ) -> dict[str, Any] | None:
        # Mirrors the real CTE: a row with NO window (a G-15 release) comes
        # back unchanged; a windowed row needs a live prior and is bumped from
        # GREATEST(current window, now).
        row = self.rows.get(asset_key)
        if row is None:
            return None
        if row.get("observation_window_until") is None:
            return dict(row)
        if row.get("prior_workflow_id") is None:
            return None
        base = max(row["observation_window_until"], datetime.now(UTC))
        row["observation_window_until"] = base + timedelta(minutes=additional_minutes)
        return dict(row)


class _FakeConn:
    """asyncpg-style fake matching ``_PsycopgConnectionWrapper``.

    Dispatches on SQL fingerprints. Only the legacy queries used by
    ``src/api/workflow_routing/repository.py`` need to be handled — the G-15
    release writes are covered with the repository mocked
    (tests/test_workflow_routing_release.py) and against Postgres
    (tests/integration/test_migration_036_workflow_routing_release.py).
    """

    def __init__(self, store: _FakeStore) -> None:
        self.store = store

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        sql_lower = " ".join(sql.lower().split())
        if "insert into workflow_routing_pointer" in sql_lower and "do update" in sql_lower:
            # promote(): args = (asset_key, new_workflow_id, mins, thresholds)
            return self.store.promote(args[0], args[1], args[2], args[3])
        if "set active_workflow_id = prior_workflow_id" in sql_lower:
            return self.store.rollback(args[0])
        if "set observation_window_until" in sql_lower:
            # args = (asset_key, additional_minutes)
            return self.store.extend_observation_window(args[0], args[1])
        if "set prior_workflow_id = null" in sql_lower:
            return self.store.clear_prior(args[0])
        if sql_lower.startswith("select ") and "where asset_key = $1" in sql_lower:
            return self.store.get(args[0])
        raise AssertionError(f"Unhandled fetchrow SQL in test fake: {sql!r}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        sql_lower = sql.lower()
        if "where prior_workflow_id is not null" in sql_lower:
            return self.store.list_observed()
        raise AssertionError(f"Unhandled fetch SQL in test fake: {sql!r}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> _FakeStore:
    return _FakeStore()


@pytest.fixture(autouse=True)
def _patch_pool(store: _FakeStore):
    """Replace ``get_pg_connection`` at the repository module's import site.

    The repository imports the symbol at module load time
    (``from src.db.db_config import get_pg_connection``), so patching the
    attribute on ``src.api.workflow_routing.repository`` is sufficient and
    keeps the rest of the WM suite untouched.
    """

    @asynccontextmanager
    async def fake_get_pg_connection():
        yield _FakeConn(store)

    with patch(
        "src.api.workflow_routing.repository.get_pg_connection",
        fake_get_pg_connection,
    ):
        yield


@pytest.fixture()
def app():
    """Build a fresh FastAPI app with auth disabled (matches test_app_config).

    `get_tenant_id` is also a route-level Depends, not middleware, so disabling
    the auth middleware alone leaves it raising 401. Override the dep with a
    fixed test tenant id so the routing-pointer tests can focus on the
    business logic.
    """
    import src.api.app_config as app_cfg
    from src.api.workflow_routing.routes import (
        _require_routing_pointers_enabled,
        _require_routing_writer,
    )
    from src.auth import get_tenant_id

    with (
        patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "false",
                "ENABLE_RATE_LIMITING": "false",
                "ENVIRONMENT": "test",
            },
        ),
        patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
        patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
    ):
        a = app_cfg.create_secure_app()
        # create_secure_app() is not what production serves (worker_management_api:app
        # is), and it lacks production's http_exception_handler, which wraps every
        # HTTPException as {"error": {...}}. Install production's handlers so the
        # bodies pinned below are the ones clients receive.
        install_production_exception_handlers(a)
        # A PLATFORM tenant: the reads answer every other tenant as a miss
        # (strict tenant separation, tests/test_workflow_routing_release.py).
        a.dependency_overrides[get_tenant_id] = lambda: "aaaaaaaa-0000-4000-8000-000000000001"
        # The service-internal write gate (worker_management#230) is exercised
        # end to end, with the real middleware and a real SA JWT, in
        # tests/test_workflow_routing_write_gate.py. Here it is lifted so these
        # tests can focus on pointer semantics.
        a.dependency_overrides[_require_routing_writer] = lambda: None
        # Same reasoning for the pointer-creation gate (rollback-watch timer
        # retired, WORKFLOW_ROUTING_POINTERS_ENABLED default false): its
        # 409-vs-enabled behaviour is exercised in
        # tests/test_workflow_routing_write_gate.py. Lifted here so promote's
        # business logic (and rollback/extend/clear-prior, which depend on a
        # promoted row existing) can be tested without touching the flag.
        a.dependency_overrides[_require_routing_pointers_enabled] = lambda: None
        yield a


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _promote(client: TestClient, **overrides: Any):
    body: dict[str, Any] = {
        "asset_key": ASSET,
        "new_workflow_id": _wid(),
        "observation_window_minutes": 15,
        "rollback_thresholds": {"failure_rate": 0.1},
    }
    body.update(overrides)
    return client.post("/api/workflow-routing/promote", json=body)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromote:
    """POST /api/workflow-routing/promote semantics."""

    def test_first_promote_creates_row_with_null_prior(self, client: TestClient) -> None:
        new_id = _wid()
        resp = _promote(client, new_workflow_id=new_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["active_workflow_id"] == new_id
        assert body["prior_workflow_id"] is None
        assert body["rollback_thresholds"] == {"failure_rate": 0.1}

    def test_second_promote_shifts_active_to_prior(self, client: TestClient) -> None:
        v1 = _wid()
        v2 = _wid()
        first = _promote(client, new_workflow_id=v1)
        assert first.status_code == 200
        second = _promote(client, new_workflow_id=v2)
        assert second.status_code == 200
        body = second.json()
        assert body["active_workflow_id"] == v2
        assert body["prior_workflow_id"] == v1

    def test_observation_window_zero_collapses_to_promoted_at(self, client: TestClient) -> None:
        """Lane F opt-out: window=0 → observation_window_until == promoted_at."""
        resp = _promote(client, observation_window_minutes=0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["promoted_at"] == body["observation_window_until"]

    @pytest.mark.parametrize(
        "bad_id",
        [
            "A" * 15,  # one short
            "A" * 17,  # one long
            "aaaaaaaa-0000-4000-8000-0000000000ff",  # a uuid is not an n8n id
            "abcdefgh-jklmnop",  # 16 chars, not alphanumeric
            "not-a-uuid",
        ],
    )
    def test_promote_rejects_a_non_16_char_workflow_id(
        self, client: TestClient, bad_id: str
    ) -> None:
        resp = _promote(client, new_workflow_id=bad_id)
        assert resp.status_code == 422, resp.text

    def test_promote_accepts_a_16_char_workflow_id(self, client: TestClient) -> None:
        resp = _promote(client, new_workflow_id="Zz9Yy8Xx7Ww6Vv5U")
        assert resp.status_code == 200, resp.text
        assert resp.json()["active_workflow_id"] == "Zz9Yy8Xx7Ww6Vv5U"

    def test_promote_is_409_while_a_canary_is_armed(
        self, client: TestClient, store: _FakeStore
    ) -> None:
        """Repointing under an armed canary would bypass /promote-canary."""
        assert _promote(client).status_code == 200
        store.rows[ASSET]["canary_workflow_id"] = _wid()
        resp = _promote(client)
        assert resp.status_code == 409, resp.text
        assert resp.json() == wire_error(409, {"error": "canary_already_armed"}, "CONFLICT")


class TestLegacyAssetKey:
    """Legacy bodies are keyed by asset_key since 036 (G-15 D1)."""

    def test_a_bare_list_maps_to_the_cell_asset_key(self, client: TestClient) -> None:
        body = {"list": "READY", "new_workflow_id": _wid(), "observation_window_minutes": 0}
        resp = client.post("/api/workflow-routing/promote", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["asset_key"] == "cell:READY"

    def test_a_card_type_without_asset_key_is_400(self, client: TestClient) -> None:
        body = {
            "card_type": "workflow_statistics_analysis",
            "list": "READY",
            "new_workflow_id": _wid(),
            "observation_window_minutes": 0,
        }
        resp = client.post("/api/workflow-routing/promote", json=body)
        assert resp.status_code == 400, resp.text
        assert resp.json() == wire_error(400, {"error": "asset_key_required"}, "BAD_REQUEST")

    def test_a_malformed_asset_key_is_400(self, client: TestClient) -> None:
        resp = client.post("/api/workflow-routing/clear-prior", json={"asset_key": "nomap"})
        assert resp.status_code == 400, resp.text
        assert resp.json() == wire_error(400, {"error": "asset_key_invalid"}, "BAD_REQUEST")


class TestRollback:
    """POST /api/workflow-routing/rollback semantics."""

    def test_rollback_swaps_active_and_prior(self, client: TestClient) -> None:
        v1 = _wid()
        v2 = _wid()
        _promote(client, new_workflow_id=v1)
        _promote(client, new_workflow_id=v2)

        resp = client.post(
            "/api/workflow-routing/rollback",
            json={"asset_key": ASSET, "reason": "p95 regression"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Post-swap: active is the prior, prior is NULL.
        assert body["active_workflow_id"] == v1
        assert body["prior_workflow_id"] is None

    def test_rollback_404_when_no_prior(self, client: TestClient) -> None:
        # Single promote → no prior → cannot roll back.
        _promote(client)
        resp = client.post(
            "/api/workflow-routing/rollback",
            json={"asset_key": ASSET, "reason": ""},
        )
        assert resp.status_code == 404

    def test_rollback_404_while_a_canary_is_armed(
        self, client: TestClient, store: _FakeStore
    ) -> None:
        _promote(client)
        _promote(client)
        store.rows[ASSET]["canary_workflow_id"] = _wid()
        resp = client.post("/api/workflow-routing/rollback", json={"asset_key": ASSET})
        assert resp.status_code == 404, resp.text

    def test_rollback_404_when_no_row(self, client: TestClient) -> None:
        resp = client.post(
            "/api/workflow-routing/rollback",
            json={"asset_key": "cell:nope", "reason": ""},
        )
        assert resp.status_code == 404


# The workflowAutoRollback payload contract. Pinned to the receiver, fr
# backend/src/routes/internal-workflow-rollback.ts (fr#1427 + its follow-up):
#   {event: "workflowAutoRollback", asset_key: str (non-empty), card_type: str|None,
#    list: str|None, active_workflow_id: str|None, rolled_back_from: str|None,
#    reason: str}
# Change one side only together with the other (worker_management#338).
_ROLLBACK_PAYLOAD_KEYS = {
    "event",
    "asset_key",
    "card_type",
    "list",
    "active_workflow_id",
    "rolled_back_from",
    "reason",
}


class TestRollbackWebhookPayload:
    """POST /rollback → the payload handed to ``_emit_rollback_webhook`` (#338).

    MUTATIONS (each turns a test here red): take ``card_type``/``list`` from the
    request body (None in both accepted shapes); read ``rolled_back_from`` from
    the post-swap ``prior_workflow_id`` (always NULL).
    """

    @pytest.fixture()
    def sent(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        from src.api.workflow_routing import routes

        captured: list[dict[str, Any]] = []

        async def capture(payload: dict[str, Any]) -> None:
            captured.append(payload)

        monkeypatch.setattr(routes, "_emit_rollback_webhook", capture)
        return captured

    @pytest.mark.parametrize(
        "request_body",
        [
            pytest.param({"asset_key": ASSET, "reason": "p95 regression"}, id="asset_key_only"),
            pytest.param({"list": "READY", "reason": "p95 regression"}, id="list_only"),
        ],
    )
    def test_payload_comes_from_the_pointer_row(
        self,
        client: TestClient,
        store: _FakeStore,
        sent: list[dict[str, Any]],
        request_body: dict[str, Any],
    ) -> None:
        v1, v2 = _wid(), _wid()
        _promote(client, new_workflow_id=v1)
        _promote(client, new_workflow_id=v2)
        # The legacy (card_type, list) columns of the row; neither request
        # shape carries both, so only the row can supply them.
        store.rows[ASSET]["card_type"] = "research"
        store.rows[ASSET]["list"] = "READY"

        resp = client.post("/api/workflow-routing/rollback", json=request_body)
        assert resp.status_code == 200, resp.text

        assert len(sent) == 1
        payload = sent[0]
        assert set(payload) == _ROLLBACK_PAYLOAD_KEYS
        assert payload["event"] == "workflowAutoRollback"
        assert payload["asset_key"] == ASSET
        assert payload["card_type"] == "research"
        assert payload["list"] == "READY"
        assert payload["active_workflow_id"] == v1  # the old prior, now active
        assert payload["rolled_back_from"] == v2  # the pre-swap active, never NULL
        assert payload["reason"] == "p95 regression"
        # The public response shape is unchanged: no extra key leaks into it.
        assert "rolled_back_from" not in resp.json()

    def test_no_payload_when_the_rollback_is_refused(
        self, client: TestClient, sent: list[dict[str, Any]]
    ) -> None:
        _promote(client)
        resp = client.post("/api/workflow-routing/rollback", json={"asset_key": ASSET})
        assert resp.status_code == 404
        assert sent == []


class TestClearPrior:
    """POST /api/workflow-routing/clear-prior semantics."""

    def test_clear_prior_nulls_prior(self, client: TestClient) -> None:
        v1 = _wid()
        v2 = _wid()
        _promote(client, new_workflow_id=v1)
        _promote(client, new_workflow_id=v2)

        resp = client.post(
            "/api/workflow-routing/clear-prior",
            json={"asset_key": ASSET},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["active_workflow_id"] == v2
        assert body["prior_workflow_id"] is None

    def test_clear_prior_404_when_row_missing(self, client: TestClient) -> None:
        resp = client.post(
            "/api/workflow-routing/clear-prior",
            json={"asset_key": "cell:missing"},
        )
        assert resp.status_code == 404


class TestExtend:
    """POST /api/workflow-routing/extend semantics."""

    def test_extend_moves_window_forward(self, client: TestClient) -> None:
        # A live canary requires a prior_workflow_id, so promote twice: the
        # second promote shifts v1 → prior and arms the observation window.
        _promote(client, new_workflow_id=_wid(), observation_window_minutes=15)
        promote = _promote(client, new_workflow_id=_wid(), observation_window_minutes=15)
        assert promote.status_code == 200, promote.text
        assert promote.json()["prior_workflow_id"] is not None
        before = datetime.fromisoformat(promote.json()["observation_window_until"])

        resp = client.post(
            "/api/workflow-routing/extend",
            json={
                "asset_key": ASSET,
                "additional_minutes": 30,
            },
        )
        assert resp.status_code == 200, resp.text
        after = datetime.fromisoformat(resp.json()["observation_window_until"])
        # Window still open (15m out) → extend the tail by exactly 30 minutes.
        assert after - before == timedelta(minutes=30)

    def test_extend_404_when_prior_cleared(self, client: TestClient) -> None:
        # Promote twice to arm a canary, then clear the prior (window lapsed,
        # promotion locked in). The row still exists and active_workflow_id is
        # set, but there is no live canary to keep observing → extend must 404,
        # matching how /rollback behaves once the prior is gone.
        _promote(client, new_workflow_id=_wid())
        _promote(client, new_workflow_id=_wid())
        cleared = client.post(
            "/api/workflow-routing/clear-prior",
            json={"asset_key": ASSET},
        )
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["prior_workflow_id"] is None

        resp = client.post(
            "/api/workflow-routing/extend",
            json={
                "asset_key": ASSET,
                "additional_minutes": 30,
            },
        )
        assert resp.status_code == 404

    def test_extend_404_when_no_pointer(self, client: TestClient) -> None:
        resp = client.post(
            "/api/workflow-routing/extend",
            json={"asset_key": "cell:never_promoted", "additional_minutes": 30},
        )
        assert resp.status_code == 404

    def test_extend_on_a_row_with_no_window_is_a_200_no_op(
        self, client: TestClient, store: _FakeStore
    ) -> None:
        """A G-15 release has no clock (decisions 5/7); approval-callback still
        calls /extend, so it must answer 200 and change nothing."""
        active = _wid()
        store.rows[ASSET] = {
            "asset_key": ASSET,
            "active_workflow_id": active,
            "prior_workflow_id": None,
            "promoted_at": datetime.now(UTC),
            "observation_window_until": None,
            "rollback_thresholds": {},
        }
        resp = client.post(
            "/api/workflow-routing/extend",
            json={"asset_key": ASSET, "additional_minutes": 30},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["observation_window_until"] is None
        assert resp.json()["active_workflow_id"] == active
        assert store.rows[ASSET]["observation_window_until"] is None

    def test_extend_rejects_zero_minutes(self, client: TestClient) -> None:
        _promote(client)
        resp = client.post(
            "/api/workflow-routing/extend",
            json={
                "asset_key": ASSET,
                "additional_minutes": 0,
            },
        )
        assert resp.status_code == 422

    def test_extend_rejects_over_ceiling(self, client: TestClient) -> None:
        _promote(client)
        resp = client.post(
            "/api/workflow-routing/extend",
            json={
                "asset_key": ASSET,
                "additional_minutes": 10_081,
            },
        )
        assert resp.status_code == 422

    def test_extend_rejects_negative_minutes(self, client: TestClient) -> None:
        _promote(client)
        resp = client.post(
            "/api/workflow-routing/extend",
            json={
                "asset_key": ASSET,
                "additional_minutes": -5,
            },
        )
        assert resp.status_code == 422


class TestActiveAndObserved:
    """GET endpoints used by n8n entry workflows + the rollback monitor."""

    def test_get_active_returns_pointer(self, client: TestClient) -> None:
        v1 = _wid()
        _promote(client, new_workflow_id=v1)
        resp = client.get(
            "/api/workflow-routing/active",
            params={"asset_key": ASSET},
        )
        assert resp.status_code == 200
        assert resp.json()["active_workflow_id"] == v1

    def test_get_active_404_when_missing(self, client: TestClient) -> None:
        resp = client.get(
            "/api/workflow-routing/active",
            params={"asset_key": "cell:nope"},
        )
        assert resp.status_code == 404

    def test_observed_only_returns_rows_with_prior(self, client: TestClient) -> None:
        # Row 1: promoted twice → has prior → should appear in /observed.
        _promote(client, asset_key=ASSET, new_workflow_id=_wid())
        _promote(client, asset_key=ASSET, new_workflow_id=_wid())
        # Row 2: promoted once → no prior → should NOT appear.
        _promote(client, asset_key="cell:DOING", new_workflow_id=_wid())

        resp = client.get("/api/workflow-routing/observed")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["asset_key"] == ASSET
        assert rows[0]["prior_workflow_id"] is not None

    def test_observed_excludes_a_row_with_no_window(
        self, client: TestClient, store: _FakeStore
    ) -> None:
        """A /promote-canary row has a prior but NO clock: the dormant
        rollback-watch must never see it (decision 7)."""
        store.rows["activity:AA Activity - Reasoning"] = {
            "asset_key": "activity:AA Activity - Reasoning",
            "active_workflow_id": _wid(),
            "prior_workflow_id": _wid(),
            "promoted_at": datetime.now(UTC),
            "observation_window_until": None,
            "rollback_thresholds": {},
        }
        resp = client.get("/api/workflow-routing/observed")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_active_legacy_list_param_maps_to_the_cell_key(self, client: TestClient) -> None:
        v1 = _wid()
        _promote(client, asset_key="cell:doing", new_workflow_id=v1)
        resp = client.get("/api/workflow-routing/active", params={"list": "doing"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["asset_key"] == "cell:doing"
        assert resp.json()["active_workflow_id"] == v1
        assert resp.json()["canary"] is None

    def test_get_active_with_card_type_and_no_asset_key_is_400(self, client: TestClient) -> None:
        resp = client.get(
            "/api/workflow-routing/active", params={"card_type": "c", "list": "doing"}
        )
        assert resp.status_code == 400, resp.text
        assert resp.json() == wire_error(400, {"error": "asset_key_required"}, "BAD_REQUEST")


# ---------------------------------------------------------------------------
# Observability: who calls the fleet-wide routing pointer (worker_management#230)
# ---------------------------------------------------------------------------


class TestRoutingCallCounter:
    """``worker_management_workflow_routing_calls_total`` must actually fire.

    This counter is the whole reason the gate in #230 is decidable: routing
    requests reach the log only as uvicorn access lines carrying no identity,
    and ``kubectl logs`` serves one rotating file (the readable window on
    2026-09-08 was 49 minutes). So the tenant behind a routing mutation was
    unmeasurable. These tests pin that it is now measured — and measured with a
    BOUNDED tenant label, which is the property that keeps it safe to ship.
    """

    @staticmethod
    def _calls(recorder) -> list[dict[str, Any]]:
        return [c.kwargs for c in recorder.call_args_list]

    @pytest.fixture()
    def recorder(self):
        """Capture ``record_workflow_routing_call`` at the ROUTE's import site.

        ``routes.py`` calls ``get_metrics_client()`` per request, so patching
        the client's method reaches the real call path rather than a stand-in.
        """
        from src.telemetry import get_metrics_client

        # ⛔ create=True is load-bearing for the RED half of this suite. Without
        # it, patching a not-yet-existing method raises AttributeError and the
        # test fails for the WRONG reason — proving only that the symbol is
        # absent, never that the request failed to record anything. With it, the
        # un-patched build runs the real request and the assertions below fail on
        # a genuine "the counter was never called".
        with patch.object(
            type(get_metrics_client()),
            "record_workflow_routing_call",
            create=True,
        ) as rec:
            yield rec

    def test_promote_is_counted_with_endpoint_and_method(
        self, client: TestClient, recorder
    ) -> None:
        assert _promote(client).status_code == 200
        calls = self._calls(recorder)
        assert len(calls) == 1, calls
        assert calls[0]["endpoint"] == "/api/workflow-routing/promote"
        assert calls[0]["method"] == "POST"

    def test_a_measured_live_caller_tenant_is_recorded_verbatim(self, app) -> None:
        """⭐ THE TEST THAT PROVES THE MEASUREMENT IS USEFUL.

        A request that authenticated with a LICENCE token must land its real
        tenant in the label — otherwise the counter ships, increments, and still
        cannot answer step 2 of #230. The tenant used here is ``n8n-dev``
        (``aaaaaaaa-0000-4000-8000-000000000001``), measured live in the WM pod
        log on 2026-09-08 and registered ``kind='internal'`` in
        ``licenses.tenants``; it is the deployment's own DEFAULT_TENANT_ID and
        the tenant the n8n callers' licence tokens belong to.
        """
        from src.telemetry import get_metrics_client

        n8n_dev = "aaaaaaaa-0000-4000-8000-000000000001"

        # Stand in for AuthenticationMiddleware's token branch, which stamps a
        # licence id alongside the introspected tenant.
        #
        # ⚠️ THE FLAGS ARE PART OF THE SHAPE, not decoration. `_stamp_identity`
        # writes `identity_proven`/`tenant_proven` as bool literals on the SAME
        # branch that sets `license_id`; the real middleware never produces a
        # `license_id` without them. An earlier version of this double omitted
        # both, which made it model a request production cannot emit — and that
        # is precisely the shape that let a caller-supplied uuid reach the
        # label. A double that cannot express production hides the defect it
        # was written to catch.
        @app.middleware("http")
        async def _stamp(request, call_next):
            request.state.tenant_id = n8n_dev
            request.state.license_id = "lic-abc"
            request.state.identity_proven = True
            request.state.tenant_proven = True
            return await call_next(request)

        with patch.object(
            type(get_metrics_client()), "record_workflow_routing_call", create=True
        ) as rec:
            c = TestClient(app, raise_server_exceptions=False)
            assert c.post(
                "/api/workflow-routing/clear-prior",
                json={"asset_key": ASSET},
            ).status_code in (200, 404)

        assert rec.call_count == 1, f"counter never fired: {rec.call_args_list}"
        kwargs = rec.call_args.kwargs
        assert kwargs["tenant"] == n8n_dev, kwargs
        assert kwargs["auth_path"] == "license", kwargs
        assert kwargs["endpoint"] == "/api/workflow-routing/clear-prior", kwargs

    def test_a_licence_id_without_proven_provenance_cannot_carry_a_raw_tenant(self, app) -> None:
        """⛔ THE FAIL-OPEN PIN. Guards the direction, not just the outcome.

        The raw-uuid branch requires ``tenant_proven is True``, not merely a
        truthy ``license_id``. Without that conjunct the branch fails OPEN: any
        shape carrying a ``license_id`` puts ``state.tenant_id`` — a value the
        caller supplied and ``_effective_tenant`` checked for SPELLING only —
        straight into a Prometheus label, which is the unbounded-series hole
        this counter's cardinality bound exists to forbid.

        This shape is UNREACHABLE in production today: ``_stamp_identity``
        writes both flags as bool literals on the same branch that sets
        ``license_id``. That is exactly why it needs a test. A guard whose
        violating input cannot occur yet is a guard nothing pins, and a
        mutation battery proved it: reverting this branch to ``license_id``
        alone left the whole suite green. Unreachable today is not unreachable
        after the next refactor.
        """
        from src.telemetry import get_metrics_client

        attacker_uuid = "aaaaaaaa-dead-7000-8000-ffffffffffff"

        # A licence id with NO provenance flags — what a partial rollback, or a
        # future branch that forgets to stamp, would produce.
        @app.middleware("http")
        async def _stamp(request, call_next):
            request.state.tenant_id = attacker_uuid
            request.state.license_id = "lic-forged"
            return await call_next(request)

        with patch.object(
            type(get_metrics_client()), "record_workflow_routing_call", create=True
        ) as rec:
            c = TestClient(app, raise_server_exceptions=False)
            c.get(
                "/api/workflow-routing/active",
                params={"asset_key": ASSET},
            )

        # POSITIVE CONTROL first: an empty call list would make the assertion
        # below pass vacuously, which is the failure mode this suite exists to
        # avoid.
        assert rec.call_count == 1, f"counter never fired: {rec.call_args_list}"
        kwargs = rec.call_args.kwargs
        assert attacker_uuid not in kwargs.values(), (
            "a caller-supplied uuid reached a label with no proven provenance — "
            f"the cardinality bound is open: {kwargs}"
        )
        assert kwargs["tenant"] == "<unproven>", kwargs

    def test_an_asserted_tenant_never_reaches_the_label(self, app) -> None:
        """⛔ The cardinality guard, and the reason this counter is safe to ship.

        On the tokenless trusted-network branch the caller NAMES its own tenant
        and ``resolve_tenant_id`` accepts any canonical uuid — so putting that
        value in a label would let a peer on the pod network mint unbounded time
        series. It must collapse to the sentinel. The two neighbouring counters
        refuse a tenant label outright for exactly this reason; this one is only
        allowed to carry a tenant because of this branch.
        """
        from src.telemetry import get_metrics_client

        attacker_chosen = "aaaaaaaa-dead-7000-8000-ffffffffffff"

        @app.middleware("http")
        async def _stamp(request, call_next):
            request.state.tenant_id = attacker_chosen
            request.state.license_id = None  # no licence => nothing proved it
            return await call_next(request)

        with patch.object(
            type(get_metrics_client()), "record_workflow_routing_call", create=True
        ) as rec:
            c = TestClient(app, raise_server_exceptions=False)
            c.get(
                "/api/workflow-routing/active",
                params={"asset_key": ASSET},
                headers={"X-Tenant-Id": attacker_chosen},
            )

        assert rec.call_count == 1, f"counter never fired: {rec.call_args_list}"
        kwargs = rec.call_args.kwargs
        assert kwargs["tenant"] == "<unproven>", kwargs
        assert attacker_chosen not in kwargs.values(), kwargs
        assert kwargs["auth_path"] == "asserted", kwargs

    # ─────────────────────────────────────────────────────────────────────
    # The DELEGATING caller. ``AuthenticationMiddleware`` nulls ``license_id``
    # on this branch on purpose (see the "DO NOT fall back to the introspected
    # licence here" comment): attribution belongs to the ADOPTED tenant, not to
    # the service account whose token was actually verified. So provenance can
    # NOT be re-derived from ``license_id`` — the flags must be read.
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stamp(app, **state: Any):
        """Install a middleware that stands in for AuthenticationMiddleware.

        Sets exactly the ``request.state`` attributes named — and nothing else —
        so a test can reproduce a branch's stamp shape verbatim, including the
        shapes that OMIT a flag.
        """

        @app.middleware("http")
        async def _m(request, call_next):
            for k, v in state.items():
                setattr(request.state, k, v)
            return await call_next(request)

        return _m

    def _record_one(self, app) -> dict[str, Any]:
        from src.telemetry import get_metrics_client

        with patch.object(
            type(get_metrics_client()), "record_workflow_routing_call", create=True
        ) as rec:
            TestClient(app, raise_server_exceptions=False).get(
                "/api/workflow-routing/active",
                params={"asset_key": ASSET},
                headers={"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000001"},
            )
        assert rec.call_count == 1, f"counter never fired: {rec.call_args_list}"
        return dict(rec.call_args.kwargs)

    def test_a_delegating_caller_is_not_labelled_the_same_as_an_anonymous_one(self, app) -> None:
        """⭐ THE DELEGATED POPULATION MUST NOT BE INVISIBLE INSIDE THE TOKENLESS ONE.

        A delegating service account DID prove who it is — a licence token was
        introspected ``active`` — and the middleware then nulled ``license_id``
        deliberately, because the request was re-scoped to somebody else's
        tenant. Deriving ``auth_path`` from ``license_id`` alone therefore files
        it under the same value as a caller that presented no credential at all.

        This is the same conflation ``_modality_router._withhold_reason``
        already refuses to make (``adopted_tenant`` vs ``unproven_identity``),
        and for the same reason: the two populations have different owners and
        different fixes — "present a token" vs "present a token FOR THAT
        TENANT". Merging them makes the smaller one unmeasurable.
        """
        adopted = "aaaaaaaa-c0de-7000-8000-000000000042"
        self._stamp(
            app,
            tenant_id=adopted,
            license_id=None,  # nulled by the delegation branch, on purpose
            identity_proven=True,  # a real licence token WAS verified
            tenant_proven=False,  # …but the tenant came from a spelling-only header
        )
        kwargs = self._record_one(app)
        assert kwargs["auth_path"] != "asserted", (
            "a delegating caller is being counted as an anonymous trusted-network "
            f"caller; the delegated population is invisible: {kwargs}"
        )
        assert kwargs["auth_path"] == "delegated", kwargs

    def test_a_delegated_adopted_tenant_never_reaches_the_label(self, app) -> None:
        """⛔ The cardinality half, and the sharper of the two bugs.

        A delegating caller that DOES forward ``X-License-Id`` keeps a truthy
        ``license_id`` (``fwd_license or None``) while ``tenant_id`` is the
        ADOPTED value — validated on SPELLING alone, exactly like the bare
        ``X-Tenant-Id`` the sibling test guards against. Classifying on
        ``license_id`` alone therefore emits ``auth_path="license"`` AND puts a
        caller-chosen uuid in the ``tenant`` label, breaking the bound the
        counter's own docstring claims ("emits a raw tenant ONLY when it came
        from licence INTROSPECTION").
        """
        adopted = "aaaaaaaa-dead-7000-8000-ffffffffffff"
        self._stamp(
            app,
            tenant_id=adopted,
            license_id="lic-forwarded-by-the-caller",
            identity_proven=True,
            tenant_proven=False,
        )
        kwargs = self._record_one(app)
        assert kwargs["tenant"] == "<unproven>", kwargs
        assert adopted not in kwargs.values(), kwargs
        assert kwargs["auth_path"] == "delegated", kwargs

    def test_an_explicitly_unproven_caller_is_still_asserted(self, app) -> None:
        """NEGATIVE CONTROL: the tokenless branch stamps BOTH flags False.

        ``_stamp_nothing_proven`` sets ``identity_proven=False`` and
        ``tenant_proven=False``. That caller must keep reading ``asserted`` —
        the delegated value may not swallow the population it was split out of.
        """
        self._stamp(
            app,
            tenant_id="aaaaaaaa-0000-4000-8000-000000000001",
            license_id=None,
            identity_proven=False,
            tenant_proven=False,
        )
        assert self._record_one(app)["auth_path"] == "asserted"

    def test_a_fully_proven_licence_caller_is_unchanged(self, app) -> None:
        """NEGATIVE CONTROL: the ordinary non-delegating token branch.

        ``tenant_proven=not delegated`` is True here, so this must still be
        ``license`` with the real tenant in the label.
        """
        n8n_dev = "aaaaaaaa-0000-4000-8000-000000000001"
        self._stamp(
            app,
            tenant_id=n8n_dev,
            license_id="lic-abc",
            identity_proven=True,
            tenant_proven=True,
        )
        kwargs = self._record_one(app)
        assert kwargs["auth_path"] == "license", kwargs
        assert kwargs["tenant"] == n8n_dev, kwargs

    def test_auth_path_values_are_a_closed_set(self) -> None:
        """Every value this dependency can emit must be declared.

        ``auth_path`` is a Prometheus label; an undeclared value is an
        unreviewed series. Pinned so a future branch cannot add one silently.
        """
        from src.telemetry.agent_metrics import WORKFLOW_ROUTING_AUTH_PATHS

        assert set(WORKFLOW_ROUTING_AUTH_PATHS) == {
            "license",
            "delegated",
            "asserted",
            "default",
            "none",
        }
        assert len(WORKFLOW_ROUTING_AUTH_PATHS) == len(set(WORKFLOW_ROUTING_AUTH_PATHS))

    def test_reads_are_counted_too(self, client: TestClient, recorder) -> None:
        """/observed is the only endpoint with live traffic — it must be visible."""
        assert client.get("/api/workflow-routing/observed").status_code == 200
        assert self._calls(recorder)[0]["endpoint"] == "/api/workflow-routing/observed"

    def test_a_telemetry_failure_cannot_break_the_request(self, client, recorder) -> None:
        """⛔ Observe-only means the request survives a broken counter.

        This dependency sits on the dispatch hot path. If a metrics backend
        fault could 500 a routing request, the instrument would be worse than
        the blindness it cures.
        """
        recorder.side_effect = RuntimeError("metrics backend down")
        assert _promote(client).status_code == 200
