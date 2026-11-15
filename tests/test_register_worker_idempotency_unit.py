"""Unit tests for the get-or-create idempotency path in
``WorkerRegistryDB.register_worker``.

``register_worker`` does a check-then-insert: it SELECTs within the caller's
visibility scope ``(tenant_id, subject, visibility)`` and, on a miss, blindly
INSERTs. The ``unique_tenant_worker_name (tenant_id, worker_name)`` constraint
is *narrower* than that lookup, so the INSERT can trip it even though the
SELECT missed. These tests pin the two recovery outcomes without a real
Postgres by driving a fake connection that raises ``UniqueViolation`` on the
INSERT:

* re-SELECT finds a row  -> idempotent return of the surviving worker (a lost
  create race); no clobbering UPDATE is issued.
* re-SELECT finds nothing -> :class:`WorkerNameConflictError` (a genuine
  cross-subject name conflict), raised without a retry storm.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import psycopg
import pytest

from src.db.worker_registry_db import WorkerNameConflictError, WorkerRegistryDB
from tests.db_test_helper import _FakeTransaction


class _RaceConn:
    """Fake psycopg-style connection simulating a UNIQUE collision on INSERT.

    The initial scoped lookup (``fetchrow`` call #1) misses; the INSERT raises
    ``UniqueViolation``; the recovery re-SELECT (``fetchrow`` call #2+) returns
    ``recover_row`` (``None`` models a genuine cross-owner conflict).
    """

    def __init__(self, *, recover_row: dict[str, Any] | None) -> None:
        self.recover_row = recover_row
        self.select_calls = 0
        self.insert_calls = 0
        self.update_calls = 0

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.select_calls += 1
        if self.select_calls == 1:
            return None  # initial lookup misses
        return self.recover_row

    async def execute(self, sql: str, *args: Any) -> str:
        head = sql.lstrip().upper()
        if head.startswith("INSERT"):
            self.insert_calls += 1
            raise psycopg.errors.UniqueViolation(
                'duplicate key value violates unique constraint "unique_tenant_worker_name"'
            )
        if head.startswith("UPDATE"):
            self.update_calls += 1
        return "OK"

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()


def _patch_conn(conn: _RaceConn):
    """Patch ``get_pg_connection`` in the registry module to yield ``conn``."""

    @asynccontextmanager
    async def _yield_conn():
        yield conn

    return patch("src.db.worker_registry_db.get_pg_connection", _yield_conn)


class TestRegisterWorkerRaceRecovery:
    @pytest.mark.asyncio
    async def test_lost_create_race_returns_existing_row_idempotently(self):
        """INSERT collides but the row is visible to the caller: return it, don't 500."""
        # Models what the recovery SELECT actually returns — it reads the row's
        # own config/is_agent/columns, not just its id. A double narrower than
        # the query it stands for is how the branch below shipped returning the
        # caller's payload unnoticed.
        conn = _RaceConn(
            recover_row={
                "worker_id": "winner-id",
                "operational_status": "offline",
                "config": {"model": "gpt-4"},
                "is_agent": False,
                "output_schema": None,
                "output_validator": None,
                "config_revision": None,
            }
        )
        db = WorkerRegistryDB()

        with _patch_conn(conn):
            result = await db.register_worker(
                tenant_id="t1",
                subject_id="t1",
                worker_name="racer",
                db_path="/tmp",
                is_agent=False,
                config={"model": "gpt-4"},
                worker_id="loser-id",
            )

        # Idempotent get-or-create: the surviving (winner's) worker_id is returned,
        # NOT the id this call generated.
        assert result["worker_id"] == "winner-id"
        assert result["worker_name"] == "racer"
        assert result["operational_status"] == "offline"
        # Exactly one INSERT attempt and NO clobbering UPDATE — get semantics,
        # the first writer's config is preserved.
        assert conn.insert_calls == 1
        assert conn.update_calls == 0

    @pytest.mark.asyncio
    async def test_the_loser_is_told_what_the_WINNER_stored(self):
        """The race loser must not be handed its own payload back.

        "The first writer wins; we do not clobber its config" — so the loser's
        body is exactly what did NOT get stored. This branch used to return
        ``config=<the caller's dict>`` and ``is_agent=<the caller's flag>``,
        which described a row that does not exist. Since 2026-08-28 both
        crud_endpoints handlers build their response body AND their
        worker_states audit metadata from this dict, so that lie reaches the
        wire: a caller sees its own rejected values echoed back as though they
        had been persisted, and four services in this estate re-PUT what they
        read.
        """
        conn = _RaceConn(
            recover_row={
                "worker_id": "winner-id",
                "operational_status": "offline",
                "config": {"model": "winner-model", "system_prompt": "WINNER"},
                "is_agent": True,
                "output_schema": {"type": "object"},
                "output_validator": "winner-validator",
                "config_revision": "winner-rev",
            }
        )
        db = WorkerRegistryDB()

        with _patch_conn(conn):
            result = await db.register_worker(
                tenant_id="t1",
                subject_id="t1",
                worker_name="racer",
                db_path="/tmp",
                is_agent=False,
                config={"model": "loser-model", "system_prompt": "LOSER"},
                worker_id="loser-id",
                output_schema='{"loser": true}',
                output_validator="loser-validator",
                config_revision="loser-rev",
            )

        assert result["config"] == {"model": "winner-model", "system_prompt": "WINNER"}
        assert result["is_agent"] is True, "the caller passed False; the row holds True"
        assert result["output_schema"] == {"type": "object"}
        assert result["output_validator"] == "winner-validator"
        assert result["config_revision"] == "winner-rev"
        # Same keys the UPDATE branch returns — a handler must not have to know
        # which branch it got.
        assert {
            "config",
            "is_agent",
            "output_schema",
            "output_validator",
            "config_revision",
        } <= set(result)

    @pytest.mark.asyncio
    async def test_cross_owner_conflict_raises_typed_error_without_retry_storm(self):
        """INSERT collides on a row the caller cannot see: raise WorkerNameConflictError."""
        conn = _RaceConn(recover_row=None)
        db = WorkerRegistryDB()

        with _patch_conn(conn), pytest.raises(WorkerNameConflictError) as exc_info:
            await db.register_worker(
                tenant_id="t1",
                subject_id="bob",
                worker_name="taken",
                db_path="/tmp",
                is_agent=False,
                config={},
            )

        assert exc_info.value.worker_name == "taken"
        assert exc_info.value.tenant_id == "t1"
        # Not retried: @with_retry only catches psycopg errors, so exactly one
        # INSERT attempt (one initial SELECT + one recovery re-SELECT).
        assert conn.insert_calls == 1
        assert conn.select_calls == 2

    @pytest.mark.asyncio
    async def test_conflict_error_is_not_a_psycopg_error(self):
        """The typed conflict must not be a psycopg.Error (else @with_retry retries it)."""
        assert not issubclass(WorkerNameConflictError, psycopg.Error)
