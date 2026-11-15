"""Unit tests for the state re-key in ``WorkerRegistryDB.register_worker``.

``WorkerStateManager.resolve_worker_id`` deliberately falls back to using a
worker's NAME as its worker_id when the worker is not in the registry
("legacy compatibility", ``src/services/worker_state_manager.py:172-178``).
That fallback stays -- a whole class of unregistered workers depends on it.

The defect it causes is downstream: when such a worker LATER registers and is
minted a UUID, its earlier rows keep the name as their worker_id.
``get_worker_states_by_id`` binds the same value to both placeholders
(``worker_id = $3 OR worker_name = $4``), so a lookup BY NAME still finds
everything but a lookup BY UUID -- what every caller uses once the worker is
registered -- matches neither column on the older rows.

Measured in production 2026-08-12: 396 rows across 25 workers, e.g.
``video-frame-analyzer`` with 154 name-keyed rows against 8525 uuid-keyed ones.

These tests pin the re-key without a real Postgres, following the fake-connection
pattern of ``test_register_worker_idempotency_unit.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest

from src.db.worker_registry_db import WorkerRegistryDB, _rows_affected

TENANT = "tenant-a"
SUBJECT = "subject-a"
EXISTING_ID = "019ddabb-7e86-7143-aad7-a98c8aff1741"


class _RecordingConn:
    """Fake connection that records every statement issued.

    ``existing_row`` drives which register_worker branch runs: a dict takes the
    UPDATE path, ``None`` takes the INSERT path.
    """

    def __init__(self, *, existing_row: dict[str, Any] | None) -> None:
        self.existing_row = existing_row
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.statements.append((sql, args))
        return self.existing_row

    async def execute(self, sql: str, *args: Any) -> str:
        self.statements.append((sql, args))
        head = sql.lstrip().upper()
        if head.startswith("UPDATE WORKER_STATES"):
            return "UPDATE 154"
        if head.startswith("UPDATE"):
            return "UPDATE 1"
        return "INSERT 0 1"

    def rekey_statements(self) -> list[tuple[str, tuple[Any, ...]]]:
        return [
            s for s in self.statements if s[0].lstrip().upper().startswith("UPDATE WORKER_STATES")
        ]


@asynccontextmanager
async def _conn_ctx(conn: Any) -> Any:
    yield conn


async def _register(conn: Any, *, worker_name: str, worker_id: str | None = None) -> dict[str, Any]:
    with patch(
        "src.db.worker_registry_db.get_pg_connection",
        lambda: _conn_ctx(conn),
    ):
        return await WorkerRegistryDB().register_worker(
            tenant_id=TENANT,
            subject_id=SUBJECT,
            worker_name=worker_name,
            db_path="/tmp/x",
            is_agent=True,
            config={},
            worker_id=worker_id,
        )


class TestRekeyOnRegistration:
    @pytest.mark.asyncio
    async def test_new_registration_rekeys_name_scoped_states(self) -> None:
        """The case the fix exists for: state written before the worker existed."""
        conn = _RecordingConn(existing_row=None)
        result = await _register(conn, worker_name="video-frame-analyzer")

        rekeys = conn.rekey_statements()
        assert len(rekeys) == 1, "a new registration must re-key its name-keyed states"

        sql, args = rekeys[0]
        # Re-keyed ONTO the freshly minted worker_id...
        assert args[0] == result["worker_id"]
        # ...scoped to this tenant/subject, and only rows where BOTH columns
        # carry the name -- so a row belonging to any other worker is untouched.
        assert args[1] == TENANT
        assert args[3] == "video-frame-analyzer"
        assert "worker_id::text = $4" in sql
        assert "worker_name = $4" in sql

    @pytest.mark.asyncio
    async def test_worker_id_is_compared_as_text(self) -> None:
        """The ``::text`` cast is load-bearing and must not be "tidied" away.

        Production declares worker_states.worker_id as TEXT -- it must, because
        the resolve_worker_id fallback stores worker NAMES in it.

        Until 2026-08-13 postgres_schema.sql declared that column UUID, and CI
        builds its ephemeral database from that file, so CI and production
        disagreed on the type. Without the cast postgres inferred $4 as ``uuid``
        from ``worker_id = $4`` and the next term died with "operator does not
        exist: character varying = uuid". The file now declares TEXT, so that
        specific divergence is gone.

        The cast and this test stay anyway. It costs nothing, and it is the pin
        that fails first if the column is ever declared UUID again -- which is a
        live possibility, because "make worker_id a real UUID" is still an open
        end-state (see the 2026-08-13 amendment in postgres_schema.sql). This is
        a unit-level pin because the pg-marked suite that would otherwise catch
        it is deselected by a default local run.
        """
        conn = _RecordingConn(existing_row=None)
        await _register(conn, worker_name="promo-review")

        sql = conn.rekey_statements()[0][0]
        assert "worker_id::text = $4" in sql, "must compare worker_id as text"
        assert "AND worker_id = $4" not in sql, "uncast comparison breaks on a UUID column"

    @pytest.mark.asyncio
    async def test_existing_registration_also_rekeys_so_it_self_heals(self) -> None:
        """Re-registering an already-registered worker repairs historical rows.

        This is what removes the need for a separate backfill migration: the 396
        production rows heal the next time each of those 25 workers registers.
        """
        conn = _RecordingConn(
            existing_row={"worker_id": EXISTING_ID, "operational_status": "active"}
        )
        await _register(conn, worker_name="video-frame-analyzer")

        rekeys = conn.rekey_statements()
        assert len(rekeys) == 1
        assert rekeys[0][1][0] == EXISTING_ID

    @pytest.mark.asyncio
    async def test_uuid_shaped_name_is_never_rekeyed(self) -> None:
        """A name that IS a UUID could collide with another worker's real id.

        resolve_worker_id has a secondary lookup returning such a name verbatim
        as the worker_id, so re-keying it could move a DIFFERENT worker's
        history. Refuse rather than guess.
        """
        conn = _RecordingConn(existing_row=None)
        await _register(conn, worker_name=EXISTING_ID)

        assert conn.rekey_statements() == [], "a UUID-shaped name must not be re-keyed"

    @pytest.mark.asyncio
    async def test_rekey_is_scoped_not_global(self) -> None:
        """Guard against the re-key ever becoming an unscoped UPDATE.

        An unscoped `UPDATE worker_states SET worker_id = ...` would rewrite
        every tenant's history. Pin that the statement carries the scope
        predicate and a WHERE at all.
        """
        conn = _RecordingConn(existing_row=None)
        await _register(conn, worker_name="promo-analyze")

        sql = conn.rekey_statements()[0][0].upper()
        assert " WHERE " in sql
        assert "$2" in sql and "$3" in sql, "must bind tenant and subject into the scope clause"


class TestRowsAffected:
    """`execute` returns a STATUS STRING, not a cursor -- see db_config.py:154."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("UPDATE 396", 396),
            ("UPDATE 0", 0),
            ("INSERT 0 1", 1),
            ("", 0),
            ("UPDATE", 0),
            (None, 0),
            (object(), 0),
        ],
    )
    def test_parses_or_degrades_to_zero(self, status: Any, expected: int) -> None:
        assert _rows_affected(status) == expected

    def test_a_cursor_like_object_does_not_silently_report_zero_rows(self) -> None:
        """Regression pin for the bug this helper replaced.

        The first cut read ``getattr(result, "rowcount", 0)``. Because execute()
        returns a string, that always yielded 0 and the log line could never
        report a real count. If someone reintroduces rowcount-reading, the
        string cases above go red.
        """
        assert _rows_affected("UPDATE 396") == 396
        assert not hasattr("UPDATE 396", "rowcount")
