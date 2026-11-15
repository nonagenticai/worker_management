"""Unit tests for db/worker_registry_db.py with InMemoryDB.

Verifies real data flow: register workers, retrieve them,
test case-insensitive lookup, config handling, etc.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from src.db.worker_registry_db import WorkerRegistryDB
from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

# ---------------------------------------------------------------------------
# Register and get roundtrip
# ---------------------------------------------------------------------------


class TestRegisterWorker:
    @pytest.mark.asyncio
    async def test_register_new_worker(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            result = await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="my-worker",
                db_path="/tmp",
                is_agent=False,
                config={"model": "gpt-4"},
                worker_id="w-100",
                status="created",
            )

        assert result["worker_name"] == "my-worker"
        assert result["tenant_id"] == "c1"
        assert result["worker_id"] == "w-100"
        assert result["status"] == "created"
        assert result["config"] == {"model": "gpt-4"}
        assert result["operational_status"] == "offline"

    @pytest.mark.asyncio
    async def test_register_then_get_roundtrip(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="my-worker",
                db_path="/tmp",
                is_agent=True,
                config={"engine": "claude"},
                worker_id="w-1",
            )

            fetched = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="my-worker")

        assert fetched is not None
        assert fetched["worker_name"] == "my-worker"
        assert fetched["is_agent"] is True
        assert fetched["config"] == {"engine": "claude"}
        assert fetched["worker_id"] == "w-1"

    @pytest.mark.asyncio
    async def test_reregister_updates_config(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            first = await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="my-worker",
                db_path="/tmp",
                is_agent=False,
                config={"version": 1},
                worker_id="w-1",
            )

            second = await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="my-worker",
                db_path="/tmp2",
                is_agent=True,
                config={"version": 2},
            )

        # Should reuse the same worker_id
        assert second["worker_id"] == first["worker_id"]
        assert second["config"] == {"version": 2}
        assert second["is_agent"] is True


# ---------------------------------------------------------------------------
# Get worker - case insensitive
# ---------------------------------------------------------------------------


class TestGetWorker:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="MyWorker",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            result = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="MyWorker")

        assert result is not None
        assert result["worker_name"] == "MyWorker"

    @pytest.mark.asyncio
    async def test_case_insensitive_fallback(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="MyWorker",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            # The code does a second query with LOWER(worker_name).
            # Our InMemoryDB handles __lower columns.
            result = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="myworker")

        assert result is not None
        assert result["worker_name"] == "MyWorker"

    @pytest.mark.asyncio
    async def test_not_found(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            result = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_config_json_string_parsed(self):
        """If config is stored as a JSON string (legacy), it should be parsed to dict."""
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            # Intentional: legacy code paths stored config as a JSON string;
            # this test verifies the read side parses it back to a dict.
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                db_path="/tmp",
                is_agent=False,
                config=cast("Any", json.dumps({"parsed": True})),
                worker_id="w-1",
            )
            result = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="w1")

        assert result is not None
        assert result["config"] == {"parsed": True}

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="shared-name",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )

            result_c1 = await db.get_worker(
                tenant_id="c1", subject_id="c1", worker_name="shared-name"
            )
            result_c2 = await db.get_worker(
                tenant_id="c2", subject_id="c2", worker_name="shared-name"
            )

        assert result_c1 is not None
        assert result_c2 is None


# ---------------------------------------------------------------------------
# Get worker by ID
# ---------------------------------------------------------------------------


class TestGetWorkerById:
    @pytest.mark.asyncio
    async def test_found(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            result = await db.get_worker_by_id(tenant_id="c1", subject_id="c1", worker_id="w-1")

        assert result is not None
        assert result["worker_id"] == "w-1"

    @pytest.mark.asyncio
    async def test_not_found(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            result = await db.get_worker_by_id(
                tenant_id="c1", subject_id="c1", worker_id="nonexistent"
            )

        assert result is None


# ---------------------------------------------------------------------------
# List workers
# ---------------------------------------------------------------------------


class TestListWorkers:
    @pytest.mark.asyncio
    async def test_returns_worker_names(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="alpha",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="beta",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-2",
            )
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="gamma",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-3",
            )

            result = await db.list_workers(tenant_id="c1", subject_id="c1")

        assert len(result) == 3
        assert set(result) == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_list_with_limit_offset(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            for i in range(5):
                await db.register_worker(
                    tenant_id="c1",
                    subject_id="c1",
                    worker_name=f"worker-{i:02d}",
                    db_path="/tmp",
                    is_agent=False,
                    config={},
                    worker_id=f"w-{i}",
                )

            limited = await db.list_workers(tenant_id="c1", subject_id="c1", limit=2)
            offset = await db.list_workers(tenant_id="c1", subject_id="c1", limit=2, offset=2)

        assert len(limited) == 2
        assert len(offset) == 2

    @pytest.mark.asyncio
    async def test_filters_invalid_names(self):
        """Null bytes, empty strings, and whitespace-only names are filtered."""
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="valid",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            # Manually insert bad records into the in-memory table
            from tests.db_test_helper import _FakeRecord

            bad_rows = [
                _FakeRecord(
                    worker_name="",
                    tenant_id="c1",
                    subject_id="c1",
                    worker_id="w-bad1",
                ),
                _FakeRecord(
                    worker_name="has\x00null",
                    tenant_id="c1",
                    subject_id="c1",
                    worker_id="w-bad2",
                ),
                _FakeRecord(
                    worker_name="   ",
                    tenant_id="c1",
                    subject_id="c1",
                    worker_id="w-bad3",
                ),
            ]
            mem._table("workers").extend(bad_rows)

            result = await db.list_workers(tenant_id="c1", subject_id="c1")

        assert result == ["valid"]

    @pytest.mark.asyncio
    async def test_list_multi_tenant(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="a",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            await db.register_worker(
                tenant_id="c2",
                subject_id="c2",
                worker_name="b",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-2",
            )

            result_c1 = await db.list_workers(tenant_id="c1", subject_id="c1")
            result_c2 = await db.list_workers(tenant_id="c2", subject_id="c2")

        assert result_c1 == ["a"]
        assert result_c2 == ["b"]


# ---------------------------------------------------------------------------
# Delete worker
# ---------------------------------------------------------------------------


class TestDeleteWorker:
    @pytest.mark.asyncio
    async def test_delete_existing(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="doomed",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            result = await db.delete_worker(tenant_id="c1", subject_id="c1", worker_name="doomed")
            after = await db.get_worker(tenant_id="c1", subject_id="c1", worker_name="doomed")

        assert result is True
        assert after is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            result = await db.delete_worker(tenant_id="c1", subject_id="c1", worker_name="nope")

        assert result is False


# ---------------------------------------------------------------------------
# Delete all workers
# ---------------------------------------------------------------------------


class TestDeleteAllWorkers:
    @pytest.mark.asyncio
    async def test_deletes_all_for_tenant(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="a",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="b",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-2",
            )
            await db.register_worker(
                tenant_id="c2",
                subject_id="c2",
                worker_name="c",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-3",
            )

            count = await db.delete_all_workers(tenant_id="c1", subject_id="c1")
            remaining_c1 = await db.list_workers(tenant_id="c1", subject_id="c1")
            remaining_c2 = await db.list_workers(tenant_id="c2", subject_id="c2")

        assert count == 2
        assert remaining_c1 == []
        assert remaining_c2 == ["c"]


# ---------------------------------------------------------------------------
# Get all workers (full details)
# ---------------------------------------------------------------------------


class TestGetAllWorkers:
    @pytest.mark.asyncio
    async def test_returns_full_details(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                db_path="/data",
                is_agent=True,
                config={"model": "claude"},
                worker_id="wid-1",
            )

            result = await db.get_all_workers(tenant_id="c1", subject_id="c1")

        assert len(result) == 1
        w = result[0]
        assert w["worker_name"] == "w1"
        assert w["is_agent"] is True
        assert w["config"] == {"model": "claude"}


# ---------------------------------------------------------------------------
# Update worker status
# ---------------------------------------------------------------------------


class TestUpdateWorkerStatus:
    @pytest.mark.asyncio
    async def test_update_status_only(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            result = await db.update_worker_status(
                tenant_id="c1", subject_id="c1", worker_name="w1", status="active"
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_with_operational_status(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            await db.register_worker(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                db_path="/tmp",
                is_agent=False,
                config={},
                worker_id="w-1",
            )
            result = await db.update_worker_status(
                tenant_id="c1",
                subject_id="c1",
                worker_name="w1",
                status="active",
                operational_status="online",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_false(self):
        mem = InMemoryDB()
        db = WorkerRegistryDB()

        with patch_get_pg_connection("src.db.worker_registry_db", mem):
            result = await db.update_worker_status(
                tenant_id="c1", subject_id="c1", worker_name="nope", status="active"
            )

        assert result is False
