"""Unit tests for worker/graphs/graph_operations.py.

Each test mocks GraphStateDatabase at the module level and verifies
that return values flow through correctly, not just that mocks were called.
"""

from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from fastapi import HTTPException

from src.api.tenant_constants import default_tenant_uuid
from src.worker.graphs.graph_operations import (
    check_database_health,
    delete_all_graphs_service,
    delete_graph_service,
    get_graph_service,
    list_graphs_service,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CREATOR = "creator-1"
GRAPH_ID = "550e8400-e29b-41d4-a716-446655440000"
GRAPH_ID_2 = "660e8400-e29b-41d4-a716-446655440001"

GRAPH_DATA = {
    "id": GRAPH_ID,
    "tenant_id": CREATOR,
    "name": "Test Graph",
    "description": "A test graph",
    "nodes": [],
    "edges": [],
    "variables": {},
    "created_at": "2026-01-01T00:00:00",
    "updated_at": "2026-01-01T00:00:00",
}

GRAPH_DATA_2 = {
    "id": GRAPH_ID_2,
    "tenant_id": CREATOR,
    "name": "Second Graph",
    "description": "",
    "nodes": [],
    "edges": [],
    "variables": {},
    "created_at": "2026-01-02T00:00:00",
    "updated_at": "2026-01-02T00:00:00",
}


def _patch_db():
    """Context manager that patches GraphStateDatabase at the operations module."""
    return patch("src.worker.graphs.graph_operations.GraphStateDatabase")


# ===================================================================
# get_graph_service
# ===================================================================


class TestGetGraphService:
    @pytest.mark.asyncio
    async def test_found_returns_graph_response_with_correct_fields(self):
        with _patch_db() as MockDB:
            MockDB.return_value.get_graph = AsyncMock(return_value=GRAPH_DATA)
            result = await get_graph_service(CREATOR, CREATOR, GRAPH_ID)

        assert result.name == "Test Graph"
        assert result.description == "A test graph"
        assert str(result.id) == GRAPH_ID
        assert result.nodes == []
        assert result.edges == []
        assert result.variables == {}

    @pytest.mark.asyncio
    async def test_not_found_raises_404(self):
        with _patch_db() as MockDB:
            MockDB.return_value.get_graph = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_graph_service(CREATOR, CREATOR, GRAPH_ID)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_db_runtime_error_raises_500_with_original_message(self):
        with _patch_db() as MockDB:
            MockDB.return_value.get_graph = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )
            with pytest.raises(HTTPException) as exc_info:
                await get_graph_service(CREATOR, CREATOR, GRAPH_ID)

        assert exc_info.value.status_code == 500
        assert "connection refused" in exc_info.value.detail


# ===================================================================
# list_graphs_service
# ===================================================================


class TestListGraphsService:
    @pytest.mark.asyncio
    async def test_returns_list_of_graph_responses(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(
                return_value=[{"id": GRAPH_ID}, {"id": GRAPH_ID_2}]
            )
            # get_graph is called for each row
            MockDB.return_value.get_graph = AsyncMock(side_effect=[GRAPH_DATA, GRAPH_DATA_2])
            result = await list_graphs_service(CREATOR, CREATOR)

        assert len(result) == 2
        assert result[0].name == "Test Graph"
        assert result[1].name == "Second Graph"

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(return_value=[])
            result = await list_graphs_service(CREATOR, CREATOR)

        assert result == []
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_partial_failure_skips_bad_graphs(self):
        """If get_graph fails for one row, that row is skipped."""
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(
                return_value=[{"id": GRAPH_ID}, {"id": GRAPH_ID_2}]
            )
            MockDB.return_value.get_graph = AsyncMock(
                side_effect=[RuntimeError("corrupt"), GRAPH_DATA_2]
            )
            result = await list_graphs_service(CREATOR, CREATOR)

        # First graph failed to load, second succeeded
        assert len(result) == 1
        assert result[0].name == "Second Graph"

    @pytest.mark.asyncio
    async def test_db_error_raises_500(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(side_effect=RuntimeError("timeout"))
            with pytest.raises(HTTPException) as exc_info:
                await list_graphs_service(CREATOR, CREATOR)

        assert exc_info.value.status_code == 500


# ===================================================================
# delete_graph_service
# ===================================================================


class TestDeleteGraphService:
    @pytest.mark.asyncio
    async def test_success_deletes_graph_and_states(self):
        with _patch_db() as MockDB:
            db = MockDB.return_value
            db.get_graph = AsyncMock(return_value=GRAPH_DATA)
            db.add_graph_state = AsyncMock()
            db.delete_graph_states = AsyncMock(return_value=3)
            db.delete_graph = AsyncMock(return_value=True)

            await delete_graph_service(CREATOR, CREATOR, GRAPH_ID)

        # Verify delete_graph_states was called for cleanup
        db.delete_graph_states.assert_called_once_with(
            tenant_id=CREATOR, subject_id=CREATOR, graph_id=GRAPH_ID
        )
        # Verify the graph itself was deleted
        db.delete_graph.assert_called_once_with(
            tenant_id=CREATOR, subject_id=CREATOR, graph_id=GRAPH_ID
        )

    @pytest.mark.asyncio
    async def test_not_found_raises_404(self):
        with _patch_db() as MockDB:
            MockDB.return_value.get_graph = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await delete_graph_service(CREATOR, CREATOR, GRAPH_ID)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_graph_returns_false_raises_500(self):
        """When db.delete_graph returns False (row disappeared), raise 500."""
        with _patch_db() as MockDB:
            db = MockDB.return_value
            db.get_graph = AsyncMock(return_value=GRAPH_DATA)
            db.add_graph_state = AsyncMock()
            db.delete_graph_states = AsyncMock(return_value=0)
            db.delete_graph = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await delete_graph_service(CREATOR, CREATOR, GRAPH_ID)

        assert exc_info.value.status_code == 500


# ===================================================================
# delete_all_graphs_service
# ===================================================================


class TestDeleteAllGraphsService:
    @pytest.mark.asyncio
    async def test_returns_correct_counts(self):
        with _patch_db() as MockDB:
            db = MockDB.return_value
            db.list_graphs = AsyncMock(return_value=[{"id": GRAPH_ID}, {"id": GRAPH_ID_2}])
            db.add_graph_state = AsyncMock()
            db.delete_all_graph_states = AsyncMock(return_value=7)
            db.delete_all_graphs = AsyncMock(return_value=2)

            result = await delete_all_graphs_service(CREATOR, CREATOR)

        assert result["deleted_count"] == 2
        assert result["states_deleted"] == 7

    @pytest.mark.asyncio
    async def test_zero_graphs_returns_zero_counts(self):
        with _patch_db() as MockDB:
            db = MockDB.return_value
            db.list_graphs = AsyncMock(return_value=[])
            db.add_graph_state = AsyncMock()
            db.delete_all_graph_states = AsyncMock(return_value=0)
            db.delete_all_graphs = AsyncMock(return_value=0)

            result = await delete_all_graphs_service(CREATOR, CREATOR)

        assert result["deleted_count"] == 0
        assert result["states_deleted"] == 0

    @pytest.mark.asyncio
    async def test_state_cleanup_failure_still_deletes_graphs(self):
        """If delete_all_graph_states fails, graphs should still be deleted."""
        with _patch_db() as MockDB:
            db = MockDB.return_value
            db.list_graphs = AsyncMock(return_value=[{"id": GRAPH_ID}])
            db.add_graph_state = AsyncMock()
            db.delete_all_graph_states = AsyncMock(side_effect=RuntimeError("state table locked"))
            db.delete_all_graphs = AsyncMock(return_value=1)

            result = await delete_all_graphs_service(CREATOR, CREATOR)

        assert result["deleted_count"] == 1
        # states_deleted should be 0 because the state cleanup failed
        assert result["states_deleted"] == 0

    @pytest.mark.asyncio
    async def test_db_error_raises_500(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(side_effect=RuntimeError("disk full"))
            with pytest.raises(HTTPException) as exc_info:
                await delete_all_graphs_service(CREATOR, CREATOR)

        assert exc_info.value.status_code == 500


# ===================================================================
# check_database_health
# ===================================================================


class TestCheckDatabaseHealth:
    @pytest.mark.asyncio
    async def test_healthy_returns_true(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(return_value=[])
            result = await check_database_health(CREATOR, CREATOR)

        assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false(self):
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(
                side_effect=psycopg.OperationalError("connection failed")
            )
            result = await check_database_health(CREATOR, CREATOR)

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_default_tenant_id(self):
        """check_database_health defaults to the deployment's default tenant.

        It used to default to the literal ``"test"``, which reached
        ``WHERE tenant_id = $1``. ``tenant_id`` is a ``uuid`` column since
        migration 024, so that default would now raise 22P02 and this *health
        check* would report the database unhealthy because of its own argument.
        """
        with _patch_db() as MockDB:
            MockDB.return_value.list_graphs = AsyncMock(return_value=[])
            result = await check_database_health()

        assert result is True
        MockDB.return_value.list_graphs.assert_called_once_with(
            tenant_id=default_tenant_uuid(), subject_id=default_tenant_uuid()
        )
