"""API endpoint integration tests with mock-psycopg connections.

Tests the full API stack: endpoint -> service -> DB class -> mock connection.
Uses the same mock-connection pattern as test_graph_state_db.py and
test_worker_registry_db_unit.py instead of mocking DB classes directly.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Environment must be set before importing app
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from fastapi.testclient import TestClient

from src.api.worker_management_api import app
from src.auth import get_identity_proven, get_tenant_id

TENANT = "test-tenant-abc"
GRAPH_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GRAPH_ID_2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
NODE_ID_1 = "11111111-1111-1111-1111-111111111111"
NODE_ID_2 = "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRow(dict):
    """Dict subclass that also supports attribute-style access for psycopg rows."""

    def __getitem__(self, key):
        return super().__getitem__(key)


def _make_conn(
    *,
    execute_return: str = "DELETE 0",
    fetchrow_return: dict | None = None,
    fetch_return: list | None = None,
    fetchval_return=None,
):
    """Create a mock psycopg connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.fetchval = AsyncMock(return_value=fetchval_return)
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return conn


def _mock_pg_factory(conn):
    """Return a callable that produces a fresh async context manager each time."""

    @asynccontextmanager
    async def _ctx():
        yield conn

    def factory():
        return _ctx()

    return factory


def _patch_graph_db(conn):
    """Patch get_pg_connection in graph_state_db module."""
    return patch("src.db.graph_state_db.get_pg_connection", _mock_pg_factory(conn))


def _patch_state_db(conn):
    """Patch get_pg_connection in workers_state_db module."""
    return patch("src.db.workers_state_db.get_pg_connection", _mock_pg_factory(conn))


def _patch_registry_db(conn):
    """Patch get_pg_connection in worker_registry_db module."""
    return patch("src.db.worker_registry_db.get_pg_connection", _mock_pg_factory(conn))


async def _fake_tenant_id(request=None):
    return TENANT


def _now():
    return datetime.now(UTC).replace(tzinfo=None)


_DEFAULT_NODE = {
    "id": NODE_ID_1,
    "name": "node1",
    "node_type": "worker",
    "worker_name": "test_worker",
    "function_name": None,
    "task_template": None,
    "dependencies": [],
    "data": {},
    "timeout": 300,
    "retry_count": 0,
    "framework_config": {},
}


def _graph_row(graph_id=GRAPH_ID, name="test-graph", nodes=None, edges=None):
    """Build a FakeRow for a graph SELECT.

    Pass nodes=[] explicitly for an empty-nodes graph.
    """
    now = _now()
    if nodes is None:
        nodes = [_DEFAULT_NODE]
    if edges is None:
        edges = []
    return FakeRow(
        {
            "id": graph_id,
            "tenant_id": TENANT,
            "name": name,
            "description": "",
            "nodes": nodes,
            "edges": edges,
            "created_at": now,
            "updated_at": now,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient with auth dependency overridden and lifespan DB init skipped."""
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    # ENABLE_AUTH=false installs no middleware, so `get_identity_proven`
    # falls back to its fail-closed default of False. This suite simulates
    # an ordinary authenticated caller (it exercises graph + worker-state
    # CRUD, not the tokenless hole), so state that explicitly rather than
    # let TestWorkerStatesPost collide with POST /workers_states' new
    # identity_proven gate (src/api/worker_states.py::save_workers_state).
    app.dependency_overrides[get_identity_proven] = lambda: True
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


# ===========================================================================
# Health endpoints
# ===========================================================================


class TestHealthEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "Worker Management API" in data["message"]

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "worker-management-api"

    def test_health_live(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_ready_db_ok(self, client):
        """Readiness returns 200 when DB is reachable."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        @asynccontextmanager
        async def fake_connection():
            yield mock_conn

        mock_pool.connection = fake_connection
        with patch("src.db.db_config.get_pg_pool", return_value=mock_pool):
            resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["database"] is True

    def test_health_ready_db_down(self, client):
        """Readiness returns 503 when DB pool is None."""
        with patch("src.db.db_config.get_pg_pool", return_value=None):
            resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not ready"
        assert data["checks"]["database"] is False


# ===========================================================================
# Graph Management API  (src/api/graph_management_api.py)
# ===========================================================================


class TestCreateGraph:
    def test_create_graph_201(self, client):
        """POST /graphs/ creates a graph and returns 201 with data."""
        conn = _make_conn()
        # create_graph -> execute (INSERT)
        # add_graph_state -> fetchrow RETURNING id
        # get_graph -> fetchrow (SELECT)
        graph_select_row = _graph_row()

        call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call: add_graph_state RETURNING id
            if call_count == 1:
                return FakeRow({"id": 1})
            # Second call: get_graph SELECT
            return graph_select_row

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        with _patch_graph_db(conn):
            resp = client.post(
                "/graphs/",
                json={
                    "name": "test-graph",
                    "nodes": [
                        {
                            "id": NODE_ID_1,
                            "name": "node1",
                            "node_type": "worker",
                            "worker_name": "test_worker",
                        }
                    ],
                    "edges": [],
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-graph"
        assert len(data["nodes"]) == 1

    def test_create_graph_validation_error(self, client):
        """POST /graphs/ with missing name gives 422."""
        resp = client.post("/graphs/", json={"nodes": [], "edges": []})
        assert resp.status_code == 422


class TestListGraphs:
    def test_list_graphs_empty(self, client):
        """GET /graphs/ returns empty list."""
        conn = _make_conn(fetch_return=[])
        with _patch_graph_db(conn):
            resp = client.get("/graphs/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_graphs_with_data(self, client):
        """GET /graphs/ returns graph summaries."""
        conn = _make_conn()
        # list_graphs -> fetch returns summary rows
        list_rows = [
            FakeRow({"id": GRAPH_ID, "name": "graph1"}),
        ]
        graph_detail = _graph_row(name="graph1")

        fetch_call_count = 0

        async def fetch_side_effect(*args, **kwargs):
            nonlocal fetch_call_count
            fetch_call_count += 1
            if fetch_call_count == 1:
                return list_rows
            return []

        conn.fetch = AsyncMock(side_effect=fetch_side_effect)
        conn.fetchrow = AsyncMock(return_value=graph_detail)

        with _patch_graph_db(conn):
            resp = client.get("/graphs/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "graph1"
        assert "id" in data[0]

    def test_list_graphs_include_details(self, client):
        """GET /graphs/?include_details=true returns full graph objects."""
        conn = _make_conn()
        list_rows = [FakeRow({"id": GRAPH_ID, "name": "graph1"})]
        graph_detail = _graph_row(name="graph1")

        fetch_call_count = 0

        async def fetch_side_effect(*args, **kwargs):
            nonlocal fetch_call_count
            fetch_call_count += 1
            if fetch_call_count == 1:
                return list_rows
            return []

        conn.fetch = AsyncMock(side_effect=fetch_side_effect)
        conn.fetchrow = AsyncMock(return_value=graph_detail)

        with _patch_graph_db(conn):
            resp = client.get("/graphs/?include_details=true")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert "nodes" in data[0]

    def test_list_graphs_pagination(self, client):
        """GET /graphs/?limit=1&offset=0 applies pagination."""
        conn = _make_conn()
        list_rows = [
            FakeRow({"id": GRAPH_ID, "name": "g1"}),
            FakeRow({"id": GRAPH_ID_2, "name": "g2"}),
        ]
        g1_detail = _graph_row(name="g1")
        g2_detail = _graph_row(graph_id=GRAPH_ID_2, name="g2")

        fetch_call_count = 0

        async def fetch_side_effect(*args, **kwargs):
            nonlocal fetch_call_count
            fetch_call_count += 1
            if fetch_call_count == 1:
                return list_rows
            return []

        fetchrow_call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal fetchrow_call_count
            fetchrow_call_count += 1
            if fetchrow_call_count == 1:
                return g1_detail
            return g2_detail

        conn.fetch = AsyncMock(side_effect=fetch_side_effect)
        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        with _patch_graph_db(conn):
            resp = client.get("/graphs/?limit=1&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "g1"


class TestGetGraph:
    def test_get_graph_found(self, client):
        """GET /graphs/{id} returns graph when it exists."""
        conn = _make_conn(fetchrow_return=_graph_row())
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-graph"
        assert len(data["nodes"]) == 1

    def test_get_graph_not_found(self, client):
        """GET /graphs/{id} returns 404 when graph does not exist."""
        conn = _make_conn(fetchrow_return=None)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}")
        assert resp.status_code == 404

    def test_get_graph_bad_uuid(self, client):
        """GET /graphs/{bad} returns 422 for invalid UUID."""
        resp = client.get("/graphs/not-a-uuid")
        assert resp.status_code == 422


class TestUpdateGraph:
    def test_update_graph_success(self, client):
        """PATCH /graphs/{id} updates and returns updated graph."""
        conn = _make_conn()
        existing = FakeRow(
            {"name": "old-name", "description": "", "nodes": [], "edges": [], "variables": {}}
        )
        updated_detail = _graph_row(name="new-name")

        fetchrow_call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal fetchrow_call_count
            fetchrow_call_count += 1
            # 1: update_graph -> SELECT existing
            if fetchrow_call_count == 1:
                return existing
            # 2: add_graph_state -> RETURNING id
            if fetchrow_call_count == 2:
                return FakeRow({"id": 1})
            # 3: get_graph -> SELECT
            return updated_detail

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
        conn.execute = AsyncMock(return_value="UPDATE 1")

        with _patch_graph_db(conn):
            resp = client.patch(f"/graphs/{GRAPH_ID}", json={"name": "new-name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"

    def test_update_graph_not_found(self, client):
        """PATCH /graphs/{id} returns 404 when not found."""
        conn = _make_conn(fetchrow_return=None)
        with _patch_graph_db(conn):
            resp = client.patch(f"/graphs/{GRAPH_ID}", json={"name": "x"})
        assert resp.status_code == 404


class TestDeleteGraph:
    def test_delete_graph_success(self, client):
        """DELETE /graphs/{id} returns 204 on success."""
        conn = _make_conn()
        graph_detail = _graph_row()

        fetchrow_call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal fetchrow_call_count
            fetchrow_call_count += 1
            # 1: delete_graph_service -> get_graph check
            if fetchrow_call_count == 1:
                return graph_detail
            # 2: add_graph_state RETURNING id
            if fetchrow_call_count == 2:
                return FakeRow({"id": 1})
            return None

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
        conn.execute = AsyncMock(return_value="DELETE 1")

        with _patch_graph_db(conn):
            resp = client.delete(f"/graphs/{GRAPH_ID}")
        assert resp.status_code == 204

    def test_delete_graph_not_found(self, client):
        """DELETE /graphs/{id} returns 404 when not found."""
        conn = _make_conn(fetchrow_return=None)
        with _patch_graph_db(conn):
            resp = client.delete(f"/graphs/{GRAPH_ID}")
        assert resp.status_code == 404

    def test_delete_graph_bad_uuid(self, client):
        """DELETE /graphs/{bad} returns 422 for invalid UUID."""
        resp = client.delete("/graphs/xyz")
        assert resp.status_code == 422


class TestDeleteAllGraphs:
    def test_delete_all_graphs(self, client):
        """DELETE /graphs/ returns count of deleted graphs."""
        conn = _make_conn()
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="DELETE 3")

        with _patch_graph_db(conn):
            resp = client.delete("/graphs/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_count"] == 3


class TestDeleteAllGraphStates:
    def test_delete_all_graph_states(self, client):
        """DELETE /graphs/states/all returns count of deleted states."""
        conn = _make_conn(execute_return="DELETE 5")
        with _patch_graph_db(conn):
            resp = client.delete("/graphs/states/all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["states_deleted"] == 5
        assert "Deleted 5" in data["message"]


class TestGraphStates:
    def test_get_graph_states(self, client):
        """GET /graphs/{id}/states returns state records."""
        now = _now()
        conn = _make_conn(
            fetch_return=[
                FakeRow(
                    {
                        "id": 1,
                        "graph_id": GRAPH_ID,
                        "created_at": now,
                        "state_data": {"event": "created"},
                    }
                )
            ]
        )
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["graph_id"] == GRAPH_ID

    def test_delete_graph_states(self, client):
        """DELETE /graphs/{id}/states returns count."""
        conn = _make_conn(execute_return="DELETE 2")
        with _patch_graph_db(conn):
            resp = client.delete(f"/graphs/{GRAPH_ID}/states")
        assert resp.status_code == 200
        data = resp.json()
        assert data["states_deleted"] == 2
        assert data["graph_id"] == GRAPH_ID


class TestGraphStructure:
    def test_structure_single_node(self, client):
        """GET /graphs/{id}/structure returns single_node type."""
        conn = _make_conn(fetchrow_return=_graph_row())
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/structure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["structure_type"] == "single_node"
        assert data["name"] == "test-graph"
        assert len(data["nodes"]) == 1

    def test_structure_empty_graph(self, client):
        """GET /graphs/{id}/structure with empty nodes returns empty type."""
        row = _graph_row(nodes=[])
        conn = _make_conn(fetchrow_return=row)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/structure")
        # Will get 500 because GraphResponse requires at least concept of nodes,
        # but empty nodes list should trigger structure_type = "empty"
        # Actually - GraphCreateRequest validates nodes, but GraphResponse doesn't
        # The service layer builds GraphResponse from DB data, so empty is fine
        assert resp.status_code == 200
        data = resp.json()
        assert data["structure_type"] == "empty"

    def test_structure_linear(self, client):
        """GET /graphs/{id}/structure with linear chain returns linear type."""
        nodes = [
            {
                "id": NODE_ID_1,
                "name": "n1",
                "node_type": "worker",
                "worker_name": "w1",
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
            {
                "id": NODE_ID_2,
                "name": "n2",
                "node_type": "worker",
                "worker_name": "w2",
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
        ]
        edges = [{"source": NODE_ID_1, "target": NODE_ID_2, "condition": None, "metadata": None}]
        row = _graph_row(nodes=nodes, edges=edges)
        conn = _make_conn(fetchrow_return=row)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/structure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["structure_type"] == "linear"
        assert len(data["edges"]) == 1

    def test_structure_not_found(self, client):
        """GET /graphs/{id}/structure returns 404 when graph missing."""
        conn = _make_conn(fetchrow_return=None)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/structure")
        assert resp.status_code == 404


class TestGraphValidation:
    def test_validate_valid_graph(self, client):
        """GET /graphs/{id}/validate returns valid=True for simple graph."""
        nodes = [
            {
                "id": NODE_ID_1,
                "name": "n1",
                "node_type": "worker",
                "worker_name": "w1",
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
        ]
        row = _graph_row(nodes=nodes, edges=[])
        conn = _make_conn(fetchrow_return=row)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["summary"]["total_nodes"] == 1

    def test_validate_missing_worker_causes_500(self, client):
        """GET /graphs/{id}/validate returns 500 when DB has invalid node data.

        NodeSchema model_validator rejects worker nodes without worker_name,
        so GraphResponse construction fails, resulting in 500.
        """
        nodes = [
            {
                "id": NODE_ID_1,
                "name": "bad_worker",
                "node_type": "worker",
                "worker_name": None,
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
        ]
        row = _graph_row(nodes=nodes)
        conn = _make_conn(fetchrow_return=row)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/validate")
        assert resp.status_code == 500

    def test_validate_orphan_nodes(self, client):
        """GET /graphs/{id}/validate detects orphaned nodes."""
        nodes = [
            {
                "id": NODE_ID_1,
                "name": "n1",
                "node_type": "worker",
                "worker_name": "w1",
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
            {
                "id": NODE_ID_2,
                "name": "n2",
                "node_type": "worker",
                "worker_name": "w2",
                "function_name": None,
                "task_template": None,
                "dependencies": [],
                "data": {},
                "timeout": 300,
                "retry_count": 0,
                "framework_config": {},
            },
        ]
        # No edges -> both nodes are orphans
        row = _graph_row(nodes=nodes, edges=[])
        conn = _make_conn(fetchrow_return=row)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/validate")
        assert resp.status_code == 200
        data = resp.json()
        # Should have orphan warnings
        orphan_warnings = [w for w in data["warnings"] if w["type"] == "orphaned_nodes"]
        assert len(orphan_warnings) == 2

    def test_validate_not_found(self, client):
        """GET /graphs/{id}/validate returns 404 when graph missing."""
        conn = _make_conn(fetchrow_return=None)
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/validate")
        assert resp.status_code == 404

    def test_validate_bad_uuid(self, client):
        """GET /graphs/{bad}/validate returns 422."""
        resp = client.get("/graphs/not-a-uuid/validate")
        assert resp.status_code == 422


# ===========================================================================
# Worker States API  (src/api/worker_states.py)
# ===========================================================================


class TestWorkerStatesGet:
    def test_get_empty(self, client):
        """GET /workers_states returns empty list."""
        conn = _make_conn(fetch_return=[])
        with _patch_state_db(conn):
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_with_data(self, client):
        """GET /workers_states returns state records."""
        now = _now()
        conn = _make_conn(
            fetch_return=[
                FakeRow(
                    {
                        "id": 1,
                        "worker_id": "w1",
                        "worker_name": "Worker1",
                        "tenant_id": TENANT,
                        "is_agent": False,
                        "state_type": "idle",
                        "worker_state": {"key": "val"},
                        "timestamp": now,
                        "metadata": {},
                    }
                )
            ]
        )
        with _patch_state_db(conn):
            resp = client.get("/workers_states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["worker_name"] == "Worker1"
        assert data[0]["state_type"] == "idle"
        assert data[0]["worker_state"] == {"key": "val"}

    def test_get_pagination_is_pushed_into_sql(self, client):
        """GET /workers_states?limit=1&offset=1 pages IN SQL, not by slicing in Python.

        The previous version of this test asserted the opposite contract: the
        fake connection ignores the query's bind parameters and always returned
        BOTH rows, so ``len(data) == 1`` could only ever have been produced by
        the endpoint slicing a full-history read in Python. That slice WAS the
        defect -- ``worker_states`` is append-only and ~1.3GB in production, so
        the unbounded ``SELECT`` ran regardless of the page size the caller
        asked for. The assertion pinned it as correct.

        So the fake now HONOURS ``LIMIT``/``OFFSET`` the way postgres would,
        and the test asserts both halves of the real contract: the caller's
        limit/offset reach the database as bind parameters, and the endpoint
        returns that page verbatim.
        """
        now = _now()
        rows = [
            FakeRow(
                {
                    "id": 1,
                    "worker_id": "w1",
                    "worker_name": "W1",
                    "tenant_id": TENANT,
                    "is_agent": False,
                    "state_type": "idle",
                    "worker_state": {},
                    "timestamp": now,
                    "metadata": {},
                }
            ),
            FakeRow(
                {
                    "id": 2,
                    "worker_id": "w2",
                    "worker_name": "W2",
                    "tenant_id": TENANT,
                    "is_agent": False,
                    "state_type": "completed",
                    "worker_state": {},
                    "timestamp": now,
                    "metadata": {},
                }
            ),
        ]
        seen: dict[str, object] = {}

        async def _fetch(query, *args):
            seen["query"] = query
            seen["args"] = args
            limit, offset = args[2], args[3]
            return rows[offset : offset + limit]

        conn = _make_conn()
        conn.fetch = AsyncMock(side_effect=_fetch)
        with _patch_state_db(conn):
            resp = client.get("/workers_states?limit=1&offset=1")

        assert resp.status_code == 200
        assert "LIMIT $3 OFFSET $4" in str(seen["query"])
        assert seen["args"][2:] == (1, 1)
        data = resp.json()
        assert len(data) == 1
        assert data[0]["worker_name"] == "W2"


class TestWorkerStatesPost:
    def test_save_single_state(self, client):
        """POST /workers_states saves a single state."""
        conn = _make_conn()

        call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeRow({"id": call_count})

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        with _patch_state_db(conn):
            resp = client.post(
                "/workers_states",
                json={
                    "worker_states": [
                        {
                            "worker_id": "w1",
                            "worker_name": "Worker1",
                            "state_type": "processing",
                            "worker_state": {"step": 1},
                        }
                    ]
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["message"] == "Workers' states saved successfully"
        assert len(data["results"]) == 1
        assert data["results"][0]["worker_id"] == "w1"
        assert data["results"][0]["id"] == 1

    def test_save_batch(self, client):
        """POST /workers_states saves multiple states."""
        conn = _make_conn()

        call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeRow({"id": call_count})

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        with _patch_state_db(conn):
            resp = client.post(
                "/workers_states",
                json={
                    "worker_states": [
                        {
                            "worker_id": "w1",
                            "worker_name": "W1",
                            "state_type": "idle",
                            "worker_state": {"x": 1},
                        },
                        {
                            "worker_id": "w2",
                            "worker_name": "W2",
                            "state_type": "completed",
                            "worker_state": {"x": 2},
                        },
                    ]
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["results"]) == 2

    def test_save_validation_error_missing_worker_id(self, client):
        """POST /workers_states rejects missing worker_id."""
        resp = client.post(
            "/workers_states",
            json={
                "worker_states": [
                    {
                        "worker_name": "W1",
                        "state_type": "idle",
                        "worker_state": {"x": 1},
                    }
                ]
            },
        )
        assert resp.status_code == 422

    def test_save_validation_error_bad_state_type(self, client):
        """POST /workers_states rejects invalid state_type."""
        resp = client.post(
            "/workers_states",
            json={
                "worker_states": [
                    {
                        "worker_id": "w1",
                        "worker_name": "W1",
                        "state_type": "INVALID",
                        "worker_state": {"x": 1},
                    }
                ]
            },
        )
        assert resp.status_code == 422

    def test_save_validation_error_empty_list(self, client):
        """POST /workers_states rejects empty worker_states list."""
        resp = client.post("/workers_states", json={"worker_states": []})
        assert resp.status_code == 422


# ===========================================================================
# Graph Execution API  (src/api/graph_execution_api.py)
# ===========================================================================


class TestGraphExecution:
    def test_execute_graph_bad_uuid(self, client):
        """POST /execution/graphs/{bad}/execute returns 422."""
        resp = client.post(
            "/execution/graphs/not-a-uuid/execute",
            json={"variables": {}},
        )
        assert resp.status_code == 422

    def test_execute_graph_not_found(self, client):
        """POST /execution/graphs/{id}/execute returns 404 when graph missing."""
        # Mock the execution service to raise "not found"
        mock_service = MagicMock()
        mock_service.execute_graph = AsyncMock(side_effect=ValueError("Graph not found"))
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            return_value=mock_service,
        ):
            resp = client.post(
                f"/execution/graphs/{GRAPH_ID}/execute",
                json={"variables": {}},
            )
        assert resp.status_code == 404

    def test_execution_health(self, client):
        """GET /execution/health returns healthy status."""
        mock_service = MagicMock()
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            return_value=mock_service,
        ):
            resp = client.get("/execution/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["execution_engine"] == "pydantic_ai"

    def test_session_status_not_found(self, client):
        """GET /execution/sessions/{id}/status returns 404 when session missing."""
        mock_service = MagicMock()
        mock_service.get_session_status = AsyncMock(return_value=None)
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            return_value=mock_service,
        ):
            resp = client.get(f"/execution/sessions/{GRAPH_ID}/status")
        assert resp.status_code == 404

    def test_stepwise_execute_bad_uuid(self, client):
        """POST /execution/graphs/{bad}/execute/stepwise returns 422."""
        resp = client.post(
            "/execution/graphs/not-a-uuid/execute/stepwise",
            json={"variables": {}},
        )
        assert resp.status_code == 422

    def test_stepwise_execute_not_found(self, client):
        """POST /execution/graphs/{id}/execute/stepwise returns 404 when graph missing."""
        mock_service = MagicMock()
        mock_service.execute_graph_stepwise = AsyncMock(side_effect=ValueError("Graph not found"))
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            return_value=mock_service,
        ):
            resp = client.post(
                f"/execution/graphs/{GRAPH_ID}/execute/stepwise",
                json={"variables": {}},
            )
        assert resp.status_code == 404


class TestExecutionSessionsList:
    """Tests for GET /execution/sessions/list (eval-set sampler endpoint)."""

    def _patch_db_class(self, sessions: list[dict]):
        """Return a context manager replacing GraphExecutionSessionDB.list_sessions."""
        instance = MagicMock()
        instance.list_sessions = AsyncMock(return_value=sessions)
        return patch(
            "src.api.graph_execution_api.GraphExecutionSessionDB",
            return_value=instance,
        ), instance

    def test_list_sessions_basic(self, client):
        """200 + UUID/datetime fields are stringified for stable JSON."""
        from uuid import UUID

        sample = {
            "session_id": UUID("11111111-1111-1111-1111-111111111111"),
            "graph_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "request_payload": {"messages": [{"role": "user", "content": "hi"}]},
            "final_response": {"choices": [{"message": {"content": "hello"}}]},
            "correlation_id": "c-1",
            "validation_passed": True,
            "status": "completed",
            "created_at": datetime(2026, 4, 30, 12, 0, 0),
            "completed_at": datetime(2026, 4, 30, 12, 0, 30),
        }
        ctx, _ = self._patch_db_class([sample])
        with ctx:
            resp = client.get("/execution/sessions/list")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["session_id"] == "11111111-1111-1111-1111-111111111111"
        assert body[0]["created_at"] == "2026-04-30T12:00:00"
        assert body[0]["request_payload"] == sample["request_payload"]
        assert body[0]["validation_passed"] is True

    def test_list_sessions_filters_forwarded(self, client):
        """All query params reach the DB layer with correct types."""
        ctx, instance = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                params={
                    "graph_id": GRAPH_ID,
                    "status": "completed",
                    "validation_passed": "true",
                    "since": "2026-04-01T00:00:00Z",
                    "until": "2026-05-01T00:00:00Z",
                    "limit": 50,
                },
            )
        assert resp.status_code == 200
        instance.list_sessions.assert_awaited_once()
        kwargs = instance.list_sessions.await_args.kwargs
        assert kwargs["graph_id"] == GRAPH_ID
        assert kwargs["status"] == "completed"
        assert kwargs["validation_passed"] is True
        assert kwargs["limit"] == 50
        assert kwargs["since"].year == 2026
        assert kwargs["since"].month == 4
        assert kwargs["until"].month == 5

    def test_list_sessions_bad_graph_id(self, client):
        """Non-UUID graph_id returns 422."""
        ctx, _ = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                params={"graph_id": "not-a-uuid"},
            )
        assert resp.status_code == 422

    def test_list_sessions_invalid_window(self, client):
        """`since` >= `until` returns 400."""
        ctx, _ = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                params={
                    "since": "2026-05-01T00:00:00Z",
                    "until": "2026-04-01T00:00:00Z",
                },
            )
        assert resp.status_code == 400

    def test_list_sessions_limit_too_high(self, client):
        """limit > 500 fails query validation (422)."""
        resp = client.get("/execution/sessions/list", params={"limit": 1000})
        assert resp.status_code == 422

    def test_list_sessions_correlation_id_filter_forwarded(self, client):
        """L2: correlation_id query param reaches the DB layer."""
        ctx, instance = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                params={"correlation_id": "trace-abc"},
            )
        assert resp.status_code == 200
        assert instance.list_sessions.await_args.kwargs["correlation_id"] == "trace-abc"

    def test_list_sessions_offset_forwarded(self, client):
        """M2: offset query param reaches the DB layer for paging."""
        ctx, instance = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                params={"limit": 50, "offset": 50},
            )
        assert resp.status_code == 200
        assert instance.list_sessions.await_args.kwargs["offset"] == 50
        assert instance.list_sessions.await_args.kwargs["limit"] == 50

    def test_list_sessions_negative_offset_rejected(self, client):
        """offset < 0 fails query validation (422)."""
        resp = client.get("/execution/sessions/list", params={"offset": -1})
        assert resp.status_code == 422

    def test_list_sessions_isolates_tenants(self, client):
        """T1: tenant_id from auth dep reaches the DB layer verbatim.

        The DB layer's scope_clause() (visibility.py) constrains rows to
        ``tenant_id = $1`` on every call, so this test pins the contract
        that whatever ``get_tenant_id`` resolves the inbound caller as is
        what the DB filter receives — proving callers cannot influence
        the tenant scope through query params.
        """
        ctx, instance = self._patch_db_class([])
        with ctx:
            resp = client.get(
                "/execution/sessions/list",
                # Try (vainly) to inject a tenant via query string — the
                # endpoint dependency does not honour it; tenant comes
                # only from get_tenant_id.
                params={"tenant_id": "00000000-0000-0000-0000-000000000099"},
            )
        assert resp.status_code == 200
        kwargs = instance.list_sessions.await_args.kwargs
        # Whatever conftest's auth dep injected — definitely NOT the
        # query-string value.
        assert kwargs["tenant_id"] != "00000000-0000-0000-0000-000000000099"
        # subject_id is also injected from auth, never from input.
        assert "subject_id" in kwargs


# ===========================================================================
# Edge cases / additional coverage
# ===========================================================================


class TestEdgeCases:
    def test_get_graph_states_empty(self, client):
        """GET /graphs/{id}/states with no states returns empty list."""
        conn = _make_conn(fetch_return=[])
        with _patch_graph_db(conn):
            resp = client.get(f"/graphs/{GRAPH_ID}/states?limit=5")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_delete_all_graphs_states_zero(self, client):
        """DELETE /graphs/states/all with no states returns 0 count."""
        conn = _make_conn(execute_return="DELETE 0")
        with _patch_graph_db(conn):
            resp = client.delete("/graphs/states/all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["states_deleted"] == 0

    def test_update_graph_bad_uuid(self, client):
        """PATCH /graphs/{bad} returns 422."""
        resp = client.patch("/graphs/xyz", json={"name": "new"})
        assert resp.status_code == 422

    def test_structure_bad_uuid(self, client):
        """GET /graphs/{bad}/structure returns 422."""
        resp = client.get("/graphs/invalid/structure")
        assert resp.status_code == 422

    def test_states_bad_uuid(self, client):
        """GET /graphs/{bad}/states returns 422."""
        resp = client.get("/graphs/bad-id/states")
        assert resp.status_code == 422

    def test_delete_states_bad_uuid(self, client):
        """DELETE /graphs/{bad}/states returns 422."""
        resp = client.delete("/graphs/not-uuid/states")
        assert resp.status_code == 422

    def test_create_graph_with_description(self, client):
        """POST /graphs/ with description."""
        conn = _make_conn()
        graph_detail = _graph_row(name="with-desc")
        call_count = 0

        async def fetchrow_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeRow({"id": 1})
            return graph_detail

        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        with _patch_graph_db(conn):
            resp = client.post(
                "/graphs/",
                json={
                    "name": "with-desc",
                    "description": "A test description",
                    "nodes": [
                        {
                            "id": NODE_ID_1,
                            "name": "n1",
                            "node_type": "worker",
                            "worker_name": "w1",
                        }
                    ],
                    "edges": [],
                },
            )
        assert resp.status_code == 201
        assert resp.json()["name"] == "with-desc"

    def test_execution_health_service_error(self, client):
        """GET /execution/health returns 500 when service init fails."""
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            side_effect=RuntimeError("init failed"),
        ):
            resp = client.get("/execution/health")
        assert resp.status_code == 500

    def test_session_status_service_exception(self, client):
        """GET /execution/sessions/{id}/status returns 500 on unexpected error."""
        mock_service = MagicMock()
        mock_service.get_session_status = AsyncMock(side_effect=RuntimeError("unexpected"))
        with patch(
            "src.api.graph_execution_api.get_execution_service",
            return_value=mock_service,
        ):
            resp = client.get(f"/execution/sessions/{GRAPH_ID}/status")
        assert resp.status_code == 500
