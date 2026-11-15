"""Tests for src.db.graph_state_db module.

Uses InMemoryDB to verify real data flow: insert data, read it back,
verify correctness.  No mock-asserts-mock-was-called theater.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from src.db.graph_state_db import GraphStateDatabase, _clean_item
from tests.db_test_helper import InMemoryDB, patch_get_pg_connection

# ---------------------------------------------------------------------------
# _clean_item
# ---------------------------------------------------------------------------


class TestCleanItem:
    def test_uuid_to_string(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        assert _clean_item(uid) == "12345678-1234-5678-1234-567812345678"

    def test_dict_recursion(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _clean_item({"key": uid, "nested": {"inner": uid}})
        assert result["key"] == str(uid)
        assert result["nested"]["inner"] == str(uid)

    def test_list_recursion(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _clean_item([uid, "hello", 42])
        assert result == [str(uid), "hello", 42]

    def test_tuple_recursion(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _clean_item((uid, "hello"))
        assert result == (str(uid), "hello")

    def test_primitive_passthrough(self):
        assert _clean_item("hello") == "hello"
        assert _clean_item(42) == 42
        assert _clean_item(None) is None

    def test_pydantic_model(self):
        from pydantic import BaseModel

        class Dummy(BaseModel):
            name: str
            uid: str

        m = Dummy(name="test", uid="abc")
        result = _clean_item(m)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["uid"] == "abc"

    def test_nested_pydantic_in_list(self):
        from pydantic import BaseModel

        class Node(BaseModel):
            id: str
            label: str

        nodes = [Node(id="n1", label="first"), Node(id="n2", label="second")]
        result = _clean_item(nodes)
        assert result == [{"id": "n1", "label": "first"}, {"id": "n2", "label": "second"}]


# ---------------------------------------------------------------------------
# GraphStateDatabase — create + get roundtrip
# ---------------------------------------------------------------------------


class TestCreateAndGetGraph:
    @pytest.mark.asyncio
    async def test_create_then_get_roundtrip(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a",
                subject_id="tenant-a",
                graph_id="g-1",
                name="My Graph",
                description="A test graph",
                nodes=[{"id": "n1", "type": "start"}],
                edges=[{"from": "n1", "to": "n2"}],
                variables={"timeout": 30},
            )

            result = await db.get_graph(tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1")

        assert result is not None
        assert result["id"] == "g-1"
        assert result["tenant_id"] == "tenant-a"
        assert result["name"] == "My Graph"
        assert result["description"] == "A test graph"
        assert result["nodes"] == [{"id": "n1", "type": "start"}]
        assert result["edges"] == [{"from": "n1", "to": "n2"}]
        assert result["variables"] == {"timeout": 30}
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a",
                subject_id="tenant-a",
                graph_id="g-2",
                name="Minimal",
            )

            result = await db.get_graph(tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-2")

        assert result is not None
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["variables"] == {}
        assert result["description"] == ""

    @pytest.mark.asyncio
    async def test_create_cleans_uuid_in_nodes(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()
        uid = UUID("12345678-1234-5678-1234-567812345678")

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-3",
                name="G",
                nodes=[{"id": uid, "name": "n1"}],
            )

            result = await db.get_graph(tenant_id="t1", subject_id="t1", graph_id="g-3")

        assert result is not None
        assert result["nodes"][0]["id"] == str(uid)

    @pytest.mark.asyncio
    async def test_get_nonexistent_graph_returns_none(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            result = await db.get_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="nonexistent"
            )

        assert result is None


# ---------------------------------------------------------------------------
# Multi-tenant isolation
# ---------------------------------------------------------------------------


class TestMultiTenantIsolation:
    @pytest.mark.asyncio
    async def test_get_graph_wrong_tenant_returns_none(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1", name="A's graph"
            )

            # tenant-b should not see tenant-a's graph
            result = await db.get_graph(tenant_id="tenant-b", subject_id="tenant-b", graph_id="g-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_graphs_only_own_tenant(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1", name="A1"
            )
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-2", name="A2"
            )
            await db.create_graph(
                tenant_id="tenant-b", subject_id="tenant-b", graph_id="g-3", name="B1"
            )

            result_a = await db.list_graphs(tenant_id="tenant-a", subject_id="tenant-a")
            result_b = await db.list_graphs(tenant_id="tenant-b", subject_id="tenant-b")

        assert len(result_a) == 2
        assert len(result_b) == 1
        assert result_b[0]["name"] == "B1"

    @pytest.mark.asyncio
    async def test_delete_all_graphs_only_own_tenant(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1", name="A1"
            )
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-2", name="A2"
            )
            await db.create_graph(
                tenant_id="tenant-b", subject_id="tenant-b", graph_id="g-3", name="B1"
            )

            count = await db.delete_all_graphs(tenant_id="tenant-a", subject_id="tenant-a")

            remaining_b = await db.list_graphs(tenant_id="tenant-b", subject_id="tenant-b")

        assert count == 2
        assert len(remaining_b) == 1
        assert remaining_b[0]["name"] == "B1"


# ---------------------------------------------------------------------------
# Update graph
# ---------------------------------------------------------------------------


class TestUpdateGraph:
    @pytest.mark.asyncio
    async def test_update_name_and_nodes(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                name="Old Name",
                nodes=[{"id": "n1"}],
                edges=[{"from": "n1", "to": "n2"}],
            )

            result = await db.update_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                name="New Name",
                nodes=[{"id": "n1"}, {"id": "n3"}],
            )

            updated = await db.get_graph(tenant_id="t1", subject_id="t1", graph_id="g-1")

        assert result is True
        assert updated is not None
        assert updated["name"] == "New Name"
        assert len(updated["nodes"]) == 2

    @pytest.mark.asyncio
    async def test_update_preserves_unset_fields(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                name="Original",
                nodes=[{"id": "n1"}],
                edges=[{"from": "n1", "to": "n2"}],
            )

            # Only update name, not nodes/edges
            result = await db.update_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                name="Updated",
            )

            updated = await db.get_graph(tenant_id="t1", subject_id="t1", graph_id="g-1")

        assert result is True
        assert updated is not None
        assert updated["name"] == "Updated"
        # nodes and edges should be preserved
        assert updated["nodes"] == [{"id": "n1"}]
        assert updated["edges"] == [{"from": "n1", "to": "n2"}]

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_false(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            result = await db.update_graph(
                tenant_id="t1",
                subject_id="t1",
                graph_id="nonexistent",
                name="Nope",
            )

        assert result is False


# ---------------------------------------------------------------------------
# Delete graph
# ---------------------------------------------------------------------------


class TestDeleteGraph:
    @pytest.mark.asyncio
    async def test_delete_existing_graph(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-1", name="To Delete")

            result = await db.delete_graph(tenant_id="t1", subject_id="t1", graph_id="g-1")
            after = await db.get_graph(tenant_id="t1", subject_id="t1", graph_id="g-1")

        assert result is True
        assert after is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            result = await db.delete_graph(tenant_id="t1", subject_id="t1", graph_id="nope")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_wrong_tenant_returns_false(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1", name="A's"
            )

            result = await db.delete_graph(
                tenant_id="tenant-b", subject_id="tenant-b", graph_id="g-1"
            )
            still_exists = await db.get_graph(
                tenant_id="tenant-a", subject_id="tenant-a", graph_id="g-1"
            )

        assert result is False
        assert still_exists is not None


# ---------------------------------------------------------------------------
# Delete all graphs
# ---------------------------------------------------------------------------


class TestDeleteAllGraphs:
    @pytest.mark.asyncio
    async def test_delete_all_returns_count(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-1", name="A")
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-2", name="B")
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-3", name="C")

            count = await db.delete_all_graphs(tenant_id="t1", subject_id="t1")
            remaining = await db.list_graphs(tenant_id="t1", subject_id="t1")

        assert count == 3
        assert remaining == []


# ---------------------------------------------------------------------------
# List graphs
# ---------------------------------------------------------------------------


class TestListGraphs:
    @pytest.mark.asyncio
    async def test_list_returns_id_and_name(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-1", name="First")
            await db.create_graph(tenant_id="t1", subject_id="t1", graph_id="g-2", name="Second")

            result = await db.list_graphs(tenant_id="t1", subject_id="t1")

        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"First", "Second"}
        for r in result:
            assert "id" in r
            assert "name" in r

    @pytest.mark.asyncio
    async def test_list_empty_tenant(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            result = await db.list_graphs(tenant_id="nobody", subject_id="nobody")

        assert result == []


# ---------------------------------------------------------------------------
# Graph states (add + get + delete roundtrip)
# ---------------------------------------------------------------------------


class TestGraphStates:
    @pytest.mark.asyncio
    async def test_add_and_get_graph_states(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            state_id = await db.add_graph_state(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                state_payload={"step": 1, "data": "hello"},
            )

            states = await db.get_graph_states(tenant_id="t1", subject_id="t1", graph_id="g-1")

        assert state_id >= 1
        assert len(states) == 1
        assert states[0]["graph_id"] == "g-1"
        assert states[0]["state_payload"]["step"] == 1
        assert states[0]["state_payload"]["data"] == "hello"

    @pytest.mark.asyncio
    async def test_add_multiple_states_with_limit(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            for i in range(5):
                await db.add_graph_state(
                    tenant_id="t1",
                    subject_id="t1",
                    graph_id="g-1",
                    state_payload={"step": i},
                )

            limited = await db.get_graph_states(
                tenant_id="t1", subject_id="t1", graph_id="g-1", limit=3
            )
            all_states = await db.get_graph_states(
                tenant_id="t1", subject_id="t1", graph_id="g-1", limit=100
            )

        assert len(limited) == 3
        assert len(all_states) == 5

    @pytest.mark.asyncio
    async def test_delete_graph_states(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.add_graph_state(
                tenant_id="t1", subject_id="t1", graph_id="g-1", state_payload={"a": 1}
            )
            await db.add_graph_state(
                tenant_id="t1", subject_id="t1", graph_id="g-1", state_payload={"a": 2}
            )
            await db.add_graph_state(
                tenant_id="t1", subject_id="t1", graph_id="g-2", state_payload={"b": 1}
            )

            count = await db.delete_graph_states(tenant_id="t1", subject_id="t1", graph_id="g-1")
            remaining_g1 = await db.get_graph_states(
                tenant_id="t1", subject_id="t1", graph_id="g-1"
            )
            remaining_g2 = await db.get_graph_states(
                tenant_id="t1", subject_id="t1", graph_id="g-2"
            )

        assert count == 2
        assert len(remaining_g1) == 0
        assert len(remaining_g2) == 1

    @pytest.mark.asyncio
    async def test_delete_all_graph_states(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.add_graph_state(
                tenant_id="t1", subject_id="t1", graph_id="g-1", state_payload={"a": 1}
            )
            await db.add_graph_state(
                tenant_id="t1", subject_id="t1", graph_id="g-2", state_payload={"b": 1}
            )
            await db.add_graph_state(
                tenant_id="t2", subject_id="t2", graph_id="g-3", state_payload={"c": 1}
            )

            count = await db.delete_all_graph_states(tenant_id="t1", subject_id="t1")
            remaining_t1 = await db.get_graph_states(
                tenant_id="t1", subject_id="t1", graph_id="g-1"
            )
            remaining_t2 = await db.get_graph_states(
                tenant_id="t2", subject_id="t2", graph_id="g-3"
            )

        assert count == 2
        assert len(remaining_t1) == 0
        assert len(remaining_t2) == 1

    @pytest.mark.asyncio
    async def test_add_graph_state_cleans_uuids(self):
        mem = InMemoryDB()
        db = GraphStateDatabase()
        uid = UUID("12345678-1234-5678-1234-567812345678")

        with patch_get_pg_connection("src.db.graph_state_db", mem):
            await db.add_graph_state(
                tenant_id="t1",
                subject_id="t1",
                graph_id="g-1",
                state_payload={"worker_id": uid, "status": "done"},
            )

            states = await db.get_graph_states(tenant_id="t1", subject_id="t1", graph_id="g-1")

        assert states[0]["state_payload"]["worker_id"] == str(uid)
