"""Graph state database module (PostgreSQL with multi-tenancy).

Per-user-within-tenant: every operation takes (tenant_id, subject_id) and the
SELECT/UPDATE/DELETE WHERE clause goes through ``scope_clause()``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Optional, cast
from uuid import UUID

from pydantic import BaseModel

from src.api.tenant_constants import tenant_label
from src.db.db_config import get_pg_connection
from src.db.workers_state_db import with_retry
from src.repositories.visibility import scope_clause

logger = logging.getLogger(__name__)


def _clean_item(item: Any) -> Any:
    """Recursively convert pydantic / UUID / nested collections to JSON-safe primitives.

    - Pydantic BaseModel -> dict (model_dump)
    - UUID -> str
    - list/tuple -> clean each element
    - dict -> clean each value.
    """
    if isinstance(item, BaseModel):
        raw = item.model_dump()
        cleaned_model_dict: dict[str, Any] = {}
        for k, v in raw.items():
            cleaned_model_dict[k] = _clean_item(v)
        return cleaned_model_dict
    elif isinstance(item, dict):
        cleaned_dict: dict[str, Any] = {}
        for k, v in item.items():
            cleaned_dict[k] = _clean_item(v)
        return cleaned_dict
    elif isinstance(item, list):
        return [_clean_item(elem) for elem in item]
    elif isinstance(item, tuple):
        return tuple(_clean_item(elem) for elem in item)
    elif isinstance(item, UUID):
        return str(item)
    else:
        return item


class GraphStateDatabase:
    """PostgreSQL-backed database for graph definitions / state history.

    All operations are scoped under (tenant_id, subject_id, visibility).
    """

    @with_retry()
    async def create_graph(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        graph_id: str,
        name: str,
        description: str = "",
        nodes: Optional[list[dict[str, Any]]] = None,
        edges: Optional[list[dict[str, Any]]] = None,
        variables: Optional[dict[str, Any]] = None,
        visibility: str = "tenant",
    ) -> None:
        """Insert a new graph definition into the ``graphs`` table."""
        timestamp = datetime.now(UTC).replace(tzinfo=None)
        effective_subject = subject_id or tenant_id

        if nodes is None:
            nodes = []
        if edges is None:
            edges = []
        if variables is None:
            variables = {}

        nodes_clean = [_clean_item(n) for n in nodes]
        edges_clean = [_clean_item(e) for e in edges]
        variables_clean = _clean_item(variables)

        async with get_pg_connection() as conn:
            await conn.execute(
                """
                INSERT INTO graphs (id, tenant_id, subject_id, visibility,
                                    tenant_name, name, description, nodes, edges, variables,
                                    created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                str(graph_id),
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                name,
                description,
                nodes_clean,
                edges_clean,
                variables_clean,
                timestamp,
                timestamp,
            )

            logger.info(
                "Created graph: tenant_id=%s, subject_id=%s, graph_id=%s, name=%s",
                tenant_id,
                effective_subject,
                graph_id,
                name,
            )

    @with_retry()
    async def update_graph(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        graph_id: str,
        nodes: Optional[list[dict[str, Any]]] = None,
        edges: Optional[list[dict[str, Any]]] = None,
        description: Optional[str] = None,
        name: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update an existing graph's properties (visible+mutable under scope)."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT name, description, nodes, edges, variables FROM graphs
                WHERE {scope_clause(1, 2)} AND id = $3
                """,
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
            )

            if not row:
                return False

            old_name = row["name"]
            old_description = row["description"] or ""
            old_nodes = row["nodes"] or []
            old_edges = row["edges"] or []
            old_variables = row["variables"] or {}

            new_name = name if name is not None else old_name
            new_description = description if description is not None else old_description

            new_nodes = [_clean_item(n) for n in nodes] if nodes is not None else old_nodes

            new_edges = [_clean_item(e) for e in edges] if edges is not None else old_edges

            new_variables = _clean_item(variables) if variables is not None else old_variables

            timestamp = datetime.now(UTC).replace(tzinfo=None)

            _sql = f"UPDATE graphs SET name = $1, description = $2, nodes = $3, edges = $4, variables = $5, updated_at = $6 WHERE {scope_clause(7, 8)} AND id = $9"  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
            await conn.execute(
                _sql,
                new_name,
                new_description,
                new_nodes,
                new_edges,
                new_variables,
                timestamp,
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
            )

            logger.info(
                "Updated graph: tenant_id=%s, subject_id=%s, graph_id=%s",
                tenant_id,
                subject_id,
                graph_id,
            )
            return True

    @with_retry()
    async def delete_graph(self, *, tenant_id: str, subject_id: str, graph_id: str) -> bool:
        """Delete a graph definition by id (visible+mutable under scope)."""
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM graphs WHERE {scope_clause(1, 2)} AND id = $3",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
            )

            deleted_count = int(result.split()[-1]) if result else 0
            success = deleted_count > 0

            if success:
                logger.info(
                    "Deleted graph: tenant_id=%s, subject_id=%s, graph_id=%s",
                    tenant_id,
                    subject_id,
                    graph_id,
                )

            return success

    @with_retry()
    async def delete_all_graphs(self, *, tenant_id: str, subject_id: str) -> int:
        """Delete all graph definitions visible+mutable under scope."""
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM graphs WHERE {scope_clause(1, 2)}",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
            )

            deleted_count = int(result.split()[-1]) if result else 0

            logger.info(
                "Deleted all graphs for tenant=%s subject=%s: count=%d",
                tenant_id,
                subject_id,
                deleted_count,
            )
            return deleted_count

    @with_retry()
    async def delete_graph_states(self, *, tenant_id: str, subject_id: str, graph_id: str) -> int:
        """Delete all state records for a specific graph."""
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM graph_states WHERE {scope_clause(1, 2)} AND graph_id = $3",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
            )

            deleted_count = int(result.split()[-1]) if result else 0

            logger.info(
                "Deleted graph states: tenant=%s subject=%s graph_id=%s, count=%d",
                tenant_id,
                subject_id,
                graph_id,
                deleted_count,
            )
            return deleted_count

    @with_retry()
    async def delete_all_graph_states(self, *, tenant_id: str, subject_id: str) -> int:
        """Delete all graph state records visible+mutable under scope."""
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM graph_states WHERE {scope_clause(1, 2)}",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
            )

            deleted_count = int(result.split()[-1]) if result else 0

            logger.info(
                "Deleted all graph states for tenant=%s subject=%s: count=%d",
                tenant_id,
                subject_id,
                deleted_count,
            )
            return deleted_count

    @with_retry()
    async def get_graph(
        self, *, tenant_id: str, subject_id: str, graph_id: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve a graph by id, visible under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, tenant_id, subject_id, visibility,
                       name, description, nodes, edges, variables, created_at, updated_at
                FROM graphs
                WHERE {scope_clause(1, 2)} AND id = $3
                """,
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
            )

            if not row:
                return None

            return {
                "id": str(row["id"]),
                "tenant_id": row.get("tenant_id"),
                "subject_id": row.get("subject_id"),
                "visibility": row.get("visibility", "tenant"),
                "name": row["name"],
                "description": row.get("description", "") or "",
                "nodes": row["nodes"] or [],
                "edges": row["edges"] or [],
                "variables": row.get("variables", {}) or {},
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }

    @with_retry()
    async def list_graphs(self, *, tenant_id: str, subject_id: str) -> list[dict[str, Any]]:
        """List all graphs visible under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, name
                FROM graphs
                WHERE {scope_clause(1, 2)}
                ORDER BY created_at DESC
                """,
                tenant_id,
                subject_id or tenant_id,
            )

            return [{"id": str(row["id"]), "name": row["name"]} for row in rows]

    @with_retry()
    async def add_graph_state(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        graph_id: str,
        state_payload: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,  # noqa: ARG002, RUF100
        visibility: str = "tenant",
    ) -> int:
        """Insert a new state record for a graph."""
        effective_subject = subject_id or tenant_id
        async with get_pg_connection() as conn:
            timestamp = datetime.now(UTC).replace(tzinfo=None)

            clean_payload = {k: _clean_item(v) for k, v in state_payload.items()}

            result = await conn.fetchrow(
                """
                INSERT INTO graph_states (tenant_id, subject_id, visibility,
                                          tenant_name, graph_id, state_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                str(graph_id),
                clean_payload,
                timestamp,
                timestamp,
            )

            state_id = int(cast("dict[str, Any]", result)["id"])
            logger.debug("Added graph state with ID %d", state_id)
            return state_id

    @with_retry()
    async def get_execution_state(
        self, *, tenant_id: str, subject_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        """Get the latest execution state for a session."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT state_data, created_at
                FROM graph_states
                WHERE {scope_clause(1, 2)} AND state_data->>'session_id' = $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tenant_id,
                subject_id or tenant_id,
                session_id,
            )

            if row:
                state_data = dict(row["state_data"] or {})
                state_data["timestamp"] = (
                    row["created_at"].isoformat() if row["created_at"] else None
                )
                return state_data

            return None

    @with_retry()
    async def get_execution_events(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        session_id: str,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get execution events for a session."""
        async with get_pg_connection() as conn:
            if event_type:
                rows = await conn.fetch(
                    f"""
                    SELECT state_data, created_at
                    FROM graph_states
                    WHERE {scope_clause(1, 2)}
                      AND state_data->>'session_id' = $3
                      AND state_data->>'event_type' = $4
                    ORDER BY created_at ASC
                    """,
                    tenant_id,
                    subject_id or tenant_id,
                    session_id,
                    event_type,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT state_data, created_at
                    FROM graph_states
                    WHERE {scope_clause(1, 2)} AND state_data->>'session_id' = $3
                    ORDER BY created_at ASC
                    """,
                    tenant_id,
                    subject_id or tenant_id,
                    session_id,
                )

            events = []
            for row in rows:
                event_data = dict(row["state_data"] or {})
                event_data["timestamp"] = (
                    row["created_at"].isoformat() if row["created_at"] else None
                )
                events.append(event_data)

            return events

    @with_retry()
    async def get_graph_states(
        self, *, tenant_id: str, subject_id: str, graph_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve the most recent ``limit`` state records for a graph."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, graph_id, created_at, state_data
                FROM graph_states
                WHERE {scope_clause(1, 2)} AND graph_id = $3
                ORDER BY created_at DESC
                LIMIT $4
                """,
                tenant_id,
                subject_id or tenant_id,
                str(graph_id),
                limit,
            )

            return [
                {
                    "id": row["id"],
                    "graph_id": str(row["graph_id"]),
                    "timestamp": row["created_at"].isoformat() if row["created_at"] else None,
                    "state_payload": row["state_data"] or {},
                    "metadata": {},
                }
                for row in rows
            ]

    @with_retry()
    async def list_execution_sessions(
        self, *, tenant_id: str, subject_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List execution sessions with their latest state."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT state_data->>'session_id' as session_id,
                       MAX(created_at) as latest_timestamp,
                       graph_id
                FROM graph_states
                WHERE {scope_clause(1, 2)} AND state_data->>'session_id' IS NOT NULL
                GROUP BY state_data->>'session_id', graph_id
                ORDER BY latest_timestamp DESC
                LIMIT $3
                """,
                tenant_id,
                subject_id or tenant_id,
                limit,
            )

            return [
                {
                    "session_id": row["session_id"],
                    "latest_timestamp": row["latest_timestamp"].isoformat()
                    if row["latest_timestamp"]
                    else None,
                    "graph_id": str(row["graph_id"]),
                }
                for row in rows
            ]
