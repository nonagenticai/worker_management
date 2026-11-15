"""Graph operations module.

Service-layer functions for graph CRUD operations under the
per-user-within-tenant isolation model. Every entry point takes
``(tenant_id, subject_id)`` and forwards them to the DB layer; the
``visibility`` of newly-created rows is supplied separately by the API
layer (``Depends(get_visibility)``) and defaults to ``'tenant'``.
"""

import logging
from uuid import uuid7

import psycopg
from fastapi import HTTPException

from src.db.graph_state_db import GraphStateDatabase
from src.worker.graphs.graph_models import (
    GraphCreateRequest,
    GraphResponse,
    GraphUpdateRequest,
)

logger = logging.getLogger(__name__)


async def check_database_health(tenant_id: str = "", subject_id: str = "") -> bool:
    """Check if the database connection is healthy and tables are accessible.

    The default was the literal ``"test"``, which reached ``WHERE tenant_id = $1``.
    ``tenant_id`` is a ``uuid`` column since migration 024, so that default would
    now raise 22P02 and a *health check* would report the database unhealthy
    because of its own argument. It resolves to the deployment's default tenant
    instead.
    """
    from src.api.tenant_constants import DEFAULT_TENANT_UUID

    tenant_id = tenant_id or DEFAULT_TENANT_UUID
    try:
        db = GraphStateDatabase()
        await db.list_graphs(tenant_id=tenant_id, subject_id=subject_id or tenant_id)
        return True
    except psycopg.Error, OSError, RuntimeError:
        logger.exception("Database health check failed")
        return False


async def create_graph_service(
    tenant_id: str,
    subject_id: str,
    request: GraphCreateRequest,
    visibility: str = "tenant",
    tenant_name: str = "",
) -> GraphResponse:
    """Create a new graph (visibility-tagged) for (tenant_id, subject_id)."""
    try:
        graph_id = str(uuid7())
        db = GraphStateDatabase()
        await db.create_graph(
            tenant_id=tenant_id,
            subject_id=subject_id,
            tenant_name=tenant_name,
            graph_id=graph_id,
            name=request.name,
            description=request.description or "",
            nodes=[node.model_dump() for node in request.nodes],
            edges=[edge.model_dump() for edge in request.edges],
            variables=request.variables or {},
            visibility=visibility,
        )
        # Optionally add initial state record
        await db.add_graph_state(
            tenant_id=tenant_id,
            subject_id=subject_id,
            tenant_name=tenant_name,
            graph_id=graph_id,
            state_payload={
                "event": "created",
                "nodes": [n.model_dump() for n in request.nodes],
            },
            metadata=None,
            visibility=visibility,
        )
        graph_data = await db.get_graph(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        if not graph_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve graph after creation")
        return GraphResponse(**_to_response_dict(graph_data))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating graph")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


async def get_graph_service(tenant_id: str, subject_id: str, graph_id: str) -> GraphResponse:
    """Retrieve a specific graph visible under (tenant_id, subject_id)."""
    try:
        db = GraphStateDatabase()
        graph_data = await db.get_graph(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        if not graph_data:
            raise HTTPException(status_code=404, detail="Graph not found")
        return GraphResponse(**_to_response_dict(graph_data))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving graph {graph_id}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


async def list_graphs_service(tenant_id: str, subject_id: str) -> list[GraphResponse]:
    """List all graphs visible under (tenant_id, subject_id)."""
    try:
        db = GraphStateDatabase()
        rows = await db.list_graphs(tenant_id=tenant_id, subject_id=subject_id)
        if not rows:
            return []

        results = []
        for row in rows:
            try:
                graph_data = await db.get_graph(
                    tenant_id=tenant_id, subject_id=subject_id, graph_id=row["id"]
                )
                if graph_data:
                    results.append(GraphResponse(**_to_response_dict(graph_data)))
            except (psycopg.Error, ValueError, KeyError, RuntimeError) as e:
                logger.warning(f"Failed to load graph {row['id']}: {e}")
                continue
        return results
    except Exception as e:
        logger.exception("Error listing graphs")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


async def update_graph_service(
    tenant_id: str, subject_id: str, graph_id: str, request: GraphUpdateRequest
) -> GraphResponse:
    """Update an existing graph visible+mutable under (tenant_id, subject_id)."""
    try:
        db = GraphStateDatabase()
        if request.edges is not None or request.nodes is not None:
            existing_graph = await db.get_graph(
                tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
            )
            if not existing_graph:
                raise HTTPException(status_code=404, detail="Graph not found")

        success = await db.update_graph(
            tenant_id=tenant_id,
            subject_id=subject_id,
            graph_id=graph_id,
            name=request.name,
            description=request.description,
            nodes=([n.model_dump() for n in request.nodes] if request.nodes else None),
            edges=([e.model_dump() for e in request.edges] if request.edges else None),
            variables=request.variables,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Graph not found")

        await db.add_graph_state(
            tenant_id=tenant_id,
            subject_id=subject_id,
            graph_id=graph_id,
            state_payload={"event": "updated", **request.model_dump()},
            metadata=None,
        )
        graph_data = await db.get_graph(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        if not graph_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve graph after update")
        return GraphResponse(**_to_response_dict(graph_data))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating graph {graph_id}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


async def delete_graph_service(tenant_id: str, subject_id: str, graph_id: str) -> None:
    """Delete a specific graph visible+mutable under (tenant_id, subject_id)."""
    try:
        db = GraphStateDatabase()
        graph_data = await db.get_graph(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        if not graph_data:
            raise HTTPException(status_code=404, detail="Graph not found")

        await db.add_graph_state(
            tenant_id=tenant_id,
            subject_id=subject_id,
            graph_id=graph_id,
            state_payload={"event": "deleted"},
            metadata=None,
        )

        states_deleted = await db.delete_graph_states(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        logger.info(f"Deleted {states_deleted} state records for graph {graph_id}")

        success = await db.delete_graph(
            tenant_id=tenant_id, subject_id=subject_id, graph_id=graph_id
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete graph")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting graph {graph_id}")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


async def delete_all_graphs_service(tenant_id: str, subject_id: str) -> dict:
    """Delete all graphs visible+mutable under (tenant_id, subject_id)."""
    try:
        db = GraphStateDatabase()
        rows = await db.list_graphs(tenant_id=tenant_id, subject_id=subject_id)
        graph_ids = [row["id"] for row in rows]

        for graph_id in graph_ids:
            try:
                await db.add_graph_state(
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    graph_id=graph_id,
                    state_payload={"event": "deleted"},
                    metadata=None,
                )
            except (psycopg.Error, RuntimeError) as e:
                logger.warning(f"Failed to record deletion state for graph {graph_id}: {e}")

        states_deleted = 0
        try:
            states_deleted = await db.delete_all_graph_states(
                tenant_id=tenant_id, subject_id=subject_id
            )
            logger.info(f"Deleted {states_deleted} graph state records")
        except (psycopg.Error, RuntimeError) as state_error:
            logger.warning(f"Failed to delete all graph states: {state_error}")

        deleted_count = await db.delete_all_graphs(tenant_id=tenant_id, subject_id=subject_id)

        return {"deleted_count": deleted_count, "states_deleted": states_deleted}
    except Exception as e:
        logger.exception("Error deleting all graphs")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e


def _to_response_dict(graph_data: dict) -> dict:
    """Strip scope-internal fields before constructing a GraphResponse."""
    drop = {"tenant_id", "subject_id", "visibility"}
    return {k: v for k, v in graph_data.items() if k not in drop}
