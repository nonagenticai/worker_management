"""Graph management API module.

This module provides comprehensive REST endpoints for managing graphs in the workflow system.
It includes operations for creating, reading, updating, deleting graphs, as well as
state management, validation, and structural analysis capabilities.
"""

import re
from collections import deque
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import ValidationError

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_GRAPH_MANAGEMENT_CREATE,
    ROUTE_GRAPH_MANAGEMENT_DELETE,
    ROUTE_GRAPH_MANAGEMENT_DELETE_ALL,
    ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES,
    ROUTE_GRAPH_MANAGEMENT_DELETE_STATES,
    ROUTE_GRAPH_MANAGEMENT_GET,
    ROUTE_GRAPH_MANAGEMENT_LIST,
    ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY,
    ROUTE_GRAPH_MANAGEMENT_STRUCTURE,
    ROUTE_GRAPH_MANAGEMENT_UPDATE,
    ROUTE_GRAPH_MANAGEMENT_VALIDATE,
    tokenless_tenant_gate,
)
from src.db.graph_state_db import GraphStateDatabase
from src.security.request_overrides import FUNCTION_CODE_NOT_SUPPORTED, has_function_code
from src.worker.graphs.graph_models import (
    GraphCreateRequest,
    GraphResponse,
    GraphUpdateRequest,
    NodeSchema,
)
from src.worker.graphs.graph_operations import (
    create_graph_service,
    delete_all_graphs_service,
    delete_graph_service,
    get_graph_service,
    list_graphs_service,
    update_graph_service,
)

router = APIRouter(prefix="/graphs", tags=["Graphs"])


def validate_graph_id(graph_id: str) -> str:
    """Validate that graph_id is a valid UUID string.

    Args:
        graph_id: String to validate as UUID

    Returns:
        str: Validated graph ID

    Raises:
        HTTPException: 422 if graph_id is not a valid UUID format
    """
    try:
        # This will raise ValueError if not a valid UUID
        UUID(graph_id)
        return graph_id
    except (ValueError, TypeError) as exc:  # fmt: skip
        raise HTTPException(
            status_code=422,
            detail=f"Invalid graph ID format. Expected UUID, got: {graph_id}",
        ) from exc


@router.post(
    "/",
    response_model=GraphResponse,
    status_code=201,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 422, 500]}},
    operation_id="create_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_CREATE))],
)
async def create_graph_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: GraphCreateRequest,
    http_request: Request,
) -> GraphResponse:
    """Create a new graph with name, description, nodes, and edges.

    Raises HTTPException for oversized requests (413), validation errors (422), or server errors (500).
    """
    # Validate request body size
    if http_request.headers.get("content-length"):
        content_length = int(http_request.headers.get("content-length", 0))
        max_size = 10 * 1024 * 1024  # 10MB max for graphs
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Maximum size is {max_size} bytes.",
            )

    try:
        return await create_graph_service(
            ctx.tenant_id,
            ctx.subject_id,
            request,
            visibility=ctx.visibility,
            tenant_name=ctx.tenant_name,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e!s}") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.get(
    "/",
    response_model=list[dict[str, Any]],
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 422, 500]}},
    operation_id="list_graphs",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_LIST))],
)
async def list_graphs_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    include_details: str = Query(
        "false", description="Include full graph details", pattern="^(true|false|0|1)$"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of graphs to return"),
    offset: int = Query(0, ge=0, description="Number of graphs to skip"),
) -> list[dict[str, Any]]:
    """List graphs with pagination for a specific tenant (ID and name by default, or full details).

    Args:
        tenant_id: Tenant identifier for multi-tenancy
        include_details: Whether to include full graph details or just summary
        limit: Maximum number of graphs to return (1-1000, default: 100)
        offset: Number of graphs to skip for pagination (default: 0)

    Returns:
        List[Dict[str, Any]]: Paginated list of graphs with summary or full details

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        # Manually validate and convert include_details parameter
        include_details_bool = include_details.lower() in ["true", "1"]

        graphs = await list_graphs_service(ctx.tenant_id, ctx.subject_id)
        # Apply pagination
        paginated_graphs = graphs[offset : offset + limit]

        if include_details_bool:
            return [graph.model_dump() for graph in paginated_graphs]
        else:
            # Return simplified list for overview
            return [
                {
                    "id": str(graph.id),
                    "name": graph.name,
                    "description": graph.description,
                    "node_count": len(graph.nodes),
                    "created_at": graph.created_at.isoformat(),
                    "updated_at": graph.updated_at.isoformat(),
                }
                for graph in paginated_graphs
            ]
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.delete(
    "/states/all",
    status_code=200,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="delete_all_graph_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES))],
)
async def delete_all_graph_states_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, Any]:
    """Delete all graph state records visible+mutable under (tenant_id, subject_id).

    Returns:
        dict: Message and count of deleted state records

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        db = GraphStateDatabase()
        states_deleted = await db.delete_all_graph_states(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id
        )
        return {
            "message": f"Deleted {states_deleted} graph state records",
            "states_deleted": states_deleted,
        }
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.delete(
    "/",
    status_code=200,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="delete_all_graphs",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_DELETE_ALL))],
)
async def delete_all_graphs_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, Any]:
    """Delete all graphs visible+mutable under (tenant_id, subject_id).

    Returns:
        dict: Message and counts of deleted graphs and states

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        result = await delete_all_graphs_service(ctx.tenant_id, ctx.subject_id)
        deleted_count = result["deleted_count"]
        states_deleted = result["states_deleted"]
        return {
            "message": f"Deleted {deleted_count} graphs",
            "deleted_count": deleted_count,
            "states_deleted": states_deleted,
        }
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.get(
    "/{graph_id}",
    response_model=GraphResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_GET))],
)
async def get_graph_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> GraphResponse:
    """Retrieve a specific graph by ID for a specific tenant.

    Args:
        graph_id: UUID string of the graph to retrieve
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        GraphResponse: Complete graph information including nodes and edges

    Raises:
        HTTPException: 422 for invalid UUID, 404 for not found, 500 for server errors
    """
    try:
        # Validate UUID format
        validated_id = validate_graph_id(graph_id)
        return await get_graph_service(ctx.tenant_id, ctx.subject_id, validated_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.patch(
    "/{graph_id}",
    response_model=GraphResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="update_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_UPDATE))],
)
async def update_graph_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: GraphUpdateRequest,
    http_request: Request,
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> GraphResponse:
    """Update an existing graph for a specific tenant (supports incremental updates).

    Args:
        request: Update request with fields to modify
        http_request: HTTP request object for validation
        graph_id: UUID string of the graph to update
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        GraphResponse: Updated graph information

    Raises:
        HTTPException: 413 if request body is too large (>10MB)
        HTTPException: 422 for validation errors, 404 for not found, 500 for server errors
    """
    # Validate request body size
    if http_request.headers.get("content-length"):
        content_length = int(http_request.headers.get("content-length", 0))
        max_size = 10 * 1024 * 1024  # 10MB max for graphs
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Maximum size is {max_size} bytes.",
            )

    try:
        # Validate UUID format
        validated_id = validate_graph_id(graph_id)
        return await update_graph_service(ctx.tenant_id, ctx.subject_id, validated_id, request)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e!s}") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.delete(
    "/{graph_id}/states",
    status_code=200,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="delete_graph_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_DELETE_STATES))],
)
async def delete_graph_states_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> dict[str, Any]:
    """Delete all execution state history for a specific graph.

    Graph definition remains intact. Returns count of deleted states.

    Raises HTTPException for invalid UUID (422) or database errors (500).
    """
    try:
        validated_id = validate_graph_id(graph_id)
        db = GraphStateDatabase()
        states_deleted = await db.delete_graph_states(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, graph_id=validated_id
        )
        return {
            "message": f"Deleted {states_deleted} state records for graph {validated_id}",
            "graph_id": validated_id,
            "states_deleted": states_deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.delete(
    "/{graph_id}",
    status_code=204,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="delete_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_DELETE))],
)
async def delete_graph_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> None:
    """Permanently delete a graph and all its execution states.

    Raises HTTPException for invalid UUID (422), graph not found (404), or server errors (500).
    """
    try:
        # Validate UUID format
        validated_id = validate_graph_id(graph_id)
        await delete_graph_service(ctx.tenant_id, ctx.subject_id, validated_id)
        return
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.get(
    "/{graph_id}/structure",
    response_model=dict[str, Any],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_graph_structure",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_STRUCTURE))],
)
async def get_graph_structure_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> dict[str, Any]:
    """Get simplified graph structure for visualization.

    Returns lightweight representation with basic node/edge info and topology type.

    Raises HTTPException for invalid UUID (422), graph not found (404), or server errors (500).
    """
    try:
        validated_id = validate_graph_id(graph_id)
        graph = await get_graph_service(ctx.tenant_id, ctx.subject_id, validated_id)

        # Build simplified structure
        nodes_simple = [
            {
                "id": str(node.id),
                "name": node.name,
                "node_type": node.node_type,
                "worker_name": node.worker_name,
                "function_name": node.function_name,
                "has_template": bool(node.task_template),
            }
            for node in graph.nodes
        ]

        edges_simple = [
            {
                "source": str(edge.source),
                "target": str(edge.target),
                "condition": edge.condition,
            }
            for edge in graph.edges
        ]

        # Determine structure type based on graph topology
        structure_type = "unknown"
        if len(graph.nodes) == 0:
            structure_type = "empty"
        elif len(graph.nodes) == 1:
            structure_type = "single_node"
        elif len(graph.edges) == 0:
            structure_type = "disconnected"
        else:
            # Analyze graph structure
            node_connections = {}
            for edge in graph.edges:
                source_id = str(edge.source)
                target_id = str(edge.target)

                if source_id not in node_connections:
                    node_connections[source_id] = {"in": 0, "out": 0}
                if target_id not in node_connections:
                    node_connections[target_id] = {"in": 0, "out": 0}

                node_connections[source_id]["out"] += 1
                node_connections[target_id]["in"] += 1

            # Determine structure type
            has_branching = any(conn["out"] > 1 for conn in node_connections.values())
            has_merging = any(conn["in"] > 1 for conn in node_connections.values())

            if has_branching and has_merging:
                structure_type = "complex"
            elif has_branching:
                structure_type = "branching"
            elif has_merging:
                structure_type = "merging"
            elif len(graph.edges) == len(graph.nodes) - 1:
                structure_type = "linear"
            else:
                structure_type = "complex"

        return {
            "id": str(graph.id),
            "name": graph.name,
            "description": graph.description,
            "nodes": nodes_simple,
            "edges": edges_simple,
            "variables": graph.variables,
            "structure_type": structure_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


@router.get(
    "/{graph_id}/states",
    response_model=list[dict[str, Any]],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_graph_states",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY))],
)
async def get_graph_state_history_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
    limit: int = Query(10, ge=1, le=100, description="Number of states to return"),
) -> list[dict[str, Any]]:
    """Get the execution state history for a graph for a specific tenant.

    Args:
        graph_id: UUID of the graph to get state history for
        tenant_id: Tenant identifier for multi-tenancy
        limit: Number of states to return (1-100)

    Returns:
        List of state records

    Raises:
        HTTPException: 422 for invalid UUID, 404 for not found, 500 for server errors
    """
    try:
        # Validate UUID format
        validated_id = validate_graph_id(graph_id)
        db = GraphStateDatabase()
        return await db.get_graph_states(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            graph_id=validated_id,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e


def _validate_edge_references(
    graph: GraphResponse,
    node_ids: set[str],
    node_map: dict[str, NodeSchema],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    """Validate edge endpoints, flag self-loops, and build adjacency maps."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    outgoing: dict[str, list[str]] = {nid: [] for nid in node_ids}
    incoming: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in graph.edges:
        src = str(edge.source)
        tgt = str(edge.target)
        if src not in node_ids:
            errors.append(
                {
                    "type": "invalid_edge_source",
                    "message": f"Edge source {src} does not reference an existing node",
                    "node_id": src,
                }
            )
        if tgt not in node_ids:
            errors.append(
                {
                    "type": "invalid_edge_target",
                    "message": f"Edge target {tgt} does not reference an existing node",
                    "node_id": tgt,
                }
            )
        # Only populate adjacency for valid references
        if src in node_ids and tgt in node_ids:
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

        # Self-loop detection
        if src == tgt and src in node_ids:
            warnings.append(
                {
                    "type": "self_loop",
                    "message": (
                        f"Edge from node {node_map[src].name} ({src}) to itself creates a self-loop"
                    ),
                    "node_id": src,
                }
            )

    return errors, warnings, outgoing, incoming


def _validate_orphaned_nodes(
    graph: GraphResponse,
    node_ids: set[str],
    node_map: dict[str, NodeSchema],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Flag nodes with no edges at all (skipped for a single-node graph)."""
    warnings: list[dict[str, Any]] = []
    if len(graph.nodes) > 1:
        for nid in node_ids:
            if len(incoming[nid]) == 0 and len(outgoing[nid]) == 0:
                warnings.append(
                    {
                        "type": "orphaned_nodes",
                        "message": (
                            f"Node {node_map[nid].name} ({nid}) has no incoming or outgoing edges"
                        ),
                        "node_id": nid,
                    }
                )
    return warnings


def _validate_unreachable_nodes(
    node_ids: set[str],
    node_map: dict[str, NodeSchema],
    outgoing: dict[str, list[str]],
    start_nodes: list[str],
) -> list[dict[str, Any]]:
    """BFS from start nodes; flag any node not reached."""
    warnings: list[dict[str, Any]] = []
    reachable: set[str] = set()
    queue: deque = deque(start_nodes)
    reachable.update(start_nodes)
    while queue:
        current = queue.popleft()
        for neighbor in outgoing.get(current, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    for nid in node_ids - reachable:
        warnings.append(
            {
                "type": "unreachable_node",
                "message": (
                    f"Node {node_map[nid].name} ({nid}) is not reachable from any start node"
                ),
                "node_id": nid,
            }
        )
    return warnings


def _validate_condition_branches(
    node_map: dict[str, NodeSchema],
    outgoing: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Condition nodes need >=2 outgoing edges (true/false branches)."""
    errors: list[dict[str, Any]] = []
    for nid, node in node_map.items():
        if node.node_type == "condition":
            out_count = len(outgoing[nid])
            if out_count < 2:
                errors.append(
                    {
                        "type": "condition_missing_branches",
                        "message": (
                            f"Condition node {node.name} ({nid}) has {out_count} "
                            f"outgoing edge(s) but needs at least 2 (true/false branches)"
                        ),
                        "node_id": nid,
                    }
                )
    return errors


def _validate_template_variables(
    graph: GraphResponse,
    node_map: dict[str, NodeSchema],
) -> list[dict[str, Any]]:
    """Flag task-template variables not defined in graph variables."""
    warnings: list[dict[str, Any]] = []
    graph_vars = set((graph.variables or {}).keys())
    template_var_pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
    for nid, node in node_map.items():
        if node.task_template:
            referenced_vars = set(template_var_pattern.findall(node.task_template))
            undefined_vars = referenced_vars - graph_vars
            if undefined_vars:
                warnings.append(
                    {
                        "type": "undefined_template_variable",
                        "message": (
                            f"Node {node.name} ({nid}) references variables "
                            f"not defined in graph variables: {sorted(undefined_vars)}"
                        ),
                        "node_id": nid,
                    }
                )
    return warnings


def _validate_cycles(
    node_ids: set[str],
    node_map: dict[str, NodeSchema],
    outgoing: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Detect circular dependencies via DFS coloring."""
    errors: list[dict[str, Any]] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in node_ids}
    parent: dict[str, str | None] = {nid: None for nid in node_ids}
    cycles_found: list[list[str]] = []

    def dfs_cycle(u: str) -> None:
        color[u] = GRAY
        for v in outgoing.get(u, []):
            if color[v] == GRAY:
                # Back edge: reconstruct cycle path
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    p = parent.get(cur)
                    if p is None:
                        break
                    cur = p
                cycle.append(v)
                cycle.reverse()
                cycles_found.append(cycle)
            elif color[v] == WHITE:
                parent[v] = u
                dfs_cycle(v)
        color[u] = BLACK

    for nid in node_ids:
        if color[nid] == WHITE:
            dfs_cycle(nid)

    for cycle in cycles_found:
        cycle_desc = " -> ".join(f"{node_map[c].name} ({c})" if c in node_map else c for c in cycle)
        errors.append(
            {
                "type": "circular_dependency",
                "message": (f"Circular dependency detected: {cycle_desc}"),
                "node_id": cycle[0],
                "cycle_path": cycle,
                "cycle_length": len(cycle) - 1,
                "affected_nodes": cycle[:-1],
            }
        )
    return errors


def _validate_worker_function_assignment(graph: GraphResponse) -> list[dict[str, Any]]:
    """Worker nodes need worker_name; function nodes need function_name; no node carries code."""
    errors: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type == "worker" and not node.worker_name:
            errors.append(
                {
                    "type": "missing_worker",
                    "message": f"Worker node {node.name} missing worker_name",
                    "node_id": str(node.id),
                }
            )
        elif node.node_type == "function" and not node.function_name:
            errors.append(
                {
                    "type": "missing_function",
                    "message": f"Function node {node.name} missing function_name",
                    "node_id": str(node.id),
                }
            )
        if has_function_code(node.data):
            errors.append(
                {
                    "type": FUNCTION_CODE_NOT_SUPPORTED,
                    "message": f"Node {node.name} carries function_code, which is never executed",
                    "node_id": str(node.id),
                }
            )
    return errors


@router.get(
    "/{graph_id}/validate",
    response_model=dict[str, Any],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="validate_graph",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_GRAPH_MANAGEMENT_VALIDATE))],
)
async def validate_graph_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    graph_id: str = Path(
        ...,
        description="UUID of the graph",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
) -> dict[str, Any]:
    """Validate graph structure, templates, and connectivity.

    Performs comprehensive validation including edge references, self-loops,
    orphan/unreachable nodes, condition branch completeness, template variables,
    and cycle detection.

    Returns:
        dict with valid (bool), errors (list), warnings (list), and summary (dict)
    """
    try:
        validated_id = validate_graph_id(graph_id)
        graph = await get_graph_service(ctx.tenant_id, ctx.subject_id, validated_id)

        node_ids = {str(node.id) for node in graph.nodes}
        node_map: dict[str, NodeSchema] = {str(node.id): node for node in graph.nodes}

        # Edge references + self-loops, returning the adjacency maps the other rules need.
        edge_errors, edge_warnings, outgoing, incoming = _validate_edge_references(
            graph, node_ids, node_map
        )

        # Start nodes have no incoming edges; end nodes have no outgoing edges.
        start_nodes = [nid for nid in node_ids if len(incoming[nid]) == 0]
        end_nodes = [nid for nid in node_ids if len(outgoing[nid]) == 0]

        errors: list[dict[str, Any]] = [
            *edge_errors,
            *_validate_condition_branches(node_map, outgoing),
            *_validate_cycles(node_ids, node_map, outgoing),
            *_validate_worker_function_assignment(graph),
        ]
        warnings: list[dict[str, Any]] = [
            *edge_warnings,
            *_validate_orphaned_nodes(graph, node_ids, node_map, incoming, outgoing),
            *_validate_unreachable_nodes(node_ids, node_map, outgoing, start_nodes),
            *_validate_template_variables(graph, node_map),
        ]

        return {
            "graph_id": validated_id,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "total_nodes": len(graph.nodes),
                "total_edges": len(graph.edges),
                "start_nodes": start_nodes,
                "end_nodes": end_nodes,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}") from e
