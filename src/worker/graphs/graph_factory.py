"""Graph Factory module for creating pydantic-graph instances.

This module provides a factory for creating graph structures using the pydantic-graph
library, following official documentation patterns for node creation and routing.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid7

from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext
from pydantic_graph.step import NodeStep

from src.security.request_overrides import FunctionCodeNotSupportedError, has_function_code
from src.worker.graphs.nodes.worker_node import record_quality_score
from src.worker.graphs.pydantic_graph_state import WorkflowState
from src.worker.worker_manager import WorkerManager

if TYPE_CHECKING:
    from src.worker.graphs.graph_models import EdgeSchema, NodeSchema

logger = logging.getLogger(__name__)


def _union_of(*members: Any) -> Any:
    """Build a `Union[...]` from classes known only at runtime.

    pydantic-graph 2 reads a node's outgoing edges off its `run` return
    annotation, and these node classes are generated per graph, so the union
    cannot be written literally. Built as `Any` because a type checker cannot
    accept a runtime list of classes inside a type expression.
    """
    hint: Any = members[0]
    for member in members[1:]:
        hint = hint | member
    return hint


@dataclass
class GraphMetadata:
    """Metadata storage for Graph instances.

    `node_classes` and `start_node_id` exist so `create_start_node` never has to
    reach into `Graph`'s private `_nodes` to find the start class: pydantic-graph
    2 exposes only the built graph, not its node registry.
    """

    node_schemas: dict[str, NodeSchema]
    edge_map: dict[str, list[dict[str, Any]]]
    name: str
    node_classes: dict[str, type[BaseNode[WorkflowState, None, Any]]] = field(default_factory=dict)
    start_node_id: str = ""


class GraphFactory:
    """Factory for creating pydantic-graph instances following official documentation."""

    def __init__(self, worker_manager: WorkerManager | None = None) -> None:
        """Initialize GraphFactory with optional WorkerManager.

        Args:
            worker_manager: Optional WorkerManager instance. If None, creates new one.
        """
        self._worker_manager = worker_manager
        self._graph_metadata: dict[int, GraphMetadata] = {}

    def _get_worker_manager(self) -> WorkerManager:
        """Get or create the WorkerManager instance.

        Returns:
            WorkerManager: The worker manager instance

        Raises:
            RuntimeError: If WorkerManager cannot be initialized
        """
        if self._worker_manager is None:
            try:
                self._worker_manager = WorkerManager()
                logger.info("WorkerManager initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize WorkerManager")
                raise RuntimeError(f"Cannot initialize WorkerManager: {e}") from e
        return self._worker_manager

    def create_graph(
        self,
        nodes: list[NodeSchema],
        edges: list[EdgeSchema],
        name: str | None = None,
        validate: bool = True,  # noqa: FBT001, FBT002, RUF100
        start_node_id: str | None = None,
    ) -> Graph[WorkflowState, None, Any, Any]:
        """Build a pydantic-graph 2 graph from node and edge schemas.

        The start node has to be known at build time: pydantic-graph 2 wires a
        single edge out of its `StartNode`, and a second edge turns it into a
        fork that dispatches to the first branch regardless of the instance
        passed in. Callers that execute from a caller-chosen node must pass
        `start_node_id` here, not just to `create_start_node`.
        """
        if validate:
            issues = GraphFactory.validate_graph_structure(nodes, edges)
            if issues:
                raise ValueError(f"Graph validation failed: {'; '.join(issues)}")

        # Build edge mapping for routing logic
        edge_map = self._build_edge_map(edges)

        # Create node mapping for easy lookup
        node_map = {str(node.id): node for node in nodes}

        # Create node CLASSES dynamically following official pydantic_graph patterns
        node_classes = self._create_node_classes(nodes, node_map, edge_map)

        graph_name = name or f"graph_{uuid7()}"
        logger.info("Creating pydantic Graph with %d node classes", len(node_classes))

        # pydantic-graph 2 replaced `Graph(nodes=(...))` — which inferred routing
        # from each node's `run` return *value* — with a GraphBuilder whose edges
        # are declared up front. Our routing is still dynamic (each node returns
        # the next node instance), so we publish the same edge set twice:
        #   1. as a return-type hint on each generated node class, which is how
        #      the builder learns its outgoing edges, and
        #   2. as the start edge from the builder's StartNode.
        # A node that returns `End(...)` needs an edge to EndNode to be legal.
        registry = {
            cast("Any", nc)()._node_schema_id: nc  # injected dataclass field
            for nc in node_classes
        }
        for nc in node_classes:
            schema_id = cast("Any", nc)()._node_schema_id
            targets = [
                registry[e["target"]]
                for e in edge_map.get(schema_id, [])
                if e["target"] in registry
            ]
            cast("Any", nc).run.__annotations__["return"] = _union_of(*targets, End[Any])

        start_node_id = self._determine_start_node(
            start_node_id, node_map, edge_map, allow_cycle_fallback=True
        )
        builder: GraphBuilder[WorkflowState, None, Any, Any] = GraphBuilder(
            name=graph_name,
            state_type=WorkflowState,
            deps_type=type(None),
            input_type=Any,
            output_type=Any,
        )
        for nc in node_classes:
            builder.add(builder.node(nc))
        builder.add_edge(builder.start_node, cast("Any", NodeStep)(registry[start_node_id]))

        graph = builder.build()
        logger.info("Graph created successfully, type: %s", type(graph))

        # Store metadata separately
        self._graph_metadata[id(graph)] = GraphMetadata(
            node_schemas=node_map,
            edge_map=edge_map,
            name=graph_name,
            node_classes=registry,
            start_node_id=start_node_id,
        )

        logger.info(
            "pydantic-graph 2 built with %d node classes: %s",
            len(node_classes),
            graph_name,
        )
        return graph

    def create_start_node(
        self,
        graph: Graph[WorkflowState, None, Any, Any],
        start_node_id: str | None = None,
    ) -> BaseNode[WorkflowState, None, Any]:
        """Create the starting node instance following CORRECT pydantic-graph patterns."""
        metadata = self._graph_metadata.get(id(graph))
        node_schemas = metadata.node_schemas if metadata else {}
        edge_map = metadata.edge_map if metadata else {}

        start_node_id = self._determine_start_node(start_node_id, node_schemas, edge_map)
        if metadata and metadata.start_node_id and start_node_id != metadata.start_node_id:
            raise ValueError(
                f"Graph was built to start at {metadata.start_node_id!r}, not {start_node_id!r}; "
                "pass start_node_id to create_graph()"
            )

        return self._find_and_create_start_node_instance(graph, start_node_id)

    def _determine_start_node(
        self,
        start_node_id: str | None,
        node_schemas: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        *,
        allow_cycle_fallback: bool = False,
    ) -> str:
        """Determine the start node ID from provided ID or graph structure."""
        if start_node_id:
            if start_node_id not in node_schemas:
                raise ValueError(f"Start node {start_node_id} not found in graph")
            return start_node_id

        # Find node with no incoming edges
        target_nodes = set()
        for edges_list in edge_map.values():
            target_nodes.update(e["target"] for e in edges_list)

        start_candidates = [node_id for node_id in node_schemas if node_id not in target_nodes]

        if not start_candidates:
            if not allow_cycle_fallback or not node_schemas:
                raise ValueError("No valid start node found (no nodes without incoming edges)")
            # Building needs *an* edge out of StartNode, and a cyclic schema still
            # has to be constructible (pydantic-graph 1 could build it too). Pick
            # the first node; asking to execute from a resolved start still raises.
            logger.warning(
                "Graph has no node without incoming edges; wiring the start edge "
                "to the first node so it can be built"
            )
            return next(iter(node_schemas))

        return start_candidates[0]

    def _find_and_create_start_node_instance(
        self, graph: Graph[WorkflowState, None, Any, Any], start_node_id: str
    ) -> BaseNode[WorkflowState, None, Any]:
        """Instantiate the registered class for `start_node_id`.

        pydantic-graph 2 keeps its node registry private, so the factory records
        the class per schema id in `GraphMetadata` at build time instead of
        reading `Graph._nodes` back out.
        """
        metadata = self._graph_metadata.get(id(graph))
        node_class = metadata.node_classes.get(start_node_id) if metadata else None
        if node_class is None:
            raise ValueError(f"Could not find node class for start node {start_node_id}")
        return node_class()

    def _create_node_classes(
        self,
        nodes: list[NodeSchema],
        node_map: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
    ) -> list[type[BaseNode[WorkflowState, None, Any]]]:
        """Create node classes for all provided node schemas.

        Args:
            nodes: List of node schemas to create classes for
            node_map: Mapping of node IDs to schemas
            edge_map: Mapping of source nodes to target node IDs (with conditions)

        Returns:
            List of BaseNode class types
        """
        node_classes = []
        for node_schema in nodes:
            node_class = self._create_node_class(node_schema, node_map, edge_map)
            node_classes.append(node_class)

        # Build node class registry: map node schema ID -> class
        node_class_registry: dict[str, type[BaseNode[WorkflowState, None, Any]]] = {}
        for nc in node_classes:
            # Each class creates instances with _node_schema_id; extract it from the default.
            # The attribute is a dynamically-added dataclass field, opaque to the type checker.
            instance: Any = nc()
            node_class_registry[instance._node_schema_id] = nc

        # Inject the registry into every node class so run() can look up siblings.
        for nc in node_classes:
            nc_any: Any = nc
            nc_any._node_class_registry = node_class_registry

        return node_classes

    def _create_node_class(
        self,
        schema: NodeSchema,
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create a node CLASS that follows official pydantic_graph patterns.

        This creates a proper node class that:
        1. Inherits from BaseNode[WorkflowState, None, ReturnType]
        2. Has embedded routing logic in the run() method
        3. Returns next node instances or End() as per official docs

        Args:
            schema: Node schema definition
            all_nodes: Mapping of all node IDs to schemas
            edge_map: Mapping of source nodes to target node IDs (with conditions)

        Returns:
            Node class type appropriate for the schema's node_type
        """
        # Get next edges (with conditions) and extract node IDs for routing
        next_edges = edge_map.get(str(schema.id), [])
        next_node_ids = [e["target"] for e in next_edges]

        node_type_creators = {
            "worker": self._create_worker_node_class,
            "function": self._create_function_node_class,
            "condition": self._create_condition_node_class,
            "inference_provider": self._create_inference_provider_node_class,
            "load_adapter": self._create_load_adapter_node_class,
            "eval_subgraph": self._create_eval_subgraph_node_class,
        }

        creator = node_type_creators.get(schema.node_type)
        if creator is None:
            raise ValueError(f"Unsupported node type: {schema.node_type}")

        return creator(schema, next_node_ids, all_nodes, edge_map, next_edges)

    @classmethod
    def _create_inference_provider_node_class(
        cls,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create an InferenceProviderNode class for the given schema.

        Thin dispatch wrapper around
        :func:`src.worker.graphs.nodes.inference_provider_node.create_inference_provider_node_class`
        so the actual closure + dataclass definition lives in the nodes
        package.
        """
        from src.worker.graphs.nodes.inference_provider_node import (
            create_inference_provider_node_class,
        )

        return create_inference_provider_node_class(
            schema, next_node_ids, all_nodes, edge_map, next_edges
        )

    @classmethod
    def _create_load_adapter_node_class(
        cls,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create a LoadAdapterNode class for the given schema.

        Thin dispatch wrapper around
        :func:`src.worker.graphs.nodes.load_adapter_node.create_load_adapter_node_class`.
        """
        from src.worker.graphs.nodes.load_adapter_node import (
            create_load_adapter_node_class,
        )

        return create_load_adapter_node_class(
            schema, next_node_ids, all_nodes, edge_map, next_edges
        )

    @classmethod
    def _create_eval_subgraph_node_class(
        cls,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create an EvalSubgraphNode class for the given schema.

        Thin dispatch wrapper around
        :func:`src.worker.graphs.nodes.eval_subgraph_node.create_eval_subgraph_node_class`.
        """
        from src.worker.graphs.nodes.eval_subgraph_node import (
            create_eval_subgraph_node_class,
        )

        return create_eval_subgraph_node_class(
            schema, next_node_ids, all_nodes, edge_map, next_edges
        )

    def _create_worker_node_class(
        self,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create a WorkerNode CLASS following official pydantic_graph patterns.

        Args:
            schema: Node schema for this worker node
            next_node_ids: List of next node IDs for routing
            all_nodes: All node schemas in the graph
            edge_map: Edge mapping for navigation
            next_edges: List of edge dicts with target and condition

        Returns:
            WorkerNode class type
        """
        if next_edges is None:
            next_edges = [{"target": nid, "condition": None} for nid in next_node_ids]

        # Capture worker manager in closure
        def get_worker_manager_func() -> WorkerManager:
            return self._get_worker_manager()

        @dataclass
        class WorkerNode(BaseNode[WorkflowState, None, str]):
            """Official pydantic_graph WorkerNode implementation."""

            # Class metadata
            _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
            _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
            _all_nodes: dict[str, NodeSchema] = field(default_factory=all_nodes.copy, init=False)
            _edge_map: dict[str, list[dict[str, Any]]] = field(
                default_factory=edge_map.copy, init=False
            )
            _next_edges: list[dict[str, Any]] = field(
                default_factory=lambda: [e.copy() for e in next_edges], init=False
            )

            def _evaluate_edge_conditions(
                self, ctx: GraphRunContext[WorkflowState, None], result: Any
            ) -> str:
                """Evaluate edge conditions to determine next node.

                Loops through edges with conditions, evaluates each against the
                current result using safe AST evaluation (same as ConditionNode).
                Edges without conditions serve as the default fallback.
                """
                eval_context: dict[str, Any] = {}
                if hasattr(ctx.state, "variables") and ctx.state.variables:
                    eval_context.update(ctx.state.variables)
                if hasattr(ctx.state, "node_outputs") and ctx.state.node_outputs:
                    eval_context.update(ctx.state.node_outputs)
                    for node_id, output in ctx.state.node_outputs.items():
                        node_schema = self._all_nodes.get(node_id)
                        if node_schema and hasattr(node_schema, "name"):
                            eval_context[node_schema.name] = output
                eval_context["result"] = result

                default_target = self._next_node_ids[0]  # fallback

                for edge in self._next_edges:
                    condition = edge.get("condition")
                    if not condition:
                        default_target = edge["target"]
                        continue
                    try:
                        tree = ast.parse(condition, mode="eval")
                        safe_node_types = (
                            ast.Compare,
                            ast.BoolOp,
                            ast.UnaryOp,
                            ast.Attribute,
                            ast.Name,
                            ast.Constant,
                            ast.List,
                            ast.Tuple,
                            ast.Dict,
                            ast.And,
                            ast.Or,
                            ast.Not,
                            ast.Eq,
                            ast.NotEq,
                            ast.Lt,
                            ast.LtE,
                            ast.Gt,
                            ast.GtE,
                            ast.In,
                            ast.NotIn,
                            ast.Is,
                            ast.IsNot,
                        )
                        for node in ast.walk(tree):
                            if not isinstance(node, (ast.Expression, ast.Load, *safe_node_types)):
                                raise ValueError(  # noqa: RUF100, TRY004
                                    f"Unsafe AST node type in edge condition: {type(node).__name__}"
                                )
                        cond_result = eval(  # nosec B307 — controlled input  # noqa: S307
                            compile(tree, "<edge_condition>", "eval"),
                            {
                                "__builtins__": {
                                    "True": True,
                                    "False": False,
                                    "None": None,
                                }
                            },
                            eval_context,
                        )
                        if bool(cond_result):
                            logger.info(
                                "WorkerNode %s: edge condition '%s' matched, routing to %s",
                                self._node_schema_id,
                                condition,
                                edge["target"],
                            )
                            return edge["target"]
                    except Exception as e:
                        logger.warning(
                            "WorkerNode %s: edge condition '%s' eval failed: %s",
                            self._node_schema_id,
                            condition,
                            e,
                        )
                        continue

                return default_target

            def _build_template_context(
                self, ctx: GraphRunContext[WorkflowState, None]
            ) -> dict[str, Any]:
                """Build template context from state variables and node outputs."""
                template_context = {}

                # Add state variables
                if hasattr(ctx.state, "variables") and ctx.state.variables:
                    template_context.update(ctx.state.variables)

                # Add node outputs (for template references like {node_name.result})
                if hasattr(ctx.state, "node_outputs") and ctx.state.node_outputs:
                    # Map node outputs by both ID and name for template access
                    for node_id, output in ctx.state.node_outputs.items():
                        # Add by node ID
                        template_context[node_id] = output

                        # Try to find node name and add by name for easier templates
                        node_schema = self._all_nodes.get(node_id)
                        if node_schema and hasattr(node_schema, "name"):
                            # Add with .result suffix for backwards compatibility
                            template_context[f"{node_schema.name}.result"] = output

                return template_context

            async def run(
                self, ctx: GraphRunContext[WorkflowState, None]
            ) -> BaseNode[WorkflowState, None, Any] | End[str]:
                """Execute worker task and route to next node - OFFICIAL pydantic_graph pattern."""
                # Get WorkerManager instance
                try:
                    worker_manager = get_worker_manager_func()
                except RuntimeError:
                    logger.exception("Cannot proceed without WorkerManager")
                    raise

                worker_name = schema.worker_name or ""
                task_template = schema.task_template or ""

                logger.info("OFFICIAL WorkerNode executing: %s", worker_name)

                try:
                    # Execute the worker task

                    # Render task template with current state variables and node outputs
                    template_context = self._build_template_context(ctx)
                    rendered_task = task_template

                    try:
                        rendered_task = task_template.format(**template_context)
                        logger.debug("Template rendered: %s -> %s", task_template, rendered_task)
                    except (KeyError, ValueError) as e:
                        logger.warning("Template rendering failed: %s, using original template", e)

                    # Send message to worker. Forward the run's identity so the
                    # loopback POST to /workers/{name}/message carries a raw
                    # X-Tenant-Id/X-Subject-Id: the auth middleware's trusted-network
                    # branch then enforces the tenant's byo_required posture on this
                    # LLM stage instead of falling back to DEFAULT_TENANT_ID
                    # (local_allowed) and silently running on platform GPU.
                    result = await worker_manager.send_message_to_worker(
                        worker_name=worker_name,
                        message=rendered_task,
                        context={"node_id": str(schema.id)},
                        tenant_id=getattr(ctx.state, "tenant_id", None),
                        subject_id=getattr(ctx.state, "subject_id", None),
                    )

                    # Store result in state
                    if hasattr(ctx.state, "node_outputs"):
                        ctx.state.node_outputs[str(schema.id)] = result

                    # Lift a validator-emitted quality_score (e.g. qa_validator)
                    # into the inference bag so the finalize path persists it onto
                    # graph_execution_sessions.quality_score. The structured worker
                    # output arrives under result["output"] (see WorkerManager);
                    # workers that don't emit the field are untouched. This is the
                    # WorkerNode the /execution engine actually runs — the standalone
                    # nodes/worker_node.WorkerNode is not on this path.
                    if isinstance(result, dict):
                        record_quality_score(
                            ctx.state,
                            result.get("output"),
                            worker_name=worker_name,
                            node_id=str(schema.id),
                        )

                    response_text = (
                        result.get("response", "No response")
                        if isinstance(result, dict)
                        else str(result)
                    )
                    logger.info(
                        "OFFICIAL WorkerNode %s completed with result length: %d",
                        worker_name,
                        len(response_text),
                    )

                    # Route to next node with edge condition evaluation
                    if not self._next_node_ids:
                        logger.info("OFFICIAL WorkerNode %s ending execution", worker_name)
                        return End(response_text)

                    # Evaluate edge conditions to pick the right next node
                    next_node_id = self._evaluate_edge_conditions(ctx, result)
                    next_node_schema = self._all_nodes.get(next_node_id)

                    if not next_node_schema:
                        logger.error("Next node %s not found", next_node_id)
                        return End(response_text)

                    # Look up next node class from registry and return instance
                    registry = getattr(type(self), "_node_class_registry", {})
                    next_node_class = registry.get(next_node_id)
                    if next_node_class is None:
                        logger.error(
                            "WorkerNode %s: next node class %s not in registry",
                            worker_name,
                            next_node_id,
                        )
                        return End(response_text)

                    logger.info(
                        "OFFICIAL WorkerNode %s routing to: %s (%s)",
                        worker_name,
                        next_node_id,
                        next_node_schema.node_type,
                    )
                    return next_node_class()

                except Exception as e:
                    logger.exception("OFFICIAL WorkerNode %s failed", worker_name)
                    return End("Error: %s" % str(e))

        # Set a unique name for the class
        WorkerNode.__name__ = f"WorkerNode_{schema.id}".replace("-", "_")
        WorkerNode.__qualname__ = WorkerNode.__name__

        return WorkerNode

    @classmethod
    def _create_function_node_class(
        cls,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create a FunctionNode CLASS following official pydantic_graph patterns.

        Args:
            schema: Node schema for this function node
            next_node_ids: List of next node IDs for routing
            all_nodes: All node schemas in the graph
            edge_map: Edge mapping for navigation
            next_edges: List of edge dicts with target and condition

        Returns:
            FunctionNode class type
        """
        if next_edges is None:
            next_edges = [{"target": nid, "condition": None} for nid in next_node_ids]

        @dataclass
        class FunctionNode(BaseNode[WorkflowState, None, Any]):
            """Official pydantic_graph FunctionNode implementation."""

            # Class metadata
            _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
            _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
            _all_nodes: dict[str, NodeSchema] = field(default_factory=all_nodes.copy, init=False)
            _edge_map: dict[str, list[dict[str, Any]]] = field(
                default_factory=edge_map.copy, init=False
            )
            _next_edges: list[dict[str, Any]] = field(
                default_factory=lambda: [e.copy() for e in next_edges], init=False
            )

            def _evaluate_edge_conditions(
                self, ctx: GraphRunContext[WorkflowState, None], result: Any
            ) -> str:
                """Evaluate edge conditions to determine next node.

                Same safe AST evaluation pattern as ConditionNode.
                """
                eval_context: dict[str, Any] = {}
                if hasattr(ctx.state, "variables") and ctx.state.variables:
                    eval_context.update(ctx.state.variables)
                if hasattr(ctx.state, "node_outputs") and ctx.state.node_outputs:
                    eval_context.update(ctx.state.node_outputs)
                    for node_id, output in ctx.state.node_outputs.items():
                        node_schema = self._all_nodes.get(node_id)
                        if node_schema and hasattr(node_schema, "name"):
                            eval_context[node_schema.name] = output
                eval_context["result"] = result

                default_target = self._next_node_ids[0]  # fallback

                for edge in self._next_edges:
                    condition = edge.get("condition")
                    if not condition:
                        default_target = edge["target"]
                        continue
                    try:
                        tree = ast.parse(condition, mode="eval")
                        safe_node_types = (
                            ast.Compare,
                            ast.BoolOp,
                            ast.UnaryOp,
                            ast.Attribute,
                            ast.Name,
                            ast.Constant,
                            ast.List,
                            ast.Tuple,
                            ast.Dict,
                            ast.And,
                            ast.Or,
                            ast.Not,
                            ast.Eq,
                            ast.NotEq,
                            ast.Lt,
                            ast.LtE,
                            ast.Gt,
                            ast.GtE,
                            ast.In,
                            ast.NotIn,
                            ast.Is,
                            ast.IsNot,
                        )
                        for node in ast.walk(tree):
                            if not isinstance(node, (ast.Expression, ast.Load, *safe_node_types)):
                                raise ValueError(  # noqa: RUF100, TRY004
                                    f"Unsafe AST node type in edge condition: {type(node).__name__}"
                                )
                        cond_result = eval(  # nosec B307 — controlled input  # noqa: S307
                            compile(tree, "<edge_condition>", "eval"),
                            {
                                "__builtins__": {
                                    "True": True,
                                    "False": False,
                                    "None": None,
                                }
                            },
                            eval_context,
                        )
                        if bool(cond_result):
                            logger.info(
                                "FunctionNode %s: edge condition '%s' matched, routing to %s",
                                self._node_schema_id,
                                condition,
                                edge["target"],
                            )
                            return edge["target"]
                    except Exception as e:
                        logger.warning(
                            "FunctionNode %s: edge condition '%s' eval failed: %s",
                            self._node_schema_id,
                            condition,
                            e,
                        )
                        continue

                return default_target

            async def run(
                self, ctx: GraphRunContext[WorkflowState, None]
            ) -> BaseNode[WorkflowState, None, Any] | End[Any]:
                """Execute function and route to next node - OFFICIAL pydantic_graph pattern."""
                logger.info("OFFICIAL FunctionNode executing: %s", schema.id)

                try:
                    # Request-supplied Python is NEVER executed. This node used
                    # to exec() ``data.function_code`` with ``__import__`` in its
                    # builtins (``import os`` worked). NodeSchema now refuses the
                    # field with a typed 422; this is the fail-closed twin for
                    # anything that reaches here without passing through it.
                    if has_function_code(schema.data):
                        raise FunctionCodeNotSupportedError(
                            f"Function node '{schema.id}' carries function_code; "
                            "request-supplied Python is never executed"
                        )

                    # No code runs: a function node passes its data context on.
                    data_context = {}
                    if hasattr(ctx.state, "variables") and ctx.state.variables:
                        data_context.update(ctx.state.variables)
                    if hasattr(ctx.state, "node_outputs") and ctx.state.node_outputs:
                        data_context.update(ctx.state.node_outputs)
                    result = data_context

                    # Store result in state
                    if hasattr(ctx.state, "node_outputs"):
                        ctx.state.node_outputs[str(schema.id)] = result

                    logger.info("OFFICIAL FunctionNode %s completed successfully", schema.id)

                    # Route to next node with edge condition evaluation
                    if not self._next_node_ids:
                        logger.info("OFFICIAL FunctionNode %s ending execution", schema.id)
                        return End(result)

                    # Evaluate edge conditions to pick the right next node
                    next_node_id = self._evaluate_edge_conditions(ctx, result)
                    next_node_schema = self._all_nodes.get(next_node_id)

                    if not next_node_schema:
                        logger.error("Next node %s not found", next_node_id)
                        return End(result)

                    # Look up next node class from registry and return instance
                    registry = getattr(type(self), "_node_class_registry", {})
                    next_node_class = registry.get(next_node_id)
                    if next_node_class is None:
                        logger.error(
                            "FunctionNode %s: next node class %s not in registry",
                            schema.id,
                            next_node_id,
                        )
                        return End(result)

                    logger.info(
                        "OFFICIAL FunctionNode %s routing to next node: %s (%s)",
                        schema.id,
                        next_node_id,
                        next_node_schema.node_type,
                    )
                    return next_node_class()

                except FunctionCodeNotSupportedError:
                    # Fail the execution, never End() it as if it completed.
                    raise
                except Exception as e:
                    logger.exception("OFFICIAL FunctionNode %s failed", schema.id)
                    return End("Error: %s" % str(e))

        # Set a unique name for the class
        FunctionNode.__name__ = f"FunctionNode_{schema.id}".replace("-", "_")
        FunctionNode.__qualname__ = FunctionNode.__name__

        return FunctionNode

    @classmethod
    def _create_condition_node_class(
        cls,
        schema: NodeSchema,
        next_node_ids: list[str],
        all_nodes: dict[str, NodeSchema],
        edge_map: dict[str, list[dict[str, Any]]],
        next_edges: list[dict[str, Any]] | None = None,  # noqa: ARG003, RUF100
    ) -> type[BaseNode[WorkflowState, None, Any]]:
        """Create a ConditionNode CLASS following official pydantic_graph patterns.

        Args:
            schema: Node schema for this condition node
            next_node_ids: List of next node IDs for routing
            all_nodes: All node schemas in the graph
            edge_map: Edge mapping for navigation
            next_edges: List of edge dicts with target and condition

        Returns:
            ConditionNode class type
        """

        @dataclass
        class ConditionNode(BaseNode[WorkflowState, None, bool]):
            """Official pydantic_graph ConditionNode implementation."""

            # Class metadata
            _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
            _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
            _all_nodes: dict[str, NodeSchema] = field(default_factory=all_nodes.copy, init=False)
            _edge_map: dict[str, list[dict[str, Any]]] = field(
                default_factory=edge_map.copy, init=False
            )

            async def run(
                self, ctx: GraphRunContext[WorkflowState, None]
            ) -> BaseNode[WorkflowState, None, Any] | End[bool]:
                """Execute condition and route to next node - OFFICIAL pydantic_graph pattern."""
                condition = schema.data.get("condition", "True") if schema.data else "True"

                logger.info("OFFICIAL ConditionNode evaluating: %s", condition)

                try:
                    # Create safe evaluation environment
                    eval_context = {}
                    if hasattr(ctx.state, "variables") and ctx.state.variables:
                        eval_context.update(ctx.state.variables)
                    if hasattr(ctx.state, "node_outputs") and ctx.state.node_outputs:
                        eval_context.update(ctx.state.node_outputs)

                        # Also add node outputs by node name for easier reference
                        for node_id, output in ctx.state.node_outputs.items():
                            node_schema = self._all_nodes.get(node_id)
                            if node_schema and hasattr(node_schema, "name"):
                                eval_context[node_schema.name] = output

                    # Evaluate condition safely
                    tree = ast.parse(condition, mode="eval")

                    # SECURITY: Validate AST only contains safe operations
                    # Only allow: comparisons, boolean ops, attribute access, literals
                    safe_node_types = (
                        ast.Compare,
                        ast.BoolOp,
                        ast.UnaryOp,
                        ast.Attribute,
                        ast.Name,
                        ast.Constant,
                        ast.List,
                        ast.Tuple,
                        ast.Dict,
                        ast.And,
                        ast.Or,
                        ast.Not,
                        ast.Eq,
                        ast.NotEq,
                        ast.Lt,
                        ast.LtE,
                        ast.Gt,
                        ast.GtE,
                        ast.In,
                        ast.NotIn,
                        ast.Is,
                        ast.IsNot,
                    )

                    for node in ast.walk(tree):
                        if not isinstance(node, (ast.Expression, ast.Load, *safe_node_types)):
                            raise ValueError(  # noqa: RUF100, TRY004
                                f"SECURITY: Unsafe AST node type in condition: {type(node).__name__}"
                            )

                    # Simple AST evaluation for safety
                    # SECURITY: Very restricted __builtins__ (only True/False/None)
                    result = eval(  # nosec B307 — controlled input  # noqa: S307
                        compile(tree, "<condition>", "eval"),
                        {"__builtins__": {"True": True, "False": False, "None": None}},
                        eval_context,
                    )
                    result = bool(result)

                    # Store result in state
                    if hasattr(ctx.state, "node_outputs"):
                        ctx.state.node_outputs[str(schema.id)] = result

                    logger.info("OFFICIAL ConditionNode %s evaluated to: %s", schema.id, result)

                    # Route to next node following OFFICIAL pattern - FIXED
                    if not self._next_node_ids:
                        logger.info("OFFICIAL ConditionNode %s ending execution", schema.id)
                        return End(result)

                    # For condition nodes, route based on result:
                    #   True  -> first edge  (index 0)
                    #   False -> second edge (index 1), or End if no false branch
                    if result:
                        branch_node_id = self._next_node_ids[0]
                    elif len(self._next_node_ids) > 1:
                        branch_node_id = self._next_node_ids[1]
                    else:
                        logger.info(
                            "OFFICIAL ConditionNode %s evaluated False with no false branch, ending",
                            schema.id,
                        )
                        return End(result)

                    branch_node_schema = self._all_nodes.get(branch_node_id)

                    # Look up branch node class from registry and return instance
                    registry = getattr(type(self), "_node_class_registry", {})
                    branch_node_class = registry.get(branch_node_id)
                    if branch_node_class is None:
                        logger.error(
                            "ConditionNode %s: branch node class %s not in registry",
                            schema.id,
                            branch_node_id,
                        )
                        return End(result)

                    logger.info(
                        "OFFICIAL ConditionNode %s (result: %s) routing to branch node: %s (%s)",
                        schema.id,
                        result,
                        branch_node_id,
                        (branch_node_schema.node_type if branch_node_schema else "unknown"),
                    )
                    return branch_node_class()

                except Exception:
                    logger.exception("OFFICIAL ConditionNode %s failed", schema.id)
                    return End(False)  # noqa: FBT003, RUF100

        # Set a unique name for the class
        ConditionNode.__name__ = f"ConditionNode_{schema.id}".replace("-", "_")
        ConditionNode.__qualname__ = ConditionNode.__name__

        return ConditionNode

    @classmethod
    def _build_edge_map(cls, edges: list[EdgeSchema]) -> dict[str, list[dict[str, Any]]]:
        """Build mapping from source node to target node IDs with edge conditions."""
        edge_map: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            source_id = str(edge.source)
            target_id = str(edge.target)
            if source_id not in edge_map:
                edge_map[source_id] = []
            edge_map[source_id].append({"target": target_id, "condition": edge.condition})
        return edge_map

    @staticmethod
    def validate_graph_structure(nodes: list[NodeSchema], edges: list[EdgeSchema]) -> list[str]:
        """Validate graph structure and return list of warnings/errors."""
        issues = []

        if not nodes:
            issues.append("Graph must have at least one node")
            return issues

        node_ids = {str(node.id) for node in nodes}

        # Validate edges reference existing nodes
        for edge in edges:
            if str(edge.source) not in node_ids:
                issues.append(f"Edge source '{edge.source}' not found in nodes")
            if str(edge.target) not in node_ids:
                issues.append(f"Edge target '{edge.target}' not found in nodes")

        # Validate node configurations
        for node in nodes:
            if node.node_type == "worker" and not node.worker_name:
                issues.append(f"Worker node '{node.id}' missing worker_name")
            elif node.node_type == "function":
                if has_function_code(node.data):
                    issues.append(
                        f"Function node '{node.id}' carries function_code, which is not supported"
                    )
            elif node.node_type == "condition":  # noqa: RUF100, SIM102
                if not node.data or not node.data.get("condition"):
                    issues.append(f"Condition node '{node.id}' missing condition")

        return issues
