"""LoadAdapterNode — ensures a LoRA adapter is loaded on the inference host.

This module provides the factory that builds per-schema
``LoadAdapterNode`` classes for the pydantic-graph execution engine.

Design
------
The node POSTs to the adapter-manager sidecar (``/adapters/load``) to
make sure a specific LoRA adapter is resident before the downstream
provider chain runs. Behaviour on failure is driven by the schema's
``require_adapter`` flag:

* ``require_adapter=False`` (default): failure is recorded in
  ``ctx.state.framework_state["adapter_loads"]`` and the chain continues
  to the next node — the downstream provider will attempt without the
  adapter.
* ``require_adapter=True``: failure short-circuits the graph with
  ``End({"status": "failed", ...})`` so that nothing further runs.

If no adapter-manager URL is configured (``WM_ADAPTER_MANAGER_URL`` unset
and no per-schema override) the node makes NO HTTP call: it records a
``status="not_configured"`` / ``error_code="ADAPTER_MANAGER_NOT_CONFIGURED"``
entry, marks the node failed, and routes exactly like a load failure
(``require_adapter`` decides between ``End`` and continuing). A URL whose
host is in ``DIRECT_GPU_HOSTS`` (e.g. a raw vLLM host) is refused at
graph-build time — applications never call vLLM directly.

On success we record a success entry in the same ``adapter_loads`` list
and hand off to the next node in declared order.

This node does not evaluate edge conditions — it has a single logical
successor in the Phase 1 inference-graph layout (adapter → providers →
eval). When zero successors exist the node ends the run.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from pydantic_graph import BaseNode, End, GraphRunContext

from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from src.worker.graphs.graph_models import NodeSchema

logger = logging.getLogger(__name__)

ADAPTER_MANAGER_NOT_CONFIGURED = "ADAPTER_MANAGER_NOT_CONFIGURED"


def _default_adapter_manager_url() -> str:
    """Resolve the adapter-manager base URL from Settings.

    Returns an empty string when unset — the adapter manager is optional.
    Callers treat empty as "not configured" and report
    ``ADAPTER_MANAGER_NOT_CONFIGURED`` without making any HTTP call.
    """
    from src.settings import get_settings

    return get_settings().wm_adapter_manager_url


def _get_adapter_config(schema: NodeSchema) -> dict[str, Any]:
    """Read and normalise the adapter config from ``schema.framework_config``.

    Raises:
        ValueError: if ``adapter_id`` is missing or not a non-empty string, or
            if the adapter-manager URL targets a ``DIRECT_GPU_HOSTS`` host.
    """
    from src.settings import DIRECT_GPU_HOSTS, is_direct_gpu_url

    raw = schema.framework_config or {}
    adapter_id = raw.get("adapter_id")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError(
            f"Load-adapter node '{schema.name}' missing 'adapter_id' in framework_config"
        )

    manager_url = raw.get("adapter_manager_url")
    if not isinstance(manager_url, str) or not manager_url:
        manager_url = os.environ.get("WM_ADAPTER_MANAGER_URL") or _default_adapter_manager_url()
    if is_direct_gpu_url(manager_url):
        raise ValueError(
            f"Load-adapter node '{schema.name}' adapter_manager_url {manager_url!r} targets "
            f"a direct GPU host ({sorted(DIRECT_GPU_HOSTS)}); route through inference-router"
        )

    require_adapter = bool(raw.get("require_adapter", False))
    timeout = int(raw.get("timeout", 10))

    return {
        "adapter_id": adapter_id,
        "adapter_manager_url": manager_url,
        "require_adapter": require_adapter,
        "timeout": timeout,
    }


def create_load_adapter_node_class(
    schema: NodeSchema,
    next_node_ids: list[str],
    _all_nodes: dict[str, NodeSchema],
    _edge_map: dict[str, list[dict[str, Any]]],
    _next_edges: list[dict[str, Any]] | None = None,
) -> type[BaseNode[WorkflowState, None, Any]]:
    """Build a per-schema ``LoadAdapterNode`` class.

    Mirrors the closure-based pattern used by
    ``create_inference_provider_node_class`` so schema metadata is
    captured in the class definition itself.

    The ``_all_nodes`` / ``_edge_map`` / ``_next_edges`` parameters are
    accepted to match the factory-dispatch signature in
    :class:`GraphFactory` but are not needed here — this node has a
    single logical successor and routes via ``_next_node_ids`` +
    ``_node_class_registry`` only.
    """
    adapter_config = _get_adapter_config(schema)

    @dataclass
    class LoadAdapterNode(BaseNode[WorkflowState, None, Any]):
        """Per-schema load-adapter node.

        POSTs to the adapter-manager ``/adapters/load`` endpoint and then
        either continues the chain (success or non-required failure) or
        short-circuits with ``End`` (required-failure).
        """

        _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
        _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
        _adapter_config: dict[str, Any] = field(
            default_factory=lambda: dict(adapter_config), init=False
        )

        def _next_successor_cls(
            self,
        ) -> type[BaseNode[WorkflowState, None, Any]] | None:
            """Return the first registered successor class, or None."""
            registry = getattr(type(self), "_node_class_registry", {})
            for next_id in self._next_node_ids:
                cls_ = registry.get(next_id)
                if cls_ is not None:
                    return cls_
            return None

        def _record_load(self, ctx: GraphRunContext[WorkflowState], entry: dict[str, Any]) -> None:
            """Append a load-attempt record to ``framework_state['adapter_loads']``."""
            bag = ctx.state.framework_state.setdefault("adapter_loads", [])
            if isinstance(bag, list):
                bag.append(entry)
            else:
                # Existing value is not a list — replace with a fresh list
                # rather than crash. This keeps the state hygienic across
                # malformed prior writes.
                ctx.state.framework_state["adapter_loads"] = [entry]

        def _not_configured(
            self, ctx: GraphRunContext[WorkflowState], adapter_id: str
        ) -> BaseNode[WorkflowState, None, Any] | End[Any]:
            """Fail the load step without any HTTP call: no adapter manager is set."""
            message = (
                "adapter manager not configured (WM_ADAPTER_MANAGER_URL unset); "
                f"cannot load adapter '{adapter_id}'"
            )
            logger.warning("LoadAdapterNode %s: %s", self._node_schema_id, message)
            self._record_load(
                ctx,
                {
                    "adapter_id": adapter_id,
                    "status": "not_configured",
                    "error_code": ADAPTER_MANAGER_NOT_CONFIGURED,
                    "latency_ms": 0,
                    "error": message,
                },
            )
            ctx.state.mark_node_failed(self._node_schema_id, message)
            failed = End(
                {
                    "status": "failed",
                    "error": "adapter manager not configured",
                    "error_code": ADAPTER_MANAGER_NOT_CONFIGURED,
                    "adapter_id": adapter_id,
                }
            )
            if self._adapter_config["require_adapter"]:
                return failed
            successor = self._next_successor_cls()
            return successor() if successor is not None else failed

        async def run(
            self, ctx: GraphRunContext[WorkflowState]
        ) -> BaseNode[WorkflowState, None, Any] | End[Any]:
            """POST to the adapter-manager and route the chain forward."""
            cfg = self._adapter_config
            adapter_id = cfg["adapter_id"]
            if not cfg["adapter_manager_url"]:
                return self._not_configured(ctx, adapter_id)
            url = f"{cfg['adapter_manager_url'].rstrip('/')}/adapters/load"

            t0 = time.monotonic()
            client = httpx.AsyncClient(timeout=cfg["timeout"])
            try:
                try:
                    response = await client.post(url, json={"adapter_id": adapter_id})
                    response.raise_for_status()
                except (TimeoutError, httpx.HTTPError) as e:
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    logger.warning(
                        "LoadAdapterNode %s failed loading adapter '%s' after %dms: %s",
                        self._node_schema_id,
                        adapter_id,
                        latency_ms,
                        e,
                    )
                    self._record_load(
                        ctx,
                        {
                            "adapter_id": adapter_id,
                            "status": "error",
                            "latency_ms": latency_ms,
                            "error": str(e),
                        },
                    )
                    ctx.state.mark_node_failed(self._node_schema_id, str(e))

                    if cfg["require_adapter"]:
                        return End(
                            {
                                "status": "failed",
                                "error": "adapter load failed",
                                "adapter_id": adapter_id,
                            }
                        )

                    successor = self._next_successor_cls()
                    if successor is not None:
                        return successor()
                    return End(
                        {
                            "status": "failed",
                            "error": "adapter load failed",
                            "adapter_id": adapter_id,
                        }
                    )
            finally:
                await client.aclose()

            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "LoadAdapterNode %s loaded adapter '%s' in %dms",
                self._node_schema_id,
                adapter_id,
                latency_ms,
            )
            self._record_load(
                ctx,
                {
                    "adapter_id": adapter_id,
                    "status": "ok",
                    "latency_ms": latency_ms,
                },
            )
            ctx.state.set_node_output(
                self._node_schema_id,
                {"adapter_id": adapter_id, "loaded": True},
            )

            successor = self._next_successor_cls()
            if successor is not None:
                return successor()
            return End({"adapter_id": adapter_id, "loaded": True})

    LoadAdapterNode.__name__ = f"LoadAdapterNode_{schema.id}".replace("-", "_")
    LoadAdapterNode.__qualname__ = LoadAdapterNode.__name__
    return LoadAdapterNode
