"""Smoke tests for graph-factory dispatch of Task 4 node types.

Asserts that ``GraphFactory._create_node_classes`` can build both a
``load_adapter`` and an ``eval_subgraph`` node from a schema dict; we
do NOT run the resulting classes here — just verify that dispatch works
and the returned object is a ``type[BaseNode]`` subclass.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_graph import BaseNode

from src.worker.graphs.graph_factory import GraphFactory
from src.worker.graphs.graph_models import NodeSchema


def _build_classes(nodes: list[NodeSchema]) -> dict[str, type[BaseNode[Any, Any, Any]]]:
    factory = GraphFactory(worker_manager=None)
    node_map = {str(n.id): n for n in nodes}
    edge_map: dict[str, list[dict[str, Any]]] = {}
    classes = factory._create_node_classes(nodes, node_map, edge_map)
    return {cast("Any", cls())._node_schema_id: cls for cls in classes}


class TestGraphFactoryDispatch:
    def test_load_adapter_dispatch_returns_basenode_subclass(self) -> None:
        node = NodeSchema(
            name="loader",
            node_type="load_adapter",
            framework_config={
                "adapter_id": "lora-xyz",
                "adapter_manager_url": "http://adapter-manager.test:8001",
                "require_adapter": False,
                "timeout": 5,
            },
        )
        classes = _build_classes([node])
        cls_ = classes[str(node.id)]
        assert isinstance(cls_, type)
        assert issubclass(cls_, BaseNode)
        # The per-schema closure captures adapter config.
        inst = cls_()
        assert cast("Any", inst)._adapter_config["adapter_id"] == "lora-xyz"

    def test_eval_subgraph_dispatch_returns_basenode_subclass(self) -> None:
        node = NodeSchema(
            name="evaler",
            node_type="eval_subgraph",
            framework_config={
                "validation_mode": "json_schema",
                "json_schema": {"type": "object"},
            },
        )
        classes = _build_classes([node])
        cls_ = classes[str(node.id)]
        assert isinstance(cls_, type)
        assert issubclass(cls_, BaseNode)
        inst = cls_()
        assert cast("Any", inst)._eval_config["validation_mode"] == "json_schema"

    def test_both_dispatch_in_single_graph(self) -> None:
        loader = NodeSchema(
            name="loader2",
            node_type="load_adapter",
            framework_config={"adapter_id": "adp-1"},
        )
        evaler = NodeSchema(
            name="evaler2",
            node_type="eval_subgraph",
            framework_config={
                "validation_mode": "json_schema",
                "json_schema": {"type": "object"},
            },
        )
        classes = _build_classes([loader, evaler])
        assert set(classes.keys()) == {str(loader.id), str(evaler.id)}
        for cls_ in classes.values():
            assert issubclass(cls_, BaseNode)
