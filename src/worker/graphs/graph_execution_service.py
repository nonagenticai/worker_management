"""Backward-compatibility shim.

All functionality has been merged into pydantic_graph_execution_service.py.
This module re-exports the public names so existing imports keep working.
"""

from src.worker.graphs.pydantic_graph_execution_service import (
    GraphExecutionService,
    PydanticGraphExecutionService,
    get_execution_service,
)

__all__ = [
    "GraphExecutionService",
    "PydanticGraphExecutionService",
    "get_execution_service",
]
