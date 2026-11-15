"""
Schemathesis hooks to handle nullable parameter generation and path parameter substitution.

Schemathesis generates literal strings "None" or "null" for nullable parameters,
which FastAPI correctly rejects. These hooks filter out such invalid values.

Also substitutes known test data IDs for path parameters so endpoints don't 404.
"""

import json
import os
import random
from typing import Any

import schemathesis

# Load test IDs at module level for path parameter substitution
_test_ids: dict[str, Any] = {}
_test_ids_path = os.path.join(os.path.dirname(__file__), "../../test_ids.json")
if os.path.exists(_test_ids_path):
    with open(_test_ids_path) as f:
        _test_ids = json.load(f)


@schemathesis.hook
def filter_query(
    _context: schemathesis.hooks.HookContext,
    query: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Filter query parameters to remove null-like string values.

    Schemathesis generates "null" strings for nullable enum parameters.
    FastAPI expects these to be omitted rather than sent as literal "null".
    """
    if query is None:
        return None

    return {k: v for k, v in query.items() if v not in ("null", "None", None, "")}


@schemathesis.hook
def filter_body(
    _context: schemathesis.hooks.HookContext,
    body: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Filter body/form parameters to remove null-like string values.

    Schemathesis generates "None" strings for nullable UUID parameters.
    FastAPI expects these to be omitted rather than sent as literal "None".
    """
    if body is None:
        return None

    if not isinstance(body, dict):
        return body

    return {k: v for k, v in body.items() if v not in ("null", "None", None)}


@schemathesis.hook
def filter_path_parameters(
    _context: schemathesis.hooks.HookContext,
    path_parameters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Substitute known test data values for path parameters.

    Schemathesis generates random values for path params like {worker_name},
    {graph_id}, etc. which always 404. This hook replaces them with known
    test data IDs loaded from test_ids.json.
    """
    if path_parameters is None:
        return None

    worker_names = _test_ids.get("worker_names", [])
    graph_ids = _test_ids.get("graph_ids", [])
    tool_names = _test_ids.get("tool_names", [])

    if "worker_name" in path_parameters and worker_names:
        path_parameters["worker_name"] = random.choice(worker_names)
    if "graph_id" in path_parameters and graph_ids:
        path_parameters["graph_id"] = random.choice(graph_ids)
    if "tool_name" in path_parameters and tool_names:
        path_parameters["tool_name"] = random.choice(tool_names)

    return path_parameters
