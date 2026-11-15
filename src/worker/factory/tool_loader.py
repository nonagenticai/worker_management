"""Tool loading logic.

Functions that fetch tools from the tool registry service
and handle tool configuration building.
"""

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MockToolTemplate:
    """Mock implementation of ToolTemplate for when import fails."""

    @staticmethod
    def from_json(data: Any) -> MockToolTemplate:  # noqa: ARG004, RUF100
        """Create a mock tool template from JSON data."""
        return MockToolTemplate()

    def create_tool(self) -> None:
        """Create a mock tool (returns None)."""
        return None


# Import ToolTemplate with fallback to mock
try:
    from src.worker.agents.tool_template import ToolTemplate
except ImportError:
    logger.exception("Could not import ToolTemplate")
    ToolTemplate: type[MockToolTemplate] = MockToolTemplate


async def _fetch_single_tool(tool_name: str, client: httpx.AsyncClient) -> Any | None:
    """Fetch a single tool from the tool registry.

    Args:
        tool_name: Name of the tool to fetch
        client: HTTP client for making requests

    Returns:
        Tool instance if successfully fetched, None otherwise
    """
    try:
        tool_registry_base_url = os.getenv("TOOL_REGISTRY_BASE_URL", "http://localhost:8889")
        tool_url = f"{tool_registry_base_url}/tools/{tool_name}"
        logger.debug("Fetching tool %s from %s", tool_name, tool_url)
        response = await client.get(tool_url)
        response.raise_for_status()
        tool_data = response.json()
        tool_template = ToolTemplate.from_json(tool_data)
        return tool_template.create_tool()
    except httpx.HTTPStatusError as http_error:
        logger.exception(
            "HTTP error fetching tool %s: %s - %s",
            tool_name,
            http_error.response.status_code,
            http_error.response.text,
        )
        return None
    except httpx.RequestError:
        logger.exception("Request error fetching tool %s", tool_name)
        return None
    except json.JSONDecodeError:
        logger.exception("JSON decode error for tool %s", tool_name)
        return None


async def _load_worker_tools(tool_names: list[str], worker_name: str) -> list[Any]:
    """Load tools from the tool registry for a worker.

    Loads tools concurrently with sensible limits and timeouts to improve
    performance while keeping resource usage controlled.

    Args:
        tool_names: List of tool names to load
        worker_name: Name of the worker for logging

    Returns:
        List of loaded tool instances
    """
    # Normalize and de-duplicate tool names early
    if not tool_names:
        return []

    normalized_names: list[str] = []
    seen = set()
    for name in tool_names:
        if not name or not isinstance(name, str):
            continue
        trimmed = name.strip()
        if not trimmed:
            logger.warning("Skipping empty tool name in tool_names list")
            continue
        if trimmed in seen:
            continue
        seen.add(trimmed)
        normalized_names.append(trimmed)

    if not normalized_names:
        return []

    # Allow tuning concurrency/timeout via env vars
    try:
        max_concurrency = int(os.getenv("TOOL_FETCH_MAX_CONCURRENCY", "5"))
        max_concurrency = max(max_concurrency, 1)
    except ValueError:
        max_concurrency = 5

    try:
        request_timeout = float(os.getenv("TOOL_FETCH_TIMEOUT_SEC", "10"))
        if request_timeout <= 0:
            request_timeout = 10.0
    except ValueError:
        request_timeout = 10.0

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _task(name: str, client: httpx.AsyncClient) -> Any | None:
        async with semaphore:
            try:
                return await _fetch_single_tool(name, client)
            except Exception:  # Defensive catch to ensure gather completes
                logger.exception("Unexpected error loading tool %s", name)
                return None

    timeout = httpx.Timeout(request_timeout)
    tools: list[Any] = []
    async with httpx.AsyncClient(timeout=timeout, headers={"Accept": "application/json"}) as client:
        tasks = [_task(name, client) for name in normalized_names]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for tool in results:
            if tool is not None:
                tools.append(tool)

    logger.info(
        "Loaded %d/%d tools for worker %s",
        len(tools),
        len(normalized_names),
        worker_name,
    )
    return tools
