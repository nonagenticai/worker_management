"""A sync function run off the event loop must still see the request identity.

`loop.run_in_executor` does not copy contextvars, so the project id set by the
auth middleware was invisible to sync graph functions and sync tools, and any
LLM call they made reached billing with no project attached.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.api.identity_context import get_project_id_from_context, set_project_id_context
from src.worker.agents.tool_template import ToolTemplate


class _In(BaseModel):
    pass


class _Out(BaseModel):
    project_id: str


def _sync_impl(_ctx: object) -> dict:
    return {"project_id": get_project_id_from_context()}


@pytest.mark.asyncio
async def test_sync_tool_impl_sees_project_id() -> None:
    set_project_id_context("aaaaaaaa-0000-7000-8000-00000000cafe")
    result = await ToolTemplate._execute_implementation(
        _sync_impl, SimpleNamespace(tool_name="ctx_probe_tool"), {}, _In, _Out
    )
    got = result.project_id if hasattr(result, "project_id") else result["project_id"]
    assert got == "aaaaaaaa-0000-7000-8000-00000000cafe"
