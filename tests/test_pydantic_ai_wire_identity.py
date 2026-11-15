"""A real pydantic-ai ``Agent.run`` puts the identity headers on the WIRE.

The other forwarding tests call the httpx hook directly. That cannot catch a
pydantic-ai / openai SDK change that sends through a client other than the one
the hook is installed on — the failure mode suspected (2026-10-07) when
inference-router logged ``worker-cdms-classify`` (UA ``pydantic-ai/1.107.0``)
with no app/project/actor. This drives the SDK end to end and inspects the
request that reaches the transport, so it fails if the SDK ever drops them.
(Live evidence for that incident: the hook DID run — the inbound caller simply
sent no X-App-Id / X-Project-Id, see ``wm_ir_identity_missing`` WARN lines.)
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic_ai import Agent

from src.api.identity_context import (
    set_actor_id_context,
    set_app_id_context,
    set_identity_context,
    set_project_id_context,
)
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIChatModel,
    InferenceRouterOpenAIProvider,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"
TENANT = "0195f48b-d000-7000-8000-000000000001"
SUBJECT = "0195f48b-d000-7000-8000-0000000000b2"
ACTOR = "0195f48b-d000-7000-8000-0000000000c3"
PROJECT = "0195f48b-d000-7000-8000-0000000000a1"
WORKER = "worker-cdms-classify"


@pytest.fixture(autouse=True)
def _clean():
    yield
    set_identity_context("", "", "")
    set_app_id_context("")
    set_project_id_context("")
    set_actor_id_context("")


def _completion(_request: httpx.Request) -> httpx.Response:
    body = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.Response(
        200, content=json.dumps(body), headers={"content-type": "application/json"}
    )


@pytest.mark.asyncio
async def test_agent_run_sends_app_project_actor_and_job_id_on_the_wire() -> None:
    set_identity_context(TENANT, "", SUBJECT)
    set_app_id_context("cdms")
    set_project_id_context(PROJECT)
    set_actor_id_context(ACTOR)

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _completion(request)

    provider = InferenceRouterOpenAIProvider(base_url=IR_URL, worker_name=WORKER, tenant_id=TENANT)
    # Swap ONLY the transport: the AsyncClient (and its request event hook) is
    # the one the provider built, so the hook runs exactly as in production.
    provider._client._client._transport = httpx.MockTransport(handler)
    agent = Agent(InferenceRouterOpenAIChatModel(model_name="m", provider=provider))

    await agent.run("hi")

    assert len(sent) == 1
    headers = sent[0].headers
    assert headers["user-agent"].startswith("pydantic-ai/")
    assert headers["X-App-Id"] == "cdms"
    assert headers["X-Project-Id"] == PROJECT
    assert headers["X-Actor-Id"] == ACTOR
    assert headers["X-Job-Id"] == WORKER
    assert headers["X-Tenant-Id"] == TENANT
