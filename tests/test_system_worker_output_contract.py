"""Regression: a SEEDED improvement worker's registered output_schema must reach the agent.

Measured on prod 2026-09-11 against the production database:

    improvement_workflow_development | 0195f48b-…-000000000001 | updated      | config.output_schema present
    improvement_workflow_development | 019de842-…-eb0d8c5396f9 | auto-created | config.output_schema ABSENT

``worker_states`` shows the traffic moving between them: the seeded row's last
run is 2026-08-27 10:55, and all 93 runs from then to 2026-09-11 08:11 landed on
the ``019de842`` row. The registry is keyed ``(tenant_id, subject_id,
worker_name)``, so the message endpoint's lookup missed and its auto-create
branch fabricated a bare row (``output_schema=None``, no ``system_prompt``,
``model_provider="vllm"``) and returned 200. Downstream,
``_output_type_from_schema(None)`` is ``str``, so every completion in the
retained window has ``output_type="string"`` -- sometimes markdown-fenced JSON,
~25% of the time the 11-token ``{"workflow_patch": []}`` -- and none of them
carry the ``rationale`` / ``expected_impact`` / ``scope_of_change`` the
registered schema requires. ``output_schema_used: null`` was an accurate
report: no schema was used.

These tests pin the two halves of the fix: the schema is recovered from the
in-image seed for BOTH a bare existing row and a fresh auto-create, and the
empty-patch case is a declarable outcome rather than a schema error.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.api.workers.system_worker_contracts import (
    registered_output_schema,
    registered_output_schemas,
)
from src.auth import get_tenant_id

_MP = "src.api.workers.message_endpoints"
TENANT = "test-tenant-contract"
WORKER_ID = "wid-contract"
SEEDED_WORKER = "improvement_workflow_development"


# ---------------------------------------------------------------------------
# The contract registry itself
# ---------------------------------------------------------------------------


def test_seed_package_is_importable_and_non_empty() -> None:
    """``registered_output_schemas`` swallows an import failure and returns {} so a
    packaging change cannot 500 a live message. That makes CI the only place the
    breakage can surface -- so assert it here, loudly."""
    schemas = registered_output_schemas()
    assert schemas, (
        "no improvement-worker schemas loaded; the seed package under scripts/ is not "
        "importable, so every registered contract is silently unenforceable"
    )
    assert SEEDED_WORKER in schemas
    # One name from each of the other loops, so a loop dropped from _LOOP_MODULES
    # is caught rather than quietly uncovered.
    for name in (
        "improvement_prompt_improvement",
        "improvement_principle_improvement",
        "improvement_procedure_improvement",
        # asset_authoring names its workers ``authoring_*``, NOT ``improvement_*`` --
        # which is exactly why the loader reads AGENT_CONFIGS instead of deriving a
        # filename from the worker name.
        "authoring_principle_development",
    ):
        assert name in schemas, f"{name} lost its registered contract"


def test_registered_schema_is_the_file_on_disk() -> None:
    schema = registered_output_schema(SEEDED_WORKER)
    assert schema is not None
    assert schema["title"] == "WorkflowDevelopmentProposal"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "workflow_patch",
        "rationale",
        "expected_impact",
        "scope_of_change",
    }


def test_an_unknown_worker_has_no_registered_contract() -> None:
    """The fallback must be confined to seeded system workers -- it is not a general
    cross-tenant config inheritance path."""
    assert registered_output_schema("test_worker") is None
    assert registered_output_schema("some_tenant_worker") is None


# ---------------------------------------------------------------------------
# Endpoint wiring
# ---------------------------------------------------------------------------


def _worker_data(*, worker_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "worker_name": worker_name,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        "config": config
        if config is not None
        else {
            # Exactly the shape of the live auto-created row: no output_schema,
            # no system_prompt, provider vllm.
            "agent_type": "pydantic_ai",
            "model_provider": "vllm",
            "model": "local-llm",
            "temperature": 0.1,
        },
        "status": "auto-created",
    }


def _run_response(output: Any = "Hello!") -> MagicMock:
    resp = MagicMock()
    resp.output = output
    resp.data = output
    resp.usage_metrics = None
    resp.session_id = None
    resp.cost_usd = None
    resp.duration_ms = None
    return resp


@asynccontextmanager
async def _noop_ctx(*_a: Any, **_kw: Any):
    yield "fake-lock"


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.state_manager") as mock_state_mgr,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
        patch(f"{_MP}.mlx_model_registry") as mock_mlx,
    ):
        mock_state_mgr.atomic_worker_operation = _noop_ctx
        mock_state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        mock_state_mgr.validate_state_transition = MagicMock(return_value=True)

        mock_db = AsyncMock()
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.add_worker_state = AsyncMock(return_value=1)
        MockStateDB.return_value = mock_db

        worker_mock = MagicMock()
        worker_mock.run = AsyncMock(return_value=_run_response())
        mock_cache.get_or_create_worker = AsyncMock(return_value=worker_mock)
        mock_cache.remove_worker = AsyncMock()
        mock_cache.remove_instance = AsyncMock()

        mock_registry.get_worker = AsyncMock(return_value=_worker_data(worker_name=SEEDED_WORKER))
        mock_registry.register_worker = AsyncMock(return_value={"worker_id": WORKER_ID})

        fake_instance = MagicMock()
        fake_instance.db_path = "/tmp/auto.db"
        mock_create.return_value = fake_instance

        mock_mlx.get_model = AsyncMock(return_value=None)
        mock_mlx.get_instance_for_model = AsyncMock(return_value=None)

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "create_worker": mock_create,
            "worker_mock": worker_mock,
        }


@pytest.fixture
def client(deps):
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(_a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _post(client, worker_name: str = SEEDED_WORKER, **extra):
    return client.post(f"/workers/{worker_name}/message", json={"message": "Hello", **extra})


def test_bare_existing_row_still_gets_the_registered_schema(client, deps):
    """The live shape: the row EXISTS (so the auto-create branch never runs) but
    carries no schema. This is the half a fix confined to auto-create would miss."""
    resp = _post(client)
    assert resp.status_code == 200

    passed = deps["cache"].get_or_create_worker.await_args.args[1]
    expected = registered_output_schema(SEEDED_WORKER)
    # Both places _create_worker_instance reads, so the agent cannot miss it.
    assert passed["output_schema"] == expected
    assert passed["config"]["output_schema"] == expected
    # ...and the response now REPORTS the schema it was constrained by, which is
    # the field that read null on all 32 canary executions.
    assert resp.json()["output_schema_used"] == expected


def test_schema_reaches_the_agent_factory_not_just_the_config(deps):
    """Prove the whole chain: the dict the endpoint hands the cache must survive the
    cache hop and arrive as a ``create_worker(output_schema=...)`` kwarg -- that is
    where pydantic-ai turns it into a StructuredDict output_type instead of ``str``."""
    import asyncio

    from src.api.workers._globals import WorkerInstanceCache
    from src.api.workers.message_endpoints import apply_registered_system_schema

    worker_data = _worker_data(worker_name=SEEDED_WORKER)
    assert apply_registered_system_schema(worker_data, SEEDED_WORKER) is True

    cache = WorkerInstanceCache()
    with patch(
        "src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())
    ) as mock_cw:
        asyncio.run(cache._create_worker_instance(SEEDED_WORKER, worker_data))

    assert mock_cw.call_args.kwargs["output_schema"] == registered_output_schema(SEEDED_WORKER)


def test_auto_create_persists_the_registered_schema(client, deps):
    """A fresh (tenant, subject) must be BORN with the contract, not repaired per-call."""
    deps["registry"].get_worker = AsyncMock(
        side_effect=[
            None,
            _worker_data(
                worker_name=SEEDED_WORKER,
                config={
                    "agent_type": "pydantic_ai",
                    "output_schema": registered_output_schema(SEEDED_WORKER),
                },
            ),
        ]
    )

    resp = _post(client)
    assert resp.status_code == 200

    expected = registered_output_schema(SEEDED_WORKER)
    # The throwaway instance built during auto-create...
    assert deps["create_worker"].call_args.kwargs["output_schema"] == expected
    # ...and the config PERSISTED to the registry.
    assert deps["registry"].register_worker.await_args.kwargs["config"]["output_schema"] == expected


def test_a_per_message_schema_still_wins(client, deps):
    """The registered default is a FALLBACK. A caller that sends its own schema must
    keep getting it, or this fix becomes a regression for every per-call caller."""
    caller_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    resp = _post(client, output_schema=caller_schema)
    assert resp.status_code == 200

    passed = deps["cache"].get_or_create_worker.await_args.args[1]
    assert passed["output_schema"] == caller_schema
    assert passed["config"]["output_schema"] == caller_schema


def test_a_configured_schema_is_never_clobbered(client, deps):
    """A row that DOES assert a schema keeps it -- the fallback only fills a hole."""
    stored = {"type": "object", "properties": {"stored": {"type": "string"}}}
    deps["registry"].get_worker = AsyncMock(
        return_value=_worker_data(
            worker_name=SEEDED_WORKER,
            config={"agent_type": "pydantic_ai", "output_schema": stored},
        )
    )
    resp = _post(client)
    assert resp.status_code == 200
    assert (
        deps["cache"].get_or_create_worker.await_args.args[1]["config"]["output_schema"] == stored
    )


def test_a_non_seeded_worker_is_untouched(client, deps):
    """No schema may be conjured for an ordinary tenant worker."""
    deps["registry"].get_worker = AsyncMock(return_value=_worker_data(worker_name="test_worker"))
    resp = _post(client, worker_name="test_worker")
    assert resp.status_code == 200

    passed = deps["cache"].get_or_create_worker.await_args.args[1]
    assert passed.get("output_schema") is None
    assert passed["config"].get("output_schema") is None
    assert resp.json()["output_schema_used"] is None


# ---------------------------------------------------------------------------
# Required keys, and the empty-patch decision
# ---------------------------------------------------------------------------


def _validator():
    import jsonschema

    return jsonschema.Draft202012Validator(registered_output_schema(SEEDED_WORKER))


_GOOD = {
    "workflow_patch": [{"op": "replace", "path": "/nodes/3/parameters/x", "value": 1}],
    "rationale": "node 3 retries on a transient 502",
    "expected_impact": {"delta_execution_success_rate": 0.05},
    "scope_of_change": "node_parameter",
}


def test_the_required_keys_are_enforced() -> None:
    """The exact shape 8 of 32 canary executions returned -- ``{"workflow_patch": []}``
    and nothing else -- must be INVALID against the contract. It read as acceptable
    only because the contract was never applied."""
    v = _validator()
    assert v.is_valid(_GOOD)
    assert not v.is_valid({"workflow_patch": []})
    for missing in ("rationale", "expected_impact", "scope_of_change"):
        assert not v.is_valid({k: val for k, val in _GOOD.items() if k != missing}), (
            f"{missing} is required but a payload without it validated"
        )
    # additionalProperties:false -- the retired prose aliases cannot come back.
    assert not v.is_valid({**_GOOD, "implementation_notes": "..."})


def test_empty_patch_is_schema_valid_and_declarable() -> None:
    """THE DECISION: no ``minItems: 1``.

    Two reasons, in order of force.

    1. It would be INERT here, provably. ``src/utils/xgrammar_schema.py`` lists
       ``minItems`` in ``_XGRAMMAR_UNSUPPORTED_KEYS``, and
       ``pydantic_ai_agent._decoder_visible_schema`` routes the object root
       through ``strip_xgrammar_unsupported`` for BOTH the decoder grammar and
       the fatal ``Draft202012Validator`` -- deliberately, so the model is never
       graded against a contract it was not shown (the 2026-07-20 outage). A
       ``minItems`` added here is stripped from both: enforced by nothing.

    2. Even if it were enforced it would be WRONG. A model with nothing to
       propose is a legitimate outcome; a decoder grammar that forbids ``[]``
       makes fabricating an operation the only way to answer, and a fabricated
       operation is promoted to a live canary. Forcing a proposal is strictly
       worse than recording a decline.

    So the empty patch stays schema-valid, and the fix is to make the decline
    SAYABLE: ``scope_of_change: "no_change"`` plus a ``rationale``. The caller
    can then tell "declined, here is why" from "malformed", instead of labelling
    both ``proposal_schema_invalid``.
    """
    schema = registered_output_schema(SEEDED_WORKER)
    assert "minItems" not in schema["properties"]["workflow_patch"]
    assert "no_change" in schema["properties"]["scope_of_change"]["enum"]

    v = _validator()
    decline = {
        "workflow_patch": [],
        "rationale": "the workflow already retries; no bounded improvement is available",
        "expected_impact": {},
        "scope_of_change": "no_change",
    }
    assert v.is_valid(decline), "an honest decline must be expressible within the contract"
    # And it is DISTINGUISHABLE from a proposal without inspecting prose.
    assert decline["scope_of_change"] == "no_change" and not decline["workflow_patch"]
    assert _GOOD["scope_of_change"] != "no_change" and _GOOD["workflow_patch"]


def test_minitems_would_be_stripped_before_it_could_enforce_anything() -> None:
    """Mechanism control for reason (1) above -- this is the measurement, not an
    appeal to the docstring. Add ``minItems`` to the contract and watch the decoder
    view drop it."""
    import copy

    from src.worker.agents.pydantic_ai_agent import _decoder_visible_schema

    hypothetical = copy.deepcopy(registered_output_schema(SEEDED_WORKER))
    hypothetical["properties"]["workflow_patch"]["minItems"] = 1

    visible = _decoder_visible_schema(hypothetical)
    assert visible is not None
    assert "minItems" not in visible["properties"]["workflow_patch"], (
        "minItems survived into the decoder view -- reason (1) for rejecting it no "
        "longer holds and the empty-patch decision must be re-argued"
    )
    # The keys that DO enforce the contract survive, so the stripping is targeted.
    assert set(visible["required"]) == set(hypothetical["required"])
    assert visible["additionalProperties"] is False
    assert "no_change" in visible["properties"]["scope_of_change"]["enum"]
