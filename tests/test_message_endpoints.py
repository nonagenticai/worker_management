"""Tests for POST /workers/{worker_name}/message (src/api/workers/message_endpoints.py).

Covers: happy paths (all output types), auto-create, model_override, output_schema
override, error handling, MCP retry logic, state transitions.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id

TENANT = "test-tenant-msg"
WORKER_ID = "wid-1234"

# Module path prefix for patching
_MP = "src.api.workers.message_endpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_ctx(*_a, **_kw):
    yield "fake-lock"


def _make_worker_data(
    *,
    worker_name="test_worker",
    worker_id=WORKER_ID,
    is_agent=True,
    config=None,
):
    return {
        "worker_name": worker_name,
        "worker_id": worker_id,
        "db_path": "/tmp/test.db",
        "is_agent": is_agent,
        "config": config
        or {
            "agent_type": "pydantic_ai",
            "model_provider": "mlx",
            "model": "mlx",
            "temperature": 0.1,
        },
        "status": "active",
    }


def _make_run_response(*, output="Hello!", output_type=str):
    """Create a mock agent response object."""
    resp = MagicMock()
    resp.output = output
    resp.data = output
    resp.usage_metrics = None
    resp.session_id = None
    resp.cost_usd = None
    resp.duration_ms = None
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deps():
    """Patch all dependencies of message_endpoints; return dict of mocks."""
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.state_manager") as mock_state_mgr,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
        patch(f"{_MP}.create_worker") as mock_create,
        patch(f"{_MP}.mlx_model_registry") as mock_mlx,
    ):
        # state_manager setup
        mock_state_mgr.atomic_worker_operation = _noop_ctx
        mock_state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        mock_state_mgr.validate_state_transition = MagicMock(return_value=True)

        # StateDatabase instance
        mock_db = AsyncMock()
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.add_worker_state = AsyncMock(return_value=1)
        MockStateDB.return_value = mock_db

        # worker_cache
        worker_mock = MagicMock()
        worker_mock.run = AsyncMock(return_value=_make_run_response())
        mock_cache.get_or_create_worker = AsyncMock(return_value=worker_mock)
        mock_cache.remove_worker = AsyncMock()
        mock_cache.remove_instance = AsyncMock()

        # worker_registry defaults: worker exists
        mock_registry.get_worker = AsyncMock(return_value=_make_worker_data())
        mock_registry.register_worker = AsyncMock(return_value={"worker_id": WORKER_ID})

        # create_worker (for auto-create)
        fake_worker_instance = MagicMock()
        fake_worker_instance.db_path = "/tmp/auto.db"
        mock_create.return_value = fake_worker_instance

        # mlx registry
        mock_mlx.get_model = AsyncMock(return_value=None)
        mock_mlx.get_instance_for_model = AsyncMock(return_value=None)

        yield {
            "registry": mock_registry,
            "cache": mock_cache,
            "state_manager": mock_state_mgr,
            "state_db": mock_db,
            "create_worker": mock_create,
            "mlx": mock_mlx,
            "worker_mock": worker_mock,
            "StateDB_cls": MockStateDB,
        }


@pytest.fixture
def client(deps):
    """TestClient with auth override and lifespan bypassed."""
    app.dependency_overrides[get_tenant_id] = lambda: TENANT

    @asynccontextmanager
    async def _fake_lifespan(a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _fake_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original


def _post_msg(client, worker_name="test_worker", **extra):
    payload = {"message": "Hello", **extra}
    return client.post(f"/workers/{worker_name}/message", json=payload)


# ===========================================================================
# Happy paths — output types
# ===========================================================================


class TestSendMessageHappyPaths:
    def test_string_output(self, client, deps):
        resp = _post_msg(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "Hello!"
        assert body["output_type"] == "string"
        # Verify state transitions were recorded
        assert deps["state_db"].add_worker_state.await_count >= 2  # processing + completed

    def test_dict_output(self, client, deps):
        deps["worker_mock"].run = AsyncMock(return_value=_make_run_response(output={"key": "val"}))
        resp = _post_msg(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == {"key": "val"}
        assert body["output_type"] == "object"

    def test_list_output(self, client, deps):
        deps["worker_mock"].run = AsyncMock(return_value=_make_run_response(output=[1, 2, 3]))
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output_type"] == "array"
        assert resp.json()["output"] == [1, 2, 3]

    def test_bool_output(self, client, deps):
        deps["worker_mock"].run = AsyncMock(return_value=_make_run_response(output=True))
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output_type"] == "boolean"
        assert resp.json()["output"] is True

    def test_enum_output(self, client, deps):
        """String output with enum in output_schema → output_type='enum'."""
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(
                config={
                    "agent_type": "pydantic_ai",
                    "output_schema": {"enum": ["a", "b", "c"]},
                }
            )
        )
        deps["worker_mock"].run = AsyncMock(return_value=_make_run_response(output="a"))
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output_type"] == "enum"

    def test_other_output_type(self, client, deps):
        """Non-standard output type (e.g., integer) → output_type='other'."""
        # Use an object type that's not str/dict/list/bool
        run_resp = MagicMock()
        run_resp.output = 42  # int → falls through to "other"
        # But wait — int might not be "other". Actually checking the code:
        # bool is checked first (bool is subclass of int), then str, dict, list
        # int is not any of those, so → "other"
        # Actually re-reading: output is extracted, then type is checked.
        # int isn't in the if-elif chain, so it goes to "other"
        run_resp.usage_metrics = None
        run_resp.session_id = None
        run_resp.cost_usd = None
        run_resp.duration_ms = None
        deps["worker_mock"].run = AsyncMock(return_value=run_resp)
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output_type"] == "other"

    def test_usage_metrics_included(self, client, deps):
        run_resp = _make_run_response()
        usage = MagicMock()
        usage.model_dump.return_value = {"tokens": 100}
        run_resp.usage_metrics = usage
        deps["worker_mock"].run = AsyncMock(return_value=run_resp)
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["usage_metrics"] == {"tokens": 100}

    def test_context_in_metadata(self, client, deps):
        """``metadata.context`` keeps its exact shape.

        ⚠️ ``metadata`` is no longer ONLY the context: it now also carries the
        prompt-attribution block (roadmap A5 — see
        ``tests/test_message_prompt_attribution.py`` and
        ``src/api/workers/prompt_attribution.py``). The context key itself is
        unchanged, which is the back-compat this asserts.
        """
        resp = _post_msg(client, context={"key": "value"})
        assert resp.status_code == 200
        assert resp.json()["metadata"]["context"] == {"key": "value"}

    def test_no_context_means_no_context_key(self, client, deps):
        """Was ``metadata is None``. A response now always carries attribution,
        so the fact being pinned is that an absent context stays ABSENT rather
        than becoming a null key."""
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert "context" not in resp.json()["metadata"]

    def test_primary_success_records_retry_attempt_zero(self, client, deps):
        """The denominator half of rescue rate.

        The rescue path already logged "recovered on LLM retry %d/%d", but a
        count of rescues with no total is not a rate. Writing 0 here means every
        completed row carries the attempt that produced it, so
        `(worker_state->>'retry_attempt')::int > 0` over all completions is the
        real ratio -- see tests/integration/test_rescue_rate_is_queryable.py.
        """
        resp = _post_msg(client)
        assert resp.status_code == 200

        completed = [
            c
            for c in deps["state_db"].add_worker_state.await_args_list
            if c.kwargs.get("state_type") == "completed"
        ]
        assert completed, "no completed state was persisted"
        assert completed[-1].kwargs["worker_state"]["retry_attempt"] == 0


# ===========================================================================
# Auto-create paths
# ===========================================================================


class TestSendMessageAutoCreate:
    def test_auto_creates_when_not_found(self, client, deps):
        """Worker not in registry → auto-create with mlx defaults."""
        # First call returns None (not found), second call after registration returns data
        deps["registry"].get_worker = AsyncMock(side_effect=[None, _make_worker_data()])
        resp = _post_msg(client)
        assert resp.status_code == 200
        # Verify create_worker was called
        deps["create_worker"].assert_awaited_once()
        # Verify register_worker was called
        deps["registry"].register_worker.assert_awaited_once()
        reg_kwargs = deps["registry"].register_worker.call_args
        assert reg_kwargs.kwargs.get("status") == "auto-created" or (
            len(reg_kwargs.args) > 0 or "auto-created" in str(reg_kwargs)
        )

    def test_auto_create_with_disable_state_storage(self, client, deps):
        """disable_state_storage=True → register_worker NOT called."""
        deps["registry"].get_worker = AsyncMock(return_value=None)
        resp = _post_msg(client, disable_state_storage=True)
        assert resp.status_code == 200
        deps["registry"].register_worker.assert_not_awaited()

    def test_auto_create_failure(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=None)
        deps["create_worker"].side_effect = RuntimeError("creation failed")
        resp = _post_msg(client)
        assert resp.status_code == 500
        body = resp.json()
        msg = body.get("error", {}).get("message", body.get("detail", ""))
        assert "auto-creation failed" in msg


# ===========================================================================
# Model override
# ===========================================================================


class TestSendMessageModelOverride:
    def test_model_found_url_resolved(self, client, deps):
        """model_override with valid model → routed via its OWN cache slot, no eviction.

        The override reaches the agent through ``_override_cache_fingerprint``, which
        is part of the cache key; evicting was both redundant and destructive (it
        closed concurrent runs' httpx pools — see
        tests/test_message_endpoint_cache_eviction_storm.py).
        """
        deps["mlx"].get_model = AsyncMock(return_value={"model": "test-model"})
        deps["mlx"].get_instance_for_model = AsyncMock(return_value="http://mlx-host:8080")
        resp = _post_msg(client, model_override="test-model")
        assert resp.status_code == 200
        deps["cache"].remove_worker.assert_not_awaited()
        kwargs = deps["cache"].get_or_create_worker.await_args.kwargs
        assert kwargs["override_key"] is not None
        # The resolved instance URL still reached the agent builder.
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["base_url_override"] == "http://mlx-host:8080"
        assert passed_config["model"] == "test-model"

    def test_model_not_found(self, client, deps):
        deps["mlx"].get_model = AsyncMock(return_value=None)
        resp = _post_msg(client, model_override="nonexistent")
        assert resp.status_code == 400
        body = resp.json()
        msg = body.get("error", {}).get("message", body.get("detail", ""))
        assert "not found in MLX registry" in msg

    def test_model_found_no_url(self, client, deps):
        """Model exists but no instance URL → proceed without override."""
        deps["mlx"].get_model = AsyncMock(return_value={"model": "m"})
        deps["mlx"].get_instance_for_model = AsyncMock(return_value=None)
        resp = _post_msg(client, model_override="m")
        assert resp.status_code == 200


# ===========================================================================
# enable_reasoning override + cache fingerprint
# ===========================================================================


def _fingerprint(**fields):
    from src.api.workers.message_endpoints import _override_cache_fingerprint
    from src.api.workers.models import MessageRequest

    return _override_cache_fingerprint(MessageRequest(message="Hello", **fields))


class TestOverrideCacheFingerprint:
    """``enable_reasoning`` must JOIN the worker-cache key.

    Without it the leak runs both ways: a caller that opts thinking out bakes that
    into the shared cached agent a later caller (who asked for nothing) reuses, and
    a caller that asks for thinking poisons an opted-out one. The three tri-state
    values must therefore produce three DIFFERENT keys.
    """

    def test_no_overrides_returns_none(self):
        """Backward compat: a bare request keeps the base cache key untouched."""
        assert _fingerprint() is None

    def test_true_false_and_none_are_all_distinct(self):
        fp_true = _fingerprint(enable_reasoning=True)
        fp_false = _fingerprint(enable_reasoning=False)
        fp_none = _fingerprint()
        assert fp_true is not None
        assert fp_false is not None
        assert fp_none is None
        assert fp_true != fp_false

    def test_false_is_not_treated_as_absent(self):
        """``False`` is falsy — an ``if not v`` style guard would collapse it into
        the no-override case and hand the opting-out caller the shared agent."""
        assert _fingerprint(enable_reasoning=False) is not None

    def test_reasoning_changes_a_key_that_already_had_overrides(self):
        """It must contribute even when another override already forces a key."""
        base = _fingerprint(max_tokens=4000)
        with_reasoning = _fingerprint(max_tokens=4000, enable_reasoning=False)
        assert base is not None
        assert with_reasoning is not None
        assert base != with_reasoning


class TestSendMessageEnableReasoningOverride:
    def test_enable_reasoning_false_reaches_worker_config(self, client, deps):
        resp = _post_msg(client, enable_reasoning=False)
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["enable_reasoning"] is False
        # And it lands in its own cache slot rather than poisoning the shared one.
        assert deps["cache"].get_or_create_worker.await_args.kwargs["override_key"] is not None

    def test_enable_reasoning_true_reaches_worker_config(self, client, deps):
        resp = _post_msg(client, enable_reasoning=True)
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["enable_reasoning"] is True

    def test_omitted_leaves_config_untouched(self, client, deps):
        """The default-preserving case: no key written, no override cache slot."""
        resp = _post_msg(client)
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert "enable_reasoning" not in passed_config
        assert deps["cache"].get_or_create_worker.await_args.kwargs["override_key"] is None

    def test_unknown_field_still_rejected(self, client):
        """MessageRequest is extra="forbid"; the new field must be DECLARED, and
        neighbouring typos must still 422 (guards against extra="allow" creep)."""
        resp = _post_msg(client, enable_reasoningg=False)
        assert resp.status_code == 422


_PYDANTIC_CFG = {
    "agent_role": "r",
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": "local-llm",
    "temperature": 0.0,
}


class TestSendMessageSeedOverride:
    """Per-message ``seed`` is passed PER RUN (``worker.run(msg, seed=)``), never
    folded into the config nor into the cache key: judges sweep seeds 0..4 and
    every cache slot holds its own never-evicted connection pool."""

    @pytest.fixture
    def seeded(self, deps):
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(config=dict(_PYDANTIC_CFG))
        )
        deps["worker_mock"].accepts_run_seed = True
        return deps

    def test_seed_is_passed_to_the_run_not_the_config(self, client, seeded):
        resp = _post_msg(client, seed=42)
        assert resp.status_code == 200
        assert seeded["worker_mock"].run.await_args.kwargs == {"seed": 42}
        passed_config = seeded["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert "seed" not in passed_config
        assert seeded["cache"].get_or_create_worker.await_args.kwargs["override_key"] is None

    def test_five_seeds_share_exactly_one_cache_slot(self, client, seeded):
        """MUTATION: put ``"seed"`` back in ``_override_cache_fingerprint`` ⇒ five
        distinct keys ⇒ red."""
        from src.api.workers._globals import _cache_key

        keys = set()
        for seed in range(5):
            assert _post_msg(client, seed=seed).status_code == 200
            call = seeded["cache"].get_or_create_worker.await_args
            keys.add(
                _cache_key(
                    call.args[0],
                    call.kwargs["tenant_id"],
                    call.args[1]["config"].get("work_tier"),
                    call.kwargs.get("product"),
                    call.kwargs["override_key"],
                )
            )
            assert seeded["worker_mock"].run.await_args.kwargs == {"seed": seed}
        assert len(keys) == 1, keys

    def test_seed_zero_is_passed_not_dropped(self, client, seeded):
        assert _post_msg(client, seed=0).status_code == 200
        assert seeded["worker_mock"].run.await_args.kwargs == {"seed": 0}

    def test_omitted_seed_runs_exactly_as_before(self, client, seeded):
        assert _post_msg(client).status_code == 200
        assert seeded["worker_mock"].run.await_args.kwargs == {}

    def test_seed_on_a_non_pydantic_worker_is_a_typed_422(self, client, deps):
        """A basic worker (no agent_role) cannot take a per-run seed. The
        pre-check refuses BEFORE any state row, so nothing is left ``processing``."""
        resp = _post_msg(client, seed=1)
        assert resp.status_code == 422
        assert "SEED_NOT_SUPPORTED" in resp.text
        deps["worker_mock"].run.assert_not_awaited()
        deps["state_db"].add_worker_state.assert_not_awaited()

    def test_a_drifted_precheck_never_strands_the_worker_in_processing(
        self, client, seeded, monkeypatch
    ):
        """Pre-check removed (it says every worker supports a seed) and the built
        worker cannot take one: only the run-time fallback fires, AFTER the
        ``processing`` row. The worker must still end in a terminal state.

        MUTATION: drop ``isinstance(he, _RefusedAfterProcessing)`` from the
        HTTPException handler ⇒ the last state row is ``processing`` ⇒ red.
        """
        monkeypatch.setattr(f"{_MP}._seed_supported", lambda _wd: True)
        seeded["worker_mock"].accepts_run_seed = False

        resp = _post_msg(client, seed=1)

        assert resp.status_code == 422
        assert "SEED_NOT_SUPPORTED" in resp.text
        seeded["worker_mock"].run.assert_not_awaited()
        states = [
            c.kwargs["state_type"] for c in seeded["state_db"].add_worker_state.await_args_list
        ]
        assert states[0] == "processing"  # control: the fallback path WAS post-processing
        assert states[-1] == "error", states

    def test_seed_must_be_an_int(self, client):
        assert _post_msg(client, seed="42").status_code == 422
        assert _post_msg(client, seed=True).status_code == 422

    def test_seed_is_not_in_the_cache_fingerprint(self):
        assert _fingerprint(seed=1) is None
        assert _fingerprint(max_tokens=4000, seed=1) == _fingerprint(max_tokens=4000)


# ===========================================================================
# system_prompt / instruction override
#
# Both fields were declared on MessageRequest, security-validated, length-capped
# and shown in the OpenAPI example -- and never read by the handler, so every
# caller sending one got a 200 and the worker RECORD's prompt instead.
# ===========================================================================


class TestPromptOverrideCacheFingerprint:
    """The prompt fields must JOIN the worker-cache key.

    ``instructions=`` is baked ONCE at ``Agent(...)`` construction, so a cached agent
    keeps whichever prompt built it. Folding the prompts into the config WITHOUT
    fingerprinting them is the naive fix: request N's system_prompt is then served to
    request N+1 from a different caller on the same worker, indefinitely.
    """

    def test_system_prompt_alone_forces_an_override_key(self):
        assert _fingerprint(system_prompt="You are a sales analyst.") is not None

    def test_instruction_alone_forces_an_override_key(self):
        assert _fingerprint(instruction="Focus on seasonal trends.") is not None

    def test_distinct_system_prompts_get_distinct_keys(self):
        """The load-bearing one: two callers with different prompts must not collide."""
        assert _fingerprint(system_prompt="ALPHA") != _fingerprint(system_prompt="BETA")

    def test_distinct_instructions_get_distinct_keys(self):
        assert _fingerprint(instruction="ALPHA") != _fingerprint(instruction="BETA")

    def test_system_prompt_and_instruction_are_not_interchangeable(self):
        """The same text in the other field builds a different agent (both are joined
        into ``instructions=``, but ordering differs), so it must key differently."""
        assert _fingerprint(system_prompt="SAME") != _fingerprint(instruction="SAME")

    def test_prompts_change_a_key_that_already_had_overrides(self):
        """They must contribute even when another override already forces a key."""
        base = _fingerprint(max_tokens=4000)
        with_prompt = _fingerprint(max_tokens=4000, system_prompt="You are terse.")
        assert base != with_prompt

    def test_key_stays_bounded_for_a_huge_prompt(self):
        """Cardinality/size guard: the material is hashed, never embedded, so the
        cache key is 16 chars regardless of a prompt capped at 1M."""
        assert len(_fingerprint(system_prompt="x" * 100_000)) == 16


class TestSendMessagePromptOverride:
    def test_system_prompt_reaches_worker_config(self, client, deps):
        resp = _post_msg(client, system_prompt="You are a sales analyst.")
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["system_prompt"] == "You are a sales analyst."
        # And it lands in its own cache slot rather than poisoning the shared one.
        assert deps["cache"].get_or_create_worker.await_args.kwargs["override_key"] is not None

    def test_instruction_reaches_worker_config(self, client, deps):
        resp = _post_msg(client, instruction="Focus on seasonal trends.")
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["instruction"] == "Focus on seasonal trends."
        assert deps["cache"].get_or_create_worker.await_args.kwargs["override_key"] is not None

    def test_both_prompts_reach_worker_config_together(self, client, deps):
        """The OpenAPI example sends BOTH; downstream they are joined into a single
        ``instructions=`` string (PydanticAIAgent._setup_agent)."""
        resp = _post_msg(
            client,
            system_prompt="You are a data analyst.",
            instruction="Focus on year-over-year growth.",
        )
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["system_prompt"] == "You are a data analyst."
        assert passed_config["instruction"] == "Focus on year-over-year growth."

    def test_omitted_prompts_leave_the_record_values_untouched(self, client, deps):
        """``None`` ⇒ keep the record's value (the documented max_tokens semantic)."""
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(
                config={
                    "agent_type": "pydantic_ai",
                    "system_prompt": "RECORD-PROMPT",
                    "instruction": "RECORD-INSTRUCTION",
                }
            )
        )
        resp = _post_msg(client)
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["system_prompt"] == "RECORD-PROMPT"
        assert passed_config["instruction"] == "RECORD-INSTRUCTION"
        assert deps["cache"].get_or_create_worker.await_args.kwargs["override_key"] is None

    def test_overriding_one_prompt_does_not_clobber_the_other(self, client, deps):
        """Each field is written under its OWN guard. A single combined guard that
        assigned both would null out the record's system_prompt for a caller who
        overrode only the instruction -- silently dropping the worker's persona."""
        deps["registry"].get_worker = AsyncMock(
            return_value=_make_worker_data(
                config={
                    "agent_type": "pydantic_ai",
                    "system_prompt": "RECORD-PROMPT",
                    "instruction": "RECORD-INSTRUCTION",
                }
            )
        )
        resp = _post_msg(client, instruction="OVERRIDE-INSTRUCTION")
        assert resp.status_code == 200
        passed_config = deps["cache"].get_or_create_worker.await_args.args[1]["config"]
        assert passed_config["instruction"] == "OVERRIDE-INSTRUCTION"
        assert passed_config["system_prompt"] == "RECORD-PROMPT"

    def test_prompt_typo_still_rejected(self, client):
        """MessageRequest is extra="forbid" -- neighbouring typos must still 422."""
        assert _post_msg(client, system_promptt="x").status_code == 422


class _FakeBuiltAgent:
    """Stand-in for an agent built by the factory, remembering what it was built with.

    Deliberately a plain class, not a MagicMock: ``WorkerInstanceCache._is_worker_healthy``
    probes ``hasattr(worker, "check_health")`` and would then await a MagicMock's
    return value. Exposing neither ``check_health`` nor ``config`` takes the
    "healthy by default" branch, which is what a cache-reuse test needs.
    """

    def __init__(self, system_prompt=None, instruction=None):
        self.system_prompt = system_prompt
        self.instruction = instruction

    async def run(self, _message):
        return _make_run_response(output=self.system_prompt or "no-prompt")


@pytest.fixture
def real_cache(deps):
    """``deps``, but with a REAL (fresh) WorkerInstanceCache instead of the mock.

    The mocked cache can only show that the endpoint HANDED an override_key down; it
    cannot show that two different prompts end up in two different agents. Swapping in
    a real cache and stubbing the factory underneath it exercises the actual keying --
    which is exactly where a fix that folds the prompt into the config but leaves
    ``_override_cache_fingerprint`` alone silently fails.
    """
    from src.api.workers._globals import WorkerInstanceCache

    built: list[dict] = []

    async def _fake_create_worker(**kwargs):
        built.append(kwargs)
        return _FakeBuiltAgent(kwargs.get("system_prompt"), kwargs.get("instruction"))

    # get_worker must return a FRESH dict per call. The shared ``deps`` fixture binds
    # one dict as return_value, so an in-place config write by request N would still be
    # visible to request N+1 -- a mock artifact that would fake the very leak under
    # test (the real registry issues a new dict from the DB row on every call).
    deps["registry"].get_worker = AsyncMock(side_effect=lambda **_kw: _make_worker_data())

    with (
        patch(f"{_MP}.worker_cache", new=WorkerInstanceCache()) as cache,
        patch(
            "src.api.workers._globals.create_worker",
            new=AsyncMock(side_effect=_fake_create_worker),
        ),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        yield {**deps, "cache": cache, "built": built}


class TestPromptOverrideCacheIsolation:
    """End-to-end through the real cache: a prompt must never outlive its request."""

    def test_distinct_system_prompts_do_not_share_an_agent(self, client, real_cache):
        alpha = _post_msg(client, system_prompt="ALPHA-PERSONA")
        beta = _post_msg(client, system_prompt="BETA-PERSONA")
        bare = _post_msg(client)  # a caller that asked for no override at all

        assert [alpha.status_code, beta.status_code, bare.status_code] == [200, 200, 200]

        # Three distinct agents were BUILT, each with the prompt of its own request --
        # and the bare caller got the record's value (None), not a leftover persona.
        assert [k.get("system_prompt") for k in real_cache["built"]] == [
            "ALPHA-PERSONA",
            "BETA-PERSONA",
            None,
        ]
        # Two override slots plus the untouched base slot.
        assert len(real_cache["cache"]._workers) == 3
        assert len([k for k in real_cache["cache"]._workers if ":ov-" in k]) == 2

    def test_same_system_prompt_reuses_one_agent(self, client, real_cache):
        """The other half: identical overrides must still SHARE a slot, or the cache
        degenerates into an unbounded per-request agent (each owns an httpx pool)."""
        assert _post_msg(client, system_prompt="STABLE-PERSONA").status_code == 200
        assert _post_msg(client, system_prompt="STABLE-PERSONA").status_code == 200

        assert len(real_cache["built"]) == 1  # second request hit the cache
        assert len(real_cache["cache"]._workers) == 1

    def test_prompt_reaches_the_factory_not_just_the_config(self, client, real_cache):
        """Proves the whole chain, not the endpoint's own bookkeeping: both fields
        leave the config dict at the cache hop and arrive as create_worker kwargs,
        which is where PydanticAIAgent joins them into ``instructions=``."""
        resp = _post_msg(client, system_prompt="You are a data analyst.", instruction="Be terse.")
        assert resp.status_code == 200

        kwargs = real_cache["built"][-1]
        assert kwargs["system_prompt"] == "You are a data analyst."
        assert kwargs["instruction"] == "Be terse."


# ===========================================================================
# Output schema override
# ===========================================================================


class TestSendMessageOutputSchemaOverride:
    def test_output_schema_on_existing_worker(self, client, deps):
        """Per-message output_schema applies via its own cache slot, without eviction."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        resp = _post_msg(client, output_schema=schema)
        assert resp.status_code == 200
        deps["cache"].remove_worker.assert_not_awaited()
        call = deps["cache"].get_or_create_worker.await_args
        assert call.kwargs["override_key"] is not None
        # apply_per_message_output_override still wrote both places the builder reads.
        passed = call.args[1]
        assert passed["output_schema"] == schema
        assert passed["config"]["output_schema"] == schema


# ===========================================================================
# Error handling
# ===========================================================================


class TestSendMessageErrors:
    def test_worker_not_agent(self, client, deps):
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(is_agent=False))
        resp = _post_msg(client)
        assert resp.status_code == 400
        body = resp.json()
        msg = body.get("error", {}).get("message", body.get("detail", ""))
        assert "not an agent" in msg

    def test_worker_run_generic_error(self, client, deps):
        deps["worker_mock"].run = AsyncMock(side_effect=RuntimeError("model crashed"))
        resp = _post_msg(client)
        assert resp.status_code == 500
        body = resp.json()
        msg = body.get("error", {}).get("message", body.get("detail", ""))
        assert "model crashed" in msg
        # Error state should have been recorded
        # Last add_worker_state call should be for error state
        last_call = deps["state_db"].add_worker_state.call_args_list[-1]
        assert last_call.kwargs.get("state_type") == "error" or "error" in str(last_call)

    def test_mcp_error_triggers_retry_success(self, client, deps):
        """MCP server error → remove + recreate, retry succeeds."""
        # First run fails with MCP error, retry succeeds
        deps["worker_mock"].run = AsyncMock(
            side_effect=[RuntimeError("MCP server is not running"), _make_run_response()]
        )
        # After remove, cache returns a new worker for retry
        new_worker = MagicMock()
        new_worker.run = AsyncMock(return_value=_make_run_response(output="retry-ok"))
        deps["cache"].get_or_create_worker = AsyncMock(
            side_effect=[deps["worker_mock"], new_worker]
        )
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock):
            resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output"] == "retry-ok"

    def test_connection_error_triggers_retry(self, client, deps):
        """Connection refused → retry logic triggered."""
        deps["worker_mock"].run = AsyncMock(side_effect=ConnectionError("connection refused"))
        new_worker = MagicMock()
        new_worker.run = AsyncMock(return_value=_make_run_response(output="retry-ok"))
        deps["cache"].get_or_create_worker = AsyncMock(
            side_effect=[deps["worker_mock"], new_worker]
        )
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock):
            resp = _post_msg(client)
        assert resp.status_code == 200

    def test_retry_also_fails(self, client, deps):
        """Both first try and retry fail → 500."""
        deps["worker_mock"].run = AsyncMock(side_effect=RuntimeError("MCP server is not running"))
        new_worker = MagicMock()
        new_worker.run = AsyncMock(side_effect=RuntimeError("still broken"))
        deps["cache"].get_or_create_worker = AsyncMock(
            side_effect=[deps["worker_mock"], new_worker]
        )
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock):
            resp = _post_msg(client)
        assert resp.status_code == 500

    def test_connection_error_succeeds_on_second_attempt(self, client, deps):
        deps["worker_mock"].run = AsyncMock(side_effect=ConnectionError("Connection error."))
        w2 = MagicMock()
        w2.run = AsyncMock(return_value=_make_run_response(output="ok2"))
        deps["cache"].get_or_create_worker = AsyncMock(side_effect=[deps["worker_mock"], w2])
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock) as sl:
            resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output"] == "ok2"
        assert sl.await_count == 1

    def test_connection_error_succeeds_on_third_attempt(self, client, deps):
        deps["worker_mock"].run = AsyncMock(side_effect=ConnectionError("Connection error."))
        w2 = MagicMock()
        w2.run = AsyncMock(side_effect=ConnectionError("Connection error."))
        w3 = MagicMock()
        w3.run = AsyncMock(return_value=_make_run_response(output="ok3"))
        deps["cache"].get_or_create_worker = AsyncMock(side_effect=[deps["worker_mock"], w2, w3])
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock) as sl:
            resp = _post_msg(client)
        assert resp.status_code == 200
        assert resp.json()["output"] == "ok3"
        assert sl.await_count == 2
        d0 = sl.await_args_list[0].args[0]
        d1 = sl.await_args_list[1].args[0]
        assert d0 < d1  # growing backoff; 25% jitter can't invert 0.5 vs 1.0 bands

    def test_connection_error_exhausts_then_502(self, client, deps):
        from src.settings import get_settings

        deps["worker_mock"].run = AsyncMock(side_effect=ConnectionError("Connection error."))
        w2 = MagicMock()
        w2.run = AsyncMock(side_effect=ConnectionError("Connection error."))
        w3 = MagicMock()
        w3.run = AsyncMock(side_effect=ConnectionError("Connection error."))
        deps["cache"].get_or_create_worker = AsyncMock(side_effect=[deps["worker_mock"], w2, w3])
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock) as sl:
            resp = _post_msg(client)
        # An exhausted ladder against an unreachable upstream is 502 BAD_GATEWAY. The
        # caller's retry gate is `>= 500`, so this is unchanged for the n8n Workers node.
        assert resp.status_code >= 500
        assert resp.status_code == 502
        # ...and the detail must now name the hop, not just repeat "Connection error."
        assert "attempts=" in resp.text
        assert sl.await_count == get_settings().llm_retry_max_attempts - 1
        last_call = deps["state_db"].add_worker_state.call_args_list[-1]
        assert last_call.kwargs.get("state_type") == "error"

    def test_generic_error_no_retry_no_sleep(self, client, deps):
        deps["worker_mock"].run = AsyncMock(side_effect=RuntimeError("model crashed"))
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock) as sl:
            resp = _post_msg(client)
        assert resp.status_code == 500
        sl.assert_not_awaited()
        assert deps["cache"].get_or_create_worker.await_count == 1

    def test_http_exception_500_records_error_state(self, client, deps):
        deps["worker_mock"].run = AsyncMock(
            side_effect=HTTPException(status_code=500, detail="internal fail")
        )
        resp = _post_msg(client)
        assert resp.status_code == 500
        # Error state should be recorded for 500s
        error_calls = [
            c for c in deps["state_db"].add_worker_state.call_args_list if "error" in str(c)
        ]
        assert len(error_calls) >= 1

    def test_http_exception_400_no_error_state(self, client, deps):
        """Non-500 HTTPException is re-raised without recording error state."""
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(is_agent=False))
        resp = _post_msg(client)
        assert resp.status_code == 400
        # Only processing state, no error state (BadRequestError is raised before run)
        error_calls = [
            c
            for c in deps["state_db"].add_worker_state.call_args_list
            if c.kwargs.get("state_type") == "error"
        ]
        assert len(error_calls) == 0

    def test_state_retrieval_failure_defaults_idle(self, client, deps):
        """If getting latest state fails, default to 'idle'."""
        deps["state_db"].get_latest_worker_state = AsyncMock(side_effect=Exception("db read error"))
        resp = _post_msg(client)
        assert resp.status_code == 200  # Should still succeed

    def test_bad_request_error_converted(self, client, deps):
        """BadRequestError raised inside endpoint is converted to HTTPException."""
        deps["registry"].get_worker = AsyncMock(return_value=_make_worker_data(is_agent=False))
        resp = _post_msg(client)
        assert resp.status_code == 400

    def test_worker_name_validation(self, client, deps):
        """Invalid worker names should be rejected by path param validation."""
        resp = _post_msg(client, worker_name="bad name with spaces")
        assert resp.status_code == 422


# ===========================================================================
# Per-product BYO posture re-resolution (A6)
# ===========================================================================


class TestReplayModeContextVar:
    """WS-1 prompt-replay: the inbound X-Replay-Mode header must set the
    per-request replay-mode ContextVar for the duration of the worker run, so
    the outbound inference-router hook (tested in test_replay_identity_headers)
    stamps the replay identity. Absent/other header → False, no leak."""

    def _run_captures_replay(self, deps):
        """Wire worker.run to capture the replay ContextVar at call time."""
        from src.api.identity_context import is_replay_mode_from_context

        captured: dict[str, bool] = {}

        async def _capture(_msg):
            captured["replay"] = is_replay_mode_from_context()
            return _make_run_response()

        deps["worker_mock"].run = AsyncMock(side_effect=_capture)
        return captured

    def test_replay_header_sets_contextvar_true(self, client, deps):
        captured = self._run_captures_replay(deps)
        resp = client.post(
            "/workers/test_worker/message",
            json={"message": "Hello"},
            headers={
                "X-Replay-Mode": "true",
                "X-Correlation-Id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
            },
        )
        assert resp.status_code == 200
        assert captured["replay"] is True

    def test_no_replay_header_contextvar_false(self, client, deps):
        captured = self._run_captures_replay(deps)
        resp = _post_msg(client)
        assert resp.status_code == 200
        assert captured["replay"] is False

    def test_replay_header_non_true_value_is_false(self, client, deps):
        """Only the literal "true" enables replay; any other value stays False."""
        captured = self._run_captures_replay(deps)
        resp = client.post(
            "/workers/test_worker/message",
            json={"message": "Hello"},
            headers={"X-Replay-Mode": "false"},
        )
        assert resp.status_code == 200
        assert captured["replay"] is False


class TestPerProductByoResolution:
    """A run naming a ``product`` re-resolves BYO posture per-(tenant, product)
    and threads product into worker creation; a run without a product keeps the
    tenant-wide posture and makes no extra licenses call (default-off)."""

    def test_product_reresolves_byo_mode_and_threads_product(self, client, deps):
        with patch(
            f"{_MP}._lookup_tenant_byo_mode",
            new=AsyncMock(return_value="byo_required"),
        ) as mock_lookup:
            resp = _post_msg(client, product="kag-graph")
        assert resp.status_code == 200
        # Per-product posture was resolved via licenses with ?app=kag-graph.
        assert mock_lookup.await_count == 1
        assert mock_lookup.call_args.args == (TENANT, "kag-graph")
        # The re-resolved posture + product were threaded to worker creation.
        kwargs = deps["cache"].get_or_create_worker.call_args.kwargs
        assert kwargs["byo_mode"] == "byo_required"
        assert kwargs["product"] == "kag-graph"

    def test_no_product_skips_lookup_and_keeps_tenant_wide(self, client, deps):
        with patch(
            f"{_MP}._lookup_tenant_byo_mode",
            new=AsyncMock(return_value="byo_required"),
        ) as mock_lookup:
            resp = _post_msg(client)  # no product
        assert resp.status_code == 200
        # No product ⇒ no per-product licenses call (zero added latency).
        assert mock_lookup.await_count == 0
        kwargs = deps["cache"].get_or_create_worker.call_args.kwargs
        assert kwargs["product"] is None


# ===========================================================================
# The completion-transition check that must STAY dead
# ===========================================================================


class TestCompletionTransitionCheckStaysDead:
    """The completion site runs NO transition check, on purpose.

    It used to call ``validate_state_transition("processing", "completed")`` —
    two string LITERALS — and PROCESSING -> COMPLETED is the first entry of
    STATE_TRANSITIONS[PROCESSING], so the branch was unreachable BY
    CONSTRUCTION. The call has been deleted, not revived.

    ⛔ Wiring ``current_state`` in there would be wrong: it is the PRE-message
    state, while the real from-state at that point is always ``processing``
    (this handler wrote that row itself). A review measured the cost — over 30
    days, 273,640 ``processing`` rows had ``previous_state`` completed 113,867 ·
    config-updated 112,428 · processing 30,356 · error 16,881 · other 42 —
    ≈119,800 spurious ``invalid_transition`` counts and ≈103,000 extra WARN
    lines per 30 days, all artefacts.
    """

    def test_no_transition_is_validated_into_completed(self, client, deps):
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value={"state_type": "created"})
        resp = _post_msg(client)
        assert resp.status_code == 200

        pairs = [c.args for c in deps["state_manager"].validate_state_transition.call_args_list]
        # The dead literal pair must never be passed to the validator.
        assert ("processing", "completed") not in pairs, pairs
        # Nor may it be "revived" by substituting the runtime pre-message state:
        # NOTHING at all is validated into ``completed``.
        assert [p for p in pairs if len(p) > 1 and p[1] == "completed"] == [], pairs
        # The one real check — the transition into ``processing`` — still runs.
        assert ("created", "processing") in pairs, pairs

    def test_invalid_transition_warns_and_still_returns_200(self, client, deps, caplog):
        """The surviving check WARNS. It does not raise, 409, or skip the write."""
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value={"state_type": "created"})
        deps["state_manager"].validate_state_transition = MagicMock(return_value=False)

        with caplog.at_level("WARNING", logger="src.api.workers.message_endpoints"):
            resp = _post_msg(client)

        assert resp.status_code == 200, "an invalid transition must NEVER reject"
        assert "Invalid state transition" in caplog.text
        assert "-> processing" in caplog.text
        # No completion warning exists to emit any more.
        assert "-> completed" not in caplog.text, caplog.text
        # And both rows were still written.
        written = [
            c.kwargs.get("state_type") for c in deps["state_db"].add_worker_state.await_args_list
        ]
        assert "processing" in written
        assert "completed" in written

    @staticmethod
    def _validated_by_state(deps):
        return {
            c.kwargs.get("state_type"): c.kwargs.get("validated")
            for c in deps["state_db"].add_worker_state.await_args_list
        }

    def test_validated_is_true_when_a_current_state_was_read(self, client, deps):
        """A validator call actually ran for the ``processing`` write ⇒ "true"."""
        deps["state_db"].get_latest_worker_state = AsyncMock(
            return_value={"state_type": "processing"}
        )
        resp = _post_msg(client)
        assert resp.status_code == 200
        by_state = self._validated_by_state(deps)
        assert by_state.get("processing") == "true"
        assert by_state.get("completed") == "by_construction"

    def test_processing_write_is_false_when_no_current_state_exists(self, client, deps):
        """No prior state ⇒ the validator never ran ⇒ that write must not claim it did.

        ⭐ ``"false"`` is the EXPOSURE number, and this site is genuinely part of
        it. The ``completed`` write is a different case entirely and stays
        ``"by_construction"``: its from-state is ``processing`` by construction
        and valid by STATE_TRANSITIONS, so it is provably good even though no
        call is made — it is neither coverage nor exposure.
        """
        deps["state_db"].get_latest_worker_state = AsyncMock(return_value=None)
        resp = _post_msg(client)
        assert resp.status_code == 200
        by_state = self._validated_by_state(deps)
        assert by_state.get("processing") == "false"
        assert by_state.get("completed") == "by_construction"

    def test_error_writes_stay_false_because_nothing_proves_them_good(self, client, deps):
        """⛔ The rescue/error writes belong in the EXPOSURE number.

        No validator runs on those paths and nothing about the site proves the
        transition good, so they must resolve to ``"false"`` — via
        ``add_worker_state``'s default, i.e. by passing no ``validated`` at all.
        Passing anything else there would be the bug.
        """
        deps["worker_mock"].run = AsyncMock(side_effect=RuntimeError("model crashed"))
        with patch(f"{_MP}.asyncio.sleep", new_callable=AsyncMock):
            _post_msg(client)

        error_calls = [
            c
            for c in deps["state_db"].add_worker_state.await_args_list
            if c.kwargs.get("state_type") == "error"
        ]
        assert error_calls, "expected at least one error state write"
        for call in error_calls:
            assert "validated" not in call.kwargs, call.kwargs

        # ...and omitting it really does resolve to the exposure value.
        import inspect

        from src.db.workers_state_db import StateDatabase

        default = inspect.signature(StateDatabase.add_worker_state).parameters["validated"].default
        assert default == "false"

    # -- ⭐ mutation check ---------------------------------------------------

    def test_completion_validated_is_by_construction_NOT_true(self, client, deps):
        """⭐ MUTATION GUARD. Fails if the completion site is changed to "true".

        The distinction is load-bearing, not decorative. ``"true"`` means a
        ``validate_state_transition`` call ACTUALLY RAN, and at the completion
        site none does — the check there was deleted as unreachable. Labelling
        it "true" would make
        ``worker_management_worker_state_writes_total{validated="true"}``
        over-report the validator's own coverage by the entire completion
        population (~250k rows / 30d), which is precisely the flattering number
        this counter exists to expose.

        It also pins the two sites APART: with a current state present, the
        ``processing`` write is "true" and the ``completed`` write is
        "by_construction" in the SAME request, so unifying them fails here too.
        """
        deps["state_db"].get_latest_worker_state = AsyncMock(
            return_value={"state_type": "processing"}
        )
        resp = _post_msg(client)
        assert resp.status_code == 200
        by_state = self._validated_by_state(deps)

        completed = by_state.get("completed")
        assert completed == "by_construction", (
            f"completion write labelled {completed!r}; only 'by_construction' is honest here"
        )
        assert completed != "true", "no validator call runs at the completion site"
        assert completed != "false", "PROCESSING -> COMPLETED is provably valid; not exposure"
        # The two sites must not be unified.
        assert by_state.get("processing") != completed
