"""Real-Postgres proof that a partial registration no longer nulls a worker's config.

Why this is a ``-m pg`` module and not a unit test
--------------------------------------------------
The subject here is *what Postgres does with the statement*: the ``jsonb ||``
operator, ``COALESCE`` type resolution across an explicitly-cast placeholder,
the ``$N -> %(pN)s`` rewrite in ``src/db/db_config.py``, and whether two
concurrent UPDATEs both survive. A hand-written double cannot answer any of
those; ``tests/db_test_helper.py`` models assignment *shape* only, and says so.

What went wrong in production
-----------------------------
``register_worker`` used to write ``config = $4`` — a wholesale replace — while
both handlers built that dict from ``request.<attr>`` unconditionally, so a
caller that OMITTED a field was indistinguishable from one that explicitly sent
``null``. Measured on prod 2026-08-28:

* ``worker-generate`` had a real prompt + model written at 2026-04-26 20:44:29;
  at **20:46:44**, two minutes later, a skeleton registration rewrote its
  ``system_prompt`` to the empty string and its ``model`` to NULL.
* ``knowledge/kag/src/wm_bootstrap.py`` PUTs 8 of ~25 config fields on every pod
  start (18x on 08-27, 24x on 08-26), so an operator edit to a kag worker had a
  half-life of hours.
* 0 of 750 rows had a non-NULL ``output_schema`` column, because neither handler
  ever passed the kwarg and the UPDATE named it unconditionally.
* 0 of the 207 ``status='updated'`` rows carried ``work_tier``, because PUT's
  payload dict had drifted from POST's and omitted four keys entirely.

Run locally:
    DATABASE_URL=postgresql://… uv run python scripts/init_db.py
    DATABASE_URL=postgresql://… uv run pytest tests/integration/test_worker_config_merge.py -m pg -v --no-cov
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool
from src.db.worker_registry_db import WorkerRegistryDB

from .pg_guard import PG_CONNECT_ERRORS, require_database_url, unreachable_pg
from .tenants import tenant_uuid

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    """Module-scoped PostgreSQL pool: skips locally, FAILS in CI, if unreachable."""
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture
async def tenant(pg_pool):
    """A throwaway tenant id; cleans up its workers on teardown."""
    tenant_id = tenant_uuid(f"int-cfgmerge-{uuid.uuid4().hex[:12]}")
    try:
        yield tenant_id
    finally:
        async with get_pg_connection() as conn:
            await conn.execute("DELETE FROM workers WHERE tenant_id = $1", tenant_id)


async def _stored(tenant_id: str, worker_name: str) -> dict:
    """Read the row straight from Postgres, bypassing every application helper."""
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            "SELECT config, is_agent, output_schema, output_validator, config_revision "
            "FROM workers WHERE tenant_id = $1 AND worker_name = $2",
            tenant_id,
            worker_name,
        )
    assert row is not None, f"{worker_name} not found in tenant {tenant_id}"
    return dict(row)


FULL_CONFIG = {
    "tool_names": [],
    "agent_role": "content generator",
    "agent_type": "pydantic_ai",
    "model_provider": "vllm",
    "model": "local-llm",
    "temperature": 0.35,
    "max_tokens": 2048,
    "system_prompt": "You expand an outline into prose.",
    "instruction": "Be concise.",
    "description": "seeded by an operator",
    "work_tier": "premium",
    "container_id": "c-123",
    "executor_endpoint": "http://executor:9000",
    "workspace": "/work",
}


async def _seed(db: WorkerRegistryDB, tenant_id: str, worker_name: str, **over) -> None:
    await db.register_worker(
        tenant_id=tenant_id,
        subject_id=tenant_id,
        worker_name=worker_name,
        db_path="/tmp/seed.db",
        is_agent=True,
        config={**FULL_CONFIG, **over},
        status="created",
    )


class TestOmittedKeysArePreserved:
    async def test_skeleton_registration_does_not_null_a_populated_config(self, tenant):
        """The worker-generate incident of 2026-04-26 20:46, replayed.

        A second registration carrying only the keys it knows about must leave
        every other key exactly as it was.
        """
        db = WorkerRegistryDB()
        await _seed(db, tenant, "worker-generate")

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="worker-generate",
            db_path="/tmp/seed.db",
            is_agent=None,
            # A bootstrapper's view of the world: it owns the role and the
            # description, and has never heard of model or system_prompt.
            config={"agent_role": "content generator", "description": "re-registered"},
            status="updated",
        )

        row = await _stored(tenant, "worker-generate")
        config = row["config"]
        assert config["model"] == "local-llm"
        assert config["system_prompt"] == "You expand an outline into prose."
        assert config["temperature"] == 0.35
        assert config["work_tier"] == "premium"
        # ...and the keys it DID send are applied.
        assert config["description"] == "re-registered"

    async def test_an_explicit_null_still_clears(self, tenant):
        """Negative control for the test above.

        Without this, "preserved" is indistinguishable from "the write was
        ignored". Sending a key with a null value must still erase it — only
        SILENCE is non-destructive.
        """
        db = WorkerRegistryDB()
        await _seed(db, tenant, "clearable")

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="clearable",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={"model": None, "system_prompt": None},
            status="updated",
        )

        config = (await _stored(tenant, "clearable"))["config"]
        assert config["model"] is None
        assert config["system_prompt"] is None
        assert config["temperature"] == 0.35, "an unrelated key must be untouched"

    async def test_the_kag_shape(self, tenant):
        """kag's real bootstrap body: 8 keys, sent on every pod start."""
        db = WorkerRegistryDB()
        await _seed(db, tenant, "kag_synthesize")

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="kag_synthesize",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={
                "agent_role": "kag /query/ask worker (kag_synthesize)",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
                "model": "default",
                "system_prompt": "Answer from the retrieved context.",
                "base_url_override": "http://inference-router:8080",
            },
            status="updated",
        )

        row = await _stored(tenant, "kag_synthesize")
        config = row["config"]
        # The 8 fields kag owns are applied...
        assert config["model"] == "default"
        assert config["base_url_override"] == "http://inference-router:8080"
        # ...and the ones it has never heard of survive its restart.
        assert config["temperature"] == 0.35
        assert config["max_tokens"] == 2048
        assert config["work_tier"] == "premium"
        assert config["container_id"] == "c-123"
        assert config["executor_endpoint"] == "http://executor:9000"
        assert config["workspace"] == "/work"


class TestSiblingColumnsAreNotNulled:
    async def test_is_agent_survives_a_caller_that_does_not_mention_it(self, tenant):
        """698 of 750 live rows are is_agent=true; a partial write must not flip them."""
        db = WorkerRegistryDB()
        await _seed(db, tenant, "agenty")
        assert (await _stored(tenant, "agenty"))["is_agent"] is True

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="agenty",
            db_path="/tmp/seed.db",
            is_agent=None,  # "the caller did not say"
            config={"description": "touched"},
            status="updated",
        )

        assert (await _stored(tenant, "agenty"))["is_agent"] is True

    async def test_output_schema_and_validator_columns_survive(self, tenant):
        """0 of 750 prod rows had these populated: every write named them unconditionally.

        Also the type-resolution test for the explicit casts — ``output_schema``
        is ``jsonb`` and ``output_validator`` is ``text``, so an uncast
        ``COALESCE($n, col)`` against a Python ``None`` fails to resolve.
        """
        db = WorkerRegistryDB()
        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="schemad",
            db_path="/tmp/seed.db",
            is_agent=True,
            config=dict(FULL_CONFIG),
            status="created",
            output_schema='{"type": "object"}',
            output_validator="def validate(x): return True",
            config_revision="rev-1",
        )

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="schemad",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={"description": "touched"},
            status="updated",
        )

        row = await _stored(tenant, "schemad")
        assert row["output_schema"] == {"type": "object"}
        assert row["output_validator"] == "def validate(x): return True"
        assert row["config_revision"] == "rev-1"


class TestMergeIsAtomic:
    async def test_concurrent_writers_touching_disjoint_keys_both_survive(self, tenant):
        """The reason the merge lives in SQL rather than in Python.

        A read-modify-write in the handler would let the later writer's stale
        read overwrite the earlier writer's key. ``config || $4`` inside the one
        UPDATE cannot.
        """
        db = WorkerRegistryDB()
        await _seed(db, tenant, "contended")

        async def write(key: str, value: str) -> None:
            await db.register_worker(
                tenant_id=tenant,
                subject_id=tenant,
                worker_name="contended",
                db_path="/tmp/seed.db",
                is_agent=None,
                config={key: value},
                status="updated",
            )

        await asyncio.gather(
            write("instruction", "from A"),
            write("description", "from B"),
            write("user_prompt", "from C"),
        )

        config = (await _stored(tenant, "contended"))["config"]
        assert config["instruction"] == "from A"
        assert config["description"] == "from B"
        assert config["user_prompt"] == "from C"
        assert config["model"] == "local-llm", "the seeded keys survived all three"


class TestTheReturnContractTheHandlersReadBackOff:
    """The seam every handler test replaces with a double.

    `crud_endpoints` builds the PUT/POST response body AND the `worker_states`
    audit metadata from what `register_worker` reports the row now holds. Four
    test modules stub that call, and until this change every one of those
    doubles returned `{"worker_id": ...}` and nothing else — which is exactly
    why they all stayed green while the handlers echoed the caller's own nulls.

    A double can only be trusted against a check of the real thing. These
    assertions are deliberately about SHAPE, in the same terms the handlers
    read: `reg_result.get("config")`, `.get("is_agent")`, `.get("config_revision")`.
    """

    async def test_the_update_branch_reports_the_persisted_row(self, tenant):
        db = WorkerRegistryDB()
        await _seed(db, tenant, "contract-update")

        result = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="contract-update",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={"description": "only this"},
            status="updated",
        )

        # Not the caller's payload — the merged row.
        assert result["config"]["description"] == "only this"
        assert result["config"]["model"] == "local-llm"
        assert result["is_agent"] is True, "None meant 'preserve', and the row held True"
        assert result["config"] == (await _stored(tenant, "contract-update"))["config"]
        # The keys crud_endpoints reads. A rename here is a silent 200-with-nulls.
        assert {"worker_id", "config", "is_agent", "config_revision"} <= set(result)

    async def test_the_insert_branch_reports_the_same_keys(self, tenant):
        """The create path must not hand back a differently-shaped dict."""
        db = WorkerRegistryDB()
        result = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="contract-insert",
            db_path="/tmp/seed.db",
            is_agent=True,
            config={"model": "m-1"},
            status="created",
        )
        assert {"worker_id", "config", "is_agent", "config_revision"} <= set(result)
        assert result["config"] == {"model": "m-1"}
        assert result["is_agent"] is True

    async def test_is_agent_is_never_none_in_the_return(self, tenant):
        """`is_agent=None` is an INPUT meaning "do not say"; it is never an output.

        `bool(reg_result.get("is_agent"))` in the handler would turn a leaked
        None into a false, so the column is NOT NULL and both branches resolve
        it before returning.
        """
        db = WorkerRegistryDB()
        created = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="contract-isagent",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={"model": "m"},
            status="created",
        )
        assert created["is_agent"] is False, "nothing to preserve on an INSERT"

        updated = await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="contract-isagent",
            db_path="/tmp/seed.db",
            is_agent=True,
            config={},
            status="updated",
        )
        assert updated["is_agent"] is True


class TestPromptSwapDurability:
    async def test_a_swapped_prompt_survives_the_next_registration(self, tenant):
        """The AA prompt-improvement loop's apply path, against a bootstrapper.

        ``PUT /workers/{name}/prompt`` is compare-and-swap guarded, but a CAS on
        the write is worthless if an unguarded re-registration overwrites it
        moments later. Production has exactly ONE ``prompt-swapped`` row so far,
        so this has not been hit yet — it would have been, at scale.
        """
        db = WorkerRegistryDB()
        await _seed(db, tenant, "improvable")

        swapped = await db.swap_prompt(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="improvable",
            new_system_prompt="IMPROVED: expand the outline, then cut 20%.",
            expected_current_prompt=FULL_CONFIG["system_prompt"],
            config_revision="canary-7",
        )
        assert swapped is not None, "CAS should have matched the seeded prompt"

        # A routine bootstrap re-registration lands right after the swap.
        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="improvable",
            db_path="/tmp/seed.db",
            is_agent=None,
            config={"agent_role": "content generator", "model_provider": "vllm"},
            status="updated",
        )

        row = await _stored(tenant, "improvable")
        assert row["config"]["system_prompt"] == "IMPROVED: expand the outline, then cut 20%."
        assert row["config_revision"] == "canary-7"


class TestFullReplaceIsStillReachable:
    async def test_merge_config_false_replaces_wholesale(self, tenant):
        """The escape hatch, so "merge" is a default rather than a cage."""
        db = WorkerRegistryDB()
        await _seed(db, tenant, "replaceable")

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name="replaceable",
            db_path="/tmp/seed.db",
            is_agent=False,
            config={"model": "only-this"},
            status="updated",
            merge_config=False,
        )

        row = await _stored(tenant, "replaceable")
        assert row["config"] == {"model": "only-this"}
        assert row["is_agent"] is False


class TestTheApiLayerAcceptsTheBodyTheDbLayerMerges:
    """The 422 regression of 2026-08-29, end to end against real Postgres.

    ``TestOmittedKeysArePreserved.test_the_kag_shape`` proves the DB layer
    merges kag's shape. It could not catch this defect, because it hands
    ``register_worker`` a config dict directly and never crosses the API
    validation that actually rejected the request.

    For 16 hours every kag seed 422'd with "model is required when
    model_provider is specified" — the cross-field rules ran at parse time
    against the body alone, while the write merged. The DB was always willing;
    the API never let the body through. This test spans both, so neither half
    can regress without it failing.
    """

    async def test_kags_seed_body_survives_validation_and_preserves_the_row(self, tenant):
        from src.api.workers.crud_endpoints import _already, _validated_config_payload
        from src.api.workers.models import WorkerCreateRequest

        db = WorkerRegistryDB()
        name = "kag_synthesize"
        await _seed(
            db,
            tenant,
            name,
            model_provider="inference_router",
            base_url_override="http://inference-router.example.com:9000",
            system_prompt="OPERATOR EDITED PROMPT",
            temperature=0.37,
        )

        # wm_bootstrap._spec_for_existing drops model / system_prompt /
        # base_url_override once the live worker has them. This is what is left.
        req = WorkerCreateRequest.model_validate(
            {
                "worker_name": name,
                "agent_role": "content generator",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
            }
        )
        stored = (await _stored(tenant, name))["config"]
        payload = await _validated_config_payload(req, lambda: _already(stored))

        await db.register_worker(
            tenant_id=tenant,
            subject_id=tenant,
            worker_name=name,
            db_path="/tmp/seed.db",
            is_agent=None,
            config=payload,
            status="updated",
        )

        after = (await _stored(tenant, name))["config"]
        assert after["model"] == "local-llm"
        assert after["system_prompt"] == "OPERATOR EDITED PROMPT"
        assert after["temperature"] == 0.37
        assert after["base_url_override"] == "http://inference-router.example.com:9000"

    async def test_the_same_body_is_still_refused_when_nothing_supplies_a_model(self, tenant):
        """The bound: relaxation comes from the stored row, not from the body."""
        from src.api.workers.crud_endpoints import _already, _validated_config_payload
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest.model_validate(
            {
                "worker_name": "kag_synthesize",
                "agent_role": "content generator",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
            }
        )
        with pytest.raises(Exception, match="model is required"):
            await _validated_config_payload(req, lambda: _already(None))


IR_SERVICE_API = "http://inference-router.inference-router.svc.cluster.local:8150/api"


class TestSeedPersistsAndReachesTheWire:
    """A PUT-shaped body carrying ``seed``: validated by the real API helper,
    merged by real Postgres, read back by the real worker cache, built into a
    real pydantic-ai agent, and observed in the outbound HTTP body.

    MUTATION: drop ``"seed"`` from ``_CONFIG_FIELDS`` ⇒ the payload never
    carries it ⇒ the stored row has no seed ⇒ the request body has no seed ⇒ red.
    """

    @staticmethod
    async def _put(db: WorkerRegistryDB, tenant_id: str, name: str, body: dict) -> None:
        from src.api.workers.crud_endpoints import _already, _validated_config_payload
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest.model_validate({"worker_name": name, **body})
        stored = (await _stored(tenant_id, name))["config"]
        payload = await _validated_config_payload(req, lambda: _already(stored))
        await db.register_worker(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            worker_name=name,
            db_path="/tmp/seed.db",
            is_agent=None,
            config=payload,
            status="updated",
        )

    @classmethod
    async def _next_message_body(cls, tenant_id: str, name: str) -> dict:
        """Send one message through the stored row and return the IR request JSON."""
        return (await cls._next_message_request(tenant_id, name))[0]

    @staticmethod
    async def _next_message_request(tenant_id: str, name: str) -> tuple[dict, dict]:
        """Send one message through the stored row; return the IR (body, headers)."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        import httpx

        from src.api.workers._globals import WorkerInstanceCache

        row = await _stored(tenant_id, name)
        bodies: list[dict] = []
        headers: list[dict] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            headers.append({k.lower(): v for k, v in request.headers.items()})
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        cache = WorkerInstanceCache()
        with (
            patch(
                "src.api.workers._globals.resolve_llm_config",
                new=AsyncMock(side_effect=lambda c, _tid, **_kw: c),
            ),
            patch(
                "src.services.inference_router_registry.inference_router_registry.get_model",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.worker.agents.pydantic_ai_agent.get_metrics_client", return_value=MagicMock()
            ),
        ):
            agent = await cache.get_or_create_worker(
                name, {"worker_name": name, "config": dict(row["config"])}, tenant_id=tenant_id
            )
            agent.agent.model.client._client._transport = httpx.MockTransport(_capture)
            await agent.run("hi")
        assert len(bodies) == 1
        return bodies[0], headers[0]

    async def _seed_ir_worker(self, db: WorkerRegistryDB, tenant_id: str, name: str) -> None:
        await _seed(
            db,
            tenant_id,
            name,
            model_provider="inference_router",
            model="local-llm",
            # The live customer shape (187 rows). A customer's override survives
            # worker build ONLY on the inference-router origin; any other host
            # is dropped (``enforce_customer_routing``), and pg CI has no
            # INFERENCE_ROUTER_URL to fall back to.
            base_url_override=IR_SERVICE_API,
            container_id=None,
            executor_endpoint=None,
            workspace=None,
            work_tier=None,
        )

    async def test_put_with_seed_persists_and_the_next_message_sends_it(self, tenant):
        db = WorkerRegistryDB()
        name = "seed_probe"
        await self._seed_ir_worker(db, tenant, name)
        before = await self._next_message_body(tenant, name)
        assert "seed" not in before  # control: nothing sent a seed before the PUT

        await self._put(db, tenant, name, {"seed": 42, "temperature": 0.0})

        assert (await _stored(tenant, name))["config"]["seed"] == 42
        after = await self._next_message_body(tenant, name)
        assert after["seed"] == 42
        assert after["temperature"] == 0.0
        assert after["model"] == "local-llm"

    async def test_put_without_seed_leaves_the_stored_seed_untouched(self, tenant):
        db = WorkerRegistryDB()
        name = "seed_probe"
        await self._seed_ir_worker(db, tenant, name)
        await self._put(db, tenant, name, {"seed": 7})
        await self._put(db, tenant, name, {"temperature": 0.5})

        cfg = (await _stored(tenant, name))["config"]
        assert cfg["seed"] == 7
        assert cfg["temperature"] == 0.5  # control: the second PUT did write

    async def test_put_ir_provider_persists_and_the_next_message_sends_x_provider(
        self, tenant, monkeypatch
    ):
        """PUT ir_provider → stored row → cached agent → IR receives X-Provider.

        MUTATIONS (each turns this red): ``_globals.py`` passes ir_provider=None;
        ``worker_factory.py`` passes ir_provider=None; ``agent_factory.py`` drops
        ``agent_kwargs["ir_provider"]``; ``"ir_provider"`` dropped from
        ``_CONFIG_FIELDS``.
        """
        db = WorkerRegistryDB()
        name = "pin_probe"
        await self._seed_ir_worker(db, tenant, name)
        _, before = await self._next_message_request(tenant, name)
        assert "x-provider" not in before  # control: no pin, no header

        await self._put(db, tenant, name, {"ir_provider": "pgx-local"})

        assert (await _stored(tenant, name))["config"]["ir_provider"] == "pgx-local"
        # Owner ruling 2026-10-01 (validator B2): the throwaway tenant is a
        # CUSTOMER, so its stored pin is dropped at worker build — no header.
        _, as_customer = await self._next_message_request(tenant, name)
        assert "x-provider" not in as_customer
        # The same stored row on a PLATFORM tenant still reaches the wire.
        from src.worker.factory import llm_config_resolver

        monkeypatch.setattr(
            llm_config_resolver, "is_commercial_platform_tenant", lambda t: t == tenant
        )
        _, after = await self._next_message_request(tenant, name)
        assert after.get("x-provider") == "pgx-local"

    async def test_seed_zero_persists_as_zero_not_null(self, tenant):
        db = WorkerRegistryDB()
        name = "seed_probe"
        await self._seed_ir_worker(db, tenant, name)
        await self._put(db, tenant, name, {"seed": 0})

        cfg = (await _stored(tenant, name))["config"]
        assert "seed" in cfg
        assert cfg["seed"] == 0
        assert (await self._next_message_body(tenant, name))["seed"] == 0

    async def test_put_with_repetition_controls_persists_and_reaches_the_wire(self, tenant):
        """Same chain for ``repetition_penalty`` / ``frequency_penalty``.

        MUTATION: drop ``"repetition_penalty"`` from ``_CONFIG_FIELDS`` ⇒ the
        stored row has none ⇒ the IR body has none ⇒ red.
        """
        db = WorkerRegistryDB()
        name = "rp_probe"
        await self._seed_ir_worker(db, tenant, name)
        before = await self._next_message_body(tenant, name)
        assert "repetition_penalty" not in before  # control
        assert "frequency_penalty" not in before

        await self._put(db, tenant, name, {"repetition_penalty": 1.1, "frequency_penalty": 0.2})

        cfg = (await _stored(tenant, name))["config"]
        assert cfg["repetition_penalty"] == 1.1
        assert cfg["frequency_penalty"] == 0.2
        after = await self._next_message_body(tenant, name)
        assert after["repetition_penalty"] == 1.1
        assert after["frequency_penalty"] == 0.2

        # Omitted on the next PUT ⇒ kept; explicit null ⇒ cleared (seed semantics).
        await self._put(db, tenant, name, {"temperature": 0.3})
        assert (await _stored(tenant, name))["config"]["repetition_penalty"] == 1.1
        await self._put(db, tenant, name, {"repetition_penalty": None})
        assert "repetition_penalty" not in await self._next_message_body(tenant, name)
