"""``MessageResponse.metadata`` reports the prompt that was ACTUALLY used.

This is the acceptance gate for roadmap item A5 (``n8n_custom_nodes``
``docs/redesign/ROADMAP.md`` §10): *"MessageResponse.metadata returns the prompt
sha actually used"*.

⚠️ WHICH READING THIS PROVES. The gate does not say whose message. These tests
prove the HAND-ISSUED reading: a direct ``POST /workers/{name}/message`` — the
shape ``sendWorkerMessage`` produces — gets back a digest of the prompt the
agent instance it ran on was carrying, including when the prompt was supplied
per-message. The stronger reading (a conveyor-originated call) cannot be
demonstrated at all right now: the conveyor is deliberately quiesced, so there
is no card traffic to observe. Nothing here claims it.

WHY A STUB THAT READS ``config``. ``worker_cache`` is mocked in these tests, so
a bare ``MagicMock`` instance would carry no prompt and the echo would fall back
to the record — which would make the per-message-override test pass for the
wrong reason. ``_instance_from_worker_data`` therefore builds its stub the way
``_globals._create_worker_instance`` builds the real agent: from
``worker_data["config"]["system_prompt"]``, the value the endpoint folds a
per-message override into. The assertions then pin BOTH that the digest matches
the override AND that it differs from the record's prompt, so an endpoint that
dropped the override again (it did once — 12 sentinel-bearing calls, 0
honoured) turns these red rather than green.
"""

from __future__ import annotations

import inspect
import os
import re
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.prompts.hashing import sha256_text
from src.api.worker_management_api import app
from src.api.workers.prompt_attribution import METADATA_KEYS
from src.auth import get_tenant_id
from src.worker.agents.cli_agent import CLIAgent, CLIAgentConfig
from src.worker.agents.pydantic_ai_agent import (
    PydanticAIAgent,
    PydanticAIAgentConfig,
    create_pydantic_ai_agent,
)
from src.worker.agents.remote_cli_agent import RemoteCLIAgent, RemoteCLIAgentConfig

# The literal `create_pydantic_ai_agent` substitutes for a falsy system_prompt.
# Pinned as a constant so the contract test below can assert the source of truth
# still says this, rather than the double quietly inventing its own default.
DEFAULT_AGENT_SYSTEM_PROMPT = "You are a helpful assistant."

TENANT = "test-tenant-attribution"
WORKER_ID = "wid-attr-1"
_MP = "src.api.workers.message_endpoints"

RECORD_PROMPT = "You are the worker recorded in the registry."
OVERRIDE_PROMPT = "You are the prompt this ONE message asked for."


@asynccontextmanager
async def _noop_ctx(*_a, **_kw):
    yield "fake-lock"


def _worker_data(*, config_revision: str | None = None, system_prompt: str | None = RECORD_PROMPT):
    config: dict[str, Any] = {
        "agent_type": "pydantic_ai",
        "model_provider": "mlx",
        "model": "mlx",
        "temperature": 0.1,
    }
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    data: dict[str, Any] = {
        "worker_name": "attr_worker",
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        "config": config,
        "status": "active",
    }
    if config_revision is not None:
        data["config_revision"] = config_revision
    return data


def _run_response(output="Hello!"):
    resp = MagicMock()
    resp.output = output
    resp.data = output
    resp.usage_metrics = None
    resp.session_id = None
    resp.cost_usd = None
    resp.duration_ms = None
    return resp


def _instance_from_worker_data(*_args, **kwargs):
    """Stand in for the real agent builder, IN THE REAL OBJECT SHAPE.

    ⛔ THIS DOUBLE'S SHAPE IS THE TEST. It used to set a top-level
    ``instance.system_prompt``. **No agent class in this service has that
    attribute** — ``PydanticAIAgent``, ``CLIAgent`` and ``RemoteCLIAgent`` all
    store their config object on ``self.config`` and the prompt lives at
    ``self.config.system_prompt``. A double of the wrong shape made the
    ``"instance"`` branch of the echo look exercised while production could only
    ever reach the ``"config"`` fallback, so the very divergence
    ``prompt_attribution`` exists to expose was invisible and the suite was
    green. ``TestAgentObjectShapeContract`` below pins the real classes so this
    double cannot drift back out of shape.

    The chain mirrored here is the production one, end to end:
    ``_globals._create_worker_instance`` reads ``config["system_prompt"]`` →
    ``create_worker`` → ``create_agent("pydantic_ai", …)`` →
    ``create_pydantic_ai_agent``, which substitutes
    ``DEFAULT_AGENT_SYSTEM_PROMPT`` when the value is falsy →
    ``PydanticAIAgentConfig`` → ``PydanticAIAgent.config``.

    The ``config`` object is a REAL ``PydanticAIAgentConfig``, not a namespace,
    so a field rename upstream breaks this double loudly instead of silently
    handing the echo a mock.
    """
    worker_data = kwargs.get("worker_data")
    if worker_data is None and len(_args) > 1:
        worker_data = _args[1]
    config = (worker_data or {}).get("config") or {}
    instance = MagicMock()
    instance.run = AsyncMock(return_value=_run_response())
    instance.config = PydanticAIAgentConfig(
        name="attr_worker",
        model_provider="mlx",
        model="mlx",
        db_path="/tmp/test.db",
        # `create_pydantic_ai_agent` does exactly this `or`, so a worker with no
        # prompt still runs on a real, known string.
        system_prompt=config.get("system_prompt") or DEFAULT_AGENT_SYSTEM_PROMPT,
        instruction=config.get("instruction"),
    )
    # A real agent has NO top-level `system_prompt`; `getattr(inst, ...)` on one
    # returns None. A bare MagicMock would answer a mock instead, which would
    # let a fallback-ordering bug pass unnoticed.
    del instance.system_prompt
    return instance


@pytest.fixture
def deps():
    with (
        patch(f"{_MP}.worker_registry") as mock_registry,
        patch(f"{_MP}.worker_cache") as mock_cache,
        patch(f"{_MP}.state_manager") as mock_state_mgr,
        patch(f"{_MP}.StateDatabase") as MockStateDB,
    ):
        mock_state_mgr.atomic_worker_operation = _noop_ctx
        mock_state_mgr.resolve_worker_id = AsyncMock(return_value=(WORKER_ID, True))
        mock_state_mgr.validate_state_transition = MagicMock(return_value=True)

        mock_db = AsyncMock()
        mock_db.get_latest_worker_state = AsyncMock(return_value=None)
        mock_db.add_worker_state = AsyncMock(return_value=1)
        MockStateDB.return_value = mock_db

        mock_cache.get_or_create_worker = AsyncMock(side_effect=_instance_from_worker_data)
        mock_cache.remove_worker = AsyncMock()
        mock_cache.remove_instance = AsyncMock()

        mock_registry.get_worker = AsyncMock(return_value=_worker_data())
        yield {"registry": mock_registry, "cache": mock_cache}


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


def _send(client, **extra):
    return client.post("/workers/attr_worker/message", json={"message": "Hello", **extra})


class TestPromptShaIsEchoed:
    def test_every_attribution_key_is_always_present(self, client):
        """An absent key is indistinguishable from an older server; an explicit
        null says 'this call could not be attributed', which is a finding."""
        body = _send(client).json()
        assert body["metadata"] is not None
        for key in METADATA_KEYS:
            assert key in body["metadata"], f"{key} missing from metadata"

    def test_sha_is_of_the_record_prompt_when_no_override(self, client):
        meta = _send(client).json()["metadata"]
        assert meta["system_prompt_sha256"] == sha256_text(RECORD_PROMPT)
        assert meta["system_prompt_source"] == "instance"

    def test_a_hand_issued_per_message_prompt_is_what_gets_digested(self, client):
        """THE GATE. A ``sendWorkerMessage`` carrying its own systemPrompt gets
        back the sha of THAT prompt — not of the prompt on the worker row."""
        meta = _send(client, system_prompt=OVERRIDE_PROMPT).json()["metadata"]
        assert meta["system_prompt_sha256"] == sha256_text(OVERRIDE_PROMPT)
        # Mutation control: if the endpoint dropped the override (as it once
        # did), the digest would be the record's and this line goes red.
        assert meta["system_prompt_sha256"] != sha256_text(RECORD_PROMPT)

    def test_stamped_revision_is_parsed_into_id_and_version(self, client, deps):
        deps["registry"].get_worker = AsyncMock(
            return_value=_worker_data(config_revision="worker-decide-keywords@4")
        )
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"] == "worker-decide-keywords@4"
        assert meta["prompt_id"] == "worker-decide-keywords"
        assert meta["prompt_version"] == 4

    def test_free_text_revision_yields_null_id_and_version(self, client, deps):
        """Today's reality: 2 of 793 workers carry a revision and neither is an
        S4 stamp. A canary worker name must NOT become a plausible version."""
        deps["registry"].get_worker = AsyncMock(
            return_value=_worker_data(config_revision="canary-decide-keywords-v2")
        )
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"] == "canary-decide-keywords-v2"
        assert meta["prompt_id"] is None
        assert meta["prompt_version"] is None
        # …and the digest is still reported, so an unstamped worker is still
        # attributable by content.
        assert meta["system_prompt_sha256"] == sha256_text(RECORD_PROMPT)

    def test_unstamped_worker_reports_null_revision_not_a_default(self, client):
        meta = _send(client).json()["metadata"]
        assert meta["worker_config_revision"] is None
        assert meta["prompt_id"] is None
        assert meta["prompt_version"] is None

    def test_worker_with_no_prompt_digests_the_substituted_framework_default(self, client, deps):
        """A promptless worker still RAN a prompt — say which one.

        ⚠️ THIS TEST USED TO PIN ``sha=None, source="none"`` and it was wrong,
        for the same reason the double was: it described a code path production
        cannot take. ``create_pydantic_ai_agent`` does
        ``system_prompt=system_prompt or "You are a helpful assistant."``, so a
        worker whose row carries no prompt runs on that literal — the text is
        baked into ``instructions=`` and the model really sees it. Reporting
        ``null`` would mean "this call could not be attributed", which is a
        FALSE statement about a call whose prompt text is known exactly.

        A consumer loses nothing: the digest of the framework default is a
        single well-known constant, so "unstamped worker on the stock prompt" is
        still trivially distinguishable from a real published prompt — and it is
        now distinguishable from "we have no idea", which ``null`` conflated it
        with.
        """
        deps["registry"].get_worker = AsyncMock(return_value=_worker_data(system_prompt=None))
        meta = _send(client).json()["metadata"]
        assert meta["system_prompt_sha256"] == sha256_text(DEFAULT_AGENT_SYSTEM_PROMPT)
        assert meta["system_prompt_source"] == "instance"
        # Negative control: this must not silently be the record's prompt.
        assert meta["system_prompt_sha256"] != sha256_text(RECORD_PROMPT)

    def test_no_prompt_anywhere_still_reports_null_sha(self, client, deps):
        """The ``"none"`` branch is still reachable, and still pinned.

        An agent object carrying no usable prompt on EITHER hop, against a row
        that has none either, is genuinely unattributable — and that, not the
        promptless-worker case above, is what ``null`` is for.
        """

        def _opaque_instance(*_a, **_kw):
            inst = MagicMock()
            inst.run = AsyncMock(return_value=_run_response())
            inst.config = None
            del inst.system_prompt
            return inst

        deps["cache"].get_or_create_worker = AsyncMock(side_effect=_opaque_instance)
        deps["registry"].get_worker = AsyncMock(return_value=_worker_data(system_prompt=None))
        meta = _send(client).json()["metadata"]
        assert meta["system_prompt_sha256"] is None
        assert meta["system_prompt_source"] == "none"

    def test_instance_without_a_usable_prompt_falls_back_to_the_record(self, client, deps):
        """Not every agent object answers a ``str`` on either hop. The echo
        falls back to the config value and SAYS it did, rather than digesting
        whatever the attributes happened to be.

        Both hops are made non-``str`` here on purpose: ``config.system_prompt``
        AND the legacy top-level attribute. If only one were, this would pass
        while the other hop was broken.
        """

        def _opaque_instance(*_a, **_kw):
            inst = MagicMock()
            inst.run = AsyncMock(return_value=_run_response())
            inst.config = MagicMock()
            inst.config.system_prompt = 12345  # not a str
            inst.system_prompt = 12345  # not a str
            return inst

        deps["cache"].get_or_create_worker = AsyncMock(side_effect=_opaque_instance)
        meta = _send(client).json()["metadata"]
        assert meta["system_prompt_sha256"] == sha256_text(RECORD_PROMPT)
        assert meta["system_prompt_source"] == "config"

    def test_a_flat_system_prompt_attribute_is_still_honoured(self, client, deps):
        """The legacy top-level read is a FALLBACK, not a deletion.

        No agent class exposes one today (``TestAgentObjectShapeContract``
        measures exactly that), but an agent that did would be carrying a real
        running prompt, and dropping the read would silently demote it to
        ``"config"``.
        """
        flat_prompt = "You are an agent that publishes its prompt flat."

        def _flat_instance(*_a, **_kw):
            inst = MagicMock()
            inst.run = AsyncMock(return_value=_run_response())
            inst.config = None
            inst.system_prompt = flat_prompt
            return inst

        deps["cache"].get_or_create_worker = AsyncMock(side_effect=_flat_instance)
        meta = _send(client).json()["metadata"]
        assert meta["system_prompt_sha256"] == sha256_text(flat_prompt)
        assert meta["system_prompt_source"] == "instance"

    def test_caller_context_is_preserved_beside_the_attribution(self, client):
        """Back-compat: ``metadata.context`` keeps its exact previous shape."""
        meta = _send(client, context={"card_id": "abc"}).json()["metadata"]
        assert meta["context"] == {"card_id": "abc"}
        assert meta["system_prompt_sha256"] == sha256_text(RECORD_PROMPT)

    def test_context_absent_stays_absent(self, client):
        assert "context" not in _send(client).json()["metadata"]

    def test_a_per_message_instruction_does_not_move_the_digest(self, client):
        """⚠️ DELIBERATE, DOCUMENTED GAP — ``instruction`` is NOT in the digest.

        What the agent really runs is the JOIN of both prompt fields:
        ``pydantic_ai_agent.py`` builds ``combined_instructions`` from
        ``config.system_prompt`` and ``config.instruction`` separated by a blank
        line, and ``MessageRequest.instruction`` is a per-message override with
        its own cache slot. So two calls that differ only in ``instruction``
        genuinely run different text and get the SAME ``system_prompt_sha256``.

        Folding the join in was rejected, and this test is the record of that
        choice rather than a comment that can rot:

        1. The key is named ``system_prompt_sha256``. A digest over
           system_prompt + instruction would make the NAME false, which is the
           same class of defect as the echo this test file exists to fix.
        2. It is the join key with ``prompt_versions.system_prompt_sha256``
           (migration 030), which digests the ``system`` text ALONE. Folding
           ``instruction`` in would make every worker that has one fail to match
           its own correctly-published ledger row — turning a true stamp into a
           false mismatch. A false negative is still a false stamp.

        The claim the echo makes is therefore narrow and TRUE: *this is the
        digest of the system prompt that ran*. It is not, and does not say it
        is, a digest of everything the model saw.
        """
        base = _send(client).json()["metadata"]
        with_instruction = _send(client, instruction="Answer only in haiku.").json()["metadata"]
        assert with_instruction["system_prompt_sha256"] == base["system_prompt_sha256"]
        assert with_instruction["system_prompt_source"] == "instance"
        # Positive control: the digest DOES move for the field it covers, so the
        # equality above is a real scoping fact and not a dead echo.
        with_prompt = _send(client, system_prompt=OVERRIDE_PROMPT).json()["metadata"]
        assert with_prompt["system_prompt_sha256"] != base["system_prompt_sha256"]


class TestAgentObjectShapeContract:
    """⛔ THE LEG THAT KEEPS THE DOUBLE HONEST.

    The defect these tests exist for was not in the echo's logic — it was that
    the double asserted a shape the real objects do not have, so a dead branch
    read as covered. Mocking can only ever prove what the mock was told. These
    assertions run against the REAL classes, so if the agents are refactored the
    double is caught diverging here instead of quietly going green again.
    """

    AGENT_CLASSES = (PydanticAIAgent, CLIAgent, RemoteCLIAgent)
    CONFIG_CLASSES = (PydanticAIAgentConfig, CLIAgentConfig, RemoteCLIAgentConfig)

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
    def test_no_agent_class_exposes_a_top_level_system_prompt(self, agent_cls):
        """If this ever goes red, the flat read in ``_instance_prompt`` stopped
        being a dead fallback and the double must grow the attribute too."""
        assert not hasattr(agent_cls, "system_prompt"), (
            f"{agent_cls.__name__} grew a top-level `system_prompt`; "
            "_instance_from_worker_data must be reshaped to match"
        )

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_every_agent_config_carries_the_system_prompt(self, config_cls):
        """The hop ``_instance_prompt`` actually reads. Positive control for the
        assertion above: the field is real, it is just not where it was read."""
        assert "system_prompt" in config_cls.model_fields, (
            f"{config_cls.__name__} no longer declares `system_prompt`; "
            "prompt attribution would silently report `config` for every call"
        )

    def test_the_factory_substitutes_this_literal_for_a_falsy_prompt(self):
        """⛔ PIN THE LITERAL PRODUCTION SUBSTITUTES, AT THE LINE THAT SUBSTITUTES IT.

        This assertion used to read
        ``PydanticAIAgentConfig.model_fields["system_prompt"].default``. That is
        a DIFFERENT literal from the one a promptless worker actually runs on:
        ``create_pydantic_ai_agent`` passes ``system_prompt=system_prompt or
        "You are a helpful assistant."`` **explicitly**, so the field default is
        never reached on that path. MEASURED at ``e81c569``: changing the
        factory's literal ALONE and touching nothing else left this file at
        **24 passed** — the contract test stayed green while
        ``test_worker_with_no_prompt_digests_the_substituted_framework_default``
        pinned a constant nothing uses. That is the same defect class this file
        exists to fix (a test pinning an object other than the one production
        reads), one tag over.

        So the config is captured at the factory's own construction site.
        ``PydanticAIAgent`` is patched out because constructing a real one
        initialises a model backend; the substitution under test happens before
        that call and is what is being measured.
        """
        captured: list[Any] = []

        class _CaptureConfig:
            def __init__(self, config: Any) -> None:
                captured.append(config)

        with patch(
            "src.worker.agents.pydantic_ai_agent.PydanticAIAgent",
            _CaptureConfig,
        ):
            create_pydantic_ai_agent(name="attr_worker", description="d", system_prompt=None)
            create_pydantic_ai_agent(
                name="attr_worker", description="d", system_prompt=RECORD_PROMPT
            )

        assert captured[0].system_prompt == DEFAULT_AGENT_SYSTEM_PROMPT, (
            "create_pydantic_ai_agent no longer substitutes DEFAULT_AGENT_SYSTEM_PROMPT; "
            "the promptless-worker test is now pinning a literal production does not run"
        )
        # Positive control: the substitution is an `or`, not an unconditional
        # overwrite, so a real prompt must survive it. Without this the
        # assertion above would also pass for a factory that ignored its
        # argument entirely.
        assert captured[1].system_prompt == RECORD_PROMPT

        # Second, WEAKER pin: the field default. Nothing on the promptless path
        # reads it (the factory always passes the field explicitly), but a
        # `PydanticAIAgentConfig` built without the kwarg elsewhere does.
        assert PydanticAIAgentConfig.model_fields["system_prompt"].default == (
            DEFAULT_AGENT_SYSTEM_PROMPT
        )

    def test_the_double_has_the_same_shape_as_a_real_agent(self):
        """⭐ THE ASSERTION THAT ACTUALLY STOPS THE REGRESSION RECURRING.

        MEASURED on this tree, not assumed: reverting
        ``_instance_from_worker_data`` to its exact pre-fix top-level shape
        while the fixed ``_instance_prompt`` is in place turns **2 of 24** red
        — this test, plus ONE endpoint test,
        ``test_worker_with_no_prompt_digests_the_substituted_framework_default``
        (a flat ``None`` makes the echo report ``sha=null``). It is NOT
        ``test_sha_is_of_the_record_prompt_when_no_override``, the test that
        originally exposed the defect: the flat read is kept as a fallback, so
        it dutifully rescues a wrongly-shaped double there.

        ⚠️ So the endpoint tests police the double's shape WEAKLY rather than
        not at all — one of them notices, by side effect, and only for the
        promptless case. What they cannot catch at ALL is the real classes
        drifting out from under the double. Measured on this same tree: giving
        ``PydanticAIAgent`` a class-attr ``system_prompt`` turns **1 of 24**
        red, and renaming ``self.config`` to ``self._cfg`` turns **1 of 24**
        red — and in both the single red test is in THIS class, with no other
        test in the file failing. That is what makes the contract load-bearing.
        """
        instance = _instance_from_worker_data(worker_data=_worker_data())
        assert not hasattr(instance, "system_prompt"), (
            "the double grew a top-level `system_prompt`; no real agent class "
            "has one, and a double that does re-hides the dead-branch defect"
        )
        assert isinstance(instance.config.system_prompt, str)
        assert instance.config.system_prompt == RECORD_PROMPT

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
    def test_every_agent_stores_its_config_on_self_config(self, agent_cls):
        """``self.config = config`` is the assignment the whole fix rests on.
        It is an ``__init__`` body, so it is asserted at the source level —
        constructing these agents needs a live model backend."""
        assert re.search(r"self\.config\s*=\s*config\b", inspect.getsource(agent_cls)), (
            f"{agent_cls.__name__} no longer assigns self.config; "
            "_instance_prompt reads instance.config.system_prompt"
        )
