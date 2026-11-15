"""Detection for BARE and DRIFTED worker records — the rows that run unconstrained.

A worker row is "bare" when it declares no ``output_schema`` in either place the
agent builder reads. Nothing then constrains or validates what the model
returns: ``_output_type_from_schema(None)`` is ``str`` and the completion comes
back freeform. karma depends on these rows to shape relationship extraction, so
a bare ``ks-relationship-extractor`` degrades extraction quality with no signal
anywhere.

Measured on prod 2026-09-11 against the production database, read-only::

    ks-relationship-extractor | 14 rows | 11 auto-created
      output_schema COLUMN null ............ 14 / 14
      config->'output_schema' absent/null ... 4 / 14   <- the one that matters

⚠️ RETRACTION, and it is load-bearing for this suite's fixtures. The finding
that opened this work was "14 of 14 have output_schema NULL, it is universal".
That is true of the ``workers.output_schema`` COLUMN and it discriminates
nothing whatsoever: the column is NULL for **779 of 779** rows in prod. It is
dead — never written. The schema that is actually read lives in
``config->'output_schema'`` (``crud_endpoints.py:610-612`` reads the column,
then falls back to config), and 145 rows carry a real one there. By the measure
that matters ks-relationship-extractor is bare on **4 of 14** rows, not 14 —
still a real defect across 4 tenants, but a quarter of the claimed size. A
fixture built on the column alone would have flagged all 779 rows and proven
only that the detector always fires.

⚠️ Second prod shape these fixtures encode: ``config->'output_schema'`` is JSON
``null`` on **615 of 779** rows — the key is PRESENT with a null value. A
``"output_schema" in config`` test reads every one of those as healthy.

What is NOT covered, deliberately
---------------------------------
Drift is only decidable where a contract is registered IN THIS IMAGE. The
``ks-*`` family's contracts are owned by a sibling service and this repo has no
copy, so for those workers bareness is detectable and drift is not. A populated
external schema books ``"healthy"`` rather than a guess — see
``test_drift_is_not_guessed_for_an_externally_owned_worker``.

⛔ Detection only. Nothing here asserts a repair, and nothing in the code under
test mutates a worker row: silently backfilling a schema would corrupt 14
tenants at once and is a separate decision.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.worker_management_api import app
from src.auth import get_tenant_id

_MP = "src.api.workers.message_endpoints"
_LOGGER = "src.api.workers.message_endpoints"
TENANT = "test-tenant-bare"
WORKER_ID = "wid-bare"

#: The live shape from prod: a system worker whose contract this repo does not
#: own, bare on 4 of its 14 rows.
KS_WORKER = "ks-relationship-extractor"
#: A locally-seeded improvement worker — the only tier where drift is decidable.
SEEDED_WORKER = "improvement_workflow_development"


# ---------------------------------------------------------------------------
# Fixtures — mirrors tests/test_system_worker_output_contract.py
# ---------------------------------------------------------------------------


def _bare_config() -> dict[str, Any]:
    """Byte-for-byte the live bare ks-relationship-extractor config.

    ``output_schema`` is present and JSON ``null`` — NOT absent. That is the
    shape 615 of 779 prod rows carry, and the one a key-presence check misses.
    """
    return {
        "agent_type": "pydantic_ai",
        "model_provider": "vllm",
        "model": "local-llm",
        "temperature": 0.1,
        "output_schema": None,
    }


def _worker_data(
    *,
    worker_name: str,
    config: dict[str, Any] | None = None,
    status: str = "auto-created",
) -> dict[str, Any]:
    return {
        "worker_name": worker_name,
        "worker_id": WORKER_ID,
        "db_path": "/tmp/test.db",
        "is_agent": True,
        # The dead column, faithfully NULL as it is on all 779 prod rows.
        "output_schema": None,
        "config": _bare_config() if config is None else config,
        "status": status,
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

        mock_registry.get_worker = AsyncMock(return_value=_worker_data(worker_name=KS_WORKER))
        mock_registry.register_worker = AsyncMock(return_value={"worker_id": WORKER_ID})

        fake_instance = MagicMock()
        fake_instance.db_path = "/tmp/auto.db"
        mock_create.return_value = fake_instance

        mock_mlx.get_model = AsyncMock(return_value=None)
        mock_mlx.get_instance_for_model = AsyncMock(return_value=None)

        yield {"registry": mock_registry, "cache": mock_cache, "worker_mock": worker_mock}


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


def _post(client, worker_name: str = KS_WORKER, **extra):
    return client.post(f"/workers/{worker_name}/message", json={"message": "Hello", **extra})


def _contract_warnings(caplog, worker_name: str) -> list[str]:
    """WARNING records naming ``worker_name`` and an output-contract problem.

    Matches on the message TEXT, not on a logger call count: this suite has to
    be runnable against the PRE-FIX code to prove it goes red, so it cannot
    import or patch any symbol the fix introduces.
    """
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and worker_name in rec.getMessage()
        and "contract" in rec.getMessage().lower()
    ]


# ---------------------------------------------------------------------------
# THE RED PROOF — today's code produces no signal for this row
# ---------------------------------------------------------------------------


def test_a_bare_ks_relationship_extractor_row_is_detected(client, caplog):
    """⭐ The headline. A bare ``ks-relationship-extractor`` row must produce a signal.

    Before the fix the ONLY bare-record warning in the service lived inside
    ``apply_registered_system_schema``, which returns early at ``if not
    schema`` whenever the worker has no locally registered contract. So the
    warning was a side effect of a successful REPAIR, never a detector, and a
    worker this repo cannot repair produced nothing at all — silence that is
    indistinguishable from health.
    """
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resp = _post(client)

    assert resp.status_code == 200, "detection must not change the outcome of the call"

    warnings = _contract_warnings(caplog, KS_WORKER)
    assert warnings, (
        f"a bare {KS_WORKER} row (config output_schema=null, the live prod shape on "
        f"4 of 14 tenants) produced NO warning about its output contract. karma's "
        f"relationship extraction is running unconstrained and nothing anywhere says so."
    )
    assert any("bare" in w for w in warnings), (
        f"the warning must name the verdict so an operator can act on it; got: {warnings}"
    )


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — mandatory. Without it the above proves only "always fires".
# ---------------------------------------------------------------------------


def test_a_healthy_worker_with_a_populated_schema_is_not_flagged(client, deps, caplog):
    """⛔ THE CONTROL. A row that DOES declare a contract must produce no warning.

    Same worker name, same endpoint, same fixtures — the ONLY difference is a
    populated ``config.output_schema``. If this also warns, the detector is a
    constant and every result above is vacuous.
    """
    healthy_config = _bare_config() | {
        "output_schema": {
            "type": "object",
            "properties": {"relationships": {"type": "array"}},
            "required": ["relationships"],
        }
    }
    deps["registry"].get_worker = AsyncMock(
        return_value=_worker_data(worker_name=KS_WORKER, config=healthy_config, status="updated")
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resp = _post(client)

    assert resp.status_code == 200
    assert not _contract_warnings(caplog, KS_WORKER), (
        "a HEALTHY row with a populated output_schema was flagged; the detector fires "
        "unconditionally and proves nothing"
    )


def test_the_negative_control_and_the_red_case_differ_only_in_the_schema():
    """Pin that the two fixtures above are one field apart.

    Without this, the control could silently drift into a different shape and
    pass for the wrong reason.
    """
    bare = _bare_config()
    healthy = _bare_config() | {"output_schema": {"type": "object"}}
    assert set(bare) == set(healthy)
    assert {k for k in bare if bare[k] != healthy[k]} == {"output_schema"}


# ---------------------------------------------------------------------------
# Classification — the unit level
# ---------------------------------------------------------------------------


def test_verdicts_for_each_row_shape():
    from src.api.workers.system_worker_contracts import (
        CONTRACT_BARE,
        CONTRACT_DRIFTED,
        CONTRACT_HEALTHY,
        registered_output_schema,
        worker_contract_verdict,
    )

    # JSON null in config — 615/779 prod rows.
    assert worker_contract_verdict(_worker_data(worker_name=KS_WORKER), KS_WORKER) == CONTRACT_BARE
    # Key absent entirely.
    assert (
        worker_contract_verdict({"config": {"agent_type": "pydantic_ai"}}, KS_WORKER)
        == CONTRACT_BARE
    )
    # No config at all.
    assert worker_contract_verdict({}, KS_WORKER) == CONTRACT_BARE
    # An EMPTY schema constrains nothing — bare as a matter of fact.
    assert worker_contract_verdict({"config": {"output_schema": {}}}, KS_WORKER) == CONTRACT_BARE
    # Populated.
    assert (
        worker_contract_verdict({"config": {"output_schema": {"type": "object"}}}, KS_WORKER)
        == CONTRACT_HEALTHY
    )
    # The dead top-level column still wins when it is somehow populated.
    assert (
        worker_contract_verdict({"output_schema": {"type": "object"}, "config": {}}, KS_WORKER)
        == CONTRACT_HEALTHY
    )
    # Drift: a registered contract exists and the row disagrees with it.
    assert (
        worker_contract_verdict(
            {"config": {"output_schema": {"type": "object", "title": "NotTheContract"}}},
            SEEDED_WORKER,
        )
        == CONTRACT_DRIFTED
    )
    # ...and the registered contract itself is healthy.
    assert (
        worker_contract_verdict(
            {"config": {"output_schema": registered_output_schema(SEEDED_WORKER)}},
            SEEDED_WORKER,
        )
        == CONTRACT_HEALTHY
    )


def test_drift_is_not_guessed_for_an_externally_owned_worker():
    """⛔ The honesty boundary. No local contract => drift is NOT decidable.

    ``ks-*`` contracts are owned by a sibling service and this image has no
    copy. A populated schema must book ``healthy``, never a fabricated
    ``drifted`` — a detector that invented a verdict it cannot support would be
    worse than the silence it replaced.
    """
    from src.api.workers.system_worker_contracts import (
        CONTRACT_HEALTHY,
        registered_output_schema,
        worker_contract_verdict,
    )

    assert registered_output_schema(KS_WORKER) is None, (
        "this repo unexpectedly owns a ks-* contract; the drift boundary must be revisited"
    )
    arbitrary = {"type": "object", "properties": {"anything": {"type": "string"}}}
    assert worker_contract_verdict({"config": {"output_schema": arbitrary}}, KS_WORKER) == (
        CONTRACT_HEALTHY
    )


def test_the_contract_label_is_bounded():
    """⛔ The cardinality bound. A raw worker_name must never become a label.

    ``POST /workers/{name}/message`` auto-creates whatever name it is handed
    and prod already carries 472 distinct names across 60 tenants.
    """
    from src.api.workers.system_worker_contracts import (
        CONTRACT_OTHER,
        classify_contract,
        contract_labels,
    )

    assert classify_contract(KS_WORKER) == KS_WORKER
    assert classify_contract(SEEDED_WORKER) == SEEDED_WORKER
    assert classify_contract("some-tenant-worker") == CONTRACT_OTHER
    assert classify_contract("ks-not-a-real-worker") == CONTRACT_OTHER, (
        "membership must be an explicit set, not a 'ks-' prefix match — a prefix "
        "test re-opens the unbounded-label hole"
    )

    labels = contract_labels()
    assert len(labels) == len(set(labels)), "duplicate label values"
    assert CONTRACT_OTHER in labels
    assert KS_WORKER in labels and SEEDED_WORKER in labels


# ---------------------------------------------------------------------------
# Generalisation: coverage is no longer a hand-written list of loop modules
# ---------------------------------------------------------------------------


def test_loop_discovery_matches_the_seeder_exactly():
    """The old ``_LOOP_MODULES`` tuple was hand-maintained and its own comment
    named the hazard: a sixth loop added to ``register.py`` and forgotten here
    would be silently uncovered. Discovery closes that, and this pins discovery
    against ``register.py``'s own merged ``AGENT_CONFIGS``."""
    from scripts.improvement_workers.register import AGENT_CONFIGS
    from src.api.workers.system_worker_contracts import loop_modules, registered_output_schemas

    assert len(loop_modules()) >= 5
    assert set(registered_output_schemas()) == set(AGENT_CONFIGS), (
        "the contracts module and the seeder disagree about which workers are registered"
    )


# ---------------------------------------------------------------------------
# The metric — it must exist at zero BEFORE anything is flagged
# ---------------------------------------------------------------------------


def _client_with_fake_meter():
    from src.telemetry.agent_metrics import AgentMetricsClient

    client = AgentMetricsClient.__new__(AgentMetricsClient)
    client.enabled = True
    client._counters = {}
    client._histograms = {}
    client._up_down_counters = {}
    client.meter = MagicMock()
    client.meter.create_counter.side_effect = lambda **_kw: MagicMock()
    client.meter.create_histogram.side_effect = lambda **_kw: MagicMock()
    client.meter.create_up_down_counter.side_effect = lambda **_kw: MagicMock()
    client._setup_metrics()
    return client


def test_the_metric_series_exist_at_zero_before_anything_is_flagged():
    """⭐ An ABSENT series is indistinguishable from a broken exporter.

    Every (contract, verdict) pair must be materialised with ``add(0, ...)``
    during setup, so "no bare rows" reads differently from "nothing is
    scraping this".
    """
    from src.api.workers.system_worker_contracts import (
        WORKER_CONTRACT_VERDICTS,
        contract_labels,
    )

    client = _client_with_fake_meter()
    counter = client._counters["worker_contract_verdict"]

    zeroed = {
        (c.args[1]["contract"], c.args[1]["verdict"])
        for c in counter.add.call_args_list
        if c.args[0] == 0
    }
    expected = {(c, v) for c in contract_labels() for v in WORKER_CONTRACT_VERDICTS}
    assert zeroed == expected, f"missing pre-materialised series: {sorted(expected - zeroed)}"

    # ...and nothing has been COUNTED yet: pre-init must not fabricate a finding.
    assert not [c for c in counter.add.call_args_list if c.args[0] != 0]


def test_the_counter_is_registered_under_a_stable_name():
    client = _client_with_fake_meter()
    names = {kw["name"] for _a, kw in client.meter.create_counter.call_args_list}
    assert "worker_management_worker_contract_verdicts_total" in names


def test_recording_a_verdict_labels_a_bounded_contract():
    from src.api.workers.system_worker_contracts import CONTRACT_BARE, CONTRACT_OTHER

    client = _client_with_fake_meter()
    counter = client._counters["worker_contract_verdict"]
    counter.add.reset_mock()

    client.record_worker_contract_verdict(KS_WORKER, CONTRACT_BARE)
    client.record_worker_contract_verdict("a-tenant-invented-name", CONTRACT_BARE)

    assert [c.args for c in counter.add.call_args_list] == [
        (1, {"contract": KS_WORKER, "verdict": CONTRACT_BARE}),
        (1, {"contract": CONTRACT_OTHER, "verdict": CONTRACT_BARE}),
    ]


def test_recording_never_raises_and_never_repairs():
    """Instrumentation must not turn a degraded completion into a failed request."""
    from src.api.workers.system_worker_contracts import CONTRACT_BARE

    client = _client_with_fake_meter()
    client._counters["worker_contract_verdict"].add.side_effect = RuntimeError("exporter down")
    client.record_worker_contract_verdict(KS_WORKER, CONTRACT_BARE)  # must not raise


def test_the_endpoint_records_the_bare_verdict(client, deps):
    """Wire-level: the endpoint must actually emit the counter, not just log."""
    from src.api.workers.system_worker_contracts import CONTRACT_BARE

    recorder = MagicMock()
    with patch(f"{_MP}.get_metrics_client", return_value=recorder):
        resp = _post(client)

    assert resp.status_code == 200
    recorder.record_worker_contract_verdict.assert_called_once_with(KS_WORKER, CONTRACT_BARE)


def test_a_per_message_schema_does_not_mask_a_bare_row(client, deps):
    """⛔ The verdict describes the ROW, not the request.

    A caller that supplies its own schema gets a constrained completion, but
    the stored row is still bare and still needs re-seeding. Classifying after
    the override would hide exactly the population this detector exists to
    find — which is why the call site sits BEFORE both the override and the
    repair.
    """
    from src.api.workers.system_worker_contracts import CONTRACT_BARE

    recorder = MagicMock()
    with patch(f"{_MP}.get_metrics_client", return_value=recorder):
        resp = _post(client, output_schema={"type": "object", "properties": {}})

    assert resp.status_code == 200
    recorder.record_worker_contract_verdict.assert_called_once_with(KS_WORKER, CONTRACT_BARE)
