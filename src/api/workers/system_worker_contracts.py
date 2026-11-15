"""The registered output contract of an improvement-loop SYSTEM worker.

Why this module exists — measured on prod 2026-09-11
----------------------------------------------------

``scripts/improvement_workers/register.py`` seeds five workflow-loop workers
(plus the prompt / asset / procedure loops) into the registry, each carrying a
Draft-2020-12 ``output_schema`` loaded from
``scripts/improvement_workers/schemas/<loop>/<card>.json``. The n8n
``doing-workflow-canary`` pattern calls ``improvement_workflow_development``
with NO per-message ``output_schema``, deliberately: its own comment says
"WM enforces the registered output_schema (workflow_patch) server-side".

That worked only for as long as the caller kept authenticating as the tenant
the seed ran under. It stopped on 2026-08-27. Measured in
``worker_management.workers``::

    improvement_workflow_development | 0195f48b-…-000000000001 | updated      | config.output_schema = t
    improvement_workflow_development | 019de842-…-eb0d8c5396f9 | auto-created | config.output_schema = f

and in ``worker_states`` the traffic moved with it: the seeded row's last run
is 2026-08-27 10:55, and every one of the 93 runs since (to 2026-09-11 08:11)
landed on the ``019de842`` row.

The registry is keyed ``(tenant_id, subject_id, worker_name)``, so the lookup in
``message_endpoints`` missed, and the endpoint's **auto-create** branch happily
fabricated a worker with ``output_schema=None`` (plus no ``system_prompt`` and
``model_provider="vllm"``) and returned 200. From there
``_globals._create_worker_instance`` read ``None``, ``_output_type_from_schema``
returned ``str``, and the model answered freeform: the retained window shows
``output_type="string"`` on every completion, some of them markdown-fenced, and
~25% of them the 11-token decline ``{"workflow_patch": []}`` — carrying none of
the ``rationale`` / ``expected_impact`` / ``scope_of_change`` the schema
requires. ``output_schema_used: null`` on the response was therefore an honest
report, not a reporting bug: no schema was used.

The same thing happened to ``improvement_prompt_improvement`` and
``improvement_principle_improvement``.

What this module fixes
----------------------

The contract of a system improvement worker is not tenant data — it is a
static, in-image definition (``scripts/`` is COPYd into the runtime image, see
``Dockerfile:91``) and it is the *same source* ``register.py`` posts from. So a
call to one of those worker names can always recover its registered schema
locally, whichever tenant the call arrives under, without reading another
tenant's row.

⛔ This is deliberately ONLY ``output_schema``. It is not a general
cross-tenant config inheritance mechanism, and it must not grow into one:
``system_prompt``, ``model``, ``model_provider`` and ``temperature`` are
equally missing on those auto-created rows (a real, separate finding), but
restoring them is a registration/seeding decision, not something a message
handler should quietly do.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# src/api/workers/system_worker_contracts.py -> src/api/workers -> src/api -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]

_WORKERS_PKG = "scripts.improvement_workers.workers"
_WORKERS_DIR = _REPO_ROOT / "scripts" / "improvement_workers" / "workers"


@lru_cache(maxsize=1)
def loop_modules() -> tuple[str, ...]:
    """Every loop module under ``scripts/improvement_workers/workers/``.

    ⚠️ This used to be a hand-written tuple of five module names, and its own
    comment named the hazard it could not prevent: "adding a loop there and
    forgetting it here would silently leave that loop's workers uncovered,
    which is the exact failure shape above". Discovering the package contents
    removes the hand-maintained half, so a sixth loop is covered the moment it
    is added rather than the moment somebody remembers this file.

    ``tests/test_bare_worker_contract_detection.py`` pins this against
    ``register.py``'s own ``AGENT_CONFIGS`` merge, so discovery and the seeder
    cannot disagree about what is registered.
    """
    try:
        return tuple(
            sorted(path.stem for path in _WORKERS_DIR.glob("*.py") if not path.stem.startswith("_"))
        )
    except OSError:
        logger.exception(
            "could not enumerate improvement-worker loop modules at %s; "
            "no registered output schema can be recovered or checked",
            _WORKERS_DIR,
        )
        return ()


@lru_cache(maxsize=1)
def registered_output_schemas() -> dict[str, dict[str, Any]]:
    """``worker_name -> output_schema`` for every seeded improvement worker.

    Read from the loop modules' own ``AGENT_CONFIGS`` rather than by guessing a
    filename from the worker name: the name→schema mapping lives there and
    nowhere else, and a convention ("strip the ``improvement_`` prefix, look for
    a same-named file") would be an incidental truth that quietly breaks the
    first time a loop names a worker differently from its schema file.

    Returns an empty mapping — never raises — if the seed package cannot be
    imported, so a packaging change can never 500 a live message. That failure
    IS logged at ERROR, and ``tests/test_system_worker_output_contract.py``
    asserts the mapping is non-empty so CI fails loudly instead.
    """
    root = str(_REPO_ROOT)
    if root not in sys.path:
        # Same hop ``scripts/improvement_workers/register.py`` makes to import
        # these modules. The runtime image has them at /app/scripts.
        sys.path.insert(0, root)

    schemas: dict[str, dict[str, Any]] = {}
    for module_name in loop_modules():
        try:
            module = __import__(
                f"{_WORKERS_PKG}.{module_name}",
                fromlist=["AGENT_CONFIGS"],
            )
        except Exception:
            logger.exception(
                "improvement-worker seed module %s is not importable; "
                "its registered output schemas cannot be enforced",
                module_name,
            )
            continue
        # Discovery is by directory now, so a shared helper module that carries
        # no AGENT_CONFIGS is a normal member of the package, not a fault.
        # ⛔ Must stay a quiet skip: an ERROR here would train operators to
        # ignore this logger, and the import failure above is the line that has
        # to stay loud.
        configs = getattr(module, "AGENT_CONFIGS", None)
        if not isinstance(configs, Mapping):
            continue
        for name, config in configs.items():
            try:
                schema = config.output_schema()
            except Exception:
                logger.exception("could not load registered output_schema for worker %s", name)
                continue
            if isinstance(schema, dict) and schema:
                schemas[name] = schema
    return schemas


def registered_output_schema(worker_name: str) -> dict[str, Any] | None:
    """The seeded ``output_schema`` for ``worker_name``, or None if it is not one."""
    return registered_output_schemas().get(worker_name)


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION: is the row this request resolved to carrying its contract?
#
# ⛔ OBSERVE-ONLY, AND DELIBERATELY SO. Nothing below mutates a worker row, and
# none of it may be turned into a write on its own evidence. Backfilling a
# schema onto a bare row is a decision about 14 tenants at once, and a detector
# that repaired what it measured could never report the population it repaired.
#
# WHY DETECTION HAD TO BE SPLIT FROM REPAIR. Before this, the only bare-record
# signal in the service was the WARNING inside
# ``message_endpoints.apply_registered_system_schema`` — and that function
# returns early at ``if not schema`` whenever the worker has no LOCALLY
# registered contract. The warning was therefore a side effect of a successful
# repair, not a detector: a worker this repo cannot repair produced no signal
# at all. Measured on prod 2026-09-11, that is the entire ``ks-*`` family.
#
# ⚠️ MEASUREMENT CORRECTION, worth keeping. The finding that opened this work
# was "14 of 14 ``ks-relationship-extractor`` rows have ``output_schema`` NULL".
# That is true of the ``workers.output_schema`` COLUMN and it discriminates
# nothing: the column is NULL for **779 of 779** rows in prod — it is dead,
# never written (``crud_endpoints.py:610-612`` already reads the column then
# falls back to ``config``). The schema that is actually read lives in
# ``config->'output_schema'``, and by that measure ks-relationship-extractor is
# bare on **4 of 14** rows, not 14. Both places are checked below, in that
# order, precisely so this detector cannot inherit the dead-column artefact.
# ═══════════════════════════════════════════════════════════════════════════

CONTRACT_BARE = "bare"
CONTRACT_DRIFTED = "drifted"
CONTRACT_HEALTHY = "healthy"

#: The closed set of verdicts. Every one is pre-materialised at zero against
#: every label in :func:`contract_labels`, so "no bare rows" and "the exporter
#: is broken" are distinguishable — an unobserved combination has no series,
#: and absence reads identically to health.
WORKER_CONTRACT_VERDICTS: tuple[str, ...] = (
    CONTRACT_HEALTHY,
    CONTRACT_BARE,
    CONTRACT_DRIFTED,
)

#: Label value for a worker that is not a known system contract.
#:
#: ⛔ The raw ``worker_name`` must NEVER reach a label. It is a path parameter
#: and ``POST /workers/{name}/message`` AUTO-CREATES whatever it is handed, so
#: it is unbounded and caller-controlled — the same untrusted input this
#: detector exists to characterise. Prod already carries **472 distinct**
#: worker names across 60 tenants (measured 2026-09-11). Bareness on an
#: unlisted worker is still COUNTED here, just not named: invisible is the one
#: thing it is not.
CONTRACT_OTHER = "<other>"

#: System workers whose output contract is owned by a SIBLING service, not by
#: this repo. Their bareness is detectable; their drift is not (see
#: :func:`worker_contract_verdict`), because there is no local contract to
#: compare a live config against.
#:
#: This is the ``ks-*`` knowledge-system family that karma calls for entity /
#: relationship extraction. The list is explicit rather than a ``ks-`` prefix
#: match ON PURPOSE: a prefix test would re-open the cardinality hole above,
#: since ``ks-<anything>`` is auto-createable by any caller.
#:
#: Enumerated from prod 2026-09-11 — every ``ks-*`` worker_name with at least
#: one bare row. Adding a name is a deliberate edit; omitting one costs
#: visibility of WHICH contract, never the count itself.
EXTERNAL_SYSTEM_WORKERS: frozenset[str] = frozenset(
    {
        "ks-abstract-concept-typer",
        "ks-business-entity-typer",
        "ks-chunking",
        "ks-code-structure",
        "ks-conflict-analyzer",
        "ks-conflict-detector",
        "ks-consistency",
        "ks-context-extractor",
        "ks-duplicate-checker",
        "ks-entity-linker",
        "ks-entity-spotter",
        "ks-entity-typer",
        "ks-evidence-aggregator",
        "ks-evidence-strength-voter",
        "ks-format-detector",
        "ks-historical-pattern-voter",
        "ks-key-value-structure",
        "ks-list-structure",
        "ks-logic-consistency-voter",
        "ks-merge-executor",
        "ks-metadata",
        "ks-narrative-structure",
        "ks-person-place-typer",
        "ks-provenance",
        "ks-quality",
        "ks-recency-voter",
        "ks-relationship-extractor",
        "ks-relationship-spotter",
        "ks-relationship-typer",
        "ks-relationship-validator",
        "ks-score-aggregator",
        "ks-segmentation",
        "ks-signal-extractor",
        "ks-source-authority-voter",
        "ks-table-structure",
        "ks-technical-entity-typer",
        "ks-vote-aggregator",
    }
)


@lru_cache(maxsize=1)
def contract_labels() -> tuple[str, ...]:
    """Every value the ``contract`` label may take, including the sentinel.

    Locally-registered system workers (schema known, drift checkable) plus
    externally-owned ones (name known, schema not) plus
    :data:`CONTRACT_OTHER`. Bounded and closed — this tuple IS the cardinality
    guarantee, and it is what the metric pre-materialises against.
    """
    return (
        *sorted(set(registered_output_schemas()) | EXTERNAL_SYSTEM_WORKERS),
        CONTRACT_OTHER,
    )


def classify_contract(worker_name: str) -> str:
    """Collapse a raw ``worker_name`` to a bounded label value.

    Returns the name itself when it is a known system worker contract, else
    :data:`CONTRACT_OTHER`. ⛔ Never return the raw value for an unknown name —
    the sentinel is the whole cardinality bound (see :data:`CONTRACT_OTHER`).
    """
    if worker_name in EXTERNAL_SYSTEM_WORKERS or worker_name in registered_output_schemas():
        return worker_name
    return CONTRACT_OTHER


def live_output_schema(worker_data: Mapping[str, Any]) -> Any:
    """The ``output_schema`` a row actually asserts, from either place it can live.

    Dedicated top-level key first, then ``config`` — the same order and the
    same two places ``crud_endpoints`` (``:610-612``) and the agent builder
    read, so this cannot disagree with what the model is actually constrained
    by. Returns ``None`` when the row asserts nothing.

    ⚠️ ``config->'output_schema'`` is JSON ``null`` on 615 of 779 prod rows —
    the key is PRESENT and its value is ``None``. A ``"output_schema" in
    config`` test would read every one of those as healthy; a value test is the
    only correct one.
    """
    schema = worker_data.get("output_schema")
    if schema is not None:
        return schema
    config = worker_data.get("config")
    if isinstance(config, Mapping):
        return config.get("output_schema")
    return None


def worker_contract_verdict(worker_data: Mapping[str, Any], worker_name: str) -> str:
    """Classify ONE resolved worker row against its registered contract.

    Returns a :data:`WORKER_CONTRACT_VERDICTS` member:

    ``"bare"``
        The row declares no output contract in either place the agent builder
        reads, so nothing constrains or validates what the model returns.
        ``_output_type_from_schema(None)`` is ``str`` and the completion comes
        back freeform. Applies to ANY worker — registered, externally owned, or
        unknown — which is the generalisation this module gained: the old
        warning could only fire for a worker this repo could also repair.

    ``"drifted"``
        The row declares a contract AND a locally registered one exists AND
        they differ. ⛔ Only decidable for locally-registered workers. For an
        externally-owned worker there is no local contract to compare against,
        so a populated schema is reported ``"healthy"`` rather than guessed at
        — this detector does not fabricate a verdict it cannot support.

    ``"healthy"``
        The row declares a contract and nothing contradicts it.

    ⚠️ An EMPTY schema (``{}``) counts as bare, which is strictly WIDER than
    the ``is not None`` predicate ``apply_registered_system_schema`` repairs
    on. That divergence is deliberate: ``{}`` constrains nothing, so it is bare
    as a matter of fact even where the repair correctly declines to touch it.
    Detection and repair answer different questions and their predicates are
    allowed to differ — in this direction only.

    Pure. Reads ``worker_data`` and never mutates it.
    """
    schema = live_output_schema(worker_data)
    if not schema:
        return CONTRACT_BARE

    registered = registered_output_schema(worker_name)
    if registered is not None and schema != registered:
        return CONTRACT_DRIFTED
    return CONTRACT_HEALTHY
