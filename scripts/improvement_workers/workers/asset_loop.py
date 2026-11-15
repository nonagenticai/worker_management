"""Worker configs for the Asset Improvement Loop (principle/requirement/decision).

Three workers, one per asset ``card_type`` in the asset improvement stream
(mirrors the single-asset ``procedure_loop`` module, one worker per improvable
asset kind). Like every other loop module, each worker is registered with a
placeholder ``system_prompt`` and a default ``output_schema``; per memory
``no-prompt-registry-wm-is-the-store`` WM is the prompt store and a PUT is
reflected on the next call.

These are the IMPROVEMENT analog of the ``asset_authoring`` DOING workers: the
authoring workers draft a NEW asset from a request utterance, whereas these
REVISE an existing base asset to fix a named quality deficit (``desired_outcome``)
while preserving the asset's identity/subject.

The schemas live alongside this file in
``schemas/asset_loop/<kind>_improvement.json`` (JSON Schema draft-07, mirroring
prompt_loop / procedure_loop). ``register.py`` discovers this module's
``AGENT_CONFIGS`` and posts it as ``output_schema`` on the worker so callers do
NOT have to pass ``output_schema`` per-call (per memory ``wm-as-sole-llm-gateway``,
callers MAY still override per-call).

The candidate is deliberately MINIMAL (``candidate_<kind>`` = ``{name, body}``)
because the downstream n8n DOING ``Parse Proposal`` reads
``candidate_<kind>.{name, body}`` and maps ``name`` -> title and ``body`` ->
statement/description/decision. Anything richer would just be dropped.

Validator-clean naming: the worker names use ``[A-Za-z0-9_-]`` only (no periods),
matching the workflow_loop / prompt_loop / procedure_loop convention enforced by
``src/api/security_validation.py:validate_identifier``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-relative path to the schemas dir (scripts/improvement_workers/schemas/asset_loop)
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "asset_loop"


@dataclass(frozen=True)
class WorkerConfig:
    """Subset of WorkerCreateRequest fields the registration script cares about.

    Mirrors ``procedure_loop.WorkerConfig`` (includes ``max_tokens``); satisfies
    the structural ``register.WorkerConfig`` protocol so the loop is added
    without touching register.py beyond the 2-line import+merge.
    """

    worker_name: str
    description: str
    system_prompt: str
    temperature: float
    model_provider: str
    model: str
    schema_filename: str
    max_tokens: int | None = None

    def output_schema(self) -> dict[str, Any]:
        """Load the JSON Schema for this worker's structured output."""
        path = SCHEMAS_DIR / self.schema_filename
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)


def _improve_prompt(kind: str) -> str:
    """Build the edit-to-fix-a-named-deficit prompt for one asset kind.

    It NAMES the exact output contract key (``candidate_<kind>``) and forbids the
    observed drift aliases -- the same discipline as prompt_loop's
    ``candidate_system_prompt`` fix (#74) and procedure_loop's ``candidate_procedure``:
    if the model returns the revision under ``revised_<kind>`` / ``candidate`` /
    ``title`` / ``statement``, the downstream asset DOING ``Parse Proposal`` skips
    it (key absent) and the loop never compounds.
    """
    key = f"candidate_{kind}"
    return (
        f"You revise an existing {kind} asset to fix a measured quality deficit. "
        f"You are given a base {kind} asset (its name and body) and a "
        "`desired_outcome` describing the deficit to fix. Revise the asset to fix "
        "that deficit while PRESERVING its identity and subject -- do NOT change the "
        f"subject and do NOT introduce a different {kind}; improve the SAME asset. "
        f"Return ONLY a JSON object with EXACTLY these keys and no others: `{key}` "
        "(an object with `name` (string, non-empty) and `body` (string, non-empty)) -- "
        f"the FULL revised {kind}; `rationale` (string, non-empty); `expected_impact` "
        "(object with numeric delta fields); `scope_of_change` (one of 'minor_wording', "
        f"'structural', 'instruction_set'). Put the revision under `{key}` with fields "
        f"`name` and `body` -- do NOT use `revised_{kind}`, `candidate`, `title`, or "
        f"`statement`; the ONLY content key is `{key}`. Reply with a single JSON object "
        "matching the registered output_schema; do not emit prose, markdown, or code fences."
    )


# fleet-wide max_tokens cap (matches prompt_loop / procedure_loop): vLLM-on-PGX
# enforces prompt+completion <= max_model_len (8192), so a larger completion
# budget 400s and falls through to a paid model. 4000 leaves headroom for the
# base asset + desired_outcome context.
_MAX_TOKENS = 4000

AGENT_CONFIGS: dict[str, WorkerConfig] = {
    "improvement_principle_improvement": WorkerConfig(
        worker_name="improvement_principle_improvement",
        description=(
            "Asset improvement loop -- principle_improvement card. Revises an "
            "existing principle to fix a named quality deficit; emits the revised "
            "principle (name + body) under candidate_principle."
        ),
        system_prompt=_improve_prompt("principle"),
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="principle_improvement.json",
        max_tokens=_MAX_TOKENS,
    ),
    "improvement_requirement_improvement": WorkerConfig(
        worker_name="improvement_requirement_improvement",
        description=(
            "Asset improvement loop -- requirement_improvement card. Revises an "
            "existing requirement to fix a named quality deficit; emits the revised "
            "requirement (name + body) under candidate_requirement."
        ),
        system_prompt=_improve_prompt("requirement"),
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="requirement_improvement.json",
        max_tokens=_MAX_TOKENS,
    ),
    "improvement_decision_improvement": WorkerConfig(
        worker_name="improvement_decision_improvement",
        description=(
            "Asset improvement loop -- decision_improvement card. Revises an "
            "existing decision to fix a named quality deficit; emits the revised "
            "decision (name + body) under candidate_decision."
        ),
        system_prompt=_improve_prompt("decision"),
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="decision_improvement.json",
        max_tokens=_MAX_TOKENS,
    ),
}
