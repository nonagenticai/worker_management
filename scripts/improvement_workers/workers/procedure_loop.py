"""Worker configs for the Procedure Improvement Loop (wf-proc improvement program).

One worker for the ``procedure_improvement`` card_type in the procedure improvement
stream owned by the ``__system_procedure__`` internal tenant. Like every other loop
module, the worker is registered with a placeholder ``system_prompt`` and a default
``output_schema``; per memory ``no-prompt-registry-wm-is-the-store`` WM is the prompt
store and a PUT is reflected on the next call.

The schema lives alongside this file in
``schemas/procedure_loop/procedure_improvement.json`` (JSON Schema draft-07, mirroring
prompt_loop). ``register.py`` discovers this module's ``AGENT_CONFIGS`` and posts it as
``output_schema`` on the worker so callers do NOT have to pass ``output_schema`` per-call
(per memory ``wm-as-sole-llm-gateway``, callers MAY still override per-call).

Validator-clean naming: the worker name uses ``[A-Za-z0-9_-]`` only (no periods),
matching the workflow_loop / prompt_loop convention enforced by
``src/api/security_validation.py:validate_identifier``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-relative path to the schemas dir (scripts/improvement_workers/schemas/procedure_loop)
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "procedure_loop"


@dataclass(frozen=True)
class WorkerConfig:
    """Subset of WorkerCreateRequest fields the registration script cares about.

    Mirrors ``prompt_loop.WorkerConfig`` (includes ``max_tokens``); satisfies the
    structural ``register.WorkerConfig`` protocol so the loop is added without
    touching register.py beyond the 2-line import+merge.
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


# Placeholder system prompt. It NAMES the output contract key explicitly and forbids the
# observed drift aliases -- the same discipline as prompt_loop's candidate_system_prompt fix
# (#74): if the model returns the revision under `revised_procedure`/`procedure`, the
# downstream procedure-canary `Parse Proposal` skips it and the loop never compounds.
_IMPROVE_PROMPT = (
    "You rewrite an agentic procedure to fix a measured failure mode or to improve its "
    "clarity. Given the current procedure (name, domain, preconditions, postconditions, and "
    "ordered steps), recent failures with redacted payloads, the eval set, and 30-day "
    "telemetry, propose a revised procedure. Be conservative: keep the procedure well-formed "
    "-- every step needs a stepNumber, name, actionType, and PDDL-checkable "
    "preconditions/postconditions, and each step's preconditions must be satisfiable by the "
    "procedure-level preconditions plus the postconditions of earlier steps. Return your "
    "revision under the `candidate_procedure` output key (do NOT use `revised_procedure` or a "
    "bare `procedure` key); also provide `rationale`, `expected_impact`, and `scope_of_change`. "
    "Reply with a single JSON object matching the registered output_schema."
)


AGENT_CONFIGS: dict[str, WorkerConfig] = {
    "improvement_procedure_improvement": WorkerConfig(
        worker_name="improvement_procedure_improvement",
        description=(
            "Procedure improvement loop -- procedure_improvement card. Rewrites an "
            "agentic procedure to fix a measured failure mode; emits a revised procedure "
            "(steps + preconditions + postconditions) under candidate_procedure."
        ),
        system_prompt=_IMPROVE_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="procedure_improvement.json",
        # 4000 = fleet cap (matches prompt_loop): vLLM-on-PGX enforces prompt+completion
        # <= max_model_len (8192), so a 16384 completion budget 400s and falls through to a
        # paid model. 4000 leaves >4000 tokens of prompt headroom for procedure context.
        max_tokens=4000,
    ),
}
