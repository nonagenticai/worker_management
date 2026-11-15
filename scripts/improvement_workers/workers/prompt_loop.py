"""Worker configs for Plan 6 (Prompt Improvement Loop).

Three workers, one per ``card_type`` in the prompt improvement stream owned
by the ``__system_prompt__`` internal tenant (see fr-backend memory
``continuous-improvement-system-level``). Each worker is registered in
worker_management with a placeholder ``system_prompt`` and a default
``output_schema``; Plan 6 itself rewrites the prompts in-place via
``PUT /workers/{name}`` once GATE-1 approvals land -- per memory
``no-prompt-registry-wm-is-the-store``, WM is the prompt store and PUT is
reflected on the next call.

The schemas live alongside this file in
``schemas/prompt_loop/<card_type>.json`` (JSON Schema Draft 2020-12).
``register.py`` discovers this module's ``AGENT_CONFIGS`` alongside
``workflow_loop.AGENT_CONFIGS`` and posts both as ``output_schema`` on each
worker so callers do NOT have to pass ``output_schema`` per-call (per memory
``wm-as-sole-llm-gateway``, callers MAY still override per-call).

Validator-clean naming: worker names use ``[A-Za-z0-9_-]`` only (no periods),
matching the workflow_loop convention enforced by
``src/api/security_validation.py:validate_identifier``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-relative path to the schemas dir (scripts/improvement_workers/schemas/prompt_loop)
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "prompt_loop"


@dataclass(frozen=True)
class WorkerConfig:
    """Subset of WorkerCreateRequest fields the registration script cares about.

    Mirrors ``workflow_loop.WorkerConfig`` but adds ``max_tokens`` because
    Plan 6 (prompt rewriter) needs a 16384-token budget while the analyzers
    sit at 8192. ``max_tokens=None`` means "do not send max_tokens in the
    payload" -- WM applies its own default in that case.
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


# Placeholder system prompts -- copied verbatim from
# fr/docs/superpowers/plans/2026-04-27-improvement-plan-6-prompt-loop.md (Stage 2).
# Plan 6 itself will rewrite these via PUT /workers/{name} on GATE-1 approval.

_STATS_PROMPT = (
    "You analyze the 30-day execution telemetry of a single LLM worker. Given "
    "exec_quality rollups, recent failures, and the worker's current system "
    "prompt, produce a structured report explaining what the data says about "
    "prompt quality. Your `candidate_system_prompt` is the existing prompt "
    "unchanged unless you are extremely confident a one-line clarification is "
    "warranted; default to `scope_of_change='minor_wording'`."
)

_IMPROVE_PROMPT = (
    "You rewrite an LLM worker's system prompt to fix a measured failure mode. "
    "Given the current prompt, recent failures (with redacted payloads + "
    "outputs), the eval set, and 30-day telemetry, propose a revised system "
    "prompt. Be conservative -- `scope_of_change='instruction_set'` only if the "
    "failure mode is structural; prefer `minor_wording` or `structural` "
    "(single-paragraph rewrite) otherwise."
)

_CONSISTENCY_PROMPT = (
    "You audit a logical app's full set of LLM workers for cross-worker "
    "inconsistency (tone, output-shape conventions, error-handling "
    "instructions, role framing). Given the catalogue (worker name + system "
    "prompt for each peer worker) plus the target worker's prompt and metrics, "
    "propose changes to the target prompt that reduce drift from the cohort. "
    "Never propose changes to multiple workers in one card."
)

# Shared output contract appended to all three prompt-loop workers. They share
# one schema (PromptImprovementProposal: candidate_system_prompt, rationale,
# expected_impact{deltas}, scope_of_change; additionalProperties:false), but the
# prompts above never named the OUTPUT keys -- so the model returned the revised
# prompt under `revised_prompt`, the downstream prompt-canary `Parse Proposal`
# correctly skipped it (key absent), and the improvement loop never compounded
# (computational verdict stuck at 1). Naming the exact keys here is the durable
# fix; the n8n flow keeps a `revised_prompt` fallback as a belt-and-suspenders
# hedge against LLM key drift.
_OUTPUT_CONTRACT = (
    " Return ONLY a JSON object with EXACTLY these keys and no others: "
    "`candidate_system_prompt` (string, non-empty) -- the FULL revised system "
    "prompt; if you are not changing the prompt, return the existing prompt "
    "verbatim here. `rationale` (string, non-empty). `expected_impact` (object "
    "with numeric `delta_p95_ms`, `delta_failure_rate`, "
    "`delta_cost_usd_per_call`). `scope_of_change` (one of 'minor_wording', "
    "'structural', 'instruction_set'). Put the revised prompt under "
    "`candidate_system_prompt` -- do NOT use `revised_prompt` or any other key."
)


AGENT_CONFIGS: dict[str, WorkerConfig] = {
    "improvement_prompt_performance_statistics_analysis": WorkerConfig(
        worker_name="improvement_prompt_performance_statistics_analysis",
        description=(
            "Prompt improvement loop -- prompt_performance_statistics_analysis "
            "card. Analyzes 30d telemetry for a single worker prompt and "
            "emits a conservative candidate revision."
        ),
        system_prompt=_STATS_PROMPT + _OUTPUT_CONTRACT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="prompt_performance_statistics_analysis.json",
        # vLLM-on-PGX (`local-llm`) enforces prompt+completion ≤ max_model_len
        # = 8192. max_tokens is the COMPLETION budget; with a non-trivial
        # prompt this previously rejected the request and silently fell
        # through to deepseek (paid). 4000 matches the established fleet
        # cap (karma_service/src/config/agents.py:11) and leaves >4000
        # tokens of prompt headroom for telemetry context.
        max_tokens=4000,
    ),
    "improvement_prompt_improvement": WorkerConfig(
        worker_name="improvement_prompt_improvement",
        description=(
            "Prompt improvement loop -- prompt_improvement card. Rewrites a "
            "worker's system_prompt to fix a measured failure mode; highest "
            "temperature + token budget of the loop because it is the rewriter."
        ),
        system_prompt=_IMPROVE_PROMPT + _OUTPUT_CONTRACT,
        temperature=0.3,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="prompt_improvement.json",
        # Was 16384 — exceeded vLLM 8192 ceiling outright, every request
        # 400'd on PGX and fell through to deepseek. 4000 = fleet cap.
        max_tokens=4000,
    ),
    "improvement_prompt_library_consistency_management": WorkerConfig(
        worker_name="improvement_prompt_library_consistency_management",
        description=(
            "Prompt improvement loop -- prompt_library_consistency_management "
            "card. Audits one app's worker prompts for cross-worker drift and "
            "proposes a target-prompt change that reduces it."
        ),
        system_prompt=_CONSISTENCY_PROMPT + _OUTPUT_CONTRACT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="prompt_library_consistency_management.json",
        # Was 16384 — same vLLM ceiling violation as above.
        max_tokens=4000,
    ),
}
