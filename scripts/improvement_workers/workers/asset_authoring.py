"""Worker configs for Plan 7 (Asset Authoring Loop).

Three workers, one per asset card_type in the asset-authoring stream. Each
worker is registered in worker_management with a placeholder ``system_prompt``
and a default ``output_schema``; per memory ``no-prompt-registry-wm-is-the-store``,
WM is the prompt store and PUT is reflected on the next call -- placeholders
are fine for the first cut and Plan 6 can rewrite them in-place later.

The schemas live alongside this file in
``schemas/asset_authoring/<card_type>.json`` (JSON Schema Draft 2020-12).
``register.py`` discovers this module's ``AGENT_CONFIGS`` alongside the
other loops and posts both as ``output_schema`` on each worker so callers do
NOT have to pass ``output_schema`` per-call (per memory
``wm-as-sole-llm-gateway``, callers MAY still override per-call).

Validator-clean naming: worker names use ``[A-Za-z0-9_-]`` only (no periods),
matching the prompt_loop / workflow_loop convention enforced by
``src/api/security_validation.py:validate_identifier``.

JSON Schema cannot express the "chosen_option in considered_options"
constraint -- that invariant is enforced in fr-backend's zod refine on the
authoring publish route. This is intentional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-relative path to the schemas dir
# (scripts/improvement_workers/schemas/asset_authoring)
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "asset_authoring"


@dataclass(frozen=True)
class WorkerConfig:
    """Subset of WorkerCreateRequest fields the registration script cares about.

    Mirrors ``prompt_loop.WorkerConfig`` / ``workflow_loop.WorkerConfig``;
    ``register.py`` consumes any object satisfying its structural
    ``WorkerConfig`` Protocol so each loop module owns its own dataclass.
    """

    worker_name: str
    description: str
    system_prompt: str
    temperature: float
    model_provider: str
    model: str
    schema_filename: str

    def output_schema(self) -> dict[str, Any]:
        """Load the JSON Schema for this worker's structured output."""
        path = SCHEMAS_DIR / self.schema_filename
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)


# Placeholder system prompts -- copied verbatim from the Plan 7 spec.
# Per memory no-prompt-registry-wm-is-the-store, WM is the prompt store and
# these can be PUT-rewritten later via Plan 6.

_PRINCIPLE_PROMPT = (
    "You are the principle-authoring worker for the Nonagentic AI asset "
    "authoring loop. Given a tenant's existing principles, requirements, and "
    "the human-authored request utterance, propose ONE new principle (name, "
    "body, rationale) that captures the rule the request implies without "
    "duplicating an existing principle. Reply with a single JSON object that "
    "exactly matches the registered output_schema; do not emit prose, "
    "markdown, or code fences."
)

_REQUIREMENT_PROMPT = (
    "You are the requirement-authoring worker for the Nonagentic AI asset "
    "authoring loop. Given the tenant's principles, existing requirements, "
    "and the request utterance, propose ONE new requirement (name, body, "
    "and >=1 specific testable acceptance_criteria). Cite the parent "
    "principle ids the requirement derives from when applicable. Reply "
    "with a single JSON object that matches the registered output_schema; "
    "do not emit prose, markdown, or code fences."
)

_DECISION_PROMPT = (
    "You are the decision-authoring worker for the Nonagentic AI asset "
    "authoring loop. Given the tenant's existing requirements + the request "
    "utterance, propose ONE decision: enumerate >=2 considered options, "
    "pick exactly one chosen_option, and state its consequences. The "
    "chosen_option MUST be one of considered_options. Reply with a single "
    "JSON object that matches the registered output_schema; do not emit "
    "prose, markdown, or code fences."
)


AGENT_CONFIGS: dict[str, WorkerConfig] = {
    "authoring_principle_development": WorkerConfig(
        worker_name="authoring_principle_development",
        description=(
            "Asset authoring loop -- principle_development DOING phase. "
            "Drafts a new principle from request + existing tenant principles."
        ),
        system_prompt=_PRINCIPLE_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="principle_development.json",
    ),
    "authoring_requirement_development": WorkerConfig(
        worker_name="authoring_requirement_development",
        description=(
            "Asset authoring loop -- requirement_development DOING phase. "
            "Drafts a new requirement from request + tenant principles."
        ),
        system_prompt=_REQUIREMENT_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="requirement_development.json",
    ),
    "authoring_decision_development": WorkerConfig(
        worker_name="authoring_decision_development",
        description=(
            "Asset authoring loop -- decision_development DOING phase. "
            "Drafts a new decision from request + tenant requirements."
        ),
        system_prompt=_DECISION_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="decision_development.json",
    ),
}
