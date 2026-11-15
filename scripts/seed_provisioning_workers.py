#!/usr/bin/env python3
"""Seed provisioning workers.

Creates the required workers for project provisioning and workflow management:
- project_classifier: Classifies project type and work needed
- meta_planner: Plans META board bootstrap
- execution_planner: Plans EXECUTION board seeding
- card_preparer: Prepares execution cards
- execution_agent: Executes card work
- qa_validator: Validates deliverables

These workers are required for the provisioning flow to work properly.

Every worker pins ``model="local-llm"`` -- the one served-model-name of the pgx
vLLM model (gitops ``scripts/host-configs/pgx/systemd/vllm.service``). Never a
concrete model name, a ``tier-*`` alias or ``default``.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

WORKER_MANAGEMENT_URL = os.getenv("WORKER_MANAGEMENT_URL", "http://localhost:8887")

# Worker definitions
PROVISIONING_WORKERS = [
    {
        "worker_name": "project_classifier",
        "description": "Classifies project type and determines work types needed",
        "agent_role": "Project Classifier",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a project classification expert. Analyze project requests and classify them by type (CODE_DEBUG, CODE_FEATURE, CODE_REARCHITECT, ANALYSIS) and determine work types needed (analysis, bug_fixing, testing, documentation, etc.).",
        "instruction": "Classify the project request and return: project_type, work_types (list), complexity_score (1-10), estimated_cards (number)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "meta_planner",
        "description": "Plans META board bootstrap for procedures, requirements, and principles",
        "agent_role": "META Bootstrap Planner",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a META bootstrap planning expert. Plan which procedures, requirements, and principles are needed for a project type. Return structured data for creating META cards.",
        "instruction": "Plan META bootstrap and return: procedure_categories (list), requirement_types (list), principle_sources (list), expected_card_count (number)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "execution_planner",
        "description": "Plans EXECUTION board seeding with initial cards",
        "agent_role": "Execution Planner",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.4,
        "system_prompt": "You are an execution planning expert. Create initial EXECUTION cards based on project type and work types. Prioritize cards and identify dependencies.",
        "instruction": "Plan execution cards and return: initial_cards (list of {title, description, label, priority}), dependencies (dict)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "card_preparer",
        "description": "Prepares execution cards with procedures and requirements",
        "agent_role": "Card Preparer",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.2,
        "system_prompt": "You are a card preparation expert. Analyze card requirements and prepare worker context with appropriate procedures and requirements.",
        "instruction": "Prepare worker for execution and return: worker_id, procedure_ids (list), requirement_ids (list), ready (boolean)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "execution_agent",
        "description": "Executes actual card work and produces deliverables",
        "agent_role": "Execution Agent",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.6,
        "system_prompt": "You are an execution agent. Complete the assigned work and produce high-quality deliverables. Follow procedures and requirements carefully.",
        "instruction": "Execute the work and return: output (string), artifacts (list), deliverable_type (string)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "qa_validator",
        "description": "Validates deliverable quality against requirements",
        "agent_role": "QA Validator",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.1,
        "system_prompt": (
            "You are a rigorous QA validation expert. Score a deliverable's quality on a "
            "strict 0.0-1.0 scale using this rubric:\n"
            "- 0.0-0.2: missing or off-topic; fails the core requirements / acceptance criteria.\n"
            "- 0.2-0.4: major gaps or errors; the core deliverable is incomplete.\n"
            "- 0.4-0.6: partial; meets some requirements but has significant issues.\n"
            "- 0.6-0.8: good; meets most requirements with minor gaps.\n"
            "- 0.8-1.0: excellent; comprehensive, accurate, fully aligned with the acceptance criteria.\n"
            "Judge on completeness, accuracy, clarity, and alignment to the acceptance criteria. "
            "A thin, vague, or placeholder deliverable scores 0.5 or below. Reserve 0.9 and above "
            "for deliverables with no material gaps. Do not inflate scores."
        ),
        "instruction": (
            "Validate the deliverable against the stated requirements and acceptance criteria, "
            "applying the rubric strictly. Return: validation_passed (boolean, true only when "
            "quality_score >= 0.6), quality_score (number between 0.0 and 1.0), issues (list of "
            "specific gaps/errors; empty only for a near-perfect deliverable), recommendations "
            '(list), decision (one of: "pass" when >= 0.6, "needs_review" when 0.4-0.6, '
            '"fail" when < 0.4)'
        ),
        # Structured output is REQUIRED here, not cosmetic: WorkerNode only lifts a
        # ``quality_score`` into the inference bag (→ persisted onto
        # graph_execution_sessions.quality_score) when the worker output is a *dict*
        # (see src/worker/graphs/nodes/worker_node.py _maybe_record_quality). With no
        # output_schema the pydantic_ai agent runs as output_type=str → output is a
        # plain string → the score is never lifted. This object schema makes the agent
        # emit a StructuredDict, so the producer chain actually fires.
        "output_schema": {
            "type": "object",
            "properties": {
                "validation_passed": {"type": "boolean"},
                "quality_score": {"type": "number"},  # 0.0–1.0; matches the instruction
                "issues": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "decision": {"type": "string"},
            },
            "required": ["validation_passed", "quality_score", "decision"],
        },
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "procedure_selector",
        "description": "Selects appropriate procedures for a project",
        "agent_role": "Procedure Selector",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a procedure selection expert. Select the most appropriate procedures from the procedure library for the given project requirements.",
        "instruction": "Select procedures and return: procedure_ids (list), rationale (string)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "requirement_selector",
        "description": "Selects relevant requirements for a project",
        "agent_role": "Requirement Selector",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a requirements selection expert. Select the most relevant quality requirements for the given project type and work.",
        "instruction": "Select requirements and return: requirement_ids (list), rationale (string)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "principle_extractor",
        "description": "Extracts principles from repository analysis",
        "agent_role": "Principle Extractor",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.4,
        "system_prompt": "You are a principle extraction expert. Analyze repository code and extract project-specific principles and best practices.",
        "instruction": "Extract principles and return: principles (list of {title, description, category}), confidence (number)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "template_preparer",
        "description": "Prepares prompt templates for project type",
        "agent_role": "Template Preparer",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a template preparation expert. Prepare optimized prompt templates for the given project type and work types.",
        "instruction": "Prepare templates and return: templates (list of {name, content, usage}), ready (boolean)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "meta_validator",
        "description": "Validates META asset completeness",
        "agent_role": "META Validator",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.2,
        "system_prompt": "You are a META completeness validator. Validate that all required META assets (procedures, requirements, principles, templates) are loaded and ready.",
        "instruction": "Validate completeness and return: complete (boolean), missing_assets (list), recommendations (list)",
        "tool_names": [],
        "deps": [],
    },
    {
        "worker_name": "meta_agent",
        "description": "Generic META board agent for various tasks",
        "agent_role": "META Agent",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.5,
        "system_prompt": "You are a META board agent. Execute META-related tasks including asset loading, validation, and preparation.",
        "instruction": "Execute META work and return: summary (string), assets_loaded (list), status (string)",
        "tool_names": [],
        "deps": [],
    },
]


async def create_worker(client: httpx.AsyncClient, worker_def: dict[str, Any]) -> bool:
    """Create a single worker.

    Args:
        client: HTTP client
        worker_def: Worker definition dictionary

    Returns:
        True if successful, False otherwise
    """
    worker_name = worker_def["worker_name"]
    print(f"Creating worker: {worker_name}...")

    try:
        response = await client.post(
            f"{WORKER_MANAGEMENT_URL}/workers",
            json=worker_def,
            timeout=30.0,
        )

        if response.status_code == 201:
            print(f"  ✓ Created: {worker_name}")
            return True
        elif response.status_code == 409:
            print(f"  ⊙ Already exists: {worker_name}")
            return True
        else:
            print(f"  ✗ Failed: {worker_name} (status {response.status_code})")
            print(f"    Response: {response.text}")
            return False

    except Exception as e:
        print(f"  ✗ Error creating {worker_name}: {e}")
        return False


async def get_existing_workers(client: httpx.AsyncClient) -> list[str]:
    """Get list of existing worker names.

    Args:
        client: HTTP client

    Returns:
        List of existing worker names
    """
    try:
        response = await client.get(
            f"{WORKER_MANAGEMENT_URL}/workers",
            timeout=10.0,
        )

        if response.status_code == 200:
            workers = response.json()
            return [w.get("worker_name", w.get("name", "")) for w in workers]
        else:
            print(f"Warning: Could not fetch existing workers (status {response.status_code})")
            return []

    except Exception as e:
        print(f"Warning: Could not fetch existing workers: {e}")
        return []


async def seed_workers():
    """Seed all provisioning workers."""
    print(f"Seeding provisioning workers to {WORKER_MANAGEMENT_URL}...")
    print(f"Total workers to create: {len(PROVISIONING_WORKERS)}")
    print()

    async with httpx.AsyncClient() as client:
        # Check existing workers
        print("Checking existing workers...")
        existing = await get_existing_workers(client)
        print(f"Found {len(existing)} existing workers: {existing}")
        print()

        # Create workers
        results = []
        for worker_def in PROVISIONING_WORKERS:
            success = await create_worker(client, worker_def)
            results.append((worker_def["worker_name"], success))

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        successful = [name for name, success in results if success]
        failed = [name for name, success in results if not success]

        print(f"Successful: {len(successful)}/{len(results)}")
        if successful:
            for name in successful:
                print(f"  ✓ {name}")

        if failed:
            print()
            print(f"Failed: {len(failed)}/{len(results)}")
            for name in failed:
                print(f"  ✗ {name}")

        print()
        print("=" * 60)

        if failed:
            print("\nSome workers failed to create. Check the errors above.")
            sys.exit(1)
        else:
            print("\n✓ All provisioning workers created successfully!")
            sys.exit(0)


def main():
    """Main entry point."""
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("These workers require OpenAI API access.")
        print()
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    asyncio.run(seed_workers())


if __name__ == "__main__":
    main()
