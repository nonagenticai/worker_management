"""Regression pin for the qa_validator scoring rubric.

The qa_validator worker feeds graph_execution_sessions.quality_score, which the
metering exec_quality ETL aggregates into governed_avg_quality and the (currently
dark) exec_quality_deficit governor term. With a generic "validate deliverables"
prompt the model returns ~1.0 for everything, so the downstream signal is a flat
constant and the governor can never act. The seed therefore carries a strict,
banded rubric. These tests fail loudly if a future edit reverts qa_validator to a
non-discriminative prompt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SEED_PATH = Path(__file__).resolve().parents[2] / "scripts" / "seed_provisioning_workers.py"


def _load_provisioning_workers() -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("seed_provisioning_workers", _SEED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROVISIONING_WORKERS


def _qa_validator() -> dict[str, Any]:
    workers = _load_provisioning_workers()
    qa = next((w for w in workers if w["worker_name"] == "qa_validator"), None)
    assert qa is not None, "qa_validator must remain a seeded provisioning worker"
    return qa


def test_qa_validator_system_prompt_has_banded_rubric() -> None:
    prompt = _qa_validator()["system_prompt"]
    lower = prompt.lower()
    assert "rubric" in lower
    # Explicit score bands anchor the model so it discriminates instead of defaulting high.
    for band in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
        assert band in prompt, f"missing rubric band {band}"
    # Thin/placeholder deliverables must be pushed low, and inflation called out.
    assert "thin" in lower
    assert "0.5 or below" in lower
    assert "do not inflate" in lower


def test_qa_validator_instruction_ties_pass_to_threshold() -> None:
    instruction = _qa_validator()["instruction"]
    assert "0.6" in instruction  # pass / validation_passed threshold
    assert "needs_review" in instruction
    assert "fail" in instruction


def test_qa_validator_keeps_structured_output_schema() -> None:
    # The score is only lifted onto graph_execution_sessions when output is a dict,
    # which requires the object output_schema (see worker_node._maybe_record_quality).
    qa = _qa_validator()
    schema = qa.get("output_schema")
    assert isinstance(schema, dict) and schema.get("type") == "object"
    assert "quality_score" in schema["properties"]
    assert "quality_score" in schema["required"]
    assert qa["temperature"] == 0.1  # reproducible scoring
