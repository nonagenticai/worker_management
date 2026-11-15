"""Plan 7 Stage 3 -- author the 3 asset-authoring worker configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.improvement_workers.workers.asset_authoring import AGENT_CONFIGS

ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = ROOT / "scripts" / "improvement_workers" / "schemas" / "asset_authoring"


def test_three_configs_present() -> None:
    expected = {
        "authoring_principle_development",
        "authoring_requirement_development",
        "authoring_decision_development",
    }
    assert set(AGENT_CONFIGS) == expected


@pytest.mark.parametrize("name", list(AGENT_CONFIGS))
def test_each_worker_config_loads_its_schema(name: str) -> None:
    cfg = AGENT_CONFIGS[name]
    schema = cfg.output_schema()
    assert schema.get("$schema", "").startswith("https://json-schema.org/")
    assert schema["type"] == "object"


def test_principle_schema_requires_rationale() -> None:
    schema = json.loads((SCHEMAS / "principle_development.json").read_text())
    assert "rationale" in schema["required"]
    assert "name" in schema["required"]
    assert "body" in schema["required"]


def test_requirement_schema_requires_min_one_acceptance_criterion() -> None:
    schema = json.loads((SCHEMAS / "requirement_development.json").read_text())
    crit = schema["properties"]["acceptance_criteria"]
    assert crit["type"] == "array"
    assert crit["minItems"] >= 1


def test_decision_schema_requires_min_two_options_plus_chosen() -> None:
    schema = json.loads((SCHEMAS / "decision_development.json").read_text())
    opts = schema["properties"]["considered_options"]
    assert opts["type"] == "array"
    assert opts["minItems"] >= 2
    for k in ("chosen_option", "consequences"):
        assert k in schema["required"]
