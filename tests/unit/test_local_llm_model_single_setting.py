"""The platform model is named ONCE: ``Settings.local_llm_model`` (default ``local-llm``).

George 2026-09-23: pgx vLLM serves one model under the served-model-name
``local-llm``; every consumer uses that name so the model can be swapped by
editing one name. These tests pin that every "use the platform model" default
in worker_management resolves through the single setting, and that no source
file re-introduces a hard-coded model id.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from src.settings import LOCAL_LLM_MODEL, Settings, reset_settings_cache

SWAPPED = "swapped-served-name"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def swapped_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("LOCAL_LLM_MODEL", SWAPPED)
    reset_settings_cache()
    yield SWAPPED
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    reset_settings_cache()


def test_default_is_the_vllm_served_model_name() -> None:
    assert LOCAL_LLM_MODEL == "local-llm"
    assert Settings().local_llm_model == LOCAL_LLM_MODEL


def test_auto_create_model_comes_from_the_setting(swapped_model: str) -> None:
    from src.api.workers.message_endpoints import _auto_create_model

    assert _auto_create_model() == swapped_model


def test_local_preset_tier_alias_coercion_falls_back_to_the_setting(
    swapped_model: str,
) -> None:
    from src.worker.factory.llm_config_resolver import _local_safe_model

    # A local preset with no flash class: the coercion target is the setting.
    preset: dict[str, Any] = {"key": "local", "is_external": False, "model_classes": {}}
    assert _local_safe_model(preset, "tier-medium", "t-1") == swapped_model


def test_catalogs_name_the_single_constant() -> None:
    from src.worker.agents.modality_catalog import modality_catalog
    from src.worker.agents.provider_catalog import provider_catalog

    local = next(p for p in provider_catalog() if p["key"] == "local")
    assert local["models"][0] == LOCAL_LLM_MODEL
    assert local["model_classes"]["flash"] == LOCAL_LLM_MODEL
    vlm_local = next(p for p in modality_catalog("vlm") if p["key"] == "local")
    assert vlm_local["models"] == [LOCAL_LLM_MODEL]


# ── source guard ────────────────────────────────────────────────────────────
# Concrete model ids that must never be a code/config literal. Field
# ``description=`` text and docstrings may cite a dated measurement ("n=12,933
# requests on <model>, 2026-08-24") — that is history, not a default.
_CONCRETE_MODEL = re.compile(r"(?i)qwen|nvfp4|awq")


def _code_string_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            skip.add(id(body[0].value))
        if isinstance(node, ast.keyword) and node.arg == "description":
            for sub in ast.walk(node.value):
                skip.add(id(sub))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append((node.lineno, node.value))
    return out


def test_no_source_literal_names_a_concrete_model_or_restates_the_alias() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC.parent)
        for lineno, value in _code_string_literals(path):
            if _CONCRETE_MODEL.search(value):
                offenders.append(f"{rel}:{lineno}: {value[:60]!r}")
            if value == LOCAL_LLM_MODEL and path.name != "settings.py":
                offenders.append(f"{rel}:{lineno}: restates {LOCAL_LLM_MODEL!r}")
    # models.py's OpenAPI example payload shows a caller what to send.
    offenders = [o for o in offenders if not o.startswith("src/api/workers/models.py")]
    assert offenders == [], "\n".join(offenders)
