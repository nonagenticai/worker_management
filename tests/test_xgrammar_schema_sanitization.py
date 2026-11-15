"""Regression tests: a worker's output_schema must reach vLLM xgrammar-compilable.

vLLM's xgrammar backend cannot compile bounded-array keywords
(``minItems``/``maxItems`` et al). It does not fail loudly — it silently
disables guided decoding, so the model answers unconstrained, the reply is
often not JSON, validation fails, and inference-router falls through to the
next provider. Live symptom: repeated
``Provider response failed validation, trying next`` for ``kag_query_expand``
while its array-free sibling ``kag_eval_judge`` never failed.

Both seams that turn a stored ``output_schema`` into a decoding constraint
must strip these keywords:
  - ``_output_type_from_schema`` (agent_type=pydantic_ai — the path every
    affected worker in the live registry actually uses)
  - ``_build_response_format`` (the inference-provider graph node)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from pydantic_ai import NativeOutput

if TYPE_CHECKING:
    import pytest

from src.utils.xgrammar_schema import _warn_if_unsatisfiable, strip_xgrammar_unsupported
from src.worker.agents.pydantic_ai_agent import _output_type_from_schema
from src.worker.graphs.nodes.inference_provider_node import _build_response_format

# Mirrors kag_query_expand's registered schema — the worker whose live
# structured decoding was silently disabled.
_HOSTILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "QueryExpansion",
    "properties": {
        "rewrites": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1},
        },
        "hyde_passage": {"type": "string", "maxLength": 2000},
    },
    "required": ["rewrites", "hyde_passage"],
}

_BANNED = ("minItems", "maxItems", "minLength", "maxLength", "pattern")


def test_strip_removes_bounded_array_keywords_recursively() -> None:
    stripped = strip_xgrammar_unsupported(_HOSTILE_SCHEMA)
    blob = str(stripped)
    for keyword in _BANNED:
        assert keyword not in blob, f"{keyword} survived the strip"
    # Structure the grammar actually needs is preserved.
    assert stripped["properties"]["rewrites"]["type"] == "array"
    assert stripped["properties"]["rewrites"]["items"] == {"type": "string"}
    assert stripped["required"] == ["rewrites", "hyde_passage"]


def test_strip_leaves_the_input_untouched() -> None:
    """The stored schema keeps its validation keywords; only the copy is relaxed."""
    strip_xgrammar_unsupported(_HOSTILE_SCHEMA)
    assert _HOSTILE_SCHEMA["properties"]["rewrites"]["minItems"] == 2


def test_pydantic_ai_output_type_reaches_vllm_without_bounded_arrays() -> None:
    """The pydantic_ai seam: every live hostile worker is agent_type=pydantic_ai."""
    wrapped = _output_type_from_schema(_HOSTILE_SCHEMA, "inference_router")
    assert isinstance(wrapped, NativeOutput)

    # Round-trip the wrapped StructuredDict back to the JSON schema that
    # pydantic-ai will hand to vLLM as response_format.
    emitted = TypeAdapter(wrapped.outputs).json_schema()
    blob = str(emitted)
    for keyword in _BANNED:
        assert keyword not in blob, (
            f"{keyword} reached vLLM — xgrammar will silently disable "
            "structured decoding for this worker"
        )
    assert emitted["properties"]["rewrites"]["type"] == "array"


def test_response_format_strips_schema_but_keeps_title_as_name() -> None:
    """Ordering guard: the name is read off the ORIGINAL schema.

    ``title`` is itself stripped, so reading the name after the strip would
    silently degrade every worker's response_format name to "output".
    """
    response_format = _build_response_format(_HOSTILE_SCHEMA)
    assert response_format is not None
    assert response_format["json_schema"]["name"] == "QueryExpansion"

    blob = str(response_format["json_schema"]["schema"])
    for keyword in _BANNED:
        assert keyword not in blob, f"{keyword} reached vLLM via response_format"


def test_response_format_still_none_for_unusable_schema() -> None:
    assert _build_response_format(None) is None
    assert _build_response_format({}) is None


# ---------------------------------------------------------------------------
# A stripped keyword is only a keyword in KEYWORD position. The same word used
# as a *property name* is data, and eating it makes the grammar unable to
# satisfy the validator.
#
# Live incident 2026-07-14/15: worker-project-decompose. ``title`` is both a
# JSON-Schema annotation and by far the most common property name in the AA
# schemas. The strip deleted the ``title`` *property* from the grammar while
# leaving "title" in ``required`` — vLLM then could not emit it, the post-hoc
# validator demanded it, ModelRetry looped, and the run died with
# "Model token limit (16384) exceeded before any response was generated".
# Every project in AA's history decomposed into nothing because of this.
# ---------------------------------------------------------------------------

# Mirrors n8n_custom_nodes project-start-analysis-orchestrate "Build Decomp
# Prompt" (#777) — the schema that reproduced the live failure exactly.
_STR: dict[str, Any] = {"type": "string"}
_STRS: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
_DECOMP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "io_map": {
            "type": "object",
            "properties": {
                "goal": _STR,
                "necessary_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": _STR,
                            "statement": _STR,
                            "csf_index": {"type": "integer"},
                        },
                        "required": ["title", "statement", "csf_index"],
                    },
                },
            },
            "required": ["goal", "necessary_conditions"],
        },
        "child_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["architecture_spike", "development_feature_new"],
                    },
                    "title": _STR,
                    "acceptance_criteria": _STRS,
                },
                "required": ["card_type", "title", "acceptance_criteria"],
            },
        },
    },
    "required": ["io_map", "child_cards"],
}


def _iter_schema_objects(node: Any) -> Any:
    """Yield every dict in the tree that declares ``properties``."""
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            yield node
        for value in node.values():
            yield from _iter_schema_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_schema_objects(item)


def test_property_named_title_survives_the_strip() -> None:
    """THE regression: ``title`` as a property name is data, not an annotation."""
    stripped = strip_xgrammar_unsupported(_DECOMP_SCHEMA)

    card = stripped["properties"]["child_cards"]["items"]
    assert "title" in card["properties"], (
        "the 'title' PROPERTY was eaten by the annotation strip — vLLM cannot "
        "emit a field the grammar does not define, so the required-property "
        "validator loops until the token budget dies"
    )
    condition = stripped["properties"]["io_map"]["properties"]["necessary_conditions"]["items"]
    assert "title" in condition["properties"]


def test_every_stripped_keyword_survives_as_a_property_name() -> None:
    """Not just ``title`` — any banned word is legal as a property name."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": _STR,
            "format": _STR,
            "pattern": _STR,
            "maximum": {"type": "integer"},
            "minLength": {"type": "integer"},
        },
        "required": ["title", "format", "pattern", "maximum", "minLength"],
    }
    stripped = strip_xgrammar_unsupported(schema)
    assert sorted(stripped["properties"]) == sorted(schema["properties"])


def test_grammar_can_satisfy_the_validator_for_every_object() -> None:
    """The invariant behind the class: required ⊆ properties, at every level.

    The validator checks the ORIGINAL schema while vLLM decodes the STRIPPED
    one. If the strip removes a required property, the two disagree and no
    completion can ever pass — an unsatisfiable grammar.
    """
    stripped = strip_xgrammar_unsupported(_DECOMP_SCHEMA)
    for obj in _iter_schema_objects(stripped):
        missing = [r for r in obj.get("required", []) if r not in obj["properties"]]
        assert not missing, f"required but not in properties: {missing}"


def test_enum_values_are_not_treated_as_schema_keywords() -> None:
    """``enum`` holds data. Never recurse into it, never filter it."""
    stripped = strip_xgrammar_unsupported(_DECOMP_SCHEMA)
    card = stripped["properties"]["child_cards"]["items"]
    assert card["properties"]["card_type"]["enum"] == [
        "architecture_spike",
        "development_feature_new",
    ]


def test_keyword_position_annotations_are_still_stripped() -> None:
    """Pins the original #118 intent — this must not regress."""
    schema: dict[str, Any] = {
        "type": "object",
        "title": "AnnotationInKeywordPosition",
        "properties": {
            "rewrites": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "title": "Rewrites",
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["rewrites"],
    }
    stripped = strip_xgrammar_unsupported(schema)
    assert "title" not in stripped
    assert "title" not in stripped["properties"]["rewrites"]
    assert "minItems" not in stripped["properties"]["rewrites"]
    assert "maxItems" not in stripped["properties"]["rewrites"]
    assert stripped["properties"]["rewrites"]["items"] == {"type": "string"}


def test_dependent_required_keys_are_property_names_not_keywords() -> None:
    """Same class as the headline bug, one keyword over.

    ``dependentRequired`` maps a property NAME to the names it requires. Its
    keys are data; filtering them by keyword is the exact mistake this module
    exists to correct.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"title": _STR, "format": _STR, "body": _STR},
        "dependentRequired": {"title": ["body"], "format": ["body"]},
    }
    stripped = strip_xgrammar_unsupported(schema)
    assert stripped["dependentRequired"] == {"title": ["body"], "format": ["body"]}


def test_data_is_copied_not_aliased() -> None:
    """The stored schema must stay intact — the validator still uses it in full."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": _STR},
        "required": ["a"],
        "enum": [{"title": "keep me"}],
    }
    stripped = strip_xgrammar_unsupported(schema)
    assert stripped["required"] is not schema["required"]
    assert stripped["enum"] is not schema["enum"]
    # ...and the data itself survives untouched, keys named after keywords included.
    assert stripped["enum"] == [{"title": "keep me"}]


def test_tripwire_stays_quiet_on_composed_schemas(caplog: pytest.LogCaptureFixture) -> None:
    """A required name may legally be declared under allOf/$ref, not locally.

    A tripwire for a P0 class earns its keep only if it never cries wolf.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": _STR},
        "allOf": [{"properties": {"b": _STR}}],
    }
    with caplog.at_level(logging.WARNING, logger="src.utils.xgrammar_schema"):
        strip_xgrammar_unsupported(schema)
    assert not caplog.records, "tripwire fired on a satisfiable composed schema"


def test_tripwire_fires_when_the_grammar_cannot_satisfy_the_validator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The #118 signature itself must still be caught."""
    schema: dict[str, Any] = {
        "type": "object",
        "required": ["keep", "gone"],
        "properties": {"keep": _STR, "gone": {"type": "string"}},
    }
    stripped = strip_xgrammar_unsupported(schema)
    del stripped["properties"]["gone"]  # simulate a strip that ate a required name
    with caplog.at_level(logging.WARNING, logger="src.utils.xgrammar_schema"):
        _warn_if_unsatisfiable(schema, stripped)
    assert any("unsatisfiable" in r.message for r in caplog.records)


def test_property_named_title_reaches_vllm_through_the_pydantic_ai_seam() -> None:
    """End-to-end at the seam the live worker actually uses."""
    wrapped = _output_type_from_schema(_DECOMP_SCHEMA, "inference_router")
    emitted = TypeAdapter(wrapped.outputs).json_schema()

    defs = emitted.get("$defs", {})

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        return defs[ref.split("/")[-1]] if ref else node

    card = resolve(resolve(emitted["properties"]["child_cards"])["items"])
    assert "title" in card["properties"], "'title' property never reached vLLM"
