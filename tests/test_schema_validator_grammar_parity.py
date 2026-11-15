"""The decoder and the validator must be constrained by the SAME schema.

This is the regression suite for the 2026-07-20 production outage. `xgrammar`
cannot compile 14 JSON-Schema keywords into a grammar, so `strip_xgrammar_unsupported`
removes them before the schema becomes the decoder's constraint. The validator,
however, was built from the *unstripped* original — so the model was graded
against a contract it had never been shown. On `inference_router`, where
`retries=0`, the first disagreement became an instant HTTP 500 and the w2v
`demo_video` pipeline failed 100% of its runs; its `retention_plan` stage never
completed once in its entire lifetime.

The invariant these tests defend: **a `ModelRetry` may only ever fire on
something the decoder was actually told about.** Anything the stripper removed
is advisory — reported via `schema_soft_violation`, never fatal.
"""

from __future__ import annotations

import logging

import pytest
from jsonschema import Draft202012Validator
from pydantic_ai import ModelRetry

from src.utils.xgrammar_schema import _XGRAMMAR_UNSUPPORTED_KEYS, stripped_keywords
from src.worker.agents.pydantic_ai_agent import _decoder_visible_schema
from tests.test_pydantic_ai_agent import _make_agent

# ---------------------------------------------------------------------------
# The 12 keywords that are BOTH stripped by xgrammar AND enforced by
# Draft202012Validator in production configuration.
#
# `format` and `title` are stripped too, but are deliberately NOT parametrised:
#   - `format` is annotation-only unless a `format_checker` is passed, and
#     production passes none. Verified on this repo's jsonschema:
#     Draft202012Validator({"type":"string","format":"email"}).iter_errors("nope") == []
#     Testing it would assert behaviour production does not have.
#   - `title` is a pure annotation; no payload can violate it.
# Adding either back would make its case vacuous — the pre-assertion below
# would fail, which is exactly the guard that catches such a mistake.
# ---------------------------------------------------------------------------
UNENFORCEABLE_CASES: list[tuple[str, dict, object]] = [
    ("minItems", {"type": "array", "minItems": 3}, []),
    ("maxItems", {"type": "array", "maxItems": 1}, ["a", "b"]),
    ("minLength", {"type": "string", "minLength": 5}, "ab"),
    ("maxLength", {"type": "string", "maxLength": 2}, "abcdef"),
    ("minimum", {"type": "integer", "minimum": 10}, 1),
    ("maximum", {"type": "integer", "maximum": 5}, 12),
    ("exclusiveMinimum", {"type": "integer", "exclusiveMinimum": 10}, 10),
    ("exclusiveMaximum", {"type": "integer", "exclusiveMaximum": 5}, 5),
    ("multipleOf", {"type": "integer", "multipleOf": 5}, 7),
    ("minProperties", {"type": "object", "minProperties": 2}, {"a": 1}),
    ("maxProperties", {"type": "object", "maxProperties": 1}, {"a": 1, "b": 2}),
    ("pattern", {"type": "string", "pattern": "^[A-Z]+$"}, "abc"),
]


def _validator_for(schema: dict):
    """Capture the validator the production `__init__` path actually registers.

    Deliberately goes through `_make_agent` rather than calling
    `_register_default_schema_validator` directly: threading the grammar schema
    from `__init__` IS the fix, so a test that bypassed `__init__` could pass
    while the wiring was broken.
    """
    _agent, _, mock_agent_inst = _make_agent(model_provider="openai", output_schema=schema)
    return mock_agent_inst.output_validator.call_args[0][0]


class TestUnenforceableKeywordsAreNotFatal:
    @pytest.mark.parametrize(
        ("keyword", "prop_schema", "bad_value"),
        UNENFORCEABLE_CASES,
        ids=[case[0] for case in UNENFORCEABLE_CASES],
    )
    def test_stripped_keyword_never_raises(self, keyword, prop_schema, bad_value):
        schema = {"type": "object", "properties": {"v": prop_schema}, "required": ["v"]}
        payload = {"v": bad_value}

        # Pre-assertion — the anti-vacuity guard. If the full schema does NOT
        # reject this payload, the case proves nothing and must be fixed rather
        # than trusted.
        assert list(Draft202012Validator(schema).iter_errors(payload)), (
            f"{keyword}: payload does not violate the FULL schema, so this case is vacuous"
        )
        # And the keyword really is one the decoder never sees.
        assert keyword in stripped_keywords(schema)

        validator = _validator_for(schema)
        assert validator(object(), payload) == payload


class TestStructuralConstraintsStayFatal:
    """The stripper touches none of these — they must still fail loudly."""

    def test_missing_required_raises(self):
        validator = _validator_for(
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        )
        with pytest.raises(ModelRetry):
            validator(object(), {})

    def test_wrong_type_raises(self):
        validator = _validator_for({"type": "object", "properties": {"a": {"type": "integer"}}})
        with pytest.raises(ModelRetry):
            validator(object(), {"a": "not an int"})

    def test_enum_violation_raises(self):
        validator = _validator_for(
            {"type": "object", "properties": {"a": {"enum": ["x", "y"]}}, "required": ["a"]}
        )
        with pytest.raises(ModelRetry):
            validator(object(), {"a": "z"})

    def test_const_violation_raises(self):
        validator = _validator_for(
            {"type": "object", "properties": {"a": {"const": "fixed"}}, "required": ["a"]}
        )
        with pytest.raises(ModelRetry):
            validator(object(), {"a": "other"})

    def test_oneof_violation_raises(self):
        validator = _validator_for(
            {
                # `type: object` alongside `oneOf` mirrors the real fr-agent
                # schema; StructuredDict rejects a bare `oneOf` root outright.
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "a"}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "b"}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                ],
            }
        )
        with pytest.raises(ModelRetry):
            validator(object(), {"kind": "c"})

    def test_additional_properties_false_raises(self):
        validator = _validator_for(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        with pytest.raises(ModelRetry):
            validator(object(), {"a": "ok", "sneaky": 1})


class TestScalarAndArrayRoots:
    """The second, quieter route into the same bug.

    Scalar and array roots never reach the stripper at all: the decoder is handed
    a bare Python type (`str`, `list[str]`) carrying no constraints whatsoever,
    while the validator enforced the full original.
    """

    def test_scalar_root_enum_is_still_fatal(self):
        """The trap: `enum` is NOT stripped, and on a scalar root the validator is its ONLY enforcement.

        `_output_type_from_schema` hands the decoder a bare `str` here, so if
        `_decoder_visible_schema` dropped `enum` while rebuilding the scalar
        root, every out-of-vocabulary label would silently become a 200 carrying
        an arbitrary string — and `message_endpoints` would tag that response
        `output_type="enum"`. That is a worse failure than the one being fixed.
        """
        schema = {"type": "string", "enum": ["alpha", "beta"]}
        assert _decoder_visible_schema(schema)["enum"] == ["alpha", "beta"]

        validator = _validator_for(schema)
        with pytest.raises(ModelRetry):
            validator(object(), "gamma")

    def test_scalar_root_const_is_still_fatal(self):
        schema = {"type": "string", "const": "only"}
        assert _decoder_visible_schema(schema)["const"] == "only"

        validator = _validator_for(schema)
        with pytest.raises(ModelRetry):
            validator(object(), "other")

    def test_array_root_item_constraint_is_advisory(self):
        validator = _validator_for({"type": "array", "items": {"type": "string", "minLength": 5}})
        assert validator(object(), ["ab"]) == ["ab"]

    def test_array_root_still_rejects_a_non_list(self):
        validator = _validator_for({"type": "array", "items": {"type": "string"}})
        with pytest.raises(ModelRetry):
            validator(object(), "not a list")

    def test_unmappable_item_type_is_not_over_constrained(self):
        """`items: {"type": "object"}` decodes as `list[Any]`.

        Naming the item type in the reconstructed schema would make the
        validator STRICTER than the decoder — reintroducing the asymmetry from
        the other direction.
        """
        schema = {"type": "array", "items": {"type": "object"}}
        assert "items" not in _decoder_visible_schema(schema)

        validator = _validator_for(schema)
        assert validator(object(), ["a", "b"]) == ["a", "b"]


class TestStripperTripwire:
    def test_unsupported_key_set_is_exactly_these_fourteen(self):
        """Adding a keyword to the stripper without extending the parity cases is a bug.

        If this fails, a new keyword became unenforceable. Decide whether it
        belongs in UNENFORCEABLE_CASES (enforced by jsonschema in production) or
        in the `format`/`title` exclusion note, then update BOTH.
        """
        assert set(_XGRAMMAR_UNSUPPORTED_KEYS) == {
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minProperties",
            "maxProperties",
            "pattern",
            "format",
            "title",
        }

    def test_every_parametrised_keyword_is_actually_stripped(self):
        parametrised = {case[0] for case in UNENFORCEABLE_CASES}
        assert parametrised <= set(_XGRAMMAR_UNSUPPORTED_KEYS)
        assert parametrised == set(_XGRAMMAR_UNSUPPORTED_KEYS) - {"format", "title"}

    def test_stripped_keywords_ignores_data_positions(self):
        """A property NAMED `minimum` is data, not a constraint."""
        schema = {
            "type": "object",
            "properties": {"minimum": {"type": "integer"}},
            "required": ["minimum"],
        }
        assert stripped_keywords(schema) == set()


class TestObservability:
    def test_unenforced_constraints_are_logged_at_registration(self, caplog):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}},
        }
        with caplog.at_level(logging.WARNING):
            _validator_for(schema)
        assert "schema_constraints_unenforced" in caplog.text
        assert "maximum" in caplog.text
        assert "minimum" in caplog.text

    def test_clean_schema_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            _validator_for({"type": "object", "properties": {"a": {"type": "string"}}})
        assert "schema_constraints_unenforced" not in caplog.text

    def test_soft_violation_is_reported_once(self, caplog):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "maximum": 5}},
            "required": ["n"],
        }
        validator = _validator_for(schema)
        with caplog.at_level(logging.WARNING):
            assert validator(object(), {"n": 12}) == {"n": 12}
        assert caplog.text.count("schema_soft_violation") == 1

    def test_conforming_output_reports_no_soft_violation(self, caplog):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "maximum": 5}},
            "required": ["n"],
        }
        validator = _validator_for(schema)
        with caplog.at_level(logging.WARNING):
            validator(object(), {"n": 3})
        assert "schema_soft_violation" not in caplog.text
