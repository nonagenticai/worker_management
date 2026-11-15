"""Regression test for inference-router schema-validation lock-out.

The OpenAI profile transformer (pydantic-ai's profiles/openai.py) adds
`additionalProperties: false` at the outer level of any `type:object`
schema when `strict is None`. For schemas using top-level `oneOf` with
no outer `properties`, this produces a logically-empty schema that
rejects every key (including the keys inside valid oneOf branches).

Fix: wrap NativeOutput with `strict=False` for the `inference_router`
provider so the transformer's strict=None default branch doesn't fire.
"""

from typing import Any

from pydantic_ai import NativeOutput, PromptedOutput

from src.worker.agents.pydantic_ai_agent import _output_type_from_schema

_ONEOF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "final_answer"},
                "text": {"type": "string", "minLength": 1},
            },
            "required": ["kind", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "tool_calls"},
                "calls": {"type": "array", "minItems": 1, "items": {}},
            },
            "required": ["kind", "calls"],
            "additionalProperties": False,
        },
    ],
}


def test_inference_router_native_output_is_strict_false() -> None:
    """inference_router wrapping must produce NativeOutput(strict=False).

    Reason: pydantic-ai's OpenAI profile mutates the schema in the
    strict=None branch by injecting empty `properties: {}` and
    `additionalProperties: false` at the outer level, which breaks
    oneOf-based discriminated unions. strict=False skips that mutation.
    """
    wrapped = _output_type_from_schema(_ONEOF_SCHEMA, "inference_router")
    assert isinstance(wrapped, NativeOutput)
    assert wrapped.strict is False, (
        "inference_router NativeOutput must have strict=False to avoid "
        "pydantic-ai's outer-additionalProperties injection on top-level "
        "oneOf schemas"
    )


def test_scalar_schema_inference_router_still_strict_false() -> None:
    """Scalar (non-object) schemas also use NativeOutput(strict=False)."""
    wrapped = _output_type_from_schema({"type": "boolean"}, "inference_router")
    assert isinstance(wrapped, NativeOutput)
    assert wrapped.strict is False


def test_mlx_provider_unchanged_uses_prompted_output() -> None:
    """mlx provider keeps PromptedOutput wrapping — strict flag does not apply."""
    wrapped = _output_type_from_schema(_ONEOF_SCHEMA, "mlx")
    assert isinstance(wrapped, PromptedOutput)
