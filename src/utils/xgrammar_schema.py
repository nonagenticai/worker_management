"""Prune JSON schemas to what vLLM's xgrammar backend can compile.

vLLM runs guided decoding through xgrammar
(``--structured-outputs-config {"backend":"xgrammar"}``). xgrammar cannot
compile several JSON-Schema keywords — most notably the bounded-array
``minItems``/``maxItems``. When it meets one it does not error: it
**silently disables structured decoding for that request**, so the model
answers unconstrained, the reply is often not JSON at all, validation
fails, and inference-router falls through to the next provider.

That failure is invisible from the worker record — the schema looks fine
in the database. So the strip happens here, at the seam where a stored
``output_schema`` becomes a *decoding constraint*, rather than on write:
the stored schema stays intact for validation and introspection, and only
the grammar is relaxed.

Relaxing the grammar is **not free**, and an earlier version of this
docstring claimed it was. It said the caller "still validates the returned
JSON against the full model (pydantic ``model_validate``)", so a reply
violating ``minItems`` would be "rejected downstream". That is false for the
``StructuredDict`` path, which is how every schema-bearing worker runs: there
is no downstream pydantic model, and the ``Draft202012Validator`` registered
in ``pydantic_ai_agent`` **is** the only enforcement.

Believing that premise cost a production outage on 2026-07-20. The validator
was built from the *unstripped* schema while the decoder got the stripped
one, so the model was graded against a contract it had never been shown; with
``retries=0`` on ``inference_router`` the first disagreement was an instant
HTTP 500, and the w2v ``demo_video`` pipeline failed 100% of runs.

The rule that replaced it: the *fatal* validator is built from the same
stripped schema the decoder saw, so the two can never disagree, and
``stripped_keywords`` below reports what was dropped so the now-unenforced
constraints are logged rather than silently lost.

**Opt-in exception, 2026-09-23: the count/length bounds** (``BOUND_KEYWORDS``).
Left advisory, a ``maxItems`` on a judge schema was a no-op — not in the
grammar, not checked afterwards. They are now always checked after decoding
against the ORIGINAL schema and the violating paths are reported in
``usage_metrics.bound_violations``. A run that opts in with
``enforce_schema_bounds`` re-asks the model with feedback naming the bound
(a ``ModelRetry``, within the worker's output retry budget) and, if the budget
runs out on a bound violation, fails with ``SchemaBoundViolationError`` (HTTP
422 ``SCHEMA_BOUND_VIOLATION``) — #358, 2026-10-10. Default OFF: RM
``principle_consistency`` has a written contract that WM does not reject an
oversized reply (it clamps on receipt). Everything else stripped stays advisory.

**Position matters.** These words are keywords only in *keyword position*.
The same word is perfectly legal as a *property name*, where it is data,
and eating it is a correctness bug rather than a relaxation: the grammar
loses a field the validator still requires, so the model cannot emit it,
validation can never pass, and the agent retries until its token budget is
gone. That is exactly what happened to ``worker-project-decompose`` on
2026-07-14 — its ``child_cards[].title`` and
``io_map.necessary_conditions[].title`` were deleted from the grammar while
``"title"`` stayed in ``required``, and every AA project decomposed into
nothing. So the walk below is position-aware: it recurses into subschemas,
preserves the *names* in ``properties``-like maps, and never descends into
data (``enum``/``const``/``required``).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

_XGRAMMAR_UNSUPPORTED_KEYS: frozenset[str] = frozenset(
    {
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
)

# The count/length bounds. Stripped from the grammar like every key above, but
# — unlike the rest — checked after decoding against the caller's ORIGINAL
# schema: reported always; retried with feedback, then FATAL, when the run opts
# in with `enforce_schema_bounds` (see `SchemaBoundViolationError`). A bound that is neither in
# the grammar nor checked afterwards is a bound that does nothing, silently:
# judge schemas carry `maxItems`/`maxLength` precisely to stop T=0 repetition
# loops, and before this they were a no-op at inference time.
#
# The other stripped keys (minimum/pattern/format/…) stay ADVISORY
# (`schema_soft_violation`), exactly as the 2026-07-20 outage fix left them.
BOUND_KEYWORDS: frozenset[str] = frozenset({"minItems", "maxItems", "minLength", "maxLength"})

#: The typed reason a bound violation surfaces with, in logs, the
#: ``aa_prompt_validation_failures_total`` ``error_type`` label, the HTTP error
#: code and the ``worker_states`` error row.
SCHEMA_BOUND_VIOLATION = "schema_bound_violation"


class SchemaBoundViolationError(Exception):
    """A model output broke a count/length bound the decoder was never given.

    Raised — never truncated, never "repaired" — when an opted-in run's output
    satisfies the decoder-visible (stripped) schema but violates one of
    ``BOUND_KEYWORDS`` in the caller's original schema, AFTER the worker's
    output retry budget is spent. Each earlier violation is re-asked as a
    ``ModelRetry`` (``SchemaBoundRetry`` in the agent module) whose feedback
    states the bound — the grammar cannot carry it, but the prompt can. This
    error itself is deliberately NOT a ``ModelRetry``: it is the terminal,
    typed outcome (HTTP 422) once re-asking has failed.

    ``violations`` holds one ``{"path", "keyword", "message"}`` per failing
    constraint; ``path`` is a JSON pointer into the OUTPUT (``""`` = root).
    """

    reason = SCHEMA_BOUND_VIOLATION

    def __init__(self, agent_name: str, violations: list[dict[str, Any]]) -> None:
        self.agent_name = agent_name
        self.violations = violations
        detail = "; ".join(
            f"{v['keyword']} at {v['path'] or '<root>'}: {v['message']}" for v in violations[:5]
        )
        super().__init__(
            f"{SCHEMA_BOUND_VIOLATION}: agent {agent_name!r} output violates a schema bound "
            f"that is enforced post-hoc only (stripped from the decoding grammar): {detail}"
        )


# name -> subschema. The KEYS are property names (data); only the VALUES are
# schemas. This is the distinction #118 missed.
_SCHEMA_MAP_KEYS: frozenset[str] = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)

# keyword -> a single subschema.
_SUBSCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    }
)

# keyword -> a list of subschemas.
_SUBSCHEMA_LIST_KEYS: frozenset[str] = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})

# Pure data. Never filter it, never recurse into it — an enum value or a
# default may be an arbitrary object whose keys mean nothing to xgrammar, and
# `dependentRequired`/`required` hold property *names*. Descending into any of
# these is the exact mistake this module exists to correct.
_DATA_KEYS: frozenset[str] = frozenset(
    {"enum", "const", "default", "examples", "required", "dependentRequired"}
)

# Keywords that can satisfy a `required` entry from somewhere other than the
# local `properties` map. Their presence makes the local-only check below
# unsound, so it stays quiet rather than crying wolf on a legal schema.
_COMPOSITION_KEYS: frozenset[str] = frozenset(
    {"allOf", "anyOf", "oneOf", "$ref", "if", "then", "else", "dependentSchemas"}
)


def _warn_if_unsatisfiable(original: dict[str, Any], stripped: dict[str, Any]) -> None:
    """Flag a grammar that cannot satisfy the validator it is paired with.

    The decoder is handed the stripped schema while the caller validates
    against the original. If a stripped-away property is still ``required``,
    the two disagree and *no* completion can pass — the agent burns its whole
    token budget on doomed retries. That invariant is the bug class; this is
    the tripwire for it.

    Stays silent on composed schemas: a ``required`` name may legally be
    declared under ``allOf``/``$ref`` rather than locally. A tripwire for a P0
    class earns its keep only if it never cries wolf.
    """
    props = stripped.get("properties")
    if not isinstance(props, dict):
        return
    required = original.get("required")
    if not isinstance(required, list):
        return
    if any(key in original for key in _COMPOSITION_KEYS):
        return
    missing = [name for name in required if isinstance(name, str) and name not in props]
    if missing:
        logger.warning(
            "xgrammar strip produced an unsatisfiable grammar: %s required but "
            "absent from properties. The decoder cannot emit these fields and "
            "validation will never pass.",
            missing,
        )


def strip_xgrammar_unsupported(node: Any) -> Any:
    """Recursively drop xgrammar-incompatible keywords from a JSON schema.

    Only keywords in *keyword position* are dropped. Property names are
    preserved verbatim even when they collide with a keyword (``title`` is
    both a common annotation and a very common field name).

    ``title`` in keyword position is stripped, so read any name you need off
    the *original* schema before calling this.
    """
    if isinstance(node, list):
        return [strip_xgrammar_unsupported(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _XGRAMMAR_UNSUPPORTED_KEYS:
            continue  # keyword position — the only place these are keywords
        if key in _DATA_KEYS:
            # Data: verbatim, no recursion, no filtering. Copied rather than
            # aliased so the caller's schema — which the validator still uses
            # in full — can never be reached through the relaxed one.
            out[key] = deepcopy(value)
        elif key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            # Keys here are NAMES chosen by the schema author — keep every one
            # of them; only their values are subschemas.
            out[key] = {name: strip_xgrammar_unsupported(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [strip_xgrammar_unsupported(sub) for sub in value]
        elif key in _SUBSCHEMA_KEYS:
            out[key] = strip_xgrammar_unsupported(value)
        else:
            out[key] = strip_xgrammar_unsupported(value)

    _warn_if_unsatisfiable(node, out)
    return out


def stripped_keywords(node: Any) -> set[str]:
    """Report which xgrammar-incompatible keywords ``strip_xgrammar_unsupported`` drops.

    Mirrors that walk exactly — same position-awareness, same refusal to
    descend into ``_DATA_KEYS`` — but collects instead of pruning. Kept
    deliberately as a parallel walk rather than folded into the stripper so
    callers that only want to *know* never pay for a deepcopy.

    A dropped keyword is a constraint the decoder was never told about, so it
    must not be enforced fatally. This is what lets the caller say so out loud
    instead of losing it silently.
    """
    found: set[str] = set()

    def walk(current: Any) -> None:
        if isinstance(current, list):
            for item in current:
                walk(item)
            return
        if not isinstance(current, dict):
            return
        for key, value in current.items():
            if key in _XGRAMMAR_UNSUPPORTED_KEYS:
                found.add(key)  # keyword position — the only place these count
            elif key in _DATA_KEYS:
                continue  # data, not schema: a `title` in here is a value
            elif key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
                for sub in value.values():
                    walk(sub)
            elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
                for sub in value:
                    walk(sub)
            else:
                walk(value)

    walk(node)
    return found


__all__ = [
    "BOUND_KEYWORDS",
    "SCHEMA_BOUND_VIOLATION",
    "SchemaBoundViolationError",
    "strip_xgrammar_unsupported",
    "stripped_keywords",
]
