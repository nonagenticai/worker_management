"""Unit tests for scripts/improvement_workers/register.py.

Uses httpx.MockTransport to fake the worker_management HTTP API; no live
service required. Validates payload shape, idempotency, and JSON Schema
correctness for the seeded workers across BOTH improvement loops:

* workflow_loop (Plan 5) -- 5 workers
* prompt_loop  (Plan 6) -- 3 workers
* asset_authoring -- 3 workers
* procedure_loop -- 1 worker
* asset_loop -- 3 workers

Tests use ``register.AGENT_CONFIGS`` (the merged 15-worker registry) for
end-to-end registration flows, and per-loop AGENT_CONFIGS for per-loop
schema/config assertions.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import httpx2 as httpx
import jsonschema
import pytest

from scripts.improvement_workers.register import (
    AGENT_CONFIGS as REGISTER_AGENT_CONFIGS,
)
from scripts.improvement_workers.register import (
    SYSTEM_WORKFLOW_TENANT_ID,
    EmptyLivePromptError,
    WriteVerificationError,
    _build_headers,
    asserted_tenant_uuid,
    build_payload,
    register_workers,
    resolve_target_tenant,
)
from scripts.improvement_workers.register import (
    main as register_main,
)
from scripts.improvement_workers.workers import workflow_loop
from scripts.improvement_workers.workers.asset_loop import (
    AGENT_CONFIGS as ASSET_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.asset_loop import (
    SCHEMAS_DIR as ASSET_LOOP_SCHEMAS_DIR,
)
from scripts.improvement_workers.workers.procedure_loop import (
    AGENT_CONFIGS as PROCEDURE_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.procedure_loop import (
    SCHEMAS_DIR as PROCEDURE_LOOP_SCHEMAS_DIR,
)
from scripts.improvement_workers.workers.prompt_loop import (
    AGENT_CONFIGS as PROMPT_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.prompt_loop import (
    SCHEMAS_DIR as PROMPT_LOOP_SCHEMAS_DIR,
)
from scripts.improvement_workers.workers.workflow_loop import (
    AGENT_CONFIGS,
    SCHEMAS_DIR,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

WM_BASE = "http://wm.test"
ROUTER_URL = "http://router.test/api"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=_transport(handler), base_url=WM_BASE)


# ---------------------------------------------------------------------------
# schema validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_path",
    sorted((SCHEMAS_DIR).glob("*.json")),
    ids=lambda p: p.name,
)
def test_schema_files_are_valid_jsonschema_2020_12(schema_path: Path) -> None:
    """Every shipped schema must be a valid JSON Schema 2020-12 document."""
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    # Raises jsonschema.SchemaError if the schema itself is invalid.
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    if schema_path.name == "workflow_development.json":
        # The development worker was converted to emit a bounded RFC-6902
        # JSON-Patch under `workflow_patch` (mirrors prompt_loop/procedure_loop);
        # it no longer shares the analyzers' proposal/predicted_deltas shape.
        for required in ("workflow_patch", "rationale", "expected_impact", "scope_of_change"):
            assert required in schema["required"], (
                f"{schema_path.name} missing required '{required}'"
            )
        return
    for required in ("proposal", "predicted_deltas", "rationale", "confidence"):
        assert required in schema["required"], f"{schema_path.name} missing required '{required}'"


def test_all_five_card_types_have_a_schema_and_config() -> None:
    """Plan 5 ships exactly five card_types; the file set must match."""
    expected = {
        "improvement_workflow_statistics_analysis",
        "improvement_workflow_performance_analysis",
        "improvement_workflow_redesign",
        "improvement_workflow_pattern_catalogue_improvement",
        "improvement_workflow_development",
    }
    assert set(AGENT_CONFIGS.keys()) == expected
    for cfg in AGENT_CONFIGS.values():
        assert (SCHEMAS_DIR / cfg.schema_filename).is_file(), (
            f"missing schema file for {cfg.worker_name}"
        )


def test_workflow_development_schema_is_a_bounded_rfc6902_patch() -> None:
    """The development worker emits a bounded RFC-6902 JSON-Patch under
    ``workflow_patch`` (mirrors prompt_loop ``candidate_system_prompt`` / procedure_loop
    ``candidate_procedure``); the old prose ``implementation_notes`` + nullable
    ``new_workflow_id`` shape is gone."""
    schema = AGENT_CONFIGS["improvement_workflow_development"].output_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["title"] == "WorkflowDevelopmentProposal"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "workflow_patch",
        "rationale",
        "expected_impact",
        "scope_of_change",
    }
    patch = schema["properties"]["workflow_patch"]
    assert patch["type"] == "array"
    # Bounded: a maxItems cap keeps the patch small/targeted.
    assert isinstance(patch.get("maxItems"), int) and patch["maxItems"] > 0
    item = patch["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"op", "path"}
    assert set(item["properties"]["op"]["enum"]) == {
        "add",
        "remove",
        "replace",
        "move",
        "copy",
        "test",
    }
    assert set(schema["properties"]["scope_of_change"]["enum"]) == {
        "node_parameter",
        "node_addition",
        "node_removal",
        "connection_rewire",
        # A decline must be EXPRESSIBLE. ``workflow_patch`` has no ``minItems``
        # (deliberately -- see test_empty_patch_is_schema_valid_and_declarable in
        # tests/test_system_worker_output_contract.py), so an empty patch is
        # schema-valid; without this member a declining model would have to pick a
        # scope it is not making, and the caller could not tell an honest "nothing
        # to change" from a malformed proposal.
        "no_change",
    }
    # The retired keys must NOT be in the new schema.
    props = schema["properties"]
    assert "implementation_notes" not in props
    assert "new_workflow_id" not in props
    assert "proposal" not in props


def test_workflow_loop_improvement_development_names_the_output_contract() -> None:
    """The improvement_workflow_development worker's system_prompt MUST name the
    ``workflow_patch`` output key and forbid the retired ``implementation_notes`` /
    ``new_workflow_id`` aliases -- otherwise the model returns the change under a
    drifted key, the downstream workflow-canary Parse Proposal skips it, and the
    loop never compounds (the prompt loop's 2026-06-16 stall, applied to workflows)."""
    cfg = AGENT_CONFIGS["improvement_workflow_development"]
    required = set(cfg.output_schema().get("required", []))
    assert "workflow_patch" in required
    sp = cfg.system_prompt
    assert "workflow_patch" in sp, (
        "improvement_workflow_development system_prompt must name the workflow_patch output key"
    )
    # Explicitly forbid the retired prose aliases.
    assert "do NOT use" in sp
    assert "implementation_notes" in sp, (
        "system_prompt must explicitly forbid the implementation_notes alias"
    )
    assert "new_workflow_id" in sp, "system_prompt must explicitly forbid the new_workflow_id alias"
    for key in ("rationale", "expected_impact", "scope_of_change"):
        assert key in sp, f"system_prompt must name {key}"
    # The other 4 workflow_loop workers keep their original (non-patch) contracts.
    for name, other in AGENT_CONFIGS.items():
        if name == "improvement_workflow_development":
            continue
        assert "workflow_patch" not in other.system_prompt, (
            f"{name} must NOT have been given the development output contract"
        )


# ---------------------------------------------------------------------------
# improvement_workflow_development: prompt <-> schema contract
#
# WHY these exist: the 2026-09-13 prompt rewrite was measured to pass all 78
# tests in this repo with the entire commit reverted -- nothing in the suite
# could see the prompt's semantics, so a regression to the prompt that made the
# worker decline on 2 of 2 live kaizen drives would have been invisible. Every
# assertion below is checked to FAIL against the origin/main (dd5e436) prompt,
# except the scope_of_change enum coverage, which dd5e436 already satisfied and
# which is kept as a forward drift guard (see its docstring).
# ---------------------------------------------------------------------------

_DEV_WORKER = "improvement_workflow_development"

# ---------------------------------------------------------------------------
# The decline: a BOUNDED REGION, not a keyword count.
#
# Two successive guards counted "blessing sentences" by keyword over the assembled prose.
# Both were defeated, measured, with this file green at 61 passed:
#
#   ATTACK 1 -- LAUNDERING. The guard exempted any sentence containing the exact
#     anti-fabrication phrase, so the blessing was appended to THAT sentence with an
#     em-dash and stayed one sentence: "...an honest decline is not -- so whenever you are
#     not completely certain a change is warranted, declining with an empty patch is always
#     the right and welcome call and you should prefer it." Phrase present, sentence exempt,
#     blessing count still 1, green.
#   ATTACK 2 -- UNDER-COUNT. A blessing that uses none of the marker words: "Emitting
#     scope_of_change = 'no_change' with a zero-length array is a healthy, first-class
#     terminal state for this loop and is never counted against you." No marker, not
#     counted, green.
#
# Both attacks live below as executable fixtures and are asserted to RED, so this guard
# cannot quietly become vacuous the way its predecessors did.
#
# The replacement is structural. `_DECLINE_POLICY` is a named constant in workflow_loop.py
# and is the only place the dev prompt may discuss declining. Three assertions:
#   (1) it appears in the assembled prompt EXACTLY once and BYTE-IDENTICALLY to the copy
#       pinned here -- so no edit to the decline wording, an em-dash continuation included,
#       can pass without editing this file and re-arguing the change;
#   (2) every sentence of the region is a whole sentence of the prompt -- so the blessing
#       cannot be smuggled in by extending the region's first or last sentence from
#       OUTSIDE the constant, which is attack 1 performed at the seam;
#   (3) outside the region the prompt carries NO decline vocabulary, with exactly one
#       accounted-for exception: the literal 'no_change' may appear once and only in the
#       sentence that enumerates all five scope_of_change values.
#
# ⚠️ WHAT THIS GUARD ACTUALLY HOLDS, stated plainly because an earlier version of this
# comment overstated it. (1) and (2) hold: they are byte-wise and lexical-judgement-free,
# so no edit to the decline wording and no extension of a region sentence from outside the
# constant can pass. (3) is PROSE MATCHING and it is defeated by any blessing that avoids
# the 19 `_DECLINE_VOCAB` terms. The earlier text confined that hole to "a blessing
# appended to the enum sentence"; it is much wider than that. `_decline_region_violations`
# does `if not hits: continue`, so a sentence carrying none of the terms is skipped
# OUTRIGHT -- a token-avoiding blessing passes ANYWHERE outside the region, and the enum
# exemption has nothing to do with it. ATTACK 3 below does exactly that and this guard does
# not see it (asserted, in that direction, by
# test_the_region_guard_does_NOT_catch_the_token_avoiding_blessing).
# THAT IS WHY THE BYTE PIN EXISTS -- see the pin section further down. This guard is kept
# as a cheap early warning for the vocabulary class and for its executable attack fixtures;
# it is NOT the thing that makes a blessing unable to land. Do not widen `_DECLINE_VOCAB`
# to try to close the hole: that is the move that has now failed three times.
# ---------------------------------------------------------------------------

_ANTI_FABRICATION_PHRASE = "never invent an operation to avoid an empty patch"

# Pinned byte-for-byte against workflow_loop._DECLINE_POLICY. If this assertion reds, the
# decline wording changed: that is the point. Do not re-sync it without reading why the
# region exists -- the worker declined on 2 of 2 live kaizen drives with a prompt that
# blessed declining four times over.
_PINNED_DECLINE_POLICY = (
    " If the "
    "workflow genuinely needs no change, emit an EMPTY `workflow_patch` array WITH "
    "`scope_of_change` = 'no_change' and a `rationale` naming what you examined. NEVER "
    "invent an operation to avoid an empty patch: a fabricated change is promoted to a live "
    "canary, an honest decline is not."
)

# Vocabulary that has no business appearing outside the decline region. Over-inclusive on
# purpose (see (3) above): "healthy", "welcome", "legitimate" and friends are here because
# they are what a blessing reaches for, not because the prompt would otherwise want them.
_DECLINE_VOCAB = (
    "declin",
    "empty patch",
    "empty `workflow_patch`",
    "zero-length",
    "nothing to change",
    "no change",
    "no_change",
    "terminal state",
    "first-class",
    "welcome",
    "against you",
    "healthy",
    "prefer it",
    "right call",
    "legitimate",
    "honest",
    "expected outcome",
    "restraint",
    "when in doubt",
)


def _enumerates_every_scope(sentence: str, scope_values: list[str]) -> bool:
    """True when ``sentence`` lists every ``scope_of_change`` value the schema declares.

    This is the one exemption in the outside-the-region sweep, and it is keyed on a
    property a blessing cannot cheaply borrow: naming all five enum members.
    """
    return all(f"'{v}'" in sentence for v in scope_values)


def _decline_region_violations(prompt: str, policy: str, scope_values: list[str]) -> list[str]:
    """Ways ``prompt`` breaks the bounded-region rule. Empty list == no violation FOUND.

    ⚠️ An empty list is NOT "the prompt does not bless declining". Legs (1) occurs-exactly-
    once-byte-identically and (2) region-sentences-are-whole-prompt-sentences hold without
    judgement. Leg (3) -- decline vocabulary outside the region -- is prose matching and is
    defeated by any blessing that avoids the 19 `_DECLINE_VOCAB` terms: the ``if not hits:
    continue`` below skips such a sentence outright, so it passes ANYWHERE outside the
    region, not merely inside the enum exemption. That is what the byte pin is for; this
    function is an early warning, not the load-bearing check.
    """
    problems: list[str] = []
    occurrences = prompt.count(policy)
    if occurrences != 1:
        problems.append(
            f"decline policy region occurs {occurrences} times in the prompt, expected 1 "
            "(it was edited, inlined, or duplicated)"
        )
        return problems
    prompt_sentences = _sentences(prompt)
    for sentence in _sentences(policy):
        if sentence not in prompt_sentences:
            problems.append(
                f"region sentence is not a whole sentence of the prompt -- something was "
                f"joined onto it from outside the constant: {sentence!r}"
            )
    outside = prompt.replace(policy, " ")
    for sentence in _sentences(outside):
        low = sentence.lower()
        hits = [v for v in _DECLINE_VOCAB if v in low]
        if not hits:
            continue
        if set(hits) <= {"no_change"} and _enumerates_every_scope(sentence, scope_values):
            continue
        problems.append(f"decline vocabulary {hits} outside the decline region: {sentence!r}")
    return problems


def _attack_laundered_blessing(prompt: str) -> str:
    """ATTACK 1, verbatim: continue the exempt anti-fabrication sentence with an em-dash."""
    tail = "an honest decline is not."
    assert prompt.count(tail) == 1, "attack fixture no longer targets the prompt"
    return prompt.replace(
        tail,
        "an honest decline is not -- so whenever you are not completely certain a change "
        "is warranted, declining with an empty patch is always the right and welcome call "
        "and you should prefer it.",
    )


def _attack_unmarked_blessing(prompt: str) -> str:
    """ATTACK 2, verbatim: a blessing that uses none of the old marker words."""
    anchor = " Reply with a single JSON object"
    assert prompt.count(anchor) == 1, "attack fixture no longer targets the prompt"
    return prompt.replace(
        anchor,
        " Emitting scope_of_change = 'no_change' with a zero-length array is a healthy, "
        "first-class terminal state for this loop and is never counted against you." + anchor,
    )


def _sentences(text: str) -> list[str]:
    """Split a prompt into sentences on '. ' boundaries.

    The worker prompts contain no abbreviations or decimals, so a period
    followed by whitespace is an unambiguous sentence boundary.
    """
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def _sentence_with(text: str, needle: str) -> str:
    """Return the single sentence containing ``needle`` (case-insensitive)."""
    hits = [s for s in _sentences(text) if needle.lower() in s.lower()]
    assert len(hits) == 1, f"expected exactly 1 sentence containing {needle!r}, found {len(hits)}"
    return hits[0]


def test_dev_prompt_states_json_pointer_semantics_and_positional_nodes() -> None:
    """``path`` must be named as an RFC-6901 JSON Pointer AND ``nodes`` as positional.

    The genuinely new content is the POSITIONAL ``nodes`` statement: nothing told the
    model that ``nodes`` is addressed by index, so it could reason its way to a correct
    change and still emit ``/nodes/HTTP Request/parameters/url``, which never applies.

    The RFC-6901 half is NOT new information to the system -- the schema's
    ``path.description`` already reads "A JSON Pointer (RFC-6901)" and descriptions reach
    the model (see test_dev_schema_descriptions_are_descriptive_not_hortatory). It is
    asserted here anyway because the prompt and the schema are separate channels and the
    prompt is the one this file guards, but the claim is only "the prompt says it", not
    "the system would otherwise be silent about it".
    """
    sp = AGENT_CONFIGS[_DEV_WORKER].system_prompt
    low = sp.lower()
    assert "rfc-6901" in low, "prompt must cite RFC-6901 for the patch `path`"
    assert "json pointer" in low, "prompt must call `path` a JSON Pointer"
    positional = _sentence_with(sp, "positional")
    assert "nodes" in positional.lower(), "the positional-addressing sentence must be about `nodes`"
    assert "node name" in positional.lower(), (
        "the prompt must rule out addressing `nodes` by node name"
    )


def test_dev_prompt_names_every_expected_impact_key_the_schema_declares() -> None:
    """Every ``expected_impact`` key is read OUT OF THE SCHEMA and required in the prompt.

    ``expected_impact`` is ``additionalProperties: false`` with no ``required``, so
    "an object" (dd5e436) invites an invented key that fails validation. Deriving the
    key list here means a schema that GAINS a key the prompt does not mention reds --
    that drift is exactly what produced this bug. FAILS on dd5e436.
    """
    cfg = AGENT_CONFIGS[_DEV_WORKER]
    impact = cfg.output_schema()["properties"]["expected_impact"]
    assert impact["additionalProperties"] is False, (
        "if expected_impact ever accepts extra keys, this test's premise is gone"
    )
    keys = list(impact["properties"])
    assert keys, "schema declares no expected_impact keys -- this test would be vacuous"
    sp = cfg.system_prompt
    missing = [k for k in keys if k not in sp]
    assert not missing, (
        f"expected_impact keys declared by the schema but absent from the prompt: {missing}"
    )


def test_dev_prompt_names_every_scope_of_change_enum_value() -> None:
    """Every ``scope_of_change`` enum member is read out of the schema and required.

    NOTE: dd5e436 already listed all five, so this one does NOT fail on the old
    prompt. It is kept as a forward drift guard: it reds when the schema gains or
    renames a member and the prompt is not updated with it.
    """
    cfg = AGENT_CONFIGS[_DEV_WORKER]
    enum = cfg.output_schema()["properties"]["scope_of_change"]["enum"]
    assert enum, "schema declares no scope_of_change values -- this test would be vacuous"
    sp = cfg.system_prompt
    missing = [v for v in enum if f"'{v}'" not in sp]
    assert not missing, (
        f"scope_of_change values declared by the schema but absent from the prompt: {missing}"
    )


def test_dev_prompt_confines_the_decline_to_one_pinned_region() -> None:
    """The decline is stated ONCE, inside ``_DECLINE_POLICY``, and nowhere else.

    It has to be present (an empty patch with ``scope_of_change`` = 'no_change' is
    schema-valid and the model must know the pairing), and it must not be stacked:
    dd5e436 carried 3 decline sentences plus "and otherwise decline" with nothing pulling
    the other way, and the worker declined on 2 of 2 live drives.

    This does NOT count blessings. Counting prose was tried twice and walked around twice
    -- see the region comment above for both attacks. What is asserted here is that the
    decline text is byte-identical to the pinned copy, that it occupies whole sentences of
    the prompt, and that no decline vocabulary survives outside it. FAILS on dd5e436, whose
    decline sentences are neither this text nor confined to one region.
    """
    cfg = AGENT_CONFIGS[_DEV_WORKER]
    sp = cfg.system_prompt
    assert workflow_loop._DECLINE_POLICY == _PINNED_DECLINE_POLICY, (
        "workflow_loop._DECLINE_POLICY changed. That is the guard firing, not a stale "
        "test: re-read the region comment, argue the new wording, then re-pin it here."
    )
    scope_values = cfg.output_schema()["properties"]["scope_of_change"]["enum"]
    problems = _decline_region_violations(sp, _PINNED_DECLINE_POLICY, scope_values)
    assert not problems, "decline region violated:\n" + "\n".join(problems)


def test_the_decline_region_guard_reds_on_both_known_attacks() -> None:
    """The guard must FAIL on the two prompts that defeated its predecessors.

    Without this the guard is only asserted in the passing direction, which is exactly how
    the previous two versions shipped: each was green against the real prompt and green
    against the blessing it was supposed to stop. Both attacks are applied to the REAL
    prompt so they cannot drift away from what they are attacking.
    """
    cfg = AGENT_CONFIGS[_DEV_WORKER]
    sp = cfg.system_prompt
    scope_values = cfg.output_schema()["properties"]["scope_of_change"]["enum"]
    for label, attacked in (
        ("laundered (em-dash continuation of the exempt sentence)", _attack_laundered_blessing(sp)),
        ("unmarked (blessing using none of the old marker words)", _attack_unmarked_blessing(sp)),
    ):
        assert attacked != sp, f"{label}: fixture made no change to the prompt"
        problems = _decline_region_violations(attacked, _PINNED_DECLINE_POLICY, scope_values)
        assert problems, f"{label}: the guard PASSED a known decline blessing"


# ---------------------------------------------------------------------------
# THE BYTE PIN -- the bytes of exactly what register.py PUTs, for ALL FIFTEEN SEEDED workers.
#
# WHY THIS EXISTS AND WHY IT IS NOT A FOURTH WORDLIST. Three vocabulary-based prose
# classifiers have now been walked around, each within minutes:
#   * ATTACK 1 (laundering) and ATTACK 2 (unmarked blessing) defeated the two guards that
#     COUNTED blessing sentences by keyword -- see the region comment above.
#   * ATTACK 3 defeats the REGION guard itself. `_decline_region_violations` does
#     `if not hits: continue`, so a sentence containing none of the 19 `_DECLINE_VOCAB`
#     terms is skipped outright and a token-avoiding blessing passes ANYWHERE outside the
#     region -- not merely inside the enum exemption the region comment describes.
#     MEASURED 2026-09-13: `_ATTACK_3_SENTENCE` inserted at the head of `_CONTRACT_CLOSING`
#     in workflow_loop.py, this file green at 87 passed.
#   * ATTACK 4 uses a channel no guard covered at all. The SCHEMA reaches the model too:
#     `description` is not in `_XGRAMMAR_UNSUPPORTED_KEYS`, so it survives into the
#     decoder-visible schema, and it was policed only by the five-phrase blacklist in
#     test_dev_schema_descriptions_are_descriptive_not_hortatory. A blessing appended to
#     `scope_of_change.description` that uses none of those five phrases passed.
#     MEASURED 2026-09-13: 87 passed.
#   * ATTACK 5 -- THE ONE THAT MADE THIS SECTION COVER FIVE WORKERS INSTEAD OF ONE.
#     The first cut of this pin covered `improvement_workflow_development` ALONE. An
#     adversarial auditor threw nine evasions at it (schema property reorder, a payload
#     field outside the pin, `description`/`agent_role`, a per-call schema override,
#     mutating either sha, one byte in either fixture, making `regenerate_pin_fixtures()`
#     touch the pin) and broke none of them -- and then simply moved one worker upstream.
#     A decline blessing inserted into `_REDESIGN_PROMPT`, immediately after "...the patch
#     should achieve.", went green at 94 passed. `improvement_workflow_redesign` FEEDS
#     `improvement_workflow_development`, so a blessing there stalls the kaizen loop one
#     stage EARLIER than the stage three rounds of guards were spent defending. The defect
#     was never the guard's strength; it was its SCOPE.
#   * ATTACK 6 -- THE FORGED MANIFEST. `_manifest` interpolated the worker NAME raw into a
#     text whose row separator is `\n` and field separator is a space, so one field could
#     impersonate several -- the classic digest-of-digests failure. An auditor built a name
#     carrying two forged rows plus the next worker's name; it reproduced the pinned
#     manifest text BYTE FOR BYTE while `improvement_workflow_redesign` was no longer read
#     from `build_payload`, and `_manifest_violation` returned `None`. Only
#     `_pinned_set_violation` caught it, which made the "two independent barriers" claim
#     false. FIXED by `json.dumps`-ing every interpolated field; regression is
#     test_the_manifest_reds_on_a_worker_name_that_forges_extra_rows, which fails on the
#     FORGERY specifically, not on a plain drop.
#   * ATTACK 7 -- THE SAME SCOPE ERROR, ONE LEVEL UP. The post-attack-5 pin read
#     `workflow_loop.AGENT_CONFIGS` -- five workers -- while `register.py` SEEDS fifteen,
#     merging five loop modules. A decline blessing in `improvement_prompt_improvement`
#     therefore shipped green, and the only tripwire was a `== 15` count test whose obvious
#     repair is to write `16`. Fixed by reading the MERGED `register.AGENT_CONFIGS`, which
#     closes the CLASS: there is no further level to be one short of, because that dict is
#     by construction everything register.py PUTs.
#
# A wordlist cannot be complete, so this pin makes NO judgement about prose at all. It
# pins, for every worker in the MERGED `register.AGENT_CONFIGS`, the bytes of (a) the assembled
# `system_prompt` and (b) the serialised `output_schema`, both read out of
# `register.build_payload` -- the function register.py actually PUTs, so the pin covers
# what is shipped rather than a reconstruction. ANY change reds: a blessing, a typo, a
# helpful clarification. That is the property all three guards were reaching for, and it is
# unfalsifiable because it classifies nothing.
#
# ⚠️ SCOPE IS THE LOAD-BEARING PART, so it is asserted rather than assumed -- and it has
# been wrong TWICE (attacks 5 and 7), each time by being one level short of what ships.
# `_PINNED_WORKERS` is written out by hand and asserted to equal
# `set(register.AGENT_CONFIGS)` -- the MERGED registry, every worker register.py seeds --
# EXACTLY (test_the_pinned_worker_set_is_exactly_the_seeded_registry). A worker added to ANY
# loop module, including one that does not exist yet, therefore REDS this file rather than
# shipping unpinned (test_the_pinned_set_check_reds_on_a_worker_from_a_non_kaizen_loop) --
# which is the difference between closing the class and shipping a third copy of the same
# fix. The manifest below is built over `_PINNED_WORKERS`, so dropping a name from that
# tuple to silence the set assertion reds the manifest instead
# (test_dropping_a_worker_from_the_pinned_set_reds_the_manifest), and a name crafted to FORGE
# the dropped worker's rows reds it too
# (test_the_manifest_reds_on_a_worker_name_that_forges_extra_rows).
#
# ⚠️ WHY ALL FIFTEEN AND NOT JUST THE KAIZEN FIVE. The other ten are not adjacent code: they
# are seeded by the SAME `register.py` run, into the SAME WM instance, under the same
# internal tenant, and their `system_prompt` and `output_schema` reach a live LLM through the
# same `build_payload` -> `PUT /workers/{name}` path. A decline blessing in
# `improvement_prompt_improvement` stalls the PROMPT loop exactly as one in
# `improvement_workflow_redesign` stalls the workflow loop. There is no property of the ten
# that makes them safer -- only the accident that the first three audits happened to be
# about workflow_loop.
#
# HOW THE TWO HALVES SPLIT, AND WHY THERE IS ONLY ONE SHA LITERAL.
#   * The per-artefact comparison is against the FIXTURE FILES in
#     `fixtures/<worker_name>/{system_prompt.txt,output_schema.json}`, so a failure prints a
#     UNIFIED DIFF naming the offending sentence rather than an opaque `abc != def`.
#   * The fixtures are NOT self-certifying: `_PINNED_MANIFEST_SHA256` is one sha256 over a
#     manifest of every worker's name plus the length and sha256 of each artefact, and it is
#     asserted over BOTH the built payloads and the fixture files. Regenerating a fixture
#     therefore does NOT re-pin anything -- a human still has to read the diff and paste a
#     new manifest sha, which is the property the per-artefact sha literals had.
#   * ⭐ ONE literal, not thirty. Thirty pinned artefacts would have minted thirty
#     `Hex High Entropy String` findings in CI's detect-secrets gate, and every deliberate
#     prompt edit would then require a `.secrets.baseline` update -- recurring churn whose
#     obvious shortcut (`detect-secrets scan --baseline .secrets.baseline`, which REWRITES
#     the audit results it is checking) is banned in this fleet. Hashing the hashes gives
#     identical detection strength for ONE baselined literal: strictly less churn than the
#     two-literal single-worker pin this replaces, while covering fifteen times as much. If
#     you re-pin, the sanctioned baseline procedure is in `_BASELINE_PROCEDURE` below --
#     it is quoted in the failure message so nobody has to invent it.
#
# The pin covers the two fields that reach the model. The rest of the PUT body (agent_role,
# model, temperature, base_url_override, ...) is covered by test_build_payload_has_expected_fields
# and the registration-flow tests; it is left out of the pin so a router-URL change does not
# force a prompt re-argument.
# ---------------------------------------------------------------------------

_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"

# Written out by hand. Asserted to equal set(register.AGENT_CONFIGS) -- the MERGED registry,
# i.e. every worker register.py seeds, across every loop module -- exactly. See the scope note.
_PINNED_WORKERS: tuple[str, ...] = (
    # workflow_loop (Plan 5)
    "improvement_workflow_statistics_analysis",
    "improvement_workflow_performance_analysis",
    "improvement_workflow_redesign",
    "improvement_workflow_pattern_catalogue_improvement",
    "improvement_workflow_development",
    # prompt_loop (Plan 6)
    "improvement_prompt_performance_statistics_analysis",
    "improvement_prompt_library_consistency_management",
    "improvement_prompt_improvement",
    # asset_authoring
    "authoring_principle_development",
    "authoring_requirement_development",
    "authoring_decision_development",
    # procedure_loop
    "improvement_procedure_improvement",
    # asset_loop
    "improvement_principle_improvement",
    "improvement_requirement_improvement",
    "improvement_decision_improvement",
)

# Pinned 2026-09-13 over all FIFTEEN seeded workers' assembled prompt + serialised schema.
# THIS LINE IS THE PIN. `regenerate_pin_fixtures()` prints it; it never writes it.
_PINNED_MANIFEST_SHA256 = "3c34517119a6ae7474060a321f5b61c47820ec51eb132f3ece90713cc8b95876"

_REGEN_COMMAND = (
    "uv run python -c \"import sys; sys.path.insert(0, 'tests/scripts'); "
    'import test_improvement_workers_register as t; t.regenerate_pin_fixtures()"'
)

# ⚠️ STEP 4 IS THE ONE PEOPLE MISS, AND OMITTING IT IS THE ACTUAL HAZARD. A re-pinner reads
# exactly two artefacts: the failing test's message, and whatever `regenerate_pin_fixtures()`
# prints. The first version of this section listed THREE steps in both and never mentioned
# `.secrets.baseline` -- so it sent every re-pinner away believing they were done, into a red
# CI whose obvious repair is the banned in-place scan. A guard that quietly routes people to
# a destructive command is worse than the churn it was avoiding. Hence one shared block,
# embedded in both messages and asserted by
# test_the_repin_instructions_name_the_detect_secrets_step.
_REPIN_STEPS = f"""\
HOW TO RE-PIN -- ALL FOUR STEPS. Step 4 is not optional and is not someone else's job.
  1. {_REGEN_COMMAND}
  2. Paste the printed _PINNED_MANIFEST_SHA256 into
     tests/scripts/test_improvement_workers_register.py, replacing the old literal.
  3. Say in the commit message WHAT changed in the prompt or schema, and WHY.
  4. That literal is 64 hex chars, so detect-secrets flags it as `Hex High Entropy String`
     and CI's `Security scan - detect-secrets baseline` WILL red. THAT IS EXPECTED, not a
     second defect -- record the finding as reviewed. Scan into a COPY and adopt ONLY the
     `results` block:
         cp .secrets.baseline /tmp/scan-copy.json
         uv run detect-secrets scan --baseline /tmp/scan-copy.json --force-use-all-plugins \\
           --exclude-files '\\.secrets\\.baseline$|^\\.github/workflows/'
     then diff `.results` with `line_number` deleted against the COMMITTED baseline and copy
     across only the changed entry: `added` must be exactly
     tests/scripts/test_improvement_workers_register.py and `removed` must be exactly the
     retired pin literal -- any other file appearing is a real finding, not churn.
     (The gate does `git checkout -- .secrets.baseline` before comparing, so compare against
     the file on disk, not through that checkout.)
     ⛔ NEVER run `detect-secrets scan --baseline .secrets.baseline`: it REWRITES the audit
        results in the very file it is auditing -- in this fleet that has already destroyed
        a 299-line results block.
     ⛔ NEVER add `pragma: allowlist secret`: it blinds that line permanently, including
        against a real credential a later edit puts there."""

# ATTACK 3, verbatim as the adversarial verifier wrote it, inserted at the head of
# `_CONTRACT_CLOSING`. It names none of the 19 `_DECLINE_VOCAB` terms.
_ATTACK_3_SENTENCE = (
    " Returning a patch of length 0 with the fifth scope value is a fully valid, "
    "frequently correct answer that costs you nothing, and you should lean that way the "
    "moment a proposal feels forced."
)

# ATTACK 4, verbatim, appended to `scope_of_change.description` in the SCHEMA -- a channel
# the prompt-side guards never see.
_ATTACK_4_SENTENCE = (
    " ...and reaching for it whenever a proposal would be speculative is exactly what "
    "this field is for."
)

# ATTACK 5, verbatim, appended to the last sentence of `_REDESIGN_PROMPT` -- the worker one
# stage UPSTREAM of the development worker, which the single-worker pin did not cover.
_ATTACK_5_SENTENCE = (
    " If the workflow already looks acceptable as it stands, proposing nothing at all is a "
    "perfectly good outcome here and costs this loop nothing."
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _put_payload(worker: str = _DEV_WORKER) -> dict[str, Any]:
    """Exactly what ``register.py`` PUTs for ``worker``.

    Built through ``build_payload`` off the MERGED ``register.AGENT_CONFIGS``, so the pin
    covers the shipped object rather than a reconstruction assembled in the test.
    """
    return build_payload(REGISTER_AGENT_CONFIGS[worker], base_url_override=ROUTER_URL)


def _schema_text(schema: dict[str, Any]) -> str:
    """Serialise an ``output_schema`` for pinning.

    ``sort_keys`` is deliberately OFF: key order is preserved from the JSON file, so a
    reordering is a change the pin SEES rather than one it canonicalises away.
    """
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def _built_texts(worker: str) -> tuple[str, str]:
    """``(system_prompt, serialised output_schema)`` as register.py PUTs them."""
    payload = _put_payload(worker)
    return payload["system_prompt"], _schema_text(payload["output_schema"])


def _pin_dir(worker: str, root: Path | None = None) -> Path:
    return (root if root is not None else _FIXTURES_ROOT) / worker


def _prompt_fixture(worker: str, root: Path | None = None) -> Path:
    return _pin_dir(worker, root) / "system_prompt.txt"


def _schema_fixture(worker: str, root: Path | None = None) -> Path:
    return _pin_dir(worker, root) / "output_schema.json"


def _fixture_texts(worker: str, root: Path | None = None) -> tuple[str, str]:
    return (
        _prompt_fixture(worker, root).read_text(encoding="utf-8"),
        _schema_fixture(worker, root).read_text(encoding="utf-8"),
    )


def _manifest(reader: Callable[[str], tuple[str, str]], workers: Sequence[str]) -> str:
    r"""The text whose sha256 is ``_PINNED_MANIFEST_SHA256``.

    Names every covered worker and the length + sha256 of each of its two decoder-visible
    artefacts. Sorted, so the manifest is a function of the SET of pinned workers and their
    bytes. Never stored on disk -- it is recomputed here and printed on failure, which is
    why it costs no additional high-entropy literal in this file.

    ⚠️ EVERY interpolated field goes through ``json.dumps``, and that is not cosmetic --
    it is the fix for ATTACK 6, the digest-of-digests forgery. The first cut of this
    function interpolated the worker NAME raw. Because the row separator is ``\n`` and the
    field separator is a space, a name is then free to contain both, so ONE entry could
    impersonate SEVERAL: an auditor built a name of the form

        "improvement_workflow_redesign\n  system_prompt <len> <sha>\n  output_schema
         <len> <sha>\nimprovement_workflow_statistics_analysis"

    whose two forged rows were the RETIRED literals for the redesign worker and whose own
    computed rows belonged to the statistics worker. That name reproduced the pinned
    manifest text BYTE FOR BYTE while `improvement_workflow_redesign` was no longer read
    from `build_payload` at all -- so its prompt could then be blessed freely and
    `_manifest_violation` stayed green (measured: returned ``None``). Only
    `_pinned_set_violation` caught it, and a pin whose set check is its single barrier is
    the vacuity this section exists to avoid.

    ``json.dumps`` escapes ``\n`` and ``"`` inside a string, so a name occupies exactly one
    line and terminates unambiguously; an artefact row always starts with two spaces and a
    fixed keyword, so it can never be mistaken for a name row. The length is an int and was
    already unforgeable, but it is routed through the same encoder so that NO field is an
    exception a later editor has to re-derive. Regression: ATTACK 6 below.
    """
    rows: list[str] = []
    for worker in sorted(workers):
        prompt, schema_text = reader(worker)
        rows.append(json.dumps(worker))
        rows.append(f"  system_prompt {json.dumps(len(prompt))} {json.dumps(_sha256(prompt))}")
        rows.append(
            f"  output_schema {json.dumps(len(schema_text))} {json.dumps(_sha256(schema_text))}"
        )
    return "\n".join(rows) + "\n"


def _prompt_diff_units(text: str) -> list[str]:
    """Sentence-per-line view of a prompt. The pieces concatenate back to the input."""
    return re.split(r"(?<=\. )", text)


def _schema_diff_units(text: str) -> list[str]:
    return text.splitlines()


def _pin_violation(
    *,
    worker: str,
    what: str,
    actual: str,
    fixture: Path,
    units: Callable[[str], list[str]],
) -> str | None:
    """Return an actionable failure message when ``actual`` is not the pinned bytes.

    The reference is the fixture FILE, so the message carries a unified diff naming the
    offending sentence. The fixture is not the whole pin: ``_PINNED_MANIFEST_SHA256`` is
    asserted over the fixture set too, so regenerating a fixture moves the failure to the
    manifest test rather than making it disappear.
    """
    if not fixture.is_file():
        return (
            f"pin fixture missing for {worker}: {fixture}\n"
            f"Regenerate it with:\n    {_REGEN_COMMAND}\n"
            f"A missing fixture is not a green pin -- it is an UNPINNED worker."
        )
    reference = fixture.read_text(encoding="utf-8")
    if actual == reference:
        return None
    diff = "\n".join(
        difflib.unified_diff(
            units(reference),
            units(actual),
            fromfile=f"pinned {fixture.name}",
            tofile=what,
            lineterm="",
        )
    )
    return (
        f"The {what} for {worker} CHANGED.\n"
        f"  pinned sha256:    {_sha256(reference)}\n"
        f"  assembled sha256: {_sha256(actual)}\n"
        f"\n"
        f"This test makes NO judgement about the change -- it only stops one landing "
        f"unread. It exists because three vocabulary-based guards before it were each "
        f"walked around within minutes by a blessing they had no word for, with the suite "
        f"green, and because the FIRST version of this pin covered one worker of five and "
        f"was walked around by moving the blessing one stage upstream (attacks 1-5 are "
        f"executable fixtures in this file).\n"
        f"\n"
        f"WHAT TO DO: read the diff below.\n"
        f"  * If the change is NOT deliberate, revert it.\n"
        f"  * If it IS deliberate, re-pin:\n"
        f"\n{_REPIN_STEPS}\n"
        f"\n"
        f"--- unified diff (pinned -> assembled) ---\n{diff}"
    )


def _prompt_pin_violation(prompt: str, worker: str = _DEV_WORKER) -> str | None:
    return _pin_violation(
        worker=worker,
        what="assembled system_prompt",
        actual=prompt,
        fixture=_prompt_fixture(worker),
        units=_prompt_diff_units,
    )


def _schema_pin_violation(schema: dict[str, Any], worker: str = _DEV_WORKER) -> str | None:
    return _pin_violation(
        worker=worker,
        what="registered output_schema",
        actual=_schema_text(schema),
        fixture=_schema_fixture(worker),
        units=_schema_diff_units,
    )


def _manifest_violation(source: str, manifest: str) -> str | None:
    """Return an actionable message when ``manifest`` is not the pinned one."""
    actual = _sha256(manifest)
    if actual == _PINNED_MANIFEST_SHA256:
        return None
    return (
        f"The pinned manifest computed from the {source} CHANGED.\n"
        f"  pinned sha256:   {_PINNED_MANIFEST_SHA256}\n"
        f"  computed sha256: {actual}\n"
        f"\n"
        f"The manifest covers every worker in _PINNED_WORKERS: its name, and the length and "
        f"sha256 of its assembled system_prompt and serialised output_schema. One of those "
        f"moved, or the covered SET moved.\n"
        f"\n"
        f"WHAT TO DO: the per-worker tests above print a unified diff naming the sentence "
        f"that changed -- read that first. If this test is the only one red, the fixtures "
        f"were regenerated without re-pinning: that is the pin working, not a stale test.\n"
        f"\n{_REPIN_STEPS}\n"
        f"\n"
        f"--- computed manifest ---\n{manifest}"
    )


def _pinned_set_violation(pinned: Sequence[str]) -> str | None:
    """Return a message when ``pinned`` is not exactly the set of workers register.py SEEDS.

    THE VACUITY CLOSER. A pin that silently covers a subset reads as coverage; that is
    precisely the defect that let attacks 5 and 7 land. Taken as a function of ``pinned`` so
    the check itself can be shown to fail
    (test_the_pinned_set_check_reds_on_a_short_or_long_set).

    ⚠️ THE REFERENCE IS ``register.AGENT_CONFIGS``, THE MERGED REGISTRY -- NOT
    ``workflow_loop.AGENT_CONFIGS``. This scope error has now been found THREE times on this
    branch, each time one level short of the thing that ships: the pin covered 1 worker of
    5, then 5 of 15. `register.py` seeds all fifteen into the same WM instance from the same
    loop, and every one of them reaches a live LLM, so the subset is never the right
    denominator. Reading the MERGED registry closes the class rather than the instance: a
    worker added to ANY loop module -- existing or not yet written -- reds here, because
    `register.AGENT_CONFIGS` is by construction everything `register.py` PUTs.
    """
    if len(set(pinned)) != len(pinned):
        return f"_PINNED_WORKERS contains a duplicate entry: {list(pinned)}"
    shipped = set(REGISTER_AGENT_CONFIGS)
    missing = sorted(shipped - set(pinned))
    extra = sorted(set(pinned) - shipped)
    if not missing and not extra:
        return None
    return (
        f"The byte pin does not cover exactly the workers register.py seeds.\n"
        f"  seeded but NOT pinned: {missing}\n"
        f"  pinned but NOT seeded: {extra}\n"
        f"\n"
        f"An unpinned worker is an unguarded prompt AND an unguarded output_schema, both of "
        f"which reach a live LLM. That is not hypothetical, and it is not a one-off: the "
        f"first version of this pin covered improvement_workflow_development alone, and a "
        f"decline blessing added to improvement_workflow_redesign -- the worker that FEEDS "
        f"it -- passed the whole suite; the second covered workflow_loop's five of the "
        f"fifteen register.py actually seeds, and a decline blessing in "
        f"improvement_prompt_improvement passed the whole suite too. The reference here is "
        f"the MERGED register.AGENT_CONFIGS precisely so a worker added in ANY loop module "
        f"reds instead of shipping uncovered.\n"
        f"\n"
        f"ADDING A WORKER REQUIRES PINNING IT. Add it to _PINNED_WORKERS, run\n"
        f"    {_REGEN_COMMAND}\n"
        f"and paste the new _PINNED_MANIFEST_SHA256. Do NOT delete a name from "
        f"_PINNED_WORKERS to make this green: that is not a repair, it is the un-pinning "
        f"this check exists to stop, and it reds the manifest instead."
    )


def regenerate_pin_fixtures(root: Path | None = None) -> None:
    """Rewrite the pinned fixture files from the CURRENT payloads and print the new pin.

    Deliberately does NOT edit this file: ``_PINNED_MANIFEST_SHA256`` IS the pin, and a
    human pastes it in having read the diff. Proven, not asserted in prose, by
    test_regenerating_the_fixtures_cannot_touch_the_pin. Invoked by hand via
    ``_REGEN_COMMAND``; ``root`` exists only so that test can redirect the writes.
    """
    for worker in _PINNED_WORKERS:
        prompt, schema_text = _built_texts(worker)
        _pin_dir(worker, root).mkdir(parents=True, exist_ok=True)
        _prompt_fixture(worker, root).write_text(prompt, encoding="utf-8")
        _schema_fixture(worker, root).write_text(schema_text, encoding="utf-8")
    manifest = _manifest(_built_texts, _PINNED_WORKERS)
    print(manifest, end="")
    print(f'_PINNED_MANIFEST_SHA256 = "{_sha256(manifest)}"')
    print()
    print(_REPIN_STEPS)


def _attack_token_avoiding_blessing(prompt: str) -> str:
    """ATTACK 3, verbatim: a blessing naming none of the `_DECLINE_VOCAB` terms."""
    anchor = " Reply with a single JSON object"
    assert prompt.count(anchor) == 1, "attack fixture no longer targets the prompt"
    return prompt.replace(anchor, _ATTACK_3_SENTENCE + anchor)


def _attack_schema_description_blessing(schema: dict[str, Any]) -> dict[str, Any]:
    """ATTACK 4, verbatim: the blessing moved into the schema's decoder-visible prose."""
    attacked = json.loads(json.dumps(schema))
    scope = attacked["properties"]["scope_of_change"]
    scope["description"] = scope["description"] + _ATTACK_4_SENTENCE
    return attacked


def _attack_redesign_prompt_blessing(prompt: str) -> str:
    """ATTACK 5, verbatim: the blessing moved one worker UPSTREAM of the development worker."""
    anchor = "(cost%, latency%, validation_delta) the patch should achieve."
    assert prompt.count(anchor) == 1, "attack fixture no longer targets the redesign prompt"
    return prompt.replace(anchor, anchor + _ATTACK_5_SENTENCE)


def _forged_manifest_name(dropped: str, successor: str) -> str:
    """ATTACK 6, verbatim: ONE worker name that impersonates a name row plus TWO artefact rows.

    The name carries `dropped`'s CURRENT rows as frozen literals and ends with `successor`'s
    name, so that when `_manifest` appends the rows it computes for this entry they land
    under `successor` instead. With raw interpolation the result was byte-identical to the
    honest manifest while `dropped` was never read from `build_payload` at all.
    """
    prompt, schema_text = _built_texts(dropped)
    return (
        f"{dropped}\n"
        f"  system_prompt {len(prompt)} {_sha256(prompt)}\n"
        f"  output_schema {len(schema_text)} {_sha256(schema_text)}\n"
        f"{successor}"
    )


def _bless_first_description(node: Any) -> bool:
    """Append `_ATTACK_4_SENTENCE` to the first `description` string found. True if one was.

    Walks in insertion order, so it is deterministic.
    """
    if isinstance(node, dict):
        value = node.get("description")
        if isinstance(value, str):
            node["description"] = value + _ATTACK_4_SENTENCE
            return True
        return any(_bless_first_description(child) for child in node.values())
    if isinstance(node, list):
        return any(_bless_first_description(child) for child in node)
    return False


def _attack_schema_prose(schema: dict[str, Any]) -> dict[str, Any]:
    """Attack 4 generalised to a schema that may carry no prose at all.

    ⚠️ RE-MEASURED 2026-09-13 over all FIFTEEN seeded schemas: only `workflow_development.json`
    has any `description` keys (5). The other FOURTEEN have ZERO, so for them the realistic
    blessing is an ADDED `description` on a property rather than an extended one -- `description` is not
    in `_XGRAMMAR_UNSUPPORTED_KEYS`, so a newly added one reaches the decoder exactly like
    an extended one would. Both shapes are schema byte changes and both must red.
    """
    attacked = json.loads(json.dumps(schema))
    if not _bless_first_description(attacked):
        first_property = next(iter(attacked["properties"].values()))
        first_property["description"] = _ATTACK_4_SENTENCE.strip()
    return attacked


def test_the_pinned_worker_set_is_exactly_the_seeded_registry() -> None:
    """The pin covers EVERY worker register.py seeds -- the assertion that de-vacuums it.

    The reference is the MERGED ``register.AGENT_CONFIGS``, not one loop module's. Adding a
    worker anywhere without pinning it reds here, by name, instead of shipping an unguarded
    prompt and schema to a live LLM.
    """
    violation = _pinned_set_violation(_PINNED_WORKERS)
    assert violation is None, violation
    # Belt and braces on the reference itself: if a future edit narrows the check back to one
    # loop module, this line is what says so out loud.
    assert set(_PINNED_WORKERS) == set(REGISTER_AGENT_CONFIGS), (
        "the pinned set is not the merged registry -- the scope error that has now been "
        "found three times on this branch"
    )


def test_the_pinned_set_check_reds_on_a_short_or_long_set() -> None:
    """The coverage assertion must itself be able to fail, in both directions."""
    for dropped in _PINNED_WORKERS:
        short = tuple(w for w in _PINNED_WORKERS if w != dropped)
        short_violation = _pinned_set_violation(short)
        assert short_violation is not None, (
            f"the coverage check PASSED a pinned set missing {dropped}"
        )
        assert dropped in short_violation, (
            f"the coverage failure must NAME {dropped}, the worker that is not pinned"
        )
    long = (*_PINNED_WORKERS, "improvement_workflow_does_not_exist")
    assert _pinned_set_violation(long) is not None, (
        "the coverage check PASSED a pinned set naming a worker that does not ship"
    )
    assert _pinned_set_violation((*_PINNED_WORKERS, _DEV_WORKER)) is not None, (
        "the coverage check PASSED a pinned set with a duplicate entry"
    )


def test_the_pinned_set_check_reds_on_a_worker_from_a_non_kaizen_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATTACK 7's closure, proven on a loop module that is NOT workflow_loop.

    The previous cut of `_pinned_set_violation` read `workflow_loop.AGENT_CONFIGS`, so a
    worker added to `prompt_loop` (or any of the other three modules) was invisible to it.
    The check now reads the merged registry, so a worker added in ANY module reds. Exercised
    here on a SIXTH prompt-loop worker -- the shape the old check could not see at all.
    """
    sixth = "improvement_prompt_a_worker_added_in_a_non_kaizen_loop"
    assert sixth not in REGISTER_AGENT_CONFIGS, "fixture name collided with a real worker"
    widened = {**REGISTER_AGENT_CONFIGS, sixth: REGISTER_AGENT_CONFIGS[_DEV_WORKER]}
    monkeypatch.setitem(globals(), "REGISTER_AGENT_CONFIGS", widened)

    violation = _pinned_set_violation(_PINNED_WORKERS)
    assert violation is not None, (
        "the coverage check PASSED a registry carrying an unpinned prompt_loop worker -- "
        "that is exactly the scope hole attack 7 walked through"
    )
    assert sixth in violation, "the coverage failure must NAME the unpinned worker"
    assert "ADDING A WORKER REQUIRES PINNING IT" in violation, (
        "the failure must say that the repair is to pin the worker"
    )
    assert "Do NOT delete a name from _PINNED_WORKERS" in violation, (
        "the failure must say that deleting a name to go green is the wrong repair"
    )


def test_dropping_a_worker_from_the_pinned_set_reds_the_manifest() -> None:
    """The set assertion is not the ONLY thing between a worker and no coverage.

    Deleting a name from `_PINNED_WORKERS` to silence the coverage test also changes the
    manifest, so that route reds too -- in a different test, with a different message.
    """
    for dropped in _PINNED_WORKERS:
        short = tuple(w for w in _PINNED_WORKERS if w != dropped)
        violation = _manifest_violation("built payloads", _manifest(_built_texts, short))
        assert violation is not None, (
            f"the manifest PASSED a pinned set with {dropped} removed -- dropping a worker "
            f"from _PINNED_WORKERS would silently un-pin it"
        )


def test_the_manifest_reds_on_a_worker_name_that_forges_extra_rows() -> None:
    """ATTACK 6: a plain drop is not the only way to leave a worker uncovered.

    ⚠️ THIS TEST IS NOT `test_dropping_a_worker_from_the_pinned_set_reds_the_manifest` WITH
    A DIFFERENT NAME, and the difference is the whole point. That test removes a name; the
    manifest then has two fewer rows and obviously moves. This one removes a name and
    replaces it with a name ENGINEERED to re-emit exactly the rows that went missing -- the
    digest-of-digests forgery. Against raw interpolation it reproduced the pinned manifest
    text byte for byte and `_manifest_violation` returned `None`, while the dropped worker's
    prompt and schema were no longer read from `build_payload` at all and could be blessed
    freely. A regression that only re-checks plain drops leaves that hole wide open.

    The forgery is run against BOTH manifests a pinned worker appears in, and the assertion
    is on `_manifest_violation` itself, not on the set check -- `_pinned_set_violation` caught
    attack 6 already and would mask a regression here.
    """
    ordered = sorted(_PINNED_WORKERS)
    forged_any = False
    for dropped, successor in itertools.pairwise(ordered):
        forged_name = _forged_manifest_name(dropped, successor)

        def _reader(
            name: str, _successor: str = successor, _forged: str = forged_name
        ) -> tuple[str, str]:
            # The forged entry's OWN rows are the SUCCESSOR's; `dropped`'s rows come from the
            # frozen literals baked into the name, so `dropped` is never read from
            # `build_payload` -- which is what "dropped from coverage" means here.
            return _built_texts(_successor if name == _forged else name)

        forged_set = (
            *(w for w in _PINNED_WORKERS if w not in (dropped, successor)),
            forged_name,
        )
        forged_manifest = _manifest(_reader, forged_set)

        assert dropped not in forged_set, "fixture did not actually drop the worker"
        violation = _manifest_violation("built payloads", forged_manifest)
        assert violation is not None, (
            f"the manifest PASSED a forged worker name that impersonates the rows of "
            f"{dropped}. An unescaped row/field separator lets ONE entry stand in for "
            f"SEVERAL: {dropped} is no longer read from build_payload, so its prompt and "
            f"schema are unpinned while this test is green. Encode every interpolated field."
        )
        forged_any = True

    assert forged_any, "the forgery loop ran zero times -- this test cannot fail"

    # ...and the reason it reds is the ENCODING, not luck: the forged name must not be able
    # to span rows once encoded.
    spanning = _forged_manifest_name(ordered[0], ordered[1])
    assert "\n" in spanning, "fixture name no longer contains the row separator"
    assert "\n" not in json.dumps(spanning), (
        "json.dumps left a raw newline in an encoded worker name -- the forgery is live again"
    )
    assert json.dumps(spanning) not in _manifest(_built_texts, _PINNED_WORKERS), (
        "the encoded forged name appears in the honest manifest"
    )


@pytest.mark.parametrize("worker", _PINNED_WORKERS)
def test_every_pinned_system_prompt_is_byte_identical_to_its_fixture(worker: str) -> None:
    """The assembled system_prompt PUT by register.py is byte-identical to the pin."""
    violation = _prompt_pin_violation(_built_texts(worker)[0], worker)
    assert violation is None, violation


@pytest.mark.parametrize("worker", _PINNED_WORKERS)
def test_every_pinned_output_schema_is_byte_identical_to_its_fixture(worker: str) -> None:
    """The output_schema PUT by register.py is byte-identical to the pin.

    Closes the second channel for every worker: schema ``description`` strings reach the
    model and were guarded, on one worker only, by a five-phrase blacklist (ATTACK 4).
    """
    violation = _schema_pin_violation(_put_payload(worker)["output_schema"], worker)
    assert violation is None, violation


def test_the_built_payloads_match_the_pinned_manifest() -> None:
    """One sha over all thirty artefacts. This is the literal a human re-pins by hand."""
    violation = _manifest_violation("built payloads", _manifest(_built_texts, _PINNED_WORKERS))
    assert violation is None, violation


def test_the_pin_fixtures_match_the_pinned_manifest() -> None:
    """The fixtures are the readable mirror of the pin, and are NOT self-certifying.

    Reported separately from the built-payload test so "the fixtures were regenerated
    without re-pinning" never reads as "a prompt changed", and so regenerating a fixture
    can never move the pin.
    """
    for worker in _PINNED_WORKERS:
        for fixture in (_prompt_fixture(worker), _schema_fixture(worker)):
            assert fixture.is_file(), f"pin fixture missing: {fixture} -- run: {_REGEN_COMMAND}"
    violation = _manifest_violation("fixture files", _manifest(_fixture_texts, _PINNED_WORKERS))
    assert violation is None, violation


def test_regenerating_the_fixtures_cannot_touch_the_pin(tmp_path: Path) -> None:
    """`regenerate_pin_fixtures()` writes fixtures and PRINTS the pin; it never writes it.

    Executable, not prose: this file's own bytes are hashed either side of the call. If a
    future edit makes the regenerator self-re-pin, the human read of the diff stops being
    part of the loop and the pin becomes a rubber stamp.
    """
    me = Path(__file__).resolve()
    before = hashlib.sha256(me.read_bytes()).hexdigest()
    regenerate_pin_fixtures(root=tmp_path)
    assert hashlib.sha256(me.read_bytes()).hexdigest() == before, (
        "regenerate_pin_fixtures() MODIFIED this test file. The sha literal is the pin and "
        "a human pastes it -- a regenerator that re-pins itself is not a pin."
    )
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    expected = sorted(
        f"{w}/{name}"
        for w in _PINNED_WORKERS
        for name in ("system_prompt.txt", "output_schema.json")
    )
    assert written == expected, f"regenerator wrote {written}, expected {expected}"
    assert _manifest(lambda w: _fixture_texts(w, tmp_path), _PINNED_WORKERS) == _manifest(
        _built_texts, _PINNED_WORKERS
    ), "the regenerated fixtures do not reproduce the built payloads"


def test_the_repin_instructions_name_the_detect_secrets_step(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Re-pinning mints a hex literal that reds CI. Both artefacts a re-pinner reads say so.

    This is asserted rather than merely written because the first cut of this section listed
    THREE steps and never mentioned `.secrets.baseline`: it told the re-pinner they were
    done, then handed them a red `Security scan - detect-secrets baseline` whose obvious
    repair is `detect-secrets scan --baseline .secrets.baseline` -- the in-place scan that
    rewrites the audit results it is checking and has destroyed a 299-line results block in
    this fleet. Routing people to a destructive command is a worse defect than the churn.

    The two artefacts are (a) the failing test's message and (b) what the regenerator prints.
    """
    required = (
        "Hex High Entropy String",
        "Security scan - detect-secrets baseline",
        "/tmp/scan-copy.json",
        "NEVER run `detect-secrets scan --baseline .secrets.baseline`",
        "NEVER add `pragma: allowlist secret`",
    )
    pin_message = _pin_violation(
        worker=_DEV_WORKER,
        what="assembled system_prompt",
        actual="not the pinned bytes",
        fixture=_prompt_fixture(_DEV_WORKER),
        units=_prompt_diff_units,
    )
    assert pin_message is not None, "the pin failed to red on obviously wrong bytes"
    manifest_message = _manifest_violation("built payloads", "not the pinned manifest\n")
    assert manifest_message is not None, "the manifest failed to red on an obviously wrong input"
    regenerate_pin_fixtures(root=tmp_path)
    printed = capsys.readouterr().out
    for label, artefact in (
        ("the per-worker pin failure", pin_message),
        ("the manifest pin failure", manifest_message),
        ("regenerate_pin_fixtures() output", printed),
    ):
        for phrase in required:
            assert phrase in artefact, (
                f"{label} does not carry {phrase!r}. A re-pinner reads only these artefacts; "
                f"dropping the detect-secrets step sends them into a red gate whose obvious "
                f"repair is the banned in-place scan."
            )
        assert "ALL FOUR STEPS" in artefact, f"{label} no longer states that step 4 is required"


@pytest.mark.parametrize("worker", _PINNED_WORKERS)
def test_the_byte_pin_reds_on_a_blessing_in_every_pinned_prompt(worker: str) -> None:
    """ATTACK 7 generalised: a decline blessing in ANY seeded worker's prompt must red.

    Attacks 3 and 5 are worker-specific fixtures anchored to text only two prompts contain.
    This one is the property behind them, asserted on all fifteen: the blessing is appended
    to the prompt as shipped, which is the shape a helpful edit actually takes. Before this
    file read the MERGED registry, `improvement_prompt_improvement` -- the worker whose ENTIRE
    JOB is rewriting other workers' prompts -- was one of the ten that could take such an
    edit with the suite green.
    """
    sp = _put_payload(worker)["system_prompt"]
    attacked = sp + _ATTACK_5_SENTENCE
    assert attacked != sp, f"{worker}: attack fixture made no change to the prompt"
    violation = _prompt_pin_violation(attacked, worker)
    assert violation is not None, f"{worker}: the BYTE PIN passed a decline blessing"
    assert worker in violation, f"{worker}: the failure must name the worker whose prompt changed"
    assert "WHAT TO DO" in violation, f"{worker}: failure message is not actionable"


def test_the_byte_pin_reds_on_every_known_prompt_attack() -> None:
    """All three development-prompt attacks -- including the one the region guard misses."""
    sp = _put_payload()["system_prompt"]
    for label, attacked in (
        ("attack 1 -- laundered em-dash continuation", _attack_laundered_blessing(sp)),
        ("attack 2 -- unmarked blessing", _attack_unmarked_blessing(sp)),
        ("attack 3 -- token-avoiding blessing", _attack_token_avoiding_blessing(sp)),
    ):
        assert attacked != sp, f"{label}: fixture made no change to the prompt"
        violation = _prompt_pin_violation(attacked)
        assert violation is not None, f"{label}: the BYTE PIN passed an altered prompt"
        assert "WHAT TO DO" in violation, f"{label}: failure message is not actionable"


def test_the_byte_pin_reds_on_the_redesign_prompt_blessing() -> None:
    """ATTACK 5: the blessing moved to the worker that FEEDS the development worker.

    This is the attack that defeated the single-worker version of this pin, at 94 passed.
    Both halves are asserted: the per-worker pin names `improvement_workflow_redesign` in
    its failure, and the whole-set manifest moves too.
    """
    worker = "improvement_workflow_redesign"
    sp = _put_payload(worker)["system_prompt"]
    attacked = _attack_redesign_prompt_blessing(sp)
    assert attacked != sp, "attack 5 fixture made no change to the redesign prompt"
    violation = _prompt_pin_violation(attacked, worker)
    assert violation is not None, "the BYTE PIN passed a decline blessing in _REDESIGN_PROMPT"
    assert worker in violation, "the failure must name the worker whose prompt changed"
    assert "WHAT TO DO" in violation, "failure message is not actionable"

    # ...and the set-level pin sees it too, so deleting the per-worker test is not a way out.
    def _swapped(name: str) -> tuple[str, str]:
        prompt, schema_text = _built_texts(name)
        return (attacked if name == worker else prompt), schema_text

    assert (
        _manifest_violation("built payloads", _manifest(_swapped, _PINNED_WORKERS)) is not None
    ), "the manifest PASSED a blessed redesign prompt"
    # Honest record of WHY this was invisible before, and it is NOT that the region guard is
    # weak: every decline guard in this file -- the region guard, the vocabulary sweep, the
    # anti-fabrication sentence check, and the first version of the byte pin -- reads the
    # DEVELOPMENT worker's prompt, and attack 5 leaves that prompt byte-identical. The defect
    # was SCOPE, not strength, which is why the fix is _PINNED_WORKERS and not a wordlist.
    dev_prompt = _put_payload(_DEV_WORKER)["system_prompt"]
    assert _prompt_pin_violation(dev_prompt) is None, (
        "attack 5 is only interesting because it leaves the development prompt untouched"
    )
    scope_values = AGENT_CONFIGS[_DEV_WORKER].output_schema()["properties"]["scope_of_change"][
        "enum"
    ]
    assert not _decline_region_violations(dev_prompt, _PINNED_DECLINE_POLICY, scope_values), (
        "the development-worker guards are green while a decline blessing sits in the "
        "prompt of the worker that feeds it -- that is exactly the hole _PINNED_WORKERS "
        "closes, and it must stay demonstrable"
    )


def test_the_byte_pin_reds_on_the_schema_description_attack() -> None:
    """ATTACK 4 reds on the pin -- and is recorded as invisible to the phrase blacklist."""
    schema = _put_payload()["output_schema"]
    attacked = _attack_schema_description_blessing(schema)
    assert attacked != schema, "attack fixture made no change to the schema"
    violation = _schema_pin_violation(attacked)
    assert violation is not None, "the BYTE PIN passed a blessed schema description"
    assert "WHAT TO DO" in violation, "failure message is not actionable"
    # Honest record of the hole the pin closes: the pre-existing five-phrase blacklist in
    # test_dev_schema_descriptions_are_descriptive_not_hortatory does NOT see this.
    blob = json.dumps(attacked).lower()
    for phrase in (
        "legitimate answer",
        "honest, expected outcome",
        "deliberate decline",
        "first-class outcome",
        "healthy terminal state",
    ):
        assert phrase not in blob, (
            "attack 4 now trips the phrase blacklist. If you added a phrase to make that "
            "happen, revert it -- the blacklist is the class of guard that has been "
            "defeated three times; the pin is what holds."
        )


@pytest.mark.parametrize("worker", _PINNED_WORKERS)
def test_the_byte_pin_reds_on_a_blessing_in_every_pinned_schema(worker: str) -> None:
    """Attack 4 generalised: every worker's schema prose is a channel, so every one is pinned.

    Only the development schema has `description` keys today (see `_attack_schema_prose`),
    so for the other fourteen the blessing is an ADDED description -- which is how prose would
    actually get into them. Either way it is a byte change in the registered schema, and
    either way it must red.
    """
    schema = _put_payload(worker)["output_schema"]
    attacked = _attack_schema_prose(schema)
    assert attacked != schema, f"{worker}: attack fixture made no change to the schema"
    violation = _schema_pin_violation(attacked, worker)
    assert violation is not None, f"{worker}: the BYTE PIN passed a blessed schema description"
    assert worker in violation, f"{worker}: the failure must name the worker whose schema changed"


def test_the_region_guard_does_NOT_catch_the_token_avoiding_blessing() -> None:
    """Recorded, not fixed: attack 3 passes `_decline_region_violations`.

    `if not hits: continue` skips any sentence carrying none of the 19 `_DECLINE_VOCAB`
    terms, so the region guard cannot see this blessing ANYWHERE outside the region. The
    honest current behaviour is asserted here so the region guard's limit is executable
    rather than prose, and so nobody "fixes" it by widening the wordlist -- that is the
    move that has now failed three times. The byte pin is what catches attack 3.
    """
    cfg = AGENT_CONFIGS[_DEV_WORKER]
    sp = cfg.system_prompt
    scope_values = cfg.output_schema()["properties"]["scope_of_change"]["enum"]
    attacked = _attack_token_avoiding_blessing(sp)
    assert attacked != sp, "attack fixture made no change to the prompt"
    assert not _decline_region_violations(attacked, _PINNED_DECLINE_POLICY, scope_values), (
        "the region guard now catches attack 3. If you widened _DECLINE_VOCAB to make that "
        "happen, revert it: a wordlist cannot be complete and this is the fourth attempt. "
        "If it is incidental to a legitimate change, update this fixture and say why."
    )
    assert _prompt_pin_violation(attacked) is not None, (
        "attack 3 escapes BOTH guards -- the byte pin must catch what the region guard cannot"
    )


def test_the_byte_pin_is_not_vacuous() -> None:
    """The pin must red on an empty build and on another worker's prompt/schema.

    A pin that cannot fail is worse than no pin: it reads as coverage. Both directions are
    exercised, for every pinned worker -- degenerate input, and a plausible-but-wrong build
    taken from a SIBLING worker. The sibling half is the one that would have caught the
    single-worker pin's real defect.
    """
    for worker in _PINNED_WORKERS:
        assert _prompt_pin_violation("", worker) is not None, (
            f"{worker}: pin passed an EMPTY prompt"
        )
        assert _schema_pin_violation({}, worker) is not None, (
            f"{worker}: pin passed an EMPTY schema"
        )
        for other in _PINNED_WORKERS:
            if other == worker:
                continue
            other_prompt, _ = _built_texts(other)
            assert _prompt_pin_violation(other_prompt, worker) is not None, (
                f"the {worker} pin passed {other}'s prompt -- it is not pinning {worker}"
            )
            assert (
                _schema_pin_violation(_put_payload(other)["output_schema"], worker) is not None
            ), f"the {worker} pin passed {other}'s schema -- it is not pinning {worker}"


def test_dev_prompt_states_the_anti_fabrication_consequence() -> None:
    """The prompt must carry the anti-fabrication CONSEQUENCE, once, as its own sentence.

    "Do not invent an operation" is an instruction; "a fabricated change is promoted to a
    live canary, an honest decline is not" is the reason, and the reason is what makes it
    persuasive -- it is the only place the prompt tells the model that fabricating has a
    cost the decline does not. dd5e436 had it; d314d52 dropped the consequence clause and
    kept only a tail clause hanging off a sentence whose main clause blesses declining,
    which is the weakest possible placement and landed exactly as a strong pro-proposal
    counterweight was added. FAILS on d314d52.

    Exactly-once matters twice over: it keeps the prompt from re-stacking, and -- since the
    sentence lives inside the pinned decline region -- it is what makes an em-dash
    continuation of it a byte-level change the region guard cannot miss.
    """
    sp = AGENT_CONFIGS[_DEV_WORKER].system_prompt
    hits = [s for s in _sentences(sp) if _ANTI_FABRICATION_PHRASE in s.lower()]
    assert len(hits) == 1, (
        f"expected exactly 1 anti-fabrication sentence, found {len(hits)}: {hits}"
    )
    sentence = hits[0].lower()
    assert "promoted to a live canary" in sentence, (
        f"anti-fabrication sentence must state the CONSEQUENCE, not just the ban: {hits[0]!r}"
    )
    assert "an honest decline is not" in sentence, (
        f"the consequence must contrast fabricating WITH declining: {hits[0]!r}"
    )
    # It must stand alone: the ban is not a subordinate clause of the decline instruction.
    # Structural, not lexical -- the region is exactly two sentences and this is the second.
    region = _sentences(workflow_loop._DECLINE_POLICY)
    assert len(region) == 2, f"decline region must be exactly 2 sentences, got {len(region)}"
    assert region[1] == hits[0], (
        "the anti-fabrication sentence must be the second sentence of the decline region, "
        "not a clause folded into the sentence that instructs the decline"
    )
    assert _ANTI_FABRICATION_PHRASE not in region[0].lower(), (
        "the decline instruction and the anti-fabrication consequence must not be one sentence"
    )


def test_dev_prompt_defect_class_counterweight_is_not_gated_on_telemetry() -> None:
    """The defect classes are stated unconditionally; only the confinement is gated.

    Both live declines came from the branch where telemetry names no failing node.
    If the defect-class list is gated on "when the evidence does implicate a node",
    that branch gets the bare "justify it without inventing evidence" sentence and no
    counterweight at all -- which is the shape that produced the declines. So: the
    defect-class sentence must carry NO conditional, the confinement sentence must
    keep one, and the order must be gate -> unconditional counterweight -> the
    no-invention clause. FAILS on dd5e436 (no defect-class sentence exists).
    """
    sp = AGENT_CONFIGS[_DEV_WORKER].system_prompt
    defect = _sentence_with(sp, "an external call with no retry")
    assert "whenever you read the workflow" in defect.lower(), (
        "the defect-class list must be introduced unconditionally"
    )
    assert not re.search(r"\bif\b", defect, re.IGNORECASE), (
        f"defect-class sentence is conditional: {defect!r}"
    )
    for gate in ("when the evidence", "only if", "names specific failing"):
        assert gate not in defect.lower(), f"defect-class sentence is gated on {gate!r}: {defect!r}"

    # ...while "confine the patch" MUST stay gated on telemetry naming the nodes.
    confine = _sentence_with(sp, "confine the patch")
    assert "only if" in confine.lower() and "telemetry" in confine.lower(), (
        f"the confinement must remain gated on telemetry: {confine!r}"
    )

    # The no-invention clause is kept verbatim and must come AFTER the counterweight.
    no_invent = _sentence_with(sp, "inventing evidence")
    sentences = _sentences(sp)
    assert sentences.index(confine) < sentences.index(defect) < sentences.index(no_invent), (
        "order must be: gated confinement -> ungated defect classes -> no-invention clause"
    )


def test_dev_prompt_defect_classes_are_all_settled_by_a_key_in_the_workflow_json() -> None:
    """Every named defect class must be decidable off the workflow JSON -- no judgement calls.

    The counterweight NARROWS the model's licence only because each class is settled by a
    key that is already in the node's own definition: ``retryOnFail``, ``timeout``,
    ``onError``. A class with no such key -- the removed "a hardcoded value that belongs in
    a parameter" -- is pure judgement, and a model under pressure to propose something can
    satisfy it on ANY workflow, which turns the counterweight into a licence to invent (it
    carried a "3.8% of 754 live workflow backups" figure here that does NOT reproduce under
    any predicate tried -- it is deleted rather than restated, see workflow_loop.py). Do not
    re-add it, or any other class whose test is an opinion.
    """
    sp = AGENT_CONFIGS[_DEV_WORKER].system_prompt
    defect = _sentence_with(sp, "an external call with no retry")
    for phrase in ("no retry", "timeout", "continue on failure"):
        assert phrase in defect.lower(), f"defect-class list lost {phrase!r}: {defect!r}"
    for key in ("retryOnFail", "timeout", "onError"):
        assert key in sp, f"prompt must name the JSON key {key!r} that settles a defect class"
    assert "hardcoded" not in sp.lower(), (
        "the 'hardcoded value that belongs in a parameter' class has no objective test and "
        "was deliberately removed -- see the comment above _DEV_PROMPT before re-adding it"
    )


def test_dev_schema_descriptions_are_descriptive_not_hortatory() -> None:
    """Schema ``description`` strings reach the model, so they must not re-bless declining.

    ``description`` is NOT in ``_XGRAMMAR_UNSUPPORTED_KEYS``, so it survives into the
    decoder-visible schema and is read alongside the system prompt. Three descriptions
    used to argue FOR the decline ("a legitimate answer", "an honest, expected
    outcome", "a deliberate decline"), silently re-stacking what the prompt rebalanced.
    FAILS against the dd5e436 schema.
    """
    from src.worker.agents.pydantic_ai_agent import _decoder_visible_schema

    schema = AGENT_CONFIGS[_DEV_WORKER].output_schema()
    visible = _decoder_visible_schema(schema)
    assert visible is not None
    # The mechanism: descriptions are NOT stripped, hence they must be descriptive.
    assert "description" in visible["properties"]["scope_of_change"], (
        "descriptions no longer reach the model -- this test's premise is gone"
    )
    blob = json.dumps(schema).lower()
    for phrase in (
        "legitimate answer",
        "honest, expected outcome",
        "deliberate decline",
        "first-class outcome",
        "healthy terminal state",
    ):
        assert phrase not in blob, (
            f"schema description argues FOR declining ({phrase!r}); keep descriptions "
            "descriptive, not hortatory"
        )
    # They must still SAY what the field means.
    assert "no_change" in schema["properties"]["scope_of_change"]["description"]
    assert "empty" in schema["properties"]["workflow_patch"]["description"].lower()


def test_each_config_loads_its_output_schema_and_each_prompt_is_substantial() -> None:
    """Loading the schema must succeed and the placeholder prompt must be >= 3 sentences."""
    for cfg in AGENT_CONFIGS.values():
        schema = cfg.output_schema()
        jsonschema.Draft202012Validator.check_schema(schema)
        # Loose sentence count: split on ". " and require at least 3 chunks.
        assert cfg.system_prompt.count(". ") >= 2, (
            f"{cfg.worker_name} system_prompt should have >= 3 sentences"
        )
        assert cfg.temperature == pytest.approx(0.2)
        assert cfg.model_provider == "inference_router"
        assert cfg.model == "local-llm"


# ---------------------------------------------------------------------------
# payload shape
# ---------------------------------------------------------------------------


def test_build_payload_has_expected_fields() -> None:
    cfg = AGENT_CONFIGS["improvement_workflow_redesign"]
    payload = build_payload(cfg, base_url_override=ROUTER_URL)
    assert payload["worker_name"] == "improvement_workflow_redesign"
    assert payload["model_provider"] == "inference_router"
    assert payload["model"] == "local-llm"
    assert payload["temperature"] == pytest.approx(0.2)
    assert payload["base_url_override"] == ROUTER_URL
    assert isinstance(payload["output_schema"], dict)
    assert payload["output_schema"]["title"] == "WorkflowRedesignProposal"
    assert payload["agent_type"] == "pydantic_ai"
    assert "system_prompt" in payload
    assert payload["system_prompt"].strip() != ""


# ---------------------------------------------------------------------------
# dry-run: NO POST/PUT issued
# ---------------------------------------------------------------------------


def test_dry_run_issues_only_GETs_and_marks_would_create() -> None:
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        # Pretend nothing exists yet.
        return httpx.Response(404, json={"detail": "not found"})

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    methods = {m for m, _ in seen}
    assert methods == {"GET"}, f"dry-run must not POST/PUT; saw {seen}"
    assert len(seen) == len(AGENT_CONFIGS)
    assert set(actions.values()) == {"would-create"}
    assert set(actions.keys()) == set(AGENT_CONFIGS.keys())


def test_dry_run_with_existing_and_force_marks_would_update() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(200, json={"worker_name": req.url.path.rsplit("/", 1)[-1]})
        raise AssertionError(f"dry-run must not call {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=True,
            client=client,
            configs=AGENT_CONFIGS,
        )
    assert set(actions.values()) == {"would-update"}


# ---------------------------------------------------------------------------
# real POST flow
# ---------------------------------------------------------------------------


def test_post_payload_is_sent_with_tenant_headers() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            captured.append(
                {
                    "url": str(req.url),
                    "headers": dict(req.headers),
                    "body": json.loads(req.content.decode("utf-8")),
                }
            )
            return httpx.Response(201, json={"worker_name": "ok"})
        raise AssertionError(f"unexpected method {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="tok-abc",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(actions.values()) == {"created"}
    assert len(captured) == len(AGENT_CONFIGS)

    sent_names = {c["body"]["worker_name"] for c in captured}
    assert sent_names == set(AGENT_CONFIGS.keys())

    for c in captured:
        assert c["headers"]["x-tenant-id"] == "Bearer tok-abc"
        assert c["headers"]["x-internal-tenant-id"] == SYSTEM_WORKFLOW_TENANT_ID
        body = c["body"]
        assert body["base_url_override"] == ROUTER_URL
        assert body["model_provider"] == "inference_router"
        assert body["model"] == "local-llm"
        assert body["temperature"] == pytest.approx(0.2)
        assert isinstance(body["output_schema"], dict)
        assert body["output_schema"]["type"] == "object"


def test_existing_worker_without_force_is_skipped() -> None:
    """A HEALTHY existing worker is skipped without --force.

    ⚠️ The stub body used to be ``{"worker_name": "x"}`` -- no ``system_prompt``
    at all -- and this test asserted that WM reported ``skipped-exists`` for it.
    That is the 2026-08-27 defect written down as the spec: a worker registered
    with a NULL prompt is precisely the row that ran the live propose loop
    prompt-less for 17 days, and "skip" is the word that hid it. The stub now
    carries a prompt, so this test still covers the skip path for a worker that
    is actually usable; the prompt-less case is
    ``test_existing_worker_with_null_prompt_is_a_failure_not_a_skip`` below,
    which is a FAILURE and not a skip.
    """
    seen_methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_methods.append(req.method)
        if req.method == "GET":
            return httpx.Response(200, json={"worker_name": "x", "system_prompt": "LIVE PROMPT"})
        raise AssertionError(f"must not call {req.method} without --force")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )
    assert set(actions.values()) == {"skipped-exists"}
    assert set(seen_methods) == {"GET"}


def test_existing_worker_with_force_is_PUT() -> None:
    methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        methods.append(req.method)
        if req.method == "GET":
            return httpx.Response(200, json={"worker_name": "x"})
        if req.method == "PUT":
            return httpx.Response(200, json={"worker_name": "x"})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )
    assert set(actions.values()) == {"updated"}
    assert "PUT" in methods


def test_unexpected_get_status_is_raised_loudly() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "kaboom"})

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )


# ---------------------------------------------------------------------------
# --force-prompt: GET-then-merge protection for live system_prompt
# ---------------------------------------------------------------------------


def _live_response_for(name: str, *, system_prompt: str | None) -> dict[str, Any]:
    """Mimic a worker_management GET /workers/{name} response body."""
    body: dict[str, Any] = {
        "worker_name": name,
        "agent_role": f"workflow improvement loop -- {name}",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "default",
        "temperature": 0.2,
        "description": "live description",
        # Live worker has a stale/different output_schema; seed must win.
        "output_schema": {"type": "object", "title": "STALE_LIVE_SCHEMA"},
    }
    if system_prompt is not None:
        body["system_prompt"] = system_prompt
    return body


def test_force_without_force_prompt_preserves_live_system_prompt() -> None:
    """--force without --force-prompt: PUT body keeps the live system_prompt."""
    captured_puts: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(200, json=_live_response_for(name, system_prompt="LIVE PROMPT"))
        if req.method == "PUT":
            captured_puts.append(
                {
                    "url": str(req.url),
                    "body": json.loads(req.content.decode("utf-8")),
                }
            )
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            force_prompt=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(actions.values()) == {"updated-preserved-prompt"}
    assert len(captured_puts) == len(AGENT_CONFIGS)
    for c in captured_puts:
        assert c["body"]["system_prompt"] == "LIVE PROMPT", (
            "live system_prompt must be preserved across PUT"
        )
        # Schema is seed-authoritative even when prompt is preserved.
        assert c["body"]["output_schema"]["title"] != "STALE_LIVE_SCHEMA"


def test_force_with_force_prompt_overwrites_system_prompt_from_seed() -> None:
    """--force --force-prompt: PUT body uses the seed system_prompt."""
    captured_puts: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(200, json=_live_response_for(name, system_prompt="LIVE PROMPT"))
        if req.method == "PUT":
            captured_puts.append({"body": json.loads(req.content.decode("utf-8"))})
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            force_prompt=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(actions.values()) == {"updated"}
    assert len(captured_puts) == len(AGENT_CONFIGS)
    seed_prompts = {cfg.system_prompt for cfg in AGENT_CONFIGS.values()}
    for c in captured_puts:
        assert c["body"]["system_prompt"] != "LIVE PROMPT", (
            "with --force-prompt the live prompt must NOT win"
        )
        assert c["body"]["system_prompt"] in seed_prompts


def test_force_without_force_prompt_falls_back_to_seed_when_live_prompt_missing() -> None:
    """If live worker has empty/missing system_prompt, seed prompt must be used."""
    captured_puts: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            # Half of the workers have empty string, half are missing the field
            # entirely -- both must fall back to seed.
            if name.endswith(("redesign", "development")):
                return httpx.Response(200, json=_live_response_for(name, system_prompt=None))
            return httpx.Response(200, json=_live_response_for(name, system_prompt=""))
        if req.method == "PUT":
            captured_puts.append({"body": json.loads(req.content.decode("utf-8"))})
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            force_prompt=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    # When live prompt is empty/missing, we treat that as "no operator edit"
    # and write the seed prompt -- so the action is the plain "updated" label.
    assert set(actions.values()) == {"updated"}
    seed_prompts = {cfg.system_prompt for cfg in AGENT_CONFIGS.values()}
    for c in captured_puts:
        prompt = c["body"]["system_prompt"]
        assert prompt in seed_prompts, "must fall back to seed prompt"
        assert prompt.strip() != ""


def test_dry_run_force_without_force_prompt_reports_preserve_label() -> None:
    """--dry-run --force without --force-prompt: no PUT, label says preserve."""
    seen_methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_methods.append(req.method)
        if req.method == "GET":
            name = req.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=_live_response_for(name, system_prompt="LIVE PROMPT"))
        raise AssertionError(f"dry-run must not call {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=True,
            force_prompt=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(seen_methods) == {"GET"}
    assert set(actions.values()) == {"would-update-preserve-prompt"}


def test_dry_run_force_with_force_prompt_reports_overwrite_label() -> None:
    """--dry-run --force --force-prompt: no PUT, label says overwrite."""
    seen_methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_methods.append(req.method)
        if req.method == "GET":
            name = req.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=_live_response_for(name, system_prompt="LIVE PROMPT"))
        raise AssertionError(f"dry-run must not call {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=True,
            force_prompt=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(seen_methods) == {"GET"}
    assert set(actions.values()) == {"would-update"}


def test_force_without_force_prompt_uses_seed_output_schema_not_live() -> None:
    """Even when system_prompt is preserved, output_schema is seed-authoritative."""
    captured_puts: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            body = _live_response_for(name, system_prompt="LIVE PROMPT")
            # Force a clearly stale live schema.
            body["output_schema"] = {
                "type": "object",
                "title": "STALE_LIVE_SCHEMA",
                "properties": {"junk": {"type": "string"}},
            }
            return httpx.Response(200, json=body)
        if req.method == "PUT":
            captured_puts.append({"body": json.loads(req.content.decode("utf-8"))})
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            force_prompt=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert len(captured_puts) == len(AGENT_CONFIGS)
    for c in captured_puts:
        sent_schema = c["body"]["output_schema"]
        assert sent_schema.get("title") != "STALE_LIVE_SCHEMA", (
            "PUT must use seed output_schema, not the live (stale) one"
        )
        # Sanity: the seed schema is the one loaded from the schema file.
        assert sent_schema["type"] == "object"
        # And the preserved prompt is still there.
        assert c["body"]["system_prompt"] == "LIVE PROMPT"


# ===========================================================================
# Plan 6 -- prompt_loop coverage
# ===========================================================================


PROMPT_LOOP_NAMES = {
    "improvement_prompt_performance_statistics_analysis",
    "improvement_prompt_improvement",
    "improvement_prompt_library_consistency_management",
}

PROCEDURE_LOOP_NAMES = {
    "improvement_procedure_improvement",
}

ASSET_LOOP_NAMES = {
    "improvement_principle_improvement",
    "improvement_requirement_improvement",
    "improvement_decision_improvement",
}


def test_register_module_merges_every_loop_without_collision() -> None:
    """``register.AGENT_CONFIGS`` is the union of every loop module, losing nothing.

    ⚠️ THE LITERAL `== 15` THAT USED TO BE HERE IS DELIBERATELY GONE, and it was not merely
    redundant. `register.AGENT_CONFIGS` is a dict merge, so a name collision between two
    loops SILENTLY drops a worker; the count test existed to catch that. But once the byte
    pin covers the merged registry
    (test_the_pinned_worker_set_is_exactly_the_seeded_registry), a count literal is a TRAP:
    adding a sixteenth worker reds it, and its obvious repair is to type `16` -- which is
    exactly the "delete a name to go green" move the pin's failure message spends a
    paragraph forbidding, wearing a different hat. It is replaced by the property the count
    was standing in for: the merged size EQUALS the sum of the per-loop sizes, which is
    false the instant two loops collide and stays true no matter how many workers exist. It
    also reds if a SIXTH loop module is merged into register.py without being listed here,
    because the sum would then be short.
    """
    from scripts.improvement_workers.workers.asset_authoring import (
        AGENT_CONFIGS as ASSET_AUTHORING_CONFIGS,
    )

    per_loop = (
        AGENT_CONFIGS,
        PROMPT_LOOP_CONFIGS,
        ASSET_AUTHORING_CONFIGS,
        PROCEDURE_LOOP_CONFIGS,
        ASSET_LOOP_CONFIGS,
    )
    assert len(REGISTER_AGENT_CONFIGS) == sum(len(loop) for loop in per_loop), (
        "register.AGENT_CONFIGS is smaller than the loops it merges -- a name collision "
        "silently dropped a worker -- or larger, meaning a loop module was merged into "
        "register.py but is not listed in this test."
    )
    assert set(REGISTER_AGENT_CONFIGS.keys()) == (
        set(AGENT_CONFIGS.keys())
        | PROMPT_LOOP_NAMES
        | set(ASSET_AUTHORING_CONFIGS.keys())
        | PROCEDURE_LOOP_NAMES
        | ASSET_LOOP_NAMES
    )
    # No collision between any of the loops.
    assert set(AGENT_CONFIGS.keys()).isdisjoint(set(PROMPT_LOOP_CONFIGS.keys()))
    assert set(AGENT_CONFIGS.keys()).isdisjoint(set(ASSET_AUTHORING_CONFIGS.keys()))
    assert set(PROMPT_LOOP_CONFIGS.keys()).isdisjoint(set(ASSET_AUTHORING_CONFIGS.keys()))
    assert set(PROCEDURE_LOOP_CONFIGS.keys()).isdisjoint(
        set(REGISTER_AGENT_CONFIGS.keys()) - PROCEDURE_LOOP_NAMES
    )
    # asset_loop is disjoint from every other loop.
    assert set(ASSET_LOOP_CONFIGS.keys()) == ASSET_LOOP_NAMES
    assert set(ASSET_LOOP_CONFIGS.keys()).isdisjoint(
        set(REGISTER_AGENT_CONFIGS.keys()) - ASSET_LOOP_NAMES
    )


def test_prompt_loop_has_exactly_three_card_types() -> None:
    """Plan 6 ships exactly three card_types; the file set must match."""
    assert set(PROMPT_LOOP_CONFIGS.keys()) == PROMPT_LOOP_NAMES
    for cfg in PROMPT_LOOP_CONFIGS.values():
        assert (PROMPT_LOOP_SCHEMAS_DIR / cfg.schema_filename).is_file(), (
            f"missing schema file for {cfg.worker_name}"
        )


def test_prompt_loop_worker_names_are_validator_clean() -> None:
    """Worker names must match WM's [A-Za-z0-9_-]+ identifier validator (no periods)."""
    import re

    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    for name in PROMPT_LOOP_CONFIGS:
        assert pattern.match(name), f"{name} contains characters outside [A-Za-z0-9_-]"
        assert "." not in name, f"{name} contains a period (Plan 6 names are dot-free)"


@pytest.mark.parametrize(
    "schema_path",
    sorted(PROMPT_LOOP_SCHEMAS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_prompt_loop_schema_files_are_valid_jsonschema(schema_path: Path) -> None:
    """Every shipped prompt_loop schema must be valid JSON Schema (draft-07)."""
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.Draft7Validator.check_schema(schema)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["type"] == "object"
    expected_required = {
        "candidate_system_prompt",
        "rationale",
        "expected_impact",
        "scope_of_change",
    }
    assert set(schema["required"]) == expected_required
    assert schema["additionalProperties"] is False
    # scope_of_change enum is fixed for the loop.
    assert set(schema["properties"]["scope_of_change"]["enum"]) == {
        "minor_wording",
        "structural",
        "instruction_set",
    }


def test_prompt_loop_each_config_loads_its_output_schema() -> None:
    """Loading each prompt_loop schema must succeed and prompts must be substantial."""
    for cfg in PROMPT_LOOP_CONFIGS.values():
        schema = cfg.output_schema()
        jsonschema.Draft7Validator.check_schema(schema)
        # Loose sentence count: at least 3 sentences in the placeholder prompt.
        assert cfg.system_prompt.count(". ") >= 2, (
            f"{cfg.worker_name} system_prompt should have >= 3 sentences"
        )
        assert cfg.model_provider == "inference_router"
        assert cfg.model == "local-llm"
        # Capped at 4000 (fleet-wide, matches karma) so vLLM-on-PGX
        # (--max-model-len=8192) doesn't 400-then-fall-through to deepseek.
        assert cfg.max_tokens == 4000


def test_prompt_loop_system_prompts_name_the_output_contract() -> None:
    """Every prompt_loop worker shares the PromptImprovementProposal schema
    (required key ``candidate_system_prompt``). Its system_prompt MUST name that
    output key explicitly and forbid the ``revised_prompt`` alias -- otherwise the
    model emits the revision under ``revised_prompt``, the downstream
    prompt-canary ``Parse Proposal`` skips it, and the loop never compounds
    (the 2026-06-16 computational-count-stuck-at-1 bug)."""
    for cfg in PROMPT_LOOP_CONFIGS.values():
        required = set(cfg.output_schema().get("required", []))
        assert "candidate_system_prompt" in required, (
            f"{cfg.worker_name} schema unexpectedly dropped candidate_system_prompt"
        )
        sp = cfg.system_prompt
        assert "candidate_system_prompt" in sp, (
            f"{cfg.worker_name} system_prompt must name the candidate_system_prompt "
            "output key (else the model returns revised_prompt and the loop stalls)"
        )
        assert "do NOT use `revised_prompt`" in sp, (
            f"{cfg.worker_name} system_prompt must explicitly forbid revised_prompt"
        )
        # The contract also names the other required keys.
        for key in ("rationale", "expected_impact", "scope_of_change"):
            assert key in sp, f"{cfg.worker_name} system_prompt must name {key}"


def test_prompt_loop_temperature_and_token_budget_match_spec() -> None:
    """prompt_improvement is the rewriter -- highest temp; all configs share
    the 4000-token fleet cap (vLLM 8192 ceiling, prompt+completion combined)."""
    rewriter = PROMPT_LOOP_CONFIGS["improvement_prompt_improvement"]
    assert rewriter.temperature == pytest.approx(0.3)
    assert rewriter.max_tokens == 4000

    stats = PROMPT_LOOP_CONFIGS["improvement_prompt_performance_statistics_analysis"]
    assert stats.temperature == pytest.approx(0.2)
    assert stats.max_tokens == 4000

    consistency = PROMPT_LOOP_CONFIGS["improvement_prompt_library_consistency_management"]
    assert consistency.temperature == pytest.approx(0.2)
    assert consistency.max_tokens == 4000


def test_build_payload_attaches_prompt_loop_schema_and_max_tokens() -> None:
    """The payload's output_schema must equal the on-disk schema content."""
    cfg = PROMPT_LOOP_CONFIGS["improvement_prompt_improvement"]
    payload = build_payload(cfg, base_url_override=ROUTER_URL)

    assert payload["worker_name"] == "improvement_prompt_improvement"
    assert payload["temperature"] == pytest.approx(0.3)
    assert payload["max_tokens"] == 4000
    assert payload["model_provider"] == "inference_router"
    assert payload["model"] == "local-llm"

    # output_schema in the payload must equal the on-disk schema verbatim.
    on_disk_path = PROMPT_LOOP_SCHEMAS_DIR / cfg.schema_filename
    with on_disk_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert payload["output_schema"] == on_disk
    assert payload["output_schema"]["title"] == "PromptImprovementProposal"


def test_build_payload_omits_max_tokens_when_workflow_loop_config_used() -> None:
    """workflow_loop configs do not declare max_tokens; payload must omit it."""
    cfg = AGENT_CONFIGS["improvement_workflow_redesign"]
    payload = build_payload(cfg, base_url_override=ROUTER_URL)
    assert "max_tokens" not in payload


def test_prompt_loop_post_creates_all_three_workers() -> None:
    """End-to-end POST of the 3 prompt_loop workers (idempotent on 404 -> POST)."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            captured.append(json.loads(req.content.decode("utf-8")))
            return httpx.Response(201, json={"worker_name": "ok"})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="tok-prompt",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=PROMPT_LOOP_CONFIGS,
        )

    assert set(actions.values()) == {"created"}
    assert {body["worker_name"] for body in captured} == PROMPT_LOOP_NAMES
    for body in captured:
        # Schema must attach correctly per worker.
        cfg = PROMPT_LOOP_CONFIGS[body["worker_name"]]
        with (PROMPT_LOOP_SCHEMAS_DIR / cfg.schema_filename).open("r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert body["output_schema"] == on_disk
        assert body["max_tokens"] == 4000


def test_prompt_loop_idempotent_post_then_skip_on_re_register() -> None:
    """Second register call (without --force) must skip already-present workers."""
    state: dict[str, dict[str, Any]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            if name in state:
                return httpx.Response(200, json=state[name])
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            body = json.loads(req.content.decode("utf-8"))
            state[body["worker_name"]] = body
            return httpx.Response(201, json={"worker_name": body["worker_name"]})
        raise AssertionError(f"unexpected {req.method}")

    # First pass: all 3 created.
    with _client(handler) as client:
        first = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=PROMPT_LOOP_CONFIGS,
        )
    assert set(first.values()) == {"created"}
    assert set(state.keys()) == PROMPT_LOOP_NAMES

    # Second pass: all 3 skipped (no force).
    with _client(handler) as client:
        second = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=PROMPT_LOOP_CONFIGS,
        )
    assert set(second.values()) == {"skipped-exists"}


def test_prompt_loop_force_preserves_live_system_prompt() -> None:
    """G2 protection: prompt_loop workers also keep their live system_prompt by default."""
    captured_puts: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(
                200,
                json={
                    "worker_name": name,
                    "agent_role": f"improvement loop -- {name}",
                    "agent_type": "pydantic_ai",
                    "model_provider": "inference_router",
                    "model": "default",
                    "temperature": 0.2,
                    "system_prompt": "OPERATOR-EDITED PROMPT",
                    "description": "live",
                    "output_schema": {"type": "object", "title": "STALE"},
                },
            )
        if req.method == "PUT":
            captured_puts.append(json.loads(req.content.decode("utf-8")))
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            dry_run=False,
            force_prompt=False,
            client=client,
            configs=PROMPT_LOOP_CONFIGS,
        )

    assert set(actions.values()) == {"updated-preserved-prompt"}
    assert len(captured_puts) == len(PROMPT_LOOP_CONFIGS)
    for body in captured_puts:
        assert body["system_prompt"] == "OPERATOR-EDITED PROMPT"
        # Schema is seed-authoritative.
        assert body["output_schema"]["title"] != "STALE"
        # max_tokens still propagated.
        assert body["max_tokens"] == 4000


def test_prompt_loop_dry_run_lists_three_would_create() -> None:
    """--dry-run alone (full registry, all 15 workers) lists the 3 prompt_loop names."""
    seen_names: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            seen_names.append(req.url.path.rsplit("/", 1)[-1])
            return httpx.Response(404, json={"detail": "not found"})
        raise AssertionError(f"dry-run must not call {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=True,
            client=client,
            # Default configs == register.AGENT_CONFIGS (the merged 15).
        )

    assert len(actions) == 15
    assert PROMPT_LOOP_NAMES.issubset(set(actions.keys()))
    assert set(actions.values()) == {"would-create"}
    assert PROMPT_LOOP_NAMES.issubset(set(seen_names))


# ===========================================================================
# procedure_loop coverage (wf-proc improvement program -- structural rung)
# ===========================================================================


def test_procedure_loop_has_exactly_one_card_type() -> None:
    """The procedure loop ships exactly one worker; its schema file must exist."""
    assert set(PROCEDURE_LOOP_CONFIGS.keys()) == PROCEDURE_LOOP_NAMES
    for cfg in PROCEDURE_LOOP_CONFIGS.values():
        assert (PROCEDURE_LOOP_SCHEMAS_DIR / cfg.schema_filename).is_file(), (
            f"missing schema file for {cfg.worker_name}"
        )


def test_procedure_loop_worker_names_are_validator_clean() -> None:
    """Worker names must match WM's [A-Za-z0-9_-]+ identifier validator (no periods)."""
    import re

    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    for name in PROCEDURE_LOOP_CONFIGS:
        assert pattern.match(name), f"{name} contains characters outside [A-Za-z0-9_-]"
        assert "." not in name, f"{name} contains a period"


@pytest.mark.parametrize(
    "schema_path",
    sorted(PROCEDURE_LOOP_SCHEMAS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_procedure_loop_schema_files_are_valid_jsonschema(schema_path: Path) -> None:
    """Every shipped procedure_loop schema must be valid JSON Schema (draft-07)."""
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.Draft7Validator.check_schema(schema)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["type"] == "object"
    expected_required = {"candidate_procedure", "rationale", "expected_impact", "scope_of_change"}
    assert set(schema["required"]) == expected_required
    assert schema["additionalProperties"] is False
    # The candidate_procedure carries steps in the shape qa-verify-procedure validates.
    step_props = schema["properties"]["candidate_procedure"]["properties"]["steps"]["items"][
        "properties"
    ]
    for key in ("stepNumber", "name", "actionType", "preconditions", "postconditions"):
        assert key in step_props, f"step schema missing {key}"


def test_procedure_loop_each_config_loads_its_output_schema() -> None:
    """Loading the schema must succeed; the placeholder prompt must be substantial."""
    for cfg in PROCEDURE_LOOP_CONFIGS.values():
        schema = cfg.output_schema()
        jsonschema.Draft7Validator.check_schema(schema)
        assert cfg.system_prompt.count(". ") >= 2, (
            f"{cfg.worker_name} system_prompt should have >= 3 sentences"
        )
        assert cfg.temperature == pytest.approx(0.2)
        assert cfg.model_provider == "inference_router"
        assert cfg.model == "local-llm"
        assert cfg.max_tokens == 4000


def test_procedure_loop_system_prompt_names_the_output_contract() -> None:
    """The worker's system_prompt MUST name the ``candidate_procedure`` output key and forbid
    the ``revised_procedure`` alias -- otherwise the model emits the revision under a drifted
    key, the downstream procedure-canary ``Parse Proposal`` skips it, and the loop never
    compounds (the prompt loop's 2026-06-16 stall, applied to procedures)."""
    for cfg in PROCEDURE_LOOP_CONFIGS.values():
        required = set(cfg.output_schema().get("required", []))
        assert "candidate_procedure" in required
        sp = cfg.system_prompt
        assert "candidate_procedure" in sp, (
            f"{cfg.worker_name} system_prompt must name the candidate_procedure output key"
        )
        assert "do NOT use `revised_procedure`" in sp, (
            f"{cfg.worker_name} system_prompt must explicitly forbid revised_procedure"
        )
        for key in ("rationale", "expected_impact", "scope_of_change"):
            assert key in sp, f"{cfg.worker_name} system_prompt must name {key}"


def test_build_payload_attaches_procedure_loop_schema_and_max_tokens() -> None:
    """The payload's output_schema must equal the on-disk schema content."""
    cfg = PROCEDURE_LOOP_CONFIGS["improvement_procedure_improvement"]
    payload = build_payload(cfg, base_url_override=ROUTER_URL)
    assert payload["worker_name"] == "improvement_procedure_improvement"
    assert payload["temperature"] == pytest.approx(0.2)
    assert payload["max_tokens"] == 4000
    on_disk_path = PROCEDURE_LOOP_SCHEMAS_DIR / cfg.schema_filename
    with on_disk_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert payload["output_schema"] == on_disk
    assert payload["output_schema"]["title"] == "ProcedureImprovementProposal"


def test_procedure_loop_post_creates_the_worker() -> None:
    """End-to-end POST of the procedure_loop worker (idempotent on 404 -> POST)."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            captured.append(json.loads(req.content.decode("utf-8")))
            return httpx.Response(201, json={"worker_name": "ok"})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="tok-proc",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=PROCEDURE_LOOP_CONFIGS,
        )

    assert set(actions.values()) == {"created"}
    assert {body["worker_name"] for body in captured} == PROCEDURE_LOOP_NAMES


# ===========================================================================
# asset_loop coverage (principle/requirement/decision improvement)
# ===========================================================================


def test_asset_loop_has_exactly_three_card_types() -> None:
    """The asset loop ships exactly three workers; each schema file must exist."""
    assert set(ASSET_LOOP_CONFIGS.keys()) == ASSET_LOOP_NAMES
    for cfg in ASSET_LOOP_CONFIGS.values():
        assert (ASSET_LOOP_SCHEMAS_DIR / cfg.schema_filename).is_file(), (
            f"missing schema file for {cfg.worker_name}"
        )


def test_asset_loop_worker_names_are_validator_clean() -> None:
    """Worker names must match WM's [A-Za-z0-9_-]+ identifier validator (no periods)."""
    import re

    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    for name in ASSET_LOOP_CONFIGS:
        assert pattern.match(name), f"{name} contains characters outside [A-Za-z0-9_-]"
        assert "." not in name, f"{name} contains a period"


@pytest.mark.parametrize(
    "schema_path",
    sorted(ASSET_LOOP_SCHEMAS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_asset_loop_schema_files_are_valid_jsonschema(schema_path: Path) -> None:
    """Every shipped asset_loop schema must be valid JSON Schema (draft-07)."""
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.Draft7Validator.check_schema(schema)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    # The single content key is candidate_<kind>; it is minimal (name + body).
    kind = schema_path.name.split("_improvement.json")[0]
    candidate_key = f"candidate_{kind}"
    expected_required = {candidate_key, "rationale", "expected_impact", "scope_of_change"}
    assert set(schema["required"]) == expected_required
    candidate = schema["properties"][candidate_key]
    assert candidate["type"] == "object"
    assert candidate["additionalProperties"] is False
    assert set(candidate["required"]) == {"name", "body"}
    assert candidate["properties"]["name"]["minLength"] == 1
    assert candidate["properties"]["body"]["minLength"] == 1
    # scope_of_change enum is fixed across the asset loop.
    assert set(schema["properties"]["scope_of_change"]["enum"]) == {
        "minor_wording",
        "structural",
        "instruction_set",
    }


def test_asset_loop_each_config_loads_its_output_schema() -> None:
    """Loading each schema must succeed; the placeholder prompt must be substantial."""
    for cfg in ASSET_LOOP_CONFIGS.values():
        schema = cfg.output_schema()
        jsonschema.Draft7Validator.check_schema(schema)
        assert cfg.system_prompt.count(". ") >= 2, (
            f"{cfg.worker_name} system_prompt should have >= 3 sentences"
        )
        assert cfg.temperature == pytest.approx(0.2)
        assert cfg.model_provider == "inference_router"
        assert cfg.model == "local-llm"
        assert cfg.max_tokens == 4000


def test_asset_loop_system_prompts_name_the_output_contract() -> None:
    """Every asset_loop worker's system_prompt MUST name its ``candidate_<kind>`` output
    key and forbid drift aliases (revised_<kind>/candidate/title/statement) -- otherwise
    the model emits the revision under a drifted key, the downstream asset DOING
    ``Parse Proposal`` skips it, and the loop never compounds (the prompt/procedure loops'
    2026-06-16 stall, applied to assets)."""
    for name, cfg in ASSET_LOOP_CONFIGS.items():
        kind = name.removeprefix("improvement_").removesuffix("_improvement")
        candidate_key = f"candidate_{kind}"
        required = set(cfg.output_schema().get("required", []))
        assert candidate_key in required, f"{cfg.worker_name} schema is missing {candidate_key}"
        sp = cfg.system_prompt
        assert candidate_key in sp, (
            f"{cfg.worker_name} system_prompt must name the {candidate_key} output key"
        )
        # Explicitly forbid the drift aliases.
        assert "do NOT use" in sp
        assert f"revised_{kind}" in sp, (
            f"{cfg.worker_name} system_prompt must forbid the revised_{kind} alias"
        )
        for alias in ("candidate", "title", "statement"):
            assert alias in sp, f"{cfg.worker_name} system_prompt must forbid the {alias} alias"
        for key in ("rationale", "expected_impact", "scope_of_change"):
            assert key in sp, f"{cfg.worker_name} system_prompt must name {key}"


def test_build_payload_attaches_asset_loop_schema_and_max_tokens() -> None:
    """The payload's output_schema must equal the on-disk schema content."""
    cfg = ASSET_LOOP_CONFIGS["improvement_principle_improvement"]
    payload = build_payload(cfg, base_url_override=ROUTER_URL)
    assert payload["worker_name"] == "improvement_principle_improvement"
    assert payload["temperature"] == pytest.approx(0.2)
    assert payload["max_tokens"] == 4000
    on_disk_path = ASSET_LOOP_SCHEMAS_DIR / cfg.schema_filename
    with on_disk_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert payload["output_schema"] == on_disk
    assert payload["output_schema"]["title"] == "PrincipleImprovementProposal"


def test_asset_loop_post_creates_all_three_workers() -> None:
    """End-to-end POST of the 3 asset_loop workers (idempotent on 404 -> POST)."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            captured.append(json.loads(req.content.decode("utf-8")))
            return httpx.Response(201, json={"worker_name": "ok"})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="tok-asset",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=ASSET_LOOP_CONFIGS,
        )

    assert set(actions.values()) == {"created"}
    assert {body["worker_name"] for body in captured} == ASSET_LOOP_NAMES
    for body in captured:
        cfg = ASSET_LOOP_CONFIGS[body["worker_name"]]
        with (ASSET_LOOP_SCHEMAS_DIR / cfg.schema_filename).open("r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert body["output_schema"] == on_disk
        assert body["max_tokens"] == 4000


# ===========================================================================
# Wrong-tenant silence -- the 2026-09-13 incident
#
# `improvement_workflow_development` ran with `system_prompt = NULL` and
# `output_schema = NULL` on tenant bbbbbbbb-0000-4000-8000-0000000000a5 (the
# one production traffic uses) from 2026-08-27. Three sessions' worth of
# prompt fixes went to a row the running system never reads, and on
# 2026-09-13 a `register.py --force --force-prompt` believed to have reseeded
# production put its 4106-char prompt on the SEED tenant
# aaaaaaaa-0000-4000-8000-000000000001 -- and reported the reseed as verified.
#
# ⛔ THE DEFECT IS NOT "THE WRONG UUID IS HARDCODED". `SYSTEM_WORKFLOW_TENANT_ID`
# is sent only as `X-Internal-Tenant-Id` and does not select the row's tenant;
# WM's auth middleware does, from the licence token or (tokenless) from WM's
# own DEFAULT_TENANT_ID. The defect is that register.py could not TELL you
# which tenant it wrote to, so a write to the wrong one was indistinguishable
# from success. Every test below fails against the pre-fix file, and each one
# fails for a different half of that sentence.
# ===========================================================================


PROD_TENANT = "bbbbbbbb-0000-4000-8000-0000000000a5"
SEED_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"


def _states_row(tenant: str) -> dict[str, Any]:
    """One row as ``GET /workers_states`` returns it (workers_state_db._row_to_dict)."""
    return {
        "id": 1,
        "worker_id": "11111111-1111-4111-8111-111111111111",
        "worker_name": "improvement_workflow_development",
        "tenant_id": tenant,
        "subject_id": tenant,
        "visibility": "tenant",
        "is_agent": True,
        "state_type": "completed",
        "worker_state": {"status": "completed"},
        "timestamp": "2026-09-13T00:00:00+00:00",
        "metadata": {},
    }


def _tenant_probe_handler(
    tenant: str | None,
    inner: Callable[[httpx.Request], httpx.Response] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer the tenant probe with ``tenant`` (``None`` => an empty, undeterminable read).

    Everything else is delegated to ``inner``; with no ``inner`` any other
    request is an assertion failure, which is how the "aborts before any write"
    tests prove no write was attempted rather than merely unobserved.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/workers_states":
            return httpx.Response(200, json=[] if tenant is None else [_states_row(tenant)])
        if inner is None:
            raise AssertionError(
                f"no request beyond the tenant probe expected; saw {req.method} {req.url.path}"
            )
        return inner(req)

    return handler


@contextlib.contextmanager
def _patched_client(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[httpx.Client]:
    """Make ``main()``'s own ``httpx.Client()`` a MockTransport client.

    ``main`` owns its client so the tenant probe and the writes share one
    identity; the tests therefore have to reach it here rather than by passing
    ``client=``.
    """
    client = _client(handler)
    with mock.patch.object(httpx, "Client", return_value=client):
        try:
            yield client
        finally:
            client.close()


def _run_main(*argv: str) -> int:
    """Run the CLI end-to-end and return its exit code.

    ``--wm-url`` and ``--license-token`` are pinned so an operator's
    environment cannot change what the test measures.
    """
    args = ["register.py", "--wm-url", WM_BASE, "--license-token", "", *argv]
    with mock.patch.object(sys, "argv", args):
        return register_main()


def test_resolve_target_tenant_reads_the_tenant_back_from_worker_management() -> None:
    """The resolved tenant comes from a tenant-scoped READ, not from anything we sent.

    Fails on the pre-fix file: ``resolve_target_tenant`` does not exist, so
    there was no way for an operator -- or the next test -- to learn which
    tenant a write would land on.
    """
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return httpx.Response(200, json=[_states_row(PROD_TENANT)])

    with _client(handler) as client:
        resolved = resolve_target_tenant(wm_url=WM_BASE, license_token="tok", client=client)

    assert resolved == PROD_TENANT
    assert seen == [("GET", "/workers_states")], (
        "the tenant must be read from a tenant-scoped endpoint; a probe that writes, "
        "or that reads nothing, is not a resolution"
    )


def test_resolve_target_tenant_is_not_an_echo_of_the_internal_tenant_header() -> None:
    """The answer tracks WM, not ``SYSTEM_WORKFLOW_TENANT_ID`` nor any header we send.

    This is the misreading that cost the 2026-09-13 session its reseed: the
    internal-tenant header is on every request and names none of the tenants
    the row can land on.
    """
    with _client(_tenant_probe_handler(SEED_TENANT)) as client:
        resolved = resolve_target_tenant(wm_url=WM_BASE, license_token="", client=client)

    assert resolved == SEED_TENANT
    assert resolved != SYSTEM_WORKFLOW_TENANT_ID, (
        "X-Internal-Tenant-Id is not the tenant the row is written to"
    )


def test_expect_tenant_mismatch_aborts_before_any_write(capsys: pytest.CaptureFixture[str]) -> None:
    """--expect-tenant <prod> against a deployment that resolves to the seed tenant: no PUT.

    ⭐ THE ASSERTION THAT MATTERS IS ``writes == []``. Exiting non-zero after
    clobbering the wrong row would be a louder version of the same incident;
    the guard is only a guard if it fires BEFORE the first write.

    Fails on the pre-fix file: there is no ``--expect-tenant``, argparse exits
    2 on the unknown flag, and nothing would have compared anything.
    """
    writes: list[str] = []

    def inner(req: httpx.Request) -> httpx.Response:
        writes.append(f"{req.method} {req.url.path}")
        return httpx.Response(200, json={})

    with _patched_client(_tenant_probe_handler(SEED_TENANT, inner)):
        rc = _run_main("--force", "--force-prompt", "--expect-tenant", PROD_TENANT)

    assert rc != 0, "a tenant mismatch must exit non-zero"
    assert writes == [], f"the guard must abort before any request beyond the probe; saw {writes}"
    err = capsys.readouterr().err
    assert PROD_TENANT in err and SEED_TENANT in err, (
        "the failure must name BOTH the tenant asked for and the tenant resolved -- "
        "naming only one is what makes a wrong-target write readable as success"
    )


def test_expect_tenant_match_is_reported_and_proceeds() -> None:
    """The guard is not vacuous: the matching case runs, and says which tenant it ran on."""
    puts: list[str] = []

    def inner(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            puts.append(json.loads(req.content.decode("utf-8"))["worker_name"])
            return httpx.Response(201, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _patched_client(_tenant_probe_handler(PROD_TENANT, inner)):
        rc = _run_main("--dry-run", "--expect-tenant", PROD_TENANT)

    assert rc == 0
    assert puts == [], "--dry-run must still write nothing"


def test_undeterminable_tenant_aborts_a_write_run(capsys: pytest.CaptureFixture[str]) -> None:
    """Fail-closed: an empty tenant-scoped read must never proceed to a write.

    "Cannot determine target tenant" is exactly the state the 2026-09-13 run
    was in while reporting success, so it is the state that must refuse.

    Fails on the pre-fix file: nothing resolves a tenant, so a write run
    proceeds unconditionally and exits 0.
    """
    with _patched_client(_tenant_probe_handler(None)):
        rc = _run_main("--force", "--force-prompt")

    assert rc != 0, "an undeterminable tenant must not proceed to a write"
    assert "cannot determine target tenant" in capsys.readouterr().err


def test_undeterminable_tenant_aborts_even_a_dry_run_when_expect_tenant_is_asked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Asking for the guard and being unable to evaluate it is itself a refusal."""
    with _patched_client(_tenant_probe_handler(None)):
        rc = _run_main("--dry-run", "--expect-tenant", PROD_TENANT)

    assert rc != 0
    assert "cannot determine target tenant" in capsys.readouterr().err


def test_the_resolved_tenant_is_printed_before_any_write(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 1: normal output AND --dry-run name the tenant, ahead of the actions.

    Fails on the pre-fix file, whose entire output could not name a tenant at
    all -- which is why three sessions of prompt fixes read as delivered.
    """

    def inner(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        raise AssertionError(f"dry-run must not call {req.method}")

    with _patched_client(_tenant_probe_handler(PROD_TENANT, inner)):
        rc = _run_main("--dry-run")

    assert rc == 0
    out = capsys.readouterr().out
    assert PROD_TENANT in out, "the run must say which tenant it is addressing"
    assert out.index(PROD_TENANT) < out.index("would-create"), (
        "the tenant must be reported BEFORE the per-worker actions, not in a footer "
        "nobody reads after deciding the run worked"
    )


# ---------------------------------------------------------------------------
# post-write read-back on the SAME resolved tenant
# ---------------------------------------------------------------------------


def test_read_back_mismatch_after_put_raises_and_names_the_worker() -> None:
    """A PUT that reads back differently is a MISMATCH, not an ``updated``.

    Fails on the pre-fix file: ``register_workers`` returned ``{"...": "updated"}``
    for every worker whose PUT returned 200, and never looked at what was
    stored. Reseeding one row and reading another is how the reseed was
    reported done while being undone.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            # The store never takes the write: every read returns the stale row.
            return httpx.Response(
                200, json={"worker_name": name, "system_prompt": "STALE", "output_schema": None}
            )
        if req.method == "PUT":
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client, pytest.raises(WriteVerificationError) as excinfo:
        register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            force_prompt=True,
            dry_run=False,
            verify_writes=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    verdicts = excinfo.value.verdicts
    assert set(verdicts) == set(AGENT_CONFIGS), "every worker that failed read-back must be named"
    for name, verdict in verdicts.items():
        assert verdict.startswith("MISMATCH"), f"{name}: {verdict}"
        assert "system_prompt" in verdict


def test_read_back_mismatch_exits_non_zero_from_main(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mismatch must reach the exit code, not just a log line."""

    def inner(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(
                200, json={"worker_name": name, "system_prompt": "STALE", "output_schema": None}
            )
        if req.method == "PUT":
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _patched_client(_tenant_probe_handler(PROD_TENANT, inner)):
        rc = _run_main("--force", "--force-prompt", "--expect-tenant", PROD_TENANT)

    assert rc != 0, "a read-back mismatch must exit non-zero"
    assert "MISMATCH" in capsys.readouterr().err


def test_read_back_verifies_when_the_store_actually_takes_the_write() -> None:
    """Not vacuous: a store that keeps what it was sent verifies, and main exits 0."""
    stored: dict[str, dict[str, Any]] = {}

    def inner(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            if name in stored:
                return httpx.Response(200, json=stored[name])
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            body = json.loads(req.content.decode("utf-8"))
            stored[body["worker_name"]] = body
            return httpx.Response(201, json={"worker_name": body["worker_name"]})
        raise AssertionError(f"unexpected {req.method}")

    with _patched_client(_tenant_probe_handler(PROD_TENANT, inner)):
        rc = _run_main("--expect-tenant", PROD_TENANT)

    assert rc == 0
    assert set(stored) == set(REGISTER_AGENT_CONFIGS)


def test_the_read_back_uses_the_same_headers_as_the_write() -> None:
    """Read back on the tenant that was written to, or the check proves nothing.

    A read-back that resolves to a different tenant than the write is the
    incident wearing a verification badge.
    """
    seen: list[tuple[str, str, str]] = []
    stored: dict[str, dict[str, Any]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        seen.append((req.method, req.url.path, req.headers.get("x-tenant-id", "")))
        if req.method == "GET":
            if name in stored:
                return httpx.Response(200, json=stored[name])
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            body = json.loads(req.content.decode("utf-8"))
            stored[body["worker_name"]] = body
            return httpx.Response(201, json={"worker_name": body["worker_name"]})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        register_workers(
            wm_url=WM_BASE,
            license_token="tok-prod",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            verify_writes=True,
            client=client,
            configs=PROCEDURE_LOOP_CONFIGS,
        )

    tenant_headers = {header for _, _, header in seen}
    assert tenant_headers == {"Bearer tok-prod"}, (
        "every request -- probe, write and read-back -- must carry one identity; "
        f"saw {tenant_headers}"
    )
    # GET(404) -> POST -> GET(read-back): the read-back is a real second read.
    assert [method for method, _, _ in seen] == ["GET", "POST", "GET"]


# ---------------------------------------------------------------------------
# a NULL live prompt is a failure, not a skip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("live_prompt", [None, "", "   "])
def test_existing_worker_with_null_prompt_is_a_failure_not_a_skip(live_prompt: str | None) -> None:
    """Absent, NULL and blank all mean the running system has no instructions.

    Fails on the pre-fix file: all three returned ``skipped-exists`` and exit
    0. Nothing else in the estate reconciles these rows -- no CI, no Flux, no
    cron -- which is how ``improvement_workflow_development`` sat prompt-less
    for 17 days behind green checks.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(200, json=_live_response_for(name, system_prompt=live_prompt))
        raise AssertionError(f"must not call {req.method} without --force")

    with _client(handler) as client, pytest.raises(EmptyLivePromptError) as excinfo:
        register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert excinfo.value.workers == sorted(AGENT_CONFIGS), (
        "one run must name every prompt-less worker, not stop at the first"
    )
    assert set(excinfo.value.actions.values()) == {"exists-empty-prompt"}, (
        "the reported action must not be a skip"
    )


def test_null_prompt_is_reported_in_dry_run_too() -> None:
    """--dry-run must surface the prompt-less row; it is the diagnostic run."""

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(200, json=_live_response_for(name, system_prompt=None))
        raise AssertionError(f"dry-run must not call {req.method}")

    with _client(handler) as client, pytest.raises(EmptyLivePromptError) as excinfo:
        register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(excinfo.value.actions.values()) == {"would-fail-empty-prompt"}


def test_null_prompt_exits_non_zero_from_main(capsys: pytest.CaptureFixture[str]) -> None:
    """The prompt-less row must reach the exit code, not only a log line."""

    def inner(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(200, json=_live_response_for(name, system_prompt=None))
        raise AssertionError(f"must not call {req.method} without --force")

    with _patched_client(_tenant_probe_handler(PROD_TENANT, inner)):
        rc = _run_main("--expect-tenant", PROD_TENANT)

    assert rc != 0
    out = capsys.readouterr()
    assert "NULL/empty system_prompt" in out.err
    assert "exists-empty-prompt" in out.out, "the per-worker report must still be printed"


def test_a_healthy_prompt_still_skips_so_the_failure_is_not_blanket() -> None:
    """Control: the new failure fires on prompt-less rows ONLY.

    Without this, a change that raised on EVERY existing worker would pass
    every test above while making the script unusable.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            return httpx.Response(
                200, json=_live_response_for(name, system_prompt="A REAL LIVE PROMPT")
            )
        raise AssertionError(f"must not call {req.method} without --force")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=False,
            dry_run=False,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(actions.values()) == {"skipped-exists"}


def test_force_prompt_repairs_a_null_prompt_rather_than_failing() -> None:
    """The documented repair must actually work: --force --force-prompt writes the seed."""
    stored: dict[str, dict[str, Any]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        if req.method == "GET":
            if name in stored:
                return httpx.Response(200, json=stored[name])
            return httpx.Response(200, json=_live_response_for(name, system_prompt=None))
        if req.method == "PUT":
            stored[name] = json.loads(req.content.decode("utf-8"))
            return httpx.Response(200, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _client(handler) as client:
        actions = register_workers(
            wm_url=WM_BASE,
            license_token="",
            base_url_override=ROUTER_URL,
            force=True,
            force_prompt=True,
            dry_run=False,
            verify_writes=True,
            client=client,
            configs=AGENT_CONFIGS,
        )

    assert set(actions.values()) == {"updated"}
    seed_prompts = {cfg.system_prompt for cfg in AGENT_CONFIGS.values()}
    for body in stored.values():
        assert body["system_prompt"] in seed_prompts
        assert body["system_prompt"].strip() != ""


# ---------------------------------------------------------------------------
# --tenant: asserting the target tenant on a tokenless in-cluster run
#
# The failure being fixed: workers/improvement-worker-reconcile has NEVER
# succeeded. Tokenless, it sent no X-Tenant-Id at all, so WM's trusted-network
# branch fell back to the Deployment's DEFAULT_TENANT_ID (the SEED tenant,
# verified live 2026-09-15) and --expect-tenant <prod> correctly refused.
# ---------------------------------------------------------------------------


def _captured_tenant_header(headers: dict[str, str]) -> str | None:
    """``X-Tenant-Id`` as sent, case-insensitively, or ``None`` if absent."""
    for key, value in headers.items():
        if key.lower() == "x-tenant-id":
            return value
    return None


def test_tokenless_run_sends_a_bare_tenant_uuid_header() -> None:
    """⭐ THE FIX. Tokenless + ``--tenant <uuid>`` puts a BARE uuid in X-Tenant-Id.

    Bare, i.e. no ``Bearer `` prefix: WM's tokenless branch hands the raw header
    to ``tenant_constants.resolve_tenant_id``, which parses a uuid. A
    ``Bearer …`` value would be treated as a licence token by the branch above
    it and introspected, which is a different (and failing) path.

    Fails on the pre-fix file: ``_build_headers`` sets X-Tenant-Id only when a
    licence token is present, so every request went out with no tenant claim
    and inherited the deployment's DEFAULT_TENANT_ID.
    """
    seen: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(_captured_tenant_header(dict(req.headers)))
        return httpx.Response(200, json=[_states_row(PROD_TENANT)])

    with _client(handler) as client:
        resolved = resolve_target_tenant(
            wm_url=WM_BASE,
            license_token="",
            assert_tenant=PROD_TENANT,
            client=client,
        )

    assert resolved == PROD_TENANT
    assert seen == [PROD_TENANT], (
        f"a tokenless run must assert the tenant as a bare uuid; sent {seen!r}"
    )
    assert seen[0] is not None and not seen[0].startswith("Bearer "), (
        "a bare assertion must NOT wear the licence-token spelling -- that routes the "
        "request into the introspection branch instead of the trusted-network branch"
    )


def test_tokenless_run_without_the_flag_still_sends_no_tenant_header() -> None:
    """The regression control: omitting --tenant must not start asserting something.

    This is the pre-fix behaviour, and it stays reachable on purpose -- it is
    what makes the CronJob's historical failure reproducible, and silently
    inventing a default here would be the same class of bug as WM's own
    fallback.
    """
    headers = _build_headers("")
    assert _captured_tenant_header(headers) is None
    assert headers["X-Internal-Tenant-Id"] == SYSTEM_WORKFLOW_TENANT_ID


def test_a_tokened_call_is_unchanged_by_the_new_flag() -> None:
    """⭐ CONTROL. With a licence token the header is byte-identical to pre-fix.

    Without this the test above could pass while having quietly rewritten the
    licence-token path, which ten-plus off-cluster seeding runs depend on.
    """
    headers = _build_headers("tok-abc")
    assert _captured_tenant_header(headers) == "Bearer tok-abc"
    assert headers["X-Internal-Tenant-Id"] == SYSTEM_WORKFLOW_TENANT_ID


def test_a_token_and_an_asserted_tenant_together_are_refused() -> None:
    """Both live in X-Tenant-Id, so honouring one silently drops the other.

    A silently dropped tenant assertion is exactly the failure this file exists
    to prevent, so the ambiguity is refused rather than resolved by precedence.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_headers("tok-abc", PROD_TENANT)


@pytest.mark.parametrize(
    "name",
    ["n8n-dev", "george-dev", "fr-test", "__system_workflow__", "n8n-dev "],
)
def test_a_tenant_NAME_is_refused_and_never_mapped_through_tenant_constants(
    name: str,
) -> None:
    """⛔ THE ROOT-CAUSE GUARD. A name must never be resolved into a uuid here.

    ``src/api/tenant_constants.py::TENANT_UUIDS`` maps "n8n-dev" to the SEED
    tenant ``aaaaaaaa-…0001``, while the authoritative licences registry has
    ``n8n-dev`` as ``aaaaaaaa-…``. WM would accept the name and resolve it via
    that map; doing so would aim at production and land on the seed tenant --
    the precise bug. So this script refuses names outright, and the refusal
    must NAME both uuids so the operator is not left guessing which to pass.

    Mutation target: a version that called ``resolve_tenant_id(name)`` instead
    would return SEED_TENANT for "n8n-dev" rather than raising.
    """
    with pytest.raises(ValueError) as excinfo:
        asserted_tenant_uuid(name)

    message = str(excinfo.value)
    assert "NEVER the name" in message or "not a uuid" in message
    if name.strip() == "n8n-dev":
        assert "SEED" in message and "DIFFERENT" in message, (
            "the refusal must say the name resolves to two DIFFERENT tenants across the two "
            "registries; naming neither is how an operator retries with the same wrong value. "
            "⛔ It prints no uuid: the values are environment-supplied, and this repository "
            "may not carry a tenant id."
        )


def test_the_name_to_uuid_path_is_not_reachable_from_the_cli() -> None:
    """End-to-end: ``--tenant n8n-dev`` exits non-zero and issues NO request.

    ⭐ ``requests == []`` is the load-bearing assertion. Exiting non-zero after
    having already probed (or written) the seed tenant would be the same
    incident with a louder ending.
    """
    requests: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(f"{req.method} {req.url.path}")
        return httpx.Response(200, json=[_states_row(SEED_TENANT)])

    with _patched_client(handler):
        rc = _run_main("--dry-run", "--tenant", "n8n-dev", "--expect-tenant", PROD_TENANT)

    assert rc != 0, "a tenant NAME must not be accepted"
    assert requests == [], f"nothing may be sent before --tenant validates; saw {requests}"


@pytest.mark.parametrize(
    "bad",
    [
        "00000000-0000-0000-0000-000000000000",  # nil / default-anon
        "aaaaaaaac92a7e378fdda eb0d8c5396f9",  # not parseable
        "urn:uuid:bbbbbbbb-0000-4000-8000-0000000000a5",  # non-canonical spelling
        "{bbbbbbbb-0000-4000-8000-0000000000a5}",  # braced
        "aaaaaaaac92a7e378fddeb0d8c5396f9",  # undashed  # pragma: allowlist secret
    ],
)
def test_unusable_tenant_values_raise_rather_than_fall_back(bad: str) -> None:
    """Never substitute a default for an unusable value -- stop the run.

    The nil uuid parses perfectly and is registered as ``default-anon``, so a
    shape check alone lets it through; it is refused by value, matching WM's
    own boundary.
    """
    with pytest.raises(ValueError):
        asserted_tenant_uuid(bad)


def test_a_canonical_uuid_is_accepted_and_returned_lowercase() -> None:
    """Not vacuous: the validator must still pass the value the CronJob needs."""
    assert asserted_tenant_uuid(PROD_TENANT) == PROD_TENANT
    assert asserted_tenant_uuid(f"  {PROD_TENANT.upper()}  ") == PROD_TENANT


def test_the_asserted_tenant_reaches_every_write_not_just_the_probe() -> None:
    """The probe and the writes must share one identity, or the guard grades a copy.

    A fix that asserted the tenant only on ``resolve_target_tenant`` would make
    ``--expect-tenant`` pass while every POST still landed on the seed tenant --
    green, and wrong in exactly the original direction.
    """
    tenants: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        tenants.append(_captured_tenant_header(dict(req.headers)))
        if req.url.path == "/workers_states":
            return httpx.Response(200, json=[_states_row(PROD_TENANT)])
        if req.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "POST":
            name = json.loads(req.content.decode("utf-8"))["worker_name"]
            return httpx.Response(201, json={"worker_name": name})
        raise AssertionError(f"unexpected {req.method}")

    with _patched_client(handler):
        rc = _run_main("--tenant", PROD_TENANT, "--expect-tenant", PROD_TENANT, "--dry-run")

    assert rc == 0
    assert len(tenants) > 1, "the run must have issued the probe and at least one GET"
    assert set(tenants) == {PROD_TENANT}, (
        f"every request must carry the asserted tenant, not just the probe; saw {set(tenants)}"
    )


def test_the_cronjob_invocation_now_succeeds_end_to_end() -> None:
    """The live CronJob's exact shape, plus --tenant, against a WM that honours it.

    Fails on the pre-fix file: argparse exits 2 on the unknown ``--tenant``.
    Without ``--tenant`` at all it is the red the CronJob has shown every run.
    """
    with _patched_client(_tenant_probe_handler(PROD_TENANT, _dry_run_get_404)):
        rc = _run_main("--dry-run", "--tenant", PROD_TENANT, "--expect-tenant", PROD_TENANT)
    assert rc == 0

    # The control: same invocation with --tenant removed still reproduces the
    # historical failure against a deployment whose default is the seed tenant.
    with _patched_client(_tenant_probe_handler(SEED_TENANT)):
        rc_without = _run_main("--dry-run", "--expect-tenant", PROD_TENANT)
    assert rc_without != 0


def _dry_run_get_404(req: httpx.Request) -> httpx.Response:
    if req.method == "GET":
        return httpx.Response(404, json={"detail": "not found"})
    raise AssertionError(f"--dry-run must not call {req.method}")


def test_a_401_on_the_probe_names_REQUIRE_AUTHENTICATED_TENANT(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 2: an armed tenant gate must fail loudly and legibly.

    ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` is default-False and is
    unset on the live Deployment (verified 2026-09-15), but it exists to be
    armed per-caller. When it is, a bare assertion is 401'd. The run must still
    fail closed -- and must say WHY, because "HTTP 401" alone is
    indistinguishable from a network fault and the remedy (a real licence
    token) is not guessable from a status code.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/workers_states":
            return httpx.Response(401, json={"detail": "unauthenticated tenant assertion"})
        raise AssertionError(f"nothing may run after a refused probe; saw {req.url.path}")

    with _patched_client(handler):
        rc = _run_main("--dry-run", "--tenant", PROD_TENANT, "--expect-tenant", PROD_TENANT)

    assert rc != 0, "a refused tenant assertion must never proceed"
    err = capsys.readouterr().err
    assert "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT" in err, (
        "the one setting that turns a working bare header into a 401 must be named"
    )
    assert "--license-token" in err, "the remedy must be named, not just the symptom"


def test_a_401_without_an_assertion_does_not_blame_the_tenant_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Control for the hint: it must not fire where it is not the explanation.

    With no ``--tenant``, ``_audit_unauthenticated_tenant_assertion`` returns
    ``None`` (it counts assertions, not their absence), so the tenant gate is
    NOT the cause and naming it would send the operator to the wrong knob.
    """
    with _patched_client(lambda _req: httpx.Response(401, json={})):
        rc = _run_main("--dry-run", "--expect-tenant", PROD_TENANT)

    assert rc != 0
    err = capsys.readouterr().err
    assert "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT" not in err


def test_the_asserted_tenant_is_printed_separately_from_the_read_back_tenant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An INPUT and a READ-BACK that happen to agree must still be two lines.

    Collapsing them is how a reader loses the ability to tell an honoured
    assertion from an ignored one.
    """
    with _patched_client(_tenant_probe_handler(PROD_TENANT, _dry_run_get_404)):
        rc = _run_main("--dry-run", "--tenant", PROD_TENANT, "--expect-tenant", PROD_TENANT)

    assert rc == 0
    out = capsys.readouterr().out
    assert "asserted tenant:" in out
    assert "target tenant:" in out
    assert out.index("asserted tenant:") < out.index("target tenant:"), (
        "what we claimed must be printed before what came back"
    )


def test_expect_tenant_still_catches_a_deployment_that_IGNORES_the_assertion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⭐ The guard is weaker once we assert, but it is NOT vacuous -- prove it.

    Asserting a tenant the server honours does make the read-back largely
    confirm our own claim. The residual value is this case: a WM that ignores
    the header (older build, or the assertion silently dropped) resolves to its
    own DEFAULT_TENANT_ID and the mismatch still aborts before any write.
    """
    writes: list[str] = []

    def inner(req: httpx.Request) -> httpx.Response:
        writes.append(f"{req.method} {req.url.path}")
        return httpx.Response(200, json={})

    # Server ignores X-Tenant-Id and answers with its own default.
    with _patched_client(_tenant_probe_handler(SEED_TENANT, inner)):
        rc = _run_main(
            "--force", "--force-prompt", "--tenant", PROD_TENANT, "--expect-tenant", PROD_TENANT
        )

    assert rc != 0
    assert writes == [], f"the guard must still abort before any write; saw {writes}"
    assert SEED_TENANT in capsys.readouterr().err
