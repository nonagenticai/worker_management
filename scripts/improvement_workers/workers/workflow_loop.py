"""Worker configs for Plan 5 (Workflow Improvement Loop).

Five workers, one per ``card_type`` in the workflow improvement stream owned
by the ``__system_workflow__`` internal tenant (see fr-backend memory
``continuous-improvement-system-level``). Each worker is registered in
worker_management with a placeholder ``system_prompt`` and a default
``output_schema``; Plan 6 (prompt improvement loop) will rewrite the prompts
in-place via ``PUT /workers/{name}`` -- per memory ``no-prompt-registry-wm-is-the-store``,
WM is the prompt store and PUT reflects on the next call.

The schemas are JSON Schema Draft 2020-12 documents stored alongside this
file in ``schemas/workflow_loop/<card_type>.json``. ``register.py`` loads
them and posts them as ``output_schema`` on each worker so callers do NOT
need to pass ``output_schema`` per-call (per memory ``wm-as-sole-llm-gateway``,
callers MAY still override per-call).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-relative path to the schemas dir (scripts/improvement_workers/schemas/workflow_loop)
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "workflow_loop"


@dataclass(frozen=True)
class WorkerConfig:
    """Subset of WorkerCreateRequest fields the registration script cares about."""

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


# Placeholder system prompts. Plan 6 will rewrite these via PUT /workers/{name}.
# Each prompt is intentionally generic, >=3 sentences, and asks the model to
# emit a JSON object matching the registered output_schema.
_STATS_PROMPT = (
    "You are the workflow statistics analysis worker for the Nonagentic AI workflow "
    "improvement loop. Given a workflow definition and a window of execution telemetry "
    "(per-node validation rates, latencies, costs, error counts), compute aggregate "
    "statistics, surface anomalies, and recommend which downstream improvement card "
    "(performance_analysis, redesign, pattern_catalogue_improvement, or development) "
    "should run next. Reply with a single JSON object that exactly matches the "
    "registered output_schema; do not emit prose, markdown, or code fences."
)

_PERF_PROMPT = (
    "You are the workflow performance analysis worker for the Nonagentic AI workflow "
    "improvement loop. Given a workflow and its recent execution telemetry, identify "
    "the bottleneck nodes (by p95 latency, cost-per-call, retry rate, or validation "
    "failure rate), explain which metric is degraded and by how much, and recommend "
    "which downstream improvement card should act on the finding. Reply with a single "
    "JSON object that exactly matches the registered output_schema; do not emit prose, "
    "markdown, or code fences."
)

_REDESIGN_PROMPT = (
    "You are the workflow redesign worker for the Nonagentic AI workflow improvement "
    "loop. Given a workflow definition plus statistics and performance findings from "
    "upstream cards, propose a minimal RFC-6902 JSON Patch against the workflow that "
    "removes a measurable bottleneck or raises validation rate, and predict the deltas "
    "(cost%, latency%, validation_delta) the patch should achieve. Reply with a single "
    "JSON object that exactly matches the registered output_schema; do not emit prose, "
    "markdown, or code fences."
)

_PATTERN_PROMPT = (
    "You are the workflow pattern-catalogue improvement worker for the Nonagentic AI "
    "workflow improvement loop. Given the current pattern catalogue and a set of "
    "redesign proposals or recurring shapes seen across workflows, propose new patterns "
    "to add (with name + structural definition) and obsolete patterns to deprecate "
    "(with reason). Reply with a single JSON object that exactly matches the registered "
    "output_schema; do not emit prose, markdown, or code fences."
)

# The development worker's prompt LEADS with the patch contract. It previously opened by
# asking for prose `implementation_notes` + a `new_workflow_id` -- a task the registered
# output_schema forbids (additionalProperties:false), so the most-attended position in the
# prompt described work that is dropped on arrival. Measured 2026-09-13: the worker declined
# on 2 of 2 kaizen drives. It also never said `path` is an RFC-6901 JSON Pointer and that
# `nodes` is positional, so a well-reasoned patch could still be inapplicable; and it blessed
# declining four times over with nothing pulling the other way, so the named defect classes
# below are the deliberate counterweight.
# The defect-class list is stated UNCONDITIONALLY ("Whenever you read the workflow"), while
# "confine the patch to those nodes" stays gated on telemetry naming them. An earlier draft
# gated BOTH on "when the evidence does implicate a node", which left the no-telemetry branch
# -- the branch both live declines took -- with the bare "justify it without inventing
# evidence" sentence and no counterweight at all. The defect classes are readable off the
# workflow JSON, so looking for them is never an invitation to invent telemetry; the
# no-invention clause is kept verbatim for that reason.
#
# ⚠️ AN ABSENT KEY IS NOT A DEFECT, and the first cut of this counterweight got that
# wrong. "An external call with no `retryOnFail`" is satisfied by essentially the whole
# estate, so on its own it swaps "always declines" for "always proposes the same patch".
# MEASURED 2026-09-13, both populations, predicates stated so a reader can re-run them:
#   population A -- the 754 workflow JSON backups under n8n_custom_nodes/n8n_live_backup/
#     **/*.json (6,036 nodes). A historical snapshot; includes inactive and superseded
#     workflows.
#   population B -- LIVE ACTIVE workflows, which is what a drive actually targets:
#     `FROM workflow_entity w, LATERAL jsonb_array_elements(w.nodes::jsonb) n WHERE
#      w.active` on the live n8n Postgres db (228 workflows, 3,353 nodes).
#
#     key present on a node        A (6,036 nodes)     B (3,353 nodes)
#       retryOnFail                   0    0.0%          76    2.3%
#       onError                     577    9.6%         418   12.5%
#       parameters.options.timeout   49    0.8%          52    1.6%
#
# So `retryOnFail` is ABSENT on 100.0% (A) / 97.7% (B) of nodes: reading that absence is
# reading the corpus, not this workflow. The two populations disagree by construction --
# prefer B for any claim about what a drive will meet. The prompt therefore says outright
# that a missing key is not itself a finding, and gives an ordering: telemetry first, then
# a node whose own JSON shows the failure is BOTH real and unattended.
# That criterion -- an external call (`@nonagentic/*`, `n8n-nodes-base.httpRequest`,
# `n8n-nodes-base.executeWorkflow`) whose `onError` is `continueRegularOutput` or whose
# `continueOnFail` is true, with no node within 2 connection hops downstream whose JSON
# mentions "error" -- fires on: A 769 nodes (12.7%) across 213/754 workflows (28.2%);
# B 477 nodes (14.2%) across 148/228 workflows (64.9%).
# ⚠️ Read that honestly: it narrows WHICH NODE (14.2% of live nodes, against 38.6% for
# the naive "external call with no retry"), it does NOT restore "usually decline" at the
# drive level -- 98.7% of live-active workflows contain at least one naive candidate and
# 64.9% contain a criterion candidate. No structural criterion measured here got below
# that, because the estate really does swallow external-call errors nearly everywhere.
# Declining stays reachable on the 35.1% of live-active workflows where no node qualifies.
# ONLY THREE classes are listed, and a fourth -- "a hardcoded value that belongs in a
# parameter" -- was deliberately REMOVED. Do not re-add it. The counterweight narrows
# rather than widens precisely because each class is settled by a key already in the
# node's JSON (`retryOnFail`, `timeout`, `onError`): the model reads a KEY, it does not
# form an opinion. ⚠️ Read that against the absent-key block above: a key the model can
# read is necessary, not sufficient -- `retryOnFail` is a key AND absent estate-wide, so
# it settles nothing on its own, which is why the prompt now pairs it with a condition on
# the node's role. "Belongs in a parameter" has no such key -- it is pure judgement, so a
# model under pressure to propose something can satisfy it on ANY workflow, which is the
# licence to invent that this whole block exists to close. The obvious objective proxy
# ("the same literal appears in >=2 nodes") does not rescue it: trivial literals ('POST',
# 'main', an expression fragment) repeat in almost every workflow, so the test is passed
# by nearly everything and the licence survives it. And unlike the three that remain, its
# REMEDY is not decidable from the workflow either -- n8n has no workflow-parameter
# primitive, so the fix is a new node plus a rewire (risking the external-contract rule
# and the 25-op cap) or an `{{ $vars.X }}` reference whose variable may not exist on the
# instance, which produces a patch that looks right and breaks at run time.
# ⚠️ This comment used to close with "measured over
# 754 live workflow backups it was the smallest class at 3.8% (classes 1-3: 36.1/33.3/
# 42.4%, union 50.1%)". NONE of those five figures reproduces -- re-measured over the same
# 754 backups under every predicate tried (external-call-scoped, any-node-scoped, per-node
# and per-workflow) they came out 5.0/0.5/20.4 and 99.6/97.3/20.4, never 36.1/33.3/42.4.
# No predicate that yields them could be recovered, so they are DELETED rather than
# restated: a figure nobody can reproduce is worse than none. The argument for removing
# the class never rested on its size anyway -- it rests on there being no key that
# settles it.
_DEV_PROMPT = (
    "Your output key is `workflow_patch`: an RFC-6902 JSON-Patch ARRAY (a list of "
    "{op, path[, from, value]} operations) applied to the BASE workflow JSON you are given, "
    "expressing ONE small, targeted, BOUNDED improvement in at most 25 operations. Every "
    "`path` is an RFC-6901 JSON Pointer into that base document, and `nodes` is an ARRAY "
    "addressed POSITIONALLY, never by node name -- to retarget the URL of the node at index "
    "7 the path is `/nodes/7/parameters/url`. You are the workflow development worker for "
    "the Nonagentic AI workflow improvement loop: you turn an approved redesign, the pattern "
    "catalogue, and the workflow's own execution evidence into that patch -- you do NOT "
    "write prose implementation notes and you do NOT propose new workflows. Never alter the "
    "workflow's external contract: leave webhook/trigger entrypoints, httpMethod, "
    "responseMode, and executeWorkflowTrigger inputSource exactly as they are. When a "
    "telemetry block is present (execution counts, failure counts, per-node failure "
    "attribution, recent error messages) read it FIRST and target the node the evidence "
    "actually implicates. If -- and only if -- that telemetry NAMES specific failing nodes, "
    "confine the patch to those nodes. Whenever you read the workflow -- telemetry or no "
    "telemetry, a node named or no node named -- look hard for these concrete defect "
    "classes: an external call with no retry on a transient failure, a missing timeout, and "
    "an error output left unread on a node set to continue on failure. Each one is settled "
    "by a key that is already in the node's own JSON -- `retryOnFail`, `timeout`, `onError` "
    "-- so pointing at one is reading evidence rather than inventing it, and they are what "
    "this loop exists to catch: when you find one, propose the fix rather than deferring "
    "it. A key being ABSENT is not by itself a finding: `retryOnFail` is unset on nearly "
    'every node running in this estate, so "it has no `retryOnFail`" describes the corpus '
    "rather than this workflow. What makes one of these a real defect is the node's own "
    "JSON showing the failure is BOTH real and unattended -- an external call (an HTTP or "
    "API call, a sub-workflow execution, a `@nonagentic` node) whose `onError` is "
    "`continueRegularOutput` or whose `continueOnFail` is true, so a failure is swallowed, "
    "and with nothing downstream of it reading that error. Order your attention "
    "accordingly: patch what the telemetry implicates first; then a node whose own "
    "configuration shows a concrete unattended risk like that one; and treat a key that is "
    "merely missing, on a node this workflow shows no sign of expecting to fail, as no "
    "finding at all. When no telemetry is shown, or it names no failing node, judge the "
    "workflow on its own structure and propose a change only if you can justify it without "
    "inventing evidence -- a defect class you can point at in the workflow JSON is exactly "
    "such a justification."
)


# Output contract appended ONLY to the improvement_workflow_development worker. Mirrors
# prompt_loop / procedure_loop (#74 / #75): NAME the exact output key and FORBID the observed
# drift aliases. The development worker emits a bounded RFC-6902 JSON-Patch under
# `workflow_patch` (not prose `implementation_notes` + a nullable `new_workflow_id`); if the
# model returns the change under any other key the downstream workflow-canary Parse Proposal
# skips it and the loop never compounds. The patch shape itself is stated in _DEV_PROMPT
# above, so this block only names the OTHER fields, the drift aliases, and the decline.
# `expected_impact` is enumerated key-by-key because the schema is additionalProperties:false
# with no `required`: "an object" invites an invented key (and _REDESIGN_PROMPT, above,
# teaches a DIFFERENT set of names), while {} is always legal.
# ⚠️ The decline instruction deliberately no longer mentions `no_change_declined`. That literal
# is DERIVED by the n8n canary, not emitted by the model -- see n8n_custom_nodes
# `scripts/build-doing-workflow-canary.py:231`, which composes it from
# `scope_of_change == 'no_change'` plus the rationale. Naming a downstream label in the prompt
# tells the model to produce something it cannot produce; the contract that matters here is
# `scope_of_change`, and that is what the prompt names. The linkage is recorded here so a
# future reader does not re-add it.
# The anti-fabrication CONSEQUENCE ("a fabricated change is promoted to a live canary, an
# honest decline is not") is its own sentence, restored from dd5e436. It is the strongest
# argument in the prompt against padding the patch, and it must NOT be folded back into the
# decline sentence as a tail clause: a main clause that blesses declining with the warning
# trailing behind it is read as permission with a caveat, which is what d314d52 shipped --
# it dropped the consequence exactly as a strong pro-proposal counterweight was added.
# Keeping it separate costs one sentence.
#
# ⚠️ THE DECLINE LIVES IN ITS OWN NAMED CONSTANT, `_DECLINE_POLICY`, AND NOWHERE ELSE.
# That is not cosmetic and it is not free to undo. Two successive guards tried to police the
# decline by COUNTING blessing sentences by keyword over the assembled prose, and both were
# walked around with the test suite green (see the region guard in
# tests/scripts/test_improvement_workers_register.py for the two attacks, kept there as
# executable fixtures): once by appending the blessing to the exempt anti-fabrication
# sentence with an em-dash so it stayed ONE sentence, and once by writing a blessing that
# uses none of the marker words. Counting prose cannot work. The region guard that replaced
# them bounds a REGION: `_DECLINE_POLICY` is pinned byte-for-byte in the test, every one of
# its sentences must be a whole sentence of the assembled prompt, and no decline vocabulary
# may appear outside it. So: do not inline this constant back into the contract string, do
# not add a second sentence about declining anywhere else in this module, and if you must
# change the decline wording, change the pinned copy in the test in the same commit and say
# why.
#
# ⚠️ THE REGION GUARD WAS DEFEATED TOO, and this comment used to imply it was sufficient.
# Read its three legs honestly: the byte-identical-occurrence and whole-sentence legs hold,
# but the "no decline vocabulary outside the region" leg is PROSE MATCHING, and the guard
# skips outright any sentence carrying none of its 19 listed terms. So a blessing written
# to avoid those terms passes ANYWHERE outside the region. ATTACK 3, recorded verbatim in
# the test as `_ATTACK_3_SENTENCE`, is exactly that, inserted at the head of
# `_CONTRACT_CLOSING` below; measured 2026-09-13 it went green at 87 passed. A second
# channel was never guarded at all: the SCHEMA in schemas/workflow_loop/
# workflow_development.json reaches the model (`description` is not stripped by
# `_XGRAMMAR_UNSUPPORTED_KEYS`), and a blessing appended to `scope_of_change.description`
# also went green.
#
# ⭐ WHAT NOW HOLDS IS A BYTE PIN, not a classifier. The test asserts the sha256 of the
# ASSEMBLED `system_prompt` and of the registered `output_schema` -- both taken from
# `register.build_payload`, i.e. what is actually PUT -- against literals pinned in
# tests/scripts/test_improvement_workers_register.py, with the reference text in
# tests/scripts/fixtures/improvement_workflow_development/. ANY edit to this prompt or that
# schema reds the suite, including a typo or a helpful clarification, and the failure tells
# you to read the diff and to re-pin deliberately with a commit message saying what and why.
# Do NOT add a fourth wordlist; three have been walked around within minutes each.
#
# ⚠️ CORRECTED 2026-09-13 -- the alias sentence used to read "Any such key is DROPPED before
# your proposal is read, so work you put under it is LOST." BOTH halves of that mechanism
# were false, and the direction of the error mattered: it understated the penalty.
#   * NOT DROPPED. schemas/workflow_loop/workflow_development.json is
#     `additionalProperties: false` at the top level with four `required` keys, so an extra
#     key does not survive as ignored baggage -- it makes the WHOLE object invalid and the
#     response is refused. Losing one key and losing the entire proposal are different
#     outcomes, and the worse one is the true one.
#   * NOT UNREAD EITHER. Three of the five aliases named here -- `patch`, `operations`,
#     `changes` -- are explicitly accepted by the n8n canary's Parse Proposal
#     (n8n_custom_nodes `scripts/build-doing-workflow-canary.py`, in the `PARSE_PROPOSAL`
#     node body -- cited by SYMBOL, not line, because the line number ALREADY rotted once:
#     this comment said `:185`, which was true when written and became `:194` on origin/main
#     on 2026-09-12. Grep for the expression rather than trusting a number:
#     `proposal.workflow_patch || proposal.patch || proposal.operations || proposal.changes`).
#     So the claim "before your proposal is read" was wrong about the reader too.
# The INSTRUCTION is unchanged and still right -- `workflow_patch` is the only key that can
# get through validation, so it is the only one worth writing to. Only the stated mechanism
# is corrected, and the correction makes it stronger, not softer.
_CONTRACT_FIELDS = (
    " Put the patch under `workflow_patch` and nowhere else -- do NOT use "
    "`implementation_notes`, `new_workflow_id`, `changes`, `patch`, or `operations`. The "
    "schema is `additionalProperties: false`, so any such key makes the WHOLE object fail "
    "validation and your entire proposal is refused -- you lose all of it, not just that "
    "key. Also "
    "provide `rationale` (a non-empty string naming what you changed and why), "
    "`expected_impact`, and `scope_of_change` (one of 'node_parameter', 'node_addition', "
    "'node_removal', 'connection_rewire', 'no_change'). `expected_impact` is an object that "
    "accepts ONLY these three optional keys -- `delta_execution_success_rate` (a number), "
    "`delta_latency_ms` (a number), and `breaking_changes` (a boolean); any other key fails "
    "validation, and an empty object {} is acceptable when you cannot predict a delta."
)

# The ONLY place in this prompt that is allowed to mention declining. Two sentences: the
# instruction, then the anti-fabrication consequence as its own sentence. Pinned verbatim by
# the test -- read the block above before editing a single character of it.
_DECLINE_POLICY = (
    " If the "
    "workflow genuinely needs no change, emit an EMPTY `workflow_patch` array WITH "
    "`scope_of_change` = 'no_change' and a `rationale` naming what you examined. NEVER "
    "invent an operation to avoid an empty patch: a fabricated change is promoted to a live "
    "canary, an honest decline is not."
)

_CONTRACT_CLOSING = (
    " Reply with a single JSON object that exactly matches "
    "the registered output_schema; do not emit prose, markdown, or code fences."
)

_OUTPUT_CONTRACT = _CONTRACT_FIELDS + _DECLINE_POLICY + _CONTRACT_CLOSING


AGENT_CONFIGS: dict[str, WorkerConfig] = {
    "improvement_workflow_statistics_analysis": WorkerConfig(
        worker_name="improvement_workflow_statistics_analysis",
        description=(
            "Workflow improvement loop -- statistics_analysis card. Aggregates "
            "execution telemetry and routes to the next improvement card."
        ),
        system_prompt=_STATS_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="workflow_statistics_analysis.json",
    ),
    "improvement_workflow_performance_analysis": WorkerConfig(
        worker_name="improvement_workflow_performance_analysis",
        description=(
            "Workflow improvement loop -- performance_analysis card. Identifies "
            "bottleneck nodes and routes to the next improvement card."
        ),
        system_prompt=_PERF_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="workflow_performance_analysis.json",
    ),
    "improvement_workflow_redesign": WorkerConfig(
        worker_name="improvement_workflow_redesign",
        description=(
            "Workflow improvement loop -- redesign card. Emits an RFC-6902 JSON "
            "Patch against a workflow with predicted deltas."
        ),
        system_prompt=_REDESIGN_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="workflow_redesign.json",
    ),
    "improvement_workflow_pattern_catalogue_improvement": WorkerConfig(
        worker_name="improvement_workflow_pattern_catalogue_improvement",
        description=(
            "Workflow improvement loop -- pattern_catalogue_improvement card. "
            "Proposes new patterns to add and old patterns to deprecate."
        ),
        system_prompt=_PATTERN_PROMPT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="workflow_pattern_catalogue_improvement.json",
    ),
    "improvement_workflow_development": WorkerConfig(
        worker_name="improvement_workflow_development",
        description=(
            "Workflow improvement loop -- development card. Translates an approved "
            "redesign into a bounded RFC-6902 JSON-Patch emitted under workflow_patch."
        ),
        system_prompt=_DEV_PROMPT + _OUTPUT_CONTRACT,
        temperature=0.2,
        model_provider="inference_router",
        model="local-llm",
        schema_filename="workflow_development.json",
    ),
}
