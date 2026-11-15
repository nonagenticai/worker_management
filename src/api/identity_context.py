"""Per-request identity ContextVars for cross-async-context propagation.

The auth middleware already stashes identity on `request.state.{tenant_id,
license_id, subject_id}`, which is enough for code paths that have direct
access to the FastAPI Request (e.g. the stateless `/chat/completions`
endpoint that builds a fresh agent per request).

The worker-dispatch path (`/workers/{name}/message`) is different: it reuses
a cached `InferenceRouterOpenAIProvider` across requests via
`WorkerInstanceCache`, keyed only by worker_name. Two concurrent requests
from different end-users would both use the same cached `AsyncOpenAI`
client; baking identity into `default_headers` at construction time would
attribute every subsequent request to whoever happened to warm the cache
first.

ContextVars solve this without breaking the cache: the middleware sets the
identity in the current async context, and an httpx event hook on the
shared client reads from the ContextVar at request-send time. ContextVars
are task-local in asyncio, so two concurrent requests on the same cached
client see their own identity.

Pattern:
    1. `AuthenticationMiddleware` calls `set_identity_context(...)` after
       introspect, BEFORE `await call_next(request)`.
    2. Anywhere downstream — including code reached via a long chain of
       awaits — `get_subject_id_from_context()` returns the right value.
    3. `InferenceRouterOpenAIProvider`'s outbound httpx event hook reads
       from these ContextVars to set `X-Tenant-Id` / `X-License-Id` /
       `X-Subject-Id` on the request being sent to inference-router.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

from cachetools import TTLCache

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Trusted effective-tenant side-channel header (single source of truth for WM). A
# delegating service account (subject ∈ AUTH_DELEGATING_SUBJECTS) may carry the
# run's real tenant in this header so BYO resolution + cost events attribute to
# the run tenant, not the SA tenant. The inbound middleware honours it; the
# outbound httpx hooks re-stamp it so a downstream re-delegating hop sees it too.
# NOTE: WM has no w2v_common dependency (it deliberately carries its own auth
# stack), so this frozen wire constant is intentionally duplicated, not imported.
EFFECTIVE_TENANT_HEADER = "X-Effective-Tenant-Id"

# The all-zero uuid inference-router coerces an unresolvable tenant to. It is a
# SENTINEL, never a customer: a call carrying it is unattributable, so outbound
# helpers omit the tenant header entirely rather than assert nothing-as-a-value.
# Same "frozen wire constant, deliberately not imported" rule as the header
# above — but hosted HERE rather than in the agent layer so the CLI-agent HTTP
# paths and the pydantic-ai httpx hook cannot drift to two different sentinels.
NIL_TENANT_ID = "00000000-0000-0000-0000-000000000000"

# HOW the tenant in `_tenant_id_var` was established. A CLOSED set on purpose:
# these values reach a metric label AND a billing field, so an open-ended string
# here would be the unbounded-cardinality hole every other closed set in
# `telemetry/agent_metrics.py` exists to avoid. Kept HERE rather than in
# agent_metrics because this module is where the fact is produced, and
# agent_metrics already imports nothing from src (no cycle either direction).
#
# ⛔ WHY THIS EXISTS AT ALL. `emit_byo_cost_event_task` bills whatever tenant is
# in this ContextVar. On `auth_middleware`'s delegation branch that value is the
# `X-Effective-Tenant-Id` header, which `_effective_tenant` validates for
# SPELLING ONLY (`resolve_tenant_id`: a uuid, or a tenant name this deployment
# knows). `AUTH_DELEGATING_SUBJECTS` says WHICH SUBJECTS may delegate; nothing
# anywhere says which TENANTS a given subject may act for. The middleware knows
# the difference — it stamps `request.state.tenant_proven = not delegated` — but
# `request.state` does not survive into the agent's async context, so until this
# ContextVar existed the emitter could not tell an owner from an adoptee and the
# billed row carried no trace of the distinction.
#
#   license   — a licence token was introspected AND the tenant it established
#               is the one the request is scoped to. The only PROVEN value.
#   delegated — a licence token was introspected (WHO is proven) but the request
#               was re-scoped to another tenant via `X-Effective-Tenant-Id`.
#               Attribution to that tenant is INTENDED; its verification does
#               not exist. Fix: entitle the caller for that tenant.
#   asserted  — the tokenless trusted-network branch: the caller named its own
#               tenant in a bare `X-Tenant-Id`. Kept APART from `delegated`
#               because the two have different owners and different fixes
#               ("present a token" vs "present a token FOR THAT TENANT"), and
#               merging them makes the smaller population invisible inside the
#               larger — the same rule as `MODALITY_KEY_WITHHELD_ADOPTED_TENANT`.
#   default   — tokenless AND no tenant header: this deployment's
#               DEFAULT_TENANT_ID. A worse finding than `asserted`, so not
#               merged into it.
#   bypass    — `bypass_auth` is on. A TEST affordance; never production.
#   unknown   — nobody stated a provenance. THE DEFAULT, and the direction is
#               the whole safety property: an emit with no request context, a
#               rollback to a peer predating this parameter, or a future auth
#               branch that forgets to pass it must all land here — visible and
#               reconcilable — and never be read as `license`.
TENANT_PROVENANCE_LICENSE = "license"
TENANT_PROVENANCE_DELEGATED = "delegated"
TENANT_PROVENANCE_ASSERTED = "asserted"
TENANT_PROVENANCE_DEFAULT = "default"
TENANT_PROVENANCE_BYPASS = "bypass"
TENANT_PROVENANCE_UNKNOWN = "unknown"
TENANT_PROVENANCES: tuple[str, ...] = (
    TENANT_PROVENANCE_LICENSE,
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_BYPASS,
    TENANT_PROVENANCE_UNKNOWN,
)
# The values for which the tenant was established by the caller's OWN
# credential. A frozenset of ONE today, and deliberately expressed as a
# membership test rather than `== TENANT_PROVENANCE_LICENSE`: a value added to
# the set above defaults to UNPROVEN unless it is explicitly listed here.
PROVEN_TENANT_PROVENANCES: frozenset[str] = frozenset({TENANT_PROVENANCE_LICENSE})

# Empty-string default — never None, never the literal "None" — so downstream
# callers that read these vars unconditionally get a header-safe value.
_tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")
_license_id_var: ContextVar[str] = ContextVar("license_id", default="")
_subject_id_var: ContextVar[str] = ContextVar("subject_id", default="")
# ⛔ DEFAULT IS `unknown`, NEVER `license`. See TENANT_PROVENANCE_UNKNOWN above.
_tenant_provenance_var: ContextVar[str] = ContextVar(
    "tenant_provenance", default=TENANT_PROVENANCE_UNKNOWN
)
# The outcome of THIS request's tenant byo_mode lookup (auth_middleware's
# `_record_byo_lookup` — the one chokepoint every lookup outcome already
# passes through). Set here so a route-level gate (`src/auth/tokenless_gate.py`)
# can tell "served, byo_mode genuinely resolved" apart from "served, licences
# was unresolved and this pod failed open inside the grace window" without a
# second HTTP round-trip or re-deriving it from the resolved mode string
# (grace-open and a genuine `local_allowed` resolve to the identical value,
# so the mode alone cannot distinguish them).
# ⛔ Deliberately NOT imported from `src.telemetry.agent_metrics` (that module
# imports FROM this one — `TENANT_PROVENANCE_UNKNOWN` above — so importing back
# would be circular). The value is one of `TENANT_BYO_LOOKUP_OUTCOMES`; this
# var only stores it, it does not validate it, same as `_caller_client_id_var`
# below. Empty-string default: "" means no lookup ran on this request at all,
# which is the common case (only the tokenless trusted-network branch looks
# this up).
_tenant_byo_lookup_outcome_var: ContextVar[str] = ContextVar(
    "tenant_byo_lookup_outcome", default=""
)
# Caller-identity (M3 of caller-identity-rollout): the upstream service that
# called this one, identified by its secure-apps Keycloak SA. Populated from
# x-caller-client-id + x-caller-roles request headers injected by license-authz
# after verifying the inbound X-Caller-Authorization JWT.
_caller_client_id_var: ContextVar[str] = ContextVar("caller_client_id", default="")
_caller_roles_var: ContextVar[tuple[str, ...]] = ContextVar("caller_roles", default=())
# Cost-correlation (fleet-wide X-Correlation-Id + optional X-Stage). Lets the
# caller (e.g. w2v_orchestrator stamping correlation_id=run_id, stage=DAG stage)
# attribute LLM cost events back to a single operation. Empty default keeps every
# emit/header read header-safe (no None). Orthogonal to auth — set once per request
# from inbound headers, independent of the token/trusted-network branch taken.
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
_stage_var: ContextVar[str] = ContextVar("stage", default="")
# Calling APPLICATION's own id (fleet-wide X-App-Id), distinct from
# caller_client_id (the calling MICROSERVICE's Keycloak SA) and from
# tenant/subject/license (WHO). This is WHICH APP the run belongs to —
# inference-router reads it (routers/completions.py) but never defaults it,
# because it fans in across 19 tenants and cannot guess. worker_management is
# a caller of inference-router with zero app_id plumbing today, so this
# ContextVar only ever holds what the INBOUND caller stamped on X-App-Id.
# ⛔ DO NOT default this to a non-empty value, here or anywhere it is read.
# worker_management itself fans in from many upstream callers (n8n,
# requirements_management, asset-evaluator, w2v_orchestrator, ...) — a
# guessed/defaulted app_id would collapse all of THEIR distinct origins into
# one bucket on every downstream cost event, which is worse than a NULL
# app_id (NULL is at least honestly "unknown"; a wrong default lies).
_app_id_var: ContextVar[str] = ContextVar("app_id", default="")
# Calling PROJECT's id (fleet-wide X-Project-Id) — the kanban project a run
# belongs to, so downstream spend can be priced per project. Orthogonal to
# app_id (WHICH APP) and to tenant/subject/license (WHO): one app serves many
# projects. Relayed verbatim from the inbound header, exactly like app_id.
# ⛔ DO NOT default this to a non-empty value, and DO NOT validate the uuid
# here: inference-router already validates X-Project-Id at its header-reading
# site (malformed → treated as absent), and a second validator is a second
# place to disagree. Empty default keeps the omit-when-empty rule below honest.
_project_id_var: ContextVar[str] = ContextVar("project_id", default="")
# ACTING person's id (fleet-wide X-Actor-Id) — WHO ACTED: the Keycloak `sub` of
# the person who initiated the work. Distinct from `_subject_id_var` (WHO PAYS:
# a service licence can pay for a human's call). inference-router reads it at
# its header site and stamps `actor_id` on the cost event; billing-service
# persists it to `billing_cost_events.actor_id` and rolls it up per member.
# VALIDATED HERE (owner ruling 2026-10-07): it must be a canonical, non-nil
# UUID (a Keycloak `sub`). A malformed value is DROPPED, never coerced — the
# accepted value is then forwarded UNCHANGED (same string, same case).
# ⛔ This ContextVar is never defaulted to the subject. It holds only what an
# inbound caller asserted. The OUTBOUND value is `get_effective_actor_id()`,
# which falls back to WM's OWN service-account sub (`SERVICE_ACTOR_ID`) when no
# caller actor exists: the owner ruling is that every cost event carries a real
# actor — the human when a human acted, otherwise the acting service's own sub —
# and never the NULL sentinel.
_actor_id_var: ContextVar[str] = ContextVar("actor_id", default="")
# Caller's JOB id (fleet-wide X-Job-Id) — the unit of work the caller is
# running (n8n execution, w2v run step, ...). Optional for the fleet. Relayed
# from the inbound header after a shape check (bounded, safe charset — it
# reaches a billing metadata column and a log line); anything else is dropped.
# The OUTBOUND / cost-event value is `get_effective_job_id(worker_name)`, which
# falls back to the WM worker name (WM always knows which job type ran).
_job_id_var: ContextVar[str] = ContextVar("job_id", default="")
# True while serving an AUTHENTICATED EXTERNAL request (licence-token, delegated
# or trusted-network caller), and in every task spawned from it. Owner decision
# 2026-10-07: such a request that brought no valid actor stays ACTOR-LESS
# downstream — it is NOT filled with SERVICE_ACTOR_ID, so the caller's gap stays
# visible to inference-router enforcement. Default False = WM-originated work.
_external_request_var: ContextVar[bool] = ContextVar("external_request", default=False)
# Spend SCOPE (fleet-wide X-Spend-Scope) — the SIBLING of `_project_id_var`, and
# a REPORTING dimension only. It exists so billing can tell two things apart
# that look identical today in the per-project coverage number:
#
#   project — this spend belongs to a project, and the caller says so. The
#             `X-Project-Id` that should accompany it is a SEPARATE fact; if it
#             is missing anyway, that is a genuine attribution FAILURE.
#   tenant  — this spend is tenant-wide work BY DESIGN and there is nothing to
#             attribute it to. A permanent, correct residual — not a failure.
#
# Without this axis every unattributed call lands in one bucket and the
# permanent residual reads as a defect nobody can ever close.
#
# ⛔ REPORTING ONLY, AND THAT CONTAINMENT IS THE WHOLE SAFETY ARGUMENT. Nothing
# may branch on this value: not pricing, not markup resolution, not routing, and
# ⛔ NOT whether `X-Project-Id` is forwarded. The two are ORTHOGONAL — a request
# carrying `X-Spend-Scope: tenant` AND a valid project id forwards BOTH, and the
# scope must never suppress the id. A forged or wrong scope may at worst
# mislabel a DENOMINATOR; it must never be able to move money. The moment
# anything downstream of this reads it to decide a charge, that argument is void
# and this header needs a real provenance story instead.
#
# ⛔ A CLOSED set of exactly two, and — UNLIKE `_project_id_var` directly above —
# validated HERE. That is a deliberate difference, not an inconsistency: the
# project id is an OPAQUE identifier whose only authority is inference-router's
# uuid check, so a second validator there would be a second place to disagree.
# A spend scope is a CLOSED VOCABULARY that reaches a metric label and a report
# dimension, so the unbounded-cardinality rule that governs
# `TENANT_PROVENANCES` applies and there is nothing for two services to disagree
# about: `tenant` and `project` are the entire universe. Anything else is DROPPED
# and LOGGED — never forwarded, and ⛔ never coerced to a default. Coercion is
# the one outcome that would defeat the purpose: silently minting `tenant` for a
# garbage value manufactures "correct by design" out of "we do not know", which
# is exactly the confusion this axis was added to end. Dropping leaves the call
# in the honest unknown bucket, where it is visible and reconcilable.
SPEND_SCOPE_TENANT = "tenant"
SPEND_SCOPE_PROJECT = "project"
SPEND_SCOPES: frozenset[str] = frozenset({SPEND_SCOPE_TENANT, SPEND_SCOPE_PROJECT})

#: Bounds on the invalid-scope WARNING in :func:`set_spend_scope_context`.
#:
#: ⛔ The value logged there is CALLER-CONTROLLED and the line fires once per
#: REQUEST, so neither bound is optional. `auth_middleware` already established
#: this pair for the same hazard (`_assertion_log_seen`, maxsize 2048 / 300s);
#: these are its numbers, deliberately, so the two sites cannot drift into
#: different answers to the same question.
_SPEND_SCOPE_LOG_MAX_CHARS = 64
_SPEND_SCOPE_LOG_TTL = 300
_spend_scope_log_seen: TTLCache = TTLCache(maxsize=2048, ttl=_SPEND_SCOPE_LOG_TTL)
# Empty default, exactly like every relayed header above: empty ⇒ OMITTED
# outbound, never sent as `X-Spend-Scope: ""`. Omission stays a first-class
# answer — a caller that says nothing about scope is not the same as one that
# says `tenant`, and must not be recorded as though it were.
_spend_scope_var: ContextVar[str] = ContextVar("spend_scope", default="")
# Caller-requested inference-router PROVIDER pin (inbound ``X-Provider``),
# forwarded to IR as ``X-Provider`` so the call rides IR's single-provider path
# (no failover). Relayed only when well-formed; empty ⇒ no header, as today.
_provider_pin_var: ContextVar[str] = ContextVar("provider_pin", default="")
PROVIDER_PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Prompt-replay identity (WS-1 of the asset-evaluator → WM → inference-router
# replay chain). Set from the inbound `X-Replay-Mode` header on send_message
# when asset-evaluator drives a prompt replay through WM. When true, the
# outbound inference-router httpx hook stamps `X-Replay-Mode: true` +
# `X-Worker-Version` (alongside the already-forwarded `X-Correlation-Id`) so IR
# writes a `replay_invocation_log` row. Bool default False keeps every normal
# (non-replay) request untouched — no replay headers are emitted.
_replay_mode_var: ContextVar[bool] = ContextVar("replay_mode", default=False)

# AA-TRACE join keys (spec AA-TRACE §2/§3/§6.5-7). Every worker_management →
# inference-router call forwards these UNCHANGED from the inbound request, in
# this exact case, so `inference_call` rows join back to the card/run that
# caused them. `X-Project-Id` is the fifth join key; it already has its own
# ContextVar (`_project_id_var`) and its own forwarding, so it is NOT repeated
# here — a second copy of one header is a second place for it to diverge.
#
# ⛔ RELAYED BYTE-FOR-BYTE, NEVER VALIDATED, NEVER DEFAULTED, NEVER INCREMENTED.
# Not even `X-AA-Hop`: the spec makes the hop count the CALLER's to stamp, and
# a WM that rewrote it would be a second author of a value the join depends on.
# Same omit-when-empty rule as every relayed header above: an absent inbound
# header is ABSENT outbound, never `""`. No length cap, matching every other
# relayed id in this file.
#
# worker_management OWNS the forwarding test for THIS hop
# (tests/test_outbound_trace_and_prompt_headers.py): the ship-tenancy lane's
# coverage stops at n8n → service and never sees the WM → IR leg.
AA_TRACE_FORWARD_HEADERS: tuple[str, ...] = (
    "X-AA-Trace-Id",
    "X-AA-Card-Id",
    "X-AA-Parent-Exec",
    "X-AA-Hop",
)
# One ContextVar holding the (header, value) pairs that were PRESENT inbound.
# A tuple, not a dict: ContextVar values must be treated as immutable, and a
# mutable default would be shared by every context that never set it.
_aa_trace_var: ContextVar[tuple[tuple[str, str], ...]] = ContextVar("aa_trace", default=())

# Prompt attribution (AA-TRACE §6.5-7; U6 contract "Attribution per call").
# Set by the worker LLM-call path immediately BEFORE each call, read by the
# outbound inference-router hook, which stamps `X-AA-Prompt-Version:
# <prompt_id>@<n>` and `X-AA-Prompt-Sha256: <hex>` so inference-router stores
# them in `inference_call.prompt_version_id` / `prompt_body_sha256`.
# ⛔ None ⇒ the header is OMITTED, never sent empty and never guessed: an
# unversioned (legacy-config) call must land as NULL so it is COUNTED as
# unversioned rather than laundered into an attributed-looking row.
# Reset to None per request by `AuthenticationMiddleware._stash_identity_context`
# so one request's prompt can never be attributed to the next request's call.
_prompt_version_id_var: ContextVar[str | None] = ContextVar("prompt_version_id", default=None)
_prompt_body_sha256_var: ContextVar[str | None] = ContextVar("prompt_body_sha256", default=None)

# END-USER PROOF on a DELEGATED request (owner ruling 2026-10-01). A delegating
# service account (svc-fr-backend, ...) re-scopes a request to a tenant with
# ``X-Effective-Tenant-Id``; that header is a CLAIM. When the caller also
# forwards the signed-in user's portal-realm JWT in ``X-End-User-Token`` and it
# verifies (signature via the portal JWKS, issuer, expiry) with a ``tenant_id``
# claim equal to the effective tenant, the middleware stamps that tenant here,
# together with the delegating subject that forwarded it. Only
# ``llm_config_resolver.tenant_key_entitled`` reads it, to let a delegated
# request decrypt THAT tenant's BYO key. ``""`` = no proof. Cleared by every
# ``set_identity_context`` call so a proof can never outlive its request.
_end_user_tenant_var: ContextVar[str] = ContextVar("end_user_tenant", default="")
_end_user_delegator_var: ContextVar[str] = ContextVar("end_user_delegator", default="")
# Why there is no proof (``absent``, ``invalid:ExpiredSignatureError``,
# ``tenant_mismatch``, ...) — surfaced in the BYO-key-withheld error so the
# caller is told what to fix. Never contains token material.
_end_user_proof_reason_var: ContextVar[str] = ContextVar("end_user_proof_reason", default="")
# The VERIFIED end-user token's own facts (owner requirement 2026-10-02),
# recorded whenever the forwarded ``X-End-User-Token`` passed signature, issuer
# and expiry on a delegated request, EVEN WHEN its ``tenant_id`` differs from
# the effective tenant. ⛔ These are NOT a tenant proof and
# ``tenant_key_entitled`` never reads them. Only two consumers exist:
# ``llm_config_resolver.owner_key_test_entitled`` (a portal ``super_admin`` may
# TEST a customer tenant's stored key, nothing else) and the owner audit log in
# ``tenant_config/_router.py``. Cleared by every ``set_identity_context``.
_end_user_sub_var: ContextVar[str] = ContextVar("end_user_sub", default="")
_end_user_token_tenant_var: ContextVar[str] = ContextVar("end_user_token_tenant", default="")
_end_user_super_admin_var: ContextVar[bool] = ContextVar("end_user_super_admin", default=False)
_end_user_verified_delegator_var: ContextVar[str] = ContextVar(
    "end_user_verified_delegator", default=""
)
# The INTROSPECTED licence subject of the service account that re-scoped this
# request with ``X-Effective-Tenant-Id`` (provenance ``delegated`` only). It is a
# verified fact (licenses introspection), unlike the effective tenant itself,
# which is the header claim. Read ONLY by ``tenant_proof.tenant_proof_refusal``
# when deciding whether WM may mint the END tenant's inference-router token for
# a delegating service on ``WM_DELEGATING_SUBJECTS`` (mirror of the router's
# ``AUTH_DELEGATING_SUBJECTS``). ``""`` = not delegated. Cleared by every
# ``set_identity_context``.
_delegating_subject_var: ContextVar[str] = ContextVar("delegating_subject", default="")


def set_identity_context(
    tenant_id: str | None,
    license_id: str | None,
    subject_id: str | None,
    *,
    tenant_provenance: str = TENANT_PROVENANCE_UNKNOWN,
) -> None:
    """Populate per-request identity ContextVars.

    Called by `AuthenticationMiddleware` once per request, after
    license-token introspection. None values become empty strings so
    downstream callers can use the values directly as header values without
    `None`-handling.

    ``tenant_provenance`` says HOW ``tenant_id`` was established — one of
    :data:`TENANT_PROVENANCES`. It is keyword-only so it can never be passed by
    accident in a positional slot, and its default is
    :data:`TENANT_PROVENANCE_UNKNOWN` rather than being REQUIRED because the
    safe direction matters more here than forced exhaustiveness: a caller that
    omits it is recorded as "nobody said", which is honest and reconcilable,
    whereas a required argument would only convert the same omission into a
    TypeError on a hot auth path. ⛔ Do not give it a default of
    ``TENANT_PROVENANCE_LICENSE`` under any circumstances — that is the one
    value that asserts a proof nobody performed, and it is read as such by
    ``telemetry/cost_events.py`` when it stamps a billing record.

    An unrecognised value is coerced to ``unknown``: this reaches a metric
    label, and the closed set is the only thing bounding its cardinality.
    """
    _tenant_id_var.set(tenant_id or "")
    _license_id_var.set(license_id or "")
    _subject_id_var.set(subject_id or "")
    _tenant_provenance_var.set(
        tenant_provenance if tenant_provenance in TENANT_PROVENANCES else TENANT_PROVENANCE_UNKNOWN
    )
    # A new identity never inherits an earlier request's end-user proof.
    _end_user_tenant_var.set("")
    _end_user_delegator_var.set("")
    _end_user_proof_reason_var.set("")
    _end_user_sub_var.set("")
    _end_user_token_tenant_var.set("")
    _end_user_super_admin_var.set(False)
    _end_user_verified_delegator_var.set("")
    _delegating_subject_var.set("")


def set_delegating_subject(subject: str | None) -> None:
    """Record the introspected subject that delegated (re-scoped) this request.

    Called by ``AuthenticationMiddleware`` on the delegated licence branch only,
    AFTER ``set_identity_context`` (which clears it).
    """
    _delegating_subject_var.set(subject or "")


def get_delegating_subject_from_context() -> str:
    """The verified delegating service-account subject, or ``""`` if not delegated."""
    return _delegating_subject_var.get()


def set_end_user_identity(
    *, sub: str, token_tenant: str, super_admin: bool, delegator: str
) -> None:
    """Record a VERIFIED end-user token's facts (signature, issuer, expiry passed).

    Called by ``end_user_token.stamp_end_user_proof`` on the delegated branch only.
    NOT a tenant proof: see the ContextVar comment above.
    """
    _end_user_sub_var.set(sub or "")
    _end_user_token_tenant_var.set(token_tenant or "")
    _end_user_super_admin_var.set(bool(super_admin))
    _end_user_verified_delegator_var.set(delegator or "")


def get_end_user_identity() -> tuple[str, str, bool, str]:
    """``(sub, token_tenant, is_super_admin, delegator)`` of the verified end-user token.

    All empty / False when no token verified on this request.
    """
    return (
        _end_user_sub_var.get(),
        _end_user_token_tenant_var.get(),
        _end_user_super_admin_var.get(),
        _end_user_verified_delegator_var.get(),
    )


def set_end_user_proof(*, tenant_id: str, delegator: str, reason: str) -> None:
    """Record the outcome of verifying a delegated request's ``X-End-User-Token``.

    ``tenant_id``/``delegator`` are set ONLY on success (``reason == "ok"``);
    pass ``""`` for both otherwise. Called by ``AuthenticationMiddleware`` on the
    delegated branch, after ``set_identity_context``.
    """
    _end_user_tenant_var.set(tenant_id or "")
    _end_user_delegator_var.set(delegator or "")
    _end_user_proof_reason_var.set(reason or "")


def get_end_user_proof() -> tuple[str, str]:
    """``(verified_tenant_id, delegating_subject)``; both ``""`` when unproven."""
    return _end_user_tenant_var.get(), _end_user_delegator_var.get()


def get_end_user_proof_reason() -> str:
    """Why the delegated request has (or lacks) an end-user proof; ``""`` = not checked."""
    return _end_user_proof_reason_var.get()


def set_caller_context(
    caller_client_id: str | None,
    caller_roles: tuple[str, ...] | None,
) -> None:
    """Populate per-request caller-identity ContextVars.

    Called by `AuthenticationMiddleware` from x-caller-client-id +
    x-caller-roles request headers (license-authz already verified the JWT
    that produced them upstream). Empty header → empty values.
    """
    _caller_client_id_var.set(caller_client_id or "")
    _caller_roles_var.set(caller_roles or ())


def set_correlation_context(
    correlation_id: str | None,
    stage: str | None,
) -> None:
    """Populate per-request cost-correlation ContextVars.

    Called by `AuthenticationMiddleware` from inbound `X-Correlation-Id` +
    `X-Stage` headers, independent of the auth branch taken. Empty header →
    empty value (opt-in: callers that don't stamp the header emit no
    correlation, preserving today's behavior).
    """
    _correlation_id_var.set(correlation_id or "")
    _stage_var.set(stage or "")


def apply_body_correlation_id(body_correlation_id: object) -> None:
    """Let a request-BODY ``correlation_id`` reach inference-router as ``X-Correlation-Id``.

    The outbound IR hook forwards whatever ``_correlation_id_var`` holds, which
    ``AuthenticationMiddleware`` fills from the inbound header. Resolution is the
    SAME rule graph execution uses (``first_valid_correlation_id``): a valid-UUID
    header wins; a malformed one (the n8n execution id ``529894`` was measured
    winning on 2026-08-14) does not beat a valid body value. A body value of None
    changes nothing, so every existing caller is untouched. The stage is kept.
    """
    if body_correlation_id is None:
        return
    # Local import: src.db must not become an import-time dependency of this
    # leaf module, which the auth middleware and the agent layer both import.
    from src.db.graph_execution_db import first_valid_correlation_id

    header_value = _correlation_id_var.get() or None
    resolved = first_valid_correlation_id(header_value, str(body_correlation_id))
    if resolved and resolved != header_value:
        _correlation_id_var.set(resolved)


def set_app_id_context(app_id: str | None) -> None:
    """Populate the per-request calling-application ContextVar.

    Called by `AuthenticationMiddleware` from the inbound `X-App-Id` header,
    independent of the auth branch taken (same shape as
    `set_correlation_context`). Empty header → empty value: worker_management
    never invents an app_id, it only ever relays what the caller sent.
    """
    _app_id_var.set(app_id or "")


def set_project_id_context(project_id: str | None) -> None:
    """Populate the per-request calling-project ContextVar.

    Called by `AuthenticationMiddleware` from the inbound `X-Project-Id`
    header, independent of the auth branch taken (same shape as
    `set_app_id_context`). Empty header → empty value: worker_management never
    invents a project_id, it only ever relays what the caller sent.
    """
    _project_id_var.set(project_id or "")


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$")


def valid_job_id(value: str | None) -> str:
    """Return ``value`` stripped if it is a bounded, safe-charset job id, else ``""``."""
    candidate = (value or "").strip()
    return candidate if _JOB_ID_RE.match(candidate) else ""


def set_job_id_context(job_id: str | None) -> None:
    """Populate the per-request caller job-id ContextVar (X-Job-Id), dropping junk."""
    _job_id_var.set(valid_job_id(job_id))


def get_job_id_from_context() -> str:
    """The inbound caller's validated X-Job-Id, or ``""``."""
    return _job_id_var.get()


def get_effective_job_id(worker_name: str = "") -> str:
    """The job id for outbound calls and WM's own cost events.

    The inbound caller's X-Job-Id when it sent a valid one, else the WM worker
    name (the job TYPE that ran). Empty only when neither is known.
    """
    return _job_id_var.get() or (worker_name or "")


def valid_project_id(value: str | None) -> str:
    """``value`` if it parses as a uuid, else ``""`` — inference-router's rule.

    IR (`completions._project_id_or_none`) treats a malformed X-Project-Id as
    absent rather than failing the call; WM's own cost events apply the SAME
    rule so the two meters cannot disagree about what a project id is.
    """
    candidate = (value or "").strip()
    if not candidate:
        return ""
    try:
        uuid.UUID(candidate)
    except ValueError:
        return ""
    return candidate


#: Canonical hyphenated UUID, any case. ``uuid.UUID()`` alone also accepts
#: braces, ``urn:uuid:`` and bare hex, which downstream parsers may not, so the
#: shape is pinned before the value is forwarded unchanged.
_ACTOR_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def valid_actor_id(value: str | None) -> str:
    """Return ``value`` unchanged if it is a canonical non-nil UUID, else ``""``."""
    candidate = (value or "").strip()
    if not candidate or not _ACTOR_ID_RE.match(candidate):
        return ""
    if candidate == NIL_TENANT_ID:
        # The nil uuid is the fleet's "nobody" sentinel, never an actor.
        return ""
    return candidate


def set_actor_id_context(actor_id: str | None) -> str:
    """Populate the per-request acting-person ContextVar; return the accepted value.

    Called by `AuthenticationMiddleware._relay_actor_id` from the inbound
    `X-Actor-Id` header ONLY for a caller trusted to assert it (a delegating
    subject or the trusted network); cleared for every other caller. A value
    that is not a canonical non-nil UUID is DROPPED (the var is cleared, ``""``
    returned). Empty header → empty value: worker_management never invents an
    inbound actor.
    """
    accepted = valid_actor_id(actor_id)
    _actor_id_var.set(accepted)
    return accepted


def set_spend_scope_context(spend_scope: str | None) -> None:
    """Populate the per-request spend-scope ContextVar, DROPPING anything invalid.

    Called by `AuthenticationMiddleware` from the inbound `X-Spend-Scope`
    header, independent of the auth branch taken (same shape as
    `set_project_id_context`). Empty header → empty value: worker_management
    never invents a scope.

    ⛔ The validation lives HERE, in the setter, rather than at the inbound
    header read — so there is exactly ONE validator, and a future second entry
    point (another middleware branch, a worker-dispatch endpoint reading the
    header directly) cannot get an unvalidated value into the ContextVar by
    forgetting to repeat the check. A validator at the call site is a validator
    that the next call site does not have.

    ⛔ An unrecognised value is DROPPED (→ empty → omitted outbound) and logged,
    NEVER coerced to `tenant` or any other default. See the ⛔ note on
    `_spend_scope_var` for why coercion is the one outcome that defeats the
    purpose of the axis. Matching is EXACT and case-SENSITIVE: `TENANT` is not
    `tenant`. This is a machine-to-machine wire value from a small set of known
    callers, not user input, so a caller sending the wrong case has a bug that
    should be visible in its own logs rather than papered over here — and a
    normalising `.lower()` is the first step onto the same slope as coercion.
    """
    value = spend_scope or ""
    if value and value not in SPEND_SCOPES:
        # WARNING, not debug: this is a caller bug that silently costs the
        # billing report a dimension, and it is invisible at the caller (the
        # request succeeds either way — the header is reporting-only). The value
        # is %r-quoted and the closed set named so the log line alone is enough
        # to fix the caller.
        #
        # ⛔ TRUNCATED AND DEDUPED, following `auth_middleware._assertion_log_seen`.
        # An earlier version of this comment said "low cardinality by
        # construction" and left the value untruncated and unthrottled. That was
        # true of CARDINALITY and false of VOLUME, which is the bound that
        # matters: the value is caller-controlled and this fires once per
        # REQUEST, so one misconfigured caller sending a large header writes an
        # unbounded line on every call into the single rotating file
        # `kubectl logs` serves — evicting every other workload's logs in the
        # pod. Bound it on bytes and on rate, not on distinct values.
        logged = value[:_SPEND_SCOPE_LOG_MAX_CHARS]
        if logged not in _spend_scope_log_seen:
            _spend_scope_log_seen[logged] = True
            logger.warning(
                "dropping invalid X-Spend-Scope=%r: not one of %s. The header is "
                "reporting-only, so the request proceeds unchanged and any "
                "X-Project-Id it carries is still forwarded — but this call will "
                "land in the unknown-scope bucket rather than being labelled. "
                "Fix the caller to send exactly one of the two values, or to omit "
                "the header entirely if it has nothing to say about scope.",
                logged,
                sorted(SPEND_SCOPES),
            )
        value = ""
    _spend_scope_var.set(value)


def valid_provider_pin(value: object) -> str:
    """``value`` when it is a well-formed IR provider name, else ``""`` (dropped)."""
    return value if isinstance(value, str) and PROVIDER_PIN_RE.match(value) else ""


def set_provider_pin_context(provider: str | None) -> None:
    """Set the per-request IR provider pin from an inbound ``X-Provider`` header.

    A malformed value is DROPPED (empty ⇒ no header forwarded), never repaired.
    """
    _provider_pin_var.set(valid_provider_pin(provider))


def get_provider_pin_from_context() -> str:
    """The caller's validated IR provider pin, or ``""`` when none was sent."""
    return _provider_pin_var.get()


def set_replay_mode_context(replay_mode: bool) -> None:  # noqa: FBT001
    """Populate the per-request prompt-replay-mode ContextVar.

    Called by the `send_message` endpoint from the inbound `X-Replay-Mode`
    header. Always set (True only when the header is literally "true"), so the
    overwhelming majority of requests explicitly reset it to False and a prior
    request's replay flag can never leak into a reused async context.
    """
    _replay_mode_var.set(replay_mode)


def is_replay_mode_from_context() -> bool:
    """Return whether the current request is a prompt replay (`X-Replay-Mode`)."""
    return _replay_mode_var.get()


def set_aa_trace_context(headers: Mapping[str, str]) -> None:
    """Capture the inbound AA-TRACE join-key headers into the ContextVar.

    Called by `AuthenticationMiddleware._stash_identity_context` with the
    inbound request's (case-insensitive) header mapping, for EVERY branch —
    always, so a header absent on this request clears whatever a previous
    request in a reused context held. Only present, non-empty headers are
    kept; the value is stored byte-for-byte. See the ⛔ note on
    `AA_TRACE_FORWARD_HEADERS`.
    """
    _aa_trace_var.set(
        tuple(
            (header, value)
            for header in AA_TRACE_FORWARD_HEADERS
            if (value := headers.get(header) or "")
        )
    )


def get_aa_trace_from_context() -> dict[str, str]:
    """The AA-TRACE headers present on the current inbound request (exact case)."""
    return dict(_aa_trace_var.get())


def set_prompt_attribution(
    prompt_version_id: str | None,
    prompt_body_sha256: str | None,
) -> None:
    """Record WHICH prompt the next inference-router call in this context sends.

    ``prompt_version_id`` is ``f"{prompt_id}@{version}"`` (or None for a legacy
    unversioned prompt); ``prompt_body_sha256`` is the hex sha256 of the
    rendered system prompt actually sent. Call it before EACH LLM call — the
    outbound hook reads whatever is current at send time. Empty string is
    normalised to None so it is omitted, never sent as an empty header.
    """
    _prompt_version_id_var.set(prompt_version_id or None)
    _prompt_body_sha256_var.set(prompt_body_sha256 or None)


def get_prompt_attribution_from_context() -> tuple[str | None, str | None]:
    """``(prompt_version_id, prompt_body_sha256)`` for the current context."""
    return _prompt_version_id_var.get(), _prompt_body_sha256_var.get()


def get_trace_and_prompt_headers() -> dict[str, str]:
    """The AA-TRACE + prompt-attribution outbound headers, omit-when-empty.

    The ONE place both outbound paths take these from — the pydantic-ai httpx
    hook (`make_identity_event_hook`) and `get_identity_headers` (CLI-agent /
    remote-container) — so the two transports cannot attribute the same call
    two different ways. `X-Project-Id` is deliberately not here; it is already
    stamped by both paths from `_project_id_var`.
    """
    headers = dict(_aa_trace_var.get())
    prompt_version_id, prompt_body_sha256 = get_prompt_attribution_from_context()
    if prompt_version_id:
        headers["X-AA-Prompt-Version"] = prompt_version_id
    if prompt_body_sha256:
        headers["X-AA-Prompt-Sha256"] = prompt_body_sha256
    return headers


def get_tenant_id_from_context() -> str:
    """Return the tenant id stored in the current request's contextvars."""
    return _tenant_id_var.get()


def get_license_id_from_context() -> str:
    """Return the license id stored in the current request's contextvars."""
    return _license_id_var.get()


def get_subject_id_from_context() -> str:
    """Return the subject id stored in the current request's contextvars."""
    return _subject_id_var.get()


def get_tenant_provenance_from_context() -> str:
    """Return HOW the current request's tenant was established.

    One of :data:`TENANT_PROVENANCES`; ``unknown`` when nothing stated it. Read
    by ``telemetry/cost_events.py`` so a billing record can say whether the
    tenant it charges was proven or merely adopted.
    """
    return _tenant_provenance_var.get()


def set_tenant_byo_lookup_outcome(outcome: str) -> None:
    """Record this request's tenant byo_mode lookup outcome.

    Called from exactly one place — ``auth_middleware._record_byo_lookup`` —
    so every lookup this pod makes (resolved, stale, grace_open, fail_closed)
    is visible here without a second call site to keep in sync.
    """
    _tenant_byo_lookup_outcome_var.set(outcome)


def get_tenant_byo_lookup_outcome_from_context() -> str:
    """Return this request's tenant byo_mode lookup outcome, or ``""``.

    ``""`` means no lookup ran on this request — the common case outside the
    tokenless trusted-network branch. Read by
    ``src/auth/tokenless_gate.py`` to tell a genuinely resolved
    ``local_allowed`` apart from one this pod fell open to inside the
    licences-unresolved grace window (see ``TENANT_BYO_LOOKUP_GRACE_OPEN``);
    the two produce the identical ``byo_mode`` string, so nothing else can.
    """
    return _tenant_byo_lookup_outcome_var.get()


def get_correlation_id_from_context() -> str:
    """Return the cost-correlation id stored in the current request's contextvars."""
    return _correlation_id_var.get()


def get_stage_from_context() -> str:
    """Return the cost-correlation stage stored in the current request's contextvars."""
    return _stage_var.get()


def get_app_id_from_context() -> str:
    """Return the calling application's id (X-App-Id) for the current request.

    Empty when the caller did not stamp X-App-Id — never a default/guessed
    value. See the ⛔ note on `_app_id_var` above for why.
    """
    return _app_id_var.get()


def get_project_id_from_context() -> str:
    """Return the calling project's id (X-Project-Id) for the current request.

    Empty when the caller did not stamp X-Project-Id — never a default/guessed
    value, and never validated here. See the ⛔ note on `_project_id_var`.
    """
    return _project_id_var.get()


def get_actor_id_from_context() -> str:
    """Return the acting person's id (X-Actor-Id) for the current request.

    Empty when the caller did not stamp X-Actor-Id — never defaulted. See the
    ⛔ note on `_actor_id_var`.
    """
    return _actor_id_var.get()


_service_actor_warned: set[str] = set()


def get_service_actor_id() -> str:
    """worker_management's OWN service-account sub (env ``SERVICE_ACTOR_ID``).

    Empty when unset or malformed (a malformed value is logged at ERROR once and
    treated as unset — never forwarded).
    """
    from src.settings import get_settings

    raw = get_settings().service_actor_id
    accepted = valid_actor_id(raw)
    if raw and not accepted and raw not in _service_actor_warned:
        _service_actor_warned.add(raw)
        logger.error(
            "SERVICE_ACTOR_ID is not a canonical non-nil UUID; ignored. "
            "WM-originated calls will carry no X-Actor-Id until it is fixed."
        )
    return accepted


def set_external_request_context(*, external: bool) -> None:
    """Mark the current context as serving an authenticated external caller."""
    _external_request_var.set(external)


def is_external_request_context() -> bool:
    """True inside an authenticated external request (or a task spawned from it)."""
    return _external_request_var.get()


def get_effective_actor_id() -> str:
    """The actor to put on every OUTBOUND call and on WM's own cost events.

    The inbound caller's validated ``X-Actor-Id`` (a human, or the calling
    service's own sub) when one was passed — bound to the request and so to any
    worker run it triggers. An authenticated EXTERNAL request without one stays
    actor-less (``""``) — never filled (owner decision 2026-10-07). Only
    WM-originated work (no inbound request: background, scheduled, internal
    workers) uses WM's own service account (``SERVICE_ACTOR_ID``).
    """
    actor = _actor_id_var.get()
    if actor:
        return actor
    if _external_request_var.get():
        # An external caller that sent no valid actor stays actor-less: filling
        # it with WM's own id would hide the caller's gap from enforcement.
        return ""
    return get_service_actor_id()


def get_spend_scope_from_context() -> str:
    """Return the spend scope (X-Spend-Scope) for the current request.

    One of :data:`SPEND_SCOPES`, or empty when the caller said nothing — or when
    it said something invalid, which `set_spend_scope_context` dropped. Those
    two collapse to the same empty value ON PURPOSE: both mean "nobody usably
    stated a scope", and inventing a third state here would only give the report
    a bucket it cannot act on. The DIFFERENCE between them is preserved where it
    is actionable — in the WARNING log, which names a caller to fix.

    ⛔ REPORTING ONLY. Do not branch on this to price, route, or decide whether
    a project id is forwarded. See the ⛔ note on `_spend_scope_var`.
    """
    return _spend_scope_var.get()


def get_identity_headers(worker_name: str = "") -> dict[str, str]:
    """The canonical outbound identity header dict for the current request.

    ⛔ OMIT-WHEN-EMPTY, NEVER EMPTY-STRING. The previous shape returned all five
    headers unconditionally, so an unresolved tenant went out as
    ``X-Tenant-Id: ""``. An empty header is a PRESENT header: the callee's
    ``Header(default=...)`` cannot fire, and inference-router coerces the empty
    string to the nil sentinel instead of seeing "absent" and refusing. This is
    the same rule -- and the same bug -- documented at length on
    ``make_identity_event_hook`` in ``pydantic_ai_agent.py``; that hook learned
    it the expensive way and this helper must not relearn it. An absent header
    is honest; an empty one is a claim that the tenant is nothing.

    The nil sentinel is dropped for the same reason: it is not a customer, and
    forwarding it launders an unattributable call into an attributed-looking
    one.

    ``X-App-Id`` and ``X-Effective-Tenant-Id`` are included here because the
    httpx hook already forwards both and a second outbound path that dropped
    them would attribute the same run two different ways depending on which
    transport it happened to take.

    ``worker_name`` is WHOSE WORK this is -- the unit of work inside THIS
    service, matching the ``X-Worker-Name`` the pydantic-ai clients already
    stamp and the ``worker_name`` field on our own cost events. It is a
    parameter rather than a ContextVar because it is a property of the caller,
    not of the request. Empty ⇒ omitted, never guessed.

    ⛔ This function does NOT return a licence credential, and OUTBOUND it must
    not: ``X-Tenant-Id`` carries the tenant UUID and nothing else. The
    ``X-Tenant-Id: Bearer <token>`` form is this service's INBOUND convention
    only (``auth_middleware.py``); applying it outbound is a defect that was
    live until 2026-09-09. inference-router has no introspection middleware,
    so it read the Bearer string as a malformed tenant, DISCARDED the cost
    event (~183/day unbilled), and logged the raw token — a credential leak.
    The licence travels separately in ``X-License-Id``, which IR actually
    reads and gates on (``require_licensed_caller``). ⛔ Do NOT "restore" the
    Bearer form here or in ``_tenant_header_value``, and never put a credential
    on ``Authorization`` either. ⚠️ The outbound credential this warned about no
    longer exists anywhere: ``outbound_license.py`` and the two per-tenant
    ``OUTBOUND_LICENSE_TOKEN_*`` env vars were RETIRED 2026-09-11 because the
    resolved token reached no wire. The INBOUND rule is unchanged and lives in
    ``auth_middleware._extract_license_token``.
    """
    ctx_tenant = _tenant_id_var.get()
    headers: dict[str, str] = {}
    if ctx_tenant and ctx_tenant != NIL_TENANT_ID:
        headers["X-Tenant-Id"] = ctx_tenant
        # The trusted effective-tenant side-channel, re-stamped so a downstream
        # hop that re-delegates attributes to the run tenant, not to the service
        # account it authenticates as.
        headers[EFFECTIVE_TENANT_HEADER] = ctx_tenant
    headers.update(
        {
            header: value
            for header, value in (
                ("X-License-Id", _license_id_var.get()),
                ("X-Subject-Id", _subject_id_var.get()),
                ("X-Correlation-Id", _correlation_id_var.get()),
                ("X-Stage", _stage_var.get()),
                # Relayed verbatim, NEVER defaulted: worker_management fans in
                # from many upstream apps and a guessed app_id collapses their
                # distinct origins into one bucket. See the ⛔ note on
                # `_app_id_var`.
                ("X-App-Id", _app_id_var.get()),
                # Relayed verbatim and UNVALIDATED: inference-router validates
                # X-Project-Id to a uuid at its own header-reading site and
                # drops it if malformed. Validating again here would be a
                # second place for the two services to disagree.
                ("X-Project-Id", _project_id_var.get()),
                # Reporting-only sibling of X-Project-Id, already validated to
                # the closed {tenant, project} set by
                # `set_spend_scope_context` (dropped → empty → omitted here by
                # the same rule as every header above). ⛔ Emitted INDEPENDENTLY
                # of X-Project-Id: the two are orthogonal facts and a request
                # carrying both must send both. Do NOT make either conditional
                # on the other.
                ("X-Spend-Scope", _spend_scope_var.get()),
                # WHO ACTED (per-member billing): the inbound actor unchanged,
                # else WM's own service-account sub. Never the subject. See the
                # ⛔ note on `_actor_id_var`.
                ("X-Actor-Id", get_effective_actor_id()),
                ("X-Worker-Name", worker_name),
            )
            if value
        }
    )
    # AA-TRACE join keys + prompt attribution, from the one shared builder so
    # this path and the httpx hook cannot drift. Already omit-when-empty.
    headers.update(get_trace_and_prompt_headers())
    return headers


def get_caller_client_id_from_context() -> str:
    """Return the calling service's Keycloak SA client_id (azp) for this request."""
    return _caller_client_id_var.get()


def get_caller_roles_from_context() -> tuple[str, ...]:
    """Return the calling service's secure-apps roles for this request."""
    return _caller_roles_var.get()
