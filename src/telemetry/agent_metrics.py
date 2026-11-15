"""Agent metrics telemetry client for sending metrics to Grafana.

This module provides OpenTelemetry metrics instrumentation for agent execution,
including token usage, execution time, and model performance metrics.
"""

import logging
import os
import time
from typing import Any

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# The closed provenance set lives beside the ContextVar that PRODUCES it, in
# `src.api.identity_context`, which imports nothing from `src` — so this
# direction is the only one and no cycle is possible. Imported rather than
# re-declared: two copies of a "closed" set is how a label quietly opens.
from src.api.identity_context import TENANT_PROVENANCE_UNKNOWN, TENANT_PROVENANCES

logger = logging.getLogger(__name__)

#: The ONE ``PrometheusMetricReader`` for this process.
#:
#: ⛔ CONSTRUCTING ONE REGISTERS A COLLECTOR ON prometheus_client's DEFAULT
#: REGISTRY. A second construction raises ``ValueError: Duplicated timeseries in
#: CollectorRegistry``. ``AgentMetricsClient`` is a lazy singleton in normal
#: operation, but tests build it directly and a reload would build it again, so
#: the reader is cached here rather than trusting that. Cached at module scope
#: (not on the class) because the registry it mutates is module-scoped too.
_PROMETHEUS_READER: PrometheusMetricReader | None = None


def _prometheus_reader() -> PrometheusMetricReader:
    """The process-wide Prometheus pull reader, built at most once."""
    global _PROMETHEUS_READER  # noqa: PLW0603 -- process-wide registry is global state
    if _PROMETHEUS_READER is None:
        _PROMETHEUS_READER = PrometheusMetricReader()
    return _PROMETHEUS_READER


#: Label value ``_route_shape`` returns when the router matched nothing. Kept in
#: sync with ``src.api.auth_middleware._UNMATCHED_PATH`` — duplicated rather than
#: imported because ``agent_metrics`` deliberately imports nothing from
#: ``src.api`` except the cycle-free ``identity_context`` (see the note above).
_UNMATCHED_PATH = "<unmatched>"

# Closed label set for ``worker_management_metering_events_dropped_total``.
# A closed set is what keeps ``reason`` a bounded label; it is also what makes
# pre-materialisation possible, so a zero is a measured zero rather than a
# missing series. See the block comment at the counter definition for why the
# three must never be collapsed into one.
METERING_DROP_REASON_NO_TENANT = "no_tenant"
METERING_DROP_REASON_UNCONFIGURED = "metering_unconfigured"
METERING_DROP_REASON_NO_EVENT_LOOP = "no_event_loop"
METERING_DROP_REASONS: tuple[str, ...] = (
    METERING_DROP_REASON_NO_TENANT,
    METERING_DROP_REASON_UNCONFIGURED,
    METERING_DROP_REASON_NO_EVENT_LOOP,
)

# ═══════════════════════════════════════════════════════════════════════════
# TENANT BYO-MODE LOOKUP: what this service DECIDED when licences could not
# answer. Closed label sets — these are metric label values.
#
# ``_lookup_tenant_byo_mode`` asks licences for a tenant's commercial posture
# and the answer decides whether a run may ride platform (owner-GPU) inference.
# Every branch of that call now books exactly one series here, so the
# unresolved RATE is knowable before anyone tightens the decision further.
#
# ⛔ ``outcome`` is the DECISION, not the failure. A failure that was covered by
# a warm last-known-good is a different event from one that fell through to a
# fabricated posture, and collapsing them would hide precisely the transition an
# operator needs to see during a licences outage.
#
#   resolved              — licences answered 200 and its value was used. The
#                           DENOMINATOR: without it "3 unresolved" is a number
#                           with no scale.
#   stale_last_known_good — licences was unreachable/non-200 and this pod still
#                           held a successful answer for this (tenant, product)
#                           inside TENANT_BYO_STALE_TTL. The tenant's OWN last
#                           posture was re-used, so a byo_required tenant stays
#                           denied through a blip instead of silently becoming
#                           local_allowed. No verdict was invented.
#   grace_open            — unresolved AND no last-known-good, inside the
#                           bounded grace window: the run proceeds on platform
#                           routing. THIS IS THE FAIL-OPEN, and it is now
#                           counted rather than silent. A standing non-zero rate
#                           here is unenforced byo_required.
#   fail_closed           — unresolved, no last-known-good, and the grace window
#                           has elapsed: the posture is fabricated as
#                           ``byo_required`` and the run is refused (402). Real
#                           denial of real traffic — alert on it, and read it
#                           together with ``failure``.
TENANT_BYO_LOOKUP_RESOLVED = "resolved"
TENANT_BYO_LOOKUP_STALE = "stale_last_known_good"
TENANT_BYO_LOOKUP_GRACE_OPEN = "grace_open"
TENANT_BYO_LOOKUP_FAIL_CLOSED = "fail_closed"
TENANT_BYO_LOOKUP_OUTCOMES: tuple[str, ...] = (
    TENANT_BYO_LOOKUP_RESOLVED,
    TENANT_BYO_LOOKUP_STALE,
    TENANT_BYO_LOOKUP_GRACE_OPEN,
    TENANT_BYO_LOOKUP_FAIL_CLOSED,
)

# WHY the posture was unresolvable. Two remedies, so two values — never one.
#
#   none      — nothing failed (pairs only with ``resolved``).
#   non_200   — licences answered, with something other than 200. On the
#                measured population this is overwhelmingly a 404 for a selector
#                that is not a registered tenant, i.e. a CALLER defect that is
#                permanent and will not heal on retry. Remedy: fix the caller.
#   transport — timeout, connection error, or an unparseable body. licences (or
#                the path to it) is unhealthy. Remedy: fix licences. This one
#                heals on its own; the other does not.
TENANT_BYO_LOOKUP_FAILURE_NONE = "none"
TENANT_BYO_LOOKUP_FAILURE_NON_200 = "non_200"
TENANT_BYO_LOOKUP_FAILURE_TRANSPORT = "transport"
TENANT_BYO_LOOKUP_FAILURES: tuple[str, ...] = (
    TENANT_BYO_LOOKUP_FAILURE_NONE,
    TENANT_BYO_LOOKUP_FAILURE_NON_200,
    TENANT_BYO_LOOKUP_FAILURE_TRANSPORT,
)

# ⛔ Only the REACHABLE pairs are pre-materialised — 7 of the 4×3 cross product,
# not 12. ``resolved`` can only ever carry ``failure=none`` and the three
# unresolved outcomes can never carry it, so seeding the other five would
# publish measured-looking zeros for questions this instrument never asks.
# Same rule, same reason, as WORKER_STATE_UNKNOWN_PAIRS below.
TENANT_BYO_LOOKUP_PAIRS: tuple[tuple[str, str], ...] = (
    (TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_NONE),
    *(
        (_outcome, _failure)
        for _outcome in (
            TENANT_BYO_LOOKUP_STALE,
            TENANT_BYO_LOOKUP_GRACE_OPEN,
            TENANT_BYO_LOOKUP_FAIL_CLOSED,
        )
        for _failure in (
            TENANT_BYO_LOOKUP_FAILURE_NON_200,
            TENANT_BYO_LOOKUP_FAILURE_TRANSPORT,
        )
    ),
)

# ═══════════════════════════════════════════════════════════════════════════
# TOKENLESS TENANT REQUEST SURFACE: originally the 8 routes measured by
# ``WorkerManagementTokenlessTenantSurfaceGrew`` (worker_management#233),
# single-sourced through ``src/auth/tokenless_gate.py``.
#
# ⚠️ ROUND 16 (worker_management CLAUDE.md billing-licensing-estate-repair
# lane A3): live measurement 2026-09-17 showed all 16 series of the ORIGINAL 8
# routes at 0 for 6h while the sibling
# ``worker_management_unauthenticated_tenant_assertion_total`` counter showed
# 697 tokenless-served requests on `/workers/{worker_name}/message` — a route
# this gate never touched. That route plus every other unwired tenant-reading
# route on the tokenless-eligible surface (message send, tenant-config LLM
# GET/PUT/DELETE, tenant-config modality CRUD + resolve, the 5 typed
# worker-creation convenience endpoints, graph execution, graph management,
# and graph-sessions correlation) were added below. The set is no longer "8"
# — it is now every route this gate is wired to, read from
# ``TOKENLESS_TENANT_REQUEST_ROUTES`` itself, not a remembered count.
#
# ``route`` is a FIXED name per route, never the raw path and never a caller-
# supplied path segment (worker_name, provider name outside the two named
# below, modality name) — the exact unbounded-cardinality mistake the sibling
# ``unauthenticated_tenant_assertion`` counter's ``path`` label used to make
# before it was rebound to the matched route TEMPLATE (see
# ``auth_middleware._route_shape``). Here there is no template to fall back
# on that is both bounded AND meaningful per-route, so every value is
# hand-enumerated instead — closed, and only grows when a route is added to
# ``TOKENLESS_TENANT_ROUTES`` in ``tokenless_gate.py``.
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_CATALOG = "tenant_config_llm_catalog"
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_ALL = "tenant_config_llm_all"
TOKENLESS_ROUTE_PROVIDERS_LIST = "providers_list"
TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER = "providers_models_inference_router"
TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM = "providers_models_vllm"
TOKENLESS_ROUTE_WORKERS_WORKER_STATUS = "workers_worker_status"
TOKENLESS_ROUTE_WORKERS_WORKER_STATES = "workers_worker_states"
TOKENLESS_ROUTE_WORKERS_STATES_LIST = "workers_states_list"

# --- Round 16 additions -----------------------------------------------------
# `/workers/{worker_name}/message` — the route the alert's own sibling counter
# showed was actually carrying tokenless traffic (697 served / 6h) while this
# gate measured 0. The single most important label in this round.
TOKENLESS_ROUTE_WORKERS_MESSAGE = "workers_message"

# `src/api/tenant_config/_router.py` — GET/PUT/DELETE `/tenant-config/llm`
# (the tenant-scoped default/tiered LLM config CRUD; `/llm/all` and
# `/llm/catalog` were already wired in PR #273).
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_GET = "tenant_config_llm_get"
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_PUT = "tenant_config_llm_put"
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_DELETE = "tenant_config_llm_delete"
# POST /tenant-config/llm/test (owner ruling 2026-10-01): the BYO key test.
TOKENLESS_ROUTE_TENANT_CONFIG_LLM_TEST = "tenant_config_llm_test"
# ⛔ GET /tenant-config/llm/audit is deliberately NOT in this set. It refuses
# every tokenless caller (``_router._require_licence_or_delegated_caller``), so
# it is not part of the tokenless surface and must never be counted on it. A
# label here would pre-seed a series that can only ever read 0.

# `src/api/tenant_config/_modality_router.py` — the modality-parameterised
# twin of the LLM router above. One label per VERB, not per (verb, modality):
# `modality` is a bounded path segment (`MODALITIES`,
# `src/db/tenant_modality_config_db.py`) but folding it into the label would
# multiply a 6-value closed set by 5 modalities for no operator benefit — the
# alert reads per-operation, not per-modality.
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_GET = "tenant_config_modality_get"
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_ALL = "tenant_config_modality_all"
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_CATALOG = "tenant_config_modality_catalog"
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_PUT = "tenant_config_modality_put"
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_DELETE = "tenant_config_modality_delete"
# `/tenant-config/modality/{modality}/resolve` — COUNT ONLY. This route has
# its own `identity_proven && tenant_proven` guard
# (`_withhold_reason`/`_key_for_caller` in `_modality_router.py`) that decides
# whether the decrypted BYO key is withheld from the response; the tokenless
# gate below is a `dependencies=[]` entry that never touches the response and
# (with `require_authenticated_tenant` staying False) never raises either, so
# wiring it here changes nothing about that guard or its response shape —
# see `tokenless_gate.py` module docstring and this round's PR description.
TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_RESOLVE = "tenant_config_modality_resolve"

# `src/api/workers/convenience_endpoints.py` — the 5 typed worker-creation
# shortcuts, each `Depends(get_tenant_context)` + a call into
# `create_worker_endpoint`.
TOKENLESS_ROUTE_WORKERS_CREATE_PYDANTIC_AI = "workers_create_pydantic_ai"
TOKENLESS_ROUTE_WORKERS_CREATE_CLAUDE_CODE = "workers_create_claude_code"
TOKENLESS_ROUTE_WORKERS_CREATE_CODEX = "workers_create_codex"
TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CLAUDE = "workers_create_remote_claude"
TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CODEX = "workers_create_remote_codex"

# `src/api/graph_execution_api.py` — 4 of its 5 routes; `GET /execution/health`
# takes no `ctx`/`get_tenant_context` dependency at all (a health probe, same
# exemption class as `/health`/`/healthz`/`/ready`) so it is deliberately left
# unwired — see `tests/test_auth/test_tokenless_gate.py::TestWiring` for the
# recorded reason.
TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE = "graph_execution_execute"
TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE = "graph_execution_execute_stepwise"
TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSION_STATUS = "graph_execution_session_status"
TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSIONS_LIST = "graph_execution_sessions_list"

# `src/api/graph_management_api.py` — all 11 routes read `ctx.tenant_id`.
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_CREATE = "graph_management_create"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_LIST = "graph_management_list"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES = "graph_management_delete_all_states"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL = "graph_management_delete_all"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_GET = "graph_management_get"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_UPDATE = "graph_management_update"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_STATES = "graph_management_delete_states"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE = "graph_management_delete"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STRUCTURE = "graph_management_structure"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY = "graph_management_state_history"
TOKENLESS_ROUTE_GRAPH_MANAGEMENT_VALIDATE = "graph_management_validate"

# `src/api/graph_sessions_correlation_api.py` — its one route.
TOKENLESS_ROUTE_GRAPH_SESSIONS_CORRELATION = "graph_sessions_correlation"

# --- Round 16, PR #276 audit follow-up --------------------------------------
# The audit that found `/workers/{worker_name}/message` unwired (above) did
# not itself sweep the REST of `src/api/workers/crud_endpoints.py`,
# `state_endpoints.py`, and `src/api/worker_states.py`. A second, exhaustive
# sweep for `Depends(get_tenant_context)` not already in
# `TOKENLESS_TENANT_REQUEST_ROUTES` found 13 more. One label per route.
TOKENLESS_ROUTE_WORKERS_CREATE = "workers_create"
TOKENLESS_ROUTE_WORKERS_LIST = "workers_list"
TOKENLESS_ROUTE_WORKERS_TYPES_LIST = "workers_types_list"
TOKENLESS_ROUTE_WORKERS_UPDATE = "workers_update"
TOKENLESS_ROUTE_WORKERS_DELETE_ALL = "workers_delete_all"
TOKENLESS_ROUTE_WORKERS_DELETE = "workers_delete"
TOKENLESS_ROUTE_WORKERS_PROMPT_SWAP = "workers_prompt_swap"
TOKENLESS_ROUTE_WORKERS_ALL_STATES = "workers_all_states"
TOKENLESS_ROUTE_WORKERS_ALL_STATES_DELETE = "workers_all_states_delete"
TOKENLESS_ROUTE_WORKERS_STATE_LATEST = "workers_state_latest"
TOKENLESS_ROUTE_WORKERS_STATES_DELETE = "workers_states_delete"
TOKENLESS_ROUTE_WORKERS_INSTANCE_CLEANUP = "workers_instance_cleanup"
TOKENLESS_ROUTE_WORKERS_STATES_CREATE = "workers_states_create"

TOKENLESS_TENANT_REQUEST_ROUTES: tuple[str, ...] = (
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_CATALOG,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_ALL,
    TOKENLESS_ROUTE_PROVIDERS_LIST,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM,
    TOKENLESS_ROUTE_WORKERS_WORKER_STATUS,
    TOKENLESS_ROUTE_WORKERS_WORKER_STATES,
    TOKENLESS_ROUTE_WORKERS_STATES_LIST,
    TOKENLESS_ROUTE_WORKERS_MESSAGE,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_GET,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_PUT,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_DELETE,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_TEST,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_GET,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_ALL,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_CATALOG,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_PUT,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_DELETE,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_RESOLVE,
    TOKENLESS_ROUTE_WORKERS_CREATE_PYDANTIC_AI,
    TOKENLESS_ROUTE_WORKERS_CREATE_CLAUDE_CODE,
    TOKENLESS_ROUTE_WORKERS_CREATE_CODEX,
    TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CLAUDE,
    TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CODEX,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSION_STATUS,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSIONS_LIST,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_CREATE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_LIST,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_GET,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_UPDATE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_STATES,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STRUCTURE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_VALIDATE,
    TOKENLESS_ROUTE_GRAPH_SESSIONS_CORRELATION,
    TOKENLESS_ROUTE_WORKERS_CREATE,
    TOKENLESS_ROUTE_WORKERS_LIST,
    TOKENLESS_ROUTE_WORKERS_TYPES_LIST,
    TOKENLESS_ROUTE_WORKERS_UPDATE,
    TOKENLESS_ROUTE_WORKERS_DELETE_ALL,
    TOKENLESS_ROUTE_WORKERS_DELETE,
    TOKENLESS_ROUTE_WORKERS_PROMPT_SWAP,
    TOKENLESS_ROUTE_WORKERS_ALL_STATES,
    TOKENLESS_ROUTE_WORKERS_ALL_STATES_DELETE,
    TOKENLESS_ROUTE_WORKERS_STATE_LATEST,
    TOKENLESS_ROUTE_WORKERS_STATES_DELETE,
    TOKENLESS_ROUTE_WORKERS_INSTANCE_CLEANUP,
    TOKENLESS_ROUTE_WORKERS_STATES_CREATE,
)
# ⚠️ ``providers_models_other`` is a REACHABLE 9th value (any provider name on
# ``GET /providers/{name}/models`` other than the two the alert named), kept
# out of the pre-seeded/enumerated 8 on purpose — see the pre-materialisation
# comment beside the counter below for why.
TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER = "providers_models_other"

#   served      — the request reached this per-route gate and was not
#                  refused HERE. That is all it means: it says nothing about
#                  the eventual response, including a later 500/502.
#   refused     — would mean ``require_authenticated_tenant`` caused a 401
#                  HERE, but ``AuthenticationMiddleware`` already refuses an
#                  asserted tenant before routing when the flag is on (see
#                  ``tokenless_gate.py``'s module docstring), so on live
#                  traffic this label is unreachable and reads 0 on every
#                  route forever — it is not a measurement of anything. A
#                  real refusal shows up on
#                  ``worker_management_unauthenticated_tenant_assertion_total``
#                  instead.
#   grace_open  — served, AND this request's tenant byo_mode lookup itself
#                  fell open inside the licences-unresolved grace window (see
#                  ``TENANT_BYO_LOOKUP_GRACE_OPEN`` above). A distinct value,
#                  not folded into ``served``: this is the compound case where
#                  an unproven tenant assertion ALSO rode a fail-open BYO
#                  posture, and collapsing it into plain ``served`` would hide
#                  exactly the traffic an operator most needs to see before
#                  arming the flag.
TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED = "served"
TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED = "refused"
TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN = TENANT_BYO_LOOKUP_GRACE_OPEN
TOKENLESS_TENANT_REQUEST_OUTCOMES: tuple[str, ...] = (
    TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED,
    TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED,
    TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN,
)

# Why a decrypted BYO provider key was withheld from a
# /tenant-config/modality/{modality}/resolve response. CLOSED on purpose —
# these are metric label values.
#
#   unproven_identity — the caller reached the handler on the tokenless
#                       trusted-network branch (or on any branch that did not
#                       verify a licence token), so the tenant it named is its
#                       own claim. A key exists; it was not returned.
#   adopted_tenant    — the caller DID prove who it is (a verified licence
#                       token) but the tenant it is scoped to is not the one
#                       that licence established: it is a delegating service
#                       account that re-scoped the request with
#                       ``X-Effective-Tenant-Id``, a header validated on
#                       SPELLING alone. Proving you are svc-kag is not proving
#                       you may hold tenant X's provider credential, so these
#                       are counted APART from ``unproven_identity`` — the two
#                       populations have different owners and different fixes
#                       (give the caller a token vs. give the caller a token
#                       FOR THAT TENANT), and merging them would make the
#                       delegated population invisible inside the tokenless one.
#   indeterminate     — the handler could not establish WHETHER the identity
#                       was proven. Withheld anyway (fail closed), and kept as
#                       its own value so "we chose not to" never hides inside
#                       "we could not tell". ⚠️ This value is expected to stay
#                       at ZERO in production, because both flags are dataclass
#                       fields on ``TenantContext`` and therefore always
#                       present. It is kept deliberately: it is the reason a
#                       PARTIAL context shape — one that carries
#                       ``identity_proven`` but not ``tenant_proven``, exactly
#                       what a rollback to the previous release looks like to a
#                       peer that imports this module — withholds instead of
#                       being read as entitled. A nonzero rate here is a
#                       deployment-shape defect, not a caller defect.
MODALITY_KEY_WITHHELD_UNPROVEN = "unproven_identity"
MODALITY_KEY_WITHHELD_ADOPTED_TENANT = "adopted_tenant"
MODALITY_KEY_WITHHELD_INDETERMINATE = "indeterminate"
# D5: an owner_provided (managed-BYO) key requested by anything other than a
# verified, listed service caller (the tenant's own licence token included).
MODALITY_KEY_WITHHELD_OWNER_KEY = "owner_key_not_service_caller"
MODALITY_KEY_WITHHELD_REASONS: tuple[str, ...] = (
    MODALITY_KEY_WITHHELD_UNPROVEN,
    MODALITY_KEY_WITHHELD_ADOPTED_TENANT,
    MODALITY_KEY_WITHHELD_INDETERMINATE,
    MODALITY_KEY_WITHHELD_OWNER_KEY,
)

# How the tenant on a /api/workflow-routing/* request was established. CLOSED on
# purpose — these are metric label values, so an open-ended string here would be
# the unbounded-cardinality hole the tenant label itself is carefully avoiding.
#
#   license   — a licence token was introspected AND the tenant it established
#               is the one the request is scoped to. This is the ONLY value for
#               which the ``tenant`` label carries a real uuid, because it is the
#               only path on which the tenant was proven rather than asserted.
#   delegated — a licence token was introspected (so WHO the caller is IS
#               proven) but the request was then re-scoped to a different
#               tenant via ``X-Effective-Tenant-Id``, a header
#               ``_effective_tenant`` validates on SPELLING alone. Split out of
#               ``asserted`` for exactly the reason
#               ``MODALITY_KEY_WITHHELD_ADOPTED_TENANT`` above is split out of
#               ``MODALITY_KEY_WITHHELD_UNPROVEN``: the two populations have
#               different owners and different fixes ("present a token" vs
#               "present a token FOR THAT TENANT"), and merging them makes the
#               delegated one invisible inside the tokenless one. It cannot be
#               re-derived from ``license_id``, because
#               ``AuthenticationMiddleware`` NULLS that on this branch on
#               purpose (customer spend must not be stamped on the service
#               account's entitlement). ⛔ The ``tenant`` label reads
#               ``<unproven>`` here: the adopted value is caller-chosen, so
#               admitting it would be the unbounded-cardinality hole the
#               ``asserted`` collapse exists to close.
#   asserted  — the caller sent a bare X-Tenant-Id on the trusted-network branch,
#               i.e. it chose its own tenant and nothing verified the claim. The
#               value is NOT put in the label; ``tenant`` reads "<unproven>".
#   default   — no tenant was supplied at all, so the request fell back to this
#               deployment's DEFAULT_TENANT_ID. A routing MUTATION arriving this
#               way is its own finding: the caller is sending no licence token.
#   none      — no tenant could be established (auth disabled, or an unmatched
#               request). Present so "we never saw this" is distinguishable from
#               "the branch does not exist".
WORKFLOW_ROUTING_AUTH_PATH_LICENSE = "license"
WORKFLOW_ROUTING_AUTH_PATH_DELEGATED = "delegated"
WORKFLOW_ROUTING_AUTH_PATH_ASSERTED = "asserted"
WORKFLOW_ROUTING_AUTH_PATH_DEFAULT = "default"
WORKFLOW_ROUTING_AUTH_PATH_NONE = "none"
WORKFLOW_ROUTING_AUTH_PATHS: tuple[str, ...] = (
    WORKFLOW_ROUTING_AUTH_PATH_LICENSE,
    WORKFLOW_ROUTING_AUTH_PATH_DELEGATED,
    WORKFLOW_ROUTING_AUTH_PATH_ASSERTED,
    WORKFLOW_ROUTING_AUTH_PATH_DEFAULT,
    WORKFLOW_ROUTING_AUTH_PATH_NONE,
)

# The sentinel that replaces a caller-asserted tenant in the label.
WORKFLOW_ROUTING_TENANT_UNPROVEN = "<unproven>"

# ─────────────────────────────────────────────────────────────────────────────
# Worker state-machine observability. CLOSED label sets, for the same
# bounded-cardinality reason as everything above.
#
# ⛔ BOTH counters below are OBSERVE-ONLY AND MUST STAY THAT WAY. The worker
# state machine currently accepts every transition: ``validate_state_transition``
# fails OPEN, and three of the four write paths never call it at all. Turning
# that into a refusal was measured at ~142,560 rejected invocations / 30 days,
# ≈55% of all traffic. So these count what already happens; nothing here may
# ever decide an outcome.
#
# The verdicts a single ``validate_state_transition`` call can reach. The three
# ``unknown_*`` arms are the FAIL-OPEN path — the function returns ``True`` on
# all of them, which is precisely the behaviour that has never been measured.
# They are kept apart rather than collapsed into one "unknown" because they need
# different fixes: an unknown *source* is a value some writer already persisted,
# an unknown *target* is a value a writer is about to persist, and "both" is a
# state machine that has been bypassed end to end.
WORKER_STATE_VERDICT_VALID = "valid"
WORKER_STATE_VERDICT_INVALID = "invalid_transition"
WORKER_STATE_VERDICT_UNKNOWN_FROM = "unknown_from"
WORKER_STATE_VERDICT_UNKNOWN_TO = "unknown_to"
WORKER_STATE_VERDICT_UNKNOWN_BOTH = "unknown_both"
WORKER_STATE_VERDICTS: tuple[str, ...] = (
    WORKER_STATE_VERDICT_VALID,
    WORKER_STATE_VERDICT_INVALID,
    WORKER_STATE_VERDICT_UNKNOWN_FROM,
    WORKER_STATE_VERDICT_UNKNOWN_TO,
    WORKER_STATE_VERDICT_UNKNOWN_BOTH,
)

# The sentinel for a persisted ``state_type`` that is not a WorkerStateType
# member. ⛔ THIS IS WHAT KEEPS ``state_type_class`` BOUNDED. ``worker_states``
# .``state_type`` is ``text NOT NULL`` with no CHECK and no PG enum, so the
# column holds whatever a caller wrote; a live census of 1,092,122 rows
# (2026-09-09) found ``config-updated`` (153,087 — note the HYPHEN),
# ``updated`` (3,448) and ``prompt-swapped``. ⚠️ ``config-updated`` became an
# enum member on 2026-09-13 and ``updated`` on 2026-09-17; both now classify as
# THEMSELVES. ``prompt-swapped`` is the last of the three still collapsing here,
# and is therefore the only remaining live exemplar of a non-member.
# Labelling the raw value would hand cardinality to whatever a writer invents.
WORKER_STATE_TYPE_UNKNOWN = "<unknown>"

# The known ``state_type`` values, i.e. the WorkerStateType members. Duplicated
# here ON PURPOSE rather than imported: ``worker_state_manager`` emits the
# verdict counter above and therefore imports this module, so importing the enum
# back would be a cycle. ``test_agent_metrics`` asserts this tuple equals the
# enum, so the duplication cannot drift silently.
WORKER_STATE_TYPE_CLASSES: tuple[str, ...] = (
    "created",
    "processing",
    "completed",
    "error",
    "idle",
    "active",
    "default",
    "config-updated",
    # ⭐ Added 2026-09-17 with the enum member (#256 lane / follows #248).
    "updated",
    WORKER_STATE_TYPE_UNKNOWN,
)

# ─────────────────────────────────────────────────────────────────────────────
# The (from, to) PAIR vocabulary for the fail-open arm.
#
# ⭐ WHY THIS IS NOT THE ENUM, AND IS DELIBERATELY WIDER THAN IT. The question
# this vocabulary exists to answer is: "if a candidate were added to
# ``WorkerStateType`` tomorrow, which pairs that return ``True`` today would
# start returning ``False``?" Every such pair must already have a
# ``STATE_TRANSITIONS`` entry or the enum change is a silent behaviour change.
# That question is UNANSWERABLE in the enum's own vocabulary, and also in
# ``WORKER_STATE_TYPE_CLASSES``, because both collapse a non-member to
# ``"<unknown>"`` — a pair labelled ``("<unknown>", "<unknown>")`` cannot be
# joined against ``STATE_TRANSITIONS`` at all.
#
# ⭐ IT HAS NOW ANSWERED IT TWICE. The census graded ``config-updated``: 72
# series over 24h, exactly one non-zero, ``(config-updated, processing)`` at
# 4,615 of 4,616 unknown verdicts. It was made a member on 2026-09-13 with the
# matching ``STATE_TRANSITIONS`` entry. It then graded ``updated``:
# ``(updated, processing)`` = 8 over 7d and 2 over 48h — small, but NON-ZERO,
# which is the bar this vocabulary was built to test, and it was promoted on
# 2026-09-17 on the same rule.
#
# ⚠️ SO THIS TUPLE IS NOW EXACTLY THE 9 ENUM MEMBERS, with
# ``WORKER_STATE_PAIR_CANDIDATES`` empty. It is momentarily NOT larger than the
# enum, and that is success rather than drift: the vocabulary is wider than the
# enum WHENEVER a promotion is under evaluation, and there is none today. The
# per-side count is unchanged at 11 (9 known + 2 sentinels) because a promoted
# value moves between the two halves rather than being added to their sum.
#
# ⛔ It is an ALLOW-LIST, not a passthrough, and that is the entire cardinality
# bound. ``worker_states.state_type`` is ``text NOT NULL`` with no CHECK, and
# both arguments to ``validate_state_transition`` are caller-supplied, so
# anything outside it collapses to ``"other"``. The 2026-09-09 census of
# 1,092,122 rows already found a third non-member, ``prompt-swapped``; it books
# as ``"other"`` ON PURPOSE. A value earns a label of its own only when someone
# proposes making it an enum member — that proposal is what this census grades.
WORKER_STATE_PAIR_OTHER = "other"
WORKER_STATE_PAIR_NONE = "none"

# The values STILL under evaluation for enum membership. ⚠️ ``config-updated``
# carries a HYPHEN and ``updated`` does not; they were two distinct live values
# (153,087 and 3,448 rows), not two spellings of one. ⛔ A promoted value must be
# REMOVED from here: it arrives via ``_WORKER_STATE_ENUM_VALUES`` and leaving it
# would duplicate it in ``WORKER_STATE_PAIR_KNOWN``, silently breaking the
# 11-per-side cardinality bound below.
#
# ⭐ EMPTY as of 2026-09-17, and an empty tuple is the RIGHT state rather than a
# defect: ``config-updated`` was promoted on 2026-09-13 and ``updated`` on
# 2026-09-17, so the census has now graded both values it was built to grade.
# ⚠️ ``prompt-swapped`` is deliberately NOT here. It is in the ROW census and has
# still never been seen transitioning, and this tuple is for a value somebody is
# actively PROPOSING to promote — adding it on row count alone is exactly the
# inference ``WORKER_STATE_PAIR_KNOWN``'s own comment forbids. It books as
# ``"other"``, which is a real, readable observation, and it earns a label of its
# own the day a promotion is proposed for it.
WORKER_STATE_PAIR_CANDIDATES: tuple[str, ...] = ()

# The enum members, as strings, derived from the tuple above so the two cannot
# drift apart. ``test_state_type_classes_match_the_enum`` pins that tuple to the
# enum, so pinning this to that tuple transitively pins it to the enum too.
_WORKER_STATE_ENUM_VALUES: tuple[str, ...] = tuple(
    _c for _c in WORKER_STATE_TYPE_CLASSES if _c != WORKER_STATE_TYPE_UNKNOWN
)

# The values that reach a pair label as themselves.
WORKER_STATE_PAIR_KNOWN: tuple[str, ...] = (
    *_WORKER_STATE_ENUM_VALUES,
    *WORKER_STATE_PAIR_CANDIDATES,
)

# Everything either side of a pair label can hold: 9 known + 2 sentinels = 11,
# so 11 x 11 = 121 is the HARD ceiling on this counter's series count.
WORKER_STATE_PAIR_VALUES: tuple[str, ...] = (
    *WORKER_STATE_PAIR_KNOWN,
    WORKER_STATE_PAIR_NONE,
    WORKER_STATE_PAIR_OTHER,
)

# The pairs this counter can actually reach, and therefore the exact set that is
# pre-materialised at zero. The counter fires ONLY on the ``unknown_*`` arm, so
# at least one side must have failed ``WorkerStateType(...)``: a pair whose two
# sides are both enum members is unreachable here by construction, and seeding
# it would publish 81 measured-looking zeros for a question this instrument
# never asks. 121 - 81 = 40.
#
# ⚠️ THIS NUMBER HAS NOW ROTTED TWICE IN FOUR DAYS — it read 72, then 57, and is
# 40 as of 2026-09-17. Every promotion into ``WorkerStateType`` shrinks it by
# construction (``2n+1`` pairs leave the reachable set for the n+1'th member),
# which is the whole point: those calls no longer reach the fail-open arm at all.
# ⛔ Do not quote a remembered figure here or in a test — derive it. The pairs
# that leave this set are exactly the ones that arrive on
# ``WORKER_STATE_INVALID_PAIRS`` below, and the two must always sum to 121;
# ``test_invalid_pair_zeros_cover_exactly_the_81_enum_pairs`` asserts that sum
# rather than either half alone, so a future promotion fails ONE arithmetic
# assertion instead of silently making both halves wrong.
WORKER_STATE_UNKNOWN_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (_from, _to)
    for _from in WORKER_STATE_PAIR_VALUES
    for _to in WORKER_STATE_PAIR_VALUES
    if not (_from in _WORKER_STATE_ENUM_VALUES and _to in _WORKER_STATE_ENUM_VALUES)
)

# The pairs the INVALID-arm census can reach, and the exact set it
# pre-materialises at zero. Precisely the complement of the tuple above: a
# ``invalid_transition`` verdict is only reachable once BOTH sides resolved to
# ``WorkerStateType`` members, so neither sentinel can ever appear on this
# counter and it is exactly ``9 x 9 = 81``.
#
# ⚠️ IT IS THE COMPLEMENT, NOT THE COMPLEMENT-MINUS-THE-VALID-ONES, and that is a
# deliberate over-seed of ~26 series that can never move. Narrowing it would
# mean importing ``STATE_TRANSITIONS`` from ``src.services.worker_state_manager``
# — which imports THIS module, so it is a cycle — or re-declaring the transition
# table here, which is a second source of truth that drifts silently the first
# time somebody widens an entry. A zero that means "this pair is legal" is
# readable from ``STATE_TRANSITIONS`` in one glance; a wrong transition table in
# the telemetry module is not detectable at all. The cheaper wrong is the
# obvious one.
WORKER_STATE_INVALID_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (_from, _to) for _from in _WORKER_STATE_ENUM_VALUES for _to in _WORKER_STATE_ENUM_VALUES
)

# What is actually known about the transition behind a write. THREE-valued, and
# strings rather than bools for a reason bigger than rendering: a two-valued
# ``validated`` cannot be read honestly.
#
#   "true"            — a ``validate_state_transition`` call ACTUALLY RAN for
#                       this write and returned. Coverage, measured.
#   "false"           — ⭐ THE EXPOSURE NUMBER. No call ran AND nothing about the
#                       site proves the transition good: the from-state is
#                       unknown, unread, or read and then defaulted away. This
#                       is the series to sum when sizing what closing the
#                       fail-open would cost, and the only one that means "the
#                       state machine never saw this".
#   "by_construction" — ⛔ NOT exposure, and must never be added to it. No call
#                       ran, but the from-state is KNOWN at the site and the
#                       transition is provably in STATE_TRANSITIONS. The
#                       completion write is the whole population of this today:
#                       the handler wrote the ``state_type="processing"`` row
#                       itself moments earlier, and PROCESSING -> COMPLETED is
#                       the first entry of ``STATE_TRANSITIONS[PROCESSING]``.
#
# Booking those ~250k provably-safe completions as "true" would over-report the
# validator's own coverage — the flattering number this counter exists to
# prevent — and booking them as "false" would invent an exposure that is not
# there. Hence three values, not two. Cardinality stays bounded: 9 state classes
# x 3 = 27 series, all pre-materialised at zero. (24 before ``config-updated``
# became the 8th enum member on 2026-09-13 and so earned a class of its own.)
WORKER_STATE_WRITE_VALIDATED_TRUE = "true"
WORKER_STATE_WRITE_VALIDATED_FALSE = "false"
WORKER_STATE_WRITE_VALIDATED_BY_CONSTRUCTION = "by_construction"
WORKER_STATE_WRITE_VALIDATED: tuple[str, ...] = (
    WORKER_STATE_WRITE_VALIDATED_TRUE,
    WORKER_STATE_WRITE_VALIDATED_FALSE,
    WORKER_STATE_WRITE_VALIDATED_BY_CONSTRUCTION,
)


def classify_worker_state_type(state_type: str | None) -> str:
    """Collapse a raw ``state_type`` to a bounded label value.

    Returns the value itself when it is a known :data:`WORKER_STATE_TYPE_CLASSES`
    member, else :data:`WORKER_STATE_TYPE_UNKNOWN`. ⛔ Never label the raw value:
    the column is free text and the sentinel is the whole cardinality bound.
    """
    if state_type in WORKER_STATE_TYPE_CLASSES and state_type != WORKER_STATE_TYPE_UNKNOWN:
        return str(state_type)
    return WORKER_STATE_TYPE_UNKNOWN


def classify_worker_state_pair_side(state: str | None) -> str:
    """Collapse one side of a transition to a bounded pair label.

    Returns the value itself when it is a :data:`WORKER_STATE_PAIR_KNOWN`
    member, :data:`WORKER_STATE_PAIR_NONE` when the side is absent, else
    :data:`WORKER_STATE_PAIR_OTHER`. ⛔ Never label the raw value: both sides are
    caller-supplied free text and the ``"other"`` sentinel is the whole
    cardinality bound.

    ⚠️ A side that is literally the four characters ``"none"`` books as
    ``"other"``, NOT as ``"none"``. The ``none`` label asserts A SIDE WAS
    MISSING — nothing was passed at all — and that is a different finding from a
    caller that persisted the word. Conflating them would make the one label
    that names a broken call site unreadable.
    """
    if state is None or state == "":
        return WORKER_STATE_PAIR_NONE
    if state in WORKER_STATE_PAIR_KNOWN:
        # ⛔ NOT `str(state)`. `WorkerStateType` is a `(str, Enum)` MIXIN, so a
        # member compares equal to its value — it passes the allow-list above —
        # but `str()` on it yields the QUALIFIED NAME. Measured:
        # `str(WorkerStateType.CREATED)` == 'WorkerStateType.CREATED', not
        # 'created'. That value escapes the bound this function exists to
        # enforce and takes the ceiling from 11 per side (121 pairs) to 18 per
        # side (324) — silently, because the allow-list check itself still
        # passes. `str.__str__` returns the plain value for BOTH a raw `str` and
        # a str-mixin member, with no isinstance ladder and no import of the
        # enum (which would be a cycle — see WORKER_STATE_PAIR_KNOWN).
        return str.__str__(state)
    return WORKER_STATE_PAIR_OTHER


def classify_worker_state_validated(validated: str | None) -> str:
    """Collapse a caller's ``validated`` to a bounded label value.

    Returns the value itself when it is a :data:`WORKER_STATE_WRITE_VALIDATED`
    member, else :data:`WORKER_STATE_WRITE_VALIDATED_FALSE`.

    ⛔ The fallback direction is deliberate and must not be flipped. An
    unrecognised value means the site did not say anything this module can
    verify, which is exactly what ``"false"`` asserts — no call ran and nothing
    proves the transition good. Falling back to ``"true"`` or
    ``"by_construction"`` would let a typo at a call site silently manufacture
    coverage, which is the failure this counter exists to detect. Over-reporting
    exposure is recoverable; over-reporting coverage is not.
    """
    if validated in WORKER_STATE_WRITE_VALIDATED:
        return str(validated)
    return WORKER_STATE_WRITE_VALIDATED_FALSE


class AgentMetricsClient:
    """Client for recording and exporting agent execution metrics to Grafana."""

    def __init__(self, service_name: str = "worker-management-agents") -> None:
        """Initialize the metrics client with OTLP exporter.

        Args:
            service_name: Name of the service for resource identification
        """
        self.enabled = False
        self.meter = None
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._up_down_counters: dict[str, Any] = {}

        # Check if OpenTelemetry is explicitly disabled
        otel_disabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
        if otel_disabled:
            logger.info("AgentMetricsClient disabled via OTEL_SDK_DISABLED environment variable")
            return

        try:
            # ⛔ THE PULL READER IS UNCONDITIONAL AND THE PUSH READER IS NOT.
            #
            # Before 2026-09-16 this whole method returned early — leaving
            # ``self.enabled = False`` and EVERY counter in this file dead —
            # whenever ``OTLP_METRICS_ENDPOINT`` was unset, or when a 2-second
            # httpx probe of Grafana failed. Both are reachable in production,
            # and the running pod was measured in exactly that state: the
            # arming instrument for
            # ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` existed in
            # source and produced nothing.
            #
            # A Prometheus PULL reader has no such dependency: it holds the
            # series in-process and Prometheus comes and gets them. So it is
            # attached FIRST and always, and the OTLP push reader is now a
            # best-effort ADDITION rather than the precondition for having any
            # telemetry at all. Grafana being down must never again mean this
            # service is unmeasurable.
            readers: list[Any] = [_prometheus_reader()]

            # Resolve OTLP metrics endpoint from env, falling back to derived
            # traces endpoint, finally to Settings (which empty-defaults and
            # fail-fasts in production via ``Settings.model_post_init``).
            from src.settings import get_settings

            metrics_endpoint = os.getenv(
                "OTLP_METRICS_ENDPOINT",
                os.getenv("OTLP_TRACES_ENDPOINT", "").replace("/traces", "/metrics"),
            )

            if not metrics_endpoint or "traces" in metrics_endpoint:
                metrics_endpoint = get_settings().otlp_metrics_endpoint

            if not metrics_endpoint:
                logger.info(
                    "OTLP push export disabled — OTLP_METRICS_ENDPOINT not configured. "
                    "Metrics remain available on the Prometheus scrape endpoint."
                )
            else:
                logger.info(f"Initializing AgentMetricsClient with endpoint: {metrics_endpoint}")
                otlp_reader = self._build_otlp_reader(metrics_endpoint)
                if otlp_reader is not None:
                    readers.append(otlp_reader)

            # Create resource with service name
            resource = Resource.create({SERVICE_NAME: service_name})

            # Set up meter provider
            provider = MeterProvider(resource=resource, metric_readers=readers)

            # Publish it as the global provider for OTHER instrumentation
            # (fastapi/httpx/sqlalchemy auto-instrumentation reads the global).
            # ⚠️ This is a no-op if something already set one: OTel logs
            # "Overriding of current MeterProvider is not allowed" and keeps the
            # incumbent. That is fine for them and FATAL for us, which is why
            # the next line does NOT go through the global.
            metrics.set_meter_provider(provider)

            # ⛔ TAKE THE METER FROM THE PROVIDER WE JUST BUILT, NOT THE GLOBAL.
            #
            # This used to be ``metrics.get_meter(__name__)``. That reads the
            # GLOBAL provider — which, per the note above, may be an incumbent
            # that has no PrometheusMetricReader attached. In that case every
            # instrument below is created on a provider nothing scrapes, and the
            # failure is totally silent: the counters exist, ``enabled`` is
            # True, ``add()`` succeeds, and the scrape is empty. That is the
            # exact shape of the bug this whole change is fixing, so it must not
            # be reachable by mere import order.
            self.meter = provider.get_meter(__name__)

            # Initialize metrics instruments
            self._setup_metrics()

            self.enabled = True
            logger.info("AgentMetricsClient initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize AgentMetricsClient: {e}")
            logger.info(
                "Agent metrics will not be sent to Grafana, but service will continue normally"
            )

    @staticmethod
    def _build_otlp_reader(metrics_endpoint: str) -> Any:
        """The OTLP push reader, or ``None`` when Grafana cannot be reached.

        ⛔ RETURNING ``None`` IS NOT A FAILURE ANY MORE. It used to abort the
        whole client. Now it only drops the PUSH leg: the Prometheus pull reader
        is already attached, so the series stay recordable and scrapable while
        Grafana is down. The 2-second probe survives because a
        ``PeriodicExportingMetricReader`` pointed at a dead collector logs an
        export traceback every 30s forever, which is the noise it was added to
        prevent.
        """
        try:
            import httpx

            test_url = metrics_endpoint.replace("/v1/metrics", "/")
            httpx.get(test_url, timeout=2)
            logger.debug(f"Successfully connected to metrics endpoint: {test_url}")
        except Exception as conn_err:
            logger.warning(
                f"Cannot reach Grafana at {metrics_endpoint}: {conn_err}. "
                f"OTLP push export is off; metrics remain on the Prometheus endpoint."
            )
            return None

        exporter = OTLPMetricExporter(
            endpoint=metrics_endpoint,
            timeout=5,  # 5 second timeout for export
        )
        # Periodic export (every 30 seconds)
        return PeriodicExportingMetricReader(
            exporter, export_interval_millis=30000, export_timeout_millis=30000
        )

    def _setup_metrics(self) -> None:
        """Set up metric instruments for recording agent data."""
        if not self.meter:
            return

        # Token usage counters
        self._counters["tokens_total"] = self.meter.create_counter(
            name="agent_tokens_total",
            description="Total tokens consumed by agents",
            unit="tokens",
        )

        self._counters["tokens_prompt"] = self.meter.create_counter(
            name="agent_tokens_prompt",
            description="Prompt tokens consumed by agents",
            unit="tokens",
        )

        self._counters["tokens_completion"] = self.meter.create_counter(
            name="agent_tokens_completion",
            description="Completion tokens consumed by agents",
            unit="tokens",
        )

        # Execution counters
        self._counters["executions_total"] = self.meter.create_counter(
            name="agent_executions_total",
            description="Total number of agent executions",
            unit="executions",
        )

        self._counters["executions_failed"] = self.meter.create_counter(
            name="agent_executions_failed",
            description="Number of failed agent executions",
            unit="executions",
        )

        # max_tokens cap-hit counter.
        #
        # A "cap hit" means the completion ended because it ran out of token
        # budget, not because the model was done. That is a TRUNCATED answer:
        #  - structured worker: the JSON is cut mid-string, pydantic-ai
        #    re-prompts, and every retry truncates identically at the same
        #    budget (a full decode each time -- the DGF-007 600s run).
        #  - plain-text worker: the truncated answer is returned to the caller
        #    as if it were complete. A silent falsehood.
        # Before this counter existed the only way to find a victim was to grep
        # pod logs and hand-histogram them (that is how ks-relationship-extractor
        # was found failing 17x in 5h at its cap).
        # NOTE ON ``unit``: keep it "1" (dimensionless). The OTLP -> Prometheus
        # exporter appends the unit to the metric name UNLESS the name already
        # contains that unit token, and only then forces the ``_total`` suffix.
        # With unit="hits" this counter shipped as
        # ``agent_max_tokens_cap_hit_hits_total`` -- so every dashboard and alert
        # written against the name declared right here matched nothing, silently,
        # which is the exact failure mode this counter was added to expose.
        # Measured live 2026-08-13; same bug hit agent_active_count and
        # aa_prompt_execution_total below. Do not "restore" a descriptive unit.
        self._counters["max_tokens_cap_hit"] = self.meter.create_counter(
            name="agent_max_tokens_cap_hit_total",
            description=(
                "Agent completions that hit the configured max_tokens cap and were "
                "therefore truncated. Detected from the provider-reported "
                "finish_reason='length' where available, else from "
                "completion_tokens >= the max_tokens actually sent."
            ),
            unit="1",
        )

        # Tokenless trusted-network tenant assertions.
        #
        # Counts requests that reached ``auth_middleware``'s trusted-network
        # branch, presented NO licence token, and still selected their own
        # tenant with a bare ``X-Tenant-Id`` header. That identity is the
        # caller's own claim; nothing proved it. This counter is the
        # "observe now, refuse later" instrument for closing that branch —
        # it must reach a stable, fully-attributed set of callers BEFORE
        # ``require_authenticated_tenant`` can be turned on anywhere.
        #
        # ⛔ NO TENANT AND NO CALLER IN THE LABELS. Tenant ids and peer IPs
        # are caller-controlled on exactly this path, so labelling them is an
        # unbounded-cardinality hole opened by the same untrusted input the
        # counter exists to watch. They go in the WARNING log line, which is
        # deduplicated, not into a time series. Labels stay low-cardinality:
        #   path     — route SHAPE, id-ish segments collapsed to ``{id}``
        #   enforced — "true" when the request was refused 401, else "false"
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["unauthenticated_tenant_assertion"] = self.meter.create_counter(
            name="worker_management_unauthenticated_tenant_assertion_total",
            description=(
                "Requests served (or refused) on the tokenless trusted-network branch "
                "that asserted a tenant via the X-Tenant-Id header with no licence "
                "credential. enforced='false' means it was served as before."
            ),
            unit="1",
        )

        # ⚠️ PRE-MATERIALISED SO "ABSENT" CANNOT BE READ AS "ZERO".
        #
        # This counter is the arming gate for
        # ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``: someone will read
        # it and decide, from what it shows, whether closing the tokenless
        # branch is safe. An OTel counter emits NO series until its first
        # ``add()``, so an un-fired instrument and a never-deployed one are
        # indistinguishable on the scrape — and the reading that would arm the
        # flag is exactly "it is zero". That confusion is the failure mode, so
        # both ``enforced`` values are booked at zero here.
        #
        # ``path`` is bounded by the route table and NOT statically enumerable,
        # so only the ``<unmatched>`` leg is booked. That is a REAL reachable
        # value, not an invented sentinel: an ENFORCED refusal returns 401
        # before the router runs, so every enforced refusal lands on exactly
        # this label (see ``auth_middleware._route_shape``). The per-route
        # breakdown still only appears once real traffic mints it — a
        # NON-``<unmatched>`` path being absent genuinely means no such request
        # has been seen.
        for _enforced in ("true", "false"):
            self._counters["unauthenticated_tenant_assertion"].add(
                0, {"path": _UNMATCHED_PATH, "enforced": _enforced}
            )

        # Tokenless tenant request surface, per ROUTE (see the closed label
        # sets above this class). Unlike the sibling counter above, ``route``
        # IS statically enumerable — hand-picked values from
        # ``TOKENLESS_TENANT_REQUEST_ROUTES``, not a route table — so all of
        # them are pre-materialised, not just one sentinel leg.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["tokenless_tenant_request"] = self.meter.create_counter(
            name="worker_management_tokenless_tenant_request_total",
            description=(
                "Requests to the routes in TOKENLESS_TENANT_REQUEST_ROUTES "
                "(originally the 8 named by WorkerManagementTokenlessTenantSurfaceGrew; "
                "widened round 16 to every tokenless-tenant-reading route once live "
                "measurement showed the alert's busiest route, "
                "workers_message, was unwired) that reached the tokenless "
                "trusted-network branch and asserted a tenant. 'served' means the "
                "request reached this per-route gate without being refused here — "
                "not a claim about the eventual response, which can still be a "
                "500/502. outcome='refused' would mean require_authenticated_tenant "
                "caused a 401 HERE, but the middleware already refuses an asserted "
                "tenant before routing when the flag is on, so this label reads 0 on "
                "every route forever and is not a live measurement; see "
                "worker_management_unauthenticated_tenant_assertion_total for the "
                "real refusal count. 'grace_open' means served AND the request rode "
                "a licences-unresolved fail-open BYO posture."
            ),
            unit="1",
        )
        # ⚠️ PRE-MATERIALISED for the same reason as the sibling counter above:
        # this is the per-route instrument an operator reads before arming
        # ``require_authenticated_tenant``, and an OTel counter with no ``add()``
        # yet is indistinguishable from one that was never deployed. Every
        # route in ``TOKENLESS_TENANT_REQUEST_ROUTES`` × ``served``/``refused``
        # is booked at zero; ``grace_open`` is NOT pre-seeded here (nor is the
        # reachable ``providers_models_other`` route)
        # — both are real, reachable values, but seeding a zero for either
        # would publish a measured-looking answer to a question ("did THIS
        # route ever grace-open", "did any OTHER provider name get queried
        # tokenless") this alert never asked and this deploy may never trigger.
        for _route in TOKENLESS_TENANT_REQUEST_ROUTES:
            for _outcome in (
                TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED,
                TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED,
            ):
                self._counters["tokenless_tenant_request"].add(
                    0, {"route": _route, "outcome": _outcome}
                )

        # Identity fields omitted on the tokenless trusted-network branch.
        #
        # The sibling counter above answers "who asserts a tenant they never
        # proved". This one answers the question underneath it: on that same
        # branch, ``license_id`` and ``subject_id`` are set to ``None``
        # unconditionally, silently, with no log line and no metric — so every
        # cost event a tokenless service-to-service caller produces is
        # structurally licence-less and subject-less, and nothing counted it.
        # That omission is the recorded mechanism behind the historically ~53%
        # NULL-licence rate in ``billing_service``'s cost flusher: the rows are
        # not lost downstream, they are born without an identity here.
        #
        # ⛔ NO TENANT AND NO PEER IP IN THE LABELS — same reasoning as the
        # counter above: both are caller-controlled on exactly this path, so
        # labelling them is an unbounded-cardinality hole. Labels stay
        # low-cardinality and bounded:
        #   field  — "license_id" or "subject_id" (2 values). Emitted per
        #            field, not once per request, so a future branch that drops
        #            only one of the two stays visible instead of averaging out.
        #   reason — why it was omitted. Distinguishes the caller that asserted
        #            its own tenant from the one that fell back to this
        #            deployment's DEFAULT_TENANT_ID; those are different
        #            populations with different migration work.
        #   caller — the ONE identifier on this request that was actually
        #            proven: the verified, allowlisted ``azp`` of an
        #            X-Caller-Authorization JWT. Bounded by the client
        #            allowlist, never caller-chosen. "<unverified>" is itself
        #            the useful answer and is the hardest class to migrate.
        #   path   — route SHAPE, id-ish segments collapsed to ``{id}``. Kept
        #            because for the "<unverified>" class it is the only
        #            remaining discriminator between distinct callers.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["identity_omitted"] = self.meter.create_counter(
            name="worker_management_identity_omitted_total",
            description=(
                "Identity fields (license_id / subject_id) left unset because the request "
                "authenticated on the tokenless trusted-network branch, which has no "
                "introspected identity. Every cost event from such a request is booked "
                "with a NULL licence and no subject. Observe-only: this counts what is "
                "already happening and changes no outcome."
            ),
            unit="1",
        )

        # BYO provider keys withheld from an unproven caller on
        # /tenant-config/modality/{modality}/resolve.
        #
        # That route is the only one in this service that returns a DECRYPTED
        # tenant credential in a response body, and the tokenless
        # trusted-network branch admits a caller on pod-network membership
        # alone (`workers` is ambient L4 — there is no waypoint and therefore
        # no mTLS identity to check either). Until every S2S caller of it
        # presents its own licence token, the key is withheld from callers
        # whose identity was never proven — and withholding SILENTLY would
        # simply move the problem, so it is counted here.
        #
        # ⭐ THIS COUNTER IS THE MIGRATION INSTRUMENT, not a security alarm.
        # Non-zero means some real caller wanted a key and did not get one:
        # that caller needs a token (or the route needs an allowlisted
        # service identity), and this is how it gets found. Zero across a
        # window with traffic means no legitimate caller depends on the
        # tokenless key path.
        #
        # ⛔ DELIBERATELY NOT PRE-MATERIALISED AT ZERO, unlike the metering
        # counters above. A seeded zero here would read as "measured: no
        # legitimate caller needs this", which is precisely the claim this
        # counter is supposed to earn rather than assert. The denominator is
        # the sibling series
        # ``worker_management_unauthenticated_tenant_assertion_total{path=
        # "/tenant-config/modality/{id}/resolve"}`` — tokenless calls to this
        # route. Absence of THIS series alongside a non-zero denominator is
        # the healthy reading; absence of both means no traffic, not safety.
        #
        # Labels stay bounded and are NOT caller-controlled: `modality` is
        # checked against MODALITIES before the handler reaches this point,
        # and `reason` is a closed set (see MODALITY_KEY_WITHHELD_REASONS).
        # Deliberately no tenant label: on this path the tenant is the
        # caller's own unproven claim, so it is unbounded by construction.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["modality_key_withheld"] = self.meter.create_counter(
            name="worker_management_modality_key_withheld_total",
            description=(
                "Decrypted BYO provider keys withheld from a /tenant-config/modality/"
                "{modality}/resolve response because the caller's identity was never "
                "proven (no verified licence token). The rest of the resolution is "
                "still returned, with hasKey reporting that a key exists."
            ),
            unit="1",
        )

        # ═══════════════════════════════════════════════════════════════════
        # TENANT BYO-MODE LOOKUP OUTCOMES.
        #
        # ⭐ THE FAIL-OPEN THIS COUNTER EXISTS FOR. ``_lookup_tenant_byo_mode``
        # asks licences whether a tenant must bring its own LLM key. Until this
        # counter it treated EVERY non-200 — 404, 401, 5xx, timeout — as
        # permission to proceed, returning ``local_allowed``: the most
        # permissive value in the enum, which routes the run onto owner GPU.
        # Measured on the deployed pods 2026-08-19: 8 of 79 lookups answered
        # 404, so ~10% of S2S runs skipped byo_required enforcement entirely,
        # with nothing anywhere recording that they had.
        #
        # ⛔ AND IT IS A CROSS-SERVICE HAZARD, not a local one. licences answers
        # this route 200 with a FABRICATED ``byo_required`` when its own DB
        # lookup faults (models.py ``_lookup_byo_mode``) — deliberately, because
        # this caller fails open on anything else. So the two services are
        # coupled: the day licences starts returning 503 for an unresolvable
        # posture, every one of those becomes a free platform-GPU grant here.
        # Counting the decision is what makes that coupling visible from this
        # side instead of only from theirs.
        #
        # Labels are bounded by construction and are NOT caller-controlled:
        # both come from closed tuples in this module. ⛔ DELIBERATELY NO
        # ``tenant`` OR ``product`` LABEL — the tenant on this path is a bare
        # X-Tenant-Id the caller chose (that is the whole reason this lookup
        # exists), so it is unbounded by construction. The tenant id is in the
        # WARNING instead, which is where attribution belongs.
        #
        # All 7 reachable (outcome, failure) pairs are pre-materialised at zero
        # below — including ``resolved``, which is the denominator. An
        # unobserved label has NO series, and absence reads identically to
        # health; this estate has several counters that look green only because
        # they were never born.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["tenant_byo_mode_lookup"] = self.meter.create_counter(
            name="worker_management_tenant_byo_mode_lookups_total",
            description=(
                "Tenant BYO-posture lookups against the licences service, by the decision "
                "this service made. outcome='resolved' licences answered and was believed; "
                "'stale_last_known_good' it could not be reached and this pod's previous "
                "answer for the same (tenant, product) was re-used; 'grace_open' it could "
                "not be reached, nothing was known, and the run was allowed onto platform "
                "inference inside the bounded grace window (byo_required is NOT enforced "
                "for that run); 'fail_closed' the grace window had elapsed and the run was "
                "refused. failure names why the lookup was unresolvable."
            ),
            unit="1",
        )
        for _byo_outcome, _byo_failure in TENANT_BYO_LOOKUP_PAIRS:
            self._counters["tenant_byo_mode_lookup"].add(
                0, {"outcome": _byo_outcome, "failure": _byo_failure}
            )

        # Execution time histogram
        self._histograms["execution_duration"] = self.meter.create_histogram(
            name="agent_execution_duration_seconds",
            description="Agent execution duration in seconds",
            unit="s",
        )

        # ═══════════════════════════════════════════════════════════════════
        # BYO cost-event metering: the drop counter and its DENOMINATOR.
        #
        # ⭐ A silent drop is an unmeasurable loss. ``telemetry/cost_events.py``
        # had three paths that return WITHOUT emitting a BYO cost event — one
        # with no log at all, two logged at DEBUG. Services run at
        # LOG_LEVEL=INFO, where DEBUG is deletion, not a quieter log. So real
        # BYO spend left the platform, was never metered, and nothing anywhere
        # said a line had gone missing: billing simply never saw it.
        #
        # ⛔ The three reasons are NOT the same event and must never share a
        # series — one benign reason sitting permanently non-zero would force
        # an alert on the whole counter to be silenced, taking the real signal
        # with it:
        #
        #   metering_unconfigured — ``METERING_REDIS_URL`` is empty. Metering
        #                           is DELIBERATELY off. Expected and benign,
        #                           and non-zero for the whole life of any
        #                           deployment that does not set the URL.
        #   no_tenant             — metering is LIVE and the LLM call really
        #                           happened, but no tenant resolved. This is
        #                           real spend with no owner. THIS is the
        #                           defect, and the series to alert on.
        #   no_event_loop         — emitted from a sync context, so the emit
        #                           coroutine was closed unrun. Also real
        #                           unmetered spend, but its remedy is a code
        #                           path change, not an attribution fix.
        #
        # All three labels are pre-materialised at zero below so that "zero
        # drops" is distinguishable from "this code path was never deployed" —
        # an unobserved label has no series at all, and absence reads
        # identically to health.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["metering_events_dropped"] = self.meter.create_counter(
            name="worker_management_metering_events_dropped_total",
            description=(
                "Completed BYO LLM calls whose cost_event was never emitted onto the "
                "billing cost bus. Every increment is real spend that left the platform "
                "and will never be billed."
            ),
            unit="1",
        )

        # The denominator. A drop count alone cannot be read as a rate: 40
        # drops is a rounding error against 4M emits and an outage against 40.
        # Deliberately UNLABELLED so it shares no label with the drop counter
        # and can be divided by the sum of it without a join.
        self._counters["metering_events_emitted"] = self.meter.create_counter(
            name="worker_management_metering_events_emitted_total",
            description=(
                "BYO cost_events successfully scheduled onto the billing cost bus. The "
                "denominator for worker_management_metering_events_dropped_total."
            ),
            unit="1",
        )

        # ⛔ HOW MUCH OF THE BILL IS ATTRIBUTED TO AN UNVERIFIED TENANT.
        #
        # `emit_byo_cost_event_task` bills whatever tenant is in the identity
        # ContextVar. On `auth_middleware`'s delegation branch that value came
        # from `X-Effective-Tenant-Id`, which `_effective_tenant` validates for
        # SPELLING ONLY — `AUTH_DELEGATING_SUBJECTS` lists WHICH SUBJECTS may
        # delegate, and nothing in this service checks which TENANTS a given
        # subject may act for. Four service accounts delegate in production.
        #
        # This counter is the answer to "how big is that population", which was
        # unanswerable before it existed: the emitted counter above is
        # deliberately unlabelled (it is the denominator for the drop counter
        # and may share no label with it), and the billing row carried no
        # provenance at all. Separate counter rather than a new label on the
        # denominator, so the divide-without-a-join property survives.
        #
        # ⛔ OBSERVE-ONLY. It changes no authorization decision and must not be
        # turned into one on its own evidence — refusing a delegated emit drops
        # real revenue from a supported path.
        #
        #   provenance — one of TENANT_PROVENANCES (6 values, CLOSED). ⛔ No
        #                tenant, subject, provider or model: those are exactly
        #                the caller-supplied unbounded values this label exists
        #                to characterise WITHOUT admitting.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["metering_event_tenant_provenance"] = self.meter.create_counter(
            name="worker_management_metering_events_by_tenant_provenance_total",
            description=(
                "BYO cost_events scheduled onto the billing cost bus, labelled by HOW the "
                "tenant being billed was established. Anything other than 'license' is a "
                "billing record attributed to a tenant this service never verified the "
                "caller may act for. Observe-only; changes no authorization decision."
            ),
            unit="1",
        )

        # Pre-materialise every series at zero (see the block comment above).
        # ⚠️ The provenance series MUST be pre-materialised: "no delegated spend"
        # and "the provenance path never ran" are the finding and its absence,
        # and an un-incremented labelled counter emits HELP/TYPE with no samples
        # — so without this, Prometheus cannot tell them apart.
        for _reason in METERING_DROP_REASONS:
            self._counters["metering_events_dropped"].add(0, {"reason": _reason})
        self._counters["metering_events_emitted"].add(0)
        for _provenance in TENANT_PROVENANCES:
            self._counters["metering_event_tenant_provenance"].add(0, {"provenance": _provenance})

        # ═══════════════════════════════════════════════════════════════════
        # WHO MUTATES THE FLEET-WIDE WORKFLOW ROUTING POINTER.
        #
        # ⛔ THIS COUNTER EXISTS BECAUSE THE LOG COULD NOT ANSWER THE QUESTION.
        # ``workflow_routing_pointer`` is keyed ``(card_type, list)`` with no
        # tenant column (migration 014; confirmed against the production
        # database), so a single ``POST /api/workflow-routing/promote`` repoints
        # dispatch for EVERY tenant. Deciding whether that surface can be gated
        # requires knowing which tenants actually call it — and on 2026-09-08
        # that was unanswerable:
        #   * the routing endpoints appear in the log only as uvicorn access
        #     lines, which carry no identity at all;
        #   * the two lines that DO carry a tenant
        #     (``auth_middleware``'s "Authenticated via license token" and
        #     "Trusted network access") are ``logger.debug`` and services run at
        #     INFO, so both were emitted ZERO times in 13,856 lines;
        #   * ``kubectl logs`` serves only the CURRENT file and this pod rotates
        #     fast — the whole readable window was 49 minutes.
        # Promoting those lines to INFO would not have fixed it: a log line you
        # cannot read back 24h later does not close a measurement gap. A counter
        # survives pod rotation, so this is a metric and not a log.
        #
        # ⛔ CARDINALITY — WHY ``tenant`` IS SAFE HERE WHEN THE TWO COUNTERS
        # ABOVE BOTH REFUSED IT. Those two count the tokenless trusted-network
        # branch, where the caller NAMES ITS OWN TENANT and
        # ``resolve_tenant_id`` accepts any canonical uuid — caller-chosen, so
        # unbounded, so correctly kept out of a label. This counter emits a raw
        # tenant ONLY when it came from licence INTROSPECTION, which can only
        # return a tenant that owns a licence (59 rows in ``licenses.tenants``
        # measured 2026-09-08: 33 service / 18 internal / 5 customer / 3 test).
        # Every other provenance collapses to the ``<unproven>`` sentinel, so a
        # caller on the pod network cannot mint series by inventing uuids. ⛔ A
        # DELEGATED request is NOT "licence introspection" for this purpose even
        # though a token was introspected: its tenant was replaced by
        # ``X-Effective-Tenant-Id`` afterwards, so it collapses to the sentinel
        # too — see WORKFLOW_ROUTING_AUTH_PATH_DELEGATED. The bound is:
        # 7 endpoints x 2 methods x (registered tenants + 1) x 5 auth_paths.
        #
        #   endpoint  — the MATCHED route template, never the raw path. The
        #               router has six routes and they carry no ids, so this is
        #               bounded by construction; "<unmatched>" is the sentinel.
        #   tenant    — the introspected tenant uuid, or "<unproven>" when the
        #               request did not present a licence token. NEVER the
        #               caller-asserted value.
        #   auth_path — HOW the tenant was established. This is the label that
        #               makes the answer actionable rather than merely
        #               interesting: a mutation arriving as "default" means the
        #               caller sent no licence token at all and landed on this
        #               deployment's DEFAULT_TENANT_ID, which is a different
        #               (and worse) finding than a real tenant mutating routing.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["workflow_routing_call"] = self.meter.create_counter(
            name="worker_management_workflow_routing_calls_total",
            description=(
                "Requests to /api/workflow-routing/*, labelled by matched endpoint, HTTP "
                "method, introspected tenant and how that tenant was established. Exists to "
                "answer 'which tenants mutate the fleet-wide routing pointer', which the "
                "access log cannot answer. Observe-only: it changes no authorization "
                "decision. See worker_management#230."
            ),
            unit="1",
        )

        # ═══════════════════════════════════════════════════════════════════
        # THE WORKER STATE MACHINE, AS IT ACTUALLY BEHAVES.
        #
        # ⛔ OBSERVE-ONLY. Neither counter below changes an outcome, and neither
        # may be turned into one on its own evidence. Making the state machine
        # REFUSE an invalid transition was measured at ~142,560 rejected
        # invocations over 30 days, ≈55% of all traffic: a 409 storm, not a fix.
        #
        # 1) verdicts — what ``validate_state_transition`` actually decided.
        #    The function is called from exactly two sites and it FAILS OPEN:
        #    an unparseable state logs a warning and returns ``True``. So today
        #    "no warnings in the log" and "every transition was legal" are
        #    indistinguishable, and the log cannot be read back over 30 days
        #    anyway. A counter survives pod rotation; a log line does not.
        #
        #      verdict — one of WORKER_STATE_VERDICTS (5 values, closed).
        #                ⛔ No worker, no tenant and no raw state value: the
        #                state column is free text, so any of those would be an
        #                unbounded label fed by exactly the untrusted input this
        #                counter exists to characterise.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_state_transition_verdict"] = self.meter.create_counter(
            name="worker_management_worker_state_transition_verdicts_total",
            description=(
                "Verdicts returned by WorkerStateManager.validate_state_transition. The "
                "three unknown_* verdicts are the fail-open path: the call returned True "
                "and the write proceeded. Observe-only; changes no outcome."
            ),
            unit="1",
        )

        # 2) writes — the choke point. EVERY real worker-state row is written
        #    through ``StateDatabase.add_worker_state``, which takes a plain
        #    ``str`` and INSERTs it unvalidated and uncoerced. The Pydantic
        #    surface (``/workers_states``) validates but carries no real
        #    traffic, so validation and traffic are on different code paths.
        #
        #    ⭐ THIS IS THE COUNTER THAT MAKES THE GAP VISIBLE. Three of the
        #    write paths (crud_endpoints' "config-updated"/"created", "updated",
        #    "prompt-swapped") never call the validator at all — on the live
        #    census that is ~14% of rows — and nothing anywhere recorded that.
        #
        #      state_type_class — the value iff it is a WorkerStateType member,
        #                         else "<unknown>". 8 values. See
        #                         classify_worker_state_type for why the raw
        #                         value must never be labelled.
        #      validated        — 3 values, closed. "true" = a validator call
        #                         actually ran. ⭐ "false" = THE EXPOSURE
        #                         NUMBER: no call ran and nothing proves the
        #                         transition good. ⛔ "by_construction" = no
        #                         call ran but the from-state is known at the
        #                         site and the transition is provably in
        #                         STATE_TRANSITIONS — this is NOT exposure and
        #                         must never be summed into it. See
        #                         WORKER_STATE_WRITE_VALIDATED for why two
        #                         values could not be read honestly.
        #
        #    All 27 combinations are pre-materialised at zero below, so a silent
        #    path is distinguishable from an unscraped one: an unobserved label
        #    has NO series, and absence reads identically to health.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_state_write"] = self.meter.create_counter(
            name="worker_management_worker_state_writes_total",
            description=(
                "Rows inserted into worker_states, by state_type class and by what is known "
                "about the transition: validated='true' a validator call ran, "
                "validated='by_construction' no call ran but the from-state is known and the "
                "transition provably valid, validated='false' is a "
                "write the state machine never saw. Observe-only; changes no outcome."
            ),
            unit="1",
        )

        # 3) unknown PAIRS — the (from, to) census, and ONLY for the fail-open
        #    arm. ⭐ THIS IS THE COUNTER THAT CAN DECIDE THE ENUM CHANGE, and
        #    counter (1) cannot: its labels are verdict-only, so it says HOW
        #    MANY calls fell through and never WHICH transition fell through.
        #    Adding ``CONFIG_UPDATED``/``UPDATED`` to ``WorkerStateType`` turns
        #    every observed pair from a fail-open ``True`` into a
        #    ``STATE_TRANSITIONS`` lookup, so a pair with no entry silently
        #    becomes ``False``. The safety precondition is therefore a PAIR
        #    census, and nothing shipped today can produce one.
        #
        #    ⭐ WHY A SEPARATE COUNTER RATHER THAN PAIR LABELS ON (1). Two
        #    reasons, and either alone would settle it. First, the pair
        #    dimension has decision value ONLY on the ``unknown_*`` arm: a
        #    ``valid``/``invalid_transition`` verdict already resolved both
        #    sides against the enum, so labelling its pair multiplies series
        #    for a question that is answered. Second, (1) has already SHIPPED
        #    (#235) with a closed 5-series label set; widening an exported
        #    metric's labels in place breaks every existing query and recording
        #    rule against it, and would force the pre-materialised zeros of a
        #    counter nobody asked to change to be re-verified. A new name costs
        #    a join in PromQL and nothing else.
        #
        #      from_state / to_state — one of WORKER_STATE_PAIR_VALUES (11
        #                values, closed). ⛔ NEVER the raw value: both sides are
        #                caller-supplied free text. The allow-list is wider than
        #                the enum ON PURPOSE — see WORKER_STATE_PAIR_KNOWN for
        #                why a census that cannot spell ``config-updated``
        #                cannot grade the change that would add it.
        #                ⛔ Still no worker and no tenant, for the same
        #                unbounded-label reason as (1).
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_state_transition_unknown_pair"] = self.meter.create_counter(
            name="worker_management_worker_state_transition_unknown_pairs_total",
            description=(
                "The (from, to) census of transitions that reached the fail-open path in "
                "WorkerStateManager.validate_state_transition, i.e. the calls counted by the "
                "unknown_from/unknown_to/unknown_both verdicts. Both sides are collapsed to a "
                "closed allow-list; 'other' is any value outside it and 'none' is a missing "
                "side. Read this before adding any member to WorkerStateType: every pair with "
                "a non-zero count must already have a STATE_TRANSITIONS entry, or the enum "
                "change turns a True into a False. Observe-only; changes no outcome."
            ),
            unit="1",
        )

        # 4) invalid PAIRS — the sibling of (3), on the arm (3) deliberately
        #    cannot see. ⭐ WHY IT IS NEEDED: `invalid_transition` is not a rare
        #    error path. Measured over 24h it is **1,753 of 16,409 verdicts
        #    (10.7%)**, against `valid` 14,656 — a tenth of all traffic is a
        #    transition the state machine declares illegal — and (1) can only
        #    say HOW MANY, never WHICH. Without the pair labels the only
        #    candidate explanation anyone has (`processing -> processing`, noted
        #    as undecided when #248 shipped) is unfalsifiable.
        #
        #    ⛔ COUNTER ONLY. It changes no return value, raises nothing, and
        #    specifically does NOT close the fail-open. The `decide` rung is
        #    still ahead, and it is cheap to defer: the sole caller
        #    (`message_endpoints.py:569`) DISCARDS the bool — the
        #    `add_worker_state` write runs outside the branch — so refusing
        #    here would today change exactly nothing anyway. This counter is
        #    what makes that rung decidable on data instead of on the one
        #    hypothesis somebody happened to write down.
        #
        #    ⛔ A SIBLING COUNTER, NOT LABELS ON (3), for the same two reasons
        #    (3) is not labels on (1). The reachable sets are DISJOINT and
        #    complementary — (3) seeds the 40 pairs with at least one non-member
        #    side, this seeds the 81 enum x enum pairs — so merging them would
        #    make "which arm did this pair arrive on?" unanswerable from the
        #    series alone, and that arm is the entire difference between "the
        #    state machine was bypassed" and "the state machine ran and said
        #    no". And (3) has SHIPPED (#248 graded an enum member against it);
        #    a new name costs a join in PromQL and breaks no existing query.
        #
        #      from_state / to_state — one of WORKER_STATE_PAIR_VALUES, same
        #                closed 11-value allow-list as (3). Both sides are
        #                enum members by construction on this arm, so the
        #                ``none``/``other`` sentinels are unreachable here —
        #                a series appearing on one of them means the caller
        #                called this from the wrong arm.
        #                ⛔ Still no worker and no tenant.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_state_transition_invalid_pair"] = self.meter.create_counter(
            name="worker_management_worker_state_transition_invalid_pairs_total",
            description=(
                "The (from, to) census of transitions that resolved against the enum and were "
                "then REFUSED by STATE_TRANSITIONS, i.e. the calls counted by the "
                "invalid_transition verdict (1,753 of 16,409 verdicts / 24h at the time this "
                "was added). Both sides are enum members by construction. Observe-only: the "
                "verdict is unchanged, nothing is raised, and the fail-open is NOT closed by "
                "this counter."
            ),
            unit="1",
        )

        # Pre-materialise all four series sets at zero (see the block comments
        # above). ⛔ Each pair counter seeds only the pairs IT can reach, and the
        # two sets are disjoint halves of the same 121: the unknown arm seeds the
        # 40 pairs with at least one non-member side (was 57 before ``updated``
        # became the 9th enum member on 2026-09-17), and the invalid arm the 81
        # enum x enum pairs. Seeding either counter with the other's half would
        # publish measured-looking zeros for a question that instrument never
        # asks. See WORKER_STATE_UNKNOWN_PAIRS and WORKER_STATE_INVALID_PAIRS.
        for _verdict in WORKER_STATE_VERDICTS:
            self._counters["worker_state_transition_verdict"].add(0, {"verdict": _verdict})
        for _state_class in WORKER_STATE_TYPE_CLASSES:
            for _validated in WORKER_STATE_WRITE_VALIDATED:
                self._counters["worker_state_write"].add(
                    0, {"state_type_class": _state_class, "validated": _validated}
                )
        for _from_state, _to_state in WORKER_STATE_UNKNOWN_PAIRS:
            self._counters["worker_state_transition_unknown_pair"].add(
                0, {"from_state": _from_state, "to_state": _to_state}
            )
        for _from_state, _to_state in WORKER_STATE_INVALID_PAIRS:
            self._counters["worker_state_transition_invalid_pair"].add(
                0, {"from_state": _from_state, "to_state": _to_state}
            )

        # ═══════════════════════════════════════════════════════════════════
        # BARE / DRIFTED WORKER CONTRACTS.
        #
        # ⛔ OBSERVE-ONLY. This counts what a request RESOLVED to; it never
        # repairs a row and must not become a write. See the block comment in
        # ``system_worker_contracts`` for why detection is split from repair.
        #
        # A "bare" row declares no output_schema in either place the agent
        # builder reads, so nothing constrains the completion. Until now the
        # only signal was a WARNING that fires INSIDE the repair path, which
        # means it could not fire for a worker this repo cannot repair — on
        # prod 2026-09-11 that is every ``ks-*`` worker, the family karma
        # depends on for relationship extraction. A bare ks-relationship-
        # extractor degrades extraction silently, with no signal anywhere.
        #
        #   contract — the worker name iff it is a KNOWN system contract, else
        #              "<other>". ⛔ Never the raw worker_name: it is a path
        #              param and the message endpoint auto-creates whatever it
        #              is handed (472 distinct names live), so labelling it raw
        #              hands cardinality to the untrusted input this counter
        #              exists to characterise. See CONTRACT_OTHER.
        #   verdict  — one of WORKER_CONTRACT_VERDICTS (3 values, closed).
        #              "drifted" is only decidable for a locally-registered
        #              contract; an externally-owned worker with a populated
        #              schema books "healthy" rather than a guess.
        #
        # Every (contract, verdict) pair is pre-materialised at zero below, so
        # a bare population of zero is distinguishable from an unscraped
        # exporter: an unobserved label has NO series, and absence reads
        # identically to health.
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_contract_verdict"] = self.meter.create_counter(
            name="worker_management_worker_contract_verdicts_total",
            description=(
                "Resolved worker rows by output-contract health: verdict='bare' the row "
                "declares no output_schema in either place the agent builder reads, so the "
                "completion is unconstrained; verdict='drifted' the row's schema differs from "
                "the contract registered in-image; verdict='healthy' otherwise. The contract "
                "label is the worker name only for known system contracts, else '<other>'. "
                "Observe-only; changes no outcome and repairs no row."
            ),
            unit="1",
        )

        # ⚠️ FUNCTION-LOCAL IMPORT, and it must stay local. At module scope this
        # is a cycle: ``src.api.workers.__init__`` imports ``_router`` ->
        # ``message_endpoints`` -> ``src.telemetry.agent_metrics``, so importing
        # the contracts module from this file's top level re-enters a
        # half-initialised ``agent_metrics``. ``AgentMetricsClient`` is built
        # lazily by ``get_metrics_client()``, long after the API package is
        # fully imported, so the import is safe HERE. (``agent_metrics``'s own
        # precedent for this hazard was to duplicate the closed tuple — not an
        # option for a 37-name list that would then drift silently.)
        from src.api.workers.system_worker_contracts import (
            WORKER_CONTRACT_VERDICTS,
            contract_labels,
        )

        for _contract in contract_labels():
            for _contract_verdict in WORKER_CONTRACT_VERDICTS:
                self._counters["worker_contract_verdict"].add(
                    0, {"contract": _contract, "verdict": _contract_verdict}
                )

        # ── Silent auto-provisioning on the message endpoint ────────────────
        #
        # ``POST /workers/{worker_name}/message`` auto-creates any name it is
        # handed, with ``system_prompt=None`` and ``instruction=None``. A caller
        # that mistypes a worker name — or sends to one that was renamed or
        # never seeded — therefore gets a PROMPTLESS generic worker and an HTTP
        # **200**. There is no error for a caller's error-reader to read: the
        # n8n Workers node's reader is correct and structurally blind here,
        # because the failure arrives as a success.
        #
        # Measured shape (n8n_custom_nodes#1522): ``worker-doing-step`` is
        # dispatched by the medium-step DOING branch and has NO ROW at any
        # timestamp, while its siblings ``worker-doing-turn`` / ``-prose`` are
        # upserted continuously through the SAME path. Worker names are not a
        # checked vocabulary, so a typo degrades silently.
        #
        # ⛔ This counter does NOT stop the auto-create — the live callers above
        # depend on it. It makes the event legible, and pairs with the WARNING
        # the endpoint emits beside it (which carries worker_name and tenant,
        # where cardinality costs nothing).
        #
        #   contract — ``classify_contract(worker_name)``: the worker name iff
        #              it is a KNOWN system contract, else "<other>". ⛔ Never
        #              the raw worker_name, for exactly the reason given on the
        #              contract-verdict counter above — this endpoint mints
        #              whatever it is handed and 472 distinct names are live, so
        #              a raw label hands unbounded cardinality to the untrusted
        #              input this counter exists to characterise.
        #
        # Every label is pre-materialised at zero below: an unobserved label has
        # no series at all, and absence would read identically to "nobody is
        # typo-ing a worker name".
        # NOTE ON ``unit``: keep it "1". See the max_tokens_cap_hit note above.
        self._counters["worker_auto_created"] = self.meter.create_counter(
            name="worker_management_worker_auto_created_total",
            description=(
                "Workers provisioned by the message endpoint's auto-create fallback, i.e. "
                "sends to a worker name that did not exist. Each one is created WITHOUT a "
                "system prompt or instruction and the call still returns 200, so this is the "
                "only signal that a caller's worker vocabulary has drifted. The contract "
                "label is the worker name only for known system contracts, else '<other>'. "
                "Observe-only; it does not block the auto-create."
            ),
            unit="1",
        )

        for _contract in contract_labels():
            self._counters["worker_auto_created"].add(0, {"contract": _contract})

        # Active agents gauge (up/down counter)
        self._up_down_counters["active_agents"] = self.meter.create_up_down_counter(
            name="agent_active_count",
            description="Number of currently active agents",
            # "agents" shipped this as ``agent_active_count_agents``. See the
            # unit note on the max_tokens_cap_hit counter above.
            unit="1",
        )

        # ==========================================
        # AA-specific metrics for meta-improvement
        # ==========================================

        # Prompt execution counter (with prompt_id, worker_name, status labels)
        self._counters["aa_prompt_execution"] = self.meter.create_counter(
            name="aa_prompt_execution_total",
            description="Total AA prompt executions",
            # "executions" shipped this as ``aa_prompt_execution_executions_total``
            # -- the name contains "execution", not "executions", so the exporter
            # appended it. See the unit note on the max_tokens_cap_hit counter.
            unit="1",
        )

        # Prompt latency histogram
        self._histograms["aa_prompt_latency"] = self.meter.create_histogram(
            name="aa_prompt_latency_seconds",
            description="AA prompt execution latency in seconds",
            unit="s",
        )

        # Prompt tokens counter (with token_type label: input/output)
        self._counters["aa_prompt_tokens"] = self.meter.create_counter(
            name="aa_prompt_tokens_total",
            description="Total tokens used by AA prompts",
            unit="tokens",
        )

        # Prompt validation failures counter
        self._counters["aa_prompt_validation_failures"] = self.meter.create_counter(
            name="aa_prompt_validation_failures_total",
            description="Total AA prompt validation failures",
            unit="failures",
        )

        # Prompt cost counter (in cents)
        self._counters["aa_prompt_cost"] = self.meter.create_counter(
            name="aa_prompt_cost_cents",
            description="Total cost of AA prompt executions in cents",
            unit="cents",
        )

        logger.info("Agent metrics instruments initialized (including AA metrics)")

    def record_execution(
        self,
        agent_name: str,
        agent_type: str,
        model_provider: str | None = None,
        model_name: str | None = None,
        duration_seconds: float | None = None,
        success: bool = True,  # noqa: FBT001, FBT002, RUF100
        token_usage: dict[str, int] | None = None,
        error_classification: str | None = None,
    ) -> None:
        """Record agent execution metrics.

        Args:
            agent_name: Name of the agent
            agent_type: Type of agent (pydantic_ai, claude_code, etc.)
            model_provider: Model provider (openai, mlx, deepseek, etc.)
            model_name: Model name
            duration_seconds: Execution duration in seconds
            success: Whether execution succeeded
            token_usage: Dictionary with token usage data (prompt_tokens, completion_tokens, total_tokens)
            error_classification: IR-aligned label (e.g. ``upstream_5xx``,
                ``upstream_timeout``, ``transient_network``) for failed
                executions, so Grafana can slice ``executions_failed``
                by cause. Ignored when success=True.
        """
        if not self.enabled or not self.meter:
            return

        try:
            # Common attributes for all metrics
            attributes = {
                "agent_name": agent_name,
                "agent_type": agent_type,
            }

            if model_provider:
                attributes["model_provider"] = model_provider
            if model_name:
                attributes["model_name"] = model_name
            if not success and error_classification:
                attributes["error_classification"] = error_classification

            # Record execution count
            if success:
                self._counters["executions_total"].add(1, attributes)
            else:
                self._counters["executions_failed"].add(1, attributes)

            # Record execution duration
            if duration_seconds is not None:
                self._histograms["execution_duration"].record(duration_seconds, attributes)

            # Record token usage
            if token_usage:
                if token_usage.get("total_tokens"):
                    self._counters["tokens_total"].add(token_usage["total_tokens"], attributes)
                if token_usage.get("prompt_tokens"):
                    self._counters["tokens_prompt"].add(token_usage["prompt_tokens"], attributes)
                if token_usage.get("completion_tokens"):
                    self._counters["tokens_completion"].add(
                        token_usage["completion_tokens"], attributes
                    )

            logger.debug(
                f"Recorded metrics for agent {agent_name}: "
                f"success={success}, duration={duration_seconds}s, "
                f"tokens={token_usage.get('total_tokens') if token_usage else 0}"
            )

        except Exception as e:
            logger.warning(f"Failed to record agent metrics: {e}")

    def record_max_tokens_cap_hit(
        self,
        agent_name: str,
        model_provider: str | None = None,
    ) -> None:
        """Record that an agent completion was truncated by its max_tokens cap.

        Args:
            agent_name: Name of the agent whose completion was truncated
            model_provider: Model provider (inference_router, mlx, openai, ...)
        """
        if not self.enabled or not self.meter:
            return

        try:
            attributes = {"agent_name": agent_name}
            if model_provider:
                attributes["model_provider"] = model_provider
            self._counters["max_tokens_cap_hit"].add(1, attributes)
        except Exception as e:
            logger.warning(f"Failed to record max_tokens cap hit: {e}")

    def record_agent_start(self, agent_name: str, agent_type: str) -> None:
        """Record that an agent has started execution.

        Args:
            agent_name: Name of the agent
            agent_type: Type of agent
        """
        if not self.enabled or not self.meter:
            return

        try:
            attributes = {"agent_name": agent_name, "agent_type": agent_type}
            self._up_down_counters["active_agents"].add(1, attributes)
        except Exception as e:
            logger.warning(f"Failed to record agent start: {e}")

    def record_agent_end(self, agent_name: str, agent_type: str) -> None:
        """Record that an agent has ended execution.

        Args:
            agent_name: Name of the agent
            agent_type: Type of agent
        """
        if not self.enabled or not self.meter:
            return

        try:
            attributes = {"agent_name": agent_name, "agent_type": agent_type}
            self._up_down_counters["active_agents"].add(-1, attributes)
        except Exception as e:
            logger.warning(f"Failed to record agent end: {e}")

    # ==========================================
    # AA-specific metric recording methods
    # ==========================================

    def record_aa_prompt_execution(
        self,
        prompt_id: str,
        worker_name: str,
        status: str,
        latency_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_cents: float | None = None,
        model: str | None = None,
        error_classification: str | None = None,
    ) -> None:
        """Record AA prompt execution metrics.

        Args:
            prompt_id: Identifier for the prompt/template
            worker_name: Name of the worker executing the prompt
            status: Execution status. The SHIPPED set is
                (success, empty_output, vacuous_output, timeout, failed,
                cancelled, recovered) — verified against every call site
                2026-08-25, `vacuous_output` added 2026-09-03, and
                `cancelled` + `recovered` added 2026-09-04.
                `cancelled` is emitted from an explicit
                `except asyncio.CancelledError` in pydantic_ai_agent.run, which
                re-raises: CancelledError is a BaseException, so it matched
                neither the TimeoutError nor the Exception handler and a
                cancelled run previously recorded a start and an end and
                nothing else.
                `recovered` means the run raised, _handle_mcp_error_recovery
                retried it, and the retry ANSWERED. It exists because the
                `failed` record used to be emitted BEFORE recovery ran, with no
                compensating record afterwards, so every rescued run was booked
                100% failed. The two are mutually exclusive and emitted from one
                `finally`, so the exception path still books exactly one record.
                ⚠️ Both are new label VALUES, not a redefinition of an existing
                one — the live consumer (n8n workflow ckwHNOGL4mZxv1a1) groups
                by status and assigns keys dynamically, so a new value is folded
                in rather than dropped.
                `vacuous_output` means the response schema DECLARED required
                fields and the answer carried none of them — a well-formed
                WRONG answer, which was previously booked as `success`. It is
                deliberately a NEW value rather than a widening of
                `empty_output`: `{}` from a schema with no `required` is a
                correct "found nothing" (ks-relationship-extractor,
                ks-entity-spotter, principle_consistency) and still books
                success. See _is_vacuous_against_schema in
                worker/agents/pydantic_ai_agent.py for the exact predicate and
                why every one of its branches fails closed. This line previously read
                "(success, failed, validation_failed, timeout)" and was wrong
                in both directions: `validation_failed` was documented but
                never emitted by anything, and `empty_output` (#192) was
                emitted but undocumented. Validation failures are counted on
                aa_prompt_validation_failures_total instead, because they can
                occur several times inside ONE run (pydantic-ai retries inside
                a single agent.run) and a per-run status cannot express that.
            latency_seconds: Execution latency in seconds
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            cost_cents: Cost in cents
            model: Model name used
            error_classification: IR-aligned label (e.g. ``upstream_5xx``,
                ``upstream_timeout``, ``transient_network``) for non-success
                executions, so the ``aa_prompt_execution`` counter can be
                sliced by cause in Grafana. Ignored for ``status='success'``.
        """
        if not self.enabled or not self.meter:
            return

        try:
            # Base attributes
            attributes = {
                "prompt_id": prompt_id,
                "worker_name": worker_name,
                "status": status,
            }
            if model:
                attributes["model"] = model
            if status != "success" and error_classification:
                attributes["error_classification"] = error_classification

            # Record execution counter
            self._counters["aa_prompt_execution"].add(1, attributes)

            # Record latency histogram
            if latency_seconds is not None:
                self._histograms["aa_prompt_latency"].record(
                    latency_seconds,
                    {
                        "prompt_id": prompt_id,
                        "worker_name": worker_name,
                    },
                )

            # Record token usage with token_type label
            if input_tokens is not None:
                self._counters["aa_prompt_tokens"].add(
                    input_tokens,
                    {
                        "prompt_id": prompt_id,
                        "worker_name": worker_name,
                        "token_type": "input",  # nosec B105 -- OpenTelemetry attribute value, not a credential
                    },
                )
            if output_tokens is not None:
                self._counters["aa_prompt_tokens"].add(
                    output_tokens,
                    {
                        "prompt_id": prompt_id,
                        "worker_name": worker_name,
                        "token_type": "output",  # nosec B105 -- OpenTelemetry attribute value, not a credential
                    },
                )

            # Record cost
            if cost_cents is not None and cost_cents > 0:
                cost_attributes = {
                    "prompt_id": prompt_id,
                    "worker_name": worker_name,
                }
                if model:
                    cost_attributes["model"] = model
                self._counters["aa_prompt_cost"].add(int(cost_cents * 100) / 100, cost_attributes)

            logger.debug(
                f"Recorded AA prompt metrics: prompt_id={prompt_id}, "
                f"worker={worker_name}, status={status}, latency={latency_seconds}s"
            )

        except Exception as e:
            logger.warning(f"Failed to record AA prompt metrics: {e}")

    def record_aa_prompt_validation_failure(
        self,
        prompt_id: str,
        worker_name: str,
        error_type: str,
    ) -> None:
        """Record AA prompt validation failure.

        Args:
            prompt_id: Identifier for the prompt/template
            worker_name: Name of the worker
            error_type: Type of validation error (json_parse, schema_invalid, etc.)
        """
        if not self.enabled or not self.meter:
            return

        try:
            attributes = {
                "prompt_id": prompt_id,
                "worker_name": worker_name,
                "error_type": error_type,
            }
            self._counters["aa_prompt_validation_failures"].add(1, attributes)
            logger.debug(f"Recorded AA validation failure: {prompt_id}, error_type={error_type}")
        except Exception as e:
            logger.warning(f"Failed to record AA validation failure: {e}")

    def record_unauthenticated_tenant_assertion(self, path: str, *, enforced: bool) -> None:
        """Count one tokenless trusted-network request that asserted its own tenant.

        Args:
            path: The MATCHED ROUTE TEMPLATE, NOT the raw path — e.g.
                ``/workers/{worker_name}/message``, never
                ``/workers/some-name/message`` (see
                ``auth_middleware._route_shape``, which reads it from
                ``scope["route"]`` so the label set is bounded by the route
                table). A raw path carries caller-chosen worker names and would
                make this a per-request time series.
            enforced: True when ``require_authenticated_tenant`` caused the
                request to be refused 401; False when it was served exactly as
                before. Keyword-only so the label can never be passed as a
                bare positional bool at a call site.

        Deliberately takes no tenant and no caller identifier: both are
        caller-controlled on this path. They belong in the log line.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["unauthenticated_tenant_assertion"].add(
                1,
                {"path": path, "enforced": "true" if enforced else "false"},
            )
        except Exception as e:
            logger.warning(f"Failed to record unauthenticated tenant assertion: {e}")

    def record_tokenless_tenant_request(self, route: str, outcome: str) -> None:
        """Count one request to one of the tokenless-tenant-surface routes.

        Args:
            route: One of ``TOKENLESS_TENANT_REQUEST_ROUTES`` (or the
                reachable ``TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER`` bucket) —
                a FIXED name, never the raw path. The only caller is
                ``src/auth/tokenless_gate.py``, which is the one place that
                decides the label; this method does not validate membership
                so a future named route needs a change in exactly one
                file, not two.
            outcome: One of ``TOKENLESS_TENANT_REQUEST_OUTCOMES``.

        Deliberately takes no tenant and no caller identifier — same reasoning
        as ``record_unauthenticated_tenant_assertion`` above.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["tokenless_tenant_request"].add(1, {"route": route, "outcome": outcome})
        except Exception as e:
            logger.warning(f"Failed to record tokenless tenant request: {e}")

    def record_identity_omission(self, field: str, *, reason: str, caller: str, path: str) -> None:
        """Count one identity field left unset on the tokenless auth branch.

        Args:
            field: The omitted field — ``"license_id"`` or ``"subject_id"``.
                One call per field, so the two never average into a single
                series that hides one of them.
            reason: Why it was omitted. Keyword-only. See
                ``auth_middleware._OMISSION_REASON_*`` for the closed set; it
                is a closed set precisely so this stays a bounded label.
            caller: The VERIFIED caller client id (JWT ``azp``), or
                ``"<unverified>"``. Keyword-only. Never the raw
                ``x-caller-client-id`` header, which is attacker-supplied on
                this path.
            path: The MATCHED ROUTE TEMPLATE, NOT the raw path (see
                ``auth_middleware._route_shape``). Keyword-only. This counter
                carries FOUR labels, so a raw path here multiplies out.

        Deliberately takes no tenant and no peer IP: both are caller-controlled
        on this path. They belong in the log line, not in a time series.

        Observe-only. This records a state the service already produces; it
        never changes whether a request is served.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["identity_omitted"].add(
                1,
                {"field": field, "reason": reason, "caller": caller, "path": path},
            )
        except Exception as e:
            logger.warning(f"Failed to record identity omission: {e}")

    def record_workflow_routing_call(
        self, *, endpoint: str, method: str, tenant: str, auth_path: str
    ) -> None:
        """Count one request to ``/api/workflow-routing/*``.

        Args:
            endpoint: The MATCHED route template (e.g.
                ``/api/workflow-routing/promote``), never the raw request path.
                Keyword-only. ``"<unmatched>"`` when no route matched.
            method: HTTP method. Keyword-only.
            tenant: The INTROSPECTED tenant uuid, or ``"<unproven>"`` when the
                request did not authenticate with a licence token. Keyword-only.
                ⛔ Never pass a caller-asserted tenant here — that is what makes
                this label bounded; see the block comment at the counter's
                creation for why the two neighbouring counters refuse a tenant
                label outright.
            auth_path: One of :data:`WORKFLOW_ROUTING_AUTH_PATHS`. Keyword-only,
                and a closed set for the same bounded-label reason.

        Observe-only, and deliberately so: it records who is already calling
        these endpoints and changes no authorization decision. The gate this
        measurement exists to inform is worker_management#230 — do NOT add one
        here on the strength of this counter alone until the data is in.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["workflow_routing_call"].add(
                1,
                {
                    "endpoint": endpoint,
                    "method": method,
                    "tenant": tenant,
                    "auth_path": auth_path,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record workflow routing call: {e}")

    def record_worker_state_transition_verdict(self, verdict: str) -> None:
        """Count one ``validate_state_transition`` decision.

        Args:
            verdict: One of :data:`WORKER_STATE_VERDICTS`. Keep it inside that
                closed set — it is the only thing keeping this a bounded label,
                and a verdict outside it also has no pre-materialised series.

        Deliberately takes no worker, tenant or raw state value: ``state_type``
        is free text in the database, so any of those would be an unbounded
        label fed by the very input this counter exists to characterise. The
        identifying detail belongs in the caller's warning log line.

        Never raises, and NEVER changes the verdict it is handed. The three
        ``unknown_*`` verdicts still return ``True`` at the call site — see the
        block comment at the counter's creation for the ≈55%-of-traffic
        measurement that forbids turning any of them into a refusal.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["worker_state_transition_verdict"].add(1, {"verdict": verdict})
        except Exception as e:
            logger.warning(f"Failed to record worker state transition verdict: {e}")

    def record_worker_state_transition_unknown_pair(
        self, from_state: str | None, to_state: str | None
    ) -> None:
        """Count one ``(from, to)`` pair that reached the fail-open path.

        Args:
            from_state: The RAW current state. Classified here via
                :func:`classify_worker_state_pair_side`; it reaches the label
                only as an allow-list member, ``"none"`` or ``"other"``.
            to_state: The RAW target state, classified the same way.

        ⭐ Call this ONLY from the ``unknown_*`` arm. On a ``valid`` or
        ``invalid_transition`` verdict both sides already resolved against the
        enum, so the pair carries no decision value and would only multiply
        series; that arm is why this is a second counter rather than two extra
        labels on :meth:`record_worker_state_transition_verdict`.

        Deliberately takes no worker and no tenant, for the same reason that
        method does: the identifying detail belongs in the caller's warning log
        line, where cardinality costs nothing.

        Never raises, and NEVER changes the verdict the caller is about to
        return. The fail-open still returns ``True`` — this counter exists to
        make the evidence for changing that readable, not to change it.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["worker_state_transition_unknown_pair"].add(
                1,
                {
                    "from_state": classify_worker_state_pair_side(from_state),
                    "to_state": classify_worker_state_pair_side(to_state),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record worker state transition unknown pair: {e}")

    def record_worker_state_transition_invalid_pair(
        self, from_state: str | None, to_state: str | None
    ) -> None:
        """Count one ``(from, to)`` pair that the state machine REFUSED.

        Args:
            from_state: The RAW current state. Classified here via
                :func:`classify_worker_state_pair_side`; it reaches the label
                only as an allow-list member, ``"none"`` or ``"other"``.
            to_state: The RAW target state, classified the same way.

        ⭐ The sibling of :meth:`record_worker_state_transition_unknown_pair`,
        and the arm that one is forbidden to see. Call this ONLY on the
        ``invalid_transition`` verdict — both sides resolved against the enum
        and ``STATE_TRANSITIONS`` then said no. That was **1,753 of 16,409
        verdicts over 24h (10.7%)**, and the verdict counter can only say how
        many; this says which.

        ⛔ OBSERVE-ONLY, AND NOT A STEP TOWARD REFUSING ANYTHING BY ITSELF. It
        does not change the returned verdict, it does not raise, and it does not
        close the fail-open in ``validate_state_transition``. The sole caller of
        that function discards the bool it returns, so there is nothing to close
        at this rung — this counter exists so the rung that eventually does can
        be argued from a distribution rather than from a hypothesis.

        Deliberately takes no worker and no tenant, same as its two siblings.

        Never raises.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["worker_state_transition_invalid_pair"].add(
                1,
                {
                    "from_state": classify_worker_state_pair_side(from_state),
                    "to_state": classify_worker_state_pair_side(to_state),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record worker state transition invalid pair: {e}")

    def record_worker_state_write(self, state_type: str | None, *, validated: str) -> None:
        """Count one row written to ``worker_states``.

        Args:
            state_type: The RAW value being persisted. It is classified here via
                :func:`classify_worker_state_type` and only ever reaches the
                label as a known member or ``"<unknown>"``; the raw value is
                never labelled.
            validated: One of :data:`WORKER_STATE_WRITE_VALIDATED` — ``"true"``
                (a ``validate_state_transition`` call actually ran), ``"false"``
                (⭐ the exposure number: no call ran and nothing proves the
                transition good) or ``"by_construction"`` (⛔ NOT exposure: no
                call ran, but the from-state is known at the site and the
                transition is provably in ``STATE_TRANSITIONS``). Anything else
                is collapsed to ``"false"`` by
                :func:`classify_worker_state_validated` — never upward, so a
                typo cannot manufacture coverage. Keyword-only, so it can never
                be passed as a bare positional at a call site; ⛔ a ``bool``
                here is a bug, and one that would silently label ``"false"``.

        Observe-only. This records a write the service has already decided to
        perform; it never changes whether the write happens.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["worker_state_write"].add(
                1,
                {
                    "state_type_class": classify_worker_state_type(state_type),
                    "validated": classify_worker_state_validated(validated),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record worker state write: {e}")

    def record_worker_contract_verdict(self, worker_name: str, verdict: str) -> None:
        """Count one resolved worker row's output-contract health.

        Args:
            worker_name: The RAW name. It is collapsed here via
                :func:`~src.api.workers.system_worker_contracts.classify_contract`
                and only ever reaches the label as a known system contract or
                ``"<other>"``; ⛔ the raw value is never labelled — it is an
                auto-createable path parameter, so labelling it raw is an
                unbounded label fed by untrusted input.
            verdict: One of
                :data:`~src.api.workers.system_worker_contracts.WORKER_CONTRACT_VERDICTS`.
                Keep it inside that closed set — a verdict outside it also has
                no pre-materialised series.

        Observe-only. This records the state of a row the service has already
        decided to run; it never repairs the row, never changes the response,
        and never raises. A bare contract is already degrading the completion —
        instrumentation must not also fail the request.
        """
        if not self.enabled or not self.meter:
            return

        try:
            # Local import for the cycle documented at the counter's creation.
            from src.api.workers.system_worker_contracts import classify_contract

            self._counters["worker_contract_verdict"].add(
                1, {"contract": classify_contract(worker_name), "verdict": verdict}
            )
        except Exception as e:
            logger.warning(f"Failed to record worker contract verdict: {e}")

    def record_worker_auto_created(self, worker_name: str) -> None:
        """Count one worker silently provisioned by the message endpoint.

        Args:
            worker_name: The RAW name. Collapsed here via
                :func:`~src.api.workers.system_worker_contracts.classify_contract`
                and only ever labelled as a known system contract or
                ``"<other>"``; ⛔ the raw value is never labelled — it is an
                auto-createable path parameter, so labelling it raw is an
                unbounded label fed by untrusted input. The identifying detail
                belongs in the WARNING the endpoint logs beside this call.

        Observe-only. The auto-create is the DEFAULT and stays that way — live
        callers (``worker-doing-turn``, ``worker-doing-prose``) are upserted
        through this path continuously. A caller that owns a closed vocabulary
        opts out per-message with ``require_existing=true``. Never raises:
        instrumenting a degraded provision must not fail the request.
        """
        if not self.enabled or not self.meter:
            return

        try:
            # Local import for the cycle documented at the counter's creation.
            from src.api.workers.system_worker_contracts import classify_contract

            self._counters["worker_auto_created"].add(
                1, {"contract": classify_contract(worker_name)}
            )
        except Exception as e:
            logger.warning(f"Failed to record worker auto-creation: {e}")

    def record_modality_key_withheld(self, modality: str, *, reason: str) -> None:
        """Count one decrypted BYO provider key NOT returned to an unproven caller.

        Args:
            modality: The resolved modality (tts / asr / image / embedding /
                vlm). Bounded — the handler has already rejected anything
                outside ``MODALITIES`` with a 404 before this can be reached.
            reason: Why the key was withheld. Keyword-only, closed set — see
                :data:`MODALITY_KEY_WITHHELD_REASONS`.

        Every failure is swallowed: a metrics backend that is down must never
        turn a request that succeeds into one that fails. ⛔ And note the
        ordering at the call site — the key is withheld FIRST and counted
        second, so a broken counter can never be the reason a secret is
        returned.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["modality_key_withheld"].add(
                1,
                {"modality": modality, "reason": reason},
            )
        except Exception as e:
            logger.warning(f"Failed to record modality key withheld: {e}")

    def record_tenant_byo_mode_lookup(self, outcome: str, *, failure: str) -> None:
        """Count one tenant BYO-posture lookup by the decision it produced.

        Args:
            outcome: One of :data:`TENANT_BYO_LOOKUP_OUTCOMES`. The DECISION,
                not the HTTP result — ``stale_last_known_good`` and
                ``grace_open`` are both "licences did not answer", and they are
                different events because one re-used the tenant's own posture
                and the other invented one.
            failure: One of :data:`TENANT_BYO_LOOKUP_FAILURES`. Keyword-only so
                a call site cannot silently swap the two arguments — they are
                both ``str`` and both bounded, so positional confusion would
                produce two plausible-looking series and no error.

        Keep both values inside their closed tuples: a value outside them also
        has no pre-materialised series, so it would report as an absence.

        Every failure is swallowed. ⛔ Note the ordering at the call site: the
        posture is decided and RETURNED independently of this call, so a metrics
        backend that is down can never be the reason a run is allowed or
        refused.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["tenant_byo_mode_lookup"].add(
                1,
                {"outcome": outcome, "failure": failure},
            )
        except Exception as e:
            logger.warning(f"Failed to record tenant byo_mode lookup: {e}")

    def record_metering_event_dropped(self, reason: str) -> None:
        """Count one BYO LLM call whose cost_event never reached the cost bus.

        Args:
            reason: One of :data:`METERING_DROP_REASONS`. Keep it inside that
                closed set — it is the only thing keeping this a bounded label,
                and a reason outside it also has no pre-materialised series.

        Deliberately takes no tenant, provider or model: this counter answers
        "how much billable work is being lost", and the identifying detail
        belongs in the WARNING log line the caller emits alongside it, where
        cardinality costs nothing.

        Never raises. Metering already failed on this path; instrumentation
        must not turn a lost cost event into a failed agent run.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["metering_events_dropped"].add(1, {"reason": reason})
        except Exception as e:
            logger.warning(f"Failed to record metering event drop: {e}")

    def record_metering_event_tenant_provenance(self, provenance: str) -> None:
        """Count one emitted cost_event under HOW its tenant was established.

        Args:
            provenance: One of :data:`src.api.identity_context.TENANT_PROVENANCES`.
                Anything outside that closed set is coerced to ``unknown`` — it
                is the only thing bounding this label's cardinality, and a value
                outside it also has no pre-materialised series.

        Deliberately takes no tenant, subject, provider or model: this counter
        answers "how much billable work is attributed to a tenant nobody
        verified", and the identifying detail belongs in the log line the
        emitter writes beside it, where cardinality costs nothing.

        Never raises. Same rule as its siblings — instrumentation must not turn
        a billed LLM call into a failed agent run.
        """
        if not self.enabled or not self.meter:
            return

        value = provenance if provenance in TENANT_PROVENANCES else TENANT_PROVENANCE_UNKNOWN
        try:
            self._counters["metering_event_tenant_provenance"].add(1, {"provenance": value})
        except Exception as e:
            logger.warning(f"Failed to record metering event provenance: {e}")

    def record_metering_event_emitted(self) -> None:
        """Count one BYO cost_event successfully scheduled onto the cost bus.

        The DENOMINATOR for :meth:`record_metering_event_dropped`. Without it a
        drop count cannot be read as a rate, and "40 drops" is unreadable — it
        is a rounding error against 4M emits and an outage against 40.

        Never raises, for the same reason as its sibling.
        """
        if not self.enabled or not self.meter:
            return

        try:
            self._counters["metering_events_emitted"].add(1)
        except Exception as e:
            logger.warning(f"Failed to record metering event emit: {e}")


# Global singleton instance
_metrics_client: AgentMetricsClient | None = None


def get_metrics_client() -> AgentMetricsClient:
    """Get or create the global metrics client instance.

    Returns:
        AgentMetricsClient instance
    """
    global _metrics_client  # noqa: PLW0603 -- lazy-init singleton
    if _metrics_client is None:
        _metrics_client = AgentMetricsClient()
    return _metrics_client


class AgentExecutionTimer:
    """Context manager for timing and recording agent execution metrics."""

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        model_provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize execution timer.

        Args:
            agent_name: Name of the agent
            agent_type: Type of agent
            model_provider: Model provider
            model_name: Model name
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.model_provider = model_provider
        self.model_name = model_name
        self.start_time: float | None = None
        self.metrics_client = get_metrics_client()

    def __enter__(self) -> AgentExecutionTimer:
        """Start timing and record agent start."""
        self.start_time = time.time()
        self.metrics_client.record_agent_start(self.agent_name, self.agent_type)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and record metrics."""
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None

            self.metrics_client.record_execution(
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                model_provider=self.model_provider,
                model_name=self.model_name,
                duration_seconds=duration,
                success=success,
            )

        self.metrics_client.record_agent_end(self.agent_name, self.agent_type)

    def record_tokens(self, token_usage: dict[str, int]) -> None:
        """Record token usage for this execution.

        Args:
            token_usage: Dictionary with token usage data
        """
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_client.record_execution(
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                model_provider=self.model_provider,
                model_name=self.model_name,
                duration_seconds=duration,
                success=True,
                token_usage=token_usage,
            )
