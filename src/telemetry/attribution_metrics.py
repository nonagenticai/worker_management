"""prometheus_client counters for attribution, metering and auth refusals.

Served on the metrics port (``METRICS_PORT``, 9887) by ``metrics_server``.
Every label set that is a closed vocabulary is materialised at 0 by
:func:`preinit_attribution_metrics` (called from the app lifespan AND at import)
so an ``increase()`` / ``absent()`` rule sees the series before the first event.

Nothing here ever carries a credential: reasons are closed sets, tenant labels
are tenant uuids, and the key fingerprint lives on the cost event, not here.
"""

from __future__ import annotations

from prometheus_client import Counter

# --- cost events -----------------------------------------------------------

#: The emitter REFUSED to publish a cost event (it was not dropped by accident:
#: it would have double-billed). ``platform_routed``: the model was built for a
#: provider inference-router serves, so IR is the meter of record.
COST_EMIT_REFUSED_PLATFORM_ROUTED = "platform_routed"
COST_EMIT_REFUSED_REASONS: tuple[str, ...] = (COST_EMIT_REFUSED_PLATFORM_ROUTED,)

COST_EMIT_REFUSED_TOTAL = Counter(
    "wm_cost_emit_refused_total",
    "worker_management cost events deliberately NOT emitted, by reason "
    "(platform_routed = inference-router meters this call; emitting would double-bill).",
    ["reason"],
)

#: Funding vocabulary stamped on every WM cost event (owner decision D1).
FUNDING_LOCAL_GPU = "local_gpu"
FUNDING_PLATFORM_EXTERNAL = "platform_external"
FUNDING_TENANT_BYO = "tenant_byo"
FUNDINGS: tuple[str, ...] = (FUNDING_LOCAL_GPU, FUNDING_PLATFORM_EXTERNAL, FUNDING_TENANT_BYO)

COST_EVENT_FUNDING_TOTAL = Counter(
    "wm_cost_event_funding_total",
    "worker_management cost events scheduled for emission, by funding "
    "(tenant_byo | platform_external | local_gpu).",
    ["funding"],
)

#: One call that used a key stored on a tenant's tenant_llm_config row (own
#: BYO or managed BYO). ``tenant`` is the tenant uuid (bounded by the tenant
#: registry), ``role`` the config row role (``__default__`` / ``tier:<t>``).
TENANT_KEY_USE_TOTAL = Counter(
    "wm_tenant_key_use_total",
    "LLM calls made with a key stored on a tenant's LLM config row (own or managed "
    "BYO), by tenant and config role. The per-call key fingerprint is on the cost event.",
    ["tenant", "role"],
)
TENANT_KEY_USE_PREINIT_TENANT = "none"
TENANT_KEY_USE_PREINIT_ROLES: tuple[str, ...] = ("__default__",)

# --- actor trust (D4) --------------------------------------------------------

ACTOR_REJECTED_MALFORMED = "malformed"
ACTOR_REJECTED_CROSS_TENANT = "cross_tenant"
ACTOR_REJECTED_UNKNOWN_SUBJECT = "unknown_subject"
ACTOR_REJECTED_TENANT_UNDECIDED = "tenant_undecided"
ACTOR_REJECTED_LOOKUP_UNAVAILABLE = "lookup_unavailable"
ACTOR_REJECTED_UNTRUSTED = "untrusted"
ACTOR_REJECTED_REASONS: tuple[str, ...] = (
    ACTOR_REJECTED_MALFORMED,
    ACTOR_REJECTED_CROSS_TENANT,
    ACTOR_REJECTED_UNKNOWN_SUBJECT,
    ACTOR_REJECTED_TENANT_UNDECIDED,
    ACTOR_REJECTED_LOOKUP_UNAVAILABLE,
    ACTOR_REJECTED_UNTRUSTED,
)

ACTOR_REJECTED_TOTAL = Counter(
    "worker_management_actor_rejected_total",
    "Inbound X-Actor-Id values DROPPED, by reason (malformed | cross_tenant | "
    "unknown_subject | tenant_undecided | lookup_unavailable | untrusted).",
    ["reason"],
)

# --- auth refusals -----------------------------------------------------------

# Reason LABELS, not credentials (S105 keyword-matches "token").
AUTH_REFUSED_MISSING_TOKEN = "missing_token"  # noqa: S105
AUTH_REFUSED_LICENSE_INACTIVE = "license_inactive"
AUTH_REFUSED_TENANT_ASSERTION = "tenant_assertion_requires_licence"
AUTH_REFUSED_UNUSABLE_TENANT = "unusable_tenant"
AUTH_REFUSED_CALLER_TOKEN_INVALID = "caller_token_invalid"  # noqa: S105
AUTH_REFUSED_WRONG_APP = "wrong_app"
# Licence branch, X-End-User-Token present but refused (401 / 403 / 503).
AUTH_REFUSED_END_USER_TOKEN_INVALID = "end_user_token_invalid"  # noqa: S105
AUTH_REFUSED_END_USER_FOREIGN_TENANT = "end_user_foreign_tenant"
AUTH_REFUSED_END_USER_UNVERIFIABLE = "end_user_unverifiable"
AUTH_REFUSED_REASONS: tuple[str, ...] = (
    AUTH_REFUSED_MISSING_TOKEN,
    AUTH_REFUSED_LICENSE_INACTIVE,
    AUTH_REFUSED_TENANT_ASSERTION,
    AUTH_REFUSED_UNUSABLE_TENANT,
    AUTH_REFUSED_CALLER_TOKEN_INVALID,
    AUTH_REFUSED_WRONG_APP,
    AUTH_REFUSED_END_USER_TOKEN_INVALID,
    AUTH_REFUSED_END_USER_FOREIGN_TENANT,
    AUTH_REFUSED_END_USER_UNVERIFIABLE,
)

AUTH_REFUSED_TOTAL = Counter(
    "worker_management_auth_refused_total",
    "worker_management requests refused by the auth middleware (401; end_user_* also 403/503), by reason.",
    ["reason"],
)


# --- licence-tenant binding ---------------------------------------------------
# A tenant header (plain `X-Tenant-Id` or `X-Effective-Tenant-Id`) that names a
# tenant OTHER than the presented licence's own, from a NON-delegating subject.
# `carrier` is where the licence travelled: `authorization` (gateway / n8n shape,
# always refused 403) or `x_tenant_id` (`X-Tenant-Id: Bearer`, refused only once
# WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT is on; served as the licence's
# own tenant before that). Enforce the second leg only once its `served` series
# has read 0 for a full window.
LICENCE_CARRIER_AUTHORIZATION = "authorization"
LICENCE_CARRIER_X_TENANT_ID = "x_tenant_id"
LICENCE_CARRIERS: tuple[str, ...] = (LICENCE_CARRIER_AUTHORIZATION, LICENCE_CARRIER_X_TENANT_ID)
MISMATCH_HEADER_PLAIN = "x_tenant_id"
MISMATCH_HEADER_EFFECTIVE = "x_effective_tenant_id"
MISMATCH_HEADERS: tuple[str, ...] = (MISMATCH_HEADER_PLAIN, MISMATCH_HEADER_EFFECTIVE)
MISMATCH_OUTCOME_REFUSED = "refused"
MISMATCH_OUTCOME_SERVED = "served"
MISMATCH_OUTCOMES: tuple[str, ...] = (MISMATCH_OUTCOME_REFUSED, MISMATCH_OUTCOME_SERVED)

LICENCE_TENANT_MISMATCH_TOTAL = Counter(
    "worker_management_licence_tenant_mismatch_total",
    "Requests whose tenant header named a tenant other than the presented licence's "
    "own, from a non-delegating subject, by licence carrier, header and outcome.",
    ["carrier", "header", "outcome"],
)


# --- tenant×app licence (George 2026-10-15) ----------------------------------
# Which app the presented licence was issued for. `app` is the introspected
# `app.key` (bounded by the licences app registry; anything not key-shaped is
# `other`, an absent app is `none`), `path` the matched route template. No
# tenant or subject labels: who still sends a non-`workers` licence is in the
# deduped `licence_app` log line, not here.
LICENCE_APP_EXPECTED = "workers"
LICENCE_APP_NONE = "none"
LICENCE_APP_OTHER = "other"
LICENCE_APP_OK = "tenant_app_ok"
LICENCE_APP_DELEGATED = "delegated"
LICENCE_APP_WRONG_APP = "wrong_app"
LICENCE_APP_NO_APP = "no_app"
LICENCE_APP_OUTCOMES: tuple[str, ...] = (
    LICENCE_APP_OK,
    LICENCE_APP_DELEGATED,
    LICENCE_APP_WRONG_APP,
    LICENCE_APP_NO_APP,
)

LICENCE_APP_TOTAL = Counter(
    "wm_licence_app_total",
    "Licence-authenticated worker_management requests by the licence's app, route "
    "template and outcome (tenant_app_ok | delegated | wrong_app | no_app).",
    ["app", "path", "outcome"],
)


def preinit_attribution_metrics() -> None:
    """Materialise every closed-vocabulary series above at 0."""
    for reason in COST_EMIT_REFUSED_REASONS:
        COST_EMIT_REFUSED_TOTAL.labels(reason=reason)
    for funding in FUNDINGS:
        COST_EVENT_FUNDING_TOTAL.labels(funding=funding)
    for reason in ACTOR_REJECTED_REASONS:
        ACTOR_REJECTED_TOTAL.labels(reason=reason)
    for reason in AUTH_REFUSED_REASONS:
        AUTH_REFUSED_TOTAL.labels(reason=reason)
    for carrier in LICENCE_CARRIERS:
        for header in MISMATCH_HEADERS:
            for outcome in MISMATCH_OUTCOMES:
                LICENCE_TENANT_MISMATCH_TOTAL.labels(
                    carrier=carrier, header=header, outcome=outcome
                )
    # One series per outcome so `increase()` sees each before its first event.
    for app, outcome in (
        (LICENCE_APP_EXPECTED, LICENCE_APP_OK),
        (LICENCE_APP_OTHER, LICENCE_APP_DELEGATED),
        (LICENCE_APP_OTHER, LICENCE_APP_WRONG_APP),
        (LICENCE_APP_NONE, LICENCE_APP_NO_APP),
    ):
        LICENCE_APP_TOTAL.labels(app=app, path="<unmatched>", outcome=outcome)
    # Tenant is runtime data; the `none` leg makes the family exist at startup.
    for role in TENANT_KEY_USE_PREINIT_ROLES:
        TENANT_KEY_USE_TOTAL.labels(tenant=TENANT_KEY_USE_PREINIT_TENANT, role=role)


preinit_attribution_metrics()
