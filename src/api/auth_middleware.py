"""License Token Authentication Middleware for Worker Management API.

Uses license token (`X-Tenant-Id: Bearer <token>`) as the ONLY auth method.
No Keycloak, no JWT, no RBAC - valid license = full access.
"""

import hashlib
import ipaddress
import logging
import os
import re
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import httpx
from cachetools import TTLCache
from fastapi import HTTPException, Request, Response, status
from prometheus_client import Counter
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_BYPASS,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    get_app_id_from_context,
    get_caller_client_id_from_context,
    set_aa_trace_context,
    set_actor_id_context,
    set_app_id_context,
    set_caller_context,
    set_correlation_context,
    set_delegating_subject,
    set_external_request_context,
    set_identity_context,
    set_job_id_context,
    set_project_id_context,
    set_prompt_attribution,
    set_spend_scope_context,
    set_tenant_byo_lookup_outcome,
    valid_actor_id,
)
from src.api.tenant_constants import (
    BYPASS_TENANT_UUID,
    default_tenant_uuid,
    resolve_asserted_tenant_id,
    resolve_tenant_id,
    tenant_uuids,
)
from src.auth.keycloak_roles import (
    CallerTokenInvalidError,
    allowed_caller_clients,
    verified_caller_identity,
)
from src.settings import get_settings, parse_project_id_caller_licences
from src.telemetry import get_metrics_client
from src.telemetry.agent_metrics import (
    TENANT_BYO_LOOKUP_FAIL_CLOSED,
    TENANT_BYO_LOOKUP_FAILURE_NON_200,
    TENANT_BYO_LOOKUP_FAILURE_NONE,
    TENANT_BYO_LOOKUP_FAILURE_TRANSPORT,
    TENANT_BYO_LOOKUP_GRACE_OPEN,
    TENANT_BYO_LOOKUP_RESOLVED,
    TENANT_BYO_LOOKUP_STALE,
)
from src.telemetry.attribution_metrics import (
    ACTOR_REJECTED_LOOKUP_UNAVAILABLE,
    ACTOR_REJECTED_MALFORMED,
    ACTOR_REJECTED_TOTAL,
    ACTOR_REJECTED_UNTRUSTED,
    AUTH_REFUSED_CALLER_TOKEN_INVALID,
    AUTH_REFUSED_END_USER_FOREIGN_TENANT,
    AUTH_REFUSED_END_USER_TOKEN_INVALID,
    AUTH_REFUSED_END_USER_UNVERIFIABLE,
    AUTH_REFUSED_LICENSE_INACTIVE,
    AUTH_REFUSED_MISSING_TOKEN,
    AUTH_REFUSED_TENANT_ASSERTION,
    AUTH_REFUSED_TOTAL,
    AUTH_REFUSED_UNUSABLE_TENANT,
    AUTH_REFUSED_WRONG_APP,
    LICENCE_APP_DELEGATED,
    LICENCE_APP_EXPECTED,
    LICENCE_APP_NO_APP,
    LICENCE_APP_NONE,
    LICENCE_APP_OK,
    LICENCE_APP_OTHER,
    LICENCE_APP_TOTAL,
    LICENCE_APP_WRONG_APP,
    LICENCE_CARRIER_AUTHORIZATION,
    LICENCE_CARRIER_X_TENANT_ID,
    LICENCE_TENANT_MISMATCH_TOTAL,
    MISMATCH_HEADER_EFFECTIVE,
    MISMATCH_HEADER_PLAIN,
    MISMATCH_OUTCOME_REFUSED,
    MISMATCH_OUTCOME_SERVED,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Trusted "effective tenant" side-channel (mirrors w2v_common's AuthMiddleware).
# WM authenticates S2S callers via their service-account license token, which
# introspects to the SERVICE-account tenant — NOT the customer/run tenant. A
# trusted service account (its introspected subject is in AUTH_DELEGATING_SUBJECTS)
# may carry the run's real tenant in this header so BYO resolution + the LLM cost
# event attribute to the run's tenant, not the service account. A customer's
# subject is never a delegator, so a customer cannot spoof. Empty config ⇒ inert.
# (EFFECTIVE_TENANT_HEADER is defined in identity_context — single source of truth.)


async def _stamp_end_user_proof_if_delegated(
    request: Request,
    *,
    delegated: bool,
    tenant_id: str,
    delegator: object,
    licence_end_user: dict | None = None,
) -> None:
    """On a DELEGATED request, verify ``X-End-User-Token`` (owner ruling 2026-10-01).

    See ``src.auth.end_user_token``. Never refuses; records the outcome for
    ``llm_config_resolver.tenant_key_entitled``. On the plain licence branch the
    token was already verified (and refused if bad) by
    ``AuthenticationMiddleware._licence_end_user``; ``licence_end_user`` is that
    verified payload, recorded here because ``set_identity_context`` (which
    runs just before) clears the end-user ContextVars.
    """
    if not delegated:
        if licence_end_user is not None:
            from src.auth.end_user_token import stamp_licence_end_user

            stamp_licence_end_user(licence_end_user, tenant_id)
        return
    # The VERIFIED (introspected) subject that re-scoped the tenant -- read by
    # tenant_proof for the IR-token decision (WM_DELEGATING_SUBJECTS).
    set_delegating_subject(str(delegator or ""))
    from src.auth.end_user_token import stamp_end_user_proof

    await stamp_end_user_proof(request, tenant_id, str(delegator or ""))


def _delegating_subjects() -> frozenset[str]:
    """Subject IDs (CSV in AUTH_DELEGATING_SUBJECTS) permitted to delegate a tenant.

    Read from the environment at call time so config changes/tests apply without a
    reload. Empty (default) disables the X-Effective-Tenant-Id side-channel.

    ⚠️ LIVE COUNT, MEASURED 2026-09-12 — **4**, not 3: ``svc-kag-karma``
    (01a0138e-…), ``svc-kag`` (01a03c8f-…), ``svc-requirements`` (01a0138c-…)
    and ``svc-fr-backend`` (019ffe9f-…, added by gitops #3813 on 2026-09-11).
    Three agreeing instruments: the running ``worker-management`` pod's env, the
    live Deployment, and gitops ``origin/main``
    ``clusters/strato/services/workers/deployment.yaml``. The "3" in this
    branch's second commit message came from a STALE LOCAL gitops checkout —
    read the env off the cluster, never off a working copy, and note that this
    set only grows, so a remembered number is a floor at best. Every one of the
    four is a caller that reaches ``/tenant-config/modality/*/resolve`` with
    ``tenant_proven=False`` once it starts presenting a token.
    """
    raw = os.getenv("AUTH_DELEGATING_SUBJECTS", "") or ""
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


# License introspection caches. Positive (5 min) + stale-while-error (15 min):
# the stale cache records the last-known-good ACTIVE result so a transient
# licenses blip (timeout / 5xx) serves last-known-good instead of a spurious
# 401. Mirrors w2v_common's AuthMiddleware (#70) + license-authz (#27), which
# this custom middleware never inherited.
_license_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_license_stale_cache: TTLCache = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_STALE_TTL", "900"))
)
# The licenses `/license/introspect` endpoint returns 200 {active:false} for a
# genuinely bad token (never a 4xx), so any timeout/connection-error/non-200
# here is an INFRA failure, not a token verdict. Bumped 2s->10s (a 2s timeout
# under load was conflated with an invalid token — the growth_bundle 401 class).
_INTROSPECT_TIMEOUT = float(os.getenv("LICENSE_INTROSPECT_TIMEOUT", "10.0"))

# A 200 {active:false, reason:"token_not_found"} during an ADDITIVE token
# rotation is a propagation race (the new token's row has not committed/
# replicated yet, or a pod still carries a stale Secret value) — NOT a
# rejection. The licenses rotation for the shared w2v SA token is additive
# (POST /licenses/{id}/tokens; provision_service_token.py --rotate never
# revokes), and a revoked/deactivated token returns reason "token_invalid" /
# "license_inactive" instead — so ONLY "token_not_found" is treated as a
# transient blip: served from the warm stale cache if present, else returned as
# None (→ retryable 503) so the caller's retry ladder rides out the race.
# Bounded from the FIRST sighting so a permanently-missing token still resolves
# to an authoritative 401 after the window. 0 disables (restores instant-401).
_TOKEN_NOT_FOUND_GRACE = float(os.getenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "120"))
_license_token_not_found_since: TTLCache = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_STALE_TTL", "900"))
)


def _bearer_value(raw: str | None) -> str | None:
    """The token in ``Bearer <token>``, or ``None``.

    Scheme compared case-insensitively (RFC 9110 sec. 11.1); ``Bearer`` with
    nothing after it is no token, so it falls through like an absent header.
    """
    value = (raw or "").strip()
    if value[:7].lower() != "bearer ":
        return None
    return value[7:].strip() or None


def _licence_from_authorization(request: Request) -> bool:
    """True iff this request's licence is read from ``Authorization``.

    ``X-Tenant-Id: Bearer`` has precedence, so the licence came from
    ``Authorization`` exactly when ``X-Tenant-Id`` carried none.
    """
    return _bearer_value(request.headers.get("X-Tenant-Id")) is None and (
        _bearer_value(request.headers.get("Authorization")) is not None
    )


def _plain_tenant_header(request: Request) -> str:
    """A non-Bearer ``X-Tenant-Id`` value (stripped), or ``""``.

    Only the auth scheme itself ("Bearer" / "Bearer <tok>") is not a tenant; a
    value that merely starts with "bearer" (e.g. "bearerco") is still a claim.
    """
    raw = (request.headers.get("X-Tenant-Id") or "").strip()
    low = raw.lower()
    return "" if low == "bearer" or low.startswith("bearer ") else raw


def _same_tenant(claim: str, licence_tenant: str) -> bool:
    """Whether a tenant CLAIM names the licence's own (canonical uuid) tenant.

    Textual match (trimmed, case-insensitive) first, else the claim resolved
    through :func:`resolve_tenant_id`. Names are accepted HERE on purpose: this
    only ever decides "is it the licence's own tenant" -- a claim that passes is
    never adopted (the licence tenant is what the request runs under), so a
    lenient spelling cannot widen anything; it can only avoid refusing a caller
    that spelled its own tenant differently.
    """
    left, right = claim.strip().lower(), licence_tenant.strip().lower()
    if left == right:
        return True
    return resolve_tenant_id(claim) == right


def _license_introspect_url() -> str:
    """Resolve the license-introspection endpoint from Settings each call."""
    return get_settings().license_introspect_url


# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/metrics",
}


def _default_tenant_id() -> str:
    """``DEFAULT_TENANT_ID`` from the environment, resolved and validated.

    The header path below is fail-closed; without this the no-header path was
    not, so a typo in one ConfigMap would silently make every tokenless write in
    the deployment land on an unusable tenant — and, since `tenant_id` is a
    `uuid` column, fail with 22P02 at the first query rather than at startup.
    Failing here names the variable and the value.
    """
    raw = os.getenv("DEFAULT_TENANT_ID") or default_tenant_uuid()
    resolved = resolve_tenant_id(raw) if raw else None
    if resolved is None:
        raise ValueError(
            f"DEFAULT_TENANT_ID={raw!r} is neither a UUID nor a known tenant name. "
            "tenant_id is a uuid column; a name must be one of "
            f"{sorted(tenant_uuids())}. (Empty means the Deployment sets neither "
            "DEFAULT_TENANT_ID nor TENANT_UUIDS — the tokenless no-header path "
            "refuses rather than picking a tenant.)"
        )
    return resolved


# ⛔ NO MODULE-LEVEL CACHE OF IT. It used to be ``DEFAULT_TENANT_ID =
# _default_tenant_id()`` right here, which resolves the tenant at IMPORT time —
# so a deployment (or a test process) with no env could not even import this
# module, and the refusal happened far from the request it protects. It is now
# resolved per call: absent config refuses the one request that needed a default
# instead of taking the whole service down at import.

_TRUSTED_CIDRS_RAW = os.getenv(
    "TRUSTED_CIDRS",
    "127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8",
)
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network(cidr.strip(), strict=False)
    for cidr in _TRUSTED_CIDRS_RAW.split(",")
    if cidr.strip()
]


def _is_trusted_network(host: str | None) -> bool:
    """Check if the client IP is within a trusted CIDR range.

    There is deliberately NO ``host == "testclient"`` arm here.

    ``testclient`` is Starlette's ``TestClient`` sentinel peer name (and
    ``httpx.ASGITransport``'s default). It is not an IP address and belongs to no
    CIDR, so that arm was not a narrowing of TRUSTED_CIDRS — it was orthogonal to
    them, granting the tokenless trusted-network branch to any request whose peer
    reported that literal string, and it shipped in the production image. The
    branch it unlocks binds ``request.state.tenant_id`` straight from a bare
    ``X-Tenant-Id`` header, so it was a tenant-selection primitive, not just an
    auth skip.

    Removed 2026-08-02, following ``tool_registry`` #23 and ``kag_common``
    0.20.0 (#25), which closed the identical hatch. Precedent for the shape that
    remains — a pure network-membership decision — is
    ``graph_adapter_service/src/auth/resolver.py``.

    Tests obtain auth the way a real caller does: pin the ASGI transport to a real
    peer address inside TRUSTED_CIDRS (``client=("127.0.0.1", 0)`` for
    ``TestClient``), or send a license token. See
    ``tests/test_testclient_peer_is_not_an_auth_bypass.py``.
    """
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


# Dedup window for the "unauthenticated tenant assertion" WARNING below.
#
# That line fires on a path 10+ live workloads use for EVERY request, so an
# unsampled WARNING would be the loudest thing this service logs and would bury
# the signal it exists to carry. The key is (tenant, peer, route shape, verified
# caller), i.e. exactly the tuple an operator needs to go and fix ONE caller —
# so every distinct caller still gets named, once per window, and a NEW caller
# appearing is visible within the window rather than lost in the flood.
#
# ⛔ The COUNTER is never sampled. Rate is read from
# ``worker_management_unauthenticated_tenant_assertion_total``; the log is for attribution
# only. Do not "simplify" by counting inside this guard.
# Authenticated API requests that arrived WITHOUT a usable X-Actor-Id (absent,
# malformed, or asserted by a caller not trusted to assert it). The request then
# stays ACTOR-LESS downstream -- an authenticated external caller is never filled
# with WM's SERVICE_ACTOR_ID (that is only for WM-originated work, #353) -- so
# this counter names the caller to fix. WHY a present value was dropped is
# counted separately on `worker_management_actor_rejected_total{reason}`.
# `caller` is the verified caller client id (azp-allowlisted, bounded), else the
# inbound X-App-Id when it is a well-formed registry key, else `unknown` — the
# same precedence as `wm_ir_call_missing_project_total`.
ACTOR_MISSING_UNKNOWN_CALLER = "unknown"
_ACTOR_MISSING_CALLER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ACTOR_MISSING_TOTAL = Counter(
    "worker_management_actor_missing_total",
    "Authenticated worker_management API requests without a valid, trusted "
    "X-Actor-Id (absent, malformed or untrusted), by caller.",
    ["caller"],
)


def preinit_actor_missing_metrics() -> None:
    """Materialise ``worker_management_actor_missing_total`` at 0 at startup.

    One child per allowlisted caller client plus ``unknown``, so the series
    (and an ``increase()`` rule over it) exist before the first miss.
    """
    ACTOR_MISSING_TOTAL.labels(caller=ACTOR_MISSING_UNKNOWN_CALLER)
    for client in allowed_caller_clients():
        ACTOR_MISSING_TOTAL.labels(caller=client)


def _actor_missing_caller() -> str:
    for candidate in (get_caller_client_id_from_context(), get_app_id_from_context()):
        value = (candidate or "").strip().lower()
        if value and _ACTOR_MISSING_CALLER_RE.match(value):
            return value
    return ACTOR_MISSING_UNKNOWN_CALLER


preinit_actor_missing_metrics()

# Authenticated API requests whose X-Project-Id is absent or not a canonical,
# non-nil UUID (owner ruling 2026-10-08: every conveyor call carries its fr
# projectId). ALWAYS counted; `require_project_id_header` (default OFF) changes
# only the OUTCOME — served vs 400 — never the count, exactly like
# `require_authenticated_tenant`. The flag is flipped by gitops once the
# counter has read zero for the callers that matter.
#
# ⛔ `caller` is ONLY what this middleware AUTHENTICATED, never X-App-Id: a
# self-declared label could hide a caller's misses under someone else's name
# and make the "zero missing ⇒ safe to enforce" read a lie. Precedence: the
# verified, allowlisted X-Caller-Authorization `azp`; else the auth branch that
# admitted the request (`licence` = an introspected-active licence token,
# `trusted_network` = the tokenless TRUSTED_CIDRS branch); else `unknown`
# (bypass_auth). A licence caller gets the configured name for its
# INTROSPECTED license_id (`project_id_caller_licences`), else
# `licence:<license_id>` — n8n sends only its licence token (no
# X-Caller-Authorization), so the licence is its strongest verified identity.
PROJECT_ID_CALLER_LICENCE = "licence"
PROJECT_ID_CALLER_TRUSTED_NETWORK = "trusted_network"
PROJECT_ID_CALLER_UNKNOWN = "unknown"
PROJECT_ID_REASON_ABSENT = "absent"
PROJECT_ID_REASON_INVALID = "invalid"
_PROJECT_ID_REASONS = (PROJECT_ID_REASON_ABSENT, PROJECT_ID_REASON_INVALID)
PROJECT_ID_MISSING_TOTAL = Counter(
    "worker_management_project_id_missing_total",
    "Authenticated worker_management API requests whose X-Project-Id is absent or "
    "not a canonical non-nil UUID, by authenticated caller and reason.",
    ["caller", "reason"],
)
PROJECT_ID_REFUSED_TOTAL = Counter(
    "worker_management_project_id_refused_total",
    "Requests refused 400 for a missing/invalid X-Project-Id "
    "(REQUIRE_PROJECT_ID_HEADER on), by authenticated caller and reason.",
    ["caller", "reason"],
)


def preinit_project_id_metrics() -> None:
    """Materialise both X-Project-Id counters at 0 for every bounded label pair."""
    # Not `licence:<id>`: those are per-licence and appear on first use.
    callers = {
        PROJECT_ID_CALLER_TRUSTED_NETWORK,
        PROJECT_ID_CALLER_UNKNOWN,
        *allowed_caller_clients(),
        *_project_id_caller_licences().values(),
    }
    for caller in callers:
        for reason in _PROJECT_ID_REASONS:
            PROJECT_ID_MISSING_TOTAL.labels(caller=caller, reason=reason)
            PROJECT_ID_REFUSED_TOTAL.labels(caller=caller, reason=reason)


@lru_cache(maxsize=4)
def _parsed_caller_licences(raw: str) -> dict[str, str]:
    """Parse once per distinct config value; a malformed map warns ONCE and is empty.

    Settings refuses a malformed map only while REQUIRE_PROJECT_ID_HEADER is on;
    with it off, observability config must never take WM down.
    """
    try:
        return parse_project_id_caller_licences(raw)
    except ValueError as exc:
        logger.warning(
            "WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES ignored (%s); licence callers "
            "are labelled licence:<license_id> until it is fixed",
            exc,
        )
        return {}


def _project_id_caller_licences() -> dict[str, str]:
    return _parsed_caller_licences(get_settings().project_id_caller_licences)


_LICENCE_ID_LABEL_RE = re.compile(r"[^a-z0-9-]")


def _project_id_licence_caller(license_result: dict[str, Any] | None) -> str:
    """Mapped name for the INTROSPECTED license_id, else ``licence:<license_id>``.

    ⛔ From ``license_result`` only — never the request's ``license_id``, which
    on the delegating branch is the caller-forwarded X-License-Id. Unmapped ids
    stay individually visible (an unknown n8n instance is not folded into a
    shared bucket); cardinality is bounded by the number of licences. The id is
    a uuid from licences, but it is sanitised to [a-z0-9-] and capped anyway so
    a surprise value can never mint unbounded label text.
    """
    verified = str((license_result or {}).get("license_id") or "").strip().lower()
    mapped = _project_id_caller_licences().get(verified)
    if mapped:
        return mapped
    label_id = _LICENCE_ID_LABEL_RE.sub("", verified)[:36]
    return f"{PROJECT_ID_CALLER_LICENCE}:{label_id}" if label_id else PROJECT_ID_CALLER_LICENCE


def _project_id_caller(auth_branch: str) -> str:
    """Verified azp if one was established, else the authenticating branch."""
    verified = (get_caller_client_id_from_context() or "").strip().lower()
    if verified and _ACTOR_MISSING_CALLER_RE.match(verified):
        return verified
    return auth_branch


def _project_id_problem(raw: str | None) -> str:
    """``""`` when ``raw`` is a usable project id, else the reason it is not.

    Same canonical non-nil UUID rule as X-Actor-Id (``valid_actor_id``), not the
    laxer ``valid_project_id``: that one is IR's "treat malformed as absent" read
    for cost events and accepts braces/urn/bare hex, which this gate must not
    bless as "carries its fr projectId".
    """
    if not (raw or "").strip():
        return PROJECT_ID_REASON_ABSENT
    return "" if valid_actor_id(raw) else PROJECT_ID_REASON_INVALID


preinit_project_id_metrics()

_ASSERTION_LOG_TTL = float(os.getenv("UNAUTH_TENANT_LOG_TTL", "300"))
_assertion_log_seen: TTLCache = TTLCache(maxsize=2048, ttl=_ASSERTION_LOG_TTL)

# Same dedup, for the NO-HEADER (default-tenant) line that
# ``_audit_identity_omission`` logs. Separate cache so the two lines cannot
# suppress each other. Key: (peer, route shape, verified caller). WM#324 step 1:
# the only attribution a header-less, token-less caller leaves is its TCP peer
# (ztunnel is L4 and preserves the source pod IP), so log it; resolve offline.
_omission_log_seen: TTLCache = TTLCache(maxsize=2048, ttl=_ASSERTION_LOG_TTL)

# One `licence_app` line per (subject, app) per hour: names who still presents a
# non-`workers` licence without putting the subject on a metric label.
_licence_app_log_seen: TTLCache = TTLCache(
    maxsize=2048, ttl=float(os.getenv("LICENCE_APP_LOG_TTL", "3600"))
)

#: An introspected ``app.key`` used as a label must look like a key; anything
#: else is ``other`` so a malformed registry row cannot mint series.
_APP_KEY_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _introspected_app_key(license_result: dict[str, Any]) -> str | None:
    """``app.key`` from licences' introspect ``app`` object (``{id, key, name}``), else None."""
    app = license_result.get("app")
    if not isinstance(app, dict):
        return None
    key = app.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def _licence_app_label(app_key: str | None) -> str:
    """Bounded ``app`` label: the key, ``none`` when absent, ``other`` when not key-shaped."""
    if app_key is None:
        return LICENCE_APP_NONE
    return app_key if _APP_KEY_LABEL_RE.match(app_key) else LICENCE_APP_OTHER


#: Label value for a request that never reached a route (404s, and refusals
#: this middleware itself returns before the router ever runs).
#:
#: A LITERAL on purpose. The obvious alternative — collapse the raw path with a
#: character heuristic and keep a couple of segments — is exactly the bug this
#: constant replaced: a heuristic reads a string the CALLER chose, so there is
#: always some spelling that survives it. ``GET /zzunroutablecallerchosenpath``
#: is alphabetic in every segment; under the old rule it became a Prometheus
#: series named after itself. A constant cannot be spelled into anything.
_UNMATCHED_PATH = "<unmatched>"

#: ``request.state`` slot holding the metric emissions that are waiting for the
#: route template. See :func:`_defer_route_labelled_metric`.
#:
#: ⛔ PER-REQUEST, and underscore-prefixed. ``request.state`` is shared with
#: handlers, so a plausible-looking name here is a foot-gun: an endpoint that
#: assigned ``request.state.deferred_route_labelled_metrics = []`` would silence
#: BOTH counters for that request and nothing would fail. Nothing in ``src/``
#: writes this today; the underscore is what keeps it that way. It must also
#: stay per-request rather than a module-global queue — under concurrency a
#: shared queue drains another request's emitters against THIS request's route
#: template (see ``test_concurrent_requests_do_not_cross_label_each_other``).
_DEFERRED_METRICS_ATTR = "_deferred_route_labelled_metrics"


def _route_shape(request: Request) -> str:
    """The MATCHED ROUTE TEMPLATE for this request, for use as a metric label.

    Raw paths on this service carry worker names, session ids and graph ids, so
    labelling with the raw path opens a per-request time series — the same
    unbounded-cardinality mistake as labelling the tenant. ``POST
    /workers/{worker_name}/message`` is the sharp edge: it AUTO-CREATES whatever
    name it is handed, so the path segment is not merely high-cardinality, it is
    attacker-chosen. ``src/api/workers/system_worker_contracts.py`` states the
    rule for exactly this parameter: "⛔ The raw ``worker_name`` must NEVER reach
    a label."

    This used to be ``_metric_path(path)``, a character heuristic that collapsed
    a segment to ``{id}`` only when it was NOT alphabetic-after-stripping ``-``
    and ``_``. Every alphabetic name therefore passed through RAW; only
    digit-bearing ones collapsed. Prod already holds
    ``path="/workers/zz-probe-nonexistent-worker/message"``, minted by a single
    one-off probe, and ``/providers/{name}/models`` emitted a distinct shape per
    provider name. A gitops ratchet alert counts these shapes, so one tokenless
    call to a new alphabetic name pinned a security alert for seven days.

    The template comes from the route table, so the label set is bounded by
    construction — which is what the ratchet alert already assumed.

    ⚠️ ``scope["route"]`` is the ONLY workable source here. Walking
    ``request.app.routes`` and re-matching does NOT work on FastAPI >= 0.137:
    each ``include_router`` contributes one opaque ``_IncludedRouter`` whose
    ``matches()`` returns ``Match.FULL`` with an EMPTY child scope and no
    ``path``, so such a walk labels 100% of real traffic ``<unmatched>`` while
    every unit test that registers routes directly on the app still passes.
    Starlette/FastAPI set ``scope["route"]`` when the router matches — i.e.
    AFTER this middleware's ``dispatch`` has run its auth logic — which is why
    callers must defer (see :func:`_defer_route_labelled_metric`) rather than
    read this before ``call_next``. Verified against fastapi 0.137.1 /
    starlette 1.3.1 (``fastapi/routing.py``: ``child_scope["route"] = self``).
    """
    path = getattr(request.scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path
    return _UNMATCHED_PATH


def _defer_route_labelled_metric(request: Request, emit: Callable[[str], None]) -> None:
    """Queue ``emit(route_shape)`` until the router has resolved the template.

    The auth decisions in ``dispatch`` all happen BEFORE ``call_next``, but
    ``scope["route"]`` only exists AFTER it, so anything that wants the template
    as a label has to be queued here and flushed from ``dispatch``'s ``finally``.
    """
    getattr(request.state, _DEFERRED_METRICS_ATTR).append(emit)


def _flush_route_labelled_metrics(request: Request) -> None:
    """Emit every deferred metric with the now-known route template.

    Every failure is swallowed, per emitter. This runs in a ``finally`` on the
    response path: a metrics backend that is down or a mock that raises must
    never turn a request that succeeded into one that fails, and one broken
    emitter must not silence the next.
    """
    pending = getattr(request.state, _DEFERRED_METRICS_ATTR, None)
    if not pending:
        return
    shape = _route_shape(request)
    for emit in pending:
        try:
            emit(shape)
        except Exception:  # pragma: no cover - defensive; observe-only
            logger.debug("deferred route-labelled metric failed", exc_info=True)
    pending.clear()


# Closed set of reasons for the identity-omission counter. CLOSED on purpose:
# these are metric label values, so an open-ended string here would be an
# unbounded-cardinality hole. Add a constant when a genuinely new branch omits
# identity — never pass an ad-hoc string.
#
# Both values describe the SAME structural fact (the tokenless trusted-network
# branch has no introspected identity, so licence and subject are unset); they
# differ in what the caller did about the tenant, which is what separates the
# two migration problems:
#   ..._ASSERTED_TENANT — the caller chose its own tenant with a bare header.
#       These are the callers that must move to a licence token before
#       WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT can be armed.
#   ..._DEFAULT_TENANT  — no X-Tenant-Id at all; the tenant came from this
#       deployment's own DEFAULT_TENANT_ID. Arming the tenant gate does NOT
#       affect these (the gate only fires when a header is present), but their
#       cost events are still licence-less, so they are counted here.
_OMISSION_REASON_ASSERTED_TENANT = "trusted_network_tokenless_asserted_tenant"
_OMISSION_REASON_DEFAULT_TENANT = "trusted_network_tokenless_default_tenant"

# The identity fields the trusted-network branch leaves unset. Emitted
# per-field so a future change that drops only one stays visible.
_OMITTED_IDENTITY_FIELDS = ("license_id", "subject_id")


class AuthenticationError(Exception):
    """Authentication related errors."""


class _RequestRefused(Exception):
    """Base for refusals raised inside ``_authenticate``; carries the response to send."""

    def __init__(self, response: Response, reason: str = "request refused") -> None:
        super().__init__(reason)
        self.response = response


class _CallerTokenRefused(_RequestRefused):
    """A PRESENT ``X-Caller-Authorization`` token failed validation (401)."""


class _ProjectIdRefused(_RequestRefused):
    """X-Project-Id absent/invalid while ``require_project_id_header`` is on (400)."""


class _TenantRefused(_RequestRefused):
    """A tenant identifier could not be resolved; carries the response to send.

    The three places that resolve a tenant (the tokenless header, the
    delegation header, and the introspected value) all sit at different depths
    inside ``dispatch``. Returning a ``Response`` up through each of them added
    a return per site to an already branch-heavy method; raising lets one
    handler turn any of them into the right refusal, and makes it impossible to
    add a fourth resolution site that forgets to check.
    """

    def __init__(self, response: Response) -> None:
        super().__init__(response, "tenant refused")


async def _introspect_license(token: str) -> dict[str, Any] | None:
    """Introspect license token to get tenant context.

    Calls the licenses service to validate the token and retrieve tenant info.
    Results are cached for 5 minutes to reduce latency.

    Args:
        token: The license token (without Bearer prefix)

    Returns:
        The introspection dict (``active`` True or False) on an authoritative
        200 response; the stale last-known-good dict on a transient failure with
        a warm stale cache; or ``None`` when introspection is transiently
        UNAVAILABLE and no stale entry exists. ``None`` therefore means "could
        not reach a verdict" (licenses slow/unreachable) — NEVER "token
        rejected" (a bad token comes back as a 200 ``active:false`` dict). The
        caller maps ``None`` to a retryable 503, not a 401.
    """
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    if cache_key in _license_cache:
        return cast("dict[str, Any]", _license_cache[cache_key])

    try:
        async with httpx.AsyncClient(timeout=_INTROSPECT_TIMEOUT) as client:
            resp = await client.post(_license_introspect_url(), json={"token": token})
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                if data.get("active"):
                    _license_cache[cache_key] = data
                    # Record the good result for stale-while-error fallback.
                    _license_stale_cache[cache_key] = data
                    # Recovery: a later token_not_found flip gets a fresh window.
                    _license_token_not_found_since.pop(cache_key, None)
                    return data
                # 200 active:false is authoritative for a genuinely bad token.
                # EXCEPT reason "token_not_found" during an additive rotation is a
                # propagation race, not a rejection — ride it out as transient,
                # bounded from the first sighting.
                if data.get("reason") == "token_not_found" and _TOKEN_NOT_FOUND_GRACE > 0:
                    now = time.monotonic()
                    first = _license_token_not_found_since.get(cache_key)
                    if first is None:
                        _license_token_not_found_since[cache_key] = first = now
                    if now - first < _TOKEN_NOT_FOUND_GRACE:
                        logger.warning(
                            "AUDIT introspect token_not_found treated as transient "
                            "(rotation race) cache_key=%s elapsed=%.0fs window=%.0fs",
                            cache_key,
                            now - first,
                            _TOKEN_NOT_FOUND_GRACE,
                        )
                        stale = _license_stale_cache.get(cache_key)
                        if stale is not None:
                            # Warm replica: serve last-known-good.
                            return cast("dict[str, Any]", stale)
                        # Cold replica (no warm stale): None -> caller retryable 503.
                        return None
                    # Window elapsed: re-pin the ORIGINAL first-sighting so its
                    # TTL refreshes under continuous traffic. Otherwise the
                    # since-entry evicts at its TTL and the next token_not_found
                    # re-opens a fresh window — a permanently-missing token would
                    # flap 503<->401 forever instead of settling to 401.
                    _license_token_not_found_since[cache_key] = first
                    logger.warning(
                        "introspect token_not_found grace expired cache_key=%s "
                        "window=%.0fs; returning authoritative inactive",
                        cache_key,
                        _TOKEN_NOT_FOUND_GRACE,
                    )
                return data
            logger.warning("License introspection returned status %d", resp.status_code)
    except Exception as e:
        logger.warning(f"License introspection failed: {e}")

    # Transient failure (timeout / connection error / non-200 — the endpoint
    # returns 200 for every token verdict, so this is always infra, never a
    # rejection). Serve last-known-good if we have one so a briefly slow/
    # unreachable licenses service does not spuriously 401 a token that was
    # valid moments ago; else return None (caller -> retryable 503).
    stale = _license_stale_cache.get(cache_key)
    if stale is not None:
        logger.warning(
            "License introspection unavailable; serving stale last-known-good (active=%s)",
            cast("dict[str, Any]", stale).get("active"),
        )
        return cast("dict[str, Any]", stale)
    return None


# Tenant byo_mode cache (5 min TTL) for trusted-network S2S calls that carry no
# token (requirements/kag/coding → WM forward only a raw X-Tenant-Id). Without
# this lookup those runs defaulted to local_allowed and bypassed BYO enforcement.
#
# ⛔ AUTHORITATIVE ANSWERS ONLY. Nothing decided by _unresolved_byo_mode below
# is ever written here. Caching a fabricated posture for 300s is exactly the
# defect that makes the licences side's own fail-closed fabrication expensive
# (one blip → up to five minutes of hard 402 for tenants that are not
# byo_required); repeating it here would multiply it, and would outlive the
# outage that caused it.
_tenant_byo_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

# Last-known-GOOD posture per (tenant, product), for stale-while-error.
#
# Deliberately a SEPARATE, longer-lived cache from the positive one above, and
# not just a longer TTL on it: the positive cache answers "what is this tenant's
# posture" and must expire quickly so a posture CHANGE takes effect within 5
# minutes, while this one answers "what did licences last tell us", which is the
# only non-fabricated thing available when licences cannot be reached.
#
# This mirrors ``_license_stale_cache`` above — same mechanism, same default
# window, same reason — rather than inventing a second shape for the same idea.
# Its own env var so the two can be tuned apart: the token path's stale window
# trades a spurious 401, this one trades an unenforced entitlement.
_TENANT_BYO_STALE_TTL = int(os.getenv("TENANT_BYO_STALE_TTL", "900"))
_tenant_byo_stale_cache: TTLCache = TTLCache(maxsize=1000, ttl=_TENANT_BYO_STALE_TTL)

# First sighting of an UNRESOLVED posture per (tenant, product), anchoring the
# grace window below. Mirrors ``_license_token_not_found_since``, including the
# re-pin on expiry, so a permanently-unresolvable selector settles on one answer
# instead of flapping. Evicted on the next successful lookup, so the window
# always measures a CONTINUOUS run of failures, never a total across an outage
# that already healed.
_tenant_byo_unresolved_since: TTLCache = TTLCache(maxsize=1000, ttl=_TENANT_BYO_STALE_TTL)

# How long a (tenant, product) whose posture CANNOT be resolved and for which
# this pod holds no last-known-good may keep running on platform inference.
#
# ⭐ THIS IS THE COST LINE OF THE WHOLE CHANGE. Read it before tuning it.
#
# Neither absolute is acceptable here:
#   * Fail OPEN forever (the behaviour this replaces) means a licences outage —
#     or, on the measured population, a caller sending a selector that is not a
#     tenant at all — is an unbounded, unbilled grant of owner GPU, invisible
#     except in a log line.
#   * Fail CLOSED immediately means every licences blip 402s live worker runs,
#     and 402 is deliberately excluded from requirements_management's retryable
#     statuses, so those runs do not come back on their own.
#
# So: last-known-good first (no invented verdict at all, and it covers every
# tenant this pod has served in the last TENANT_BYO_STALE_TTL), then a bounded
# window in which an UNKNOWN tenant still proceeds, then denial.
#
# WHAT IT COSTS WHEN LICENCES IS DOWN, stated plainly: for ~15 minutes nothing
# changes for a warm tenant and an unknown tenant still runs. Past that — i.e.
# licences unreachable CONTINUOUSLY for this tenant for the whole window — runs
# for tenants this pod has never resolved are refused with a 402 whose text
# talks about configuring a provider, which is not why it was refused. That is
# the known sharp edge, and it is the reason the window is generous.
#
# WHY THAT TERMINAL IS CONSISTENT: the token-carrying path ALREADY behaves this
# way. ``_license_stale_cache`` serves last-known-good for 900s and then the
# request gets a retryable refusal. A tokenless S2S run outliving that window
# with MORE entitlement than a token-carrying one is the asymmetry, not the fix.
#
# TUNING: one knob, no separate bypass flag. Raise it to restore an effectively
# unbounded fail-open (the ``grace_open`` series keeps counting, so the fail-open
# stays visible rather than becoming a silent allowlist); set it to 0 to deny as
# soon as there is no last-known-good.
_TENANT_BYO_UNRESOLVED_GRACE = float(os.getenv("TENANT_BYO_UNRESOLVED_GRACE_SECONDS", "900"))


def _licenses_base_url() -> str:
    """Derive the licenses service base URL from the introspect endpoint."""
    url = _license_introspect_url()
    return url.rsplit("/license/introspect", 1)[0] if "/license/introspect" in url else url


def _unresolved_byo_mode(
    cache_key: tuple[str, str | None],
    tenant_id: str,
    product: str | None,
    failure: str,
) -> str:
    """Decide a posture when licences could NOT be asked. Never invents silently.

    Split out of :func:`_lookup_tenant_byo_mode` so there is exactly ONE place
    that turns "I could not ask" into a mode, and so the three arms below cannot
    drift apart as branches get added to the call. It performs no I/O.

    The ladder, in order:

    1. **Last-known-good** — this pod successfully resolved this
       ``(tenant, product)`` inside ``TENANT_BYO_STALE_TTL``. Re-use THAT. This
       is not a fallback verdict; it is the tenant's own posture, so a
       ``byo_required`` tenant stays denied through a blip instead of silently
       becoming ``local_allowed``, which is what used to happen on every non-200.
       ⛔ It is NOT written back into ``_tenant_byo_cache``: a stale value must
       not outlive the outage by another 300s.
    2. **Bounded grace** — nothing is known about this tenant (cold pod, or a
       tenant this pod has never served). Proceed on platform routing, but only
       for ``TENANT_BYO_UNRESOLVED_GRACE_SECONDS`` measured from the FIRST
       unresolved sighting. This is the old behaviour, now bounded and counted.
    3. **Fail closed** — the window elapsed. Return ``byo_required``: a grant
       that cannot be verified is not a grant, and by this point the failure is
       not a blip. On the measured population the arm that actually lands here
       is a caller sending a selector that is not a registered tenant (a 404
       that will never heal), which is precisely the ~10% of S2S runs that were
       skipping enforcement.

    ⛔ The return value stays inside the existing three-value posture enum. A
    fourth "unresolved" value would have to be understood by ``filter_for_mode``
    and by every ``== "byo_required"`` comparison in the routers, i.e. it would
    change what ``byo_mode`` MEANS. The fact that a posture was fabricated lives
    on the counter and in the log, not in the value.
    """
    metrics = get_metrics_client()
    stale = _tenant_byo_stale_cache.get(cache_key)
    if stale is not None:
        logger.warning(
            "byo_mode_unresolved tenant_id=%r product=%r failure=%s — serving "
            "last-known-good %r (licences unreachable; posture NOT re-verified)",
            tenant_id,
            product,
            failure,
            stale,
        )
        _record_byo_lookup(metrics, TENANT_BYO_LOOKUP_STALE, failure)
        return cast("str", stale)

    now = time.monotonic()
    first = _tenant_byo_unresolved_since.get(cache_key)
    if first is None:
        _tenant_byo_unresolved_since[cache_key] = first = now
    elapsed = now - cast("float", first)
    if elapsed < _TENANT_BYO_UNRESOLVED_GRACE:
        logger.warning(
            "byo_mode_unresolved tenant_id=%r product=%r failure=%s elapsed=%.0fs "
            "window=%.0fs — FAILING OPEN to local_allowed inside the grace window; "
            "byo_required is NOT enforced for this run",
            tenant_id,
            product,
            failure,
            elapsed,
            _TENANT_BYO_UNRESOLVED_GRACE,
        )
        _record_byo_lookup(metrics, TENANT_BYO_LOOKUP_GRACE_OPEN, failure)
        return "local_allowed"

    # Re-pin the ORIGINAL first sighting so its TTL refreshes under continuous
    # traffic. Without this the since-entry evicts at its own TTL and the next
    # failure opens a FRESH window, so a permanently-unresolvable selector would
    # flap open→closed→open forever instead of settling. Same fix, same reason,
    # as ``_license_token_not_found_since`` above.
    _tenant_byo_unresolved_since[cache_key] = first
    logger.warning(
        "byo_mode_unresolved tenant_id=%r product=%r failure=%s elapsed=%.0fs "
        "window=%.0fs — grace expired, FAILING CLOSED to byo_required; this run "
        "is refused because the posture could not be verified, not because the "
        "tenant is commercial",
        tenant_id,
        product,
        failure,
        elapsed,
        _TENANT_BYO_UNRESOLVED_GRACE,
    )
    _record_byo_lookup(metrics, TENANT_BYO_LOOKUP_FAIL_CLOSED, failure)
    return "byo_required"


def _record_byo_lookup(metrics: Any, outcome: str, failure: str) -> None:
    """Book one lookup outcome, swallowing everything.

    ⛔ A metrics backend that is down must never change a posture decision — the
    caller has already chosen its return value before this runs. The client's
    own recorder swallows too; this second guard exists because the client is
    monkeypatched in tests and in ``_modality_router``, so the object here is not
    always that class.
    """
    try:
        metrics.record_tenant_byo_mode_lookup(outcome, failure=failure)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to record byo_mode lookup outcome: {e}")
    # Stash for the REQUEST, not just the metric — a route-level gate reached
    # later in this same dispatch (src/auth/tokenless_gate.py) needs to know
    # WHICH outcome this run got, and the counter above records the fact
    # without keeping it queryable. Never let a ContextVar write break the
    # lookup it is describing.
    try:
        set_tenant_byo_lookup_outcome(outcome)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to stash byo_mode lookup outcome in context: {e}")


async def _lookup_tenant_byo_mode(tenant_id: str, product: str | None = None) -> str:
    """Resolve a tenant's byo_mode by tenant_id via the licenses service.

    Used by the trusted-network (no-token) auth path so byo_required is enforced
    for S2S worker runs (requirements/kag/coding → WM) that forward only a raw
    X-Tenant-Id, not a token. An authoritative 200 is cached 5 min.

    ⚠️ THIS NO LONGER FAILS OPEN ON EVERY NON-200. It used to: any 404 / 401 /
    5xx / timeout returned ``local_allowed``, the most permissive value in the
    enum, which routes the run onto owner GPU. "I could not ask" and "licences
    says this tenant may use platform inference" were the same answer. They are
    now different: see :func:`_unresolved_byo_mode` for the ladder and for what
    the change costs while licences is down.

    ⛔ This coupling is CROSS-SERVICE and the two halves must be read together.
    licences answers this route 200 with a fabricated ``byo_required`` when its
    own DB lookup faults — chosen deliberately BECAUSE this caller fails open on
    anything else, so making the route 503 would have converted a fail-closed
    guess over there into an allow over here. This function is that argument's
    other half; the licences default can stay fail-closed on its merits rather
    than because this side cannot cope with a 503.

    ``product`` (optional): when given, resolve the PER-PRODUCT posture by
    passing ``?app=<product>`` to licenses — an override for that product wins
    over the tenant-wide value; absent an override licenses returns the
    tenant-wide value (backward compatible). Cache is keyed on
    ``(tenant_id, product)`` so per-product and tenant-wide results don't collide.
    """
    if not tenant_id:
        # No tenant was ASSERTED at all — a different defect from an
        # unresolvable one, on a path that never reaches licences. Unchanged,
        # and deliberately not booked on the lookup counter: no lookup happened,
        # and counting it would inflate the denominator the unresolved rate is
        # measured against.
        return "local_allowed"
    cache_key = (tenant_id, product)
    if cache_key in _tenant_byo_cache:
        return cast("str", _tenant_byo_cache[cache_key])
    # Bound before the ``try`` so the value can never be unbound at the return.
    # The initial value is the ``except`` arm's own value on purpose: the only
    # way to reach the return WITHOUT the non-200 leg having overwritten it is
    # to have raised, which is precisely what ``transport`` names.
    failure = TENANT_BYO_LOOKUP_FAILURE_TRANSPORT
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            params = {"app": product} if product else None
            resp = await client.get(
                f"{_licenses_base_url()}/tenants/{tenant_id}/byo-mode", params=params
            )
            if resp.status_code == 200:
                mode = str(resp.json().get("byo_mode") or "local_allowed")
                _tenant_byo_cache[cache_key] = mode
                # Record the good result for stale-while-error, and clear any
                # open unresolved window: the window must measure a CONTINUOUS
                # run of failures, not a total across an outage that healed.
                _tenant_byo_stale_cache[cache_key] = mode
                _tenant_byo_unresolved_since.pop(cache_key, None)
                _record_byo_lookup(
                    get_metrics_client(),
                    TENANT_BYO_LOOKUP_RESOLVED,
                    TENANT_BYO_LOOKUP_FAILURE_NONE,
                )
                return mode
            # NON-200 IS THE DANGEROUS BRANCH, AND IT USED TO BE SILENT.
            #
            # A 404/401/5xx is a successful HTTP exchange, so it never reaches the
            # `except` below. Control fell straight through to `return
            # "local_allowed"` and the run proceeded as if the tenant had no BYO
            # requirement — with NOTHING written anywhere. The symptom is not a
            # broken run; it is byo_required enforcement quietly not happening,
            # which is invisible until someone audits the licences side.
            #
            # MEASURED on the deployed pods 2026-08-19 (licenses,
            # licenses-fb44865c7-lrvw7, 6463 log lines): 8 of 79 byo-mode lookups
            # answered 404 — ~10% of S2S runs skipped enforcement. licenses logged
            # the refusal at ITS end (`byo_mode_tenant_unresolved`); this service,
            # the one that acts on the answer, logged nothing. Two selectors were
            # responsible, and neither is a registered tenant (73 rows in
            # `tenants`): `1769980520185202330` (not a UUID at all) and
            # `ad976f5b-…` (a UUIDv4 — tenant ids are UUIDv7).
            #
            # ⚠️ The line below used to end "FAILING OPEN to local_allowed", and
            # that was the whole decision. It no longer is: a non-200 now hands
            # off to ``_unresolved_byo_mode``, which prefers this pod's own
            # last-known-good and bounds the fail-open it can still choose. This
            # log stays because a COUNT cannot answer "which id failed", and
            # that id — not the rate — is the whole investigation. The two
            # measured selectors were `1769980520185202330` (not a UUID at all)
            # and `ad976f5b-…` (a UUIDv4 where tenant ids are UUIDv7).
            logger.warning(
                "byo_mode_lookup_non_200 tenant_id=%r product=%r status=%s "
                "— posture UNRESOLVED; see the byo_mode_unresolved line for the decision",
                tenant_id,
                product,
                resp.status_code,
            )
            failure = TENANT_BYO_LOOKUP_FAILURE_NON_200
    except Exception as e:
        # Timeout, connection error, or an unparseable body. licences (or the
        # path to it) is unhealthy — a different remedy from the non-200 above,
        # which is usually a caller sending a selector that is not a tenant, so
        # the two get different label values and never share a series.
        logger.warning(f"Tenant byo_mode lookup failed for {tenant_id} (product={product}): {e}")
        failure = TENANT_BYO_LOOKUP_FAILURE_TRANSPORT
    return _unresolved_byo_mode(cache_key, tenant_id, product, failure)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """License Token Authentication Middleware.

    Extracts Bearer token from X-Tenant-Id header and validates via license introspection.
    """

    def __init__(self, app: ASGIApp, bypass_auth: bool = False) -> None:
        super().__init__(app)
        self.bypass_auth = bypass_auth  # For testing

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Authenticate, then emit the observations that needed a route template.

        Split from :meth:`_authenticate` purely so the deferred-metric bracket
        is impossible to miss: every auth decision happens BEFORE the router
        runs, but the ``path`` label on this middleware's two counters is the
        MATCHED ROUTE TEMPLATE, which only exists AFTER it. The auditors queue
        their emissions on ``request.state`` and this ``finally`` flushes them —
        on every exit, served or refused or raised.
        """
        setattr(request.state, _DEFERRED_METRICS_ATTR, [])
        try:
            return await self._authenticate(request, call_next)
        finally:
            # ⛔ A ``finally``, not a line before the ``return``. ``_authenticate``
            # has a blanket ``except Exception``, so a raising handler is
            # converted to a 500 and either placement would work — but
            # ``CancelledError`` is a ``BaseException`` and sails past it. A
            # tokenless caller that times out and hangs up is exactly the traffic
            # these counters exist to measure, so dropping it would under-count
            # the callers that have to be migrated. Pinned by
            # ``test_a_cancelled_request_is_still_counted``.
            #
            # On the refusal and 404 legs no route was ever matched, so the label
            # is ``_UNMATCHED_PATH``. The counter's per-route breakdown really is
            # lost there; the LOG's is not, because the raw path joins the dedup
            # key on precisely that leg (see ``_audit_unauthenticated_tenant_assertion``).
            _flush_route_labelled_metrics(request)

    def _stash_identity_context(self, request: Request) -> None:
        """Relay the fleet-wide cost-attribution headers into ContextVars.

        Orthogonal to auth: set for EVERY authenticated branch. Every value is
        relayed verbatim and never invented — a stamped default would collapse
        distinct origins into one bucket on every cost event this service, and
        inference-router downstream of it, emits.

        Extracted from ``_authenticate`` so adding a dimension here does not
        push that method back over PLR0915. ⛔ Adding the 4th (``project_id``)
        is what tipped it from 70 to 71 statements; the fix is fewer statements,
        never a raised ceiling or a ``noqa``.
        """
        # Cost-correlation (fleet-wide): stash X-Correlation-Id + X-Stage so
        # the BYO cost emitter (and the outbound hook to inference-router) can
        # attribute LLM cost events back to the caller's operation (for w2v,
        # correlation_id=run_id, stage=DAG stage). Orthogonal to auth — set
        # for every authenticated branch. Empty header = no correlation.
        set_correlation_context(
            request.headers.get("x-correlation-id", ""),
            request.headers.get("x-stage", ""),
        )

        # Calling application id (fleet-wide X-App-Id): relayed, never
        # invented. worker_management fans in from many upstream apps
        # (n8n, requirements_management, asset-evaluator,
        # w2v_orchestrator, ...) — stamping a default here would collapse
        # all of their distinct origins into one bucket on every cost
        # event this service (and inference-router downstream of it)
        # emits. Orthogonal to auth — set for every authenticated branch,
        # exactly like correlation above. Empty header = no app_id.
        set_app_id_context(request.headers.get("x-app-id", ""))

        # Calling project id (fleet-wide X-Project-Id): the kanban project
        # a run belongs to, so downstream spend can be priced per project.
        # Relayed, never invented — same shape as app_id above, and
        # deliberately NOT uuid-validated here: inference-router validates
        # it at its own header-reading site (malformed → treated as
        # absent), and a second validator is a second place to disagree.
        # Orthogonal to auth — set for every authenticated branch. Empty
        # header = no project_id.
        set_project_id_context(request.headers.get("x-project-id", ""))
        # Caller job id (fleet-wide X-Job-Id, optional): relayed after a shape
        # check; the outbound/cost-event value falls back to the worker name.
        set_job_id_context(request.headers.get("x-job-id", ""))

        # Spend scope (fleet-wide X-Spend-Scope): the REPORTING-ONLY sibling of
        # X-Project-Id, letting billing distinguish "we failed to attribute this
        # spend to a project" from "this is tenant-wide work by design and there
        # is nothing to attribute it to" — today those collapse into one bucket
        # and the permanent residual reads as a failure.
        #
        # Relayed, never invented — same shape as project_id above. ⛔ But
        # UNLIKE project_id it IS validated, in `set_spend_scope_context`: this
        # is a closed two-value vocabulary reaching a report dimension, not an
        # opaque id whose only authority lives downstream. Anything other than
        # exactly `tenant` or `project` is dropped there and logged, never
        # coerced to a default. The validation deliberately does NOT live at
        # this call site — one validator in the setter covers every future
        # entry point; one here would only cover this one.
        #
        # ⛔ ORTHOGONAL TO project_id: a request carrying both forwards both.
        # Orthogonal to auth — set for every authenticated branch. Empty header
        # = no scope stated, which stays distinct from `tenant`.
        set_spend_scope_context(request.headers.get("x-spend-scope", ""))

        # AA-TRACE join keys (spec AA-TRACE §2/§3): X-AA-Trace-Id, X-AA-Card-Id,
        # X-AA-Parent-Exec, X-AA-Hop — relayed byte-for-byte, never validated or
        # incremented (X-Project-Id, the fifth, is relayed above). Always set, so
        # an absent header clears any value a reused context still held.
        set_aa_trace_context(request.headers)
        # Prompt attribution is per-CALL, set by the worker path before each LLM
        # call; reset here so no request can inherit a previous one's prompt.
        set_prompt_attribution(None, None)

        # Acting person (fleet-wide X-Actor-Id): ⛔ NOT relayed here, and NOT
        # orthogonal to auth like the dimensions above. It is CLEARED here and
        # set only by `_relay_actor_id` once the auth branch has decided whether
        # the caller may assert a person on someone else's behalf. See there.
        set_actor_id_context("")
        set_external_request_context(external=False)

    @staticmethod
    async def _relay_actor_id(
        request: Request,
        *,
        trusted: bool,
        auth_branch: str,
        caller_tenant: str = "",
        own_sub: str = "",
        end_user_sub: str = "",
    ) -> None:
        """Honour inbound X-Actor-Id only when the caller may assert it (D4).

        X-Actor-Id is WHO ACTED — billing rolls it into
        ``billing_cost_actor_daily`` per member, so an actor a caller cannot
        vouch for would bill one member's spend to another.

        Accepted when:
          * ``trusted`` — a DELEGATING licence subject (``AUTH_DELEGATING_SUBJECTS``)
            or the tokenless ``TRUSTED_CIDRS`` S2S branch, which already assert
            the TENANT on someone's behalf (a strictly larger claim);
          * otherwise (an ordinary licence caller, e.g. svc-cdmsfirst-n8n):
            (a) the actor IS the caller's own registered service-account sub —
                licences' introspection ``subject_keycloak_sub`` (``own_sub``),
                cached with the introspection; or
            (b) the actor is a Keycloak user whose licences membership
                (``subject.tenant_id``) equals the caller's tenant — cached
                lookup with WM's own credential (``actor_membership``).
        A licence-branch end user VERIFIED from ``X-End-User-Token``
        (``end_user_sub``) is the actor when the header is absent, and a header
        equal to it is accepted. Anything else is DROPPED — never coerced, never
        defaulted to the licence subject — counted on ``worker_management_actor_rejected_total{reason}``
        and logged at INFO. Every request that ends without an accepted actor is
        also counted on ``worker_management_actor_missing_total{caller}``.
        """
        actor = (request.headers.get("x-actor-id", "") or "").strip() or end_user_sub
        accepted = ""
        # Reached only for an AUTHENTICATED external caller (token or trusted
        # network). From here on a missing actor is the CALLER's gap: it stays
        # absent downstream instead of falling back to SERVICE_ACTOR_ID.
        set_external_request_context(external=True)
        set_actor_id_context("")
        reason = ""
        if actor:
            shaped = valid_actor_id(actor)
            if not shaped:
                reason = ACTOR_REJECTED_MALFORMED
            elif (
                trusted
                or (own_sub and shaped.lower() == own_sub.strip().lower())
                or (end_user_sub and shaped.lower() == end_user_sub.strip().lower())
            ):
                accepted = set_actor_id_context(shaped)
            elif not caller_tenant:
                reason = ACTOR_REJECTED_UNTRUSTED
            else:
                from src.auth.actor_membership import SAME_TENANT, actor_tenant_verdict

                try:
                    verdict = await actor_tenant_verdict(shaped, caller_tenant)
                except Exception:
                    logger.warning("actor membership lookup raised", exc_info=True)
                    verdict = ACTOR_REJECTED_LOOKUP_UNAVAILABLE
                if verdict == SAME_TENANT:
                    accepted = set_actor_id_context(shaped)
                else:
                    reason = verdict
        if reason:
            ACTOR_REJECTED_TOTAL.labels(reason=reason).inc()
            logger.info(
                "X-Actor-Id rejected reason=%s caller=%s tenant=%s trusted=%s path=%s len=%d",
                reason,
                _actor_missing_caller(),
                caller_tenant or "<none>",
                trusted,
                request.url.path[:200],
                len(actor),
            )
        if not accepted:
            ACTOR_MISSING_TOTAL.labels(caller=_actor_missing_caller()).inc()
        # The X-Project-Id gate rides on this call because it is the ONE call
        # every authenticated external request makes exactly once (licence via
        # `_stamp_identity`, tokenless via the trusted-network branch) — placing
        # it in `_authenticate` itself tips PLR0915. May raise `_ProjectIdRefused`.
        # bypass_auth never reaches here: a test affordance, not a caller.
        AuthenticationMiddleware._gate_project_id(request, auth_branch)

    @staticmethod
    def _gate_project_id(request: Request, auth_branch: str) -> None:
        """Count a missing/invalid X-Project-Id; raise a 400 refusal iff the flag is on.

        Called once per AUTHENTICATED request, from ``_relay_actor_id`` — after
        the credential is checked, so an unauthenticated request is a 401, never
        a 400, and is not counted. The relay in ``_stash_identity_context`` is untouched: a valid
        value is forwarded exactly as sent (no case normalisation — IR parses
        any case and ``valid_actor_id`` likewise forwards unchanged).
        Observability never fails a request: a counter error is logged and the
        request proceeds.
        """
        reason = _project_id_problem(request.headers.get("x-project-id"))
        if not reason:
            return
        caller = _project_id_caller(auth_branch)
        try:
            PROJECT_ID_MISSING_TOTAL.labels(caller=caller, reason=reason).inc()
        except Exception:
            logger.warning("project_id_missing counter failed", exc_info=True)
        if not get_settings().require_project_id_header:
            return
        from fastapi.responses import JSONResponse

        try:
            PROJECT_ID_REFUSED_TOTAL.labels(caller=caller, reason=reason).inc()
        except Exception:
            logger.warning("project_id_refused counter failed", exc_info=True)
        logger.info(
            "X-Project-Id refused reason=%s caller=%s path=%s",
            reason,
            caller,
            request.url.path[:200],
        )
        # Same envelope as every other 400 this middleware returns
        # (``_bad_tenant_response``), plus ``reason`` so a caller can tell
        # "forgot it" from "sent garbage" without parsing the message.
        raise _ProjectIdRefused(
            JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "PROJECT_ID_REQUIRED",
                        "reason": reason,
                        "message": "X-Project-Id must be the fr project UUID this call belongs to",
                        "status": 400,
                    }
                },
            ),
            "project id required",
        )

    async def _apply_verified_caller_identity(self, request: Request) -> None:
        """Set the verified caller context; raise a 401 refusal iff a PRESENT caller token is invalid."""
        try:
            caller_client_id, caller_roles = await verified_caller_identity(request)
        except CallerTokenInvalidError as invalid:
            raise _CallerTokenRefused(
                self._caller_token_invalid_response(invalid.failure_class),
                "caller token invalid",
            ) from None
        set_caller_context(caller_client_id, caller_roles)

    async def _authenticate(self, request: Request, call_next: Callable) -> Response:
        """Process authentication via license token."""
        try:
            # Skip authentication for public endpoints
            if self._is_public_endpoint(request.url.path):
                return await call_next(request)

            # Caller-identity. This used to read x-caller-client-id +
            # x-caller-roles straight off the request, on the premise that
            # license-authz had verified an upstream JWT and injected them.
            # For worker_management that premise was false in both directions:
            # the ingress EnvoyFilter never stripped those two headers and
            # discards the verified ones license-authz returns, and the
            # `workers` namespace is ambient with no waypoint, so no in-mesh
            # ext_authz runs here either. The header was attacker-supplied end
            # to end, and graph_sessions_correlation_api traded it for an
            # unscoped cross-tenant read.
            #
            # Verify the JWT ourselves instead, on the channel this codebase
            # already designed for it (X-Caller-Authorization — see
            # src/auth/caller_identity.py, which emits exactly that outbound).
            # There is deliberately NO fallback to the old headers: a
            # `verified or header` line would keep the vulnerability alive
            # behind a green test.
            #
            # A PRESENT caller token that fails validation is a 401 — not a
            # silent downgrade to the request's own licence/trusted-network
            # auth. Absent header, JWKS unreachable, and a VALID token whose
            # azp is not allowlisted still return ("", ()) and carry on
            # (see keycloak_roles' failure policy). Checked before bypass_auth
            # so the verdict is identical in tests and production.
            await self._apply_verified_caller_identity(request)

            self._stash_identity_context(request)

            # Skip auth if bypassed (for testing)
            if self.bypass_auth:
                # `tenant_id` is a `uuid` column (migration 024), so the literal
                # "test_user" this used to stamp is now a hard 22P02 on the first
                # statement — i.e. bypass_auth would break every test that
                # touches Postgres. BYPASS_TENANT_UUID is the deterministic uuid5
                # of that same label, so the value is still recognisably "the
                # bypass tenant" and the crosswalk explains it if one ever leaks.
                # `user_id` stays the readable label: it is an identity string,
                # not a tenant column.
                request.state.user = {"user_id": "test_user", "tenant_id": BYPASS_TENANT_UUID}
                request.state.tenant_id = BYPASS_TENANT_UUID
                request.state.tenant_name = "test_user"
                request.state.license_id = None
                request.state.subject_id = None
                # Bypass is a TEST affordance, not a proof of identity. Nothing
                # was presented and nothing was verified, so it is stamped
                # unproven exactly like the tokenless branch — a handler that
                # gates a secret on `identity_proven` must not become open
                # simply because auth was switched off. Same for the tenant:
                # BYPASS_TENANT_UUID is a constant this middleware chose, not a
                # tenant any caller established, so it is not proven either.
                self._stamp_nothing_proven(request)
                set_identity_context(
                    BYPASS_TENANT_UUID,
                    None,
                    None,
                    # Its own value, not `unknown`: a cost event stamped
                    # `bypass` says auth was SWITCHED OFF, which is a
                    # deployment-shape finding, not a caller one.
                    tenant_provenance=TENANT_PROVENANCE_BYPASS,
                )
                return await call_next(request)

            # Extract token from X-Tenant-Id header (Bearer format)
            token = self._extract_license_token(request)
            x_tenant_id = request.headers.get("X-Tenant-Id")

            # If a token is provided, always validate it (even from trusted networks)
            if token:
                license_result = await _introspect_license(token)
                if license_result is None:
                    # Transient introspection failure (licenses slow/unreachable,
                    # cold stale cache) — NOT a rejected token (a bad token
                    # returns 200 active:false -> a dict -> 401 below). Return a
                    # retryable 503 so the caller's retry path rides out the blip
                    # across worker_management replicas, instead of a spurious 401
                    # hard-failing a long run's tail LLM stage on a cold replica
                    # (the growth_bundle 401 class). 503 never grants access.
                    return self._service_unavailable_response(
                        "License introspection temporarily unavailable; retry"
                    )
                if not license_result.get("active"):
                    return self._unauthorized_response(
                        "Invalid or expired license token",
                        reason=AUTH_REFUSED_LICENSE_INACTIVE,
                        request=request,
                        upstream_reason=str(license_result.get("reason") or ""),
                    )

                # tenant_id and subject_id are distinct identities — never
                # conflate them. (Pre-2026-04-27 this line preferred subject_id
                # as a fallback because introspect always returned subject_id
                # = None; now that license-authz returns a real Subject UUID
                # for end-user tokens, that fallback would stamp the user's
                # UUID as the tenant.)
                tenant_id = self._canonical_introspected_tenant(license_result.get("tenant_id"))

                # Per Phase 5a of the per-user metering plan: stash subject_id
                # (the canonical end-user UUID from the licenses Subject row)
                # alongside tenant_id so outbound S2S calls can forward it as
                # X-Subject-Id and inference-router metering can attribute LLM
                # cost to the originating user. May be None for tokens that
                # are not bound to a Subject (e.g. machine-to-machine licenses
                # or pre-Phase-1 tokens); call sites must default to "" in
                # that case (never the literal string "None").
                subject_id_raw = license_result.get("subject_id")
                subject_id: str | None = str(subject_id_raw) if subject_id_raw is not None else None
                license_id_raw = license_result.get("license_id")
                license_id: str | None = str(license_id_raw) if license_id_raw is not None else None

                # Effective-tenant delegation: a trusted service account may
                # re-scope this request to the run's real tenant via
                # X-Effective-Tenant-Id, so BYO resolution + the cost event follow
                # the run tenant rather than the service-account tenant. Gated on
                # the introspected subject being a configured delegator; a customer
                # subject is never listed (no spoofing). Empty config ⇒ no-op.
                delegated = False
                auth_subject = subject_id_raw
                # TENANT BINDING (port of requirements_management #222/#223) --
                # see `_bind_tenant_claims`. Raises 403 for a non-delegating
                # caller naming a foreign tenant; `eff_raw` is the tenant a
                # DELEGATING caller asked to act as ("" = none).
                # ...then the tenant×app licence check (George 2026-10-15), see
                # `_bind_tenant_and_app`; under enforce it raises a 401.
                delegating, eff_raw = self._bind_tenant_and_app(
                    request, str(tenant_id), auth_subject, license_result
                )
                if delegating:
                    # ⛔ THIS HEADER IS A SECOND MOUTH ON THE SAME PIPE. It was
                    # `.strip()`ed and adopted with no validation at all, so a
                    # delegating service account could re-scope a request to any
                    # string it liked — `sales`, or `'; DROP--` — and that value
                    # reached `scope_clause` and the database. `tenant_id` is a
                    # `uuid` column now, so the same input is a 22P02 blamed on
                    # Postgres. Resolving it here means anything unresolvable is
                    # refused, exactly as on the tokenless path. Found by an
                    # adversarial agent AFTER I had "closed" the boundary — one
                    # entry point is not the boundary.
                    #
                    # ⛔ AND IT HAPPENED AGAIN, 2026-09-16. This comment used to
                    # end "means a NAME still delegates", and that sentence was
                    # true and was the bug: `_effective_tenant("n8n-dev")`
                    # returned the SEED tenant. The tokenless-header fix for
                    # issue #264 closed one door and left this one open, with the
                    # PR's own scope section not even listing it — caught by a
                    # second adversarial audit. Both are now
                    # `resolve_asserted_tenant_id`. THE LESSON DID NOT TAKE THE
                    # FIRST TIME: when a value class is wrong, fix every site
                    # that admits it, not the site the ticket named.
                    eff_tenant = self._effective_tenant(eff_raw)
                    if eff_tenant and eff_tenant != str(tenant_id):
                        # ⛔ `raw=` IS THE WHOLE POINT OF THIS LINE, AND IT WAS
                        # MISSING UNTIL 2026-09-16. Until now this logged only
                        # `eff_tenant` — the RESOLVED uuid — so a caller sending
                        # the NAME `n8n-dev` and one sending the uuid
                        # `aaaaaaaa-…0001` produced BYTE-IDENTICAL lines. That
                        # made the one question anybody needs to ask about this
                        # header unanswerable after the fact: measured over 30d,
                        # 58,710 of 72,351 delegated requests (81.1%) resolved to
                        # a uuid that a name could equally have produced, and
                        # nothing recorded which spelling arrived.
                        #
                        # The tokenless branch has carried its raw value all
                        # along, which is the only reason THAT door could be
                        # closed on evidence (1,501 assertions / 30d, exactly one
                        # name-valued, and it was our own probe). This line is
                        # that same instrument for this door. Truncated to 64
                        # chars like its sibling: it is caller-controlled input
                        # being written to a log.
                        #
                        # ⚠️ Do not "tidy" this by dropping `raw` when it equals
                        # the resolved value. Equality IS the measurement — it is
                        # how a uuid-sender is told apart from a name-sender, and
                        # a line that omits the field when they match leaves
                        # exactly the ambiguity this exists to remove.
                        logger.info(
                            "effective-tenant delegation: subject=%s acting as "
                            "tenant=%s (raw=%r resolved_from_name=%s) "
                            "(service-account tenant=%s)",
                            auth_subject,
                            eff_tenant,
                            eff_raw[:64],
                            eff_raw.strip() in tenant_uuids(),
                            tenant_id,
                        )
                        tenant_id = eff_tenant
                        fwd_subject = (request.headers.get("X-Subject-Id") or "").strip()
                        fwd_license = (request.headers.get("X-License-Id") or "").strip()
                        subject_id = fwd_subject or subject_id
                        # ⛔ DO NOT fall back to the introspected licence here.
                        # `tenant_id` has just been re-scoped to the CUSTOMER, but
                        # the introspected licence belongs to the SERVICE ACCOUNT.
                        # Keeping it stamps customer spend on someone else's
                        # entitlement — attributed, and wrong, which is worse than
                        # visibly unattributed because nothing downstream can tell.
                        #
                        # Measured 2026-09-02, when `WORKER_MANAGEMENT_LICENSE_TOKEN`
                        # was first set on `requirements` (gitops #3274) and this
                        # branch went live for the conflict-reconciler sweeper:
                        #   licence-owner != billed-tenant rows, 12h BEFORE:    11
                        #   licence-owner != billed-tenant rows, 11.5h AFTER: 3,411
                        # 3,397 of those carried svc-requirements' "Coding Module"
                        # licence across NINE tenants that do not own it, and drew
                        # 10,771,443 tokens against a licence whose quota_monthly
                        # is 1,000,000 — 10.8x, in half a day.
                        #
                        # `None` is the honest answer: this request has no licence
                        # belonging to the tenant it is billed to. billing's
                        # `resolve_license(tenant_id, service_key)` backstop can
                        # then resolve one the ADOPTED tenant actually owns, and
                        # where it cannot the row stays unattributed and visible to
                        # BillingUnattributedVolumeSustained. A caller that DOES
                        # forward `X-License-Id` is unaffected — `fwd_license` wins,
                        # exactly as before.
                        #
                        # ⚠️ SCOPE, stated: `subject_id` above has the same shape of
                        # problem (a service-account subject on a customer-scoped
                        # request). It is left alone deliberately — billing keys
                        # attribution on `license_id` only
                        # (`billing_service src/models/db/rollups.py`
                        # `license_attributed_clause`), so changing it here would be
                        # an unmeasured behaviour change riding along with a
                        # measured one.
                        license_id = fwd_license or None
                        delegated = True
                else:
                    # A non-delegating caller asserting a DIFFERENT tenant is a
                    # spoof attempt or a caller misconfigured against this
                    # receiver's AUTH_DELEGATING_SUBJECTS. Surface it (the
                    # introspected tenant still wins). The S2S hook stamps this
                    # header unconditionally, so an equal-tenant value is a benign
                    # no-op on the normal non-delegating path — don't warn on it
                    # (fleet-wide spam that drowns the real-re-scope signal).
                    # Deliberately NOT resolved: on this branch the header is
                    # ignored (the introspected tenant wins), and a value that is
                    # never used must not be able to fail the request. Validating
                    # here would turn a misconfigured-but-harmless caller into a
                    # 400. Only the delegating branch above, where the value is
                    # actually adopted, is fail-closed.
                    eff_tenant = (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
                    if eff_tenant and eff_tenant != str(tenant_id):
                        # ⚠️ `subject=` ALONE WAS NOT ACTIONABLE (#242). Measured
                        # over 7 days this fires 17 times, all from ONE subject,
                        # most recently 2026-09-16T17:20Z -- so the population is
                        # a single misconfigured caller, and two of the facts
                        # needed to find it were missing: WHICH route it is
                        # hitting, and WHICH tenant it actually got stamped with
                        # (the introspected one always wins on this branch).
                        #
                        # ⛔ THE REJECTED HEADER VALUE ITSELF (`eff_tenant`, the
                        # tenant the caller WANTED) IS DELIBERATELY NOT LOGGED,
                        # and this comment used to claim the pair `header wanted
                        # X, ran as Y` was the whole diagnosis while only Y ever
                        # shipped. Corrected rather than closed by adding X: on
                        # THIS branch `eff_tenant` is unvalidated caller-supplied
                        # free text -- the block above says so ("Deliberately NOT
                        # resolved"; the introspected tenant wins and the header
                        # is never used) -- so logging it would add a second
                        # attacker-controlled substring to this line. For a
                        # population of ONE known subject the wanted-tenant is
                        # recoverable once from that caller's config; it does not
                        # need to be on every request. If the population ever
                        # stops being one, add it length-capped and `%r`-quoted,
                        # not raw.
                        #
                        # ⚠️ `request.url.path` IS PERCENT-DECODED, so it is
                        # already one such substring. uvicorn `unquote`s the raw
                        # path into `scope["path"]` (httptools_impl.py:259,
                        # h11_impl.py:201), so a caller holding a valid token can
                        # request `/api/workers/x%0asubject=...` and forge a
                        # newline plus a fake field into this log. Pre-existing
                        # class, token-holder-only, and NOT closed here -- stated
                        # so it is a known cost of this line rather than
                        # something a later reader discovers.
                        #
                        # ⛔ STILL NOT A 401, DELIBERATELY. Rejecting here is
                        # blocked on a Keycloak change; until that lands this
                        # stays exactly as loud as it was -- three log fields
                        # instead of one, identical control flow, identical
                        # response. Adding the refusal in the same edit as the
                        # fields would ship an unmeasured behaviour change riding
                        # along with a pure observability one.
                        logger.warning(
                            "effective-tenant header rejected (subject not delegating): "
                            "subject=%s path=%s stamped_tenant=%s",
                            auth_subject or "<missing>",
                            request.url.path,
                            tenant_id,
                        )

                # ⛔ `not delegated` IS THE WHOLE SECOND FLAG, AND IT IS NOT THE
                # SAME QUESTION AS `identity_proven`. On this branch a licence
                # token was introspected `active`, so WHO the caller is was
                # proven either way. But when `delegated` is True, `tenant_id`
                # above is no longer the tenant that licence established — it
                # was REPLACED by the `X-Effective-Tenant-Id` value, which
                # `_effective_tenant` checked for SPELLING only, exactly like the
                # tokenless branch's bare `X-Tenant-Id`. See
                # `src.auth.get_tenant_proven` for why the two facts are stamped
                # separately and `_modality_router._key_for_caller` for the one
                # consumer that needs the difference.
                # LICENCE BRANCH END USER (George 2026-10-15): for a NON-delegating
                # caller `_stamp_identity` verifies X-End-User-Token against the
                # licence tenant and adopts the user as subject (refusing a bad
                # one); it returns the verified payload, else None. Hence
                # `request.state.subject_id` below, not the local `subject_id`.
                licence_end_user = await self._stamp_identity(
                    request,
                    str(tenant_id),
                    license_id,
                    subject_id,
                    license_result,
                    tenant_proven=not delegated,
                    # WHO ACTED: fully trusted only from a DELEGATING subject —
                    # the same predicate as X-Effective-Tenant-Id above, not
                    # `delegated` (that also needs the tenant header). Any other
                    # licence caller may name only its own SA sub or a member of
                    # its own tenant (D4).
                    actor_trusted=bool(auth_subject)
                    and str(auth_subject) in _delegating_subjects(),
                    verify_end_user=not delegating,
                )
                # BYO LLM entitlement (per-tenant, from the licenses service via
                # introspection). Absent => 'local_allowed' (fail-open to today's
                # behavior; commercial tenants are explicitly set 'byo_required').
                # BYO mode must reflect the EFFECTIVE tenant: when delegating, the
                # SA token's byo_mode (often platform_only) would wrongly block the
                # run tenant's BYO. Re-resolve for the effective tenant (fail-open).
                if delegated:
                    request.state.byo_mode = await _lookup_tenant_byo_mode(str(tenant_id))
                else:
                    request.state.byo_mode = str(license_result.get("byo_mode") or "local_allowed")
                # ⛔ `tenant_proven` DIES AT THE EDGE OF `request.state`. The
                # stamp above records that this tenant was adopted rather than
                # proven, but `request.state` is reachable only from code that
                # holds the Request — and the BYO cost event is emitted deep in
                # the agent run (`pydantic_ai_agent` -> `cost_events`), which
                # holds only these ContextVars. So the emitter billed the
                # adopted tenant with NOTHING on the record distinguishing it
                # from a licensed one. Carrying the same fact here is what makes
                # the difference survive into the billing row.
                #
                # ⚠️ NOT a refusal, and deliberately not one. Attribution to the
                # adopted tenant is INTENDED — that is why `license_id` is
                # nulled above rather than kept. Four live service accounts
                # delegate; refusing their events would drop real revenue and
                # replace a lying field with its inverse. Until someone decides
                # who may act for whom, the honest move is to SAY which it was.
                set_identity_context(
                    str(tenant_id),
                    license_id,
                    request.state.subject_id,
                    tenant_provenance=(
                        TENANT_PROVENANCE_DELEGATED if delegated else TENANT_PROVENANCE_LICENSE
                    ),
                )
                # Owner ruling 2026-10-01: a delegated request may carry the
                # signed-in user's portal JWT in X-End-User-Token. Verified here
                # (signature/issuer/expiry/tenant_id == effective tenant) and
                # recorded for `tenant_key_entitled`; never refuses the request.
                await _stamp_end_user_proof_if_delegated(
                    request,
                    delegated=delegated,
                    tenant_id=str(tenant_id),
                    delegator=auth_subject,
                    licence_end_user=licence_end_user,
                )
                logger.debug(
                    f"Authenticated via license token: tenant={tenant_id} subject={subject_id}"
                )
                return await call_next(request)

            # No token — check if request is from a trusted network
            client_host = request.client.host if request.client else None
            if _is_trusted_network(client_host):
                # ⛔ THIS LINE IS THE MOUTH OF THE PIPE. Until 2026-08-25 it read
                #     tenant_id = x_tenant_id.strip() if x_tenant_id else DEFAULT_TENANT_ID
                # i.e. whatever a caller on the pod network put in the header was
                # stored as a tenant id, verbatim and unvalidated. That is how
                # 93,842 rows of `worker_states` came to hold values like
                # `sales`, `auto`, `__system_workflow__` and bare snowflake
                # integers. `tenant_id` is now a real `uuid` column, so an
                # unresolved label would fail every statement of the request with
                # `invalid input syntax for type uuid` — a 500 blamed on the
                # database instead of a 400 blamed on the caller.
                #
                # ⛔ A NAME NO LONGER WORKS, AS OF 2026-09-16, AND THAT IS THE
                # POINT. It used to: `resolve_tenant_id` maps `n8n-dev` to
                # `…0001`, while the licences registry has `n8n-dev` as
                # `aaaaaaaa-…` and `…0001` as the SEED tenant
                # `n8n-dev-portal-realm`. A caller aiming at production landed on
                # the seed tenant with a 200 and wrote there. `fr-test` collides
                # the same way. `resolve_asserted_tenant_id` refuses names for
                # that reason — there is no correct resolution to choose, so the
                # only non-misattributing answer is to refuse.
                #
                # Anything unresolvable is likewise refused here, deliberately
                # rather than falling back to DEFAULT_TENANT_ID — see
                # `resolve_tenant_id`'s docstring for why a silent fallback is
                # the worse failure.
                tenant_id = self._trusted_network_tenant(x_tenant_id)
                # Resolving the value only proves it is SPELLED like a tenant.
                # Nothing here proved the caller is entitled to it. Record that
                # fact on every such request, and refuse it once
                # WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT is on. See the
                # method docstring — the default is OFF on purpose.
                refusal = self._audit_unauthenticated_tenant_assertion(
                    request, x_tenant_id, str(tenant_id), client_host
                )
                if refusal is not None:
                    return refusal
                request.state.user = {
                    "user_id": str(tenant_id),
                    "tenant_id": str(tenant_id),
                    "sub": str(tenant_id),
                }
                request.state.tenant_id = str(tenant_id)
                request.state.tenant_name = request.headers.get("X-Tenant-Name", "")
                # ⛔ THE TENANT ON THIS BRANCH IS A CLAIM, NOT A PROOF. Membership
                # of TRUSTED_CIDRS admitted the caller and a bare X-Tenant-Id
                # header named the tenant; `_trusted_network_tenant` checked only
                # that the value is SPELLED like a tenant this deployment knows.
                # Record that distinction as a first-class fact on the request so
                # a handler holding something that must not be handed to an
                # unproven caller (a decrypted BYO provider key — see
                # `src/api/tenant_config/_modality_router.py::resolve_modality_config`)
                # can gate on WHICH BRANCH RAN, rather than re-deriving it from
                # the CIDR or from the absence of a licence id.
                #
                # This is a STAMP, not a gate: it changes no request outcome here.
                # The refusal for this branch is
                # `require_authenticated_tenant`, which stays default-OFF.
                # And the tenant is unproven for the SAME reason, so both flags
                # say no here. They are still SEPARATE flags: the delegating
                # token branch is the case where they disagree, and a consumer
                # reading only one of them would be wrong on exactly that case.
                self._stamp_nothing_proven(request)
                # Trusted-network path has no introspected user identity;
                # leave subject_id / license_id unset so outbound calls
                # forward "".
                #
                # ⛔ THESE TWO LINES ARE WHERE THE BILL LOSES ITS OWNER. They
                # were, until now, the only unlogged and uncounted step in this
                # whole branch: the tenant assertion above at least warns and
                # increments, but the licence and the subject were dropped in
                # silence. Every cost event downstream of a tokenless
                # service-to-service caller is therefore structurally
                # licence-less and subject-less — the recorded mechanism behind
                # billing's historically ~53% NULL-licence rate. Count it, so
                # the omission has a number instead of being inferred from a
                # NULL rate measured three services away.
                self._audit_identity_omission(request, x_tenant_id)
                request.state.license_id = None
                request.state.subject_id = None
                request.state.license = None
                # BYO enforcement for tokenless S2S runs: the token path reads
                # byo_mode from introspect, but trusted-network callers
                # (requirements/kag/coding → WM) forward only a raw X-Tenant-Id.
                # Resolve byo_mode by tenant_id so a byo_required tenant can't
                # reach the platform LLM through a worker app. Fail-open inside.
                request.state.byo_mode = await _lookup_tenant_byo_mode(str(tenant_id))
                # `asserted` when the caller NAMED a tenant, `default` when it
                # named none and landed on this deployment's DEFAULT_TENANT_ID.
                # Split for the same reason `WORKFLOW_ROUTING_AUTH_PATH_DEFAULT`
                # is split from `_ASSERTED`: a bill attributed to the default
                # tenant because nobody said otherwise is a different (and
                # worse) finding than one attributed to a tenant a caller chose.
                set_identity_context(
                    str(tenant_id),
                    None,
                    None,
                    tenant_provenance=(
                        TENANT_PROVENANCE_ASSERTED
                        if (x_tenant_id or "").strip()
                        else TENANT_PROVENANCE_DEFAULT
                    ),
                )
                await self._relay_actor_id(
                    request, trusted=True, auth_branch=PROJECT_ID_CALLER_TRUSTED_NETWORK
                )
                logger.debug(f"Trusted network access from {client_host}: tenant={tenant_id}")
                return await call_next(request)

            # Not trusted, no token — reject
            return self._unauthorized_response(
                "Missing authentication token",
                reason=AUTH_REFUSED_MISSING_TOKEN,
                request=request,
            )

        except _RequestRefused as refused:
            # One handler for every tenant-resolution site (tokenless header,
            # delegation header, introspected value) and for an invalid PRESENT
            # caller token. Placed BEFORE the blanket `except Exception` below,
            # which would otherwise turn a caller's bad tenant or bad caller
            # token into a 500 blamed on this service.
            return refused.response
        except Exception:
            logger.exception("Authentication middleware error")
            return self._internal_error_response("Authentication service error")

    @staticmethod
    async def _licence_end_user(request: Request, licence_tenant: str) -> dict | None:
        """Verified ``X-End-User-Token`` payload for a NON-delegating licence caller.

        ``None`` when the header is absent (today's behaviour: the licence's own
        subject). A PRESENT token is verified by
        ``end_user_token.verify_licence_end_user`` (the delegated branch's
        ``_verified_payload`` + ``token_proves_tenant``, against the licence
        tenant) and anything short of ``ok`` raises ``_RequestRefused``:
        401 ``END_USER_TOKEN_INVALID`` (bad signature/issuer/expiry/typ/azp/sub),
        403 ``END_USER_TENANT_MISMATCH`` (valid user, not a member of the
        licence's tenant), 503 ``END_USER_TOKEN_UNVERIFIABLE`` (JWKS unreachable
        or the portal realm unconfigured; retryable). Never echoes the token.
        """
        from src.auth.end_user_token import (
            END_USER_TOKEN_HEADER,
            LICENCE_END_USER_FOREIGN_TENANT,
            LICENCE_END_USER_UNVERIFIABLE,
            PROOF_ABSENT,
            PROOF_OK,
            verify_licence_end_user,
        )

        payload, reason = await verify_licence_end_user(
            request.headers.get(END_USER_TOKEN_HEADER), licence_tenant
        )
        if reason == PROOF_ABSENT:
            return None
        if reason == PROOF_OK and payload is not None:
            return payload
        if reason in LICENCE_END_USER_UNVERIFIABLE:
            code, http, metric = (
                "END_USER_TOKEN_UNVERIFIABLE",
                status.HTTP_503_SERVICE_UNAVAILABLE,
                AUTH_REFUSED_END_USER_UNVERIFIABLE,
            )
        elif reason in LICENCE_END_USER_FOREIGN_TENANT:
            code, http, metric = (
                "END_USER_TENANT_MISMATCH",
                status.HTTP_403_FORBIDDEN,
                AUTH_REFUSED_END_USER_FOREIGN_TENANT,
            )
        else:
            code, http, metric = (
                "END_USER_TOKEN_INVALID",
                status.HTTP_401_UNAUTHORIZED,
                AUTH_REFUSED_END_USER_TOKEN_INVALID,
            )
        raise _RequestRefused(
            AuthenticationMiddleware._end_user_refused_response(
                request, code, http, metric, reason
            ),
            code,
        )

    @staticmethod
    def _end_user_refused_response(
        request: Request, code: str, http_status: int, metric: str, reason: str
    ) -> Response:
        """The licence-branch ``X-End-User-Token`` refusal (see ``_licence_end_user``)."""
        from fastapi.responses import JSONResponse

        AuthenticationMiddleware._record_auth_refusal(metric, request, reason)
        headers: dict[str, str] = {}
        if http_status == status.HTTP_401_UNAUTHORIZED:
            # error="invalid_token" names the END-USER token, so a caller's
            # licence-replay path (plain `Bearer` challenge) is not triggered.
            headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'
        elif http_status == status.HTTP_503_SERVICE_UNAVAILABLE:
            headers["Retry-After"] = "1"
        return JSONResponse(
            status_code=http_status,
            content={
                "error": {
                    "code": code,
                    "message": (
                        f"X-End-User-Token refused: {reason}. Send a valid portal access "
                        "token of a member of the licence's tenant, or omit the header."
                    ),
                    "failure_class": reason,
                    "status": http_status,
                }
            },
            headers=headers,
        )

    def _bind_tenant_and_app(
        self,
        request: Request,
        licence_tenant: str,
        auth_subject: object,
        license_result: dict[str, Any],
    ) -> tuple[bool, str]:
        """``_bind_tenant_claims``, then ``_check_licence_app`` on its delegation verdict.

        One call site so the app check runs right after the delegation decision
        (``delegated`` is one of its outcomes) and before any identity stamp
        (enforce refuses by raising ``_RequestRefused``).
        """
        delegating, eff_raw = self._bind_tenant_claims(request, licence_tenant, auth_subject)
        self._check_licence_app(
            request, license_result, auth_subject, delegated=delegating and bool(eff_raw)
        )
        return delegating, eff_raw

    def _check_licence_app(
        self,
        request: Request,
        license_result: dict[str, Any],
        auth_subject: object,
        *,
        delegated: bool,
    ) -> None:
        """Classify the licence's app; raise a 401 ``_RequestRefused`` only under enforce.

        George 2026-10-15: no app-to-app licences, only tenant×app ones, so a
        licence presented to worker_management must be issued for ``workers``
        (inference-router's ``_judge`` / ``EXPECTED_APP_KEY`` pattern). Outcomes
        on ``wm_licence_app_total``: ``tenant_app_ok`` (app is workers, no
        delegation), ``delegated`` (a delegating subject using an effective-tenant
        claim -- ALLOWED in both modes; its removal is a later step),
        ``wrong_app`` (another app), ``no_app`` (introspect returned no app).
        Delegation is judged first: fr's licence is a ``coding`` licence and must
        keep working through the transition.
        """
        app_key = _introspected_app_key(license_result)
        if delegated:
            outcome = LICENCE_APP_DELEGATED
        elif app_key is None:
            outcome = LICENCE_APP_NO_APP
        elif app_key == LICENCE_APP_EXPECTED:
            outcome = LICENCE_APP_OK
        else:
            outcome = LICENCE_APP_WRONG_APP
        app_label = _licence_app_label(app_key)

        def _emit(shape: str) -> None:
            LICENCE_APP_TOTAL.labels(app=app_label, path=shape, outcome=outcome).inc()

        _defer_route_labelled_metric(request, _emit)
        if outcome == LICENCE_APP_OK:
            return
        enforced = get_settings().wm_licence_app_mode == "enforce"
        refuse = enforced and outcome in (LICENCE_APP_WRONG_APP, LICENCE_APP_NO_APP)
        subject = str(auth_subject or "<missing>")[:64]
        key = (subject, app_label)
        if key not in _licence_app_log_seen:
            _licence_app_log_seen[key] = True
            logger.warning(
                "licence_app outcome=%s subject=%s app=%s expected=%s mode=%s action=%s",
                outcome,
                subject,
                app_label,
                LICENCE_APP_EXPECTED,
                "enforce" if enforced else "observe",
                "REFUSED (401)" if refuse else "SERVED",
            )
        if refuse:
            raise _RequestRefused(self._wrong_app_response(request, app_key), "wrong_app")

    def _wrong_app_response(self, request: Request, app_key: str | None) -> Response:
        """401 ``wrong_app``: the licence is valid but not issued for ``workers``."""
        from fastapi.responses import JSONResponse

        self._record_auth_refusal(AUTH_REFUSED_WRONG_APP, request, app_key or "no_app")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "wrong_app",
                    "message": (
                        f"Licence is not issued for app '{LICENCE_APP_EXPECTED}'; present "
                        f"a tenant '{LICENCE_APP_EXPECTED}' licence"
                    ),
                    "status": 401,
                    "expected_app": LICENCE_APP_EXPECTED,
                    "app": app_key,
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _bind_tenant_claims(
        self, request: Request, licence_tenant: str, auth_subject: object
    ) -> tuple[bool, str]:
        """Bind the request's tenant headers to the introspected licence.

        Returns ``(delegating, requested effective tenant)``: whether the
        INTROSPECTED subject is on AUTH_DELEGATING_SUBJECTS, and -- for a
        delegating subject only -- the stripped tenant it asked to act as
        (X-Effective-Tenant-Id, else the plain X-Tenant-Id claim; ``""`` = none).
        Raises ``_TenantRefused`` (403) instead when a NON-delegating caller names
        a foreign tenant (see ``_refuse_foreign_tenant_claims``).

        The istio ingress gateway validates ``Authorization: Bearer <licence>``,
        leaves it in place and injects a PLAIN ``X-Tenant-Id: <uuid>``. With the
        licence in Authorization that plain header is a tenant CLAIM: never
        ignored, never silently replaced by the licence tenant. A delegating
        subject's plain header stands in for X-Effective-Tenant-Id (which wins if
        both are sent) and takes the delegation path. Anyone else's claim -- and
        any X-Effective-Tenant-Id alongside it -- must be the licence's OWN
        tenant. A plain header equal to the licence tenant is no delegation
        claim (the gateway injects that on every request). With the licence in
        ``X-Tenant-Id: Bearer`` there is no plain claim.
        """
        delegating = bool(auth_subject) and str(auth_subject) in _delegating_subjects()
        from_authorization = _licence_from_authorization(request)
        header_eff = (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
        plain_tenant = _plain_tenant_header(request) if from_authorization else ""
        if not delegating:
            mismatch = self._refuse_foreign_tenant_claims(
                request,
                licence_tenant,
                from_authorization=from_authorization,
                claims=(
                    (MISMATCH_HEADER_PLAIN, plain_tenant),
                    (MISMATCH_HEADER_EFFECTIVE, header_eff),
                ),
            )
            if mismatch is not None:
                raise _TenantRefused(mismatch)
            return False, ""
        # A plain header equal to the licence tenant (in any spelling) needs no
        # special case: `_effective_tenant` resolves it to the licence tenant and
        # the delegation branch treats that as no re-scope.
        return True, header_eff or plain_tenant

    def _refuse_foreign_tenant_claims(
        self,
        request: Request,
        licence_tenant: str,
        *,
        from_authorization: bool,
        claims: tuple[tuple[str, str], ...],
    ) -> Response | None:
        """403 a NON-delegating caller whose tenant header names another tenant.

        Licence in ``Authorization`` (the gateway / n8n shape, new in this
        service): always refused -- before this change those requests were not
        authenticated at all, they were admitted on network trust as whatever
        the plain header said, which is the hole being closed.

        Licence in ``X-Tenant-Id: Bearer``: only ``X-Effective-Tenant-Id`` can
        mismatch. That has been SERVED as the licence's own tenant (with a
        WARNING) for months, and a live caller still sends it (subject
        01a0dd02-…, /workers/promo-*/message, 14x on 2026-10-07), so it is
        refused only once WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT is on.
        Both legs are counted on worker_management_licence_tenant_mismatch_total
        so the served leg can be watched down to 0 before enforcing it.
        """
        carrier = (
            LICENCE_CARRIER_AUTHORIZATION if from_authorization else LICENCE_CARRIER_X_TENANT_ID
        )
        refuse = from_authorization or get_settings().require_authenticated_tenant
        for header, claim in claims:
            if not claim or _same_tenant(claim, licence_tenant):
                continue
            outcome = MISMATCH_OUTCOME_REFUSED if refuse else MISMATCH_OUTCOME_SERVED
            try:
                LICENCE_TENANT_MISMATCH_TOTAL.labels(
                    carrier=carrier, header=header, outcome=outcome
                ).inc()
            except Exception:  # pragma: no cover - observe-only
                logger.debug("licence tenant mismatch accounting failed", exc_info=True)
            if not refuse:
                continue
            logger.warning(
                "licence tenant mismatch: a non-delegating caller named tenant %r in %s "
                "but its licence (carrier=%s) belongs to tenant=%s; REFUSED (403). "
                "path=%s peer=%s",
                claim[:64],
                header,
                carrier,
                licence_tenant,
                request.url.path[:200],
                request.client.host if request.client else "-",
            )
            return self._tenant_mismatch_response(
                f"Tenant '{claim[:64]}' does not match the tenant of the presented licence"
            )
        return None

    def _tenant_mismatch_response(self, detail: str) -> Response:
        """Return 403: the credential is valid, the tenant it was asked to act as is not its own."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": {"code": "TENANT_MISMATCH", "message": detail, "status": 403}},
        )

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public and doesn't require authentication."""
        clean_path = path.rstrip("/")
        if not clean_path:
            clean_path = "/"

        if clean_path in PUBLIC_ENDPOINTS:
            return True

        # Check path prefixes
        public_prefixes = ["/docs", "/redoc", "/static", "/health"]
        return any(clean_path.startswith(prefix) for prefix in public_prefixes)

    def _extract_license_token(self, request: Request) -> str | None:
        """The licence from ``X-Tenant-Id: Bearer <token>``, else ``Authorization: Bearer``.

        ``X-Tenant-Id`` keeps precedence. ``Authorization`` is consulted because
        the istio ingress gateway validates the licence there, strips client
        identity headers, injects a plain ``X-Tenant-Id: <uuid>`` and leaves
        ``Authorization`` in place -- reading only ``X-Tenant-Id`` admitted the
        gateway (and any in-mesh pod with a bare header) as that tenant on the
        trusted-network branch with no credential checked.

        History: on 2026-04-26 FR forwarded a Keycloak JWT in ``Authorization``
        and it was introspected as a licence (``token_not_found`` 401s), which is
        why this read ``X-Tenant-Id`` only. That outcome is now the intended
        one: a non-licence in ``Authorization`` is a hard 401, never a silent
        fall-through to network trust (George 2026-10-14). FR's workers client
        sends ``sendAuthorization: false`` and coding-ai's wm_client sends no
        ``Authorization``; caller identity JWTs ride ``X-Caller-Authorization``.
        """
        return _bearer_value(request.headers.get("X-Tenant-Id")) or _bearer_value(
            request.headers.get("Authorization")
        )

    @staticmethod
    def _record_auth_refusal(
        reason: str, request: Request | None, upstream_reason: str = ""
    ) -> None:
        """Count one 401 and log ONE structured line. Never the token. Never raises."""
        try:
            AUTH_REFUSED_TOTAL.labels(reason=reason).inc()
            peer = request.client.host if request is not None and request.client else ""
            logger.info(
                "auth_refused reason=%s upstream_reason=%s method=%s path=%s peer=%s "
                "caller=%s app_id=%s",
                reason,
                upstream_reason[:64] or "-",
                request.method if request is not None else "-",
                request.url.path[:200] if request is not None else "-",
                peer or "-",
                get_caller_client_id_from_context() or "-",
                (request.headers.get("x-app-id", "")[:64] if request is not None else "") or "-",
            )
        except Exception:  # pragma: no cover - observe-only
            logger.debug("auth refusal accounting failed", exc_info=True)

    def _unauthorized_response(
        self,
        detail: str,
        *,
        reason: str,
        request: Request | None = None,
        upstream_reason: str = "",
    ) -> Response:
        """Return 401 Unauthorized, counted on worker_management_auth_refused_total."""
        from fastapi.responses import JSONResponse

        self._record_auth_refusal(reason, request, upstream_reason)

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": detail,
                    "status": 401,
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _audit_unauthenticated_tenant_assertion(
        self,
        request: Request,
        x_tenant_id: str | None,
        tenant_id: str,
        client_host: str | None,
    ) -> Response | None:
        """Record — and, when gated on, refuse — a tenant nobody proved.

        ⛔ WHAT IS ACTUALLY WRONG HERE, so nobody "cleans this up".

        The branch this is called from admits a caller purely on network
        membership (``TRUSTED_CIDRS`` defaults to the whole pod network) and
        then lets that caller NAME ITS OWN TENANT in a bare ``X-Tenant-Id``
        header. The only check is syntactic — ``resolve_tenant_id`` accepts any
        canonical uuid — so there is no allowlist, no subject check and no
        licence introspection. The identity is then stamped with a NULL
        licence, and every downstream cost event and every tenant-scoped query
        for the rest of the request runs under it. Measured 2026-08-31:
        worker-management bills 9 distinct tenant_ids across 27 worker types on
        this path, two of which exist in no licences registry at all, and the
        admitted namespaces include ``n8n``, which runs user-authored
        workflows.

        The fix is NOT to honour a forwarded ``X-License-Id`` here. That would
        let any admitted peer assert a licence as well as a tenant — a strictly
        larger hole wearing the costume of a fix. kag refuses exactly that, in
        as many words: "nobody's proven assertion"
        (``kag/src/auth/tenant.py:85-88``). The only real fix is that each of
        these callers presents its own licence token in
        ``X-Tenant-Id: Bearer <token>`` (or stops being trusted), at which
        point the token branch above handles it and this code is never reached.

        Until then: observe now, refuse later. This method is the observation,
        and ``require_authenticated_tenant`` is the refusal — default OFF,
        because ten-plus live workloads are on this path today and a
        default-closed flag would 401 production on the next pod roll. The
        warning and the counter fire in BOTH states.

        Returns:
            A 401 ``Response`` when the gate is closed, else ``None`` (serve on,
            exactly as before).
        """
        # No header ⇒ no assertion. The absent-header case falls back to
        # DEFAULT_TENANT_ID, which is this deployment's own configuration
        # rather than the caller's claim; it is a different (and separately
        # arguable) problem and is deliberately NOT counted here, so the
        # counter means one thing only.
        if not x_tenant_id:
            return None

        enforced = get_settings().require_authenticated_tenant
        path = request.url.path

        # The ONE caller identifier on this request that was actually proven:
        # the `azp` of a verified X-Caller-Authorization JWT, set by
        # `verified_caller_identity` earlier in dispatch. Deliberately not the
        # raw x-caller-client-id header, which is attacker-supplied here (see
        # the comment on that call site). "<unverified>" is itself the useful
        # answer — a caller with no verified identity at all is the hardest
        # class to migrate off this branch and should be found first.
        caller = get_caller_client_id_from_context() or "<unverified>"
        metrics = get_metrics_client()
        method = request.method
        forwarded_for = (request.headers.get("x-forwarded-for") or "")[:128] or "<none>"
        user_agent = (request.headers.get("user-agent") or "")[:128]
        correlation_id = (request.headers.get("x-correlation-id") or "")[:128]

        # Both the counter label and the WARNING's dedup key are keyed on the
        # route SHAPE, and that is only knowable once the router has matched —
        # after this method has already decided whether to refuse. So the whole
        # observation is deferred to dispatch's ``finally``; the refusal below
        # is NOT deferred, because it must happen before the endpoint runs.
        def _emit(shape: str) -> None:
            # Counted unsampled, in both states, BEFORE the dedup guard below.
            metrics.record_unauthenticated_tenant_assertion(shape, enforced=enforced)

            # ⛔ THE 5th KEY COMPONENT IS NOT REDUNDANT WITH ``shape``.
            # ``shape`` is the matched route template, which is exactly what an
            # operator needs to go and fix one caller — EXCEPT when nothing
            # matched, where it is the constant ``<unmatched>`` and carries no
            # attribution at all. That is not a corner case: an ENFORCED refusal
            # returns before the router runs, so in enforced mode EVERY refusal
            # shares one key and only the first target ever gets named (measured:
            # three refusals to /alpha, /bravo, /charlie produced ONE warning).
            # So on the unmatched leg only, the raw path joins the key, and each
            # distinct refusal target is named once per window again.
            #
            # This is a LOG key, never a metric label — that is the whole reason
            # a caller-controlled string is admissible here and not in the
            # counter. It is bounded twice over: truncated to the same 200 chars
            # the line already logs, and capped by ``_assertion_log_seen``'s own
            # ``maxsize=2048`` / 300s TTL. On the MATCHED leg the component is
            # empty, so the hot path 10+ workloads use keeps exactly the tight
            # dedup it has today.
            unmatched_target = path[:200] if shape == _UNMATCHED_PATH else ""
            log_key = (tenant_id, client_host or "?", shape, caller, unmatched_target)
            if log_key in _assertion_log_seen:
                return
            _assertion_log_seen[log_key] = True
            logger.warning(
                "unauthenticated tenant assertion: a tokenless caller on the trusted network "
                "asserted tenant=%s (X-Tenant-Id=%r) and was %s. "
                "peer=%s forwarded_for=%s method=%s path=%s caller_client_id=%s "
                "user_agent=%r correlation_id=%r. "
                "No licence was presented and none was verified, so this tenant is the "
                "caller's own claim, not a proven one; the request is stamped with a NULL "
                "licence and bills that tenant. Before "
                "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT can be turned on, THIS peer "
                "must present its own licence token in `X-Tenant-Id: Bearer <token>` or be "
                "removed from TRUSTED_CIDRS. Do not 'fix' this by forwarding X-License-Id.",
                tenant_id,
                (x_tenant_id or "")[:64],
                "REFUSED (401)" if enforced else "SERVED",
                client_host or "<unknown>",
                forwarded_for,
                method,
                path[:200],
                caller,
                user_agent,
                correlation_id,
            )

        _defer_route_labelled_metric(request, _emit)

        if not enforced:
            return None
        return self._unauthorized_response(
            "A tenant asserted via X-Tenant-Id requires a licence token; send "
            "`X-Tenant-Id: Bearer <token>`.",
            reason=AUTH_REFUSED_TENANT_ASSERTION,
            request=request,
        )

    def _audit_identity_omission(self, request: Request, x_tenant_id: str | None) -> None:
        """Count the licence/subject omission on the tokenless auth branch.

        ⛔ OBSERVE-ONLY, AND RETURNS NOTHING ON PURPOSE. Its sibling
        ``_audit_unauthenticated_tenant_assertion`` returns an optional 401
        because it is the enforcement point for a flag; this one has no flag,
        no refusal, and no return value, so there is no path by which it can
        change whether a request is served. That asymmetry is deliberate —
        do not "make it consistent" by giving this one a gate. Arming a gate
        here would 401 exactly the mis-modelled service callers this counter
        exists to find, which is the opposite of finding them.

        Every failure is swallowed. A metrics backend that is down, misbehaving
        or unconfigured must never turn a request that succeeds today into one
        that fails.
        """
        try:
            # Bounded label: the verified, allowlisted JWT `azp` set by
            # `verified_caller_identity` earlier in dispatch — NOT the raw
            # x-caller-client-id header, which the caller controls here.
            caller = get_caller_client_id_from_context() or "<unverified>"
            reason = (
                _OMISSION_REASON_ASSERTED_TENANT if x_tenant_id else _OMISSION_REASON_DEFAULT_TENANT
            )
            if not x_tenant_id:
                # LOG ONLY (WM#324). The asserted case is already logged, with
                # its peer, by ``_audit_unauthenticated_tenant_assertion``; the
                # no-header case logged nothing, so its callers could not be
                # attributed. Queued BEFORE the metrics client is acquired, as
                # its own emitter, so a broken metrics backend cannot silence it.
                _defer_route_labelled_metric(request, self._omission_peer_logger(request, caller))
            # Acquired EAGERLY, inside this try: a metrics client that cannot
            # even be constructed must be swallowed here, on the request path
            # (tests/test_identity_omission_counter.py pins that directly).
            # Only the labelled emission itself is deferred, because the route
            # template does not exist until after ``call_next``.
            metrics = get_metrics_client()

            def _emit(shape: str) -> None:
                for field in _OMITTED_IDENTITY_FIELDS:
                    metrics.record_identity_omission(
                        field, reason=reason, caller=caller, path=shape
                    )

            _defer_route_labelled_metric(request, _emit)
        except Exception:  # pragma: no cover - defensive; observe-only
            # Deliberately not re-raised and deliberately not a warning per
            # request: this is instrumentation, and the request is unaffected.
            logger.debug("identity omission counter failed", exc_info=True)

    @staticmethod
    def _omission_peer_logger(request: Request, caller: str) -> Callable[[str], None]:
        """Deferred INFO line naming the peer of a no-header tokenless request.

        Field names match the asserted-tenant WARNING (``peer=``,
        ``forwarded_for=``, ``caller_client_id=``) so one Loki query reads both.
        Deduplicated per (peer, route shape, caller) like that line. The peer is
        a LOG field only, never a metric label (unbounded).
        """
        client_host = request.client.host if request.client else None
        forwarded_for = (request.headers.get("x-forwarded-for") or "")[:128] or "<none>"
        user_agent = (request.headers.get("user-agent") or "")[:128]
        correlation_id = (request.headers.get("x-correlation-id") or "")[:128]
        method = request.method
        path = request.url.path

        def _log(shape: str) -> None:
            unmatched_target = path[:200] if shape == _UNMATCHED_PATH else ""
            log_key = (client_host or "?", shape, caller, unmatched_target)
            if log_key in _omission_log_seen:
                return
            _omission_log_seen[log_key] = True
            logger.info(
                "identity omitted: a tokenless caller on the trusted network sent no "
                "X-Tenant-Id and was served as the default tenant. "
                "peer=%s forwarded_for=%s method=%s path=%s route=%s caller_client_id=%s "
                "user_agent=%r correlation_id=%r",
                client_host or "<unknown>",
                forwarded_for,
                method,
                path[:200],
                shape,
                caller,
                user_agent,
                correlation_id,
            )

        return _log

    @staticmethod
    def _stamp_nothing_proven(request: Request) -> None:
        """Record that this branch verified NEITHER the caller nor its tenant.

        The bypass branch and the tokenless trusted-network branch both land
        here. Kept as one helper rather than four assignments so a third such
        branch cannot stamp one flag and forget the other — a request carrying
        ``identity_proven=False`` with no ``tenant_proven`` at all is read as
        ``indeterminate`` by ``_modality_router._withhold_reason``, which is a
        deployment-shape alarm nobody should be able to trigger by accident.

        Both consumers default to False on a missing attribute anyway, so this
        changes no outcome; it makes the fact explicit and greppable.
        """
        request.state.identity_proven = False
        request.state.tenant_proven = False

    def _trusted_network_tenant(self, x_tenant_id: str | None) -> str:
        """Canonical uuid for the tokenless trusted-network branch, or refuse.

        The reasoning lives at the call site.
        """
        if not x_tenant_id:
            return _default_tenant_id()
        resolved = resolve_asserted_tenant_id(x_tenant_id)
        if resolved is None:
            raise _TenantRefused(
                self._bad_tenant_response(
                    "X-Tenant-Id must be a canonical tenant UUID. A tenant NAME is not "
                    "accepted on an unauthenticated request: the same name resolves to "
                    "different tenants in the portal realm and in the licences registry, "
                    "so it cannot identify one. Send the UUID, or present a licence token "
                    "in `X-Tenant-Id: Bearer <token>`.",
                    "X-Tenant-Id",
                    x_tenant_id,
                    reason=(
                        "a tenant NAME, which is ambiguous across registries"
                        if x_tenant_id.strip() in tenant_uuids()
                        else "neither a UUID nor a known tenant name"
                    ),
                )
            )
        return resolved

    @staticmethod
    async def _stamp_identity(
        request: Request,
        tenant_id: str,
        license_id: object,
        subject_id: object,
        license_result: dict[str, Any] | None,
        *,
        tenant_proven: bool,
        actor_trusted: bool,
        verify_end_user: bool = False,
    ) -> dict | None:
        """Bind the resolved identity onto ``request.state``.

        ``verify_end_user`` (non-delegating licence callers only): verify
        ``X-End-User-Token`` FIRST (``_licence_end_user``, raises on a present
        bad token) and, when valid, bind the user's ``sub`` as ``subject_id``
        and as the default actor. Returns the verified payload, else None.

        Extracted from ``dispatch`` only to keep it under the statement cap.

        ``tenant_proven`` is keyword-only and has NO DEFAULT on purpose. A
        default would have to be one of two wrong things: True, which makes a
        future call site that forgets it hand out another tenant's credential,
        or False, which silently downgrades the ordinary non-delegating token
        caller and would be debugged as "the key gate is broken". Requiring the
        caller to state it means the delegation question is answered where the
        delegation decision is made, and a new branch cannot omit it.
        """
        end_user = (
            await AuthenticationMiddleware._licence_end_user(request, tenant_id)
            if verify_end_user
            else None
        )
        end_user_sub = str(end_user["sub"]).strip() if end_user is not None else ""
        if end_user_sub:
            subject_id = end_user_sub
        request.state.user = {
            "user_id": tenant_id,
            "tenant_id": tenant_id,
            "sub": tenant_id,
        }
        request.state.tenant_id = tenant_id
        request.state.tenant_name = request.headers.get("X-Tenant-Name", "")
        request.state.license_id = license_id
        request.state.subject_id = subject_id
        request.state.license = license_result
        # Reached ONLY from the token branch, after `_introspect_license`
        # returned `active: true`. This is the one place in this middleware
        # where a credential was actually presented and actually verified, so
        # it is the one place that may stamp the identity as proven. Consumers
        # read it through `src.auth.get_identity_proven`, which defaults to
        # False when the attribute is absent — so every other branch, every
        # public endpoint and every deployment running without this middleware
        # is unproven by omission rather than by remembering to say so.
        request.state.identity_proven = True
        # ⛔ NOT `= True`. This branch proved WHO, and only sometimes WHICH. See
        # the call site: `tenant_proven=not delegated`.
        request.state.tenant_proven = tenant_proven
        # ``actor_trusted`` is keyword-only with NO default for the same reason
        # as ``tenant_proven``: the caller answers "may this licence assert
        # someone else's X-Actor-Id?" where the delegation decision is made. A
        # non-delegating licence may still name its OWN service-account sub
        # (introspected ``subject_keycloak_sub``) or a member of its tenant (D4).
        await AuthenticationMiddleware._relay_actor_id(
            request,
            trusted=actor_trusted,
            caller_tenant=tenant_id,
            own_sub=str((license_result or {}).get("subject_keycloak_sub") or ""),
            auth_branch=_project_id_licence_caller(license_result),
            end_user_sub=end_user_sub,
        )
        return end_user

    def _canonical_introspected_tenant(self, tenant_id: object) -> str:
        """Normalise the tenant licences introspected, or refuse the request.

        Licences is trusted, but it is still an upstream returning a string into
        a ``uuid`` column. A malformed value would fail every statement of the
        request with 22P02 and read as a database fault rather than as bad
        introspection data; a non-canonical spelling would look like a second
        tenant in logs and metrics. Normalising costs nothing and keeps the
        failure attributable to the thing that caused it.

        ⛔ NAMES ARE REFUSED HERE TOO, AND "LICENCES IS TRUSTED" IS THE REASON
        TO REFUSE, NOT TO ALLOW. An earlier version of this fix argued that
        licences "picked which registry it meant", so a name from introspection
        could be honoured. That is backwards. Resolution here goes through
        ``TENANT_UUIDS``, which is the PORTAL REALM's map — so an introspected
        ``"n8n-dev"`` would resolve to ``aaaaaaaa-…0001``
        (``n8n-dev-portal-realm``, the seed tenant) while licences' OWN registry
        has ``n8n-dev`` as ``bbbbbbbb-0000-4000-8000-0000000000a5``. Honouring
        the name would misattribute a licence to a different tenant than the
        authority that issued it says it belongs to — the same defect as the
        tokenless header, on the AUTHENTICATED path, where it is least visible.

        ⭐ AND IT COSTS NOTHING, because a name cannot legitimately arrive.
        Measured on worker4 2026-09-16: ``licenses.tenants.id`` and
        ``license.tenant_id`` are both ``uuid NOT NULL`` with an FK between
        them, and no tenant NAME is uuid-shaped (0 rows match
        ``name ~ '^[0-9a-f]{8}-'``). So every well-formed introspection response
        already carries a uuid and is unaffected; only a broken upstream reaches
        the refusal, and a loud 401 is the right answer to that.
        """
        resolved = resolve_asserted_tenant_id(str(tenant_id or ""))
        if resolved is None:
            if tenant_id:
                logger.error("licence introspection returned an unusable tenant_id: %r", tenant_id)
            raise _TenantRefused(
                self._unauthorized_response(
                    "License token missing or unusable tenant context",
                    reason=AUTH_REFUSED_UNUSABLE_TENANT,
                )
            )
        return resolved

    def _effective_tenant(self, eff_raw: str) -> str:
        """Resolve the X-Effective-Tenant-Id delegation header (may be empty).

        ⛔ A NAME IS STILL ACCEPTED HERE, AND THAT IS A DELIBERATE DEFERRAL —
        NOT AN OVERSIGHT, AND NOT A JUDGEMENT THAT IT IS SAFE.

        This header has the SAME defect as the tokenless one. It is
        caller-supplied and REPLACES the tenant the licence established, so a
        name misattributes identically: measured on this branch,
        ``_effective_tenant("n8n-dev")`` returns ``aaaaaaaa-…0001`` — the seed
        tenant — and the request is served 200. That a licence was presented does
        not make THIS tenant proven; the call site already says so in the only
        way that matters, ``tenant_proven=not delegated``. The token proved WHO;
        this header re-scoped WHICH, on the caller's say-so.

        ⛔ SO WHY IS IT STILL HERE? BECAUSE IT CANNOT BE SHOWN SAFE WITH TODAY'S
        LOGGING, AND THE TRAFFIC IS TOO LARGE TO FIND OUT WITH.

        The tokenless door could be closed because its log line records the RAW
        header value, so Loki can separate name-senders from uuid-senders: 1,501
        assertions over 30d, exactly 1 name-valued, and that one was this issue's
        own probe. This door had no such line. Until the commit that adds it,
        ``logger.info("effective-tenant delegation: …")`` recorded only the
        ALREADY-RESOLVED tenant, so a caller sending ``n8n-dev`` and one sending
        ``aaaaaaaa-…0001`` produced BYTE-IDENTICAL log lines.

        Measured over 30d, grouping those lines by resolved tenant:

            72,351 delegation requests
            58,710 of them (81.1%) resolved to a uuid a NAME could also have
            produced — …0001 (50,304, `n8n-dev`), …0003 (4,729, `george-dev`),
            …0099 (3,614, `fr-test`), …0101 (63, `ahtapol`)

        Whether ANY of those 58,710 carried a name is UNOBSERVABLE with the
        logging that existed when they were served. Refusing names here would
        turn an unknown share of 81% of delegated traffic into 400s.

        ⚠️ AND THE CALLER ENUMERATION DOES NOT COVER IT. An earlier version of
        this docstring argued the door was safe because the three enumerable
        senders — hirag_service, discovery-engine, kag_orchestrator — each
        resolve to a uuid at their own boundary and so "none can emit a name".
        That is still true of those three, and it is not sufficient: the logs
        carry FIVE distinct delegating subjects, and the second-largest,
        ``8382f0b3-c082-4f97-8d47-120ce5a18e05`` (9,270 requests), is NOT in the
        live ``AUTH_DELEGATING_SUBJECTS`` at all — the config moved inside the
        window. A repo-side enumeration cannot speak for a sender it cannot name.

        THE PATH OUT, which is mechanical rather than a matter of judgement: the
        call site now logs ``eff_raw`` beside the resolved value. Once a full
        retention window has passed, re-run the group-by on the RAW value; if the
        name count is 0, this becomes ``resolve_asserted_tenant_id`` exactly like
        the other two doors. Tracked as its own issue — see the PR for #264.
        """
        if not eff_raw:
            return ""
        resolved = resolve_tenant_id(eff_raw)
        if resolved is None:
            raise _TenantRefused(
                self._bad_tenant_response(
                    f"{EFFECTIVE_TENANT_HEADER} is neither a UUID nor a known tenant name.",
                    EFFECTIVE_TENANT_HEADER,
                    eff_raw,
                )
            )
        return resolved

    def _bad_tenant_response(
        self,
        detail: str,
        header: str,
        value: str,
        *,
        reason: str = "neither a UUID nor a known tenant name",
    ) -> Response:
        """Return 400 Bad Request — the caller named a tenant that cannot exist.

        ⛔ THE LOG LINE IS NOT DECORATION. This refusal is a behaviour CHANGE:
        before 2026-08-25 these values were adopted verbatim, so any caller still
        sending one starts getting 400s the moment this deploys. Without naming
        the header and the value, an operator sees a rise in 400s with nothing to
        act on, and the first real diagnosis happens in a support ticket. WARNING
        rather than INFO precisely because it means "somebody upstream is broken
        and does not know yet".

        The value is truncated: it is attacker-influenced input being written to
        a log, and an unbounded field is how a log store gets flooded.

        Deliberately 400 and not 401: nothing was wrong with the caller's
        *authority* (it is on a trusted network), and nothing about retrying will
        help. The request is malformed. 401 would send a service into a token-
        refresh loop over a typo, and 500 would blame the database for a value
        the caller chose.
        """
        from fastapi.responses import JSONResponse

        logger.warning(
            "tenant rejected: header=%s value=%r is %s "
            "(known names, NOT accepted on the tokenless X-Tenant-Id header: %s). The caller "
            "is misconfigured; it was adopted verbatim before 2026-08-25.",
            header,
            value[:64],
            reason,
            ",".join(sorted(tenant_uuids())),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "INVALID_TENANT",
                    "message": detail,
                    "status": 400,
                }
            },
        )

    def _caller_token_invalid_response(self, failure_class: str) -> Response:
        """Return 401 for a PRESENT but invalid ``X-Caller-Authorization`` token.

        Distinct code from AUTHENTICATION_REQUIRED so a caller can tell its
        caller token (not its licence) was refused. Names the failure class
        only; the token is never echoed.
        """
        from fastapi.responses import JSONResponse

        self._record_auth_refusal(AUTH_REFUSED_CALLER_TOKEN_INVALID, None, failure_class)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "CALLER_TOKEN_INVALID",
                    "message": (
                        f"X-Caller-Authorization token rejected: {failure_class}. "
                        "Omit the header or send a valid, unexpired secure-apps SA token."
                    ),
                    "failure_class": failure_class,
                    "status": 401,
                }
            },
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )

    def _service_unavailable_response(self, detail: str) -> Response:
        """Return 503 Service Unavailable — a RETRYABLE auth outcome.

        Used when license introspection could not reach a verdict (licenses
        slow/unreachable, cold stale cache). Distinct from 401: the token was
        never rejected, so the caller should retry rather than treat it as
        invalid. 503 lands in w2v_common's client ``_RETRYABLE_STATUS_CODES``,
        so downstream callers retry with backoff for free. ``Retry-After`` hints
        a short pause. Never grants access.
        """
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "INTROSPECTION_UNAVAILABLE",
                    "message": detail,
                    "status": 503,
                }
            },
            headers={"Retry-After": "1"},
        )

    def _internal_error_response(self, detail: str) -> Response:
        """Return 500 Internal Server Error response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "AUTHENTICATION_SERVICE_ERROR",
                    "message": detail,
                    "status": 500,
                }
            },
        )


# Simple helper for route handlers
def get_current_user(request: Request) -> dict[str, Any]:
    """Get current authenticated user from request state."""
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return cast("dict[str, Any]", request.state.user)
