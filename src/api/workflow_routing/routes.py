"""FastAPI router for ``/api/workflow-routing/*``.

Authentication: license token in ``X-Tenant-Id: Bearer <token>``. The four
POSTs additionally require ``service-internal`` from a verified
``X-Caller-Authorization`` SA JWT (worker_management#230; see below). The global
``AuthenticationMiddleware`` (see ``src/api/auth_middleware.py``) is
responsible for actually validating the token; this module just declares
the standard ``Depends(get_tenant_id)`` so the dependency wires through and
the OpenAPI surface advertises tenant scoping.

NOTE: license tokens travel ONLY in ``X-Tenant-Id``, never ``Authorization``
— see the ``upstream-license-token-pattern`` memory and the comment in
``auth_middleware._extract_license_token`` for why.

Since migration 036 (G-15) a pointer is keyed by ``asset_key`` =
``'<map>:<name>'`` (the dispatch key) and every workflow id is a 16-char n8n id.

Endpoints:
- GET  /api/workflow-routing/active?asset_key=   (legacy: ?list= → 'cell:<list>')
- GET  /api/workflow-routing/observed
- POST /api/workflow-routing/seed-active      ┐ G-15 release path: service role
- POST /api/workflow-routing/release          │ AND a release tenant (…0001/…0002),
- POST /api/workflow-routing/promote-canary   │ see ``_require_release_tenant``
- POST /api/workflow-routing/discard-canary   ┘
- POST /api/workflow-routing/promote          ┐ legacy (Lane D), service role only
- POST /api/workflow-routing/rollback         │
- POST /api/workflow-routing/extend           │
- POST /api/workflow-routing/clear-prior      ┘
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from prometheus_client import Counter
from pydantic import BaseModel, Field, field_validator

from src.api import identity_context
from src.api.tenant_constants import PLATFORM_TENANT_IDS, is_platform_tenant
from src.api.workflow_routing import repository
from src.auth import get_tenant_id
from src.auth.caller_identity import require_any_role
from src.settings import get_settings
from src.telemetry import get_metrics_client
from src.telemetry.agent_metrics import (
    WORKFLOW_ROUTING_AUTH_PATH_ASSERTED,
    WORKFLOW_ROUTING_AUTH_PATH_DEFAULT,
    WORKFLOW_ROUTING_AUTH_PATH_DELEGATED,
    WORKFLOW_ROUTING_AUTH_PATH_LICENSE,
    WORKFLOW_ROUTING_AUTH_PATH_NONE,
    WORKFLOW_ROUTING_TENANT_UNPROVEN,
)

# ⛔ ``Request`` MUST be imported at RUNTIME, not under ``TYPE_CHECKING``.
# This module has ``from __future__ import annotations``, so every annotation is
# a string that FastAPI resolves with ``get_type_hints`` when it builds the
# dependency. Behind a TYPE_CHECKING guard the name does not exist at runtime,
# FastAPI cannot resolve it to the Request type, and it falls back to treating
# the parameter as a REQUEST FIELD — every route on this router then answers 422
# instead of running. Caught by the existing suite (19 red), not by review.

logger = logging.getLogger(__name__)


# ⛔⛔ AUTHORIZATION ON THIS MODULE (worker_management#230). ⛔⛔
#
# Reads (``/active``, ``/observed``) keep plain licence auth. The four WRITES
# (``/promote``, ``/rollback``, ``/extend``, ``/clear-prior``) additionally
# require ``service-internal`` via ``require_role`` — roles read ONLY from a
# signature-verified, ``CALLER_ALLOWED_CLIENTS``-allowlisted
# ``X-Caller-Authorization`` SA JWT (``src/auth/keycloak_roles.py``), never
# from a request header. Same mechanism as prompt DISCARD/activate
# (``src/api/prompts/routes.py``). ``prompt-asset-bootstrap`` is deliberately
# NOT accepted: repointing dispatch is not creating an initial prompt.
# The four G-15 release writes add a TENANT gate on top — see THE RELEASE GATE
# below.
#
# WHY: ``Depends(get_tenant_id)`` is authentication only — any active licence
# resolves to a tenant — while ``workflow_routing_pointer`` has NO tenant
# column, so one POST repointed dispatch for EVERY tenant; and it is
# internet-reachable (gitops ``istio/tunnel-mtls/virtualservice.yaml`` maps
# ``/workers/`` -> ``/``; the edge gates by host + licence, never by path).
#
# WHY THE "GATE LAST" ORDER IN #230 NO LONGER APPLIES. #230 warned that gating
# would 403 live automation (``canary-workflow-rollback-watch`` -> /rollback,
# /clear-prior; ``improve::approval-callback`` -> /extend), whose
# ``@nonagentic/n8n-nodes-workers`` credential sends no SA JWT. Re-measured
# 2026-09-29 before gating:
#   * ``workflow_routing_pointer`` holds 0 rows in production (worker4), and
#     nothing in n8n_custom_nodes@origin/main calls ``/promote`` — so no row
#     can ever exist for the watcher or the callback to act on;
#   * ``worker_management_workflow_routing_calls_total`` over 21 days: ONLY
#     ``GET /observed`` (auth_path=license); zero series for any POST;
#   * ``aa_rollback_watch_observed_pointers`` max over 30 days = 0.
# Both dependants act only on a row ``/promote`` created, so they are dormant,
# not live. ⚠️ Consequence: before anyone ARMS the canary loop (promotes via a
# service-internal SA), ``n8n-nodes-workers`` must send
# ``X-Caller-Authorization`` too — otherwise promote works and the watcher's
# rollback 403s (loudly: its "Check Rollback" node throws on >= 400).
ROUTING_WRITER_ROLE = "service-internal"
# G-15: the NARROW role for the dedicated n8n client ``n8n-workflow-routing``
# (one service = one identity). It grants routing writes only, so n8n never
# needs the broad ``service-internal`` role to release a workflow canary.
ROUTING_WRITER_SCOPED_ROLE = "workflow-routing-writer"
_require_routing_writer = require_any_role(ROUTING_WRITER_ROLE, ROUTING_WRITER_SCOPED_ROLE)


# ⛔⛔ POINTER CREATION IS SEPARATELY GATED (George, 2026-09-30). ⛔⛔
#
# ``canary-workflow-rollback-watch`` — the 5-min n8n timer that watched
# promoted rows and fired /rollback on a threshold breach — is RETIRED.
# ``workflow_routing_pointer`` holds 0 rows in production, so no row exists
# for it to have been watching anyway. Until an event-driven rollback
# verdict routine replaces that timer, no NEW pointer may be created:
# promoting without anything watching the observation window is a canary
# with no one checking the cage.
#
# Gated: every route that CREATES pointer state — legacy ``/promote``
# (``repository.promote``'s ``INSERT … ON CONFLICT DO UPDATE``), G-15
# ``/seed-active`` (inserts the row) and G-15 ``/release`` (arms a canary on
# a row, which is new pointer state a dispatcher can route to). Each lists
# this dependency AFTER its auth checks, so an unauthorised caller never
# learns the flag state.
#
# NOT gated: ``/rollback``, ``/extend``, ``/clear-prior``, ``/promote-canary``
# and ``/discard-canary`` only ever mutate a row that already exists. With 0
# rows live each already 404s/409s on its own existing-row check; the live
# ``improve::approval-callback`` workflow calls ``/extend`` and rollback, and
# ``/discard-canary`` must stay available to DISARM — gating those would
# change behaviour for no reason tied to this flag's purpose.
def _require_routing_pointers_enabled() -> None:
    if not get_settings().workflow_routing_pointers_enabled:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "routing_pointers_disabled",
                "message": (
                    "creating workflow routing pointers is disabled until an "
                    "event-driven rollback verdict exists (rollback-watch timer "
                    "retired); set WORKFLOW_ROUTING_POINTERS_ENABLED=true only "
                    "together with that verdict"
                ),
            },
        )


async def _record_routing_call(request: Request) -> None:
    """Count this workflow-routing request. OBSERVE-ONLY — never refuses.

    This is the instrument that makes worker_management#230 decidable: it
    answers "which tenants actually call these endpoints", which the access log
    cannot, because routing requests appear there only as uvicorn lines carrying
    no identity, and because ``kubectl logs`` serves one rotating file.

    ⛔ Returns ``None`` on every path and swallows every exception. It is
    declared alongside ``Depends(get_tenant_id)`` on a router whose endpoints
    include the dispatch hot path; a telemetry failure must never be able to
    turn a served request into a 500. It also must never be mistaken for a gate:
    there is no branch here that can reject anything.
    """
    try:
        state = request.state
        # Provenance decides BOTH labels. Only an introspected licence proves a
        # tenant, and only a proven tenant may enter the label — an asserted one
        # is caller-chosen and would make this an unbounded time series.
        #
        # ⛔ THE DELEGATION TEST COMES FIRST, AND IT READS THE FLAGS — NOT
        # ``license_id``. Provenance is NOT recoverable from ``license_id`` on
        # this router: ``AuthenticationMiddleware`` deliberately NULLS it on the
        # delegation branch (see its "DO NOT fall back to the introspected
        # licence here" comment — keeping it stamps customer spend on the
        # service account's entitlement). So a delegating caller, whose identity
        # genuinely WAS proven, arrives here looking byte-identical to a caller
        # that presented nothing at all, and a caller that forwards
        # ``X-License-Id`` arrives looking fully licensed while carrying an
        # ADOPTED tenant that ``_effective_tenant`` checked for spelling only.
        # Ordered and reasoned exactly like
        # ``_modality_router._withhold_reason``'s ``adopted_tenant`` rung, which
        # this mirrors: the delegated population must not be merged into the
        # tokenless one, and the adopted tenant must not reach the label.
        #
        # ⚠️ READ THE DEFAULTS PRECISELY — an earlier draft of this comment was
        # measured FALSE. `tenant_proven` defaults to False, and `is not True`
        # is satisfied by that default, so a request stamping `identity_proven`
        # with `tenant_proven` ABSENT is labelled `delegated` — it does NOT fall
        # through. That is the correct outcome (identity proven, tenant not),
        # but it is a match, not a fall-through, and a half-stamped rollback
        # lands here rather than below.
        # A request with NO middleware at all still falls through: both reads
        # default False, so the first conjunct fails.
        #
        # The raw-uuid branch below is provenance-POSITIVE for the same reason:
        # it requires `tenant_proven is True` rather than merely a truthy
        # `license_id`. Testing `license_id` alone fails OPEN — a shape that
        # stamps `license_id` without the flags would put a caller-supplied
        # uuid into the label. Unreachable today (`_stamp_identity` writes both
        # flags as bool literals beside it), but the default direction must be
        # the safe one, and only an INTROSPECTED tenant may enter the label.
        if (
            getattr(state, "identity_proven", False) is True
            and getattr(state, "tenant_proven", False) is not True
        ):
            tenant = WORKFLOW_ROUTING_TENANT_UNPROVEN
            auth_path = WORKFLOW_ROUTING_AUTH_PATH_DELEGATED
        elif getattr(state, "license_id", None) and getattr(state, "tenant_proven", False) is True:
            tenant = str(getattr(state, "tenant_id", "") or WORKFLOW_ROUTING_TENANT_UNPROVEN)
            auth_path = WORKFLOW_ROUTING_AUTH_PATH_LICENSE
        elif getattr(state, "tenant_id", None):
            tenant = WORKFLOW_ROUTING_TENANT_UNPROVEN
            auth_path = (
                WORKFLOW_ROUTING_AUTH_PATH_ASSERTED
                if request.headers.get("X-Tenant-Id")
                else WORKFLOW_ROUTING_AUTH_PATH_DEFAULT
            )
        else:
            tenant = WORKFLOW_ROUTING_TENANT_UNPROVEN
            auth_path = WORKFLOW_ROUTING_AUTH_PATH_NONE

        # The MATCHED route template, never the raw path: bounded by the ten
        # routes this router declares.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "<unmatched>"

        get_metrics_client().record_workflow_routing_call(
            endpoint=endpoint,
            method=request.method,
            tenant=tenant,
            auth_path=auth_path,
        )
    except Exception:  # pragma: no cover - defensive; observe-only
        logger.debug("workflow-routing call counter failed", exc_info=True)


router = APIRouter(
    prefix="/api/workflow-routing",
    tags=["Workflow Routing"],
    dependencies=[Depends(get_tenant_id), Depends(_record_routing_call)],
)


# ⛔⛔ THE RELEASE GATE (G-15 DESIGN D4; phase-10 REVIEW Q3). ⛔⛔
#
# /seed-active, /release, /promote-canary and /discard-canary repoint (or arm a
# canary on) what dispatch runs for an asset for EVERY tenant — the table has
# no tenant column. Each requires BOTH, as two dependencies on its own route:
#   1. ``_require_routing_writer`` — the verified ``service-internal`` SA role
#      (the #230 gate above, reused unchanged), and
#   2. ``_require_release_tenant`` — the caller's tenant is the system tenant
#      ``…0001`` or the workflow-improvement stream tenant ``…0002``. NARROWER
#      than ``PLATFORM_TENANT_IDS``: "written only by improvement cards", and WM
#      cannot verify a card, so the tenant is the narrowest proof it has.
# Both refuse with 403 BEFORE any database access. phase-10 requires a test and
# a mutation control PER GATE, so the pair is written out on each route rather
# than shared through one list or router — removing it from one route must red
# that route's tests and no other's (tests/test_workflow_routing_release.py).
#
# ⚠️ On the SA path the tenant is ASSERTED (trusted-network ``X-Tenant-Id``),
# not introspected: the role is the proof of identity, the tenant only narrows
# which verified callers may release. A licence alone never carries the role.
RELEASE_TENANT_IDS: frozenset[str] = frozenset(
    {
        "0195f48b-d000-7000-8000-000000000001",  # system
        "0195f48b-d000-7000-8000-000000000002",  # workflow-improvement stream
    }
)
if not RELEASE_TENANT_IDS <= PLATFORM_TENANT_IDS:  # narrower, never wider
    raise RuntimeError("RELEASE_TENANT_IDS must be a subset of PLATFORM_TENANT_IDS")
_PLATFORM_ROUTING_FORBIDDEN = {"error": "platform_routing_forbidden"}


def _require_release_tenant(tenant_id: Annotated[str, Depends(get_tenant_id)]) -> None:
    """403 unless the caller's tenant may release workflows.

    Fixed detail, deliberately: it names neither the caller's tenant nor the
    accepted ones (same rule as ``platform_prompt_forbidden``).
    """
    if (tenant_id or "").strip().lower() in RELEASE_TENANT_IDS:
        return
    logger.warning("Refused workflow release call from tenant %s", tenant_id)
    raise HTTPException(status_code=403, detail=_PLATFORM_ROUTING_FORBIDDEN)


# The RELEASABLE SET (movies-ab ruling, G-15 DESIGN): an asset may be seeded,
# released or promoted only as ``activity:<name>`` with ``<name>`` in
# WM_WF_RELEASE_ALLOWLIST (JSON list of workflow names, the same frozen list as
# ncn's AA_WF_PROMOTE_ALLOWLIST). ⛔ FAIL CLOSED: unset, empty, unparsable or a
# non-list ⇒ nothing is releasable. /discard-canary is deliberately NOT checked:
# disarming is always safe, and an asset dropped from the set must still be
# disarmable.
RELEASABLE_PREFIX = "activity:"
_ASSET_NOT_RELEASABLE = {"error": "asset_not_releasable"}


def _releasable_names() -> frozenset[str]:
    raw = (get_settings().wm_wf_release_allowlist or "").strip()
    if not raw:
        return frozenset()
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("WM_WF_RELEASE_ALLOWLIST is not JSON; nothing is releasable")
        return frozenset()
    if not isinstance(parsed, list) or not all(isinstance(n, str) and n for n in parsed):
        logger.warning("WM_WF_RELEASE_ALLOWLIST is not a list of names; nothing is releasable")
        return frozenset()
    return frozenset(n for n in parsed if isinstance(n, str))


def _require_releasable(asset_key: str) -> None:
    """403 ``asset_not_releasable`` unless ``asset_key`` is in the releasable set."""
    name = asset_key[len(RELEASABLE_PREFIX) :] if asset_key.startswith(RELEASABLE_PREFIX) else None
    if name is None or name not in _releasable_names():
        raise HTTPException(status_code=403, detail=_ASSET_NOT_RELEASABLE)


def _captured_trace_id() -> str | None:
    """The inbound ``X-AA-Trace-Id`` captured by ``AuthenticationMiddleware``.

    Read through the module at call time so a test can patch it. Stored
    byte-for-byte (AA-TRACE §2: relayed, never rewritten); blank means absent.
    """
    value = identity_context.get_aa_trace_from_context().get("X-AA-Trace-Id", "")
    return value if value.strip() else None


def _require_trace_id() -> str:
    """400 ``trace_id_required`` when the release call carries no trace.

    phase-10 decision 4 ("trace_id on every release"); Q6 for /seed-active.
    """
    trace_id = _captured_trace_id()
    if trace_id is None:
        raise HTTPException(status_code=400, detail={"error": "trace_id_required"})
    return trace_id


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

# An n8n workflow id: 16 alphanumerics (measured 10-01: 364/364 ids on worker4).
# Mirrors migration 036's ``wrp_workflow_id_shape`` CHECK, so a bad id is a 422
# here rather than a CheckViolation 500 at the database.
WORKFLOW_ID_PATTERN = r"^[A-Za-z0-9]{16}$"
# '<map>:<name>' (G-15 D1): the env map the dispatch site reads, then the key.
# No control characters (a NUL reaches Postgres and 500s instead of a 4xx).
ASSET_KEY_PATTERN = r"^(activity|pattern|cell):[^\s\x00-\x1f\x7f][^\x00-\x1f\x7f]*$"
CARD_REF_PATTERN = r"^[^\x00-\x1f\x7f]+$"


def _validate_workflow_id(value: str, field: str) -> str:
    if not re.fullmatch(WORKFLOW_ID_PATTERN, value):
        raise ValueError(f"{field} must be a 16-character n8n workflow id: {value!r}")
    return value


def _validate_asset_key(value: str) -> str:
    if not re.fullmatch(ASSET_KEY_PATTERN, value):
        raise ValueError(f"asset_key must be '<activity|pattern|cell>:<name>': {value!r}")
    return value


class CanaryOut(BaseModel):
    """The armed canary of a pointer (all four set, or the whole object null)."""

    workflow_id: str
    pct: int
    canary_id: str
    released_by_card_id: str


class WorkflowRoutingPointer(BaseModel):
    """Public shape of a ``workflow_routing_pointer`` row (036).

    ``canary`` is nested and null when none is armed; WM makes no routing
    decision — the resolver buckets the card (G-15 D4).
    """

    asset_key: str
    card_type: str | None = None
    list: str | None = None
    active_workflow_id: str
    prior_workflow_id: str | None = None
    canary: CanaryOut | None = None
    promoted_at: str | None = None
    observation_window_until: str | None = None
    rollback_thresholds: dict[str, Any] = Field(default_factory=dict)
    released_by_tenant_id: str | None = None
    trace_id: str | None = None
    promoted_trace_id: str | None = None
    released_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> WorkflowRoutingPointer:
        """Build the public shape from a repository row (flat canary columns)."""
        canary = None
        if row.get("canary_workflow_id") is not None:
            canary = CanaryOut(
                workflow_id=row["canary_workflow_id"],
                pct=row["canary_pct"],
                canary_id=row["canary_id"],
                released_by_card_id=row["released_by_card_id"],
            )
        fields = {k: v for k, v in row.items() if k in cls.model_fields and k != "canary"}
        return cls(**fields, canary=canary)


class _AssetKeyBody(BaseModel):
    asset_key: str = Field(..., min_length=3, max_length=300)

    @field_validator("asset_key")
    @classmethod
    def _asset_key_shape(cls, v: str) -> str:
        return _validate_asset_key(v)


class SeedActiveRequest(_AssetKeyBody):
    """Body for POST /seed-active: the asset's CURRENT env-map workflow id."""

    active_workflow_id: str

    @field_validator("active_workflow_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        return _validate_workflow_id(v, "active_workflow_id")


class ReleaseRequest(_AssetKeyBody):
    """Body for POST /release — the per-asset ARMING row (decision 2)."""

    canary_workflow_id: str
    canary_pct: int = Field(..., ge=1, le=50)  # a canary is never the majority arm
    canary_id: str = Field(..., min_length=1, max_length=200, pattern=CARD_REF_PATTERN)
    released_by_card_id: str = Field(..., min_length=1, max_length=200, pattern=CARD_REF_PATTERN)

    @field_validator("canary_workflow_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        return _validate_workflow_id(v, "canary_workflow_id")


class CanaryActionRequest(_AssetKeyBody):
    """Body for POST /promote-canary and /discard-canary."""

    canary_id: str = Field(..., min_length=1, max_length=200, pattern=CARD_REF_PATTERN)
    card_id: str = Field(..., min_length=1, max_length=200, pattern=CARD_REF_PATTERN)


class _LegacyKeyBody(BaseModel):
    """Legacy (card_type, list) bodies, keyed by ``asset_key`` since 036.

    ``asset_key`` wins when present; otherwise a bare ``list`` maps to
    ``cell:<list>`` (the only (card_type, list)-shaped dispatch, G-15 D1). A
    ``card_type`` without ``asset_key`` names no asset → 400.
    """

    asset_key: str | None = Field(None, min_length=3, max_length=300)
    card_type: str | None = Field(None, min_length=1, max_length=200)
    list: str | None = Field(None, min_length=1, max_length=200)

    def resolved_asset_key(self) -> str:
        """The asset this legacy body names (400 when it names none)."""
        return _resolve_asset_key(self.asset_key, self.card_type, self.list)


class PromoteRequest(_LegacyKeyBody):
    """Body for POST /promote (legacy)."""

    new_workflow_id: str = Field(..., description="16-character n8n workflow id")
    observation_window_minutes: int = Field(
        ...,
        ge=0,
        le=10_080,  # 1 week ceiling — guards against fat-fingered Infinity windows
        description=(
            "Minutes until prior_workflow_id is auto-cleared. 0 = Lane F opt-out "
            "(promote and lock in immediately, rollback monitor does not arm)."
        ),
    )
    rollback_thresholds: dict[str, Any] = Field(default_factory=dict)

    @field_validator("new_workflow_id")
    @classmethod
    def _validate_workflow_id(cls, v: str) -> str:
        # Validated so a typo fails at 422 rather than at the database's
        # wrp_workflow_id_shape CHECK with an opaque 500.
        return _validate_workflow_id(v, "new_workflow_id")


class RollbackRequest(_LegacyKeyBody):
    """Body for POST /rollback.

    ``reason`` is forwarded to the fr-backend webhook; the database table
    does not persist it (events live in metering_service rollups, not WM).
    """

    reason: str = Field("", max_length=1000)


class ExtendRequest(_LegacyKeyBody):
    """Body for POST /extend.

    ``additional_minutes`` is bounded [1, 10080] (1 week ceiling, mirroring
    ``PromoteRequest.observation_window_minutes``); 0 is rejected at 422.
    """

    additional_minutes: int = Field(
        ...,
        ge=1,
        le=10_080,  # 1 week ceiling — same guard as promote's window
        description="Minutes to add on top of the later of (now, current window end).",
    )


class ClearPriorRequest(_LegacyKeyBody):
    """Body for POST /clear-prior."""


def _refuse_release_asset(asset_key: str) -> str:
    """Legacy writes touch ONLY ``cell:`` pointers (the one shape they ever served).

    ``activity:`` (and any future ``pattern:``) pointers are written ONLY through
    the gated release path (/seed-active, /release, /promote-canary,
    /discard-canary: release tenant + releasable set, phase-10 Q3). The legacy
    routes keep only the service role, so an allow-list of ``cell:`` -- not a
    deny of ``activity:`` -- keeps a newly wired dispatch kind from silently
    becoming writable around those gates (G-15 audit finding).
    """
    if not asset_key.startswith("cell:"):
        raise HTTPException(status_code=403, detail={"error": "use_release_path"})
    return asset_key


def _resolve_asset_key(asset_key: str | None, card_type: str | None, list_name: str | None) -> str:
    """The asset a request names.

    ``asset_key``, else ``cell:<list>`` when no ``card_type`` is given (G-15
    D1); anything else is a 400.
    """
    if asset_key:
        try:
            return _validate_asset_key(asset_key)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "asset_key_invalid"}) from None
    if list_name and not card_type:
        return f"cell:{list_name}"
    raise HTTPException(status_code=400, detail={"error": "asset_key_required"})


_OUTCOME_HTTP: dict[str, tuple[int, str]] = {
    repository.NOT_FOUND: (404, "no_pointer"),
    repository.ACTIVE_CONFLICT: (409, "active_conflict"),
    repository.CANARY_ARMED: (409, "canary_already_armed"),
    repository.CANARY_IS_ACTIVE: (409, "canary_is_active"),
    repository.NO_CANARY: (409, "no_canary"),
    repository.CANARY_ID_MISMATCH: (409, "canary_id_mismatch"),
}


def _pointer_or_raise(outcome: str, row: dict[str, Any] | None) -> WorkflowRoutingPointer:
    if outcome in (repository.OK, repository.UNCHANGED) and row is not None:
        return WorkflowRoutingPointer.from_row(row)
    status, error = _OUTCOME_HTTP.get(outcome, (500, "unexpected_outcome"))
    raise HTTPException(status_code=status, detail={"error": error})


# ---------------------------------------------------------------------------
# Webhook helper (workflowAutoRollback → fr-backend, worker_management#338)
# ---------------------------------------------------------------------------

# fr ``POST /api/internal/workflow-rollback`` (fr#1427) reads exactly this
# header and compares it with fr's ``FR_WM_ROLLBACK_TOKEN``: 503 when fr's token
# is unset, 401 on a bad/absent token, 400 on a bad payload, 200
# ``{"accepted": true}`` (one ``canary_rollback_events`` row) on success.
ROLLBACK_TOKEN_HEADER = "X-Internal-Token"  # noqa: S105 - a header NAME, not a secret
ROLLBACK_WEBHOOK_TIMEOUT = 2.0
ROLLBACK_WEBHOOK_OUTCOMES = ("ok", "http_error", "transport_error", "unconfigured", "error")

ROLLBACK_WEBHOOK_TOTAL = Counter(
    "wm_workflow_rollback_webhook_total",
    "workflowAutoRollback deliveries to fr, by outcome "
    "(ok | http_error | transport_error | unconfigured | error).",
    ["outcome"],
)
# Pre-initialise every outcome at 0 so all five series are scraped before the
# first delivery — an alert on ``increase(...{outcome!="ok"})`` needs the series
# to exist, and a rollback is rare enough that it otherwise never would.
for _outcome in ROLLBACK_WEBHOOK_OUTCOMES:
    ROLLBACK_WEBHOOK_TOTAL.labels(outcome=_outcome)


class _RollbackWebhookSeams:
    """The transport the fr call uses (``None`` = the network)."""

    transport: httpx.AsyncBaseTransport | None = None


def set_rollback_webhook_transport_for_tests(transport: httpx.AsyncBaseTransport | None) -> None:
    """Route the rollback webhook through ``transport`` (``None`` restores the network)."""
    _RollbackWebhookSeams.transport = transport


async def _emit_rollback_webhook(payload: dict[str, Any]) -> None:
    """POST a ``workflowAutoRollback`` event to fr-backend's internal receiver.

    URL (used verbatim) and caller token come from settings
    ``FR_WORKFLOW_ROLLBACK_URL`` / ``FR_WM_ROLLBACK_TOKEN``; the token is sent as
    ``X-Internal-Token``. Every attempt is counted in
    ``wm_workflow_rollback_webhook_total{outcome}``:

    * url or token empty      -> ``unconfigured`` (nothing sent; WARNING — in
      prod it means a mis-provisioned Deployment)
    * url httpx cannot parse  -> ``unconfigured`` (``httpx.InvalidURL``, which is
      NOT an ``httpx.HTTPError``) and a non-ASCII token (``UnicodeEncodeError``
      building the header) — both raised before anything is sent
    * 2xx                     -> ``ok``
    * any other status        -> ``http_error`` (status logged, body not)
    * httpx / OS error        -> ``transport_error`` (class name logged only)
    * ANY other ``Exception`` -> ``error`` (class name logged only) — the final
      catch-all. Enumerating types failed twice (``ExceptionGroup[OverflowError]``
      from an out-of-range port, ``idna.IDNAError`` from ``http://xn--/``), so
      this boundary does not rely on the list being complete. ``BaseException``
      (``asyncio.CancelledError``, ``KeyboardInterrupt``) still propagates.

    NON-blocking by design: the database swap is the source of truth and fr
    only records the event, so a failure here must never fail an emergency
    rollback. "Fail-visible" is the metric (and its gitops alert), not a raise.
    The token is never logged.
    """
    settings = get_settings()
    url = settings.fr_workflow_rollback_url.strip()
    token = settings.fr_wm_rollback_token.strip()
    if not url or not token:
        ROLLBACK_WEBHOOK_TOTAL.labels(outcome="unconfigured").inc()
        logger.warning(
            "fr-backend rollback webhook not sent: FR_WORKFLOW_ROLLBACK_URL %s, "
            "FR_WM_ROLLBACK_TOKEN %s",
            "set" if url else "unset",
            "set" if token else "unset",
        )
        return
    try:
        async with httpx.AsyncClient(
            timeout=ROLLBACK_WEBHOOK_TIMEOUT, transport=_RollbackWebhookSeams.transport
        ) as client:
            resp = await client.post(url, json=payload, headers={ROLLBACK_TOKEN_HEADER: token})
    except (httpx.InvalidURL, UnicodeEncodeError) as exc:
        # Misconfiguration, not transport: neither subclasses httpx.HTTPError,
        # so without this they would raise INTO the rollback request. Class
        # name only — the message can quote the url or the token.
        ROLLBACK_WEBHOOK_TOTAL.labels(outcome="unconfigured").inc()
        logger.warning("fr-backend rollback webhook not sent (bad config): %s", type(exc).__name__)
        return
    except (httpx.HTTPError, OSError) as exc:
        # Class name only: an httpx error string can carry the request.
        ROLLBACK_WEBHOOK_TOTAL.labels(outcome="transport_error").inc()
        logger.warning("fr-backend rollback webhook failed: %s", type(exc).__name__)
        return
    # must-never-raise boundary; outcome counted. Exception only, never
    # BaseException: cancellation must still propagate.
    except Exception as exc:
        ROLLBACK_WEBHOOK_TOTAL.labels(outcome="error").inc()
        # Class name only — str()/repr() can carry the url or the token.
        logger.warning("fr-backend rollback webhook failed (unexpected): %s", type(exc).__name__)
        return
    if 200 <= resp.status_code < 300:
        ROLLBACK_WEBHOOK_TOTAL.labels(outcome="ok").inc()
        return
    ROLLBACK_WEBHOOK_TOTAL.labels(outcome="http_error").inc()
    logger.warning("fr-backend rollback webhook returned HTTP %s", resp.status_code)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


# ⛔ STRICT TENANT SEPARATION ON THE READS (owner ruling 2026-09-30): a pointer
# row is SYSTEM data — it carries ``released_by_card_id``,
# ``released_by_tenant_id``, trace ids, ``canary_id`` and candidate workflow ids
# — and the table has no tenant column, so a row is never a customer's own.
# Only a ``PLATFORM_TENANT_IDS`` caller may read it. Every other tenant is
# answered EXACTLY as a genuine miss (same status, body and headers: the 404
# below on /active, an empty list on /observed), so the refusal discloses
# neither that a pointer exists nor that a gate fired, and a customer resolver
# falls back to its env map as it would with no pointer. The tenant is the one
# the release gate reads (``Depends(get_tenant_id)``: the introspected licence
# tenant, or the asserted ``X-Tenant-Id`` on the trusted-network SA path), and
# the check runs BEFORE any database access.
_NO_POINTER_DETAIL = "No routing pointer for this asset_key"


def _may_read_routing(tenant_id: str | None) -> bool:
    if is_platform_tenant(tenant_id):
        return True
    logger.info("Workflow-routing read answered as a miss for non-platform tenant %s", tenant_id)
    return False


@router.get(
    "/active",
    response_model=WorkflowRoutingPointer,
    operation_id="workflow_routing_active",
)
async def get_active(
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    asset_key: str | None = Query(None, min_length=3, max_length=300),
    card_type: str | None = Query(None, min_length=1, max_length=200),
    list: str | None = Query(None, min_length=1, max_length=200, alias="list"),
) -> WorkflowRoutingPointer:
    """Return the pointer for an asset: active, prior and the armed canary (or null).

    Read by the ncn resolver on dispatch. 404 when no pointer exists — the
    resolver falls back to its env map (G-15 D4). ``?list=`` alone is the
    legacy form (→ ``cell:<list>``); ``?card_type`` without ``asset_key`` → 400.
    A non-platform tenant always gets that same 404 (strict tenant separation).
    """
    key = _resolve_asset_key(asset_key, card_type, list)
    row = await repository.get_active_by_asset_key(key) if _may_read_routing(tenant_id) else None
    if row is None:
        raise HTTPException(status_code=404, detail=_NO_POINTER_DETAIL)
    return WorkflowRoutingPointer.from_row(row)


@router.get(
    "/observed",
    response_model=list[WorkflowRoutingPointer],
    operation_id="workflow_routing_observed",
)
async def list_observed(
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> list[WorkflowRoutingPointer]:
    """List every pointer row currently inside a legacy observation window.

    Used by the ``canary-workflow-rollback-watch`` n8n monitor (retired by R7,
    decision 7). Rows released through the G-15 path have no window and are
    never listed. A non-platform tenant always gets ``[]`` — the same answer
    as no row being observed (strict tenant separation).
    """
    if not _may_read_routing(tenant_id):
        return []
    rows = await repository.list_observed()
    return [WorkflowRoutingPointer.from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Release writes (G-15 D4). Each carries the release gate pair ON ITS OWN ROUTE.
# ---------------------------------------------------------------------------


@router.post(
    "/seed-active",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_seed_active",
    # Pointer-creating write: gated by WORKFLOW_ROUTING_POINTERS_ENABLED, and
    # the flag runs LAST so an unauthorised caller gets 401/403 and never a
    # 409 that would disclose the flag state.
    dependencies=[
        Depends(_require_routing_writer),
        Depends(_require_release_tenant),
        Depends(_require_routing_pointers_enabled),
    ],
)
async def seed_active(
    body: SeedActiveRequest,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> WorkflowRoutingPointer:
    """Create the pointer for an asset from its CURRENT env-map id (Q6).

    Needed once per releasable asset before any release (phase-10: every
    member seeded before run A). Idempotent: the same id again is a 200 with
    nothing written; a DIFFERENT id over an existing row is 409. Not a release.
    """
    _require_releasable(body.asset_key)
    trace_id = _require_trace_id()
    outcome, row = await repository.seed_active(
        asset_key=body.asset_key,
        active_workflow_id=body.active_workflow_id,
        tenant_id=tenant_id,
        trace_id=trace_id,
    )
    if outcome == repository.NOT_FOUND:
        # A concurrent seed committed after this statement's snapshot.
        raise HTTPException(status_code=409, detail={"error": "concurrent_seed"})
    return _pointer_or_raise(outcome, row)


@router.post(
    "/release",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_release",
    # Pointer-creating write: gated by WORKFLOW_ROUTING_POINTERS_ENABLED, and
    # the flag runs LAST so an unauthorised caller gets 401/403 and never a
    # 409 that would disclose the flag state.
    dependencies=[
        Depends(_require_routing_writer),
        Depends(_require_release_tenant),
        Depends(_require_routing_pointers_enabled),
    ],
)
async def release(
    body: ReleaseRequest,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> WorkflowRoutingPointer:
    """Arm a canary for an asset — the per-asset ARMING row (decision 2).

    409 when a canary is already armed or the candidate IS the active; 404
    when the asset was never seeded. Stamps the release ``trace_id`` (400 if
    absent, decision 4), ``released_by_tenant_id`` and ``released_at``. Arming
    is not routing: ncn routes to it only while AA_WORKFLOW_CANARY_ROUTABLE.
    """
    _require_releasable(body.asset_key)
    trace_id = _require_trace_id()
    outcome, row = await repository.release(
        asset_key=body.asset_key,
        canary_workflow_id=body.canary_workflow_id,
        canary_pct=body.canary_pct,
        canary_id=body.canary_id,
        released_by_card_id=body.released_by_card_id,
        tenant_id=tenant_id,
        trace_id=trace_id,
    )
    return _pointer_or_raise(outcome, row)


@router.post(
    "/promote-canary",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_promote_canary",
    dependencies=[Depends(_require_routing_writer), Depends(_require_release_tenant)],
)
async def promote_canary(body: CanaryActionRequest) -> WorkflowRoutingPointer:
    """active := canary, prior := old active, canary cleared (decision 6).

    THE one promoter's pointer call (done-workflow-promote). 409 when no
    canary is armed, ``canary_id`` names a different one (a stale card), or
    ``card_id`` is not the card that released it. ``promoted_trace_id`` is stamped from ``X-AA-Trace-Id`` when present.
    """
    _require_releasable(body.asset_key)
    outcome, row = await repository.promote_canary(
        asset_key=body.asset_key,
        canary_id=body.canary_id,
        card_id=body.card_id,
        trace_id=_captured_trace_id(),
    )
    logger.info(
        "promote-canary %s canary_id=%s card_id=%s -> %s",
        body.asset_key,
        body.canary_id,
        body.card_id,
        outcome,
    )
    return _pointer_or_raise(outcome, row)


@router.post(
    "/discard-canary",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_discard_canary",
    dependencies=[Depends(_require_routing_writer), Depends(_require_release_tenant)],
)
async def discard_canary(body: CanaryActionRequest) -> WorkflowRoutingPointer:
    """Clear the armed canary; the active is unchanged. Same 409s as promote.

    No releasable-set check: disarming is always safe (see ``_require_releasable``).
    """
    outcome, row = await repository.discard_canary(
        asset_key=body.asset_key,
        canary_id=body.canary_id,
    )
    logger.info(
        "discard-canary %s canary_id=%s card_id=%s -> %s",
        body.asset_key,
        body.canary_id,
        body.card_id,
        outcome,
    )
    return _pointer_or_raise(outcome, row)


# ---------------------------------------------------------------------------
# Legacy writes (Lane D). Service role only (#230), unchanged.
# ---------------------------------------------------------------------------


@router.post(
    "/promote",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_promote",
    # Order matters: the writer-role check runs FIRST so an unauthenticated
    # caller still gets 401/403, never a 409 that would leak whether
    # pointer creation is currently enabled.
    dependencies=[Depends(_require_routing_writer), Depends(_require_routing_pointers_enabled)],
)
async def promote(
    body: PromoteRequest,
) -> WorkflowRoutingPointer:
    """Promote a new workflow version for an asset (legacy).

    First-time promote: insert with ``prior_workflow_id = NULL``.
    Subsequent promote: shift current active → prior, set new active.
    ``observation_window_minutes = 0`` opts out of the canary window.
    409 while a canary is armed: that release is promoted or discarded only
    through /promote-canary or /discard-canary.
    """
    row = await repository.promote(
        asset_key=_refuse_release_asset(body.resolved_asset_key()),
        new_workflow_id=body.new_workflow_id,
        observation_window_minutes=body.observation_window_minutes,
        rollback_thresholds=body.rollback_thresholds,
    )
    if row is None:
        raise HTTPException(status_code=409, detail={"error": "canary_already_armed"})
    return WorkflowRoutingPointer.from_row(row)


@router.post(
    "/rollback",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_rollback",
    dependencies=[Depends(_require_routing_writer)],
)
async def rollback(
    body: RollbackRequest,
) -> WorkflowRoutingPointer:
    """Swap ``active_workflow_id`` back to ``prior_workflow_id``.

    404 when no row exists, the row has no prior version, or a canary is
    armed (discard it first). The caller must not blindly retry.

    On success, fires the ``workflowAutoRollback`` webhook to fr-backend
    (non-blocking; every outcome counted in ``wm_workflow_rollback_webhook_total``).
    """
    asset_key = _refuse_release_asset(body.resolved_asset_key())
    row = await repository.rollback(asset_key=asset_key)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No routing pointer with a prior version to roll back to (or a canary is armed)",
        )
    # Contract: fr backend/src/routes/internal-workflow-rollback.ts (fr#1427 +
    # its follow-up). Every pointer field comes from the ROW: the request's
    # card_type/list are None in both accepted shapes (asset_key-only, and
    # list-only → cell:<list>), and after the swap the row's prior is NULL, so
    # ``rolled_back_from`` is the pre-swap active the repository reads (#338).
    await _emit_rollback_webhook(
        {
            "event": "workflowAutoRollback",
            "asset_key": row["asset_key"],
            "card_type": row["card_type"],
            "list": row["list"],
            "active_workflow_id": row["active_workflow_id"],
            "rolled_back_from": row["rolled_back_from"],
            "reason": body.reason,
        }
    )
    return WorkflowRoutingPointer.from_row(row)


@router.post(
    "/extend",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_extend",
    dependencies=[Depends(_require_routing_writer)],
)
async def extend_observation_window(
    body: ExtendRequest,
) -> WorkflowRoutingPointer:
    """Prolong a legacy canary observation window.

    Bumps ``observation_window_until`` by ``additional_minutes`` from the later
    of *now* and the current window end. A row with NO window (every G-15
    release — no clock, decisions 5/7) is returned unchanged with 200: kept,
    not 410, because ``improve::approval-callback`` calls it by raw HTTP.

    404 when there is no row, or a windowed row has no live prior.
    """
    row = await repository.extend_observation_window(
        asset_key=_refuse_release_asset(body.resolved_asset_key()),
        additional_minutes=body.additional_minutes,
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No live canary observation window to extend for this asset_key",
        )
    return WorkflowRoutingPointer.from_row(row)


@router.post(
    "/clear-prior",
    response_model=WorkflowRoutingPointer,
    status_code=200,
    operation_id="workflow_routing_clear_prior",
    dependencies=[Depends(_require_routing_writer)],
)
async def clear_prior(
    body: ClearPriorRequest,
) -> WorkflowRoutingPointer:
    """Clear ``prior_workflow_id`` after the observation window passes.

    Called by the rollback-watch n8n workflow once
    ``now() > observation_window_until`` without a threshold breach. 404
    when no row exists.
    """
    row = await repository.clear_prior(asset_key=_refuse_release_asset(body.resolved_asset_key()))
    if row is None:
        raise HTTPException(status_code=404, detail="No routing pointer for this asset_key")
    return WorkflowRoutingPointer.from_row(row)
