"""FastAPI router for ``/prompts*`` — the prompt version ledger (roadmap A5 / S2).

Specified by ``docs/redesign/design/improvement-design.md`` §2.3 in
``nonagenticai/n8n_custom_nodes``.

WHAT THIS IS FOR. S4 (``n8n_custom_nodes`` #1603) made git the source of truth
for worker prompts and made generated Ensure nodes stamp
``configRevision=<prompt_id>@<version>``. That stamp is an ASSERTION: nothing
could check it, because there was no server-side record of what
``<prompt_id>@<version>`` contains and no response field saying which prompt a
call actually ran. This router is the record; ``MessageResponse.metadata``
(``src/api/workers/message_endpoints.py``) is the report. Together they make a
worker call attributable to an exact prompt id and version, verifiable by
comparing two digests rather than by trusting a label.

⛔ BARE PATHS. ``/prompts``, never ``/api/prompts`` and never ``/api/v1/…`` —
this service mounts bare paths (``/workers``, ``/providers``, ``/graphs``) and
the ingress VirtualService rewrites ``/workers/`` to ``/``.

⛔⛔ AUTHORIZATION — A KNOWN, SERVICE-WIDE GAP. READ BEFORE "FIXING" IT HERE. ⛔⛔

``Depends(get_tenant_id)`` below is AUTHENTICATION ONLY: it resolves any active
licence token to a tenant and checks no role (``src/auth/tenant.py``;
``AuthenticationMiddleware``'s own first line is "No Keycloak, no JWT, no RBAC -
valid license = full access"). So today any holder of a valid licence token could
read and PUT prompt versions (closed for customers — see PLATFORM GATE below),
and these routes are internet-reachable through the
``api.nonagentic.ai/workers/`` prefix rewrite like every other bare route here.

Do NOT close that by adding ``require_role(...)`` to this file. It is the same
gap, with the same remedy in the same order, as
``src/api/workflow_routing/routes.py`` documents at length and as
``worker_management#230`` / the fleet-authz plan's tier C6 sequence
: ``require_role`` reads roles that exist only on a request carrying a
signature-verified ``X-Caller-Authorization`` SA JWT, and ZERO live callers
send one — every gateway-routed request lands on the tokenless branch. Arming a
role check on one new route while the north-south path is unmigrated buys
nothing and diverges this service from a migration being done service-wide.

EXCEPTION — two moves are gated on the elevated ``service-internal`` role
(``PROMPT_ADMIN_ROLE`` below): a DISCARDED target (terminal) and the deprecated
``activate`` (archives every other PUBLISHED version). Neither has a live
caller, so gating them costs nothing; see the comment on ``PROMPT_ADMIN_ROLE``.

⛔ PLATFORM GATE (owner ruling 2026-09-30, strict tenant separation) — the "any
licence holder" above is CLOSED for EVERY route, reads included, without
waiting for the service-wide migration, because this is not tenant data:
``prompt_versions`` carries no customer data — every row, and every byte of
prompt text, is a SYSTEM asset (its owner, migration 035, is a platform
tenant) — and system data is never visible to a customer. Workers run the PUBLISHED version, so a write changes what the
platform's workers do for every tenant; a read hands a customer the system's
prompt text. So EVERY route (the GETs, PUT, POST …/status, POST …/activate)
needs the caller's tenant in ``PLATFORM_TENANT_IDS``
(``src/api/tenant_constants.py``, a mirror of fr's list) OR
``PROMPT_ADMIN_ROLE`` / ``PROMPT_BOOTSTRAP_ROLE``; otherwise 403
``{"detail": {"error": "platform_prompt_forbidden"}}``, before the repository
is touched. It sits at ROUTER level, so a route added later is gated without
anyone remembering to. This is a TENANT check, not a role check, so the warning
above does not apply to it: the tenant is what every live caller does carry.
Measured before gating the reads: 0 ``/prompts`` calls in 72h of WM access logs
(positive control: 925 ``POST /workers``), so no live reader breaks. The role
gates below stay layered on top. The per-call pin on ``POST
/workers/{name}/message`` is closed for customers the same way
(``src/api/workers/prompt_pin.py``: ``tenant_forbidden``).

What bounds the damage MEANWHILE is in the data model, not in a flag, and it is
the guarantee this ledger actually needs:

  * a published ``(prompt_id, version)`` can NEVER be rewritten — a PUT whose
    content digest differs from the stored one is refused 409 (repository:
    ``PromptVersionConflict``). History is append-only;
  * there is no DELETE route at all;
  * every field is length-bounded (``models.py``) and ``prompt_id`` is
    charset-bounded, so the write surface cannot be used for unbounded storage;
  * publishing is not activating. A PUT lands ``DRAFT``; only an explicit
    status move to PUBLISHED (or the deprecated activate) changes what
    ``GET /prompts/{id}`` serves.

LIFECYCLE (migration 031, P4 contract — twin of procedure_management D4 R9/R10):
DRAFT -> CANARY -> PUBLISHED <-> ARCHIVED, any -> DISCARDED (terminal). The
matrix lives in ``repository.TRANSITIONS``. ``PROMPTS_READ_REQUIRES_PUBLISHED``
(default off, exposed on /health) makes the unpinned read resolve the HIGHEST
PUBLISHED version; off, it serves the most recently activated PUBLISHED one —
today's single "active" row.

Endpoints:
- GET  /prompts/search?q=&tag=&status=&limit=&prompt_id=&worker_name=
- GET  /prompts/{prompt_id}                       (unpinned: the resolved PUBLISHED version)
- GET  /prompts/{prompt_id}/versions/{version}    (pinned: any status)
- PUT  /prompts/{prompt_id}/versions/{version}    (publisher; lands DRAFT)
- POST /prompts/{prompt_id}/versions/{version}/status    {"status": ...}
- POST /prompts/{prompt_id}/versions/{version}/activate  (DEPRECATED alias)
- GET  /prompts/{prompt_id}/versions/{version}/transitions (the ledger, migration 033)

TRANSITION LEDGER (migration 033). Every status change — the PUT's creation
(NULL -> DRAFT), every status move, and activate's publish plus each archive it
causes — appends one ``prompt_version_transitions`` row in the same
transaction, with the actor taken from the VERIFIED principal
(``src/api/prompts/actor.py``), never from a header or the body.

INITIAL CREATE (P4 owner ruling 2026-09-28). ``POST …/status
{"status":"PUBLISHED","reason":"initial_create"}`` publishes a DRAFT directly —
creation, not promotion, so it fakes no CANARY hop — ONLY with the elevated
role and ONLY for a prompt that has never had a PUBLISHED or ARCHIVED version
(rows and ledger both checked). Otherwise 409 with the top-level body
``{"error":"initial_create_not_allowed","prompt_id":…,"version":…,"reason":…}``. Every
other DRAFT -> PUBLISHED is still refused by the matrix.

⛔ RELEASE CARD (REALIGN R7, coordinator ruling 2026-10-03). Every prompt
release goes through the board: a move INTO CANARY or PUBLISHED — ``POST
…/status`` (``initial_create`` included) and the deprecated ``POST …/activate``
— must carry ``actor_card_id``, the Planka card id authorising it (exactly 19
ASCII digits). Missing or malformed is 422 ``{"detail":{"error":
"prompt_release_card_required"}}``, checked AFTER the role gates (a role-less
caller still gets its 403) and BEFORE any write. The id is recorded on the
ledger row (migration 039) and returned by ``…/transitions``. ARCHIVED,
DISCARDED and DRAFT targets need none. There is NO exemption for any HTTP
caller, the bootstrap SA included. The only writers without a card are
in-process ``actor_kind=migration`` jobs (``scripts/adopt_live_worker_prompts.py``
adopting already-live text — there is no card to name; its rows say
``adopt_live``/``migration``), and migration 039's CHECK refuses a card-less
release row from anyone else.

⛔ RELEASE CARD MUST EXIST (worker_management#336). After the format check
and before any write, WM asks fr whether the card exists and is owned by a
PLATFORM tenant (:mod:`src.api.prompts.release_card`): 422
``prompt_release_card_unknown`` (fr 404) / ``prompt_release_card_not_platform``
(tenant outside WM's own platform set); 503 ``prompt_release_card_unverifiable``
when no answer can be had (fr down, timeout, unset env, …) — nothing written.
A positive answer is cached per pod for 60 s.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse

from src.api.common_responses import STANDARD_ERROR_MODELS, ErrorResponse
from src.api.prompts import release_card, repository
from src.api.prompts.actor import resolve_actor
from src.api.prompts.hashing import sha256_text, version_payload_sha256
from src.api.prompts.models import (
    PromptActivateRequest,
    PromptActivateResponse,
    PromptSearchHit,
    PromptSearchResponse,
    PromptStatusChange,
    PromptStatusResponse,
    PromptTransition,
    PromptTransitionsResponse,
    PromptVersionResponse,
    PromptVersionUpsert,
    PromptVersionWriteResponse,
    is_valid_actor_card_id,
    validate_prompt_id,
)
from src.api.tenant_constants import is_platform_tenant, system_tenant_id
from src.auth import caller_identity, get_tenant_id
from src.auth.caller_identity import require_any_role, require_role
from src.settings import get_settings

logger = logging.getLogger(__name__)

# Annotated explicitly: a dict comprehension over STANDARD_ERROR_MODELS narrows
# to `dict[int, ...]`, which is not assignable to FastAPI's
# `dict[int | str, dict[str, Any]]` and makes `ty` red on every decorator.
_ERRORS: dict[int | str, dict[str, Any]] = {
    k: v for k, v in STANDARD_ERROR_MODELS.items() if k in (400, 401, 403, 404, 409, 422, 500)
}
# Every route can refuse 403 (the platform gate). Write routes can also refuse
# 403 for the elevated moves and 503 (031 not applied).
_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    **_ERRORS,
    503: {"model": ErrorResponse},
}

# ⛔ ELEVATED AUTH for the two moves that cannot be undone or that change what
# every caller is served: a DISCARDED target (terminal — nothing leaves
# DISCARDED) and the deprecated ``activate`` (archives every other PUBLISHED
# version). Plain licence auth (``get_tenant_id``) admits ANY licence holder,
# so those two additionally require the verified service role this service
# already uses for its other privileged path (``graph_sessions_correlation_api``):
# ``require_role`` reads roles ONLY from a signature-verified, azp-allowlisted
# ``X-Caller-Authorization`` SA JWT (``src/auth/keycloak_roles.py``), never
# from a request header. The warning in this module's docstring about
# ``require_role`` is about the ORDINARY routes, whose live callers send no SA
# JWT; these two have no live caller (the only consumer, n8n_custom_nodes'
# prompt-publisher, issues GET and PUT only and its test pins that), so gating
# them breaks nothing and is fail-closed from day one. Every other transition
# keeps the router's licence auth plus the platform write gate below.
PROMPT_ADMIN_ROLE = "service-internal"
_require_prompt_admin = require_role(PROMPT_ADMIN_ROLE)

# ⛔ The ONE narrower role, accepted on EXACTLY ONE move: ``initial_create``.
# It exists for the ``claude-asset-bootstrap`` Keycloak SA (ncn#1780), which
# creates INITIAL prompt assets. That SA must NOT carry ``service-internal``:
# licenses and improvement-loop-service accept ``service-internal`` on role
# alone, so granting it would be a fleet-wide over-grant. Only this service
# maps ``prompt-asset-bootstrap``, and only here — DISCARDED, ``activate`` and
# every other ``service-internal`` gate (e.g. the unscoped cross-tenant read in
# ``graph_sessions_correlation_api``) still demand ``service-internal`` and 403
# / scope-restrict a bootstrap token. The role is still read only from a
# verified, azp-allowlisted SA JWT, so the SA's azp must be in
# ``CALLER_ALLOWED_CLIENTS`` for the role to count at all.
PROMPT_BOOTSTRAP_ROLE = "prompt-asset-bootstrap"
_require_initial_create = require_any_role(PROMPT_ADMIN_ROLE, PROMPT_BOOTSTRAP_ROLE)

# ⛔ PLATFORM GATE — on EVERY route, reads and writes (router-level, below).
# ``prompt_versions`` has no tenant column: these are SYSTEM assets. A write
# changes what the platform's workers do for every tenant; a read shows a
# customer system prompt text, which strict tenant separation (owner ruling
# 2026-09-30) forbids. Licence auth alone let ANY customer licence do both. A
# call now needs the caller's tenant in ``PLATFORM_TENANT_IDS`` (mirror of fr's
# list) OR one of the two verified SA roles above, read from the same source as
# ``require_role``. Measured live before this gate: the only writers ever were
# ``actor_kind=migration`` and ``claude-asset-bootstrap`` (bootstrap role), and
# there were 0 ``/prompts`` reads in 72h, so it breaks no legitimate caller.
# The role gates above stay layered on top.
_PLATFORM_ROLES = frozenset({PROMPT_ADMIN_ROLE, PROMPT_BOOTSTRAP_ROLE})
_PLATFORM_FORBIDDEN = {"error": "platform_prompt_forbidden"}


def _require_platform_caller(tenant_id: Annotated[str, Depends(get_tenant_id)]) -> None:
    """403 unless the caller is a platform tenant or holds a prompt role.

    Fixed detail, deliberately: it names neither the caller's tenant nor the
    accepted roles, so a refused customer learns nothing it could probe with.
    """
    if is_platform_tenant(tenant_id):
        return
    # Looked up through the module at call time (not imported by name), as
    # ``require_role`` does, so tests patching it reach this gate too.
    if not _PLATFORM_ROLES.isdisjoint(caller_identity.get_caller_roles_from_context()):
        return
    logger.warning("Refused prompt call from non-platform tenant %s", tenant_id)
    raise HTTPException(status_code=403, detail=_PLATFORM_FORBIDDEN)


# ⛔ BOTH dependencies at ROUTER level, not per-route: a route added later
# without ``get_tenant_id`` would be unauthenticated, and one without the
# platform gate would show system prompt data to a customer — and nothing would
# say so. Router dependencies run before a route's own, so on ``activate`` a
# customer still gets the fixed platform 403 before the admin-role check.
router = APIRouter(
    prefix="/prompts",
    tags=["Prompts"],
    dependencies=[Depends(get_tenant_id), Depends(_require_platform_caller)],
)


def _require_lifecycle_schema() -> None:
    """503 while the startup probe says migration 031 is not applied.

    The production migration runner is non-strict, so a failed 031 leaves the
    pod serving on the 030 CHECK ('active','retired'), where every lifecycle
    write would be a CheckViolation 500. ``unknown`` (never probed) is let
    through: the database CHECK is still the backstop, and a CheckViolation is
    mapped to the same 503 by :func:`_schema_refusal`.
    """
    if repository.lifecycle_schema_state() == repository.SCHEMA_MISSING_031:
        raise _schema_refusal()


def _schema_refusal() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "prompt_lifecycle_schema_missing: migration 031_prompt_version_lifecycle is "
            "not applied to prompt_versions (its status CHECK does not admit PUBLISHED); "
            "prompt writes are refused until it is. Reads are unaffected."
        ),
    )


def _ledger_refusal() -> HTTPException:
    """503 when migration 033 (the transition ledger) is not applied.

    The runner is non-strict, so a failed 033 leaves a pod whose every write
    would otherwise 500 on the missing table. The write is refused WHOLE — the
    ledger row shares the status change's transaction, so no status moves
    unrecorded.
    """
    return HTTPException(
        status_code=503,
        detail=(
            "prompt_transition_ledger_missing: migration 033_prompt_version_transitions "
            "is not applied; prompt writes are refused until it is. Reads are unaffected."
        ),
    )


_OWNER_MISMATCH = {"error": "prompt_owner_mismatch"}


def _owner_refusal(exc: repository.PromptOwnerMismatch) -> HTTPException:
    """403 for a write to a prompt another tenant owns (migration 035).

    Same envelope as the platform gate's 403 (``{"detail": {"error": …}}``).
    The detail names neither the owner nor the caller.
    """
    logger.warning("Refused prompt write: %s", exc)
    return HTTPException(status_code=403, detail=_OWNER_MISMATCH)


def _owner_schema_refusal() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "prompt_owner_schema_missing: migration 035_prompt_owner is not applied; "
            "prompt writes are refused until it is. Reads are unaffected."
        ),
    )


def _card_schema_refusal() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "prompt_release_card_schema_missing: migration 039_prompt_release_card is "
            "not applied; prompt writes and ledger reads are refused until it is."
        ),
    )


def _missing_column_refusal(exc: Exception) -> HTTPException:
    """UndefinedColumn -> the 503 naming the migration whose column is missing."""
    if "actor_card_id" in str(exc):
        return _card_schema_refusal()
    return _owner_schema_refusal()


def _check_refusal(exc: psycopg.errors.CheckViolation) -> HTTPException:
    """CheckViolation -> 422 for 039's release-card backstop, else the 031 503."""
    if getattr(exc.diag, "constraint_name", None) == repository.RELEASE_CARD_CONSTRAINT:
        return HTTPException(status_code=422, detail=_RELEASE_CARD_REQUIRED)
    return _schema_refusal()


def _missing_table_refusal(exc: Exception) -> HTTPException:
    """UndefinedTable -> the 503 naming the migration whose table is missing."""
    if "prompt_owner" in str(exc):
        return _owner_schema_refusal()
    return _ledger_refusal()


def _holds_prompt_role() -> bool:
    """True when the verified caller holds a prompt role (the system SAs)."""
    return not _PLATFORM_ROLES.isdisjoint(caller_identity.get_caller_roles_from_context())


# ⛔ U6 step 4: a move INTO CANARY or PUBLISHED needs a prompt role. With
# WM_PROMPT_SOURCE=published the highest PUBLISHED version of a BOUND prompt is
# the live system prompt of its workers within the resolver TTL, so
# DRAFT -> CANARY -> PUBLISHED by ANY platform licence (the router gate alone)
# would rewrite what workers run. Applied to every prompt, not only bound ones:
# a binding can be added after the publish, and "bound" is a race the route
# would have to win. The two roles are the same ones the PUT already treats as
# privileged (``_holds_prompt_role``): the live writers are actor_kind=migration
# (in-process, never this route) and ``claude-asset-bootstrap``
# (``prompt-asset-bootstrap``). Moves OUT (ARCHIVED; DISCARDED has its own
# admin gate) keep the platform gate: retiring a version never adds one.
_PUBLISH_ROLE_REQUIRED = {"error": "prompt_publish_role_required"}
_ROLE_GATED_TARGETS = frozenset({repository.CANARY, repository.PUBLISHED})


# ⛔ REALIGN R7: the release's board card (see the module docstring).
_RELEASE_CARD_REQUIRED = {"error": "prompt_release_card_required"}


def _require_release_card(to_status: str, actor_card_id: Any) -> None:
    """422 ``prompt_release_card_required`` for a CANARY/PUBLISHED target without a card.

    Called after the role gates and before any write.
    """
    if to_status in _ROLE_GATED_TARGETS and not is_valid_actor_card_id(actor_card_id):
        logger.warning("Refused prompt move to %s: no valid actor_card_id", to_status)
        raise HTTPException(status_code=422, detail=_RELEASE_CARD_REQUIRED)


async def _verify_release_card(to_status: str, actor_card_id: Any) -> None:
    """worker_management#336: the card must EXIST and be a PLATFORM card (asks fr).

    Called after :func:`_require_release_card` (so the id is well-formed) and
    before any write. 422 ``prompt_release_card_unknown`` /
    ``prompt_release_card_not_platform``; 503 ``prompt_release_card_unverifiable``
    on any failure to get an answer. See :mod:`src.api.prompts.release_card`.
    """
    if to_status in _ROLE_GATED_TARGETS:
        await release_card.verify_release_card(actor_card_id)


def _require_publish_role(to_status: str) -> None:
    """403 ``prompt_publish_role_required`` for a CANARY/PUBLISHED target.

    Passes when the verified caller holds a prompt role.
    """
    if to_status in _ROLE_GATED_TARGETS and not _holds_prompt_role():
        logger.warning("Refused prompt move to %s: caller holds no prompt role", to_status)
        raise HTTPException(status_code=403, detail=_PUBLISH_ROLE_REQUIRED)


def _initial_create_refusal(exc: repository.InitialCreateNotAllowed) -> JSONResponse:
    """The ruling's 409 body — TOP-LEVEL keys, not wrapped in ``detail``.

    Same shape as the other pin errors (``src/api/workers/prompt_pin.py``:
    ``{"error", "prompt_id", "version"}``) plus ``reason``.
    """
    return JSONResponse(
        status_code=409,
        content={
            "error": "initial_create_not_allowed",
            "prompt_id": exc.prompt_id,
            "version": exc.version,
            "reason": exc.reason,
        },
    )


# The ledger is small (one row per prompt version, ~40 prompts today) but the
# cap is what stops a caller asking for the whole table in one request.
_MAX_SEARCH_LIMIT = 100
# main's pre-031 text for an unknown ``status=``, kept byte-identical while
# PROMPTS_READ_REQUIRES_PUBLISHED is off (flag-off = today's responses).
_LEGACY_SEARCH_STATUS_ERROR = "status must be one of active, retired, any"
# ``status=`` values the search route accepts, mapped to ledger statuses.
# ``active`` / ``retired`` are the pre-031 spellings, kept so existing callers
# (the default included) mean what they meant: ``active`` = what is live,
# ``retired`` = everything that is not. ``any`` = no filter.
_SEARCH_STATUS: dict[str, list[str] | None] = {
    **{s.lower(): [s] for s in repository.STATUSES},
    "active": [repository.PUBLISHED],
    "retired": [s for s in repository.STATUSES if s != repository.PUBLISHED],
    "any": None,
}


def _prompt_id(raw: str) -> str:
    try:
        return validate_prompt_id(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# ⚠️ ROUTE ORDER IS LOAD-BEARING. FastAPI matches in registration order, so the
# static ``/search`` segment MUST be declared before ``/{prompt_id}`` or a
# search request is served as a lookup of a prompt literally named "search".
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=PromptSearchResponse,
    operation_id="search_prompts",
    responses=_ERRORS,
    summary="Full-text search over published prompt versions",
)
async def search_prompts(
    q: Annotated[str | None, Query(max_length=500, description="Free text query")] = None,
    tag: Annotated[
        str | None, Query(max_length=255, description="Exact purpose_tags match")
    ] = None,
    status: Annotated[
        str,
        Query(
            description="DRAFT | CANARY | PUBLISHED | ARCHIVED | DISCARDED | any, or the "
            "legacy active (=PUBLISHED) / retired (=not PUBLISHED). Default: active."
        ),
    ] = "active",
    limit: Annotated[int, Query(ge=1, le=_MAX_SEARCH_LIMIT)] = 20,
    prompt_id: Annotated[
        str | None,
        Query(
            description="Exact prompt_id match, combinable with every other filter. "
            "The way to list ONE prompt's versions or prove it absent: an unfiltered "
            "listing is capped at `limit` across all prompts and has no offset."
        ),
    ] = None,
    worker_name: Annotated[
        str | None,
        Query(
            max_length=255,
            description="Exact match against an ELEMENT of the row's worker_names, "
            "combinable with every other filter; `limit` still applies.",
        ),
    ] = None,
) -> PromptSearchResponse:
    """Ranked hits, ordered ``ts_rank desc, prompt_id asc`` (design §2.3).

    Defaults to ``status=active`` because a retrieval shortlist must be
    actionable: a hit on a retired version would send a caller to a prompt no
    worker is running. Pass ``status=any`` to include retired versions — that is
    what replay and a publisher's "what is already here" need.
    """
    key = status.lower()
    if key not in _SEARCH_STATUS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"status must be one of {', '.join(sorted(_SEARCH_STATUS))} (case-insensitive)"
                if get_settings().prompts_read_requires_published
                else _LEGACY_SEARCH_STATUS_ERROR
            ),
        )
    # Passed ONLY when given, so a caller that omits it gets exactly today's
    # query and response.
    extra: dict[str, str] = {} if prompt_id is None else {"prompt_id": _prompt_id(prompt_id)}
    if worker_name is not None:
        extra["worker_name"] = worker_name
    hits = await repository.search(q=q, tag=tag, status=_SEARCH_STATUS[key], limit=limit, **extra)
    return PromptSearchResponse(
        query=q,
        tag=tag,
        status=status,
        prompt_id=extra.get("prompt_id"),
        count=len(hits),
        hits=[PromptSearchHit(**hit) for hit in hits],
    )


@router.get(
    "/{prompt_id}",
    response_model=PromptVersionResponse,
    operation_id="get_active_prompt",
    responses=_ERRORS,
    summary="The live (PUBLISHED) version of a prompt",
)
async def get_active_prompt(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
) -> PromptVersionResponse:
    """The unpinned read. 404 when the prompt has no PUBLISHED version.

    Flag on (``PROMPTS_READ_REQUIRES_PUBLISHED``): the HIGHEST PUBLISHED
    version. Flag off: today's behaviour, the most recently activated PUBLISHED
    version. DRAFT/CANARY/ARCHIVED/DISCARDED are never resolved here either
    way; the 404 is honest — there is no prompt a caller should be running. Use
    the version route to read a specific one.
    """
    pid = _prompt_id(prompt_id)
    flag = get_settings().prompts_read_requires_published
    row = await repository.get_active(pid, highest_published=flag)
    if row is None:
        # Flag off: main's text, byte-identical. Flag on: the lifecycle wording.
        detail = (
            f"No PUBLISHED version for prompt '{pid}'. Unpublished versions are "
            f"readable at /prompts/{pid}/versions/{{version}}."
            if flag
            else f"No ACTIVE version for prompt '{pid}'. Published-but-not-activated "
            f"versions are readable at /prompts/{pid}/versions/{{version}}."
        )
        raise HTTPException(status_code=404, detail=detail)
    return PromptVersionResponse(**row)


@router.get(
    "/{prompt_id}/versions/{version}",
    response_model=PromptVersionResponse,
    operation_id="get_prompt_version",
    responses=_ERRORS,
    summary="One prompt version, any status (the pinned read)",
)
async def get_prompt_version(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
    version: Annotated[int, PathParam(ge=1, description="Version number")],
) -> PromptVersionResponse:
    """Every status stays readable — that is what replay and a pinned stamp read."""
    pid = _prompt_id(prompt_id)
    row = await repository.get_version(pid, version)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Prompt version '{pid}@{version}' not found")
    return PromptVersionResponse(**row)


@router.put(
    "/{prompt_id}/versions/{version}",
    response_model=PromptVersionWriteResponse,
    operation_id="publish_prompt_version",
    responses=_WRITE_ERRORS,
    summary="Publish a prompt version (immutable; idempotent on content)",
)
async def publish_prompt_version(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
    version: Annotated[int, PathParam(ge=1, description="Version number")],
    body: PromptVersionUpsert,
    request: Request,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> PromptVersionWriteResponse:
    """Publish ``(prompt_id, version)``, or confirm it is already published.

    Idempotent ON CONTENT, which is what lets the publisher (roadmap A16) just
    PUT every registry entry every run instead of doing its own GET-and-compare:

      * absent                  → inserted, ``written=true``
      * present, same content   → 200, ``written=false, unchanged=true``
      * present, other content  → **409**. Versions are immutable. You bump, you
        never rewrite — the whole value of a version ledger is that a historical
        attribution can still be checked later.

    Both digests are computed HERE, from the body, by
    ``src/api/prompts/hashing.py`` — the same module the live-call echo uses.
    A client-supplied ``expected_sha256`` is only ever an assertion to compare
    against (422 on mismatch); it is never stored as given.

    OWNERSHIP (migration 035): a NEW ``prompt_id`` becomes owned by the
    caller's tenant; a version of a prompt another tenant owns is **403**
    ``{"detail": {"error": "prompt_owner_mismatch"}}`` unless the caller holds
    a prompt role (``PROMPT_ADMIN_ROLE`` / ``PROMPT_BOOTSTRAP_ROLE``).
    """
    pid = _prompt_id(prompt_id)
    _require_lifecycle_schema()
    payload_sha = version_payload_sha256(
        question=body.question,
        system=body.system,
        template=body.template,
        output_schema=body.output_schema,
        worker_names=body.worker_names,
        purpose_tags=body.purpose_tags,
    )
    if body.expected_sha256 is not None and body.expected_sha256 != payload_sha:
        raise HTTPException(
            status_code=422,
            detail=(
                f"expected_sha256 {body.expected_sha256} does not match the digest the "
                f"server computes for this payload ({payload_sha}). The caller's "
                "canonicalisation has drifted from src/api/prompts/hashing.py."
            ),
        )

    try:
        row, written = await repository.upsert_version(
            prompt_id=pid,
            version=version,
            sha256=payload_sha,
            system_prompt_sha256=sha256_text(body.system),
            question=body.question,
            system=body.system,
            template=body.template,
            output_schema=body.output_schema,
            worker_names=body.worker_names,
            purpose_tags=body.purpose_tags,
            git_sha=body.git_sha,
            actor=resolve_actor(request),
            # A NEW prompt is owned by the caller's tenant, EXCEPT for the system
            # service accounts (service-internal / prompt-asset-bootstrap): their
            # licence tenant is incidental (the bootstrap SA's is the AA conveyor
            # tenant), and every prompt they ever wrote is backfilled to the system
            # tenant in 035 -- so new ones are system-owned too (U6 ownership).
            owner_tenant_id=system_tenant_id() if _holds_prompt_role() else tenant_id,
            privileged=_holds_prompt_role(),
        )
    except repository.PromptOwnerMismatch as exc:
        raise _owner_refusal(exc) from exc
    except repository.PromptVersionConflict as exc:
        logger.warning("Refused prompt rewrite: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.errors.CheckViolation as exc:
        raise _schema_refusal() from exc
    except psycopg.errors.UndefinedTable as exc:
        raise _missing_table_refusal(exc) from exc
    except psycopg.errors.UndefinedColumn as exc:
        raise _missing_column_refusal(exc) from exc

    logger.info(
        "prompt version %s@%s %s (sha256=%s)",
        pid,
        version,
        "published" if written else "already published, unchanged",
        payload_sha,
    )
    return PromptVersionWriteResponse(
        prompt_id=row["prompt_id"],
        version=row["version"],
        sha256=row["sha256"],
        system_prompt_sha256=row["system_prompt_sha256"],
        status=row["status"],
        config_revision=row["config_revision"],
        written=written,
        unchanged=not written,
    )


@router.post(
    "/{prompt_id}/versions/{version}/status",
    response_model=PromptStatusResponse,
    operation_id="transition_prompt_version_status",
    responses=_WRITE_ERRORS,
    summary="Move a prompt version along its lifecycle",
)
async def transition_prompt_version_status(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
    version: Annotated[int, PathParam(ge=1, description="Version number")],
    body: PromptStatusChange,
    request: Request,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> PromptStatusResponse | JSONResponse:
    """Apply one lifecycle move, checked against the transition matrix.

    DRAFT->CANARY|DISCARDED, CANARY->PUBLISHED|DISCARDED,
    PUBLISHED->ARCHIVED|DISCARDED, ARCHIVED->PUBLISHED|DISCARDED. Anything else
    (including a same-status move) is 409 and writes nothing; an unpublished
    version is 404.

    Unlike the deprecated activate, publishing here does NOT archive the other
    PUBLISHED versions: several may be PUBLISHED and the flag-on resolver picks
    the highest.

    A DISCARDED target is terminal and needs the elevated role
    (:data:`PROMPT_ADMIN_ROLE`, 403 otherwise). A CANARY or PUBLISHED target
    needs a prompt role (:data:`PROMPT_ADMIN_ROLE` or
    :data:`PROMPT_BOOTSTRAP_ROLE`; 403 ``{"detail":{"error":
    "prompt_publish_role_required"}}`` otherwise). ARCHIVED needs only the
    router's platform gate (platform tenant or a prompt role).

    ``reason`` (optional, <= 200 chars) is recorded on the ledger row.
    ``reason="initial_create"`` with ``status=PUBLISHED`` is the sanctioned
    initial publish from DRAFT: elevated role OR :data:`PROMPT_BOOTSTRAP_ROLE`
    (403 otherwise) and no PUBLISHED
    or ARCHIVED history on any version of the prompt (409 otherwise, top-level
    ``{"error":"initial_create_not_allowed",…}``).

    OWNERSHIP (migration 035): same rule as the PUT — a version of a prompt
    another tenant owns is 403 ``prompt_owner_mismatch`` unless the caller
    holds a prompt role.

    RELEASE CARD (REALIGN R7): a CANARY or PUBLISHED target needs
    ``actor_card_id`` (19 digits; 422 ``{"detail":{"error":
    "prompt_release_card_required"}}`` otherwise, after the role checks and
    before any write), recorded on the ledger row.
    """
    pid = _prompt_id(prompt_id)
    # DISCARDED first: a DISCARDED target with reason=initial_create is still a
    # discard and still needs the admin role, never the bootstrap one.
    if body.status == repository.DISCARDED:
        _require_prompt_admin(request)
    elif body.reason == repository.INITIAL_CREATE:
        _require_initial_create(request)
    _require_publish_role(body.status)
    _require_release_card(body.status, body.actor_card_id)
    await _verify_release_card(body.status, body.actor_card_id)
    _require_lifecycle_schema()
    try:
        row, previous_status = await repository.transition(
            pid,
            version,
            body.status,
            actor=resolve_actor(request),
            reason=body.reason,
            actor_card_id=body.actor_card_id,
            caller_tenant_id=tenant_id,
            privileged=_holds_prompt_role(),
        )
    except repository.PromptOwnerMismatch as exc:
        raise _owner_refusal(exc) from exc
    except repository.InitialCreateNotAllowed as exc:
        logger.warning("Refused initial_create: %s", exc)
        return _initial_create_refusal(exc)
    except repository.InvalidStatusTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.errors.CheckViolation as exc:
        raise _check_refusal(exc) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise _missing_table_refusal(exc) from exc
    except psycopg.errors.UndefinedColumn as exc:
        raise _missing_column_refusal(exc) from exc
    if row is None or previous_status is None:
        raise HTTPException(status_code=404, detail=f"Prompt version '{pid}@{version}' not found")
    logger.info(
        "prompt version %s@%s status %s -> %s (reason=%r)",
        pid,
        version,
        previous_status,
        row["status"],
        body.reason,
    )
    return PromptStatusResponse(
        prompt_id=row["prompt_id"],
        version=row["version"],
        status=row["status"],
        previous_status=previous_status,
        config_revision=row["config_revision"],
        activated_at=row["activated_at"],
    )


@router.post(
    "/{prompt_id}/versions/{version}/activate",
    response_model=PromptActivateResponse,
    operation_id="activate_prompt_version",
    responses=_WRITE_ERRORS,
    summary="DEPRECATED: publish this version and archive every other PUBLISHED one",
    deprecated=True,
    # The router's platform gate runs FIRST, so a customer's refusal is the same
    # fixed body on every route; the admin role is then still required on top.
    dependencies=[Depends(_require_prompt_admin)],
)
async def activate_prompt_version(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
    version: Annotated[int, PathParam(ge=1, description="Version number")],
    request: Request,
    body: PromptActivateRequest | None = None,
) -> PromptActivateResponse:
    """Publish this version and archive the others (DEPRECATED alias).

    Alias of ``POST …/status {"status": "PUBLISHED"}`` that ALSO
    archives every other PUBLISHED version, in one transaction — preserving the
    pre-031 "exactly one live version" semantics for existing callers. Like the
    old route it publishes from DRAFT/CANARY/ARCHIVED directly; a DISCARDED
    version is 409.

    Requires the elevated role (:data:`PROMPT_ADMIN_ROLE`; 403 otherwise): it
    archives every other PUBLISHED version, i.e. it changes what every caller
    is served.

    Separate from the PUT on purpose: publishing writes a record, activating
    changes what callers get. A lower version may be activated — that is a
    deliberate rollback, and refusing it would make rollback impossible.

    RELEASE CARD (REALIGN R7): always a release, so the body must carry
    ``{"actor_card_id": "<19 digits>"}`` (422 ``prompt_release_card_required``
    otherwise, after the role check and before any write). It is recorded on
    every ledger row the call appends.
    """
    pid = _prompt_id(prompt_id)
    card = body.actor_card_id if body is not None else None
    _require_release_card(repository.PUBLISHED, card)
    await _verify_release_card(repository.PUBLISHED, card)
    _require_lifecycle_schema()
    try:
        row, previous_version, changed = await repository.activate(
            pid, version, actor=resolve_actor(request), actor_card_id=card
        )
    except repository.InvalidStatusTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.errors.CheckViolation as exc:
        raise _check_refusal(exc) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise _ledger_refusal() from exc
    except psycopg.errors.UndefinedColumn as exc:
        raise _missing_column_refusal(exc) from exc
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt version '{pid}@{version}' not found; publish it before activating",
        )
    logger.info(
        "prompt version %s@%s activated (previous active: %s, changed=%s)",
        pid,
        version,
        previous_version,
        changed,
    )
    return PromptActivateResponse(
        prompt_id=row["prompt_id"],
        version=row["version"],
        status=row["status"],
        config_revision=row["config_revision"],
        previous_active_version=previous_version,
        changed=changed,
    )


@router.get(
    "/{prompt_id}/versions/{version}/transitions",
    response_model=PromptTransitionsResponse,
    operation_id="list_prompt_version_transitions",
    responses=_ERRORS,
    summary="The status-transition ledger of one prompt version",
)
async def list_prompt_version_transitions(
    prompt_id: Annotated[str, PathParam(description="Registry prompt id")],
    version: Annotated[int, PathParam(ge=1, description="Version number")],
) -> PromptTransitionsResponse:
    """Every recorded status change of ``prompt_id@version``, oldest first.

    Same auth as the other reads (the router's licence auth + platform gate:
    a customer is 403, the ledger is system data). 404 when the
    version does not exist. Rows written before migration 033 are limited to
    the backfilled creation row (``actor_kind=migration``). 503 (the write
    paths' ``prompt_transition_ledger_missing``) when migration 033 is absent.
    """
    pid = _prompt_id(prompt_id)
    try:
        rows = await repository.get_transitions(pid, version)
    except psycopg.errors.UndefinedTable as exc:
        raise _ledger_refusal() from exc
    except psycopg.errors.UndefinedColumn as exc:
        raise _card_schema_refusal() from exc
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Prompt version '{pid}@{version}' not found")
    return PromptTransitionsResponse(
        prompt_id=pid,
        version=version,
        count=len(rows),
        transitions=[PromptTransition(**r) for r in rows],
    )
