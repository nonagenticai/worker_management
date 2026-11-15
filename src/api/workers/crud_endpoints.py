"""CRUD endpoints for worker management (create, list, get, update, delete)."""

import hashlib
import logging
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import unquote
from uuid import uuid7

from fastapi import Depends, HTTPException, Query, Request, Response
from fastapi import Path as PathParam

from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.error_handlers import APIError, APIValidationError, ConflictError
from src.api.prompts.hashing import format_prompt_revision, parse_prompt_revision
from src.api.security_validation import SecurityValidationError, validate_identifier
from src.api.tenant_constants import is_platform_tenant
from src.api.workers._globals import worker_cache, worker_registry
from src.api.workers._router import router
from src.api.workers.message_endpoints import AUTO_CREATE_MODEL_PROVIDER, _auto_create_model
from src.api.workers.models import (
    PromptSwapRequest,
    PromptSwapResponse,
    WorkerCreateRequest,
    WorkerResponse,
    derive_agent_type,
    validate_effective_worker,
)
from src.api.workers.prompt_pin import load_runnable_version, unversioned_revision
from src.auth import TenantContext, get_tenant_context
from src.auth.tokenless_gate import (
    ROUTE_WORKERS_CREATE,
    ROUTE_WORKERS_DELETE,
    ROUTE_WORKERS_DELETE_ALL,
    ROUTE_WORKERS_LIST,
    ROUTE_WORKERS_PROMPT_SWAP,
    ROUTE_WORKERS_TYPES_LIST,
    ROUTE_WORKERS_UPDATE,
    ROUTE_WORKERS_WORKER_STATUS,
    tokenless_tenant_gate,
)
from src.db.worker_registry_db import WorkerNameConflictError
from src.db.workers_state_db import StateDatabase
from src.settings import get_settings
from src.worker.worker_factory import create_worker

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


logger = logging.getLogger(__name__)


# Every config key the API persists, named once. POST and PUT used to carry two
# near-identical dict literals which had silently drifted: PUT omitted
# container_id / executor_endpoint / workspace / work_tier, so a PUT *deleted*
# those four keys from a config a POST had set. Measured on prod 2026-08-28:
# 445 rows carry container_id, and 0 of the 207 status='updated' rows carry
# work_tier. One tuple means the two handlers cannot drift again.
_CONFIG_FIELDS: tuple[str, ...] = (
    "tool_names",
    "agent_role",
    "agent_type",
    "model_provider",
    "model",
    "temperature",
    "seed",
    "ir_provider",
    "repetition_penalty",
    "frequency_penalty",
    "enforce_schema_bounds",
    "max_tokens",
    "system_prompt",
    "instruction",
    "user_prompt",
    "description",
    "deps",
    "mcp_config",
    "mcp_config_preset",
    "mcp_servers",
    "base_url_override",
    "api_key_override",
    "tools_dir_override",
    "output_schema",
    "output_validator",
    "output_retries",
    "container_id",
    "executor_endpoint",
    "workspace",
    "work_tier",
)

# Both handlers stored these as ``None`` rather than ``""`` (``request.X if
# request.X else None``). Preserved verbatim so a consumer reading the config
# sees the shape it always has.
_EMPTY_STRING_IS_NONE: frozenset[str] = frozenset({"agent_role", "agent_type"})


def _config_payload(request: WorkerCreateRequest, *, only_set: bool = True) -> dict[str, Any]:
    """The config keys this request actually asserts.

    With ``only_set`` (the default) a key is emitted ONLY when the client sent
    it, via Pydantic's ``model_fields_set``. That is what makes the jsonb merge
    in :meth:`WorkerRegistryDB.register_worker` non-destructive: a field the
    caller stayed silent about is preserved, while a field it sent as an
    explicit JSON ``null`` is still in ``model_fields_set`` and therefore still
    clears.

    Before this, both handlers read ``request.<attr>`` unconditionally, so
    "omitted" and "explicitly null" were indistinguishable — a bootstrapper
    sending 8 of these 25 fields nulled the other 17. `knowledge/kag`'s
    startup PUT does exactly that on every pod start.

    ``only_set=False`` reproduces the old full payload, for the one caller that
    is genuinely authoritative for every field (the auto-create follow-through).
    """
    sent = request.model_fields_set
    payload: dict[str, Any] = {}
    for field in _CONFIG_FIELDS:
        if only_set and field not in sent:
            continue
        value = getattr(request, field)
        if field in _EMPTY_STRING_IS_NONE and not value:
            value = None
        payload[field] = value
    return payload


async def _validated_config_payload(
    request: WorkerCreateRequest,
    load_existing_config: Callable[[], Awaitable[dict[str, Any] | None]],
    *,
    only_set: bool = True,
) -> dict[str, Any]:
    """The config keys to write, having validated the worker they will PRODUCE.

    The cross-field rules ("model is required when model_provider is
    specified", and friends) used to run at FastAPI parse time against the
    request body alone. That was right while a write replaced the whole config
    — the body was the worker. Since the jsonb merge landed, the body is a
    SUBSET of the resulting worker, so those rules had to move here, where the
    stored row is known. See ``validate_effective_worker``.

    ``agent_type`` is folded back into the payload when the derivation changes
    it, because a derived value must still be PERSISTED: ``_globals.py`` reads
    ``config.get("agent_type")`` straight from the row, so a create that did
    not store it would silently downgrade a pydantic-ai worker to a basic one.
    When the derivation agrees with the stored row the key is left ABSENT, so
    merge semantics are preserved for every caller that stays silent about it.

    ``existing_config`` is the config the row will HOLD when this write lands —
    which on the PUT auto-create path is the config that branch just wrote, NOT
    an empty dict. Validating that path against ``{}`` would 422 a body the
    auto-create had itself just made valid.
    """
    payload = _config_payload(request, only_set=only_set)

    existing_config: dict[str, Any] = {}
    try:
        # Exactly what parse time used to check: the fields the CLIENT sent,
        # not any a convenience endpoint injected afterwards. A caller that
        # works today therefore cannot start failing. POST is the hot path
        # (19,272 `config-updated` rows against 3,092 from PUT), and this
        # branch costs it no extra query.
        validate_effective_worker(
            {k: v for k, v in payload.items() if k in request._client_sent_fields}
        )
    except ValueError:
        # Not coherent alone — but it may be a legitimate PARTIAL update of a
        # row that already supplies the rest. NOW it is worth a read.
        loaded = await load_existing_config()
        existing_config = loaded if isinstance(loaded, dict) else {}
        merged = {**existing_config, **payload}
        try:
            validate_effective_worker(merged)
        except ValueError as exc:
            if not _violation_predates_this_request(existing_config, exc):
                raise APIValidationError(str(exc)) from exc
            # Grandfathered: the stored row ALREADY broke this rule and this
            # request did not introduce it. Measured on prod 2026-08-29: 44 of
            # 751 rows carry an ``agent_type`` with an empty ``agent_role``,
            # and 44 carry ``pydantic_ai`` with no ``model_provider``. Raising
            # here would make those 44 workers permanently un-updatable — a
            # strictly worse defect than the one this validation exists to fix.
            # A request can still never INTRODUCE a violation, nor make an
            # existing one worse.
            return payload

    derived_agent_type = derive_agent_type({**existing_config, **payload})
    if "agent_type" in payload or derived_agent_type != existing_config.get("agent_type"):
        payload["agent_type"] = derived_agent_type

    return payload


async def _already(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """The PUT path has already read the row — hand it over, don't re-query."""
    return value


def _violation_predates_this_request(existing_config: dict[str, Any], exc: ValueError) -> bool:
    """Did the stored row already fail this exact rule, before this request?"""
    if not existing_config:
        return False
    try:
        validate_effective_worker(existing_config)
    except ValueError as prior:
        return str(prior) == str(exc)
    return False


# The six fields the worker_states audit trail records. Kept as its own tuple:
# this trail is the only reason the config clobber was provable at all, and it
# is the post-deploy verification signal — so it must record what the row now
# HOLDS, not what the caller happened to send.
_AUDIT_CONFIG_FIELDS: tuple[str, ...] = (
    "agent_role",
    "agent_type",
    "model_provider",
    "model",
    "system_prompt",
    "instruction",
)


def _audit_config(persisted_config: dict[str, Any]) -> dict[str, Any]:
    """The audit-metadata view of a worker's config, read from the stored row."""
    return {field: persisted_config.get(field) for field in _AUDIT_CONFIG_FIELDS}


def _is_agent_arg(request: WorkerCreateRequest) -> bool | None:
    """``is_agent`` for this request, or ``None`` for "the caller did not say".

    ``is_agent`` is derived from ``agent_role``, so it is asserted exactly when
    ``agent_role`` is. Passing the old unconditional ``bool(request.agent_role)``
    flipped an existing agent row to ``False`` on any request that omitted the
    field — 698 of 750 live rows are ``is_agent = true``.
    """
    if "agent_role" not in request.model_fields_set:
        return None
    return bool(request.agent_role)


# Worker Routes
@router.post(
    "",
    response_model=WorkerResponse,
    status_code=201,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 409, 422, 500]}
    },
    operation_id="create_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_CREATE))],
)
async def create_worker_endpoint(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
) -> WorkerResponse:
    """Create a new worker or AI agent with specified configuration.

    Workers can be simple tool executors or full AI agents with LLM capabilities.
    The worker is registered in the database and an initial state is saved.

    This **upserts**: if the worker already exists its config is updated, and
    that update is partial by field presence — an omitted field keeps its stored
    value, an explicit JSON ``null`` clears it. Same rule as
    ``PUT /workers/{worker_name}``.

    Raises HTTPException for validation errors (422), missing credentials (401), or creation failures (500).
    """
    try:
        # Additional validation
        if not request.worker_name or not request.worker_name.strip():
            raise HTTPException(status_code=422, detail="Worker name cannot be empty")

        # Generate worker_id first so we can use it consistently
        worker_id = str(uuid7())

        # Normalize worker name to ensure consistency
        normalized_worker_name = request.worker_name.lower().replace(" ", "_")

        # Create worker instance
        worker = await create_worker(
            # The END tenant this worker is built for: inference-router model
            # validation lists as THIS tenant (no WM/platform tenant fallback).
            tenant_id=ctx.tenant_id,
            worker_name=normalized_worker_name,
            tool_names=request.tool_names,
            agent_role=request.agent_role if request.agent_role else None,
            agent_type=request.agent_type if request.agent_type else None,
            model_provider=request.model_provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            instruction=request.instruction,
            user_prompt=request.user_prompt,
            description=request.description,
            deps=request.deps,
            mcp_config=request.mcp_config,
            mcp_config_preset=request.mcp_config_preset,
            mcp_servers=request.mcp_servers,
            base_url_override=request.base_url_override,
            api_key_override=request.api_key_override,
            tools_dir_override=request.tools_dir_override,
            output_schema=request.output_schema,
            output_validator_code=request.output_validator,
            output_retries=request.output_retries,
            disable_state_storage=request.disable_state_storage,
            container_id=request.container_id,
            executor_endpoint=request.executor_endpoint,
            workspace=request.workspace,
        )

        # Only the keys this request actually asserts — see _config_payload.
        # This endpoint upserts, and it is the HOT path (prod audit 2026-08-28:
        # 19,272 `config-updated` rows against 3,092 from PUT), so a partial
        # body here must not null the fields it stayed silent about.
        # This endpoint upserts, so the row it will produce is the stored
        # config merged with this body — that is what must satisfy the
        # cross-field rules, not the body alone. One indexed lookup.
        async def _load_existing_config() -> dict[str, Any] | None:
            """Only called when this body is not coherent on its own."""
            row = await worker_registry.get_worker(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                worker_name=request.worker_name,
            )
            return (row or {}).get("config")

        config_to_register = await _validated_config_payload(request, _load_existing_config)

        # Log identifying fields only -- NEVER the config values. The config
        # can carry `api_key_override` / `base_url_override` (see
        # `_overrides_for_caller` above) and other caller-supplied fields, so
        # logging the dict itself puts credentials in the log stream at INFO.
        # The key set is enough to debug what a request asserted.
        logger.info(
            "Registering worker %s (agent_type=%s): config keys=%s",
            normalized_worker_name,
            request.agent_type,
            sorted(config_to_register.keys()),
        )

        # Register worker in database (upserts: creates if new, merges config if exists)
        reg_result = await worker_registry.register_worker(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            tenant_name=ctx.tenant_name,
            worker_name=request.worker_name,
            worker_id=worker_id,
            db_path=worker.db_path,
            is_agent=_is_agent_arg(request),
            config=config_to_register,
            visibility=ctx.visibility,
            config_revision=request.config_revision,
        )

        # Detect upsert: register_worker returns the existing worker_id on update
        existing_worker_id = reg_result.get("worker_id")
        is_update = existing_worker_id and existing_worker_id != worker_id
        if is_update:
            # Use existing worker_id to preserve state history
            worker_id = existing_worker_id
            # Evict stale cached instance so next request uses updated config
            await worker_cache.remove_worker(normalized_worker_name)
            logger.info(
                f"Updated existing worker '{request.worker_name}' config "
                f"(provider={request.model_provider}, model={request.model}), cache evicted"
            )

        # What the row actually holds after the merge — not the request's view.
        persisted_config = reg_result.get("config") or {}
        persisted_is_agent = bool(reg_result.get("is_agent"))

        # Save state entry (config-updated for existing workers, created for new ones)
        state_persistence_failed = False
        state_type = "config-updated" if is_update else "created"
        try:
            db = StateDatabase()
            await db.add_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                tenant_name=ctx.tenant_name,
                worker_id=worker_id,
                worker_name=request.worker_name,
                is_agent=persisted_is_agent,
                state_type=state_type,
                worker_state={"status": state_type},
                visibility=ctx.visibility,
                metadata={"config": _audit_config(persisted_config)},
            )
        except Exception as state_error:
            state_persistence_failed = True

            logger.error(
                f"Failed to save worker state: {state_error}",
                exc_info=True,
                extra={
                    "worker_id": worker_id,
                    "worker_name": request.worker_name,
                    "traceback": traceback.format_exc(),
                },
            )

        worker_status = (
            "updated"
            if is_update
            else ("created_without_state" if state_persistence_failed else "created")
        )
        return _worker_response(
            record={**reg_result, "config": persisted_config},
            ctx=ctx,
            worker_id=worker_id,
            name=request.worker_name,
            status=worker_status,
            db_path=worker.db_path,
            is_agent=persisted_is_agent,
            echo_overrides=False,
        )
    except APIError:
        raise  # 4xx envelopes are the answer, not a 500
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except WorkerNameConflictError as e:
        # worker_name already owned by another subject in this tenant: a genuine
        # 409, not a 500. (The get-or-create race is resolved inside
        # register_worker and returns idempotently — it never reaches here.)
        raise ConflictError("Worker", e.worker_name) from e
    except Exception as e:
        error_msg = str(e)

        # Log the full traceback for debugging
        logger.exception(f"Error creating worker: {error_msg}")
        logger.exception(f"Full traceback: {traceback.format_exc()}")

        # Handle specific API key errors with appropriate status codes
        if "API key required" in error_msg or "OPENAI_API_KEY" in error_msg:
            raise HTTPException(
                status_code=422,
                detail="API key required for the specified model provider. Please provide an API key or set the appropriate environment variable.",
            ) from e
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid API credentials provided") from e

        # Generic server error for other cases
        raise HTTPException(status_code=500, detail=f"Failed to create worker: {error_msg}") from e


# Type-specific worker creation endpoints


def _secrets_withheld_reason(ctx: TenantContext) -> str | None:
    """Why this caller may not read a worker's secret-bearing fields, or ``None``.

    The secret-bearing fields are ``api_key_override``, ``base_url_override``
    and ``mcp_config``. Returning ``None`` requires BOTH flags to be explicitly
    True — the same two-flag rule ``_modality_router._withhold_reason`` applies
    to the decrypted modality key:

    * ``identity_proven`` — a licence token was presented and introspected.
    * ``tenant_proven`` — that token's OWN tenant is ``ctx.tenant_id``, i.e. the
      tenant the worker row was read under (``get_worker`` is
      ``WHERE tenant_id = ctx.tenant_id``).

    ⛔ ``identity_proven`` ALONE IS NOT ENOUGH. A delegating service account
    (subject in ``AUTH_DELEGATING_SUBJECTS``) presents a real token — identity
    proven — and then re-scopes the request with ``X-Effective-Tenant-Id``, a
    header validated on spelling alone (``tenant_proven=False``). Gating on
    identity alone handed that caller every tenant's worker secrets. There is no
    platform-admin role in this service; a caller that needs another tenant's
    worker secrets needs a token FOR that tenant.

    ``is True`` rather than truthiness, and ``getattr`` with a False default, so
    every shape that is not an explicit double-True withholds.
    """
    if getattr(ctx, "identity_proven", False) is not True:
        return "unproven_identity"
    if getattr(ctx, "tenant_proven", False) is not True:
        return "adopted_tenant"
    return None


def _overrides_for_caller(
    config: dict,
    *,
    withheld_reason: str | None,
    worker_name: str,
    tenant_id: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(api_key_override, base_url_override)``, withheld if not entitled.

    ``withheld_reason`` comes from :func:`_secrets_withheld_reason`; ``None``
    means the caller proved both its identity and this tenant.

    Mirrors the redaction shape PR #244 established for
    ``/tenant-config/modality/{modality}/resolve``
    (``src/api/tenant_config/_modality_router.py::_key_for_caller``): withhold
    only the secrets, leave the rest of the record and the 200 intact, and only
    for a caller that never proved its identity. A proven caller keeps reading
    its own tenant's worker unchanged.

    ⛔ BOTH fields are secrets here, and the earlier reasoning that
    ``base_url_override`` was safe to expose was wrong. It is not only written
    by ``llm_config_resolver.py`` / ``message_endpoints.py`` from a plain
    endpoint URL -- it is ALSO written straight from caller input by
    ``POST /workers`` and ``PUT /workers/{name}``, and the model validator
    rejects only the empty string. Nothing stops a tenant storing
    ``https://<user>:<secret>@host/v1``, or a URL carrying a token in a query
    parameter, for a self-hosted BYO endpoint. Handing that to an unproven mesh
    pod is the same leak as ``api_key_override``, so it gets the same treatment
    rather than a promise about how callers "should" use the field.
    """
    api_key_override = config.get("api_key_override")
    base_url_override = config.get("base_url_override")
    if withheld_reason is None:
        return api_key_override, base_url_override

    withheld = [
        name
        for name, value in (
            ("api_key_override", api_key_override),
            ("base_url_override", base_url_override),
        )
        if value is not None
    ]
    if withheld:
        logger.warning(
            "worker response withheld %s for worker=%s tenant=%s reason=%s: "
            "caller did not prove both its identity and this tenant (a delegating "
            "service account's own token does not prove the tenant it adopts).",
            ",".join(withheld),
            worker_name,
            tenant_id,
            withheld_reason,
        )
    return None, None


def _worker_response(
    *,
    record: dict[str, Any],
    ctx: TenantContext,
    worker_id: str | None,
    name: str,
    status: str,
    db_path: str | None,
    is_agent: bool,
    echo_overrides: bool = True,
) -> WorkerResponse:
    """Build a ``WorkerResponse`` from what the row HOLDS — the ONE builder.

    ``record`` is either ``worker_registry.get_worker(...)`` (GET) or
    ``worker_registry.register_worker(...)``'s return (POST / PUT upsert); both
    carry the persisted ``config`` jsonb plus the ``output_schema`` /
    ``output_validator`` / ``config_revision`` columns.

    Why one builder: POST and PUT each hand-built their response and each
    dropped fields GET returned — ``max_tokens`` among them, so every caller
    comparing the upsert's echo against its own spec (the n8n infer atomics'
    worker_config_drift postcondition, prod 2026-09-24) saw ``max_tokens: null``
    against a row that stored the requested value. A field added to
    ``WorkerResponse`` is now added once, for every route.
    """
    config = record.get("config") or {}

    # Columns first, then config: config IS the storage for these today (the
    # columns are unwired — see get_worker_status), the column read covers the
    # in-memory per-message override and the day they are wired.
    output_schema = record.get("output_schema")
    if output_schema is None:
        output_schema = config.get("output_schema")
    output_validator = record.get("output_validator")
    if output_validator is None:
        output_validator = config.get("output_validator")

    # POST / PUT never echoed the two overrides (nor mcp_config) and still do not
    # (``echo_overrides=False``): an upsert's response lands in the caller's
    # execution log (n8n stores every node output), so echoing a key the caller
    # just sent only copies a secret into a second store. GET keeps its
    # proven-identity rule.
    withheld_reason = _secrets_withheld_reason(ctx)
    if echo_overrides:
        api_key_override, base_url_override = _overrides_for_caller(
            config,
            withheld_reason=withheld_reason,
            worker_name=name,
            tenant_id=ctx.tenant_id,
        )
    else:
        api_key_override, base_url_override = None, None

    return WorkerResponse(
        id=worker_id,
        name=name,
        status=status,
        db_path=db_path,
        is_agent=is_agent,
        agent_type=config.get("agent_type"),
        model_provider=config.get("model_provider"),
        model=config.get("model"),
        system_prompt=config.get("system_prompt"),
        instruction=config.get("instruction"),
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
        tool_names=config.get("tool_names") or [],
        output_schema=output_schema,
        output_validator=output_validator,
        output_retries=config.get("output_retries"),
        # mcp_config can carry an auth dict (check_request_mcp_config keeps it
        # whole), so it follows the overrides' rule exactly: never on the upsert
        # echo, and on GET only to a caller that proved its identity AND this
        # tenant (see _secrets_withheld_reason).
        mcp_config=(config.get("mcp_config") or config.get("mcp_config_preset"))
        if echo_overrides and withheld_reason is None
        else None,
        base_url_override=base_url_override,
        api_key_override=api_key_override,
        config_revision=record.get("config_revision"),
    )


@router.get(
    "/types",
    response_model=dict[str, list[str]],
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 500]}},
    operation_id="list_worker_types",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_TYPES_LIST))],
)
async def list_worker_types(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],  # noqa: ARG001
) -> dict[str, list[str]]:
    """List all available agent types that can be used to create workers."""
    try:
        import src.worker.agents  # noqa: F401, RUF100
        from src.worker.agents.agent_registry import list_available_agent_types

        return {"agent_types": list_available_agent_types()}
    except ImportError:
        # Fallback if registry not available
        return {
            "agent_types": [
                "pydantic_ai",
                "claude_code",
                "codex",
                "remote_claude",
                "remote_codex",
            ]
        }


# Every query parameter ``GET /workers`` understands. Anything else is refused
# rather than ignored — see ``_reject_unknown_query_params``.
LIST_WORKERS_QUERY_PARAMS: frozenset[str] = frozenset({"prefix", "limit", "offset"})

# Hard ceiling on one response, and — when the caller names no page — the size
# the whole visible population must fit inside.
#
# The default used to be 100. Measured on the live service 2026-09-23: one
# tenant has 165 workers, so `GET /workers` returned 100 of 165 and nothing in
# the response said so. Two ways out were considered and one was measured to be
# unsafe:
#
#   * Refuse the truncated default with a 4xx. Loud, but it converts a wrong
#     answer into an outage for callers that are content with a page, including
#     an n8n scheduled sweep that has no error path.
#   * Make the default COMPLETE. The default now returns every visible row up to
#     this ceiling, so it cannot be a partial answer; a population that would
#     exceed the ceiling is refused with a 400 naming the total, never silently
#     cut. Explicit limit/offset still page, and say so in the headers.
#
# This is deliberately NOT an unbounded default — an unbounded default is the
# other hazard, and this bound is the same one explicit limits already carried
# (`le=1000`), so it is not a new number. Largest tenant measured: 229 rows, so
# the refusal arm is currently unreachable and is a genuine signal if it fires.
LIST_WORKERS_MAX_LIMIT = 1000


def _reject_unknown_query_params(request: Request, allowed: frozenset[str], route: str) -> None:
    """422 on any query parameter the route does not implement.

    FastAPI ignores unknown query parameters by default. For a *filter* that is
    the worst possible behaviour: ``GET /workers?prefix=foo`` returned the whole
    unfiltered first page, byte-identical to ``GET /workers``, so a caller asking
    for a cohort silently received an arbitrary slice of everything and had no
    way to tell. Measured 2026-09-23 against the live service: ``prefix=aa``,
    ``prefix=zzz_nonexistent_prefix``, ``owner_tenant_id=<all-zero uuid>`` and no
    filter at all produced four identical md5s.

    ``owner_tenant_id`` is refused rather than implemented on purpose. This list
    is already scoped to the caller's verified tenant by
    :func:`~src.repositories.visibility.scope_clause`; a caller-supplied tenant
    filter is therefore either redundant with that scope or an attempt to read
    across it. A query parameter must never be able to widen tenant scope, and
    the ``workers`` table has no owner column distinct from ``tenant_id`` for one
    to mean anything narrower.
    """
    unknown = sorted(set(request.query_params.keys()) - allowed)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported query parameter(s) for {route}: {', '.join(unknown)}. "
                f"Supported: {', '.join(sorted(allowed))}. "
                "This endpoint refuses filters it cannot apply instead of ignoring them."
            ),
        )


@router.get(
    "",
    response_model=list[str],
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 405, 422, 500]}
    },
    operation_id="list_workers",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_LIST))],
)
async def list_workers(
    request: Request,
    response: Response,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    prefix: str | None = Query(
        None,
        max_length=100,
        description=(
            "Case-insensitive worker-name prefix, applied server-side. An "
            "unmatched prefix returns an empty list; it never falls back to the "
            "unfiltered population."
        ),
    ),
    limit: int | None = Query(
        None,
        ge=1,
        le=LIST_WORKERS_MAX_LIMIT,
        description=(
            "Page size. Omit it to receive the complete matching population "
            f"(up to {LIST_WORKERS_MAX_LIMIT}); a population larger than that is "
            "refused with 400 rather than truncated."
        ),
    ),
    offset: int = Query(0, ge=0, description="Number of workers to skip"),
) -> list[str]:
    """List worker names visible to the caller's tenant, filtered and paginated.

    The response body is, and stays, a bare JSON array of names. That is not
    inertia: ``skill_builder``'s ``_find_available_worker`` guards on
    ``isinstance(payload, list)`` and its docstring records the outage an
    envelope caused last time (0 executions against 37 registered skills), and
    both n8n nodes spread an array into one item per worker. So the truncation
    signal is carried in headers plus the *absence* of truncation by default,
    not by wrapping the body.

    Guarantees, none of which held before:

    * **A supplied filter is applied or refused, never ignored.** ``prefix`` is a
      real ``WHERE``; any other query parameter is a 422.
    * **A default response is complete.** With no ``limit``, either every
      matching row is returned or the request is refused with a 400 naming the
      total. There is no silent short page.
    * **A paged response says it is one.** Every 200 carries ``X-Total-Count``,
      ``X-Has-More``, ``X-Limit`` and ``X-Offset``.

    Raises:
        HTTPException: 422 for unknown query params or pagination overflow; 400
            when the population exceeds what one unpaged response may carry.
    """
    _reject_unknown_query_params(request, LIST_WORKERS_QUERY_PARAMS, "GET /workers")

    effective_limit = LIST_WORKERS_MAX_LIMIT if limit is None else limit

    # Validate pagination boundary to prevent integer overflow
    if offset + effective_limit > 2**31 - 1:
        raise HTTPException(status_code=422, detail="Pagination parameters would cause overflow")

    normalized_prefix = prefix.strip() if prefix else None

    total = await worker_registry.count_workers(
        tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, prefix=normalized_prefix
    )

    if limit is None and total > LIST_WORKERS_MAX_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{total} workers match this query, more than the {LIST_WORKERS_MAX_LIMIT} "
                "one unpaged response may carry. Refusing to return a truncated list that "
                "cannot be told apart from a complete one. Pass an explicit limit and "
                "offset to page through, or narrow the query with prefix."
            ),
        )

    workers = await worker_registry.list_workers(
        tenant_id=ctx.tenant_id,
        subject_id=ctx.subject_id,
        limit=effective_limit,
        offset=offset,
        prefix=normalized_prefix,
    )

    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Limit"] = str(effective_limit)
    response.headers["X-Offset"] = str(offset)
    response.headers["X-Has-More"] = "true" if offset + len(workers) < total else "false"
    return workers


@router.get(
    "/{worker_name}",
    response_model=WorkerResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="get_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_WORKER_STATUS))],
)
async def get_worker_status(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    worker_name: str = PathParam(
        ...,
        description="Name of the worker to retrieve",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> WorkerResponse:
    """Get detailed status and configuration of a specific worker.

    Status is determined from the latest worker state record if available.

    Raises HTTPException for invalid names (422), worker not found (404), or internal errors (500).
    """
    try:
        # Additional validation for edge cases that FastAPI might miss
        if not worker_name or not worker_name.strip():
            raise HTTPException(status_code=422, detail="Worker name cannot be empty")

        # Allow Unicode characters in worker names - no ASCII restriction

        # URL decode the worker name first
        worker_name = unquote(worker_name)

        # Check for invalid characters and patterns that might bypass FastAPI validation
        if (
            worker_name.lower() in ["states", "message", "null", "health", "", "/"]
            or "\x00" in worker_name
            or len(worker_name) == 0
            or worker_name.startswith("/")
            or worker_name.endswith("/")
        ):
            raise HTTPException(status_code=422, detail="Invalid worker name")

        # Check length constraints
        if len(worker_name) > 100:
            raise HTTPException(status_code=422, detail="Worker name too long")

        # Additional pattern validation - Unicode-aware validation
        # Use explicit character checking instead of regex for better Unicode support
        if len(worker_name) > 0:
            # Check first character: must be Unicode letter, digit, or underscore
            if not (worker_name[0].isalnum() or worker_name[0] == "_"):
                raise HTTPException(
                    status_code=422,
                    detail="Worker name must start with a Unicode letter, digit, or underscore",
                )

            # Check remaining characters: letters, digits, underscores, hyphens
            for char in worker_name[1:]:
                if not (char.isalnum() or char in "_-"):
                    raise HTTPException(
                        status_code=422,
                        detail="Worker name can only contain Unicode letters, digits, underscores, and hyphens",
                    )

        worker_data = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
        )
        if not worker_data:
            raise HTTPException(status_code=404, detail="Worker not found")

        # Ensure worker_data has required fields
        if (
            "worker_name" not in worker_data
            or "db_path" not in worker_data
            or "config" not in worker_data
        ):
            raise HTTPException(
                status_code=500, detail="Invalid worker data retrieved from database"
            )

        # Get latest state from StateDatabase
        latest_state = None
        try:
            db = StateDatabase()
            latest_state = await db.get_latest_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                worker_id=worker_data.get("worker_id", worker_name),
            )
        except Exception as e:
            logger.warning(f"Could not retrieve latest state for worker {worker_name}: {e!s}")

        # Use the latest state status if available
        status = "unknown"
        if (
            latest_state
            and "worker_state" in latest_state
            and "status" in latest_state["worker_state"]
        ):
            status = latest_state["worker_state"]["status"]
        else:
            status = worker_data.get("status", "unknown")

        # worker_data first, then config. NOT because the columns are canonical —
        # they are empty. Measured on prod 2026-08-29: output_schema /
        # output_validator are non-NULL on 0 of 751 rows because no handler ever
        # passes those kwargs to register_worker, while ~140 rows carry the schema
        # inside config. config IS the storage here; the worker_data read exists
        # for the in-memory per-message override (message_endpoints
        # ._apply_message_override) and for the day the columns are actually wired.
        # ⛔ PLAINTEXT SECRET ON A TOKENLESS ROUTE. `config["api_key_override"]`
        # is a per-worker LLM API key stored in the clear (unlike
        # `tenant_modality_config.api_key_encrypted`, this column is never
        # encrypted at rest). `ctx` comes from `get_tenant_context`, which
        # covers BOTH the header-omitted and header-asserted shapes of the
        # tokenless trusted-network branch — see
        # `AuthenticationMiddleware._trusted_network_tenant`: no header stamps
        # the literal DEFAULT_TENANT_ID, a header present passes a syntax-only
        # name/uuid check with no allowlist. Either way `identity_proven` is
        # False (`_stamp_nothing_proven`), so withhold the key rather than
        # hand any pod on the mesh — including n8n, which runs user-authored
        # workflows — a live credential for whatever tenant it names.
        #
        # Mirrors the redaction shape PR #244 established for
        # `/tenant-config/modality/{modality}/resolve`
        # (`src/api/tenant_config/_modality_router.py::_key_for_caller`):
        # withhold only the secret, not the rest of the record, and only for
        # an unproven caller — a caller that DID prove its identity keeps
        # reading its own tenant's worker unchanged. Unlike that route this
        # one has no `hasKey`/`keyWithheld` sibling fields to preserve, so
        # there is nothing else to mirror; the field is set to `None`, same
        # as `_key_for_caller`'s masking shape.
        #
        # `base_url_override` is redacted TOO, and the earlier reasoning for
        # leaving it exposed was wrong. It is not only written by
        # `llm_config_resolver.py` / `message_endpoints.py` from a plain
        # endpoint URL -- it is ALSO written straight from caller input by
        # `POST /workers` (:283) and `PUT /workers/{name}` (:795, :927), and
        # `models.py`'s validator rejects only the empty string. Nothing stops
        # a tenant storing `https://<user>:<secret>@host/v1` or a URL carrying a
        # token in a query parameter for a self-hosted BYO endpoint. Handing
        # that to an unproven mesh pod is the same leak as `api_key_override`,
        # so it gets the same treatment rather than a promise about how
        # callers "should" use the field.
        # Both the column-then-config read and the override redaction live in
        # _worker_response, shared with POST/PUT so the three cannot drift apart.
        return _worker_response(
            record=worker_data,
            ctx=ctx,
            worker_id=worker_data.get("worker_id"),
            name=worker_data["worker_name"],
            status=status,
            db_path=worker_data["db_path"],
            is_agent=worker_data["is_agent"],
        )
    except APIError:
        raise  # 4xx envelopes are the answer, not a 500
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception:
        logger.exception(f"Error getting worker status for {worker_name}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.put(
    "/{worker_name}",
    response_model=WorkerResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="update_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_UPDATE))],
)
async def update_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: WorkerCreateRequest,
    worker_name: str = PathParam(
        ...,
        description="Name of the worker to update",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> WorkerResponse:
    """Update an existing worker's configuration.

    **Partial by field presence.** A config field you OMIT is left as it is; a
    field you send as an explicit JSON ``null`` is cleared. Send only the keys
    you own. (Until 2026-08-28 this replaced the whole config, so omitting a
    field nulled it — a service re-registering 8 of ~25 fields on every startup
    wiped the other 17, including an operator's hand-edited model and prompt.)

    ⚠️ **Two exceptions to "omitted is left as it is", both known:**

    * ``agent_type`` is DERIVED, not merely defaulted: when you send
      ``agent_role`` and ``model_provider`` without it,
      ``validate_effective_worker`` derives ``"pydantic_ai"`` and
      ``_validated_config_payload`` folds it back into the write — so that
      request DOES overwrite a stored ``agent_type``. The derivation cannot
      simply be dropped, because a create must store it. Send ``agent_type``
      explicitly if you have a non-``pydantic_ai`` worker. (When the derived
      value matches the stored one the key is left absent, so a caller that
      stays silent still merges.)
    * ``config_revision`` is preserved on omit and cannot be cleared through
      this endpoint — see ``register_worker``'s docstring.

    Worker must exist. Request worker_name must match path parameter. Worker cache is cleared after update.

    Raises HTTPException for validation errors (422), worker not found (404), or update failures (500).
    """
    try:
        # URL decode the worker name first, then run the same identifier policy
        # the body validator enforces — including reserved-name rejection
        # (path param bypasses WorkerCreateRequest.validate_worker_name).
        decoded_worker_name = unquote(worker_name)
        try:
            decoded_worker_name = validate_identifier(
                decoded_worker_name, "worker_name", max_length=100, allow_unicode=True
            )
        except SecurityValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Check if worker exists before updating
        existing_worker = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=decoded_worker_name
        )

        # The config the row will hold once this write lands — the basis for
        # the cross-field validation in _validated_config_payload.
        effective_base_config: dict[str, Any] | None = None

        # Auto-create worker if it doesn't exist (similar to send_message)
        if not existing_worker:
            logger.info(
                f"Worker '{decoded_worker_name}' not found during update. Auto-creating with platform-model defaults."
            )

            # Use request values if provided, otherwise use defaults
            auto_agent_role = (
                request.agent_role
                if request.agent_role
                else f"AI Assistant for {decoded_worker_name}"
            )
            # Same platform-model pair as send_message's auto-create path: the
            # old "mlx"/"mlx" default pointed at the retired MLX upstream.
            auto_model_provider = (
                request.model_provider if request.model_provider else AUTO_CREATE_MODEL_PROVIDER
            )
            auto_model = request.model if request.model else _auto_create_model()
            auto_temperature = request.temperature if request.temperature is not None else 0.1

            try:
                # Generate worker ID for auto-created worker
                auto_worker_id = str(uuid7())

                # Create the worker instance
                auto_worker = await create_worker(
                    # The END tenant this worker is built for: inference-router model
                    # validation lists as THIS tenant (no WM/platform tenant fallback).
                    tenant_id=ctx.tenant_id,
                    worker_name=decoded_worker_name,
                    tool_names=request.tool_names if request.tool_names else [],
                    agent_role=auto_agent_role,
                    agent_type=(request.agent_type if request.agent_type else "pydantic_ai"),
                    model_provider=auto_model_provider,
                    model=auto_model,
                    temperature=auto_temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                    instruction=request.instruction,
                    user_prompt=request.user_prompt,
                    description=(
                        request.description
                        if request.description
                        else f"Auto-created worker for {decoded_worker_name}"
                    ),
                    deps=request.deps,
                    mcp_config=request.mcp_config,
                    mcp_config_preset=request.mcp_config_preset,
                    mcp_servers=request.mcp_servers,
                    base_url_override=request.base_url_override,
                    api_key_override=request.api_key_override,
                    tools_dir_override=request.tools_dir_override,
                    output_schema=request.output_schema,
                    output_validator_code=request.output_validator,
                    output_retries=request.output_retries,
                    disable_state_storage=request.disable_state_storage,
                )

                # Prepare configuration for registry
                auto_worker_config = {
                    "tool_names": request.tool_names if request.tool_names else [],
                    "agent_role": auto_agent_role,
                    "agent_type": (request.agent_type if request.agent_type else "pydantic_ai"),
                    "model_provider": auto_model_provider,
                    "model": auto_model,
                    "temperature": auto_temperature,
                    "max_tokens": request.max_tokens,
                    "system_prompt": request.system_prompt,
                    "instruction": request.instruction,
                    "user_prompt": request.user_prompt,
                    "description": (
                        request.description
                        if request.description
                        else f"Auto-created worker for {decoded_worker_name}"
                    ),
                    "deps": request.deps,
                    "mcp_config": request.mcp_config,
                    "mcp_config_preset": request.mcp_config_preset,
                    "mcp_servers": request.mcp_servers,
                    "base_url_override": request.base_url_override,
                    "api_key_override": request.api_key_override,
                    "tools_dir_override": request.tools_dir_override,
                    "output_schema": request.output_schema,
                    "output_validator": request.output_validator,
                    "output_retries": request.output_retries,
                    "disable_state_storage": request.disable_state_storage,
                }

                # What the row will hold when the follow-through register runs
                # below — only if this branch actually writes it.
                if not request.disable_state_storage:
                    effective_base_config = auto_worker_config

                # Register worker in database (unless state storage is disabled)
                if not request.disable_state_storage:
                    await worker_registry.register_worker(
                        tenant_id=ctx.tenant_id,
                        subject_id=ctx.subject_id,
                        tenant_name=ctx.tenant_name,
                        worker_name=request.worker_name,
                        worker_id=auto_worker_id,
                        db_path=auto_worker.db_path,
                        is_agent=True,
                        config=auto_worker_config,
                        status="auto-created",
                        visibility=ctx.visibility,
                        config_revision=request.config_revision,
                    )

                logger.info(
                    f"Successfully auto-created worker '{decoded_worker_name}' with mlx defaults"
                )

                # Set as existing worker for the rest of the update logic
                existing_worker = await worker_registry.get_worker(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    worker_name=decoded_worker_name,
                )
                existing_worker_id = auto_worker_id

            except Exception as create_error:
                logger.exception(
                    f"Failed to auto-create worker '{decoded_worker_name}' during update"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Worker not found and auto-creation failed: {create_error!s}",
                ) from create_error
        else:
            # Validate that the path worker_name matches the request worker_name
            if request.worker_name != decoded_worker_name:
                raise HTTPException(
                    status_code=422,
                    detail=f"Worker name in path '{decoded_worker_name}' must match worker_name in request body '{request.worker_name}'",
                )

            # Keep the existing worker_id
            existing_worker_id = existing_worker.get("worker_id")
            effective_base_config = existing_worker.get("config")

        # Normalize worker name to ensure consistency
        normalized_worker_name = request.worker_name.lower().replace(" ", "_")

        # Only the keys this request asserts (see _config_payload), plus the
        # view of the config the row will actually hold once they are merged in.
        config_to_update = await _validated_config_payload(
            request, lambda: _already(effective_base_config)
        )

        # This call is here for ``worker.db_path`` and nothing else.
        #
        # ⚠️ Do NOT "improve" it by feeding it the merged (stored + requested)
        # config. That was tried and reverted: the premise is that a partial
        # request makes the factory validate a config the row will not hold, but
        # ``src/worker/factory/agent_factory.py:264-271`` catches ``ValueError``
        # AND bare ``Exception`` and returns a ``WorkerTemplate``, so the factory
        # validates nothing a caller can observe. The merged version cost an
        # extra mlx / inference-router resolve and a CLI-MCP resolution on every
        # PUT, bought no assertion any test could distinguish, and its comment
        # asserted a safety property that does not exist.
        #
        # ``db_path`` derives only from the name and ``disable_state_storage``
        # (``worker_factory.py:82-99, 322``), neither of which is merge-affected.
        worker = await create_worker(
            # The END tenant this worker is built for: inference-router model
            # validation lists as THIS tenant (no WM/platform tenant fallback).
            tenant_id=ctx.tenant_id,
            worker_name=normalized_worker_name,
            tool_names=request.tool_names,
            agent_role=request.agent_role if request.agent_role else None,
            agent_type=request.agent_type if request.agent_type else None,
            model_provider=request.model_provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            instruction=request.instruction,
            user_prompt=request.user_prompt,
            description=request.description,
            deps=request.deps,
            mcp_config=request.mcp_config,
            mcp_config_preset=request.mcp_config_preset,
            mcp_servers=request.mcp_servers,
            base_url_override=request.base_url_override,
            api_key_override=request.api_key_override,
            tools_dir_override=request.tools_dir_override,
            output_schema=request.output_schema,
            output_validator_code=request.output_validator,
            output_retries=request.output_retries,
            disable_state_storage=request.disable_state_storage,
        )

        # Update worker in database. This is reached in TWO ways, and merge is
        # right for both:
        #   * the ordinary update — the caller asserts what it owns;
        #   * the follow-through after the auto-create above, where the mlx
        #     defaults fill the fields the caller stayed silent about. That is
        #     what a default is for. Previously the caller's Nones erased them
        #     moments after they were written, so a PUT that auto-created a
        #     worker and omitted `model` stored NULL — the defaults computed
        #     just above never reached the row.
        #     (With `disable_state_storage=True` the auto-create branch writes
        #     NOTHING, so there are no defaults to preserve and this call is a
        #     plain create; the merge is a no-op over an empty row.)
        reg_result = await worker_registry.register_worker(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            tenant_name=ctx.tenant_name,
            worker_name=request.worker_name,  # Keep original name for user display
            worker_id=existing_worker_id,
            db_path=worker.db_path,
            is_agent=_is_agent_arg(request),
            config=config_to_update,
            status="updated",
            visibility=ctx.visibility,
            config_revision=request.config_revision,
        )
        persisted_config = reg_result.get("config") or {}
        persisted_is_agent = bool(reg_result.get("is_agent"))

        # Remove worker from cache to force recreation with new config
        try:
            await worker_cache.remove_worker(decoded_worker_name)
        except Exception as cache_error:
            logger.warning(
                f"Failed to remove worker {decoded_worker_name} from cache: {cache_error}"
            )

        # Save worker state for the update
        try:
            db = StateDatabase()
            await db.add_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                tenant_name=ctx.tenant_name,
                worker_id=existing_worker_id,
                worker_name=request.worker_name,  # Keep original name for user display
                is_agent=persisted_is_agent,
                state_type="updated",
                worker_state={"status": "updated"},
                visibility=ctx.visibility,
                metadata={"config": _audit_config(persisted_config)},
            )
        except Exception as state_error:
            logger.warning(f"Failed to save updated worker state: {state_error}")
            # Don't fail the worker update for state save issues

        return _worker_response(
            record={**reg_result, "config": persisted_config},
            ctx=ctx,
            worker_id=existing_worker_id,  # Include the existing worker UUID
            name=request.worker_name,  # Original name
            status="updated",
            db_path=worker.db_path,
            is_agent=persisted_is_agent,
            echo_overrides=False,
        )
    except APIError:
        raise  # 4xx envelopes are the answer, not a 500
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_msg = str(e)

        # Log the full traceback for debugging
        logger.exception(f"Error updating worker: {error_msg}")
        logger.exception(f"Full traceback: {traceback.format_exc()}")

        # Handle specific API key errors with appropriate status codes
        if "API key required" in error_msg or "OPENAI_API_KEY" in error_msg:
            raise HTTPException(
                status_code=422,
                detail="API key required for the specified model provider. Please provide an API key or set the appropriate environment variable.",
            ) from e
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid API credentials provided") from e

        # Generic server error for other cases
        raise HTTPException(status_code=500, detail=f"Failed to update worker: {error_msg}") from e


@router.delete(
    "/",
    status_code=204,
    responses={**{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 405, 500]}},
    operation_id="delete_all_workers",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_DELETE_ALL))],
)
async def delete_all_workers(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> None:
    """Delete all workers and their associated states for a specific tenant.

    Completely removes all workers for the specified tenant including:
    1. All state history records for all workers
    2. All worker instances from cache (with cleanup)
    3. All worker registrations from database

    This operation is scoped to the (tenant_id, subject_id) and does not affect other tenants' or other users' workers.
    This operation is irreversible. All workers would need to be recreated
    to use them again.

    Args:
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        None (204 No Content on success)

    Raises:
        HTTPException: 500 if deletion fails

    Use cases:
    - Clearing all workers to reset the system for testing
    - System maintenance or cleanup
    - Starting fresh with a clean slate

    Warning:
        This operation is irreversible. All workers and their state history
        for this tenant will be permanently lost.

    Note:
        The operation is designed to be resilient - it will attempt to clean up
        all resources even if some steps fail. For example, if state deletion
        fails for some workers, it will still proceed with worker removal.
    """
    try:
        # Get all workers before deletion to clean up their instances and states
        all_workers = await worker_registry.get_all_workers(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id
        )

        total_states_deleted = 0
        cache_cleanup_errors = 0

        # Clean up worker states and cache instances for each worker
        for worker_data in all_workers:
            worker_name = worker_data["worker_name"]
            worker_id = worker_data.get("worker_id", worker_name)

            # Clean up worker states
            try:
                db = StateDatabase()
                states_deleted = await db.delete_worker_state(
                    tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id
                )
                total_states_deleted += states_deleted
                logger.info(f"Deleted {states_deleted} state records for worker {worker_name}")
            except Exception as state_error:
                logger.warning(f"Failed to delete worker states for {worker_name}: {state_error}")
                # Continue with other workers even if state cleanup fails

            # Remove worker from cache
            try:
                await worker_cache.remove_worker(worker_name)
            except Exception as cache_error:
                cache_cleanup_errors += 1
                logger.warning(f"Failed to remove worker {worker_name} from cache: {cache_error}")
                # Continue with other workers even if cache cleanup fails

        # Delete all workers from registry
        workers_deleted = await worker_registry.delete_all_workers(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id
        )

        logger.info(
            f"Deleted all workers for tenant {ctx.tenant_id}: {workers_deleted} workers, {total_states_deleted} total state records"
        )
        if cache_cleanup_errors > 0:
            logger.warning(f"Cache cleanup failed for {cache_cleanup_errors} workers")

        return  # 204 No Content

    except Exception as e:
        logger.exception("Error deleting all workers")
        raise HTTPException(status_code=500, detail=f"Failed to delete all workers: {e!s}") from e


@router.delete(
    "/{worker_name}",
    status_code=204,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 422, 500]}
    },
    operation_id="delete_worker",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_DELETE))],
)
async def delete_worker(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    worker_name: str = PathParam(
        ...,
        description="Name of the worker to delete",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
) -> None:
    """Delete a specific worker and all associated states for a specific tenant.

    Completely removes a worker from the system including:
    1. All state history records
    2. Worker instance from cache (with cleanup)
    3. Worker registration from database

    This operation is irreversible. The worker would need to be recreated
    to use it again.

    Args:
        worker_name: Name of the worker to delete.
                    Must be 1-100 characters.
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        None (204 No Content on success)

    Raises:
        HTTPException: 422 if worker name is empty
        HTTPException: 404 if worker not found
        HTTPException: 500 if deletion fails

    Note:
        The operation is designed to be resilient - it will attempt to clean up
        all resources even if some steps fail. For example, if state deletion
        fails, it will still proceed with worker removal.
    """
    try:
        # Validate worker_name input
        if not worker_name:
            raise HTTPException(status_code=422, detail="Worker name cannot be empty")

        # Allow control characters but still validate basic constraints
        if len(worker_name) == 0:
            raise HTTPException(status_code=422, detail="Worker name cannot be empty")

        # URL decode the worker name to handle special characters
        worker_name = unquote(worker_name)

        # Get worker data to retrieve worker_id for state cleanup
        worker_data = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
        )
        if not worker_data:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker_id = worker_data.get("worker_id", worker_name)

        # Clean up worker states first
        states_deleted = 0
        try:
            db = StateDatabase()
            states_deleted = await db.delete_worker_state(
                tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id
            )
            logger.info(f"Deleted {states_deleted} state records for worker {worker_name}")
        except Exception as state_error:
            logger.warning(f"Failed to delete worker states for {worker_name}: {state_error}")
            # Continue with worker deletion even if state cleanup fails

        # Remove worker from cache
        try:
            await worker_cache.remove_worker(worker_name)
        except Exception as cache_error:
            logger.warning(f"Failed to remove worker {worker_name} from cache: {cache_error}")

        # Delete worker from registry
        try:
            deleted = await worker_registry.delete_worker(
                tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
            )
            if deleted:
                return  # 204 No Content - no response body
            else:
                # Worker was already not in registry, but we cleaned up states
                # This is still considered successful
                return  # 204 No Content
        except Exception:
            logger.exception(f"Registry deletion error for {worker_name}")
            # If worker was found initially but registry deletion fails,
            # we've still cleaned up states, so return success
            return  # 204 No Content

    except APIError:
        raise  # 4xx envelopes are the answer, not a 500
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception(f"Error deleting worker {worker_name}")
        raise HTTPException(status_code=500, detail=f"Failed to delete worker: {e!s}") from e


@router.put(
    "/{worker_name}/prompt",
    response_model=PromptSwapResponse,
    operation_id="swap_worker_prompt",
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [400, 401, 404, 409, 422, 500]}
    },
    summary="Atomically swap a worker's system prompt (P-MP-01 schema-immutable, OD-4 CAS)",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_PROMPT_SWAP))],
)
async def swap_worker_prompt(
    worker_name: Annotated[str, PathParam(description="Base worker whose prompt to swap")],
    request: PromptSwapRequest,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> PromptSwapResponse:
    """Replace ONLY the worker's ``system_prompt`` (the AA prompt-improvement DONE apply path).

    Unlike ``PUT /workers/{name}`` this NEVER auto-creates (a swap target must
    pre-exist → 404), changes only the prompt body (``output_schema`` is
    structurally immutable, P-MP-01), and supports an optional compare-and-swap
    on ``expected_prompt_sha256`` (409 on mismatch / concurrent change, OD-4).
    """
    try:
        decoded = unquote(worker_name)
        try:
            validate_identifier(decoded, "worker_name")
        except SecurityValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # 1. The base must already exist (swap NEVER auto-creates).
        existing = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=decoded
        )
        if not existing:
            raise HTTPException(status_code=404, detail=f"Worker '{decoded}' not found")

        stored_name = existing["worker_name"]
        current_config = existing.get("config") or {}
        current_prompt = current_config.get("system_prompt") or ""

        # 1b. P4 A3 — what text goes in, and what config_revision truthfully
        #     names it. ``prompt_version`` loads the ledger body (PUBLISHED or
        #     CANARY; else the A4 409 body, missing → the pin 404 body) and labels the worker
        #     exactly ``<id>@<N>``. A free-text swap cannot name a ledger version,
        #     so with PROMPTS_READ_REQUIRES_PUBLISHED on it is labelled
        #     ``unversioned:<sha12>``; flag off keeps today's label
        #     (``canary_version``, or unchanged when absent).
        new_prompt: str
        new_revision: str | None
        if request.prompt_version is not None:
            # ⛔ Tenancy of prompt versions (owner ruling 2026-09-30): every
            # prompt_versions row is a SYSTEM asset (no tenant column), so only a
            # platform tenant may copy one into a worker. A customer swap naming a
            # ledger version would put system prompt text into its own worker —
            # 403 before the ledger is read. Free-text swaps are unaffected.
            if not is_platform_tenant(ctx.tenant_id):
                raise HTTPException(status_code=403, detail={"error": "platform_prompt_forbidden"})
            pv_id, pv_version = parse_prompt_revision(request.prompt_version)
            if pv_id is None or pv_version is None:
                raise HTTPException(status_code=422, detail="prompt_version must be <id>@<N>")
            # Missing → 404 / unrunnable → 409, both the shared top-level pin body.
            row = await load_runnable_version(pv_id, pv_version)
            new_prompt = str(row["system"])
            new_revision = format_prompt_revision(pv_id, pv_version)
        else:
            new_prompt = request.new_system_prompt or ""
            new_revision = (
                unversioned_revision(_sha256(new_prompt))
                if get_settings().prompts_read_requires_published
                else request.canary_version
            )

        # 2. P-MP-01: output_schema is immutable. The swap never writes it, but if
        #    the caller asserts an expected schema it must equal the live one.
        if request.expected_output_schema is not None:
            live_schema = existing.get("output_schema")
            if live_schema is None:
                live_schema = current_config.get("output_schema")
            if request.expected_output_schema != (live_schema or {}):
                raise HTTPException(
                    status_code=422,
                    detail="output_schema is immutable; only the prompt body may change (P-MP-01)",
                )

        # 3. Compare-and-swap on the prompt sha (OD-4) — surfaced as 409.
        cas_predicate: str | None = None
        if request.expected_prompt_sha256 is not None:
            if _sha256(current_prompt) != request.expected_prompt_sha256:
                raise HTTPException(
                    status_code=409,
                    detail="Live prompt sha256 does not match expected_prompt_sha256 "
                    "(the base changed since the canary was cut)",
                )
            cas_predicate = current_prompt  # atomic guard against concurrent change

        # 4. Atomic swap (output_schema column never touched).
        updated = await worker_registry.swap_prompt(
            tenant_id=ctx.tenant_id,
            subject_id=ctx.subject_id,
            worker_name=stored_name,
            new_system_prompt=new_prompt,
            expected_current_prompt=cas_predicate,
            config_revision=new_revision,
        )
        if updated is None:
            # CAS predicate set but 0 rows → concurrent change between read and update.
            raise HTTPException(
                status_code=409,
                detail="Prompt changed concurrently during the swap; retry with the current sha",
            )

        # 5. Evict the cached worker instance so the next message uses the new prompt.
        try:
            await worker_cache.remove_worker(stored_name.lower())
        except Exception as cache_error:
            logger.warning("Prompt swap: cache evict failed for %s: %s", stored_name, cache_error)

        # 6. Durable audit trail.
        try:
            await StateDatabase().add_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                tenant_name=ctx.tenant_name,
                worker_id=str(updated["worker_id"]),
                worker_name=stored_name,
                is_agent=bool(updated.get("is_agent")),
                state_type="prompt-swapped",
                worker_state={"status": "prompt-swapped"},
                visibility=ctx.visibility,
                metadata={
                    "canary_version": request.canary_version,
                    "prompt_sha256": _sha256(new_prompt),
                    "prev_prompt_sha256": _sha256(current_prompt),
                    # Only a versioned swap adds this key; a free-text swap's
                    # audit row stays byte-identical to before.
                    **(
                        {"prompt_version": new_revision}
                        if request.prompt_version is not None
                        else {}
                    ),
                    "reason": request.reason,
                },
            )
        except Exception as state_error:
            logger.warning("Prompt swap: state audit failed for %s: %s", stored_name, state_error)

        return PromptSwapResponse(
            swapped=True,
            worker_name=stored_name,
            output_schema_preserved=True,
            canary_version=request.canary_version,
            prompt_sha256=_sha256(new_prompt),
        )
    except APIError:
        raise  # 4xx envelopes are the answer, not a 500
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error swapping prompt for worker {worker_name}")
        raise HTTPException(status_code=500, detail=f"Failed to swap prompt: {e!s}") from e
