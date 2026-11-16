"""Message sending endpoint for worker agents."""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Annotated, Any
from uuid import uuid7

from fastapi import Depends, Header, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel

from src.api.auth_middleware import _lookup_tenant_byo_mode
from src.api.common_responses import STANDARD_ERROR_MODELS
from src.api.error_handlers import BadRequestError, InternalServerError, NotFoundError
from src.api.identity_context import (
    apply_body_correlation_id,
    set_prompt_attribution,
    set_provider_pin_context,
    set_replay_mode_context,
)
from src.api.prompts.repository import CANARY
from src.api.security_validation import SecurityValidationError, validate_identifier
from src.api.workers._globals import state_manager, worker_cache, worker_registry
from src.api.workers._helpers import (
    _PASS_THROUGH_UPSTREAM_STATUSES,
    IR_QUEUE_REJECTED,
    _backoff_delay,
    _describe_connection_failure,
    _ir_error_meta,
    _is_connection_error,
    _is_deterministic_failure,
    _is_queue_rejected,
    _upstream_status_code,
)
from src.api.workers._router import router
from src.api.workers.models import MessageRequest, MessageResponse
from src.api.workers.prompt_attribution import (
    CANARY_SELECTED_BY_PIN,
    PROMPT_SOURCE_CANARY,
    PROMPT_SOURCE_LEGACY,
    PROMPT_SOURCE_PIN,
    PROMPT_SOURCE_PUBLISHED,
    PROMPT_SOURCE_REQUEST,
    PromptSelection,
    prompt_attribution,
    response_metadata,
    selection_attribution,
)
from src.api.workers.prompt_pin import (
    PIN_CANARY_HELD,
    PIN_TENANT_FORBIDDEN,
    PinVerdict,
    PromptPin,
    apply_pin_verdict,
    apply_prompt_pin,
    canary_held,
    pin_forbidden,
    pin_verdict,
    resolve_prompt_pin,
    unversioned_revision,
)
from src.api.workers.prompt_source import (
    ResolvedPrompt,
    apply_resolved_prompt,
    resolve_worker_prompt,
)
from src.api.workers.system_worker_contracts import (
    CONTRACT_BARE,
    CONTRACT_DRIFTED,
    registered_output_schema,
    worker_contract_verdict,
)
from src.auth import TenantContext, get_tenant_context
from src.auth.tenant_proof import (
    NOT_A_COMMERCIAL_PLATFORM_TENANT,
    owner_commercial_keys_refusal,
)
from src.auth.tokenless_gate import ROUTE_WORKERS_MESSAGE, tokenless_tenant_gate
from src.db.workers_state_db import StateDatabase
from src.security.redact import redact_secrets_in_text
from src.security.request_overrides import (
    OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
    stored_validator_code,
)
from src.services.mlx_model_registry import mlx_model_registry
from src.settings import get_settings
from src.telemetry.agent_metrics import (
    WORKER_STATE_WRITE_VALIDATED_BY_CONSTRUCTION,
    WORKER_STATE_WRITE_VALIDATED_FALSE,
    WORKER_STATE_WRITE_VALIDATED_TRUE,
    get_metrics_client,
)
from src.utils.log_bounds import summarize_for_log
from src.utils.xgrammar_schema import SCHEMA_BOUND_VIOLATION, SchemaBoundViolationError
from src.worker.agents.base_agent import SENT_USAGE_METRICS_ATTR
from src.worker.agents.provider_catalog import is_on_hardware_ir_pin
from src.worker.factory.llm_config_resolver import BYORequiredError, TenantLLMKeyUnusableError
from src.worker.worker_factory import create_worker

logger = logging.getLogger(__name__)


class _TypedHTTPException(HTTPException):
    """An HTTPException whose ``error_code`` the handler renders as ``error.code``."""

    def __init__(
        self,
        error_code: str,
        *,
        status_code: int,
        detail: str,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        if error_details:
            self.error_details = error_details


class _SentUsageHTTPException(HTTPException):
    """An HTTPException the handler renders with ``error.details`` (additive)."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        error_details: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_details = error_details


def _sent_usage_details(error: BaseException) -> dict[str, Any] | None:
    """``{"usage_metrics": ...}`` for a failed run that recorded what it SENT.

    A failed pydantic_ai run carries the read-back of the sampling keys it put on
    the wire (temperature / seed / max_tokens / penalties / correlation_id) —
    see ``PydanticAIAgent._stamp_sent_usage_metrics``. Without this, a 503 from
    inference-router left the caller with null for values it demonstrably sent.
    None when the exception carries nothing (non-pydantic worker, pre-run failure).
    """
    sent = getattr(error, SENT_USAGE_METRICS_ATTR, None)
    return {"usage_metrics": sent} if isinstance(sent, dict) else None


def _override_cache_fingerprint(
    request: MessageRequest, prompt_pin: str | None = None
) -> str | None:
    """Fingerprint the per-message overrides baked into the agent, for the cache key.

    Covers ``output_schema`` / ``output_validator`` / ``model_override`` /
    ``max_tokens`` / ``enable_reasoning`` / ``system_prompt`` / ``instruction``.
    Folded into the worker-cache key so a request carrying an override never poisons
    the shared instance a later NON-override caller reuses — the cross-caller
    schema/model/budget/reasoning/prompt leak.
    ``enable_reasoning`` is load-bearing here in BOTH directions: without it a caller
    that opts thinking OUT bakes that into the shared agent for a later caller that
    asked for nothing, and vice versa. The two prompt fields are load-bearing for the
    same reason and more sharply: ``instructions=`` is baked ONCE at ``Agent(...)``
    construction, so without them in the key one caller's system prompt would be
    served to every later caller on the same worker until the slot was recreated.
    ``work_tier`` is excluded: it already joins the cache key via
    ``config['work_tier']``. Returns ``None`` when the request carries none of the
    seven overrides, so the base key is unchanged (backward compat).

    The prompts are embedded in the hashed material rather than pre-hashed
    individually, matching how ``output_schema`` (also unbounded in size) is already
    handled: the digest is truncated to 16 hex chars, so a 1M-char prompt costs the
    same key length as an absent one.

    ``prompt_pin`` is the ``<prompt_id>@<N>`` a per-call prompt-version pin ran
    (``prompt_pin.py``). It is added to the material ONLY when set, so an
    un-pinned request hashes exactly the seven keys it always did and keeps its
    slot; a pinned one lands in its own slot. Ledger versions are immutable, so
    the revision identifies the content.
    """
    material = {
        "output_schema": request.output_schema,
        "output_validator": request.output_validator,
        "model_override": request.model_override,
        "max_tokens": request.max_tokens,
        "enable_reasoning": request.enable_reasoning,
        "system_prompt": request.system_prompt,
        "instruction": request.instruction,
    }
    if prompt_pin is not None:
        material["prompt_pin"] = prompt_pin
    if all(v is None for v in material.values()):
        return None
    blob = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ⛔ ``seed`` is deliberately NOT in the fingerprint above. A per-message seed is
# passed per RUN (``worker.run(..., seed=)`` → ``agent.run(model_settings=)``),
# so it never needs its own cached agent. Keying on it would mint one cache slot
# — each holding its own connection pool, never evicted — per distinct seed, and
# seed sweeps (judges run 0..4) would multiply slots for no behavioural reason.

SEED_NOT_SUPPORTED = "SEED_NOT_SUPPORTED"
SAMPLING_OVERRIDE_NOT_SUPPORTED = "SAMPLING_OVERRIDE_NOT_SUPPORTED"


class _RefusedAfterProcessing(_TypedHTTPException):
    """A 4xx raised AFTER the worker row was moved to ``processing``.

    The ``except HTTPException`` handler writes the terminal ``error`` row only
    for 5xx, so an ordinary 4xx from here would leave the worker stuck in
    ``processing`` forever. This type opts the refusal into that same error-row
    write while keeping its 4xx status.
    """


BYO_PROVIDER_CALL_FAILED = "BYO_PROVIDER_CALL_FAILED"
# A vendor error body can be long (whole HTML error pages have been seen); the
# customer-facing detail is capped so one failure cannot flood a response or a
# worker_states row.
_BYO_DETAIL_MAX_CHARS = 600


def ran_on_tenant_key(ran_instance: Any) -> bool:
    """True when the instance that ran was built on the tenant's OWN resolved key.

    ``is True``, not truthiness: the flag is a bool on every real config, and a
    truthy non-bool (a mock attribute) must not turn a platform failure into a
    tenant-key one.
    """
    return getattr(getattr(ran_instance, "config", None), "tenant_byo_key", False) is True


def sanitize_byo_error(ran_instance: Any, text: str) -> str:
    """Upstream error text with the tenant's key removed, and bounded.

    The exact key (``config.api_key`` / ``api_key_override``) is replaced, any
    ``key=`` URL parameter is masked, and the text is capped.
    """
    config = getattr(ran_instance, "config", None)
    secrets = [getattr(config, attr, None) for attr in ("api_key", "api_key_override")]
    text = redact_secrets_in_text(text, *(s for s in secrets if isinstance(s, str)))
    if len(text) > _BYO_DETAIL_MAX_CHARS:
        text = text[:_BYO_DETAIL_MAX_CHARS] + "...(truncated)"
    return text


def byo_failure_detail(ran_instance: Any, tenant_id: Any, detail: str) -> str:
    """Prefix a failed run's error with the tenant's BYO provider + the fix.

    Only when the instance that ran was built on the tenant's own resolved key
    (``config.tenant_byo_key``); any other run's detail is returned unchanged.
    The upstream text is sanitized (:func:`sanitize_byo_error`).
    """
    if not ran_on_tenant_key(ran_instance):
        return detail
    config = getattr(ran_instance, "config", None)
    provider = getattr(config, "model_provider", None) or "the configured provider"
    model = getattr(config, "model", None) or ""
    return (
        f"The call to your own {provider} key (tenant {tenant_id}, model {model!r}) "
        f"failed: {sanitize_byo_error(ran_instance, detail)}. Nothing was run on a "
        f"platform model instead. Fix: check the {provider} API key, its quota and "
        "the model name in the tenant LLM settings (POST /tenant-config/llm/test "
        "tests the key)."
    )


def byo_failure_http(
    ran_instance: Any, tenant_id: Any, error: BaseException, upstream_status: int | None
) -> _TypedHTTPException:
    """WM-1: a vendor rejection on the tenant's OWN key, as the typed envelope.

    502 ``BYO_PROVIDER_CALL_FAILED`` — the same code the /providers and /chat
    paths return for the same failure — with the sanitized upstream text, never
    500 ``INTERNAL_ERROR`` with raw ``str(e)``. 502 keeps every caller retry
    gate that keys on ``>= 500`` behaving exactly as the old 500 did.
    """
    config = getattr(ran_instance, "config", None)
    return _TypedHTTPException(
        BYO_PROVIDER_CALL_FAILED,
        status_code=502,
        detail=byo_failure_detail(ran_instance, tenant_id, str(error)),
        error_details={
            "tenant_id": str(tenant_id) if tenant_id else None,
            "provider": getattr(config, "model_provider", None),
            "model": getattr(config, "model", None),
            "upstream_status": upstream_status,
        },
    )


def _seed_supported(worker_data: dict[str, Any]) -> bool:
    """Whether the worker this message will build is a pydantic-ai agent.

    Only ``PydanticAIAgent.run`` accepts a per-run seed. Mirrors agent_factory:
    no ``agent_role`` ⇒ basic worker (no LLM), and a missing ``agent_type``
    defaults to ``pydantic_ai``.
    """
    config = worker_data.get("config") or {}
    if not isinstance(config, dict) or not config.get("agent_role"):
        return False
    return (config.get("agent_type") or "pydantic_ai") == "pydantic_ai"


def _run_sampling_kwargs(request: MessageRequest) -> dict[str, Any]:
    """The per-RUN sampling overrides this message carries, only the ones it set.

    Same contract for all four: passed to ``worker.run`` (never folded into the
    config nor the cache key), winning over the worker record's value, absent ⇒
    not sent. 0 is a real seed (and 0.0 a real temperature) and must survive,
    hence ``is not None``.
    """
    candidates = {
        "seed": request.seed,
        "repetition_penalty": request.repetition_penalty,
        "frequency_penalty": request.frequency_penalty,
        "temperature": request.temperature,
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _unsupported_run_override_code(run_kwargs: dict[str, Any]) -> str:
    """Return the refusal code for unsupported per-run overrides.

    ``SEED_NOT_SUPPORTED`` when a seed is among them (its established code), else
    ``SAMPLING_OVERRIDE_NOT_SUPPORTED``.
    """
    return SEED_NOT_SUPPORTED if "seed" in run_kwargs else SAMPLING_OVERRIDE_NOT_SUPPORTED


def _effective_enforce_schema_bounds(
    request: MessageRequest, worker_data: dict[str, Any] | None
) -> bool:
    """Whether THIS run fails on a count/length bound violation (opt-in, default off).

    Same precedence as seed: the per-message value wins; None ⇒ the worker
    record's ``config.enforce_schema_bounds``; absent ⇒ False (advisory).
    """
    if request.enforce_schema_bounds is not None:
        return request.enforce_schema_bounds
    config = worker_data.get("config") if isinstance(worker_data, dict) else None
    return isinstance(config, dict) and config.get("enforce_schema_bounds") is True


async def _run_worker(
    worker: Any,
    request: MessageRequest,
    worker_name: str,
    *,
    enforce_schema_bounds: bool = False,
) -> Any:
    """``worker.run`` with the per-message sampling overrides, when any were sent.

    Unset ⇒ exactly ``worker.run(message)``, as before. The ``accepts_run_seed``
    check is the backstop for :func:`_seed_supported`: a worker that cannot take
    the kwargs is refused with the same typed 422, never a TypeError 500.

    ``enforce_schema_bounds`` is only forwarded when True, and only to a worker
    whose ``run`` takes per-run kwargs (pydantic_ai — the only agent type with a
    schema validator to enforce it in).
    """
    run_kwargs = _run_sampling_kwargs(request)
    if enforce_schema_bounds and getattr(worker, "accepts_run_seed", False) is True:
        run_kwargs["enforce_schema_bounds"] = True
    if not run_kwargs:
        return await worker.run(request.message)
    if getattr(worker, "accepts_run_seed", False) is not True:
        # Only reachable if ``_seed_supported`` (the pre-check, which refuses
        # BEFORE the processing row) has drifted from the agent factory. By now
        # the worker is ``processing``, so this must end in a terminal row.
        code = _unsupported_run_override_code(run_kwargs)
        raise _RefusedAfterProcessing(
            code,
            status_code=422,
            detail=f"{code}: worker '{worker_name}' is not a pydantic_ai worker",
        )
    return await worker.run(request.message, **run_kwargs)


def apply_per_message_output_override(
    worker_data: dict[str, Any],
    output_schema: dict[str, Any],
    output_validator: str | None = None,
) -> None:
    """Apply a per-message ``output_schema`` where the agent builder will actually read it.

    Writes BOTH the config dict and the dedicated top-level keys, and the second
    half is the whole point. ``_globals.get_or_create_worker`` prefers
    ``worker_data["output_schema"]`` and falls back to ``config["output_schema"]``
    only when the top-level value is ``None`` — a column-first rule that exists
    deliberately, because registry-backed workers persist the schema to a
    dedicated DB column rather than inline.

    Writing only the config dict therefore left the per-message override
    **silently discarded** for every worker whose column is populated: the cache
    was still evicted, the agent still rebuilt — from the *stored* schema — and
    the caller still got a 200. Nothing anywhere surfaced that the schema it sent
    had been ignored. Fresh/auto-created workers escaped this only because
    ``register_worker`` leaves that column NULL.

    Mutates in place and does not persist: ``worker_data`` is a per-request dict,
    so this is an override for THIS message, never a registration change.
    """
    config = worker_data.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config["output_schema"] = output_schema
    if output_validator:
        config["output_validator"] = output_validator
    worker_data["config"] = config

    worker_data["output_schema"] = output_schema
    if output_validator:
        worker_data["output_validator"] = output_validator


def apply_registered_system_schema(worker_data: dict[str, Any], worker_name: str) -> bool:
    """Restore a system improvement worker's REGISTERED schema when its row carries none.

    See ``system_worker_contracts`` for the measurement. Short version: the
    registry is keyed ``(tenant_id, subject_id, worker_name)``, the improvement
    loop's caller started authenticating as a different tenant on 2026-08-27,
    the lookup missed, and the message endpoint auto-created a bare row with
    ``output_schema=None``. Every run since answered freeform against a contract
    that exists, is registered, and was never applied.

    Applies ONLY when the resolved row asserts no schema in either place the
    agent builder reads (the dedicated top-level key and ``config``), so an
    explicitly-configured schema — or a per-message override, which is applied
    before this and writes both — always wins. Returns whether it fired.

    ⛔ Not persisted. This is a per-request repair of one field, deliberately
    not a write-back: the correct durable fix is to re-seed the row under the
    calling tenant (``scripts/improvement_workers/register.py``), and a handler
    that silently rewrote registry rows would hide that the seeding is stale.
    """
    config = worker_data.get("config")
    config = config if isinstance(config, dict) else {}
    if worker_data.get("output_schema") is not None or config.get("output_schema") is not None:
        return False

    schema = registered_output_schema(worker_name)
    if not schema:
        return False

    apply_per_message_output_override(worker_data, schema)
    # WARNING, not INFO: firing at all means the row this tenant resolves to is
    # missing its registered contract. The schema is now enforced for THIS call,
    # but the row is still bare (and, on an auto-created row, still missing its
    # system_prompt / model_provider too) until it is re-seeded.
    logger.warning(
        "worker '%s' resolved to a row with no output_schema; applying the registered "
        "improvement-worker schema for this message. Re-seed the row for this tenant "
        "(scripts/improvement_workers/register.py) to fix it durably.",
        worker_name,
    )
    return True


def _message_for_state(message: Any) -> Any:
    """Coerce ``MessageRequest.message`` into a JSON-safe shape for state.

    Plain strings pass through. Multimodal content-block lists become a
    list of dicts via ``.model_dump()``; image data is preserved (the
    state DB row carries it as base64 text, capped by the schema). When
    images are large this can bloat conversation state — callers using
    multimodal should set ``disable_state_storage=True`` to skip the
    database write entirely.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return [block.model_dump() if isinstance(block, BaseModel) else block for block in message]
    return message


# Defaults for the auto-create fallback path (when sendWorkerMessage hits a
# worker that was never explicitly created). Historically hardcoded to
# "mlx"/"mlx", which stopped working once the dedicated MLX upstream went away.
# The provider is env-driven; the MODEL is the platform's single served-model-name,
# ``Settings.local_llm_model`` (decoupled from the underlying HF model id), read
# per request so there is exactly one place that names it.
AUTO_CREATE_MODEL_PROVIDER = os.environ.get("AUTO_CREATE_MODEL_PROVIDER", "vllm")


def _auto_create_model() -> str:
    """Model for an auto-created worker: the platform's one served-model-name."""
    return get_settings().local_llm_model


async def _pin_verdict_for_run(
    request: MessageRequest,
    worker_data: dict[str, Any],
    prompt_pin: PromptPin | None,
    ran_instance: Any,
    *,
    pin_held: bool = False,
    pin_tenant_forbidden: bool = False,
) -> PinVerdict:
    """A2/A3 verdict for a finished run, keyed on the digest of what RAN.

    The digest comes from :func:`prompt_attribution` — the same instance-first
    read the metadata echo uses — so the revision and ``system_prompt_sha256``
    in one response can never describe two different prompts.

    ``pin_held`` (U6): a CANARY pin that was held. The truthful revision is
    still decided against what ran; only ``prompt_pin`` says ``canary_held``.

    ``pin_tenant_forbidden`` (U6): a NON-platform tenant's stamp. Decided here
    WITHOUT :func:`pin_verdict`, whose label verification reads the ledger: a
    customer call never touches system data, so it claims no ledger version and
    reports ``unversioned:<sha12>`` of what ran.
    """
    ran_sha = prompt_attribution(worker_data, ran_instance)["system_prompt_sha256"]
    if pin_tenant_forbidden:
        return PinVerdict(
            PIN_TENANT_FORBIDDEN,
            unversioned_revision(ran_sha) if ran_sha is not None else None,
        )
    verdict = await pin_verdict(
        request.prompt_versions,
        worker_data,
        prompt_pin,
        ran_sha,
        message_override=request.system_prompt is not None,
    )
    if pin_held:
        return PinVerdict(PIN_CANARY_HELD, verdict.revision)
    return verdict


def _select_prompt(
    request: MessageRequest,
    prompt_pin: PromptPin | None,
    resolved: ResolvedPrompt | None,
) -> PromptSelection:
    """Which source chose this call's system prompt (U6 precedence, highest first).

    request.system_prompt > pin (CANARY reported apart) > resolved PUBLISHED >
    the worker's config. Mirrors the order the blocks in :func:`send_message`
    apply them in; a ledger revision is reported only for a ledger-backed source.
    """
    if request.system_prompt is not None:
        return PromptSelection(PROMPT_SOURCE_REQUEST)
    if prompt_pin is not None:
        if prompt_pin.status == CANARY:
            return PromptSelection(
                PROMPT_SOURCE_CANARY, prompt_pin.revision, canary_selected_by=CANARY_SELECTED_BY_PIN
            )
        return PromptSelection(PROMPT_SOURCE_PIN, prompt_pin.revision)
    if resolved is not None:
        return PromptSelection(PROMPT_SOURCE_PUBLISHED, resolved.revision)
    return PromptSelection(PROMPT_SOURCE_LEGACY)


def _state_metadata(
    request: MessageRequest, selection_meta: dict[str, Any], verdict: PinVerdict
) -> dict[str, Any]:
    """``metadata`` for a completed ``worker_states`` row (U6: always attributed).

    ``context`` keeps its previous shape and is omitted when absent, exactly as
    before; the selection keys are ALWAYS present (``prompt_version_id`` null on
    a legacy call), so an improvement query can group rows by the version that
    ran without parsing the response. ``prompt_pin`` only when the call had a
    verdict to report (``canary_held`` and ``tenant_forbidden`` included — those
    are the counted events).
    """
    meta: dict[str, Any] = {"context": request.context} if request.context else {}
    meta.update(selection_meta)
    if verdict.prompt_pin is not None:
        meta["prompt_pin"] = verdict.prompt_pin
    return meta


@router.post(
    "/{worker_name}/message",
    response_model=MessageResponse,
    responses={
        **{k: v for k, v in STANDARD_ERROR_MODELS.items() if k in [401, 404, 405, 409, 422, 500]}
    },
    operation_id="send_worker_message",
    dependencies=[Depends(tokenless_tenant_gate(ROUTE_WORKERS_MESSAGE))],
)
async def send_message(  # noqa: PLR0912, PLR0915 -- send-message handler dispatches over worker type / output format / error class; refactor to per-route helpers is a separate API change
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    request: MessageRequest,
    worker_name: str = PathParam(
        ...,
        description="Name of the worker agent to send message to",
        examples=["test_worker"],
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    ),
    x_replay_mode: str | None = Header(default=None, alias="X-Replay-Mode"),
    x_provider: str | None = Header(default=None, alias="X-Provider"),
) -> MessageResponse:
    """Send a message to an AI agent worker and get its response.

    If the worker doesn't exist it's auto-created with default settings (temperature=0.1,
    provider/model from AUTO_CREATE_MODEL_PROVIDER / Settings.local_llm_model) and — this is the
    part that bites — with NO system prompt and NO instruction, then answers 200. A mistyped
    or renamed worker name is therefore a SILENT degradation, not an error. Send
    ``require_existing=true`` to get a 404 instead; the auto-create is logged at WARNING and
    counted in ``worker_management_worker_auto_created_total`` either way.
    Optional per-message output_schema overrides worker's default for dynamic structured output.

    Raises HTTPException for non-agent workers (400), an unknown worker under
    require_existing=true (404), validation errors (422), or processing failures (500).
    """
    # Path-param `worker_name` bypasses WorkerCreateRequest.validate_worker_name —
    # without this guard, a POST to /workers/undefined/message would auto-create
    # a row with worker_name="undefined" (orphan observed 2026-05-12, tenant
    # 00000000-0000-0000-0000-000000000001). Apply the same reserved-name +
    # identifier policy the body validator enforces.
    try:
        worker_name = validate_identifier(
            worker_name, "worker_name", max_length=100, allow_unicode=True
        )
    except SecurityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # WS-1 prompt-replay: when asset-evaluator drives a prompt replay through WM,
    # it sets X-Replay-Mode: true. Stash it in the per-request ContextVar so the
    # outbound inference-router httpx hook stamps X-Replay-Mode + X-Worker-Version
    # (alongside the already-forwarded X-Correlation-Id) on the completion call,
    # letting IR write a replay_invocation_log row. Always set (False when absent)
    # so a prior request's flag can never leak into a reused async context.
    set_replay_mode_context(bool(x_replay_mode) and x_replay_mode.strip().lower() == "true")
    # Body correlation_id → the ContextVar the outbound IR hook reads (and that
    # usage_metrics.correlation_id is stamped from).
    apply_body_correlation_id(request.correlation_id)
    # Caller's inference-router provider pin → forwarded to IR as X-Provider.
    # ⛔ PROVEN platform tenants only (owner rulings 2026-10-01 + 2026-10-03,
    # fr#1385): pinning IR to a specific upstream is how a commercial provider
    # behind IR would be reached on the owner's key, so it takes the same
    # predicate as the owner's keys — membership AND a request that proved the
    # tenant. An asserted / default platform tenant is treated like a customer.
    # Dropped (not refused) for everyone else — except that an UNPROVEN platform
    # tenant keeps a pin to an on-hardware upstream (pgx-local), which reaches no
    # commercial key. A customer's pin is dropped regardless (ruling 2026-10-01).
    pin_refusal = owner_commercial_keys_refusal(ctx.tenant_id)
    if pin_refusal is None or (
        pin_refusal != NOT_A_COMMERCIAL_PLATFORM_TENANT and is_on_hardware_ir_pin(x_provider)
    ):
        set_provider_pin_context(x_provider)
    else:
        set_provider_pin_context(None)
        if x_provider:
            logger.info(
                "send_message_provider_pin_dropped tenant=%s pin=%r reason=%s",
                ctx.tenant_id,
                x_provider,
                pin_refusal,
            )

    worker_data = None
    worker_auto_created = False
    # The exact agent instance this request last ran on, or None if it never got
    # as far as running one. The retry ladder passes it to ``remove_instance`` as
    # the compare-and-evict target so a failing run can only ever retire the
    # generation IT saw fail, never a replacement a sibling has already built and
    # is running on. Declared out here because the ladder lives in an ``except``
    # that can also be reached before the instance is ever assigned.
    ran_instance: Any = None
    # ``<prompt_id>@<N>`` when this call runs a per-call prompt-version pin (P4
    # §4) or a resolved PUBLISHED version (U6), else None. Out here for the same
    # reason as ``ran_instance``: the retry ladder must rebuild into the SAME
    # override slot the primary run used.
    prompt_pin_revision: str | None = None
    prompt_pin: PromptPin | None = None
    # U6: a CANARY pin held back by WM_PROMPT_CANARY_ROUTABLE=false, and which
    # source selected this call's system prompt (legacy until decided below).
    prompt_pin_held = False
    prompt_pin_forbidden = False
    prompt_selection = PromptSelection(PROMPT_SOURCE_LEGACY)
    try:
        # Get worker from registry
        worker_data = await worker_registry.get_worker(
            tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
        )

        # Auto-create worker if it doesn't exist
        if not worker_data:
            # ── The opt-out, and it must come FIRST ────────────────────────
            # A caller that owns a closed worker vocabulary can refuse the
            # auto-create per message. Checked before anything is minted so the
            # refusal is provably a NO-OP: no row, no registry write, no agent.
            # Default False — see MessageRequest.require_existing. The live
            # callers on this path (worker-doing-turn / -prose) never set it and
            # keep auto-creating exactly as before.
            if request.require_existing:
                logger.warning(
                    "Worker '%s' does not exist; refused to auto-create because the "
                    "caller set require_existing=true",
                    worker_name,
                    extra={
                        "worker_name": worker_name,
                        "tenant_id": str(ctx.tenant_id),
                        "tenant_name": ctx.tenant_name,
                        "require_existing": True,
                        "auto_created": False,
                    },
                )
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Worker '{worker_name}' does not exist and was NOT auto-created "
                        "because the caller set require_existing=true. Create the worker "
                        "explicitly (POST /workers or POST /workers/pydantic-ai), or check "
                        "the name for a typo or a rename."
                    ),
                )

            # ── DETECTION ─────────────────────────────────────────────────
            # WARNING, not INFO, and it says what is actually wrong: the row
            # about to be minted has NO system prompt and NO instruction (see
            # the create_worker call below, which passes both as None), so the
            # completion is unguided — and the caller still gets a 200. This was
            # an INFO line for the whole of its life, which is why a send to a
            # nonexistent worker has never been visible to anyone: the n8n
            # caller's error-reader is correct and structurally blind, because
            # the failure arrives as a success. See n8n_custom_nodes#1522.
            logger.warning(
                "Worker '%s' not found — AUTO-CREATING it WITHOUT a system prompt or "
                "instruction (model_provider=%s, model=%s). The send returns 200 from a "
                "promptless generic worker; if this name is a typo or a renamed worker, "
                "this is a SILENT failure. Set require_existing=true to get a 404 instead.",
                worker_name,
                AUTO_CREATE_MODEL_PROVIDER,
                _auto_create_model(),
                extra={
                    "worker_name": worker_name,
                    "tenant_id": str(ctx.tenant_id),
                    "tenant_name": ctx.tenant_name,
                    "auto_created": True,
                    "system_prompt": None,
                    "instruction": None,
                    "model_provider": AUTO_CREATE_MODEL_PROVIDER,
                    "model": _auto_create_model(),
                },
            )
            # Counted on the DECISION to auto-create, not on its success: a
            # create that then fails is still a name nobody had provisioned, and
            # booking it only on success would hide the noisiest half.
            get_metrics_client().record_worker_auto_created(worker_name)

            # Create worker with default settings
            default_agent_role = f"AI Assistant for {worker_name}"

            # Use per-message output_schema if provided; otherwise fall back to the
            # REGISTERED contract when this name is a seeded improvement worker.
            # Without that fallback this branch mints a contract-free row and
            # PERSISTS it — which is how improvement_workflow_development,
            # improvement_prompt_improvement and improvement_principle_improvement
            # all ended up running unconstrained under tenant aaaaaaaa from
            # 2026-08-27. See ``system_worker_contracts``.
            worker_output_schema = request.output_schema or registered_output_schema(worker_name)
            worker_output_validator = request.output_validator if request.output_validator else None

            try:
                # Generate worker ID
                auto_worker_id = str(uuid7())

                # Create the worker using create_worker factory function
                worker_instance = await create_worker(
                    # The END tenant this worker is built for: inference-router model
                    # validation lists as THIS tenant (no WM/platform tenant fallback).
                    tenant_id=ctx.tenant_id,
                    worker_name=worker_name,
                    tool_names=[],  # No tools by default
                    agent_role=default_agent_role,
                    agent_type="pydantic_ai",
                    model_provider=AUTO_CREATE_MODEL_PROVIDER,
                    model=_auto_create_model(),
                    temperature=0.1,
                    system_prompt=None,
                    instruction=None,
                    user_prompt=None,
                    description=f"Auto-created worker for {worker_name}",
                    deps=None,
                    mcp_config=None,
                    mcp_config_preset=None,
                    mcp_servers=None,
                    base_url_override=None,
                    api_key_override=None,
                    tools_dir_override=None,
                    output_schema=worker_output_schema,
                    output_validator_code=worker_output_validator,
                    disable_state_storage=request.disable_state_storage,
                )

                # Prepare configuration for registry
                auto_worker_config = {
                    "tool_names": [],
                    "agent_role": default_agent_role,
                    "agent_type": "pydantic_ai",
                    "model_provider": AUTO_CREATE_MODEL_PROVIDER,
                    "model": _auto_create_model(),
                    "temperature": 0.1,
                    "system_prompt": None,
                    "instruction": None,
                    "user_prompt": None,
                    "description": f"Auto-created worker for {worker_name}",
                    "deps": None,
                    "mcp_config": None,
                    "mcp_servers": None,
                    "base_url_override": None,
                    "api_key_override": None,
                    "tools_dir_override": None,
                    "output_schema": worker_output_schema,
                    "output_validator": worker_output_validator,
                }

                # Register worker in database only if state storage is enabled
                if not request.disable_state_storage:
                    await worker_registry.register_worker(
                        tenant_id=ctx.tenant_id,
                        subject_id=ctx.subject_id,
                        tenant_name=ctx.tenant_name,
                        worker_name=worker_name,
                        worker_id=auto_worker_id,
                        db_path=worker_instance.db_path,
                        is_agent=True,
                        config=auto_worker_config,
                        status="auto-created",
                        visibility=ctx.visibility,
                    )
                    logger.info(f"Successfully auto-created and registered worker '{worker_name}'")

                    # Fetch the newly created worker data from registry
                    worker_data = await worker_registry.get_worker(
                        tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_name=worker_name
                    )
                    if not worker_data:
                        raise HTTPException(
                            status_code=500,
                            detail="Worker was created but could not be retrieved from registry",
                        )
                else:
                    logger.info(
                        f"Successfully auto-created worker '{worker_name}' (state storage disabled, not registered)"
                    )
                    # Construct worker_data manually since we didn't register it
                    worker_data = {
                        "worker_name": worker_name,
                        "worker_id": auto_worker_id,
                        "db_path": worker_instance.db_path,
                        "is_agent": True,
                        "config": auto_worker_config,
                        "status": "auto-created",
                    }

                worker_auto_created = True

            except Exception as create_error:
                logger.exception(f"Failed to auto-create worker '{worker_name}'")
                raise HTTPException(
                    status_code=500,
                    detail=f"Worker not found and auto-creation failed: {create_error!s}",
                ) from create_error

        # Verify it's an agent (should always be true for auto-created workers)
        if not worker_data["is_agent"]:
            raise BadRequestError("Worker is not an agent")

        # Handle per-message model_override for MLX routing
        # If request has model_override, resolve the correct MLX instance URL
        if request.model_override:
            logger.info(
                f"Applying model_override '{request.model_override}' for worker '{worker_name}'"
            )
            # Validate model exists and resolve instance URL
            model_info = await mlx_model_registry.get_model(request.model_override)
            if not model_info:
                raise BadRequestError(
                    f"Model '{request.model_override}' not found in MLX registry. "
                    "Use GET /providers/mlx/models to list available models."
                )

            resolved_url = await mlx_model_registry.get_instance_for_model(request.model_override)
            if resolved_url:
                config = worker_data.get("config", {})
                if not isinstance(config, dict):
                    config = {}
                    worker_data["config"] = config
                config["base_url_override"] = resolved_url
                config["model"] = request.model_override
                worker_data["config"] = config
                logger.info(f"Routed model '{request.model_override}' to instance: {resolved_url}")
                # NO cache eviction here. ``model_override`` is part of
                # ``_override_cache_fingerprint``, so this request already resolves to
                # its OWN cache slot (``tenant:worker:tier[:product]:ov-<fp>``) and gets
                # a freshly built agent for the new model/URL without touching anyone
                # else's. See the eviction note above the retry ladder for why calling
                # remove_worker here was actively harmful.

        # ── DETECTION, and it must stay HERE ──────────────────────────────
        # Classify the row AS RESOLVED, before the per-message override below
        # and before ``apply_registered_system_schema`` further down. Both of
        # those mutate ``worker_data`` in memory, so a verdict taken after
        # either one would read a repaired copy and report health that the
        # stored row does not have — the detector would go blind exactly on the
        # population it exists to find.
        #
        # ⛔ Observe-only, and deliberately NOT a refusal: a bare row still runs.
        # Backfilling or rejecting is a decision about 14 tenants at once and is
        # not this code's to take. See ``system_worker_contracts``.
        _contract_verdict = worker_contract_verdict(worker_data, worker_name)
        get_metrics_client().record_worker_contract_verdict(worker_name, _contract_verdict)
        if _contract_verdict in (CONTRACT_BARE, CONTRACT_DRIFTED):
            # The identifying detail (tenant, subject) belongs here, where
            # cardinality costs nothing — never on the counter's labels.
            logger.warning(
                "worker '%s' resolved to a row with a %s output contract "
                "(tenant=%s subject=%s status=%s); the completion is not constrained by "
                "a registered contract. Re-seed the row for this tenant to fix it durably.",
                worker_name,
                _contract_verdict,
                ctx.tenant_id,
                ctx.subject_id,
                worker_data.get("status"),
            )

        # Handle per-message output_schema override
        # Apply if request has output_schema (for existing workers or when overriding auto-created defaults)
        if request.output_schema and not (worker_auto_created and not request.model_override):
            logger.info(f"Applying per-message output_schema override for worker '{worker_name}'")
            apply_per_message_output_override(
                worker_data, request.output_schema, request.output_validator
            )
            # NO cache eviction here — ``output_schema`` / ``output_validator`` are both
            # in ``_override_cache_fingerprint``, so this request keys into its own slot
            # and the agent is rebuilt with the new schema regardless.

        # Last resort for a SEEDED improvement worker whose resolved row carries no
        # schema (the cross-tenant miss + silent auto-create, measured 2026-09-11 —
        # see ``system_worker_contracts``). Runs AFTER the per-message override so a
        # caller-supplied schema always wins, and covers the rows that are ALREADY
        # bare — a fix confined to the auto-create branch above would be inert for
        # them, since ``get_worker`` now returns those rows and the branch never runs.
        # No eviction and no ``override_key``: the schema is the worker's registered
        # default, so it belongs in the base cache slot every caller shares.
        apply_registered_system_schema(worker_data, worker_name)

        # ⛔ FAIL CLOSED on a STORED validator. Request bodies carrying validator
        # code are already refused at parse time (422), but a row written before
        # that refusal could still hold code, and building the agent from it used
        # to ``exec()`` it. Refuse the message with the same typed code rather
        # than silently running without the validator the row asked for — a
        # caller relying on it must see that it is gone. Measured 2026-09-22:
        # 0 of 793 rows carry one, so this refuses nothing live.
        if stored_validator_code(worker_data):
            logger.error(
                "refusing message to worker '%s' (tenant=%s): stored output_validator "
                "code is not supported and is never executed",
                worker_name,
                ctx.tenant_id,
            )
            raise _TypedHTTPException(
                OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
                status_code=422,
                detail=(
                    f"{OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED}: worker '{worker_name}' has "
                    "stored output_validator code, which is never executed. Clear it "
                    "with PUT /workers/{name} (output_validator: null) and use "
                    "output_schema instead."
                ),
            )

        # Per-message work_tier override: fold it into the worker config so the
        # tenant's per-tier BYO LLM row is selected for THIS message. It also
        # joins the cache key (tenant:worker:tier), so distinct tiers get distinct
        # cached instances and a tier switch never reuses the wrong model.
        if request.work_tier:
            config = worker_data.get("config", {})
            if not isinstance(config, dict):
                config = {}
            config["work_tier"] = request.work_tier
            worker_data["config"] = config

        # Per-message max_tokens override: fold it into the worker config so the
        # generation budget applies for THIS message even when the runtime tenant's
        # worker row is bare auto-created (e.g. requirements_management forwards the
        # end-user tenant; max_tokens has no other path to that tenant's row). Like
        # the output_schema override it needs no eviction: ``max_tokens`` is in
        # ``_override_cache_fingerprint``, so a request carrying it lands in its own
        # cache slot and the agent there is built with the new budget.
        if request.max_tokens is not None:
            config = worker_data.get("config", {})
            if not isinstance(config, dict):
                config = {}
            if config.get("max_tokens") != request.max_tokens:
                config["max_tokens"] = request.max_tokens
                worker_data["config"] = config

        # Per-message enable_reasoning override: fold it into the worker config so a
        # caller can turn chain-of-thought OFF for THIS message. This is the only path
        # that exists for them: the auto-create branch above hardcodes
        # AUTO_CREATE_MODEL_PROVIDER (default "vllm"), so the call goes DIRECT to the
        # self-hosted backend and never touches inference-router — the only component
        # that otherwise pins enable_thinking=false. A thinking model handed a
        # structured-output schema spends its whole max_tokens budget inside <think>
        # and returns finish_reason=length with no valid answer. Like max_tokens it
        # needs no eviction: ``enable_reasoning`` is in ``_override_cache_fingerprint``,
        # so a request carrying it lands in its own cache slot.
        if request.enable_reasoning is not None:
            config = worker_data.get("config", {})
            if not isinstance(config, dict):
                config = {}
            if config.get("enable_reasoning") != request.enable_reasoning:
                config["enable_reasoning"] = request.enable_reasoning
                worker_data["config"] = config

        # Per-message seed: NOT folded into the config and NOT in the cache
        # fingerprint — it is passed per run (see ``_run_worker``). Refused up
        # front, before any state row is written, for a worker whose run cannot
        # take it.
        run_overrides = _run_sampling_kwargs(request)
        if run_overrides and not _seed_supported(worker_data):
            code = _unsupported_run_override_code(run_overrides)
            raise _TypedHTTPException(
                code,
                status_code=422,
                detail=(
                    f"{code}: {', '.join(sorted(run_overrides))} is only supported on "
                    f"pydantic_ai workers; '{worker_name}' is not one"
                ),
            )

        # Per-message system_prompt / instruction override: fold both into the worker
        # config so the agent built for THIS message carries the caller's prompts.
        # Both fields were DECLARED on MessageRequest, security-validated
        # (validate_prompts), capped at 1M and advertised in this endpoint's own
        # OpenAPI example — and then dropped: nothing here ever read
        # request.system_prompt / request.instruction, so a caller sending either got a
        # 200 carrying the worker RECORD's prompt, with nothing surfacing that the
        # override had been ignored (same silent-discard shape as the output_schema
        # column-vs-config bug above; 12 sentinel-bearing calls, 0 honoured).
        # This is the whole missing hop: ``_globals._create_worker_instance`` already
        # reads both config keys and hands them to create_worker, which joins them into
        # pydantic-ai's ``instructions=``. None ⇒ keep the record's value (the
        # max_tokens semantic) — hence a guard PER field, so overriding only
        # ``instruction`` cannot clobber the record's ``system_prompt`` to None.
        # Needs no eviction: both are in ``_override_cache_fingerprint``, so the request
        # lands in its own cache slot. That is REQUIRED here rather than a nicety —
        # ``instructions=`` is baked once at Agent construction, so a shared slot would
        # serve this caller's prompt to every later caller on the same worker.
        if request.system_prompt is not None or request.instruction is not None:
            config = worker_data.get("config", {})
            if not isinstance(config, dict):
                config = {}
            if request.system_prompt is not None:
                config["system_prompt"] = request.system_prompt
            if request.instruction is not None:
                config["instruction"] = request.instruction
            worker_data["config"] = config

        # Per-call prompt-version pin (P4 contract §4), DARK behind
        # PROMPTS_READ_REQUIRES_PUBLISHED. ``resolve_prompt_pin`` returns None —
        # and this block is a no-op — unless the flag is on, the request carries a
        # ``prompt_versions`` stamp, and that stamp names the prompt this worker's
        # config_revision is labelled with. Refused BEFORE the ``processing`` state
        # row is written, so a 409/404 (top-level ``raw_body``) leaves no
        # half-started run behind. A per-message system_prompt is never refused: it
        # wins, the pin is skipped, and the verdict says ``ignored:message_override``.
        # The pin rewrites only this request's ``worker_data`` and joins the cache
        # override key below: the worker's shared slot is never touched.
        #
        # U6 runtime prompt source. With WM_PROMPT_SOURCE=published a PLATFORM
        # worker bound in ``worker_prompt_binding`` runs the highest PUBLISHED
        # version of its prompt: it REPLACES config.system_prompt (which n8n
        # ensure-nodes keep rewriting — the second writer must not win) and
        # labels the request ``<prompt_id>@<N>``. Resolved BEFORE the pin, so a
        # pin still wins (precedence request > pin > published > config) and a
        # pin naming the bound prompt matches the new label. Never applied over
        # a per-message system_prompt; legacy flag / customer tenant / any
        # resolver failure → None, and this call is byte-identical to before.
        resolved_prompt: ResolvedPrompt | None = None
        if request.system_prompt is None:
            resolved_prompt = await resolve_worker_prompt(ctx.tenant_id, worker_name)
            if resolved_prompt is not None:
                apply_resolved_prompt(worker_data, resolved_prompt)
        # ⛔ TENANT GATE FIRST (owner ruling 2026-09-30): a NON-platform tenant's
        # stamp is never applied, PUBLISHED or CANARY, in either prompt source,
        # and is decided BEFORE ``resolve_prompt_pin`` so no ledger row is read
        # for that caller. Not a 403 — the worker runs what it would have run
        # without the stamp; the verdict says ``tenant_forbidden`` (counted).
        prompt_pin_forbidden = pin_forbidden(request.prompt_versions, ctx.tenant_id)
        prompt_pin = (
            None
            if prompt_pin_forbidden
            else await resolve_prompt_pin(
                request.prompt_versions,
                worker_data,
                request_system_prompt=request.system_prompt,
            )
        )
        # CANARY is never auto-selected and, under WM_PROMPT_SOURCE=published,
        # a CANARY pin runs only when WM_PROMPT_CANARY_ROUTABLE allows it. A held
        # pin is dropped here (counted inside ``canary_held``) and the resolved
        # PUBLISHED version — already folded in above — runs instead.
        # A NON-platform tenant's CANARY pin is held even when routable: a canary
        # is unreleased system data and is never run for a customer.
        if canary_held(prompt_pin, ctx.tenant_id):
            logger.info(
                "worker '%s': CANARY pin %s held (not routable, or non-platform tenant); "
                "running %s",
                worker_name,
                prompt_pin.revision if prompt_pin else None,
                resolved_prompt.revision if resolved_prompt else "the config prompt",
            )
            prompt_pin = None
            prompt_pin_held = True
        if prompt_pin is not None:
            apply_prompt_pin(worker_data, prompt_pin)
            prompt_pin_revision = prompt_pin.revision
            logger.info(
                "worker '%s' runs pinned prompt %s (%s) for this call",
                worker_name,
                prompt_pin_revision,
                prompt_pin.status,
            )
        elif resolved_prompt is not None:
            # Versions are immutable, so the revision identifies the content: a
            # release (new N) lands in a fresh slot and never reuses an agent
            # whose ``instructions=`` were baked from the previous version.
            prompt_pin_revision = resolved_prompt.revision
        prompt_selection = _select_prompt(request, prompt_pin, resolved_prompt)

        # Resolve worker ID and update state atomically
        worker_id, _in_registry = await state_manager.resolve_worker_id(
            ctx.tenant_id, ctx.subject_id, worker_name, worker_registry
        )

        async with state_manager.atomic_worker_operation(
            worker_id, worker_name, "message_processing"
        ):
            # Get current state for validation
            current_state = None
            db = StateDatabase()
            try:
                latest_state = await db.get_latest_worker_state(
                    tenant_id=ctx.tenant_id, subject_id=ctx.subject_id, worker_id=worker_id
                )
                if latest_state:
                    current_state = latest_state.get("state_type", "idle")
            except Exception as e:
                logger.warning(f"Could not retrieve current state for {worker_name}: {e}")
                current_state = "idle"  # Safe default

            # Validate state transition.
            # The validator only runs when a current state was actually read, so
            # that condition — not a hardcoded "true" — is what ``validated=``
            # reports on the write below. Otherwise a write whose state lookup
            # FAILED (the except above defaults it) would be booked as validated.
            # ⛔ Only "true"/"false" here: a call either ran or it did not. This
            # site can never be "by_construction" — that value means the
            # from-state is KNOWN at the site, and here it is precisely what
            # could not be established.
            transition_validated = (
                WORKER_STATE_WRITE_VALIDATED_TRUE
                if current_state
                else WORKER_STATE_WRITE_VALIDATED_FALSE
            )
            if current_state and not state_manager.validate_state_transition(
                current_state, "processing"
            ):
                logger.warning(
                    f"Invalid state transition for {worker_name}: {current_state} -> processing"
                )

            # Update worker state to processing
            await db.add_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                tenant_name=ctx.tenant_name,
                worker_id=worker_id,
                worker_name=worker_name,
                is_agent=True,
                state_type="processing",
                validated=transition_validated,
                worker_state={
                    "status": "processing",
                    "message": _message_for_state(request.message),
                    "previous_state": current_state,
                },
                visibility=ctx.visibility,
                metadata={"context": request.context} if request.context else None,
            )

        # Get or create worker instance (OPTIMIZED: reuses existing workers).
        # When the run names a product, re-resolve BYO posture per-(tenant,
        # product) — a per-product override wins over the tenant-wide byo_mode
        # (the licenses lookup is 5-min cached; no product ⇒ no extra call).
        effective_byo_mode = (
            await _lookup_tenant_byo_mode(str(ctx.tenant_id), request.product)
            if request.product
            else ctx.byo_mode
        )
        try:
            worker = await worker_cache.get_or_create_worker(
                worker_name,
                worker_data,
                tenant_id=ctx.tenant_id,
                byo_mode=effective_byo_mode,
                product=request.product,
                override_key=_override_cache_fingerprint(request, prompt_pin_revision),
            )
        except TenantLLMKeyUnusableError as exc:
            # Owner ruling 2026-10-01: the tenant's own key for this tier is
            # withheld (403) or unusable (409). Never run it on a platform model
            # instead. Raised as a refusal-after-processing so the
            # `except HTTPException` handler below writes the terminal error row.
            raise _RefusedAfterProcessing(
                exc.error_code,
                status_code=exc.status_code,
                detail=exc.message,
                error_details=exc.details(),
            ) from exc

        logger.info(f"Using worker instance: {worker}")
        ran_instance = worker

        # U6: the version + digest of the prompt this instance carries, published
        # to the outbound inference-router hook (X-AA-Prompt-Version /
        # X-AA-Prompt-Sha256) BEFORE the LLM call is made. Always set — None on a
        # legacy call — so a previous request's values can never leak through a
        # reused async context.
        selection_meta = selection_attribution(prompt_selection, worker_data, ran_instance)
        set_prompt_attribution(
            selection_meta["prompt_version_id"], selection_meta["prompt_body_sha256"]
        )

        # Send message to agent
        response = await _run_worker(
            worker,
            request,
            worker_name,
            enforce_schema_bounds=_effective_enforce_schema_bounds(request, worker_data),
        )

        # A2/A3: what this call reports about its prompt, decided against the
        # instance that actually ran (flag off + no stamp → adds nothing).
        verdict = await _pin_verdict_for_run(
            request,
            worker_data,
            prompt_pin,
            ran_instance,
            pin_held=prompt_pin_held,
            pin_tenant_forbidden=prompt_pin_forbidden,
        )

        # Extract output (can be any type)
        output = (
            response.output
            if hasattr(response, "output")
            else (response.data if hasattr(response, "data") else str(response))
        )

        # Determine output type
        # Get output schema from worker config first (needed for enum detection)
        config_dict = worker_data.get("config")
        output_schema_used = (
            config_dict.get("output_schema") if isinstance(config_dict, dict) else None
        )

        if isinstance(output, bool):
            output_type = "boolean"
        elif isinstance(output, str):
            # Check if schema defines enum/choices
            if (
                output_schema_used
                and isinstance(output_schema_used, dict)
                and "enum" in output_schema_used
            ):
                output_type = "enum"
            else:
                output_type = "string"
        elif isinstance(output, dict):
            output_type = "object"
        elif isinstance(output, list):
            output_type = "array"
        else:
            output_type = "other"

        # Bounded: this used to interpolate the whole response object. See
        # src/utils/log_bounds.py -- the prompt side was bounded first and the
        # response then became the largest remaining term (8.2% of log bytes,
        # avg 794 B, longest single line 4,890 chars, measured 2026-08-13).
        logger.info("Response from agent (%s): %s", output_type, summarize_for_log(output))

        # Update worker state to completed atomically
        async with state_manager.atomic_worker_operation(
            worker_id, worker_name, "message_completion"
        ):
            db = StateDatabase()
            # ⛔ NO transition check here, deliberately and permanently.
            #
            # This site used to call
            # ``validate_state_transition("processing", "completed")`` — two
            # string LITERALS, so the branch could never be entered. That was
            # dead code for a CORRECT reason: PROCESSING -> COMPLETED is the
            # first entry of STATE_TRANSITIONS[PROCESSING], so the guard is
            # unreachable BY CONSTRUCTION, not by accident. The call is
            # therefore deleted rather than "revived".
            #
            # ⛔ DO NOT WIRE ``current_state`` IN TO "make it real". That
            # variable is the PRE-message state, read before this handler wrote
            # the ``state_type="processing"`` row itself (the ``add_worker_state``
            # call ~80 lines above, which is also why the row written below
            # asserts ``"previous_state": "processing"``). The real from-state
            # at THIS point is always ``processing``, so ``current_state ->
            # completed`` would measure a transition that never happened.
            #
            # A review measured that mistake before it shipped. Over 30 days,
            # 273,640 ``processing`` rows carried these ``previous_state``
            # values: completed 113,867 · config-updated 112,428 ·
            # processing 30,356 · error 16,881 · other 42. Scaled by
            # completions, a ``current_state``-based check here would emit
            # ≈119,800 spurious ``invalid_transition`` counts and ≈103,000 extra
            # "Unknown state types" WARN lines per 30 days (~7,400/day) — a ~70×
            # increase in that log line and ~80% more WARNING volume overall.
            # Every one an artefact: the true transition, ``processing ->
            # completed``, is VALID. ``tests/test_message_endpoints.py`` pins
            # this by asserting the pair ("processing", "completed") is never
            # passed to the validator.
            #
            # ``validated="by_construction"`` on the write below is a statement
            # about that known-by-construction, known-valid from-state — not
            # about a validator call. ⛔ It is NOT ``"true"``: no call runs here,
            # and labelling it "true" would make the counter over-report the
            # validator's own coverage by ~250k completions / 30d — the exact
            # flattering number this whole change exists to prevent. Nor is it
            # ``"false"``: that value is the EXPOSURE number, and booking a
            # provably-valid transition into it would size a risk that is not
            # there. The OTHER site (the ``processing`` write above) passes
            # "true"/"false" instead because there the validator genuinely only
            # runs ``if current_state``; the two are different situations and
            # must not be unified.

            await db.add_worker_state(
                tenant_id=ctx.tenant_id,
                subject_id=ctx.subject_id,
                tenant_name=ctx.tenant_name,
                worker_id=worker_id,
                worker_name=worker_name,
                is_agent=True,
                state_type="completed",
                validated=WORKER_STATE_WRITE_VALIDATED_BY_CONSTRUCTION,
                visibility=ctx.visibility,
                worker_state={
                    "status": "completed",
                    "message": _message_for_state(request.message),
                    # 0 = succeeded on the primary run, no retry was needed. The
                    # rescue path below writes the attempt number that recovered
                    # it. Recording BOTH is what makes rescue rate a real ratio
                    # rather than a count of rescues with no denominator:
                    #
                    #   SELECT count(*) FILTER (WHERE (worker_state->>'retry_attempt')::int > 0)
                    #        / NULLIF(count(*), 0)::float AS rescue_rate
                    #   FROM worker_states WHERE state_type = 'completed';
                    #
                    # Until now the rescue was INFO-logged only ("recovered on LLM
                    # retry %d/%d"), so the rate could only be recovered by parsing
                    # Loki -- and Loki here retains ~16 days (memory
                    # loki-history-starts-at-the-worker4-migration), while
                    # worker_states is the durable record. No migration is needed:
                    # worker_state is JSONB.
                    "retry_attempt": 0,
                    # Only a flag-on call adds this key (the truthful revision,
                    # A3), so a flag-off row is byte-identical to before
                    # (P4 §4 "returns/records").
                    **({"worker_config_revision": verdict.revision} if verdict.revision else {}),
                    "output": (
                        output
                        if isinstance(output, (str, dict, list, int, float, bool, type(None)))
                        else str(output)
                    ),
                    "output_type": output_type,
                    "previous_state": "processing",
                    # Include usage metrics in worker state if available
                    "usage_metrics": (
                        response.usage_metrics.model_dump()
                        if hasattr(response, "usage_metrics") and response.usage_metrics
                        else None
                    ),
                    # Claude Code CLI session/cost tracking
                    "session_id": (
                        response.session_id if hasattr(response, "session_id") else None
                    ),
                    "cost_usd": (response.cost_usd if hasattr(response, "cost_usd") else None),
                    "duration_ms": (
                        response.duration_ms if hasattr(response, "duration_ms") else None
                    ),
                },
                metadata=_state_metadata(request, selection_meta, verdict),
            )

        return MessageResponse(
            output=output,
            output_type=output_type,
            output_schema_used=(
                output_schema_used if isinstance(output_schema_used, dict) else None
            ),
            # Prompt attribution (roadmap A5): which prompt version this worker
            # is labelled with, AND a digest of the prompt the instance that
            # just ran was actually carrying. ``ran_instance`` is the object
            # ``worker.run`` was called on, so the digest describes THIS
            # inference rather than the row we read before it.
            metadata=apply_pin_verdict(
                response_metadata(
                    worker_data,
                    ran_instance,
                    context=getattr(request, "context", None),
                    selection=prompt_selection,
                ),
                verdict,
            ),
            usage_metrics=(
                response.usage_metrics.model_dump()
                if hasattr(response, "usage_metrics") and response.usage_metrics
                else None
            ),
        )

    except BYORequiredError as exc:
        # Commercial (byo_required) tenant ran a worker without configuring a
        # provider. Hard-fail with an actionable error — never ride owner GPU.
        raise HTTPException(
            status_code=402,
            detail={
                "code": "byo_llm_not_configured",
                "message": "Configure your LLM provider in Settings before running agents.",
            },
        ) from exc

    except HTTPException as he:
        # Record error state for server errors before re-raising — and for a
        # refusal raised after the processing row, which would otherwise strand
        # the worker in ``processing``.
        if (he.status_code >= 500 or isinstance(he, _RefusedAfterProcessing)) and worker_data:
            try:
                db = StateDatabase()
                await db.add_worker_state(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    tenant_name=ctx.tenant_name,
                    visibility=ctx.visibility,
                    worker_id=worker_data.get("worker_id", worker_name),
                    worker_name=worker_name,
                    is_agent=True,
                    state_type="error",
                    worker_state={
                        "status": "error",
                        "message": _message_for_state(request.message),
                        "error": str(he.detail),
                    },
                    metadata=({"context": request.context} if request.context else None),
                )
            except Exception:
                logger.exception("Error updating worker state")
        raise
    except (BadRequestError, NotFoundError, InternalServerError) as e:
        # Record error state for server errors before converting
        if e.status_code >= 500 and worker_data:
            try:
                db = StateDatabase()
                await db.add_worker_state(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    tenant_name=ctx.tenant_name,
                    visibility=ctx.visibility,
                    worker_id=worker_data.get("worker_id", worker_name),
                    worker_name=worker_name,
                    is_agent=True,
                    state_type="error",
                    worker_state={
                        "status": "error",
                        "message": _message_for_state(request.message),
                        "error": str(e.detail),
                    },
                    metadata=({"context": request.context} if request.context else None),
                )
            except Exception:
                logger.exception("Error updating worker state")
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except Exception as e:
        # How many full agent runs this request actually burned. The ladder below can
        # spend llm_retry_max_attempts of them (7 in the cluster, against an in-tree
        # default of 3) and a caller seeing the single terminal error has no way to
        # know that. Measured 2026-08-31: a SUCCESSFUL `Select Best` averaged 281.9s
        # against a vLLM mean of 20.1s -- i.e. ~14 upstream calls -- so the cost of the
        # ladder is invisible in both the success and the failure path.
        _attempts_made = 1
        # Transient MCP/connection fault: recreate the worker and re-run with
        # bounded exponential backoff + jitter. llm_retry_max_attempts counts the
        # primary run (line 353), so recreate-and-run runs at most (max-1) more times.
        #
        # An IR ``queue_rejected`` is NEVER laddered: IR had no free provider slot,
        # so a recreated worker re-hits the same full queue and each retry only adds
        # load to it. It is mapped to 503 + Retry-After below instead.
        if (
            ("MCP server is not running" in str(e) or _is_connection_error(e))
            and worker_data
            and not _is_queue_rejected(e)
        ):
            _settings = get_settings()
            # llm_retry_max_attempts bounds the COUNT of attempts, not the TIME they
            # can take, and each attempt is separately allowed a full agent cap
            # (agent_run_timeout_s=600 / agent_run_timeout_mcp_s=900). At the values
            # live in the cluster (LLM_RETRY_MAX_ATTEMPTS=7, base 2.0s, max 30.0s)
            # that is 6 recreate-retries x 600s + 90s of accumulated backoff = 3690s
            # of retrying on top of a primary run that already burned up to 600s —
            # 4290s for one request. Bound the phase in wall
            # clock as well: a new attempt only starts while the budget holds, and on
            # expiry we fall through to the handler below, which raises the REAL last
            # error (``e`` is reassigned from each retry) — never a synthesised
            # success and never a placeholder message.
            _retry_deadline = time.monotonic() + _settings.llm_retry_deadline_s
            for attempt in range(1, _settings.llm_retry_max_attempts):
                _attempts_made = attempt + 1
                if time.monotonic() >= _retry_deadline:
                    logger.warning(
                        "Worker %s LLM retry budget of %.1fs exhausted after %d/%d "
                        "attempts; surfacing the last error: %s",
                        worker_name,
                        _settings.llm_retry_deadline_s,
                        attempt - 1,
                        _settings.llm_retry_max_attempts - 1,
                        e,
                    )
                    break
                await asyncio.sleep(
                    _backoff_delay(
                        attempt,
                        base_delay_s=_settings.llm_retry_base_delay_s,
                        max_delay_s=_settings.llm_retry_max_delay_s,
                        jitter_ratio=_settings.llm_retry_jitter_ratio,
                    )
                )
                logger.warning(
                    "Worker %s LLM stage error (retry %d/%d), recreating: %s",
                    worker_name,
                    attempt,
                    _settings.llm_retry_max_attempts - 1,
                    e,
                )
                try:
                    # Evict ONLY this request's own slot. remove_worker() deletes by
                    # ``tenant:worker:`` prefix and _cleanup_worker closes the agent's
                    # AsyncOpenAI pool, so the prefix form closed the http client out
                    # from under every concurrent sibling run on another variant —
                    # handing them a spurious "Connection error." that _is_connection_error
                    # scores transient, dragging them into this same ladder, where their
                    # recreate evicted ours right back. That mutual eviction is why
                    # concurrent runs of one worker failed in lockstep across the whole
                    # ladder instead of independently.
                    #
                    # ``expected=ran_instance`` closes the WITHIN-slot half of the
                    # same defect. Scoping by key stopped one variant evicting
                    # another; it does nothing when N concurrent runs share ONE
                    # slot and fail together — the first rebuilds, and the rest
                    # then evict its brand-new replacement out from under it.
                    # Compare-and-evict means a run can only retire the instance
                    # it actually ran; the losers just join the replacement.
                    await worker_cache.remove_instance(
                        worker_name,
                        worker_data,
                        tenant_id=ctx.tenant_id,
                        product=request.product,
                        override_key=_override_cache_fingerprint(request, prompt_pin_revision),
                        expected=ran_instance,
                    )
                    retry_byo_mode = (
                        await _lookup_tenant_byo_mode(str(ctx.tenant_id), request.product)
                        if request.product
                        else ctx.byo_mode
                    )
                    worker = await worker_cache.get_or_create_worker(
                        worker_name,
                        worker_data,
                        tenant_id=ctx.tenant_id,
                        byo_mode=retry_byo_mode,
                        product=request.product,
                        override_key=_override_cache_fingerprint(request, prompt_pin_revision),
                    )
                    ran_instance = worker
                    # U6: re-describe the RECREATED instance before its LLM call.
                    retry_selection_meta = selection_attribution(
                        prompt_selection, worker_data, ran_instance
                    )
                    set_prompt_attribution(
                        retry_selection_meta["prompt_version_id"],
                        retry_selection_meta["prompt_body_sha256"],
                    )
                    response = await _run_worker(
                        worker,
                        request,
                        worker_name,
                        enforce_schema_bounds=_effective_enforce_schema_bounds(
                            request, worker_data
                        ),
                    )
                    retry_verdict = await _pin_verdict_for_run(
                        request,
                        worker_data,
                        prompt_pin,
                        ran_instance,
                        pin_held=prompt_pin_held,
                        pin_tenant_forbidden=prompt_pin_forbidden,
                    )
                    logger.info(
                        "Worker %s recovered on LLM retry %d/%d",
                        worker_name,
                        attempt,
                        _settings.llm_retry_max_attempts - 1,
                    )

                    # Extract output (can be any type)
                    retry_output = (
                        response.output
                        if hasattr(response, "output")
                        else (response.data if hasattr(response, "data") else str(response))
                    )

                    # Determine output type
                    # Get output schema from worker config first (needed for enum detection)
                    retry_config_dict = worker_data.get("config")
                    retry_output_schema_used = (
                        retry_config_dict.get("output_schema")
                        if isinstance(retry_config_dict, dict)
                        else None
                    )

                    if isinstance(retry_output, bool):
                        retry_output_type = "boolean"
                    elif isinstance(retry_output, str):
                        # Check if schema defines enum/choices
                        if (
                            retry_output_schema_used
                            and isinstance(retry_output_schema_used, dict)
                            and "enum" in retry_output_schema_used
                        ):
                            retry_output_type = "enum"
                        else:
                            retry_output_type = "string"
                    elif isinstance(retry_output, dict):
                        retry_output_type = "object"
                    elif isinstance(retry_output, list):
                        retry_output_type = "array"
                    else:
                        retry_output_type = "other"

                    # Update to completed state
                    db = StateDatabase()
                    await db.add_worker_state(
                        tenant_id=ctx.tenant_id,
                        subject_id=ctx.subject_id,
                        tenant_name=ctx.tenant_name,
                        visibility=ctx.visibility,
                        worker_id=worker_data.get("worker_id", worker_name),
                        worker_name=worker_name,
                        is_agent=True,
                        state_type="completed",
                        worker_state={
                            "status": "completed",
                            "message": _message_for_state(request.message),
                            # The attempt that actually recovered this request (>= 1).
                            # Pairs with retry_attempt=0 on the primary path above so
                            # rescue rate is a ratio, not an orphan count.
                            "retry_attempt": attempt,
                            **(
                                {"worker_config_revision": retry_verdict.revision}
                                if retry_verdict.revision
                                else {}
                            ),
                            "output": (
                                retry_output
                                if isinstance(
                                    retry_output,
                                    (str, dict, list, int, float, bool, type(None)),
                                )
                                else str(retry_output)
                            ),
                            "output_type": retry_output_type,
                            # Include usage metrics in worker state if available
                            "usage_metrics": (
                                response.usage_metrics.model_dump()
                                if hasattr(response, "usage_metrics") and response.usage_metrics
                                else None
                            ),
                            # Claude Code CLI session/cost tracking
                            "session_id": (
                                response.session_id if hasattr(response, "session_id") else None
                            ),
                            "cost_usd": (
                                response.cost_usd if hasattr(response, "cost_usd") else None
                            ),
                            "duration_ms": (
                                response.duration_ms if hasattr(response, "duration_ms") else None
                            ),
                        },
                        metadata=_state_metadata(request, retry_selection_meta, retry_verdict),
                    )

                    return MessageResponse(
                        output=retry_output,
                        output_type=retry_output_type,
                        output_schema_used=(
                            retry_output_schema_used
                            if isinstance(retry_output_schema_used, dict)
                            else None
                        ),
                        # Same attribution as the primary path. ``ran_instance``
                        # was reassigned to the RECREATED worker above, so the
                        # digest describes the attempt that actually produced
                        # this output — not the one that failed.
                        metadata=apply_pin_verdict(
                            response_metadata(
                                worker_data,
                                ran_instance,
                                context=getattr(request, "context", None),
                                selection=prompt_selection,
                            ),
                            retry_verdict,
                        ),
                        usage_metrics=(
                            response.usage_metrics.model_dump()
                            if hasattr(response, "usage_metrics") and response.usage_metrics
                            else None
                        ),
                    )
                except Exception as retry_error:
                    e = retry_error
                    if isinstance(retry_error, (TenantLLMKeyUnusableError, BYORequiredError)):
                        # The REBUILD itself was refused (the tenant's key row
                        # changed mid-run, or a byo_required tenant lost its row).
                        # Never transient, never a 500: mapped below to the same
                        # 403/409/402 shapes as the first build.
                        break
                    if _is_queue_rejected(retry_error) or not (
                        "MCP server is not running" in str(retry_error)
                        or _is_connection_error(retry_error)
                    ):
                        # non-transient (or IR queue_rejected: busy, retrying only
                        # adds load) — stop retrying; fall to the handler below,
                        # which logs the authoritative traceback before raising.
                        break
                    # transient — retry (or exhaust). Terminal failure is logged
                    # with a full traceback by the fall-through handler below, so
                    # keep the per-attempt line quiet (no duplicate stack traces).
                    logger.warning(
                        "Worker %s LLM retry %d/%d failed transiently: %s",
                        worker_name,
                        attempt,
                        _settings.llm_retry_max_attempts - 1,
                        retry_error,
                    )

        # Update worker state to error if worker_data exists
        if worker_data:
            try:
                db = StateDatabase()
                await db.add_worker_state(
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    tenant_name=ctx.tenant_name,
                    visibility=ctx.visibility,
                    worker_id=worker_data.get("worker_id", worker_name),
                    worker_name=worker_name,
                    is_agent=True,
                    state_type="error",
                    worker_state={
                        "status": "error",
                        "message": _message_for_state(request.message),
                        # WM-1: a tenant-key run's vendor text can carry the key
                        # (google echoes ?key= in the URL) — scrub it here too.
                        "error": (
                            sanitize_byo_error(ran_instance, str(e))
                            if ran_on_tenant_key(ran_instance)
                            else str(e)
                        ),
                        # Typed reason + JSON paths for a post-hoc bound failure, so
                        # the row is queryable without parsing the message.
                        **(
                            {
                                "error_type": SCHEMA_BOUND_VIOLATION,
                                "violations": e.violations,
                            }
                            if isinstance(e, SchemaBoundViolationError)
                            else {}
                        ),
                    },
                    metadata=({"context": request.context} if request.context else None),
                )
            except Exception:
                logger.exception("Error updating worker state")

        # A refusal raised by the retry ladder's REBUILD (the first build's
        # refusals are converted where they are raised): the same contract as
        # that first build — 403 BYO_KEY_WITHHELD / 409 BYO_KEY_UNUSABLE with
        # details, or 402 byo_llm_not_configured — never a generic 500. The
        # error row was written just above.
        if isinstance(e, TenantLLMKeyUnusableError):
            raise _TypedHTTPException(
                e.error_code,
                status_code=e.status_code,
                detail=e.message,
                error_details=e.details(),
            ) from e
        if isinstance(e, BYORequiredError):
            raise HTTPException(
                status_code=402,
                detail={
                    "code": "byo_llm_not_configured",
                    "message": "Configure your LLM provider in Settings before running agents.",
                },
            ) from e

        # A count/length bound (maxItems/maxLength/minItems/minLength) that the
        # decoder could not enforce and the post-hoc check rejected. Deterministic
        # in the request — at T=0 the same output recurs — so 422 (the n8n Workers
        # node does not retry 4xx), with a code a caller can branch on.
        if isinstance(e, SchemaBoundViolationError):
            logger.warning(
                "Error sending message to worker %s: responding HTTP 422 (%s)",
                worker_name,
                SCHEMA_BOUND_VIOLATION,
            )
            raise _TypedHTTPException(
                SCHEMA_BOUND_VIOLATION.upper(),
                status_code=422,
                detail=str(e),
                error_details=_sent_usage_details(e),
            ) from e

        # Classify FIRST, log SECOND. The ERROR line has to be able to name the
        # status we are about to return, and it cannot do that from above the
        # branches.
        upstream_status = _upstream_status_code(e)
        ir_classification, ir_retry_after = _ir_error_meta(e)
        response_headers: dict[str, str] | None = None

        # IR queue_rejected: no free provider slot. BUSY, not broken -- 503 with a
        # Retry-After and IR's classification forwarded, so the caller backs off
        # instead of reading a 502/500 as an outage. Placed FIRST: it outranks the
        # connection-text guess and whatever status IR used (502 before the IR
        # change, 503 after), and the ladder above has already refused to retry it.
        if ir_classification == IR_QUEUE_REJECTED:
            status_code = 503
            detail = str(e)
            response_headers = {
                "Retry-After": ir_retry_after or str(get_settings().queue_rejected_retry_after_s),
                "X-IR-Error-Classification": IR_QUEUE_REJECTED,
            }
        # A terminal failure that is a property of the REQUEST (prompt too long for
        # the model's max_tokens; structured-output budget already exhausted) cannot
        # succeed on an identical retry. Callers retry on 5xx and not on 4xx -- the
        # n8n Workers node gates on `statusCode >= 500` -- so returning 500 here made
        # every such failure cost up to MAX_RETRIES full agent runs, and 91.8% of
        # terminal errors over the last 30 days (3347/3645) were exactly this class.
        # 422 matches the existing convention in this module for "the request as
        # given cannot be fulfilled". Unrecognised errors keep 500 and stay retryable.
        elif _is_deterministic_failure(e):
            status_code = 422
            detail = str(e)
        # An upstream that answered with a status of its own gets that status
        # forwarded rather than flattened into 500. This is deliberately placed
        # AHEAD of `_is_connection_error`: a 503 body can itself read "All
        # providers failed. Last error: Connection error.", and a structured
        # status the upstream really sent must outrank a substring guess about
        # the message text. `str(e)` already names status_code, model_name and the
        # provider-level body, so unlike the bare "Connection error." literal it
        # needs no enrichment to be diagnosable.
        elif upstream_status in _PASS_THROUGH_UPSTREAM_STATUSES:
            status_code = upstream_status
            detail = str(e)
        # An upstream that could not be reached is not an internal error, and saying
        # so costs the caller nothing: the n8n Workers node gates its retry on
        # `statusCode >= 500`, so 502 retries exactly as 500 did. What changes is that
        # the failure now names its own hop and its underlying cause instead of
        # arriving as the bare literal "Connection error.". 502 is already this
        # repo's convention for an unreachable upstream (see
        # `src/api/providers/_chat_router.py:213` and `_router.py:296`).
        elif _is_connection_error(e):
            status_code = 502
            detail = _describe_connection_failure(
                e, worker_name=worker_name, attempts=_attempts_made
            )
        else:
            status_code = 500
            detail = str(e)
        # Owner ruling 2026-10-01: when the run was on the TENANT'S OWN key, say
        # so — name the provider and the fix — instead of a bare upstream error.
        # Nothing fell back to a platform model.
        detail = byo_failure_detail(ran_instance, ctx.tenant_id, detail)
        # WM-1: a server-side failure on the tenant's own key is the VENDOR
        # rejecting it (bad key, quota, unknown model, vendor down), not an
        # internal error. Typed 502 BYO_PROVIDER_CALL_FAILED with sanitized
        # detail. A 4xx classification (deterministic request failure) keeps its
        # status; its detail is already sanitized by byo_failure_detail above.
        byo_http: _TypedHTTPException | None = None
        if status_code >= 500 and ran_on_tenant_key(ran_instance):
            byo_http = byo_failure_http(ran_instance, ctx.tenant_id, e, upstream_status)
            status_code = byo_http.status_code
            detail = str(byo_http.detail)

        # The status goes on the ERROR line itself, not only inside exc_info.
        # During the 2026-08-31 incident this service emitted 2,544 failures in an
        # hour and an operator grepping ERROR-level logs for "500" found NOTHING:
        # the only line carrying the number was `logger.info("HTTPException
        # handler: 500 ...")` in `src/api/error_handlers.py`, at INFO. The
        # "Error sending message to worker %s" prefix is preserved verbatim so
        # existing greps and dashboards keep matching; the status is appended.
        logger.error(
            "Error sending message to worker %s: responding HTTP %s (upstream_status=%s)",
            worker_name,
            status_code,
            upstream_status,
            exc_info=True,
        )
        if byo_http is not None:
            raise byo_http from e
        sent_usage = _sent_usage_details(e)
        if sent_usage is not None:
            raise _SentUsageHTTPException(
                status_code=status_code,
                detail=detail,
                error_details=sent_usage,
                headers=response_headers,
            ) from e
        raise HTTPException(status_code=status_code, detail=detail, headers=response_headers) from e
