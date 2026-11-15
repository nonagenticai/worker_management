r"""Stateless chat completions — POST /chat/completions.

Reuses the pydantic-ai factory dispatch from `pydantic_ai_agent.py` to build a
fresh model per request and drives it via `pydantic_ai.Agent`. No worker
registry, no DB writes — purely request/response.

`stream=true` returns text/event-stream with OpenAI-compat `data: {...}\\n\\n`
chunks carrying a `delta` field, terminated by `data: [DONE]\\n\\n`. The final
chunk before `[DONE]` is a `meta` frame with the model + provider the upstream
actually reported (so failover-chain providers like inference-router reveal
which backend in the chain served the call).

TENANT RULES (owner rulings 2026-10-01 and 2026-10-03). A platform tenant
(``tenant_constants.commercial_platform_tenant_ids()``, gitops
``COMMERCIAL_PLATFORM_TENANT_IDS``) may call any provider, but an EXTERNAL one
(the owner's registered commercial keys) only when this request PROVED the
tenant (``src.auth.tenant_proof.owner_commercial_keys_permitted``); otherwise 403
``PROVIDER_NOT_ENABLED_FOR_TENANT`` — the same rule the ``/providers`` catalog
applies. Every other tenant may call ONLY:

- ``inference_router`` with model ``local-llm`` (its inbound ``X-Provider`` pin
  is dropped, so a customer cannot pin inference-router to a commercial
  upstream), or
- an external provider it holds a usable BYO key for (own, or managed BYO set
  by the owner for that tenant). The call is built with THAT key
  (``tenant_byo_key=True``; the owner's env key is never read) and emits the
  same BYO cost_event the worker path does.

Anything else is 403 before a model is built.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from src.api.identity_context import apply_body_correlation_id, set_provider_pin_context

# Module reference, not names: the tenant helpers are patched in ONE place
# (``_router``) by tests, and the two routers must never disagree about who the
# caller is.
from src.api.providers import _router as _providers
from src.api.tenant_constants import is_commercial_platform_tenant
from src.api.workers._helpers import IR_QUEUE_REJECTED, _ir_error_meta, _is_queue_rejected
from src.auth.tenant_proof import owner_commercial_keys_refusal
from src.security.redact import redact_secrets_in_text
from src.services.mlx_model_registry import mlx_model_registry
from src.settings import get_settings
from src.worker.agents.provider_catalog import billing_provider_for, is_on_hardware_ir_pin
from src.worker.agents.pydantic_ai_agent import (
    PROVIDER_METADATA,
    SELF_HOSTED_PROVIDERS,
    ModelProvider,
    PydanticAIAgent,
    PydanticAIAgentConfig,
    ResponseExtractor,
    _output_type_from_schema,
    _repetition_penalty_of,
    _resolve_max_tokens,
    _with_repetition_penalty,
    resolve_billable_provider,
)
from src.worker.provider_registry import EXTERNAL_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Emit a `:keepalive\n\n` SSE comment when the upstream model has been silent
# for this many seconds. SSE comment lines (RFC 6.8.1) are spec-defined as
# ignored by clients but are real bytes on the wire — they reset idle-stream
# timers in intermediaries (Cloudflare edge ~100s, cloudflared keepAliveTimeout,
# Envoy stream_idle_timeout) so a slow first-token or long inter-token pause
# can't trigger a reset that the user sees as a ~110s "hang".
SSE_HEARTBEAT_INTERVAL_S = float(os.getenv("CHAT_SSE_HEARTBEAT_INTERVAL_S", "15"))

# Minimum (last_delta − first_delta) window required before we publish a
# tg_tps (decode-rate) figure. Below this, the observed window is dominated
# by event-loop / network-flush coalescing rather than real per-token timing
# — `stream_text(delta=True)` yields per flush, not per token, so when the
# chain (mlx_inference / vLLM → pydantic-ai's OpenAI client → asyncio) batches
# several SSE chunks into adjacent wakeups, the wall-clock between first and
# last yield collapses to tens of microseconds even though many tokens were
# generated. Reporting tg in that regime produces values like 2.4M t/s. We
# instead drop the field; the fr UI renders no tg row when it's absent.
MIN_TG_WINDOW_S = float(os.getenv("CHAT_MIN_TG_WINDOW_S", "0.05"))


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    provider: str = Field(description="One of the names from GET /providers")
    model: str = Field(description="Model id for the chosen provider", min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt (overrides any 'system' role messages)",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=16384,
        description=(
            "Generation budget, sent ONLY when supplied (omitted ⇒ no max_tokens key, "
            "exactly as before). An explicit value is clamped by the same "
            "``_resolve_max_tokens`` caps as the worker path."
        ),
    )
    seed: StrictInt | None = Field(default=None, description="Sampling seed; None ⇒ not sent.")
    repetition_penalty: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description="vLLM repetition penalty (self-hosted providers only); None ⇒ not sent.",
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="frequency_penalty; None ⇒ not sent."
    )
    correlation_id: UUID | None = Field(
        default=None,
        description=(
            "Join key forwarded to inference-router as X-Correlation-Id (a valid "
            "inbound X-Correlation-Id header wins, as on the worker path)."
        ),
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for structured output",
    )
    stream: bool = Field(default=False)


class ChatCompletionsUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # max_tokens actually sent (after _resolve_max_tokens); None ⇒ none was sent.
    max_tokens: int | None = None
    # The other sampling keys actually sent, read back from the settings the
    # agent was built with (the request body's source), never from the request
    # model: a repetition_penalty dropped for a non-self-hosted provider is
    # None here because it was not sent. None ⇒ the key was absent.
    temperature: float | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    # Provider-reported finish reason (pydantic-ai normalised); "length" ⇒ the
    # answer was truncated by max_tokens.
    finish_reason: str | None = None
    # prompt_tokens / time_to_first_token — only meaningful on the streaming
    # path, where we can observe when the first delta arrived.
    pp_tps: float | None = None
    # output_tokens / (last_delta_t - first_delta_t) — only meaningful on the
    # streaming path.
    tg_tps: float | None = None


class ChatCompletionsResponse(BaseModel):
    output: str
    provider: str
    model: str
    usage: ChatCompletionsUsage
    # Actual model name the upstream reported in its response. For routers
    # (inference-router) this surfaces which backend in the chain served the
    # call (gemma / deepseek / claude / vllm / …). None when the upstream
    # didn't echo a model name.
    resolved_model: str | None = None
    # Provider name the upstream's SDK / client reported. Redundant with
    # req.provider for direct providers; for routers it names the specific
    # SDK family the winning backend used.
    resolved_provider: str | None = None
    # inference-router's read-back of what ACTUALLY served the call, passed
    # through unchanged. None when the upstream did not report it.
    served_provider: str | None = None
    served_model: str | None = None
    served_model_root: str | None = None


def _validate_provider(name: str) -> None:
    if name not in PROVIDER_METADATA:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {name}")


async def _resolve_base_url(req: ChatCompletionsRequest) -> str | None:
    """Return the per-instance base URL for the requested provider, or None to use defaults.

    For `mlx`, look up the specific instance URL (mini1/mini2/ring) for the
    chosen model via the MLX registry so pydantic-ai talks to the right host
    instead of falling through to `OPENAI_API_BASE`. Other providers have
    fixed endpoints and return None here (the factory handles their defaults).
    """
    if req.provider != "mlx":
        return None
    try:
        return await mlx_model_registry.resolve_base_url(req.model)
    except Exception as exc:
        logger.warning("mlx_registry_resolve_failed", extra={"error": str(exc)})
        return None


async def _resolve_tenant_access(
    req: ChatCompletionsRequest, request: Request
) -> dict[str, Any] | None:
    """Apply the tenant rules; return the BYO entry for a BYO call, else None.

    Raises 403 for a provider/model this tenant may not use. Also decides the
    inbound ``X-Provider`` pin: honoured for a platform tenant only.
    """
    tenant_id = _providers.caller_tenant_id(request)
    inbound_pin = request.headers.get("x-provider")
    if is_commercial_platform_tenant(tenant_id):
        # ⛔ Membership is NOT proof (owner ruling 2026-10-03, fr#1385): every
        # privilege below that reaches the OWNER'S commercial keys — an external
        # provider, an X-Provider pin, a non-local inference-router model — is
        # granted only when THIS request proved the tenant. One predicate.
        refusal = owner_commercial_keys_refusal(tenant_id)
        if refusal is None:
            # Caller's inference-router provider pin → forwarded to IR as X-Provider.
            set_provider_pin_context(inbound_pin)
            return None
        # Refused here, typed, before a model is built — ModelFactory.
        # enforce_local_only_provider refuses an external one again at build.
        if req.provider in EXTERNAL_PROVIDERS:
            raise _providers.provider_not_enabled_for_tenant(tenant_id, req.provider, refusal)
        # An unproven platform tenant keeps owner HARDWARE (vllm / mlx /
        # inference_router local-llm, a pin to an on-hardware upstream such as
        # pgx-local) but no route to the owner's commercial upstreams behind
        # inference-router: any other / unknown pin is dropped, local-llm only.
        if is_on_hardware_ir_pin(inbound_pin):
            set_provider_pin_context(inbound_pin)
        else:
            set_provider_pin_context(None)
        if inbound_pin and not is_on_hardware_ir_pin(inbound_pin):
            logger.info(
                "chat_completions_provider_pin_dropped_unproven_platform",
                extra={"tenant_id": tenant_id, "provider_pin": inbound_pin, "reason": refusal},
            )
        if req.provider == "inference_router" and req.model != _providers.customer_platform_model():
            raise _providers.provider_not_enabled_for_tenant(tenant_id, req.provider, refusal)
        return None

    # ⛔ A customer may not pin inference-router to a specific upstream: that is
    # how a commercial provider behind IR would be reached on the owner's key.
    set_provider_pin_context(None)
    if inbound_pin:
        logger.info(
            "chat_completions_provider_pin_dropped_non_platform",
            extra={"tenant_id": tenant_id, "provider_pin": inbound_pin},
        )

    if req.provider == "inference_router":
        allowed = _providers.customer_platform_model()
        if req.model != allowed:
            raise _providers.TenantLLMHTTPException(
                403,
                "MODEL_NOT_AVAILABLE_TO_TENANT",
                (
                    f"Model '{req.model}' is not available to this tenant on "
                    f"inference_router; use '{allowed}'."
                ),
                {"tenant_id": tenant_id, "provider": req.provider, "model": req.model},
            )
        return None

    status = (
        (await _providers.tenant_byo_provider_status(request))
        if req.provider in EXTERNAL_PROVIDERS
        else {}
    )
    st = status.get(req.provider)
    if st is None:
        logger.info(
            "chat_completions_refused_not_tenant_usable",
            extra={"tenant_id": tenant_id, "provider": req.provider},
        )
        raise _providers.byo_not_configured(tenant_id, req.provider)
    if st["status"] != "usable":
        # The tenant HAS a key for this provider; say why it cannot be used.
        raise _providers.key_error_http(st["error"])
    return st["entry"]


def _byo_failed(
    req: ChatCompletionsRequest, request: Request, byo: dict[str, Any], exc: BaseException
) -> HTTPException:
    """The tenant's own key failed at the vendor: name the provider and the fix.

    Owner ruling 2026-10-01: a clear 502, never a retry on a platform model. The
    key is scrubbed from the upstream text.
    """
    return _providers.byo_call_failed(
        _providers.caller_tenant_id(request),
        req.provider,
        _providers._scrub(str(exc), byo.get("api_key")),
        model=req.model,
    )


def _emit_byo_cost(
    req: ChatCompletionsRequest,
    byo: dict[str, Any] | None,
    stream_result: Any,
    usage: ChatCompletionsUsage,
    latency_s: float,
) -> None:
    """Emit the cost_event for a direct-to-vendor chat call (worker parity).

    Same gate and fields as ``PydanticAIAgent``'s run path: EVERY call to an
    external provider is metered here — on the tenant's BYO key
    (funding=tenant_byo), a managed-BYO row or the owner's env key for a proven
    commercial-platform workspace (funding=platform_external). ⛔ The gate used
    to require a BYO ``cost_source``, so the platform-paid path emitted nothing.
    Platform-routed providers are metered by inference-router.
    """
    from src.telemetry.cost_events import (
        emit_byo_cost_event_task,
        funding_for,
        is_external_call,
    )

    if usage.input_tokens is None or not is_external_call(req.provider):
        return
    served = ResponseExtractor.extract_served_fields(stream_result)
    resolved_model, _ = _resolved_from_result(stream_result)
    served_model = served.get("served_model") or resolved_model
    cost_provider = resolve_billable_provider(
        served_provider=served.get("served_provider"),
        cost_provider=(
            billing_provider_for(byo["preset_key"], req.model, req.provider) if byo else None
        ),
        model_provider=req.provider,
        served_model=served_model,
        requested_model=req.model,
    )
    if cost_provider is None:
        logger.error(
            "byo_cost_event_skipped_unpriceable_provider worker=chat-completion "
            "model=%s provider=%s in=%s out=%s",
            req.model,
            req.provider,
            usage.input_tokens,
            usage.output_tokens,
        )
        return
    owner_provided = bool(byo.get("owner_provided")) if byo else False
    emit_byo_cost_event_task(
        provider=cost_provider,
        model=req.model,
        served_model=served_model,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        latency_ms=int(latency_s * 1000),
        owner_provided=owner_provided,
        model_provider=req.provider,
        worker_name="chat-completion",
        funding=funding_for(
            model_provider=req.provider,
            tenant_byo_key=bool(byo),
            owner_provided=owner_provided,
        ),
        key_fp=str(byo.get("key_fingerprint") or "") if byo else "",
        key_role=str(byo.get("role") or "") if byo else "",
    )


async def _build_agent(
    req: ChatCompletionsRequest, request: Request, byo: dict[str, Any] | None = None
) -> Agent:
    """Build a one-shot pydantic-ai Agent using the existing factory dispatch.

    Identity (tenant/license/subject) is pulled from `request.state` and
    threaded into the agent config so the inference-router provider can
    forward it as `X-Tenant-Id` / `X-License-Id` / `X-Subject-Id`. This is
    redundant with the ContextVar mechanism in InferenceRouterOpenAIProvider
    (the middleware sets both) but provides defense in depth: if the
    ContextVar is unset for any reason, the constructor-arg fallback still
    attributes the call correctly.
    """
    # _validate_provider has already rejected anything not in PROVIDER_METADATA,
    # which is keyed by the ModelProvider literal values. db_path is empty —
    # we never persist anything for chat/completions.
    resolved_base_url = await _resolve_base_url(req)
    config = PydanticAIAgentConfig(
        name="chat-completion",
        model_provider=cast("ModelProvider", req.provider),
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        seed=req.seed,
        repetition_penalty=req.repetition_penalty,
        frequency_penalty=req.frequency_penalty,
        system_prompt=req.system_prompt,
        base_url=(byo.get("base_url") or None) if byo else resolved_base_url,
        db_path="",
        # Same resolution the tenant gate used, so the gate and ModelFactory's
        # own platform check can never disagree about who the caller is.
        tenant_id=_providers.caller_tenant_id(request),
        license_id=getattr(request.state, "license_id", None),
        subject_id=getattr(request.state, "subject_id", None),
        # Tenant BYO: the tenant's own (or managed) key, marked so ModelFactory
        # admits the external provider and never tops it up from the env.
        api_key=byo["api_key"] if byo else None,
        tenant_byo_key=bool(byo),
        cost_source=byo.get("cost_source") if byo else None,
        cost_provider=(
            billing_provider_for(byo["preset_key"], req.model, req.provider) if byo else None
        ),
        owner_provided=bool(byo.get("owner_provided")) if byo else False,
    )
    # Reach into the factory via a throwaway PydanticAIAgent to get the
    # pydantic-ai Model; we don't call its run_agent() (that uses the wrapper
    # logic for workers). Instead we construct Agent directly.
    tmp = PydanticAIAgent.__new__(PydanticAIAgent)
    model = tmp._initialize_model(config)
    # ⛔ Byte-identical for a caller that sends none of the optional fields: the
    # body is temperature only, exactly as before — NO default max_tokens and NO
    # reasoning switch (ModelFactory.get_model_settings would add both). Each
    # optional key is added only when the caller supplied it; an explicit
    # max_tokens still goes through ``_resolve_max_tokens`` for the same caps as
    # the worker path.
    settings = ModelSettings(temperature=req.temperature)
    if req.max_tokens is not None:
        resolved_max_tokens = _resolve_max_tokens(config)
        if resolved_max_tokens is not None:
            settings["max_tokens"] = resolved_max_tokens
    if req.seed is not None:
        settings["seed"] = req.seed
    if req.frequency_penalty is not None:
        settings["frequency_penalty"] = req.frequency_penalty
    if req.repetition_penalty is not None:
        if req.provider in SELF_HOSTED_PROVIDERS:
            settings["extra_body"] = _with_repetition_penalty(None, req.repetition_penalty)
        else:
            logger.warning(
                "chat_completions_repetition_penalty_not_sent",
                extra={"provider": req.provider},
            )
    # Translate the requested output_schema into a pydantic-ai output_type.
    # _output_type_from_schema returns ``str`` when no schema is set, which
    # is the unchanged plain-text default; for boolean/array/object schemas
    # it picks per-provider mode (NativeOutput → vLLM response_format=json_schema,
    # PromptedOutput for mlx, bare type elsewhere). Retries mirror the worker
    # agent path (pydantic_ai_agent.py:505–509).
    output_type = _output_type_from_schema(req.output_schema, cast("ModelProvider", req.provider))
    retries = 10 if req.provider == "mlx" else (2 if req.provider == "inference_router" else 3)
    return Agent(
        model=model,
        system_prompt=req.system_prompt or "",
        model_settings=settings,
        output_type=output_type,
        retries=retries,
    )


def _sent_max_tokens(agent: Any) -> int | None:
    """The max_tokens on the agent's own settings — the value that went out."""
    settings = getattr(agent, "model_settings", None)
    value = settings.get("max_tokens") if isinstance(settings, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sent_sampling(agent: Any) -> dict[str, Any]:
    """temperature / seed / penalties on the agent's own settings — what went out.

    Read with ``.get`` / ``is not None`` only: 0 (T=0, seed=0) is a real sent
    value and must echo as 0, never collapse to None.
    """
    settings = getattr(agent, "model_settings", None)
    if not isinstance(settings, dict):
        return dict.fromkeys(("temperature", "seed", "repetition_penalty", "frequency_penalty"))
    return {
        "temperature": settings.get("temperature"),
        "seed": settings.get("seed"),
        "repetition_penalty": _repetition_penalty_of(cast("ModelSettings", settings)),
        "frequency_penalty": settings.get("frequency_penalty"),
    }


def _extract_prompt(messages: list[ChatMessage]) -> str:
    """Use the last user-role message as the prompt.

    Full conversation history via pydantic-ai `message_history` is a separate
    change; the fr chat page always re-sends the full transcript on each turn,
    so the last user message is sufficient for single-turn playground UX.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found in messages list")


@router.post(
    "/completions",
    response_model=None,
    operation_id="chat_completions",
)
async def chat_completions(req: ChatCompletionsRequest, request: Request) -> Any:
    _validate_provider(req.provider)
    prompt = _extract_prompt(req.messages)
    # Body correlation_id → the ContextVar the outbound IR hook forwards.
    apply_body_correlation_id(req.correlation_id)
    # Tenant rules (403s here, before any model is built) + the X-Provider pin.
    byo = await _resolve_tenant_access(req, request)

    try:
        agent = await _build_agent(req, request, byo) if byo else await _build_agent(req, request)
    except Exception as exc:
        logger.warning(
            "chat_completions_build_failed",
            extra={"error": _providers._scrub(str(exc), (byo or {}).get("api_key"))},
        )
        if byo:
            raise _byo_failed(req, request, byo, exc) from exc
        raise HTTPException(status_code=502, detail=f"Provider unavailable: {exc}") from exc

    if req.stream:
        return StreamingResponse(
            (
                _sse_stream(
                    agent,
                    prompt,
                    req,
                    byo,
                    on_byo_error=lambda exc: _byo_failed(req, request, byo, exc),
                )
                if byo
                else _sse_stream(agent, prompt, req)
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: internally stream so we can observe time-to-first-token
    # and the decode window, then buffer the full result and return it as a
    # single JSON body. `agent.run()` alone exposes token counts but no timing,
    # which is why pp_tps/tg_tps were always null on this path. We reuse the
    # exact `_compute_rates` logic the SSE path uses, so the two paths report
    # rates identically (and tg_tps is still dropped for typed/structured
    # output, where there is no honest per-token decode window).
    t0 = time.perf_counter()
    first_delta_t: float | None = None
    last_delta_t: float | None = None
    delta_count = 0
    try:
        async with agent.run_stream(prompt) as stream:
            if req.output_schema is None:
                parts: list[str] = []
                async for delta in stream.stream_text(delta=True):
                    now = time.perf_counter()
                    if first_delta_t is None:
                        first_delta_t = now
                    last_delta_t = now
                    delta_count += 1
                    parts.append(delta)
                output = "".join(parts)
            else:
                first_delta_t = time.perf_counter()
                validated = await stream.get_output()
                last_delta_t = time.perf_counter()
                output = (
                    validated if isinstance(validated, str) else json.dumps(validated, default=str)
                )
            stream_result = stream
    except Exception as exc:
        logger.warning(
            "chat_completions_run_failed",
            extra={"error": _providers._scrub(str(exc), (byo or {}).get("api_key"))},
        )
        # IR queue_rejected: no free provider slot. BUSY, not broken -- 503 with a
        # Retry-After and IR's classification forwarded (same contract as the
        # worker path in message_endpoints), so the caller backs off instead of
        # reading a 502 as an outage and retrying immediately. Checked BEFORE the
        # BYO envelope: IR is the platform path, a BYO call never raises it -- and
        # `not byo` makes that structural, so busy can never mask a BYO vendor failure.
        if _is_queue_rejected(exc) and not byo:
            _, ir_retry_after = _ir_error_meta(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Provider busy: {exc}",
                headers={
                    "Retry-After": ir_retry_after
                    or str(get_settings().queue_rejected_retry_after_s),
                    "X-IR-Error-Classification": IR_QUEUE_REJECTED,
                },
            ) from exc
        if byo:
            raise _byo_failed(req, request, byo, exc) from exc
        raise HTTPException(status_code=502, detail=f"Provider call failed: {exc}") from exc

    usage = _usage_from_result(stream_result)
    usage.max_tokens = _sent_max_tokens(agent)
    for key, value in _sent_sampling(agent).items():
        setattr(usage, key, value)
    usage.finish_reason = ResponseExtractor.finish_reason(stream_result)
    pp_tps, tg_tps = _compute_rates(usage, t0, first_delta_t, last_delta_t, delta_count)
    usage.pp_tps = round(pp_tps, 1) if pp_tps is not None else None
    usage.tg_tps = round(tg_tps, 1) if tg_tps is not None else None
    _emit_byo_cost(req, byo, stream_result, usage, time.perf_counter() - t0)
    resolved_model, resolved_provider = _resolved_from_result(stream_result)
    return ChatCompletionsResponse(
        output=output,
        provider=req.provider,
        model=req.model,
        usage=usage,
        resolved_model=resolved_model,
        resolved_provider=resolved_provider,
        **ResponseExtractor.extract_served_fields(stream_result),
    )


async def _sse_stream(  # noqa: PLR0915 -- SSE stream emits a fixed sequence of frame kinds (deltas, result, meta, [DONE]); each is its own short block
    agent: Agent,
    prompt: str,
    req: ChatCompletionsRequest,
    byo: dict[str, Any] | None = None,
    *,
    on_byo_error: Callable[[BaseException], HTTPException] | None = None,
) -> AsyncIterator[bytes]:
    r"""Stream SSE chunks from a pydantic-ai Agent.

    Emits a ``:keepalive\\n\\n`` SSE comment every
    ``SSE_HEARTBEAT_INTERVAL_S`` seconds when no real delta has flowed,
    so silent gaps (model warmup, slow first-token, long inter-token pauses)
    don't trip intermediary idle-stream timers (Cloudflare edge ~100s,
    cloudflared, Envoy). SSE comment lines are spec-ignored by clients.

    After streaming completes, emits one `meta` frame carrying: the resolved
    backend (model + provider the upstream actually reported so failover-chain
    providers reveal which backend served the call), token counts, and two
    performance rates the UI renders under the assistant bubble:

    - ``pp_tps`` = prompt_tokens / time_to_first_token (prompt-processing rate)
    - ``tg_tps`` = output_tokens / (last_delta - first_delta) (generation rate);
      omitted when the observed window is below ``MIN_TG_WINDOW_S`` or fewer
      than two distinct delta flushes were seen, since either condition means
      the upstream chain coalesced flushes and we have no honest decode-rate
      window.
    """
    first_delta_t: float | None = None
    last_delta_t: float | None = None
    delta_count: int = 0
    t0 = time.perf_counter()
    # Holder mutated by the producer task so the consumer can compute meta
    # after the producer's `async with` has cleanly exited.
    holder: dict[str, Any] = {"stream": None, "errored": False}
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _producer() -> None:
        """Drive the upstream stream into the queue. None = end-of-stream sentinel."""
        nonlocal first_delta_t, last_delta_t, delta_count
        try:
            async with agent.run_stream(prompt) as stream:
                holder["stream"] = stream
                if req.output_schema is None:
                    # Plain-text path: emit char-by-char deltas as fast as
                    # the model produces them. `stream_text(delta=True)` is
                    # the canonical pydantic-ai accessor for this — but it
                    # ONLY works when output_type is `str`. For typed
                    # output_type (bool/list[…]/dict[…]) pydantic-ai raises
                    # ``stream_text() can only be used with text responses``,
                    # so the typed path below uses a different accessor.
                    async for delta in stream.stream_text(delta=True):
                        now = time.perf_counter()
                        if first_delta_t is None:
                            first_delta_t = now
                        last_delta_t = now
                        delta_count += 1
                        payload = {"delta": delta, "provider": req.provider, "model": req.model}
                        await queue.put(f"data: {json.dumps(payload)}\n\n".encode())
                else:
                    # Typed-output path: deltas would be raw JSON characters
                    # which aren't useful to the renderer (it waits on the
                    # `result` frame for the validated value). Skip the delta
                    # loop entirely and let the stream complete; we emit a
                    # single ``progress`` heartbeat so intermediaries observe
                    # activity and the bubble can show a "thinking" state.
                    first_delta_t = time.perf_counter()
                    progress_payload = {
                        "progress": "generating",
                        "provider": req.provider,
                        "model": req.model,
                    }
                    await queue.put(f"data: {json.dumps(progress_payload)}\n\n".encode())
                    # Drain the stream by reading the final structured output.
                    # ``get_output()`` awaits the full response and returns the
                    # pydantic-validated value matching the configured
                    # output_type. This is also what the result-frame block
                    # below captures into holder["output"].
                    last_delta_t = time.perf_counter()
                # All deltas (text) or structured output (typed) consumed
                # inside the live stream context — capture the validated
                # typed output iff the caller asked for one. We don't emit
                # `result` for plain-text (output_schema=None): the raw deltas
                # already carry everything the bubble needs and adding a
                # redundant frame would force the renderer to special-case a
                # no-op echo. For typed schemas (boolean/list/object) the
                # typed renderer relies on this single `result` frame for the
                # validated value.
                if req.output_schema is not None:
                    try:
                        holder["output"] = await stream.get_output()
                    except Exception as exc:
                        # Validation failure here is non-fatal for the client —
                        # any progress frame already shipped, so the bubble
                        # has SOMETHING. The typed renderer will fall through
                        # to TextResult when `result` is absent.
                        logger.warning(
                            "chat_completions_get_output_failed",
                            extra={"error": str(exc)},
                        )
        except Exception as exc:
            holder["errored"] = True
            err: dict[str, Any]
            if byo is not None and on_byo_error is not None:
                # Owner ruling 2026-10-01: a failure on the TENANT'S OWN key is the
                # same BYO_PROVIDER_CALL_FAILED envelope the non-stream path
                # returns, with the key scrubbed — never the raw vendor text
                # (google puts the key in the URL; validator N-a).
                http_exc = on_byo_error(exc)
                message = str(http_exc.detail)
                logger.warning("chat_completions_stream_failed", extra={"error": message})
                # ``error`` STAYS a string: fr's SSE parser (streamChat.ts
                # ``parseFrame``) recognises an error frame only by
                # ``typeof obj.error === 'string'`` and silently skips anything
                # else, which would leave an empty bubble and no error at all
                # (validator 3 D4). The envelope fields ride BESIDE it.
                err = {
                    "error": message,
                    "error_code": getattr(http_exc, "error_code", "BYO_PROVIDER_CALL_FAILED"),
                    "status": http_exc.status_code,
                    "provider": req.provider,
                }
                details = getattr(http_exc, "error_details", None)
                if details:
                    err["details"] = details
            elif _is_queue_rejected(exc):
                # The 200 + keepalives are already on the wire before IR's queue
                # refusal (~30s slot wait) can arrive, so a 503 is impossible here.
                # Same frame shape as every other typed error (#316): ``error``
                # STAYS a string, the envelope rides BESIDE it.
                logger.warning("chat_completions_stream_failed", extra={"error": str(exc)})
                err = {
                    "error": f"Provider busy: {exc}",
                    "error_code": "PROVIDER_BUSY",
                    "status": 503,
                    "provider": req.provider,
                    "type": "provider_busy",
                    "classification": IR_QUEUE_REJECTED,
                    "details": {"retry_after": _stream_retry_after_s(exc)},
                }
            else:
                # Upstream text can echo a credential-bearing URL (issue #327).
                message = redact_secrets_in_text(str(exc))
                logger.warning("chat_completions_stream_failed", extra={"error": message})
                err = {"error": message, "provider": req.provider}
            await queue.put(f"data: {json.dumps(err, default=str)}\n\n".encode())
        finally:
            await queue.put(None)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_INTERVAL_S)
            except TimeoutError:
                yield b": keepalive\n\n"
                continue
            if item is None:
                break
            yield item

        if not holder["errored"]:
            stream_result = holder["stream"]
            # Emit the validated typed value before `meta` so the renderer can
            # finalise the bubble (boolean → pill, list → bulleted list,
            # object → key/value table) without re-parsing the accumulated
            # deltas. `schema` is the same fragment the caller sent — echoed
            # back so the renderer doesn't have to remember it across
            # re-renders or history navigation. Only emitted when both a
            # schema was requested and the producer captured a validated
            # value (`get_output()` succeeded); missing in plain-text and
            # validation-failure paths.
            if req.output_schema is not None and "output" in holder:
                result_frame = {
                    "result": holder["output"],
                    "schema": req.output_schema,
                }
                # default=str catches exotic types pydantic-ai validators may
                # return (datetimes, enums); the four UI shapes
                # (bool / list / dict / str) serialise natively without it.
                yield f"data: {json.dumps(result_frame, default=str)}\n\n".encode()
            resolved_model, resolved_provider = _resolved_from_result(stream_result)
            usage = _usage_from_result(stream_result)
            pp_tps, tg_tps = _compute_rates(usage, t0, first_delta_t, last_delta_t, delta_count)
            _emit_byo_cost(req, byo, stream_result, usage, time.perf_counter() - t0)
            meta_body: dict[str, Any] = {}
            if resolved_model:
                meta_body["resolved_model"] = resolved_model
            if resolved_provider:
                meta_body["resolved_provider"] = resolved_provider
            meta_body.update(ResponseExtractor.extract_served_fields(stream_result))
            if usage.input_tokens is not None:
                meta_body["input_tokens"] = usage.input_tokens
            if usage.output_tokens is not None:
                meta_body["output_tokens"] = usage.output_tokens
            sent_max_tokens = _sent_max_tokens(agent)
            if sent_max_tokens is not None:
                meta_body["max_tokens"] = sent_max_tokens
            # Same "present only when sent" rule as max_tokens; ``is not None``
            # so a sent 0 (T=0, seed=0) is emitted, not dropped.
            meta_body.update(
                {key: value for key, value in _sent_sampling(agent).items() if value is not None}
            )
            finish_reason = ResponseExtractor.finish_reason(stream_result)
            if finish_reason is not None:
                meta_body["finish_reason"] = finish_reason
            if pp_tps is not None:
                meta_body["pp_tps"] = round(pp_tps, 1)
            if tg_tps is not None:
                meta_body["tg_tps"] = round(tg_tps, 1)
            if meta_body:
                yield f"data: {json.dumps({'meta': meta_body})}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        if not producer_task.done():
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer_task


def _stream_retry_after_s(exc: BaseException) -> int:
    """IR's Retry-After as int seconds; the configured fallback when absent/non-numeric."""
    _, ir_retry_after = _ir_error_meta(exc)
    if ir_retry_after is not None and ir_retry_after.strip().isdigit():
        return int(ir_retry_after.strip())
    return get_settings().queue_rejected_retry_after_s


def _compute_rates(
    usage: ChatCompletionsUsage,
    t0: float,
    first_delta_t: float | None,
    last_delta_t: float | None,
    delta_count: int,
) -> tuple[float | None, float | None]:
    """Derive (pp_tps, tg_tps) from measured stream timings and token counts.

    Either rate is ``None`` when its inputs are unavailable: backends that
    don't report token usage or streams that produced no deltas simply drop
    the corresponding field from the meta frame.

    ``tg`` additionally requires at least two distinct delta flushes
    (``delta_count >= 2``) and an inter-delta window of at least
    ``MIN_TG_WINDOW_S``. Below those thresholds the upstream chain coalesced
    flushes into a near-instant burst — wall time then reflects scheduler
    overhead, not real decode pace, and dividing token count by it produces
    nonsense values (millions of t/s). In that regime we drop the field.
    """
    pp: float | None = None
    tg: float | None = None
    if first_delta_t is not None and usage.input_tokens:
        ttft = first_delta_t - t0
        if ttft > 0:
            pp = usage.input_tokens / ttft
    if (
        first_delta_t is not None
        and last_delta_t is not None
        and usage.output_tokens
        and delta_count >= 2
    ):
        gen_dur = last_delta_t - first_delta_t
        if gen_dur >= MIN_TG_WINDOW_S:
            tg = usage.output_tokens / gen_dur
    return pp, tg


def _usage_from_result(result: Any) -> ChatCompletionsUsage:
    """Best-effort token-usage extraction from a pydantic-ai RunResult."""
    usage = ChatCompletionsUsage()
    try:
        u = result.usage()
    except Exception:
        return usage
    if u is None:
        return usage

    def _as_int(*names: str) -> int | None:
        for name in names:
            value = getattr(u, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    usage.input_tokens = _as_int("input_tokens", "request_tokens")
    usage.output_tokens = _as_int("output_tokens", "response_tokens")
    usage.total_tokens = _as_int("total_tokens")
    return usage


def _resolved_from_result(result: Any) -> tuple[str | None, str | None]:
    """Return (model_name, provider_name) from the final ModelResponse.

    pydantic-ai records the actual upstream name on each ModelResponse;
    for failover-chain providers (inference-router) this reveals which
    backend in the chain served the call. Returns (None, None) when the
    result object is unavailable or carries no ModelResponse.
    """
    try:
        messages = result.all_messages() if result is not None else []
    except Exception:
        return None, None
    last_response = None
    for msg in messages:
        if type(msg).__name__ == "ModelResponse":
            last_response = msg
    if last_response is None:
        return None, None
    return (
        getattr(last_response, "model_name", None),
        getattr(last_response, "provider_name", None),
    )
