"""PydanticAI agent implementation for workflow system.

This module provides a PydanticAI-based agent that can execute tasks using various
LLM providers (OpenAI, DeepSeek, Groq, Cohere, MLX, inference-router) with support
for tools and MCP (Model Context Protocol) servers.
"""

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack, suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast
from uuid import UUID, uuid7

import httpx
from cohere import AsyncClientV2 as AsyncCohereClient
from dotenv import load_dotenv
from groq import AsyncGroq
from openai import AsyncOpenAI
from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, Tool, models
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelSettings, create_async_http_client
from pydantic_ai.models import anthropic as _pai_anthropic
from pydantic_ai.models import openai as _pai_openai
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.groq import GroqModel, GroqModelSettings
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.cohere import CohereProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.xai import XaiProvider

from src.security.request_overrides import (
    OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
    OutputValidatorCodeNotSupportedError,
    has_validator_code,
    pinned_tenant_transport,
    refuse_owner_infra_request,
)
from src.telemetry import get_metrics_client
from src.utils.xgrammar_schema import (
    BOUND_KEYWORDS,
    SCHEMA_BOUND_VIOLATION,
    SchemaBoundViolationError,
    strip_xgrammar_unsupported,
    stripped_keywords,
)
from src.worker.errors import extract_error_detail

# The provider registry is the single source of truth for which providers
# exist, which need a credential, and which env var carries it. It lives in a
# dependency-free module outside src.worker.agents so lightweight consumers
# (API schemas, worker templates, the optional-dependency wrapper) can import it
# without pulling in this module's heavy pydantic_ai/cohere/groq/openai chain.
# Re-exported here because ``from ...pydantic_ai_agent import ModelProvider`` is
# the historical import path used across src/ and tests/.
from src.worker.provider_registry import (  # noqa: F401 -- re-exported for existing call sites
    LOCAL_PROVIDERS,
    PROVIDER_METADATA,
    ModelProvider,
)

from .base_agent import (
    SENT_USAGE_METRICS_ATTR,
    BaseAgent,
    TokenUsageMetrics,
)
from .base_agent import AgentRunResult as WorkerAgentRunResult
from .tool_template import ToolTemplate
from .xai_stream import PinnedContextXaiModel

load_dotenv()

from src.utils.log_bounds import MAX_LOGGED_PAYLOAD_CHARS, summarize_for_log

logger = logging.getLogger(__name__)

# Prompt/response log bounding now lives in src/utils/log_bounds.py so the
# prompt side (here) and the response side (api/workers/message_endpoints.py)
# cannot drift apart -- bounding only one just moves the flood to the other.
_MAX_LOGGED_PROMPT_CHARS = MAX_LOGGED_PAYLOAD_CHARS
_summarize_for_log = summarize_for_log


# The agent-execution wall-clock budget now lives in Settings
# (``agent_run_timeout_s`` / ``agent_run_timeout_mcp_s``, env vars
# AGENT_RUN_TIMEOUT_S / AGENT_RUN_TIMEOUT_MCP_S) so it can be re-baselined
# without an image rebuild. Values are unchanged (600.0 / 900.0).
#
# It is read through ``_get_settings()`` at call time, NOT captured into a
# module-level constant at import time — a constant would freeze whatever the
# environment happened to be when this module was first imported and would make
# the env vars a lie. ``Settings`` enforces the ordering invariant
# (ir_http_read_timeout_s <= the agent cap); see ``_agent_run_timeout``.


class MLXCompletionsWrapper:
    """Wrapper for chat.completions that strips <think> tags."""

    def __init__(self, original_completions: Any) -> None:
        self._completions = original_completions

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Strip <think> tags from MLX completion responses."""
        logger.info("MLXCompletionsWrapper.create() called - about to strip <think> tags")
        response = await self._completions.create(*args, **kwargs)

        # Strip <think>...</think> tags from response content
        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                if hasattr(choice, "message") and choice.message.content:
                    original_content = choice.message.content
                    # Remove thinking tags using regex
                    cleaned_content = re.sub(
                        r"<think>.*?</think>\s*",
                        "",
                        choice.message.content,
                        flags=re.DOTALL,
                    )
                    # Update the message content
                    choice.message.content = cleaned_content.strip()
                    logger.info(
                        f"MLX response cleaned: original length={len(original_content)}, "
                        f"cleaned length={len(cleaned_content)}, had <think> tags={('<think>' in original_content)}"
                    )

        return response


class MLXChatWrapper:
    """Wrapper for chat that provides custom completions with <think> tag stripping."""

    def __init__(self, original_chat: Any) -> None:
        self._chat = original_chat
        self.completions = MLXCompletionsWrapper(original_chat.completions)


class MLXAsyncOpenAI(AsyncOpenAI):
    """Wrapper for AsyncOpenAI that strips <think> tags from MLX responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Wrap the chat attribute
        # MLXChatWrapper duck-types AsyncChat (provides .completions attribute)
        cast("Any", self).chat = MLXChatWrapper(self.chat)


class MLXOpenAIProvider(OpenAIProvider):
    """Custom OpenAI provider that uses MLXAsyncOpenAI client."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "mlx-local",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize provider with custom MLX client.

        ``http_client`` carries the shared inference-router identity hook when
        the MLX instance URL is inference-router (it is, in prod).
        """
        super().__init__(base_url=base_url, api_key=api_key)
        # Replace the client with our custom wrapper.
        # max_retries=0 for the same reason as InferenceRouterOpenAIProvider (see
        # the long comment there): the SDK retries read timeouts, and mlx is a
        # self-hosted backend on our own Apple-silicon boxes, so a retried
        # timeout is a second full decode on the machine that just failed to
        # keep up. This site needs the pin MORE than the IR one, not less: it
        # passes no http_client, so its read budget is the SDK's own hardcoded
        # DEFAULT_TIMEOUT (600s, openai/_constants.py:9) rather than
        # ir_http_read_timeout_s — which means Settings._validate_timeout_ordering
        # cannot see it and cannot protect it. Lowering agent_run_timeout_s below
        # 600 would silently re-open the retry window here while the invariant
        # check stayed green.
        self._client = MLXAsyncOpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, http_client=http_client
        )


# The fleet-wide "no tenant" sentinel, minted by inference-router
# (src/services/metering.py) whenever X-Tenant-Id is absent OR empty. The value
# is a contract, not a library, so worker_management restates it rather than
# depending on inference-router -- but it restates it ONCE, in identity_context,
# and this alias points at that single copy. Two literals of the same sentinel
# in two outbound paths is exactly how one of them would later be "tidied" to a
# different value and start laundering unattributable calls.
from src.api.identity_context import NIL_TENANT_ID as _NIL_TENANT_ID


def _drop_unattributed_header(request: httpx.Request, name: str) -> None:
    """Remove an identity header rather than send it empty.

    The client-level default headers put these on every request, so "do not
    set it" is not enough -- an unresolved value has to be actively removed or
    the constructor's value (possibly "") rides along. httpx's Headers raises
    KeyError on a missing key, hence the guard.
    """
    if name in request.headers:
        del request.headers[name]


def _tenant_header_value(resolved_tenant: str) -> str:
    """Render the ``X-Tenant-Id`` value for an ALREADY-RESOLVED, real tenant.

    ⛔ OUTBOUND, ``X-Tenant-Id`` CARRIES THE RAW TENANT UUID AND NOTHING ELSE.
    Never a ``Bearer`` form, never a licence token. The licence travels in its
    own header, ``X-License-Id``, set separately by both callers.

    ⚠️ This function used to take a ``_license_token`` second argument and drop
    it on the floor, which was itself a correction of an earlier version that
    returned ``f"Bearer {license_token}"``. Both the parameter and the whole
    ``OUTBOUND_LICENSE_TOKEN`` mechanism behind it were RETIRED 2026-09-11 --
    the value reached no wire from here or anywhere else, so it was a live
    bearer credential in the pod environment with no consumer. Do not
    reintroduce either. The two directions are not the same contract:

    * INBOUND (unchanged, still true): a caller authenticates to *us* with
      ``X-Tenant-Id: Bearer <token>``; ``AuthenticationMiddleware.
      _extract_license_token`` (src/api/auth_middleware.py) reads it from that
      header and from nowhere else. It must never be accepted on
      ``Authorization`` -- that is the 2026-04-26 production 401s, where FR's
      Keycloak bearer was introspected as a licence token and came back
      ``token_not_found``. None of that changes.

    * OUTBOUND (what this helper builds): the only destinations these two call
      sites can reach are inference-router -- ``INFERENCE_ROUTER_URL``,
      ``VLLM_URL`` (repointed at IR by gitops #2999) and the ``base_url``
      override of all 238 worker rows all resolve to
      an in-cluster inference-router Service origin -- and, on a
      dev/unset fallback, a bare vLLM. **Neither introspects anything.** An
      east-west call would therefore deliver a raw bearer token into IR's
      tenant field verbatim.

    What that cost while it lasted: IR's ``services/metering.py`` classifies an
    ``Authorization``-shaped tenant as ``credential_shaped`` and **DISCARDS the
    cost event entirely** -- not null-tenanted, discarded -- so the spend is
    unbillable AND invisible. And it is strictly worse for authorization, not
    better: ``require_caller`` / ``require_licensed_caller`` validate
    ``X-Tenant-Id`` as a **uuid**, so the ``Bearer`` form would HTTP 400 the
    moment ``INFERENCE_ROUTER_REQUIRE_CALLER_LICENSE`` is enabled.

    No authorization is lost, because nothing downstream ever consumed it. The
    licence keeps travelling in ``X-License-Id`` -- the header inference-router
    actually reads (``security.py``, ``Header(alias="X-License-Id")``) and the
    one ``require_licensed_caller`` gates on.

    ⚠️ This helper does NOT decide whether to send the header at all. The
    caller must already have established that the tenant is real (truthy and
    not the nil sentinel). An unattributable call stays unattributable and
    stays loud.

    Defined ONCE and used by both the hook and the constructor's default
    headers on purpose. Two copies of a header contract diverging is the
    documented origin of the `_create_vllm_model` identity gap above.
    """
    return resolved_tenant


async def _set_ir_tenant_bearer(request: httpx.Request, resolved_tenant: str) -> None:
    """``Authorization: Bearer <tenant bundle token>`` — IR origin only (plan §2.4).

    Behind ``WM_IR_BUNDLE_TOKENS`` (default off ⇒ this is a no-op and the wire
    is unchanged). The token is the TENANT's own short-lived inference-router
    token, so it must never reach any other origin: a vendor API or a bare vLLM
    would receive a live credential for the tenant. ``X-Tenant-Id`` is NOT
    touched here and stays the raw uuid (see :func:`_tenant_header_value`).

    The decision (flag, tenant proof, exact origin, fail-soft) lives ONCE in
    :func:`src.services.ir_bundle_tokens.ir_authorization`, shared with the
    model-listing call, so the two WM→IR paths cannot diverge. When no token
    can be had, the hook then drops the SDK placeholder
    (``ir_bundle_tokens.drop_placeholder_bearer``): never ``Bearer internal``.
    """
    from src.services import ir_bundle_tokens

    auth = await ir_bundle_tokens.ir_authorization(str(request.url), resolved_tenant)
    if auth:
        request.headers["Authorization"] = auth


IR_CALL_MISSING_PROJECT_TOTAL = Counter(
    "wm_ir_call_missing_project_total",
    "worker_management -> inference-router calls forwarded WITHOUT X-Project-Id because "
    "the inbound caller sent none, by inbound caller (verified client id, else asserted "
    "app id, else 'unknown'). WM never invents a project id; fix the named caller.",
    ["caller"],
)


def _count_missing_project(caller_client_id: str, app_id: str) -> None:
    """Count one IR call whose inbound caller sent no X-Project-Id."""
    caller = (caller_client_id or app_id or "unknown").strip()[:64] or "unknown"
    IR_CALL_MISSING_PROJECT_TOTAL.labels(caller=caller).inc()


# Per-WORKER view of the same omission, for BOTH forwarded identity headers.
# `wm_ir_call_missing_project_total` above names the inbound caller; this one
# names the WM worker (job type) the call ran under, so "which upstream app must
# fix its X-App-Id / X-Project-Id" and "which worker's traffic is unattributed"
# are both answerable. WM NEVER invents either value -- it only counts the miss.
IR_IDENTITY_HEADERS = ("app_id", "project_id")
IR_IDENTITY_UNKNOWN_WORKER = "<unknown>"
IR_IDENTITY_MISSING_TOTAL = Counter(
    "wm_ir_identity_missing_total",
    "worker_management -> inference-router calls forwarded WITHOUT an identity header "
    "because the inbound caller sent none (header=app_id|project_id), by WM worker. "
    "WM never invents X-App-Id or X-Project-Id; fix the calling app.",
    ["header", "worker"],
)


def preinit_ir_identity_metrics() -> None:
    """Materialise the WM->IR identity series at 0 so they exist from startup.

    Called from the app lifespan. Worker names are runtime data, so the
    pre-initialised child is the ``<unknown>`` worker leg -- enough for the
    metric family (and an ``absent()`` / ``increase()`` rule) to exist before
    the first miss.
    """
    for header in IR_IDENTITY_HEADERS:
        IR_IDENTITY_MISSING_TOTAL.labels(header=header, worker=IR_IDENTITY_UNKNOWN_WORKER)


preinit_ir_identity_metrics()


def _count_missing_identity(header: str, worker_name: str) -> None:
    """Count + WARN one IR call forwarded without ``header`` (never substituted)."""
    from src.api.identity_context import (
        get_caller_client_id_from_context,
        get_correlation_id_from_context,
        get_subject_id_from_context,
    )

    worker = worker_name or IR_IDENTITY_UNKNOWN_WORKER
    IR_IDENTITY_MISSING_TOTAL.labels(header=header, worker=worker).inc()
    logger.warning(
        "wm_ir_identity_missing header=%s worker=%s caller=%s subject=%s correlation=%s -- "
        "inbound request carried no %s; forwarded to inference-router WITHOUT it. "
        "WM never invents one: fix the calling app.",
        header,
        worker,
        get_caller_client_id_from_context() or "<unknown>",
        get_subject_id_from_context() or "<unknown>",
        get_correlation_id_from_context() or "<none>",
        header,
    )


def make_identity_event_hook(
    *,
    worker_name: str,
    worker_version: str,
    fallback_tenant_id: str,
    fallback_license_id: str,
    fallback_subject_id: str,
    pinned_provider: str = "",
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Build the httpx request hook that stamps caller identity on every call.

    ⛔ THIS EXISTS BECAUSE TWO CLIENTS DIVERGED AND ONE WAS FORGOTTEN, AND THAT
    DIVERGENCE WAS THE BUG. `VLLM_URL` was repointed at inference-router by
    gitops #2999, so `_create_vllm_model` became an inference-router client
    wearing a vllm label. Someone noticed half of it and added `X-Worker-Name`
    there -- but not the identity triple. Measured live 2026-08-27 on
    `principle_consistency`:

        worker_name: principle_consistency      <- the half that was added
        tenant_id:   00000000-...-000000000000  <- nil
        license_id:  unknown
        app_id:      unknown

    A stripped-header case loses ONLY the tenant. Losing license and app too is
    the signature of a client that never sent identity at all.

    It is also why worker_management #198 -- which correctly threaded the real
    tenant all the way to `config.tenant_id` -- changed nothing for this worker:
    that fix landed in `InferenceRouterOpenAIProvider`, a class this agent never
    instantiates. Post-deploy the nil ratio went 22.75% -> 40.42%.

    So the hook is defined ONCE and both clients take it. A second copy is how
    this happened; there will not be a third.
    """

    async def _identity_event_hook(request: httpx.Request) -> None:
        # Read identity ContextVars set by AuthenticationMiddleware in
        # the current async task. Empty-string default keeps the
        # contract that headers are always present, never missing.
        from src.api.identity_context import (
            EFFECTIVE_TENANT_HEADER,
            get_app_id_from_context,
            get_caller_client_id_from_context,
            get_correlation_id_from_context,
            get_effective_actor_id,
            get_license_id_from_context,
            get_project_id_from_context,
            get_provider_pin_from_context,
            get_spend_scope_from_context,
            get_stage_from_context,
            get_subject_id_from_context,
            get_tenant_id_from_context,
            get_trace_and_prompt_headers,
            is_replay_mode_from_context,
        )

        ctx_tenant = get_tenant_id_from_context()
        ctx_license = get_license_id_from_context()
        ctx_subject = get_subject_id_from_context()
        # Resolve, then SET ONLY WHAT RESOLVED. Writing an empty string here
        # is what produced `X-Tenant-Id: ""` on the wire.
        resolved_tenant = ctx_tenant or fallback_tenant_id
        resolved_license = ctx_license or fallback_license_id
        resolved_subject = ctx_subject or fallback_subject_id
        if resolved_tenant and resolved_tenant != _NIL_TENANT_ID:
            # ALWAYS the resolved tenant uuid, never `Bearer <token>`.
            # inference-router -- the only destination reachable from here --
            # has no introspection middleware and DISCARDS a cost event whose
            # tenant is credential-shaped. See `_tenant_header_value`.
            request.headers["X-Tenant-Id"] = _tenant_header_value(resolved_tenant)
            await _set_ir_tenant_bearer(request, resolved_tenant)
        else:
            # ⛔ REFUSED on the inference-router origin (owner ruling
            # 2026-10-07): WM never calls IR on its own behalf, so a call with
            # no real end tenant is not made at all -- typed
            # IRTenantRequiredError, ERROR log, wm_ir_tenantless_call_refused_total.
            # No WM/platform tenant is ever substituted.
            from src.services.ir_bundle_tokens import is_ir_origin, refuse_tenantless_ir_call

            if is_ir_origin(str(request.url)):
                refuse_tenantless_ir_call(
                    "completion", worker_name=worker_name, url=str(request.url)
                )
            # Non-IR origin (a bare vLLM in dev): loud, not silent, and never
            # substituted. There is no default tenant and no sentinel.
            _drop_unattributed_header(request, "X-Tenant-Id")
            logger.error(
                "outbound_tenant_unresolved: worker=%s ctx=%r fallback=%r -- "
                "X-Tenant-Id omitted; this call cannot be attributed to a "
                "customer and inference-router will meter it against the nil "
                "sentinel. Fix the caller, not this header.",
                worker_name or "<unknown>",
                ctx_tenant,
                fallback_tenant_id,
            )
        # Whatever happened above, the SDK placeholder (IR_CLIENT_API_KEY) never
        # reaches inference-router as if it were a licence.
        from src.services.ir_bundle_tokens import drop_placeholder_bearer

        drop_placeholder_bearer(str(request.url), request.headers, worker_name=worker_name)
        if resolved_license:
            request.headers["X-License-Id"] = resolved_license
        else:
            # Loud, and NEVER substituted -- the same rule as the tenant branch
            # above, one level down in severity. The difference in level is the
            # whole point of this comment, so nobody "makes it consistent".
            #
            # An absent TENANT is always a defect: every branch of
            # `AuthenticationMiddleware` resolves one (bypass ->
            # BYPASS_TENANT_UUID, token -> introspected, trusted-network ->
            # `resolve_tenant_id`-or-refuse), so an empty tenant means
            # something broke. Hence ERROR up there.
            #
            # An absent LICENCE is, today, an EXPECTED state on two of those
            # three branches, by explicit design:
            #   * the tokenless trusted-network branch stamps a NULL licence --
            #     "Trusted-network path has no introspected user identity;
            #     leave subject_id / license_id unset" (auth_middleware.py); and
            #   * the delegating branch sets `license_id = None` deliberately,
            #     because the introspected licence belongs to the SERVICE
            #     ACCOUNT, not to the re-scoped customer tenant. Keeping it
            #     billed 3,411 rows against nine tenants that did not own the
            #     licence, and drew 10.8x its monthly quota in half a day
            #     (measured 2026-09-02; see that branch's comment).
            # So this is not a bug in THIS service, and it must not be ERROR:
            # it would be indistinguishable from the tenant case in alerting.
            #
            # What WAS wrong is that the absence was intended while its
            # consequence -- an outbound GPU call nobody can be invoiced for --
            # was recorded NOWHERE. The tenant branch shouted; this one said
            # nothing at all. That silence is why `principle_consistency` ran
            # 2,672 of 3,502 calls unbillable over 30h without leaving one line
            # naming the worker or the tenant that caused it.
            #
            # ⛔ DO NOT "fix" this by manufacturing a licence. Not from the
            # tenant, not from a fallback, not from a forwarded header. The
            # forwarded-header version is refused in as many words in
            # `_audit_unauthenticated_tenant_assertion`: honouring an
            # `X-License-Id` from an admitted peer "would let any admitted peer
            # assert a licence as well as a tenant -- a strictly larger hole
            # wearing the costume of a fix". A tenant->licence lookup is the
            # same DEFAULT_TENANT_ID-shaped substitution: it makes the row look
            # attributed and bills a real customer for spend nobody proved was
            # theirs. Wrong-but-complete is strictly worse than missing.
            #
            # The only real fix is upstream and OUTSIDE this repo: the calling
            # service presents its own licence token in
            # `X-Tenant-Id: Bearer <token>` (as gitops #3274 did for
            # `requirements`), at which point the token branch resolves a real
            # licence and this warning stops firing for that caller. Naming the
            # worker, the tenant and the correlation id is what makes that
            # caller findable -- which is the entire job of this line.
            _drop_unattributed_header(request, "X-License-Id")
            logger.warning(
                "outbound_license_unresolved: worker=%s tenant=%s correlation=%s "
                "subject_present=%s -- X-License-Id omitted; this inference-router "
                "call cannot be invoiced against any licence and will be metered "
                "unattributed. Not substitutable here: fix the CALLER so it "
                "presents its own licence token in 'X-Tenant-Id: Bearer <token>' on its "
                "INBOUND call to this service. That is the inbound contract only -- "
                "our OUTBOUND calls send the tenant UUID in X-Tenant-Id and the "
                "licence in X-License-Id, and must never send the Bearer form.",
                worker_name or "<unknown>",
                resolved_tenant or "<unresolved>",
                get_correlation_id_from_context() or "<none>",
                bool(resolved_subject),
            )
        if resolved_subject:
            request.headers["X-Subject-Id"] = resolved_subject
        else:
            # Deliberately NOT given a log line of its own, and this is the
            # note that says so rather than leaving a third silent branch.
            # Billing attributes on `license_id` ONLY
            # (`billing_service src/models/db/rollups.py`,
            # `license_attributed_clause`), so a missing subject costs no
            # revenue and shares its cause with the licence above -- it is
            # reported as `subject_present=` on that warning instead of
            # doubling the log volume of the same event.
            _drop_unattributed_header(request, "X-Subject-Id")
        # Re-stamp the trusted effective-tenant side-channel so any downstream
        # hop that re-delegates (subject ∈ AUTH_DELEGATING_SUBJECTS) attributes
        # to the run tenant, not WM's SA tenant. IR today reads the plain
        # X-Tenant-Id, so this is defensive for multi-hop re-delegation. Only
        # set when present — never clobber with an empty value.
        if ctx_tenant:
            request.headers[EFFECTIVE_TENANT_HEADER] = ctx_tenant
        # Forward cost-correlation downstream so inference-router (which meters
        # platform-routed calls) emits the same correlation_id/stage on its
        # cost event. Only set when present — never clobber with empty values.
        ctx_correlation = get_correlation_id_from_context()
        ctx_stage = get_stage_from_context()
        if ctx_correlation:
            request.headers["X-Correlation-Id"] = ctx_correlation
        if ctx_stage:
            request.headers["X-Stage"] = ctx_stage
        # Forward the calling application's id, verbatim, when the inbound
        # caller stamped one. ⛔ NEVER default this: inference-router fans in
        # across 19 tenants and worker_management itself fans in across many
        # upstream apps (n8n, requirements_management, asset-evaluator,
        # w2v_orchestrator, ...) — a guessed app_id here would collapse all of
        # them into one bucket, which is strictly worse for attribution than
        # an absent header (IR's own X-App-Id read defaults to "unknown" and
        # leaves it there; do not "fix" that by inventing a value upstream).
        ctx_app_id = get_app_id_from_context()
        if ctx_app_id:
            request.headers["X-App-Id"] = ctx_app_id
        else:
            _drop_unattributed_header(request, "X-App-Id")
            _count_missing_identity("app_id", worker_name)
        # X-Job-Id: the inbound caller's own X-Job-Id when it sent one (relayed),
        # else the WM worker name (job type, e.g. ks-relationship-extractor).
        # Optional for the router; the SAME value WM's own cost events carry.
        from src.api.identity_context import get_effective_job_id

        job_id = get_effective_job_id(worker_name)
        if job_id:
            request.headers["X-Job-Id"] = job_id
        # Forward the kanban project this run belongs to, verbatim, so
        # inference-router can price spend per project (it already reads
        # X-Project-Id at its header-reading site and validates it there — do
        # NOT add a second validator here; two validators are two things that
        # can disagree). Same omit-when-empty rule as every header above: an
        # absent header is honest, an empty one is a claim that the project is
        # nothing. ⛔ NEVER default or guess a project id, for the same reason
        # spelled out for X-App-Id directly above.
        #
        # Owner ruling 2026-10-07: WM NEVER fabricates a project id (not the
        # tenant id, not a derived one) -- project-less calling apps send their
        # own tenant-level value. A missing inbound project is forwarded as
        # absent and COUNTED per inbound caller so the offending app is visible.
        ctx_project_id = get_project_id_from_context()
        if ctx_project_id:
            request.headers["X-Project-Id"] = ctx_project_id
        else:
            _drop_unattributed_header(request, "X-Project-Id")
            _count_missing_project(get_caller_client_id_from_context(), ctx_app_id)
            _count_missing_identity("project_id", worker_name)
        # Forward the spend SCOPE — the reporting-only sibling of X-Project-Id
        # that lets billing tell "we failed to attribute this spend to a
        # project" apart from "this is tenant-wide work by design and there is
        # nothing to attribute it to". Already validated to the closed
        # {tenant, project} set by `set_spend_scope_context`; anything else was
        # dropped there, so an empty value here means "nobody usably said" and
        # the same omit-when-empty rule applies.
        #
        # ⛔ REPORTING ONLY: nothing downstream may price, mark up or route on
        # this, and ⛔ it is stamped in its OWN `if`, deliberately NOT nested
        # under the X-Project-Id branch above. The two facts are ORTHOGONAL — a
        # caller sending `X-Spend-Scope: tenant` alongside a valid project id
        # gets BOTH forwarded, and the scope must never suppress the id. Nesting
        # these would quietly turn a reporting label into a control over
        # attribution, which is the one thing this header must never be.
        ctx_spend_scope = get_spend_scope_from_context()
        if ctx_spend_scope:
            request.headers["X-Spend-Scope"] = ctx_spend_scope
        # Forward WHO ACTED (per-member billing): inference-router reads
        # X-Actor-Id and stamps `actor_id` on the cost event. The inbound
        # actor UNCHANGED, else WM's own service-account sub (WM-originated
        # work). Same omit-when-empty rule; ⛔ NEVER defaulted to the subject.
        ctx_actor_id = get_effective_actor_id()
        if ctx_actor_id:
            request.headers["X-Actor-Id"] = ctx_actor_id
        # Provider pin: the caller's inbound X-Provider wins, else the worker
        # config's ``ir_provider``. IR honours it as a single-provider path (no
        # failover). Absent ⇒ no header, so today's request is unchanged.
        provider_pin = get_provider_pin_from_context() or pinned_provider
        if provider_pin:
            request.headers["X-Provider"] = provider_pin
        # WS-1 prompt-replay identity: when asset-evaluator drives a replay
        # through WM (X-Replay-Mode: true on send_message), stamp the replay
        # identity so IR writes a replay_invocation_log row. X-Worker-Name is
        # already always present (default header); add mode + active version,
        # and re-assert the caller's X-Correlation-Id so a replay is never
        # logged without its correlation key even on the empty-stage path.
        if is_replay_mode_from_context():
            request.headers["X-Replay-Mode"] = "true"
            request.headers["X-Worker-Version"] = worker_version
            if ctx_correlation:
                request.headers["X-Correlation-Id"] = ctx_correlation
        # AA-TRACE (§2/§6.5-7): forward the 5 inbound trace headers unchanged and
        # stamp X-AA-Prompt-Version / X-AA-Prompt-Sha256 so inference-router can
        # record which prompt version ran. Absent values are omitted, never "".
        request.headers.update(get_trace_and_prompt_headers())

    return _identity_event_hook


# The placeholder bearer every worker_management -> inference-router client
# presents in ``Authorization`` until the identity hook replaces it with the
# tenant's bundle token (``_set_ir_tenant_bearer``). It carries no authority;
# it exists because the OpenAI SDK refuses to build without a key. ONE value for
# every IR client, so the ``vllm`` path (which IS an IR client since gitops
# #2999) can no longer go out as ``Bearer EMPTY`` while the ``inference_router``
# path goes out as ``Bearer internal`` for the same run.
IR_CLIENT_API_KEY = "internal"  # pragma: allowlist secret


def ir_identity_default_headers(
    *,
    worker_name: str,
    tenant_id: str,
    license_id: str,
    subject_id: str,
) -> dict[str, str]:
    """Constructor-time identity headers for ANY worker_management -> IR client.

    Shared by :class:`InferenceRouterOpenAIProvider` and
    ``ModelFactory._create_vllm_model`` so the two clients cannot disagree about
    the header contract (two copies diverging is how the vllm identity gap
    opened). The per-request :func:`make_identity_event_hook` still has the
    final word on every send.

    Identity headers are included ONLY when they carry a value. The old shape
    put all three in unconditionally ("headers are always present, never
    missing"), which meant an unresolved tenant went out as ``X-Tenant-Id: ""``
    -- a PRESENT header, so inference-router's ``Header(default=...)`` could not
    fire, and IR coerced the empty string to the nil sentinel instead. An absent
    header is honest; an empty one is a claim that the tenant is nothing.
    X-Worker-Name is not identity and stays unconditional.
    """
    headers: dict[str, str] = {"X-Worker-Name": worker_name}
    if tenant_id:
        # Same helper the hook uses, so the default header and the per-request
        # header cannot disagree: the resolved tenant uuid, never
        # `Bearer <token>`. The hook still strips a nil tenant at send time.
        headers["X-Tenant-Id"] = _tenant_header_value(tenant_id)
    if license_id:
        headers["X-License-Id"] = license_id
    if subject_id:
        headers["X-Subject-Id"] = subject_id
    return headers


class InferenceRouterOpenAIProvider(OpenAIProvider):
    """OpenAI provider that injects identity headers for inference-router.

    Headers:
        X-Tenant-Id / X-License-Id / X-Subject-Id: identity passthrough so
            inference-router's metering attributes the LLM cost back to the
            originating tenant + license + end-user subject. Per Phase 5a of
            the per-user metering plan, X-Subject-Id MUST be co-equal with
            X-Tenant-Id (no Authorization fallback). All three default to
            empty string when unknown — never the literal "None".

    Routing key (schema_id) is no longer injected as a header. inference-router
    self-derives it from the request body's response_format.json_schema
    fingerprint (content-addressable) and auto-registers a model-registry rule
    on first sighting. Calls without response_format fall through to the
    providers.yaml default_chain. The OpenAI `model` field on the underlying
    OpenAIChatModel keeps its native meaning — match a model alias to auto-pin
    a provider, or send anything generic to ride the default chain.

    Identity is layered: the constructor accepts tenant/license/subject for
    explicit/test use AND installs an httpx event hook that overrides those
    values from `src.api.identity_context` ContextVars at request-send time.

    ⛔ THE PARAGRAPH THAT USED TO SIT HERE WAS STALE, AND IT COST REAL
    ATTRIBUTION. It said `WorkerInstanceCache` "keys agents by worker_name
    only, so ... baking identity at construction would mis-attribute every
    request after the first". That has not been true since the cache became
    tenant-scoped: `_cache_key` (src/api/workers/_globals.py) composes
    `tenant_id:worker_name:work_tier[:product]`. Believing the old sentence is
    why nothing ever threaded `tenant_id` down the factory chain, so the
    constructor's `tenant_id or ""` default was the ONLY value in play — and
    when the ContextVar layer is also empty (any dispatch whose LLM call does
    not run on the middleware's context), this provider sent
    `X-Tenant-Id: ""`. inference-router then coerced that to the nil sentinel
    (src/services/metering.py) and billing dead-lettered it. Measured
    2026-08-27: 1,400 of 13,893 routed calls in 24h (10.1%), 1,086 of them
    `principle_consistency`, whose caller `requirements_management` had the
    real tenant UUID on the row the whole time.

    So: tenant_id IS baked at construction now (safe -- it is in the cache
    key). license_id and subject_id are NOT, and must not be: neither is part
    of the key, so for those the original warning still holds exactly. They
    stay ContextVar-only.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = IR_CLIENT_API_KEY,
        tenant_id: str | None = None,
        license_id: str | None = None,
        subject_id: str | None = None,
        worker_name: str | None = None,
        worker_version: str | None = None,
        pinned_provider: str | None = None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key)
        self._pinned_provider = pinned_provider or ""
        # Constructor-supplied identity is the fallback when no per-request
        # ContextVar is set (tests, ad-hoc construction, fr /chat path that
        # builds a fresh agent per request).
        self._fallback_tenant_id = tenant_id or ""
        self._fallback_license_id = license_id or ""
        self._fallback_subject_id = subject_id or ""
        # Worker name is fixed at construction time — a single provider
        # instance is shared by all calls from the same PydanticAIAgent.
        # IR binds the value into structlog contextvars via
        # routers/completions.py, unlocking per-worker slicing on
        # provider_marked_degraded / ReadTimeout events. Closes the
        # observability gap noted in the 2026-05-19 trust.pdf incident,
        # where `worker_name: null` in IR logs forced investigators to
        # cross-correlate by timestamp instead of grepping by worker.
        self._worker_name = worker_name or ""
        # Active worker version (workers.config_revision), fixed at construction
        # time like worker_name — a single provider instance is shared by all
        # calls from the same PydanticAIAgent. Emitted as X-Worker-Version ONLY
        # when the per-request replay-mode ContextVar is set, so prompt replays
        # driven by asset-evaluator carry the identity IR needs to write a
        # replay_invocation_log row. Empty string when unknown — never "None".
        self._worker_version = worker_version or ""

        # ⛔ There is deliberately no outbound LICENCE CREDENTIAL field here.
        # `self._license_token`, and the whole `OUTBOUND_LICENSE_TOKEN[_<TENANT>]`
        # mechanism that fed it, were RETIRED 2026-09-11: the value was resolved,
        # threaded through two layers and then dropped by `_tenant_header_value`,
        # so it reached no wire while sitting in the pod environment as a live
        # bearer credential. The licence that IR actually reads is the resolved
        # licence ID on `X-License-Id`, below. Do not reintroduce a token field.

        # Built by the SAME helper the vllm path uses -- see
        # `ir_identity_default_headers` for the omit-when-empty contract.
        headers = ir_identity_default_headers(
            worker_name=self._worker_name,
            tenant_id=self._fallback_tenant_id,
            license_id=self._fallback_license_id,
            subject_id=self._fallback_subject_id,
        )

        _identity_event_hook = make_identity_event_hook(
            worker_name=self._worker_name,
            worker_version=self._worker_version,
            fallback_tenant_id=self._fallback_tenant_id,
            fallback_license_id=self._fallback_license_id,
            fallback_subject_id=self._fallback_subject_id,
            pinned_provider=self._pinned_provider,
        )

        # AsyncOpenAI accepts an http_client; install the event hook there so
        # it fires on every outbound request without us having to wrap each
        # SDK call site.
        #
        # Transport hardening (F3 of the trust.pdf fix plan): the bare
        # AsyncClient default surfaced transient ambient-mesh blips —
        # cross-namespace TLS handshake resets between workers and
        # inference-router — as "Connection error" agent-run failures.
        # AsyncHTTPTransport(retries=N) retries idempotent connect-level
        # failures (httpx.ConnectError, immediate RST) without us needing
        # application-level retry. Read timeout 120s matches normal cold-
        # cache latency of the 120B model under load; the bare-client
        # default of 5s was guaranteed-fail for real completions.
        #
        # The transport is DiagnosticAsyncHTTPTransport, a subclass that adds
        # nothing on the success path and, on failure, logs the fields the
        # 2026-08 "Connection error" investigation stalled for want of:
        # elapsed_s, target_url, a request_id joinable to IR's own logs, and
        # connection_state (new/reused/unknown). See http_diagnostics.py for
        # why each one discriminates between the surviving hypotheses. It
        # still IS an AsyncHTTPTransport, so `retries=` above keeps its
        # meaning exactly.
        #
        # `limits=` is passed explicitly for the first time. The values are
        # httpx's own defaults, so this changes no behaviour -- it makes the
        # pool ceiling a stated, reviewable number instead of an inherited
        # one. See the settings block for why it is deliberately NOT raised
        # in the same change that ships the instrument to measure it.
        #
        # ⛔ `limits=` MUST be passed to the TRANSPORT, never to the
        # AsyncClient, whenever an explicit transport is supplied.
        # `AsyncClient._init_transport` is literally
        # `if transport is not None: return transport` (httpx 0.28.1,
        # _client.py) -- so a client-level `limits=` next to a `transport=` is
        # silently DISCARDED, with no error and no warning. Written on the
        # client this block would read as a fix, review as a fix, and do
        # nothing whatsoever. That is also why the test asserts on the pool's
        # own `_max_keepalive_connections`, not on a constructor kwarg: only
        # the pool can tell you the ceiling actually took.
        from src.settings import get_settings

        from .http_diagnostics import DiagnosticAsyncHTTPTransport

        _s = get_settings()
        http_client = httpx.AsyncClient(
            event_hooks={"request": [_identity_event_hook]},
            transport=DiagnosticAsyncHTTPTransport(
                retries=_s.ir_http_transport_retries,
                limits=httpx.Limits(
                    max_connections=_s.ir_http_max_connections,
                    max_keepalive_connections=_s.ir_http_max_keepalive_connections,
                    keepalive_expiry=_s.ir_http_keepalive_expiry_s,
                ),
            ),
            timeout=httpx.Timeout(
                connect=_s.ir_http_connect_timeout_s,
                read=_s.ir_http_read_timeout_s,
                write=_s.ir_http_write_timeout_s,
                pool=_s.ir_http_pool_timeout_s,
            ),
        )
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers,
            http_client=http_client,
            # max_retries=0 is REQUIRED, not a tidy-up. The SDK default is 2
            # (openai/_constants.py:10) and its retry loop is not confined to
            # 429/5xx: openai/_base_client.py:1653 catches httpx.TimeoutException
            # and, while remaining_retries > 0, sleeps and re-sends. A read
            # timeout here means a decode that already consumed the full
            # ir_http_read_timeout_s budget on our own GPU — re-sending it costs
            # another full decode on the very provider whose saturation caused
            # the timeout, so one slow request becomes three. That is positive
            # feedback, and it is exactly backwards from what a saturated
            # provider needs.
            #
            # Nothing is lost: connect/RST blips are already covered underneath
            # by AsyncHTTPTransport(retries=ir_http_transport_retries), and
            # genuine application-level retry lives above in the bounded
            # exp-backoff recreate loop (llm_retry_max_attempts). Retry policy
            # belongs to the caller, which can see the queue depth and the
            # worker's remaining budget; the SDK can see neither.
            #
            # This has been latent rather than live only because the agent cap
            # (agent_run_timeout_s) currently EQUALS ir_http_read_timeout_s, so
            # the agent's asyncio.timeout always fires first and the SDK never
            # spends a retry. Settings._validate_timeout_ordering now enforces
            # that ordering, but pinning 0 here is what makes it safe regardless.
            max_retries=0,
        )


def resolve_billable_provider(
    *,
    served_provider: str | None,
    cost_provider: str | None,
    model_provider: str | None,
    served_model: str | None,
    requested_model: str | None,
) -> str | None:
    """Pick the provider a BYO cost_event may be billed under, or None.

    Precedence: what inference-router says it actually used > the pricing
    provider resolved from the tenant row > model_provider (the worker's
    routing name — only correct for direct external BYO).

    A *routing* name (see ``ROUTING_PROVIDERS``) has no provider_pricing row, so
    a second pass re-derives the real upstream from the now-known served model.
    If it is STILL a routing name we return None: billing computes cost_usd once
    at flush time, so emitting would create a permanently-unbilled NULL row —
    the caller must log loudly and drop the event instead (issue #137).
    """
    from src.worker.agents.provider_catalog import ROUTING_PROVIDERS, billing_provider_for

    provider = served_provider or cost_provider or model_provider
    if provider and provider in ROUTING_PROVIDERS:
        provider = billing_provider_for("local", served_model or requested_model or "", provider)
    if not provider or provider in ROUTING_PROVIDERS:
        return None
    return provider


# inference-router's non-standard top-level response body fields naming what
# ACTUALLY served a call: the provider, vLLM's served name, and the model root
# (the weights behind that name). Read back, passed through unchanged, never
# re-derived from the requested model.
IR_SERVED_FIELDS: tuple[str, ...] = ("served_model", "served_provider", "served_model_root")


class InferenceRouterStreamedResponse(OpenAIStreamedResponse):
    """Streamed counterpart of ``InferenceRouterOpenAIChatModel``'s override.

    The non-streaming hook (``_process_provider_details``) is never called on a
    streamed run: ``_process_streamed_response`` builds a StreamedResponse and
    the details come from *this* class's ``_map_provider_details(chunk)``, whose
    base implementation reads only ``chunk.choices[0]`` (logprobs/finish_reason).
    IR's ``served_model`` / ``served_provider`` are top-level *body* fields, so
    without this override they are dropped on every streamed run and issue #137
    regresses. Same merge semantics as the non-streaming override above.
    """

    def _map_provider_details(self, chunk: Any) -> dict[str, Any] | None:
        details = super()._map_provider_details(chunk) or {}
        extra = getattr(chunk, "model_extra", None) or {}
        for key in IR_SERVED_FIELDS:
            value = extra.get(key)
            if value:
                details[key] = value
        return details or None


class InferenceRouterOpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel that surfaces inference-router's resolved-routing fields.

    WM sends inference-router a routing *alias* as the OpenAI `model` field
    (e.g. ``tier-medium``); IR picks the real upstream + model server-side and
    echoes them back as two non-standard top-level body fields, ``served_model``
    and ``served_provider``. Billing prices a cost_event on
    (provider, served_model), so without these the emitted event carries the
    alias — which has no provider_pricing row — and its ``cost_usd`` is NULL
    forever (billing prices exactly once, at flush time). That is issue #137.

    The fields survive to here because the OpenAI SDK's BaseModel is
    ``extra="allow"``: they land in ``ChatCompletion.model_extra``, and
    pydantic-ai's ``_ChatCompletion`` subclass inherits that config (it only
    adds a ``title``), so they also survive the
    ``model_validate(response.model_dump())`` round-trip in
    ``_validate_completion``. No override of that hook is needed.

    ``_process_provider_details`` is pydantic-ai's documented extension point;
    whatever it returns is merged into ``ModelResponse.provider_details``.
    """

    def _process_provider_details(self, response: Any) -> dict[str, Any] | None:
        details = super()._process_provider_details(response) or {}
        extra = getattr(response, "model_extra", None) or {}
        # Absent keys stay absent (an older IR that predates the companion PR
        # returns neither) and explicit nulls are dropped, so the emit site's
        # "when present" checks stay honest rather than seeing a None value.
        for key in IR_SERVED_FIELDS:
            value = extra.get(key)
            if value:
                details[key] = value
        return details or None

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Streamed runs bypass ``_process_provider_details`` — hook them too.

        pydantic-ai's ``_process_streamed_response`` never calls that method; it
        instantiates this class instead. Documented subclass hook.
        """
        return InferenceRouterStreamedResponse


# Providers whose backend is a self-hosted OpenAI-compat server we run on
# our own hardware. These are the providers vulnerable to the 2026-05-13
# Nemotron-3 / SM120 CUTLASS TMA-WS FP4-MoE kernel crash, which triggers when
# max_tokens approaches the model's full context window under structured
# output. Outbound max_tokens for these providers is always clamped to
# SAFE_MAX_TOKENS below.
SELF_HOSTED_PROVIDERS: tuple[str, ...] = ("vllm", "inference_router", "mlx", "vlm_local")

# Ceiling for max_tokens on any request routed to a self-hosted backend.
# Set above the highest legitimate worker config observed in production
# (15000, ks-* knowledge-system workers) but far below the 131072 context
# window of Nemotron-3 where the FP4-MoE kernel crash triggers.
SAFE_MAX_TOKENS = 16384

# Fallback when a worker targeting a self-hosted backend doesn't specify
# max_tokens explicitly. Deliberately NOT tuned by the 2026-05-13 crash fix,
# which is why it sat at the historical 4000 long after that value stopped
# fitting the work.
#
# Raised 4000 -> 6000 on 2026-08-09 against measured production failure, not a
# guess. Over 7 days, three unrelated workers with no explicit config died at
# exactly this ceiling with "Model token limit (4000) exceeded before any
# response was generated": ks-relationship-extractor (40), footage-critic (2)
# and footage-director (2). They share nothing but this default.
#
# 6000 and not more. Two corrections to how this number used to be justified,
# both established by measurement on 2026-08-09:
#
# ⚠️ (1) pgx-local publishes TWO decode rates and they are not interchangeable:
#   request_time_per_output_token_seconds 191.103/11859 = 16.12 ms/tok
#     -> 62.0 tok/s, but this is INTER-TOKEN only (excludes prefill + queue)
#   request_generation_tokens_sum / e2e_request_latency_seconds_sum
#     = 1.697783e6 / 45819.15            -> 37.05 tok/s of TOTAL wall-clock
# Anything measuring elapsed time sees the second one. An earlier version of
# this comment quoted 62.4 and claimed a "2x pessimism margin"; that overstated
# the headroom by ~1.67x.
#
# ⚠️ (2) The DEGRADED breaker is NOT the binding ceiling, because it is not a
# deadline at all. `ProviderHealth.record()` runs in inference-router's
# `finally` (provider_clients.py:536-545) AFTER the call has already returned --
# a 210s call returns its answer normally. Only after 3 CONSECUTIVE breaches do
# SUBSEQUENT calls get ProviderDegradedError for a 300s cooldown. It is a
# post-hoc classifier, so no max_tokens value can "stay under" it. This comment
# previously called it "the binding ceiling"; that was wrong.
#
# ⚠️ (3) The deadline numbers this comment used to quote are STALE -- corrected
# 2026-08-19 against the live GitOps manifests, which are the source of truth for
# all four (nothing here is a code default):
#   INFERENCE_ROUTER_CHAIN_DEADLINE_SECONDS = 1200
#     (gitops-production/clusters/strato/services/inference-router/deployment.yaml)
#   AGENT_RUN_TIMEOUT_S = AGENT_RUN_TIMEOUT_MCP_S = IR_HTTP_READ_TIMEOUT_S = 1200
#     (gitops-production/clusters/strato/services/workers/deployment.yaml)
# The previous text said "the 300s chain deadline ... and above that this
# service's own agent_run_timeout_s of 600". Both were superseded; the in-repo
# Settings defaults (600/900/600 in src/settings.py) are likewise overridden by
# those env vars in production, so reading the defaults here misleads too. So the
# chain deadline is NOT what bounds 6000 tokens today -- it has ~4x the headroom
# the old arithmetic assumed.
#
# ⚠️ (4) The model is no longer Nemotron-3. pgx serves Qwen/Qwen3.8-27B-FP8 at
# max_model_len=131072 (a DENSE model, so decode behaves differently from the
# MoE the tok/s figures above were taken on -- re-measure before reusing them).
# The 131072 window is unchanged, so SAFE_MAX_TOKENS's headroom argument still
# holds, but every sentence below that reasons FROM Nemotron-3 decode rates is
# describing a model we no longer run.
#
# So: at 37.05 tok/s wall-clock, 6000 tokens is 162s -- which was already inside
# the old 300s chain deadline and is far inside the live 1200s one. The value is
# LEFT AT 6000 deliberately: the justification above is stale, not the number,
# and re-baselining it needs fresh measurement on Qwen3.8-27B rather than a
# recalculation from figures taken on a different model.
#
# And the cap is a CEILING, not a typical decode: mean generation is 143 tokens
# at 3.87s mean e2e, so full-budget decodes are the rare tail. Over 30 days
# (n=21,159 pgx-local samples) exactly 8 calls exceeded 210s, 1 exceeded 240s,
# and 0 exceeded 300s.
# Still far below SAFE_MAX_TOKENS, so the FP4-MoE kernel-crash guard is intact.
# ⚠️ RE-DERIVED 2026-08-24 — EVERY NUMBER ABOVE THIS LINE WAS MEASURED ON A MODEL
# THAT NO LONGER EXISTS. pgx was swapped google/gemma-4-26B-A4B-it (MoE, ~4B
# ACTIVE params) -> Qwen/Qwen3.8-27B-FP8 (DENSE, 27B active) on 2026-08-18.
# "26B -> 27B" reads like a rounding change; active params set the decode rate,
# so it is ~4x. Re-measured from the live pgx `/metrics` (n=12,933 requests):
#
#   request_generation_tokens_sum / e2e_request_latency_seconds_sum
#     = 8.026819e6 / 694872.8   ->  11.55 tok/s of TOTAL wall-clock  (was 37.05)
#   request_time_per_output_token_seconds_sum / _count
#     = 1193.165 / 12933 = 92.3 ms/tok -> 10.84 tok/s INTER-TOKEN    (was 62.0)
#
# The two rates have CONVERGED (62 vs 37 before, 10.8 vs 11.6 now): on the dense
# model decode dominates, so prefill and queue no longer separate them. MTP
# speculative decoding is on and accepting 93.2% of draft tokens — that rate is
# WITH the speculator, not before it.
#
# The ceilings this value has to fit inside also all moved (gitops #2905/#2989):
#   inference-router chain deadline   300  -> 1200 s
#   inference-router pgx-local httpx  600  -> 1100 s   (the innermost rung)
#   this service's agent_run_timeout_s 600 -> 1200 s
#
# So 6000 STANDS, but for a different reason than the one written above: at
# 11.55 tok/s it is 519 s, which fits under the 1100 s provider timeout with
# room for prefill and queue. Under the OLD 600 s rung it would not have — and
# that is exactly the truncation #2989 fixed. Anything above ~9500 tokens
# (~825 s) starts crowding 1100 again; re-derive here, not at the call site.
#
# ⚠️ THAT LAST SENTENCE IS NO LONGER ADVICE — IT IS ENFORCED. It was written as
# a warning and stayed inert for six days, during which seven registered workers
# sat at 10000-16000 and 59 requests decoded to a 12000-token cap and returned
# nothing (2026-08-24, see Settings.self_hosted_max_tokens_ceiling). The ~9500
# derived by hand here is now computed as
# floor(provider_timeout_s x decode_tokens_per_s x headroom) = 9528 and applied
# in _resolve_max_tokens. Re-derive it THERE, by moving one of the three inputs
# — not by editing this number, which only governs the unset case.
#
# ⚠️ And the "rare tail" claim above is now false: mean generation is 620 tokens
# at 53.7 s mean e2e (was 143 tokens / 3.87 s). That 4.3x is not more work — it
# is Qwen3.8's chat template defaulting chain-of-thought ON, which is what
# `_resolve_reasoning_extra_body` now turns off for the direct vLLM path.
DEFAULT_SELF_HOSTED_MAX_TOKENS = 6000


def _resolve_reasoning_extra_body(
    config: "PydanticAIAgentConfig",  # noqa: UP037 -- forward ref; class defined below
) -> dict[str, object] | None:
    """Resolve the ``extra_body`` payload that carries the chain-of-thought switch.

    ⚠️ THE WIRE SHAPE DIFFERS BY PROVIDER AND THE WRONG ONE IS A SILENT NO-OP —
    it is accepted, forwarded, and simply never read, so the model keeps thinking
    and nothing in the logs says otherwise. Hence three cases, not one:

    * ``inference_router`` ⇒ ``{"enable_reasoning": bool}``. IR declares the field
      on its own request schema (``inference-router/src/models/schemas.py``,
      ``enable_reasoning: bool = False``) and rewrites it into
      ``chat_template_kwargs.enable_thinking`` before it reaches vLLM
      (``provider_clients.py``, under ``force_chat_template_kwargs``). Sending the
      raw vLLM shape here would be dropped by IR's own payload builder. The sibling
      graph path already uses exactly this shape — see
      ``src/worker/graphs/nodes/inference_provider_node.py``.
    * direct self-hosted (``vllm`` / ``mlx`` / ``vlm_local``) ⇒
      ``{"chat_template_kwargs": {"enable_thinking": bool}}``, vLLM's own kwarg,
      because nothing is guaranteed to sit in front of the server and translate for
      us. This is the path every w2v LLM call takes: the message endpoint's
      auto-create branch hardcodes ``AUTO_CREATE_MODEL_PROVIDER`` (default
      ``vllm``). ⚠️ Since gitops #2999 the deployed ``VLLM_URL`` points at the
      inference-router Service, so in production that traffic IS routed — but the
      env var is the only thing making it so, and IR ignores this key rather than
      rejecting it, so the vLLM shape stays correct for the unrouted case.
    * anything else (openai / anthropic / groq / deepseek / cohere / xai / …) ⇒
      ``None``. No shape is invented for a provider whose reasoning parameter has
      not been verified; a guess would be the silent no-op described above.

    ``config.enable_reasoning is None`` means "the caller did not say". For every
    provider but one that stays absent from the payload. ``vllm`` is the exception
    and defaults to thinking OFF — see the block comment below for why an absent
    switch is not a neutral default there.
    """
    if config.enable_reasoning is None:
        # Direct vLLM is the ONE provider where sending nothing is not neutral.
        # Qwen3.8's chat template defaults thinking ON; gemma-4's, which pgx served
        # until the 2026-08-18 swap, defaulted OFF. So "send nothing" did not keep
        # behaviour as it was — it silently turned chain-of-thought on for every
        # caller that does not pass the flag, and nothing in the logs says so.
        #
        # MEASURED on the live pgx endpoint 2026-08-24, identical prompt, temp 0,
        # two runs each:
        #     no chat_template_kwargs -> 420 / 326 completion tokens, 49.5s / 38.6s
        #     enable_thinking=False   ->  28 /  28 completion tokens,  3.5s /  3.4s
        # Same 64-66 character answer. 12-15x the tokens and ~11-14x the wall clock,
        # on a GPU the whole fleet shares, at ~11-13 tok/s decode. Over 7 days this
        # service built 1009 vllm agents and only 49 carried a switch (Loki,
        # ns=workers); the other 960 paid that.
        #
        # Defaulting it off is not a new policy, it is the one already in force
        # everywhere else: inference-router's own request schema declares
        # ``enable_reasoning: bool = False``, and all 49 explicit uses observed in
        # production set False. An explicit ``True`` still reaches the model.
        #
        # ONLY ``vllm``. mlx / vlm_local are different servers serving different
        # models and neither has been measured here; inventing a default for them
        # would be the same unverified guess this function refuses to make below.
        #
        # ⚠️ THE DEPLOYED EXPOSURE CLOSED WHILE THIS WAS IN REVIEW, at a different
        # layer. gitops #2999 (merged 2026-08-24, verified live: 0 direct
        # /v1/chat/completions from ns=workers in the 5 min after the roll)
        # repointed ``VLLM_URL`` from the raw vLLM host to the inference-router
        # Service, so `model_provider: vllm` is now the ROUTED path to the local
        # model rather than vLLM itself. That does not make this branch wrong or
        # dead, and it is deliberately NOT keyed on the URL:
        #
        #   * routed  — IR's ``ChatCompletionRequest`` is a plain BaseModel, so
        #     pydantic's default ``extra="ignore"`` drops this key at parse; IR then
        #     builds the upstream payload from its own ``enable_reasoning``, which
        #     DEFAULTS FALSE (schemas.py) and is forced onto vLLM as
        #     ``chat_template_kwargs`` by ``force_chat_template_kwargs``. Thinking is
        #     off either way, and this key cannot 422.
        #   * direct  — ``VLLM_URL`` unset falls back to ``_vllm_default_url()``, and
        #     dev/test configs still point at a real vLLM. There this branch IS the
        #     fix, and it is what makes a revert of #2999 safe instead of a silent
        #     12-15x regression.
        if config.model_provider == "vllm":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None
    if config.model_provider == "inference_router":
        return {"enable_reasoning": config.enable_reasoning}
    if config.model_provider in SELF_HOSTED_PROVIDERS:
        return {"chat_template_kwargs": {"enable_thinking": config.enable_reasoning}}
    return None


def _with_repetition_penalty(extra_body: object, repetition_penalty: float) -> dict[str, Any]:
    """``extra_body`` with ``repetition_penalty`` merged in, never replaced.

    pydantic-ai merges model settings SHALLOWLY (``base | overrides``), so a
    bare ``{"repetition_penalty": x}`` would silently drop the reasoning keys
    (``enable_reasoning`` / ``chat_template_kwargs``) already in ``extra_body``.
    A non-dict ``extra_body`` is not something this module ever builds; it is
    dropped rather than guessed at.
    """
    merged: dict[str, Any] = (
        {str(k): v for k, v in extra_body.items()} if isinstance(extra_body, dict) else {}
    )
    merged["repetition_penalty"] = repetition_penalty
    return merged


def _resolve_max_tokens(config: "PydanticAIAgentConfig") -> int | None:  # noqa: UP037 -- forward ref; class defined below
    """Resolve the outbound max_tokens value for a given agent config.

    For self-hosted backends (vllm/inference_router/mlx/vlm_local) max_tokens
    is *always* set so vLLM can never fall back to "max_seq_len - prompt_len",
    which was the trigger for the 2026-05-13 FP4-MoE kernel crash. Explicit
    values are clamped to SAFE_MAX_TOKENS as a runtime defense (the validator
    on WorkerCreateRequest is a separate layer that catches new bad configs;
    this layer also handles any existing rows or in-process overrides that
    bypass the validator).

    For external providers (anthropic/openai/groq/cohere/deepseek) we honour
    config.max_tokens unchanged (or omit the field entirely when None).
    """
    if config.model_provider in SELF_HOSTED_PROVIDERS:
        if config.max_tokens is not None:
            # TWO CEILINGS, DELIBERATELY NOT COLLAPSED INTO ONE NUMBER.
            #
            # SAFE_MAX_TOKENS answers "can this crash the kernel?" (context
            # window). time_budget answers "can this ever FINISH?" (the
            # innermost timeout rung, at the measured decode rate). They move
            # for unrelated reasons: swapping the served model changes the
            # decode rate, raising a deadline changes the budget, and neither
            # touches the context window. Whichever is smaller wins, and the
            # log below says WHICH so the next reader does not have to guess.
            time_budget = _get_settings().self_hosted_max_tokens_ceiling
            clamped = min(config.max_tokens, SAFE_MAX_TOKENS, time_budget)
            if clamped != config.max_tokens:
                reason = (
                    "SAFE_MAX_TOKENS"
                    if time_budget >= SAFE_MAX_TOKENS
                    else "self_hosted_max_tokens_ceiling (time budget)"
                )
                logger.warning(
                    "Agent %s - max_tokens clamped %d -> %d for self-hosted provider %s (%s)",
                    config.name,
                    config.max_tokens,
                    clamped,
                    config.model_provider,
                    reason,
                )
                # COUNT IT. A clamp that only logs is the same defect one layer
                # up: the 83 lost generations of 2026-08-24 were invisible
                # precisely because every signal on that path was a log line.
                # This rides the existing aa_prompt_validation_failures_total
                # rather than minting a third token-related counter — the event
                # IS a config that fails a check, and error_type is the
                # dimension that separates it from the schema failures wired
                # alongside it. ⚠️ error_type stays a CLOSED enum: cardinality
                # here is workers x error_type, and a free-text value (an
                # exception message, a clamped integer) is the term that
                # explodes it.
                try:
                    get_metrics_client().record_aa_prompt_validation_failure(
                        prompt_id=config.name,
                        worker_name=config.name,
                        error_type=(
                            "max_tokens_exceeds_safe_ceiling"
                            if time_budget >= SAFE_MAX_TOKENS
                            else "max_tokens_exceeds_time_budget"
                        ),
                    )
                except Exception:  # pragma: no cover - telemetry must never break a run
                    logger.debug("max_tokens clamp not recorded", exc_info=True)
            return clamped
        return DEFAULT_SELF_HOSTED_MAX_TOKENS
    return config.max_tokens


# Base URLs for self-hosted OpenAI-compat backends are resolved from
# Settings (``VLLM_URL`` / ``VLM_LOCAL_URL`` env vars on the Deployment).
# Module-level ``VLLM_DEFAULT_URL`` / ``VLM_LOCAL_DEFAULT_URL`` are exposed
# via ``__getattr__`` so each access re-reads the current Settings value;
# this keeps existing call sites (``... or VLLM_DEFAULT_URL``) working and
# fails loud in production via Settings.model_post_init when unset.
from src.settings import LOCAL_LLM_MODEL
from src.settings import get_settings as _get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.run import AgentRunResult as FrameworkAgentRunResult


def _vllm_default_url() -> str:
    return _get_settings().vllm_url


def _vlm_local_default_url() -> str:
    return _get_settings().vlm_local_url


def __getattr__(name: str) -> str:
    if name == "VLLM_DEFAULT_URL":
        return _vllm_default_url()
    if name == "VLM_LOCAL_DEFAULT_URL":
        return _vlm_local_default_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PydanticAIAgentConfig(BaseModel):
    """Configuration for a PydanticAI agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    model_provider: ModelProvider = Field(
        ...,
        description=(
            "Model provider (openai, deepseek, groq, cohere, mlx, inference_router, "
            "anthropic, vllm, or vlm_local)"
        ),
    )
    model: str = Field(..., min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ir_provider: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description=(
            "inference-router provider to pin (sent as X-Provider: single-provider "
            "path, no failover). None ⇒ no header. A caller's inbound X-Provider wins."
        ),
    )
    seed: StrictInt | None = Field(
        default=None,
        description=(
            "Sampling seed. None (default) ⇒ NO seed key is sent, so the outbound "
            "payload is byte-for-byte unchanged. An int is sent as the "
            "OpenAI-compatible top-level `seed` via pydantic-ai's native "
            "ModelSettings['seed'] (models/openai.py reads it). Strict so a JSON "
            "bool or string in a stored config fails loudly instead of becoming 1."
        ),
    )
    repetition_penalty: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description=(
            "vLLM repetition penalty (1.0 = off, >1 penalises repeats). None "
            "(default) ⇒ NOTHING is sent. Not an OpenAI parameter, so it travels "
            "in ``extra_body`` (merged with any reasoning keys) and becomes a "
            "top-level body key — and only for self-hosted providers "
            "(SELF_HOSTED_PROVIDERS); an external API would reject the unknown key."
        ),
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "OpenAI-compatible ``frequency_penalty``. None (default) ⇒ no key is "
            "sent. Sent via pydantic-ai's native ModelSettings['frequency_penalty']."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=16384,
        description=(
            "Maximum tokens in the model response. When set, overrides the "
            "per-provider hardcoded default (4000 for mlx/inference_router). "
            "When None, providers fall back to their existing behavior. "
            "Capped at 16384 to protect self-hosted vLLM (Nemotron-3 on "
            "SM120/GB10) from a CUTLASS TMA-WS FP4-MoE kernel crash under "
            "near-full-context structured-output requests (2026-05-13)."
        ),
    )
    enable_reasoning: bool | None = Field(
        default=None,
        description=(
            "Tri-state chain-of-thought switch for reasoning-capable models. "
            "None (default) ⇒ NOTHING is emitted, so the outbound payload is "
            "byte-for-byte unchanged and the model/template default decides. "
            "True/False are forwarded only to self-hosted backends, and the wire "
            "shape differs per provider (see ModelFactory.get_model_settings): "
            "inference_router takes a flat ``extra_body.enable_reasoning`` (IR "
            "declares the field and rewrites it into "
            "``chat_template_kwargs.enable_thinking``), while vllm/mlx/vlm_local "
            "take vLLM's own ``extra_body.chat_template_kwargs.enable_thinking``. "
            "External providers get nothing — sending the wrong shape is a SILENT "
            "no-op, so no shape is invented for a provider we have not verified."
        ),
    )
    system_prompt: str | None = Field(default="You are a helpful assistant.")
    instruction: str | None = Field(
        default=None,
        description="Instruction for the agent, preferred over system_prompt for some use cases.",
    )
    user_prompt: str = Field(default="")
    tools: list[Any] = Field(default_factory=list, description="List of Tool instances")
    db_path: str
    deps: list[str] = Field(default_factory=list)
    base_url: str | None = Field(default=None, description="Base URL for the model provider API")
    api_key: str | None = Field(default=None, description="API key for the model provider")
    cost_source: str | None = Field(
        default=None,
        description=(
            "Set to 'worker-management' by resolve_llm_config when this worker runs on a "
            "tenant BYO provider (direct-to-provider, bypassing inference-router metering). "
            "Triggers a token-accurate cost_event after each run (src/telemetry/cost_events.py)."
        ),
    )
    cost_provider: str | None = Field(
        default=None,
        description=(
            "Real upstream provider the cost_event is BILLED under, set by "
            "resolve_llm_config. For the `local` preset this is the served model's "
            "pricing provider (pgx-local / deepseek), NOT the inference_router routing "
            "name in model_provider — metering's provider_pricing is keyed on it. "
            "Falls back to model_provider when unset (external BYO)."
        ),
    )
    tenant_byo_key: bool = Field(
        default=False,
        description=(
            "True ONLY when ``api_key`` was resolved from THIS tenant's own "
            "``tenant_llm_config`` row (own BYO, or managed BYO the owner set for "
            "this one tenant) by the BYO resolver (``llm_config_resolver``). It is "
            "the single thing that lets a non-platform tenant reach an external "
            "provider, and it forbids the owner env-key fallback in "
            "``ModelFactory.get_api_key``. Never set from request input."
        ),
    )
    owner_provided: bool = Field(
        default=False,
        description=(
            "Managed BYO: the platform owner supplied the tenant's external key. "
            "Set by resolve_llm_config. When True the BYO cost_event is emitted as "
            "a normal marked-up line (usage_extra.owner_provided, no byo flag) so "
            "billing applies the tenant's markup rule on real cost."
        ),
    )
    key_fingerprint: str = Field(
        default="",
        description=(
            "Audit fingerprint of the stored tenant key VERSION that serves this "
            "worker (sha256 prefix of the encrypted blob, never the key). Stamped on "
            "every cost event as usage_extra.key_fp. Empty off the tenant-key path."
        ),
    )
    byo_role: str = Field(
        default="",
        description="tenant_llm_config role of the row the key came from (wm_tenant_key_use_total).",
    )
    tools_dir: str | None = Field(
        default=None, description="Directory containing local tool definitions"
    )
    mcp_servers: list[Any] = Field(
        default_factory=list, description="List of initialized MCP Server instances"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for structured output"
    )
    output_validator_code: str | None = Field(
        default=None,
        description="UNSUPPORTED: must be None/empty; non-empty code fails closed, never runs",
    )
    output_retries: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description=(
            "Per-worker override for pydantic-ai output-validation retry "
            "budget. None preserves the provider default at Agent init time."
        ),
    )
    # Identity passthrough for inference-router metering (Phase 5a of the
    # per-user metering plan). When the agent runs in response to an inbound
    # license-authz'd request, these are populated from request.state and
    # forwarded as X-Tenant-Id / X-License-Id / X-Subject-Id on outbound LLM
    # calls so inference-router's metering attributes cost back to the
    # originating tenant + license + user. None is treated as "" downstream.
    tenant_id: str | None = Field(
        default=None,
        description="Tenant ID forwarded as X-Tenant-Id to inference-router for metering",
    )
    license_id: str | None = Field(
        default=None,
        description="License ID forwarded as X-License-Id to inference-router for metering",
    )
    subject_id: str | None = Field(
        default=None,
        description="Subject ID (end-user UUID) forwarded as X-Subject-Id for per-user metering",
    )
    # ⛔ There is deliberately no `license_token` field on this config. It was
    # RETIRED 2026-09-11 along with `outbound_license.py`: the value was
    # resolved per tenant, threaded down two factory layers onto this config,
    # and then DISCARDED by `_tenant_header_value`, whose body is just
    # `return resolved_tenant`. It reached no wire, so it was a live bearer
    # credential sitting in the pod environment and on this config object with
    # no consumer -- and one that had to be kept out of every `f"{config}"` by
    # a `repr=False` that was the only thing between it and a debug log.
    # Removing it closes that permanently. The licence IR actually reads is
    # `license_id`, forwarded as `X-License-Id`. Do not reintroduce this field.
    worker_version: str | None = Field(
        default=None,
        description=(
            "Active worker version (workers.config_revision). Forwarded as "
            "X-Worker-Version to inference-router ONLY when the request is a "
            "prompt replay (X-Replay-Mode), so IR can key a replay_invocation_log "
            "row to the exact worker version under replay. None → '' downstream."
        ),
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[Any]) -> list[Any]:
        """Validate that all items in tools list are Tool instances.

        Args:
            v: List of tools to validate

        Returns:
            Validated list of Tool instances

        Raises:
            ValueError: If any item is not a Tool instance
        """
        if not all(isinstance(tool, Tool) for tool in v):
            raise ValueError("All items in tools must be Tool instances")
        return v


# JSON-schema primitives that pydantic-ai's Agent accepts directly as
# `output_type=` without a StructuredDict wrapper. StructuredDict requires the
# root schema to be an object; for scalars/arrays we hand pydantic-ai the bare
# Python type and let it build the appropriate response_format on its own.
_JSON_TYPE_TO_PY: dict[str, Any] = {
    "boolean": bool,
    "integer": int,
    "number": float,
    "string": str,
}


def _decoder_visible_schema(output_schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Describe what the *decoder* was actually constrained by.

    The fatal validator must be built from this, not from the caller's schema.
    The decoder never sees the full schema: object roots reach it stripped of the
    keywords xgrammar cannot compile, and scalar/array roots reach it as a bare
    Python type carrying no constraints at all. Grading a model against a
    contract it was never shown is how ``demo_video``/``retention_plan`` failed
    100% of runs on 2026-07-20 — see the ``xgrammar_schema`` module docstring.

    ``enum``/``const`` are carried through deliberately. They are *not*
    xgrammar-stripped, and on a scalar root ``_output_type_from_schema`` hands
    the decoder a bare ``str``, so the validator is their only enforcement.
    Dropping them here would silently turn every out-of-vocabulary
    classification label into a 200 carrying an arbitrary string.
    """
    if not output_schema:
        return None

    raw_type = output_schema.get("type")

    visible: dict[str, Any]
    if raw_type in _JSON_TYPE_TO_PY:
        visible = {"type": raw_type}
    elif raw_type == "array":
        visible = {"type": "array"}
        items = output_schema.get("items")
        item_type_name = items.get("type") if isinstance(items, dict) else None
        # Mirror the `list[item_py]` fallback below: an item type we cannot map
        # decodes as `list[Any]`, so naming it here would make the validator
        # STRICTER than the decoder — reintroducing the very asymmetry this
        # function exists to remove.
        if item_type_name in _JSON_TYPE_TO_PY:
            visible["items"] = {"type": item_type_name}
    else:
        # object / unknown / composite — exactly what StructuredDict receives.
        return cast("dict[str, Any]", strip_xgrammar_unsupported(output_schema))

    for data_key in ("enum", "const"):
        if data_key in output_schema:
            visible[data_key] = deepcopy(output_schema[data_key])
    return visible


def _output_type_from_schema(
    output_schema: dict[str, Any] | None,
    provider: ModelProvider,
) -> Any:
    """Translate a JSON-schema fragment into a pydantic-ai ``output_type``.

    Shapes:
      - None / missing       -> ``str`` (plain text, current default)
      - ``{"type": "<scalar>"}`` -> the bare Python type (bool/int/float/str)
      - ``{"type": "array", "items": …}`` -> ``list[<item-type>]``
      - ``{"type": "object", …}`` -> ``StructuredDict(schema, name="OutputData")``

    Provider wrapping mirrors the per-provider modes already established in
    ``_setup_agent``:
      - ``inference_router`` -> ``NativeOutput(...)`` (vLLM honours OpenAI's
        ``response_format=json_schema`` natively).
      - ``mlx`` -> ``PromptedOutput(...)`` (local MLX shims don't reliably
        honour ``response_format``; prompted mode is safest).
      - everything else -> the bare type (tool / function-calling mode).
    """
    if not output_schema:
        return str

    raw_type = output_schema.get("type")
    inner: Any
    if raw_type in _JSON_TYPE_TO_PY:
        inner = _JSON_TYPE_TO_PY[raw_type]
    elif raw_type == "array":
        items = output_schema.get("items")
        item_type_name = items.get("type") if isinstance(items, dict) else None
        item_py = _JSON_TYPE_TO_PY.get(item_type_name, Any) if item_type_name else Any
        inner = list[item_py]  # type: ignore[valid-type]
    else:
        # object / unknown / composite — hand the schema to StructuredDict.
        # pydantic-ai will raise UserError("Schema must be an object") for
        # anything it can't wrap; that's the right behaviour for genuinely
        # malformed input.
        #
        # Strip first: this schema becomes a vLLM guided-decoding grammar, and
        # xgrammar silently disables structured decoding on any schema carrying
        # a keyword it cannot compile (minItems/maxItems chief among them) —
        # the model then answers unconstrained and the reply is often not JSON.
        #
        # Routed through `_decoder_visible_schema` so there is exactly ONE
        # definition of "what the decoder was told", shared with the validator
        # registered in `__init__`. When those two drifted apart, the model was
        # graded against a contract it had never seen and every run 500'd.
        from pydantic_ai import StructuredDict

        inner = StructuredDict(
            # non-None: `output_schema` was truthy-checked at the top.
            cast("dict[str, Any]", _decoder_visible_schema(output_schema)),
            name="OutputData",
            description="Structured output from the model",
        )

    if provider == "mlx":
        return PromptedOutput(inner)
    if provider == "inference_router":
        # strict=False prevents pydantic-ai's OpenAI profile transformer
        # from adding `additionalProperties: false` at the outer level on
        # type:object schemas. With the default (strict=None) the
        # transformer mutates top-level-oneOf schemas into a logically
        # empty shape that rejects every key in the response. vLLM's
        # guided decoding enforces structure at the inference layer.
        return NativeOutput(inner, strict=False)
    return inner


class ModelFactory:
    """Builds a pydantic-ai model instance (and its ModelSettings) from agent config.

    Pure config->model dispatch with no shared instance state. Kept in this module
    so the provider classes it references resolve from the module namespace (tests
    patch them as ``<module>.OpenAIChatModel`` etc.).
    """

    @staticmethod
    def enforce_local_only_provider(
        provider: str | None,
        *,
        call_site: str,
        tenant_byo_key: bool = False,
        tenant_id: str | None = None,
    ) -> None:
        """Refuse any provider that is not declared local, unless permitted.

        ⭐ THE POLICY IS AN ALLOWLIST, NOT A BLOCKLIST, AND THAT IS THE WHOLE
        POINT. It admits ``provider in LOCAL_PROVIDERS`` and refuses everything
        else — the seven paid vendors today, and equally a provider name that
        reaches the factory tomorrow with no registry row at all. A blocklist of
        the seven would be bypassed by the eighth on the day it is added, which
        is exactly the failure mode this guard exists to make impossible.

        An external provider is admitted on exactly TWO grounds (owner ruling
        2026-10-01), and neither is a property of the provider:

        1. ``tenant_byo_key`` — the credential is the TENANT'S OWN, resolved from
           its ``tenant_llm_config`` row by the BYO resolver (own BYO, or managed
           BYO the owner set for that one tenant). The flag is set only by
           ``llm_config_resolver`` (worker path) and
           ``resolve_tenant_byo_providers`` (chat path), never from request
           input. ``get_api_key`` then returns that key and NEVER the env key.
        2. the tenant is a PROVEN commercial-platform workspace
           (``src.auth.tenant_proof.owner_commercial_keys_permitted``: in the
           gitops ``COMMERCIAL_PLATFORM_TENANT_IDS`` set AND proven by this
           request — licence introspection or a verified end-user token; owner
           ruling 2026-10-03). Only this ground may use the owner's commercial
           env keys (``XAI_API_KEY`` …). A customer tenant, an unknown tenant,
           NO tenant, a platform tenant merely ASSERTED in a header, and a run
           with no proving request context are all refused: no customer may
           ever spend the owner's key. The deployment-wide
           ``ALLOW_EXTERNAL_LLM_PROVIDERS`` flag that used to gate this ground
           was REMOVED (fr#1385); it is no longer read.

        ``tenant_id`` is the config's baked tenant; when absent the request's
        identity ContextVar is read. Absent both, the tenant is unknown and
        ground 2 fails closed.

        Called from TWO independent choke points, deliberately:

        1. ``initialize_model`` — the dispatch every production path funnels
           through (``PydanticAIAgent._setup_agent`` and the /providers chat
           router both reach a model only via this method). A new ``elif``
           branch added below is therefore already covered.
        2. ``get_api_key`` — called by every external creator as its FIRST
           statement and by NO local one (the local creators use literal
           sentinels: ``"mlx-local"``, ``"vlm-local"``, or no key at all). So a
           future vendor creator invoked DIRECTLY, bypassing the dispatch
           entirely, still cannot obtain the credential it needs to reach the
           vendor. A guard at the dispatch alone would miss that call.

        Args:
            provider: ``config.model_provider``; ``None`` and unknown names are
                refused exactly like a known paid vendor.
            call_site: Where the refusal fired, for the log/error text.
            tenant_byo_key: ``config.tenant_byo_key``.
            tenant_id: ``config.tenant_id``.

        Raises:
            UserError: When the provider is not local and neither ground holds.
        """
        if provider in LOCAL_PROVIDERS:
            return
        if tenant_byo_key:
            return
        from src.api.identity_context import get_tenant_id_from_context
        from src.auth.tenant_proof import owner_commercial_keys_refusal

        effective_tenant = tenant_id or get_tenant_id_from_context() or None
        refusal = owner_commercial_keys_refusal(effective_tenant)
        if refusal is None:
            return
        logger.error(
            "external_llm_provider_refused provider=%s call_site=%s tenant=%s "
            "reason=%s tenant_byo_key=False",
            provider,
            call_site,
            effective_tenant,
            refusal,
        )
        raise UserError(
            f"Model provider {provider!r} is refused at {call_site}: tenant "
            f"{effective_tenant!r} may not use the platform's provider keys "
            f"({refusal}) and has no usable BYO key of its own for this provider. "
            "Customer tenants reach an external provider only with their own (or "
            "managed) BYO key; the platform's keys serve only a commercial-platform "
            "workspace (gitops COMMERCIAL_PLATFORM_TENANT_IDS) that the request "
            "PROVED with a licence or a verified end-user token. Permitted local "
            f"providers are {sorted(LOCAL_PROVIDERS)}. If this provider runs on our "
            "own hardware, declare it `external: False` in "
            "src/worker/provider_registry.PROVIDER_METADATA."
        )

    @staticmethod
    def _owner_env_key(env_var: str) -> str | None:
        """Read one of the OWNER's provider keys from the pod environment.

        The single read site, so a test can prove it is never reached for a
        customer tenant. Only ``get_api_key`` calls it, and only after
        ``enforce_local_only_provider`` admitted ground 2 (proven platform).
        """
        return os.getenv(env_var)

    @staticmethod
    def get_api_key(config: PydanticAIAgentConfig, env_var: str) -> str | None:
        """Return the credential for an external creator.

        ⛔ This is the SECOND local-only barrier, not a duplicate of the one in
        ``initialize_model``. Every external creator calls this to fetch its
        credential and no local creator does, so refusing here covers a vendor
        creator called directly (tests, a future helper, a new code path that
        forgets the dispatch). See ``enforce_local_only_provider``.

        ⛔ A tenant-resolved key (``config.tenant_byo_key``) is returned AS-IS and
        the env var is never consulted, even when the key is empty: an empty BYO
        key must fail as "no key", never be silently topped up with the owner's.

        Args:
            config: Agent configuration that may contain API key
            env_var: Environment variable name to check for API key

        Returns:
            API key string if found, None otherwise

        Raises:
            UserError: If the provider is not local and neither the tenant-BYO
                ground nor the proven-platform ground holds.
        """
        tenant_byo_key = bool(getattr(config, "tenant_byo_key", False))
        ModelFactory.enforce_local_only_provider(
            getattr(config, "model_provider", None),
            call_site="ModelFactory.get_api_key",
            tenant_byo_key=tenant_byo_key,
            tenant_id=getattr(config, "tenant_id", None),
        )
        if tenant_byo_key:
            return config.api_key or None
        return config.api_key or ModelFactory._owner_env_key(env_var)

    @staticmethod
    def get_model_settings(
        config: PydanticAIAgentConfig, model_instance: models.Model
    ) -> ModelSettings | None:
        """Helper to create model_settings based on config and model type.

        Args:
            config: Agent configuration containing temperature setting
            model_instance: The initialized model instance

        Returns:
            Appropriate ModelSettings instance for the model type, or None
        """
        resolved_max_tokens = _resolve_max_tokens(config)

        agent_model_settings: ModelSettings | None = None
        if config.temperature is not None:
            if isinstance(model_instance, OpenAIChatModel):
                if resolved_max_tokens is not None:
                    agent_model_settings = OpenAIChatModelSettings(
                        temperature=config.temperature, max_tokens=resolved_max_tokens
                    )
                    logger.info(
                        f"Agent {config.name} - Created OpenAIChatModelSettings for "
                        f"{config.model_provider}: temperature={config.temperature}, "
                        f"max_tokens={resolved_max_tokens}"
                    )
                else:
                    agent_model_settings = OpenAIChatModelSettings(temperature=config.temperature)
            elif isinstance(model_instance, GroqModel):
                if resolved_max_tokens is not None:
                    agent_model_settings = GroqModelSettings(
                        temperature=config.temperature, max_tokens=resolved_max_tokens
                    )
                else:
                    agent_model_settings = GroqModelSettings(temperature=config.temperature)
            elif resolved_max_tokens is not None:
                agent_model_settings = ModelSettings(
                    temperature=config.temperature, max_tokens=resolved_max_tokens
                )
            else:
                agent_model_settings = ModelSettings(temperature=config.temperature)

        # Sampling seed, applied as a post-step for the same reason as
        # ``extra_body`` below: ``seed`` is a key of the base ``ModelSettings``
        # TypedDict, and pydantic-ai's OpenAI-compatible models send it as the
        # request body's top-level ``seed``. Absent entirely when unset, so a
        # worker without a seed sends exactly what it sent before.
        if config.seed is not None:
            if agent_model_settings is None:
                agent_model_settings = ModelSettings()
            agent_model_settings["seed"] = config.seed

        # Chain-of-thought switch. Applied as a post-step rather than threaded
        # through each provider branch above because ``extra_body`` is declared on
        # the base ``ModelSettings`` TypedDict (pydantic_ai/settings.py) and is
        # therefore a valid key on every variant the branches construct; the
        # OpenAI-compat models read it at request time (models/openai.py). Left
        # absent entirely when the resolver returns None, so the default-off case
        # adds no key at all.
        reasoning_extra_body = _resolve_reasoning_extra_body(config)
        if reasoning_extra_body is not None and agent_model_settings is not None:
            agent_model_settings["extra_body"] = reasoning_extra_body
            logger.info(
                "Agent %s - enable_reasoning=%s sent to %s as extra_body=%s",
                config.name,
                config.enable_reasoning,
                config.model_provider,
                reasoning_extra_body,
            )

        # Repetition controls, same "absent ⇒ no key" rule as the seed. Applied
        # AFTER the reasoning block because repetition_penalty shares
        # ``extra_body`` with it and must merge into it, never replace it.
        if config.frequency_penalty is not None:
            if agent_model_settings is None:
                agent_model_settings = ModelSettings()
            agent_model_settings["frequency_penalty"] = config.frequency_penalty
        if config.repetition_penalty is not None:
            if config.model_provider in SELF_HOSTED_PROVIDERS:
                if agent_model_settings is None:
                    agent_model_settings = ModelSettings()
                agent_model_settings["extra_body"] = _with_repetition_penalty(
                    agent_model_settings.get("extra_body"), config.repetition_penalty
                )
            else:
                logger.warning(
                    "Agent %s - repetition_penalty=%s NOT sent: provider %s is not "
                    "self-hosted and would reject the vLLM-only key",
                    config.name,
                    config.repetition_penalty,
                    config.model_provider,
                )
        return agent_model_settings

    @staticmethod
    def initialize_model(config: PydanticAIAgentConfig) -> models.Model:  # noqa: PLR0911 -- one return per provider; a dispatch dict would obscure the flat, greppable mapping
        """Initialize the appropriate model based on the provider configuration.

        Args:
            config: Agent configuration containing model provider and settings

        Returns:
            Initialized model instance for the specified provider

        Raises:
            UserError: If model provider is unsupported, or if it is not a
                local provider and the tenant is neither BYO nor a proven
                commercial-platform workspace
            Exception: If model initialization fails
        """
        # ⛔ BEFORE THE DISPATCH, NEVER INSIDE IT. Placed here rather than in the
        # seven external branches so that the branch a future provider adds is
        # covered the moment it is written: anything not in LOCAL_PROVIDERS never
        # reaches its creator at all. Seven copies inside the branches would be
        # seven things the eighth provider does not have to opt into.
        ModelFactory.enforce_local_only_provider(
            config.model_provider,
            call_site="ModelFactory.initialize_model",
            tenant_byo_key=config.tenant_byo_key,
            tenant_id=config.tenant_id,
        )

        logger.info(
            "Initializing model for provider '%s', model name '%s'.",
            config.model_provider,
            config.model,
        )

        try:
            if config.model_provider == "openai":
                return ModelFactory._create_openai_model(config)
            elif config.model_provider == "deepseek":
                return ModelFactory._create_deepseek_model(config)
            elif config.model_provider == "groq":
                return ModelFactory._create_groq_model(config)
            elif config.model_provider == "cohere":
                return ModelFactory._create_cohere_model(config)
            elif config.model_provider == "mlx":
                return ModelFactory._create_mlx_model(config)
            elif config.model_provider == "inference_router":
                return ModelFactory._create_inference_router_model(config)
            elif config.model_provider == "anthropic":
                return ModelFactory._create_anthropic_model(config)
            elif config.model_provider == "google":
                return ModelFactory._create_google_model(config)
            elif config.model_provider == "xai":
                return ModelFactory._create_xai_model(config)
            elif config.model_provider == "vllm":
                return ModelFactory._create_vllm_model(config)
            elif config.model_provider == "vlm_local":
                return ModelFactory._create_vlm_local_model(config)
            else:
                raise UserError(f"Unsupported model provider: {config.model_provider}")
        except Exception as e:
            logger.error(
                "Error initializing model '%s' for provider '%s': %s",
                config.model,
                config.model_provider,
                str(e),
                exc_info=True,
            )
            raise

    @staticmethod
    def _byo_guarded_http_client(config: PydanticAIAgentConfig) -> httpx.AsyncClient | None:
        """An httpx client for a TENANT BYO ``base_url``, refusing owner infra per hop.

        Validator 3 D3: the tenant endpoint was checked at PUT and at resolve,
        but the SDKs follow redirects and re-resolve DNS at connect time. The
        ``request`` hook runs on every hop, so a 307 (or a re-pointed name) to an
        internal Service is refused before anything is sent. ``None`` (SDK
        default client) when the call is not on a tenant endpoint.
        """
        if not (getattr(config, "tenant_byo_key", False) and config.base_url):
            return None
        # WM-6: the transport dials the address it validated (no second DNS
        # lookup at connect), so a rebinding answer can never be the one used.
        # The hook stays as a per-hop, transport-independent belt.
        return httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=5),
            follow_redirects=True,
            event_hooks={"request": [refuse_owner_infra_request]},
            transport=pinned_tenant_transport(),
        )

    @staticmethod
    def _create_openai_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Create an OpenAI model.

        Args:
            config: Agent configuration containing API key and model settings

        Returns:
            OpenAIChatModel configured for OpenAI provider

        Raises:
            UserError: If API key is not provided
        """
        api_key = ModelFactory.get_api_key(config, "OPENAI_API_KEY")
        if not api_key:
            raise UserError("API key required for OpenAI and not in config or env (OPENAI_API_KEY)")

        logger.info(f"Initializing OpenAI model: {config.model} with API key.")

        if config.base_url:
            logger.info(f"Using custom base URL for OpenAI: {config.base_url}")
            # Deliberately KEEPS the SDK default max_retries=2, unlike the
            # self-hosted providers above. api.openai.com is someone else's
            # capacity: a retry there does not deepen a queue we own, and their
            # 429/5xx responses are exactly what the SDK's backoff is for.
            custom_openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.base_url,
                http_client=ModelFactory._byo_guarded_http_client(config),
            )
            openai_provider = OpenAIProvider(openai_client=custom_openai_client)
        else:
            openai_provider = OpenAIProvider(api_key=api_key)

        return OpenAIChatModel(model_name=config.model, provider=openai_provider)

    @staticmethod
    def _create_deepseek_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Create a DeepSeek model.

        Args:
            config: Agent configuration containing API key and model settings

        Returns:
            OpenAIChatModel configured for DeepSeek provider

        Raises:
            UserError: If API key is not provided
        """
        api_key = ModelFactory.get_api_key(config, "DEEPSEEK_API_KEY")
        if not api_key:
            raise UserError(
                "API key required for DeepSeek and not in config or env (DEEPSEEK_API_KEY)"
            )

        logger.info(f"Initializing DeepSeek model: {config.model} with API key.")

        if config.base_url:
            logger.info(f"Using custom base URL for DeepSeek: {config.base_url}")
            # Keeps the SDK default max_retries=2 — external paid API, same
            # reasoning as the OpenAI branch above.
            custom_openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.base_url,
                http_client=ModelFactory._byo_guarded_http_client(config),
            )
            deepseek_provider = DeepSeekProvider(openai_client=custom_openai_client)
        else:
            deepseek_provider = DeepSeekProvider(api_key=api_key)

        return OpenAIChatModel(model_name=config.model, provider=deepseek_provider)

    @staticmethod
    def _create_xai_model(config: PydanticAIAgentConfig) -> XaiModel:
        """Create an xAI (Grok) model.

        Uses the native xAI SDK provider (base URL https://api.x.ai/v1). Unlike the
        OpenAI-compatible providers above, the xAI SDK speaks gRPC, so ``config.base_url``
        (an HTTP base URL) has no equivalent and is ignored with a warning rather than
        fed to the gRPC client as a broken channel target.

        Args:
            config: Agent configuration containing API key and model settings

        Returns:
            XaiModel configured for the xAI provider

        Raises:
            UserError: If API key is not provided
        """
        api_key = ModelFactory.get_api_key(config, "XAI_API_KEY")
        if not api_key:
            raise UserError("API key required for xAI and not in config or env (XAI_API_KEY)")

        if config.base_url:
            logger.warning(
                "Agent %s - custom base_url %r ignored for xAI: the native xAI SDK is "
                "gRPC and has no OpenAI-style HTTP base URL override.",
                config.name,
                config.base_url,
            )

        logger.info(f"Initializing xAI model: {config.model} with API key.")

        xai_provider = XaiProvider(api_key=api_key)
        # WM#337: the SDK's stream span must attach and detach in ONE task.
        return PinnedContextXaiModel(config.model, provider=xai_provider)

    @staticmethod
    def _create_mlx_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Create an MLX model using OpenAI compatibility with <think> tag stripping.

        Args:
            config: Agent configuration containing model settings

        Returns:
            OpenAIChatModel configured for MLX local inference with custom provider
        """
        logger.info(
            "Initializing MLX model '%s' with <think> tag stripping provider.",
            config.model,
        )

        mlx_base_url = config.base_url or os.getenv("OPENAI_API_BASE")
        if not mlx_base_url:
            raise UserError(
                "No MLX instance URL configured. Set base_url via MLX model registry "
                "resolution, or set OPENAI_API_BASE environment variable."
            )
        # Ensure base_url ends with /v1 for OpenAI client compatibility
        if not mlx_base_url.rstrip("/").endswith("/v1"):
            mlx_base_url = mlx_base_url.rstrip("/") + "/v1"
        logger.info(f"Using MLX base URL: {mlx_base_url}")

        # ⛔ EVERY MLX_INSTANCE_*_URL POINTS AT INFERENCE-ROUTER (verified live
        # 2026-10-07 on the workers Deployment). This client used to send no
        # identity at all — no tenant, licence token, app, project, actor or job
        # — so IR (licence-enforcing since 2026-10-07) could neither admit nor
        # attribute it, and WM's own emitter refuses `mlx` (platform-routed), so
        # nothing metered it. It now carries the SAME identity hook and
        # placeholder bearer as the inference_router/vllm clients: IR stays the
        # single meter (funding=local_gpu on its side). A non-IR dev URL gets the
        # headers but never the tenant's bundle token (origin-gated).
        http_client = create_async_http_client()
        http_client.headers.update(
            ir_identity_default_headers(
                worker_name=config.name or "",
                tenant_id=config.tenant_id or "",
                license_id=config.license_id or "",
                subject_id=config.subject_id or "",
            )
        )
        http_client.event_hooks.setdefault("request", []).append(
            make_identity_event_hook(
                worker_name=config.name or "",
                worker_version=config.worker_version or "",
                fallback_tenant_id=config.tenant_id or "",
                fallback_license_id=config.license_id or "",
                fallback_subject_id=config.subject_id or "",
                pinned_provider=config.ir_provider or "",
            )
        )
        # Use custom provider that strips <think> tags
        provider = MLXOpenAIProvider(
            base_url=mlx_base_url,
            api_key=IR_CLIENT_API_KEY,
            http_client=http_client,
        )

        return OpenAIChatModel(model_name=config.model, provider=provider)

    @staticmethod
    def _create_vlm_local_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Route a ``vlm_local`` (vision) model THROUGH inference-router.

        ⛔ This used to dial ``VLM_LOCAL_URL`` (the raw
        vlm_backend on mini2 — verified live 2026-10-07, not an IR URL) with
        only lowercase ``x-tenant-id`` / ``x-user-id``: no licence token, no
        app / project / actor / job, and WM's emitter refuses ``vlm_local`` as
        platform-routed — so every vision call was unmetered. It now goes via
        inference-router pinned to IR's ``vlm_local`` provider (``X-Provider``
        single-provider path, no silent failover to a text model), with the
        full identity hook and the tenant's bundle token: IR is the single meter
        (funding=local_gpu on its side). While IR has no ``vlm_local`` provider
        enabled the call fails loudly at IR — the backend is dormant anyway.

        ``config.base_url`` is honoured only when it IS the inference-router
        origin; a stored raw-backend URL is ignored, never dialled.
        """
        from src.services.ir_bundle_tokens import is_ir_origin

        base = config.base_url if config.base_url and is_ir_origin(config.base_url) else None
        base = base or os.getenv("INFERENCE_ROUTER_URL")
        if not base:
            raise UserError("vlm_local is served via inference-router; set INFERENCE_ROUTER_URL.")
        logger.info(
            "Initializing vlm_local model '%s' via inference-router (pinned provider %s)",
            config.model,
            config.ir_provider or "vlm_local",
        )
        routed = config.model_copy(
            update={"base_url": base, "ir_provider": config.ir_provider or "vlm_local"}
        )
        return ModelFactory._create_inference_router_model(routed)

    @staticmethod
    def _create_groq_model(config: PydanticAIAgentConfig) -> GroqModel:
        """Create a Groq model.

        Args:
            config: Agent configuration containing API key and model settings

        Returns:
            GroqModel configured for Groq provider

        Raises:
            UserError: If API key is not provided
        """
        api_key = ModelFactory.get_api_key(config, "GROQ_API_KEY")
        if not api_key:
            raise UserError("API key required for Groq and not in config or env (GROQ_API_KEY)")

        logger.info(f"Initializing Groq model: {config.model} with API key.")

        if config.base_url:
            logger.info(f"Using custom base URL for Groq: {config.base_url}")
            custom_groq_client = AsyncGroq(
                api_key=api_key,
                base_url=config.base_url,
                http_client=ModelFactory._byo_guarded_http_client(config),
            )
            groq_provider = GroqProvider(groq_client=custom_groq_client)
        else:
            groq_provider = GroqProvider(api_key=api_key)

        return GroqModel(model_name=config.model, provider=groq_provider)

    @staticmethod
    def _create_cohere_model(config: PydanticAIAgentConfig) -> CohereModel:
        """Create a Cohere model.

        Args:
            config: Agent configuration containing API key and model settings

        Returns:
            CohereModel configured for Cohere provider

        Raises:
            UserError: If API key is not provided
        """
        api_key = ModelFactory.get_api_key(config, "COHERE_API_KEY")
        if not api_key:
            raise UserError("API key required for Cohere and not in config or env (COHERE_API_KEY)")

        logger.info(f"Initializing Cohere model: {config.model} with API key.")

        if config.base_url:
            logger.info(f"Using custom base URL for Cohere: {config.base_url}")
            # Same per-hop owner-infra refusal the other BYO clients get
            # (_byo_guarded_http_client). Cohere takes it as ``httpx_client``,
            # not ``http_client``, which is why it was missed in validator 3 D3.
            custom_cohere_client = AsyncCohereClient(
                api_key=api_key,
                base_url=config.base_url,
                httpx_client=ModelFactory._byo_guarded_http_client(config),
            )
            cohere_provider = CohereProvider(cohere_client=custom_cohere_client)
        else:
            cohere_provider = CohereProvider(api_key=api_key)

        return CohereModel(model_name=config.model, provider=cohere_provider)

    @staticmethod
    def _create_inference_router_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Create an inference-router model with identity-header injection.

        config.model is forwarded as the OpenAI `model` field; inference-router
        treats it as a model alias, or falls through to default_chain if it
        doesn't match. Schema-id-based routing is now self-derived by
        inference-router from the outbound request's
        response_format.json_schema fingerprint — no header needed.

        ⚠️ ``local-llm`` PINS TO pgx-local ONLY WHILE pgx-local IS THE SOLE
        ENABLED OWNER OF THAT ALIAS, which is a live configuration fact about
        inference-router, not a property of the alias. This line used to claim
        the alias "auto-pins to pgx-local" flatly; that was FALSE when written —
        deepseek co-owned the alias, so ``_auto_pin_by_alias`` saw an ambiguous
        owner set and the request fell through to the default chain, i.e. a
        request asking for "local" could be served by a paid API. It became true
        only on 2026-09-17, when deepseek and xai were disabled in IR and
        pgx-local was left as the only owner.

        ⛔ So it is CONTINGENT, and it reverts the moment a second owner of the
        alias is enabled in inference-router — no change here, no deploy, no
        signal on this side. Do not read it as a routing guarantee, and do not
        rely on it as the mechanism that keeps a run local: the thing that
        actually enforces that is ``enforce_local_only_provider`` above, which
        refuses to build an external client at all.

        Args:
            config: Agent configuration containing base URL and model settings

        Returns:
            OpenAIChatModel configured for inference-router provider

        Raises:
            UserError: If base URL is not provided via config or environment variable
        """
        ir_base_url = config.base_url or os.getenv("INFERENCE_ROUTER_URL")
        if not ir_base_url:
            raise UserError(
                "No inference-router URL configured. Set base_url in worker config "
                "or set INFERENCE_ROUTER_URL environment variable."
            )
        # Inference router uses /api prefix, not /v1
        if not ir_base_url.rstrip("/").endswith("/api"):
            ir_base_url = ir_base_url.rstrip("/") + "/api"

        # ⚠️ The `license_token=present|absent` field that used to close this
        # line is GONE with the credential itself (retired 2026-09-11). It was
        # presence-only by construction -- never the value -- so nothing that
        # was safe to log is lost.
        logger.info(
            f"Initializing inference-router model: model={config.model} at {ir_base_url} "
            f"tenant={config.tenant_id} subject={config.subject_id}"
        )

        provider = InferenceRouterOpenAIProvider(
            base_url=ir_base_url,
            tenant_id=config.tenant_id,
            license_id=config.license_id,
            subject_id=config.subject_id,
            worker_name=config.name,
            worker_version=config.worker_version,
            pinned_provider=config.ir_provider,
        )

        # Subclass (not a bare OpenAIChatModel) so IR's served_model /
        # served_provider reach ModelResponse.provider_details and the BYO
        # cost_event can be priced on the resolved model rather than the alias.
        return InferenceRouterOpenAIChatModel(model_name=config.model, provider=provider)

    @staticmethod
    def _create_anthropic_model(config: PydanticAIAgentConfig) -> AnthropicModel:
        """Create an Anthropic Claude model via pydantic-ai's native support."""
        api_key = ModelFactory.get_api_key(config, "ANTHROPIC_API_KEY")
        if not api_key:
            raise UserError(
                "API key required for Anthropic and not in config or env (ANTHROPIC_API_KEY)"
            )
        logger.info(f"Initializing Anthropic model: {config.model} with API key.")
        if config.base_url:
            provider = AnthropicProvider(
                api_key=api_key,
                base_url=config.base_url,
                http_client=ModelFactory._byo_guarded_http_client(config),
            )
        else:
            provider = AnthropicProvider(api_key=api_key)
        return AnthropicModel(model_name=config.model, provider=provider)

    @staticmethod
    def _create_google_model(config: PydanticAIAgentConfig) -> "GoogleModel":  # noqa: UP037 -- GoogleModel only imported under TYPE_CHECKING; no `from __future__ import annotations` in this module
        """Create a Google Gemini model via pydantic-ai's native support (BYO key)."""
        api_key = ModelFactory.get_api_key(config, "GEMINI_API_KEY")
        if not api_key:
            raise UserError(
                "API key required for Google Gemini and not in config or env (GEMINI_API_KEY)"
            )
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        logger.info(f"Initializing Google Gemini model: {config.model} with API key.")
        if config.base_url:
            provider = GoogleProvider(
                api_key=api_key,
                base_url=config.base_url,
                http_client=ModelFactory._byo_guarded_http_client(config),
            )
        else:
            provider = GoogleProvider(api_key=api_key)
        return GoogleModel(model_name=config.model, provider=provider)

    @staticmethod
    def _create_vllm_model(config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Create an in-cluster vLLM model (OpenAI-compatible, no API key).

        ⚠️ ``VLLM_URL`` NO LONGER POINTS AT vLLM. gitops #2999 (2026-08-24
        15:05:28Z) repointed it to the inference-router
        (an in-cluster inference-router Service origin), which
        serves ``/v1/chat/completions`` as an alias of its native ``/api`` route
        since inference-router#112 — the SAME handler object, not a copy.

        That made a client which had never needed an identity suddenly visible
        in a system that logs one. Measured 2026-08-24T15:30Z->08-25T06:00Z:
        ``worker_name`` was NULL on all 95 ``upstream_empty_content`` rows while
        POPULATED on every ``upstream_empty_result`` row of the same period,
        same provider, same model. The difference is not the route and not the
        router — ``src/routers/completions.py:148`` reads ``X-Worker-Name`` as a
        route-signature Header param on both routes. It is that
        ``InferenceRouterOpenAIProvider`` above sends the header and this client
        never did, because real vLLM had no such concept.

        The cost of that gap is not cosmetic: ``InferenceRouterEmptyCompletionsHigh``
        fires on exactly this traffic and structurally cannot name the agent
        responsible, because the router-side metric has no caller label either.
        The header is the only attribution that exists.
        """
        base = config.base_url or os.getenv("VLLM_URL") or _vllm_default_url()
        if not base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        logger.info(f"Initializing vLLM model: {config.model} at {base}")
        # Built through pydantic-ai's own factory, NOT a bare httpx.AsyncClient:
        # create_async_http_client() carries the OpenAI-matched timeouts and the
        # pydantic-ai User-Agent, and losing those to add one header would be a
        # bad trade. Only the header is added.
        #
        # NB passing our own client means OpenAIProvider does not set
        # ``_own_http_client``, so it is not auto-closed on provider __aexit__.
        # That matches InferenceRouterOpenAIProvider in this same file, which
        # supplies its own AsyncOpenAI for the same reason, and nothing here
        # enters providers as context managers.
        http_client = create_async_http_client()
        # Constructor-time identity headers from the SAME helper
        # InferenceRouterOpenAIProvider uses (X-Worker-Name always -- empty
        # string, never "None"; X-Tenant-Id / X-License-Id / X-Subject-Id only
        # when they carry a value).
        http_client.headers.update(
            ir_identity_default_headers(
                worker_name=config.name or "",
                tenant_id=config.tenant_id or "",
                license_id=config.license_id or "",
                subject_id=config.subject_id or "",
            )
        )
        # ⛔ X-Worker-Name ALONE WAS HALF THE FIX, AND THE MISSING HALF COST
        # REAL ATTRIBUTION. Since VLLM_URL points at inference-router (see the
        # docstring above), this is an inference-router client, and IR meters
        # what it can identify. With only the worker name it logged
        #   worker_name=<real>  tenant_id=nil  license_id=unknown  app_id=unknown
        # for every call — measured 2026-08-27, 116 of 287 routed calls in one
        # 3h window, all `principle_consistency`. Losing license and app TOO is
        # what distinguishes "never sent identity" from "sent it empty".
        #
        # The hook is the SAME object InferenceRouterOpenAIProvider installs, by
        # construction rather than by copy — two copies diverging is precisely
        # how this gap opened.
        http_client.event_hooks.setdefault("request", []).append(
            make_identity_event_hook(
                worker_name=config.name or "",
                worker_version=config.worker_version or "",
                fallback_tenant_id=config.tenant_id or "",
                fallback_license_id=config.license_id or "",
                fallback_subject_id=config.subject_id or "",
                pinned_provider=config.ir_provider or "",
            )
        )
        # ⛔ NOT ``api_key="EMPTY"``. That was the vLLM convention, and this is an  # pragma: allowlist secret
        # inference-router client: it presents the same placeholder as the
        # inference_router path, which the hook above replaces with the tenant's
        # bundle token on the IR origin. Same run => same Authorization, on both
        # provider names. An unresolved tenant/licence fails loud in the shared
        # hook (outbound_tenant_unresolved ERROR / outbound_license_unresolved
        # WARNING, header omitted, never substituted) -- exactly as on that path.
        provider = OpenAIProvider(base_url=base, api_key=IR_CLIENT_API_KEY, http_client=http_client)
        # VLLM_URL points at inference-router, so lift IR's served_* read-back
        # fields exactly as the inference_router path does (absent ⇒ unchanged).
        return InferenceRouterOpenAIChatModel(model_name=config.model, provider=provider)


class PromptBuilder:
    """Builds context-aware prompts from graph-execution context. Stateless."""

    @staticmethod
    def build_context_aware_prompt(
        task: str, graph_context: dict[str, Any], node_config: dict[str, Any]
    ) -> str:
        """Build a context-aware prompt that includes graph execution information.

        Args:
            task: The primary task for the agent to execute
            graph_context: Context information from graph execution
            node_config: Configuration specific to the current node

        Returns:
            Enhanced prompt string with context information
        """
        context_parts = [task]

        # Add graph context information
        if graph_context:
            context_info = PromptBuilder.build_graph_context_info(graph_context)
            if context_info:
                context_parts.append(f"\nGraph Execution Context:\n{chr(10).join(context_info)}")

        # Add node configuration if available
        if node_config:
            context_parts.append(f"\nNode Configuration: {node_config}")

        return "\n".join(context_parts)

    @staticmethod
    def build_graph_context_info(graph_context: dict[str, Any]) -> list[str]:
        """Build graph context information strings.

        Args:
            graph_context: Dictionary containing graph execution context

        Returns:
            List of formatted context information strings
        """
        context_info = []

        if graph_context.get("session_id"):
            context_info.append(f"Session ID: {graph_context['session_id']}")

        if graph_context.get("graph_id"):
            context_info.append(f"Graph ID: {graph_context['graph_id']}")

        if graph_context.get("node_id"):
            context_info.append(f"Current Node: {graph_context['node_id']}")

        # Include completed nodes for context
        completed_nodes = graph_context.get("completed_nodes", [])
        if completed_nodes:
            context_info.append(f"Completed nodes: {', '.join(completed_nodes)}")

        # Include previous outputs that might be relevant
        PromptBuilder._add_previous_outputs_context(graph_context, context_info)

        # Include variables
        PromptBuilder._add_variables_context(graph_context, context_info)

        return context_info

    @staticmethod
    def _add_previous_outputs_context(
        graph_context: dict[str, Any], context_info: list[str]
    ) -> None:
        """Add previous outputs to the context information.

        Args:
            graph_context: Dictionary containing previous node outputs
            context_info: List to append output information to
        """
        previous_outputs = graph_context.get("previous_outputs", {})
        if not previous_outputs:
            return

        output_summary = []
        for node_id, output in previous_outputs.items():
            # Truncate long outputs for context
            output_str = str(output)
            if len(output_str) > 200:
                output_str = output_str[:200] + "..."
            output_summary.append(f"{node_id}: {output_str}")

        if output_summary:
            context_info.append(f"Previous outputs: {'; '.join(output_summary)}")

    @staticmethod
    def _add_variables_context(graph_context: dict[str, Any], context_info: list[str]) -> None:
        """Add variables to the context information.

        Args:
            graph_context: Dictionary containing graph variables
            context_info: List to append variable information to
        """
        variables = graph_context.get("variables", {})
        if not variables:
            return

        var_summary = []
        for var_name, value in variables.items():
            var_summary.append(f"{var_name}={value}")
        context_info.append(f"Variables: {', '.join(var_summary)}")


class ResponseExtractor:
    """Extracts text output and token usage from a pydantic-ai run result."""

    @staticmethod
    def extract_served_fields(response: Any) -> dict[str, str]:
        """IR's ``served_*`` fields from the LAST model response of a run.

        Only the :data:`IR_SERVED_FIELDS` actually present are returned; ``{}``
        for a provider that does not report them or on any traversal error —
        never a guess.
        """
        try:
            for msg in reversed(response.all_messages()):
                details = getattr(msg, "provider_details", None)
                if not details:
                    continue
                return {k: details[k] for k in IR_SERVED_FIELDS if details.get(k)}
        except Exception:  # never fail a run over a routing annotation
            return {}
        return {}

    @staticmethod
    def extract_served_routing(
        response: FrameworkAgentRunResult[Any],
    ) -> tuple[str | None, str | None]:
        """Return (served_model, served_provider) reported by inference-router.

        Reads ``provider_details`` off the LAST ModelResponse in the run — see
        :class:`InferenceRouterOpenAIChatModel`, which lifts IR's non-standard
        ``served_model`` / ``served_provider`` body fields into that dict.

        KNOWN, ACCEPTED LIMITATION: a tool-calling loop makes N model calls but
        this agent emits ONE aggregated cost_event, so served_model here is
        last-write-wins across those calls. That is correct while IR resolves a
        given alias to the same upstream for every call in a run (it does); if
        IR ever fails over mid-run, the aggregate bills at the last call's
        model. Splitting the event per model call is the real fix and is out of
        scope for #137.

        Returns (None, None) when IR predates the companion PR that adds these
        fields, or on any traversal error — the caller then behaves exactly as
        it did before.
        """
        try:
            for msg in reversed(response.all_messages()):
                details = getattr(msg, "provider_details", None)
                if not details:
                    continue
                served_model = details.get("served_model")
                served_provider = details.get("served_provider")
                if served_model or served_provider:
                    return served_model, served_provider
                # A ModelResponse with details but no served_* fields is still
                # the last model call — stop rather than reading a stale older
                # response further back in the message history.
                return None, None
        except Exception as e:  # pragma: no cover -- defensive; never fail a run
            logger.debug(f"Could not extract served routing from response: {e}")
        return None, None

    @staticmethod
    def extract_text_output(
        response: FrameworkAgentRunResult[Any],
        agent_name: str,
    ) -> Any:
        """Extract text output from agent response, with fallbacks.

        When output_type is not set (defaults to str), PydanticAI may return
        None if the model response couldn't be coerced. This method tries
        multiple extraction strategies.

        Args:
            response: The agent run result from PydanticAI
            agent_name: Name of the owning agent (used only for the fallback warning)

        Returns:
            Extracted output (str, dict, or other)
        """
        # Strategy 1: Direct output
        if response.output is not None:
            return response.output

        # Strategy 2: Extract from last message content
        try:
            all_messages = response.all_messages()
            if all_messages:
                # Walk messages in reverse to find last model response
                for msg in reversed(all_messages):
                    # Check for model text response parts
                    if hasattr(msg, "parts"):
                        for part in msg.parts:
                            if hasattr(part, "content") and part.content:
                                return part.content
                    # Direct content attribute
                    if hasattr(msg, "content") and msg.content:
                        return msg.content
        except Exception as e:
            logger.debug(f"Could not extract from messages: {e}")

        # Strategy 3: Try all_messages_json and parse last assistant message
        try:
            messages_json = response.all_messages_json()
            if messages_json:
                import json

                messages = json.loads(messages_json)
                for msg in reversed(messages):
                    if msg.get("role") == "model-text-response" or msg.get("role") == "assistant":
                        parts = msg.get("parts", [])
                        for part in parts:
                            if part.get("part_kind") == "text" and part.get("content"):
                                return part["content"]
                        # Direct content
                        if msg.get("content"):
                            return msg["content"]
        except Exception as e:
            logger.debug(f"Could not extract from messages JSON: {e}")

        logger.warning(
            f"Agent {agent_name} - Could not extract output from response. "
            f"output={response.output}, type={type(response.output)}"
        )
        return None

    @staticmethod
    def finish_reason(run_result: FrameworkAgentRunResult[Any] | None) -> str | None:
        """Return the provider-reported finish reason of the last model response.

        pydantic-ai DOES surface this in the pinned version (>=1.99.0, locked at
        1.107.0): ``ModelResponse.finish_reason`` is a normalized
        ``Literal['stop', 'length', 'content_filter', 'tool_call', 'error']``
        (pydantic_ai/messages.py) that every model class populates from its own
        vendor field -- e.g. ``OpenAIChatModel._map_finish_reason`` maps the
        OpenAI/vLLM ``choices[0].finish_reason`` onto it. ``'length'`` is the
        authoritative "ran out of max_tokens" signal, so it is preferred over
        the token-equality proxy below.

        Returns None when the reason is not reachable (no run result, an older
        message shape, or a provider that did not report one) -- the caller then
        falls back to comparing completion tokens against the cap.
        """
        if run_result is None:
            return None
        try:
            for message in reversed(list(run_result.all_messages())):
                if getattr(message, "kind", None) == "response":
                    reason = getattr(message, "finish_reason", None)
                    return reason if isinstance(reason, str) else None
        except Exception as exc:  # pragma: no cover - defensive, shape-dependent
            logger.debug("Could not read finish_reason: %s", exc)
        return None

    @staticmethod
    def check_max_tokens_cap(
        agent_name: str,
        completion_tokens: int | None,
        max_tokens_cap: int | None,
        model_provider: str | None,
        run_result: FrameworkAgentRunResult[Any] | None,
    ) -> bool:
        """Warn + count when a completion was truncated by its max_tokens cap.

        A truncated completion is indistinguishable from a malformed one at the
        call site (see the DGF-007 note in ``_setup_agent``): for a structured
        worker pydantic-ai re-prompts and every retry truncates identically, and
        for a plain-text worker the truncated answer is returned as if complete.
        This is the detector that makes that visible without grepping pod logs.

        Detection order:
          1. ``finish_reason == 'length'`` -- authoritative, provider-reported.
          2. ``completion_tokens >= max_tokens_cap`` -- proxy, used only when the
             provider reported no finish reason at all. Equality is the real
             signal (a decode cannot exceed its budget); ``>=`` is defensive.

        A finish reason that is present and is NOT ``'length'`` wins over the
        proxy: the model genuinely stopped, it merely landed on the budget.

        Returns:
            True if a cap hit was detected and recorded.
        """
        if not isinstance(max_tokens_cap, int) or not isinstance(completion_tokens, int):
            return False
        try:
            reason = ResponseExtractor.finish_reason(run_result)
            if reason == "length":
                detection = "finish_reason=length"
            elif reason is None and completion_tokens >= max_tokens_cap:
                detection = "completion_tokens>=max_tokens (finish_reason unavailable)"
            else:
                return False

            logger.warning(
                "Agent %s - max_tokens cap hit (%s): provider=%s, max_tokens=%s, "
                "completion=%s. The response is TRUNCATED -- a structured worker will "
                "re-prompt and truncate identically, a plain-text worker returns the "
                "truncated answer as if complete. Raise this worker's max_tokens or "
                "shrink its output schema.",
                agent_name,
                detection,
                model_provider,
                max_tokens_cap,
                completion_tokens,
            )
            get_metrics_client().record_max_tokens_cap_hit(
                agent_name=agent_name,
                model_provider=model_provider,
            )
        except Exception as exc:
            logger.warning("Agent %s - Error checking max_tokens cap: %s", agent_name, exc)
            return False
        return True

    @staticmethod
    def extract_token_usage(
        agent_name: str,
        run_result: FrameworkAgentRunResult[Any] | None,
        agent: Any,
        max_tokens_cap: int | None = None,
        model_provider: str | None = None,
    ) -> TokenUsageMetrics | None:
        """Extract token usage metrics from agent run result with fallback to model usage.

        ``max_tokens_cap`` is the value ``_resolve_max_tokens`` returned for this
        run (threaded in from the agent instance, never re-derived here, so it
        cannot drift from what was actually sent). When it is provided, each
        usage branch also runs the cap-hit detector.
        """
        try:
            run_usage = None
            if run_result and hasattr(run_result, "usage"):
                run_usage = run_result.usage()

            if run_usage and getattr(run_usage, "has_values", lambda: False)():
                prompt_tokens = getattr(run_usage, "input_tokens", None)
                completion_tokens = getattr(run_usage, "output_tokens", None)
                total_tokens = getattr(run_usage, "total_tokens", None)

                logger.info(
                    "Agent %s - Token usage: Prompt=%s, Completion=%s, Total=%s",
                    agent_name,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
                ResponseExtractor.check_max_tokens_cap(
                    agent_name=agent_name,
                    completion_tokens=completion_tokens,
                    max_tokens_cap=max_tokens_cap,
                    model_provider=model_provider,
                    run_result=run_result,
                )
                return TokenUsageMetrics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            model_usage = getattr(getattr(agent, "model", None), "usage", None)
            if model_usage:
                prompt_tokens = getattr(model_usage, "input_tokens", None)
                if prompt_tokens is None:
                    prompt_tokens = getattr(model_usage, "prompt_tokens", None)
                completion_tokens = getattr(model_usage, "output_tokens", None)
                if completion_tokens is None:
                    completion_tokens = getattr(model_usage, "completion_tokens", None)
                total_tokens = getattr(model_usage, "total_tokens", None)

                if any(
                    token_count is not None
                    for token_count in (prompt_tokens, completion_tokens, total_tokens)
                ):
                    logger.info(
                        "Agent %s - Token usage (legacy model usage): Prompt=%s, Completion=%s, Total=%s",
                        agent_name,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )
                    ResponseExtractor.check_max_tokens_cap(
                        agent_name=agent_name,
                        completion_tokens=completion_tokens,
                        max_tokens_cap=max_tokens_cap,
                        model_provider=model_provider,
                        run_result=run_result,
                    )
                    return TokenUsageMetrics(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

            logger.info("Agent %s - Token usage information not available.", agent_name)
        except Exception as tu_ex:
            logger.warning("Agent %s - Error extracting token usage: %s", agent_name, tu_ex)
        return None


class _SamplingProvenance(TypedDict):
    requested_model: str | None
    temperature: float | None
    seed: int | None
    max_tokens: int | None
    repetition_penalty: float | None
    frequency_penalty: float | None


def _uuid_or_none(value: str | None) -> str | None:
    """``value`` in canonical form if it is a UUID, else None.

    usage_metrics persists the correlation id; a malformed inbound header (the
    n8n execution id ``529894``) is still forwarded as before, but never stored.
    """
    if not value:
        return None
    try:
        return str(UUID(value))
    except ValueError, TypeError, AttributeError:
        return None


def _repetition_penalty_of(settings: ModelSettings) -> float | None:
    extra_body = settings.get("extra_body")
    if isinstance(extra_body, dict):
        value = extra_body.get("repetition_penalty")
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return None


def _sampling_provenance(
    model_instance: models.Model, model_settings: ModelSettings | None
) -> _SamplingProvenance:
    """What the request body will carry for model / sampling / budget keys.

    Read from the settings object pydantic-ai is handed, not from the stored
    worker config: temperature is defaulted twice on the way down (agent_factory
    ``None → 0.7``, and the API's ``None → 0.1`` on auto-create), and max_tokens
    is clamped/defaulted by ``_resolve_max_tokens``, so the config value is NOT
    what was sent. A key absent from ``model_settings`` is reported as None
    because it is absent from the wire too.
    """
    settings = model_settings or {}
    requested_model = getattr(model_instance, "model_name", None)
    return _SamplingProvenance(
        requested_model=requested_model if isinstance(requested_model, str) else None,
        temperature=settings.get("temperature"),
        seed=settings.get("seed"),
        max_tokens=settings.get("max_tokens"),
        repetition_penalty=_repetition_penalty_of(settings),
        frequency_penalty=settings.get("frequency_penalty"),
    )


def _temperature_unsent_reason(
    model_instance: models.Model, model_settings: ModelSettings | None
) -> str | None:
    """Why pydantic-ai will strip ``temperature`` from this model's requests, or None.

    Mirrors pydantic-ai's own drop rules (1.107), so a temperature that never
    reaches the wire is REPORTED instead of silently ignored:

    * ``OpenAIChatModel`` (and subclasses: vllm / mlx / vlm_local /
      inference_router / deepseek / openai / xai-compat): dropped when the
      profile lists it in ``openai_unsupported_model_settings``, or when the
      profile is a reasoning model with reasoning active
      (``_drop_sampling_params_for_reasoning``: o-series / GPT-5 always,
      GPT-5.1+ unless ``openai_reasoning_effort`` is ``none``/unset).
    * ``AnthropicModel``: dropped when the profile sets
      ``anthropic_disallows_sampling_settings`` (Claude Opus 4.7/4.8).
    * Groq / Cohere / Google: always sent.
    """
    try:
        return _temperature_unsent_reason_unguarded(model_instance, model_settings)
    except Exception as exc:  # reporting only: must never break agent construction
        logger.warning(
            "could not evaluate whether model %r honours temperature: %s",
            getattr(model_instance, "model_name", "?"),
            exc,
        )
        return None


def _temperature_unsent_reason_unguarded(
    model_instance: models.Model, model_settings: ModelSettings | None
) -> str | None:
    settings = model_settings or {}
    name = getattr(model_instance, "model_name", "?")
    # The classes are read from pydantic-ai's own modules, not this module's
    # names: tests patch ``OpenAIChatModel`` here with a MagicMock.
    if isinstance(model_instance, _pai_openai.OpenAIChatModel):
        profile = OpenAIModelProfile.from_profile(model_instance.profile)
        if "temperature" in profile.openai_unsupported_model_settings:
            return f"model {name!r} profile lists temperature as unsupported"
        if profile.openai_supports_reasoning:
            effort = settings.get("openai_reasoning_effort")
            if not profile.openai_supports_reasoning_effort_none or effort not in (None, "none"):
                return f"reasoning model {name!r} rejects sampling parameters"
        return None
    if isinstance(model_instance, _pai_anthropic.AnthropicModel):
        profile = AnthropicModelProfile.from_profile(model_instance.profile)
        if profile.anthropic_disallows_sampling_settings:
            return f"model {name!r} disallows sampling settings"
    return None


_TEMPERATURE_UNSENT_LOGGED: set[tuple[str | None, str | None]] = set()


def _log_temperature_unsent_once(provider: str | None, model: str | None, reason: str) -> None:
    """WARN once per (provider, model) that a per-message temperature was not sent."""
    key = (provider, model)
    if key in _TEMPERATURE_UNSENT_LOGGED:
        return
    _TEMPERATURE_UNSENT_LOGGED.add(key)
    logger.warning(
        "per-message temperature NOT sent to provider=%s model=%s: %s "
        "(reported as usage_metrics.temperature_not_sent)",
        provider,
        model,
        reason,
    )


# ⛔ Post-hoc schema-bound enforcement is OPT-IN, per RUN. The validator is baked
# into the cached agent at construction, so the per-run choice reaches it through
# these ContextVars (task-local, so concurrent runs on one cached agent cannot see
# each other's flag). Default OFF: requirements_management's principle_consistency
# has a written contract that WM "will not reject an oversized reply" — RM clamps
# instead — and a default-on 422 would quarantine its most-conflicted procedures.
_ENFORCE_SCHEMA_BOUNDS: ContextVar[bool] = ContextVar("enforce_schema_bounds", default=False)
# Paths of count/length bound violations seen by the LAST validator call of this
# run (a retry that lands in bounds clears them). Stamped on usage_metrics either
# way, so how often bounds WOULD bite is measurable before anyone opts in.
_BOUND_VIOLATIONS: ContextVar[list[str] | None] = ContextVar("bound_violations", default=None)


def _clear_recorded_bound_violations() -> None:
    """Reset this run's recorded bound violations (last validator attempt wins).

    A retry that lands in bounds must not inherit the previous attempt's paths.
    """
    recorded = _BOUND_VIOLATIONS.get()
    if recorded is not None:
        recorded.clear()


#: ``aa_prompt_validation_failures_total`` ``error_type`` for one opted-in bound
#: violation that was sent back to the model as a ``ModelRetry`` (one per
#: attempt). The run's FINAL failure, once the retry budget is spent, is still
#: counted as ``SCHEMA_BOUND_VIOLATION``.
SCHEMA_BOUND_RETRY = "schema_bound_retry"

_BOUND_PHRASES: dict[str, tuple[str, str]] = {
    "minItems": ("at least", "item"),
    "maxItems": ("at most", "item"),
    "minLength": ("at least", "character"),
    "maxLength": ("at most", "character"),
}


class SchemaBoundRetry(ModelRetry):
    """An opted-in bound violation, sent back to the model as retry feedback.

    A ``ModelRetry`` (so pydantic-ai re-asks within the worker's output retry
    budget) that also carries the structured ``violations``, so that when the
    budget runs out the run can be failed with the typed
    ``SchemaBoundViolationError`` rather than pydantic-ai's generic
    ``UnexpectedModelBehavior`` (see ``_bound_retry_cause``).
    """

    def __init__(self, message: str, *, agent_name: str, violations: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.violations = violations


def _bound_sentence(violation: dict[str, Any]) -> str:
    """``fix_outline must have at least 1 item`` — the bound, in the model's words."""
    field = violation["path"].lstrip("/").replace("/", ".") or "the output"
    limit = violation.get("limit")
    phrase = _BOUND_PHRASES.get(violation["keyword"])
    if phrase is None or not isinstance(limit, int):
        return f"{field} violates {violation['keyword']}"
    direction, unit = phrase
    return f"{field} must have {direction} {limit} {unit}{'' if limit == 1 else 's'}"


def _bound_retry_feedback(violations: list[dict[str, Any]]) -> str:
    """Model-readable retry message: path + keyword + bound + validator message."""
    lines = ["Output violates a size bound of the required schema. Fix these and answer again:"]
    for v in violations[:10]:
        lines.append(
            f"  - {_bound_sentence(v)} (at {v['path'] or '<root>'}: "
            f"{v['keyword']}={v.get('limit')}; {v['message']})"
        )
    return "\n".join(lines)


def _bound_retry_cause(error: BaseException) -> SchemaBoundRetry | None:
    """The ``SchemaBoundRetry`` that exhausted the retry budget, if it was one.

    pydantic-ai raises ``UnexpectedModelBehavior("Exceeded maximum output
    retries")`` ``from`` the LAST attempt's failure (the ``ModelRetry``, or the
    ``ToolRetryError`` wrapping it). Only the explicit ``__cause__`` chain is
    walked — never ``__context__`` — so an earlier attempt's bound violation can
    never be blamed for a later, different failure (a structural
    ``schema_invalid``, an empty response): those stay exactly as before.
    """
    seen: set[int] = set()
    cur: BaseException | None = error.__cause__
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, SchemaBoundRetry):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
    return None


def _raise_or_record_bound_violations(
    agent_name: str, bound_leaves: list[Any], count_failure: Callable[[str], None]
) -> None:
    """Record the violating paths; opted in ⇒ also raise ``SchemaBoundRetry``.

    Not opted in is advisory exactly as before (the caller goes on to the
    soft-violation log + counter); the paths only land in
    ``usage_metrics.bound_violations``. Opted in (#358, George 2026-10-10
    "retry then fail"), the model is re-asked with feedback naming each path and
    the bound from the ORIGINAL schema; if the retry budget runs out on a bound
    violation, ``_run_agent_with_timeout`` turns it into
    ``SchemaBoundViolationError`` (HTTP 422 ``schema_bound_violation``).
    """
    violations: list[dict[str, Any]] = [
        {
            "path": "/" + "/".join(str(p) for p in lf.absolute_path) if lf.absolute_path else "",
            "keyword": str(lf.validator),
            "message": str(lf.message)[:200],
            "limit": getattr(lf, "validator_value", None),
        }
        for lf in bound_leaves
    ]
    recorded = _BOUND_VIOLATIONS.get()
    if recorded is not None:
        recorded.extend(v["path"] for v in violations)
    if _ENFORCE_SCHEMA_BOUNDS.get():
        feedback = _bound_retry_feedback(violations)
        logger.warning(
            "%s: agent %r output violates an enforced schema bound; re-asking the model. %s",
            SCHEMA_BOUND_RETRY,
            agent_name,
            feedback,
        )
        count_failure(SCHEMA_BOUND_RETRY)
        raise SchemaBoundRetry(feedback, agent_name=agent_name, violations=violations)


@dataclass(frozen=True)
class RunSampling:
    """Per-RUN sampling overrides (a per-message value, winning over the config).

    Passed to ``agent.run(model_settings=...)`` rather than baked into the cached
    agent: baking would need one cache slot (and one never-evicted connection
    pool) per distinct value. Every field None ⇒ the run is exactly
    ``agent.run(input)``.
    """

    seed: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    temperature: float | None = None

    def is_empty(self) -> bool:
        """True when no per-run override was set."""
        return (
            self.seed is None
            and self.repetition_penalty is None
            and self.frequency_penalty is None
            and self.temperature is None
        )


class PydanticAIAgent(BaseAgent):
    """Agent class for PydanticAI framework with proper MCP server lifecycle management."""

    #: ``run()`` accepts per-run ``seed`` / ``repetition_penalty`` /
    #: ``frequency_penalty`` / ``temperature``. message_endpoints only passes them to workers that
    #: declare this; other agent types' ``run`` would TypeError.
    accepts_run_seed: ClassVar[bool] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        config: PydanticAIAgentConfig,
        agent_id: UUID | None = None,
    ) -> None:
        """Initialize the PydanticAI agent with configuration.

        Args:
            config: Configuration object containing agent settings
            agent_id: Optional unique identifier for the agent

        Raises:
            ValueError: If configuration is invalid
            UserError: If model provider is unsupported
        """
        # Initialize base agent
        super().__init__(agent_id)

        self.agent: Agent
        self.agent_id: UUID
        self.config: PydanticAIAgentConfig
        self.name: str
        self.db_path: str
        self._tool_registry: dict[str, Tool]
        self.mcp_servers_ready: bool
        self._mcp_exit_stack: AsyncExitStack | None
        # max_tokens actually sent for this agent; set in _setup_agent.
        self._max_tokens_cap: int | None
        # model / sampling / max_tokens actually sent; set in _setup_agent.
        self._sampling_provenance: _SamplingProvenance | None = None
        self._agent_model_settings: ModelSettings | None = None
        # Why this agent's model cannot take a temperature (None ⇒ it can).
        self._temperature_unsent_reason: str | None = None
        # Keywords stripped from the decoding grammar; set when the schema
        # validator registers. Stamped on usage_metrics per run.
        self._grammar_stripped_keywords: list[str] | None = None

        # ⛔ TRANSPORT REFCOUNT — read `cleanup()` before touching any of these.
        # One cached instance is shared by every CONCURRENT request on the same
        # cache slot, and they all share ONE httpx pool. `cleanup()` used to
        # close that pool the moment any single caller evicted the slot, which
        # severed the sockets of the siblings still awaiting `agent.run()`.
        # `_active_runs` counts runs currently inside `run()`; `_retired` means
        # "the cache has dropped me, close as soon as the last run leaves";
        # `_transport_closed` makes the close idempotent.
        self._active_runs: int = 0
        self._retired: bool = False
        self._transport_closed: bool = False

        self._setup_agent(config, agent_id)

    def _setup_agent(self, config: PydanticAIAgentConfig, agent_id: UUID | None) -> None:
        """Set up the agent instance with the provided configuration.

        Args:
            config: Agent configuration containing model and tool settings
            agent_id: Optional unique identifier, generates new UUID if None

        Raises:
            UserError: If model initialization fails
        """
        model_instance = self._initialize_model(config)
        # The exact max_tokens this run will send, kept on the instance so the
        # token-usage path can compare the reported completion against it (see
        # ResponseExtractor.check_max_tokens_cap). `_resolve_max_tokens` is a
        # pure function of `config` and `_get_model_settings` calls it on this
        # same object, so this cannot drift from what is actually sent.
        self._max_tokens_cap = _resolve_max_tokens(config)
        agent_model_settings = self._get_model_settings(config, model_instance)
        # Sampling provenance for usage_metrics, read back from the SAME settings
        # object handed to Agent(...) below, so it cannot drift from the wire.
        self._sampling_provenance = _sampling_provenance(model_instance, agent_model_settings)
        self._temperature_unsent_reason = _temperature_unsent_reason(
            model_instance, agent_model_settings
        )
        # Kept so a per-run repetition_penalty can MERGE into this agent's own
        # extra_body (pydantic-ai merges run settings shallowly).
        self._agent_model_settings = agent_model_settings

        # Build output_type from output_schema if provided.
        # Default to str (TextOutputSchema) — NOT None which creates
        # AutoOutputSchema with a final_result tool that confuses models like
        # DeepSeek.
        #
        # _output_type_from_schema handles scalar/array/object roots and
        # per-provider mode wrapping (NativeOutput for inference_router,
        # PromptedOutput for mlx, bare type elsewhere). See the helper for
        # the full provider-mode rationale.
        output_type: Any = _output_type_from_schema(config.output_schema, config.model_provider)
        if config.output_schema:
            logger.info(
                f"Agent {config.name} output_type={type(output_type).__name__} "
                f"for provider={config.model_provider} schema={config.output_schema}"
            )

        # Create the Agent instance using composition instead of inheritance
        # Pass MCP servers to Agent - it will manage them internally
        # Retry budget: pydantic-ai re-prompts on output-validation failure.
        # - mlx: 10 — small models genuinely struggle to coerce JSON.
        # - inference_router: 0 — NativeOutput + vLLM guided-decoding makes
        #   validation failure on the FIRST attempt vanishingly rare. The
        #   historical 2-retry budget was a workaround for output-validation
        #   flake; with pgx-local at max_num_seqs=1 it amplifies a slow but
        #   correct first response into a 3× duplicate-request storm against
        #   the upstream queue (3 × 120s read = 360s ≈ the 5-min retry cycle
        #   seen on trust.pdf 2026-05-19). Transport-level robustness comes
        #   from AsyncHTTPTransport(retries=ir_http_transport_retries).
        # - default: 3.
        # Per-worker override via `output_retries` on the worker spec; see
        # WorkerCreateRequest in src/api/workers/models.py. None preserves the
        # provider default exactly so the trust.pdf-storm invariant on
        # inference_router (=0) is never silently flipped globally.
        # NOTE: `inference_router = 0` is safe to keep ONLY because of the
        # grammar-parity invariant in `_register_default_schema_validator` — the
        # validator that can fail a run is built from the schema the decoder
        # actually saw, so a first-attempt failure means a genuine structural
        # miss, not a constraint the model was never shown. Do not raise this to
        # "work around" a validation 500; fix the parity instead.
        #
        # SECOND reason not to raise it, and the one that actually bit (DGF-007):
        # this budget multiplies WALL CLOCK, not just error count, and it does so
        # INSIDE a single ``agent.run()`` — so the agent's own asyncio.timeout is the
        # only thing bounding it and the caller sees one opaque slow call, not N.
        # A `max_tokens` budget too small for the schema truncates the JSON mid-string;
        # the retry loop does not inspect ``finish_reason``, so a length-truncated
        # completion is indistinguishable from a malformed one, pydantic-ai re-prompts,
        # and the re-prompt truncates at the same budget for the same reason. (It is
        # now at least VISIBLE: ResponseExtractor.check_max_tokens_cap warns and
        # increments ``agent_max_tokens_cap_hit_total`` on every such completion. It
        # does not change the retry behaviour.) Every retry is
        # therefore a guaranteed-identical failure that costs a FULL decode:
        # retries=3 turned one 205s call into 205.0 + 212.3 + 219.8 = 600.188s, which
        # is the whole of the DGF-007 canary's runtime. The fix for a truncating
        # worker is to raise ITS max_tokens (or shrink the schema), never this.
        provider_default_retries = (
            10
            if config.model_provider == "mlx"
            else (0 if config.model_provider == "inference_router" else 3)
        )
        retries = (
            config.output_retries if config.output_retries is not None else provider_default_retries
        )

        # Route both system_prompt and instruction through pydantic-ai's
        # `instructions=` so they end up in a single system message at index 0.
        # Passing system_prompt= and instructions= together produces two
        # consecutive system messages, which strict chat templates (Qwen3.6
        # in particular: "System message must be at the beginning") 400 on.
        # When output_schema is set, pydantic-ai's prompted-output mode joins
        # its schema instruction onto the same string here, keeping the
        # one-system-at-index-0 invariant for every consumer regardless of
        # which DB field they wrote into.
        combined_instructions = (
            "\n\n".join(p for p in (config.system_prompt, config.instruction) if p) or None
        )
        self.agent = Agent(
            model=model_instance,
            tools=config.tools,
            instructions=combined_instructions,
            toolsets=config.mcp_servers,
            model_settings=agent_model_settings,
            output_type=output_type,
            retries=retries,
        )

        retries_source = (
            "per-worker override"
            if config.output_retries is not None
            else (
                "increased for small MLX model"
                if config.model_provider == "mlx"
                else "provider default"
            )
        )
        logger.info(f"Agent {config.name} configured with retries={retries} ({retries_source})")

        # Log model_settings for debugging
        if agent_model_settings:
            logger.info(
                f"Agent {config.name} - model_settings passed to Agent: {agent_model_settings}"
            )

        self.agent_id = agent_id or uuid7()
        self.config = config
        self.name = config.name
        self.db_path = config.db_path
        self._tool_registry = {}
        self.mcp_servers_ready = False
        self._mcp_exit_stack = None

        # ⛔ Request-supplied validator CODE is never executed. This used to
        # ``exec()`` it with full builtins (RCE for anyone who could reach the
        # worker API). The API refuses it with a typed 422; this is the
        # fail-closed backstop for any other path that builds an agent.
        if has_validator_code(config.output_validator_code):
            logger.error(
                "refusing to build agent %s: output_validator code is not supported "
                "and is never executed (%s)",
                config.name,
                OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
            )
            raise OutputValidatorCodeNotSupportedError(
                f"{OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED}: agent {config.name!r} carries "
                "output_validator code, which is never executed"
            )
        if config.output_schema:
            # Create default schema-aware validator with helpful error messages.
            # The second argument is the parity invariant: the validator that can
            # FAIL a run must be built from the same schema the decoder was given,
            # never from the caller's full one.
            self._register_default_schema_validator(
                config.output_schema, _decoder_visible_schema(config.output_schema)
            )

        # Only mention tools/MCP if they're actually present
        init_parts = [f"PydanticAIAgent '{self.name}' (ID: {self.agent_id}) initialized."]
        if config.tools:
            init_parts.append(f"{len(config.tools)} local tools configured.")
        if config.mcp_servers:
            init_parts.append(f"{len(config.mcp_servers)} MCP servers configured.")
        logger.info(" ".join(init_parts))

        # MCP servers will be initialized on first use to avoid fire-and-forget tasks
        # This prevents potential resource leaks from untracked async tasks

    def _count_validator_compile_failure(self, error_type: str) -> None:
        """Count a validator that could not be compiled, never raise.

        Separate from the in-closure counter because these two sites run at
        REGISTRATION, before the closure exists — and they are the only
        validation events that fire once per agent build rather than once per
        model attempt. Same metric, same closed error_type enum. Also used for
        the one per-RUN event outside the closure: an opted-in bound violation
        that outlived the retry budget (``SCHEMA_BOUND_VIOLATION``, #358).
        """
        try:
            get_metrics_client().record_aa_prompt_validation_failure(
                prompt_id=self.name,
                worker_name=self.name,
                error_type=error_type,
            )
        except Exception:  # pragma: no cover - telemetry must never break a run
            logger.debug("validator compile failure not recorded", exc_info=True)

    def _register_default_schema_validator(
        self, output_schema: dict[str, Any], grammar_schema: dict[str, Any] | None = None
    ) -> None:
        """Register default schema-aware validator backed by jsonschema.

        Uses Draft 2020-12 so `oneOf`, `const`, and `additionalProperties: false`
        are actually enforced (the previous hand-rolled implementation only
        checked top-level `type` + `required`, which let discriminated-union
        schemas like fr-agent's silently pass through malformed outputs).

        **Two validators, deliberately.** The *fatal* one is built from
        ``grammar_schema`` — exactly what the decoder was constrained by, via
        ``_decoder_visible_schema`` — so a ``ModelRetry`` can only ever fire on
        something the model was actually told about. The *advisory* one is built
        from the caller's full schema; when it alone objects, the constraint was
        one xgrammar cannot compile, so the output is returned and the violation
        logged as ``schema_soft_violation``.

        **Count/length bounds are OPT-IN retry-then-fail** (``BOUND_KEYWORDS``:
        ``minItems``, ``maxItems``, ``minLength``, ``maxLength``; #358, George
        2026-10-10 "retry then fail"). With ``enforce_schema_bounds`` on for the
        run, a violation raises ``SchemaBoundRetry`` — a ``ModelRetry`` whose
        feedback names each JSON path, keyword and the bound from the ORIGINAL
        schema (e.g. "fix_outline must have at least 1 item"), counted as
        ``schema_bound_retry``. If the worker's output retry budget
        (``output_retries`` / provider default) runs out and the LAST attempt
        still violates a bound, the run fails with the typed
        ``SchemaBoundViolationError`` (HTTP 422 ``schema_bound_violation`` + JSON
        paths, counted as ``schema_bound_violation``) — never a 500, never a
        truncation. Off (the default) they stay advisory exactly as above, but the
        violating paths are recorded in ``usage_metrics.bound_violations`` so the
        rate at which bounds would bite is measured before anyone opts in. Default
        off because RM ``principle_consistency`` relies on WM NOT rejecting an
        oversized reply (it clamps on receipt).

        Why this retry does not reopen the 2026-07-20 outage below: (1) it is
        opt-in per run — no worker that has not asked for it is ever graded
        against a stripped bound; (2) the feedback STATES the bound, so the model
        is no longer graded against a contract it was never shown — the retry
        prompt is where it is shown; (3) it is bounded by the worker's own
        ``output_retries`` (``inference_router`` = 0 ⇒ no retry at all, straight
        to the 422), so it cannot multiply wall clock beyond what the worker
        already budgets; and (4) exhaustion ends in the same typed, deterministic
        422 the opt-in returned before retries existed, which callers (the n8n
        Workers node does not retry 4xx) already handle.

        Building the fatal validator from the FULL schema is what caused the
        2026-07-20 outage: ``retries=0`` on ``inference_router`` turned the first
        unenforceable-constraint disagreement into an instant HTTP 500, and the
        w2v ``demo_video`` / ``retention_plan`` pipeline failed 100% of runs. The
        fix: enforce structure, log the rest. Consumers that truly need a bound
        clamp it on receipt (the established "document + clamp at the consumer"
        pattern).

        The two validators compile in *separate* ``try`` blocks: a problem in the
        advisory schema must never take down the fatal one, because a single
        shared ``try`` used to collapse to "no validator at all" — which is
        precisely the silent state that made the outage unreproducible on demand
        (a probe kept returning 200).

        Args:
            output_schema: the caller's full JSON schema (advisory enforcement).
            grammar_schema: what the decoder actually saw. Defaults to
                ``output_schema`` only so old call sites keep working; every real
                caller passes the value from ``_decoder_visible_schema``.
        """
        import json as json_module

        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import UnknownType
        from pydantic_ai import ModelRetry, RunContext

        # ``effective_grammar`` mirrors the decoder grammar exactly; the caller's
        # full ``output_schema`` is used only for non-fatal advisory logging.
        effective_grammar = output_schema if grammar_schema is None else grammar_schema

        # Compile the FATAL validator in its own try. It must never be taken down
        # by a problem in the advisory schema — that coupling is how a silent "no
        # validator at all" state used to arise.
        try:
            enforced_validator = Draft202012Validator(effective_grammar)
        except Exception:
            logger.exception(
                "Failed to compile the grammar-parity schema validator for agent %s — "
                "NO output validation is registered for this agent. Falling back to "
                "PydanticAI's default validation messages.",
                self.name,
            )
            # ⛔ THE SILENT STATE, AND THE ONE MOST WORTH COUNTING. This branch
            # leaves the agent running with NO output validation at all, and
            # says so only in a log line. The docstring on this method blames
            # exactly that state for making the 2026-07-20 outage
            # unreproducible. It is counted before the return, not after — there
            # is no after.
            self._count_validator_compile_failure("validator_compile_failed")
            return

        # The advisory validator is best-effort by construction: if it cannot be
        # compiled we simply lose the soft-violation signal, not enforcement.
        advisory_validator: Any = None
        try:
            advisory_validator = Draft202012Validator(output_schema)
        except Exception:
            logger.exception(
                "Failed to compile the advisory schema validator for agent %s; "
                "soft-violation reporting is disabled (enforcement is unaffected).",
                self.name,
            )
            self._count_validator_compile_failure("advisory_validator_compile_failed")

        # Announce the constraints the decoder was never given, once at
        # registration — a greppable inventory of what is now advisory-only.
        unenforced = stripped_keywords(output_schema)
        # Surfaced per run in usage_metrics (`grammar_stripped_keywords`), so a
        # reader can see "bounds enforced post-hoc only" without the logs.
        self._grammar_stripped_keywords = sorted(unenforced) or None
        advisory_only = sorted(unenforced - BOUND_KEYWORDS)
        post_hoc = sorted(unenforced & BOUND_KEYWORDS)
        if advisory_only:
            logger.warning(
                "schema_constraints_unenforced: agent %s declares %s, which vLLM guided "
                "decoding cannot compile into a grammar. These are NOT enforced — they are "
                "advisory only. State the constraint in the property `description` (the model "
                "reads that) and enforce it at the consumer.",
                self.name,
                advisory_only,
            )
        if post_hoc:
            logger.info(
                "schema_bounds_post_hoc: agent %s declares %s; stripped from the decoding "
                "grammar, checked AFTER decoding against the original schema. Advisory "
                "(paths in usage_metrics.bound_violations) unless the run opts in with "
                "enforce_schema_bounds, which fails it with %s.",
                self.name,
                post_hoc,
                SCHEMA_BOUND_VIOLATION,
            )

        def format_feedback(output: Any, errors: list[Any]) -> str:
            """Build a model-readable retry message from jsonschema errors."""
            lines = ["Output failed schema validation. Issues:"]
            for err in errors[:5]:
                loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
                lines.append(f"  - at {loc}: {err.message}")
            lines.append(
                # Show the enforced (decoder-visible) schema — never ask the
                # model to satisfy a bound the grammar cannot express.
                "Allowed schema (must match exactly one branch of any "
                "oneOf): " + json_module.dumps(effective_grammar, separators=(",", ":"))
            )
            lines.append("Your output was: " + json_module.dumps(output, default=str)[:500])
            return "\n".join(lines)

        def _count_validation_failure(error_type: str) -> None:
            """Increment aa_prompt_validation_failures_total, never raise.

            ⛔ WHY A COUNTER AND NOT A ``status=`` VALUE. #192 expressed a new
            failure mode as a new value on ``aa_prompt_execution_total``'s
            ``status`` dimension, and for a lost answer that was right. It
            cannot work here. ``retries`` is handed to the pydantic-ai ``Agent``
            constructor and the ENTIRE retry loop runs inside one
            ``await agent.run()`` — so a validation failure that a later retry
            recovers from produces ``status="success"`` and nothing else. On the
            ``mlx`` provider (retries=10) up to ten failures vanish behind one
            success. ``status`` is recorded once per RUN; this fires once per
            ATTEMPT, which is the only place the retry-absorbed failures exist.

            That is also why this recorder was worth wiring rather than
            deleting: it is the only instrument in the service that can see
            them. It sat dead with zero call sites in ``src/`` and zero series
            in Prometheus, while the two sites below carried the signal as
            ``logger.warning`` — one of which its own comment already called
            "a greppable dead-signal metric".
            """
            try:
                get_metrics_client().record_aa_prompt_validation_failure(
                    prompt_id=self.name,
                    worker_name=self.name,
                    error_type=error_type,
                )
            except Exception:  # pragma: no cover - telemetry must never break a run
                logger.debug("validation failure not recorded", exc_info=True)

        def validate(ctx: RunContext, output: Any) -> Any:  # noqa: ARG001
            _clear_recorded_bound_violations()
            try:
                errors = list(enforced_validator.iter_errors(output))
            except UnknownType:
                # Schema references a type the validator doesn't know
                # (e.g. {"type": "unknown"}). Treat as no-op rather than
                # blocking the agent on a malformed schema.
                return output
            if errors:
                feedback = format_feedback(output, errors)
                logger.warning(
                    f"Schema validation failed for agent {self.name}: "
                    f"{errors[0].message} (at "
                    f"{'/'.join(str(p) for p in errors[0].absolute_path) or '<root>'})"
                )
                _count_validation_failure("schema_invalid")
                raise ModelRetry(feedback)
            # Structurally valid. Any remaining full-schema error is a
            # grammar-stripped constraint the decoder was never given —
            # log it (a greppable dead-signal metric) but do NOT raise.
            if advisory_validator is None:
                return output
            try:
                relaxed = list(advisory_validator.iter_errors(output))
            except UnknownType:
                return output
            if relaxed:
                # For oneOf/anyOf the top-level error collapses to a useless
                # "not valid under any of the given schemas". Drill into the
                # branch the output most nearly satisfied (fewest sub-errors —
                # the intended branch of a discriminated union) so the metric
                # names the REAL constraint (e.g. `minItems`/`maximum`), not a
                # sibling branch's discriminator mismatch.
                def _leaves(e: Any) -> list[Any]:
                    if not e.context:
                        return [e]
                    by_branch: dict[Any, list[Any]] = {}
                    for sub in e.context:
                        idx = sub.schema_path[0] if sub.schema_path else -1
                        by_branch.setdefault(idx, []).append(sub)
                    return min(by_branch.values(), key=len)

                # ⛔ COUNT/LENGTH BOUNDS, AGAINST THE ORIGINAL SCHEMA.
                # They are stripped from the grammar (xgrammar), so this check —
                # run on the caller's UNSTRIPPED schema — is their only
                # enforcement. Left advisory, a judge's `maxItems` did nothing at
                # all. Opted in (#358): a ModelRetry whose feedback states the
                # bound — the grammar cannot carry it, but the prompt can — and,
                # once the retry budget is spent, the typed 422 (mapped in
                # `_run_agent_with_timeout`). Never truncated or repaired.
                bound_leaves = [
                    lf
                    for e in relaxed
                    for lf in _leaves(e)
                    if getattr(lf, "validator", None) in BOUND_KEYWORDS
                ]
                if bound_leaves:
                    _raise_or_record_bound_violations(
                        self.name, bound_leaves, _count_validation_failure
                    )

                detail = "; ".join(
                    f"{'/'.join(str(p) for p in lf.absolute_path) or '<root>'}: {lf.message}"
                    for e in relaxed[:5]
                    for lf in _leaves(e)[:3]
                )
                logger.warning(
                    f"schema_soft_violation: agent {self.name} output satisfies the "
                    f"decoder-enforced schema but violates a grammar-stripped constraint "
                    f"(non-fatal, the decoder was never given it): {detail}"
                )
                _count_validation_failure("schema_soft_violation")
            return output

        self.agent.output_validator(validate)
        logger.info(
            f"Registered default jsonschema validator for agent {self.name} "
            f"(schema type: {output_schema.get('type', 'unknown')})"
        )

    def _get_model_settings(
        self, config: PydanticAIAgentConfig, model_instance: models.Model
    ) -> ModelSettings | None:
        """Delegate to ModelFactory; kept as a method for the test patch target."""
        return ModelFactory.get_model_settings(config, model_instance)

    async def _initialize_mcp_servers(self) -> None:
        """Initialize MCP servers with proper lifecycle management.

        Raises:
            asyncio.CancelledError: If initialization is cancelled
            Exception: If MCP server initialization fails
        """
        if not self.config.mcp_servers:
            self.mcp_servers_ready = True
            return

        try:
            logger.info(
                f"Initializing {len(self.config.mcp_servers)} MCP servers for agent {self.name}"
            )

            # Create exit stack for managing MCP server lifecycle
            self._mcp_exit_stack = AsyncExitStack()

            # Start all MCP servers
            await self._start_mcp_servers()

            self.mcp_servers_ready = True
            logger.info(
                "Successfully initialized %d MCP servers for agent %s",
                len(self.config.mcp_servers),
                self.name,
            )

        except asyncio.CancelledError:
            logger.warning("MCP initialization cancelled for agent %s", self.name)
            await self._cleanup_mcp_on_error()
            raise
        except Exception:
            logger.exception("Failed to initialize MCP servers for agent %s", self.name)
            await self._cleanup_mcp_on_error()

    async def _start_mcp_servers(self) -> None:
        """Start all MCP servers in the configuration.

        Raises:
            asyncio.CancelledError: If startup is cancelled
            Exception: If any MCP server fails to start
        """
        for i, server in enumerate(self.config.mcp_servers):
            try:
                logger.info(f"Starting MCP server {i + 1}/{len(self.config.mcp_servers)}: {server}")
                # Enter the server context and wait for it to be ready
                if self._mcp_exit_stack is not None:
                    await self._mcp_exit_stack.enter_async_context(server)

                # Give the server a moment to fully initialize
                await asyncio.sleep(0.5)
                logger.info(f"Successfully started MCP server: {server}")
            except asyncio.CancelledError:
                logger.warning(f"MCP server startup cancelled for {server}")
                # Properly propagate cancellation
                raise
            except Exception as e:
                logger.error(f"Failed to start MCP server {server}: {e}", exc_info=True)
                # Continue with other servers but track the failure
                continue

    async def _cleanup_mcp_on_error(self) -> None:
        """Clean up MCP resources when initialization fails.

        Raises:
            asyncio.CancelledError: If cleanup is cancelled
        """
        self.mcp_servers_ready = False
        if self._mcp_exit_stack:
            try:
                # Close the exit stack in the same task context
                await self._mcp_exit_stack.aclose()
            except asyncio.CancelledError:
                logger.warning("MCP cleanup cancelled")
                raise
            except Exception as cleanup_error:
                logger.error("Error during MCP cleanup: %s", cleanup_error, exc_info=True)
            finally:
                self._mcp_exit_stack = None

    def _initialize_model(self, config: PydanticAIAgentConfig) -> models.Model:
        """Delegate to ModelFactory.initialize_model."""
        return ModelFactory.initialize_model(config)

    def _get_api_key(self, config: PydanticAIAgentConfig, env_var: str) -> str | None:
        """Delegate to ModelFactory.get_api_key (kept for direct-call tests)."""
        return ModelFactory.get_api_key(config, env_var)

    def _create_inference_router_model(self, config: PydanticAIAgentConfig) -> OpenAIChatModel:
        """Delegate to ModelFactory (kept for direct-call tests)."""
        return ModelFactory._create_inference_router_model(config)

    async def run_with_context(self, enhanced_context: dict[str, Any]) -> WorkerAgentRunResult:
        """Run the agent with enhanced graph execution context.

        Args:
            enhanced_context: Dictionary containing task, graph context, and node config

        Returns:
            WorkerAgentRunResult containing output and usage metrics

        Raises:
            Exception: If agent execution fails
        """
        try:
            # Extract the task from the enhanced context
            task = enhanced_context.get("task", "")
            graph_context = enhanced_context.get("graph_context", {})
            node_config = enhanced_context.get("node_config", {})

            agent_name = self.name
            agent_id_str = str(self.agent_id)

            logger.info(
                f"Running agent {agent_name} (ID: {agent_id_str}) with enhanced context. "
                f"Task: {task}, Graph ID: {graph_context.get('graph_id')}, "
                f"Node ID: {graph_context.get('node_id')}"
            )

            # Build an enhanced prompt that includes graph context
            enhanced_prompt = self._build_context_aware_prompt(task, graph_context, node_config)

            # Use the standard run method with the enhanced prompt
            return await self.run(enhanced_prompt)

        except Exception:
            agent_name = self.name
            logger.exception(f"Error in PydanticAIAgent.run_with_context for agent {agent_name}")
            raise

    def _build_context_aware_prompt(
        self, task: str, graph_context: dict[str, Any], node_config: dict[str, Any]
    ) -> str:
        """Delegate to PromptBuilder.build_context_aware_prompt."""
        return PromptBuilder.build_context_aware_prompt(task, graph_context, node_config)

    def _build_graph_context_info(self, graph_context: dict[str, Any]) -> list[str]:
        """Delegate to PromptBuilder.build_graph_context_info."""
        return PromptBuilder.build_graph_context_info(graph_context)

    def _add_previous_outputs_context(
        self, graph_context: dict[str, Any], context_info: list[str]
    ) -> None:
        """Delegate to PromptBuilder (kept for direct-call tests)."""
        PromptBuilder._add_previous_outputs_context(graph_context, context_info)

    def _add_variables_context(
        self, graph_context: dict[str, Any], context_info: list[str]
    ) -> None:
        """Delegate to PromptBuilder (kept for direct-call tests)."""
        PromptBuilder._add_variables_context(graph_context, context_info)

    def _is_vacuous_against_schema(self, output: str | None) -> bool:
        """True when the schema DECLARED required fields and the answer has none.

        ⛔ THIS IS NOT "the output is empty". That distinction is the whole
        point, and getting it wrong manufactures a defect rate out of correct
        answers. `{}` is schema-VALID for ks-relationship-extractor,
        ks-entity-spotter and principle_consistency, which declare no
        `required` — for them `{}` is a correct "found nothing" and MUST stay
        `success`. Measured 2026-09-02: 253 of 375 empty bodies over 24h were
        that shape, finish_reason "stop", validation_passed true.

        What this catches is the other shape: a schema that DOES declare
        required fields, answered with an object containing none of them. That
        is a well-formed WRONG answer, and until now it was booked as success —
        the same class of defect as a conveyor status of 'unknown' passing every
        assert.

        Every branch below fails CLOSED to False (i.e. "not vacuous"), because
        the cost of a false positive here is inventing a failure that did not
        happen:

        * no schema, or no `required`  -> not our business; that is the
          legitimate "found nothing" case above.
        * output empty or None         -> that is `lost_answer`'s territory, and
          double-booking one event under two statuses would corrupt both.
        * output is not JSON           -> a parse failure is a DIFFERENT defect
          with its own signal; claiming it here would hide it.
        * output is a JSON scalar/list -> `required` only has meaning for an
          object, so the question does not apply.

        ⚠️ A PARTIAL answer (some required keys present, others missing) is
        deliberately NOT vacuous. That is a validation failure, which this
        estate already counts on aa_prompt_validation_failures_total, and which
        can happen several times inside ONE run. Only a total absence is
        unambiguous enough to book as a per-run status.
        """
        schema = getattr(self.config, "output_schema", None)
        if not isinstance(schema, dict):
            return False
        required = schema.get("required")
        if not required:
            return False
        if not output:
            return False
        try:
            parsed = json.loads(output)
        except ValueError, TypeError:
            return False
        if not isinstance(parsed, dict):
            return False
        # "Present" means present AND carrying something. A required key set to
        # null or "" is absent in every sense the caller cares about.
        return not any(parsed.get(key) not in (None, "", [], {}) for key in required)

    def _extract_text_output(
        self,
        response: FrameworkAgentRunResult[Any],
    ) -> Any:
        """Delegate to ResponseExtractor.extract_text_output (passes agent name)."""
        return ResponseExtractor.extract_text_output(response, self.name)

    def _extract_token_usage(
        self,
        agent_name: str,
        run_result: FrameworkAgentRunResult[Any] | None,
    ) -> TokenUsageMetrics | None:
        """Delegate to ResponseExtractor.extract_token_usage (passes the wrapped agent).

        Also passes the resolved max_tokens for this agent so the extractor can
        flag a truncated (cap-hit) completion.
        """
        return ResponseExtractor.extract_token_usage(
            agent_name,
            run_result,
            self.agent,
            max_tokens_cap=getattr(self, "_max_tokens_cap", None),
            model_provider=getattr(self.config, "model_provider", None),
        )

    @staticmethod
    def _attach_served_routing(
        token_metrics: TokenUsageMetrics | None,
        run_result: FrameworkAgentRunResult[Any],
    ) -> tuple[str | None, str | None]:
        """Stamp inference-router's served_model / served_provider onto the metrics.

        Returns the pair as well, so a caller that also needs it (the BYO cost
        event) reuses this one traversal instead of repeating it.

        Deliberately total: ``extract_served_routing`` already swallows any
        traversal error and answers ``(None, None)``, so a provider that does
        not report these — every non-IR provider, and an IR predating the
        companion PR — yields None rather than raising. Nothing about a run's
        success may depend on a routing annotation being present.
        """
        served_model, served_provider = ResponseExtractor.extract_served_routing(run_result)
        if token_metrics is not None:
            token_metrics.served_model = served_model
            token_metrics.served_provider = served_provider
            token_metrics.served_model_root = ResponseExtractor.extract_served_fields(
                run_result
            ).get("served_model_root")
        return served_model, served_provider

    def _attach_sampling_provenance(
        self,
        token_metrics: TokenUsageMetrics | None,
        *,
        run_sampling: RunSampling | None = None,
        run_result: FrameworkAgentRunResult[Any] | None = None,
    ) -> TokenUsageMetrics | None:
        """Stamp what was SENT (model / sampling / budget / correlation) and how it ENDED.

        The values were captured when the agent was built (``_sampling_provenance``)
        from the exact model_settings pydantic-ai sends. When token usage could not
        be extracted (``token_metrics is None``) a metrics object is still created
        so a completed call always records what it asked for; its token fields stay
        None, which is the truthful reading. An agent built without provenance
        (never initialised) leaves the metrics untouched.

        ``run_sampling`` holds the per-run overrides passed to
        ``agent.run(model_settings=...)``; pydantic-ai merges run settings OVER the
        agent's, so each one present is the value that was sent and wins over the
        build-time value. It is read back from the SAME ``_run_model_settings``
        output the run used, so it cannot drift from the wire.

        ``correlation_id`` is the ContextVar the outbound httpx hook reads to set
        ``X-Correlation-Id`` on the IR request, so the stamped value and the header
        are the same read — persisted only when it is a valid UUID, else None. ``finish_reason`` is pydantic-ai's normalised
        provider-reported reason (``length`` ⇒ truncated by max_tokens).
        """
        provenance = getattr(self, "_sampling_provenance", None)
        if provenance is None:
            return token_metrics
        if token_metrics is None:
            token_metrics = TokenUsageMetrics()
        stripped = getattr(self, "_grammar_stripped_keywords", None)
        if stripped:
            token_metrics.grammar_stripped_keywords = list(stripped)
            if set(stripped) & BOUND_KEYWORDS:
                token_metrics.schema_bounds_enforced = _ENFORCE_SCHEMA_BOUNDS.get()
                token_metrics.bound_violations = list(_BOUND_VIOLATIONS.get() or [])
        effective: dict[str, Any] = dict(provenance)
        run_settings = self._run_model_settings(run_sampling)
        if run_settings is not None:
            for key in ("seed", "frequency_penalty", "temperature"):
                if key in run_settings:
                    effective[key] = run_settings[key]
            run_rp = _repetition_penalty_of(run_settings)
            if run_rp is not None:
                effective["repetition_penalty"] = run_rp
        token_metrics.requested_model = effective.get("requested_model")
        token_metrics.temperature = effective.get("temperature")
        unsent = getattr(self, "_temperature_unsent_reason", None)
        if unsent is not None and token_metrics.temperature is not None:
            # pydantic-ai strips the key before the request (see
            # _temperature_unsent_reason): report what the wire carried — nothing.
            token_metrics.temperature = None
            token_metrics.temperature_not_sent = unsent
        token_metrics.seed = effective.get("seed")
        token_metrics.max_tokens = effective.get("max_tokens")
        token_metrics.repetition_penalty = effective.get("repetition_penalty")
        token_metrics.frequency_penalty = effective.get("frequency_penalty")
        token_metrics.finish_reason = ResponseExtractor.finish_reason(run_result)
        from src.api.identity_context import get_correlation_id_from_context

        token_metrics.correlation_id = _uuid_or_none(get_correlation_id_from_context())
        return token_metrics

    def _stamp_sent_usage_metrics(self, error: BaseException, run_sampling: RunSampling) -> None:
        """Carry what was SENT on a run that FAILED, on the exception itself.

        A failed call (an IR 503, a timeout) returns no ``WorkerAgentRunResult``,
        so the sampling read-back that a completed call puts in usage_metrics was
        lost with it: a judge that sent temperature 0 / seed 0 and got a 503 could
        only record null for both. The values come from the SAME
        ``_attach_sampling_provenance`` read of the settings pydantic-ai was handed
        (never the worker config), with token fields None — nothing was measured.
        The HTTP layer renders it as ``error.details.usage_metrics``. Best effort:
        an exception type that refuses attributes simply carries nothing.
        """
        metrics = self._attach_sampling_provenance(None, run_sampling=run_sampling)
        if metrics is None:
            return
        with suppress(AttributeError, TypeError):
            setattr(error, SENT_USAGE_METRICS_ATTR, metrics.model_dump())

    async def _ensure_mcp_servers_ready(self) -> None:
        """Ensure MCP servers are ready for use.

        Raises:
            Exception: If MCP server initialization fails
        """
        if self.config.mcp_servers and not self.mcp_servers_ready:
            logger.info("MCP servers not ready, initializing...")
            try:
                await self._initialize_mcp_servers()

                # Wait a bit for servers to fully initialize
                if self.mcp_servers_ready:
                    await asyncio.sleep(1)  # Give servers time to stabilize
                    logger.info(f"MCP servers for agent {self.name} are now ready")
                else:
                    logger.warning("MCP servers failed to initialize, proceeding without them")
            except asyncio.CancelledError:
                logger.warning("MCP server initialization cancelled")
                self.mcp_servers_ready = False
                raise
            except Exception as e:
                logger.error(f"Failed to ensure MCP servers ready: {e}", exc_info=True)
                self.mcp_servers_ready = False
                # Don't raise - allow agent to proceed without MCP servers

    @staticmethod
    def _to_pydantic_ai_parts(prompt: str | list[Any]) -> str | list[Any]:
        """Convert API content blocks into pydantic-ai input parts.

        - String prompt: passed through unchanged (legacy text-only path).
        - List of TextBlock/ImageBlock: converted to a list of plain
          strings (text) and ``pydantic_ai.BinaryContent`` (images).
          ImageBlock entries arrive base64-encoded; we decode the bytes
          and hand them to BinaryContent which the underlying provider
          (Claude / GPT-4o / Gemini / Gemma-4) consumes via standard
          OpenAI-compat content-block messages.
        """
        if isinstance(prompt, str):
            return prompt

        import base64

        from pydantic_ai import BinaryContent

        parts: list[Any] = []
        for block in prompt:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type == "text":
                text = getattr(block, "text", None) or block.get("text", "")
                if text:
                    parts.append(text)
            elif block_type == "image":
                data = getattr(block, "data", None) or block.get("data", "")
                media_type = getattr(block, "media_type", None) or block.get(
                    "media_type", "image/jpeg"
                )
                parts.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=media_type,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown content block type {block_type!r}; expected 'text' or 'image'."
                )
        return parts

    def _agent_run_timeout(self) -> float:
        """Wall-clock budget for a single ``agent.run()``.

        Returns ``agent_run_timeout_mcp_s`` for MCP-enabled agents (extra headroom
        for tool round-trips), else ``agent_run_timeout_s``. Both are Settings
        fields (env: AGENT_RUN_TIMEOUT_MCP_S / AGENT_RUN_TIMEOUT_S), defaults
        900.0 / 600.0.

        INVARIANT — the returned budget is always >= the downstream inference-router
        read budget (``ir_http_read_timeout_s``), enforced at startup by
        ``Settings._validate_timeout_ordering``. Two things depend on it:

        1. a slow-but-working vLLM call is never cut off WM-side before the
           router/pgx can finish — the orchestrator LLM ladder (660s) sits above;
        2. more subtly, this cap's ``asyncio.timeout`` starts before connect, so
           while it outlives the read budget it always fires FIRST and the OpenAI
           SDK's retry loop never gets a turn. Invert the ordering and each read
           timeout becomes several repeat decodes against a saturated provider
           (``openai/_base_client.py:1653`` retries ``httpx.TimeoutException``).
           The clients also pin ``max_retries=0``, which is the durable fix; the
           invariant is the backstop.

        Sourced here so the main run path AND the MCP-error-recovery retry use the
        SAME value (the retry used to hardcode a stray 240s — an inverted hop well
        below the downstream budget that could cut off a legitimately-slow recovery).
        """
        settings = _get_settings()
        if self.config.mcp_servers:
            return settings.agent_run_timeout_mcp_s
        return settings.agent_run_timeout_s

    def _run_model_settings(self, run_sampling: RunSampling | None) -> ModelSettings | None:
        """The per-run ``model_settings`` for ``agent.run``, or None when there are none.

        Per-message sampling values travel HERE, not in the cached agent's
        settings: baking them in would need one cache slot (and one never-evicted
        connection pool) per distinct value. pydantic-ai merges run settings over
        the agent's SHALLOWLY, so a per-run ``repetition_penalty`` is merged into
        the agent's own ``extra_body`` (reasoning keys) rather than replacing it,
        and is only ever sent to self-hosted providers — the same rule as the
        build-time path in ``ModelFactory.get_model_settings``.
        """
        if run_sampling is None or run_sampling.is_empty():
            return None
        settings = ModelSettings()
        if run_sampling.seed is not None:
            settings["seed"] = run_sampling.seed
        if run_sampling.temperature is not None:
            settings["temperature"] = run_sampling.temperature
            unsent = getattr(self, "_temperature_unsent_reason", None)
            if unsent is not None:
                _log_temperature_unsent_once(self.config.model_provider, self.config.model, unsent)
        if run_sampling.frequency_penalty is not None:
            settings["frequency_penalty"] = run_sampling.frequency_penalty
        if run_sampling.repetition_penalty is not None:
            if self.config.model_provider in SELF_HOSTED_PROVIDERS:
                agent_settings = getattr(self, "_agent_model_settings", None) or {}
                settings["extra_body"] = _with_repetition_penalty(
                    agent_settings.get("extra_body"), run_sampling.repetition_penalty
                )
            else:
                logger.warning(
                    "Agent %s - per-run repetition_penalty=%s NOT sent: provider %s is "
                    "not self-hosted and would reject the vLLM-only key",
                    self.name,
                    run_sampling.repetition_penalty,
                    self.config.model_provider,
                )
        return settings or None

    def _run_kwargs(self, run_sampling: RunSampling | None) -> dict[str, Any]:
        """Per-run ``agent.run`` kwargs.

        Empty when there are no per-run overrides, so the call is exactly
        ``agent.run(input)`` as before.
        """
        settings = self._run_model_settings(run_sampling)
        return {} if settings is None else {"model_settings": settings}

    async def _run_agent_with_timeout(
        self,
        prompt: str | list[Any],
        agent_name: str,
        *,
        run_sampling: RunSampling | None = None,
    ) -> Any:
        """Run the agent with appropriate timeout based on MCP server status.

        Args:
            prompt: The prompt to send to the agent. Either a plain string
                (legacy) or a list of TextBlock/ImageBlock content items
                (multimodal — only supported by vision-capable models).
            agent_name: Name of the agent for logging

        Returns:
            Agent response object

        Raises:
            asyncio.TimeoutError: If agent execution times out
        """
        if self.config.mcp_servers:
            logger.info(
                "Agent %s - Running with %d MCP servers",
                agent_name,
                len(self.config.mcp_servers),
            )
        timeout = self._agent_run_timeout()

        run_input = self._to_pydantic_ai_parts(prompt)

        # Use asyncio.timeout() instead of wait_for() to avoid creating a new task
        # This prevents cancel scope violations with MCP servers that use ANYIO
        async with asyncio.timeout(timeout):
            try:
                return await self.agent.run(run_input, **self._run_kwargs(run_sampling))
            except UnexpectedModelBehavior as e:
                self._raise_if_bound_exhausted(e)
                raise

    def _raise_if_bound_exhausted(self, error: UnexpectedModelBehavior) -> None:
        """#358: map an output-retry budget spent ON a bound violation to the typed error.

        An opted-in bound violation is retried (SchemaBoundRetry); when the budget
        runs out on one, fail with the same typed error (→ HTTP 422
        schema_bound_violation + paths) the API returned before retries existed —
        never a generic 500. Returns (does nothing) for any other exhausted-retry
        failure, which the caller re-raises unchanged. Shared by the main run path
        and the MCP-recovery re-run so both surface the same 422.
        """
        bound = _bound_retry_cause(error)
        if bound is None:
            return
        bound_error = SchemaBoundViolationError(bound.agent_name, bound.violations)
        logger.warning("%s (after the output retry budget was spent)", bound_error)
        self._count_validator_compile_failure(SCHEMA_BOUND_VIOLATION)
        raise bound_error from error

    def _record_cancelled_run(
        self, metrics_client: Any, agent_name: str, duration_seconds: float
    ) -> None:
        """Book a cancelled run on both counters.

        Lifted out of ``run()``'s ``except asyncio.CancelledError`` purely to
        keep that method under the statement cap — the reasoning for each field
        lives at the call site.
        """
        logger.warning("Agent %s - run cancelled after %.4fs", agent_name, duration_seconds)
        metrics_client.record_execution(
            agent_name=agent_name,
            agent_type="pydantic_ai",
            model_provider=self.config.model_provider,
            model_name=self.config.model,
            duration_seconds=duration_seconds,
            success=False,
            # A cancellation is NOT an upstream fault. Classifying it keeps
            # executions_failed sliceable, so a dashboard can subtract deliberate
            # cancellations instead of reading them as agent defects. Omitting
            # the call altogether would be the worse distortion: the run would
            # vanish from the total AND the failed series while still having
            # recorded a start.
            error_classification="cancelled",
        )
        metrics_client.record_aa_prompt_execution(
            prompt_id=agent_name,
            worker_name=agent_name,
            status="cancelled",
            latency_seconds=duration_seconds,
            model=self.config.model,
            error_classification="cancelled",
        )

    async def _recover_and_book_outcome(
        self,
        *,
        metrics_client: Any,
        error: Exception,
        prompt: str | list[Any],
        agent_name: str,
        start_time: float,
        error_classification: str | None,
        run_sampling: RunSampling | None = None,
    ) -> WorkerAgentRunResult | None:
        """Attempt MCP recovery, then book the outcome the recovery ACTUALLY had.

        ⛔ THE ORDER IS THE FIX. ``status="failed"`` used to be emitted in
        ``run()`` ABOVE the recovery call, and a run that
        ``_handle_mcp_error_recovery`` then RESCUED returned its answer with no
        compensating record. Every MCP-recovered run was therefore booked 100%
        failed, and the recovery path had a 0% success rate by construction
        rather than by measurement.

        ``finally`` rather than a tail call, so exactly one record is emitted on
        EVERY exit: recovery returning a result, recovery returning None, and
        recovery itself raising (it re-raises TimeoutError). One
        aa_prompt_execution per run on all three — no double count, and no path
        that books nothing.

        ⚠️ Deliberately scoped to the metric that HAS a status dimension.
        ``record_execution``'s ``success=False`` is left untouched in ``run()``:
        that counter means "the run raised", which it did, and flipping it would
        redefine agent_executions_failed for every dashboard built on it — the
        same argument this file already makes on the success path.
        """
        recovery_status = "failed"
        result: WorkerAgentRunResult | None = None
        try:
            result = await self._handle_mcp_error_recovery(
                error, prompt, agent_name, run_sampling=run_sampling
            )
            if result:
                recovery_status = "recovered"
        finally:
            metrics_client.record_aa_prompt_execution(
                prompt_id=agent_name,
                worker_name=agent_name,
                status=recovery_status,
                # Recomputed, not the pre-recovery elapsed: the caller waited
                # through the retry too, so on a recovered run the original
                # duration is not the latency anyone experienced.
                latency_seconds=asyncio.get_running_loop().time() - start_time,
                model=self.config.model,
                error_classification=error_classification,
            )
        return result

    async def run(
        self,
        prompt: str | list[Any],
        *,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        enforce_schema_bounds: bool = False,
    ) -> WorkerAgentRunResult:
        """Run the agent; see ``_run_impl``.

        ``enforce_schema_bounds`` (per run, default OFF) makes a count/length
        bound violation (minItems/maxItems/minLength/maxLength, which the grammar
        cannot carry) re-ask the model with feedback naming the bound, within the
        worker's output retry budget, and fail the run with
        ``SchemaBoundViolationError`` if the budget runs out on one, instead of
        being advisory. Either way the violating paths land in
        ``usage_metrics.bound_violations``. Scoped with ContextVars and reset on
        every exit, so the flag can never leak into another run.
        """
        enforce_token = _ENFORCE_SCHEMA_BOUNDS.set(bool(enforce_schema_bounds))
        violations_token = _BOUND_VIOLATIONS.set([])
        try:
            return await self._run_impl(
                prompt,
                seed=seed,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
            )
        finally:
            _BOUND_VIOLATIONS.reset(violations_token)
            _ENFORCE_SCHEMA_BOUNDS.reset(enforce_token)

    async def _run_impl(
        self,
        prompt: str | list[Any],
        *,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
    ) -> WorkerAgentRunResult:
        """Run the agent with proper MCP server lifecycle management.

        Args:
            prompt: The prompt to send to the agent
            seed: Per-run sampling seed (a per-message override). Sent via
                ``agent.run(model_settings=...)`` so the cached agent is reused;
                None ⇒ the agent's own settings decide (its config seed, if any).
            repetition_penalty: Per-run vLLM repetition penalty, same rules as
                ``seed`` (self-hosted providers only).
            frequency_penalty: Per-run OpenAI ``frequency_penalty``, same rules.
            temperature: Per-run sampling temperature, same rules; wins over the
                agent's build-time temperature for THIS run only.

        Returns:
            WorkerAgentRunResult containing output and usage metrics

        Raises:
            TimeoutError: If agent execution times out
            ValueError: If agent returns None response
            Exception: If agent execution fails
        """
        agent_name = self.name
        agent_id_str = str(self.agent_id)
        logger.info(
            "Running agent %s (ID: %s) with prompt: %s",
            agent_name,
            agent_id_str,
            _summarize_for_log(prompt),
        )

        # Per-run (per-message) sampling overrides; see RunSampling.
        run_sampling = RunSampling(
            seed=seed,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
        )

        # Get metrics client for telemetry
        metrics_client = get_metrics_client()
        start_time = asyncio.get_running_loop().time()

        # Record agent start
        metrics_client.record_agent_start(agent_name, "pydantic_ai")

        # Take a reference on the shared HTTP transport for the whole run. This
        # statement MUST stay immediately above `try:` with nothing between them
        # that can raise, because the matching decrement lives in that `try`'s
        # `finally` — a raise in the gap would leak a reference and pin the pool
        # open forever. See `cleanup()` for why the refcount exists.
        self._active_runs += 1

        try:
            # NOTE: Don't manually initialize MCP servers - pydantic-ai Agent handles this
            # when mcp_servers are passed to Agent constructor. Manual initialization
            # causes task/cancel scope conflicts.

            # Run the agent with appropriate timeout
            response = await self._run_agent_with_timeout(
                prompt, agent_name, run_sampling=run_sampling
            )

            if response is None:
                logger.error(
                    "Agent %s - Response from agent.run() was None, which is unexpected.",
                    agent_name,
                )
                raise ValueError(
                    f"Agent '{agent_name}' received an unexpected None response "
                    "from the core run method."
                )

            logger.info(
                "Agent %s - Successfully completed run with response from agent.run()",
                agent_name,
            )

            # Extract token usage metrics
            token_metrics = self._extract_token_usage(agent_name, response)

            # ⛔ UNCONDITIONAL, AND THAT IS THE FIX. This extraction used to live
            # inside the BYO-cost branch below, guarded by
            # `cost_source == BYO_COST_SOURCE`. Every other run — which is all of
            # them, in practice — therefore answered the caller with
            # served_model=null even though inference-router had told us, in the
            # response body, exactly which model served the request. The value
            # was measured, carried through provider_details, and then discarded
            # because a BILLING condition happened to be false.
            served_model, served_provider = self._attach_served_routing(token_metrics, response)

            # Calculate execution duration
            duration_seconds = asyncio.get_running_loop().time() - start_time

            # Record metrics to Grafana
            token_usage_dict: dict[str, int] | None = None
            if token_metrics:
                token_usage_dict = {
                    "prompt_tokens": token_metrics.prompt_tokens or 0,
                    "completion_tokens": token_metrics.completion_tokens or 0,
                    "total_tokens": token_metrics.total_tokens or 0,
                }

            # ⛔ EXTRACTION RUNS BEFORE THE METRICS, AND THE ORDER IS THE FIX.
            # It used to run after, so a run that BILLED completion tokens and
            # then produced no output at all was recorded status="success" —
            # the metric said the agent worked while the caller got "".
            # Measured on the 2026-08-24 inference-router empties: 95 rows with
            # finish_reason "length", content_len 0, completion_tokens 12000 on
            # 59 of them, and latency up to 1,050,715 ms — 17.5 minutes of GPU
            # for one answer that arrived empty, every one of them counted as a
            # success here.
            #
            # Moving it above the recorders is safe BECAUSE the enclosing
            # try/except already books a failure: if _extract_text_output
            # raises, the `except Exception` branch below records
            # success=False / status="failed". Before this change an exception
            # there left a "success" already recorded and then propagated.
            output = self._extract_text_output(response)
            lost_answer = bool(output is None and token_metrics and token_metrics.completion_tokens)
            vacuous_answer = self._is_vacuous_against_schema(output)
            if lost_answer:
                logger.error(
                    "Agent %s - output is None despite %d completion tokens. "
                    "Returning empty string to prevent None propagation.",
                    agent_name,
                    token_metrics.completion_tokens if token_metrics else 0,
                )
                output = ""
            elif output is None:
                # ⛔ THE COERCION MUST NOT LIVE INSIDE `if lost_answer:`.
                # `lost_answer` additionally requires TRUTHY completion_tokens,
                # so a run that returned None having billed ZERO (or having
                # reported no) completion tokens skipped the branch entirely,
                # was booked status="success" below, and handed the caller a
                # None — the exact propagation the branch above exists to
                # prevent, arriving through the one door it does not cover.
                #
                # ⚠️ This widens the COERCION only, never the CLASSIFICATION.
                # `lost_answer` is computed above and is untouched, so
                # `empty_output` still means "the answer was lost while tokens
                # were billed" and nothing else. And `{}` / `[]` / `""` are not
                # None, so this branch cannot see them: they keep booking
                # success, exactly as the boundary on the status= argument
                # below requires. Widening this to `if not output:` would
                # manufacture a defect rate out of correct "found nothing"
                # answers — do not.
                logger.warning(
                    "Agent %s - output is None with no completion tokens billed. "
                    "Coercing to empty string; the status stays success because "
                    "no answer was lost, but the caller must never receive None.",
                    agent_name,
                )
                output = ""

            metrics_client.record_execution(
                agent_name=agent_name,
                agent_type="pydantic_ai",
                model_provider=self.config.model_provider,
                model_name=self.config.model,
                duration_seconds=duration_seconds,
                # Deliberately still True. This counter means "the run
                # completed without raising", which it did; flipping it would
                # redefine agent_executions_failed (today: the exception path)
                # and silently change every dashboard built on it. The lost
                # answer is expressed on the status dimension below, which is
                # the metric that has one.
                success=True,
                token_usage=token_usage_dict,
            )

            # Record AA-specific metrics for meta-improvement tracking
            metrics_client.record_aa_prompt_execution(
                prompt_id=agent_name,  # Use agent name as prompt_id
                worker_name=agent_name,
                # ⚠️ "empty_output" means THE ANSWER WAS LOST, not "the answer
                # was short". Do NOT widen this to cover vacuous JSON: `{}` is
                # schema-VALID for the agents that emit it most —
                # ks-relationship-extractor, ks-entity-spotter and
                # principle_consistency declare no `required`, so `{}` is a
                # correct "found nothing" (measured: 253 of 375 empty_result
                # bodies over 24h, finish_reason "stop" on 100%,
                # validation_passed true). Counting those as failures would
                # manufacture a defect rate out of correct answers.
                # ⭐ vacuous_output (2026-09-03) is a FIFTH status, deliberately
                # NOT a widening of empty_output. The comment above is right and
                # stands: `{}` from a no-`required` schema is a correct "found
                # nothing" and must keep booking success. What was invisible is
                # the OTHER shape — a schema that DOES declare required fields
                # returning an object with none of them. That is a well-formed
                # WRONG answer, and it was being counted as success.
                #
                # Precedent for adding a value rather than moving a boundary:
                # agent_metrics.py's own docstring records that the shipped set
                # is (success, empty_output, timeout, failed) and that
                # `validation_failed` was documented but never emitted.
                #
                # ⚠️ SAFE FOR THE ONE LIVE CONSUMER, verified from n8n's own DB
                # 2026-09-03: workflow ckwHNOGL4mZxv1a1 "AA Prompt Statistics
                # Collector" (active) queries
                #   sum(aa_prompt_execution_total) by (prompt_id, status)
                # — it GROUPS BY status and assigns keys dynamically, so a new
                # value is folded in rather than dropped or crashed on. There are
                # zero other consumers: `aa_prompt_execution` appears nowhere in
                # gitops-production (no rule, no dashboard).
                status=(
                    "empty_output"
                    if lost_answer
                    else "vacuous_output"
                    if vacuous_answer
                    else "success"
                ),
                latency_seconds=duration_seconds,
                input_tokens=token_metrics.prompt_tokens if token_metrics else None,
                output_tokens=(token_metrics.completion_tokens if token_metrics else None),
                model=self.config.model,
            )

            # External-call metering: ANY call to a provider off owner hardware
            # goes direct-to-vendor and bypasses inference-router's metering, so
            # emit a token-accurate cost_event here — the tenant's own key
            # (funding=tenant_byo), a managed-BYO row or the owner's env key
            # (funding=platform_external). ⛔ The gate used to be
            # `cost_source == BYO_COST_SOURCE`, which only the BYO resolver
            # stamps: every PLATFORM-paid external call (owner env key, admin-
            # pinned provider) emitted NOTHING — unmetered spend. Platform-
            # routed providers stay out: inference-router is their single meter.
            # Fire-and-forget + no-op when unconfigured. Local imports avoid a
            # module-load cycle with the factory layer.
            from src.telemetry.cost_events import (
                emit_byo_cost_event_task,
                funding_for,
                is_external_call,
            )

            if is_external_call(self.config.model_provider) and token_metrics:
                # What the upstream ACTUALLY served. self.config.model is only
                # what we asked for — for inference-router that is a routing
                # alias (tier-medium), and billing prices on served_model, so
                # emitting the alias leaves cost_usd NULL forever (issue #137).
                #
                # Read from the hoisted extraction above rather than re-running
                # it: one traversal, one answer, so the cost event and the
                # caller's usage_metrics can never disagree about which model
                # served the run.
                cost_provider = resolve_billable_provider(
                    served_provider=served_provider,
                    cost_provider=self.config.cost_provider,
                    model_provider=self.config.model_provider,
                    served_model=served_model,
                    requested_model=self.config.model,
                )
                if cost_provider is None:
                    # Unpriceable: only a routing name is available. Emitting
                    # would write a permanently-NULL cost_usd row that no later
                    # pass repairs, so drop it and make the lost spend loud and
                    # greppable instead of silent.
                    logger.error(
                        "byo_cost_event_skipped_unpriceable_provider worker=%s "
                        "configured_model=%s served_model=%s model_provider=%s "
                        "cost_provider=%s in=%s out=%s",
                        agent_name,
                        self.config.model,
                        served_model,
                        self.config.model_provider,
                        self.config.cost_provider,
                        token_metrics.prompt_tokens,
                        token_metrics.completion_tokens,
                    )
                else:
                    emit_byo_cost_event_task(
                        provider=cost_provider,
                        # Keep the requested alias on `model` for attribution;
                        # served_model is what gets priced.
                        model=self.config.model,
                        served_model=served_model,
                        input_tokens=token_metrics.prompt_tokens or 0,
                        output_tokens=token_metrics.completion_tokens or 0,
                        latency_ms=int(duration_seconds * 1000),
                        owner_provided=self.config.owner_provided,
                        funding=funding_for(
                            model_provider=self.config.model_provider,
                            tenant_byo_key=self.config.tenant_byo_key,
                            owner_provided=self.config.owner_provided,
                        ),
                        key_fp=self.config.key_fingerprint if self.config.tenant_byo_key else "",
                        key_role=self.config.byo_role,
                        # The provider the model was BUILT with, not the billing
                        # provider: `cost_provider` above has already rewritten a
                        # routing name into a real upstream, so it can no longer
                        # tell the emitter that this call went through
                        # inference-router. Passing the raw model_provider is what
                        # lets the emitter refuse a platform-routed double-emit.
                        model_provider=self.config.model_provider,
                        # WHOSE WORK this is. `agent_name` is `self.name`, which
                        # is `config.name` -- byte-identical to the value that
                        # goes out on `X-Worker-Name`, so the event and the wire
                        # header can never disagree about which worker ran.
                        worker_name=agent_name,
                    )

            # Provenance is attached at the RETURN, not beside the routing stamp
            # above: when usage was unavailable it CREATES a metrics object, and
            # the token-metric and BYO cost-event branches above gate on
            # ``token_metrics`` truthiness — attaching earlier would book a
            # zero-token cost_event for a run whose usage was never measured.
            return WorkerAgentRunResult(
                output=output,
                usage_metrics=self._attach_sampling_provenance(
                    token_metrics, run_sampling=run_sampling, run_result=response
                ),
            )

        except asyncio.CancelledError:
            # ⛔ CancelledError is a BaseException in py3, so it matched NEITHER
            # `except TimeoutError` NOR `except Exception` below, and the
            # `finally` records no outcome. A cancelled run therefore emitted a
            # start (record_agent_start) and an end (record_agent_end) with
            # NOTHING in between: it was invisible on aa_prompt_execution and on
            # both execution counters, so a wave of shutdown- or
            # disconnect-driven cancellations read as runs that simply never
            # finished rather than as runs that were cancelled.
            self._record_cancelled_run(
                metrics_client, agent_name, asyncio.get_running_loop().time() - start_time
            )
            # ⛔ RE-RAISE, ALWAYS — never swallow a cancellation. Returning
            # normally here would tell the awaiting task the cancel took effect
            # while this coroutine completed anyway; asyncio.timeout(),
            # TaskGroup and every cancel-scope above depend on it propagating.
            raise
        except TimeoutError:
            logger.exception("Agent %s - Timeout waiting for response", agent_name)
            # Record failed execution
            duration_seconds = asyncio.get_running_loop().time() - start_time
            metrics_client.record_execution(
                agent_name=agent_name,
                agent_type="pydantic_ai",
                model_provider=self.config.model_provider,
                model_name=self.config.model,
                duration_seconds=duration_seconds,
                success=False,
            )
            # Record AA timeout metric
            metrics_client.record_aa_prompt_execution(
                prompt_id=agent_name,
                worker_name=agent_name,
                status="timeout",
                latency_seconds=duration_seconds,
                model=self.config.model,
            )
            timeout_error = TimeoutError(f"Agent '{agent_name}' timed out")
            self._stamp_sent_usage_metrics(timeout_error, run_sampling)
            raise timeout_error from None
        except Exception as e:
            detail = extract_error_detail(e)
            # Elapsed is computed BEFORE the log, not after. It used to be
            # derived below, purely for the metrics call, so the error line
            # itself carried no duration at all -- and duration is the single
            # field that separates a stale-socket EOF (~1.6s) from a timeout
            # (ir_http_read_timeout_s, 600s). The 2026-08 investigation had to
            # reconstruct 13,689 Running→Error pairs from state transitions to
            # recover a number this line already had in scope.
            duration_seconds = asyncio.get_running_loop().time() - start_time
            # The join key to inference-router's logs, and to the
            # ir_http_request_failed line emitted underneath by
            # DiagnosticAsyncHTTPTransport. Read from the same ContextVar the
            # outbound hook stamps as X-Correlation-Id, so the two agree by
            # construction. connection_state is NOT available here -- it is a
            # property of one HTTP attempt, not of the agent run, and lives on
            # the transport line that shares this request_id.
            #
            # ⛔ `detail` ALREADY contributes `ir_request_id`, and it is not a
            # substitute: extract_error_detail reads it off the RESPONSE header
            # `x-ir-request-id`, so it is None for exactly the failures that
            # need it. A connection that dies before one response byte carries
            # no response, hence no header, hence no id -- which is precisely
            # how a 13,689-failure population ended up with no join key. This
            # one is request-side and survives the absence of a response.
            from src.api.identity_context import get_correlation_id_from_context

            run_detail = {
                "agent_name": agent_name,
                "elapsed_s": round(duration_seconds, 4),
                "target_base_url": self.config.base_url,
                "model_provider": self.config.model_provider,
                "request_id": get_correlation_id_from_context() or "",
                **detail,
            }
            logger.error(
                "Error in PydanticAIAgent.run for agent %s [classification=%s status=%s "
                "elapsed_s=%.4f target_base_url=%s model_provider=%s request_id=%s]: %s",
                agent_name,
                detail["error_classification"],
                detail["status_code"],
                duration_seconds,
                run_detail["target_base_url"],
                run_detail["model_provider"],
                run_detail["request_id"],
                str(e),
                exc_info=True,
                extra=run_detail,
            )

            metrics_client.record_execution(
                agent_name=agent_name,
                agent_type="pydantic_ai",
                model_provider=self.config.model_provider,
                model_name=self.config.model,
                duration_seconds=duration_seconds,
                success=False,
                error_classification=detail["error_classification"],
            )
            # ⛔ THE AA OUTCOME IS BOOKED **AFTER** RECOVERY RESOLVES, AND THE
            # ORDER IS THE FIX. status="failed" used to be emitted right here,
            # above the recovery call, and a run that _handle_mcp_error_recovery
            # then RESCUED returned its answer with no compensating record. So
            # every MCP-recovered run was booked 100% failed, and the recovery
            # path had a 0% success rate on the one metric that could have shown
            # it working — by construction, not by measurement.
            #
            # `finally` rather than a tail call, so exactly one record is emitted
            # on EVERY exit: recovery returning a result, recovery returning
            # None, and recovery itself raising (it re-raises TimeoutError).
            # One aa_prompt_execution per run on all three — no double count,
            # and no path that books nothing.
            #
            # ⚠️ Deliberately scoped to the metric that HAS a status dimension.
            # record_execution above keeps success=False untouched: that counter
            # means "the run raised", which it did, and flipping it would
            # redefine agent_executions_failed for every dashboard built on it —
            # the same argument this file already makes on the success path.
            result = await self._recover_and_book_outcome(
                metrics_client=metrics_client,
                error=e,
                prompt=prompt,
                agent_name=agent_name,
                start_time=start_time,
                error_classification=detail["error_classification"],
                run_sampling=run_sampling,
            )

            if result:
                return result

            self._stamp_sent_usage_metrics(e, run_sampling)
            raise
        finally:
            # Always record agent end
            metrics_client.record_agent_end(agent_name, "pydantic_ai")
            # Release this run's transport reference. If `cleanup()` retired the
            # instance while we were mid-flight, the LAST run out is the one that
            # actually closes the pool — never the caller that evicted the slot.
            await self._release_run()

    async def _handle_mcp_error_recovery(
        self,
        error: Exception,
        prompt: str | list[Any],
        agent_name: str,
        *,
        run_sampling: RunSampling | None = None,
    ) -> WorkerAgentRunResult | None:
        """Handle MCP error recovery by reinitializing servers and retrying.

        Args:
            error: The original error that occurred
            prompt: The prompt to retry with
            agent_name: Name of the agent for logging

        Returns:
            WorkerAgentRunResult if recovery successful, None otherwise

        Raises:
            asyncio.TimeoutError: If retry times out
        """
        if "MCP" not in str(error) or not self.config.mcp_servers:
            return None

        logger.warning("MCP error detected, attempting to reinitialize servers: %s", error)
        self.mcp_servers_ready = False

        try:
            await self._initialize_mcp_servers()
            if not self.mcp_servers_ready:
                return None

            logger.info("MCP servers reinitialized, retrying...")
            retry_input = self._to_pydantic_ai_parts(prompt)
            # Use asyncio.timeout() instead of wait_for() to avoid creating a new task.
            # Same budget as the main run path (was a stray 240s — an inverted hop
            # below the 600s downstream inference-router read budget).
            async with asyncio.timeout(self._agent_run_timeout()):
                retry_response = await self.agent.run(retry_input, **self._run_kwargs(run_sampling))

            # Extract token usage on retry
            token_metrics_retry = self._extract_token_usage(agent_name, retry_response)
            # Same stamping as the main path — this branch returns a result to
            # the SAME MessageResponse builder, so omitting it here would make
            # served_model present or absent depending on whether an MCP server
            # happened to need reinitialising mid-run.
            self._attach_served_routing(token_metrics_retry, retry_response)
            token_metrics_retry = self._attach_sampling_provenance(
                token_metrics_retry, run_sampling=run_sampling, run_result=retry_response
            )

            return WorkerAgentRunResult(
                output=retry_response.output,
                usage_metrics=token_metrics_retry,
            )
        except TimeoutError:
            logger.exception("Retry after MCP reinitialize timed out")
            raise
        except UnexpectedModelBehavior as retry_exc:
            # Same 422 as the main path when the re-run spends its budget on a
            # bound violation; anything else falls through to the generic arm.
            self._raise_if_bound_exhausted(retry_exc)
            detail = extract_error_detail(retry_exc)
            logger.error(
                "Retry after MCP reinitialize failed [classification=%s status=%s]: %s",
                detail["error_classification"],
                detail["status_code"],
                str(retry_exc),
                exc_info=True,
                extra=detail,
            )
            return None
        except Exception as retry_exc:
            detail = extract_error_detail(retry_exc)
            logger.error(
                "Retry after MCP reinitialize failed [classification=%s status=%s]: %s",
                detail["error_classification"],
                detail["status_code"],
                str(retry_exc),
                exc_info=True,
                extra=detail,
            )
            return None

    def add_tools_from_json(self, json_tools: list[dict[str, Any]]) -> None:
        """Add tools to the agent from JSON tool definitions.

        Args:
            json_tools: List of tool definitions in JSON format

        Raises:
            Exception: If tool creation or validation fails
        """
        try:
            for json_tool in json_tools:
                # Validate tool structure before processing
                tool_template = ToolTemplate.from_json(json_tool)
                tool = tool_template.create_tool()
                if tool.name in self._tool_registry:
                    logger.warning(
                        "Tool %s already exists in PydanticAIAgent registry, overwriting",
                        tool.name,
                    )
                self._tool_registry[tool.name] = tool
                logger.info(
                    "Added tool %s to PydanticAIAgent '%s' (local registry only)",
                    tool.name,
                    self.agent_id,
                )
        except Exception:
            logger.exception("Error adding tools from JSON")
            raise

    async def check_health(self) -> bool:
        """Health check for the worker.

        Returns:
            True if agent is healthy
        """
        try:
            # Basic health check - agent exists and is configured
            # NOTE: pydantic-ai manages MCP server lifecycle, so we don't check them here
            return self.agent is not None
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    async def _is_mcp_server_healthy(self, server: MCPServerSSE | MCPServerStdio) -> bool:
        """Check if MCP server is healthy.

        Args:
            server: MCP server instance to check

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Check if server is running
            if hasattr(server, "is_running"):
                return bool(getattr(server, "is_running", False))

            # Basic check - see if server object exists and has expected attributes
            return server is not None and hasattr(server, "command")

        except AttributeError as e:
            logger.debug(f"MCP server missing expected attributes: {e}")
            return False
        except Exception as e:
            logger.warning(f"MCP server health check failed: {e}")
            return False

    async def _close_model_client(self) -> None:
        """Actually close the model's ``AsyncOpenAI`` (and its httpx pool). Idempotent.

        Reachable for the main path (inference_router / OpenAI family) via
        ``agent.model.client`` (an ``AsyncOpenAI`` with an async ``close()``); other
        model types may not expose it, so this is fully defensive and never raises.

        The ``_transport_closed`` flag is set BEFORE the await, and there is no
        await between the read and the write, so two concurrent callers on one
        event loop cannot both reach ``close()``.
        """
        if self._transport_closed:
            return
        self._transport_closed = True
        model = getattr(self.agent, "model", None)
        client = getattr(model, "client", None) if model is not None else None
        close = getattr(client, "close", None) if client is not None else None
        if callable(close):
            with suppress(Exception):
                await close()

    async def _release_run(self) -> None:
        """Drop this run's transport reference; close the pool if we are the last out.

        Called from ``run()``'s ``finally``, so it runs on the success path, the
        error path AND under cancellation. The close is ``shield``ed for the
        cancellation case: a bare ``await`` in a ``finally`` while a
        ``CancelledError`` is propagating raises at the await point, which would
        skip the close entirely and leak the pool — precisely the leak the close
        exists to prevent. ``shield`` still re-raises here, but the wrapped task
        is owned by the loop and runs to completion regardless.
        """
        self._active_runs -= 1
        if self._retired and self._active_runs <= 0:
            await asyncio.shield(self._close_model_client())

    async def cleanup(self) -> None:
        """Retire this agent; close its HTTP transport once no run is still using it.

        ⛔ THIS METHOD DOES NOT NECESSARILY CLOSE ANYTHING, AND THAT IS THE FIX.

        Why the close exists at all: pydantic-ai manages MCP server lifecycle but
        NOT the model's HTTP client. Every provider builds its own ``AsyncOpenAI``
        (which owns an ``httpx`` connection pool), and the cache rebuilds an
        instance on every transient-error recreate, on a failed health check, and
        on a tenant config invalidation — so a no-op cleanup leaks one pool per
        rebuild on the long-lived shared gateway, trending toward fd/pool
        exhaustion. ``TestCleanupClosesHttpClient`` pins that the pool really is
        released. The close is load-bearing; it cannot simply be deleted.

        Why it must not be UNCONDITIONAL: one cached instance serves every
        concurrent request on the same cache slot. Closing the pool from the
        caller that happens to evict severs the sockets of the siblings that are
        mid-flight inside ``agent.run()``. Measured on the deployed #209
        diagnostics: 100% of the failures were ``httpcore.ReadError`` with
        ``response_headers_received=False`` and an EMPTY message (a LOCAL close,
        not a peer RST), p50 4.81 ms after a ``Cleaning up agent`` line (511 of
        512 within 50 ms; an equally frequent unrelated INFO line gave p50 340 ms
        and 0 within 3 ms), and 534 cleanups over just 10 distinct cache keys in
        5 minutes with one key evicted 155 times — i.e. the collisions are WITHIN
        a slot, not across the variants #208's ``remove_instance`` separated.
        Each severed sibling surfaced as ``APIConnectionError("Connection
        error.")``, which ``_is_connection_error`` scores transient, which put an
        innocent run into the retry ladder, whose own recreate cleaned up the
        next generation of siblings. That is the amplification loop.

        ⚠️ ``connection_state`` measured new 4,740 / reused 12 — the exact
        INVERSE of a stale-keepalive fault. Do not "fix" this with
        ``keepalive_expiry`` or pool sizing; those change nothing here.

        So: mark the instance retired and hand the close to whoever leaves last.
        The cache has already dropped its reference by the time this returns, so
        no NEW run can join; the in-flight ones finish on a live pool and the
        last one out calls ``_close_model_client`` from ``_release_run``. Runs
        are bounded by ``_agent_run_timeout`` (``asyncio.timeout``), so
        ``_active_runs`` cannot stay above zero forever and the deferral cannot
        become a permanent leak.
        """
        logger.info(f"Cleaning up agent {self.name}")
        self._retired = True
        if self._active_runs > 0:
            logger.info(
                "Agent %s transport close deferred: %d run(s) still in flight",
                self.name,
                self._active_runs,
            )
            return
        await self._close_model_client()

    def get_tool(self, tool_name: str) -> Tool | None:
        """Retrieve a tool from the registry by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tool_registry.get(tool_name)


def create_pydantic_ai_agent(  # noqa: PLR0913 -- agent factory accepts one arg per configurable axis (model, tools, mcp, telemetry, ...); refactor to AgentConfig dataclass is a separate API change
    name: str,
    description: str,
    model_provider: ModelProvider = "vllm",
    model: str = LOCAL_LLM_MODEL,
    temperature: float = 0.7,
    db_path: str = "",
    system_prompt: str | None = None,
    instruction: str | None = None,
    user_prompt: str = "",
    tools: list[Tool] | None = None,
    deps: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    tools_dir: str | None = None,
    mcp_servers_list: list[MCPServerSSE | MCPServerStdio] | None = None,
    output_schema: dict[str, Any] | None = None,
    output_validator_code: str | None = None,
    max_tokens: int | None = None,
    enable_reasoning: bool | None = None,  # noqa: FBT001 -- keyword-only at every call site
    output_retries: int | None = None,
    worker_version: str | None = None,
    tenant_id: str | None = None,
    seed: int | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    ir_provider: str | None = None,
    tenant_byo_key: bool = False,  # noqa: FBT001, FBT002 -- keyword-only at every call site
) -> PydanticAIAgent:
    """Create and configure a PydanticAI agent.

    Args:
        name: Agent name
        description: Agent description
        model_provider: LLM provider to use
        model: Model name
        temperature: Sampling temperature
        db_path: Database path for persistence
        system_prompt: System prompt for the agent
        instruction: Instruction for the agent
        user_prompt: Initial user prompt
        tools: List of tools available to the agent
        deps: Dependencies for the agent
        base_url: Custom base URL for the provider
        api_key: API key for the provider
        tools_dir: Directory containing tool definitions
        mcp_servers_list: List of MCP servers to use
        output_schema: JSON schema for structured output
        output_validator_code: Python code for output validator function
        max_tokens: Generation budget (None ⇒ provider default)
        enable_reasoning: Tri-state chain-of-thought switch (None ⇒ send nothing)
        seed: Sampling seed (None ⇒ no seed key is sent)
        repetition_penalty: vLLM repetition penalty (None ⇒ nothing is sent)
        frequency_penalty: OpenAI frequency_penalty (None ⇒ nothing is sent)

    Returns:
        Configured PydanticAIAgent instance
    """
    logger.info(
        "Creating PydanticAI agent '%s' with model %s:%s.",
        name,
        model_provider,
        model,
    )
    if mcp_servers_list:
        logger.info(
            "Agent '%s' will be configured with %d MCP servers.",
            name,
            len(mcp_servers_list),
        )

    config = PydanticAIAgentConfig(
        name=name,
        description=description,
        model_provider=model_provider,
        model=model,
        temperature=temperature,
        seed=seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        ir_provider=ir_provider,
        max_tokens=max_tokens,
        enable_reasoning=enable_reasoning,
        system_prompt=system_prompt or "You are a helpful assistant.",
        instruction=instruction,
        user_prompt=user_prompt,
        tools=tools or [],
        db_path=db_path,
        deps=deps or [],
        base_url=base_url,
        api_key=api_key,
        tools_dir=tools_dir,
        mcp_servers=mcp_servers_list if mcp_servers_list else [],
        output_schema=output_schema,
        output_validator_code=output_validator_code,
        output_retries=output_retries,
        worker_version=worker_version,
        # Identity baked at construction, and ONLY tenant_id. This is safe --
        # and necessary -- because WorkerInstanceCache keys instances by
        # `tenant_id:worker_name:work_tier[:product]` (src/api/workers/
        # _globals.py::_cache_key), so an instance is never shared across
        # tenants. license_id and subject_id are deliberately NOT baked: neither
        # is part of that key, so baking them WOULD mis-attribute every request
        # after the first. They stay on the per-request ContextVar layer in
        # InferenceRouterOpenAIProvider._identity_event_hook.
        tenant_id=tenant_id,
        # Tenant-resolved BYO key marker (llm_config_resolver); see the field.
        tenant_byo_key=tenant_byo_key,
    )
    return PydanticAIAgent(config=config)


# Auto-register PydanticAI agent with the registry
def _register_pydantic_ai_agent() -> None:
    """Register PydanticAI agent with the global registry."""
    try:
        from .agent_registry import register_agent

        register_agent("pydantic_ai", create_pydantic_ai_agent, PydanticAIAgent)
        logger.info("Registered pydantic_ai agent type with global registry")
    except Exception as e:
        logger.warning(f"Failed to register pydantic_ai agent: {e}")


# Register on module import
_register_pydantic_ai_agent()
