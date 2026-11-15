"""Token-accurate cost-event emitter for every EXTERNAL LLM call.

⭐ 2026-10-07 (D1/D3): every call to a provider off owner hardware emits here,
stamped with ``funding`` — ``tenant_byo`` (the tenant's own key) or
``platform_external`` (managed-BYO row or the owner's env key; the latter used
to emit NOTHING). ``local_gpu`` work is inference-router's to meter. Events also
carry ``project_id``, ``job_id`` and a key fingerprint (``usage_extra.key_fp``).

BYO (bring-your-own-provider) LLM calls build a *direct* provider client in the
worker (see ``llm_config_resolver`` + ``pydantic_ai_agent``'s ModelFactory) and
therefore bypass the inference-router — so they are never metered by the
inference-router's own cost emitter. This module emits an equivalent cost_event
to the shared metering Redis stream (``improvement-loop:costs``) so
``metering_service`` can attribute and price the usage from its effective-dated
``provider_pricing`` table.

Design mirrors ``inference-router``'s emitter:
- **fire-and-forget** so it never blocks or fails the agent run; all errors are
  caught and logged. The scheduled task is retained in a module-level set with a
  done-callback that discards it, so it can never be garbage-collected before the
  XADD completes (a bare ``create_task`` drops a strong ref — Python may GC the
  task mid-flight and silently lose the cost event);
- **no-op when ``metering_redis_url`` is unset** — the code is safe to deploy
  before the gitops env wiring lands;
- ``cost_usd`` is emitted as ``0.0`` (advisory): metering_service is the source
  of truth and recomputes USD from ``served_model`` against ``provider_pricing``;
- ``usage_extra={"byo": "true"}`` is what metering's rollup keys on to set the
  ``cost_daily.byo`` flag, and ``cost_source="worker-management"`` distinguishes
  these events from inference-router's.

⚠️ "BYO" means the call PHYSICALLY LEFT the platform. The `local` preset's
provider IS inference-router, so a `local`-preset call is metered by the router
AND was being metered again here — a duplicate billing row flagged ``byo=true``
for tokens burned on owner hardware. Two layers now prevent that:
``llm_config_resolver`` stamps ``cost_source`` only for an ``is_external``
preset, and ``emit_byo_cost_event_task`` REFUSES any ``model_provider`` in
``PLATFORM_ROUTED_PROVIDERS``. Every event also carries an ``event_key``
(uuid7, minted once per call) so billing can reject a redelivered emit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import redis.asyncio as redis

from src.api.identity_context import (
    TENANT_PROVENANCE_UNKNOWN,
    TENANT_PROVENANCES,
    get_app_id_from_context,
    get_caller_client_id_from_context,
    get_correlation_id_from_context,
    get_effective_actor_id,
    get_effective_job_id,
    get_license_id_from_context,
    get_project_id_from_context,
    get_spend_scope_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_provenance_from_context,
    valid_project_id,
)
from src.settings import get_settings
from src.telemetry import attribution_metrics as _am
from src.telemetry.agent_metrics import (
    METERING_DROP_REASON_NO_EVENT_LOOP,
    METERING_DROP_REASON_NO_TENANT,
    METERING_DROP_REASON_UNCONFIGURED,
    get_metrics_client,
)

logger = logging.getLogger(__name__)


def _record_drop(reason: str) -> None:
    """Count one cost event that never reached the bus. Never raises.

    Wrapped because this runs on a path where metering has ALREADY failed;
    an exporter problem must not escalate a lost cost event into a failed
    agent run. Deliberately swallowed at debug — the WARNING that carries the
    operator-visible message is emitted by the caller, next to this call.
    """
    try:
        get_metrics_client().record_metering_event_dropped(reason)
    except Exception:  # pragma: no cover - defensive; observe-only
        logger.debug("metering drop counter failed", exc_info=True)


def _record_emitted(tenant_provenance: str) -> None:
    """Count one cost event successfully scheduled. Never raises.

    The DENOMINATOR. Without it the drop counter cannot be read as a rate:
    40 drops is a rounding error against 4M emits and an outage against 40.

    Also counts the event under HOW its tenant was established, on a SEPARATE
    counter. The denominator above stays unlabelled on purpose — it is divided
    by the sum of the drop counter and must share no label with it — so the
    provenance breakdown cannot ride on it.
    """
    try:
        get_metrics_client().record_metering_event_emitted()
    except Exception:  # pragma: no cover - defensive; observe-only
        logger.debug("metering emit counter failed", exc_info=True)
    try:
        get_metrics_client().record_metering_event_tenant_provenance(tenant_provenance)
    except Exception:  # pragma: no cover - defensive; observe-only
        logger.debug("metering provenance counter failed", exc_info=True)


# service_key + cost_source stamped on every event this module emits. Keep in
# sync with metering_service's expectations and llm_config_resolver.BYO_COST_SOURCE.
SERVICE_KEY = "worker-management"
COST_SOURCE = "worker-management"

# Providers that are NOT a billable upstream but a hop to one on owner hardware.
# inference-router meters everything it serves, so an event emitted for one of
# these is a DUPLICATE of a row billing already has. Kept local (a frozenset of
# strings) rather than imported from the agent layer, which would create a
# module-load cycle; mirrors ``pydantic_ai_agent.SELF_HOSTED_PROVIDERS`` and
# ``llm_config_resolver.SYSTEM_ROUTED``.
PLATFORM_ROUTED_PROVIDERS: frozenset[str] = frozenset(
    {"inference_router", "vllm", "mlx", "vlm_local"}
)

FUNDING_LOCAL_GPU = _am.FUNDING_LOCAL_GPU
FUNDING_PLATFORM_EXTERNAL = _am.FUNDING_PLATFORM_EXTERNAL
FUNDING_TENANT_BYO = _am.FUNDING_TENANT_BYO


def funding_for(*, model_provider: str | None, tenant_byo_key: bool, owner_provided: bool) -> str:
    """WHO PAYS the upstream for one LLM call (owner decision D1).

    * ``local_gpu`` — the model runs on owner hardware (a platform-routed
      provider; inference-router meters it).
    * ``tenant_byo`` — the call ran on the TENANT's own key (its own BYO row).
    * ``platform_external`` — an external vendor paid with the PLATFORM's key:
      the owner's env key (proven commercial-platform workspace) or a managed
      BYO row (``owner_provided``: the owner's key stored for that tenant).
    """
    if model_provider in PLATFORM_ROUTED_PROVIDERS:
        return FUNDING_LOCAL_GPU
    if tenant_byo_key and not owner_provided:
        return FUNDING_TENANT_BYO
    return FUNDING_PLATFORM_EXTERNAL


def is_external_call(model_provider: str | None) -> bool:
    """True when the model was built for a provider OFF owner hardware.

    Every such call MUST emit a cost event from this service: nothing else
    meters it (inference-router never sees it).
    """
    return bool(model_provider) and model_provider not in PLATFORM_ROUTED_PROVIDERS


def key_fingerprint(api_key_encrypted: object) -> str:
    """Audit fingerprint of a STORED tenant key version — never of the key.

    sha256 of the row's ENCRYPTED blob (AES-GCM with a random nonce, so every
    save is a distinct version), first 16 hex chars. It identifies which stored
    key version served a call, and reveals nothing about the key itself.
    """
    if not api_key_encrypted:
        return ""
    import hashlib

    raw = (
        api_key_encrypted
        if isinstance(api_key_encrypted, bytes | bytearray | memoryview)
        else str(api_key_encrypted).encode()
    )
    return hashlib.sha256(bytes(raw)).hexdigest()[:16]


# Memoized client kept in a dict container (mutation, not rebinding) to avoid a
# module-level `global` statement.
_redis_state: dict[str, redis.Redis] = {}

# Strong references to in-flight emit tasks. asyncio only holds a *weak* reference
# to a task created by create_task(), so without this set Python may garbage-collect
# the task before its XADD completes — silently dropping a billable BYO cost event.
# The done-callback discards the task once it finishes. (Mirrors inference-router's
# TelemetrySink._bg_tasks pattern.)
_background_tasks: set[asyncio.Task[None]] = set()


def _get_redis() -> redis.Redis | None:
    """Lazily build (and memoize) the metering Redis client, or None if unset."""
    settings = get_settings()
    url = settings.metering_redis_url
    if not url:
        return None
    client = _redis_state.get("client")
    if client is None:
        client = redis.from_url(url)
        _redis_state["client"] = client
    return client


async def _emit(  # noqa: PLR0913 -- one arg per metering-event field XADD'd to the stream
    *,
    stream: str,
    tenant_id: str,
    license_id: str,
    subject_id: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    event_key: str = "",
    correlation_id: str = "",
    stage: str = "",
    caller_client_id: str = "",
    app_id: str = "",
    actor_id: str = "",
    worker_name: str = "",
    owner_provided: bool = False,
    served_model: str | None = None,
    tenant_provenance: str = TENANT_PROVENANCE_UNKNOWN,
    project_id: str = "",
    job_id: str = "",
    spend_scope: str = "",
    funding: str = "",
    key_fp: str = "",
) -> None:
    """XADD a single BYO cost event. Swallows all errors (hot-path safety).

    ``tenant_provenance`` says how ``tenant_id`` was established — see
    :data:`src.api.identity_context.TENANT_PROVENANCES`. Defaults to
    ``unknown``, never to ``license``: a direct caller that does not state it
    has not proven anything, and the whole point of the field is that an
    unproven attribution must not be able to look proven.
    """
    if not funding:
        # A direct caller that did not say: the old byo/owner_provided split.
        funding = FUNDING_PLATFORM_EXTERNAL if owner_provided else FUNDING_TENANT_BYO
    try:
        client = _get_redis()
        if client is None:
            return
        fields: dict[str, str] = {
            "tenant_id": tenant_id,
            "license_id": license_id,
            "subject_id": subject_id,
            # Cost-correlation: ties this event to the caller's operation
            # (w2v: run_id + DAG stage). Empty when the caller did not stamp
            # X-Correlation-Id; billing parses it but tolerates absence.
            "correlation_id": correlation_id,
            "stage": stage,
            # WHO MADE THIS HOP — the upstream microservice's Keycloak service
            # account, as verified by license-authz (it injects
            # `x-caller-client-id` only after checking the inbound
            # `X-Caller-Authorization` JWT). Empty when the caller presented no
            # verified identity, which is most of today's traffic; billing keeps
            # an absent caller absent rather than inventing one.
            #
            # NOT `worker_name`, which billing whitelists separately: that says
            # whose WORK this is (the origin), this says who called US. On
            # requirements -> worker-management -> inference-router they differ
            # at every hop, and that difference is what lets an app-to-app call
            # be attributed to its caller instead of to whichever service
            # happens to emit the event.
            "caller_client_id": caller_client_id,
            # WHICH APP this run belongs to — relayed verbatim from the
            # inbound X-App-Id header (see identity_context._app_id_var).
            # ⛔ Deliberately NEVER defaulted: worker_management fans in from
            # many upstream apps, and a guessed app_id would collapse all of
            # their distinct origins into one bucket. Empty when the caller
            # did not stamp X-App-Id, which is most of today's traffic.
            "app_id": app_id,
            # WHO ACTED — the Keycloak sub of the person who initiated the
            # work, relayed from the inbound X-Actor-Id. billing persists it
            # to `billing_cost_events.actor_id` (per-member rollup). Empty ⇒
            # billing records the NULL actor; ⛔ never defaulted to subject.
            "actor_id": actor_id,
            # WHICH PROJECT — the calling app's own project, from the bound
            # inbound X-Project-Id, validated exactly as inference-router does
            # (a non-uuid is absent). Never invented.
            "project_id": project_id,
            # Reporting-only sibling of project_id (billing whitelists it).
            "spend_scope": spend_scope,
            # WHO PAYS the upstream (D1): tenant_byo | platform_external |
            # local_gpu. A flat field billing may not store yet; its parser
            # lifts only known keys, so an unknown one is ignored, and the same
            # value rides in usage_extra (jsonb) below so nothing is lost.
            "funding": funding,
            # WHOSE WORK THIS IS -- the originating worker, i.e. the same value
            # this service already stamps on the outbound `X-Worker-Name`
            # header (pydantic_ai_agent.make_identity_event_hook). billing
            # whitelists `worker_name` as metadata precisely to disambiguate
            # ONE `service_key` ("worker-management") across the many distinct
            # workers that fan into it; sending nothing wasted a field that
            # exists for exactly this question.
            #
            # NOT `caller_client_id` (who called US) and NOT `app_id` (which
            # upstream APP the run belongs to) -- see the note on
            # `caller_client_id` above. On requirements -> worker-management ->
            # inference-router all three differ, and `worker_name` is the only
            # one that names the unit of work inside THIS service.
            #
            # ⛔ Never defaulted, for the same reason `app_id` is not: an
            # invented worker name would collapse distinct origins into one
            # bucket. Empty means "the emit site had no worker in scope",
            # which is honest; a guess is not.
            "worker_name": worker_name,
            "service_key": SERVICE_KEY,
            "cost_source": COST_SOURCE,
            # Idempotency key: minted ONCE per completed LLM call at the emit
            # site, so every redelivery of this stream entry (Redis streams are
            # at-least-once, and a consumer restart re-reads its pending list)
            # carries the SAME key. Billing persists it under a UNIQUE
            # constraint and skips a key it has already ingested, turning
            # at-least-once delivery into exactly-once billing.
            "event_key": event_key or str(uuid.uuid7()),
            "provider": provider,
            # `model` is what WM *requested* — for inference-router that is a
            # routing alias (e.g. "tier-medium"), kept for attribution/debugging.
            "model": model,
            # `served_model` is what billing PRICES on, so it must be the
            # concrete model IR actually resolved the alias to. Falling back to
            # `model` reproduces the old behaviour when the caller could not
            # determine it (IR predating the served_model response field).
            "served_model": str(served_model or model),
            "input_tokens": str(input_tokens),
            "output_tokens": str(output_tokens),
            "cache_creation_input_tokens": "0",
            "cache_read_input_tokens": "0",
            # Advisory only — metering recomputes USD from provider_pricing.
            "cost_usd": "0.00000000",
            "latency_ms": str(latency_ms),
            "schema_id": "",
            # The caller's X-Job-Id when it sent one, else the WM worker name
            # (the job TYPE that ran) — same value as the outbound X-Job-Id.
            "job_id": job_id,
            "fallback_used": "false",
            "provider_request_id": "",
            # metering's rollup_engine sets cost_daily.byo from usage_extra["byo"].
            # Managed BYO instead emits owner_provided (no byo) → a normal
            # marked-up line rather than a force-zero'd byo line.
            # ⛔ `tenant_provenance` IS THE ONLY THING ON THIS ROW THAT SAYS
            # WHETHER THE TENANT BEING BILLED WAS PROVEN. Without it a
            # delegated event — whose tenant came from `X-Effective-Tenant-Id`,
            # validated for spelling only — is byte-identical to a licensed one
            # apart from `tenant_id`, which is precisely the value in question
            # and therefore cannot be its own evidence. `license_id` cannot
            # substitute: `auth_middleware` deliberately NULLS it on delegation,
            # so a delegated row looks the same as a tokenless one there.
            #
            # ⛔ STAMPED ON *EVERY* EVENT, INCLUDING PROVEN ONES. Stamping only
            # the unproven ones would make "absent" mean two different things —
            # "proven" and "emitted by a pod that predates this field" — during
            # any rolling deploy or rollback, which is the same absent-vs-zero
            # trap the pre-materialised metric series below avoid. Billing's
            # rollup engine keys `cost_daily.byo` off `usage_extra["byo"]` and
            # `usage_extra["owner_provided"]` and reads no other key, so an
            # additional key changes no price, no rollup and no invoice line; it
            # is queryable as `usage_extra->>'tenant_provenance'` for
            # reconciliation. The value comes from a CLOSED set, never from
            # caller input.
            "usage_extra": json.dumps(
                {
                    **(
                        {"byo": "true"}
                        if funding == FUNDING_TENANT_BYO
                        else {"owner_provided": "true"}
                    ),
                    "funding": funding,
                    # Per-call BYO key audit: fingerprint of the stored key
                    # VERSION (sha256 prefix of the encrypted blob), never the
                    # key. Empty for the owner's env key.
                    **({"key_fp": key_fp} if key_fp else {}),
                    "tenant_provenance": (
                        tenant_provenance
                        if tenant_provenance in TENANT_PROVENANCES
                        else TENANT_PROVENANCE_UNKNOWN
                    ),
                },
                sort_keys=True,
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await client.xadd(stream, cast("dict[Any, Any]", fields), maxlen=100_000, approximate=True)
        logger.info(
            "byo_cost_event_emitted tenant=%s tenant_provenance=%s funding=%s project=%s "
            "job=%s provider=%s model=%s served_model=%s in=%d out=%d",
            tenant_id,
            tenant_provenance,
            funding,
            project_id or "-",
            job_id or "-",
            provider,
            model,
            fields["served_model"],
            input_tokens,
            output_tokens,
        )
    except Exception:
        logger.warning("byo_cost_event_emit_failed", exc_info=True)


def emit_byo_cost_event_task(  # noqa: PLR0913 -- one keyword per cost-event fact the emit site knows
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    model_provider: str,
    owner_provided: bool = False,
    served_model: str | None = None,
    worker_name: str = "",
    funding: str | None = None,
    key_fp: str = "",
    key_role: str = "",
) -> None:
    """Fire-and-forget a cost event for an EXTERNAL call, for the current identity.

    Reads tenant/license/subject from the identity ContextVars set by the auth
    middleware (the same attribution path inference-router uses). Returns
    immediately; never raises. Emits nothing when ``metering_redis_url`` is
    unset or no tenant is in context -- but "emits nothing" is no longer
    "silent": each of the three non-emitting paths logs at WARNING and
    increments ``worker_management_metering_events_dropped_total`` under its own
    ``reason``, because every one of them is real BYO spend that will never be
    billed. The success path increments the emitted denominator, without which
    a drop count cannot be read as a rate.

    ``owner_provided`` True (managed BYO — the platform owner supplied the key)
    emits ``usage_extra.owner_provided="true"`` and NO ``byo`` flag, so billing
    prices real cost and applies the tenant's markup rule instead of forcing the
    markup to 0 under ``byo_platform_fee``.

    ``served_model`` is the concrete model the upstream actually served when the
    caller requested a routing *alias* (inference-router resolves e.g.
    ``tier-medium`` server-side and echoes the resolved name back). Billing
    prices on ``served_model``, so passing the alias there yields a permanent
    ``cost_usd IS NULL`` row. None ⇒ fall back to ``model`` (old behaviour).

    ``worker_name`` is the originating worker -- the same value this service
    stamps on the outbound ``X-Worker-Name`` header. billing whitelists it as
    metadata to split one ``service_key`` ("worker-management") across the many
    workers that fan into it; every event emitted before this parameter existed
    carried the service key and nothing else, so the spend was billable but its
    origin was not recoverable. Empty ⇒ absent, never a guessed name.

    ``model_provider`` is the provider the model was actually BUILT with — not
    the billing provider. It is REQUIRED and it is a gate that can fail: a
    platform-routed provider means inference-router already metered this call,
    so emitting here would double-bill. The caller is expected to have shut the
    gate upstream (``llm_config_resolver`` no longer stamps ``cost_source`` for
    a non-external preset); this refusal is the second, load-bearing layer, and
    it logs at ERROR so a regression is loud rather than silently expensive,
    and is counted on ``wm_cost_emit_refused_total{reason="platform_routed"}``.

    ``funding`` (D1) is who pays: callers pass :func:`funding_for`. ``None``
    (a legacy caller) derives it from ``owner_provided`` exactly as the old
    byo/owner_provided split did. ``key_fp`` / ``key_role`` are the per-call
    tenant-key audit (fingerprint of the stored key version, config role); a
    call on a tenant-row key is counted on ``wm_tenant_key_use_total``.
    """
    if funding is None:
        funding = FUNDING_PLATFORM_EXTERNAL if owner_provided else FUNDING_TENANT_BYO
    if model_provider in PLATFORM_ROUTED_PROVIDERS:
        try:
            _am.COST_EMIT_REFUSED_TOTAL.labels(reason=_am.COST_EMIT_REFUSED_PLATFORM_ROUTED).inc()
        except Exception:  # pragma: no cover - observe-only
            logger.debug("cost emit refusal counter failed", exc_info=True)
        logger.error(
            "byo_cost_event_refused_platform_routed model_provider=%s provider=%s "
            "model=%s served_model=%s in=%d out=%d — inference-router already "
            "meters this call; emitting would double-bill",
            model_provider,
            provider,
            model,
            served_model,
            input_tokens,
            output_tokens,
        )
        return
    settings = get_settings()
    if not settings.metering_redis_url:
        # Metering is deliberately off, but the LLM call still happened and its
        # cost still left the platform. Benign as a CONFIGURATION, never benign
        # as an OUTCOME — so it is counted under its own reason (which is what
        # lets an alert target `no_tenant` without this drowning it) and said
        # out loud, because at LOG_LEVEL=INFO a debug line is deletion.
        logger.warning(
            "byo_cost_event_dropped reason=%s provider=%s model=%s served_model=%s "
            "in=%d out=%d — METERING_REDIS_URL is unset; this spend will never be billed",
            METERING_DROP_REASON_UNCONFIGURED,
            provider,
            model,
            served_model,
            input_tokens,
            output_tokens,
        )
        _record_drop(METERING_DROP_REASON_UNCONFIGURED)
        return
    tenant_id = get_tenant_id_from_context()
    # ⛔ READ BESIDE THE TENANT, NEVER DERIVED FROM IT. The two facts travel
    # together or the record lies: a tenant with no provenance reads as proven
    # to anyone who does not know the field can be missing. Read HERE, at the
    # call site, for the same reason `caller_client_id` is — this runs inside
    # the request's async context, before `create_task` below.
    tenant_provenance = get_tenant_provenance_from_context()
    if not tenant_id:
        # ⭐ THE defect case: metering is LIVE, the LLM call really happened,
        # and there is no owner to bill. This was logged at DEBUG, and services
        # run at LOG_LEVEL=INFO — so every one of these vanished without a
        # trace. Real BYO spend, no tenant, no record, no bill.
        logger.warning(
            "byo_cost_event_dropped reason=%s provider=%s model=%s served_model=%s "
            "in=%d out=%d — no tenant in context; real BYO spend is unattributable "
            "and will never be billed",
            METERING_DROP_REASON_NO_TENANT,
            provider,
            model,
            served_model,
            input_tokens,
            output_tokens,
        )
        _record_drop(METERING_DROP_REASON_NO_TENANT)
        return
    coro = _emit(
        stream=settings.metering_costs_stream,
        # Minted HERE, once per completed LLM call, so every redelivery of the
        # resulting stream entry carries the same key and billing's UNIQUE
        # constraint collapses it. uuid7 is time-ordered, which keeps billing's
        # key index append-only. It does NOT dedupe across producers — two
        # services metering the same physical call still mint two keys; that
        # class is what the `model_provider` refusal above prevents.
        event_key=str(uuid.uuid7()),
        tenant_id=tenant_id,
        license_id=get_license_id_from_context(),
        subject_id=get_subject_id_from_context(),
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        correlation_id=get_correlation_id_from_context(),
        stage=get_stage_from_context(),
        # Read HERE, at the call site, exactly like every other identity field
        # above — the ContextVar is live in the request context and this runs
        # BEFORE `create_task` below hands the coroutine to the event loop.
        caller_client_id=get_caller_client_id_from_context(),
        app_id=get_app_id_from_context(),
        actor_id=get_effective_actor_id(),
        # Passed by the caller, not read from a ContextVar: the worker is a
        # property of the emit SITE (which agent ran), not of the request.
        worker_name=worker_name,
        owner_provided=owner_provided,
        served_model=served_model,
        tenant_provenance=tenant_provenance,
        # Same rules as the outbound IR hook: the bound inbound X-Project-Id
        # (validated like IR does), the caller's X-Job-Id else the worker.
        project_id=valid_project_id(get_project_id_from_context()),
        job_id=get_effective_job_id(worker_name),
        spend_scope=get_spend_scope_from_context(),
        funding=funding,
        key_fp=key_fp,
    )
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        # No running loop (sync context). `coro.close()` throws the emit away
        # UNRUN — the comment above claimed "run to completion best-effort",
        # which it never did. Also real unmetered spend, but its remedy is a
        # call-site change, not an attribution fix, so it gets its own reason.
        coro.close()
        logger.warning(
            "byo_cost_event_dropped reason=%s provider=%s model=%s served_model=%s "
            "in=%d out=%d — emitted from a sync context with no running event loop; "
            "the emit was discarded unrun and this spend will never be billed",
            METERING_DROP_REASON_NO_EVENT_LOOP,
            provider,
            model,
            served_model,
            input_tokens,
            output_tokens,
        )
        _record_drop(METERING_DROP_REASON_NO_EVENT_LOOP)
        return
    # Retain a strong ref until the task completes so it can't be GC'd mid-XADD.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    try:
        _am.COST_EVENT_FUNDING_TOTAL.labels(funding=funding).inc()
        if key_fp:
            _am.TENANT_KEY_USE_TOTAL.labels(tenant=tenant_id, role=key_role or "__default__").inc()
    except Exception:  # pragma: no cover - observe-only
        logger.debug("funding/key-use counters failed", exc_info=True)
    # The denominator, incremented only once the emit is genuinely SCHEDULED —
    # past every drop branch above. Counting earlier would inflate it with the
    # very calls the drop counter exists to count.
    _record_emitted(tenant_provenance)
