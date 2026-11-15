"""Worker instance cache and global state management."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.api.tenant_constants import is_commercial_platform_tenant
from src.db.worker_registry_db import WorkerRegistryDB
from src.security.request_overrides import prewarm_owner_infra_dns
from src.services.worker_state_manager import WorkerStateManager
from src.worker.factory.llm_config_resolver import (
    enforce_customer_routing,
    resolve_llm_config,
    tenant_key_entitled,
)
from src.worker.worker_factory import create_worker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


def _cache_key(
    worker_name: str,
    tenant_id: str | None,
    work_tier: str | None = None,
    product: str | None = None,
    override_key: str | None = None,
    *,
    key_entitled: bool = False,
) -> str:
    """Compose the per-tenant, per-tier, per-product cache key.

    Worker instances are keyed by ``tenant_id:worker_name:work_tier`` because a
    tenant's BYO LLM key (and its per-tier provider/model) is baked into the
    instantiated agent — a name-only key would leak one tenant's decrypted key
    into another tenant's reused instance, and a tier-less key would serve the
    small-tier model for a high-tier request. ``work_tier`` defaults to ``_``
    (the tenant ``__default__`` row). tenant_id (UUID), worker_name (validated
    identifier) and the tier are all colon-free, so the key splits cleanly.

    ``product`` (optional, licenses App.key) is appended only when set: the BYO
    *posture* (byo_required / platform_only / local_allowed) is resolved
    per-(tenant, product), so two products can bake different providers (BYO row
    vs platform) into their agents for the same (tenant, tier) — sharing an
    instance across products would serve one product's resolved config to
    another. Absent a product the key is byte-identical to the pre-product form
    (backward compatible). App.key is a colon-free slug, so the key still splits.
    """
    if not tenant_id:
        return worker_name
    key = f"{tenant_id}:{worker_name}:{work_tier or '_'}"
    if product:
        key = f"{key}:{product}"
    # ``override_key`` is a fingerprint of the PER-MESSAGE overrides (output_schema
    # / output_validator / model_override / max_tokens / enable_reasoning /
    # system_prompt / instruction) that get BAKED into the instantiated agent.
    # Without it these keyed under the base slot, so a request carrying an override
    # rebuilt the shared instance and POISONED it for every later NON-override caller
    # (cross-caller schema/model/budget/reasoning/prompt leak). Appending it gives
    # each distinct override its own slot; absent an override the key is
    # byte-identical to the pre-override form (backward compatible).
    #
    # CARDINALITY NOTE: nothing expires an override slot, and each cached agent owns
    # an AsyncOpenAI httpx pool. This was previously masked, not solved: the endpoint
    # force-evicted the whole ``tenant:worker:`` prefix on every override request, so
    # old slots were collected as a side effect of a bug that also closed concurrent
    # runs' connections. With that eviction gone the slots persist. That is safe while
    # the override material is low-cardinality (a call site sends the SAME schema /
    # max_tokens every time, so it reuses one slot), and it is NOT safe if a caller
    # starts minting per-request schemas. A bound would have to be refcount-aware —
    # evicting a slot that an in-flight run is using is exactly the defect that was
    # just removed — so it is deliberately not attempted here.
    #
    # ⚠️ SCOPE: the above is about ``_workers`` (the INSTANCE slots) and is still
    # true. The sibling ``_worker_locks`` dict, which was leaking on exactly the
    # same caller-enumerable key, is now bounded: ``WorkerInstanceCache._held_lock``
    # reference counts it and reaps each entry at zero users. Do not read a bounded
    # lock dict as a bounded instance cache — they are two different leaks and only
    # one of them is closed.
    if override_key:
        key = f"{key}:ov-{override_key}"
    # ⛔ worker_management#245: an instance built for a caller that PROVED the
    # tenant may carry the tenant's decrypted BYO key; one built for an
    # adopted/asserted caller must not. Without their own slots, an unproven
    # caller would simply reuse the proven caller's cached agent — and its key.
    # Appended last so the tenant/worker prefix matches used for eviction are
    # unchanged, and absent entitlement the key is byte-identical to before.
    if key_entitled:
        key = f"{key}:byo-key"
    return key


@dataclass
class _LockEntry:
    """One per-cache-key :class:`asyncio.Lock` plus its in-flight user count.

    ``users`` counts the current holder AND everyone queued behind it, so it is
    non-zero for exactly as long as the entry is load-bearing.
    """

    lock: asyncio.Lock
    users: int = 0


class WorkerInstanceCache:
    """Cache for active worker instances to avoid recreation with proper MCP lifecycle management.

    ⛔ NOT THREAD-SAFE. The guarantee is **single-event-loop atomicity**:
    ``_worker_locks`` is a plain :class:`dict`, ``users += 1`` is a
    read-modify-write, and :class:`asyncio.Lock` is valid on one loop only.
    What makes :meth:`_held_lock` correct is that neither its
    get/create/increment block nor its decrement/reap block contains an
    ``await``, so no other coroutine on the same loop can be scheduled part-way
    through either. A thread hop (``asyncio.to_thread``, ``run_in_executor``),
    a second event loop, or an ``await`` inserted into either block breaks it.
    """

    def __init__(self) -> None:
        self._workers: dict[str, Any] = {}
        self._worker_locks: dict[str, _LockEntry] = {}
        self._locks_lock: asyncio.Lock = asyncio.Lock()  # Lock for managing locks dictionary

    @asynccontextmanager
    async def _held_lock(self, cache_key: str) -> AsyncIterator[asyncio.Lock]:
        """Hold the per-cache-key instance lock, reaping the entry once nobody wants it.

        ⭐ REFERENCE COUNTED, AND IT HAS TO BE. ``_worker_locks`` used to be
        populated get-or-create and never emptied, and its keys are
        CALLER-ENUMERABLE: :func:`_cache_key` folds in ``override_key``, which
        :func:`~src.api.workers.message_endpoints._override_cache_fingerprint`
        derives from per-message fields the caller supplies — ``system_prompt``,
        ``instruction``, ``output_schema``, ``output_validator``,
        ``model_override``, ``max_tokens``, ``enable_reasoning``. One
        authenticated caller varying ``system_prompt`` per request therefore
        minted one permanent :class:`asyncio.Lock` per request. See the
        CARDINALITY NOTE on :func:`_cache_key`, which already conceded this is
        unsafe "if a caller starts minting per-request schemas".

        ⛔ DO NOT "simplify" this to deleting the entry when the operation that
        created it finishes. A coroutine may be holding this lock and others may
        be queued inside ``lock.acquire()``; dropping the entry lets the next
        caller mint a FRESH lock for the same key and walk into the critical
        section beside them. For this cache the critical section is
        create-or-reuse-or-evict of ONE instance slot, so that converts a memory
        leak into a duplicate-instantiation / evict-the-live-instance bug —
        strictly worse. ``users`` is what makes removal safe: it counts the
        holder plus every waiter, and the entry is removed only at zero, where
        the lock is provably unlocked, unwaited and stateless, so a later caller
        re-creating it is indistinguishable.

        ⛔ ALL FOUR call sites must come through here.
        :meth:`remove_worker`, :meth:`remove_instance` and
        :meth:`invalidate_tenant` used to read ``_worker_locks`` with a raw
        ``.get()`` and fall back to an UNLOCKED cleanup when the key was absent.
        That was already a latent race; with reaping it would become the normal
        case, because at rest no key has an entry at all — every eviction would
        run unlocked against a concurrent :meth:`get_or_create_worker`, and
        ``remove_worker``'s ``if cache_key in self._worker_locks`` guard would
        make it evict NOTHING.

        ``_locks_lock`` is retained around the get/create/increment because it
        is this class's existing contract, but it is not what makes that block
        atomic — the absence of ``await`` inside it is. The decrement
        deliberately does NOT take ``_locks_lock``: acquiring it is an
        ``await``, and an ``await`` in a ``finally`` while a
        :exc:`asyncio.CancelledError` is propagating raises at the await point,
        which would skip the decrement and strand the entry forever — precisely
        the leak being fixed. The decrement/check/delete has no ``await``, so it
        is already atomic against other coroutines on this loop.

        ⚠️ MEASURED 2026-09-15, because the paragraph above reads stronger than
        the evidence: putting ``async with self._locks_lock`` back around the
        ``finally`` produces NO observable difference today. An uncontended
        :class:`asyncio.Lock` acquire returns on a fast path WITHOUT yielding,
        and ``_locks_lock`` is never held across a suspension, so it is never
        contended, so that ``await`` never actually suspends and never delivers
        a cancellation. The edit stays forbidden — it is one contended acquire
        away from doing exactly what the paragraph says — but it is a
        behaviourally EQUIVALENT mutant, and no test that runs the code can
        catch it. ``tests/test_worker_instance_cache_held_lock_await_free.py``
        therefore asserts the await-free rule on the AST as well as through
        arrival-offset and cancellation probes.

        ⚠️ This bounds ``_worker_locks`` ONLY. ``_workers`` itself is still
        unbounded per override slot; that is the instance-eviction problem the
        CARDINALITY NOTE describes and it is deliberately not addressed here.
        """
        async with self._locks_lock:
            entry = self._worker_locks.get(cache_key)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock())
                self._worker_locks[cache_key] = entry
            entry.users += 1
        try:
            async with entry.lock:
                yield entry.lock
        finally:
            entry.users -= 1
            # ``is entry`` asserts an invariant rather than preventing a race:
            # while ``users > 0`` no successor can be installed under this key,
            # because a newcomer that finds this entry increments it instead of
            # replacing it, and the only removal is this branch at zero. It is
            # kept because it is free and because it fails safe the day that
            # stops holding — a second removal path, an entry replaced in place,
            # or an ``await`` between the decrement and the delete.
            if entry.users <= 0 and self._worker_locks.get(cache_key) is entry:
                del self._worker_locks[cache_key]

    async def get_or_create_worker(
        self,
        worker_name: str,
        worker_data: dict[str, Any],
        *,
        tenant_id: str | None = None,
        byo_mode: str = "local_allowed",
        product: str | None = None,
        override_key: str | None = None,
    ) -> Any:
        """Get existing worker or create new one if needed.

        This method implements a caching mechanism to reuse worker instances when possible,
        avoiding the overhead of recreating workers on every request. It includes proper
        MCP (Model Control Protocol) lifecycle management.

        Args:
            worker_name: Name of the worker to retrieve or create
            worker_data: Dictionary containing worker configuration including:
                - worker_name: The normalized worker name
                - config: Worker configuration dict with tool_names, agent_role, etc.

        Returns:
            Worker instance (either existing or newly created)

        Raises:
            Exception: If worker creation fails or MCP initialization fails

        Note:
            Uses per-worker locks to prevent race conditions during concurrent access.
        """
        # Tenant- and tier-scoped cache key: a tenant's BYO LLM key + per-tier
        # provider/model are baked into the instantiated agent, so instances must
        # never be shared across tenants OR across work tiers.
        tenant_id = tenant_id or worker_data.get("tenant_id")
        config = worker_data.get("config") or {}
        work_tier = config.get("work_tier") if isinstance(config, dict) else None
        # Computed ONCE and used for both the slot and the resolver, so the
        # instance in a slot always matches the entitlement the slot names.
        key_entitled = tenant_key_entitled(tenant_id)
        cache_key = _cache_key(
            worker_name, tenant_id, work_tier, product, override_key, key_entitled=key_entitled
        )

        # Use a lock per worker to prevent race conditions. ``_held_lock`` is
        # reference counted, so the entry is reaped once no holder or waiter
        # wants it -- see its docstring for why a bare dict of locks here was a
        # caller-enumerable leak.
        async with self._held_lock(cache_key):
            # Check if worker exists and is healthy
            if cache_key in self._workers:
                worker = self._workers[cache_key]
                if await self._is_worker_healthy(worker):
                    logger.info(f"Reusing existing worker instance: {cache_key}")
                    return worker
                else:
                    logger.info(f"Worker {cache_key} unhealthy, recreating...")
                    await self._cleanup_worker(cache_key)

            # Create new worker
            logger.info(f"Creating new worker instance: {cache_key}")
            worker = await self._create_worker_instance(
                worker_name,
                worker_data,
                tenant_id=tenant_id,
                byo_mode=byo_mode,
                work_tier=work_tier,
                product=product,
                key_entitled=key_entitled,
            )
            self._workers[cache_key] = worker

            # Start MCP servers if this is a PydanticAI agent with MCP servers
            if (
                hasattr(worker, "config")
                and hasattr(worker.config, "mcp_servers")
                and worker.config.mcp_servers
            ):
                logger.info(f"Initializing MCP servers for worker {worker_name}")
                # The agent will initialize its MCP servers automatically
                await asyncio.sleep(2)  # Give some time for initialization

                # Verify MCP servers are ready
                if hasattr(worker, "mcp_servers_ready") and not worker.mcp_servers_ready:
                    logger.warning(
                        f"MCP servers not ready for worker {worker_name}, but proceeding"
                    )

            return worker

    async def _is_worker_healthy(self, worker: Any) -> bool:
        """Check if worker is still healthy and responsive.

        Performs health checks on worker instances to determine if they can be reused
        or need to be recreated. For PydanticAI agents with MCP servers, also checks
        MCP server readiness.

        Args:
            worker: Worker instance to check health for

        Returns:
            True if worker is healthy and can be reused, False otherwise

        Note:
            Returns True by default if no specific health check method is available.
            Logs warnings for failed health checks but doesn't raise exceptions.
        """
        try:
            # Add health check logic here
            if hasattr(worker, "check_health"):
                return bool(await worker.check_health())

            # For PydanticAI agents with MCP servers, check MCP server status
            if (
                hasattr(worker, "config")
                and hasattr(worker.config, "mcp_servers")
                and worker.config.mcp_servers
            ) and hasattr(worker, "mcp_servers_ready"):
                return bool(worker.mcp_servers_ready)

            return True  # Basic assumption if no health check
        except Exception as e:
            logger.warning(f"Worker health check failed: {e}")
            return False

    async def _create_worker_instance(
        self,
        worker_name: str,
        worker_data: dict[str, Any],
        *,
        tenant_id: str | None = None,
        byo_mode: str = "local_allowed",
        work_tier: str | None = None,
        product: str | None = None,
        key_entitled: bool = False,
    ) -> Any:
        """Create a new worker instance using the worker factory.

        Normalizes the worker name and passes all configuration to the worker factory
        for instantiation. Supports various worker types including PydanticAI agents.

        Args:
            worker_name: Name of the worker (will be normalized)
            worker_data: Dictionary containing:
                - worker_name: Original worker name
                - config: Full worker configuration including:
                    - tool_names: List of tools the worker can use
                    - agent_role: Role description for agent workers
                    - agent_type: Type of agent (e.g., 'pydantic_ai')
                    - model_provider: LLM provider (openai, deepseek, etc.)
                    - model: Specific model to use
                    - temperature: Model temperature setting
                    - system_prompt: System prompt for the agent
                    - instruction: Additional instructions
                    - user_prompt: Default user prompt
                    - description: Worker description
                    - deps: Python dependencies
                    - mcp_config: MCP configuration
                    - mcp_servers: List of MCP servers to enable
                    - base_url_override: Custom API base URL
                    - api_key_override: Custom API key
                    - tools_dir_override: Custom tools directory

        Returns:
            Newly created worker instance

        Raises:
            Exception: If worker creation fails
        """
        normalized_name = worker_data["worker_name"].lower().replace(" ", "_")

        # Normalize config to a dictionary
        config = worker_data.get("config") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                logger.warning(
                    "Worker %s has non-JSON config; defaulting to empty config",
                    worker_name,
                )
                config = {}

        # BYO LLM: overlay the tenant's provider/key onto this worker's config
        # before instantiation, so all worker-based apps inherit it with no
        # per-app change. May raise BYORequiredError for commercial tenants
        # without a configured provider (caught and surfaced by the endpoint).
        # ``byo_mode`` is already resolved per-(tenant, product) by the endpoint
        # when a product was supplied; log product so the resolution is auditable.
        logger.info(
            "Resolving LLM config: worker=%s tenant=%s product=%s byo_mode=%s work_tier=%s",
            worker_name,
            tenant_id,
            product,
            byo_mode,
            work_tier,
        )
        config = await resolve_llm_config(
            config,
            tenant_id,
            byo_mode=byo_mode,
            work_tier=work_tier,
            key_entitled=key_entitled,
        )
        # Owner ruling 2026-10-01: a customer never pins inference-router to an
        # upstream, never runs a non-`local-llm` IR model, and never reaches a
        # local direct provider (validator B2/N6). Platform tenants untouched.
        # N4: the routing belt's owner-infra check resolves DNS; do it off the loop.
        if not is_commercial_platform_tenant(tenant_id):
            await prewarm_owner_infra_dns(config.get("base_url_override"))
        config = enforce_customer_routing(config, tenant_id)

        # Provide safe defaults to avoid KeyErrors / type errors
        tool_names = config.get("tool_names", [])
        agent_role = config.get("agent_role")
        agent_type = config.get("agent_type")
        model_provider = config.get("model_provider")
        model = config.get("model")
        temperature = config.get("temperature")
        max_tokens = config.get("max_tokens")
        # Per-message chain-of-thought switch (see MessageRequest.enable_reasoning).
        # None ⇒ absent from the payload ⇒ unchanged behaviour; it only becomes a
        # wire field for self-hosted backends, in ModelFactory.get_model_settings.
        enable_reasoning = config.get("enable_reasoning")
        # Sampling seed (worker config, or the per-message override folded in by
        # message_endpoints). None ⇒ no seed key reaches the wire.
        seed = config.get("seed")
        # inference-router provider pin (worker config ``ir_provider``) → X-Provider.
        ir_provider = config.get("ir_provider")
        # Repetition controls from the worker config. A per-message value is NOT
        # folded in here: like the per-message seed it is passed per RUN.
        repetition_penalty = config.get("repetition_penalty")
        frequency_penalty = config.get("frequency_penalty")
        system_prompt = config.get("system_prompt")
        instruction = config.get("instruction")
        user_prompt = config.get("user_prompt", "")
        description = config.get("description")
        deps = config.get("deps", [])
        mcp_config = config.get("mcp_config")
        mcp_config_preset = config.get("mcp_config_preset")
        mcp_servers = config.get("mcp_servers")
        base_url_override = config.get("base_url_override")
        api_key_override = config.get("api_key_override")
        # Set only by resolve_llm_config when api_key_override is the tenant's
        # own decrypted BYO key; never read from stored worker config input.
        tenant_byo_key = bool(config.get("tenant_byo_key")) and bool(api_key_override)
        tools_dir_override = config.get("tools_dir_override")
        # Read worker_data first, then the config dict. The comment that used to
        # sit here said these are "persisted as dedicated top-level registry
        # columns" and that config was the legacy fallback. That is BACKWARDS,
        # measured on prod 2026-08-29: the output_schema / output_validator
        # COLUMNS are non-NULL on 0 of 751 rows, while ~140 rows carry an
        # output_schema inside config. No API caller passes those kwargs to
        # register_worker, so nothing ever populates the columns — config is the
        # real storage, not the fallback.
        #
        # ⚠️ The worker_data lookup is still LOAD-BEARING, so do not "simplify" it
        # away: message_endpoints._apply_message_override writes
        # worker_data["output_schema"] IN MEMORY so a per-message schema override
        # wins over the stored config. Deleting this read would silently break
        # that, and no column is involved.
        #
        # The original bug this ordering fixed is real and unchanged: reading
        # solely from config returned None for every registry-backed worker, so
        # the rebuilt pydantic-ai agent got output_type=str → freeform generation
        # → degenerate JSON.
        output_schema = worker_data.get("output_schema")
        if output_schema is None:
            output_schema = config.get("output_schema")
        output_validator = worker_data.get("output_validator")
        if output_validator is None:
            output_validator = config.get("output_validator")
        output_retries = worker_data.get("output_retries")
        if output_retries is None:
            output_retries = config.get("output_retries")

        # WS-1 prompt-replay: the worker's active version (workers.config_revision,
        # a dedicated registry column) is forwarded to the pydantic-ai provider so
        # it can stamp X-Worker-Version on outbound IR calls under replay mode.
        # None for legacy/auto-created rows without a revision — treated as "".
        worker_version = worker_data.get("config_revision")

        worker = await create_worker(
            worker_name=normalized_name,
            tool_names=tool_names,
            agent_role=agent_role,
            agent_type=agent_type,
            model_provider=model_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            seed=seed,
            ir_provider=ir_provider,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            system_prompt=system_prompt,
            instruction=instruction,
            user_prompt=user_prompt,
            description=description,
            deps=deps,
            mcp_config=mcp_config,
            mcp_config_preset=mcp_config_preset,
            mcp_servers=mcp_servers,
            base_url_override=base_url_override,
            api_key_override=api_key_override,
            tenant_byo_key=tenant_byo_key,
            tools_dir_override=tools_dir_override,
            output_schema=output_schema,
            output_validator_code=output_validator,
            output_retries=output_retries,
            worker_version=worker_version,
            # The tenant this instance is being built FOR. It was already
            # resolved above (it is the cache key and drives resolve_llm_config)
            # and was simply never handed to the agent, so every outbound
            # inference-router call from a cached worker went out with
            # `X-Tenant-Id: ""`.
            tenant_id=tenant_id,
            # ⚠️ A `license_token=resolve_outbound_license_token(tenant_id)`
            # kwarg stood here. RETIRED 2026-09-11 together with
            # `outbound_license.py` and the two per-tenant
            # OUTBOUND_LICENSE_TOKEN_* env vars on the workers Deployment
            # (gitops #3825). The resolved value was threaded down two factory
            # layers onto `worker.config.license_token` and then discarded by
            # `_tenant_header_value` -- it reached no wire, so it was a live
            # bearer credential in the pod environment with no consumer. The
            # licence inference-router reads is `license_id` on `X-License-Id`,
            # resolved per-request from the identity ContextVars.
        )

        # Carry the BYO routing decision (set by resolve_llm_config) onto the
        # agent config so run() can emit a token-accurate cost_event. Done here
        # rather than threading through the factory layers; only pydantic_ai
        # workers carry a cost_source-bearing config.
        cost_source = config.get("cost_source")
        if cost_source and getattr(worker, "config", None) is not None:
            try:
                worker.config.cost_source = cost_source
                # Real pricing provider for the cost_event (e.g. pgx-local /
                # deepseek for the local preset); set alongside cost_source so
                # the emitter bills under a provider metering can price.
                cost_provider = config.get("cost_provider")
                if cost_provider:
                    worker.config.cost_provider = cost_provider
                # Managed-BYO provenance: owner-provided key → the run's cost
                # event is emitted marked-up (owner_provided), not force-zero'd.
                worker.config.owner_provided = bool(config.get("owner_provided"))
                # Per-call tenant-key audit (fingerprint of the stored key
                # version, never the key) + the row role it came from.
                worker.config.key_fingerprint = str(config.get("key_fingerprint") or "")
                worker.config.byo_role = str(config.get("byo_role") or "")
            except AttributeError, ValueError:
                # A failed stamp means the run's cost_event is emitted without a
                # priceable provider (NULL cost_usd forever, issue #137) — that must
                # not be invisible in prod, where DEBUG is off. Name both fields.
                logger.warning(
                    "cost_source/cost_provider not settable on %s config "
                    "(cost_source=%s, cost_provider=%s) — cost_event may be unpriceable",
                    normalized_name,
                    cost_source,
                    config.get("cost_provider"),
                )

        return worker

    async def _cleanup_worker(self, worker_name: str) -> None:
        """Clean up a worker instance and remove it from cache.

        Performs graceful cleanup of worker resources, including calling the worker's
        cleanup method if available. Removes the worker from the cache after cleanup.

        Args:
            worker_name: Name of the worker to clean up

        Note:
            Logs warnings for cleanup errors but ensures worker is removed from cache
            even if cleanup fails. This prevents zombie workers from remaining cached.
        """
        if worker_name in self._workers:
            worker = self._workers[worker_name]
            try:
                # Add cleanup logic here
                if hasattr(worker, "cleanup"):
                    logger.info(f"Cleaning up worker {worker_name}")
                    await worker.cleanup()
                    logger.info(f"Worker {worker_name} cleanup completed")
            except Exception as e:
                logger.warning(f"Error cleaning up worker {worker_name}: {e}")
            finally:
                del self._workers[worker_name]

    async def remove_worker(self, worker_name: str, *, tenant_id: str | None = None) -> None:
        """Remove worker from cache with proper locking.

        Thread-safe method to remove a worker from the cache. Uses the worker's lock
        to ensure no concurrent operations are happening during removal.

        Args:
            worker_name: Name of the worker to remove from cache
            tenant_id: Tenant owning the instance (composite cache key).

        Note:
            This is the public interface for worker removal. It ensures proper
            synchronization before calling the internal _cleanup_worker method.
        """
        if tenant_id is not None:
            # Evict every tier AND product variant for this (tenant, worker):
            # keys look like ``tenant_id:worker_name:<tier>[:<product>]``, so a
            # tenant-scoped prefix match clears them all.
            prefix = f"{tenant_id}:{worker_name}:"
            keys = [k for k in self._workers if k.startswith(prefix)]
        else:
            # Caller didn't scope by tenant (e.g. config-mutate / delete paths):
            # evict the legacy name-only key AND every tenant/tier instance of
            # this worker name (the worker name is the 2nd colon segment), so a
            # config change can't leave a stale cached agent for any tenant.
            keys = [
                k for k in self._workers if k == worker_name or k.split(":")[1:2] == [worker_name]
            ]
        for cache_key in keys:
            # Unconditionally take the key's lock. The old ``if cache_key in
            # self._worker_locks`` guard silently skipped any slot that had no
            # lock entry; once entries are reaped at zero users that is EVERY
            # slot at rest, so the guard would have made this method evict
            # nothing at all.
            async with self._held_lock(cache_key):
                await self._cleanup_worker(cache_key)

    async def remove_instance(
        self,
        worker_name: str,
        worker_data: dict[str, Any],
        *,
        tenant_id: str | None = None,
        product: str | None = None,
        override_key: str | None = None,
        expected: Any | None = None,
    ) -> None:
        """Evict EXACTLY the one slot ``get_or_create_worker`` would return, nothing else.

        ``remove_worker`` deletes by ``tenant:worker:`` prefix, which is right for a
        registry/config mutation (every variant is stale) and catastrophic for a
        per-request recreate. ``_cleanup_worker`` calls ``worker.cleanup()``, and
        ``PydanticAIAgent.cleanup`` closes the model's ``AsyncOpenAI`` httpx pool —
        so a prefix eviction issued by ONE in-flight request closes the connection
        pool of every SIBLING request currently awaiting ``agent.run()`` on another
        variant of the same worker. Those siblings then raise
        ``APIConnectionError("Connection error.")``, which ``_is_connection_error``
        classifies as transient, which puts them into the retry ladder, where their
        own recreate evicts by prefix again — the runs fail in lockstep for the full
        length of the ladder, and none of the original faults were theirs.

        The key is composed exactly as ``get_or_create_worker`` composes it (same
        ``_cache_key`` call, same ``work_tier`` read out of ``worker_data``), so the
        two cannot drift apart.

        ``expected`` makes this a COMPARE-and-evict: pass the instance the caller
        actually ran (and found broken) and the slot is dropped only while it still
        holds that same object. Scoping by key alone is not enough once several
        concurrent runs share ONE slot: when N of them fail together, the first
        evicts and rebuilds, and each of the other N-1 then evicts the brand-new
        replacement that the first one is already running on. Measured on the
        deployed #209 diagnostics: 534 cleanups across only 10 distinct cache keys
        in 5 minutes, one key evicted 155 times — an instance-churn storm in which
        almost every eviction discarded a healthy instance nobody had complained
        about. With ``expected`` set, a caller can only ever retire the generation
        it saw fail; the losers of the race fall through to
        ``get_or_create_worker`` and simply join the replacement.

        ``expected=None`` keeps the unconditional behaviour for callers that hold no
        instance (config mutation, delete, tenant invalidation).
        """
        config = worker_data.get("config") or {}
        work_tier = config.get("work_tier") if isinstance(config, dict) else None
        slot_tenant = tenant_id or worker_data.get("tenant_id")
        cache_key = _cache_key(
            worker_name,
            slot_tenant,
            work_tier,
            product,
            override_key,
            key_entitled=tenant_key_entitled(slot_tenant),
        )

        def _is_stale_target() -> bool:
            """True when this eviction would hit something other than what the caller ran."""
            if cache_key not in self._workers:
                return True
            return expected is not None and self._workers[cache_key] is not expected

        if _is_stale_target():
            return
        # Always take the key's lock. This used to fall back to an UNLOCKED
        # ``_cleanup_worker`` whenever ``_worker_locks`` had no entry for the
        # key -- a race against a concurrent ``get_or_create_worker`` even
        # before the entries were reaped, and the normal path once they are.
        async with self._held_lock(cache_key):
            # Re-check UNDER the lock. The pre-lock check is only a fast path:
            # acquiring the lock is a suspension point, and the whole point of
            # this method is that concurrent runs contend for one slot, so the
            # winner can have swapped the instance while we waited.
            if _is_stale_target():
                logger.info(
                    "Skipping eviction of %s: slot no longer holds the failed instance",
                    cache_key,
                )
                return
            await self._cleanup_worker(cache_key)

    async def invalidate_tenant(self, tenant_id: str) -> None:
        """Drop all cached instances for a tenant (best-effort, after a config change).

        Called when a tenant's BYO LLM config changes so the next run rebuilds
        the agent against the new provider/key instead of reusing a stale one.
        """
        prefix = f"{tenant_id}:"
        for cache_key in [k for k in self._workers if k.startswith(prefix)]:
            # Always locked, for the same reason as ``remove_instance``: the
            # ``.get()``-or-unlocked shape raced ``get_or_create_worker``, and
            # with reaped entries the unlocked branch would be the only one
            # ever taken.
            async with self._held_lock(cache_key):
                await self._cleanup_worker(cache_key)


# Global cache instance
worker_cache = WorkerInstanceCache()

# Initialize worker registry
try:
    worker_registry = WorkerRegistryDB()
except Exception as e:
    logger.exception("Failed to initialize worker registry")
    raise RuntimeError(f"Worker registry initialization failed: {e}") from e

# Global state manager instance
state_manager = WorkerStateManager()
