"""Which PUBLISHED prompt version a platform worker runs (U6, migration 034).

Until this module, a worker's runtime system prompt was ``workers.config
->>'system_prompt'`` and nothing else. A prompt release (a new PUBLISHED row in
``prompt_versions``) therefore changed nothing that ran, and n8n "ensure" nodes
keep REWRITING that config text — a second writer that would undo any release
made through the config.

With ``WM_PROMPT_SOURCE=published`` a PLATFORM-tenant worker that has a
``worker_prompt_binding`` row runs the HIGHEST PUBLISHED version of its bound
prompt instead, and the config text is ignored for it. Precedence per call (see
``message_endpoints.send_message``)::

    request.system_prompt  >  pin (prompt_pin.py)  >  RESOLVED PUBLISHED  >  config

⛔ CANARY IS NEVER RESOLVED HERE. Only ``get_active(highest_published=True)`` is
read, which matches PUBLISHED rows only. A CANARY runs solely through an explicit
pin, and only when ``WM_PROMPT_CANARY_ROUTABLE`` allows it (``prompt_pin.py``).

⛔ A CUSTOMER TENANT IS NEVER RESOLVED, even when a binding row exists for it —
the tenant gate runs BEFORE any read. System prompts are platform data (owner
ruling 2026-09-30).

⛔ A PROMPT OWNED BY ANOTHER TENANT IS NEVER RESOLVED (migration 035, strict
tenant separation). The binding read joins ``prompt_owner``; when the prompt's
owner is not the binding's tenant (or it has no owner), the worker keeps its
legacy config prompt, ``wm_prompt_binding_cross_tenant_refused_total`` is
incremented and the first refusal per process is logged at WARNING. 035's
composite FK (binding.tenant_id, prompt_id) -> prompt_owner should make this
impossible; this is defense in depth, so a non-zero counter means the FK was
bypassed or dropped.

FAIL-SAFE. Every failure — the table missing (034 not applied), the pool down,
a binding naming a prompt with no PUBLISHED version — answers ``None`` and the
caller keeps the legacy config prompt. Never a 500: a broken resolver must not
take message traffic down with it. A read ERROR is logged at WARNING once per
process and counted on every occurrence (``wm_prompt_resolve_errors_total``), so
a standing failure is visible without flooding the log.

CACHE. Resolutions (including "no binding") are cached per pod for
``WM_PROMPT_RESOLVE_TTL_S`` (default 5 s), keyed ``(tenant_id, worker_name)``. A
release therefore reaches every pod within that bound — "picked up on the next
call" anywhere in this package means the next call AFTER that TTL, i.e. within
``WM_PROMPT_RESOLVE_TTL_S`` (default 5 s) of the release — and a hot worker costs
one read per TTL rather than one per message. Errors are NOT cached: the next
call retries. ``worker_name`` is caller-controlled, so the cache is BOUNDED at
:data:`CACHE_MAX_ENTRIES` with oldest-first eviction: a flood of distinct names
costs extra reads, never unbounded memory. :func:`clear_cache` is for tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter

from src.api.prompts import repository as prompt_repository
from src.api.prompts.hashing import format_prompt_revision, sha256_text
from src.api.tenant_constants import is_platform_tenant
from src.db.db_config import get_pg_connection
from src.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "ResolvedPrompt",
    "apply_resolved_prompt",
    "clear_cache",
    "count_config_synced",
    "resolve_worker_prompt",
    "sync_bound_worker_configs",
]

# The WM_PROMPT_SOURCE value that turns resolution on.
_FLAG_PUBLISHED = "published"

RESOLVE_ERRORS = Counter(
    "wm_prompt_resolve_errors_total",
    "worker -> published prompt resolutions that FAILED and fell back to the "
    "legacy config prompt (WM_PROMPT_SOURCE=published only).",
)

CROSS_TENANT_REFUSED = Counter(
    "wm_prompt_binding_cross_tenant_refused_total",
    "worker -> published prompt resolutions REFUSED because the bound prompt is "
    "owned by another tenant (or by none); the legacy config prompt ran. Should "
    "stay 0: migration 035's FK forbids such a binding.",
)

CONFIG_SYNCED = Counter(
    "wm_prompt_config_synced_total",
    "Bound platform workers whose workers.config.system_prompt was rewritten to "
    "the resolved PUBLISHED text by a COMMITTED prompt-version transition "
    "(the config mirror; WM_PROMPT_SOURCE=published only).",
    ["prompt_id"],
)

# LEFT JOIN: a binding whose prompt has no owner row still comes back (owner
# NULL) and is refused below, rather than silently reading as "no binding".
_BINDING_SQL = (
    "SELECT b.prompt_id, o.owner_tenant_id "
    "FROM worker_prompt_binding b "
    "LEFT JOIN prompt_owner o ON o.prompt_id = b.prompt_id "
    "WHERE b.tenant_id = $1 AND b.worker_name = $2"
)


@dataclass(frozen=True)
class ResolvedPrompt:
    """The PUBLISHED version a bound worker runs for this call."""

    prompt_id: str
    version: int
    system: str
    sha256: str  # sha256 hex of ``system`` — the join key with system_prompt_sha256
    status: str

    @property
    def revision(self) -> str:
        """``<prompt_id>@<N>`` — reported as ``prompt_version_id``."""
        return format_prompt_revision(self.prompt_id, self.version)


# Upper bound on cached resolutions per process. ``worker_name`` comes from the
# request path, so without a cap the cache grows with every distinct name sent.
CACHE_MAX_ENTRIES = 4096

# (tenant_id, worker_name) -> (monotonic expiry, resolution or None). Insertion
# order is eviction order: a (re)write moves the key to the end, and the oldest
# entry is dropped first once the cap is reached.
_cache: dict[tuple[str, str], tuple[float, ResolvedPrompt | None]] = {}
_error_logged = False
_cross_tenant_logged = False


def clear_cache() -> None:
    """Drop every cached resolution and re-arm the once-per-process logs."""
    global _error_logged, _cross_tenant_logged  # noqa: PLW0603 - process-wide log latches
    _cache.clear()
    _error_logged = False
    _cross_tenant_logged = False


def owner_matches(owner_tenant_id: Any, tenant: str) -> bool:
    """True only when the prompt's owner IS the binding's tenant (035)."""
    return owner_tenant_id is not None and _tenant_key(owner_tenant_id) == tenant


def _refuse_cross_tenant(tenant: str, worker_name: str, prompt_id: str, owner: Any) -> None:
    global _cross_tenant_logged  # noqa: PLW0603 - once-per-process log latch
    CROSS_TENANT_REFUSED.inc()
    if not _cross_tenant_logged:
        _cross_tenant_logged = True
        logger.warning(
            "worker '%s' (tenant %s) is bound to prompt '%s', owned by tenant %s; "
            "REFUSED (strict tenant separation), running the legacy config prompt. "
            "Further refusals are counted in wm_prompt_binding_cross_tenant_refused_total.",
            worker_name,
            tenant,
            prompt_id,
            owner,
        )


def _cache_put(key: tuple[str, str], expires: float, resolved: ResolvedPrompt | None) -> None:
    """Insert ``key`` as the newest entry, evicting oldest-first beyond the cap."""
    _cache.pop(key, None)
    while len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (expires, resolved)


def _tenant_key(tenant_id: Any) -> str:
    return str(tenant_id).strip().lower() if tenant_id is not None else ""


async def _read(tenant: str, worker_name: str) -> ResolvedPrompt | None:
    """Binding -> highest PUBLISHED version, or ``None`` when either is absent."""
    async with get_pg_connection() as conn:
        return await _read_on(conn, tenant, worker_name)


async def _read_on(
    conn: Any, tenant: str, worker_name: str, *, report: bool = True
) -> ResolvedPrompt | None:
    """:func:`_read` on the caller's connection.

    THE resolution, used by both the runtime and the config mirror
    (:func:`sync_bound_worker_configs`).

    ``report=False`` (the mirror) skips the refusal counter and the logs: the
    mirror re-resolves inside a write transaction that may still roll back, and
    an unresolvable binding there simply leaves the config untouched.
    """
    binding = await conn.fetchrow(_BINDING_SQL, tenant, worker_name)
    if binding is None:
        return None
    prompt_id = str(binding["prompt_id"])
    owner = binding["owner_tenant_id"]
    if not owner_matches(owner, tenant):
        if report:
            _refuse_cross_tenant(tenant, worker_name, prompt_id, owner)
        return None
    row = await prompt_repository.get_active_on(conn, prompt_id, highest_published=True)
    if row is None:
        # A binding whose prompt has nothing PUBLISHED: integrity is checked here,
        # not by an FK (034 header). The worker keeps its config prompt.
        if report:
            logger.warning(
                "worker '%s' (tenant %s) is bound to prompt '%s', which has no PUBLISHED "
                "version; running the legacy config prompt",
                worker_name,
                tenant,
                prompt_id,
            )
        return None
    status = str(row["status"])
    if status != prompt_repository.PUBLISHED:  # defensive: get_active reads PUBLISHED only
        return None
    system = row["system"]
    if not isinstance(system, str):
        return None
    return ResolvedPrompt(
        prompt_id=prompt_id,
        version=int(row["version"]),
        system=system,
        sha256=sha256_text(system),
        status=status,
    )


async def resolve_worker_prompt(tenant_id: Any, worker_name: str) -> ResolvedPrompt | None:
    """The PUBLISHED prompt this worker runs, or ``None`` for the legacy config prompt.

    ``None`` unless ALL hold: ``WM_PROMPT_SOURCE=published``, ``tenant_id`` is a
    platform tenant, a binding exists, its prompt is OWNED by ``tenant_id``
    (035), and it has a PUBLISHED version.
    The first two are checked before any read or cache lookup. Never raises.
    """
    settings = get_settings()
    if settings.wm_prompt_source != _FLAG_PUBLISHED:
        return None
    tenant = _tenant_key(tenant_id)
    if not is_platform_tenant(tenant):
        return None

    key = (tenant, worker_name)
    ttl = settings.wm_prompt_resolve_ttl_s
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]

    try:
        resolved = await _read(tenant, worker_name)
    except Exception:
        global _error_logged  # noqa: PLW0603 - once-per-process log latch
        RESOLVE_ERRORS.inc()
        if not _error_logged:
            _error_logged = True
            logger.warning(
                "published prompt resolution failed for worker '%s' (tenant %s); "
                "falling back to the legacy config prompt. Further failures are "
                "counted in wm_prompt_resolve_errors_total, not logged.",
                worker_name,
                tenant,
                exc_info=True,
            )
        return None

    if ttl > 0:
        _cache_put(key, now + ttl, resolved)
    return resolved


def apply_resolved_prompt(worker_data: dict[str, Any], resolved: ResolvedPrompt) -> None:
    """Fold the resolved version into THIS request's ``worker_data``.

    The config's ``system_prompt`` — whatever the n8n ensure-node last wrote
    there — is REPLACED, and ``config_revision`` becomes ``<prompt_id>@<N>`` so
    the A3 label check and the attribution echo name the version that runs.
    ``config`` is COPIED first (same rule as ``prompt_pin.apply_prompt_pin``):
    the registry's dict is never mutated in place. The caller adds the revision
    to the worker-cache override key, so the version gets its own slot.
    """
    config = worker_data.get("config")
    config = dict(config) if isinstance(config, dict) else {}
    config["system_prompt"] = resolved.system
    worker_data["config"] = config
    worker_data["config_revision"] = resolved.revision


# ---------------------------------------------------------------------------
# The config MIRROR (U6, LEDGER C2-32)
# ---------------------------------------------------------------------------
# Without it every legitimate publish left ``workers.config.system_prompt`` on
# the old text: ``wm_prompt_binding_drift_total`` fired on normal use (blaming a
# config edit), and reverting the flag to ``legacy`` would have put bound
# workers back on stale text. With it, config.system_prompt of a bound,
# resolvable platform worker IS the resolved published text after every
# transition that can change it, written in the SAME transaction as the status
# move — so drift > 0 means only "config was edited after the last sync", and a
# revert to legacy runs the bytes that were running.

# Serialises every mirror of ONE prompt. Taken AFTER the transition's own row
# locks and BEFORE the resolve: a second transaction on the same prompt (even on
# a different version, which the row lock does not cover) waits here until the
# first commits, and its resolve — a fresh READ COMMITTED snapshot — then sees
# both moves. Without it two interleaved moves each resolve from their own
# snapshot and the later committer can write a text that is no longer the
# resolved winner.
_SYNC_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('wm_prompt_config_sync:' || $1, 0))"

_SYNC_TABLES_SQL = (
    "SELECT to_regclass('worker_prompt_binding') IS NOT NULL "
    "AND to_regclass('prompt_owner') IS NOT NULL"
)

_BOUND_TO_PROMPT_SQL = (
    "SELECT tenant_id, worker_name FROM worker_prompt_binding "
    "WHERE prompt_id = $1 ORDER BY tenant_id, worker_name"
)

# jsonb_set on the existing object: every other config key is preserved. Only a
# row whose text DIFFERS is written (IS DISTINCT FROM also covers a missing key).
_MIRROR_SQL = (
    "UPDATE workers SET "
    "config = jsonb_set(COALESCE(config, '{}'::jsonb), '{system_prompt}', to_jsonb($1::text)), "
    "updated_at = now() "
    "WHERE tenant_id::text = $2 AND worker_name = $3 "
    "AND (config ->> 'system_prompt') IS DISTINCT FROM $1 "
    "RETURNING worker_name"
)


async def sync_bound_worker_configs(conn: Any, prompt_id: str) -> int:
    """Mirror the resolved text of ``prompt_id`` into its bound workers' config.

    Runs on ``conn`` inside the caller's open transaction (a transition), so the
    mirror commits or rolls back WITH the status move. Returns how many worker
    rows were rewritten; the caller counts them in ``wm_prompt_config_synced_total``
    after it commits (:func:`count_config_synced`).

    The text is NOT the version just transitioned: each bound worker is
    re-resolved through :func:`_read_on` — the runtime's own resolution — so an
    archive falls back to the next-highest PUBLISHED, and nothing PUBLISHED left
    (or a cross-tenant / ownerless binding) leaves the config untouched.

    Exactly the resolver's gates: nothing unless ``WM_PROMPT_SOURCE=published``
    (in ``legacy`` the config IS what runs, and a publish must not become a
    deploy), and a CUSTOMER tenant is never touched. An unbound worker has no
    binding row and is never selected.
    """
    if get_settings().wm_prompt_source != _FLAG_PUBLISHED:
        return 0
    if not await conn.fetchval(_SYNC_TABLES_SQL):
        return 0
    await conn.execute(_SYNC_LOCK_SQL, prompt_id)
    synced = 0
    for binding in await conn.fetch(_BOUND_TO_PROMPT_SQL, prompt_id):
        tenant = _tenant_key(binding["tenant_id"])
        if not is_platform_tenant(tenant):
            continue
        worker_name = str(binding["worker_name"])
        resolved = await _read_on(conn, tenant, worker_name, report=False)
        if resolved is None or resolved.prompt_id != prompt_id:
            continue
        rows = await conn.fetch(_MIRROR_SQL, resolved.system, tenant, worker_name)
        synced += len(rows)
    return synced


def count_config_synced(prompt_id: str, synced: int) -> None:
    """Count ``synced`` mirrored workers — call only AFTER the transition committed."""
    if synced > 0:
        CONFIG_SYNCED.labels(prompt_id=prompt_id).inc(synced)
