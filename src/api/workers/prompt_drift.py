"""Drift between a bound worker's config prompt and its PUBLISHED version (U6, #314).

With ``WM_PROMPT_SOURCE=published`` a bound PLATFORM worker runs the highest
PUBLISHED version of its prompt and IGNORES ``workers.config.system_prompt``, by
design (``prompt_source.py``). The n8n ``createWorker`` ensure-nodes keep
rewriting that config text, so the two can silently diverge: an ensure-node
edit that used to change what ran stops doing so, and nothing says it stopped.

This module measures that divergence, PERIODICALLY, off the message path:

* ``wm_prompt_binding_drift{tenant_id, worker_name}`` — 1 when
  ``sha256(config.system_prompt)`` differs from the sha of the bound prompt's
  highest PUBLISHED version, 0 when equal. One series per RESOLVED binding.
* ``wm_prompt_binding_drift_total`` — how many resolved bindings drift.
* ``wm_prompt_binding_unresolved_total`` — bindings the runtime could NOT
  resolve: the prompt has no PUBLISHED version, the worker row is missing, or
  the prompt is owned by ANOTHER tenant (or none; migration 035 — the resolver
  refuses it, ``prompt_source.owner_matches``).
  The runtime runs the config prompt for those (fail-safe), so they carry no
  drift series — but a non-zero count is itself something to look at.
* ``wm_prompt_binding_checked_total`` — bindings evaluated by the last refresh,
  so "0 drift" is distinguishable from "0 bindings".
* ``wm_prompt_drift_last_success_timestamp_seconds`` — when the last refresh
  succeeded; a stale value means the numbers above are stale.
* ``wm_prompt_drift_refresh_errors_total`` — refreshes that FAILED.

⚠️ IT RUNS WHILE THE FLAG IS STILL ``legacy``. That is the point: the flip must
be observable BEFORE it happens, so the gauge is computed from the tables, never
from what a call resolved. It reads no setting except its own interval.

SAME COMPARISON AS THE RUNTIME. "Highest PUBLISHED" is selected exactly as
``prompt_repository.get_active(highest_published=True)`` selects it (highest
version among ``status IN ('PUBLISHED', 'active')``), and a winner that is not
``PUBLISHED`` counts as unresolved — the resolver returns ``None`` for it too.
Both digests are :func:`src.api.prompts.hashing.sha256_text`, the one the
resolver stamps on ``ResolvedPrompt.sha256``. No normalisation: a whitespace
difference IS drift, because the runtime would send different bytes.

CUSTOMER TENANTS ARE SKIPPED. A customer binding is never resolved
(``prompt_source.resolve_worker_prompt`` gates on :func:`is_platform_tenant`
before any read), so its "drift" changes nothing that runs; it gets no series.

CARDINALITY. Bindings are platform-only (~70 rows on 2026-10-01, written by
``scripts/adopt_live_worker_prompts.py``), so one series per (tenant_id,
worker_name) is bounded by that table, not by request traffic. Labels that
disappear (a binding removed, or no longer resolvable) are REMOVED on the next
refresh, so a stale 1 can never linger.

FAIL-SAFE. A refresh error (pool down, query error) is logged at WARNING once
per process, counted on every occurrence, and the PREVIOUS values are kept —
never raised, never a crash. A database without ``worker_prompt_binding``
(migration 034 not applied) is 0 bindings, not an error.

Every pod computes the same values from the same tables, so aggregate with
``max()`` across pods, never ``sum()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter, Gauge

from src.api.prompts import repository as prompt_repository
from src.api.prompts.hashing import sha256_text
from src.api.tenant_constants import is_platform_tenant
from src.api.workers.prompt_source import owner_matches
from src.db.db_config import get_pg_connection
from src.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "DriftSnapshot",
    "compute_drift",
    "drift_loop",
    "read_bindings",
    "refresh_once",
    "start_drift_task",
]

DRIFT = Gauge(
    "wm_prompt_binding_drift",
    "1 when sha256(workers.config.system_prompt) differs from the sha of the bound "
    "prompt's highest PUBLISHED version, 0 when equal. One series per resolved "
    "platform binding (worker_prompt_binding, ~70 rows); removed when the binding is.",
    ["tenant_id", "worker_name"],
)
DRIFT_TOTAL = Gauge(
    "wm_prompt_binding_drift_total",
    "Resolved platform prompt bindings whose config.system_prompt differs from the "
    "bound PUBLISHED version (as of the last successful refresh).",
)
UNRESOLVED_TOTAL = Gauge(
    "wm_prompt_binding_unresolved_total",
    "Platform prompt bindings the runtime cannot resolve: the prompt has no PUBLISHED "
    "version, or the bound worker row is missing (as of the last successful refresh).",
)
CHECKED_TOTAL = Gauge(
    "wm_prompt_binding_checked_total",
    "Platform prompt bindings evaluated by the last successful drift refresh.",
)
LAST_SUCCESS = Gauge(
    "wm_prompt_drift_last_success_timestamp_seconds",
    "Unix time of the last successful prompt-binding drift refresh.",
)
REFRESH_ERRORS = Counter(
    "wm_prompt_drift_refresh_errors_total",
    "Prompt-binding drift refreshes that FAILED; the previous gauge values were kept.",
)

# One read, no parameters. ``to_regclass`` first, so a database without 034
# answers "no bindings" instead of an UndefinedTable error every interval.
_TABLE_SQL = "SELECT to_regclass('worker_prompt_binding') IS NOT NULL"

# binding -> owner (LEFT: an owner that is not the binding's tenant, or none,
# is "unresolved" — the resolver refuses it) ->
# worker (LEFT: a missing worker is "unresolved", not dropped) ->
# highest PUBLISHED-or-legacy-'active' version (LEFT: none is "unresolved").
# The lateral pick is ``get_active(highest_published=True)``'s WHERE + ORDER BY,
# verbatim; see the module docstring. ``workers.tenant_id`` is a uuid and the
# binding's a TEXT, compared as text: a binding tenant that is not a canonical
# lowercase uuid can never match — and the resolver, which looks the binding up
# by the lowercased tenant, could never resolve it either.
_BINDINGS_SQL = """
SELECT b.tenant_id,
       b.worker_name,
       b.prompt_id,
       o.owner_tenant_id                  AS owner_tenant_id,
       (w.worker_name IS NOT NULL)        AS worker_found,
       w.config ->> 'system_prompt'       AS config_system,
       p.version                          AS published_version,
       p.status                           AS published_status,
       p.system                           AS published_system
FROM worker_prompt_binding b
LEFT JOIN prompt_owner o
       ON o.prompt_id = b.prompt_id
LEFT JOIN workers w
       ON w.tenant_id::text = b.tenant_id
      AND w.worker_name = b.worker_name
LEFT JOIN LATERAL (
    SELECT pv.version, pv.status, pv.system
    FROM prompt_versions pv
    WHERE pv.prompt_id = b.prompt_id
      AND pv.status IN ('PUBLISHED', 'active')
    ORDER BY pv.version DESC
    LIMIT 1
) p ON TRUE
ORDER BY b.tenant_id, b.worker_name
"""


@dataclass(frozen=True)
class DriftSnapshot:
    """One refresh's verdict. ``drift`` holds RESOLVED bindings only."""

    drift: dict[tuple[str, str], int]
    unresolved: int
    checked: int

    @property
    def drift_total(self) -> int:
        """How many resolved bindings currently drift (value of the total gauge)."""
        return sum(self.drift.values())


# The label sets exported by the last successful refresh: the next refresh
# removes any that are gone, so a removed binding's series disappears with it.
_exported: set[tuple[str, str]] = set()
_error_logged = False


def _reset_for_tests() -> None:
    """Drop every exported series and re-arm the once-per-process error log."""
    global _error_logged  # noqa: PLW0603 - process-wide log latch, reset by tests
    DRIFT.clear()
    _exported.clear()
    _error_logged = False


async def read_bindings() -> list[dict[str, Any]]:
    """Every binding row joined to its worker and highest PUBLISHED version.

    Read-only. ``[]`` when ``worker_prompt_binding`` does not exist (pre-034).
    """
    async with get_pg_connection() as conn:
        if not await conn.fetchval(_TABLE_SQL):
            return []
        rows = await conn.fetch(_BINDINGS_SQL)
    return [dict(r) for r in rows]


def compute_drift(rows: list[dict[str, Any]]) -> DriftSnapshot:
    """Pure: binding rows -> per-binding 0/1 drift plus the unresolved count."""
    drift: dict[tuple[str, str], int] = {}
    unresolved = 0
    checked = 0
    for row in rows:
        tenant = str(row["tenant_id"])
        if not is_platform_tenant(tenant):
            # Never resolved at runtime, so its config cannot "drift" from
            # anything that runs. Not exported (system data stays platform).
            continue
        checked += 1
        published = row.get("published_system")
        if (
            not owner_matches(row.get("owner_tenant_id"), tenant.strip().lower())
            or not row.get("worker_found")
            or row.get("published_status") != prompt_repository.PUBLISHED
            or not isinstance(published, str)
        ):
            # The resolver answers None for each of these and the worker keeps
            # its config prompt: nothing has diverged, but nothing is bound either.
            unresolved += 1
            continue
        config = row.get("config_system")
        # A config with no system_prompt at all still diverges: legacy runs no
        # prompt text, published runs the version's text.
        same = isinstance(config, str) and sha256_text(config) == sha256_text(published)
        drift[(tenant, str(row["worker_name"]))] = 0 if same else 1
    return DriftSnapshot(drift=drift, unresolved=unresolved, checked=checked)


def _publish(snapshot: DriftSnapshot) -> None:
    """Write ``snapshot`` to the gauges, removing label sets that disappeared."""
    for labels in _exported - set(snapshot.drift):
        DRIFT.remove(*labels)
    for (tenant, worker), value in snapshot.drift.items():
        DRIFT.labels(tenant_id=tenant, worker_name=worker).set(value)
    _exported.clear()
    _exported.update(snapshot.drift.keys())
    DRIFT_TOTAL.set(snapshot.drift_total)
    UNRESOLVED_TOTAL.set(snapshot.unresolved)
    CHECKED_TOTAL.set(snapshot.checked)
    LAST_SUCCESS.set(time.time())


async def refresh_once() -> DriftSnapshot | None:
    """Read, compare, export. ``None`` on error (previous values kept). Never raises."""
    global _error_logged  # noqa: PLW0603 - once-per-process log latch
    try:
        snapshot = compute_drift(await read_bindings())
    except Exception:
        REFRESH_ERRORS.inc()
        if not _error_logged:
            _error_logged = True
            logger.warning(
                "prompt-binding drift refresh failed; keeping the previous values. "
                "Further failures are counted in wm_prompt_drift_refresh_errors_total, "
                "not logged.",
                exc_info=True,
            )
        return None
    _publish(snapshot)
    return snapshot


async def drift_loop(interval_seconds: float) -> None:
    """Refresh forever, the FIRST refresh before the first sleep.

    Same ordering as ``worker_states_retention.retention_loop``: a pod that rolls
    more often than the interval must still export a value. ``refresh_once``
    never raises, so the only exit is cancellation at shutdown.
    """
    while True:
        await refresh_once()
        await asyncio.sleep(interval_seconds)


def start_drift_task() -> asyncio.Task[None] | None:
    """Start :func:`drift_loop` as a background task, or ``None`` when disabled."""
    interval = get_settings().wm_prompt_drift_interval_s
    if interval <= 0:
        logger.info("prompt-binding drift metric is disabled (WM_PROMPT_DRIFT_INTERVAL_S=0)")
        return None
    task = asyncio.create_task(drift_loop(interval))
    logger.info("prompt-binding drift metric started: every %ss", interval)
    return task
