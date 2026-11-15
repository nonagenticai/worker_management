"""Retention for ``tenant_llm_config_audit`` (migration 037) — WM-5.

The customer-visible BYO-config history had no retention: every PUT / DELETE /
TEST appends a row and nothing ever removed one. Volume is low (a few rows per
tenant per config change), so this is a slow, bounded prune, not an urgent one:
rows older than ``tenant_llm_audit_retention_days`` (default 400 — a year of
history plus a quarter of slack, so a customer reviewing "the last 12 months"
always sees all of it) are deleted in batches, once a day.

Scheduling reuses :func:`src.services.worker_states_retention.retention_loop`
(work BEFORE the first sleep, back-off on error), started from the app
lifespan beside the worker_states sweep.

Each batch is ONE explicit transaction whose ``statement_timeout`` is
``SET LOCAL``: it ends with the transaction, so nothing leaks onto the next
client of a transaction-pooling PgBouncer connection.

Cross-tenant by design (there is no RLS in this repo); the only predicate is
age, which is tenant-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional

from src.db.db_config import get_pg_connection
from src.settings import get_settings

logger = logging.getLogger(__name__)

_BATCH_STATEMENT_TIMEOUT_MS = 30_000

_DELETE_BATCH_SQL = """
WITH doomed AS (
    SELECT id FROM tenant_llm_config_audit
     WHERE created_at < $1
     ORDER BY id
     LIMIT $2
),
removed AS (
    DELETE FROM tenant_llm_config_audit t
     USING doomed d
     WHERE t.id = d.id
    RETURNING t.id
)
SELECT count(*) AS deleted FROM removed
"""


@dataclass(frozen=True)
class AuditPruneResult:
    """What one prune pass did. ``exhausted`` = no expired rows remain."""

    deleted: int
    batches: int
    exhausted: bool


async def prune_audit_once(
    *,
    retention_days: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batches: Optional[int] = None,
    now: Optional[datetime] = None,
) -> AuditPruneResult:
    """Delete audit rows older than the window, in bounded batches."""
    s = get_settings()
    days = retention_days if retention_days is not None else s.tenant_llm_audit_retention_days
    if days < 1:
        raise ValueError("tenant_llm_audit_retention_days must be >= 1")
    size = batch_size or s.tenant_llm_audit_retention_batch_size
    limit = max_batches or s.tenant_llm_audit_retention_max_batches_per_pass
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)

    deleted = 0
    batches = 0
    exhausted = False
    for _ in range(limit):
        async with get_pg_connection() as conn, conn.transaction():
            await conn.execute(f"SET LOCAL statement_timeout = {_BATCH_STATEMENT_TIMEOUT_MS}")
            row = await conn.fetchrow(_DELETE_BATCH_SQL, cutoff, size)
        batches += 1
        n = int((row or {}).get("deleted") or 0)
        deleted += n
        if n < size:
            exhausted = True
            break
        await asyncio.sleep(0)
    log = logger.info if deleted else logger.debug
    log(
        "tenant_llm_config_audit retention: deleted %d row(s) older than %dd in %d batch(es)%s",
        deleted,
        days,
        batches,
        "" if exhausted else " (backlog remains; next pass continues)",
    )
    return AuditPruneResult(deleted=deleted, batches=batches, exhausted=exhausted)


def start_audit_retention_task() -> Optional[asyncio.Task[None]]:
    """Start the daily prune as a background task, or return None when disabled."""
    from src.services.worker_states_retention import retention_loop

    s = get_settings()
    if not s.tenant_llm_audit_retention_enabled:
        logger.info(
            "tenant_llm_config_audit retention is disabled (TENANT_LLM_AUDIT_RETENTION_ENABLED)"
        )
        return None
    task = asyncio.create_task(
        retention_loop(
            sweep=prune_audit_once,
            interval_seconds=s.tenant_llm_audit_retention_interval_seconds,
            label="tenant_llm_config_audit",
        )
    )
    logger.info(
        "tenant_llm_config_audit retention started: keep %dd, every %ds",
        s.tenant_llm_audit_retention_days,
        s.tenant_llm_audit_retention_interval_seconds,
    )
    return task
