"""Repository for ``workflow_routing_pointer``.

Plain psycopg 3 queries against the existing ``AsyncConnectionPool`` (no
SQLAlchemy ORM, no Alembic — mirrors the rest of the worker_management data
layer; see ``src/db/db_config.py``).

Since migration 036 (G-15) the table is keyed by ``asset_key`` —
``'<map>:<name>'``, the DISPATCH key (``activity:AA Activity - Reasoning``,
``cell:doing``) — and every workflow id is a 16-char n8n id held as TEXT. A row
carries the ``active_workflow_id`` dispatch runs, a ``prior_workflow_id`` for
rollback, and at most ONE armed canary (``canary_workflow_id`` / ``canary_pct``
/ ``canary_id`` / ``released_by_card_id``, all-or-none by CHECK). The routing
DECISION is not made here: the ncn resolver reads the row and buckets the card
(G-15 D2/D4 — one place decides).

Every write is ONE statement, so a refusal (409) and the write it refuses are
decided against the same row version. Where a refusal needs a reason, the
statement is a CTE whose second arm reads the row only when the UPDATE matched
nothing; the caller maps ``outcome`` to an HTTP status.

Every method below operates inside the connection wrapper provided by
``get_pg_connection`` so it composes with the existing pool, autocommit, and
``Jsonb()`` adapter behavior the rest of the codebase depends on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.db.db_config import get_pg_connection

# Outcomes a write returns alongside the row (mapped to HTTP in routes.py).
OK = "ok"
UNCHANGED = "unchanged"  # idempotent replay: nothing written
NOT_FOUND = "not_found"
ACTIVE_CONFLICT = "active_conflict"  # seed: a row exists with a different active
CANARY_ARMED = "canary_armed"
CANARY_IS_ACTIVE = "canary_is_active"
NO_CANARY = "no_canary"
CANARY_ID_MISMATCH = "canary_id_mismatch"

_COLUMNS = (
    "asset_key, card_type, list, active_workflow_id, prior_workflow_id, "
    "promoted_at, observation_window_until, rollback_thresholds, "
    "canary_workflow_id, canary_pct, canary_id, released_by_card_id, "
    "released_by_tenant_id, trace_id, promoted_trace_id, released_at"
)
# The same list qualified by the alias ``p`` (for the CTEs' read-back arm).
_P_COLUMNS = ", ".join(f"p.{c.strip()}" for c in _COLUMNS.split(","))

# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------


def _row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Coerce a psycopg dict_row result into the JSON shape the API returns.

    Ids are TEXT since 036; ``datetime`` values are stringified so FastAPI's
    default JSON encoder can serialize without custom encoders. Returns
    ``None`` unchanged so callers can branch on "not found".
    """
    if row is None:
        return None
    return {
        "asset_key": row["asset_key"],
        "card_type": row.get("card_type"),
        "list": row.get("list"),
        "active_workflow_id": _str_or_none(row["active_workflow_id"]),
        "prior_workflow_id": _str_or_none(row.get("prior_workflow_id")),
        "promoted_at": _stringify_dt(row.get("promoted_at")),
        "observation_window_until": _stringify_dt(row.get("observation_window_until")),
        "rollback_thresholds": row.get("rollback_thresholds") or {},
        "canary_workflow_id": _str_or_none(row.get("canary_workflow_id")),
        "canary_pct": row.get("canary_pct"),
        "canary_id": row.get("canary_id"),
        "released_by_card_id": row.get("released_by_card_id"),
        "released_by_tenant_id": row.get("released_by_tenant_id"),
        "trace_id": row.get("trace_id"),
        "promoted_trace_id": row.get("promoted_trace_id"),
        "released_at": _stringify_dt(row.get("released_at")),
    }


def _outcome(row: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """Split a CTE result carrying an ``outcome`` column into (outcome, row)."""
    if row is None:
        return NOT_FOUND, None
    return str(row["outcome"]), _row_to_dict(row)


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _stringify_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


async def get_active_by_asset_key(asset_key: str) -> dict[str, Any] | None:
    """Return the pointer row for ``asset_key`` or ``None`` (→ 404).

    Read by the ncn resolver ``r.workflow.resolve`` on dispatch; a 404 makes it
    fall back to the env map (G-15 D4, phase-10 condition A).
    """
    sql = f"SELECT {_COLUMNS} FROM workflow_routing_pointer WHERE asset_key = $1"
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key)
    return _row_to_dict(row)


async def list_observed() -> list[dict[str, Any]]:
    """Return every pointer row inside a legacy observation window.

    ``prior_workflow_id IS NOT NULL`` as before, AND a non-NULL window: a row
    promoted by ``promote_canary`` has a prior but NO clock (phase-10 decisions
    5/7), so the dormant ``canary-workflow-rollback-watch`` must never see it —
    otherwise it would clear or roll back a release it does not own.
    """
    sql = (
        f"SELECT {_COLUMNS} FROM workflow_routing_pointer "
        "WHERE prior_workflow_id IS NOT NULL "
        "  AND observation_window_until IS NOT NULL "
        "ORDER BY promoted_at"
    )
    async with get_pg_connection() as conn:
        rows = await conn.fetch(sql)
    return [r for r in (_row_to_dict(row) for row in rows) if r is not None]


# ---------------------------------------------------------------------------
# Release writes (G-15 D4) — gated in routes.py BEFORE any of these run
# ---------------------------------------------------------------------------


async def seed_active(
    *, asset_key: str, active_workflow_id: str, tenant_id: str, trace_id: str
) -> tuple[str, dict[str, Any] | None]:
    """Create the pointer for an asset from its CURRENT env-map id.

    Idempotent: an existing row with the SAME active is ``UNCHANGED`` (nothing
    rewritten, its original trace kept); with a DIFFERENT active it is
    ``ACTIVE_CONFLICT`` (→ 409). Not a release: no canary is touched.
    ``NOT_FOUND`` here means a concurrent seed committed after this statement's
    snapshot — the caller answers 409 and the retry sees the row.
    """
    sql = (
        "WITH ins AS ( "
        "    INSERT INTO workflow_routing_pointer "
        "        (asset_key, active_workflow_id, promoted_at, rollback_thresholds, "
        "         released_by_tenant_id, trace_id, released_at) "
        "    VALUES ($1, $2, now(), '{}'::jsonb, $3, $4, now()) "
        "    ON CONFLICT (asset_key) DO NOTHING "
        f"    RETURNING {_COLUMNS} "
        ") "
        f"SELECT '{OK}' AS outcome, ins.* FROM ins "
        "UNION ALL "
        f"SELECT CASE WHEN p.active_workflow_id = $2 THEN '{UNCHANGED}' "
        f"            ELSE '{ACTIVE_CONFLICT}' END, {_P_COLUMNS} "
        "  FROM workflow_routing_pointer p "
        " WHERE p.asset_key = $1 AND NOT EXISTS (SELECT 1 FROM ins)"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key, active_workflow_id, tenant_id, trace_id)
    return _outcome(row)


async def release(
    *,
    asset_key: str,
    canary_workflow_id: str,
    canary_pct: int,
    canary_id: str,
    released_by_card_id: str,
    tenant_id: str,
    trace_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Arm a canary on an existing pointer — the per-asset arming row (decision 2).

    Matches only a row with NO armed canary whose active differs from the
    candidate, so ``CANARY_ARMED`` / ``CANARY_IS_ACTIVE`` (→ 409) and the write
    are decided on one row version. ``NOT_FOUND`` = never seeded (→ 404).
    """
    sql = (
        "WITH upd AS ( "
        "    UPDATE workflow_routing_pointer "
        "       SET canary_workflow_id = $2, canary_pct = $3, canary_id = $4, "
        "           released_by_card_id = $5, released_by_tenant_id = $6, "
        "           trace_id = $7, released_at = now() "
        "     WHERE asset_key = $1 "
        "       AND canary_workflow_id IS NULL "
        "       AND active_workflow_id <> $2 "
        f"    RETURNING {_COLUMNS} "
        ") "
        f"SELECT '{OK}' AS outcome, upd.* FROM upd "
        "UNION ALL "
        f"SELECT CASE WHEN p.canary_workflow_id IS NOT NULL THEN '{CANARY_ARMED}' "
        f"            ELSE '{CANARY_IS_ACTIVE}' END, {_P_COLUMNS} "
        "  FROM workflow_routing_pointer p "
        " WHERE p.asset_key = $1 AND NOT EXISTS (SELECT 1 FROM upd)"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            sql,
            asset_key,
            canary_workflow_id,
            canary_pct,
            canary_id,
            released_by_card_id,
            tenant_id,
            trace_id,
        )
    return _outcome(row)


def _canary_outcome_arm() -> str:
    """Read-back arm shared by promote/discard: why the UPDATE matched nothing."""
    return (
        f"SELECT CASE WHEN p.canary_workflow_id IS NULL THEN '{NO_CANARY}' "
        f"            ELSE '{CANARY_ID_MISMATCH}' END, {_P_COLUMNS} "
        "  FROM workflow_routing_pointer p "
        " WHERE p.asset_key = $1 AND NOT EXISTS (SELECT 1 FROM upd)"
    )


async def promote_canary(
    *, asset_key: str, canary_id: str, card_id: str, trace_id: str | None
) -> tuple[str, dict[str, Any] | None]:
    """active := canary, prior := old active, canary cleared (decision 6).

    ⚠️ Race labels (G-15 audit finding 2): the LOSER of two concurrent calls
    re-reads under a snapshot taken before the winner committed, so its refusal
    label can be misleading (a retried promote of the same canary may report
    ``canary_id_mismatch`` rather than ``no_canary``). The outcome is always a
    single winner; callers must treat ANY refusal as "not promoted by me" and
    re-read the pointer, never branch on the label.

    THE one promoter. Every SET reads the OLD row, so the swap is one statement.

    Matches only the armed canary with this ``canary_id``: a stale card gets
    ``CANARY_ID_MISMATCH``, an unarmed row ``NO_CANARY`` (both → 409). No clock:
    ``observation_window_until`` is NULLed (decisions 5/7).

    Only the card that RELEASED the canary may promote it
    (``released_by_card_id = card_id``; G-15 audit finding 8). Discard is
    deliberately not card-bound: a new release must be able to clear a
    foreign, rejected canary first.
    """
    sql = (
        "WITH upd AS ( "
        "    UPDATE workflow_routing_pointer "
        "       SET prior_workflow_id = active_workflow_id, "
        "           active_workflow_id = canary_workflow_id, "
        "           canary_workflow_id = NULL, canary_pct = NULL, canary_id = NULL, "
        "           released_by_card_id = NULL, "
        "           promoted_at = now(), observation_window_until = NULL, "
        "           promoted_trace_id = $3 "
        "     WHERE asset_key = $1 "
        "       AND canary_workflow_id IS NOT NULL "
        "       AND canary_id = $2 "
        "       AND released_by_card_id = $4 "
        f"    RETURNING {_COLUMNS} "
        ") "
        f"SELECT '{OK}' AS outcome, upd.* FROM upd "
        "UNION ALL " + _canary_outcome_arm()
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key, canary_id, trace_id, card_id)
    return _outcome(row)


async def discard_canary(*, asset_key: str, canary_id: str) -> tuple[str, dict[str, Any] | None]:
    """Clear the armed canary; the active is unchanged.

    Same match and outcomes as :func:`promote_canary`.
    """
    sql = (
        "WITH upd AS ( "
        "    UPDATE workflow_routing_pointer "
        "       SET canary_workflow_id = NULL, canary_pct = NULL, canary_id = NULL, "
        "           released_by_card_id = NULL "
        "     WHERE asset_key = $1 "
        "       AND canary_workflow_id IS NOT NULL "
        "       AND canary_id = $2 "
        f"    RETURNING {_COLUMNS} "
        ") "
        f"SELECT '{OK}' AS outcome, upd.* FROM upd "
        "UNION ALL " + _canary_outcome_arm()
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key, canary_id)
    return _outcome(row)


# ---------------------------------------------------------------------------
# Legacy writes (Improvement Plan 4 / Lane D) — kept on the 036 schema
# ---------------------------------------------------------------------------


async def promote(
    *,
    asset_key: str,
    new_workflow_id: str,
    observation_window_minutes: int,
    rollback_thresholds: dict[str, Any],
) -> dict[str, Any] | None:
    """Legacy promote for ``asset_key`` (upsert; active → prior).

    - First promote: insert with ``prior_workflow_id = NULL``.
    - Subsequent promotes: shift ``active_workflow_id`` → ``prior_workflow_id``
      and set ``active_workflow_id = new_workflow_id``.
    - ``observation_window_minutes = 0`` is the Lane F opt-out.

    ⛔ Returns ``None`` (→ 409) while a canary is ARMED: repointing under it
    could make the canary equal the active (``wrp_canary_differs``) and would
    bypass ``promote_canary``. The ``DO UPDATE … WHERE`` keeps that one statement.
    """
    sql = (
        "INSERT INTO workflow_routing_pointer "
        "    (asset_key, active_workflow_id, prior_workflow_id, "
        "     promoted_at, observation_window_until, rollback_thresholds) "
        "VALUES ($1, $2, NULL, now(), now() + make_interval(mins => $3), $4) "
        "ON CONFLICT (asset_key) DO UPDATE SET "
        "    prior_workflow_id = workflow_routing_pointer.active_workflow_id, "
        "    active_workflow_id = EXCLUDED.active_workflow_id, "
        "    promoted_at = now(), "
        "    observation_window_until = now() + make_interval(mins => $3), "
        "    rollback_thresholds = EXCLUDED.rollback_thresholds "
        "  WHERE workflow_routing_pointer.canary_workflow_id IS NULL "
        f"RETURNING {_COLUMNS}"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            sql,
            asset_key,
            new_workflow_id,
            observation_window_minutes,
            rollback_thresholds,
        )
    return _row_to_dict(row)


async def rollback(*, asset_key: str) -> dict[str, Any] | None:
    """Swap ``active_workflow_id`` back to ``prior_workflow_id``.

    Returns the updated row, or ``None`` (→ 404) when no row exists, it has no
    prior, or a canary is ARMED (the swap could make canary == active). Single
    statement, atomic against concurrent promotes.

    The returned dict carries ONE extra key, ``rolled_back_from``: the
    ``active_workflow_id`` BEFORE the swap (the version being rolled away
    from). ``RETURNING`` sees only post-UPDATE values — where ``prior`` is
    already NULL — so the pre-swap id is read by the ``FOR UPDATE`` sub-select
    in the SAME statement (row-locked, so no concurrent promote can slip
    between the read and the swap). ``WorkflowRoutingPointer.from_row`` ignores
    the extra key; the rollback webhook reports it (#338).
    """
    sql = (
        "UPDATE workflow_routing_pointer p "
        "   SET active_workflow_id = prior_workflow_id, "
        "       prior_workflow_id = NULL "
        "  FROM (SELECT asset_key, active_workflow_id AS rolled_back_from "
        "          FROM workflow_routing_pointer "
        "         WHERE asset_key = $1 "
        "           FOR UPDATE) o "
        " WHERE p.asset_key = o.asset_key "
        "   AND p.prior_workflow_id IS NOT NULL "
        "   AND p.canary_workflow_id IS NULL "
        f"RETURNING {_P_COLUMNS}, o.rolled_back_from"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key)
    out = _row_to_dict(row)
    if out is not None and row is not None:
        out["rolled_back_from"] = _str_or_none(row["rolled_back_from"])
    return out


async def extend_observation_window(
    *, asset_key: str, additional_minutes: int
) -> dict[str, Any] | None:
    """Prolong a legacy observation window, or no-op on a row with NO clock.

    - Window set and a live prior: bump from ``GREATEST(window, now())`` (the
      monotonic rule, unchanged).
    - Window NULL (every 036 release — no clock, decisions 5/7): return the row
      UNCHANGED → 200. ``improve::approval-callback`` still calls /extend by raw
      HTTP (``approval_callback.json``), so this is a no-op, not a 410.
    - Otherwise ``None`` (→ 404): no row, or a windowed row whose prior was
      already cleared (extending it would be a silent no-op the caller did not
      intend).
    """
    sql = (
        "WITH upd AS ( "
        "    UPDATE workflow_routing_pointer "
        "       SET observation_window_until = "
        "           GREATEST(observation_window_until, now()) "
        "           + make_interval(mins => $2) "
        "     WHERE asset_key = $1 "
        "       AND prior_workflow_id IS NOT NULL "
        "       AND observation_window_until IS NOT NULL "
        f"    RETURNING {_COLUMNS} "
        ") "
        "SELECT upd.* FROM upd "
        "UNION ALL "
        f"SELECT {_P_COLUMNS} FROM workflow_routing_pointer p "
        " WHERE p.asset_key = $1 AND p.observation_window_until IS NULL "
        "   AND NOT EXISTS (SELECT 1 FROM upd)"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key, additional_minutes)
    return _row_to_dict(row)


async def clear_prior(*, asset_key: str) -> dict[str, Any] | None:
    """Clear ``prior_workflow_id``. Returns the row, or ``None`` if none exists."""
    sql = (
        "UPDATE workflow_routing_pointer "
        "   SET prior_workflow_id = NULL "
        " WHERE asset_key = $1 "
        f"RETURNING {_COLUMNS}"
    )
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, asset_key)
    return _row_to_dict(row)
