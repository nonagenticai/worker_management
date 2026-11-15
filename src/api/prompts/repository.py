"""Repository for ``prompt_versions`` (migrations 030, 031), its ledger (033) and owner (035).

Plain psycopg 3 through the existing ``AsyncConnectionPool`` — no SQLAlchemy,
no Alembic — mirroring ``src/api/workflow_routing/repository.py`` and the rest
of this service's data layer (``src/db/db_config.py``).

⚠️ Array parameters MUST carry an explicit ``::text[]`` cast on the
placeholder. The pool's asyncpg-compatibility wrapper cannot tell a native
array parameter from a JSONB list value without it, and silently wraps the
bare list in ``Jsonb()`` — which then fails with "jsonb cannot be cast to
text[]". See ``_PsycopgConnectionWrapper._array_param_indices`` and
``tests/integration/test_array_column_binding.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.api.prompts import metrics as prompt_metrics
from src.api.prompts.hashing import format_prompt_revision
from src.db.db_config import get_pg_connection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.api.prompts.actor import Actor

# Every column the API ever returns, in one place so a new column cannot be
# added to one query and forgotten in another. ``search_tsv`` is deliberately
# absent — it is an index input, not API surface.
_COLUMNS = (
    "prompt_id, version, sha256, system_prompt_sha256, question, system, template, "
    "output_schema, worker_names, purpose_tags, git_sha, status, published_at, activated_at"
)


# ---------------------------------------------------------------------------
# Lifecycle (migration 031, P4 contract §2 — the D4 R9 matrix, verbatim)
# ---------------------------------------------------------------------------

DRAFT = "DRAFT"
CANARY = "CANARY"
PUBLISHED = "PUBLISHED"
ARCHIVED = "ARCHIVED"
DISCARDED = "DISCARDED"
STATUSES = (DRAFT, CANARY, PUBLISHED, ARCHIVED, DISCARDED)

# from-status -> the ONLY statuses it may move to. Anything else, including a
# same-status "transition", is a 409. DISCARDED is terminal.
TRANSITIONS: dict[str, frozenset[str]] = {
    DRAFT: frozenset({CANARY, DISCARDED}),
    CANARY: frozenset({PUBLISHED, DISCARDED}),
    PUBLISHED: frozenset({ARCHIVED, DISCARDED}),
    ARCHIVED: frozenset({PUBLISHED, DISCARDED}),
    DISCARDED: frozenset(),
}

# Statuses a pinned EXECUTION may run (P4 §4). A pinned READ returns any status.
RUNNABLE_STATUSES = frozenset({PUBLISHED, CANARY})

# ---------------------------------------------------------------------------
# ROLLOUT WINDOW: legacy spellings (migration 031 is TRANSITIONAL)
# ---------------------------------------------------------------------------
# During a rolling deploy a pre-031 pod keeps running next to this code and
# still WRITES the 030 spellings: its PUT inserts ``'retired'`` and its activate
# writes ``'active'`` / ``'retired'``. 031's CHECK therefore admits both sets,
# and EVERY read here maps a legacy value to its lifecycle status with the same
# rule 031 used for existing rows, so a row an old pod wrote mid-rollout reads
# (and transitions) exactly like one 031 migrated:
#   active                              -> PUBLISHED
#   retired AND activated_at IS NULL    -> DRAFT
#   retired AND activated_at NOT NULL   -> ARCHIVED
# This code never WRITES a legacy value. The follow-up migration
# ``032_prompt_version_lifecycle_tighten.sql`` (to land once no pre-031 pod can
# be running) re-maps any leftover legacy rows and drops 'active'/'retired'
# from the CHECK; after it, this mapping is dead code and can go.
LEGACY_ACTIVE = "active"
LEGACY_RETIRED = "retired"
# The same mapping as a SQL expression, for filters on the effective status.
EFFECTIVE_STATUS_SQL = (
    "(CASE status WHEN 'active' THEN 'PUBLISHED' "
    "WHEN 'retired' THEN CASE WHEN activated_at IS NULL THEN 'DRAFT' ELSE 'ARCHIVED' END "
    "ELSE status END)"
)


def effective_status(status: Any, activated_at: Any) -> str:
    """The lifecycle status of a stored ``status`` (legacy spellings mapped)."""
    raw = str(status)
    if raw == LEGACY_ACTIVE:
        return PUBLISHED
    if raw == LEGACY_RETIRED:
        return DRAFT if activated_at is None else ARCHIVED
    return raw


# ---------------------------------------------------------------------------
# Schema readiness: is 031 applied? (probed once at startup, cached)
# ---------------------------------------------------------------------------
# The startup runner is NON-strict in production: a 031 that fails is logged
# and the pod starts anyway, on the 030 CHECK ('active','retired'). Every write
# here would then die as a CheckViolation 500. The probe below reads the live
# CHECK once at startup; the write routes answer 503 with a clear error while
# it says ``missing_031``, and /health exposes the value.
SCHEMA_READY = "ready"
SCHEMA_MISSING_031 = "missing_031"
SCHEMA_UNKNOWN = "unknown"
_schema_state: str = SCHEMA_UNKNOWN


def lifecycle_schema_state() -> str:
    """The cached probe result: ``ready`` | ``missing_031`` | ``unknown`` (not probed)."""
    return _schema_state


def set_lifecycle_schema_state(state: str) -> None:
    """Replace the cached probe result (startup, and tests)."""
    global _schema_state  # noqa: PLW0603 - a process-wide cache, set once at startup
    _schema_state = state


async def probe_lifecycle_schema(table: str = "prompt_versions") -> str:
    """Read ``table``'s status CHECK and cache whether it admits ``PUBLISHED``.

    ``missing_031`` when the table is absent or its
    ``prompt_versions_status_check`` does not name PUBLISHED (the 030 CHECK);
    ``ready`` otherwise. A probe that raises
    leaves the cache ``unknown`` and re-raises: the caller logs it, and the
    database's own CHECK stays the backstop.
    """
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            "SELECT to_regclass($1) IS NOT NULL AS present, "
            "(SELECT string_agg(pg_get_constraintdef(c.oid), ' ') FROM pg_constraint c "
            " WHERE c.conrelid = to_regclass($1) AND c.contype = 'c' "
            "   AND c.conname = 'prompt_versions_status_check') AS check_def",
            table,
        )
    present = bool(row and row["present"])
    check_def = (row or {}).get("check_def")
    # No CHECK at all admits PUBLISHED too; only an absent table or a CHECK
    # that does not name PUBLISHED (the 030 one) refuses this code's writes.
    admits = check_def is None or f"'{PUBLISHED}'" in check_def
    state = SCHEMA_READY if present and admits else SCHEMA_MISSING_031
    set_lifecycle_schema_state(state)
    return state


class InvalidStatusTransition(Exception):
    """``from_status -> to_status`` is not in :data:`TRANSITIONS`. Surfaced as 409."""

    def __init__(self, prompt_id: str, version: int, from_status: str, to_status: str) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.from_status = from_status
        self.to_status = to_status
        allowed = ", ".join(sorted(TRANSITIONS.get(from_status, frozenset()))) or "none"
        super().__init__(
            f"{prompt_id}@{version} is {from_status}; {from_status} -> {to_status} is not "
            f"a permitted transition (allowed from {from_status}: {allowed})"
        )


class PromptVersionConflict(Exception):
    """A published (prompt_id, version) exists with DIFFERENT content.

    Versions are immutable: you bump, you never rewrite. Surfaced as 409.
    """

    def __init__(
        self, prompt_id: str, version: int, live_sha256: str, incoming_sha256: str
    ) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.live_sha256 = live_sha256
        self.incoming_sha256 = incoming_sha256
        super().__init__(
            f"{prompt_id}@{version} is already published with sha256 {live_sha256}; "
            f"refusing to rewrite it with {incoming_sha256}. Publish a new version instead."
        )


# ---------------------------------------------------------------------------
# Ownership (migration 035): every prompt has ONE owning tenant
# ---------------------------------------------------------------------------


class PromptOwnerMismatch(Exception):
    """The caller's tenant is not the prompt's owner (and holds no prompt role).

    Surfaced as 403 ``{"error": "prompt_owner_mismatch"}``. Strict tenant
    separation: tenant A must never write — and so never change what runs for —
    a prompt owned by tenant B.
    """

    def __init__(self, prompt_id: str, owner_tenant_id: str, caller_tenant_id: str) -> None:
        self.prompt_id = prompt_id
        self.owner_tenant_id = owner_tenant_id
        self.caller_tenant_id = caller_tenant_id
        super().__init__(
            f"prompt {prompt_id} is owned by tenant {owner_tenant_id}; "
            f"tenant {caller_tenant_id} may not write it"
        )


def normalize_tenant(tenant_id: Any) -> str:
    """The stored form of a tenant id: stripped, lowercase text (035 header)."""
    return str(tenant_id).strip().lower()


async def _claim_owner(conn: Any, prompt_id: str, caller_tenant: str, *, privileged: bool) -> str:
    """Return the owner of ``prompt_id``, making ``caller_tenant`` the owner if it has none.

    Runs on ``conn`` inside the caller's transaction, so a NEW prompt's owner row
    and its first version commit (or roll back) together. ``ON CONFLICT DO
    NOTHING`` then re-read: two tenants racing to create one id both insert, the
    loser waits, finds the winner's row and is refused as a non-owner.

    An EXISTING prompt owned by another tenant raises
    :class:`PromptOwnerMismatch` unless ``privileged`` (a verified prompt role:
    the system SAs). A privileged write never changes the owner.
    """
    await conn.execute(
        "INSERT INTO prompt_owner (prompt_id, owner_tenant_id) VALUES ($1, $2) "
        "ON CONFLICT (prompt_id) DO NOTHING",
        prompt_id,
        caller_tenant,
    )
    owner = await conn.fetchval(
        "SELECT owner_tenant_id FROM prompt_owner WHERE prompt_id = $1", prompt_id
    )
    owner = normalize_tenant(owner)
    if owner != caller_tenant and not privileged:
        raise PromptOwnerMismatch(prompt_id, owner, caller_tenant)
    return owner


async def _check_owner(conn: Any, prompt_id: str, caller_tenant: str) -> None:
    """Raise :class:`PromptOwnerMismatch` unless ``caller_tenant`` owns ``prompt_id``."""
    owner = await conn.fetchval(
        "SELECT owner_tenant_id FROM prompt_owner WHERE prompt_id = $1", prompt_id
    )
    owner_key = "" if owner is None else normalize_tenant(owner)
    if owner_key != caller_tenant:
        raise PromptOwnerMismatch(prompt_id, owner_key or "<none>", caller_tenant)


# ---------------------------------------------------------------------------
# Transition ledger (migration 033) and the sanctioned initial publish
# ---------------------------------------------------------------------------

# The ``reason`` that makes DRAFT -> PUBLISHED legal: a Claude session CREATING
# the first live version of a prompt (P4 owner ruling 2026-09-28). Creation, not
# promotion — so it must not fake a CANARY hop, and the ledger row says so.
INITIAL_CREATE = "initial_create"
# Reasons the service writes itself (not caller free text).
REASON_ACTIVATE = "activate"
REASON_ACTIVATE_SUPERSEDED = "activate_superseded"
# ``scripts/adopt_live_worker_prompts.py``: v1 of a worker's prompt IS the live
# text read from ``workers.config`` (no seeding from git). Same creation
# semantics as INITIAL_CREATE, recorded under its own name so the ledger says
# where the text came from. Reserved in models.RESERVED_TRANSITION_REASONS, so
# the public API can never send it.
REASON_ADOPT_LIVE = "adopt_live"
# Every reason that takes the initial-create branch of :func:`transition`
# (DRAFT -> PUBLISHED, refused once the prompt has any publish history).
INITIAL_CREATE_REASONS = frozenset({INITIAL_CREATE, REASON_ADOPT_LIVE})

# Refusal reasons for :class:`InitialCreateNotAllowed`.
INITIAL_CREATE_HISTORY = "prompt_has_publish_history"
INITIAL_CREATE_NOT_DRAFT = "version_is_not_draft"
INITIAL_CREATE_NOT_PUBLISH = "initial_create_targets_published_only"


class InitialCreateNotAllowed(Exception):
    """``reason=initial_create`` refused. Surfaced as 409 with a top-level body."""

    def __init__(self, prompt_id: str, version: int, reason: str) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.reason = reason
        super().__init__(f"initial_create refused for {prompt_id}@{version}: {reason}")


@asynccontextmanager
async def _write_conn(conn: Any | None) -> AsyncIterator[Any]:
    """The connection a write path runs on.

    ``conn`` given: the CALLER's connection, inside the caller's open
    transaction — nothing is opened or committed here, so several writes can
    share one transaction (the adoption job). ``None``: a pooled connection in
    its own transaction, exactly as before.
    """
    if conn is not None:
        yield conn
        return
    async with get_pg_connection() as own, own.transaction():
        yield own


_INSERT_TRANSITION_SQL = (
    "INSERT INTO prompt_version_transitions "
    "(prompt_id, version, from_status, to_status, reason, actor_sub, actor_kind, actor_card_id) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
)

# Migration 039's CHECK: a CANARY/PUBLISHED ledger row without ``actor_card_id``
# (and not ``actor_kind='migration'``). The route checks first and refuses 422;
# this name lets it map the database backstop to the same 422.
RELEASE_CARD_CONSTRAINT = "prompt_version_transitions_release_card_check"


async def _record_transition(
    conn: Any,
    prompt_id: str,
    version: int,
    from_status: str | None,
    to_status: str,
    reason: str | None,
    actor: Actor,
    actor_card_id: str | None = None,
) -> None:
    """Append one ledger row on ``conn`` — the caller's open transaction.

    ``actor_card_id`` is the release's Planka card (migration 039, REALIGN R7).
    """
    await conn.execute(
        _INSERT_TRANSITION_SQL,
        prompt_id,
        version,
        from_status,
        to_status,
        reason,
        actor.sub,
        actor.kind,
        actor_card_id,
    )


async def _is_bound(conn: Any, prompt_id: str) -> bool:
    """True when a ``worker_prompt_binding`` row references ``prompt_id``.

    Probes ``to_regclass`` first: on a database without migration 034 the
    table is absent, and a bare reference would abort the caller's transaction
    — a metric label must never refuse a publish.
    """
    if await conn.fetchval("SELECT to_regclass('worker_prompt_binding')") is None:
        return False
    return bool(
        await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM worker_prompt_binding WHERE prompt_id = $1)",
            prompt_id,
        )
    )


async def _mirror_bound_configs(conn: Any, prompt_id: str) -> int:
    """Rewrite bound workers' ``config.system_prompt`` to the resolved text (U6 mirror).

    On ``conn`` — the transition's own open transaction — so the mirror and the
    status move commit or roll back together. Imported lazily:
    ``src.api.workers.prompt_source`` imports this module.
    """
    from src.api.workers import prompt_source  # import cycle: prompt_source imports this module

    return await prompt_source.sync_bound_worker_configs(conn, prompt_id)


def _count_config_synced(prompt_id: str, synced: int) -> None:
    from src.api.workers import prompt_source  # import cycle: prompt_source imports this module

    prompt_source.count_config_synced(prompt_id, synced)


async def _has_publish_history(conn: Any, prompt_id: str) -> bool:
    """True when ANY version of ``prompt_id`` was EVER PUBLISHED or ARCHIVED.

    Two independent witnesses, either is enough:

    * the ROWS — a version whose (effective) status is PUBLISHED or ARCHIVED, or
      that carries ``activated_at`` (stamped on every entry to PUBLISHED and
      never cleared, so it survives a later DISCARD). This covers history made
      before the ledger existed (migration 033 backfills creation rows only);
    * the LEDGER — any row entering or leaving PUBLISHED/ARCHIVED. This covers
      history the rows no longer show.
    """
    row = await conn.fetchrow(
        "SELECT EXISTS (SELECT 1 FROM prompt_versions WHERE prompt_id = $1 AND "
        "  (status IN ('PUBLISHED', 'ARCHIVED', 'active') OR activated_at IS NOT NULL)"
        ") AS in_rows, "
        "EXISTS (SELECT 1 FROM prompt_version_transitions WHERE prompt_id = $1 AND "
        "  (to_status IN ('PUBLISHED', 'ARCHIVED') OR from_status IN ('PUBLISHED', 'ARCHIVED'))"
        ") AS in_ledger",
        prompt_id,
    )
    return bool(row["in_rows"]) or bool(row["in_ledger"])


async def get_transitions(prompt_id: str, version: int) -> list[dict[str, Any]] | None:
    """The ledger rows of one version, oldest first; ``None`` if the version is absent."""
    async with get_pg_connection() as conn:
        exists = await conn.fetchrow(
            "SELECT 1 AS one FROM prompt_versions WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
        )
        if exists is None:
            return None
        rows = await conn.fetch(
            "SELECT id, prompt_id, version, from_status, to_status, reason, actor_sub, "
            "actor_kind, actor_card_id, at FROM prompt_version_transitions "
            "WHERE prompt_id = $1 AND version = $2 ORDER BY at, id",
            prompt_id,
            version,
        )
    return [
        {
            "id": int(r["id"]),
            "prompt_id": r["prompt_id"],
            "version": int(r["version"]),
            "from_status": r["from_status"],
            "to_status": r["to_status"],
            "reason": r["reason"],
            "actor_sub": r["actor_sub"],
            "actor_kind": r["actor_kind"],
            "actor_card_id": r["actor_card_id"],
            "at": _stringify_dt(r["at"]),
        }
        for r in rows
    ]


def _stringify_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Coerce a psycopg ``dict_row`` into the JSON shape the API returns."""
    if row is None:
        return None
    prompt_id = row["prompt_id"]
    version = int(row["version"])
    out: dict[str, Any] = {
        "prompt_id": prompt_id,
        "version": version,
        # CHAR(64) comes back blank-padded on some drivers; strip defensively so
        # a digest comparison can never fail on trailing whitespace.
        "sha256": str(row["sha256"]).strip(),
        "system_prompt_sha256": str(row["system_prompt_sha256"]).strip(),
        "question": row.get("question"),
        "system": row["system"],
        "template": row.get("template"),
        "output_schema": row.get("output_schema"),
        "worker_names": list(row.get("worker_names") or []),
        "purpose_tags": list(row.get("purpose_tags") or []),
        "git_sha": row.get("git_sha"),
        "status": effective_status(row["status"], row.get("activated_at")),
        "published_at": _stringify_dt(row.get("published_at")),
        "activated_at": _stringify_dt(row.get("activated_at")),
        "config_revision": format_prompt_revision(prompt_id, version),
    }
    return out


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


async def get_version(prompt_id: str, version: int) -> dict[str, Any] | None:
    """One version, ANY status — the pinned read. Replay reads non-live ones."""
    sql = f"SELECT {_COLUMNS} FROM prompt_versions WHERE prompt_id = $1 AND version = $2"
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(sql, prompt_id, version)
    return _row_to_dict(row)


async def get_active(prompt_id: str, *, highest_published: bool = False) -> dict[str, Any] | None:
    """The version an UNPINNED read resolves to, or ``None`` when nothing is live.

    Only PUBLISHED versions are ever resolved; DRAFT, CANARY, ARCHIVED and
    DISCARDED never are. Which PUBLISHED one depends on the flag
    (``PROMPTS_READ_REQUIRES_PUBLISHED``, passed in by the route):

    * ``highest_published=True`` — the HIGHEST PUBLISHED version (P4 §3, the
      D4 R9 resolver).
    * ``highest_published=False`` — today's behaviour: the single version the
      last ``activate`` made live, i.e. the most recently activated PUBLISHED
      row. While ``activate`` is the only way to publish (it archives the
      others) the two agree; they diverge only once the status route has made
      several versions PUBLISHED, which is what the flag is for.

    Publishing is not activating. A prompt whose versions are all still DRAFT
    answers ``None`` here on purpose — silently promoting the highest version
    would make a PUT a deploy.
    """
    async with get_pg_connection() as conn:
        return await get_active_on(conn, prompt_id, highest_published=highest_published)


async def get_active_on(
    conn: Any, prompt_id: str, *, highest_published: bool = False
) -> dict[str, Any] | None:
    """:func:`get_active` on the CALLER's connection (and so its open transaction).

    The one SQL both share: the config mirror (``prompt_source.
    sync_bound_worker_configs``) resolves inside a transition's transaction,
    where a fresh pooled connection could not see the uncommitted move.
    """
    order = "version DESC" if highest_published else "activated_at DESC NULLS LAST, version DESC"
    sql = (
        f"SELECT {_COLUMNS} FROM prompt_versions "
        f"WHERE prompt_id = $1 AND status IN ('PUBLISHED', 'active') "
        f"ORDER BY {order} LIMIT 1"
    )
    row = await conn.fetchrow(sql, prompt_id)
    return _row_to_dict(row)


async def search(
    *,
    q: str | None,
    tag: str | None,
    status: str | list[str] | None,
    limit: int,
    prompt_id: str | None = None,
    worker_name: str | None = None,
) -> list[dict[str, Any]]:
    """Full-text search, ordered ``ts_rank desc, prompt_id asc`` (design §2.3).

    ``prompt_id`` (optional) is an EXACT match, ANDed with every other filter.
    It is what lets a caller prove a prompt is ABSENT: an unfiltered listing is
    capped at ``limit`` across ALL prompts and has no offset, so once the table
    outgrows the cap "not in the first page" stops meaning "not there".

    An empty ``q`` is a LISTING, not an error: rank is 0 for every row and the
    secondary ``prompt_id asc`` key makes the order total and stable. That keeps
    one route answering both "find me a prompt about X" and "what is in here".

    ``worker_name`` (optional) matches when that exact name is an ELEMENT of the
    row's ``worker_names`` array (``= ANY``), bound as a parameter, ANDed with
    every other filter and still capped by ``limit``.
    """
    where = ["TRUE"]
    args: list[Any] = []

    if q:
        args.append(q)
        where.append(f"search_tsv @@ websearch_to_tsquery('english', ${len(args)})")
    if tag:
        args.append([tag])
        where.append(f"purpose_tags && ${len(args)}::text[]")
    if prompt_id is not None:
        args.append(prompt_id)
        where.append(f"prompt_id = ${len(args)}")
    if worker_name is not None:
        args.append(worker_name)
        where.append(f"${len(args)} = ANY(worker_names)")
    if status:
        args.append([status] if isinstance(status, str) else list(status))
        # The EFFECTIVE status, so a legacy row an old pod wrote mid-rollout
        # matches the lifecycle filter it reads as.
        where.append(f"{EFFECTIVE_STATUS_SQL} = ANY(${len(args)}::text[])")

    if q:
        args.append(q)
        rank_expr = f"ts_rank(search_tsv, websearch_to_tsquery('english', ${len(args)}))"
    else:
        rank_expr = "0::float4"

    args.append(limit)
    sql = (
        "SELECT prompt_id, version, sha256, system_prompt_sha256, question, purpose_tags, "
        f"worker_names, status, published_at, activated_at, {rank_expr} AS rank "
        "FROM prompt_versions "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY rank DESC, prompt_id ASC, version DESC LIMIT ${len(args)}"
    )
    async with get_pg_connection() as conn:
        rows = await conn.fetch(sql, *args)

    hits: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        version = int(row["version"])
        hits.append(
            {
                "prompt_id": prompt_id,
                "version": version,
                "sha256": str(row["sha256"]).strip(),
                "system_prompt_sha256": str(row["system_prompt_sha256"]).strip(),
                "question": row.get("question"),
                "purpose_tags": list(row.get("purpose_tags") or []),
                "worker_names": list(row.get("worker_names") or []),
                "status": effective_status(row["status"], row.get("activated_at")),
                "published_at": _stringify_dt(row.get("published_at")),
                "config_revision": format_prompt_revision(prompt_id, version),
                "rank": float(row.get("rank") or 0.0),
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


async def upsert_version(  # noqa: PLR0913 -- one argument per ledger column, by design
    *,
    prompt_id: str,
    version: int,
    sha256: str,
    system_prompt_sha256: str,
    question: str | None,
    system: str,
    template: str | None,
    output_schema: dict[str, Any] | None,
    worker_names: list[str],
    purpose_tags: list[str],
    git_sha: str | None,
    actor: Actor,
    owner_tenant_id: str,
    privileged: bool = False,
    conn: Any | None = None,
) -> tuple[dict[str, Any], bool]:
    """Publish ``(prompt_id, version)``. Returns ``(row, written)``.

    ``INSERT … ON CONFLICT DO NOTHING`` then re-read, rather than a read-then-
    write: two publisher runs racing on the same version would both see "absent"
    under the latter and the loser would get a primary-key error instead of a
    clean idempotent answer.

    Three outcomes, and only one of them writes:
      * absent          → inserted, ``written=True``
      * present, same sha256 → nothing written, ``written=False``
      * present, different sha256 → :class:`PromptVersionConflict` (409)

    ⛔ There is no UPDATE path here, by design. A published version is the
    thing replay reads; if it could be rewritten, every historical attribution
    would be unverifiable after the fact.

    The row is born ``DRAFT``: readable and replayable, but not what
    ``GET /prompts/{id}`` serves until it is PUBLISHED (status route or the
    deprecated activate).

    An insert also appends the creation row (NULL -> DRAFT) to
    ``prompt_version_transitions`` in the same transaction; an idempotent
    re-PUT changes nothing and records nothing.

    OWNERSHIP (migration 035). ``owner_tenant_id`` is the CALLER's tenant. A
    NEW ``prompt_id`` is claimed for it (``prompt_owner`` row, same
    transaction); an EXISTING one must already be owned by it, else
    :class:`PromptOwnerMismatch` (403) and nothing is written — unless
    ``privileged`` (a verified prompt role), which may write a version of any
    prompt but never changes its owner. The version row always carries the
    prompt's owner (the composite FK enforces it).

    ``conn`` (optional): run on the caller's connection and open transaction
    instead of opening one (see :func:`_write_conn`).
    """
    caller_tenant = normalize_tenant(owner_tenant_id)
    insert_sql = f"""
        INSERT INTO prompt_versions (
            prompt_id, version, sha256, system_prompt_sha256, question, system,
            template, output_schema, worker_names, purpose_tags, git_sha, status,
            owner_tenant_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::text[], $10::text[], $11, 'DRAFT', $12)
        ON CONFLICT (prompt_id, version) DO NOTHING
        RETURNING {_COLUMNS}
    """
    async with _write_conn(conn) as wconn:
        owner = await _claim_owner(wconn, prompt_id, caller_tenant, privileged=privileged)
        inserted = await wconn.fetchrow(
            insert_sql,
            prompt_id,
            version,
            sha256,
            system_prompt_sha256,
            question,
            system,
            template,
            output_schema,
            list(worker_names),
            list(purpose_tags),
            git_sha,
            owner,
        )
        if inserted is not None:
            row = _row_to_dict(inserted)
            assert row is not None  # noqa: S101 - RETURNING guarantees a row here
            await _record_transition(wconn, prompt_id, version, None, DRAFT, None, actor)
            return row, True

        existing = await wconn.fetchrow(
            f"SELECT {_COLUMNS} FROM prompt_versions WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
        )

    row = _row_to_dict(existing)
    if row is None:  # pragma: no cover - only if the row vanished between statements
        raise PromptVersionConflict(prompt_id, version, "unknown", sha256)
    if row["sha256"] != sha256:
        raise PromptVersionConflict(prompt_id, version, row["sha256"], sha256)
    return row, False


async def transition(
    prompt_id: str,
    version: int,
    to_status: str,
    *,
    actor: Actor,
    reason: str | None = None,
    actor_card_id: str | None = None,
    caller_tenant_id: str | None = None,
    privileged: bool = False,
    conn: Any | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Move one version along the lifecycle. Returns ``(row, from_status)``.

    ``row`` is ``None`` when the version was never published (404). A move not
    in :data:`TRANSITIONS` raises :class:`InvalidStatusTransition` (409) and
    writes nothing.

    The row is locked ``FOR UPDATE`` so two concurrent transitions of one
    version serialise and the second is judged against the first's result, not
    against a stale read.

    Entering PUBLISHED stamps ``activated_at = now()``: it is "when this version
    last went live", which is what the flag-off resolver orders by and what the
    down migration / a re-applied 031 use to tell ARCHIVED from DRAFT.

    ``reason == INITIAL_CREATE`` is the one move OUTSIDE the matrix: DRAFT ->
    PUBLISHED, allowed only when no version of the prompt was ever PUBLISHED or
    ARCHIVED (:func:`_has_publish_history`); otherwise
    :class:`InitialCreateNotAllowed`. The route checks the elevated role before
    calling this. Every row of the prompt is locked first, so two concurrent
    initial creates of different versions serialise and the second sees the
    first's history.

    Every applied move appends one ``prompt_version_transitions`` row in the
    same transaction, carrying ``reason`` and ``actor``.

    Every reason in :data:`INITIAL_CREATE_REASONS` (``initial_create``, and
    ``adopt_live`` from the adoption job) takes the initial-create branch with
    all of its checks; the ledger records the reason actually given.

    OWNERSHIP (migration 035): with ``caller_tenant_id`` given and not
    ``privileged``, a version of a prompt the caller's tenant does not own is
    refused with :class:`PromptOwnerMismatch` (403) before anything moves.
    ``caller_tenant_id=None`` skips the check: in-process callers (the adoption
    job) have already established ownership through :func:`upsert_version`.

    ``conn`` (optional): run on the caller's connection and open transaction
    instead of opening one (see :func:`_write_conn`). A refusal then raises
    without rolling back — the caller owns the transaction, and so also owns
    ``wm_prompt_published_bound_total``: nothing is counted on a caller's
    ``conn`` (it may still roll back); the caller counts after it commits.

    RELEASE CARD (REALIGN R7, migration 039): ``actor_card_id`` is recorded on
    the ledger row of a move into CANARY or PUBLISHED, and ignored for any other
    target. The ROUTE refuses a release without a valid one (422); here the
    database CHECK is the backstop — a release row without a card is refused
    (``CheckViolation`` on :data:`RELEASE_CARD_CONSTRAINT`) unless the actor is
    ``migration`` (the in-process adoption job, which has no card to name).

    CONFIG MIRROR (U6, C2-32): a move into or out of PUBLISHED re-resolves every
    worker bound to the prompt and rewrites its ``config.system_prompt`` to the
    resolved text, in this same transaction
    (``prompt_source.sync_bound_worker_configs``). ``wm_prompt_config_synced_total``
    is counted after commit, and — like the publish counter — never on a
    caller's ``conn``.
    """
    if to_status not in STATUSES:
        raise ValueError(f"unknown status {to_status!r}")
    initial_create = reason in INITIAL_CREATE_REASONS
    async with _write_conn(conn) as wconn:
        if initial_create:
            await wconn.fetch(
                "SELECT version FROM prompt_versions WHERE prompt_id = $1 "
                "ORDER BY version FOR UPDATE",
                prompt_id,
            )
        target = await wconn.fetchrow(
            "SELECT status, activated_at FROM prompt_versions "
            "WHERE prompt_id = $1 AND version = $2 FOR UPDATE",
            prompt_id,
            version,
        )
        if target is None:
            return None, None
        if caller_tenant_id is not None and not privileged:
            await _check_owner(wconn, prompt_id, normalize_tenant(caller_tenant_id))
        from_status = effective_status(target["status"], target["activated_at"])
        if initial_create:
            if to_status != PUBLISHED:
                raise InitialCreateNotAllowed(prompt_id, version, INITIAL_CREATE_NOT_PUBLISH)
            if from_status != DRAFT:
                raise InitialCreateNotAllowed(prompt_id, version, INITIAL_CREATE_NOT_DRAFT)
            if await _has_publish_history(wconn, prompt_id):
                raise InitialCreateNotAllowed(prompt_id, version, INITIAL_CREATE_HISTORY)
        elif to_status not in TRANSITIONS.get(from_status, frozenset()):
            raise InvalidStatusTransition(prompt_id, version, from_status, to_status)
        await wconn.execute(
            "UPDATE prompt_versions SET status = $3, "
            "activated_at = CASE WHEN $4::boolean THEN now() ELSE activated_at END "
            "WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
            to_status,
            to_status == PUBLISHED,
        )
        await _record_transition(
            wconn,
            prompt_id,
            version,
            from_status,
            to_status,
            reason,
            actor,
            actor_card_id if to_status in RUNNABLE_STATUSES else None,
        )
        published_now = to_status == PUBLISHED and from_status != PUBLISHED
        bound = published_now and await _is_bound(wconn, prompt_id)
        # Only a move into or out of PUBLISHED can change the resolved text
        # (CANARY is never resolved); any other move leaves the config alone.
        synced = 0
        if PUBLISHED in (from_status, to_status):
            synced = await _mirror_bound_configs(wconn, prompt_id)
        row = await wconn.fetchrow(
            f"SELECT {_COLUMNS} FROM prompt_versions WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
        )
    # After the block: on our own connection the move is committed by now, so a
    # rolled-back publish is never counted. On a caller's ``conn`` this function
    # cannot know whether the caller will commit, so it counts NOTHING: the
    # caller counts after its own commit (the adoption job is the only such
    # caller; see ``scripts/adopt_live_worker_prompts.py``).
    if published_now and conn is None:
        prompt_metrics.count_published(prompt_id, bound=bound)
    if conn is None:
        _count_config_synced(prompt_id, synced)
    return _row_to_dict(row), from_status


async def activate(
    prompt_id: str, version: int, *, actor: Actor, actor_card_id: str | None = None
) -> tuple[dict[str, Any] | None, int | None, bool]:
    """DEPRECATED alias: PUBLISH ``version`` and ARCHIVE every other PUBLISHED one.

    Returns ``(row, previous_active_version, changed)``; ``row`` is ``None``
    when that version was never published (404).

    Kept so existing callers keep today's "exactly one live version" semantics
    after migration 031 dropped the one-active index. It therefore does NOT go
    through the transition matrix for the target: a DRAFT (today's freshly-PUT
    ``retired``) or CANARY or ARCHIVED version is published directly, as
    ``activate`` always did. The one refusal is DISCARDED, which is terminal:
    :class:`InvalidStatusTransition` (409). Use ``POST …/status`` for
    matrix-checked moves.

    All rows of the prompt are locked first so two concurrent activates cannot
    each archive the other and leave zero — or two — PUBLISHED.

    Activating a LOWER version is permitted: that is a deliberate rollback.
    ``previous_active_version`` is the highest other version that was PUBLISHED
    (and is now ARCHIVED), or ``None``.

    Ledger: one row per version that actually moved, in the same transaction —
    each other PUBLISHED -> ARCHIVED (reason ``activate_superseded``) and the
    target -> PUBLISHED (reason ``activate``). An already-PUBLISHED target with
    nothing to archive records nothing.

    No ownership check (migration 035): the route admits only the elevated
    ``service-internal`` role, which may act on any prompt.

    RELEASE CARD (REALIGN R7, migration 039): ``actor_card_id`` is recorded on
    EVERY row this call appends — the publish and each archive it causes are
    one release, authorised by one card. The route requires it (422).
    """
    async with get_pg_connection() as conn, conn.transaction():
        rows = await conn.fetch(
            "SELECT version, status, activated_at FROM prompt_versions WHERE prompt_id = $1 "
            "ORDER BY version FOR UPDATE",
            prompt_id,
        )
        by_version = {
            int(r["version"]): effective_status(r["status"], r["activated_at"]) for r in rows
        }
        target_status = by_version.get(version)
        if target_status is None:
            return None, None, False
        if target_status == DISCARDED:
            raise InvalidStatusTransition(prompt_id, version, DISCARDED, PUBLISHED)

        others = sorted(
            (v for v, st in by_version.items() if st == PUBLISHED and v != version),
            reverse=True,
        )
        previous_version = others[0] if others else None

        archived = await conn.fetch(
            "UPDATE prompt_versions SET status = 'ARCHIVED' "
            "WHERE prompt_id = $1 AND status IN ('PUBLISHED', 'active') AND version <> $2 "
            "RETURNING version",
            prompt_id,
            version,
        )
        for r in sorted(archived, key=lambda r: int(r["version"])):
            await _record_transition(
                conn,
                prompt_id,
                int(r["version"]),
                PUBLISHED,
                ARCHIVED,
                REASON_ACTIVATE_SUPERSEDED,
                actor,
                actor_card_id,
            )
        already_published = target_status == PUBLISHED
        if not already_published:
            await conn.execute(
                "UPDATE prompt_versions SET status = 'PUBLISHED', activated_at = now() "
                "WHERE prompt_id = $1 AND version = $2",
                prompt_id,
                version,
            )
            await _record_transition(
                conn,
                prompt_id,
                version,
                target_status,
                PUBLISHED,
                REASON_ACTIVATE,
                actor,
                actor_card_id,
            )
        bound = not already_published and await _is_bound(conn, prompt_id)
        synced = 0
        if not already_published or archived:
            synced = await _mirror_bound_configs(conn, prompt_id)
        row = await conn.fetchrow(
            f"SELECT {_COLUMNS} FROM prompt_versions WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
        )

    # Nothing moved only when this version was ALREADY published and no other
    # version had to be archived to get there.
    changed = (not already_published) or bool(others)
    if not already_published:
        prompt_metrics.count_published(prompt_id, bound=bound)
    _count_config_synced(prompt_id, synced)
    return _row_to_dict(row), previous_version, changed
