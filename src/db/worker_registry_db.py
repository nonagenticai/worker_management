"""Worker registry database module (PostgreSQL with multi-tenancy).

This module provides database operations for managing worker registrations.
All operations are scoped by ``(tenant_id, subject_id, visibility)`` per the
per-user-within-tenant isolation model:

- ``visibility='tenant'`` rows are visible/mutable by any member of the tenant.
- ``visibility='user'`` rows are visible/mutable only by the creating subject.

See ``src/repositories/visibility.py`` for the predicate helpers and
``migrations/010_per_user_within_tenant_isolation.sql`` for the schema.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid7

import psycopg

from src.api.tenant_constants import tenant_label
from src.db.db_config import get_pg_connection
from src.db.workers_state_db import with_retry
from src.repositories.visibility import scope_clause

logger = logging.getLogger(__name__)


def _looks_like_uuid(value: str) -> bool:
    """True when ``value`` parses as a UUID.

    Used to refuse the re-key below when a worker's NAME is itself a UUID
    string. ``WorkerStateManager.resolve_worker_id`` has a secondary lookup that
    returns such a name verbatim as the worker_id, so a name-shaped-like-a-UUID
    could collide with a DIFFERENT worker's real worker_id and the re-key would
    move that worker's history. Pathological but cheap to exclude.
    """
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _rows_affected(status: Any) -> int:
    """Row count from an asyncpg-style command status such as ``"UPDATE 396"``.

    ``_PsycopgConnectionWrapper.execute`` returns the cursor's statusmessage as
    a plain string (``src/db/db_config.py:154-160``), NOT a cursor -- so there
    is no ``.rowcount`` attribute to read. Returns 0 for anything unparseable
    rather than raising: this feeds a log line, and a mis-shaped status must
    never take down a registration.
    """
    if not isinstance(status, str):
        return 0
    tail = status.rsplit(" ", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


class WorkerNameConflictError(Exception):
    """A worker_name is already owned within the tenant by an unseeable row.

    Raised by :meth:`WorkerRegistryDB.register_worker` when the blind INSERT
    trips the ``unique_tenant_worker_name (tenant_id, worker_name)`` constraint
    but the caller-scoped re-SELECT finds nothing to return idempotently — i.e.
    the colliding row belongs to another subject's ``visibility='user'`` worker
    in the same tenant. The lookup identity (tenant_id, subject, visibility) is
    narrower than the constraint identity (tenant_id, worker_name), so this is
    reachable even though the initial lookup missed.

    This is intentionally NOT a ``psycopg.Error`` so the ``@with_retry``
    decorator does not retry it (the conflict is deterministic — retrying only
    burns connections before failing) and so API layers can map it to 409.
    """

    def __init__(self, tenant_id: str, worker_name: str) -> None:
        self.tenant_id = tenant_id
        self.worker_name = worker_name
        super().__init__(
            f"worker_name {worker_name!r} already exists in tenant {tenant_id!r} "
            f"and is owned by another subject"
        )


class WorkerRegistryDB:
    """API for the worker registry database with PostgreSQL and multi-tenancy support."""

    @staticmethod
    async def _rekey_name_scoped_states(
        conn: Any,
        *,
        tenant_id: str,
        effective_subject: str,
        worker_name: str,
        worker_id: str,
    ) -> int:
        """Re-point this worker's name-keyed state rows at its real worker_id.

        ``WorkerStateManager.resolve_worker_id`` falls back to using the
        worker_name AS the worker_id for anything not in the registry
        ("legacy compatibility"). That fallback is deliberate and stays -- a
        whole class of unregistered workers relies on it. But when such a worker
        LATER registers and is minted a UUID, its earlier rows keep the name as
        their worker_id and the two halves of its history stop being reachable
        together:

        ``get_worker_states_by_id`` binds the SAME value to both placeholders
        (``worker_id = $3 OR worker_name = $4``), so a lookup BY NAME still finds
        everything, but a lookup BY UUID -- which is what every caller uses once
        the worker is registered -- matches neither column on the older rows.
        Measured in production 2026-08-12: 396 rows across 25 workers were
        invisible this way (video-frame-analyzer: 154 name-keyed vs 8525
        uuid-keyed).

        Runs on both the create and the update path so it is self-healing: the
        pre-existing split rows are repaired the next time each worker
        registers, without a separate backfill migration. Idempotent -- after
        the first run no row matches.

        Returns the number of rows re-keyed (0 on the overwhelmingly common
        path, where the worker never had name-keyed state).
        """
        if _looks_like_uuid(worker_name) or worker_name == worker_id:
            return 0

        # ``worker_id::text`` is defensive, and deliberately kept. PRODUCTION
        # declares worker_states.worker_id as TEXT -- it has to, because the
        # fallback above stores worker NAMES in it. Until 2026-08-13
        # postgres_schema.sql declared it ``UUID`` and CI builds its ephemeral
        # database from that file, so the two disagreed: without the cast postgres
        # inferred $4 as ``uuid`` from ``worker_id = $4`` and the very next term
        # failed with "operator does not exist: character varying = uuid". The
        # file now declares TEXT, so that divergence is closed -- but the cast
        # stays, because it is free, it makes the statement correct against BOTH
        # shapes, and "make worker_id a real UUID" is still an open end-state
        # (see the 2026-08-13 amendment in postgres_schema.sql). It also cannot
        # match a UUID-typed column against a name, which is the right behaviour.
        _sql = (  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
            f"UPDATE worker_states SET worker_id = $1 "
            f"WHERE {scope_clause(2, 3)} AND worker_id::text = $4 AND worker_name = $4"
        )
        result = await conn.execute(_sql, worker_id, tenant_id, effective_subject, worker_name)

        # ``_PsycopgConnectionWrapper.execute`` returns an asyncpg-style STATUS
        # STRING ("UPDATE 396"), not a cursor (src/db/db_config.py:154-160), so
        # there is no .rowcount to read -- reading one silently yields 0 and the
        # log below would never report a count.
        rekeyed = _rows_affected(result)
        if rekeyed:
            logger.info(
                "Re-keyed %d name-keyed worker_states row(s) onto worker_id: "
                "tenant_id=%s, worker_name=%s, worker_id=%s",
                rekeyed,
                tenant_id,
                worker_name,
                worker_id,
            )
        return rekeyed

    @with_retry()
    async def register_worker(  # noqa: PLR0913 -- SQL upsert with one arg per worker column
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        worker_name: str,
        db_path: str,
        is_agent: Optional[bool],
        config: dict[str, Any],
        worker_id: Optional[str] = None,
        status: str = "created",
        output_schema: Optional[str] = None,
        output_validator: Optional[str] = None,
        visibility: str = "tenant",
        config_revision: Optional[str] = None,
        merge_config: bool = True,
    ) -> dict[str, Any]:
        """Register (or upsert by worker_name) a worker.

        Lookup uses the visibility scope clause: an existing tenant-visible
        worker with this name is updated regardless of caller subject; an
        existing user-visible worker with this name is only matched if the
        caller is the owner. Otherwise a new row is inserted with the
        requested ``visibility``.

        ``merge_config`` (default True) makes the UPDATE branch **additive**:
        ``config`` is shallow-merged over the stored jsonb, and the columns
        ``is_agent`` / ``output_schema`` / ``output_validator`` /
        ``config_revision`` are only written when the caller passes a non-None
        value. Callers therefore send only the keys they own; a key they omit
        is preserved rather than nulled.

        The merge is done **inside the one UPDATE statement** (``config || $4``),
        not read-modify-write in Python, so two concurrent registrations that
        touch disjoint keys both survive. A Python-side merge cannot give that.

        Pass ``merge_config=False`` for a true full replace — the caller's body
        becomes the whole record. Correct only when the caller is authoritative
        for every field. **No caller passes it today**; it exists so "merge" is
        a default rather than a cage, and it is the mutant the ``-m pg`` suite
        uses to prove the merge is load-bearing.

        ⚠️ Under a merge ``config_revision`` becomes **unclearable** through this
        function: ``COALESCE`` means a ``None`` preserves. That matches the
        ``swap_prompt`` path below, which has always done it, and it is a
        deliberate trade — no caller clears it (748 of 750 prod rows are NULL,
        and the field is round-tripped, never interpreted) while every partial
        caller would otherwise null it. If a clear is ever needed, add an
        explicit sentinel rather than reading "omitted" as "clear".

        The INSERT branch is unaffected: there is nothing to merge over.
        """
        effective_subject = subject_id or tenant_id

        async with get_pg_connection() as conn:
            now = datetime.now(UTC).replace(tzinfo=None)

            existing = await conn.fetchrow(
                f"""
                SELECT worker_id, operational_status
                FROM workers
                WHERE {scope_clause(1, 2)} AND worker_name = $3
                """,
                tenant_id,
                effective_subject,
                worker_name,
            )

            if existing:
                existing_worker_id = existing["worker_id"]

                if merge_config:
                    # Shallow jsonb merge, right-hand wins, evaluated server-side
                    # so concurrent writers touching disjoint keys both survive.
                    # COALESCE guards a NULL config (none in prod, but the column
                    # is nullable and `NULL || x` is NULL).
                    # Every COALESCE argument carries an explicit cast: psycopg
                    # sends a Python ``None`` as an untyped NULL, and COALESCE
                    # resolves its result type at plan time — without the cast
                    # ``COALESCE($6, output_schema)`` mixes text with the jsonb
                    # column and fails to resolve. The bare ``col = $n`` form
                    # never hit this because assignment casts are implicit.
                    _config_set = "config = COALESCE(config, '{}'::jsonb) || $4::jsonb"
                    _is_agent_set = "is_agent = COALESCE($2::boolean, is_agent)"
                    _schema_set = "output_schema = COALESCE($6::jsonb, output_schema)"
                    _validator_set = "output_validator = COALESCE($7::text, output_validator)"
                    _revision_set = "config_revision = COALESCE($8::text, config_revision)"
                else:
                    _config_set = "config = $4"
                    _is_agent_set = "is_agent = $2"
                    _schema_set = "output_schema = $6"
                    _validator_set = "output_validator = $7"
                    _revision_set = "config_revision = $8"

                _sql = (
                    f"UPDATE workers SET db_path = $1, {_is_agent_set}, status = $3, "
                    f"{_config_set}, updated_at = $5, {_schema_set}, {_validator_set}, "
                    f"{_revision_set} WHERE {scope_clause(9, 10)} AND worker_name = $11 "
                    "RETURNING config, is_agent, output_schema, output_validator, config_revision"
                )  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                # `workers.is_agent` is NOT NULL, so the full-replace branch has
                # to substitute a value for "the caller did not say"; only the
                # merge branch can pass NULL through to COALESCE.
                persisted = await conn.fetchrow(
                    _sql,
                    db_path,
                    is_agent if merge_config else bool(is_agent),
                    status,
                    config,
                    now,
                    output_schema,
                    output_validator,
                    config_revision,
                    tenant_id,
                    effective_subject,
                    worker_name,
                )

                logger.info(
                    "Updated worker: tenant_id=%s, worker_name=%s, worker_id=%s",
                    tenant_id,
                    worker_name,
                    existing_worker_id,
                )

                # Self-heal the historical split (see _rekey_name_scoped_states).
                await self._rekey_name_scoped_states(
                    conn,
                    tenant_id=tenant_id,
                    effective_subject=effective_subject,
                    worker_name=worker_name,
                    worker_id=str(existing_worker_id),
                )

                # Report what the row now HOLDS, not what the caller sent. Under
                # a merge those differ, and every consumer downstream (the PUT
                # response body, the worker_states audit metadata) would
                # otherwise echo the caller's nulls over values the DB kept.
                persisted = persisted or {}
                return {
                    "worker_id": str(existing_worker_id),
                    "worker_name": worker_name,
                    "tenant_id": tenant_id,
                    "subject_id": effective_subject,
                    "db_path": db_path,
                    "is_agent": persisted.get("is_agent", is_agent),
                    "status": status,
                    "config": persisted.get("config") or config,
                    "output_schema": persisted.get("output_schema", output_schema),
                    "output_validator": persisted.get("output_validator", output_validator),
                    "config_revision": persisted.get("config_revision", config_revision),
                    "operational_status": existing["operational_status"],
                    "updated_at": now.isoformat(),
                }

            new_worker_id = worker_id or str(uuid7())

            # There is nothing to preserve on an INSERT, and the column is
            # NOT NULL DEFAULT false (postgres_schema.sql:149) — "the caller
            # did not say" means false for a brand-new row.
            new_is_agent = bool(is_agent)

            try:
                await conn.execute(
                    """
                    INSERT INTO workers
                    (worker_id, tenant_id, subject_id, visibility,
                     tenant_name, worker_name, db_path, is_agent, status, config,
                     created_at, updated_at, output_schema, output_validator,
                     config_revision)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    str(new_worker_id),
                    tenant_id,
                    effective_subject,
                    visibility,
                    tenant_label(tenant_id, tenant_name),
                    worker_name,
                    db_path,
                    new_is_agent,
                    status,
                    config,
                    now,
                    now,
                    output_schema,
                    output_validator,
                    config_revision,
                )
            except psycopg.errors.UniqueViolation:
                # The lookup above missed but the INSERT tripped
                # unique_tenant_worker_name (tenant_id, worker_name). Two cases:
                #
                #   1. A concurrent create for the same (tenant, subject-scope,
                #      worker_name) committed between our SELECT and INSERT (the
                #      classic get-or-create race). Re-SELECT within the caller's
                #      visibility scope now finds the surviving row — return it
                #      idempotently instead of surfacing a 500. (Get semantics:
                #      the first writer wins; we do not clobber its config.)
                #
                #   2. The colliding row is another subject's visibility='user'
                #      worker in this tenant — invisible to the scoped SELECT and
                #      un-ownable by the caller. The re-SELECT finds nothing;
                #      raise a typed conflict the API maps to 409.
                # Selects the row's OWN config/is_agent/columns, not just its id:
                # the winner's record is what survives ("the first writer wins;
                # we do not clobber its config"), so echoing the loser's payload
                # back would have this function report a row that does not exist.
                # Both callers in crud_endpoints build their response body and
                # their worker_states audit metadata off this dict.
                raced = await conn.fetchrow(
                    f"""
                    SELECT worker_id, operational_status, config, is_agent,
                           output_schema, output_validator, config_revision
                    FROM workers
                    WHERE {scope_clause(1, 2)} AND worker_name = $3
                    ORDER BY created_at
                    """,  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                    tenant_id,
                    effective_subject,
                    worker_name,
                )
                if raced is None:
                    logger.warning(
                        "register_worker conflict: worker_name=%s already owned by "
                        "another subject in tenant_id=%s",
                        worker_name,
                        tenant_id,
                    )
                    raise WorkerNameConflictError(tenant_id, worker_name) from None

                raced_worker_id = raced["worker_id"]
                logger.info(
                    "register_worker lost a create race; returning existing worker: "
                    "tenant_id=%s, worker_name=%s, worker_id=%s",
                    tenant_id,
                    worker_name,
                    raced_worker_id,
                )
                return {
                    "worker_id": str(raced_worker_id),
                    "worker_name": worker_name,
                    "tenant_id": tenant_id,
                    "subject_id": effective_subject,
                    "db_path": db_path,
                    # ``.get`` rather than ``[...]``: this recovery row is the
                    # narrowest thing in the function to reach, and a KeyError
                    # here turns a survivable race into a 500.
                    "is_agent": bool(raced.get("is_agent")),
                    "status": status,
                    "config": raced.get("config") or {},
                    "output_schema": raced.get("output_schema"),
                    "output_validator": raced.get("output_validator"),
                    "config_revision": raced.get("config_revision"),
                    "operational_status": raced.get("operational_status"),
                    "updated_at": now.isoformat(),
                }

            logger.info(
                "Registered new worker: tenant_id=%s, worker_name=%s, worker_id=%s",
                tenant_id,
                worker_name,
                new_worker_id,
            )

            # The case this fix exists for: state was written under the NAME
            # before this worker had a worker_id at all.
            await self._rekey_name_scoped_states(
                conn,
                tenant_id=tenant_id,
                effective_subject=effective_subject,
                worker_name=worker_name,
                worker_id=str(new_worker_id),
            )

            return {
                "worker_id": new_worker_id,
                "worker_name": worker_name,
                "tenant_id": tenant_id,
                "subject_id": effective_subject,
                "db_path": db_path,
                "is_agent": new_is_agent,
                "status": status,
                "config": config,
                "config_revision": config_revision,
                "operational_status": "offline",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }

    @with_retry()
    async def swap_prompt(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        worker_name: str,
        new_system_prompt: str,
        expected_current_prompt: Optional[str] = None,
        config_revision: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Atomically replace ONLY ``config['system_prompt']`` on a worker row.

        Implements the OD-4 atomic compare-and-swap. ``output_schema`` (the
        dedicated column AND the ``config.output_schema`` key) is never named in
        the SET clause, so it is structurally immutable across the swap (P-MP-01).

        When ``expected_current_prompt`` is provided the UPDATE only applies if
        the live prompt still equals it (``IS NOT DISTINCT FROM``) — so a
        concurrent change since the caller read the worker yields 0 rows.

        Returns the updated row dict, or ``None`` if no row matched (caller
        distinguishes not-found from CAS-mismatch via a prior ``get_worker``).
        """
        effective_subject = subject_id or tenant_id
        async with get_pg_connection() as conn:
            now = datetime.now(UTC).replace(tzinfo=None)
            returning = (
                "RETURNING worker_id, worker_name, tenant_id, subject_id, visibility, "
                "db_path, is_agent, status, config, created_at, updated_at, "
                "operational_status, output_schema, output_validator, config_revision"
            )
            if expected_current_prompt is not None:
                _sql = (  # nosec B608 — scope_clause returns parameterized SQL
                    "UPDATE workers SET "
                    "config = jsonb_set(config, '{system_prompt}', to_jsonb($1::text)), "
                    "updated_at = $2, config_revision = COALESCE($3, config_revision) "
                    f"WHERE {scope_clause(4, 5)} AND worker_name = $6 "
                    "AND (config->>'system_prompt') IS NOT DISTINCT FROM $7 "
                    f"{returning}"
                )
                args: tuple[Any, ...] = (
                    new_system_prompt,
                    now,
                    config_revision,
                    tenant_id,
                    effective_subject,
                    worker_name,
                    expected_current_prompt,
                )
            else:
                _sql = (  # nosec B608 — scope_clause returns parameterized SQL
                    "UPDATE workers SET "
                    "config = jsonb_set(config, '{system_prompt}', to_jsonb($1::text)), "
                    "updated_at = $2, config_revision = COALESCE($3, config_revision) "
                    f"WHERE {scope_clause(4, 5)} AND worker_name = $6 "
                    f"{returning}"
                )
                args = (
                    new_system_prompt,
                    now,
                    config_revision,
                    tenant_id,
                    effective_subject,
                    worker_name,
                )

            row = await conn.fetchrow(_sql, *args)
            if not row:
                return None
            logger.info(
                "Swapped prompt: tenant_id=%s, worker_name=%s, config_revision=%s",
                tenant_id,
                worker_name,
                config_revision,
            )
            return self._row_to_dict(row)

    @with_retry()
    async def get_worker(
        self, *, tenant_id: str, subject_id: str, worker_name: str
    ) -> Optional[dict[str, Any]]:
        """Get a worker by name visible under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT worker_id, worker_name, tenant_id, subject_id, visibility,
                       db_path, is_agent, status,
                       config, created_at, updated_at, operational_status,
                       output_schema, output_validator, config_revision
                FROM workers
                WHERE {scope_clause(1, 2)} AND worker_name = $3
                """,
                tenant_id,
                subject_id or tenant_id,
                worker_name,
            )

            if not row:
                normalized_name = worker_name.lower()
                row = await conn.fetchrow(
                    f"""
                    SELECT worker_id, worker_name, tenant_id, subject_id, visibility,
                           db_path, is_agent, status,
                           config, created_at, updated_at, operational_status,
                           output_schema, output_validator, config_revision
                    FROM workers
                    WHERE {scope_clause(1, 2)} AND LOWER(worker_name) = $3
                    """,
                    tenant_id,
                    subject_id or tenant_id,
                    normalized_name,
                )

            if not row:
                return None

            return self._row_to_dict(row)

    @with_retry()
    async def get_worker_by_id(
        self, *, tenant_id: str, subject_id: str, worker_id: str
    ) -> Optional[dict[str, Any]]:
        """Get a worker by id visible under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT worker_id, worker_name, tenant_id, subject_id, visibility,
                       db_path, is_agent, status,
                       config, created_at, updated_at, operational_status,
                       output_schema, output_validator, config_revision
                FROM workers
                WHERE {scope_clause(1, 2)} AND worker_id = $3
                """,
                tenant_id,
                subject_id or tenant_id,
                str(worker_id),
            )

            if not row:
                return None

            return self._row_to_dict(row)

    @staticmethod
    def _listable_name_clause() -> str:
        """SQL fragment keeping only names ``list_workers`` is willing to return.

        This used to live ONLY as a Python post-filter applied *after* ``LIMIT``,
        which had two consequences: a page could come back shorter than ``limit``
        for reasons the caller could not see, and any ``COUNT(*)`` taken beside it
        would disagree with the page. Expressing the predicate in SQL lets the
        page query and :meth:`count_workers` share one definition, so a total is
        comparable to the rows it describes.

        A NUL byte is not checked here: a PostgreSQL ``text`` value cannot contain
        one, so that arm of the Python filter was always dead.
        """
        return (
            "worker_name IS NOT NULL AND btrim(worker_name) <> '' "
            "AND octet_length(worker_name) <= 200"
        )

    @classmethod
    def _list_filter_sql(cls, *, prefix: Optional[str], prefix_param: int) -> str:
        """WHERE-tail shared by the page query and the count query.

        The prefix match is ``starts_with()`` rather than ``LIKE``, deliberately:
        ``LIKE`` would treat ``%`` and ``_`` in a caller-supplied prefix as
        wildcards, and worker names contain ``_`` constantly (the n8n cohort op
        derives prefixes like ``fr_planner_``), so ``LIKE`` would silently widen
        the very filter this exists to make exact. It is matched case-insensitively
        to agree with ``get_worker``, which looks names up via ``LOWER()``.
        """
        clause = f" AND {cls._listable_name_clause()}"
        if prefix:
            clause += f" AND starts_with(LOWER(worker_name), LOWER(${prefix_param}))"
        return clause

    @with_retry()
    async def list_workers(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        limit: int = 100,
        offset: int = 0,
        prefix: Optional[str] = None,
    ) -> list[str]:
        """List worker names visible under (tenant_id, subject_id).

        Args:
            tenant_id: Owning tenant; bound into :func:`scope_clause`.
            subject_id: Caller subject for user-visible rows.
            limit: Page size.
            offset: Page offset.
            prefix: Optional case-insensitive worker-name prefix. ``None``/empty
                means no name filter. A supplied value is a real ``WHERE`` — it is
                never accepted and then dropped, which is the defect this argument
                exists to close.
        """
        args: list[Any] = [tenant_id, subject_id or tenant_id, limit, offset]
        if prefix:
            args.append(prefix)
        filter_sql = self._list_filter_sql(prefix=prefix, prefix_param=5)
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""SELECT worker_name FROM workers
                WHERE {scope_clause(1, 2)}{filter_sql}
                ORDER BY worker_name
                LIMIT $3 OFFSET $4""",
                *args,
            )

            # Second net. On real Postgres ``_listable_name_clause()`` has already
            # excluded these rows, and a ``text`` column cannot hold a NUL byte at
            # all — so this loop is a no-op in production. It is kept because
            # tests/db_test_helper.py's in-memory fake parses only ``col = $N``
            # predicates and silently ignores the SQL clause, so removing it would
            # turn test_worker_registry_db_unit.py::test_filters_invalid_names red
            # for a reason that has nothing to do with the database.
            valid_workers = []
            for row in rows:
                worker_name = row["worker_name"]
                if (
                    worker_name
                    and isinstance(worker_name, str)
                    and "\x00" not in worker_name
                    and len(worker_name.encode("utf-8")) <= 200
                    and worker_name.strip()
                ):
                    valid_workers.append(worker_name)

            return valid_workers

    @with_retry()
    async def count_workers(
        self, *, tenant_id: str, subject_id: str, prefix: Optional[str] = None
    ) -> int:
        """Total rows ``list_workers`` would return for this scope, ignoring paging.

        Shares :meth:`_list_filter_sql` with the page query so the number is a
        total *of the same population*, not of a nearby one.
        """
        args: list[Any] = [tenant_id, subject_id or tenant_id]
        if prefix:
            args.append(prefix)
        filter_sql = self._list_filter_sql(prefix=prefix, prefix_param=3)
        async with get_pg_connection() as conn:
            total = await conn.fetchval(
                f"""SELECT COUNT(*) FROM workers
                WHERE {scope_clause(1, 2)}{filter_sql}""",
                *args,
            )
            return int(total or 0)

    @with_retry()
    async def delete_worker(self, *, tenant_id: str, subject_id: str, worker_name: str) -> bool:
        """Delete a worker visible+mutable under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM workers WHERE {scope_clause(1, 2)} AND worker_name = $3",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
                worker_name,
            )

            deleted_count = int(result.split()[-1]) if result else 0

            if deleted_count > 0:
                logger.info(
                    "Deleted worker: tenant_id=%s, worker_name=%s",
                    tenant_id,
                    worker_name,
                )
                return True

            return False

    @with_retry()
    async def delete_all_workers(self, *, tenant_id: str, subject_id: str) -> int:
        """Delete all workers visible+mutable under (tenant_id, subject_id).

        Tenant-visible rows are deleted regardless of caller subject; user-
        visible rows are only deleted when owned by ``subject_id``. Other
        tenants are never touched.
        """
        async with get_pg_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM workers WHERE {scope_clause(1, 2)}",  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                tenant_id,
                subject_id or tenant_id,
            )

            deleted_count = int(result.split()[-1]) if result else 0

            logger.info(
                "Deleted all workers visible to tenant=%s subject=%s: count=%d",
                tenant_id,
                subject_id,
                deleted_count,
            )

            return deleted_count

    @with_retry()
    async def get_all_workers(self, *, tenant_id: str, subject_id: str) -> list[dict[str, Any]]:
        """Get all workers visible under (tenant_id, subject_id) with full details."""
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT worker_id, worker_name, tenant_id, subject_id, visibility,
                       db_path, is_agent, status,
                       config, created_at, updated_at, operational_status,
                       output_schema, output_validator, config_revision
                FROM workers
                WHERE {scope_clause(1, 2)}
                ORDER BY worker_name
                """,
                tenant_id,
                subject_id or tenant_id,
            )

            return [self._row_to_dict(row) for row in rows]

    @with_retry()
    async def update_worker_status(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        worker_name: str,
        status: str,
        operational_status: Optional[str] = None,
    ) -> bool:
        """Update worker status if visible+mutable under (tenant_id, subject_id)."""
        async with get_pg_connection() as conn:
            now = datetime.now(UTC).replace(tzinfo=None)

            if operational_status:
                _sql = f"UPDATE workers SET status = $1, operational_status = $2, updated_at = $3 WHERE {scope_clause(4, 5)} AND worker_name = $6"  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                result = await conn.execute(
                    _sql,
                    status,
                    operational_status,
                    now,
                    tenant_id,
                    subject_id or tenant_id,
                    worker_name,
                )
            else:
                _sql = f"UPDATE workers SET status = $1, updated_at = $2 WHERE {scope_clause(3, 4)} AND worker_name = $5"  # nosec B608 — scope_clause returns parameterized SQL, see src/repositories/visibility.py
                result = await conn.execute(
                    _sql,
                    status,
                    now,
                    tenant_id,
                    subject_id or tenant_id,
                    worker_name,
                )

            updated_count = int(result.split()[-1]) if result else 0
            return updated_count > 0

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Convert a workers row dict to the response shape."""
        config_data = row["config"] or {}
        if isinstance(config_data, str):
            try:
                config_data = json.loads(config_data)
            except json.JSONDecodeError:
                logger.warning(
                    "Worker %s config is not valid JSON; defaulting to empty dict",
                    row.get("worker_name"),
                )
                config_data = {}

        return {
            "worker_id": str(row["worker_id"]),
            "worker_name": row["worker_name"],
            "tenant_id": row.get("tenant_id"),
            "subject_id": row.get("subject_id"),
            "visibility": row.get("visibility", "tenant"),
            "db_path": row["db_path"],
            "is_agent": row["is_agent"],
            "status": row["status"],
            "config": config_data,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "operational_status": row["operational_status"] or "offline",
            "output_schema": row["output_schema"],
            "output_validator": row["output_validator"],
            "config_revision": row.get("config_revision"),
        }
