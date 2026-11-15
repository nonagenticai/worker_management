"""Database configuration module for PostgreSQL.

PostgreSQL is the ONLY supported database.
Uses psycopg v3 async with connection pooling.
"""

import contextlib
import logging
import os
import re
import ssl
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString, Optional, cast

import psycopg
from psycopg import pq
from psycopg.abc import Buffer  # noqa: TC002 - runtime annotation on a Loader subclass
from psycopg.adapt import Loader
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg.types.string import TextLoader
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from src.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# PostgreSQL connection pool (initialized on first use)
_pg_pool: Optional[AsyncConnectionPool] = None


# --- run_migrations() startup-safety guards ---------------------------------
#
# How long a single migration may wait for a lock before failing fast. Sized
# against the startupProbe that used to kill the pod instead: failureThreshold
# 30 x periodSeconds 10 = 300s, plus terminationGracePeriodSeconds 60 = 359s
# to exit 137. 10s is two orders of magnitude inside that, so even a chain in
# which every one of the 23 files blocks still finishes well under the probe
# budget, while being far longer than any lock a healthy peer legitimately
# holds.
_MIGRATION_LOCK_TIMEOUT = "10s"

# Caps an orphaned migrator. A pod SIGKILLed mid-migration leaves its server
# connection idle IN a transaction holding ACCESS EXCLUSIVE; this makes
# Postgres reap it rather than letting it block every subsequent boot.
_MIGRATION_IDLE_TX_TIMEOUT = "60s"

# A migration whose first 40 lines carry this directive runs OUTSIDE a
# transaction, in autocommit. The only reason to want that is
# `CREATE INDEX CONCURRENTLY` / `DROP INDEX CONCURRENTLY`, which PostgreSQL
# refuses inside a transaction block. Such a file gets NO lock_timeout guard
# (SET LOCAL needs a transaction) and NO rollback -- CONCURRENTLY takes only
# SHARE UPDATE EXCLUSIVE so it does not block writers, but a failure leaves an
# INVALID index behind that `IF NOT EXISTS` will then skip forever.
#
# ⚠️ A no-transaction file must contain EXACTLY ONE statement -- see
# `_no_transaction_violation`. The drop-then-create pair this comment used to
# recommend cannot run: psycopg sends a multi-statement string through the
# simple-query protocol, which wraps it in an implicit transaction, and
# PostgreSQL then rejects it with `DROP INDEX CONCURRENTLY cannot run inside a
# transaction block`. Measured against postgres:18, 2026-08-28. So the correct
# shape is one statement per file:
#
#     -- migrate:no-transaction
#     CREATE INDEX CONCURRENTLY idx_foo ON foo (bar);
#
# which leaves the INVALID-leftover problem to be solved where it can actually
# be solved -- migration 028, which reaps `indisvalid = false` indexes on every
# boot (`migrate:always`).
#
# 026 and 027 are deliberately NOT converted: both are applied and valid in
# production (by hand, 2026-08-27) and `postgres_schema.sql` already declares
# them for fresh databases, so a drop-then-create would destroy a working
# index and reopen a window without it, to prove a point.
_MIGRATION_NO_TRANSACTION_DIRECTIVE = "migrate:no-transaction"

# A migration whose header carries this directive is re-executed on EVERY boot
# and is never skipped on the strength of the ledger.
#
# The ledger exists because an already-applied file re-taking ACCESS EXCLUSIVE
# every boot is what caused the 2026-08-27 convoy, so this is a narrow opt-out,
# not a general escape hatch: a DDL file marked `migrate:always` re-creates the
# exact lock storm the ledger removes.
#
# It exists for files whose value IS the re-assertion -- a control loop rather
# than a one-time change. The only current holder is 028, which reaps INVALID
# indexes: recorded once and skipped thereafter it would never reap anything
# again, which is worse than not existing, because it would read like
# protection while providing none. Its steady-state cost is one catalog SELECT.
_MIGRATION_ALWAYS_DIRECTIVE = "migrate:always"

# --- What the ledger ASSUMES about the files it retires ---------------------
#
# The ledger assumes every migration is a ONE-TIME change: applied once,
# recorded, never executed again. That assumption is not free, and it is worth
# stating because it is invisible at the call site.
#
# Before the ledger, the runner re-executed every file on every boot, so any
# file whose predicate still matched rows kept repairing them. Three files in
# the current chain read like standing repairs and are NOT -- each for a
# different reason, and each reason was measured on 2026-08-28 rather than
# assumed:
#
#   * 025 is BOUNDED. Its DELETE carries
#     `AND created_at < TIMESTAMPTZ '2026-08-26 00:00:00+00'`, so a row created
#     after the incident window cannot match. Re-running it was provably a
#     no-op, and its own test treats re-running as the hazard that bound exists
#     to contain.
#   * 017 is GUARDED AT THE BOUNDARY. Its DELETE predicate is unbounded, but
#     `src/api/security_validation.py` rejects reserved names on the
#     crud/message/state endpoints before a row can be written. The two sets
#     were compared programmatically: 24 names in 017's DELETE, 24 in the
#     validator, zero difference in either direction. 0 rows match on prod.
#   * 024 is UNREPRESENTABLE. Its repair is reachable only while `tenant_id` is
#     text; prod `information_schema` reports `workers.tenant_id` is `uuid`, so
#     a non-canonical spelling cannot be stored at all. Its test reaches the
#     repair path only by ALTERing the column back to varchar first.
#
# ⚠️ A future repair migration that is NONE of those three -- unbounded in its
# predicate, unguarded at the API boundary, and still representable in the
# schema -- is a standing repair, and the ledger WILL silently retire it into a
# one-time historical event on its second boot, with every test still green.
# Such a file needs an explicit re-run mechanism (`migrate:always`), and the
# author must justify the exemption: an `always` DDL file re-creates the exact
# lock storm this ledger removes.
#
# Ledger of which files have actually been applied, keyed by FILENAME.
#
# Deliberately NOT the pre-existing `schema_migrations` table. That one is
# keyed on a free-text `version` whose values are inconsistent across the
# chain -- '1.0.4', '0.8.0', '010', '026' all appear, and 013 exists purely to
# retro-fit a UNIQUE constraint and backfill rows whose INSERT had failed. Any
# skip decision keyed on it would be guessing, and a wrong skip means a
# migration silently never runs on a fresh database. A separate table keyed on
# the filename the runner actually iterates is exact by construction, and
# leaves the legacy bookkeeping untouched for whatever still reads it.
_MIGRATION_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migration_files (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""

# Matches a top-level `BEGIN;` / `COMMIT;` occupying a line on its own.
#
# Anchored to the start/end of the file precisely so a `DO $$ BEGIN ... END $$`
# block can never match: that BEGIN is mid-file and carries no semicolon.
# Verified against all 27 files in migrations/ -- the 15 that are wrapped have
# exactly one top-level BEGIN and one top-level COMMIT, and the COMMIT is
# always the final statement. No file commits mid-way.
_LEADING_BEGIN_RE = re.compile(r"\A(\s*(?:--[^\n]*\n|\s)*)BEGIN\s*;", re.IGNORECASE)
_TRAILING_COMMIT_RE = re.compile(r"COMMIT\s*;\s*\Z", re.IGNORECASE)


def _declares_directive(sql_text: str, directive: str) -> bool:
    """True only if ``directive`` occupies a whole SQL comment line of its own.

    ⛔ Deliberately NOT a substring test over the header.

    A file that MENTIONS a directive is not a file that DECLARES it, and in this
    repository that distinction is not hypothetical: migration files here carry
    long prose headers that discuss the runner's own mechanics. 027's header
    already quotes `CREATE INDEX CONCURRENTLY IF NOT EXISTS` and explains the
    INVALID-index trap; 028's header has to name `migrate:always` to explain why
    it carries it. Under a substring test, any such file that mentioned
    `migrate:no-transaction` within its first 40 lines -- including one saying it
    deliberately does NOT use it -- would silently be run with no lock_timeout
    guard and no rollback.

    Anchored on the whole stripped line, so `-- migrate:always` declares and
    `-- unlike 028, this file is not migrate:always` does not.
    """
    return any(
        re.fullmatch(rf"--\s*{re.escape(directive)}\s*", line.strip())
        for line in sql_text.splitlines()[:40]
    )


def _wants_no_transaction(sql_text: str) -> bool:
    """True when the file opts out of the runner's per-file transaction."""
    return _declares_directive(sql_text, _MIGRATION_NO_TRANSACTION_DIRECTIVE)


def _runs_always(sql_text: str) -> bool:
    """True when the file is a control loop and must never be skipped."""
    return _declares_directive(sql_text, _MIGRATION_ALWAYS_DIRECTIVE)


def _no_transaction_violation(sql_text: str) -> Optional[str]:
    """Return why a ``no-transaction`` file cannot run, or ``None`` if it can.

    psycopg sends a multi-statement string through the simple-query protocol,
    which wraps the whole string in an implicit transaction -- so a two-statement
    "non-transactional" file fails with `DROP INDEX CONCURRENTLY cannot run
    inside a transaction block`, a message that points at transactions rather
    than at the real cause, which is the semicolon. Measured against postgres:18
    on 2026-08-28.

    Catching it here names the actual problem at the point of decision, before
    the statement reaches a backend, instead of letting a file that can never
    apply be logged as a routine migration failure on every boot -- which is
    exactly how 026 and 027 went unnoticed for weeks.

    Dollar-quoting is refused rather than parsed: a ``$$ ... $$`` body may
    contain semicolons, so counting statements around one is not reliable, and
    nothing needs a dollar-quoted CONCURRENTLY file (a ``DO`` block cannot run
    CONCURRENTLY anyway).
    """
    body = "\n".join(line for line in sql_text.splitlines() if not line.strip().startswith("--"))
    if re.search(r"\$[A-Za-z_]*\$", body):
        return "it is dollar-quoted, which this runner will not statement-split"
    statements = [s for s in body.split(";") if s.strip()]
    if len(statements) > 1:
        return (
            f"it contains {len(statements)} statements; a `"
            f"{_MIGRATION_NO_TRANSACTION_DIRECTIVE}` file must contain exactly one, "
            "because psycopg wraps a multi-statement string in an implicit "
            "transaction and PostgreSQL then refuses CONCURRENTLY inside it"
        )
    return None


def _strip_outer_transaction(sql_text: str) -> str:
    """Remove a file's own outer ``BEGIN;`` / ``COMMIT;`` so the runner owns it.

    WHY THIS EXISTS (production incident, 2026-08-27)
    -------------------------------------------------
    The runner wraps each file in ``conn.transaction()``. psycopg had already
    opened a transaction for the session-level ``SET`` above, so that wrapper
    was a SAVEPOINT, and each file's own ``BEGIN`` landed inside an open
    transaction -- **668 occurrences of "WARNING: there is already a
    transaction in progress"** in the worker4 postgres log. The BEGIN was a
    no-op; the file's ``COMMIT`` then committed the OUTER transaction.

    That matters because ``DATABASE_URL`` points at PgBouncer in
    ``pool_mode = transaction``, which pins a server connection only for the
    duration of a transaction. Committing mid-chain hands the connection back,
    so subsequent files are not guaranteed to run on the backend that received
    the session-level ``SET lock_timeout``. Measured consequence: migration 011
    acquired ACCESS EXCLUSIVE on ``workers``, then blocked on ``worker_states``
    for **67 minutes** with no lock timeout ever firing, while on other boots
    the same file failed cleanly at exactly 10.0s. A queued ACCESS EXCLUSIVE
    blocks every later lock request, so 19 sessions piled up behind it -- all
    ``workers`` reads and all ``worker_states`` INSERTs -- until the backend
    was cancelled by hand.

    With the file's own boundaries removed, the runner owns the transaction,
    ``SET LOCAL`` is guaranteed to apply to the same backend as the DDL (a
    transaction cannot migrate between backends), and no COMMIT can release
    the connection mid-chain. This supersedes the "deliberately NOT SET LOCAL"
    reasoning that used to sit at the call site: SET LOCAL was unusable only
    *because* of the embedded COMMIT this function removes.
    """
    stripped = _LEADING_BEGIN_RE.sub(r"\1", sql_text, count=1)
    return _TRAILING_COMMIT_RE.sub("", stripped).rstrip()


async def _ensure_migration_ledger(conn: psycopg.AsyncConnection) -> None:
    """Create the filename ledger if absent. Cheap: a tiny, uncontended table."""
    async with conn.transaction():
        await conn.execute(cast("LiteralString", _MIGRATION_LEDGER_DDL))


async def _applied_filenames(conn: psycopg.AsyncConnection) -> set[str]:
    """Filenames already applied, so a re-run is a no-op instead of a lock wait.

    This is what stops an already-applied file from re-taking ACCESS EXCLUSIVE
    on a hot table every single boot. Measured before this change: 8 of 26
    files failed on EVERY pod start with `canceling statement due to lock
    timeout`, 7 of them already-applied `ADD COLUMN IF NOT EXISTS` no-ops that
    still demand the lock -- ~160s of boot spent waiting for locks that were
    never going to be granted, and a warning list that made a genuine failure
    indistinguishable from the routine noise.

    Only files this runner has SEEN succeed are listed. A database that
    pre-dates the ledger therefore re-runs everything once; the files that
    succeed are recorded and never run again, and the ones that fail keep
    being retried, which is the correct behaviour for a migration that has
    genuinely not been applied. Nothing is assumed about the legacy
    `schema_migrations` rows.
    """
    async with conn.transaction():
        rows = await (await conn.execute("SELECT filename FROM schema_migration_files")).fetchall()
    # Rows are dicts, not tuples: the pool sets `row_factory=dict_row` (see
    # `init_pg_pool` conn_kwargs). Indexing by position raises `KeyError: 0`.
    # ty infers a tuple from psycopg's signature, so cast -- same treatment as
    # `_PsycopgConnectionWrapper.fetchval`.
    return {cast("dict[str, Any]", row)["filename"] for row in rows}


async def _apply_outside_transaction(
    conn: psycopg.AsyncConnection,
    sql_path: Path,
    sql_text: str,
) -> None:
    """Run a file in autocommit, for `CREATE INDEX CONCURRENTLY`.

    PostgreSQL refuses CONCURRENTLY inside a transaction block, and the runner
    wraps every other file in one -- which is why migrations 026 and 027 were
    written as plain `CREATE INDEX` and then failed on every boot for weeks
    against a table too busy to ever grant SHARE.

    No `SET LOCAL` guard is possible here (there is no transaction to scope it
    to) and none is wanted: CONCURRENTLY takes SHARE UPDATE EXCLUSIVE, which
    does not conflict with the writers a plain CREATE INDEX would block. The
    cost is that a failure is not rolled back, so the file must be written to
    tolerate its own INVALID leftovers -- drop, then create.

    The ledger row is written after the fact rather than atomically with the
    DDL, for the same reason: there is no transaction to share. A crash in the
    gap re-runs the file next boot, which the drop-then-create shape makes safe.
    """
    previous = conn.autocommit
    await conn.set_autocommit(True)
    try:
        await conn.execute(cast("LiteralString", sql_text))
        await conn.execute(
            "INSERT INTO schema_migration_files (filename) VALUES (%s) "
            "ON CONFLICT (filename) DO NOTHING",
            (sql_path.name,),
        )
    finally:
        # RESTORE, do not force False. `init_pg_pool` opens this pool with
        # `autocommit=True` (see conn_kwargs), so forcing False here would hand
        # a silently different connection back to the pool.
        await conn.set_autocommit(previous)


# --- Why there is deliberately NO advisory lock around the migration loop ----
#
# The obvious hardening -- pg_advisory_lock(<const>) before the loop,
# pg_advisory_unlock after -- is UNSAFE in this deployment, and the two
# transaction-scoped variants are either unsafe or a false guarantee:
#
# 1. Session-scoped pg_advisory_lock(): worker_management's DATABASE_URL is
#    the PgBouncer service on port 6432, and that pooler runs
#    `pool_mode = transaction` (gitops clusters/strato/services/pgbouncer/
#    pgbouncer.ini:104) with no server_reset_query -- PgBouncer skips DISCARD
#    ALL in transaction pooling by default. A session-scoped advisory lock is
#    released only by explicit unlock or backend exit, so a pod SIGKILLed
#    mid-migration would strand the lock on a server connection that PgBouncer
#    then hands to unrelated clients, and every future migrator would block on
#    it forever. idle_in_transaction_session_timeout cannot save us: the
#    stranded connection is idle, not idle-in-transaction. That is strictly
#    worse than the bug being fixed.
#
# 2. pg_advisory_xact_lock() wrapping the WHOLE loop in one transaction: safe
#    with respect to pooling, but it holds every migration's ACCESS EXCLUSIVE
#    lock across all 11 tables for the duration of the entire chain instead of
#    releasing per file, which enlarges exactly the blast radius we are trying
#    to shrink. It also makes the whole chain all-or-nothing, discarding the
#    per-file isolation the loop is built around.
#
# 3. pg_advisory_xact_lock() inside each per-file transaction: safe, but 12 of
#    the 23 migration files issue their own `COMMIT;`, which releases the
#    xact-scoped lock partway through the file. It would read like
#    serialisation while silently not serialising over half the chain -- the
#    kind of false guarantee that causes the next incident.
#
# lock_timeout already removes the failure mode the advisory lock was meant to
# address: concurrent migrators (guaranteed by maxSurge: 1) no longer pile up
# unboundedly, because each blocked migration now fails fast into the
# non-strict handler and the boot completes. Every migration is idempotent, and
# DDL is transactional in PostgreSQL, so a timed-out migration rolls back
# cleanly and is retried on the next boot. Correctness over completeness.


class _PsycopgConnectionWrapper:
    """Wraps a psycopg AsyncConnection to provide asyncpg-style API.

    The rest of the codebase uses asyncpg-style methods (fetch, fetchrow,
    execute, fetchval) with $N positional placeholders.  This wrapper
    translates to psycopg cursor calls so every DB module works unchanged.
    """

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn

    @staticmethod
    def _translate_placeholders(sql: str) -> LiteralString:
        """Convert ``$1, $2, ...`` numbered placeholders to psycopg named ``%(pN)s``.

        Named placeholders preserve asyncpg's ``$N`` semantics: a number may
        appear out of textual order (e.g. ``SET x = $3 ... WHERE y = $1``) or be
        reused (``... $3 ... <= $3``), and each occurrence binds to the same arg.
        A naive ``$N`` -> ``%s`` positional substitution silently mis-binds such
        queries (psycopg fills ``%s`` in textual order, ignoring the number).
        """
        return cast("LiteralString", re.sub(r"\$(\d+)", r"%(p\1)s", sql))

    # ``$N`` cast to a PostgreSQL array type, e.g. ``$1::text[]`` or
    # ``$2 :: int [ ]``. Captures the placeholder number.
    _ARRAY_CAST_RE = re.compile(r"\$(\d+)\s*::\s*[A-Za-z_]\w*\s*\[\s*\]")

    @classmethod
    def _array_param_indices(cls, sql: str) -> set[int]:
        """Placeholder numbers explicitly cast to a PostgreSQL array.

        A bare Python ``list`` is ambiguous: it may be a JSONB column value
        (``nodes = $5`` where ``$5`` is ``[]``) or a native array parameter
        (``col = ANY($1::text[])``). asyncpg disambiguated from server-side
        type oids; this wrapper has only the SQL text, so an explicit array
        cast on the placeholder is the signal to pass the list through as a
        native PG array instead of wrapping it as ``Jsonb`` (``jsonb`` cannot
        cast to ``text[]``).
        """
        return {int(m.group(1)) for m in cls._ARRAY_CAST_RE.finditer(sql)}

    @classmethod
    def _adapt_args(cls, sql: str, args: tuple[Any, ...]) -> Optional[dict[str, Any]]:
        """Map positional args to the ``%(pN)s`` names used by the SQL.

        Wraps dict/list values with ``Jsonb()`` for JSONB columns (asyncpg
        auto-serializes; psycopg3 needs it explicit). The exception is a
        ``list`` whose placeholder is explicitly array-cast in ``sql`` (see
        :meth:`_array_param_indices`): it is passed through unwrapped so
        psycopg adapts it to a native PG array. ``args[0]`` -> ``p1``,
        ``args[1]`` -> ``p2``, ... Returns ``None`` for no args.
        """
        if not args:
            return None
        array_params = cls._array_param_indices(sql)
        adapted: dict[str, Any] = {}
        for i, a in enumerate(args):
            if isinstance(a, list) and (i + 1) in array_params:
                adapted[f"p{i + 1}"] = a  # native PG array param
            elif isinstance(a, (dict, list)):
                adapted[f"p{i + 1}"] = Jsonb(a)
            else:
                adapted[f"p{i + 1}"] = a
        return adapted

    async def execute(self, sql: str, *args: Any) -> str:
        """Execute a statement and return a status string like 'INSERT 0 1'."""
        translated = self._translate_placeholders(sql)
        cur = await self._conn.execute(translated, self._adapt_args(sql, args))
        # Build an asyncpg-compatible status string from the cursor statusmessage
        msg = cur.statusmessage or ""
        return msg

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        translated = self._translate_placeholders(sql)
        cur = await self._conn.execute(translated, self._adapt_args(sql, args))
        rows = await cur.fetchall()
        return [dict(r) for r in rows] if rows else []

    async def fetchrow(self, sql: str, *args: Any) -> Optional[dict[str, Any]]:
        """Execute a query and return the first row as a dict, or None."""
        translated = self._translate_placeholders(sql)
        cur = await self._conn.execute(translated, self._adapt_args(sql, args))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Execute a query and return the first column of the first row."""
        translated = self._translate_placeholders(sql)
        cur = await self._conn.execute(translated, self._adapt_args(sql, args))
        row = await cur.fetchone()
        if row is None:
            return None
        # row is a dict (dict_row factory); ty infers tuple — cast for safety
        return next(iter(cast("dict[str, Any]", row).values()))

    def transaction(self) -> Any:
        """Return a transaction context manager (psycopg v3 style)."""
        return self._conn.transaction()


# ---------------------------------------------------------------------------
# uuid columns must still read back as `str`.
# ---------------------------------------------------------------------------
# Migration 024 made `tenant_id` a real Postgres `uuid`. By default psycopg3
# then hands back a `uuid.UUID` OBJECT rather than the `str` every caller in
# this service has always received -- and that is a silent change, not a loud
# one: `row["tenant_id"] == some_str` simply becomes False, a dict keyed by
# tenant stops matching, and `json.dumps` raises somewhere far from the query.
# Nothing type-checks it, because the DB layer is untyped `dict[str, Any]`.
#
# Registering the text loader for the `uuid` OID keeps the Python side exactly
# as it was: the DATABASE gets a real type (and rejects `n8n-dev`), while ~40
# call sites, the API schemas and the outbound `X-Tenant-Id` headers need no
# change at all. Writes are unaffected -- Postgres parses a `str` into a uuid on
# the way in, which is what makes the column a guard in the first place.
#
# This mirrors what `sa.Uuid(as_uuid=False)` does for the SQLAlchemy services.
class _UuidBinaryAsText(Loader):
    """Binary-format `uuid` -> canonical lowercase `str`.

    psycopg picks text or binary format per query, so registering only the text
    loader would leave binary results returning `uuid.UUID` — the same silent
    inconsistency the text loader exists to remove, but intermittent and so
    harder to find. `TextBinaryLoader` is NOT the answer: it UTF-8-decodes the
    payload, and a uuid arrives in binary as 16 raw bytes, so it raises
    `UnicodeDecodeError` (measured, not assumed).
    """

    format = pq.Format.BINARY

    def load(self, data: Buffer) -> str:
        return str(uuid.UUID(bytes=bytes(data)))


async def _uuid_reads_back_as_text(conn: psycopg.AsyncConnection) -> None:
    """Return `uuid` columns as `str`, as every caller here already expects.

    Must be `async`: `AsyncConnectionPool` awaits its `configure` callback, and a
    plain `def` here makes every `getconn()` fail with `PoolTimeout` after 30s
    rather than with anything that names the callback.

    Both formats are registered. psycopg chooses text or binary per query, so
    registering only the text loader would leave binary-format results returning
    `uuid.UUID` — the same silent inconsistency, now intermittent and therefore
    harder to find than the problem it was fixing.
    """
    conn.adapters.register_loader("uuid", TextLoader)
    conn.adapters.register_loader("uuid", _UuidBinaryAsText)


def _ca_bundle() -> str | None:
    """The configured CA bundle, or None when it is not there.

    The estate image mounts one at DB_CA_CERT_PATH; a clone usually has none, and
    passing a missing path to create_default_context raises FileNotFoundError — which
    is how an open-source checkout died before it could connect.
    """
    path = getattr(get_settings(), "db_ca_cert_path", "") or ""
    if path and os.path.exists(path):
        return path
    logger.warning("CA bundle %r not found; using the system trust store", path)
    return None


def _build_ssl_context(settings: Any, sslmode: str) -> ssl.SSLContext | None:
    """The SSL context for `sslmode`, or None when SSL is disabled.

    Extracted from `init_pg_pool` so that routine stays readable, and because the
    two places that used to hardcode the CA path now share one fallback.
    """
    try:
        if sslmode == "disable":
            logger.info("SSL disabled for PostgreSQL connection")
            return None
        if sslmode in ("allow", "prefer"):
            # Try SSL but fall back to non-SSL if it fails
            ctx = ssl.create_default_context(cafile=_ca_bundle())
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            logger.info("SSL enabled with allow/prefer mode")
            return ctx
        # require, verify-ca, or verify-full modes
        logger.info("Setting up SSL context for mode: %s", sslmode)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        ca = _ca_bundle()
        if ca is not None:
            ctx.load_verify_locations(cafile=ca)
        logger.info("Loaded CA certificate")

        # Load client certificate and key for mutual TLS authentication
        client_cert = settings.client_cert_path
        client_key = settings.client_key_path
        logger.info(
            "Checking for client certificates: cert=%s (exists=%s), key=%s (exists=%s)",
            client_cert,
            os.path.exists(client_cert),
            client_key,
            os.path.exists(client_key),
        )
        if os.path.exists(client_cert) and os.path.exists(client_key):
            ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
            logger.info("Loaded client certificate for mutual TLS: %s", client_cert)
        else:
            logger.warning("Client certificates not found, mutual TLS not configured")

        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    except Exception as ssl_err:
        logger.error("Error creating SSL context: %s", ssl_err, exc_info=True)
        raise
    return ctx


async def init_pg_pool() -> AsyncConnectionPool:
    """Initialize PostgreSQL connection pool using psycopg v3 async.

    Returns:
        AsyncConnectionPool: The initialized connection pool

    Raises:
        ValueError: If DATABASE_URL is not set
        Exception: If connection fails
    """
    global _pg_pool  # noqa: PLW0603 -- lazy-init singleton pool

    if _pg_pool is not None:
        logger.debug("PostgreSQL pool already initialized")
        return _pg_pool

    settings = get_settings()
    db_url_raw = settings.database_url
    if not db_url_raw:
        logger.error("DATABASE_URL not set in environment")
        raise ValueError("DATABASE_URL environment variable is required")

    url = make_url(db_url_raw)
    host = url.host or "localhost"
    port = url.port or 5432
    database = url.database or "worker_management"
    user = url.username or ""
    password = url.password or ""

    pool_min = settings.postgres_pool_min
    pool_max = settings.postgres_pool_max
    connect_timeout = settings.postgres_connect_timeout

    # Create SSL context based on POSTGRES_SSLMODE setting
    sslmode = settings.postgres_sslmode.lower()
    logger.info("PostgreSQL SSL mode: %s", sslmode)
    ssl_context = _build_ssl_context(settings, sslmode)

    # Build psycopg connection string
    sslmode_conninfo = "disable" if ssl_context is None else sslmode
    conninfo = (
        f"host={host} port={port} dbname={database} "
        f"user={user} password={password} "
        f"sslmode={sslmode_conninfo} "
        f"connect_timeout={connect_timeout}"
    )

    # psycopg connection kwargs
    conn_kwargs: dict[str, Any] = {
        "row_factory": dict_row,
        "autocommit": True,
    }
    if ssl_context is not None:
        conn_kwargs["sslcontext"] = ssl_context

    try:
        logger.info(
            "Initializing PostgreSQL pool: host=%s, port=%s, db=%s, user=%s",
            host,
            port,
            database,
            user,
        )

        _pg_pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=pool_min,
            max_size=pool_max,
            kwargs=conn_kwargs,
            max_idle=300,
            check=AsyncConnectionPool.check_connection,
            configure=_uuid_reads_back_as_text,
        )
        await _pg_pool.open()

        # Test the connection
        async with _pg_pool.connection() as conn:
            cur = await conn.execute(cast("LiteralString", "SELECT version()"))
            row = await cur.fetchone()
            # dict_row factory makes fetchone() return dict, not tuple
            version = cast("str", cast("dict[str, Any]", row)["version"]) if row else "unknown"
            logger.info("PostgreSQL connection successful: %s", version)

        logger.info(
            "PostgreSQL pool initialized: min=%d, max=%d connections",
            pool_min,
            pool_max,
        )
        return _pg_pool

    except Exception:
        logger.exception("Failed to initialize PostgreSQL pool")
        _pg_pool = None
        raise


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------
#
# Background: production was hitting a 500 on ``GET /workers`` because
# ``workers.tenant_id`` did not exist in the live database. The Kubernetes
# init-container approach (``configmap-migrations.yaml`` in the gitops repo)
# was hand-maintained and silently fell out of sync — migrations 008, 009,
# and 010 were committed to ``migrations/`` in this repo but were never
# copied into the ConfigMap, so they never ran. This mirrors the
# ``secrets-service`` PR #1 fix, but here the migration unit is plain SQL
# files, not Alembic revisions.
#
# We now ship the SQL files inside the application image (``COPY migrations/``
# in the Dockerfile) and apply them in lifespan startup against the same
# pool the app uses. Every migration is required to be idempotent (uses
# ``IF NOT EXISTS`` / ``DO`` blocks), so running the full set on every
# startup is safe and re-runnable. The existing ``schema_migrations`` table
# records which versions have been applied, but the source of truth for
# "did this column get added?" is the database itself, not a row in the
# tracking table.


def _resolve_migrations_dir(override: Optional[Path] = None) -> Path:
    """Locate the directory holding ordered ``NNN_*.sql`` migration files.

    Resolution order:
    1. ``override`` argument (used by tests to point at a temp dir).
    2. ``WORKER_MGMT_MIGRATIONS_DIR`` env var (escape hatch for ops).
    3. ``<repo_root>/migrations`` derived from ``__file__`` — works in dev,
       in CI, and inside the production container as long as the Dockerfile
       copies ``migrations/`` to ``/app/migrations``.
    """
    if override is not None:
        return override

    env_override = os.environ.get("WORKER_MGMT_MIGRATIONS_DIR")
    if env_override:
        return Path(env_override)

    # src/db/db_config.py -> src/db -> src -> repo_root
    return Path(__file__).resolve().parent.parent.parent / "migrations"


def _list_migration_files(migrations_dir: Path) -> list[Path]:
    """Return migration files sorted by filename (lexicographic = ordered)."""
    if not migrations_dir.is_dir():
        return []
    return sorted(p for p in migrations_dir.iterdir() if p.suffix == ".sql")


async def run_migrations(
    *,
    migrations_dir: Optional[Path] = None,
    pool: Optional[AsyncConnectionPool] = None,
    strict: Optional[bool] = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Apply every not-yet-applied ``migrations/*.sql`` file, in filename order.

    Each file is executed inside its own transaction, whose boundaries this
    runner owns -- a file's own ``BEGIN;``/``COMMIT;`` is stripped first (see
    :func:`_strip_outer_transaction`). A file may opt out of the transaction
    entirely with the ``migrate:no-transaction`` directive, which is what makes
    ``CREATE INDEX CONCURRENTLY`` possible.

    A file that has already been applied is skipped on the strength of the
    ``schema_migration_files`` ledger, so a re-run costs nothing rather than
    re-taking ACCESS EXCLUSIVE on a hot table. Migrations are still required to
    be idempotent (``ADD COLUMN IF NOT EXISTS``, ``CREATE INDEX IF NOT
    EXISTS``, ``DO`` blocks gated on ``pg_constraint`` lookups, etc.), because
    a database that pre-dates the ledger re-runs the whole set once.

    A file may opt out of that skip with ``migrate:always``, for the narrow case
    of a control loop whose value is continuous re-assertion rather than a
    one-time change (migration 028, which reaps INVALID indexes). Both
    directives are recognised only as a whole comment line of their own -- see
    :func:`_declares_directive` for why a substring test is unsafe here.

    Two error policies, controlled by ``strict``:

    * ``strict=True`` — first failing migration aborts the whole run and
      re-raises. Used by tests pinning the canonical, Postgres-clean path.
    * ``strict=False`` (default in production) — log each failure as a
      warning and continue with the next file. This matches the legacy
      init-container behavior (``psql -f $f || echo "...may have errors"``),
      and is what kept prod alive while early migrations carried SQLite
      artifacts (``CREATE TRIGGER IF NOT EXISTS``, ``hex(randomblob())``,
      etc.) that PostgreSQL rejects. Without this leniency, a single
      broken legacy file would block migration 010 — which is the actual
      production fix we need to apply (``workers.tenant_id``).

    Default resolution: ``WORKER_MGMT_MIGRATIONS_STRICT=1`` env var, falling
    back to ``False``.

    Args:
        migrations_dir: optional override (used by tests).
        pool: optional pool override (used by tests). Defaults to the
            module-level ``_pg_pool`` populated by :func:`init_pg_pool`.
        strict: see above.

    Returns:
        tuple[list[str], list[tuple[str, str]]]:
            (applied_filenames, [(failed_filename, error_message), ...]).

    Raises:
        RuntimeError: if the pool is not initialized.
        psycopg.Error: re-raised from the first failing migration when
            ``strict=True``.
    """
    target_pool = pool if pool is not None else _pg_pool
    if target_pool is None:
        raise RuntimeError(
            "PostgreSQL pool not initialized. Call init_pg_pool() before run_migrations()."
        )

    if strict is None:
        strict = os.environ.get("WORKER_MGMT_MIGRATIONS_STRICT", "").lower() in {
            "1",
            "true",
            "yes",
        }

    resolved_dir = _resolve_migrations_dir(migrations_dir)
    files = _list_migration_files(resolved_dir)
    if not files:
        logger.warning(
            "No migration files found in %s — startup will use whatever "
            "schema is already in the database.",
            resolved_dir,
        )
        return [], []

    applied: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    async with target_pool.connection() as conn:
        # --- Startup-safety guards: bound how long DDL may block. -----------
        #
        # Without these the chain ran with lock_timeout=0 and
        # statement_timeout=0 (verified live: `show lock_timeout` -> 0), i.e.
        # "block forever". Migrations 010 and 019 take ACCESS EXCLUSIVE across
        # 11 tables, so one pod SIGKILLed mid-migration left its PgBouncer
        # server connection alive and mid-transaction, still holding ACCESS
        # EXCLUSIVE. The next pod blocked on it, was killed by the startupProbe
        # (30 x 10s) + terminationGracePeriodSeconds 60 = 359s -> exit 137, and
        # leaked another blocked backend. The convoy grew: observed 2026-08-08,
        # one backend held the lock 2h29m52s and 60 distinct backends unblocked
        # within 10ms of its death.
        #
        # lock_timeout turns "block until the probe kills us" into a fast, clean
        # failure that the non-strict handler below records and steps over, so a
        # boot can never again be consumed by waiting on a lock.
        # idle_in_transaction_session_timeout caps any orphan that still gets
        # created: a SIGKILLed pod's server connection is left idle IN a
        # transaction, so Postgres now reaps it (and releases its locks) instead
        # of holding them until someone notices.
        #
        # `SET LOCAL`, inside each per-file transaction (see
        # `_strip_outer_transaction` for the incident that forced this).
        #
        # This REPLACES a session-level `SET` here, whose stated reason was
        # that a SET LOCAL "is discarded at that inner COMMIT" in the 15 files
        # carrying their own `BEGIN; ... COMMIT;`. True as written -- but the
        # session-level form was not a fix, because that same inner COMMIT
        # returns the connection to a `pool_mode = transaction` PgBouncer and
        # the session state stops being reliably attached to the backend
        # running the DDL. Removing the embedded boundaries removes the reason
        # SET LOCAL was rejected, and SET LOCAL cannot be separated from its
        # own transaction the way session state can. It also needs no RESET:
        # it is scoped to a transaction that always ends here, so nothing can
        # leak a 10s lock_timeout onto unrelated traffic through the pool.
        pool_autocommit = conn.autocommit
        await _ensure_migration_ledger(conn)
        already_applied = await _applied_filenames(conn)
        try:
            # Each migration in its own transaction so a single bad file does
            # not roll back unrelated additive ALTERs from the same batch.
            for sql_path in files:
                sql_text = sql_path.read_text(encoding="utf-8")
                if not sql_text.strip():
                    logger.debug("Skipping empty migration: %s", sql_path.name)
                    continue
                # `migrate:always` files are control loops, not one-time
                # changes, so the ledger must not retire them. Everything else
                # is skipped once recorded -- that is the whole point of it.
                if sql_path.name in already_applied and not _runs_always(sql_text):
                    logger.debug("Skipping applied migration: %s", sql_path.name)
                    skipped.append(sql_path.name)
                    continue

                if _wants_no_transaction(sql_text):
                    problem = _no_transaction_violation(sql_text)
                    if problem:
                        # Refuse before the statement reaches a backend. The
                        # server's own error for this names transactions, not
                        # the semicolon that caused it.
                        logger.error(
                            "Migration %s declares `%s` but %s; refusing to run it",
                            sql_path.name,
                            _MIGRATION_NO_TRANSACTION_DIRECTIVE,
                            problem,
                        )
                        if strict:
                            raise RuntimeError(f"{sql_path.name}: {problem}")
                        failed.append((sql_path.name, problem))
                        continue

                logger.info("Applying migration: %s", sql_path.name)
                try:
                    if _wants_no_transaction(sql_text):
                        await _apply_outside_transaction(conn, sql_path, sql_text)
                    else:
                        body = _strip_outer_transaction(sql_text)
                        async with conn.transaction():
                            await conn.execute(
                                cast(
                                    "LiteralString",
                                    f"SET LOCAL lock_timeout = '{_MIGRATION_LOCK_TIMEOUT}'",
                                )
                            )
                            await conn.execute(
                                cast(
                                    "LiteralString",
                                    "SET LOCAL idle_in_transaction_session_timeout = "
                                    f"'{_MIGRATION_IDLE_TX_TIMEOUT}'",
                                )
                            )
                            await conn.execute(cast("LiteralString", body))
                            # Same transaction as the DDL: the ledger row and
                            # the schema change commit together or not at all.
                            await conn.execute(
                                "INSERT INTO schema_migration_files (filename) "
                                "VALUES (%s) ON CONFLICT (filename) DO NOTHING",
                                (sql_path.name,),
                            )
                except psycopg.Error as exc:
                    if strict:
                        raise
                    # Match the legacy init-container behavior: log and keep going.
                    # Kept as a forward safety belt so a future bad migration logs
                    # instead of crashing startup. The chain itself is Postgres-
                    # clean since the 001 removal (it was the only SQLite-flavored
                    # file; its objects already live in postgres_schema.sql).
                    #
                    # A lock timeout arrives here too: SQLSTATE 55P03 raises
                    # psycopg.errors.LockNotAvailable, a psycopg.Error subclass.
                    # So a blocked migration is recorded in `failed` and the run
                    # continues, rather than hanging until the startupProbe fires.
                    msg = str(exc).strip().splitlines()[0]
                    logger.warning(
                        "Migration %s failed (continuing): %s",
                        sql_path.name,
                        msg,
                    )
                    failed.append((sql_path.name, msg))
                    continue
                applied.append(sql_path.name)
        finally:
            # Autocommit is the only session state this function can still
            # leave behind, and only if `_apply_outside_transaction` died
            # between its toggles. Everything else is now SET LOCAL, which
            # PostgreSQL discards at transaction end, so there is nothing to
            # hand back to the PgBouncer pool. Best-effort: a dead connection
            # has nothing to restore, and that must not mask the real error.
            with contextlib.suppress(psycopg.Error):
                if conn.autocommit is not pool_autocommit:
                    await conn.set_autocommit(pool_autocommit)

    logger.info(
        "Migrations: applied=%d skipped=%d failed=%d (dir=%s)",
        len(applied),
        len(skipped),
        len(failed),
        resolved_dir,
    )
    if failed:
        logger.warning("Migration failures (non-strict mode): %s", failed)
    return applied, failed


async def close_pg_pool() -> None:
    """Close PostgreSQL connection pool and clean up resources."""
    global _pg_pool  # noqa: PLW0603 -- lazy-init singleton pool

    if _pg_pool is None:
        logger.debug("PostgreSQL pool not initialized, nothing to close")
        return

    try:
        await _pg_pool.close()
        logger.info("PostgreSQL pool closed successfully")
    except Exception:
        logger.exception("Error closing PostgreSQL pool")
    finally:
        _pg_pool = None


def get_pg_pool() -> Optional[AsyncConnectionPool]:
    """Get the current PostgreSQL connection pool.

    Returns:
        AsyncConnectionPool or None: The connection pool if initialized

    Note:
        Pool must be initialized with init_pg_pool() before calling this.
    """
    return _pg_pool


@asynccontextmanager
async def get_pg_connection() -> AsyncIterator[_PsycopgConnectionWrapper]:
    """Get a PostgreSQL database connection from the pool.

    Yields a wrapper object that provides asyncpg-compatible methods
    (fetch, fetchrow, fetchval, execute) on top of psycopg v3.

    Usage:
        async with get_pg_connection() as conn:
            result = await conn.fetch(
                "SELECT * FROM workers WHERE tenant_id = $1 AND "
                "(visibility = 'tenant' OR (visibility = 'user' AND subject_id = $2))",
                tenant_id,
                subject_id,
            )

    Yields:
        _PsycopgConnectionWrapper: A wrapped database connection

    Raises:
        RuntimeError: If pool is not initialized
    """
    if _pg_pool is None:
        logger.error("PostgreSQL pool not initialized. Call init_pg_pool() first.")
        raise RuntimeError(
            "PostgreSQL pool not initialized. Call init_pg_pool() during application startup."
        )

    async with _pg_pool.connection() as connection:
        yield _PsycopgConnectionWrapper(connection)


async def test_pg_connection() -> bool:
    """Test PostgreSQL connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        async with get_pg_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            return bool(result == 1)
    except Exception:
        logger.exception("PostgreSQL connection test failed")
        return False


def get_postgres_config() -> dict:
    """Get PostgreSQL configuration from environment.

    Returns:
        dict: Configuration dictionary
    """
    settings = get_settings()
    db_url_raw = settings.database_url
    if not db_url_raw:
        raise ValueError("DATABASE_URL environment variable is required")

    url = make_url(db_url_raw)

    return {
        "host": url.host or "localhost",
        "port": url.port or 5432,
        "database": url.database or "worker_management",
        "user": url.username or "",
        "password": url.password or "",
        "pool_min": settings.postgres_pool_min,
        "pool_max": settings.postgres_pool_max,
        "command_timeout": settings.postgres_command_timeout,
        "connect_timeout": settings.postgres_connect_timeout,
    }
