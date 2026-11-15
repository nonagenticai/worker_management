"""Shared gate for the real-Postgres (``@pytest.mark.pg``) integration suite.

Why this exists
---------------
Every ``pg``-marked module used to open with its own copy of::

    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set — real Postgres 18 required")
    try:
        await init_pg_pool()
    except Exception as exc:
        pytest.skip(f"Could not connect to Postgres at DATABASE_URL: {exc}")

The first skip is correct for a laptop with no database. The **second** one was
dangerous: it turned a genuine Postgres outage, a typo'd DSN or a CI sidecar
that failed to come up into a *green* run — including
``test_migration_chain.py``, the only proof the migration chain applies cleanly.
A gate that can silently disarm itself is worse than no gate, because it is
trusted.

``except Exception`` also swallowed everything else: an ``ImportError`` from a
renamed module, an ``AttributeError`` from a changed signature, a ``TypeError``
in the fixture itself — all reported as "could not connect to Postgres".

What this changes
-----------------
Local developers keep the skip. **CI cannot skip.** When ``CI`` is truthy
(GitHub Actions always exports ``CI=true``) an absent ``DATABASE_URL`` or an
unreachable database is a hard ``pytest.fail``, not a skip.

The connection guard is also narrowed to the exceptions that actually mean
"the database is not reachable". ``psycopg_pool.PoolTimeout`` — what
``init_pg_pool`` raises when the pool cannot fill — subclasses
``psycopg.OperationalError``, so the two-entry tuple below covers it; anything
else propagates as a real error.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from src.db.db_config import get_postgres_config
from src.settings import get_settings

from .target_guard import refusal

__all__ = [
    "PG_CONNECT_ERRORS",
    "pg_conninfo",
    "require_database_url",
    "unreachable_pg",
]

# Exceptions that genuinely mean "Postgres is not reachable".
# psycopg_pool.PoolTimeout subclasses psycopg.OperationalError (verified against
# psycopg-pool in this repo's lockfile), so init_pg_pool's failure mode is
# covered without importing psycopg_pool here.
PG_CONNECT_ERRORS = (psycopg.OperationalError, OSError)


def _require_pg() -> bool:
    """True when a missing/unreachable Postgres must be a failure, not a skip.

    Read at call time rather than import time so a test can monkeypatch ``CI``
    and so the value cannot be frozen by import ordering.
    """
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def require_database_url() -> None:
    """Skip locally / fail in CI when ``DATABASE_URL`` is not set.

    A ``DATABASE_URL`` naming a production host is a hard failure (never a
    skip): the pg suite runs migrations and writes rows. See target_guard.py.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        reason = refusal("DATABASE_URL", url)
        if reason is not None:
            pytest.fail(f"pg suite refused: {reason}")
        return
    msg = "DATABASE_URL not set — real Postgres 18 required"
    if _require_pg():
        pytest.fail(
            f"{msg}. CI must never skip the pg suite: the archetype's DB-integration "
            f"pass exports DATABASE_URL from the ephemeral postgres:18 sidecar, so an "
            f"empty value means the sidecar or its env wiring is broken."
        )
    pytest.skip(msg)


def unreachable_pg(exc: BaseException) -> None:
    """Skip locally / fail in CI when the database cannot be reached.

    Call from the ``except`` arm of the pool-init guard. Never returns.
    """
    msg = f"Could not connect to Postgres at DATABASE_URL: {exc}"
    if _require_pg():
        pytest.fail(
            f"{msg}. In CI this is a real outage or a broken DSN, not a reason "
            f"to pass: skipping here would hide the migration-chain proof."
        )
    pytest.skip(msg)


def pg_conninfo(**overrides: Any) -> str:
    """libpq conninfo for a raw connection to the database the app pool uses.

    ``DATABASE_URL`` is a **SQLAlchemy URL, not a libpq DSN**, and the two are
    not interchangeable. The fleet archetype exports::

        DATABASE_URL=postgresql+psycopg://{user}:{password}@127.0.0.1:{port}/{db}

    (gitops-production ``.github/workflows/python-service.yml``, step "Start
    ephemeral CI database"). ``psycopg.AsyncConnection.connect()`` only
    URI-parses a string beginning ``postgresql://`` or ``postgres://``; the
    ``+psycopg`` driver suffix makes it something else, so libpq falls back to
    keyword/value parsing and raises::

        psycopg.ProgrammingError: missing "=" after
        "postgresql+psycopg://…/worker_management_ci" in connection info string

    That is exactly how ``test_migration_lock_safety.py`` reddened main while
    passing locally against a plain ``postgresql://`` DSN.

    So do what the application itself does rather than parsing a URL by hand:
    :func:`src.db.db_config.get_postgres_config` runs the same SQLAlchemy
    ``make_url`` that :func:`~src.db.db_config.init_pg_pool` uses (it
    understands any ``+driver`` suffix), and the discrete parameters it returns
    are assembled here with :func:`psycopg.conninfo.make_conninfo`, which
    quotes and escapes values instead of string-concatenating them. The
    resulting connection is byte-for-byte the same server, database and role as
    the pool under test — only a different backend, which is the point.

    ``sslmode`` mirrors ``init_pg_pool``'s ``sslmode_conninfo``: the pool
    passes an :class:`ssl.SSLContext` for every mode except ``disable`` (where
    it is ``None``), so the effective mode is always
    ``settings.postgres_sslmode``. A raw connection lets libpq handle it
    natively. CI sets ``POSTGRES_SSLMODE=disable`` on the DB-integration step.

    Keyword ``overrides`` are passed through to ``make_conninfo`` for
    per-connection options (e.g. ``connect_timeout``).
    """
    cfg = get_postgres_config()
    return make_conninfo(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        sslmode=get_settings().postgres_sslmode.lower(),
        connect_timeout=cfg["connect_timeout"],
        **overrides,
    )


async def forget_all_migrations() -> None:
    """Empty the applied-migrations ledger.

    Safe before the table exists: a fresh database has nothing to forget.
    """
    from src.db.db_config import get_pg_connection

    async with get_pg_connection() as conn:
        if await conn.fetchval("SELECT to_regclass('schema_migration_files')") is None:
            return
        await conn.execute("DELETE FROM schema_migration_files")


async def apply_migrations(*, strict: bool = True):
    """Run the chain the way the migration-CONTENT tests need it.

    ``run_migrations`` records every applied file in ``schema_migration_files``
    and skips anything listed there. That is the point in production -- 8 of 26
    files were re-taking ACCESS EXCLUSIVE on live tables every pod boot and
    failing on a lock timeout every time.

    But these modules assert on what a migration DOES to data: 024 converging
    uuid spellings, 025 deleting exactly two rows. Several of them seed rows and
    then expect the file to run AGAIN, and others simply assume nothing has been
    applied yet -- which stops being true the moment an earlier module in the
    same job populates the ledger against the shared CI database. A bare call
    then applies nothing and the assertions fail for reasons that have nothing
    to do with the migration under test.

    Clearing first keeps the subject of these tests the migration content. The
    ledger's own behaviour is covered where it belongs: ``tests/
    test_run_migrations.py`` and the chain's second-run assertion, which
    deliberately calls ``run_migrations`` directly to prove the skip happens.
    """
    from src.db.db_config import run_migrations

    await forget_all_migrations()
    return await run_migrations(strict=strict)
