"""Per-user-within-tenant visibility helpers (raw-SQL flavor).

Two scoping axes:
- tenant_id: the organization that owns the data
- subject_id: the user (Subject UUID from licenses) within that tenant

Rows have a ``visibility`` column ('tenant' default | 'user'). List queries
return tenant-visible rows + the caller's own user-visible rows. Mutations
on user-visible rows are gated to the owning subject.

This module is the worker_management mirror of
``procedure_management/src/repositories/visibility.py`` (the Phase 1 pilot).
The pilot used SQLAlchemy expression helpers; worker_management runs raw SQL
through a psycopg-via-asyncpg-style wrapper, so we expose the predicate as a
SQL fragment built from positional placeholders ($1, $2, ...).
"""

from __future__ import annotations

from typing import Any


def scope_clause(tenant_param: int, subject_param: int) -> str:
    """SQL fragment for ``WHERE``: tenant + visibility OR clause.

    Produces a fragment of the form::

        tenant_id = $T AND (visibility = 'tenant'
                            OR (visibility = 'user' AND subject_id = $S))

    Caller is responsible for binding ``$T`` to ``tenant_id`` and ``$S`` to
    ``subject_id`` at the corresponding positional indexes.

    Args:
        tenant_param: 1-based positional index for the ``tenant_id`` value.
        subject_param: 1-based positional index for the ``subject_id`` value.

    Returns:
        SQL fragment to be inlined into a ``WHERE`` clause.
    """
    return (
        f"tenant_id = ${tenant_param} "
        f"AND (visibility = 'tenant' "
        f"OR (visibility = 'user' AND subject_id = ${subject_param}))"
    )


def can_mutate(row: Any, *, subject_id: str) -> bool:
    """True if caller may mutate ``row``.

    - tenant-visible rows: any tenant member (caller is in the tenant by
      virtue of having reached this row through ``scope_clause``).
    - user-visible rows: only the creating subject.

    Accepts both dict-shaped rows (psycopg ``dict_row`` factory output) and
    object-style rows. Missing ``visibility`` defaults to ``'tenant'`` to
    preserve backfilled-row semantics.

    Args:
        row: Mapping or object exposing ``visibility`` and ``subject_id``.
        subject_id: Caller's subject_id (empty string for tenant-only callers).
    """
    if isinstance(row, dict):
        visibility = row.get("visibility", "tenant")
        row_subject = row.get("subject_id")
    else:
        visibility = getattr(row, "visibility", "tenant")
        row_subject = getattr(row, "subject_id", None)

    if visibility == "tenant":
        return True
    return bool(row_subject == subject_id)
