"""Readable tenant labels that are still valid ``tenant_id`` values.

``tenant_id`` is a real Postgres ``uuid`` column (migration
``024_tenant_id_is_a_uuid.sql``), so the integration fixtures can no longer
seed a readable literal like ``int-wreg-9f2c1a``. They used to, and that is
part of how production accumulated 93,842 rows whose tenant_id was not a uuid:
the shape was normal everywhere, including in the tests.

``tenant_uuid()`` keeps the *labelling* habit — a caller still writes what the
tenant is for — while producing something the column accepts. It is a pure
function of the label, so a test that wants two tenants to differ gets two
different uuids, and a test that wants the same tenant twice gets the same one.

The namespace is the one migration 024 uses, so a label seeded here maps to the
same uuid the migration would have derived for it.

⚠️ What is lost: the uuid no longer *reads* as the test that created it. When
debugging rows left behind in a shared CI database, seed ``tenant_name`` with
the label as well — that column exists precisely to carry the human string.
"""

from __future__ import annotations

import uuid

from src.api.tenant_constants import TENANT_NAMESPACE


def tenant_uuid(label: str) -> str:
    """A stable ``uuid`` for a readable tenant label."""
    return str(uuid.uuid5(TENANT_NAMESPACE, label))
