"""``get_tenant_id``'s introspection path resolves the TENANT, never the subject.

``src/auth/tenant.py`` used to read ``subject_id or tenant_id`` from licence
introspection. A licences subject is a user, and one user holds licences in
several tenants, so that line:

* stamped a user's uuid as a tenant that does not exist (how 73 workers and 38
  graphs came to be keyed by subject ``8382f0b3-...`` -- migration 040), and
* collapsed every tenant that user belongs to into ONE tenant, i.e. a
  cross-tenant read/write.

The path is a fallback behind ``AuthenticationMiddleware`` (which already reads
``tenant_id`` only), but middleware registration is fail-open, so this resolver
has to be correct on its own.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.auth.tenant import get_tenant_id

SUBJECT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
TENANT_A = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
TENANT_B = "01a1113b-674f-72b0-a84b-dc746d39afca"


def _request() -> SimpleNamespace:
    # Empty state: no middleware ran, so the introspection path is reached.
    return SimpleNamespace(state=SimpleNamespace(), headers={"X-Tenant-Id": "Bearer tok"})


async def _resolve(introspected: dict) -> str:
    with patch(
        "src.auth.tenant._introspect_license",
        new_callable=AsyncMock,
        return_value={"active": True, **introspected},
    ):
        return await get_tenant_id(_request())


@pytest.mark.asyncio
async def test_distinct_subject_and_tenant_resolve_to_the_tenant() -> None:
    assert await _resolve({"subject_id": SUBJECT, "tenant_id": TENANT_A}) == TENANT_A


@pytest.mark.asyncio
async def test_one_subject_in_two_tenants_resolves_to_two_tenants() -> None:
    """The cross-tenant case: same user, two licences, two tenants."""
    a = await _resolve({"subject_id": SUBJECT, "tenant_id": TENANT_A})
    b = await _resolve({"subject_id": SUBJECT, "tenant_id": TENANT_B})
    assert (a, b) == (TENANT_A, TENANT_B)
    assert SUBJECT not in (a, b)


@pytest.mark.asyncio
async def test_a_subject_without_a_tenant_is_refused_not_promoted() -> None:
    """No fallback: a response with no tenant_id has no tenant (401)."""
    with pytest.raises(HTTPException) as excinfo:
        await _resolve({"subject_id": SUBJECT})
    assert excinfo.value.status_code == 401
