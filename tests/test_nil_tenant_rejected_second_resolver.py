"""The SECOND tenant resolver refuses the nil uuid too — and is not shadowed.

⛔ WHY THIS FILE EXISTS SEPARATELY FROM ``test_nil_tenant_rejected_inbound.py``.
That file guards ``tenant_constants.resolve_tenant_id``, the funnel behind
``auth_middleware``'s three inbound sites, and its commit claimed that was THE
funnel. It is not. ``src/auth/tenant.py::get_tenant_id`` is a second, independent
resolver, reached through ``Depends(get_tenant_context)`` from ~50 route sites
across 12 API modules, and it never calls ``resolve_tenant_id`` at all.

It is normally shadowed by the middleware's ``request.state.tenant_id`` fast
path — but middleware registration is FAIL-OPEN:
``worker_management_api.py:401-406`` wraps ``add_middleware`` in
``try/except Exception: logger.exception(...)``, so a registration failure
downgrades every route to this resolver while the process keeps serving 200s.
A guard that only holds while another component is healthy is not a guard.

Also pinned here: the nil uuid is no longer PRIVILEGED. It used to sit in
``llm_config_resolver._SYSTEM_TENANTS``, exempting it from the ``byo_required``
hard-fail and routing it to seeded platform LLM config — owner GPU spend that no
customer can be charged for, reachable from ``ctx.tenant_id``.

Every test asserts a raised status or a returned value, never a mock call count.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request

from src.auth.tenant import get_tenant_id
from src.worker.factory.llm_config_resolver import _SYSTEM_TENANTS, _system_tenants

NIL = "00000000-0000-0000-0000-000000000000"
REAL_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"


def _request(state: dict[str, Any], headers: dict[str, str] | None = None) -> Request:
    """A real ``Request`` (not a stub) whose ``.state`` is prebound, as the
    middleware would leave it — or as it is left when the middleware never ran."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/probe",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "state": state,
        }
    )


# --------------------------------------------------------------------------- #
# Path 1 — request.state.tenant_id (the middleware fast path).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_state_tenant_id_of_nil_is_refused() -> None:
    with pytest.raises(HTTPException) as excinfo:
        await get_tenant_id(_request({"tenant_id": NIL}))
    assert excinfo.value.status_code == 401
    assert "nil uuid" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_state_tenant_id_of_a_real_tenant_still_passes() -> None:
    """CONTROL — otherwise "refuse everything" would pass the test above."""
    assert await get_tenant_id(_request({"tenant_id": REAL_TENANT})) == REAL_TENANT


# --------------------------------------------------------------------------- #
# Path 2 — request.state.user, dict and object forms.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["tenant_id", "user_id", "sub"])
async def test_state_user_dict_carrying_nil_is_refused(key: str) -> None:
    """All three keys this path coalesces over are entry points, not just the first."""
    with pytest.raises(HTTPException) as excinfo:
        await get_tenant_id(_request({"user": {key: NIL}}))
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_state_user_dict_with_a_real_tenant_still_passes() -> None:
    assert await get_tenant_id(_request({"user": {"tenant_id": REAL_TENANT}})) == REAL_TENANT


@pytest.mark.asyncio
async def test_state_user_object_carrying_nil_is_refused() -> None:
    class _User:
        tenant_id = NIL

    with pytest.raises(HTTPException) as excinfo:
        await get_tenant_id(_request({"user": _User()}))
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_state_user_object_with_a_real_tenant_still_passes() -> None:
    class _User:
        tenant_id = REAL_TENANT

    assert await get_tenant_id(_request({"user": _User()})) == REAL_TENANT


# --------------------------------------------------------------------------- #
# Path 3 — direct licence introspection (the fully unshadowed path: no
# middleware ran at all, so nothing populated request.state).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_introspected_licence_resolving_to_nil_is_refused() -> None:
    """The nil uuid as the introspected ``tenant_id`` is refused, whatever the
    subject. (``subject_id`` is never read as a tenant on this path any more, so
    a nil SUBJECT beside a nil tenant changes nothing.)"""
    for claim in ({"tenant_id": NIL}, {"subject_id": NIL, "tenant_id": NIL}):
        introspect = {"active": True, **claim}
        with patch(
            "src.auth.tenant._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await get_tenant_id(_request({}, {"X-Tenant-Id": "Bearer tok"}))
        assert excinfo.value.status_code == 401
        assert "nil uuid" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_an_introspected_licence_with_a_real_tenant_still_passes() -> None:
    """CONTROL for the unshadowed path — the resolver still works."""
    introspect = {"active": True, "tenant_id": REAL_TENANT}
    with patch(
        "src.auth.tenant._introspect_license",
        new_callable=AsyncMock,
        return_value=introspect,
    ):
        got = await get_tenant_id(_request({}, {"X-Tenant-Id": "Bearer tok"}))
    assert got == REAL_TENANT


# --------------------------------------------------------------------------- #
# The nil uuid is no longer a PRIVILEGED tenant.
# --------------------------------------------------------------------------- #


def test_the_env_allow_list_cannot_re_add_the_nil_uuid() -> None:
    """Removing the literal is cosmetic if one ConfigMap typo puts it back.

    ``WM_SYSTEM_TENANT_IDS`` is GitOps-set and nothing validated it. Tested
    through the pure builder, NOT an ``importlib.reload``: reloading that module
    rebinds its module-level ``_db`` and detaches every other test's monkeypatch.
    """
    built = _system_tenants(f"{NIL},{REAL_TENANT}")
    assert NIL not in built
    # CONTROL: the env still works for a value that is a real tenant.
    assert REAL_TENANT in built


def test_the_env_allow_list_rejects_the_nil_uuid_whatever_its_case() -> None:
    assert NIL.upper() not in _system_tenants(NIL.upper())


def test_the_reserved_platform_sentinel_survives() -> None:
    """CONTROL: only the nil uuid lost its exemption."""
    assert "00000000-0000-0000-0000-000000000001" in _system_tenants("")
    assert NIL not in _SYSTEM_TENANTS


def test_the_middleware_registration_really_is_fail_open() -> None:
    """The premise of this whole file, pinned as a fact rather than a comment.

    If this ever becomes fail-CLOSED, the second resolver stops being reachable
    in production and this file's urgency drops — but until then, it is live.
    """
    src = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/src/api/worker_management_api.py"
    )
    with open(src) as fh:
        text = fh.read()
    assert 'app.add_middleware(cast("Any", AuthenticationMiddleware), bypass_auth=False)' in text
    assert 'logger.exception("Failed to enable authentication middleware")' in text
