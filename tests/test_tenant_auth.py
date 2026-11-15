"""Tests for src.auth.tenant — tenant extraction and BOLA prevention.

Covers:
- get_tenant_id: fast-path via request.state, dict user, header introspection, 401
- get_resource_with_tenant_check: tenant_id match, 404 missing, 404 BOLA
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.tenant import get_resource_with_tenant_check, get_tenant_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(*, state_attrs: dict | None = None, headers: dict | None = None):
    """Build a minimal mock Request with configurable state and headers."""
    request = MagicMock()
    # Build a real SimpleNamespace so hasattr/getattr work naturally
    state = SimpleNamespace(**(state_attrs or {}))
    request.state = state
    request.headers = headers or {}
    return request


# ===================================================================
# get_tenant_id
# ===================================================================


class TestGetTenantId:
    """Tests for the get_tenant_id dependency."""

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_request_state_tenant_id(self):
        """Priority 1 fast path: request.state.tenant_id is returned directly."""
        request = _make_request(state_attrs={"tenant_id": "tenant-abc"})
        result = await get_tenant_id(request)
        assert result == "tenant-abc"

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_state_user_dict_tenant_id(self):
        """Priority 1: user dict with tenant_id key."""
        request = _make_request(state_attrs={"user": {"tenant_id": "tenant-from-dict"}})
        result = await get_tenant_id(request)
        assert result == "tenant-from-dict"

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_state_user_dict_user_id(self):
        """Priority 1: user dict falls back to user_id when tenant_id is absent."""
        request = _make_request(state_attrs={"user": {"user_id": "uid-fallback"}})
        result = await get_tenant_id(request)
        assert result == "uid-fallback"

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_state_user_dict_sub(self):
        """Priority 1: user dict falls back to sub when tenant_id and user_id are absent."""
        request = _make_request(state_attrs={"user": {"sub": "sub-fallback"}})
        result = await get_tenant_id(request)
        assert result == "sub-fallback"

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_state_user_object(self):
        """Priority 1: user is an object with tenant_id attribute."""
        user_obj = SimpleNamespace(tenant_id="obj-tenant")
        request = _make_request(state_attrs={"user": user_obj})
        result = await get_tenant_id(request)
        assert result == "obj-tenant"

    @pytest.mark.asyncio
    async def test_extracts_tenant_from_state_user_object_sub(self):
        """Priority 1: user object falls back to sub attribute."""
        user_obj = SimpleNamespace(sub="obj-sub")
        request = _make_request(state_attrs={"user": user_obj})
        result = await get_tenant_id(request)
        assert result == "obj-sub"

    @pytest.mark.asyncio
    async def test_extracts_tenant_via_header_introspection(self):
        """Priority 2: the introspected ``tenant_id`` wins, never the subject."""
        request = _make_request(headers={"X-Tenant-Id": "Bearer some-license-token"})

        with patch(
            "src.auth.tenant._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "subject_id": "introspected-subject",
                "tenant_id": "introspected-tenant",
            },
        ):
            result = await get_tenant_id(request)

        assert result == "introspected-tenant"

    @pytest.mark.asyncio
    async def test_extracts_tenant_via_header_introspection_tenant_id_field(self):
        """Priority 2: introspection returns tenant_id instead of subject_id."""
        request = _make_request(headers={"X-Tenant-Id": "Bearer token-xyz"})

        with patch(
            "src.auth.tenant._introspect_license",
            new_callable=AsyncMock,
            return_value={"active": True, "tenant_id": "introspected-tid"},
        ):
            result = await get_tenant_id(request)

        assert result == "introspected-tid"

    @pytest.mark.asyncio
    async def test_raises_401_when_no_tenant_context(self):
        """No state tenant, no user, no header -> 401."""
        request = _make_request()

        with pytest.raises(HTTPException) as exc_info:
            await get_tenant_id(request)

        assert exc_info.value.status_code == 401
        assert "tenant" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_503_when_introspection_transiently_unavailable(self):
        """Header present but introspection returns None (transient/unavailable,
        NOT a token verdict) -> retryable 503, not a spurious 401. Closes the
        growth_bundle 401 class on the dependency path (mirrors #94)."""
        request = _make_request(headers={"X-Tenant-Id": "Bearer valid-but-unverifiable"})

        with (
            patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_tenant_id(request)

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_raises_401_when_introspection_inactive(self):
        """Header present but license is inactive -> 401."""
        request = _make_request(headers={"X-Tenant-Id": "Bearer inactive-token"})

        with (
            patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value={"active": False},
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_tenant_id(request)

        assert exc_info.value.status_code == 401


# ===================================================================
# get_resource_with_tenant_check
# ===================================================================


class TestGetResourceWithTenantCheck:
    """Tests for BOLA-prevention resource access."""

    @pytest.mark.asyncio
    async def test_returns_resource_when_tenant_matches_via_tenant_id(self):
        """Resource with matching tenant_id is returned."""
        resource = {"id": "res-1", "tenant_id": "tenant-a", "data": "payload"}
        db_func = AsyncMock(return_value=resource)

        result = await get_resource_with_tenant_check("res-1", "tenant-a", db_func, "Worker")

        assert result == resource
        db_func.assert_awaited_once_with("res-1")

    @pytest.mark.asyncio
    async def test_returns_resource_when_tenant_matches_with_full_row_shape(self):
        """A resource carrying the full ownership shape still matches on tenant_id."""
        resource = {
            "id": "res-2",
            "tenant_id": "tenant-b",
            "subject_id": "tenant-b",
            "visibility": "tenant",
        }
        db_func = AsyncMock(return_value=resource)

        result = await get_resource_with_tenant_check("res-2", "tenant-b", db_func, "Worker")

        assert result == resource

    @pytest.mark.asyncio
    async def test_returns_object_resource_when_tenant_matches(self):
        """Resource as an object (not dict) with tenant_id attribute."""
        resource = SimpleNamespace(id="res-3", tenant_id="tenant-c")
        db_func = AsyncMock(return_value=resource)

        result = await get_resource_with_tenant_check("res-3", "tenant-c", db_func, "Graph")

        assert result is resource

    @pytest.mark.asyncio
    async def test_returns_object_resource_with_subject_attr(self):
        """Resource object with tenant_id (+subject_id) attributes matches on tenant_id."""
        resource = SimpleNamespace(id="res-4", tenant_id="tenant-d", subject_id="tenant-d")
        db_func = AsyncMock(return_value=resource)

        result = await get_resource_with_tenant_check("res-4", "tenant-d", db_func, "State")

        assert result is resource

    @pytest.mark.asyncio
    async def test_raises_404_when_resource_not_found(self):
        """db_func returns None -> 404."""
        db_func = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_resource_with_tenant_check("nonexistent", "tenant-x", db_func, "Worker")

        assert exc_info.value.status_code == 404
        assert "Worker not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_404_not_403_when_tenant_mismatch(self):
        """BOLA prevention: tenant mismatch returns 404 (not 403) to avoid leaking existence."""
        resource = {"id": "res-5", "tenant_id": "tenant-owner"}
        db_func = AsyncMock(return_value=resource)

        with pytest.raises(HTTPException) as exc_info:
            await get_resource_with_tenant_check("res-5", "tenant-attacker", db_func, "Worker")

        # Must be 404, NOT 403 - BOLA prevention hides existence
        assert exc_info.value.status_code == 404
        assert "Worker not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_404_on_mismatch_with_full_row_shape(self):
        """BOLA prevention also applies to a resource carrying the full ownership shape."""
        resource = {
            "id": "res-6",
            "tenant_id": "real-owner",
            "subject_id": "real-owner",
            "visibility": "tenant",
        }
        db_func = AsyncMock(return_value=resource)

        with pytest.raises(HTTPException) as exc_info:
            await get_resource_with_tenant_check("res-6", "attacker", db_func, "Graph")

        assert exc_info.value.status_code == 404
        assert "Graph not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_404_when_resource_has_no_tenant(self):
        """Resource exists but has no tenant_id -> 404 (mismatch with None)."""
        resource = {"id": "res-7", "data": "orphan"}
        db_func = AsyncMock(return_value=resource)

        with pytest.raises(HTTPException) as exc_info:
            await get_resource_with_tenant_check("res-7", "any-tenant", db_func, "Resource")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_custom_resource_type_in_error_message(self):
        """resource_type parameter is used in error messages."""
        db_func = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_resource_with_tenant_check("x", "t", db_func, "CustomThing")

        assert "CustomThing not found" in exc_info.value.detail
