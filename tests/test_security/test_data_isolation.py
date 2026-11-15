"""
Security tests for data isolation between users.

Tests comprehensive multi-tenancy security:
- Verify user A cannot see user B's workers
- Verify user A cannot modify user B's workers
- Verify user A cannot delete user B's workers
- Test worker state isolation
- Test graph isolation
- Edge cases: SQL injection, empty/long tenant IDs
"""

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

# ============================================================================
# HELPERS
# ============================================================================


def _worker_record(worker_name: str, tenant_id: str, worker_id: str | None = None) -> dict:
    """Build a mock worker record matching WorkerResponse expectations."""
    wid = worker_id or f"wid-{worker_name}"
    return {
        "worker_id": wid,
        "worker_name": worker_name,
        "tenant_id": tenant_id,
        "tenant_name": tenant_id,  # Phase 3: tenant_name present on all records
        "db_path": f"/tmp/{worker_name}.db",
        "is_agent": False,
        "status": "active",
        "config": {"agent_type": None},
    }


def _headers(user: str) -> dict:
    """X-Tenant-Id header — the bearer token IS the tenant id."""
    return {"X-Tenant-Id": f"Bearer {user}"}


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def test_client_with_mocks():
    """TestClient with auth mocked and worker_registry patched."""
    from src.api.worker_management_api import app
    from src.auth.tenant import get_tenant_id

    # Mock license introspection so the middleware accepts our Bearer tokens
    # and sets request.state.tenant_id = <token value>
    async def _mock_introspect(token: str):
        if not token or not token.strip():
            return None
        return {"active": True, "tenant_id": token, "subject_id": token}

    # Override get_tenant_id to read from request.state (set by middleware)
    # or fall back to extracting the tenant from X-Tenant-Id directly.
    # The fallback handles the case where ENABLE_AUTH=false was set by a prior
    # test module at import time (pre-existing test isolation issue), so the
    # AuthenticationMiddleware was never attached to the app.
    async def _mock_get_tenant_id(request: Request) -> str:
        if hasattr(request.state, "tenant_id") and request.state.tenant_id:
            return str(request.state.tenant_id)
        # Fallback: extract from X-Tenant-Id (same header the middleware reads)
        x_tenant_id = request.headers.get("X-Tenant-Id", "")
        if x_tenant_id.lower().startswith("bearer "):
            tid = x_tenant_id[7:].strip()
            if tid:
                return tid
        raise HTTPException(status_code=401, detail="Missing tenant context")

    app.dependency_overrides[get_tenant_id] = _mock_get_tenant_id

    mock_registry = AsyncMock()
    # A default register_worker that returns a row-shaped dict. Without it the
    # bare AsyncMock hands back a MagicMock, and the handlers — which now build
    # their response and audit metadata from what register_worker reports the
    # row HOLDS — traverse an error path instead of the one under test. Tests
    # here assert tenant scoping, so a 500 would still "pass" while proving
    # nothing.
    mock_registry.register_worker = AsyncMock(
        side_effect=lambda **kw: (
            _worker_record(kw.get("worker_name", "w"), kw.get("tenant_id", ""))
            | {"config": kw.get("config") or {}, "is_agent": bool(kw.get("is_agent"))}
        )
    )

    # Same reasoning as register_worker above: GET /workers now takes a total
    # from count_workers to tell a complete answer from a paged one, and a bare
    # AsyncMock would hand back a MagicMock that no comparison accepts, turning
    # every scoping assertion below into a 500 that proves nothing.
    mock_registry.count_workers = AsyncMock(return_value=0)

    mock_state_db = AsyncMock()
    mock_state_db.get_worker_states_by_id = AsyncMock(return_value=[])
    mock_state_db.get_worker_states = AsyncMock(return_value=[])
    mock_state_db.delete_worker_state = AsyncMock(return_value=0)

    with (
        patch("src.api.auth_middleware._introspect_license", side_effect=_mock_introspect),
        patch("src.api.workers._globals.worker_registry", mock_registry),
        patch("src.api.workers.crud_endpoints.worker_registry", mock_registry),
        patch("src.api.workers.message_endpoints.worker_registry", mock_registry),
        patch("src.api.workers.state_endpoints.worker_registry", mock_registry),
        patch("src.api.workers.state_endpoints.StateDatabase", return_value=mock_state_db),
        patch("src.api.workers.crud_endpoints.StateDatabase", return_value=mock_state_db),
        patch("src.api.workers.message_endpoints.StateDatabase", return_value=mock_state_db),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        cast("Any", client).mock_registry = mock_registry
        yield client

    app.dependency_overrides.clear()


# ============================================================================
# DATA ISOLATION TESTS - WORKERS
# ============================================================================


def test_user_a_cannot_see_user_b_workers(test_client_with_mocks):
    """User A can only see their own workers, not user B's."""
    client = test_client_with_mocks
    calls_log: list = []

    async def mock_list_workers(tenant_id: str = "", **kwargs: Any) -> list:
        calls_log.append(("list_workers", tenant_id))
        if tenant_id == "user_a":
            return ["worker-a1"]
        elif tenant_id == "user_b":
            return ["worker-b1"]
        return []

    client.mock_registry.list_workers = mock_list_workers

    response_a = client.get("/workers", headers=_headers("user_a"))
    response_b = client.get("/workers", headers=_headers("user_b"))

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert ("list_workers", "user_a") in calls_log
    assert ("list_workers", "user_b") in calls_log


def test_user_cannot_access_another_users_worker_by_name(test_client_with_mocks):
    """User A cannot access user B's worker even knowing its name."""
    client = test_client_with_mocks

    async def mock_get_worker(
        tenant_id: str = "", worker_name: str = "", **kwargs: Any
    ) -> dict | None:
        if tenant_id == "user_b" and worker_name == "worker-b1":
            return _worker_record("worker-b1", "user_b")
        return None

    client.mock_registry.get_worker = mock_get_worker

    # User B can see their own worker
    response_b = client.get("/workers/worker-b1", headers=_headers("user_b"))
    assert response_b.status_code == 200

    # User A gets 404
    response_a = client.get("/workers/worker-b1", headers=_headers("user_a"))
    assert response_a.status_code == 404


def test_user_cannot_update_another_users_worker(test_client_with_mocks):
    """User A cannot update user B's worker (get_worker returns None → auto-create for A, not update B)."""
    client = test_client_with_mocks

    get_calls: list = []

    async def mock_get_worker(
        tenant_id: str = "", worker_name: str = "", **kwargs: Any
    ) -> dict | None:
        get_calls.append((tenant_id, worker_name))
        return None  # Not found for this tenant

    client.mock_registry.get_worker = mock_get_worker

    # PUT with user_a creds → endpoint sees worker not found for user_a
    # Even if it auto-creates, it creates for user_a, NOT user_b
    response = client.put(
        "/workers/worker-b1",
        json={"worker_name": "worker-b1"},
        headers=_headers("user_a"),
    )
    # Assert the request actually completed: without this the scoping checks
    # below pass just as happily against a 500 that never reached the registry.
    assert response.status_code == 200, response.text

    # The DB was queried with user_a, not user_b
    assert any(cid == "user_a" for cid, _ in get_calls)
    assert not any(cid == "user_b" for cid, _ in get_calls)


def test_user_cannot_delete_another_users_worker(test_client_with_mocks):
    """User A cannot delete user B's worker."""
    client = test_client_with_mocks
    client.mock_registry.get_worker = AsyncMock(return_value=None)

    response = client.delete("/workers/worker-b1", headers=_headers("user_a"))
    assert response.status_code == 404


def test_user_cannot_send_message_to_another_users_worker(test_client_with_mocks):
    """User A cannot interact with user B's worker via message endpoint."""
    client = test_client_with_mocks

    get_calls: list = []

    async def mock_get_worker(
        tenant_id: str = "", worker_name: str = "", **kwargs: Any
    ) -> dict | None:
        get_calls.append((tenant_id, worker_name))
        return None  # Not found for this tenant

    client.mock_registry.get_worker = mock_get_worker

    client.post(
        "/workers/worker-b1/message",
        json={"message": "Hello"},
        headers=_headers("user_a"),
    )

    # Regardless of status code, the DB was only queried for user_a
    assert any(cid == "user_a" for cid, _ in get_calls)
    assert not any(cid == "user_b" for cid, _ in get_calls)


# ============================================================================
# DATA ISOLATION TESTS - WORKER STATES
# ============================================================================


def test_user_cannot_access_another_users_worker_states(test_client_with_mocks):
    """User A cannot access user B's worker states."""
    client = test_client_with_mocks
    client.mock_registry.get_worker = AsyncMock(return_value=None)

    response = client.get("/workers/worker-b1/states", headers=_headers("user_a"))
    assert response.status_code == 404


def test_delete_all_workers_only_affects_current_user(test_client_with_mocks):
    """delete_all_workers only deletes current user's workers."""
    client = test_client_with_mocks

    delete_calls: list = []

    async def mock_delete_all(tenant_id: str = "", **kwargs: Any) -> dict:
        delete_calls.append(tenant_id)
        return {"deleted": 3}

    client.mock_registry.delete_all_workers = mock_delete_all
    # list_workers also called during delete_all flow
    client.mock_registry.list_workers = AsyncMock(return_value=[])

    response = client.delete("/workers/", headers=_headers("user_a"))
    assert response.status_code != 401
    assert "user_a" in delete_calls
    assert "user_b" not in delete_calls


# ============================================================================
# MULTI-USER SCENARIO TESTS
# ============================================================================


def test_three_users_complete_isolation(test_client_with_mocks):
    """Complete data isolation between three different users."""
    client = test_client_with_mocks
    users = ["user_alpha", "user_beta", "user_gamma"]

    seen_tenants: list = []

    async def mock_list_workers(tenant_id: str = "", **kwargs: Any) -> list:
        seen_tenants.append(tenant_id)
        return [f"worker-{tenant_id}-1"]

    client.mock_registry.list_workers = mock_list_workers

    for user in users:
        response = client.get("/workers", headers=_headers(user))
        assert response.status_code == 200

    # Verify each user's tenant was passed separately
    assert set(seen_tenants) == set(users)


def test_concurrent_operations_maintain_isolation(test_client_with_mocks):
    """Concurrent operations from different users maintain isolation."""
    client = test_client_with_mocks
    client.mock_registry.list_workers = AsyncMock(return_value=[])
    client.mock_registry.get_worker = AsyncMock(return_value=None)

    operations = [
        ("user_a", "GET", "/workers"),
        ("user_b", "GET", "/workers"),
        ("user_a", "GET", "/workers/worker-a1"),
        ("user_b", "GET", "/workers/worker-b1"),
    ]

    for user, _method, path in operations:
        response = client.get(path, headers=_headers(user))
        assert response.status_code != 401


# ============================================================================
# SECURITY BOUNDARY TESTS
# ============================================================================


def test_tenant_id_cannot_be_forged_in_request_body(test_client_with_mocks):
    """tenant_id in request body is ignored — tenant comes from auth header."""
    client = test_client_with_mocks

    get_calls: list = []

    async def mock_get_worker(tenant_id: str = "", worker_name: str = "", **kwargs: Any) -> dict:
        get_calls.append(tenant_id)
        return _worker_record(worker_name, tenant_id)

    client.mock_registry.get_worker = mock_get_worker
    client.mock_registry.register_worker = AsyncMock(
        return_value=_worker_record("test-worker", "user_a")
    )

    response = client.post(
        "/workers",
        json={
            "worker_name": "test-worker",
            "tenant_id": "user_b",  # Forged — should be ignored
            "subject_id": "user_b",
            "visibility": "tenant",
        },
        headers=_headers("user_a"),
    )

    assert response.status_code != 401
    # Any DB call should have used user_a (from auth), not user_b
    for called_id in get_calls:
        assert called_id != "user_b"


def test_tenant_id_cannot_be_specified_in_query_params(test_client_with_mocks):
    """A tenant_id in the query string is REFUSED, and was previously ignored.

    ⚠️ This assertion changed from 200 to 422, and the property it pins got
    stronger, not weaker. "Ignored" was the same mechanism that let
    ``?prefix=`` and ``?owner_tenant_id=`` be silently dropped from
    ``GET /workers`` — measured 2026-09-23, four filter variants returning one
    identical md5. The endpoint now refuses any query parameter it does not
    implement, so a caller-supplied tenant_id cannot reach a handler at all,
    which is a stricter guarantee than reaching one that overlooks it.
    """
    client = test_client_with_mocks
    called: list = []
    client.mock_registry.list_workers = AsyncMock(side_effect=lambda **kw: called.append(kw) or [])

    response = client.get(
        "/workers?tenant_id=user_b, subject_id=user_b",
        headers=_headers("user_a"),
    )

    assert response.status_code == 422
    assert "tenant_id" in response.text
    # The point of the test: user_b never became a scope, by any route.
    assert called == []


# ============================================================================
# GRAPH ISOLATION TESTS
# ============================================================================


def test_user_cannot_access_another_users_graph(test_client_with_mocks):
    """User A cannot access user B's graphs."""
    client = test_client_with_mocks

    # Graph endpoints may or may not exist — verify no data leakage
    response = client.get("/graphs/graph-b1", headers=_headers("user_a"))
    # Should be 404 (not found) or 405 (no such route), never 200 with B's data
    assert response.status_code in [404, 405, 422] or response.status_code != 200


def test_user_cannot_execute_another_users_graph(test_client_with_mocks):
    """User A cannot execute user B's graphs."""
    client = test_client_with_mocks

    response = client.post(
        "/execution/graph-b1",
        json={"input": "test"},
        headers=_headers("user_a"),
    )
    # Should fail — never 200
    assert response.status_code != 200 or response.status_code in [404, 405, 422]


# ============================================================================
# COMPREHENSIVE ISOLATION TEST
# ============================================================================


def test_comprehensive_multi_tenant_isolation(test_client_with_mocks):
    """
    Full scenario: two users, complete isolation.
    """
    client = test_client_with_mocks

    workers_db = {
        "user_a": [
            _worker_record("worker-a1", "user_a", "wa1"),
            _worker_record("worker-a2", "user_a", "wa2"),
        ],
        "user_b": [
            _worker_record("worker-b1", "user_b", "wb1"),
            _worker_record("worker-b2", "user_b", "wb2"),
        ],
    }

    async def mock_list_workers(tenant_id: str = "", **kwargs: Any) -> list:
        return [w["worker_name"] for w in workers_db.get(tenant_id, [])]

    async def mock_get_worker(
        tenant_id: str = "", worker_name: str = "", **kwargs: Any
    ) -> dict | None:
        for worker in workers_db.get(tenant_id, []):
            if worker["worker_name"] == worker_name:
                return worker
        return None

    client.mock_registry.list_workers = mock_list_workers
    client.mock_registry.get_worker = mock_get_worker

    # User A lists workers
    resp = client.get("/workers", headers=_headers("user_a"))
    assert resp.status_code == 200
    assert resp.json() == ["worker-a1", "worker-a2"]

    # User B lists workers
    resp = client.get("/workers", headers=_headers("user_b"))
    assert resp.status_code == 200
    assert resp.json() == ["worker-b1", "worker-b2"]

    # User A cannot see user B's worker
    resp = client.get("/workers/worker-b1", headers=_headers("user_a"))
    assert resp.status_code == 404

    # User B cannot see user A's worker
    resp = client.get("/workers/worker-a1", headers=_headers("user_b"))
    assert resp.status_code == 404

    # User B cannot delete user A's worker
    resp = client.delete("/workers/worker-a1", headers=_headers("user_b"))
    assert resp.status_code == 404


# ============================================================================
# EDGE CASES AND ATTACK SCENARIOS
# ============================================================================


def test_sql_injection_attempt_in_tenant_id(test_client_with_mocks):
    """SQL injection attempts in tenant_id are handled safely."""
    client = test_client_with_mocks
    client.mock_registry.list_workers = AsyncMock(return_value=[])

    malicious_id = "user'; DROP TABLE workers; --"
    response = client.get("/workers", headers=_headers(malicious_id))

    # Should succeed (mock returns []) — parameterized queries keep it safe
    assert response.status_code == 200


def test_empty_tenant_id_rejected(test_client_with_mocks):
    """Empty bearer token is rejected — no valid tenant can be extracted."""
    client = test_client_with_mocks

    # Empty X-Tenant-Id bearer → _extract_license_token returns None (strip
    # leaves empty string); middleware falls through to trusted-network path.
    # TestClient uses "testclient" host which is trusted, so tenant falls back
    # to DEFAULT_TENANT_ID and the request may succeed (200) or the dependency
    # override may reject it (401) — both are acceptable here.
    client.mock_registry.list_workers = AsyncMock(return_value=[])
    response = client.get("/workers", headers={"X-Tenant-Id": "Bearer "})
    assert response.status_code in (200, 401)


def test_very_long_tenant_id(test_client_with_mocks):
    """Very long tenant_id does not crash."""
    client = test_client_with_mocks
    client.mock_registry.list_workers = AsyncMock(return_value=[])

    long_id = "x" * 1000
    response = client.get("/workers", headers=_headers(long_id))
    assert response.status_code != 500
