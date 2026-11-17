"""Tests for Phase 3: tenant_name column across all owned tables.

Verifies:
- get_tenant_name dependency returns correct values from request state
- AuthenticationMiddleware stashes tenant_name on request.state
- DB layer methods accept tenant_name parameter with correct defaults
- Fallback: tenant_name="" in DB layer resolves to tenant_id
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from src.auth.dependencies import get_tenant_name

# ---------------------------------------------------------------------------
# get_tenant_name dependency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_tenant_name_from_state() -> None:
    """Returns the tenant_name stashed on request.state by middleware."""
    request = MagicMock(spec=Request)
    request.state.tenant_name = "acme-corp"
    result = await get_tenant_name(request)
    assert result == "acme-corp"


@pytest.mark.asyncio
async def test_get_tenant_name_missing_attr_returns_empty() -> None:
    """Returns empty string when request.state has no tenant_name attribute."""
    request = MagicMock(spec=Request)
    # Simulate missing attribute (hasattr → False via MagicMock spec)
    del request.state.tenant_name
    result = await get_tenant_name(request)
    assert result == ""


@pytest.mark.asyncio
async def test_get_tenant_name_none_attr_returns_empty() -> None:
    """Returns empty string when tenant_name is None on request.state."""
    request = MagicMock(spec=Request)
    request.state.tenant_name = None
    result = await get_tenant_name(request)
    assert result == ""


@pytest.mark.asyncio
async def test_get_tenant_name_empty_string_returns_empty() -> None:
    """Passes through an existing empty string."""
    request = MagicMock(spec=Request)
    request.state.tenant_name = ""
    result = await get_tenant_name(request)
    assert result == ""


# ---------------------------------------------------------------------------
# Middleware tenant_name stashing
# ---------------------------------------------------------------------------


class _FakeHeaders:
    """Minimal header dict that supports .get() with case-insensitive lookups."""

    def __init__(self, raw: dict[str, str]) -> None:
        self._data = {k.lower(): v for k, v in raw.items()}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key.lower(), default)

    def __getitem__(self, key: str) -> str:
        return self._data[key.lower()]

    def __contains__(self, key: object) -> bool:
        return str(key).lower() in self._data


def _make_request_mock(headers: dict[str, str], client_host: str = "testclient") -> MagicMock:
    """Build a mock Request with the given headers and client."""
    request = MagicMock()
    request.url.path = "/workers"
    request.headers = _FakeHeaders(headers)
    client = MagicMock()
    client.host = client_host
    request.client = client
    request.state = MagicMock()
    return request


@pytest.mark.asyncio
async def test_middleware_stashes_tenant_name_from_header_trusted_network() -> None:
    """Middleware stashes X-Tenant-Name on request.state for trusted-network paths."""
    from src.api.auth_middleware import AuthenticationMiddleware

    middleware = AuthenticationMiddleware(app=MagicMock(), bypass_auth=False)

    request = _make_request_mock(
        headers={"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003", "X-Tenant-Name": "acme"},
        client_host="127.0.0.1",
    )

    response_mock = MagicMock()

    async def call_next(req: Any) -> Any:
        return response_mock

    await middleware.dispatch(request, call_next)
    assert request.state.tenant_name == "acme"


@pytest.mark.asyncio
async def test_middleware_stashes_empty_tenant_name_when_header_absent_trusted_network() -> None:
    """Middleware stashes empty string when X-Tenant-Name header is absent."""
    from src.api.auth_middleware import AuthenticationMiddleware

    middleware = AuthenticationMiddleware(app=MagicMock(), bypass_auth=False)

    request = _make_request_mock(
        headers={"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000003"},
        client_host="127.0.0.1",
    )

    response_mock = MagicMock()

    async def call_next(req: Any) -> Any:
        return response_mock

    await middleware.dispatch(request, call_next)
    assert request.state.tenant_name == ""


@pytest.mark.asyncio
async def test_middleware_bypass_stashes_the_bypass_label() -> None:
    """Bypass-auth path also stashes tenant_name (the bypass label)."""
    from src.api.auth_middleware import AuthenticationMiddleware

    middleware = AuthenticationMiddleware(app=MagicMock(), bypass_auth=True)
    request = _make_request_mock(headers={})

    response_mock = MagicMock()

    async def call_next(req: Any) -> Any:
        return response_mock

    await middleware.dispatch(request, call_next)
    # The bypass path stamps the readable label. It used to stamp "" while
    # putting the literal "test_user" in tenant_id; now tenant_id is a uuid
    # (BYPASS_TENANT_UUID) and the label carries the name. What this test exists
    # to pin is that the attribute is SET AT ALL on this path, so `get_tenant_name`
    # can never read a stale or missing one — and it still pins exactly that.
    assert request.state.tenant_name == "test_user"


# ---------------------------------------------------------------------------
# DB layer: register_worker accepts tenant_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_worker_accepts_tenant_name() -> None:
    """register_worker passes tenant_name through to the INSERT."""
    from src.db.worker_registry_db import WorkerRegistryDB

    registry = WorkerRegistryDB()

    # Record EVERY statement, not just the most recent one. register_worker
    # issues more than one execute() on the create path -- it also re-keys any
    # name-keyed worker_states rows onto the new worker_id (see
    # WorkerRegistryDB._rekey_name_scoped_states). Keeping only the last call
    # made this assert read the re-key's args instead of the INSERT's, which is
    # a property of the fake, not of the behaviour under test.
    calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fake_execute(query: str, *args: Any) -> str:
        calls.append((query, args))
        return "INSERT 1"

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(side_effect=fake_execute)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.worker_registry_db.get_pg_connection", return_value=mock_ctx):
        await registry.register_worker(
            tenant_id="t1",
            subject_id="s1",
            tenant_name="acme-corp",
            worker_name="my-worker",
            db_path="/tmp/x.db",
            is_agent=False,
            config={},
            visibility="tenant",
        )

    inserts = [c for c in calls if c[0].lstrip().upper().startswith("INSERT")]
    assert len(inserts) == 1, "expected exactly one INSERT into workers"

    # tenant_name should appear in INSERT args
    assert "acme-corp" in inserts[0][1]


@pytest.mark.asyncio
async def test_register_worker_falls_back_to_tenant_id_when_name_empty() -> None:
    """When tenant_name='', the DB receives tenant_id as the tenant_name value."""
    from src.db.worker_registry_db import WorkerRegistryDB

    registry = WorkerRegistryDB()

    captured: dict[str, Any] = {}

    async def fake_execute(query: str, *args: Any) -> str:
        captured["args"] = args
        return "INSERT 1"

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(side_effect=fake_execute)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.worker_registry_db.get_pg_connection", return_value=mock_ctx):
        await registry.register_worker(
            tenant_id="my-tenant",
            subject_id="s1",
            tenant_name="",  # empty → fallback to tenant_id
            worker_name="my-worker",
            db_path="/tmp/x.db",
            is_agent=False,
            config={},
            visibility="tenant",
        )

    # With empty tenant_name, the value inserted should be tenant_id
    assert "my-tenant" in captured.get("args", ())


# ---------------------------------------------------------------------------
# DB layer: add_worker_state accepts tenant_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_worker_state_accepts_tenant_name() -> None:
    """add_worker_state passes tenant_name through to the INSERT."""
    from src.db.workers_state_db import StateDatabase

    db = StateDatabase()

    captured: dict[str, Any] = {}

    async def fake_fetchrow(query: str, *args: Any) -> dict[str, Any]:
        captured["args"] = args
        return {"id": 42}

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(side_effect=fake_fetchrow)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.workers_state_db.get_pg_connection", return_value=mock_ctx):
        state_id = await db.add_worker_state(
            tenant_id="t1",
            subject_id="s1",
            tenant_name="acme-corp",
            worker_id="w1",
            worker_name="wname",
            is_agent=False,
            state_type="created",
            worker_state={"status": "ok"},
        )

    assert state_id == 42
    assert "acme-corp" in captured.get("args", ())


# ---------------------------------------------------------------------------
# DB layer: create_graph accepts tenant_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_graph_accepts_tenant_name() -> None:
    """create_graph passes tenant_name through to the INSERT."""
    from src.db.graph_state_db import GraphStateDatabase

    db = GraphStateDatabase()

    captured: dict[str, Any] = {}

    async def fake_execute(query: str, *args: Any) -> str:
        captured["args"] = args
        return "INSERT 1"

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=fake_execute)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.graph_state_db.get_pg_connection", return_value=mock_ctx):
        await db.create_graph(
            tenant_id="t1",
            subject_id="s1",
            tenant_name="acme-corp",
            graph_id="00000000-0000-0000-0000-000000000001",
            name="test-graph",
        )

    assert "acme-corp" in captured.get("args", ())
