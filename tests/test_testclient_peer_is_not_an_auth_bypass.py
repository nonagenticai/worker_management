"""Regression guard: the `testclient` peer name is not an authentication bypass.

`_is_trusted_network` used to return True for the literal peer host `"testclient"`,
Starlette's `TestClient` sentinel (and `httpx.ASGITransport`'s default). That arm was
NOT a narrowing of TRUSTED_CIDRS — `"testclient"` is not an IP and matches no CIDR — so
it granted the tokenless trusted-network branch to any request reporting that peer name,
from any network, and it shipped in the production image.

The branch it unlocked does more than skip auth: it binds `request.state.tenant_id`
directly from a bare `X-Tenant-Id` header, so it was a tenant-SELECTION primitive.

Same hatch, same fix, in `tool_registry` (#23) and `kag_common` 0.20.0 (#25).

⚠️ On the RED side, do not accept an ImportError or an AttributeError as proof. The
assertions below are on the function's actual return value and on a real HTTP status
code, so a rename fails loudly rather than passing vacuously.
"""

from __future__ import annotations

import ipaddress

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth_middleware import (
    TRUSTED_NETWORKS,
    AuthenticationMiddleware,
    _is_trusted_network,
)


def test_testclient_peer_is_not_trusted() -> None:
    """The literal sentinel is rejected, and it is rejected as a NON-ADDRESS."""
    assert _is_trusted_network("testclient") is False
    # False-positive control: the rejection must come from `testclient` not being an
    # address, not from TRUSTED_NETWORKS having collapsed to empty — an empty CIDR list
    # would make this test pass while proving nothing about the hatch.
    assert TRUSTED_NETWORKS, "TRUSTED_CIDRS parsed to nothing; this test would be vacuous"
    with pytest.raises(ValueError):
        ipaddress.ip_address("testclient")


def test_a_real_trusted_peer_is_still_trusted() -> None:
    """The positive half: removing the hatch must not break genuine trusted-network callers."""
    assert _is_trusted_network("127.0.0.1") is True
    assert _is_trusted_network("10.42.0.1") is True


def test_untrusted_peer_is_not_trusted() -> None:
    assert _is_trusted_network("203.0.113.9") is False


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware)

    @application.get("/api/probe")
    async def probe() -> dict[str, str]:  # pragma: no cover - body only runs when admitted
        return {"ok": "yes"}

    return application


def test_end_to_end_testclient_peer_with_bare_tenant_header_is_401(app: FastAPI) -> None:
    """The exploit shape, end to end through the real middleware.

    A tokenless request naming an arbitrary tenant, whose peer is the TestClient
    sentinel. Before the fix this returned 200 with `request.state.tenant_id` bound to
    the attacker-supplied value.
    """
    with TestClient(app) as client:
        resp = client.get(
            "/api/probe",
            # fr-test is a REAL tenant, and not this caller's. The header has to
            # be perfectly valid: if it were not, the 401 could be attributed to
            # the tenant resolver rejecting a bogus value rather than to the peer
            # name, and the escape hatch would be silently un-pinned.
            headers={"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000099"},
        )
    assert resp.status_code == 401, (
        f"testclient peer admitted with {resp.status_code}; the production escape hatch is back."
    )


def test_end_to_end_real_trusted_peer_with_bare_tenant_header_is_200(app: FastAPI) -> None:
    """Non-vacuity control for the test above.

    Identical request, only the peer address differs. If this were also 401 the 401
    above would prove nothing about the peer name — it could be any unrelated rejection
    (a missing route, a middleware ordering change, a config fail-closed).
    """
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get(
            "/api/probe",
            # fr-test is a REAL tenant, and not this caller's. The header has to
            # be perfectly valid: if it were not, the 401 could be attributed to
            # the tenant resolver rejecting a bogus value rather than to the peer
            # name, and the escape hatch would be silently un-pinned.
            headers={"X-Tenant-Id": "aaaaaaaa-0000-4000-8000-000000000099"},
        )
    assert resp.status_code == 200, (
        f"trusted-network peer got {resp.status_code}; the 401 above is not "
        "attributable to the peer name."
    )
