"""The nil uuid is not a tenant — refused BY VALUE on all three inbound sites.

⛔ WHAT WAS OPEN. ``worker_management`` had no nil check anywhere. Its only
mention of the value went the OTHER way: ``worker/factory/llm_config_resolver.py``
lists it in ``_SYSTEM_TENANTS``, i.e. as a privileged platform tenant that must
never 402. The two ``if ctx_tenant:`` guards in ``worker/agents/pydantic_ai_agent.py``
are OUTBOUND header-stamping and gate on truthiness only, so nothing stopped a
request from simply *asserting* the nil tenant and being served — on the tokenless
trusted-network branch (reachable from any pod in ``10.0.0.0/8``), on the
``X-Effective-Tenant-Id`` delegation header, or via an introspected licence claim.
The nil uuid is *registered* in ``licenses.tenants`` as ``default-anon``, so it
also survives a dirty-row audit while collapsing unattributed callers into one
bucket; an anonymous principal was once granted admin on exactly this tenant.

Seven sibling services already reject it at their inbound boundary
(``w2v_orchestrator/src/auth/run_tenant.py``,
``video_production_service/src/api/tenant.py`` and five more). This is the same
rule, applied in ``tenant_constants.resolve_tenant_id`` — the single funnel all
three sites already share.

Every test asserts the REFUSAL (the HTTP status the boundary answers) or the
ADMISSION of a real tenant. No mock call counts: this estate has shipped
call-count tests that passed straight through the defect they were written for.

Mutation proof: delete ``if parsed.int == 0: return None`` from
``resolve_tenant_id`` — the refusal tests below go red (200 instead of 400/401)
while every CONTROL stays green.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.api.auth_middleware as auth_middleware_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.tenant_constants import NIL_TENANT_UUID, resolve_tenant_id

NIL = "00000000-0000-0000-0000-000000000000"
REAL_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
SA_SUBJECT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, object]:
        return {"tenant_id": request.state.tenant_id}

    return application


def _call(app: FastAPI, headers: dict[str, str]) -> tuple[int, dict]:
    """Hit the app from a REAL peer inside TRUSTED_CIDRS (no ``testclient`` hatch)."""
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/api/probe", headers=headers)
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body


# --------------------------------------------------------------------------- #
# The resolver itself — the single funnel.
# --------------------------------------------------------------------------- #


def test_the_constant_is_the_nil_uuid() -> None:
    assert NIL_TENANT_UUID == NIL


def test_resolve_tenant_id_refuses_the_nil_uuid() -> None:
    assert resolve_tenant_id(NIL) is None
    assert resolve_tenant_id(NIL.upper()) is None
    assert resolve_tenant_id(f"  {NIL}  ") is None


def test_resolve_tenant_id_still_resolves_a_real_tenant_and_a_known_name() -> None:
    """CONTROL: the guard refuses the sentinel, not tenants in general."""
    assert resolve_tenant_id(REAL_TENANT) == REAL_TENANT
    assert resolve_tenant_id("n8n-dev") == REAL_TENANT


# --------------------------------------------------------------------------- #
# Site 1 — the tokenless trusted-network X-Tenant-Id header (400).
# --------------------------------------------------------------------------- #


def test_a_trusted_network_caller_asserting_the_nil_tenant_is_400(app: FastAPI) -> None:
    status, _ = _call(app, {"X-Tenant-Id": NIL})
    assert status == 400


def test_a_trusted_network_caller_naming_a_real_tenant_still_passes(app: FastAPI) -> None:
    """CONTROL for the site above — "refuse every header" would otherwise pass it."""
    with patch(
        "src.api.auth_middleware._lookup_tenant_byo_mode",
        new_callable=AsyncMock,
        return_value="local_allowed",
    ):
        status, body = _call(app, {"X-Tenant-Id": REAL_TENANT})
    assert status == 200
    assert body["tenant_id"] == REAL_TENANT


# --------------------------------------------------------------------------- #
# Site 2 — the introspected licence claim (401: the TOKEN is unusable).
# --------------------------------------------------------------------------- #


def test_a_licence_whose_tenant_claim_is_the_nil_uuid_is_401(app: FastAPI) -> None:
    introspect = {"active": True, "tenant_id": NIL, "subject_id": "user-uuid-1"}
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ),
    ):
        status, _ = _call(app, {"X-Tenant-Id": "Bearer a-real-licence-token"})
    assert status == 401


def test_a_licence_with_a_real_tenant_claim_still_passes(app: FastAPI) -> None:
    """CONTROL for the licensed branch."""
    introspect = {"active": True, "tenant_id": REAL_TENANT, "subject_id": "user-uuid-1"}
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ),
    ):
        status, body = _call(app, {"X-Tenant-Id": "Bearer a-real-licence-token"})
    assert status == 200
    assert body["tenant_id"] == REAL_TENANT


# --------------------------------------------------------------------------- #
# Site 3 — the X-Effective-Tenant-Id delegation header (400).
#
# This is the site that matters most: it is the one header whose VALUE a caller
# supplies and this service then ADOPTS as the request tenant. A delegating
# service account re-scoping a run onto the nil tenant is precisely the
# unattributable-spend hole w2v_orchestrator's `billable_tenant_id` refuses at
# the other end of the same hop.
# --------------------------------------------------------------------------- #


def test_a_delegating_sa_may_not_re_scope_onto_the_nil_tenant(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    introspect = {
        "active": True,
        "tenant_id": "22222222-2222-4222-8222-222222222222",
        "subject_id": SA_SUBJECT,
        "byo_mode": "platform_only",
    }
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        status, _ = _call(
            app,
            {"X-Tenant-Id": "Bearer sa-token", "X-Effective-Tenant-Id": NIL},
        )
    assert status == 400


def test_a_delegating_sa_may_still_re_scope_onto_a_real_tenant(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONTROL: delegation itself is untouched — only the sentinel is refused."""
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    introspect = {
        "active": True,
        "tenant_id": "22222222-2222-4222-8222-222222222222",
        "subject_id": SA_SUBJECT,
        "byo_mode": "platform_only",
    }
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        status, body = _call(
            app,
            {"X-Tenant-Id": "Bearer sa-token", "X-Effective-Tenant-Id": REAL_TENANT},
        )
    assert status == 200
    assert body["tenant_id"] == REAL_TENANT
