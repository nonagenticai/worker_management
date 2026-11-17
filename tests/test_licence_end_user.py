"""Licence branch honours a verified ``X-End-User-Token`` (George 2026-10-15).

Licences are tenant×app only: fr (and other callers) will present a TENANT
``workers`` licence whose own subject is the tenant×app SERVICE subject, and name
the signed-in user in ``X-End-User-Token``. Before this change the licence branch
ignored that header, so ``subject_id`` was the service subject: workers registered
under it, ``visibility='user'`` scoping lost the user, and cost attribution went
to the service.

Driven through the REAL ``AuthenticationMiddleware``; only licence introspection
and the portal JWKS network fetch are stubbed.

MUTATIONS (run 2026-10-15): ``verify_end_user=not delegating`` -> ``False`` in
``_authenticate`` ⇒ 10 of 15 red (subject stays the service subject; garbage /
foreign / unverifiable tokens served 200). -> ``True`` ⇒ the delegated-path
garbage-token test goes red (delegation must stay never-refusing).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from src.api.identity_context import (
    get_actor_id_from_context,
    get_end_user_identity,
    get_subject_id_from_context,
    get_tenant_provenance_from_context,
    set_identity_context,
)
from src.auth import end_user_token, keycloak_roles
from src.auth.dependencies import TenantContext, get_tenant_context
from tests._platform_tenant import CUSTOMER_TENANT_A, CUSTOMER_TENANT_B

SERVICE_SUBJECT = "019ffe9f-0000-7000-8000-00000000a99a"  # tenant×app service subject
DELEGATOR = "019ffe9f-0000-7000-8000-00000000f00d"  # a delegating SA (fr today)
USER_SUB = "5d0c1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"  # Keycloak sub of the end user
ISSUER = "https://auth.nonagentic.ai/realms/portal"
KID = "portal-key-1"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    from src.settings import reset_settings_cache

    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.setenv(
        "KEYCLOAK_JWKS_URL",
        "http://keycloak.keycloak.svc.cluster.local:8080/realms/secure-apps/protocol/openid-connect/certs",
    )
    monkeypatch.delenv("END_USER_JWKS_URL", raising=False)
    monkeypatch.delenv("END_USER_TOKEN_ISSUER", raising=False)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", DELEGATOR)
    reset_settings_cache()
    end_user_token._jwks_cache.clear()
    yield
    end_user_token._jwks_cache.clear()
    set_identity_context(None, None, None)
    reset_settings_cache()


@pytest.fixture
def portal_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def portal_jwks(portal_key, monkeypatch: pytest.MonkeyPatch):
    pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(portal_key.public_key()))
    pub.update({"kid": KID, "alg": "RS256", "use": "sig"})

    async def fake_fetch(url: str) -> dict:
        return {"keys": [pub]}

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", fake_fetch)


def _mint(key, *, tenant: str = CUSTOMER_TENANT_A, sub: str = USER_SUB, **extra: Any) -> str:
    claims: dict[str, Any] = {
        "sub": sub,
        "iss": ISSUER,
        "exp": int(time.time()) + 600,
        "typ": "Bearer",
        "azp": "fr-frontend",
        "tenant_id": tenant,
        **extra,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def _licence(subject: str = SERVICE_SUBJECT, tenant: str = CUSTOMER_TENANT_A) -> dict[str, Any]:
    return {
        "active": True,
        "tenant_id": tenant,
        "subject_id": subject,
        "license_id": "lic-workers-1",
        "app": {"id": "a1", "key": "workers", "name": "Workers"},
        "byo_mode": "local_allowed",
    }


def _probe_app() -> FastAPI:
    from src.api.auth_middleware import AuthenticationMiddleware

    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @app.get("/api/workers")
    async def probe(request: Request, ctx: TenantContext = Depends(get_tenant_context)):  # noqa: B008
        sub, token_tenant, _sa, delegator = get_end_user_identity()
        return {
            "state_subject": request.state.subject_id,
            "ctx_subject": ctx.subject_id,
            "ctx_tenant": ctx.tenant_id,
            "ctx_subject_var": get_subject_id_from_context(),
            "actor": get_actor_id_from_context(),
            "provenance": get_tenant_provenance_from_context(),
            "eu_sub": sub,
            "eu_token_tenant": token_tenant,
            "eu_delegator": delegator,
            "license_id": request.state.license_id,
        }

    return app


def _get(headers: dict[str, str], licence: dict[str, Any] | None = None):
    client = TestClient(_probe_app(), raise_server_exceptions=False)
    with (
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=licence or _licence(),
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        return client.get("/api/workers", headers=headers)


AUTHZ: dict[str, str] = {"Authorization": "Bearer tenant-workers-licence"}


class TestLicenceBranchEndUser:
    def test_valid_user_token_makes_the_user_the_subject(self, portal_key, portal_jwks):
        resp = _get({**AUTHZ, "X-End-User-Token": f"Bearer {_mint(portal_key)}"})
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["state_subject"] == USER_SUB
        assert out["ctx_subject"] == USER_SUB
        assert out["ctx_subject_var"] == USER_SUB  # cost attribution ContextVar
        assert out["actor"] == USER_SUB
        assert out["ctx_tenant"] == CUSTOMER_TENANT_A
        assert out["provenance"] == "license"  # tenant_key_entitled unchanged
        assert out["eu_sub"] == USER_SUB
        assert out["eu_delegator"] == ""  # nothing was delegated
        assert out["license_id"] == "lic-workers-1"

    def test_member_via_member_tenant_ids_is_accepted(self, portal_key, portal_jwks):
        token = _mint(portal_key, tenant=CUSTOMER_TENANT_B, member_tenant_ids=[CUSTOMER_TENANT_A])
        resp = _get({**AUTHZ, "X-End-User-Token": token})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ctx_subject"] == USER_SUB

    def test_user_from_another_tenant_is_refused_403(self, portal_key, portal_jwks):
        resp = _get({**AUTHZ, "X-End-User-Token": _mint(portal_key, tenant=CUSTOMER_TENANT_B)})
        assert resp.status_code == 403, resp.text
        err = resp.json()["error"]
        assert err["code"] == "END_USER_TENANT_MISMATCH"
        assert err["failure_class"] == "tenant_mismatch"

    @pytest.mark.parametrize("token", ["garbage", "Bearer not.a.jwt"])
    def test_garbage_token_is_refused_401(self, token, portal_jwks):
        resp = _get({**AUTHZ, "X-End-User-Token": token})
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "END_USER_TOKEN_INVALID"
        # Not the licence challenge: a caller's licence replay must not fire.
        assert resp.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'

    def test_token_signed_by_another_key_is_refused_401(self, portal_jwks):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        resp = _get({**AUTHZ, "X-End-User-Token": _mint(other)})
        assert resp.status_code == 401, resp.text

    def test_id_token_is_refused_401(self, portal_key, portal_jwks):
        resp = _get({**AUTHZ, "X-End-User-Token": _mint(portal_key, typ="ID")})
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["failure_class"] == "not_access_token"

    def test_jwks_unreachable_is_503_not_a_fallback(self, portal_key, monkeypatch):
        import httpx2 as httpx

        async def boom(url: str) -> dict:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(keycloak_roles, "_fetch_jwks", boom)
        resp = _get({**AUTHZ, "X-End-User-Token": _mint(portal_key)})
        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "END_USER_TOKEN_UNVERIFIABLE"

    def test_without_token_unchanged(self, portal_jwks):
        resp = _get(AUTHZ)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["state_subject"] == SERVICE_SUBJECT
        assert out["ctx_subject"] == SERVICE_SUBJECT
        assert out["actor"] == ""
        assert out["eu_sub"] == ""
        assert out["provenance"] == "license"


class TestDelegatedPathUnchanged:
    """A delegating SA keeps X-Subject-Id + never-refusing proof semantics."""

    base: ClassVar[dict[str, str]] = {
        "Authorization": "Bearer sa-token",
        "X-Effective-Tenant-Id": CUSTOMER_TENANT_A,
        "X-Subject-Id": USER_SUB,
    }

    def test_garbage_end_user_token_is_not_refused_on_delegation(self, portal_jwks):
        resp = _get(
            {**self.base, "X-End-User-Token": "garbage"},
            licence=_licence(subject=DELEGATOR, tenant=CUSTOMER_TENANT_B),
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["ctx_tenant"] == CUSTOMER_TENANT_A
        assert out["ctx_subject"] == USER_SUB  # from X-Subject-Id, as before
        assert out["provenance"] == "delegated"
        assert out["eu_sub"] == ""

    def test_valid_end_user_token_on_delegation_records_delegator(self, portal_key, portal_jwks):
        resp = _get(
            {**self.base, "X-End-User-Token": _mint(portal_key)},
            licence=_licence(subject=DELEGATOR, tenant=CUSTOMER_TENANT_B),
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["provenance"] == "delegated"
        assert out["eu_delegator"] == DELEGATOR


# ------------------------------------------------- register_worker under the user

_MP = "src.api.workers.crud_endpoints"


def _workers_client():
    from src.api.auth_middleware import AuthenticationMiddleware
    from src.api.workers._router import router

    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    app.include_router(router)

    @asynccontextmanager
    async def _lifespan(a):
        yield

    app.router.lifespan_context = _lifespan
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def registry():
    import src.api.workers.crud_endpoints  # noqa: F401 -- registers the routes

    with (
        patch(f"{_MP}.worker_registry") as reg,
        patch(f"{_MP}.worker_cache") as cache,
        patch(f"{_MP}.StateDatabase") as state_db,
        patch(f"{_MP}.create_worker") as create,
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=_licence(),
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        reg.register_worker = AsyncMock(side_effect=lambda **kw: {"worker_id": kw["worker_id"]})
        reg.get_worker = AsyncMock(return_value=None)
        cache.remove_worker = AsyncMock()
        state_db.return_value = AsyncMock()
        fake = MagicMock()
        fake.db_path = "/tmp/licence-end-user.db"
        create.return_value = fake
        yield reg


def _post_worker(headers: dict[str, str]):
    with _workers_client() as client:
        return client.post(
            "/workers",
            headers={**AUTHZ, "X-Visibility": "user", **headers},
            json={"worker_name": "mine", "model_provider": "openai", "model": "gpt-4"},
        )


def test_register_worker_lands_under_the_user(registry, portal_key, portal_jwks):
    resp = _post_worker({"X-End-User-Token": _mint(portal_key)})
    assert resp.status_code == 201, resp.text
    kw = registry.register_worker.await_args.kwargs
    assert kw["subject_id"] == USER_SUB
    assert kw["tenant_id"] == CUSTOMER_TENANT_A
    assert kw["visibility"] == "user"


def test_register_worker_without_token_lands_under_the_licence_subject(registry, portal_jwks):
    resp = _post_worker({})
    assert resp.status_code == 201, resp.text
    assert registry.register_worker.await_args.kwargs["subject_id"] == SERVICE_SUBJECT


def test_register_worker_refused_for_garbage_token(registry, portal_jwks):
    resp = _post_worker({"X-End-User-Token": "garbage"})
    assert resp.status_code == 401, resp.text
    registry.register_worker.assert_not_awaited()


def test_verify_licence_end_user_never_raises_on_absent():
    payload, reason = asyncio.run(end_user_token.verify_licence_end_user(None, CUSTOMER_TENANT_A))
    assert payload is None
    assert reason == end_user_token.PROOF_ABSENT
