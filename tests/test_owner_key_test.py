"""Owner requirement 2026-10-02: the owner (portal ``super_admin``) tests a
CUSTOMER tenant's stored BYO key from fr's admin tenant page.

fr delegates as svc-fr-backend (``X-Effective-Tenant-Id`` = the customer) and
forwards the owner's own portal JWT, whose ``tenant_id`` is the OWNER's tenant.

Pinned here, through the REAL ``AuthenticationMiddleware`` and the REAL
``/tenant-config`` router (only licence introspection, the portal JWKS fetch,
the DB and the vendor call are stubbed):

- super_admin JWT + foreign tenant: ``POST /llm/test`` reaches the vendor, ok.
- non-super_admin JWT + foreign tenant: withheld, the vendor is never called.
- super_admin JWT: ``tenant_key_entitled`` stays False, so the resolver,
  ``/providers`` and ``/chat`` (all of which read only that) still withhold.
- missing / forged / expired JWT: withheld.
- owner PUT / DELETE / test is audit-logged with the actor ``sub``, never the key.

MUTATIONS: drop ``owner_key_test_entitled`` from the test route ⇒ the allowed
case goes red. Widen ``tenant_key_entitled`` with the super_admin arm ⇒ the
"stays False" case goes red. Skip the role check ⇒ the non-super_admin case goes
red. Drop the audit call ⇒ the audit cases go red.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx2 as httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.identity_context import (
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    set_end_user_identity,
    set_identity_context,
)
from src.auth import end_user_token, keycloak_roles
from src.security.secret_box import encrypt_secret
from src.worker.factory import llm_config_resolver as r
from tests._platform_tenant import CUSTOMER_TENANT_A, CUSTOMER_TENANT_B, PLATFORM_TENANT
from tests._tenant_llm_audit_fake import AuditFakeMixin

SA_SUBJECT = "019ffe9f-0000-7000-8000-00000000f00d"  # svc-fr-backend stand-in
SA_TENANT = "22222222-2222-4222-8222-222222222222"
ISSUER = "https://auth.nonagentic.ai/realms/portal"
KID = "portal-key-1"
OWNER_SUB = "owner-sub-0001"
TENANT_KEY = "sk-customer-a-own-openai-key"  # pragma: allowlist secret


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
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
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


def _mint(
    key,
    *,
    roles: tuple[str, ...] = ("super_admin",),
    tenant: str = PLATFORM_TENANT,
    exp: int = 600,
    top_level_roles: tuple[str, ...] = (),
    client_roles: dict[str, tuple[str, ...]] | None = None,
) -> str:
    claims: dict[str, Any] = {
        "sub": OWNER_SUB,
        "iss": ISSUER,
        "exp": int(time.time()) + exp,
        "tenant_id": tenant,
        "realm_access": {"roles": list(roles)},
        # A real Keycloak access token from fr's SPA client (WM-3 pins both).
        "typ": "Bearer",
        "azp": "fr-frontend",
    }
    if top_level_roles:
        claims["roles"] = list(top_level_roles)
    if client_roles:
        claims["resource_access"] = {c: {"roles": list(r)} for c, r in client_roles.items()}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def _row(**kw: Any) -> dict[str, Any]:
    base = {
        "tenant_id": CUSTOMER_TENANT_A,
        "role": "__default__",
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "params": {},
        "api_key_encrypted": encrypt_secret(TENANT_KEY),
        "cost_source": "worker-management",
        "owner_provided": False,
    }
    base.update(kw)
    return base


class _DB(AuditFakeMixin):
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {
            (CUSTOMER_TENANT_A, "__default__"): _row()
        }

    async def list_configs(self, tenant_id: str) -> list[dict[str, Any]]:
        return [dict(v) for (t, _), v in self.rows.items() if t == tenant_id]

    async def get_config(self, tenant_id: str, role: str = "__default__") -> dict[str, Any] | None:
        row = self.rows.get((str(tenant_id), role))
        return dict(row) if row else None

    async def upsert_config(self, *, tenant_id: str, role: str, **kw: Any) -> None:
        self.take_audit(tenant_id, role, kw)
        self.rows[(str(tenant_id), role)] = {"tenant_id": str(tenant_id), "role": role, **kw}

    async def delete_config(
        self, tenant_id: str, role: str, *, audit_actor_kind: str | None = None
    ) -> bool:
        gone = self.rows.pop((str(tenant_id), role), None)
        self.audit_delete(tenant_id, role, gone, audit_actor_kind)
        return gone is not None


def _vendor_ok():
    calls: list[str] = []

    async def fake_get(url: str, *, headers=None, timeout=10.0):
        calls.append(url)
        if (headers or {}).get("Authorization") == f"Bearer {TENANT_KEY}":
            return {"data": [{"id": "gpt-4o"}]}
        raise httpx.HTTPStatusError(
            "401 Unauthorized", request=httpx.Request("GET", url), response=httpx.Response(401)
        )

    return fake_get, calls


def _app() -> FastAPI:
    from src.api.auth_middleware import AuthenticationMiddleware
    from src.api.tenant_config._router import router

    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)
    app.include_router(router)

    @app.get("/api/workers")
    async def probe():
        from src.api.workers._globals import _cache_key

        own = r.tenant_key_entitled(CUSTOMER_TENANT_A)
        return {
            "tenant_key_entitled": own,
            "owner_key_test_entitled": r.owner_key_test_entitled(CUSTOMER_TENANT_A),
            "cache_key": _cache_key("w", CUSTOMER_TENANT_A, key_entitled=own),
        }

    return app


@pytest.fixture
def client(monkeypatch):
    from src.api.tenant_config import _router as tc

    db = _DB()
    monkeypatch.setattr(tc, "_db", db)
    monkeypatch.setattr(r, "_db", db)
    tc._key_test_cooldown.clear()
    with (
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "tenant_id": SA_TENANT,
                "subject_id": SA_SUBJECT,
                "byo_mode": "platform_only",
            },
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        yield TestClient(_app(), raise_server_exceptions=False), db
    tc._key_test_cooldown.clear()


BASE: dict[str, str] = {
    "X-Tenant-Id": "Bearer sa-token",
    "X-Effective-Tenant-Id": CUSTOMER_TENANT_A,
}


def _test(c: TestClient, headers: dict[str, str]):
    fake_get, calls = _vendor_ok()
    with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
        resp = c.post("/tenant-config/llm/test", json={}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert TENANT_KEY not in resp.text
    return resp.json(), calls


class TestOwnerKeyTestEndpoint:
    def test_super_admin_with_a_foreign_tenant_may_test(self, client, portal_key, portal_jwks):
        c, _ = client
        body, calls = _test(c, {**BASE, "X-End-User-Token": f"Bearer {_mint(portal_key)}"})
        assert body["ok"] is True, body
        assert body["provider"] == "openai"
        assert len(calls) == 1

    def test_non_super_admin_with_a_foreign_tenant_is_withheld(
        self, client, portal_key, portal_jwks
    ):
        c, _ = client
        token = _mint(portal_key, roles=("tenant_admin",))
        body, calls = _test(c, {**BASE, "X-End-User-Token": token})
        assert body["ok"] is False
        assert "X-End-User-Token" in body["error"]
        assert calls == []

    def test_top_level_roles_claim_is_not_the_realm_role(self, client, portal_key, portal_jwks):
        """fr reads ``realm_access.roles`` only; a top-level ``roles`` does not count."""
        c, _ = client
        token = _mint(portal_key, roles=(), top_level_roles=("super_admin",))
        body, calls = _test(c, {**BASE, "X-End-User-Token": token})
        assert body["ok"] is False
        assert calls == []

    @pytest.mark.parametrize("client_id", ["fr", "account", "worker-management"])
    def test_client_role_super_admin_is_not_the_realm_role(
        self, client, portal_key, portal_jwks, client_id
    ):
        """``resource_access.<client>.roles`` is a CLIENT role, not the realm role.

        Pins that only ``realm_access.roles`` grants the owner path: reading
        client roles as super_admin would let any client admin test a foreign key.
        """
        c, _ = client
        token = _mint(portal_key, roles=(), client_roles={client_id: ("super_admin",)})
        body, calls = _test(c, {**BASE, "X-End-User-Token": token})
        assert body["ok"] is False
        assert calls == []

    @pytest.mark.parametrize("case", ["missing", "forged", "expired"])
    def test_missing_or_invalid_token_is_withheld(self, client, portal_key, portal_jwks, case):
        c, _ = client
        headers = dict(BASE)
        if case == "forged":
            other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            headers["X-End-User-Token"] = _mint(other)
        elif case == "expired":
            headers["X-End-User-Token"] = _mint(portal_key, exp=-120)
        body, calls = _test(c, headers)
        assert body["ok"] is False
        assert calls == []


class TestOwnerNeverGetsInferenceEntitlement:
    def test_super_admin_does_not_make_tenant_key_entitled_true(
        self, client, portal_key, portal_jwks
    ):
        c, _ = client
        resp = c.get("/api/workers", headers={**BASE, "X-End-User-Token": _mint(portal_key)})
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["owner_key_test_entitled"] is True
        assert out["tenant_key_entitled"] is False
        assert not out["cache_key"].endswith(":byo-key")

    def test_resolver_still_withholds_under_an_owner_identity(self, monkeypatch):
        """The resolver path (/chat, workers) reads tenant_key_entitled only."""
        set_identity_context(
            CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )
        set_end_user_identity(
            sub=OWNER_SUB, token_tenant=PLATFORM_TENANT, super_admin=True, delegator=SA_SUBJECT
        )
        assert r.owner_key_test_entitled(CUSTOMER_TENANT_A) is True
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is False


class TestOwnerPredicateUnit:
    def _owner(self, *, provenance=TENANT_PROVENANCE_DELEGATED, delegator=SA_SUBJECT, **kw):
        set_identity_context(CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=provenance)
        args = {"sub": OWNER_SUB, "token_tenant": PLATFORM_TENANT, "super_admin": True}
        args.update(kw)
        set_end_user_identity(delegator=delegator, **args)

    def test_delegator_removed_from_config_loses_it(self, monkeypatch):
        self._owner()
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "someone-else")
        assert r.owner_key_test_entitled(CUSTOMER_TENANT_A) is False

    def test_licence_provenance_is_not_the_owner_path(self):
        self._owner(provenance=TENANT_PROVENANCE_LICENSE)
        assert r.owner_key_test_entitled(CUSTOMER_TENANT_A) is False

    def test_other_tenant_than_the_request_is_refused(self):
        self._owner()
        assert r.owner_key_test_entitled(PLATFORM_TENANT) is False

    def test_no_sub_is_refused(self):
        self._owner(sub="")
        assert r.owner_key_test_entitled(CUSTOMER_TENANT_A) is False

    def test_a_new_identity_clears_it(self):
        self._owner()
        set_identity_context(
            CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )
        assert r.owner_key_test_entitled(CUSTOMER_TENANT_A) is False


class TestOwnerAuditLog:
    logger_name: ClassVar[str] = "src.api.tenant_config._router"

    def _owner_lines(self, caplog) -> list[str]:
        return [
            rec.getMessage()
            for rec in caplog.records
            if rec.name == self.logger_name and "owner_action" in rec.getMessage()
        ]

    def test_put_delete_and_test_are_audited_without_the_key(
        self, client, portal_key, portal_jwks, caplog
    ):
        c, db = client
        headers = {**BASE, "X-End-User-Token": _mint(portal_key)}
        caplog.set_level(logging.INFO, logger=self.logger_name)
        fake_get, _ = _vendor_ok()
        with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
            put = c.put(
                "/tenant-config/llm?tier=medium",
                json={"provider": "anthropic", "model": "claude-x", "apiKey": TENANT_KEY},
                headers=headers,
            )
            assert put.status_code == 200, put.text
            assert c.post("/tenant-config/llm/test", json={}, headers=headers).status_code == 200
            assert c.delete("/tenant-config/llm?tier=medium", headers=headers).status_code == 204
        lines = self._owner_lines(caplog)
        actions = [ln.split("action=")[1].split()[0] for ln in lines]
        assert actions == ["put", "test", "delete"], lines
        for ln in lines:
            assert f"tenant={CUSTOMER_TENANT_A}" in ln
            assert f"actor_sub={OWNER_SUB}" in ln
            assert "actor_super_admin=True" in ln
        assert "provider=anthropic" in lines[0] and "tier=medium" in lines[0]
        assert "provider=anthropic" in lines[2]
        assert "owner_key_test=True" in lines[1]
        assert all(TENANT_KEY not in rec.getMessage() for rec in caplog.records)
        assert (CUSTOMER_TENANT_A, "tier:medium") not in db.rows

    def test_owner_action_records_the_stored_key_fingerprint_not_the_key(
        self, client, portal_key, portal_jwks, caplog
    ):
        c, db = client
        headers = {**BASE, "X-End-User-Token": _mint(portal_key)}
        caplog.set_level(logging.INFO, logger=self.logger_name)
        fake_get, _ = _vendor_ok()
        with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
            put = c.put(
                "/tenant-config/llm?tier=medium",
                json={"provider": "anthropic", "model": "claude-x", "apiKey": TENANT_KEY},
                headers=headers,
            )
            assert put.status_code == 200, put.text
            stored = db.rows[(CUSTOMER_TENANT_A, "tier:medium")]["api_key_encrypted"]
            assert stored and TENANT_KEY not in str(stored)
            assert (
                c.post("/tenant-config/llm/test", json={"tier": "medium"}, headers=headers)
            ).status_code == 200
            assert c.delete("/tenant-config/llm?tier=medium", headers=headers).status_code == 204
            # No key involved: a keyless row deleted again records key_id=none.
            assert c.delete("/tenant-config/llm?tier=medium", headers=headers).status_code == 204
        raw = stored if isinstance(stored, bytes) else str(stored).encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()[:12]
        lines = self._owner_lines(caplog)
        assert [ln.split("action=")[1].split()[0] for ln in lines] == [
            "put",
            "test",
            "delete",
            "delete",
        ], lines
        for ln in lines[:3]:
            assert f"key_id={expected}" in ln, ln
        assert "key_id=none" in lines[3], lines[3]
        plain_hash = hashlib.sha256(TENANT_KEY.encode("utf-8")).hexdigest()
        for rec in caplog.records:
            msg = rec.getMessage()
            assert TENANT_KEY not in msg
            assert plain_hash[:12] not in msg

    def test_same_tenant_user_is_not_an_owner_action(self, client, portal_key, portal_jwks, caplog):
        c, _ = client
        caplog.set_level(logging.INFO, logger=self.logger_name)
        token = _mint(portal_key, roles=("tenant_admin",), tenant=CUSTOMER_TENANT_A)
        body, _ = _test(c, {**BASE, "X-End-User-Token": token})
        assert body["ok"] is True  # the tenant's own proven user, unchanged path
        assert self._owner_lines(caplog) == []


class TestCustomerVisibleAudit:
    """Owner ruling 2026-10-02: the CUSTOMER sees who set / tested / removed its key.

    Through the REAL middleware and router: every PUT / DELETE / TEST writes an
    audit row naming the KIND of actor (never the ``sub``) and the key id (never
    the key), and ``GET /tenant-config/llm/audit`` serves the effective tenant's
    rows only, newest first, capped at 200.

    MUTATIONS: drop the audit kwargs from PUT ⇒ ``test_every_caller_kind_is_recorded``
    red. Classify any verified user as workspace_admin (drop the tenant
    comparison) ⇒ the foreign non-super_admin ``service`` row goes red. Query another tenant ⇒ the scoping case goes red.
    Drop the ``min(limit, 200)`` ⇒ the cap case goes red. Swallow the TEST audit
    failure ⇒ the 503 case goes red.
    """

    def _owner(self, portal_key) -> dict[str, str]:
        return {**BASE, "X-End-User-Token": f"Bearer {_mint(portal_key)}"}

    def _admin(self, portal_key) -> dict[str, str]:
        token = _mint(portal_key, roles=("tenant_admin",), tenant=CUSTOMER_TENANT_A)
        return {**BASE, "X-End-User-Token": f"Bearer {token}"}

    def test_every_caller_kind_is_recorded(self, client, portal_key, portal_jwks):
        c, db = client
        fake_get, _ = _vendor_ok()
        with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
            # platform_operator: set a key on tier medium, test it, remove it.
            owner = self._owner(portal_key)
            put = c.put(
                "/tenant-config/llm?tier=medium",
                json={"provider": "openai", "model": "gpt-4o", "apiKey": TENANT_KEY},
                headers=owner,
            )
            assert put.status_code == 200, put.text
            assert (
                c.post("/tenant-config/llm/test", json={"tier": "medium"}, headers=owner)
            ).status_code == 200
            assert c.delete("/tenant-config/llm?tier=medium", headers=owner).status_code == 204
            # Deleting an absent row removes nothing and records nothing.
            assert c.delete("/tenant-config/llm?tier=medium", headers=owner).status_code == 204
            # workspace_admin: a model-only edit of the default row (key preserved).
            admin = self._admin(portal_key)
            put = c.put(
                "/tenant-config/llm", json={"provider": "openai", "model": "gpt-4.1"}, headers=admin
            )
            assert put.status_code == 200, put.text
            # service: the delegating service with no end-user token.
            put = c.put(
                "/tenant-config/llm?tier=small",
                json={"provider": "local", "model": "local-llm", "apiKey": ""},
                headers=BASE,
            )
            assert put.status_code == 200, put.text
            # service: a verified user of ANOTHER tenant without super_admin is
            # neither the operator nor the customer's own admin.
            foreign = _mint(portal_key, roles=("tenant_admin",), tenant=PLATFORM_TENANT)
            put = c.put(
                "/tenant-config/llm?tier=small",
                json={"provider": "local", "model": "local-llm"},
                headers={**BASE, "X-End-User-Token": f"Bearer {foreign}"},
            )
            assert put.status_code == 200, put.text
        got = [(r["action"], r["actor_kind"], r["role"]) for r in db.audit_rows]
        assert got == [
            ("set_key", "platform_operator", "tier:medium"),
            ("test", "platform_operator", "tier:medium"),
            ("remove", "platform_operator", "tier:medium"),
            ("update", "workspace_admin", "__default__"),
            ("clear_key", "service", "tier:small"),
            ("update", "service", "tier:small"),
        ], got
        assert all(r["tenant_id"] == CUSTOMER_TENANT_A for r in db.audit_rows)

        resp = c.get("/tenant-config/llm/audit", headers=BASE)
        assert resp.status_code == 200, resp.text
        entries = resp.json()["entries"]
        assert [(e["action"], e["actor"], e["tier"]) for e in entries] == [
            ("update", "service", "small"),
            ("clear_key", "service", "small"),
            ("update", "workspace_admin", None),
            ("remove", "platform_operator", "medium"),
            ("test", "platform_operator", "medium"),
            ("set_key", "platform_operator", "medium"),
        ]
        assert set(entries[0]) == {"at", "tier", "action", "actor", "provider", "keyId"}
        assert entries[1]["keyId"] == "none"
        assert entries[5]["keyId"] != "none" and len(entries[5]["keyId"]) == 12
        # set / test / remove of one key all name it the same way.
        assert entries[3]["keyId"] == entries[4]["keyId"] == entries[5]["keyId"]

    def test_no_key_and_no_sub_in_the_rows_or_the_response(self, client, portal_key, portal_jwks):
        c, db = client
        owner = self._owner(portal_key)
        put = c.put(
            "/tenant-config/llm",
            json={"provider": "openai", "model": "gpt-4o", "apiKey": TENANT_KEY},
            headers=owner,
        )
        assert put.status_code == 200, put.text
        resp = c.get("/tenant-config/llm/audit", headers=owner)
        assert resp.status_code == 200, resp.text
        plain_hash = hashlib.sha256(TENANT_KEY.encode("utf-8")).hexdigest()
        stored = json.dumps(db.audit_rows, default=str)
        for blob in (stored, resp.text):
            assert TENANT_KEY not in blob
            assert plain_hash[:12] not in blob
            assert OWNER_SUB not in blob
            assert "sub" not in json.loads(resp.text)["entries"][0]

    def test_audit_read_is_scoped_to_the_effective_tenant(self, client, portal_key, portal_jwks):
        c, db = client
        db._append(
            tenant_id=CUSTOMER_TENANT_B,
            role="__default__",
            action="set_key",
            actor_kind="platform_operator",
            provider="b-secret-provider",
            key_id="none",
        )
        db._append(
            tenant_id=CUSTOMER_TENANT_A,
            role="__default__",
            action="test",
            actor_kind="service",
            provider="openai",
            key_id="none",
        )
        resp = c.get("/tenant-config/llm/audit", headers=BASE)  # effective = A
        assert resp.status_code == 200, resp.text
        assert [e["provider"] for e in resp.json()["entries"]] == ["openai"]
        assert "b-secret-provider" not in resp.text

    def test_limit_is_capped_at_200(self, client, portal_key, portal_jwks, monkeypatch):
        c, db = client
        seen: list[int] = []
        real = db.list_audit

        async def spy(tenant_id: str, limit: int = 50):
            seen.append(limit)
            return await real(tenant_id, limit)

        monkeypatch.setattr(db, "list_audit", spy)
        assert c.get("/tenant-config/llm/audit?limit=1000", headers=BASE).status_code == 200
        assert c.get("/tenant-config/llm/audit", headers=BASE).status_code == 200
        assert seen == [200, 50]
        assert c.get("/tenant-config/llm/audit?limit=0", headers=BASE).status_code == 422

    def test_a_failed_test_audit_refuses_the_test(self, client, portal_key, portal_jwks):
        c, db = client

        async def broken(**_kw: Any) -> None:
            raise RuntimeError("audit table unavailable")

        db.record_audit = broken  # type: ignore[method-assign]
        fake_get, calls = _vendor_ok()
        with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
            resp = c.post("/tenant-config/llm/test", json={}, headers=self._owner(portal_key))
        assert resp.status_code == 503, resp.text
        assert calls == []  # never reached the vendor unrecorded

    @pytest.mark.parametrize(
        "token",
        [
            None,  # delegated, no end-user proof: the key is withheld
            "tenant_admin_of_another_tenant",
        ],
    )
    def test_a_refused_test_is_recorded_as_test_refused(
        self, client, portal_key, portal_jwks, token
    ):
        """WM-4. A test refused BEFORE any live call (key withheld) is
        ``test_refused`` in the customer's history, never ``test``.
        MUTATION: write the row before the verdict (pre-WM-4) ⇒ ``test`` ⇒ red."""
        c, db = client
        headers = dict(BASE)
        if token is not None:
            minted = _mint(portal_key, roles=("tenant_admin",), tenant=PLATFORM_TENANT)
            headers["X-End-User-Token"] = f"Bearer {minted}"
        body, calls = _test(c, headers)
        assert body["ok"] is False
        assert calls == []
        assert [r["action"] for r in db.audit_rows] == ["test_refused"]
        resp = c.get("/tenant-config/llm/audit", headers=BASE)
        assert resp.status_code == 200, resp.text
        assert [e["action"] for e in resp.json()["entries"]] == ["test_refused"]

    def test_an_allowed_test_is_still_recorded_as_test(self, client, portal_key, portal_jwks):
        """Positive control for WM-4."""
        c, db = client
        body, calls = _test(c, self._owner(portal_key))
        assert body["ok"] is True
        assert len(calls) == 1
        assert [r["action"] for r in db.audit_rows] == ["test"]


class TestAuditRefusesTokenless:
    """``GET /tenant-config/llm/audit`` is NOT on the tokenless surface (#233 class).

    Measured live after WM #320 (7fafcfd3): a trusted-CIDR caller with a bare
    ``X-Tenant-Id: <any tenant>`` (provenance ``asserted``) read that tenant's
    audit, because ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` is OFF.
    Its only intended caller is fr, DELEGATED. Through the REAL middleware.

    MUTATIONS: re-admit ``asserted`` in ``_require_licence_or_delegated_caller``
    ⇒ the tokenless case goes red (200 + rows). Re-wire
    ``tokenless_tenant_gate`` on the route ⇒ the metric case goes red (counted).
    Drop the delegated branch ⇒ the delegated case goes red (401).
    """

    TRUSTED_PEER: ClassVar[tuple[str, int]] = ("10.42.1.7", 54321)

    def _seed(self, db: _DB) -> None:
        db._append(
            tenant_id=CUSTOMER_TENANT_A,
            role="__default__",
            action="set_key",
            actor_kind="platform_operator",
            provider="a-secret-provider",
            key_id="none",
        )

    def test_tokenless_asserted_caller_is_refused_uncounted_and_reads_nothing(self, client):
        import src.auth.tokenless_gate as tokenless_gate_module
        from src.settings import get_settings

        _, db = client
        self._seed(db)
        assert get_settings().require_authenticated_tenant is False  # the live posture
        listed: list[str] = []
        real = db.list_audit

        async def spy(tenant_id: str, limit: int = 50):
            listed.append(tenant_id)
            return await real(tenant_id, limit)

        db.list_audit = spy  # type: ignore[method-assign]
        metrics = MagicMock()
        with (
            patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics),
            TestClient(_app(), client=self.TRUSTED_PEER, raise_server_exceptions=False) as c,
        ):
            resp = c.get("/tenant-config/llm/audit", headers={"X-Tenant-Id": CUSTOMER_TENANT_A})
        assert resp.status_code == 401, resp.text
        assert "a-secret-provider" not in resp.text
        assert "entries" not in resp.json()
        assert listed == []  # the handler never ran; no row was read
        # Not tokenless surface: worker_management_tokenless_tenant_request_total
        # must not move for this route.
        metrics.record_tokenless_tenant_request.assert_not_called()

    def test_tokenless_default_tenant_caller_is_refused(self, client):
        _, db = client
        self._seed(db)
        with TestClient(_app(), client=self.TRUSTED_PEER, raise_server_exceptions=False) as c:
            resp = c.get("/tenant-config/llm/audit")
        assert resp.status_code == 401, resp.text
        assert "a-secret-provider" not in resp.text

    def test_licence_caller_reads_its_own_tenant(self, client):
        c, db = client
        self._seed(db)
        own = {
            "active": True,
            "tenant_id": CUSTOMER_TENANT_A,
            "subject_id": "customer-a-licence-subject",
            "byo_mode": "local_allowed",
        }
        with patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=own,
        ):
            resp = c.get("/tenant-config/llm/audit", headers={"X-Tenant-Id": "Bearer a-token"})
        assert resp.status_code == 200, resp.text
        assert [e["provider"] for e in resp.json()["entries"]] == ["a-secret-provider"]

    def test_delegated_fr_caller_reads_the_effective_tenant(self, client):
        c, db = client
        self._seed(db)
        resp = c.get("/tenant-config/llm/audit", headers=BASE)  # svc-fr-backend → A
        assert resp.status_code == 200, resp.text
        assert [e["provider"] for e in resp.json()["entries"]] == ["a-secret-provider"]

    @pytest.mark.parametrize("provenance", ["asserted", "default", "bypass", "unknown"])
    def test_every_unproven_provenance_is_refused(self, provenance: str):
        import asyncio
        import types

        from fastapi import HTTPException

        from src.api.tenant_config._router import _require_licence_or_delegated_caller

        set_identity_context(CUSTOMER_TENANT_A, None, None, tenant_provenance=provenance)
        req = types.SimpleNamespace(state=types.SimpleNamespace())
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_require_licence_or_delegated_caller(req))  # type: ignore[arg-type]
        assert exc.value.status_code == 401

    def test_delegated_from_an_unlisted_subject_is_refused(self):
        import asyncio
        import types

        from fastapi import HTTPException

        from src.api.tenant_config._router import _require_licence_or_delegated_caller

        set_identity_context(
            CUSTOMER_TENANT_A, None, None, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )
        state = types.SimpleNamespace(identity_proven=True, license={"subject_id": "not-listed"})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                _require_licence_or_delegated_caller(types.SimpleNamespace(state=state))  # type: ignore[arg-type]
            )
        assert exc.value.status_code == 401
        ok = types.SimpleNamespace(identity_proven=True, license={"subject_id": SA_SUBJECT})
        asyncio.run(_require_licence_or_delegated_caller(types.SimpleNamespace(state=ok)))  # type: ignore[arg-type]
