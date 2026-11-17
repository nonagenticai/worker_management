"""Owner rulings 2026-10-01 (fix round 2) + validator defects B1, B2, N1, N4-N6.

Each class names the item it pins and the mutation that turns it red.

- #3  end-user JWT proof for BYO-key decrypt on DELEGATED calls (X-End-User-Token)
- #4  a withheld/unusable tenant key or a failed vendor call is a CLEAR error,
      never a silent fallback to platform / local-llm routing
- #6  POST /tenant-config/llm/test
- B1  a customer graph may not read an owner env key (inference_provider_node)
- B2  customer worker path: stored ir_provider dropped, non-local-llm IR model coerced
- N1  worker-path tenant_byo_key wiring (_globals → create_worker → agent kwargs)
- N4  no env fallback on an empty BYO key in the model-list fetchers
- N5  IR model cache: no global lock across the fetch; a failure is not cached for the TTL
- N6  local direct providers on the customer worker path
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
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
    set_end_user_proof,
    set_identity_context,
)
from src.auth import end_user_token, keycloak_roles
from src.security.secret_box import encrypt_secret
from src.worker.factory import llm_config_resolver as r
from tests._platform_tenant import (
    CUSTOMER_TENANT_A,
    CUSTOMER_TENANT_B,
    PLATFORM_TENANT,
    platform_tenant_context,
    proven_tenant,
)
from tests._tenant_llm_audit_fake import AuditFakeMixin

SA_SUBJECT = "019ffe9f-0000-7000-8000-00000000f00d"  # a delegating service account
SA_TENANT = "22222222-2222-4222-8222-222222222222"
ISSUER = "https://auth.nonagentic.ai/realms/portal"
KID = "portal-key-1"
TENANT_KEY = "sk-tenant-a-own-openai-key"  # pragma: allowlist secret


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
    jwks = {"keys": [pub]}
    fetched: list[str] = []

    async def fake_fetch(url: str) -> dict:
        fetched.append(url)
        return jwks

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", fake_fetch)
    return fetched


def _mint(
    key,
    *,
    tenant: str | None = CUSTOMER_TENANT_A,
    iss: str = ISSUER,
    exp: int = 600,
    typ: str | None = "Bearer",
    azp: str | None = "fr-frontend",
):
    claims: dict[str, Any] = {"sub": "end-user-1", "iss": iss, "exp": int(time.time()) + exp}
    if tenant is not None:
        claims["tenant_id"] = tenant
    # The shape of a real Keycloak ACCESS token fr forwards (WM-3).
    if typ is not None:
        claims["typ"] = typ
    if azp is not None:
        claims["azp"] = azp
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def _run(coro):
    return asyncio.run(coro)


# =============================================================== #3 verification


class TestEndUserTokenVerification:
    """MUTATION: skip the signature / issuer / tenant check ⇒ a case below goes red."""

    def test_valid_portal_token_for_the_effective_tenant_proves_it(self, portal_key, portal_jwks):
        tenant, reason = _run(
            end_user_token.verify_end_user_tenant(f"Bearer {_mint(portal_key)}", CUSTOMER_TENANT_A)
        )
        assert (tenant, reason) == (CUSTOMER_TENANT_A, "ok")
        # The portal JWKS is derived from the configured secure-apps one.
        assert portal_jwks == [
            "http://keycloak.keycloak.svc.cluster.local:8080/realms/portal/protocol/openid-connect/certs"
        ]

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"exp": -60}, "invalid:ExpiredSignatureError"),
            ({"iss": "https://evil.example/realms/portal"}, "invalid:InvalidIssuerError"),
            (
                {"iss": "https://auth.nonagentic.ai/realms/secure-apps"},
                "invalid:InvalidIssuerError",
            ),
            ({"tenant": CUSTOMER_TENANT_B}, "tenant_mismatch"),
            ({"tenant": None}, "no_tenant_claim"),
            # WM-3: an ID token (Keycloak ``typ`` = ``ID``), a refresh token and a
            # token without ``typ`` are not access tokens.
            ({"typ": "ID"}, "not_access_token"),
            ({"typ": "Refresh"}, "not_access_token"),
            ({"typ": None}, "not_access_token"),
            # WM-3: an access token minted for a client outside the allowlist
            # (another portal-realm client, or none) cannot prove a tenant.
            ({"azp": "w2v-frontend"}, "client_not_allowed"),
            ({"azp": "account-console"}, "client_not_allowed"),
            ({"azp": None}, "client_not_allowed"),
        ],
    )
    def test_rejections(self, portal_key, portal_jwks, kwargs, reason):
        tenant, got = _run(
            end_user_token.verify_end_user_tenant(_mint(portal_key, **kwargs), CUSTOMER_TENANT_A)
        )
        assert (tenant, got) == ("", reason)

    @pytest.mark.parametrize("azp", ["fr-frontend", "kag-frontend", "portal-bff"])
    def test_default_allowlist_accepts_the_live_user_facing_clients(
        self, portal_key, portal_jwks, azp
    ):
        tenant, reason = _run(
            end_user_token.verify_end_user_tenant(
                f"Bearer {_mint(portal_key, azp=azp)}", CUSTOMER_TENANT_A
            )
        )
        assert (tenant, reason) == (CUSTOMER_TENANT_A, "ok")

    def test_allowlist_is_read_from_settings_and_empty_is_deny_all(
        self, portal_key, portal_jwks, monkeypatch
    ):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("END_USER_TOKEN_ALLOWED_CLIENTS", "custom-spa")
        reset_settings_cache()
        assert _run(
            end_user_token.verify_end_user_tenant(
                _mint(portal_key, azp="custom-spa"), CUSTOMER_TENANT_A
            )
        ) == (CUSTOMER_TENANT_A, "ok")
        assert _run(
            end_user_token.verify_end_user_tenant(_mint(portal_key), CUSTOMER_TENANT_A)
        ) == ("", "client_not_allowed")
        monkeypatch.setenv("END_USER_TOKEN_ALLOWED_CLIENTS", "")
        reset_settings_cache()
        assert _run(
            end_user_token.verify_end_user_tenant(_mint(portal_key), CUSTOMER_TENANT_A)
        ) == ("", "client_not_allowed")

    @pytest.mark.parametrize(("typ", "proven"), [("ID", False), ("Bearer", True)])
    def test_id_token_records_no_owner_identity(self, portal_key, portal_jwks, typ, proven):
        """An ID token must not even yield the verified-identity facts the owner
        key-test path reads (``sub`` / ``super_admin``), not just fail the tenant
        proof. The ``Bearer`` row is the positive control."""
        from src.api.identity_context import get_end_user_identity

        token = _mint(portal_key, typ=typ)

        async def go():
            set_identity_context(
                CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
            )
            req = SimpleNamespace(
                headers={end_user_token.END_USER_TOKEN_HEADER: f"Bearer {token}"},
                state=SimpleNamespace(),
            )
            await end_user_token.stamp_end_user_proof(req, CUSTOMER_TENANT_A, SA_SUBJECT)
            return req.state.end_user_tenant_proven, get_end_user_identity()[0]

        got_proven, sub = _run(go())
        assert got_proven is proven
        assert sub == ("end-user-1" if proven else "")

    def test_token_signed_by_another_key_is_rejected(self, portal_jwks):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        tenant, reason = _run(
            end_user_token.verify_end_user_tenant(_mint(other), CUSTOMER_TENANT_A)
        )
        assert tenant == ""
        assert reason == "invalid:InvalidSignatureError"

    def test_absent_and_garbage(self, portal_jwks):
        assert _run(end_user_token.verify_end_user_tenant(None, CUSTOMER_TENANT_A)) == (
            "",
            "absent",
        )
        tenant, reason = _run(
            end_user_token.verify_end_user_tenant("Bearer x.y", CUSTOMER_TENANT_A)
        )
        assert tenant == "" and reason.startswith("invalid:")


# =============================================================== #3 entitlement


class TestDelegatedEntitlement:
    """tenant_key_entitled: licence as before; delegated ONLY with an end-user proof
    of the same tenant from a still-configured delegator.

    MUTATION: drop the delegated arm ⇒ the proven case goes red. MUTATION: skip
    the delegator re-check ⇒ test_delegator_removed... goes red."""

    def _delegated(self, tenant: str = CUSTOMER_TENANT_A) -> None:
        set_identity_context(
            tenant, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )

    def test_delegated_without_proof_is_not_entitled(self):
        self._delegated()
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is False
        assert "X-End-User-Token" in r.key_withheld_reason(CUSTOMER_TENANT_A)

    def test_delegated_with_proof_is_entitled_for_that_tenant_only(self):
        self._delegated()
        set_end_user_proof(tenant_id=CUSTOMER_TENANT_A, delegator=SA_SUBJECT, reason="ok")
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is True
        assert r.tenant_key_entitled(CUSTOMER_TENANT_B) is False

    def test_delegator_removed_from_config_loses_entitlement(self, monkeypatch):
        self._delegated()
        set_end_user_proof(tenant_id=CUSTOMER_TENANT_A, delegator=SA_SUBJECT, reason="ok")
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "someone-else")
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is False

    def test_a_new_identity_clears_a_previous_proof(self):
        self._delegated()
        set_end_user_proof(tenant_id=CUSTOMER_TENANT_A, delegator=SA_SUBJECT, reason="ok")
        self._delegated()  # next request on the same context
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is False

    def test_licence_provenance_unchanged(self):
        set_identity_context(
            CUSTOMER_TENANT_A, "lic", "sub", tenant_provenance=TENANT_PROVENANCE_LICENSE
        )
        assert r.tenant_key_entitled(CUSTOMER_TENANT_A) is True
        assert r.tenant_key_entitled(CUSTOMER_TENANT_B) is False


def _middleware_app() -> FastAPI:
    from src.api.auth_middleware import AuthenticationMiddleware

    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @app.get("/api/workers")
    async def probe():
        from src.api.workers._globals import _cache_key

        entitled = r.tenant_key_entitled(CUSTOMER_TENANT_A)
        return {
            "entitled": entitled,
            "cache_key": _cache_key("w", CUSTOMER_TENANT_A, key_entitled=entitled),
        }

    return app


def _drive(headers: dict[str, str], expect: int = 200) -> dict[str, Any]:
    client = TestClient(_middleware_app(), raise_server_exceptions=False)
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
        resp = client.get("/api/workers", headers=headers)
    assert resp.status_code == expect, resp.text
    return resp.json()


class TestDelegatedProofThroughTheRealMiddleware:
    """The REAL AuthenticationMiddleware (only licence introspection + the JWKS
    network fetch are stubbed). MUTATION: remove the middleware's stamp call ⇒
    the proven case goes red; the cache key's ``:byo-key`` suffix follows."""

    base: ClassVar[dict[str, str]] = {
        "X-Tenant-Id": "Bearer sa-token",
        "X-Effective-Tenant-Id": CUSTOMER_TENANT_A,
    }

    def test_forwarded_end_user_token_entitles_and_keys_the_cache(self, portal_key, portal_jwks):
        out = _drive({**self.base, "X-End-User-Token": f"Bearer {_mint(portal_key)}"})
        assert out["entitled"] is True
        assert out["cache_key"].endswith(":byo-key")

    def test_no_end_user_token_is_not_entitled(self, portal_jwks):
        out = _drive(self.base)
        assert out["entitled"] is False
        assert not out["cache_key"].endswith(":byo-key")

    def test_token_for_another_tenant_is_not_entitled(self, portal_key, portal_jwks):
        token = _mint(portal_key, tenant=CUSTOMER_TENANT_B)
        out = _drive({**self.base, "X-End-User-Token": token})
        assert out["entitled"] is False

    def test_non_delegating_caller_cannot_use_the_header(
        self, portal_key, portal_jwks, monkeypatch
    ):
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "")
        # Since the licence-branch end-user check (George 2026-10-15) a
        # non-delegating caller's X-End-User-Token is verified against the
        # LICENCE tenant (SA_TENANT); this user is CUSTOMER_TENANT_A's, so the
        # request is refused outright rather than served unentitled.
        out = _drive({**self.base, "X-End-User-Token": _mint(portal_key)}, expect=403)
        assert out["error"]["code"] == "END_USER_TENANT_MISMATCH"


# =============================================================== #4 no silent fallback


class _RowsDB(AuditFakeMixin):
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows

    async def list_configs(self, tenant_id: str) -> list[dict[str, Any]]:
        return [dict(x) for x in self.rows.get(tenant_id, [])]

    async def get_config(self, tenant_id: str, role: str = "__default__") -> dict[str, Any] | None:
        for x in self.rows.get(tenant_id, []):
            if x["role"] == role:
                return dict(x)
        return None


def _row(provider: str = "openai", model: str = "gpt-4o", key: str | None = TENANT_KEY, **kw):
    base = {
        "tenant_id": CUSTOMER_TENANT_A,
        "role": "tier:medium",
        "provider": provider,
        "model": model,
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "params": {},
        "api_key_encrypted": encrypt_secret(key) if key else None,
        "cost_source": "worker-management",
        "owner_provided": False,
    }
    base.update(kw)
    return base


class TestNoSilentFallback:
    """MUTATION: restore the WARN-and-fall-through ⇒ resolve returns a platform
    config instead of raising ⇒ red."""

    @pytest.mark.asyncio
    async def test_withheld_tier_key_is_403_naming_provider_tier_and_fix(self, monkeypatch):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row()]}))
        set_identity_context(
            CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )
        cfg = {"model_provider": "inference_router", "model": "local-llm"}
        with pytest.raises(r.TenantLLMKeyUnusableError) as exc_info:
            await r.resolve_llm_config(cfg, CUSTOMER_TENANT_A, work_tier="medium")
        err = exc_info.value
        assert (err.status_code, err.error_code) == (403, "BYO_KEY_WITHHELD")
        assert "openai" in err.message and "'medium'" in err.message
        assert "X-End-User-Token" in err.message
        assert TENANT_KEY not in err.message
        assert err.details()["provider"] == "openai"
        assert cfg == {"model_provider": "inference_router", "model": "local-llm"}

    @pytest.mark.asyncio
    async def test_delegated_with_end_user_proof_gets_the_tier_key(self, monkeypatch):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row()]}))
        set_identity_context(
            CUSTOMER_TENANT_A, None, SA_SUBJECT, tenant_provenance=TENANT_PROVENANCE_DELEGATED
        )
        set_end_user_proof(tenant_id=CUSTOMER_TENANT_A, delegator=SA_SUBJECT, reason="ok")
        cfg = await r.resolve_llm_config(
            {"model_provider": "inference_router", "model": "local-llm"},
            CUSTOMER_TENANT_A,
            work_tier="medium",
        )
        assert cfg["model_provider"] == "openai"
        assert cfg["api_key_override"] == TENANT_KEY
        assert cfg["tenant_byo_key"] is True

    @pytest.mark.asyncio
    async def test_undecryptable_key_is_409(self, monkeypatch):
        bad = _row()
        bad["api_key_encrypted"] = "not-a-valid-blob"
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [bad]}))
        with pytest.raises(r.TenantLLMKeyUnusableError) as exc_info:
            await r.resolve_llm_config(
                {"model_provider": "inference_router", "model": "local-llm"},
                CUSTOMER_TENANT_A,
                work_tier="medium",
                key_entitled=True,
            )
        assert (exc_info.value.status_code, exc_info.value.reason) == (409, "decrypt_failed")
        assert "PUT /tenant-config/llm?tier=medium" in exc_info.value.message


@asynccontextmanager
async def _noop_ctx(*_a: Any, **_k: Any):
    yield


@pytest.fixture
def message_client():
    """POST /workers/{name}/message with persistence mocked (the IR-passthrough harness)."""
    from src.api.worker_management_api import app
    from src.auth import get_tenant_id

    msg = "src.api.workers.message_endpoints"
    holder: dict[str, Any] = {}
    with (
        patch(f"{msg}.worker_registry") as registry,
        patch(f"{msg}.worker_cache") as cache,
        patch(f"{msg}.state_manager") as state_mgr,
        patch(f"{msg}.StateDatabase") as state_db_cls,
        patch(f"{msg}.mlx_model_registry") as mlx,
    ):
        state_mgr.atomic_worker_operation = _noop_ctx
        state_mgr.resolve_worker_id = AsyncMock(return_value=("wid-1", True))
        state_mgr.validate_state_transition = MagicMock(return_value=True)
        state_db = AsyncMock()
        state_db.get_latest_worker_state = AsyncMock(return_value=None)
        state_db.add_worker_state = AsyncMock(return_value=1)
        state_db_cls.return_value = state_db

        async def _get(*_a: Any, **_k: Any) -> Any:
            if isinstance(holder.get("raise"), BaseException):
                raise holder["raise"]
            return holder["agent"]

        cache.get_or_create_worker = AsyncMock(side_effect=_get)
        cache.remove_worker = AsyncMock()
        cache.remove_instance = AsyncMock()
        registry.get_worker = AsyncMock(
            return_value={
                "worker_name": "byo_probe",
                "worker_id": "wid-1",
                "db_path": "/tmp/byo.db",
                "is_agent": True,
                "config": {"agent_role": "r", "agent_type": "pydantic_ai"},
                "status": "active",
            }
        )
        mlx.get_model = AsyncMock(return_value=None)
        mlx.get_instance_for_model = AsyncMock(return_value=None)
        app.dependency_overrides[get_tenant_id] = lambda: CUSTOMER_TENANT_A

        @asynccontextmanager
        async def _fake_lifespan(_a: Any):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _fake_lifespan
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, holder, state_db
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original


class TestWorkerPathHttpContract:
    def test_withheld_key_is_403_with_code_details_and_an_error_row(self, message_client):
        """MUTATION: drop the TenantLLMKeyUnusableError → _RefusedAfterProcessing
        conversion ⇒ a generic 500 and/or no error row ⇒ red."""
        client, holder, state_db = message_client
        holder["raise"] = r.TenantLLMKeyUnusableError(
            tenant_id=CUSTOMER_TENANT_A,
            provider="openai",
            tier="medium",
            role="tier:medium",
            reason="withheld",
            withheld_detail="X-End-User-Token: absent",
        )
        resp = client.post("/workers/byo_probe/message", json={"message": "hi"})
        assert resp.status_code == 403, resp.text
        body = resp.json()["error"]
        assert body["code"] == "BYO_KEY_WITHHELD"
        assert "openai" in body["message"]
        assert body["details"]["provider"] == "openai"
        assert body["details"]["tier"] == "medium"
        states = [c.kwargs.get("state_type") for c in state_db.add_worker_state.await_args_list]
        assert states[-1] == "error"

    def test_failed_vendor_call_on_the_tenant_key_names_provider_and_fix(self, message_client):
        """MUTATION: drop byo_failure_detail ⇒ the bare upstream text ⇒ red."""
        client, holder, _ = message_client

        class _Agent:
            config = SimpleNamespace(tenant_byo_key=True, model_provider="openai", model="gpt-4o")

            async def run(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("Error code: 401 - invalid_api_key")

        holder["agent"] = _Agent()
        resp = client.post("/workers/byo_probe/message", json={"message": "hi"})
        assert resp.status_code >= 400
        text = resp.text
        assert "your own openai key" in text
        assert "POST /tenant-config/llm/test" in text
        assert "Nothing was run on a platform model" in text

    def test_vendor_rejection_on_tenant_key_is_typed_502_with_scrubbed_detail(self, message_client):
        """WM-1. MUTATION: drop the byo_failure_http branch ⇒ 500 INTERNAL_ERROR
        with the raw vendor text (key included) ⇒ red."""
        client, holder, state_db = message_client
        key = "AIzaSy-tenant-own-google-key-123"  # pragma: allowlist secret

        class _Agent:
            config = SimpleNamespace(
                tenant_byo_key=True, model_provider="google", model="gemini-2.5-pro", api_key=key
            )

            async def run(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError(
                    "Client error '400 Bad Request' for url "
                    f"'https://generativelanguage.googleapis.com/v1beta/models?key={key}': "
                    "API key not valid" + " pad" * 400
                )

        holder["agent"] = _Agent()
        resp = client.post("/workers/byo_probe/message", json={"message": "hi"})
        assert resp.status_code == 502, resp.text
        err = resp.json()["error"]
        assert err["code"] == "BYO_PROVIDER_CALL_FAILED"
        assert isinstance(err["message"], str)
        assert "your own google key" in err["message"]
        assert "API key not valid" in err["message"]
        assert "(truncated)" in err["message"]
        assert key not in resp.text
        assert err["details"]["provider"] == "google"
        assert err["details"]["model"] == "gemini-2.5-pro"
        # The terminal error row is scrubbed as well.
        last = state_db.add_worker_state.await_args_list[-1].kwargs
        assert last["state_type"] == "error"
        assert key not in json.dumps(last["worker_state"])

    def test_platform_run_failure_keeps_internal_error(self, message_client):
        """Positive control for WM-1: a non-BYO run's unclassified failure is
        still 500 INTERNAL_ERROR with its text unchanged."""
        client, holder, _ = message_client

        class _Agent:
            config = SimpleNamespace(tenant_byo_key=False, model_provider="openai", model="m")

            async def run(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("boom-internal")

        holder["agent"] = _Agent()
        resp = client.post("/workers/byo_probe/message", json={"message": "hi"})
        assert resp.status_code == 500, resp.text
        assert resp.json()["error"]["code"] == "INTERNAL_ERROR"
        assert "boom-internal" in resp.json()["error"]["message"]

    def test_non_byo_failure_detail_unchanged(self):
        from src.api.workers.message_endpoints import byo_failure_detail

        agent = SimpleNamespace(config=SimpleNamespace(tenant_byo_key=False))
        assert byo_failure_detail(agent, CUSTOMER_TENANT_A, "boom") == "boom"


# =============================================================== chat (#4 on /chat)


@pytest.fixture
def chat_client(monkeypatch):
    from src.api.worker_management_api import app

    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c


@pytest.fixture(autouse=True)
def _fresh_live_cache():
    from src.api.providers import _router
    from src.api.tenant_config import _router as tc

    _router._byo_live_ok.clear()
    _router._byo_live_fail.clear()
    tc._key_test_cooldown.clear()
    yield
    _router._byo_live_ok.clear()
    _router._byo_live_fail.clear()
    tc._key_test_cooldown.clear()


def _as_customer(entitled: bool):
    return (
        patch("src.api.providers._router.caller_tenant_id", return_value=CUSTOMER_TENANT_A),
        patch.object(r, "tenant_key_entitled", lambda t: entitled and t == CUSTOMER_TENANT_A),
    )


class TestChatErrors:
    def _post(self, client):
        return client.post(
            "/chat/completions",
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    def test_withheld_key_on_chat_is_403_withheld_not_add_your_key(self, chat_client, monkeypatch):
        """MUTATION: chat gate reads only the usable map ⇒ 'add your own key' ⇒ red."""
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]}))
        p1, p2 = _as_customer(entitled=False)
        with p1, p2:
            resp = self._post(chat_client)
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "BYO_KEY_WITHHELD"

    def test_byo_failure_carrying_ir_busy_meta_is_never_masked_as_503(
        self, chat_client, monkeypatch
    ):
        """MUTATION: drop `and not byo` from the busy check => a BYO vendor failure that happens
        to carry IR's queue_rejected classification is reported as 503 busy => red."""
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]}))
        p1, p2 = _as_customer(entitled=True)
        upstream = RuntimeError("Error code: 503 - queue full")
        upstream.response = SimpleNamespace(  # type: ignore[attr-defined]
            headers={"X-IR-Error-Classification": "queue_rejected", "Retry-After": "7"}
        )
        outer = RuntimeError("status_code: 503, body: queue full")
        outer.__cause__ = upstream
        agent = MagicMock()
        agent.run_stream = MagicMock(side_effect=outer)
        with (
            p1,
            p2,
            patch(
                "src.api.providers._chat_router._build_agent",
                new_callable=AsyncMock,
                return_value=agent,
            ),
        ):
            resp = self._post(chat_client)
        assert resp.status_code == 502, resp.text
        assert resp.json()["error"]["code"] == "BYO_PROVIDER_CALL_FAILED"
        assert "X-IR-Error-Classification" not in resp.headers

    def test_failed_vendor_call_on_chat_is_502_byo_call_failed(self, chat_client, monkeypatch):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]}))
        p1, p2 = _as_customer(entitled=True)

        async def boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError(f"401 invalid key {TENANT_KEY}")

        agent = MagicMock()
        agent.run_stream = MagicMock(side_effect=RuntimeError(f"401 invalid key {TENANT_KEY}"))
        with (
            p1,
            p2,
            patch(
                "src.api.providers._chat_router._build_agent",
                new_callable=AsyncMock,
                return_value=agent,
            ),
        ):
            resp = self._post(chat_client)
        assert resp.status_code == 502, resp.text
        err = resp.json()["error"]
        assert err["code"] == "BYO_PROVIDER_CALL_FAILED"
        assert "openai" in err["message"]
        assert TENANT_KEY not in resp.text  # scrubbed


# =============================================================== #6 key test


class TestKeyTestEndpoint:
    """POST /tenant-config/llm/test. MUTATION: return ok without the live call /
    ignore a vendor rejection ⇒ the bad-key case goes red."""

    @pytest.fixture
    def tc_client(self, monkeypatch):
        from src.api.tenant_config import _router as tc
        from src.api.worker_management_api import app
        from src.auth import TenantContext, get_tenant_context

        db = _RowsDB(
            {
                CUSTOMER_TENANT_A: [
                    _row(role="__default__"),
                    _row(provider="local", model="local-llm", key=None, role="tier:small"),
                ]
            }
        )
        monkeypatch.setattr(tc, "_db", db)
        monkeypatch.setattr(r, "_db", db)
        state = {"entitled": True}
        monkeypatch.setattr(tc, "tenant_key_entitled", lambda _t: state["entitled"])
        app.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=CUSTOMER_TENANT_A,
            subject_id=CUSTOMER_TENANT_A,
            tenant_name="a",
            visibility="tenant",
            byo_mode="local_allowed",
        )
        with (
            patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
            patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            yield c, state
        app.dependency_overrides.clear()

    def _vendor(self, ok: bool):
        async def fake_get(url: str, *, headers=None, timeout=10.0):
            if ok and (headers or {}).get("Authorization") == f"Bearer {TENANT_KEY}":
                return {"data": [{"id": "gpt-4o"}, {"id": "gpt-5"}]}
            raise httpx.HTTPStatusError(
                "401 Unauthorized",
                request=httpx.Request("GET", url),
                response=httpx.Response(401),
            )

        return fake_get

    def test_good_key_is_ok_with_latency_and_no_key_in_body(self, tc_client):
        client, _ = tc_client
        with patch("src.api.providers._router._httpx_json_get", side_effect=self._vendor(True)):
            resp = client.post("/tenant-config/llm/test", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["provider"] == "openai" and body["model"] == "gpt-4o"
        assert isinstance(body["latency_ms"], int)
        assert "error" not in body
        assert TENANT_KEY not in resp.text

    def test_bad_key_goes_red(self, tc_client):
        client, _ = tc_client
        with patch("src.api.providers._router._httpx_json_get", side_effect=self._vendor(False)):
            resp = client.post("/tenant-config/llm/test", json={"provider": "openai"})
        body = resp.json()
        assert body["ok"] is False
        assert "401" in body["error"]

    def test_withheld_key_says_how_to_prove_the_tenant(self, tc_client):
        client, state = tc_client
        state["entitled"] = False
        with patch("src.api.providers._router._httpx_json_get", new_callable=AsyncMock) as vendor:
            resp = client.post("/tenant-config/llm/test", json={"tier": "medium"})
        body = resp.json()
        assert body["ok"] is False
        assert "X-End-User-Token" in body["error"]
        vendor.assert_not_called()

    def test_model_not_listed_is_a_warning_not_a_red(self, tc_client):
        client, _ = tc_client
        with patch("src.api.providers._router._httpx_json_get", side_effect=self._vendor(True)):
            resp = client.post(
                "/tenant-config/llm/test", json={"provider": "openai", "model": "gpt-9"}
            )
        body = resp.json()
        assert body["ok"] is True
        assert "gpt-9" in body["warning"]

    def test_local_row_is_tested_against_inference_router(self, tc_client):
        from src.services.inference_router_registry import InferenceRouterModel

        client, _ = tc_client
        with patch(
            "src.api.providers._router.inference_router_registry.refresh_model_cache",
            new_callable=AsyncMock,
            return_value=[InferenceRouterModel(id="local-llm")],
        ) as refresh:
            resp = client.post("/tenant-config/llm/test", json={"tier": "small"})
        body = resp.json()
        assert body == {**body, "ok": True, "provider": "inference_router", "model": "local-llm"}
        refresh.assert_awaited_once_with(CUSTOMER_TENANT_A)

    def test_no_row_is_not_ok(self, tc_client):
        client, _ = tc_client
        resp = client.post("/tenant-config/llm/test", json={"provider": "anthropic"})
        assert resp.json()["ok"] is False

    def test_tier_and_provider_together_is_422(self, tc_client):
        client, _ = tc_client
        resp = client.post("/tenant-config/llm/test", json={"tier": "small", "provider": "x"})
        assert resp.status_code == 422


# =============================================================== B1


class TestGraphNodeEnvKey:
    """MUTATION: restore ``os.environ.get(cfg["api_key_env"])`` ⇒ the customer
    case reads the owner's key ⇒ red."""

    cfg: ClassVar[dict[str, str]] = {"name": "p", "api_key_env": "XAI_API_KEY"}

    def test_customer_state_tenant_is_refused_and_env_never_read(self, monkeypatch):
        from src.worker.graphs.nodes import inference_provider_node as n

        monkeypatch.setenv("XAI_API_KEY", "xai-OWNER")
        reads: list[str] = []
        real_get = os.environ.get

        def spy(key: str, default: Any = None) -> Any:
            if key == "XAI_API_KEY":
                reads.append(key)
            return real_get(key, default)

        with patch.object(n.os.environ, "get", side_effect=spy):
            with pytest.raises(n.CustomerEnvKeyRefusedError) as exc_info:
                n._provider_env_key(self.cfg, CUSTOMER_TENANT_A)
        assert reads == []
        assert "xai-OWNER" not in str(exc_info.value)

    def test_unknown_tenant_is_refused(self, monkeypatch):
        from src.worker.graphs.nodes import inference_provider_node as n

        monkeypatch.setenv("XAI_API_KEY", "xai-OWNER")
        with pytest.raises(n.CustomerEnvKeyRefusedError):
            n._provider_env_key(self.cfg, None)

    def test_platform_state_with_a_customer_request_is_refused(self, monkeypatch):
        from src.worker.graphs.nodes import inference_provider_node as n

        monkeypatch.setenv("XAI_API_KEY", "xai-OWNER")
        set_identity_context(CUSTOMER_TENANT_A, None, None)
        with pytest.raises(n.CustomerEnvKeyRefusedError):
            n._provider_env_key(self.cfg, PLATFORM_TENANT)

    def test_platform_tenant_still_reads_its_key(self, monkeypatch):
        """A licence-PROVEN platform tenant (real predicate, real contextvars)."""
        from src.worker.graphs.nodes import inference_provider_node as n

        monkeypatch.setenv("XAI_API_KEY", "xai-OWNER")
        with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
            assert n._provider_env_key(self.cfg, PLATFORM_TENANT) == "xai-OWNER"


# =============================================================== B2 + N1 + N6 (worker build)


def _build(monkeypatch, tenant: str, config: dict[str, Any], *, resolved: dict | None = None):
    """Run WorkerInstanceCache._create_worker_instance; capture create_worker kwargs."""
    from src.api.workers import _globals

    captured: dict[str, Any] = {}

    async def fake_resolve(cfg, tenant_id, **_k):
        if resolved is not None:
            cfg.update(resolved)
        return cfg

    async def fake_create_worker(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace())

    monkeypatch.setattr(_globals, "resolve_llm_config", fake_resolve)
    monkeypatch.setattr(_globals, "create_worker", fake_create_worker)
    cache = _globals.WorkerInstanceCache()
    _run(
        cache._create_worker_instance(
            "w", {"worker_name": "w", "config": dict(config)}, tenant_id=tenant
        )
    )
    return captured


class TestCustomerWorkerRouting:
    """MUTATION: drop the enforce_customer_routing call in _globals ⇒ red."""

    def test_stored_ir_provider_pin_is_dropped_for_a_customer(self, monkeypatch):
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {
                "agent_role": "r",
                "model_provider": "inference_router",
                "model": "local-llm",
                "ir_provider": "xai",
            },
        )
        assert got["ir_provider"] is None

    @pytest.mark.parametrize("model", ["deepseek-v4-pro", "tier-medium", "ks-entity-spotter"])
    def test_non_local_llm_ir_model_is_coerced_for_a_customer(self, monkeypatch, model):
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {"agent_role": "r", "model_provider": "inference_router", "model": model},
        )
        assert (got["model_provider"], got["model"]) == ("inference_router", "local-llm")

    @pytest.mark.parametrize("provider", ["vllm", "mlx", "vlm_local", None])
    def test_local_direct_provider_moves_to_inference_router(self, monkeypatch, provider):
        """N6. MUTATION: drop the local-direct branch ⇒ red."""
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {
                "agent_role": "r",
                "model_provider": provider,
                "model": "qwen-x",
                "base_url_override": "http://mini1:8080",
            },
        )
        assert (got["model_provider"], got["model"]) == ("inference_router", "local-llm")
        assert got["base_url_override"] is None

    def test_platform_tenant_untouched(self, monkeypatch):
        with proven_tenant(PLATFORM_TENANT):  # fr#1385: only a PROVEN platform tenant
            got = _build(
                monkeypatch,
                PLATFORM_TENANT,
                {
                    "agent_role": "r",
                    "model_provider": "vllm",
                    "model": "deepseek-v4-pro",
                    "ir_provider": "deepseek",
                },
            )
        assert (got["model_provider"], got["model"], got["ir_provider"]) == (
            "vllm",
            "deepseek-v4-pro",
            "deepseek",
        )

    def test_cli_agent_types_are_not_rewritten(self, monkeypatch):
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {"agent_role": "r", "agent_type": "claude_code", "model": "claude-sonnet-4-6"},
        )
        assert got["model"] == "claude-sonnet-4-6"

    def test_byo_resolution_flag_reaches_create_worker(self, monkeypatch):
        """N1 (_globals). MUTATION: force tenant_byo_key=False in _globals ⇒ red."""
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {"agent_role": "r", "model_provider": "inference_router", "model": "local-llm"},
            resolved={
                "model_provider": "openai",
                "model": "gpt-4o",
                "api_key_override": TENANT_KEY,
                "tenant_byo_key": True,
            },
        )
        assert got["tenant_byo_key"] is True
        assert got["api_key_override"] == TENANT_KEY
        assert (got["model_provider"], got["model"]) == ("openai", "gpt-4o")

    def test_flag_without_a_key_is_false(self, monkeypatch):
        got = _build(
            monkeypatch,
            CUSTOMER_TENANT_A,
            {
                "agent_role": "r",
                "model_provider": "inference_router",
                "model": "local-llm",
                "tenant_byo_key": True,
            },
        )
        assert got["tenant_byo_key"] is False


class TestAgentFactoryForwardsFlag:
    """N1 (agent_factory). MUTATION: stop forwarding tenant_byo_key ⇒ red."""

    def _kwargs(self, *, byo: bool, key: str | None) -> dict[str, Any]:
        from src.worker.factory import agent_factory as af

        wc, ac = af._create_worker_configs(
            "w",
            [],
            "role",
            "pydantic_ai",
            "openai",
            "gpt-4o",
            0.1,
            None,
            None,
            None,
            None,
            None,
            None,
            key,
            None,
            tenant_byo_key=byo,
        )
        captured: dict[str, Any] = {}

        def fake_create_agent(agent_type: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        with patch("src.worker.agents.agent_registry.create_agent", side_effect=fake_create_agent):
            _run(af._create_agent_worker(wc, ac, [], "", [], tenant_id=CUSTOMER_TENANT_A))
        return captured

    def test_flag_forwarded_with_a_key(self):
        kw = self._kwargs(byo=True, key=TENANT_KEY)
        assert kw["tenant_byo_key"] is True
        assert kw["api_key"] == TENANT_KEY

    def test_flag_absent_without_a_key(self):
        assert "tenant_byo_key" not in self._kwargs(byo=True, key=None)


# =============================================================== N4


class TestFetcherEmptyKey:
    """MUTATION: restore ``api_key or _require_env(...)`` ⇒ the owner key is read ⇒ red."""

    @pytest.mark.parametrize(
        "fetch_name",
        [
            "_fetch_openai_models",
            "_fetch_deepseek_models",
            "_fetch_groq_models",
            "_fetch_xai_models",
            "_fetch_cohere_models",
            "_fetch_google_models",
            "_fetch_anthropic_models",
        ],
    )
    def test_empty_byo_key_never_falls_back_to_env(self, fetch_name):
        from src.api.providers import _router

        with (
            patch.object(_router, "_require_env") as env,
            patch.object(_router, "_httpx_json_get", new_callable=AsyncMock) as http_get,
        ):
            with pytest.raises(RuntimeError, match="empty tenant key"):
                _run(getattr(_router, fetch_name)(api_key=""))
        env.assert_not_called()
        http_get.assert_not_called()

    def test_platform_call_without_a_key_still_uses_env(self, monkeypatch):
        from src.api.providers import _router

        monkeypatch.setenv("OPENAI_API_KEY", "sk-owner")
        with patch.object(
            _router, "_httpx_json_get", new_callable=AsyncMock, return_value={"data": []}
        ) as http_get:
            _run(_router._fetch_openai_models())
        assert http_get.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-owner"


# =============================================================== N5


class TestIRModelCache:
    def _registry(self, monkeypatch, handler):
        from src.services import inference_router_registry as m
        from src.settings import reset_settings_cache

        monkeypatch.setenv("INFERENCE_ROUTER_URL", "http://inference-router.invalid")
        reset_settings_cache()
        real_client = httpx.AsyncClient

        def factory(*a: Any, **k: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(handler), **k)

        monkeypatch.setattr(m.httpx, "AsyncClient", factory)
        return m.InferenceRouterRegistry()

    @pytest.mark.asyncio
    async def test_tenants_do_not_wait_on_each_others_fetch(self, monkeypatch):
        """MUTATION: hold one global lock across the fetch ⇒ B waits for A, which
        waits for B ⇒ timeout ⇒ red."""
        a_started = asyncio.Event()
        b_started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            tenant = request.headers.get("x-tenant-id")
            if tenant == CUSTOMER_TENANT_A:
                a_started.set()
                await asyncio.wait_for(b_started.wait(), 2)
            else:
                b_started.set()
            return httpx.Response(200, json={"data": [{"id": "local-llm"}]})

        reg = self._registry(monkeypatch, handler)
        a, b = await asyncio.wait_for(
            asyncio.gather(
                reg.get_all_models(CUSTOMER_TENANT_A), reg.get_all_models(CUSTOMER_TENANT_B)
            ),
            5,
        )
        assert [m.id for m in a] == [m.id for m in b] == ["local-llm"]

    @pytest.mark.asyncio
    async def test_same_tenant_concurrent_misses_share_one_fetch(self, monkeypatch):
        calls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.headers.get("x-tenant-id") or "")
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={"data": [{"id": "local-llm"}]})

        reg = self._registry(monkeypatch, handler)
        await asyncio.gather(*(reg.get_all_models(CUSTOMER_TENANT_A) for _ in range(5)))
        assert calls == [CUSTOMER_TENANT_A]

    @pytest.mark.asyncio
    async def test_a_failed_fetch_is_not_cached_for_the_ttl(self, monkeypatch):
        """MUTATION: cache [] in the model cache on failure ⇒ the recovered IR is
        not seen for the full TTL ⇒ red."""
        from src.services import inference_router_registry as m

        monkeypatch.setattr(m, "_FAILURE_CACHE_TTL_S", 0.05)
        state = {"up": False, "calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if not state["up"]:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "local-llm"}]})

        reg = self._registry(monkeypatch, handler)
        assert await reg.get_all_models(CUSTOMER_TENANT_A) == []
        assert await reg.get_all_models(CUSTOMER_TENANT_A) == []  # inside the short window
        assert state["calls"] == 1
        state["up"] = True
        await asyncio.sleep(0.1)
        assert [x.id for x in await reg.get_all_models(CUSTOMER_TENANT_A)] == ["local-llm"]
        assert state["calls"] == 2


# =============================================================== N2 (documented only)


def test_n2_absent_tenant_header_is_still_the_platform_default_documented():
    """N2 is PRE-EXISTING and deliberately NOT changed here: a tokenless
    trusted-network request with no X-Tenant-Id gets DEFAULT_TENANT_ID, which is
    a platform tenant. ``is_commercial_platform_tenant`` itself fails closed on absent /
    unknown input; the residual is the middleware default. Pinned so a change
    to either side is a conscious one."""
    from src.api.tenant_constants import is_commercial_platform_tenant

    assert is_commercial_platform_tenant(None) is False
    assert is_commercial_platform_tenant("") is False
    assert is_commercial_platform_tenant("aaaaaaaa-0000-4000-8000-000000000001") is True


# =============================================================== fix round 3
# Validator 2 (2026-10-01): D1 + N-a, N-b, N-c (M15), N-d, N-f.


class TestD1KeyedLocalRowIsNotByo:
    """D1. A keyed `local` (inference-router) row must not set tenant_byo_key, and
    enforce_customer_routing must coerce it to local-llm regardless.

    MUTATION (resolver): ``tenant_byo_key = bool(usable_key)`` ⇒ the flag is True
    and the repro reaches IR with grok-4-fast ⇒ red.
    MUTATION (routing): early-return on ``tenant_byo_key`` for any provider ⇒
    the routing cases go red.
    """

    @pytest.mark.asyncio
    async def test_repro_keyed_local_row_with_paid_ir_model_is_coerced(self, monkeypatch):
        monkeypatch.setattr(
            r,
            "_db",
            _RowsDB({CUSTOMER_TENANT_A: [_row(provider="local", model="grok-4-fast", key="x")]}),
        )
        cfg = await r.resolve_llm_config(
            {"agent_role": "r", "model_provider": "inference_router", "model": "local-llm"},
            CUSTOMER_TENANT_A,
            work_tier="medium",
            key_entitled=True,
        )
        assert cfg["model_provider"] == "inference_router"
        assert cfg["tenant_byo_key"] is False
        cfg = r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)
        assert (cfg["model_provider"], cfg["model"]) == ("inference_router", "local-llm")
        assert cfg.get("ir_provider") is None

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("inference_router", "grok-4-fast"),
            ("inference_router", "default"),
            ("vllm", "qwen-x"),
            ("mlx", "qwen-x"),
            (None, "qwen-x"),
        ],
    )
    def test_routing_does_not_skip_ir_or_local_direct_on_the_byo_flag(self, provider, model):
        cfg = {
            "agent_role": "r",
            "model_provider": provider,
            "model": model,
            "api_key_override": "x",
            "tenant_byo_key": True,
            "ir_provider": "xai",
        }
        out = r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)
        assert (out["model_provider"], out["model"]) == ("inference_router", "local-llm")
        assert out["ir_provider"] is None

    def test_routing_still_skips_a_real_external_byo_key(self):
        """The other half: a customer on its own OpenAI key keeps it."""
        cfg = {
            "agent_role": "r",
            "model_provider": "openai",
            "model": "gpt-4o",
            "api_key_override": TENANT_KEY,
            "tenant_byo_key": True,
        }
        out = r.enforce_customer_routing(cfg, CUSTOMER_TENANT_A)
        assert (out["model_provider"], out["model"]) == ("openai", "gpt-4o")

    @pytest.mark.asyncio
    async def test_external_row_still_sets_the_flag(self, monkeypatch):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row()]}))
        cfg = await r.resolve_llm_config(
            {"model_provider": "inference_router", "model": "local-llm"},
            CUSTOMER_TENANT_A,
            work_tier="medium",
            key_entitled=True,
        )
        assert cfg["tenant_byo_key"] is True


class _WritableDB(_RowsDB):
    async def upsert_config(self, *, tenant_id: str, role: str = "__default__", **cols: Any):
        self.take_audit(tenant_id, role, cols)
        self.rows.setdefault(tenant_id, [])
        self.rows[tenant_id] = [x for x in self.rows[tenant_id] if x["role"] != role]
        self.rows[tenant_id].append(
            {"tenant_id": tenant_id, "role": role, "updated_at": None, **cols}
        )


class TestD1PutRefusesNonLocalLlmOnLocalRow:
    """D1 write side. MUTATION: drop the customer local-llm check in PUT ⇒ 200 ⇒ red."""

    @pytest.fixture
    def put_client(self, monkeypatch):
        from src.api.tenant_config import _router as tc
        from src.api.worker_management_api import app
        from src.auth import TenantContext, get_tenant_context

        db = _WritableDB({})
        monkeypatch.setattr(tc, "_db", db)
        holder = {"tenant": CUSTOMER_TENANT_A}
        app.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=holder["tenant"],
            subject_id=holder["tenant"],
            tenant_name="t",
            visibility="tenant",
            byo_mode="local_allowed",
        )
        with (
            patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
            patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            yield c, db, holder
        app.dependency_overrides.clear()

    @pytest.mark.parametrize("model", ["grok-4-fast", "default"])
    def test_customer_local_row_with_other_model_is_422_and_not_written(self, put_client, model):
        client, db, _ = put_client
        resp = client.put(
            "/tenant-config/llm?tier=medium",
            json={"provider": "local", "model": model, "apiKey": "x"},
        )
        assert resp.status_code == 422, resp.text
        msg = resp.json()["error"]["message"]
        assert model in msg and "local-llm" in msg
        assert db.rows == {}

    def test_customer_local_llm_row_is_accepted(self, put_client):
        client, db, _ = put_client
        resp = client.put(
            "/tenant-config/llm?tier=medium", json={"provider": "local", "model": "local-llm"}
        )
        assert resp.status_code == 200, resp.text
        assert db.rows[CUSTOMER_TENANT_A][0]["model"] == "local-llm"

    def test_platform_tenant_keeps_the_ir_model_choice(self, put_client):
        client, db, holder = put_client
        holder["tenant"] = PLATFORM_TENANT
        resp = client.put(
            "/tenant-config/llm?tier=medium", json={"provider": "local", "model": "grok-4-fast"}
        )
        assert resp.status_code == 200, resp.text
        assert db.rows[PLATFORM_TENANT][0]["model"] == "grok-4-fast"


class TestM15LiveCacheIsPerTenant:
    """N-c / M15. MUTATION: drop tenant from _byo_cache_key ⇒ tenant B reuses
    tenant A's cached verdict, the second fetch never happens ⇒ red."""

    def test_same_key_two_tenants_fetch_separately(self):
        from src.api.providers import _router as pr

        entry = {"api_key": TENANT_KEY, "base_url": None}
        fetch = AsyncMock(side_effect=[["gpt-4o"], RuntimeError("401 bad key")])
        with patch.object(pr, "_fetch_row_models", fetch):
            assert _run(pr.live_row_models(CUSTOMER_TENANT_A, "openai", entry)) == ["gpt-4o"]
            with pytest.raises(pr.ByoLiveCheckError):
                _run(pr.live_row_models(CUSTOMER_TENANT_B, "openai", entry))
            # Tenant A is still served from ITS cache entry.
            assert _run(pr.live_row_models(CUSTOMER_TENANT_A, "openai", entry)) == ["gpt-4o"]
        assert fetch.await_count == 2
        assert [c.args[0] for c in fetch.await_args_list] == [CUSTOMER_TENANT_A, CUSTOMER_TENANT_B]


class TestRetryRebuildRefusal:
    """N-b. The retry ladder's REBUILD raising a key refusal must map to the same
    403/409/402 as the first build. MUTATION: drop the rebuild mapping ⇒ a
    generic 500 ⇒ red."""

    def _drive(self, message_client, monkeypatch, refusal: BaseException):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("LLM_RETRY_BASE_DELAY_S", "0")
        monkeypatch.setenv("LLM_RETRY_MAX_DELAY_S", "0")
        reset_settings_cache()
        client, holder, state_db = message_client

        class _Agent:
            config = SimpleNamespace(tenant_byo_key=False)

            async def run(self, *_a: Any, **_k: Any) -> Any:
                # First run dies transiently ⇒ the ladder rebuilds, and the
                # rebuild is refused (the row changed mid-run).
                holder["raise"] = refusal
                raise RuntimeError("Connection error.")

        holder["agent"] = _Agent()
        resp = client.post("/workers/byo_probe/message", json={"message": "hi"})
        states = [c.kwargs.get("state_type") for c in state_db.add_worker_state.await_args_list]
        return resp, states

    @pytest.mark.parametrize(
        ("reason", "status", "code"),
        [
            ("withheld", 403, "BYO_KEY_WITHHELD"),
            ("decrypt_failed", 409, "BYO_KEY_UNUSABLE"),
        ],
    )
    def test_key_refusal_on_rebuild_keeps_its_shape(
        self, message_client, monkeypatch, reason, status, code
    ):
        refusal = r.TenantLLMKeyUnusableError(
            tenant_id=CUSTOMER_TENANT_A,
            provider="openai",
            tier="medium",
            role="tier:medium",
            reason=reason,
        )
        resp, states = self._drive(message_client, monkeypatch, refusal)
        assert resp.status_code == status, resp.text
        body = resp.json()["error"]
        assert body["code"] == code
        assert body["details"]["provider"] == "openai"
        assert states[-1] == "error"

    def test_byo_required_on_rebuild_is_402(self, message_client, monkeypatch):
        """WM-2. MUTATION: drop 402 typing in error_handlers ⇒ code "ERROR" with
        the dict nested under ``message`` ⇒ red."""
        resp, _ = self._drive(message_client, monkeypatch, r.BYORequiredError("no row"))
        assert resp.status_code == 402, resp.text
        err = resp.json()["error"]
        assert err["code"] == "byo_llm_not_configured"
        assert err["message"] == "Configure your LLM provider in Settings before running agents."
        assert err["status"] == 402


class TestStreamingChatByoError:
    """N-a. MUTATION: stream the raw ``str(exc)`` ⇒ the key appears and the
    envelope code is absent ⇒ red."""

    def test_stream_failure_on_tenant_key_is_scrubbed_envelope(self, chat_client, monkeypatch):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]}))
        p1, p2 = _as_customer(entitled=True)
        agent = MagicMock()
        agent.run_stream = MagicMock(
            side_effect=RuntimeError(f"401 https://vendor/v1?key={TENANT_KEY}")
        )
        with (
            p1,
            p2,
            patch(
                "src.api.providers._chat_router._build_agent",
                new_callable=AsyncMock,
                return_value=agent,
            ),
        ):
            resp = chat_client.post(
                "/chat/completions",
                json={
                    "provider": "openai",
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        assert TENANT_KEY not in resp.text
        frames = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        errs = [f for f in frames if "error" in f]
        assert len(errs) == 1
        frame = errs[0]
        # Validator 3 D4: ``error`` stays the scrubbed STRING (fr parses it);
        # the envelope fields sit beside it.
        assert isinstance(frame["error"], str)
        assert frame["error_code"] == "BYO_PROVIDER_CALL_FAILED"
        assert frame["status"] == 502
        assert "openai" in frame["error"] and "***" in frame["error"]
        assert frame["details"]["provider"] == "openai"
        assert frame["provider"] == "openai"
        # fr's parser (frontend/src/routes/chat/streamChat.ts parseFrame) turns
        # it into a visible error, not a skipped frame.
        assert _fr_parse_frame(f"data: {json.dumps(frame)}") == {
            "kind": "error",
            "message": frame["error"],
        }


def _fr_parse_frame(frame: str) -> dict | None:
    """Line-for-line port of fr origin/main ``frontend/src/routes/chat/streamChat.ts``
    ``parseFrame`` (2026-10-01). An error frame is recognised ONLY when
    ``typeof obj.error === 'string'``; any other shape returns null and fr skips
    the frame silently, leaving an empty assistant bubble and no error."""
    line = frame.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "":
        return None
    if payload == "[DONE]":
        return {"kind": "done"}
    try:
        obj = json.loads(payload)
    except ValueError:
        return {"kind": "malformed", "raw": payload}
    if isinstance(obj.get("error"), str):
        return {"kind": "error", "message": obj["error"]}
    if isinstance(obj.get("delta"), str):
        return {"kind": "delta", "delta": obj["delta"]}
    if isinstance(obj.get("meta"), dict):
        return {"kind": "meta", "meta": obj["meta"]}
    if "result" in obj:
        return {"kind": "result", "value": obj["result"]}
    return None


class TestStreamErrorFrameStaysParseableByFr:
    """Validator 3 D4. MUTATION: emit ``{"error": {code, message, ...}}`` (the
    object envelope) ⇒ ``_fr_parse_frame`` returns None ⇒ red."""

    def test_fr_parser_port_skips_an_object_error(self):
        # Pins the port itself: the shape fr cannot read IS skipped.
        assert _fr_parse_frame('data: {"error": {"code": "X", "message": "m"}}') is None
        assert _fr_parse_frame('data: {"error": "m"}') == {"kind": "error", "message": "m"}

    def test_every_error_frame_on_the_byo_stream_is_an_fr_error_event(
        self, chat_client, monkeypatch
    ):
        monkeypatch.setattr(r, "_db", _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]}))
        p1, p2 = _as_customer(entitled=True)
        agent = MagicMock()
        agent.run_stream = MagicMock(side_effect=RuntimeError("upstream 500"))
        with (
            p1,
            p2,
            patch(
                "src.api.providers._chat_router._build_agent",
                new_callable=AsyncMock,
                return_value=agent,
            ),
        ):
            resp = chat_client.post(
                "/chat/completions",
                json={
                    "provider": "openai",
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        events = [_fr_parse_frame(block) for block in resp.text.split("\n\n") if block.strip()]
        kinds = [e["kind"] for e in events if e]
        assert kinds == ["error", "done"], (kinds, resp.text)


class TestKeyTestCooldown:
    """N-d. MUTATION: drop the cooldown ⇒ the vendor is called twice ⇒ red."""

    @pytest.fixture
    def kt_client(self, monkeypatch):
        from src.api.tenant_config import _router as tc
        from src.api.worker_management_api import app
        from src.auth import TenantContext, get_tenant_context

        db = _RowsDB({CUSTOMER_TENANT_A: [_row(role="__default__")]})
        monkeypatch.setattr(tc, "_db", db)
        monkeypatch.setattr(r, "_db", db)
        monkeypatch.setattr(tc, "tenant_key_entitled", lambda _t: True)
        app.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=CUSTOMER_TENANT_A,
            subject_id=CUSTOMER_TENANT_A,
            tenant_name="a",
            visibility="tenant",
            byo_mode="local_allowed",
        )
        with (
            patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
            patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            yield c
        app.dependency_overrides.clear()

    def test_second_test_within_window_reuses_the_last_result(self, kt_client):
        calls: list[str] = []

        async def fake_get(url: str, *, headers=None, timeout=10.0):
            calls.append(url)
            if len(calls) == 1:
                raise httpx.HTTPStatusError(
                    "401 Unauthorized",
                    request=httpx.Request("GET", url),
                    response=httpx.Response(401),
                )
            return {"data": [{"id": "gpt-4o"}]}

        with patch("src.api.providers._router._httpx_json_get", side_effect=fake_get):
            first = kt_client.post("/tenant-config/llm/test", json={}).json()
            second = kt_client.post("/tenant-config/llm/test", json={}).json()
        assert len(calls) == 1
        assert first["ok"] is False and second["ok"] is False
        assert second["error"] == first["error"]

    def test_cooldown_is_keyed_by_tenant(self):
        from src.api.providers import _router as pr

        entry = {"api_key": TENANT_KEY, "base_url": None}
        assert pr._byo_cache_key(CUSTOMER_TENANT_A, "openai", entry) != pr._byo_cache_key(
            CUSTOMER_TENANT_B, "openai", entry
        )


class TestEmptyIssuerRefusesProof:
    """N-f. MUTATION: restore ``if expected_iss and ...`` / drop the early
    refusal ⇒ a valid token proves the tenant with no issuer pin ⇒ red."""

    def test_empty_issuer_setting_refuses_proof(self, portal_key, portal_jwks, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("END_USER_TOKEN_ISSUER", "")
        reset_settings_cache()
        tenant, reason = _run(
            end_user_token.verify_end_user_tenant(f"Bearer {_mint(portal_key)}", CUSTOMER_TENANT_A)
        )
        assert (tenant, reason) == ("", "issuer_unconfigured")

    def test_verify_itself_refuses_an_empty_issuer(self, portal_key, monkeypatch):
        from src.settings import reset_settings_cache

        monkeypatch.setenv("END_USER_TOKEN_ISSUER", "  ")
        reset_settings_cache()
        pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(portal_key.public_key()))
        pub.update({"kid": KID, "alg": "RS256", "use": "sig"})
        with pytest.raises(jwt.InvalidIssuerError):
            end_user_token._verify(
                _mint(portal_key),
                {"keys": [pub]},
                expected_realm="portal",
                expected_issuer="",
            )
