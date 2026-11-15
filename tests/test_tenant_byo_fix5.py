"""Validator fix5 (2026-10-01): modality base_url, managed-row lock, Cohere hook.

Owner ruling: no customer tenant may use the owner's commercial keys, except
managed BYO (``owner_provided``) for the one tenant it is set for, with no
leakage. Each class names the defect it pins and the mutation that turns it red.

- M1  ``PUT /tenant-config/modality/{m}`` stored any ``baseUrl``, and ``/resolve``
      handed it back with the key (the OWNER's key for a managed row), so the
      key could be sent to a tenant-chosen or owner-infra endpoint.
- M2  a managed row's provider / base_url / key / mark were tenant-editable,
      on the modality AND the LLM PUT.
- M3  the Cohere BYO client had no per-hop owner-infra hook.
"""

from __future__ import annotations

import datetime
import os
import secrets
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx
import pytest
from fastapi.testclient import TestClient

from src.security import managed_byo
from src.security import request_overrides as ro
from src.security.managed_byo import MANAGED_BYO_ROW_LOCKED
from src.security.request_overrides import (
    TENANT_BASE_URL_NOT_ALLOWED,
    TenantBaseUrlNotAllowedError,
    refuse_owner_infra_request,
)
from src.security.secret_box import encrypt_secret
from src.settings import reset_settings_cache
from tests._platform_tenant import CUSTOMER_TENANT_A, PLATFORM_TENANT
from tests._tenant_llm_audit_fake import AuditFakeMixin

IR_V1 = "http://inference-router.inference-router.svc.cluster.local:8150/v1"
VENDOR = "https://api.vendor-byo.example/v1"
VENDOR_ORIGIN = "https://api.vendor-byo.example:443"
OTHER_VENDOR = "https://api.other-vendor.example/v1"
VENDOR_IP = "93.184.215.14"
# Not a credential: an opaque placeholder standing in for the OWNER's key.
OWNER_KEY_PLACEHOLDER = "owner-managed-cred-placeholder"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.setenv(
        "TENANT_BASE_URL_ALLOWLIST",
        f"{CUSTOMER_TENANT_A}={VENDOR_ORIGIN},{CUSTOMER_TENANT_A}=https://api.other-vendor.example:443",
    )
    reset_settings_cache()
    table = {"api.vendor-byo.example": (VENDOR_IP,), "api.other-vendor.example": (VENDOR_IP,)}
    monkeypatch.setattr(ro, "_resolve_host_ips", lambda host: table.get(host, ()))
    ro._dns_cache.clear()
    yield table
    reset_settings_cache()


def _as_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        managed_byo,
        "get_caller_roles_from_context",
        lambda: (managed_byo.MANAGED_BYO_ADMIN_ROLE,),
    )


def _ctx(tenant: str):
    from src.auth import TenantContext

    return TenantContext(
        tenant_id=tenant,
        subject_id=tenant,
        tenant_name="t",
        visibility="tenant",
        byo_mode="local_allowed",
        identity_proven=True,
        tenant_proven=True,
    )


class _ModalityDB:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.upserts = 0

    async def get_config(self, tenant_id, modality, role="__default__"):
        row = self.rows.get((str(tenant_id), modality, role))
        return dict(row) if row else None

    async def upsert_config(self, *, tenant_id, modality, role="__default__", **cols):
        self.upserts += 1
        self.rows[(str(tenant_id), modality, role)] = {
            "tenant_id": str(tenant_id),
            "modality": modality,
            "role": role,
            "updated_at": datetime.datetime(2026, 10, 1, tzinfo=datetime.UTC),
            **cols,
        }


class _LLMDB(AuditFakeMixin):
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.upserts = 0

    async def get_config(self, tenant_id, role="__default__"):
        row = self.rows.get((str(tenant_id), role))
        return dict(row) if row else None

    async def upsert_config(self, *, tenant_id, role="__default__", **cols):
        self.take_audit(tenant_id, role, cols)
        self.upserts += 1
        self.rows[(str(tenant_id), role)] = {"tenant_id": str(tenant_id), "role": role, **cols}


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch):
    from src.api.tenant_config import _modality_router as mr
    from src.api.tenant_config import _router as tc
    from src.api.worker_management_api import app
    from src.auth import get_tenant_context

    mdb, ldb = _ModalityDB(), _LLMDB()
    monkeypatch.setattr(mr, "_db", mdb)
    monkeypatch.setattr(tc, "_db", ldb)
    tenant = {"id": CUSTOMER_TENANT_A}
    app.dependency_overrides[get_tenant_context] = lambda: _ctx(tenant["id"])
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c, mdb, ldb, tenant
    app.dependency_overrides.clear()


def _code(resp: httpx.Response) -> str | None:
    body = resp.json()
    return body.get("error_code") or (body.get("error") or {}).get("code")


# ───────────────────────────────────────────── M1: modality base_url at PUT


class TestM1ModalityPutValidatesBaseUrl:
    """MUTATION: drop the ``_checked_base_url`` call in ``put_modality_config`` ⇒
    every refusal below is a 200 and the row is written ⇒ red."""

    @pytest.mark.parametrize(
        ("modality", "provider", "url"),
        [
            ("embedding", "openai_compatible", IR_V1),
            ("embedding", "openai_compatible", "http://10.43.0.10:8000/v1"),
            ("vlm", "openai_compatible", "http://mlx-inference.example.com:8080/v1"),
            ("tts", "self_hosted", "http://tts.example.com:9202"),  # the preset default
            ("embedding", "openai_compatible", "https://not-allowlisted.example/v1"),
        ],
    )
    def test_customer_put_is_422_and_not_written(self, app_client, modality, provider, url):
        c, mdb, _, _ = app_client
        resp = c.put(
            f"/tenant-config/modality/{modality}",
            json={
                "provider": provider,
                "model": "m",
                "baseUrl": url,
                "apiKey": OWNER_KEY_PLACEHOLDER,
            },
        )
        assert resp.status_code == 422, resp.text
        assert _code(resp) == TENANT_BASE_URL_NOT_ALLOWED
        assert mdb.upserts == 0

    def test_name_resolving_to_private_ip_is_refused(self, app_client, _env):
        c, mdb, _, _ = app_client
        _env["api.vendor-byo.example"] = ("10.0.0.5",)
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": VENDOR, "apiKey": "k"},
        )
        assert resp.status_code == 422, resp.text
        assert mdb.upserts == 0

    def test_customer_self_hosted_owner_gpu_default_is_refused(self, app_client):
        c, mdb, _, _ = app_client
        resp = c.put(
            "/tenant-config/modality/asr",
            json={
                "provider": "self_hosted",
                "model": "x",
                "baseUrl": "http://asr.example.com:9201",
            },
        )
        assert resp.status_code == 422, resp.text
        assert mdb.upserts == 0

    def test_allowlisted_public_vendor_passes(self, app_client):
        """Control: a tenant's own allowlisted public endpoint still works."""
        c, mdb, _, _ = app_client
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": VENDOR, "apiKey": "k"},
        )
        assert resp.status_code == 200, resp.text
        assert mdb.upserts == 1

    def test_platform_tenant_keeps_the_preset_owner_gpu_default(self, app_client):
        """Control: owner hardware on the owner's own tenant."""
        c, _, _, tenant = app_client
        tenant["id"] = PLATFORM_TENANT
        resp = c.put(
            "/tenant-config/modality/tts",
            json={
                "provider": "self_hosted",
                "model": "k",
                "baseUrl": "http://tts.example.com:9202",
            },
        )
        assert resp.status_code == 200, resp.text

    def test_platform_tenant_other_internal_url_is_refused(self, app_client):
        c, _, _, tenant = app_client
        tenant["id"] = PLATFORM_TENANT
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": IR_V1, "apiKey": "k"},
        )
        assert resp.status_code == 422, resp.text


# ───────────────────────────────────────────── M1: modality base_url at resolve


def _seed(mdb: _ModalityDB, *, url: str, owner_provided: bool, tenant: str = CUSTOMER_TENANT_A):
    mdb.rows[(tenant, "embedding", "__default__")] = {
        "tenant_id": tenant,
        "modality": "embedding",
        "role": "__default__",
        "provider": "openai_compatible",
        "model": "m",
        "base_url": url,
        "params": {},
        "api_key_encrypted": encrypt_secret(OWNER_KEY_PLACEHOLDER),
        "owner_provided": owner_provided,
        "cost_source": None,
    }


class TestM1ModalityResolveRefusesStoredBaseUrl:
    """MUTATION: drop the resolve-time ``_checked_base_url`` ⇒ the stored URL is
    returned with the decrypted key (200) ⇒ red."""

    @pytest.mark.parametrize("owner_provided", [True, False])
    @pytest.mark.parametrize(
        "url", [IR_V1, "http://tts.example.com:9202", "https://not-allowlisted.example/v1"]
    )
    def test_stored_row_is_409_with_no_key_and_no_fallback(self, app_client, url, owner_provided):
        c, mdb, _, _ = app_client
        _seed(mdb, url=url, owner_provided=owner_provided)
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.status_code == 409, resp.text
        assert _code(resp) == "base_url_not_allowlisted"
        assert OWNER_KEY_PLACEHOLDER not in resp.text
        assert "configured" not in resp.json()  # not a silent platform fallback

    def test_rebound_name_is_refused_at_resolve(self, app_client, _env):
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        _env["api.vendor-byo.example"] = ("100.100.100.100",)  # tailnet
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.status_code == 409, resp.text
        assert OWNER_KEY_PLACEHOLDER not in resp.text

    def test_allowlisted_row_resolves(self, app_client, monkeypatch):
        """Control: the managed key still reaches its own tenant's allowed endpoint
        (via the verified service caller only — D5, see test_tenant_byo_fix6)."""
        from src.api.tenant_config import _modality_router as mr

        monkeypatch.setattr(mr, "get_caller_client_id_from_context", lambda: "modality-gateway")
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        resp = c.get("/tenant-config/modality/embedding/resolve")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["base_url"] == VENDOR
        assert body["owner_provided"] is True
        assert body["api_key"] == OWNER_KEY_PLACEHOLDER


# ───────────────────────────────────────────── M2: managed rows are owner-locked


class TestM2ModalityManagedRowLocked:
    """MUTATION: drop ``assert_managed_row_change_allowed`` from the modality PUT
    ⇒ every tenant change below is a 200 ⇒ red."""

    @pytest.mark.parametrize(
        "change",
        [
            {"baseUrl": OTHER_VENDOR},
            {"provider": "openai", "baseUrl": None},
            {"ownerProvided": False},
            {"apiKey": "tenant-own"},  # pragma: allowlist secret
            {"apiKey": ""},
        ],
    )
    def test_tenant_change_is_403_and_row_unchanged(self, app_client, change):
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        before = dict(mdb.rows[(CUSTOMER_TENANT_A, "embedding", "__default__")])
        body = {"provider": "openai_compatible", "model": "m", **change}
        body = {k: v for k, v in body.items() if v is not None}
        resp = c.put("/tenant-config/modality/embedding", json=body)
        assert resp.status_code == 403, resp.text
        assert _code(resp) == MANAGED_BYO_ROW_LOCKED
        assert mdb.upserts == 0
        assert mdb.rows[(CUSTOMER_TENANT_A, "embedding", "__default__")] == before

    def test_tenant_may_not_mark_its_own_row_managed(self, app_client):
        c, mdb, _, _ = app_client
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={
                "provider": "openai_compatible",
                "model": "m",
                "baseUrl": VENDOR,
                "apiKey": "k",
                "ownerProvided": True,
            },
        )
        assert resp.status_code == 403, resp.text
        assert mdb.upserts == 0

    def test_tenant_may_change_model_and_resend_same_base_url(self, app_client):
        """Control: what is not the owner's decision stays editable."""
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m2", "baseUrl": VENDOR + "/"},
        )
        assert resp.status_code == 200, resp.text
        row = mdb.rows[(CUSTOMER_TENANT_A, "embedding", "__default__")]
        assert (row["model"], row["owner_provided"]) == ("m2", True)

    def test_owner_path_still_changes_a_managed_row(self, app_client, monkeypatch):
        """Control: the owner path keeps working."""
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        _as_owner(monkeypatch)
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": OTHER_VENDOR},
        )
        assert resp.status_code == 200, resp.text
        assert mdb.rows[(CUSTOMER_TENANT_A, "embedding", "__default__")]["base_url"] == (
            OTHER_VENDOR
        )

    def test_owner_path_is_still_validated(self, app_client, monkeypatch):
        """The owner role unlocks the row, never the base_url rule."""
        c, mdb, _, _ = app_client
        _seed(mdb, url=VENDOR, owner_provided=True)
        _as_owner(monkeypatch)
        resp = c.put(
            "/tenant-config/modality/embedding",
            json={"provider": "openai_compatible", "model": "m", "baseUrl": IR_V1},
        )
        assert resp.status_code == 422, resp.text
        assert mdb.upserts == 0


class TestM2LLMManagedRowLocked:
    """MUTATION: drop ``assert_managed_row_change_allowed`` from the LLM PUT ⇒ red."""

    @staticmethod
    def _seed(ldb: _LLMDB) -> None:
        ldb.rows[(CUSTOMER_TENANT_A, "__default__")] = {
            "tenant_id": CUSTOMER_TENANT_A,
            "role": "__default__",
            "provider": "openai_compatible",
            "model": "gpt-4o",
            "base_url": VENDOR,
            "temperature": None,
            "max_tokens": None,
            "params": {},
            "api_key_encrypted": encrypt_secret(OWNER_KEY_PLACEHOLDER),
            "owner_provided": True,
        }

    @pytest.mark.parametrize(
        "change",
        [
            {"baseUrl": OTHER_VENDOR},
            {"provider": "anthropic", "model": "claude-sonnet-5"},
            {"ownerProvided": False},
            {"apiKey": "tenant-own"},  # pragma: allowlist secret
        ],
    )
    def test_tenant_change_is_403(self, app_client, change):
        c, _, ldb, _ = app_client
        self._seed(ldb)
        resp = c.put(
            "/tenant-config/llm",
            json={"provider": "openai_compatible", "model": "gpt-4o", **change},
        )
        assert resp.status_code == 403, resp.text
        assert _code(resp) == MANAGED_BYO_ROW_LOCKED
        assert ldb.upserts == 0

    def test_tenant_may_change_model(self, app_client):
        c, _, ldb, _ = app_client
        self._seed(ldb)
        resp = c.put("/tenant-config/llm", json={"provider": "openai_compatible", "model": "o3"})
        assert resp.status_code == 200, resp.text
        assert ldb.rows[(CUSTOMER_TENANT_A, "__default__")]["owner_provided"] is True

    def test_owner_path_changes_it(self, app_client, monkeypatch):
        c, _, ldb, _ = app_client
        self._seed(ldb)
        _as_owner(monkeypatch)
        resp = c.put(
            "/tenant-config/llm",
            json={"provider": "openai_compatible", "model": "gpt-4o", "baseUrl": OTHER_VENDOR},
        )
        assert resp.status_code == 200, resp.text


def test_owner_role_comes_only_from_the_verified_caller_roles():
    """No role in the ContextVar (a tenant licence, tokenless, delegated) ⇒ not owner."""
    assert managed_byo.is_managed_byo_owner_caller() is False


# ───────────────────────────────────────────── M3: Cohere BYO client hook


class TestM3CohereByoClientRefusesOwnerInfraRedirect:
    """MUTATION: drop ``httpx_client=`` from the Cohere BYO client ⇒ the client
    has no request hook and follows the 307 into the cluster ⇒ red."""

    @pytest.mark.asyncio
    async def test_cohere_byo_client_has_the_hook_and_refuses_the_redirect(self, monkeypatch):
        from src.worker.agents import pydantic_ai_agent as pa
        from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

        built: dict[str, Any] = {}
        real = pa.AsyncCohereClient

        def _record(*args: Any, **kwargs: Any):
            built.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(pa, "AsyncCohereClient", _record)
        cfg = PydanticAIAgentConfig(
            name="t",
            model_provider="cohere",
            model="command-r",
            base_url=VENDOR,
            api_key="tenant-cohere",  # pragma: allowlist secret
            tenant_byo_key=True,
            db_path="",
        )
        ModelFactory._create_cohere_model(cfg)
        client = built.get("httpx_client")
        assert isinstance(client, httpx.AsyncClient)
        assert refuse_owner_infra_request in client.event_hooks["request"]

        hits: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hits.append(str(request.url))
            if request.url.host == "api.vendor-byo.example":
                return httpx.Response(307, headers={"location": IR_V1 + "/chat"})
            return httpx.Response(200, json={"ok": True})

        client._transport = httpx.MockTransport(handler)
        with pytest.raises(TenantBaseUrlNotAllowedError):
            await client.post(VENDOR + "/chat", json={})
        assert hits == [VENDOR + "/chat"]
        await client.aclose()
