"""worker_management#245: a tenant's stored LLM key is decrypted ONLY for a
caller that proved that tenant with its own licence.

``resolve_llm_config`` used to decrypt ``tenant_llm_config.api_key_encrypted``
for whatever ``tenant_id`` it was handed — including the ADOPTED tenant of a
delegating service account (``X-Effective-Tenant-Id``, checked for spelling
only) and the asserted tenant of a tokenless trusted-network caller. These pin:

* every non-licence provenance gets NO decrypted key, and ``decrypt_secret`` is
  never even called (not "decrypted then dropped");
* a licence-proven caller still gets its own key;
* a licence proving tenant A never unlocks tenant B's key;
* a key-less row (``local``) is unaffected;
* the worker cache never hands a proven caller's key-bearing instance to an
  unproven caller of the same tenant.
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest

from src.api.identity_context import (
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_BYPASS,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_LICENSE,
    TENANT_PROVENANCE_UNKNOWN,
    set_identity_context,
)
from src.api.workers import _globals
from src.security.secret_box import encrypt_secret
from src.worker.factory import llm_config_resolver as r
from src.worker.factory.llm_config_resolver import (
    TenantLLMKeyUnusableError,
    resolve_llm_config,
    tenant_key_entitled,
)

pytestmark = pytest.mark.asyncio

TENANT_A = "019edc1e-0000-7000-8000-00000000000a"
TENANT_B = "019edc1e-0000-7000-8000-00000000000b"
KEYS = {TENANT_A: "sk-tenant-A", TENANT_B: "sk-tenant-B"}

UNPROVEN = [
    TENANT_PROVENANCE_DELEGATED,
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_BYPASS,
    TENANT_PROVENANCE_UNKNOWN,
]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    yield
    set_identity_context(None, None, None)


class _PerTenantDB:
    """Each tenant holds its OWN keyed anthropic row."""

    async def get_config(self, tenant_id: str, role: str = "__default__") -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": None,
            "temperature": None,
            "max_tokens": None,
            "api_key_encrypted": encrypt_secret(KEYS[tenant_id]),
            "cost_source": "worker-management",
        }


@pytest.fixture
def decrypts(monkeypatch):
    """Record every tenant whose key reached ``decrypt_secret``."""
    calls: list[str] = []
    real = r.decrypt_secret

    def _spy(blob: str) -> str:
        out = real(blob)
        calls.append(out)
        return out

    monkeypatch.setattr(r, "_db", _PerTenantDB())
    monkeypatch.setattr(r, "decrypt_secret", _spy)
    return calls


def _as(tenant: str, provenance: str) -> None:
    set_identity_context(tenant, "lic-1", "sub-1", tenant_provenance=provenance)


def _system_cfg() -> dict[str, Any]:
    return {"model_provider": "inference_router", "model": "default"}


@pytest.mark.parametrize("provenance", UNPROVEN)
async def test_unproven_tenant_gets_no_decrypted_key(decrypts, provenance):
    """Owner ruling 2026-10-01: a withheld key is a CLEAR 403, never a silent
    fall-through to seeded system routing (it used to return the platform
    config here)."""
    _as(TENANT_A, provenance)
    with pytest.raises(TenantLLMKeyUnusableError) as exc_info:
        await resolve_llm_config(_system_cfg(), TENANT_A)
    assert decrypts == [], f"{provenance}: key was decrypted"
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "BYO_KEY_WITHHELD"
    assert "anthropic" in exc_info.value.message
    assert KEYS[TENANT_A] not in exc_info.value.message


async def test_adopted_tenant_that_requires_byo_hard_fails_rather_than_riding_the_key(decrypts):
    _as(TENANT_A, TENANT_PROVENANCE_DELEGATED)
    with pytest.raises(TenantLLMKeyUnusableError) as exc_info:
        await resolve_llm_config(_system_cfg(), TENANT_A, byo_mode="byo_required")
    assert exc_info.value.withheld
    assert decrypts == []


async def test_proven_tenant_still_gets_its_own_key(decrypts):
    _as(TENANT_A, TENANT_PROVENANCE_LICENSE)
    out = await resolve_llm_config(_system_cfg(), TENANT_A)
    assert out["api_key_override"] == KEYS[TENANT_A]
    assert out["model_provider"] == "anthropic"
    assert decrypts == [KEYS[TENANT_A]]


async def test_a_licence_for_tenant_a_never_unlocks_tenant_b(decrypts):
    """The graph worker node can resolve a tenant from graph state, not the
    request; a licence proves exactly one tenant."""
    _as(TENANT_A, TENANT_PROVENANCE_LICENSE)
    with pytest.raises(TenantLLMKeyUnusableError) as exc_info:
        await resolve_llm_config(_system_cfg(), TENANT_B)
    assert exc_info.value.withheld
    assert KEYS[TENANT_B] not in decrypts
    assert decrypts == []


async def test_keyless_local_row_is_still_honoured_for_an_unproven_caller(monkeypatch):
    class _LocalDB:
        async def get_config(self, tenant_id, role="__default__"):
            return {
                "provider": "local",
                "model": "local-llm",
                "base_url": None,
                "temperature": None,
                "max_tokens": None,
                "api_key_encrypted": None,
            }

    monkeypatch.setattr(r, "_db", _LocalDB())
    _as(TENANT_A, TENANT_PROVENANCE_ASSERTED)
    out = await resolve_llm_config(_system_cfg(), TENANT_A)
    assert out["model"] == "local-llm"


async def test_entitlement_predicate():
    assert tenant_key_entitled(None) is False
    _as(TENANT_A, TENANT_PROVENANCE_LICENSE)
    assert tenant_key_entitled(TENANT_A) is True
    assert tenant_key_entitled(TENANT_B) is False
    for provenance in UNPROVEN:
        _as(TENANT_A, provenance)
        assert tenant_key_entitled(TENANT_A) is False, provenance


async def test_cache_never_hands_a_proven_instance_to_an_unproven_caller(monkeypatch):
    seen: list[bool] = []

    async def _fake_create(self, worker_name, worker_data, **kw):
        seen.append(kw["key_entitled"])
        return object()

    monkeypatch.setattr(_globals.WorkerInstanceCache, "_create_worker_instance", _fake_create)
    monkeypatch.setattr(_globals.WorkerInstanceCache, "_is_worker_healthy", _healthy)
    cache = _globals.WorkerInstanceCache()
    data = {"worker_name": "w", "config": {}}

    _as(TENANT_A, TENANT_PROVENANCE_LICENSE)
    proven = await cache.get_or_create_worker("w", data, tenant_id=TENANT_A)
    _as(TENANT_A, TENANT_PROVENANCE_DELEGATED)
    adopted = await cache.get_or_create_worker("w", data, tenant_id=TENANT_A)

    assert adopted is not proven
    assert seen == [True, False]


async def _healthy(*_args: object) -> bool:
    return True
