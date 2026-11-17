"""`model_provider: vllm` must put the SAME auth + attribution on the wire as
`model_provider: inference_router` for the same run.

Owner ruling: every GPU call is attributed to a real tenant. gitops #2999
repointed `VLLM_URL` at inference-router, so the `vllm` provider is an IR client
under another name -- ~93 auto-created worker rows use it and are recreated as
`vllm` by their callers, so this is fixed in code, not in rows.

The two clients used to diverge on `Authorization`: `api_key="EMPTY"` on the  # pragma: allowlist secret
vllm path vs the IR placeholder on the other. These tests drive a real request
through BOTH constructed object graphs (MockTransport, not object inspection)
and compare what reached the wire, plus the loud-failure log lines when the
tenant or licence cannot be resolved.
"""

from __future__ import annotations

import logging

import httpx2 as httpx
import pytest

from src.services import ir_bundle_tokens
from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig

REAL_TENANT = "e148e8af-3751-49bf-a0fa-56ab46d31ddc"
IR_BASE = "http://inference-router.inference-router.svc.cluster.local:8150"

# Every auth / attribution header either path may stamp.
IDENTITY_HEADERS = (
    "authorization",
    "x-tenant-id",
    "x-license-id",
    "x-subject-id",
    "x-worker-name",
    "x-provider",
)

_COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "local-llm",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


def _cfg(provider: str, identity: dict[str, str]) -> PydanticAIAgentConfig:
    return PydanticAIAgentConfig(
        name="principle_consistency",
        model_provider="vllm" if provider == "vllm" else "inference_router",
        model="local-llm",
        db_path="",
        base_url=IR_BASE,
        ir_provider="pgx-local",
        tenant_id=identity.get("tenant_id"),
        license_id=identity.get("license_id"),
        subject_id=identity.get("subject_id"),
    )


async def _wire(provider: str, **identity: str) -> dict[str, str]:
    """Send one completion through the real client for `provider`; return the
    identity headers that actually left the process."""
    seen: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["__path"] = request.url.path
        seen.update({k: v for k, v in request.headers.items() if k in IDENTITY_HEADERS})
        return httpx.Response(200, json=_COMPLETION)

    cfg = _cfg(provider, identity)
    model = (
        ModelFactory._create_vllm_model(cfg)
        if provider == "vllm"
        else ModelFactory._create_inference_router_model(cfg)
    )
    openai_client = model._provider.client
    openai_client._client._transport = httpx.MockTransport(_capture)
    await openai_client.chat.completions.create(
        model="local-llm", messages=[{"role": "user", "content": "hi"}]
    )
    return seen


def _strip_path(h: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in h.items() if k != "__path"}


@pytest.fixture
def bundle_token(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """The tenant's IR bundle token is mintable for this run."""
    calls: list[tuple[str, str | None]] = []

    async def _ir_authorization(url: str, tenant_id: str | None) -> str | None:
        calls.append((url, tenant_id))
        return "Bearer bundle-tok-for-tenant" if tenant_id == REAL_TENANT else None

    monkeypatch.setattr(ir_bundle_tokens, "ir_authorization", _ir_authorization)
    return calls


@pytest.mark.asyncio
async def test_vllm_wire_identity_equals_inference_router_with_bundle_token(bundle_token):
    identity = {"tenant_id": REAL_TENANT, "license_id": "lic-1", "subject_id": "subj-1"}
    ir = await _wire("inference_router", **identity)
    vllm = await _wire("vllm", **identity)

    # Base URL behaviour is unchanged: /v1 (pydantic_ai) vs /api (IR native).
    assert ir["__path"] == "/api/chat/completions"
    assert vllm["__path"] == "/v1/chat/completions"

    assert _strip_path(vllm) == _strip_path(ir)
    assert vllm["authorization"] == "Bearer bundle-tok-for-tenant"
    assert vllm["x-tenant-id"] == REAL_TENANT
    assert vllm["x-license-id"] == "lic-1"
    assert vllm["x-subject-id"] == "subj-1"
    # The bundle-token decision was asked for the tenant on BOTH paths.
    assert [t for _, t in bundle_token] == [REAL_TENANT, REAL_TENANT]


@pytest.mark.asyncio
async def test_vllm_never_sends_bearer_empty_when_no_bundle_token(monkeypatch):
    """Flag off / no token minted: both paths fall back to the SAME placeholder."""

    async def _none(url: str, tenant_id: str | None) -> None:
        return None

    monkeypatch.setattr(ir_bundle_tokens, "ir_authorization", _none)
    identity = {"tenant_id": REAL_TENANT, "license_id": "lic-1", "subject_id": "subj-1"}
    ir = await _wire("inference_router", **identity)
    vllm = await _wire("vllm", **identity)

    assert vllm["authorization"] != "Bearer EMPTY"
    assert _strip_path(vllm) == _strip_path(ir)


@pytest.mark.asyncio
async def test_missing_licence_fails_loud_identically(bundle_token, caplog):
    caplog.set_level(logging.WARNING)
    ir = await _wire("inference_router", tenant_id=REAL_TENANT)
    ir_logs = [r for r in caplog.records if "outbound_license_unresolved" in r.getMessage()]
    caplog.clear()
    vllm = await _wire("vllm", tenant_id=REAL_TENANT)
    vllm_logs = [r for r in caplog.records if "outbound_license_unresolved" in r.getMessage()]

    assert "x-license-id" not in vllm  # omitted, never substituted
    assert _strip_path(vllm) == _strip_path(ir)
    assert len(vllm_logs) == len(ir_logs) == 1
    assert vllm_logs[0].levelno == ir_logs[0].levelno == logging.WARNING


@pytest.mark.asyncio
async def test_missing_tenant_fails_loud_identically(bundle_token, caplog):
    caplog.set_level(logging.WARNING)
    ir = await _wire("inference_router", license_id="lic-1")
    ir_logs = [r for r in caplog.records if "outbound_tenant_unresolved" in r.getMessage()]
    caplog.clear()
    vllm = await _wire("vllm", license_id="lic-1")
    vllm_logs = [r for r in caplog.records if "outbound_tenant_unresolved" in r.getMessage()]

    assert "x-tenant-id" not in vllm
    # No tenant => no bundle token => the shared placeholder, on both paths.
    assert vllm["authorization"] == ir["authorization"] != "Bearer bundle-tok-for-tenant"
    assert _strip_path(vllm) == _strip_path(ir)
    assert len(vllm_logs) == len(ir_logs) == 1
    assert vllm_logs[0].levelno == ir_logs[0].levelno == logging.ERROR
