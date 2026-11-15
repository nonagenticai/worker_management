"""The router-routed double-emit: worker-management must meter BYO only.

`cost_source` is the ONLY trigger for `emit_byo_cost_event_task`. The `local`
preset's provider is `inference_router`, which meters every call it serves, so
stamping `cost_source` on a `local`-preset run made worker_management write a
SECOND billing row for the same LLM call — flagged `byo=true` for tokens that
burned on owner hardware. Measured in prod 2026-08-26: 14,833 of 14,840
correlation-bearing `billing_cost_events` rows emitted under
`service_key='worker-management'`  # pragma: allowlist secret  (a service
NAME, not a credential; detect-secrets keyword-matches the `_key=` shape)
had an exact `inference-router` twin.

Two layers are asserted here, because the failure is expensive and silent:
  1. `resolve_llm_config` stamps `cost_source` only for an `is_external` preset;
  2. `emit_byo_cost_event_task` REFUSES a platform-routed `model_provider`,
     even if a future caller reopens layer 1.
Plus the `event_key` idempotency field billing needs to reject a redelivery.
"""

import json
import secrets
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.identity_context import TENANT_PROVENANCE_LICENSE, set_identity_context
from src.security.secret_box import encrypt_secret
from src.telemetry import cost_events
from src.worker.agents.provider_catalog import get_preset, provider_catalog
from src.worker.factory import llm_config_resolver as r

pytestmark = pytest.mark.asyncio

TENANT = "019edc1e-0000-7000-8000-000000000001"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))


@pytest.fixture(autouse=True)
def _proven_caller():
    """A licence-proven caller for ``TENANT``: the only one entitled to a
    decrypted BYO key (worker_management#245)."""
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_provenance=TENANT_PROVENANCE_LICENSE)
    yield
    set_identity_context(None, None, None)


class _FakeDB:
    def __init__(self, row):
        self._row = row

    async def get_config(self, tenant_id, role="__default__"):
        return self._row


def _row(**over):
    base = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "api_key_encrypted": encrypt_secret("sk-tenant-key"),
        "cost_source": "worker-management",
    }
    base.update(over)
    return base


def _stub_settings(url: str = "redis://metering:6379/0"):
    return SimpleNamespace(metering_redis_url=url, metering_costs_stream="improvement-loop:costs")


# --------------------------------------------------------------------------
# Layer 1 — the resolver does not arm the emit gate for router-routed presets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["local-llm", "default", "tier-medium", "deepseek-v4-pro"])
async def test_local_preset_never_stamps_cost_source(monkeypatch, model):
    """The `local` preset routes THROUGH inference-router, which already meters.

    Pre-fix this returned cost_source="worker-management" for every one of these
    models, arming a duplicate emit for a call billing already had a row for.
    """
    monkeypatch.setattr(
        r, "_db", _FakeDB(_row(provider="local", model=model, api_key_encrypted=None))
    )
    config = await r.resolve_llm_config({}, tenant_id=TENANT)

    assert config.get("cost_source") is None, (
        f"local/{model} armed the BYO emit gate — inference-router meters this "
        "call, so a second cost_event is a duplicate billing row"
    )


async def test_platform_only_local_row_also_stays_unarmed(monkeypatch):
    """platform_only drops EXTERNAL rows but lets a `local` row fall through."""
    monkeypatch.setattr(
        r, "_db", _FakeDB(_row(provider="local", model="local-llm", api_key_encrypted=None))
    )
    config = await r.resolve_llm_config({}, tenant_id=TENANT, byo_mode="platform_only")

    assert config.get("cost_source") is None


async def test_genuine_byo_preset_still_stamps_cost_source(monkeypatch):
    """The regression guard: a real off-platform call must STILL be metered.

    inference-router never sees this call, so worker_management is the only
    meter of record. Suppressing it would silently stop billing real spend.
    """
    monkeypatch.setattr(r, "_db", _FakeDB(_row()))
    config = await r.resolve_llm_config({}, tenant_id=TENANT)

    assert config["cost_source"] == r.BYO_COST_SOURCE
    assert config["model_provider"] == "anthropic"


@pytest.mark.parametrize("key", sorted(p["key"] for p in provider_catalog() if p["is_external"]))
async def test_every_external_preset_is_still_metered(monkeypatch, key):
    """Whole-catalog sweep: suppression must hit ONLY the non-external preset."""
    from src.settings import reset_settings_cache

    preset = get_preset(key)
    monkeypatch.setattr(
        r,
        "_db",
        _FakeDB(
            _row(
                provider=key,
                model=(preset["models"] or ["gpt-4o-mini"])[0],
                base_url="https://byo.example/v1" if preset["requires_base_url"] else None,
            )
        ),
    )
    # A BYO endpoint must be on the tenant's allowlist (request_overrides).
    monkeypatch.setenv("TENANT_BASE_URL_ALLOWLIST", f"{TENANT}=https://byo.example")
    reset_settings_cache()
    try:
        config = await r.resolve_llm_config({}, tenant_id=TENANT)
    finally:
        monkeypatch.delenv("TENANT_BASE_URL_ALLOWLIST")
        reset_settings_cache()

    assert config["cost_source"] == r.BYO_COST_SOURCE, f"{key} lost its meter"


# --------------------------------------------------------------------------
# Layer 2 — the emitter refuses a platform-routed provider outright
# --------------------------------------------------------------------------


# Spelled out rather than read off the module, so this file still COLLECTS
# against the pre-fix tree — a collection abort would report as one red error
# instead of the seven behavioural failures that prove the defect.
PLATFORM_ROUTED = ["inference_router", "mlx", "vllm", "vlm_local"]


async def test_platform_routed_set_is_the_one_the_emitter_uses():
    assert set(PLATFORM_ROUTED) == set(cost_events.PLATFORM_ROUTED_PROVIDERS)


@pytest.mark.parametrize("model_provider", PLATFORM_ROUTED)
async def test_emitter_refuses_platform_routed_provider(monkeypatch, caplog, model_provider):
    """A gate that can fail: refuse, and say so at ERROR."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)
    monkeypatch.setattr(cost_events, "get_settings", _stub_settings)
    monkeypatch.setattr(
        cost_events, "get_tenant_id_from_context", lambda: "019edc1e-0000-7000-8000-000000000001"
    )

    with caplog.at_level("ERROR"):
        cost_events.emit_byo_cost_event_task(
            provider="pgx-local",
            model="tier-medium",
            served_model="local-llm",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
            model_provider=model_provider,
        )

    fake_client.xadd.assert_not_awaited()
    assert "byo_cost_event_refused_platform_routed" in caplog.text


async def test_emitter_still_emits_for_genuine_byo(monkeypatch):
    """The mirror-image guard: a direct third-party call is still metered."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)
    monkeypatch.setattr(cost_events, "get_settings", _stub_settings)
    monkeypatch.setattr(
        cost_events, "get_tenant_id_from_context", lambda: "019edc1e-0000-7000-8000-000000000001"
    )
    monkeypatch.setattr(cost_events, "get_license_id_from_context", lambda: "l-1")
    monkeypatch.setattr(cost_events, "get_subject_id_from_context", lambda: "s-1")
    monkeypatch.setattr(cost_events, "get_correlation_id_from_context", lambda: "")
    monkeypatch.setattr(cost_events, "get_stage_from_context", lambda: "")

    cost_events.emit_byo_cost_event_task(
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        latency_ms=1000,
        model_provider="anthropic",
    )
    for task in list(cost_events._background_tasks):
        await task

    fake_client.xadd.assert_awaited_once()
    fields = fake_client.xadd.call_args[0][1]
    assert fields["provider"] == "anthropic"
    assert json.loads(fields["usage_extra"])["byo"] == "true"


# --------------------------------------------------------------------------
# The idempotency key billing enforces uniqueness on
# --------------------------------------------------------------------------


async def test_every_event_carries_a_unique_parseable_event_key(monkeypatch):
    """Without a key, an at-least-once redelivery is an unrejectable duplicate."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    keys = []
    for _ in range(2):
        await cost_events._emit(
            stream="improvement-loop:costs",
            tenant_id=TENANT,
            license_id="l-1",
            subject_id="s-1",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
        )
        keys.append(fake_client.xadd.call_args[0][1]["event_key"])

    for k in keys:
        assert uuid.UUID(k).version == 7, "event_key must be a uuid7 (time-ordered)"
    assert keys[0] != keys[1], "two distinct calls must not share an idempotency key"


async def test_event_key_is_stable_across_the_redeliveries_of_one_call(monkeypatch):
    """One call ⇒ one key. A redelivery of that entry replays the SAME key."""
    fake_client = AsyncMock()
    monkeypatch.setattr(cost_events, "_get_redis", lambda: fake_client)

    for _ in range(3):
        await cost_events._emit(
            stream="improvement-loop:costs",
            tenant_id=TENANT,
            license_id="l-1",
            subject_id="s-1",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
            event_key="0199aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
        )
    seen = {c[0][1]["event_key"] for c in fake_client.xadd.call_args_list}
    assert seen == {"0199aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee"}


async def test_tenant_key_route_carries_a_key_fingerprint_never_the_key(monkeypatch):
    """Per-call BYO key audit: the resolver stamps the stored key VERSION's
    fingerprint (sha256 prefix of the encrypted blob) and the row role."""
    row = _row(role="tier:high")
    monkeypatch.setattr(r, "_db", _FakeDB(row))
    config = await r.resolve_llm_config({}, tenant_id=TENANT)

    assert config["tenant_byo_key"] is True
    assert config["key_fingerprint"] == cost_events.key_fingerprint(row["api_key_encrypted"])
    assert len(config["key_fingerprint"]) == 16
    assert "sk-tenant-key" not in config["key_fingerprint"]
    assert config["byo_role"] == "tier:high"


async def test_keyless_local_route_has_no_key_fingerprint(monkeypatch):
    monkeypatch.setattr(
        r, "_db", _FakeDB(_row(provider="local", model="local-llm", api_key_encrypted=None))
    )
    config = await r.resolve_llm_config({}, tenant_id=TENANT)
    assert "key_fingerprint" not in config
