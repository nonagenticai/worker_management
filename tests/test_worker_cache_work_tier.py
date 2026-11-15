"""The worker-instance cache must key by work_tier and pass it to the resolver.

A tenant's per-tier BYO row bakes a distinct provider/model/key into the agent,
so two messages at different tiers must resolve + cache as distinct instances.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.api.workers._globals import WorkerInstanceCache, _cache_key

TENANT = "019edc1e-0000-7000-8000-000000000001"
_AGENT_CONFIG = {
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": "local-llm",
}


def test_cache_key_includes_tier():
    assert _cache_key("w", TENANT, "high") == f"{TENANT}:w:high"
    assert _cache_key("w", TENANT) == f"{TENANT}:w:_"  # default tier sentinel
    assert _cache_key("w", None) == "w"  # no tenant -> name only


def test_cache_key_includes_product():
    # A product is appended only when set; the no-product key is byte-identical
    # to the pre-product form (backward compatible).
    assert _cache_key("w", TENANT, "high", "kag-graph") == f"{TENANT}:w:high:kag-graph"
    assert _cache_key("w", TENANT, "high", None) == f"{TENANT}:w:high"
    assert _cache_key("w", TENANT, None, "kag-graph") == f"{TENANT}:w:_:kag-graph"
    assert _cache_key("w", None, "high", "kag-graph") == "w"  # no tenant -> name only


def test_cache_key_includes_override_key():
    # A per-message override fingerprint is appended only when set; the no-override
    # key is byte-identical to the pre-override form (backward compatible).
    assert _cache_key("w", TENANT, "high", None, "abc123") == f"{TENANT}:w:high:ov-abc123"
    assert _cache_key("w", TENANT, "high", "kag", "abc123") == f"{TENANT}:w:high:kag:ov-abc123"
    assert _cache_key("w", TENANT, "high", None, None) == f"{TENANT}:w:high"
    assert _cache_key("w", None, "high", None, "abc123") == "w"  # no tenant -> name only


@pytest.mark.asyncio
async def test_override_key_isolates_from_base_slot():
    """Regression: a per-message override (output_schema/model_override/max_tokens)
    must key under its OWN slot, so it never poisons the shared base instance that a
    later NON-override caller reuses (the cross-caller schema/model/budget leak)."""
    cache = WorkerInstanceCache()
    ov_obj, base_obj = object(), object()
    seq = iter([ov_obj, base_obj])
    with (
        patch(
            "src.api.workers._globals.create_worker",
            new=AsyncMock(side_effect=lambda **_kw: next(seq)),
        ),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        w_ov = await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
            override_key="fp1",
        )
        w_base = await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
        )

    assert f"{TENANT}:w:_:ov-fp1" in cache._workers  # override got its own slot
    assert f"{TENANT}:w:_" in cache._workers  # base slot is separate
    assert w_ov is ov_obj
    assert w_base is base_obj
    assert w_ov is not w_base  # the override instance never leaked into the base caller


@pytest.mark.asyncio
async def test_work_tier_threaded_to_resolver_and_keyed():
    worker_data = {"worker_name": "w", "config": {**_AGENT_CONFIG, "work_tier": "high"}}
    cache = WorkerInstanceCache()
    with (
        patch("src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ) as mock_resolve,
    ):
        await cache.get_or_create_worker("w", worker_data, tenant_id=TENANT)

    # The resolver was called with the per-message tier.
    assert mock_resolve.call_args.kwargs["work_tier"] == "high"
    # And the instance is cached under the tier-qualified key.
    assert f"{TENANT}:w:high" in cache._workers


@pytest.mark.asyncio
async def test_distinct_tiers_create_distinct_instances():
    cache = WorkerInstanceCache()
    with (
        patch(
            "src.api.workers._globals.create_worker",
            new=AsyncMock(side_effect=lambda **_kw: object()),
        ),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": {**_AGENT_CONFIG, "work_tier": "small"}},
            tenant_id=TENANT,
        )
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": {**_AGENT_CONFIG, "work_tier": "high"}},
            tenant_id=TENANT,
        )

    assert {f"{TENANT}:w:small", f"{TENANT}:w:high"} <= set(cache._workers)

    # remove_worker(tenant) evicts every tier variant.
    await cache.remove_worker("w", tenant_id=TENANT)
    assert not [k for k in cache._workers if k.startswith(f"{TENANT}:w:")]


@pytest.mark.asyncio
async def test_product_threaded_to_create_instance_and_keyed():
    """A product supplied to get_or_create_worker reaches _create_worker_instance
    and qualifies the cache key, so a per-(tenant, product) BYO posture can't leak
    across products."""
    worker_data = {"worker_name": "w", "config": dict(_AGENT_CONFIG)}
    cache = WorkerInstanceCache()
    with (
        patch("src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker(
            "w", worker_data, tenant_id=TENANT, byo_mode="byo_required", product="kag-graph"
        )

    assert f"{TENANT}:w:_:kag-graph" in cache._workers


@pytest.mark.asyncio
async def test_distinct_products_create_distinct_instances():
    """Same (tenant, tier) but different products must NOT share an instance:
    product A may be byo_required (BYO key baked in) while B is local_allowed."""
    cache = WorkerInstanceCache()
    with (
        patch(
            "src.api.workers._globals.create_worker",
            new=AsyncMock(side_effect=lambda **_kw: object()),
        ),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
            product="prod-a",
        )
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
            product="prod-b",
        )
        # Same product reuses (no third instance created).
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
            product="prod-a",
        )

    assert {f"{TENANT}:w:_:prod-a", f"{TENANT}:w:_:prod-b"} <= set(cache._workers)
    assert len([k for k in cache._workers if k.startswith(f"{TENANT}:w:")]) == 2

    # remove_worker(tenant) evicts every product variant too (prefix match).
    await cache.remove_worker("w", tenant_id=TENANT)
    assert not [k for k in cache._workers if k.startswith(f"{TENANT}:w:")]


@pytest.mark.asyncio
async def test_config_max_tokens_passed_to_factory():
    """A per-message max_tokens folded into config (message_endpoints) must reach
    create_worker, so the rebuilt agent applies the caller's budget — the path
    that lets requirements_management's curated budget work under the end-user
    tenant's bare auto-created row."""
    worker_data = {"worker_name": "w", "config": {**_AGENT_CONFIG, "max_tokens": 1500}}
    cache = WorkerInstanceCache()
    with (
        patch(
            "src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())
        ) as mock_create,
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker("w", worker_data, tenant_id=TENANT)

    assert mock_create.call_args.kwargs["max_tokens"] == 1500


@pytest.mark.asyncio
async def test_enable_reasoning_read_from_config_and_passed_to_create_worker():
    """The cache is the hop where the per-message switch leaves the config dict.

    message_endpoints folds ``enable_reasoning`` into ``worker_data['config']``; if
    the cache never reads it back out, the whole chain downstream is inert while
    every other layer still looks correct.
    """
    cache = WorkerInstanceCache()
    with (
        patch("src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": {**_AGENT_CONFIG, "enable_reasoning": False}},
            tenant_id=TENANT,
            override_key="fp-reasoning",
        )
        from src.api.workers._globals import create_worker as patched

        assert patched.call_args.kwargs["enable_reasoning"] is False


@pytest.mark.asyncio
async def test_enable_reasoning_absent_from_config_passes_none():
    """Absent must stay None, never False: False is a real instruction."""
    cache = WorkerInstanceCache()
    with (
        patch("src.api.workers._globals.create_worker", new=AsyncMock(return_value=object())),
        patch(
            "src.api.workers._globals.resolve_llm_config",
            new=AsyncMock(side_effect=lambda cfg, _tid, **_kw: cfg),
        ),
    ):
        await cache.get_or_create_worker(
            "w",
            {"worker_name": "w", "config": dict(_AGENT_CONFIG)},
            tenant_id=TENANT,
        )
        from src.api.workers._globals import create_worker as patched

        assert patched.call_args.kwargs["enable_reasoning"] is None
