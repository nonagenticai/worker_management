"""The cost_source/cost_provider stamp must be loud when it fails.

``_create_worker_instance`` stamps the BYO routing decision onto the agent config
so ``run()`` can emit a priceable cost_event. That stamp used to fail into
``logger.debug`` — invisible in prod (DEBUG off) — and the message did not even
name ``cost_provider``. A silent failure here means a NULL ``cost_usd`` forever
(billing prices exactly once, at flush time). See issue #137.
"""

import logging

import pytest

from src.api.workers import _globals


class _FrozenConfig:
    """A config that rejects the stamp, as a slots/frozen agent config would."""

    __slots__ = ()


class _Worker:
    def __init__(self):
        self.config = _FrozenConfig()


@pytest.mark.asyncio
async def test_failed_cost_stamp_warns_and_names_both_fields(monkeypatch, caplog):
    async def _fake_resolve(config, tenant_id, *, byo_mode, work_tier, key_entitled):
        return {
            **config,
            "agent_type": "pydantic_ai",
            "cost_source": "byo",
            "cost_provider": "deepseek",
        }

    async def _fake_create_worker(*args, **kwargs):
        return _Worker()

    monkeypatch.setattr(_globals, "resolve_llm_config", _fake_resolve)
    monkeypatch.setattr(_globals, "create_worker", _fake_create_worker)

    cache = object.__new__(_globals.WorkerInstanceCache)
    with caplog.at_level(logging.WARNING, logger=_globals.logger.name):
        worker = await cache._create_worker_instance(
            "project_classifier", {"worker_name": "project_classifier", "config": {}}
        )

    assert isinstance(worker, _Worker)
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert records, "a failed cost stamp must be logged at WARNING, not DEBUG"
    message = records[-1].getMessage()
    assert "cost_source" in message
    assert "cost_provider" in message
    # The values must be in the message, not just the field names.
    assert "byo" in message and "deepseek" in message
