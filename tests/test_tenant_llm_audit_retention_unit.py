"""WM-5 unit: the audit prune is wired into the lifespan and bounded by settings."""

from __future__ import annotations

import asyncio

import pytest

from src.services import tenant_llm_audit_retention as ar


def test_disabled_returns_none(monkeypatch):
    from src.settings import reset_settings_cache

    monkeypatch.setenv("TENANT_LLM_AUDIT_RETENTION_ENABLED", "false")
    reset_settings_cache()
    try:
        assert ar.start_audit_retention_task() is None
    finally:
        reset_settings_cache()


def test_enabled_starts_the_loop_with_the_prune(monkeypatch):
    from src.services import worker_states_retention as wsr
    from src.settings import reset_settings_cache

    monkeypatch.setenv("TENANT_LLM_AUDIT_RETENTION_ENABLED", "true")
    monkeypatch.setenv("TENANT_LLM_AUDIT_RETENTION_INTERVAL_SECONDS", "3600")
    reset_settings_cache()
    seen: dict[str, object] = {}

    async def fake_loop(**kw: object) -> None:
        seen.update(kw)

    monkeypatch.setattr(wsr, "retention_loop", fake_loop)

    async def go() -> None:
        task = ar.start_audit_retention_task()
        assert task is not None
        await task

    try:
        asyncio.run(go())
    finally:
        reset_settings_cache()
    assert seen == {
        "sweep": ar.prune_audit_once,
        "interval_seconds": 3600,
        "label": "tenant_llm_config_audit",
    }


def test_retention_floor_refuses_a_tiny_window(monkeypatch):
    from pydantic import ValidationError

    from src.settings import Settings

    monkeypatch.setenv("TENANT_LLM_AUDIT_RETENTION_DAYS", "1")
    with pytest.raises(ValidationError):
        Settings()


def test_lifespan_starts_the_audit_prune():
    import inspect

    from src.api import worker_management_api as api

    src = inspect.getsource(api)
    assert "start_audit_retention_task()" in src
    assert "audit_retention_task" in src.split("for task in (", 1)[1].split(")", 1)[0]
