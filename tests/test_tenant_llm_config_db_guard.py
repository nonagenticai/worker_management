"""The paid-upstream guard on the ONLY function that can write tenant_llm_config.

The API layer checks the same rule and turns it into a 422 — that is the
tenant-facing half, and tests/test_tenant_config_endpoints.py covers it. This
file covers why the check is ALSO in the DB layer: `TenantLLMConfigDB.upsert_config`
is the only function in the process that writes the table, so a check here is
one no script, shell, or future caller can route around. The endpoint tests
cannot prove that — they swap the whole DB object for an in-memory fake.

⭐ Two rows written past this seam on 2026-07-19 — provider='local',
model='tier-medium', roles tier:medium and tier:high — spent $140.31 over
46,102 requests on paid DeepSeek for a tenant whose configuration said
"platform (local inference)". Nobody ever named DeepSeek.
See migrations/025_purge_local_preset_paid_tier_rows.sql.
"""

from __future__ import annotations

import pytest

from src.db.tenant_llm_config_db import TenantLLMConfigDB
from src.worker.agents.provider_catalog import PaidUpstreamForLocalPresetError

TENANT = "0195f48b-d000-7000-8000-000000000001"


def _row(**over):
    row = {
        "tenant_id": TENANT,
        "role": "tier:medium",
        "provider": "local",
        "model": "tier-medium",
        "base_url": None,
        "temperature": None,
        "max_tokens": None,
        "params": {},
        "api_key_encrypted": None,
    }
    row.update(over)
    return row


@pytest.mark.asyncio
async def test_upsert_refuses_the_shape_that_cost_140_dollars(monkeypatch):
    # The guard must fire BEFORE a connection is taken. If it fired after, a
    # refused write would still hold a pool slot, and — worse — anyone reading
    # the code would have to prove the INSERT had not already run.
    def _boom(*_a, **_k):
        raise AssertionError("a refused write must never reach the database")

    monkeypatch.setattr("src.db.tenant_llm_config_db.get_pg_connection", _boom)

    with pytest.raises(PaidUpstreamForLocalPresetError) as exc:
        await TenantLLMConfigDB().upsert_config(**_row())
    assert "deepseek" in str(exc.value)


@pytest.mark.asyncio
async def test_upsert_refuses_the_high_tier_row_as_well(monkeypatch):
    # Both offending rows carried model='tier-medium'; only the ROLE differed.
    # The rule reads provider+model and is deliberately role-blind, so it cannot
    # be evaded by writing the same pair under another tier.
    monkeypatch.setattr(
        "src.db.tenant_llm_config_db.get_pg_connection",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not connect")),
    )
    with pytest.raises(PaidUpstreamForLocalPresetError):
        await TenantLLMConfigDB().upsert_config(**_row(role="tier:high"))


@pytest.mark.asyncio
async def test_upsert_lets_a_legitimate_local_row_through(monkeypatch):
    # THE OTHER HALF OF THE GATE. Reaching the connection is the proof the guard
    # allowed the write — this stops the guard being "correct" by refusing
    # everything, which the two tests above cannot detect on their own.
    reached: dict[str, object] = {}

    class _Conn:
        async def execute(self, _sql, *args):
            reached["args"] = args

    class _Ctx:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_a):
            return False

    monkeypatch.setattr("src.db.tenant_llm_config_db.get_pg_connection", lambda *_a, **_k: _Ctx())

    await TenantLLMConfigDB().upsert_config(**_row(model="local-llm"))
    assert reached["args"][3] == "local-llm"


@pytest.mark.asyncio
async def test_upsert_lets_a_deliberate_external_paid_row_through(monkeypatch):
    # Selecting the deepseek preset with your own key is a considered purchase.
    # The guard is about a row that CLAIMS local while billing a paid API, not
    # about paid providers.
    reached: dict[str, object] = {}

    class _Conn:
        async def execute(self, _sql, *args):
            reached["args"] = args

    class _Ctx:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_a):
            return False

    monkeypatch.setattr("src.db.tenant_llm_config_db.get_pg_connection", lambda *_a, **_k: _Ctx())

    await TenantLLMConfigDB().upsert_config(
        **_row(role="__default__", provider="deepseek", model="deepseek-v4-pro")
    )
    assert reached["args"][2] == "deepseek"
