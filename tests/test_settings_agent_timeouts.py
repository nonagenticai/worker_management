"""Agent-run wall-clock budget: configurability + the ordering invariant.

Two things are pinned here.

1. ``agent_run_timeout_s`` / ``agent_run_timeout_mcp_s`` are Settings fields
   bound to bare env vars, hoisted out of module-level constants in
   ``pydantic_ai_agent.py``. Their VALUES are unchanged (600.0 / 900.0) — the
   point of the move was to make a future re-baseline an env change, not to
   perform one. ``test_defaults_are_unchanged`` is the tripwire for that.

2. The load-bearing ordering invariant::

       ir_http_read_timeout_s <= min(agent_run_timeout_s, agent_run_timeout_mcp_s)

   The agent cap's ``asyncio.timeout`` starts before connect, so while it is
   >= the downstream read budget it always fires first and the OpenAI SDK's
   retry loop never gets a turn. Today 600.0 == 600.0 and the SDK's retry
   window is exactly zero-width — which is the only reason its default
   ``max_retries=2`` has not been tripling GPU load. That equality was
   accidental and undocumented. Inverting it (lowering the read timeout below
   the agent cap is precisely what the next tuning pass wants) makes every
   read timeout cost two extra full decodes on a provider that is already
   saturated. ``Settings._validate_timeout_ordering`` now refuses to start.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from src.settings import Settings, reset_settings_cache

if TYPE_CHECKING:
    from collections.abc import Iterator

_ENV_KEYS = (
    "AGENT_RUN_TIMEOUT_S",
    "AGENT_RUN_TIMEOUT_MCP_S",
    "IR_HTTP_READ_TIMEOUT_S",
    "ENVIRONMENT",
)


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """These tests mutate env vars; keep them isolated and drop the cache."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    reset_settings_cache()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_settings_cache()


def _mk(**overrides: Any) -> Settings:
    # model_validate, not dict-kwarg unpack — the latter is rejected by ty CI.
    return Settings.model_validate(dict(overrides))


class TestDefaultsUnchanged:
    def test_defaults_are_unchanged(self) -> None:
        """This change makes the budget configurable; it does NOT re-baseline it.

        If a later commit wants 600 -> 180, it edits these numbers deliberately
        and updates this test. An incidental drift fails here.
        """
        s = _mk()
        assert s.agent_run_timeout_s == 600.0
        assert s.agent_run_timeout_mcp_s == 900.0
        assert s.ir_http_read_timeout_s == 600.0

    def test_shipped_defaults_satisfy_the_invariant(self) -> None:
        """The guard must not fire on what we actually ship."""
        s = _mk()
        assert s.ir_http_read_timeout_s <= min(s.agent_run_timeout_s, s.agent_run_timeout_mcp_s)

    def test_mcp_budget_is_the_more_generous_one(self) -> None:
        s = _mk()
        assert s.agent_run_timeout_mcp_s > s.agent_run_timeout_s


class TestEnvVarBinding:
    def test_agent_run_timeout_s_reads_env(self) -> None:
        os.environ["AGENT_RUN_TIMEOUT_S"] = "321.5"
        # Keep the invariant satisfiable while overriding only the one field.
        os.environ["IR_HTTP_READ_TIMEOUT_S"] = "300"
        assert Settings().agent_run_timeout_s == 321.5

    def test_agent_run_timeout_mcp_s_reads_env(self) -> None:
        os.environ["AGENT_RUN_TIMEOUT_MCP_S"] = "1200"
        assert Settings().agent_run_timeout_mcp_s == 1200.0

    def test_env_names_are_case_insensitive(self) -> None:
        """Settings sets case_sensitive=False; deployments use SHOUTY names."""
        # SIM112 is suppressed deliberately: the lower-case spelling IS the
        # assertion here — this test exists to prove case_sensitive=False holds
        # for the new fields, so upper-casing it would delete the test's point.
        os.environ["agent_run_timeout_mcp_s"] = "1111"  # noqa: SIM112
        try:
            assert Settings().agent_run_timeout_mcp_s == 1111.0
        finally:
            os.environ.pop("agent_run_timeout_mcp_s", None)

    def test_both_budgets_configurable_together(self) -> None:
        os.environ["IR_HTTP_READ_TIMEOUT_S"] = "180"
        os.environ["AGENT_RUN_TIMEOUT_S"] = "200"
        os.environ["AGENT_RUN_TIMEOUT_MCP_S"] = "400"
        s = Settings()
        assert (s.ir_http_read_timeout_s, s.agent_run_timeout_s, s.agent_run_timeout_mcp_s) == (
            180.0,
            200.0,
            400.0,
        )


class TestOrderingInvariantGuard:
    def test_read_timeout_above_agent_cap_is_refused(self) -> None:
        """The exact re-baseline mistake: agent cap lowered, read budget left high."""
        with pytest.raises(RuntimeError, match="agent-timeout invariant"):
            _mk(ir_http_read_timeout_s=600.0, agent_run_timeout_s=300.0)

    def test_read_timeout_above_mcp_cap_is_refused(self) -> None:
        """min() covers the MCP budget too — an MCP agent uses that cap."""
        with pytest.raises(RuntimeError, match="agent-timeout invariant"):
            _mk(
                ir_http_read_timeout_s=600.0,
                agent_run_timeout_s=900.0,
                agent_run_timeout_mcp_s=300.0,
            )

    def test_equality_is_allowed(self) -> None:
        """Equality is the CURRENT production state and must stay legal — the
        agent cap starts its clock before connect, so at equality it still
        fires first and the SDK retry window stays zero-width."""
        s = _mk(
            ir_http_read_timeout_s=450.0, agent_run_timeout_s=450.0, agent_run_timeout_mcp_s=450.0
        )
        assert s.agent_run_timeout_s == 450.0

    def test_agent_cap_above_read_budget_is_allowed(self) -> None:
        s = _mk(ir_http_read_timeout_s=120.0, agent_run_timeout_s=600.0)
        assert s.ir_http_read_timeout_s == 120.0

    def test_guard_fires_outside_production(self) -> None:
        """Deliberately NOT gated on environment=production, unlike the
        required-infra check. A dev/staging value that violates the ordering is
        exactly the value that later gets promoted, and the failure mode it
        produces (GPU-load amplification) only shows up under real load."""
        for env in ("development", "staging", "test"):
            with pytest.raises(RuntimeError, match="agent-timeout invariant"):
                _mk(
                    environment=env,
                    ir_http_read_timeout_s=600.0,
                    agent_run_timeout_s=60.0,
                )

    def test_guard_fires_in_production_too(self) -> None:
        with pytest.raises(RuntimeError, match="agent-timeout invariant"):
            _mk(environment="production", ir_http_read_timeout_s=600.0, agent_run_timeout_s=60.0)

    def test_guard_fires_from_env_vars_not_just_kwargs(self) -> None:
        """The realistic path is a Deployment env var, not a constructor kwarg."""
        os.environ["AGENT_RUN_TIMEOUT_S"] = "60"
        with pytest.raises(RuntimeError, match="agent-timeout invariant"):
            Settings()

    def test_error_names_the_offending_values(self) -> None:
        """An operator reading this in a CrashLoopBackOff needs the numbers."""
        with pytest.raises(RuntimeError) as exc:
            _mk(ir_http_read_timeout_s=600.0, agent_run_timeout_s=90.0)
        msg = str(exc.value)
        assert "ir_http_read_timeout_s" in msg
        assert "600.0" in msg
        assert "90.0" in msg
        assert "agent_run_timeout_s" in msg


class TestAgentConsumesTheSettings:
    """``PydanticAIAgent._agent_run_timeout`` must read Settings at call time.

    Called unbound against a stub carrying only ``.config.mcp_servers`` — that
    is the entire input the real method reads, and it avoids standing up a
    model/provider just to check which number comes back.
    """

    @staticmethod
    def _timeout_for(*, mcp: bool) -> float:
        from types import SimpleNamespace

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        stub = SimpleNamespace(config=SimpleNamespace(mcp_servers=["x"] if mcp else []))
        return PydanticAIAgent._agent_run_timeout(stub)  # type: ignore[arg-type]

    def test_non_mcp_agent_uses_agent_run_timeout_s(self) -> None:
        assert self._timeout_for(mcp=False) == 600.0

    def test_mcp_agent_uses_the_mcp_budget(self) -> None:
        assert self._timeout_for(mcp=True) == 900.0

    def test_non_mcp_budget_honours_env_override(self) -> None:
        """Proves the value is read through Settings at call time rather than
        frozen into a module constant at import time."""
        os.environ["IR_HTTP_READ_TIMEOUT_S"] = "100"
        os.environ["AGENT_RUN_TIMEOUT_S"] = "123"
        reset_settings_cache()
        assert self._timeout_for(mcp=False) == 123.0

    def test_mcp_budget_honours_env_override(self) -> None:
        os.environ["AGENT_RUN_TIMEOUT_MCP_S"] = "777"
        reset_settings_cache()
        assert self._timeout_for(mcp=True) == 777.0

    def test_returned_budget_never_undercuts_the_read_timeout(self) -> None:
        """The property the invariant exists to protect, asserted end-to-end
        against the shipped defaults."""
        from src.settings import get_settings

        read_budget = get_settings().ir_http_read_timeout_s
        assert self._timeout_for(mcp=False) >= read_budget
        assert self._timeout_for(mcp=True) >= read_budget
