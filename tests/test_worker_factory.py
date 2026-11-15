"""Tests for worker factory -- MLX resolution, agent creation, basic workers.

Every test verifies that values *flow through* to the returned worker or
the config objects passed downstream, not just that a mock was called.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.worker.worker_template import WorkerTemplate

if TYPE_CHECKING:
    from src.worker.factory.agent_factory import AgentConfig, WorkerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_factory():
    """Return a context-manager stack that patches tool loading and MCP setup
    so we can exercise create_worker without network I/O."""
    return (
        patch(
            "src.worker.factory.worker_factory._load_worker_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.worker.factory.worker_factory._setup_mcp_servers_for_worker",
            new_callable=AsyncMock,
            return_value=[],
        ),
    )


# ---------------------------------------------------------------------------
# MLX Resolution
# ---------------------------------------------------------------------------


class TestMLXResolution:
    """Verify that the resolved base_url actually lands in the WorkerConfig
    passed to _create_agent_worker, and that non-MLX providers skip the
    registry entirely."""

    @pytest.mark.asyncio
    async def test_mlx_specific_model_resolved_url_flows_to_worker_config(self):
        """create_worker(provider='mlx', model='qwen3-1.7b') must pass
        the resolved URL into worker_config.base_url_override."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_reg.resolve_base_url = AsyncMock(return_value="https://mlx-inference.example.com")
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="test",
                tool_names=[],
                agent_role="Test role",
                model_provider="mlx",
                model="qwen3-1.7b",
            )

            # The URL must have flowed through to the first positional arg
            worker_cfg: WorkerConfig = mock_create.call_args[0][0]
            assert worker_cfg.base_url_override == "https://mlx-inference.example.com"

    @pytest.mark.asyncio
    async def test_mlx_registry_returns_none_uses_fallback_url(self):
        """When the MLX registry cannot resolve a model it returns None.
        The original base_url_override (fallback) must be kept."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_reg.resolve_base_url = AsyncMock(return_value=None)
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="fallback_test",
                tool_names=[],
                agent_role="Test",
                model_provider="mlx",
                model="unknown-model",
                base_url_override="http://custom-fallback:8080",
            )

            # resolve_base_url returned None, so the original override is kept
            worker_cfg: WorkerConfig = mock_create.call_args[0][0]
            assert worker_cfg.base_url_override == "http://custom-fallback:8080"

    @pytest.mark.asyncio
    async def test_deepseek_provider_skips_mlx_registry(self):
        """provider='deepseek' must NOT touch mlx_model_registry."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="ds_test",
                tool_names=[],
                agent_role="Test",
                model_provider="deepseek",
                model="deepseek-chat",
            )

            mock_reg.resolve_base_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_groq_provider_skips_mlx_registry(self):
        """provider='groq' must NOT touch mlx_model_registry."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="groq_test",
                tool_names=[],
                agent_role="Test",
                model_provider="groq",
                model="llama-3-8b",
            )

            mock_reg.resolve_base_url.assert_not_called()


# ---------------------------------------------------------------------------
# Agent creation flow
# ---------------------------------------------------------------------------


class TestAgentCreationFlow:
    """Verify that agent_type, system_prompt, name, and model flow
    through create_worker into the AgentConfig / WorkerConfig objects."""

    @pytest.mark.asyncio
    async def test_pydantic_ai_agent_config_flows_through(self):
        """is_agent=True (agent_role set), agent_type='pydantic_ai'.
        Verify name, model, system_prompt land in the configs."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_reg.resolve_base_url = AsyncMock(return_value=None)
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="agent_test",
                tool_names=["tool_a"],
                agent_role="Code reviewer",
                agent_type="pydantic_ai",
                model_provider="mlx",
                model="qwen3-1.7b",
                system_prompt="Review code carefully.",
                temperature=0.3,
            )

            worker_cfg: WorkerConfig = mock_create.call_args[0][0]
            agent_cfg: AgentConfig = mock_create.call_args[0][1]

            assert worker_cfg.worker_name == "agent_test"
            assert worker_cfg.tool_names == ["tool_a"]
            assert agent_cfg.agent_type == "pydantic_ai"
            assert agent_cfg.agent_role == "Code reviewer"
            assert agent_cfg.model_provider == "mlx"
            assert agent_cfg.model == "qwen3-1.7b"
            assert agent_cfg.system_prompt == "Review code carefully."
            assert agent_cfg.temperature == 0.3

    @pytest.mark.asyncio
    async def test_basic_worker_without_agent_role_returns_worker_template(self):
        """No agent_role -> create_worker returns a WorkerTemplate, not an agent."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            from src.worker.worker_factory import create_worker

            result = await create_worker(
                worker_name="basic_worker",
                tool_names=[],
                disable_state_storage=True,
            )

            # _create_agent_worker must NOT be called for basic workers
            mock_create.assert_not_called()
            # The result should be a WorkerTemplate
            assert isinstance(result, WorkerTemplate)
            assert result.name == "basic_worker"

    @pytest.mark.asyncio
    async def test_claude_code_agent_type_flows_through(self):
        """agent_type='claude_code' must arrive in the AgentConfig."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="cc_worker",
                tool_names=[],
                agent_role="Code agent",
                agent_type="claude_code",
            )

            agent_cfg: AgentConfig = mock_create.call_args[0][1]
            assert agent_cfg.agent_type == "claude_code"

    @pytest.mark.asyncio
    async def test_default_agent_type_is_pydantic_ai(self):
        """If agent_type is None, _create_worker_configs defaults it to 'pydantic_ai'."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch("src.worker.factory.worker_factory.mlx_model_registry") as mock_reg,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_reg.resolve_base_url = AsyncMock(return_value=None)
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="default_type",
                tool_names=[],
                agent_role="Test",
                # agent_type intentionally omitted
            )

            agent_cfg: AgentConfig = mock_create.call_args[0][1]
            assert agent_cfg.agent_type == "pydantic_ai"


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------


class TestToolLoading:
    """Verify that tools loaded by _load_worker_tools are passed to the
    downstream worker or agent creation."""

    @pytest.mark.asyncio
    async def test_loaded_tools_passed_to_basic_worker(self):
        """Tools from _load_worker_tools must appear in the returned
        WorkerTemplate.tools list."""
        fake_tool = MagicMock(__name__="fake_tool")
        with (
            patch(
                "src.worker.factory.worker_factory._load_worker_tools",
                new_callable=AsyncMock,
                return_value=[fake_tool],
            ),
            patch(
                "src.worker.factory.worker_factory._setup_mcp_servers_for_worker",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from src.worker.worker_factory import create_worker

            result = await create_worker(
                worker_name="tool_test",
                tool_names=["fake_tool"],
                disable_state_storage=True,
            )

            assert isinstance(result, WorkerTemplate)
            assert fake_tool in result.tools

    @pytest.mark.asyncio
    async def test_loaded_tools_passed_to_agent_worker(self):
        """Tools from _load_worker_tools must be forwarded to _create_agent_worker."""
        fake_tool_a = MagicMock(__name__="tool_a")
        fake_tool_b = MagicMock(__name__="tool_b")
        with (
            patch(
                "src.worker.factory.worker_factory._load_worker_tools",
                new_callable=AsyncMock,
                return_value=[fake_tool_a, fake_tool_b],
            ),
            patch(
                "src.worker.factory.worker_factory._setup_mcp_servers_for_worker",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="agent_tools",
                tool_names=["tool_a", "tool_b"],
                agent_role="Needs tools",
            )

            # Third positional arg is tools
            tools_arg = mock_create.call_args[0][2]
            assert fake_tool_a in tools_arg
            assert fake_tool_b in tools_arg


# ---------------------------------------------------------------------------
# enable_reasoning plumbing
# ---------------------------------------------------------------------------


class TestEnableReasoningPlumbing:
    """The chain-of-thought switch has to survive four hops -- config dict ->
    create_worker -> AgentConfig -> create_agent kwargs. Each hop is a place it
    can be silently dropped, and a dropped switch looks exactly like a switch
    that was never set: the model just keeps thinking."""

    @pytest.mark.asyncio
    async def test_create_worker_puts_enable_reasoning_on_agent_config(self):
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="reasoner",
                tool_names=[],
                agent_role="Test role",
                model_provider="vllm",
                model="qwen",
                enable_reasoning=False,
            )

            agent_cfg: AgentConfig = mock_create.call_args[0][1]
            assert agent_cfg.enable_reasoning is False

    @pytest.mark.asyncio
    async def test_create_worker_defaults_enable_reasoning_to_none(self):
        """Not passing it must leave None, not False -- False is a real
        instruction to the model, None means 'say nothing'."""
        p1, p2 = _patch_factory()
        with (
            p1,
            p2,
            patch(
                "src.worker.factory.worker_factory._create_agent_worker",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(db_path="")

            from src.worker.worker_factory import create_worker

            await create_worker(
                worker_name="plain",
                tool_names=[],
                agent_role="Test role",
                model_provider="vllm",
                model="qwen",
            )

            agent_cfg: AgentConfig = mock_create.call_args[0][1]
            assert agent_cfg.enable_reasoning is None

    @pytest.mark.asyncio
    async def test_agent_worker_forwards_enable_reasoning_to_create_agent(self):
        from src.worker.factory.agent_factory import (
            AgentConfig as _AgentConfig,
        )
        from src.worker.factory.agent_factory import (
            WorkerConfig as _WorkerConfig,
        )
        from src.worker.factory.agent_factory import (
            _create_agent_worker,
        )

        agent_config = _AgentConfig(
            agent_role="r",
            agent_type="pydantic_ai",
            model_provider="vllm",
            model="qwen",
            enable_reasoning=False,
        )
        worker_config = _WorkerConfig(worker_name="w", tool_names=[])

        with patch("src.worker.agents.agent_registry.create_agent") as mock_create_agent:
            mock_create_agent.return_value = MagicMock()
            await _create_agent_worker(worker_config, agent_config, [], "", [])

        assert mock_create_agent.call_args.kwargs["enable_reasoning"] is False

    @pytest.mark.asyncio
    async def test_agent_worker_omits_enable_reasoning_when_unset(self):
        """Omitted, not passed as None: the kwarg must not appear at all so
        create_pydantic_ai_agent's own default stands."""
        from src.worker.factory.agent_factory import (
            AgentConfig as _AgentConfig,
        )
        from src.worker.factory.agent_factory import (
            WorkerConfig as _WorkerConfig,
        )
        from src.worker.factory.agent_factory import (
            _create_agent_worker,
        )

        agent_config = _AgentConfig(
            agent_role="r",
            agent_type="pydantic_ai",
            model_provider="vllm",
            model="qwen",
        )
        worker_config = _WorkerConfig(worker_name="w", tool_names=[])

        with patch("src.worker.agents.agent_registry.create_agent") as mock_create_agent:
            mock_create_agent.return_value = MagicMock()
            await _create_agent_worker(worker_config, agent_config, [], "", [])

        assert "enable_reasoning" not in mock_create_agent.call_args.kwargs

    def test_create_pydantic_ai_agent_kwarg_reaches_the_config(self):
        """The last hop: the factory kwarg must land on PydanticAIAgentConfig."""
        from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

        agent = create_pydantic_ai_agent(
            name="reasoner",
            description="d",
            model_provider="vllm",
            model="qwen",
            base_url="http://vllm:8000",
            enable_reasoning=False,
        )
        assert agent.config.enable_reasoning is False
