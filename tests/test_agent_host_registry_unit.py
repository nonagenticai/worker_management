"""Unit tests for services/agent_host_registry.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.agent_host_registry import AgentHostInstance, AgentHostRegistry


class TestAgentHostInstance:
    def test_defaults(self):
        inst = AgentHostInstance(name="mini1", url="http://localhost:9100")
        assert inst.healthy is False
        assert inst.active_containers == 0
        assert inst.max_containers == 10


class TestCheckInstanceHealth:
    @pytest.mark.asyncio
    async def test_healthy(self):
        registry = AgentHostRegistry()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "active_containers": 3,
            "max_containers": 10,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            instance = await registry.check_instance_health("mini1", "http://localhost:9100")
            assert instance.healthy is True
            assert instance.active_containers == 3

    @pytest.mark.asyncio
    async def test_unhealthy_status_code(self):
        registry = AgentHostRegistry()
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            instance = await registry.check_instance_health("mini1", "http://localhost:9100")
            assert instance.healthy is False

    @pytest.mark.asyncio
    async def test_connect_error(self):
        registry = AgentHostRegistry()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            instance = await registry.check_instance_health("mini1", "http://localhost:9100")
            assert instance.healthy is False

    @pytest.mark.asyncio
    async def test_timeout(self):
        registry = AgentHostRegistry()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            instance = await registry.check_instance_health("mini1", "http://localhost:9100")
            assert instance.healthy is False


class TestGetBestInstance:
    @pytest.mark.asyncio
    async def test_no_healthy_instances(self):
        registry = AgentHostRegistry()
        with patch.object(registry, "get_healthy_instances", new=AsyncMock(return_value=[])):
            result = await registry.get_best_instance()
            assert result is None

    @pytest.mark.asyncio
    async def test_selects_least_loaded(self):
        registry = AgentHostRegistry()
        i1 = AgentHostInstance(
            name="mini1", url="http://m1:9100", healthy=True, active_containers=5
        )
        i2 = AgentHostInstance(
            name="mini2", url="http://m2:9100", healthy=True, active_containers=1
        )
        with patch.object(registry, "get_healthy_instances", new=AsyncMock(return_value=[i1, i2])):
            result = await registry.get_best_instance()
            assert result is not None
            # First call with round-robin idx 0 should get i2 (sorted by least loaded)
            assert result.name == "mini2"


class TestRefresh:
    @pytest.mark.asyncio
    async def test_concurrent_refresh(self):
        registry = AgentHostRegistry()
        mock_health = AsyncMock(
            return_value=AgentHostInstance(name="test", url="http://test:9100", healthy=True)
        )
        with (
            patch.object(registry, "check_instance_health", new=mock_health),
            patch(
                "src.services.agent_host_registry._build_agent_host_instances",
                return_value={"mini1": "http://m1:9100"},
            ),
        ):
            results = await registry.refresh()
            assert len(results) == 1
