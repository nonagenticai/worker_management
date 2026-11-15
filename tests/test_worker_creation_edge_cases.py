"""Unit tests for edge cases in worker creation, especially MCP server handling."""

import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.api.workers import WorkerCreateRequest
from src.worker.worker_factory import (
    _handle_server_filtering,
    create_worker,
    resolve_mcp_config,
)

# Test creator ID (UUID format)
TEST_CREATOR_ID = "550e8400-e29b-41d4-a716-446655440000"


class TestMCPServerHandling:
    """Test cases for MCP server configuration handling."""

    def test_handle_server_filtering_no_config_files(self, tmp_path, monkeypatch):
        """Test _handle_server_filtering when no config files exist."""
        # Change to temp directory so no config files are found
        monkeypatch.chdir(tmp_path)

        result = _handle_server_filtering(["filesystem", "memory"], "test_worker")

        assert result is None

    def test_handle_server_filtering_with_config(self, tmp_path, monkeypatch):
        """Test _handle_server_filtering with a valid config file."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config directory and file
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()

        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": ["--path", "/tmp"],
                },
                "memory": {"command": "mcp-server-memory", "args": []},
                "database": {
                    "command": "mcp-server-db",
                    "args": ["--connection", "sqlite:///test.db"],
                },
            }
        }

        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(config_data))

        # Test filtering existing servers
        result = _handle_server_filtering(["filesystem", "memory"], "test_worker")

        assert result is not None
        assert "mcpServers" in result
        assert "filesystem" in result["mcpServers"]
        assert "memory" in result["mcpServers"]
        assert "database" not in result["mcpServers"]

    def test_handle_server_filtering_partial_match(self, tmp_path, monkeypatch):
        """Test _handle_server_filtering when only some servers are found."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config directory and file
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()

        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": ["--path", "/tmp"],
                }
            }
        }

        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(config_data))

        # Test with one existing and one non-existing server
        result = _handle_server_filtering(["filesystem", "memory"], "test_worker")

        assert result is not None
        assert "filesystem" in result["mcpServers"]
        assert "memory" not in result["mcpServers"]

    def test_handle_server_filtering_no_matches(self, tmp_path, monkeypatch):
        """Test _handle_server_filtering when no requested servers are found."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config directory and file
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()

        config_data = {
            "mcpServers": {
                "database": {
                    "command": "mcp-server-db",
                    "args": ["--connection", "sqlite:///test.db"],
                }
            }
        }

        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(config_data))

        # Test with non-existing servers
        result = _handle_server_filtering(["filesystem", "memory"], "test_worker")

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_mcp_config_with_servers_list(self, tmp_path, monkeypatch):
        """Test resolve_mcp_config when mcp_servers is provided as a list."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config directory and file
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()

        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": ["--path", "/tmp"],
                },
                "memory": {"command": "mcp-server-memory", "args": []},
            }
        }

        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(config_data))

        # Test resolve_mcp_config with mcp_servers list
        result = await resolve_mcp_config(
            mcp_config=None,
            mcp_config_preset=None,
            mcp_servers=["filesystem", "memory"],
            worker_name="test_worker",
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "mcpServers" in result
        assert "filesystem" in result["mcpServers"]
        assert "memory" in result["mcpServers"]


class TestWorkerCreationEdgeCases:
    """Test edge cases in worker creation."""

    def test_worker_create_request_validation(self):
        """Test WorkerCreateRequest validation for edge cases."""
        # Test with valid mcp_servers list
        request = WorkerCreateRequest(
            worker_name="test_worker",
            tool_names=["tool1"],
            agent_role="Test Agent",
            agent_type="pydantic_ai",
            model_provider="deepseek",
            model="deepseek-chat",
            system_prompt="Test system prompt",
            mcp_servers=["filesystem", "memory"],
        )
        assert request.mcp_servers == ["filesystem", "memory"]

        # Test with empty mcp_servers list
        request = WorkerCreateRequest(
            worker_name="test_worker",
            tool_names=["tool1"],
            mcp_servers=[],
        )
        assert request.mcp_servers == []

        # Test with None mcp_servers
        request = WorkerCreateRequest(
            worker_name="test_worker",
            tool_names=["tool1"],
            mcp_servers=None,
        )
        assert request.mcp_servers is None

    @pytest.mark.asyncio
    async def test_create_worker_with_mcp_servers_no_config(self, tmp_path, monkeypatch):
        """Test create_worker when mcp_servers is specified but no config exists."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock the tool loading and other dependencies
        with patch("src.worker.factory.worker_factory._load_worker_tools", return_value=[]):
            with patch("src.worker.factory.agent_factory.WorkerTemplate") as MockWorkerTemplate:
                mock_worker = MagicMock()
                mock_worker.db_path = "/tmp/test_worker.db"
                MockWorkerTemplate.return_value = mock_worker

                # Create worker with mcp_servers but no config file
                worker = await create_worker(
                    worker_name="test_worker",
                    tool_names=[],
                    mcp_servers=["filesystem", "memory"],
                )

                # Should create a basic worker without MCP servers
                assert worker is not None
                assert MockWorkerTemplate.called

    @pytest.mark.asyncio
    async def test_create_worker_handles_malformed_mcp_config(self, tmp_path, monkeypatch):
        """Test create_worker handles malformed mcp_servers data gracefully."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock dependencies
        with patch("src.worker.factory.worker_factory._load_worker_tools", return_value=[]):
            with patch("src.worker.factory.agent_factory.WorkerTemplate") as MockWorkerTemplate:
                mock_worker = MagicMock()
                mock_worker.db_path = "/tmp/test_worker.db"
                MockWorkerTemplate.return_value = mock_worker

                # Test with various malformed inputs
                test_cases = [
                    {"mcp_servers": [""]},  # Empty string in list
                    {"mcp_servers": ["valid", ""]},  # Mix of valid and empty
                    {"mcp_servers": ["server" * 50]},  # Very long server name
                    {"mcp_servers": ["server-with-special-chars!@#"]},  # Invalid chars
                ]

                for test_case in test_cases:
                    worker = await create_worker(
                        worker_name="test_worker",
                        tool_names=[],
                        mcp_servers=test_case["mcp_servers"],
                    )
                    # Should handle gracefully without crashing
                    assert worker is not None


class TestMCPConfigErrorHandling:
    """Test error handling in MCP configuration."""

    @pytest.mark.asyncio
    async def test_mcp_utils_handles_list_input(self):
        """Test that mcp_utils properly handles list input."""
        # Mock pydantic_ai module before importing mcp_utils
        with patch.dict(
            "sys.modules", {"pydantic_ai": MagicMock(), "pydantic_ai.mcp": MagicMock()}
        ):
            from src.worker.mcp_utils import create_mcp_servers_from_config

            # Should handle list input gracefully (intentional bad type).
            result = create_mcp_servers_from_config(
                cast("Any", ["filesystem", "memory"]),
                None,  # This is a list, not dict/str
            )

            assert result == []  # Should return empty list, not crash

    def test_mcp_config_json_errors(self, tmp_path, monkeypatch):
        """Test handling of JSON errors in config files."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config directory and file with invalid JSON
        config_dir = tmp_path / "mcp_configs"
        config_dir.mkdir()

        config_file = config_dir / "agent_config.json"
        config_file.write_text("{ invalid json }")

        # Should handle invalid JSON gracefully
        result = _handle_server_filtering(["filesystem"], "test_worker")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
