"""Unit tests for docker_mcp_manager.py."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.worker.docker_mcp_manager import (
    ContainerNotFoundError,
    DockerCommandError,
    DockerError,
    DockerMCPManager,
)


class TestDockerMCPManagerInit:
    def test_init_defaults(self):
        mgr = DockerMCPManager()
        assert mgr.container_prefix == "mcp-"
        assert mgr.managed_label == "mcp.managed=true"


class TestParseDockerOutput:
    def test_empty_output(self):
        mgr = DockerMCPManager()
        result = mgr._parse_docker_output("")
        assert result == []

    def test_single_container(self):
        mgr = DockerMCPManager()
        line = json.dumps(
            {
                "ID": "abc123",
                "Names": "mcp-server",
                "Image": "mcp:latest",
                "Status": "Up 2 hours",
                "State": "running",
                "CreatedAt": "2026-01-01",
                "Ports": "8080",
                "Mounts": "",
                "Networks": "bridge",
                "Labels": "mcp.managed=true",
            }
        )
        result = mgr._parse_docker_output(line)
        assert len(result) == 1
        assert result[0]["id"] == "abc123"
        assert result[0]["managed"] is True

    def test_multiple_containers(self):
        mgr = DockerMCPManager()
        lines = "\n".join(
            [
                json.dumps({"ID": "c1", "Names": "mcp-a", "Image": "img1"}),
                json.dumps({"ID": "c2", "Names": "mcp-b", "Image": "img2"}),
            ]
        )
        result = mgr._parse_docker_output(lines)
        assert len(result) == 2


class TestParseContainerLine:
    def test_valid_json(self):
        mgr = DockerMCPManager()
        line = json.dumps({"ID": "123", "Names": "mcp-test", "Image": "img", "Status": "Up"})
        result = mgr._parse_container_line(line)
        assert result is not None
        assert result["id"] == "123"

    def test_invalid_json(self):
        mgr = DockerMCPManager()
        result = mgr._parse_container_line("not json")
        assert result is None

    def test_non_dict_json(self):
        mgr = DockerMCPManager()
        result = mgr._parse_container_line(json.dumps(["list"]))
        assert result is None


class TestListManagedContainers:
    @pytest.mark.asyncio
    async def test_success(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps({"ID": "c1", "Names": "mcp-x"}).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await mgr.list_managed_containers()
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_docker_not_found(self):
        mgr = DockerMCPManager()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(DockerCommandError, match="Docker not found"):
                await mgr.list_managed_containers()

    @pytest.mark.asyncio
    async def test_docker_command_failure(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error msg"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(DockerCommandError):
                await mgr.list_managed_containers()


class TestGetContainerLogs:
    @pytest.mark.asyncio
    async def test_success(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"log line 1\nlog line 2", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await mgr.get_container_logs("mcp-test", lines=10)
            assert "log line 1" in result

    @pytest.mark.asyncio
    async def test_container_not_found(self):
        mgr = DockerMCPManager()
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"No such container"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(ContainerNotFoundError):
                await mgr.get_container_logs("mcp-nonexistent")

    @pytest.mark.asyncio
    async def test_empty_name_rejected(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="non-empty string"):
            await mgr.get_container_logs("")

    @pytest.mark.asyncio
    async def test_negative_lines_rejected(self):
        mgr = DockerMCPManager()
        with pytest.raises(ValueError, match="positive integer"):
            await mgr.get_container_logs("mcp-test", lines=-1)

    @pytest.mark.asyncio
    async def test_docker_not_found(self):
        mgr = DockerMCPManager()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(DockerCommandError, match="Docker not found"):
                await mgr.get_container_logs("mcp-test")

    @pytest.mark.asyncio
    async def test_os_error(self):
        mgr = DockerMCPManager()
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Permission denied")):
            with pytest.raises(DockerCommandError):
                await mgr.get_container_logs("mcp-test")


class TestExceptions:
    def test_docker_error_hierarchy(self):
        assert issubclass(ContainerNotFoundError, DockerError)
        assert issubclass(DockerCommandError, DockerError)
