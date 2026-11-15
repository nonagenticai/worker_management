"""Docker MCP Manager - Manages Docker containers for MCP servers."""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DockerError(Exception):
    """Base exception for Docker operations."""


class ContainerNotFoundError(DockerError):
    """Exception raised when a container is not found."""


class DockerCommandError(DockerError):
    """Exception raised when a Docker command fails."""


class DockerMCPManager:
    """Manager for Docker MCP containers.

    Provides functionality to list containers and retrieve logs
    for Docker containers with MCP prefix.
    """

    def __init__(self) -> None:
        """Initialize the Docker MCP Manager."""
        self.container_prefix: str = "mcp-"
        self.managed_label: str = "mcp.managed=true"

    async def list_managed_containers(self) -> list[dict[str, Any]]:
        """List all managed MCP containers.

        Returns:
            List of dictionaries containing container information.
            Empty list if no containers found or on error.

        Raises:
            DockerCommandError: When Docker command execution fails.
        """
        try:
            return await self._execute_docker_list_command()
        except FileNotFoundError as exc:
            logger.exception("Docker command not found. Is Docker installed?")
            raise DockerCommandError("Docker not found") from exc
        except OSError as exc:
            logger.exception("Error executing Docker command")
            raise DockerCommandError(f"Docker execution failed: {exc}") from exc

    async def _execute_docker_list_command(self) -> list[dict[str, Any]]:
        """Execute Docker command to list containers.

        Returns:
            List of container information dictionaries.
        """
        cmd = [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={self.container_prefix}",
            "--format",
            "json",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"Docker command failed: {error_msg}")
            raise DockerCommandError(f"Docker ps failed: {error_msg}")

        return self._parse_docker_output(stdout.decode().strip())

    def _parse_docker_output(self, output: str) -> list[dict[str, Any]]:
        """Parse Docker command output into container information.

        Args:
            output: Raw Docker command output.

        Returns:
            List of parsed container information dictionaries.
        """
        if not output:
            logger.info("No MCP containers found")
            return []

        containers = []
        # Parse each line as JSON (docker format=json outputs one JSON object per line)
        for line in output.split("\n"):
            if line.strip():
                container_info = self._parse_container_line(line)
                if container_info:
                    containers.append(container_info)

        logger.info(f"Found {len(containers)} managed MCP containers")
        return containers

    def _parse_container_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single line of Docker output.

        Args:
            line: JSON line from Docker output.

        Returns:
            Container information dictionary or None if parsing fails.
        """
        try:
            container_info = json.loads(line)

            # Validate that we have the required fields
            if not isinstance(container_info, dict):
                logger.warning(f"Invalid container info format: {line}")
                return None

            return {
                "id": container_info.get("ID", ""),
                "name": container_info.get("Names", ""),
                "image": container_info.get("Image", ""),
                "status": container_info.get("Status", ""),
                "state": container_info.get("State", ""),
                "created": container_info.get("CreatedAt", ""),
                "ports": container_info.get("Ports", ""),
                "mounts": container_info.get("Mounts", ""),
                "networks": container_info.get("Networks", ""),
                "labels": container_info.get("Labels", ""),
                "managed": True,  # All containers with mcp- prefix are considered managed
            }
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Failed to parse container line: {line}, error: {exc}")
            return None

    async def get_container_logs(self, container_name: str, lines: int = 50) -> str:
        """Get logs from a specific MCP container using docker CLI.

        Args:
            container_name: Name of the container to get logs from.
            lines: Number of log lines to retrieve (default: 50).

        Returns:
            Container logs as a string.

        Raises:
            ContainerNotFoundError: When the specified container doesn't exist.
            DockerCommandError: When Docker command execution fails.
            ValueError: When input parameters are invalid.
        """
        if not container_name or not isinstance(container_name, str):
            raise ValueError("Container name must be a non-empty string")

        if lines <= 0:
            raise ValueError("Lines must be a positive integer")

        try:
            return await self._execute_docker_logs_command(container_name, lines)
        except FileNotFoundError as exc:
            logger.exception("Docker command not found. Is Docker installed?")
            raise DockerCommandError("Docker not found") from exc
        except OSError as exc:
            logger.exception("Error executing Docker logs command")
            raise DockerCommandError(f"Docker logs execution failed: {exc}") from exc

    async def _execute_docker_logs_command(self, container_name: str, lines: int) -> str:
        """Execute Docker logs command.

        Args:
            container_name: Name of the container.
            lines: Number of log lines to retrieve.

        Returns:
            Container logs as a string.

        Raises:
            ContainerNotFoundError: When container is not found.
            DockerCommandError: When Docker command fails.
        """
        cmd = [
            "docker",
            "logs",
            "--tail",
            str(lines),
            "--timestamps",
            container_name,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if "No such container" in error_msg:
                logger.warning(f"Container {container_name} not found")
                raise ContainerNotFoundError(f"Container {container_name} not found")
            logger.error(f"Docker logs command failed: {error_msg}")
            raise DockerCommandError(f"Docker logs failed: {error_msg}")

        return stdout.decode("utf-8")
