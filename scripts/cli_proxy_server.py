#!/usr/bin/env python3
"""
CLI Proxy Server for Claude Code and Codex
===========================================

This proxy server runs on the host machine and forwards requests to the local
Claude Code and Codex CLIs, which have subscription authentication configured.

The container connects to this proxy via host.docker.internal.

Usage:
    python3 cli_proxy_server.py [--port PORT] [--host HOST]

Endpoints:
    POST /claude - Execute Claude Code CLI
    POST /codex  - Execute Codex CLI
    GET  /health - Health check
"""

import asyncio
import logging
import subprocess
import sys
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("ERROR: Required dependencies not installed!")
    print("Please install: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CLI Proxy Server",
    description="Proxy server for Claude Code and Codex CLI tools",
    version="1.0.0",
)


class CLIRequest(BaseModel):
    """Request model for CLI execution."""

    prompt: str = Field(..., description="The prompt to send to the CLI")
    model: str | None = Field(None, description="Model to use (optional)")
    system_prompt: str | None = Field(None, description="System prompt (optional)")
    max_tokens: int | None = Field(None, description="Max tokens (optional)")
    temperature: float | None = Field(None, description="Temperature (optional)")
    working_directory: str | None = Field(None, description="Working directory for execution")
    timeout: int = Field(300, description="Timeout in seconds")


class CLIResponse(BaseModel):
    """Response model for CLI execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int
    execution_time: float


async def execute_cli(
    cli_name: str,
    command: list[str],
    working_dir: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Execute a CLI command asynchronously.

    Args:
        cli_name: Name of the CLI (for logging)
        command: Command and arguments to execute
        working_dir: Working directory for execution
        timeout: Timeout in seconds

    Returns:
        Dictionary with execution results
    """
    import time

    start_time = time.time()

    try:
        logger.info(f"Executing {cli_name}: {' '.join(command[:3])}... (timeout={timeout}s)")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,  # Close stdin so Codex doesn't wait for input
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            logger.exception(f"{cli_name} execution timed out after {timeout}s")
            process.kill()
            await process.wait()
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {timeout} seconds",
                "exit_code": -1,
                "execution_time": time.time() - start_time,
            }

        execution_time = time.time() - start_time
        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        success = process.returncode == 0

        if success:
            logger.info(f"{cli_name} execution completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"{cli_name} execution failed with code {process.returncode}")
            logger.error(f"stderr: {stderr_str[:500]}")

        # Build error message with more context when stderr is empty
        error_msg = None
        if not success:
            if stderr_str.strip():
                error_msg = stderr_str
            else:
                # stderr is empty, provide helpful error based on exit code and output
                error_msg = f"CLI exited with code {process.returncode}. "
                if stdout_str.strip():
                    error_msg += f"Output: {stdout_str[:200]}"
                else:
                    error_msg += f"No output. Command: {' '.join(command[:5])}..."

        return {
            "success": success,
            "output": stdout_str,
            "error": error_msg,
            "exit_code": process.returncode,
            "execution_time": execution_time,
        }

    except Exception as e:
        logger.exception(f"Error executing {cli_name}")
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "execution_time": time.time() - start_time,
        }


@app.post("/claude", response_model=CLIResponse)
async def claude_endpoint(request: CLIRequest):
    """
    Execute Claude Code CLI.

    Endpoint for container to send Claude Code requests.
    """
    try:
        # Build Claude command
        command = ["claude"]

        if request.model:
            command.extend(["--model", request.model])

        if request.system_prompt:
            command.extend(["--system-prompt", request.system_prompt])

        # Add the prompt
        command.append(request.prompt)

        # Execute
        result = await execute_cli(
            "Claude Code",
            command,
            working_dir=request.working_directory,
            timeout=request.timeout,
        )

        return CLIResponse(**result)

    except Exception as e:
        logger.exception("Error in Claude endpoint")
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.post("/codex", response_model=CLIResponse)
async def codex_endpoint(request: CLIRequest):
    """
    Execute Codex CLI.

    Endpoint for container to send Codex requests.
    """
    try:
        # Build Codex command using exec subcommand
        command = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]

        if request.model:
            command.extend(["--model", request.model])

        # Note: Codex doesn't have a separate --system-prompt flag
        # The system prompt should already be prepended to the prompt by the agent

        # Add the prompt
        command.append(request.prompt)

        # Execute
        result = await execute_cli(
            "Codex",
            command,
            working_dir=request.working_directory,
            timeout=request.timeout,
        )

        return CLIResponse(**result)

    except Exception as e:
        logger.exception("Error in Codex endpoint")
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if CLIs are available
        claude_check = subprocess.run(
            ["which", "claude"],  # noqa: S607 -- PATH lookup is intentional
            capture_output=True,
            timeout=5,
            check=False,
        )

        codex_check = subprocess.run(
            ["which", "codex"],  # noqa: S607 -- PATH lookup is intentional
            capture_output=True,
            timeout=5,
            check=False,
        )

        return {
            "status": "healthy",
            "claude_available": claude_check.returncode == 0,
            "claude_path": (
                claude_check.stdout.decode().strip() if claude_check.returncode == 0 else None
            ),
            "codex_available": codex_check.returncode == 0,
            "codex_path": (
                codex_check.stdout.decode().strip() if codex_check.returncode == 0 else None
            ),
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/mcp/add")
async def mcp_add_endpoint(request: Request):
    """
    Configure MCP server for Claude Code CLI.

    Endpoint for adding MCP servers to Claude Code configuration.
    Supports both URL-based (SSE) and command-based (stdio) MCP servers.

    URL-based: { "name": "server", "url": "http://..." }
    Command-based: { "name": "server", "command": "node", "args": ["path/to/server.js"], "env": {...} }
    """
    try:
        body = await request.json()
        server_name = body.get("name")

        if not server_name:
            raise HTTPException(status_code=400, detail="server name is required")

        # Check if it's URL-based or command-based
        server_url = body.get("url")
        server_command = body.get("command")
        env_vars = body.get("env", {})

        if server_url:
            # URL-based MCP server (SSE/HTTP transport)
            logger.info(f"Adding URL-based MCP server: {server_name} -> {server_url}")
            command = ["claude", "mcp", "add", server_name, server_url]
        elif server_command:
            # Command-based MCP server (stdio transport)
            args = body.get("args", [])

            # Build command: claude mcp add -e KEY=value <name> <command> [args...]
            command = ["claude", "mcp", "add"]

            # Add environment variables using -e flags
            for key, value in env_vars.items():
                command.extend(["-e", f"{key}={value}"])

            # Add server name, command, and args
            command.extend([server_name, server_command])
            command.extend(args)

            logger.info(
                f"Adding command-based MCP server: {server_name} -> {server_command} {' '.join(args)}"
            )
            if env_vars:
                logger.info(f"With environment variables: {list(env_vars.keys())}")
        else:
            raise HTTPException(status_code=400, detail="Either 'url' or 'command' is required")

        # Log the full command for debugging
        logger.info(f"Full command: {' '.join(command)}")

        # Execute the claude mcp add command
        result = await execute_cli("Claude MCP Add", command, working_dir=None, timeout=30)

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Error in MCP add endpoint")
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "CLI Proxy Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /claude": "Execute Claude Code CLI",
            "POST /codex": "Execute Codex CLI",
            "POST /mcp/add": "Add MCP server to Claude Code",
            "GET /health": "Health check",
        },
    }


def main():
    """Run the proxy server."""
    import argparse

    parser = argparse.ArgumentParser(description="CLI Proxy Server for Claude Code and Codex")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")  # noqa: S104
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(args.log_level.upper())

    logger.info(f"Starting CLI Proxy Server on {args.host}:{args.port}")
    logger.info("Claude Code and Codex CLI requests will be forwarded to local CLIs")

    # Run server
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
