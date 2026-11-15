#!/usr/bin/env python3
"""Example: create workers with different MCP configurations.

Uses the pydantic-ai compatible worker API.
"""

import requests

# Your worker management API base URL
API_BASE = "http://localhost:8887"


def create_worker_with_mcp_preset() -> None:
    """Create a worker using a preset MCP configuration."""
    worker_config = {
        "worker_name": "github_assistant",
        "agent_role": "github_specialist",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.3,
        "system_prompt": "You are a GitHub specialist assistant.",
        "instruction": "Help users with GitHub operations using available tools.",
        "description": "A specialized agent for GitHub operations",
        # Use preset MCP configuration
        "mcp_config_preset": "github",  # Uses ./mcp_configs/github_agent_config.json
    }

    response = requests.post(f"{API_BASE}/workers", json=worker_config, timeout=30)
    print("GitHub Agent Response:", response.json())


def create_worker_with_specific_servers() -> None:
    """Create a worker using specific MCP servers from default config."""
    worker_config = {
        "worker_name": "data_analyst",
        "agent_role": "data_specialist",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.1,
        "system_prompt": "You are a data analysis specialist.",
        "instruction": "Help users with data analysis using available database and file tools.",
        "description": "A specialized agent for data analysis",
        # Use specific MCP servers
        "mcp_servers": ["sqlite", "filesystem"],  # Only enable these servers
    }

    response = requests.post(f"{API_BASE}/workers", json=worker_config, timeout=30)
    print("Data Analyst Response:", response.json())


def create_worker_with_custom_config() -> None:
    """Create a worker with a completely custom MCP configuration."""
    custom_mcp_config = {
        "mcpServers": {
            "custom_fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch", "--user-agent", "MyCustomAgent"],
            },
            "local_filesystem": {
                "command": "uvx",
                "args": ["mcp-server-filesystem", "/workspace"],
            },
        }
    }

    worker_config = {
        "worker_name": "custom_assistant",
        "agent_role": "general_assistant",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.5,
        "system_prompt": "You are a helpful assistant with custom tools.",
        "instruction": "Use your custom fetch and filesystem tools to help users.",
        "description": "An agent with custom MCP server configuration",
        # Use direct MCP config dictionary
        "mcp_config": custom_mcp_config,
    }

    response = requests.post(f"{API_BASE}/workers", json=worker_config, timeout=30)
    print("Custom Agent Response:", response.json())


def create_lightweight_worker() -> None:
    """Create a worker with no MCP servers for fast responses."""
    worker_config = {
        "worker_name": "chat_bot",
        "agent_role": "conversational_assistant",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.7,
        "system_prompt": "You are a friendly conversational assistant.",
        "instruction": "Engage in helpful conversations without using external tools.",
        "description": "A lightweight conversational agent",
        # Use preset with no MCP servers
        "mcp_config_preset": "lightweight",
    }

    response = requests.post(f"{API_BASE}/workers", json=worker_config, timeout=30)
    print("Lightweight Agent Response:", response.json())


def test_worker(worker_name: str, message: str) -> None:
    """Test a worker by sending it a message."""
    payload = {"message": message, "context": {"test": True}}

    response = requests.post(f"{API_BASE}/workers/{worker_name}/message", json=payload, timeout=30)
    if response.status_code == 200:
        print(f"\n{worker_name} Response:", response.json()["response"])
    else:
        print(f"\n{worker_name} Error:", response.json())


if __name__ == "__main__":
    print("Creating workers with different MCP configurations...\n")

    # Create different types of workers
    create_worker_with_mcp_preset()
    create_worker_with_specific_servers()
    create_worker_with_custom_config()
    create_lightweight_worker()

    print("\n" + "=" * 50)
    print("Testing workers...")

    # Test the workers
    test_worker("github_assistant", "What GitHub repositories can you help me with?")
    test_worker("data_analyst", "Can you help me analyze a database?")
    test_worker("custom_assistant", "What tools do you have available?")
    test_worker("chat_bot", "Hello! How are you doing today?")
