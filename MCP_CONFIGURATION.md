# MCP Server Configuration for Claude Code Agents

## Overview

Claude Code agents now support automatic MCP (Model Context Protocol) server configuration when using the CLI proxy. When an agent is created with `mcp_servers` specified, those MCP servers will be automatically added to the host's Claude Code CLI configuration.

## How It Works

1. When a Claude Code agent is initialized with `mcp_servers` configured
2. The agent sends configuration requests to the proxy server
3. The proxy executes `claude mcp add <name> <url>` on the host machine
4. The MCP servers become available to all subsequent Claude Code CLI executions on that host

## Configuration Format

MCP servers must be specified as dictionaries with `name` and `url` keys:

```python
from worker.agents.claude_code_agent import create_claude_code_agent

agent = create_claude_code_agent(
    name="my-agent",
    description="Agent with MCP access",
    db_path="./src/db/worker_registry.sqlite",
    mcp_servers=[
        {
            "name": "kanban",
            "url": "http://localhost:3000/sse"  # Must be accessible from HOST, not container
        },
        {
            "name": "github",
            "url": "npx -y @modelcontextprotocol/server-github"
        }
    ]
)
```

## Important Limitations

### Networking Considerations

**Problem**: The proxy runs on the HOST machine, while some MCP servers run in Docker containers.

- ❌ **Won't work**: `http://kanban-mcp:1337/sse` (Docker network hostname not accessible from host)
- ✅ **Will work**: `http://localhost:1337/sse` (if port is published to host)

**Solution**: Publish MCP server ports to the host:

```yaml
# docker-compose.yaml
services:
  kanban-mcp:
    ports:
      - "1337:1337"  # Publish to host
```

Then configure the agent with the localhost URL:
```python
mcp_servers=[
    {"name": "kanban", "url": "http://localhost:1337/sse"}
]
```

### Global Configuration

MCP servers are added to the **host's global Claude Code configuration**, not per-agent. This means:

- All agents using the proxy will have access to configured MCP servers
- MCP configuration persists across agent restarts
- Different agents cannot have completely isolated MCP configurations when using the proxy

## Alternative: Direct CLI Access

If you need MCP servers that are only accessible from within the Docker network, you have two options:

### Option 1: Use PydanticAI Agent Instead

PydanticAI agents run inside the container and can access container network resources:

```python
from worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

agent = create_pydantic_ai_agent(
    name="my-agent",
    model="local-llm",
    mcp_servers=["http://kanban-mcp:1337/sse"]  # Container network URL works!
)
```

### Option 2: Install Claude CLI in Container

If you must use Claude Code agent without the proxy:

1. Set `CLI_PROXY_ENABLED=false` in docker-compose.yaml
2. Ensure Claude Code CLI is installed in the container (already done in Dockerfile)
3. Configure authentication inside the container

## Testing

Test MCP configuration with:

```bash
python3 test_mcp_configuration.py
```

Check proxy logs to verify MCP commands:

```bash
tail -f /tmp/cli_proxy_server.log
```

Verify MCP servers are configured on host:

```bash
claude mcp list
```

## API Endpoints

The proxy server provides the following MCP-related endpoint:

### POST /mcp/add

Add an MCP server to Claude Code configuration.

**Request Body**:
```json
{
  "name": "server-name",
  "url": "http://localhost:3000/sse"
}
```

**Response**:
```json
{
  "success": true,
  "output": "",
  "error": null,
  "exit_code": 0,
  "execution_time": 0.5
}
```

## Troubleshooting

### "MCP server requires a URL/command"

**Cause**: MCP server specified as string instead of dict

**Fix**: Use dict format with name and url

### "All connection attempts failed"

**Cause**: Proxy server not running

**Fix**: Start proxy server:
```bash
python3 scripts/cli_proxy_server.py --port 9000 > /tmp/cli_proxy_server.log 2>&1 &
```

### "MCP add failed: error: missing required argument 'commandOrUrl'"

**Cause**: URL not provided in configuration

**Fix**: Ensure MCP server config includes both name and url

### MCP tools not available in Claude responses

**Possible causes**:
1. MCP server not running
2. MCP server URL not accessible from host
3. MCP server not properly configured

**Debug steps**:
1. Check proxy logs: `tail -f /tmp/cli_proxy_server.log`
2. List configured MCPs: `claude mcp list`
3. Test MCP server directly: `curl http://localhost:1337/sse`

## Examples

### Example 1: Kanban MCP (Requires Port Publishing)

```yaml
# docker-compose.yaml
services:
  kanban-mcp:
    ports:
      - "1337:1337"
```

```python
# Agent configuration
agent = create_claude_code_agent(
    name="kanban-agent",
    description="Agent with Kanban access",
    db_path="./src/db/worker_registry.sqlite",
    mcp_servers=[
        {"name": "kanban", "url": "http://localhost:1337/sse"}
    ]
)
```

### Example 2: NPX-based MCP Servers

```python
agent = create_claude_code_agent(
    name="multi-mcp-agent",
    description="Agent with multiple MCP servers",
    db_path="./src/db/worker_registry.sqlite",
    mcp_servers=[
        {"name": "github", "url": "npx -y @modelcontextprotocol/server-github"},
        {"name": "filesystem", "url": "npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir"}
    ]
)
```

### Example 3: Mixed Configuration (API + CLI)

You can also configure some MCPs via the API and others manually:

```python
# Via API during agent creation
agent = create_claude_code_agent(
    name="my-agent",
    db_path="./src/db/worker_registry.sqlite",
    mcp_servers=[
        {"name": "github", "url": "npx -y @modelcontextprotocol/server-github"}
    ]
)

# Manually on host (persists for all agents)
# claude mcp add database postgresql://localhost/mydb
```

## Summary

- ✅ Automatic MCP configuration when creating Claude Code agents
- ✅ Supports any MCP server accessible from the host
- ✅ Works with both SSE and stdio MCP servers
- ❌ Cannot access Docker-internal MCP servers without port publishing
- ❌ MCP configuration is global per host, not per-agent

For per-agent MCP isolation or Docker-internal MCP access, use PydanticAI agents instead.
