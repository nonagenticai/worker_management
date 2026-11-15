# Multi-Agent Type Support Architecture

## Overview

The worker management system now supports multiple agent types through an extensible registry pattern. This allows you to use different AI agent implementations (pydantic_ai, claude-code, codex, etc.) with the **same API endpoint and payload structure**.

## Key Features

✅ **Unified API** - Single endpoint for all agent types
✅ **Dynamic Registration** - Agents auto-register on import
✅ **Extensible** - Add new agent types easily
✅ **Graceful Degradation** - Works even if some agent types aren't available
✅ **Same Payload Structure** - Just change `agent_type` field

## Architecture Components

### 1. Base Agent Abstraction (`src/worker/agents/base_agent.py`)

Abstract base class defining the interface all agents must implement:
- `run(prompt)` - Execute with a simple prompt
- `run_with_context(enhanced_context)` - Execute with graph context
- `check_health()` - Health check
- `cleanup()` - Resource cleanup

### 2. Agent Registry (`src/worker/agents/agent_registry.py`)

Factory registry pattern for managing agent types:
- Maps agent type strings to factory functions
- Provides `create_agent(type, **kwargs)` for dynamic creation
- Lists available agent types via `list_available_agent_types()`

### 3. Agent Implementations

#### PydanticAI Agent (`pydantic_ai_agent.py`)
- LLM-based agents using PydanticAI framework
- Supports multiple providers (OpenAI, MLX, DeepSeek, Groq, Cohere)
- MCP server integration
- Token usage tracking

#### Claude Code Agent (`claude_code_agent.py`)
- Wrapper for Claude Code CLI tool
- Executes tasks via subprocess
- Supports system prompts and instructions
- Configurable timeout and working directory

#### Codex Agent (`codex_agent.py`)
- Wrapper for Codex CLI tool
- Similar to Claude Code agent
- CLI-based execution

### 4. Updated Components

#### API Validation (`src/api/workers.py`)
- Dynamic validation against registered agent types
- Fallback to hardcoded list if registry unavailable

#### Worker Factory (`src/worker/worker_factory.py`)
- Uses registry for agent creation instead of hardcoded conditionals
- Passes appropriate configuration to each agent type
- Fallback to basic worker on failure

## Usage Examples

### Creating a PydanticAI Worker

```bash
curl --location 'http://mini2.local:8887/workers' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '{
    "worker_name": "local_agent",
    "tool_names": [],
    "agent_role": "multitask_ai_specialist",
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": "local-llm",
    "temperature": 0.1,
    "system_prompt": "You are a highly capable AI assistant",
    "mcp_servers": ["kanban"]
}'
```

### Creating a Claude Code Worker

```bash
curl --location 'http://mini2.local:8887/workers' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '{
    "worker_name": "claude_code_agent",
    "tool_names": [],
    "agent_role": "coding_assistant",
    "agent_type": "claude_code",
    "system_prompt": "You are a coding assistant",
    "instruction": "Help with code generation and debugging",
    "description": "Claude Code CLI agent for coding tasks",
    "mcp_servers": ["filesystem"]
}'
```

### Creating a Codex Worker

```bash
curl --location 'http://mini2.local:8887/workers' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--data '{
    "worker_name": "codex_agent",
    "tool_names": [],
    "agent_role": "code_generator",
    "agent_type": "codex",
    "model": "deepseek-chat",
    "system_prompt": "You are an expert code generator",
    "description": "Codex CLI agent for code generation"
}'
```

## Adding New Agent Types

To add a new agent type:

1. **Create Agent Class**
   ```python
   from .base_agent import BaseAgent, AgentRunResult

   class MyNewAgent(BaseAgent):
       async def run(self, prompt: str) -> AgentRunResult:
           # Implementation
           pass

       async def run_with_context(self, enhanced_context) -> AgentRunResult:
           # Implementation
           pass

       async def check_health(self) -> bool:
           # Implementation
           pass

       async def cleanup(self) -> None:
           # Implementation
           pass
   ```

2. **Create Factory Function**
   ```python
   def create_my_new_agent(name, description, db_path, **kwargs):
       config = MyNewAgentConfig(...)
       return MyNewAgent(config)
   ```

3. **Register Agent**
   ```python
   def _register_my_new_agent():
       from .agent_registry import register_agent
       register_agent("my_new_agent", create_my_new_agent, MyNewAgent)

   _register_my_new_agent()
   ```

4. **Import in `__init__.py`**
   ```python
   try:
       from .my_new_agent import MyNewAgent, create_my_new_agent
   except ImportError:
       pass
   ```

That's it! The agent is now available via the API.

## Configuration Parameters

### Common Parameters (All Agent Types)
- `worker_name` - Unique worker identifier
- `agent_role` - Role description (required for agents)
- `agent_type` - Agent type: "pydantic_ai", "claude_code", "codex"
- `system_prompt` - System instructions
- `instruction` - Additional instructions
- `description` - Worker description
- `mcp_servers` - List of MCP servers to enable

### PydanticAI-Specific Parameters
- `model_provider` - LLM provider (openai, mlx, deepseek, groq, cohere)
- `model` - Model name
- `temperature` - Sampling temperature (0.0-2.0)
- `tool_names` - List of tools to use
- `base_url_override` - Custom API base URL
- `api_key_override` - Custom API key

### CLI-Agent-Specific Parameters (claude_code, codex)
- `working_directory` - Working directory for execution
- `claude_code_path` / `codex_path` - Path to CLI executable
- `timeout` - Execution timeout in seconds (default: 300)

## Testing

Run the test suite to verify the implementation:

```bash
python test_multi_agent_support.py
```

The test suite verifies:
- ✓ Agent registry functionality
- ✓ Agent creation for all types
- ✓ API validation accepts registered types
- ✓ API validation rejects invalid types

## Benefits

1. **Flexibility** - Choose the right agent for the task
2. **Extensibility** - Add new agents without modifying core code
3. **Compatibility** - Existing workers continue to work
4. **Simplicity** - Same API for all agent types
5. **Reliability** - Graceful degradation when agents unavailable

## Implementation Files

### New Files
- `src/worker/agents/base_agent.py` - Base agent abstraction
- `src/worker/agents/agent_registry.py` - Registry pattern implementation
- `src/worker/agents/claude_code_agent.py` - Claude Code CLI wrapper
- `src/worker/agents/codex_agent.py` - Codex CLI wrapper
- `test_multi_agent_support.py` - Test suite

### Modified Files
- `src/worker/agents/pydantic_ai_agent.py` - Now extends BaseAgent
- `src/worker/agents/__init__.py` - Imports all agents
- `src/api/workers.py` - Dynamic agent type validation
- `src/worker/worker_factory.py` - Registry-based agent creation

## Notes

- PydanticAI agent requires pydantic-ai package
- Claude Code agent requires claude-code CLI installed
- Codex agent requires codex CLI installed
- Agents gracefully fail if dependencies unavailable
- Graph functionality unchanged (scoped out as requested)
- All existing workers continue to function normally

## Future Enhancements

Potential additions:
- LangChain agent wrapper
- AutoGPT agent wrapper
- Custom script agent wrapper
- Agent pooling and load balancing
- Agent health monitoring dashboard
