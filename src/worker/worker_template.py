"""Worker template module for creating both agent and non-agent workers.

This module provides templates for creating workers that can execute tasks
using available tools with intelligent tool selection logic.
"""

import inspect
import json
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.worker.errors import extract_error_detail
from src.worker.provider_registry import PROVIDER_NAMES, ModelProvider

logger = logging.getLogger(__name__)


class WorkerRunResult(BaseModel):
    """Result from a non-agent worker execution."""

    success: bool = Field(..., description="Whether the worker execution was successful")
    data: Any = Field(..., description="Output data from the worker execution")
    message: str = Field(..., description="Status message describing the execution result")
    error: str | None = Field(None, description="Error message if execution failed")
    execution_time: float | None = Field(
        None, description="Time taken to execute the task in seconds"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tools that were used during execution",
    )


class WorkerTemplate(BaseModel):
    """Template for creating workers (non-agent Python function workers)."""

    name: str = Field(..., description="Name of the worker")
    description: str = Field(..., description="Description of what the worker does")
    tools: list[Any] = Field(..., description="List of tools available to the worker")
    output_type: type[BaseModel] | None = Field(
        None, description="Expected output type for structured output"
    )
    retries: int = Field(default=3, description="Number of retries for failed executions")
    fastapi_app: FastAPI | None = Field(
        None, description="FastAPI application instance if used as a service"
    )
    db_path: str = Field(..., description="Path to the database file for this worker")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    async def run(self, task: str) -> WorkerRunResult:
        """Execute the worker task using available tools."""
        start_time = time.time()

        logger.info("Non-agent worker %s executing task: %s", self.name, task)

        try:
            # Parse task if it's JSON
            task_data = self._parse_task(task)

            # Execute using available tools
            result = await self._execute_with_tools(task_data)

            execution_time = time.time() - start_time

            return WorkerRunResult(
                success=True,
                data=result,
                message=f"Task completed successfully by worker {self.name}",
                error=None,
                execution_time=execution_time,
                tools_used=[
                    tool.__class__.__name__ if hasattr(tool, "__class__") else str(tool)
                    for tool in self.tools
                ],
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception("Worker %s execution failed", self.name)

            return WorkerRunResult(
                success=False,
                data=None,
                message=f"Task failed in worker {self.name}",
                error=str(e),
                execution_time=execution_time,
            )

    def _parse_task(self, task: str) -> dict[str, Any]:
        """Parse task string into structured data."""
        try:
            # Try to parse as JSON first
            if task.strip().startswith("{") or task.strip().startswith("["):
                parsed_data: dict[str, Any] = json.loads(task)
                return parsed_data
            # Treat as plain text task
            return {"task": task, "type": "text"}
        except json.JSONDecodeError:
            # Fallback to text task
            return {"task": task, "type": "text"}

    async def _execute_with_tools(self, task_data: dict[str, Any]) -> Any:
        """Execute task using available tools."""
        if not self.tools:
            # No tools available - execute as simple processing
            return await self._execute_simple_task(task_data)

        # Try to match task to available tools
        task_content = task_data.get("task", "")

        # Select most appropriate tool based on task characteristics
        tool = self._select_best_tool(task_data)

        if hasattr(tool, "run"):
            # Tool has a run method
            if inspect.iscoroutinefunction(tool.run):
                return await tool.run(task_content)
            return tool.run(task_content)
        if callable(tool):
            # Tool is callable
            if inspect.iscoroutinefunction(tool):
                return await tool(task_content)
            return tool(task_content)
        # Tool doesn't have expected interface
        logger.warning(
            "Tool %s doesn't have run() method or __call__, using simple execution",
            tool,
        )
        return await self._execute_simple_task(task_data)

    def _select_best_tool(self, task_data: dict[str, Any]) -> Any:
        """Select the most appropriate tool for the given task.

        Uses heuristics to match task characteristics with tool capabilities:
        - Examines task type and content keywords
        - Checks tool names and docstrings for relevance
        - Falls back to first available tool if no clear match

        Args:
            task_data: Structured task data containing type and content

        Returns:
            The most appropriate tool from the available tools

        Raises:
            ValueError: If no tools are available
        """
        if not self.tools:
            raise ValueError("No tools available for task execution")

        task_type = task_data.get("type", "text")
        task_content = task_data.get("task", "").lower()

        # Define keyword mappings for different tool types
        keyword_mappings = {
            "file": ["file", "read", "write", "save", "load", "document"],
            "data": ["data", "csv", "json", "parse", "process", "transform"],
            "api": ["api", "http", "request", "fetch", "call", "endpoint"],
            "calculation": ["calc", "math", "compute", "calculate", "formula"],
            "text": ["text", "string", "format", "template", "content"],
            "database": ["db", "database", "sql", "query", "table", "record"],
            "web": ["web", "html", "scrape", "browser", "url", "website"],
            "image": ["image", "picture", "photo", "visual", "graphic"],
            "email": ["email", "mail", "send", "message", "notification"],
        }

        # Score each tool based on relevance to task
        tool_scores = []

        for tool in self.tools:
            score = 0
            tool_name = getattr(tool, "__name__", tool.__class__.__name__).lower()
            tool_doc = getattr(tool, "__doc__", "") or ""
            tool_doc = tool_doc.lower() if tool_doc else ""

            # Score based on task type match
            if task_type in tool_name or task_type in tool_doc:
                score += 10

            # Score based on keyword presence in task content
            for keywords in keyword_mappings.values():
                if any(keyword in task_content for keyword in keywords):
                    if any(keyword in tool_name for keyword in keywords):
                        score += 8
                    if any(keyword in tool_doc for keyword in keywords):
                        score += 5

            # Bonus for exact name matches
            if any(word in tool_name for word in task_content.split()):
                score += 15

            # Check if tool has expected interface methods
            if hasattr(tool, "run") or callable(tool):
                score += 2
            else:
                score -= 5  # Penalize tools without proper interface

            tool_scores.append((tool, score))

        # Sort by score (highest first) and return the best tool
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        best_tool = tool_scores[0][0]

        logger.info(
            "Selected tool %s with score %d for task type '%s'",
            getattr(best_tool, "__name__", best_tool.__class__.__name__),
            tool_scores[0][1],
            task_type,
        )

        return best_tool

    async def _execute_simple_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Simple task execution without tools."""
        task_content = task_data.get("task", "")

        # Basic processing - this is where you'd implement your Python function logic
        result = {
            "worker_name": self.name,
            "processed_task": task_content,
            "timestamp": datetime.now().isoformat(),
            "processing_type": "simple_worker",
            "result": f"Processed by non-agent worker: {task_content}",
        }

        logger.info("Simple task execution completed: %s", result)
        return result


class AgentTemplate(WorkerTemplate):
    """Template for creating agents with LLM capabilities."""

    agent_type: str
    agent_role: str
    # Reuses the ModelProvider Literal itself rather than re-declaring it, so
    # this template can never again reject a provider ModelFactory supports
    # (it used to omit xai, google and vlm_local). It stays a static Literal
    # for mypy/pydantic because ModelProvider *is* the Literal alias.
    model_provider: ModelProvider
    model: str
    temperature: float = 0.0
    system_prompt: str | None = None
    instruction: str | None = None
    user_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", protected_namespaces=())

    async def run(self, task: str) -> WorkerRunResult:
        """Execute agent task using PydanticAI agent."""
        start_time = time.time()

        logger.info("Agent worker %s executing task: %s", self.name, task)

        try:
            # Import here to avoid circular dependencies
            from src.worker.agents.pydantic_ai_agent import create_pydantic_ai_agent

            # Defense-in-depth re-check of what the model_provider annotation
            # already enforces. Derived from the same ModelProvider literal, so
            # there is exactly one provider list in this file, not two.
            if self.model_provider not in PROVIDER_NAMES:
                raise ValueError(
                    f"Invalid model provider: {self.model_provider}. "
                    f"Must be one of: {list(PROVIDER_NAMES)}"
                )

            # Create agent configuration
            agent = create_pydantic_ai_agent(
                name=self.name,
                description=self.description,
                model_provider=self.model_provider,
                model=self.model,
                temperature=self.temperature,
                db_path=self.db_path,
                system_prompt=self.system_prompt,
                instruction=self.instruction,
                user_prompt=self.user_prompt,
                tools=self.tools,
            )

            # Execute agent task
            agent_result = await agent.run(task)

            execution_time = time.time() - start_time

            return WorkerRunResult(
                success=True,
                data=agent_result.output,
                message=f"Agent task completed successfully by {self.name}",
                error=None,
                execution_time=execution_time,
                tools_used=[
                    tool.__class__.__name__ if hasattr(tool, "__class__") else str(tool)
                    for tool in self.tools
                ],
            )

        except Exception as e:
            execution_time = time.time() - start_time
            detail = extract_error_detail(e)
            logger.error(
                "Agent worker %s execution failed [classification=%s status=%s]: %s",
                self.name,
                detail["error_classification"],
                detail["status_code"],
                str(e),
                exc_info=True,
                extra={"worker_name": self.name, **detail},
            )

            return WorkerRunResult(
                success=False,
                data=None,
                message=f"Agent task failed in worker {self.name}",
                error=f"[{detail['error_classification']}] {e!s}",
                execution_time=execution_time,
            )
