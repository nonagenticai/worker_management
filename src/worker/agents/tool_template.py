"""Template for creating tools from JSON definitions."""

import ast
import contextvars
import hashlib
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from .schemas import SchemaField, ToolSchema

logger = logging.getLogger(__name__)

# Explicit type definition to avoid TypeVar evaluation issues
ToolPrepareFunction = Callable[[RunContext[Any], ToolDefinition], Awaitable[ToolDefinition | None]]

# Thread pool for sandboxed execution
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool_exec")

# Compilation cache with LRU eviction
_compilation_cache: dict[str, Any] = {}
_cache_lock = threading.RLock()
MAX_CACHE_SIZE = 100
EXECUTION_TIMEOUT = 30.0  # seconds

# Failure tracking and recovery
_failure_counts: dict[str, int] = {}
_failure_lock = threading.RLock()
MAX_FAILURES_PER_TOOL = 5
FAILURE_RESET_TIME = 3600  # 1 hour
_last_failure_reset = time.time()

# Performance monitoring
_execution_stats: dict[str, dict[str, Any]] = {}
_stats_lock = threading.RLock()


class ToolTemplate(BaseModel):
    """Template for creating tools from JSON definitions."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    implementation: Callable
    prepare: ToolPrepareFunction | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _wrap_function_as_impl(python_code: str) -> str:
        """Wrap existing function code to create an impl function."""
        import re

        # Find the function definition
        func_match = re.search(r"def\s+(\w+)\s*\([^)]*\)", python_code)
        if not func_match:
            raise ValueError("No function definition found in python_code")

        original_func_name = func_match.group(1)

        # Create wrapper implementation
        wrapper_code = f"""
{python_code}

async def impl(ctx, **kwargs):
    '''Implementation wrapper for {original_func_name}'''
    return {original_func_name}(**kwargs)
"""
        return wrapper_code

    @classmethod
    def from_json(cls, json_data: dict) -> ToolTemplate:
        """Create a ToolTemplate instance from JSON data."""
        # Map API response fields to ToolSchema fields
        normalized_data = json_data.copy()
        if "python_code" in normalized_data and "implementation" not in normalized_data:
            python_code = normalized_data.pop("python_code")
            # Wrap the function in an impl function that the template expects
            normalized_data["implementation"] = cls._wrap_function_as_impl(python_code)

        # Validate incoming data
        tool_schema = ToolSchema.model_validate(normalized_data)

        # Create input/output models
        # Extract properties from JSON Schema format
        input_properties = tool_schema.input_schema.get("properties", {})
        input_required = tool_schema.input_schema.get("required", [])

        input_model_properties: dict[str, SchemaField] = {
            k: SchemaField.model_validate(v) for k, v in input_properties.items()
        }
        input_model_required_fields = input_required

        input_model = cls._create_pydantic_model(
            f"{tool_schema.name}Input",
            input_model_properties,
            input_model_required_fields,
        )

        # Prepare properties for output model
        output_properties = tool_schema.output_schema.get("properties", {})
        output_required = tool_schema.output_schema.get("required", list(output_properties.keys()))

        output_model_properties: dict[str, SchemaField] = {
            k: SchemaField.model_validate(v) for k, v in output_properties.items()
        }
        output_model_required_fields = output_required

        output_model = cls._create_pydantic_model(
            f"{tool_schema.name}Output",
            output_model_properties,
            output_model_required_fields,
        )

        # Create implementation
        implementation = cls._create_implementation(
            tool_schema.implementation, input_model, output_model
        )

        return cls(
            name=tool_schema.name,
            description=tool_schema.description,
            input_schema=input_model,
            output_schema=output_model,
            implementation=implementation,
        )

    @staticmethod
    def _create_pydantic_model(
        model_name: str,
        properties_schema: dict[str, SchemaField],
        required_fields: list[str] | None = None,
    ) -> type[BaseModel]:
        """Create a Pydantic model from a schema definition of its properties."""
        fields: dict[str, Any] = {}
        actual_required_fields = set(required_fields or [])

        for field_name, field_def in properties_schema.items():
            # Pass a unique prefix for nested models to avoid name collisions
            nested_model_name_prefix = model_name + field_name.capitalize()
            field_type = ToolTemplate._get_field_type(field_def, nested_model_name_prefix)

            if field_name in actual_required_fields:
                fields[field_name] = (field_type, Field(...))  # Required
            else:
                fields[field_name] = (field_type | None, Field(None))  # Optional

        # Cast to proper type for mypy
        model = create_model(model_name, **fields)
        return cast("type[BaseModel]", model)

    @staticmethod
    def _get_field_type(field_schema: SchemaField, model_name_prefix: str) -> type:
        """Convert JSON schema type to Python type, handling nested structures."""
        type_str = field_schema.type

        if type_str == "string":
            return str
        elif type_str == "integer":
            return int
        elif type_str == "number":
            return float
        elif type_str == "boolean":
            return bool
        elif type_str == "array":
            if field_schema.items and isinstance(field_schema.items, dict):
                # Item schema is a dict, needs to be validated into SchemaField
                item_definition = SchemaField.model_validate(field_schema.items)
                # Compute item type for validation side effects
                ToolTemplate._get_field_type(item_definition, model_name_prefix + "Item")
                # Return list with proper item type
                return list[Any]
            else:
                # Default for array with no/invalid items schema
                return list[Any]
        elif type_str == "object":
            if field_schema.properties and isinstance(field_schema.properties, dict):
                # Properties is a dict of {prop_name: prop_schema_dict}
                # Convert each prop_schema_dict to SchemaField
                nested_properties_schema: dict[str, SchemaField] = {
                    k: SchemaField.model_validate(v) for k, v in field_schema.properties.items()
                }
                # Construct a unique name for the nested model
                nested_model_name = (
                    model_name_prefix  # Already has field name, make it more unique if needed
                )
                if not nested_model_name.endswith("Nested"):  # Avoid multiple Nested suffixes
                    nested_model_name += "Nested"

                return ToolTemplate._create_pydantic_model(
                    nested_model_name,
                    nested_properties_schema,
                    field_schema.required,  # Pass the 'required' list for the nested object's properties
                )
            else:
                # Default for object with no/invalid properties schema
                return dict[str, Any]
        else:
            # Fallback for unknown types
            logger.warning(
                f"Unknown schema type '{type_str}' encountered for field in {model_name_prefix}. Defaulting to Any."
            )
            return Any

    @staticmethod
    def _reset_failure_counts_if_needed() -> None:
        """Reset failure counts periodically."""
        global _last_failure_reset  # noqa: PLW0603 -- module-level rate-limit timestamp
        current_time = time.time()

        with _failure_lock:
            if current_time - _last_failure_reset > FAILURE_RESET_TIME:
                _failure_counts.clear()
                _last_failure_reset = current_time
                logger.info("Reset tool failure counts")

    @staticmethod
    def _is_tool_failed(tool_name: str) -> bool:
        """Check if a tool has exceeded failure threshold."""
        ToolTemplate._reset_failure_counts_if_needed()

        with _failure_lock:
            return _failure_counts.get(tool_name, 0) >= MAX_FAILURES_PER_TOOL

    @staticmethod
    def _record_tool_failure(tool_name: str) -> None:
        """Record a tool execution failure."""
        with _failure_lock:
            _failure_counts[tool_name] = _failure_counts.get(tool_name, 0) + 1
            failure_count = _failure_counts[tool_name]

            if failure_count >= MAX_FAILURES_PER_TOOL:
                logger.error(
                    f"Tool {tool_name} has reached failure threshold ({failure_count} failures)"
                )
            else:
                logger.warning(f"Tool {tool_name} failure count: {failure_count}")

    @staticmethod
    def _record_execution_stats(tool_name: str, execution_time: float, success: bool) -> None:  # noqa: FBT001, RUF100
        """Record execution statistics for monitoring."""
        with _stats_lock:
            if tool_name not in _execution_stats:
                _execution_stats[tool_name] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                    "avg_time": 0.0,
                }

            stats = _execution_stats[tool_name]
            stats["total_executions"] += 1
            stats["total_time"] += execution_time

            if success:
                stats["successful_executions"] += 1
            else:
                stats["failed_executions"] += 1

            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)

            stats["avg_time"] = stats["total_time"] / stats["total_executions"]

    @staticmethod
    def get_tool_stats(tool_name: str) -> dict[str, Any] | None:
        """Get execution statistics for a tool."""
        with _stats_lock:
            return _execution_stats.get(tool_name)

    @staticmethod
    def get_all_stats() -> dict[str, dict[str, Any]]:
        """Get execution statistics for all tools."""
        with _stats_lock:
            return _execution_stats.copy()

    @staticmethod
    def _get_cache_key(code: str, input_model_name: str, output_model_name: str) -> str:
        """Generate a cache key for compiled tool implementations."""
        content = f"{code}_{input_model_name}_{output_model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _evict_cache_if_needed() -> None:
        """Evict oldest entries if cache is too large."""
        if len(_compilation_cache) >= MAX_CACHE_SIZE:
            # Remove oldest half of entries (simple LRU approximation)
            items = list(_compilation_cache.items())
            to_remove = len(items) // 2
            for key, _ in items[:to_remove]:
                del _compilation_cache[key]
            logger.debug(f"Evicted {to_remove} entries from tool compilation cache")

    @staticmethod
    def _validate_code_security(code: str) -> None:
        """Enhanced security validation of Python code."""
        try:
            # Parse AST for comprehensive security analysis
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}") from e

        # Dangerous imports and calls
        dangerous_patterns = {
            "imports": {
                "os",
                "subprocess",
                "sys",
                "__builtin__",
                "builtins",
                "importlib",
                "pkgutil",
            },
            "calls": {
                "exec",
                "eval",
                "compile",
                "__import__",
                "open",
                "input",
                "raw_input",
            },
            "attributes": {"__globals__", "__locals__", "__code__", "__class__"},
        }

        for node in ast.walk(tree):
            # Check dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names = []
                if isinstance(node, ast.Import):
                    module_names = [alias.name for alias in node.names]
                elif node.module:
                    module_names = [node.module]

                for module_name in module_names:
                    if any(dangerous in module_name for dangerous in dangerous_patterns["imports"]):
                        raise ValueError(f"Dangerous import detected: {module_name}")

            # Check dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_patterns["calls"]:
                    raise ValueError(f"Dangerous function call detected: {node.func.id}")

            # Check dangerous attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in dangerous_patterns["attributes"]:
                    raise ValueError(f"Dangerous attribute access detected: {node.attr}")

    @staticmethod
    def _create_sandboxed_namespace() -> dict[str, Any]:
        """Create a secure, restricted namespace for code execution."""
        from typing import Optional as typing_Optional

        # Safe builtins only
        safe_builtins = {
            # Basic types
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            # Safe functions
            "len": len,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            # Type checking
            "isinstance": isinstance,
            "type": type,
            # String/data processing
            "ord": ord,
            "chr": chr,
            "hex": hex,
            "bin": bin,
            # Math
            "pow": pow,
            "round": round,
            # Required types for pydantic (Optional exposed for sandboxed user code)
            "Dict": dict,
            "List": list,
            "Optional": typing_Optional,
            "Any": Any,
            "RunContext": RunContext,
        }

        # Import safe standard library modules
        import datetime
        import json
        import math
        import re

        safe_modules = {
            "math": math,
            "datetime": datetime,
            "json": json,
            "re": re,
        }

        namespace: dict[str, Any] = {**safe_builtins, **safe_modules}

        # Disable dangerous builtins
        namespace["__builtins__"] = safe_builtins

        return namespace

    @staticmethod
    def _execute_in_sandbox(code: str, namespace: dict[str, Any]) -> Any:
        """Execute code in a sandboxed environment with timeout."""

        def _run_code() -> Any:
            try:
                # Compile with restricted mode
                compiled_code = compile(code, "<tool_code>", "exec")

                # Execute in sandbox
                # Container isolation provides primary security boundary
                exec(compiled_code, namespace, namespace)  # nosec B102 — intentional dynamic execution  # noqa: S102

                # Return the impl function
                impl = namespace.get("impl")
                if not impl:
                    raise ValueError("Implementation must define an 'impl' function")

                if not callable(impl):
                    raise ValueError("Implementation 'impl' must be callable")  # noqa: RUF100, TRY004

                return impl

            except Exception as e:
                logger.exception("Sandboxed execution error")
                raise ValueError(f"Code execution failed: {e}") from e

        try:
            # Execute with timeout in thread pool
            # Copy the caller's contextvars: module top-level code runs on this pool
            # thread and would otherwise execute without the request identity.
            future = _executor.submit(contextvars.copy_context().run, _run_code)
            return future.result(timeout=EXECUTION_TIMEOUT)
        except TimeoutError:
            raise ValueError(f"Code execution timeout after {EXECUTION_TIMEOUT}s") from None
        except Exception as e:
            raise ValueError(f"Sandboxed execution failed: {e}") from e

    @staticmethod
    def _create_implementation(
        code: str, input_model: type[BaseModel], output_model: type[BaseModel]
    ) -> Callable:
        """Create a secure async implementation function from code string."""
        # Generate cache key
        cache_key = ToolTemplate._get_cache_key(code, input_model.__name__, output_model.__name__)

        # Check cache first
        with _cache_lock:
            if cache_key in _compilation_cache:
                logger.debug(f"Using cached tool implementation: {cache_key}")
                cached_impl = _compilation_cache[cache_key]

                # Return wrapper with cached implementation
                async def cached_wrapper(ctx: RunContext, **kwargs: Any) -> Any:
                    return await ToolTemplate._execute_implementation(
                        cached_impl, ctx, kwargs, input_model, output_model
                    )

                return cached_wrapper

        # Security validation
        ToolTemplate._validate_code_security(code)

        # Create sandboxed namespace
        namespace = ToolTemplate._create_sandboxed_namespace()

        # Execute in sandbox
        impl = ToolTemplate._execute_in_sandbox(code, namespace)

        # Cache the compiled implementation
        with _cache_lock:
            ToolTemplate._evict_cache_if_needed()
            _compilation_cache[cache_key] = impl
            logger.debug(f"Cached tool implementation: {cache_key}")

        # Return secure wrapper
        async def wrapped_implementation(ctx: RunContext, **kwargs: Any) -> Any:
            return await ToolTemplate._execute_implementation(
                impl, ctx, kwargs, input_model, output_model
            )

        return wrapped_implementation

    @staticmethod
    async def _execute_implementation(
        impl: Callable,
        ctx: RunContext,
        kwargs: dict[str, Any],
        input_model: type[BaseModel],
        output_model: type[BaseModel],
    ) -> Any:
        """Execute implementation with proper validation and error handling."""
        tool_name = getattr(ctx, "tool_name", "unknown_tool")
        start_time = time.time()

        try:
            # Check if tool has exceeded failure threshold
            if ToolTemplate._is_tool_failed(tool_name):
                logger.error(f"Tool {tool_name} is disabled due to repeated failures")
                raise ValueError(f"Tool {tool_name} is temporarily disabled due to failures")

            # Validate input
            validated_input = input_model.model_validate(kwargs)

            # Execute implementation with isolation
            # Check if implementation is async
            if inspect.iscoroutinefunction(impl):
                result = await impl(ctx, **validated_input.model_dump())
            else:
                # Run sync function in executor to avoid blocking
                loop = None
                try:
                    import asyncio

                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop:
                    # run_in_executor does NOT copy contextvars (unlike to_thread), so
                    # a sync tool would lose the request's identity — X-Project-Id among
                    # it — and its LLM spend would reach billing with no project.
                    call_ctx = contextvars.copy_context()
                    result = await loop.run_in_executor(
                        _executor,
                        lambda: call_ctx.run(impl, ctx, **validated_input.model_dump()),
                    )
                else:
                    result = impl(ctx, **validated_input.model_dump())

            execution_time = time.time() - start_time

            # Log performance
            if execution_time > 5.0:  # Log slow executions
                logger.warning(f"Slow tool execution for {tool_name}: {execution_time:.2f}s")

            # Validate output
            validated_output = output_model.model_validate(result)

            # Record successful execution stats
            ToolTemplate._record_execution_stats(tool_name, execution_time, True)  # noqa: FBT003, RUF100

            return validated_output.model_dump()

        except Exception as e:
            execution_time = time.time() - start_time

            # Record failure
            ToolTemplate._record_tool_failure(tool_name)
            ToolTemplate._record_execution_stats(tool_name, execution_time, False)  # noqa: FBT003, RUF100

            logger.exception(f"Tool {tool_name} implementation error")
            # Don't expose internal details to users
            raise ValueError(f"Tool {tool_name} execution failed: {type(e).__name__}") from e

    def create_tool(self) -> Tool:
        """Create a tool instance from this template."""
        if self.implementation is None:
            raise ValueError(f"Tool {self.name} has no implementation")

        return Tool(
            self.implementation,
            name=self.name,
            description=self.description,
            prepare=self.prepare,
        )
