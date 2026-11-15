"""Tool conversion utilities."""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from .schemas import ToolSchema
from .tool_template import ToolTemplate

logger = logging.getLogger(__name__)


def parse_json_to_tools(json_data: list[dict]) -> list[ToolTemplate]:
    """Parse JSON data and create tool instances using ToolTemplate.

    Args:
        json_data: List of tool definitions in JSON format

    Returns:
        List of instantiated ToolTemplate objects

    Raises:
        ValidationError: If any tool definition is invalid
    """
    tools = []
    for tool_json in json_data:
        try:
            # Validate against schema
            ToolSchema.model_validate(tool_json)
            # Create tool template
            tool = ToolTemplate.from_json(tool_json)
            tools.append(tool)
        except ValidationError:
            logger.exception("Invalid tool definition")
            raise
        except ValueError, TypeError, AttributeError:
            logger.exception("Error creating tool")
            raise
    return tools


def load_local_tool(file_path: str | Path) -> ToolTemplate | None:
    """Load a tool from a local Python file.

    Args:
        file_path: Path to the Python file containing the tool definition

    Returns:
        ToolTemplate instance if successful, None if failed

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If required attributes are missing
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Tool file not found: {file_path}")
            return None

        # Import the module
        spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
        if not spec or not spec.loader:
            logger.error(f"Could not load spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        module_name = file_path.stem

        # Store original module if it exists to restore on failure
        original_module = sys.modules.get(module_name)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)

            # Extract required attributes
            required_attrs = [
                "TOOL_NAME",
                "TOOL_DESCRIPTION",
                "INPUT_SCHEMA",
                "OUTPUT_SCHEMA",
                "impl",
            ]
            if not all(hasattr(module, attr) for attr in required_attrs):
                missing = [attr for attr in required_attrs if not hasattr(module, attr)]
                logger.error(f"Missing required attributes in {file_path}: {missing}")
                raise AttributeError(f"Missing required attributes: {missing}")

            # Create tool schema
            tool_schema = {
                "name": module.TOOL_NAME,
                "description": module.TOOL_DESCRIPTION,
                "input_schema": module.INPUT_SCHEMA,
                "output_schema": module.OUTPUT_SCHEMA,
                "implementation": inspect.getsource(module.impl),
            }

            return ToolTemplate.from_json(tool_schema)

        except Exception:
            # Clean up the module if any step fails
            if original_module is not None:
                sys.modules[module_name] = original_module
            elif module_name in sys.modules:
                del sys.modules[module_name]
            raise

    except ImportError, AttributeError, ValueError, TypeError:
        logger.exception(f"Error loading tool from {file_path}")
        return None


def load_tools_from_directory(directory_path: str | Path) -> list[ToolTemplate]:
    """Load all tools from a directory.

    Args:
        directory_path: Path to directory containing tool Python files

    Returns:
        List of successfully loaded ToolTemplate instances
    """
    directory_path = Path(directory_path)
    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return []

    tools = []
    for file_path in directory_path.glob("**/*.py"):
        if file_path.name.startswith("__"):
            continue

        tool = load_local_tool(file_path)
        if tool:
            tools.append(tool)
            logger.info(f"Successfully loaded tool {tool.name} from {file_path}")

    return tools
