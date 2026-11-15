"""Schema definitions for tool conversion.

This module provides Pydantic models for defining and validating tool schemas,
including field definitions, input/output schemas, and tool implementations.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SchemaField(BaseModel):
    """Definition of a field in a schema.

    Represents a single field in a tool's input or output schema with
    type information and validation rules.
    """

    type: str
    description: str | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None
    required: list[str] | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_schema_field(cls, values: Any) -> dict[str, Any]:
        """Validate the schema field values.

        Args:
            values: Dictionary or other data structure of field values to validate

        Returns:
            Validated dictionary of field values

        Raises:
            ValueError: If field data is invalid
        """
        # Handle case where values is not a dictionary (e.g., list from JSON schema)
        if not isinstance(values, dict):
            raise ValueError(f"Schema field must be a dictionary, got {type(values)}")  # noqa: RUF100, TRY004

        # Ensure type field exists and is a string
        if not isinstance(values.get("type"), str):
            raise ValueError("Field type must be a string")  # noqa: RUF100, TRY004
        return values


class ToolSchema(BaseModel):
    """Base schema for tool JSON definition.

    Defines the complete structure for a tool including its name, description,
    input/output schemas, and Python implementation code.
    """

    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: dict[str, Any] = Field(..., description="Schema definition for tool input")
    output_schema: dict[str, Any] = Field(..., description="Schema definition for tool output")
    implementation: str = Field(..., description="Python code implementing the tool")

    model_config = ConfigDict(extra="ignore")

    @field_validator("implementation")
    @classmethod
    def validate_implementation(cls, v: str) -> str:
        """Validate that implementation is valid Python code.

        Args:
            v: Python code string to validate

        Returns:
            Validated Python code string

        Raises:
            ValueError: If code cannot be compiled
        """
        try:
            compile(v, "<string>", "exec")
            return v
        except Exception as e:
            raise ValueError(f"Invalid Python code: {e!s}") from e

    # Schema validation is now handled in ToolTemplate.from_json()
    # since we support standard JSON Schema format


class DynamicModel(BaseModel):
    """Base class for dynamically created models.

    Provides configuration for models that are created dynamically
    with arbitrary field types and extra validation disabled.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
