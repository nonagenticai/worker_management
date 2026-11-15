"""Pydantic models for parsed asset metadata.

This module defines structured models for storing parsed and analyzed data
from various asset types including procedures, requirements, code, and entities.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CodeSnippet(BaseModel):
    """Represents a parsed code snippet with metadata."""

    content: str = Field(..., description="The actual code content")
    language: str = Field(..., description="Programming language of the code")
    file_path: str | None = Field(None, description="Source file path")
    line_start: int | None = Field(None, description="Starting line number")
    line_end: int | None = Field(None, description="Ending line number")
    function_name: str | None = Field(None, description="Function/method name if applicable")
    class_name: str | None = Field(None, description="Class name if applicable")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for code extraction"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the code snippet"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of extraction"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class Entity(BaseModel):
    """Represents an extracted semantic entity."""

    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_type: str = Field(
        ...,
        description="Type of entity (e.g., 'function', 'class', 'module', 'concept')",
    )
    name: str = Field(..., description="Name of the entity")
    description: str | None = Field(None, description="Description of the entity")
    source: str = Field(..., description="Source document or file")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Entity attributes and properties"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for entity extraction",
    )
    tags: list[str] = Field(default_factory=list, description="Semantic tags")
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of extraction"
    )


class SemanticRelationship(BaseModel):
    """Represents a semantic relationship between entities."""

    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(
        ...,
        description="Type of relationship (e.g., 'calls', 'implements', 'depends_on')",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for relationship"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship metadata"
    )


class ParsedProcedure(BaseModel):
    """Represents a parsed procedure with structured metadata."""

    procedure_id: str = Field(..., description="Unique identifier for the procedure")
    title: str = Field(..., description="Procedure title")
    description: str | None = Field(None, description="Procedure description")
    steps: list[str] = Field(default_factory=list, description="Procedure steps")
    preconditions: list[str] = Field(
        default_factory=list, description="Prerequisites for the procedure"
    )
    postconditions: list[str] = Field(
        default_factory=list, description="Expected outcomes after procedure"
    )
    code_snippets: list[CodeSnippet] = Field(
        default_factory=list, description="Associated code snippets"
    )
    entities: list[Entity] = Field(
        default_factory=list, description="Entities mentioned in the procedure"
    )
    tags: list[str] = Field(default_factory=list, description="Procedure tags")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for procedure parsing",
    )
    source_file: str | None = Field(None, description="Source file path")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional procedure metadata"
    )
    parsed_at: datetime = Field(default_factory=datetime.now, description="Timestamp of parsing")


class ParsedRequirement(BaseModel):
    """Represents a parsed requirement with structured metadata."""

    requirement_id: str = Field(..., description="Unique identifier for the requirement")
    title: str = Field(..., description="Requirement title")
    description: str = Field(..., description="Requirement description")
    requirement_type: str = Field(
        ...,
        description="Type of requirement (e.g., 'functional', 'non-functional', 'constraint')",
    )
    priority: str = Field(
        default="medium", description="Priority level (e.g., 'high', 'medium', 'low')"
    )
    status: str = Field(
        default="pending",
        description="Requirement status (e.g., 'pending', 'in-progress', 'completed')",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criteria for the requirement"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of dependent requirements"
    )
    code_snippets: list[CodeSnippet] = Field(
        default_factory=list, description="Associated code snippets"
    )
    entities: list[Entity] = Field(
        default_factory=list, description="Entities mentioned in the requirement"
    )
    tags: list[str] = Field(default_factory=list, description="Requirement tags")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for requirement parsing",
    )
    source_file: str | None = Field(None, description="Source file path")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional requirement metadata"
    )
    parsed_at: datetime = Field(default_factory=datetime.now, description="Timestamp of parsing")


class ParsedPrinciple(BaseModel):
    """Represents a parsed principle or guideline with structured metadata."""

    principle_id: str = Field(..., description="Unique identifier for the principle")
    title: str = Field(..., description="Principle title")
    description: str = Field(..., description="Principle description")
    category: str = Field(
        default="general",
        description="Category of principle (e.g., 'design', 'coding', 'architecture')",
    )
    rationale: str | None = Field(None, description="Rationale behind the principle")
    examples: list[str] = Field(
        default_factory=list, description="Examples demonstrating the principle"
    )
    code_snippets: list[CodeSnippet] = Field(
        default_factory=list, description="Associated code examples"
    )
    entities: list[Entity] = Field(
        default_factory=list, description="Entities related to the principle"
    )
    tags: list[str] = Field(default_factory=list, description="Principle tags")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for principle parsing",
    )
    source_file: str | None = Field(None, description="Source file path")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional principle metadata"
    )
    parsed_at: datetime = Field(default_factory=datetime.now, description="Timestamp of parsing")
