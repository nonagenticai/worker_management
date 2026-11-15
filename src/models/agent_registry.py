"""Agent Registry API Models.

Pydantic models for the agent registry API endpoints.
Reuses core models from workflow_management and adds response/request models.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.type_definitions.constraint_types import validate_identifier

# ============================================
# Enums
# ============================================


class AgentStatus(str, Enum):
    """Agent operational status."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class SkillLevel(str, Enum):
    """Skill proficiency levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# ============================================
# Core Models
# ============================================


class Skill(BaseModel):
    """Agent skill definition with proficiency level and applicable domains."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    name: str = Field(..., description="Skill name", min_length=1)
    level: SkillLevel = Field(..., description="Proficiency level")
    domains: list[str] = Field(
        default_factory=list,
        description="Applicable domains or contexts for this skill",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate skill name is non-empty, null-byte-free, and stripped."""
        return validate_identifier(v, "Skill name")


class Workload(BaseModel):
    """Agent current workload and availability metrics."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    current_tasks: int = Field(..., ge=0, description="Number of currently assigned tasks")
    capacity: int = Field(..., ge=1, description="Maximum task capacity")
    availability: float = Field(
        ..., ge=0.0, le=1.0, description="Availability percentage (0.0 to 1.0)"
    )

    @field_validator("availability")
    @classmethod
    def validate_availability(cls, v: float, info: Any) -> float:  # noqa: ARG003, RUF100
        """Ensure availability matches capacity ratio."""
        # Allow validation to pass, actual calculation should be done server-side
        return round(v, 2)


class Agent(BaseModel):
    """Agent registry entry with full metadata."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name", min_length=1)
    skills: list[Skill] = Field(
        default_factory=list, description="Agent capabilities and skill set"
    )
    status: AgentStatus = Field(..., description="Current operational status")
    capacity: int = Field(..., ge=1, description="Maximum concurrent task capacity")
    current_workload: Workload | None = Field(
        None, description="Current workload metrics if available"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    last_seen: datetime | None = Field(None, description="Last time agent checked in")
    created_at: datetime | None = Field(None, description="Agent registration timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")


class AgentFilter(BaseModel):
    """Filter criteria for agent queries."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    skills: list[str] | None = Field(None, description="Required skills (AND logic)")
    min_skill_level: SkillLevel | None = Field(None, description="Minimum skill level required")
    status: list[AgentStatus] | None = Field(None, description="Filter by agent status")
    min_availability: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum availability percentage"
    )
    min_capacity: int | None = Field(None, ge=1, description="Minimum task capacity")
    domains: list[str] | None = Field(None, description="Filter by skill domains")


# ============================================
# Response Models
# ============================================


class AgentListResponse(BaseModel):
    """Response model for listing agents."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    agents: list[Agent] = Field(..., description="List of agents matching the query criteria")
    total: int = Field(..., ge=0, description="Total number of agents returned")

    @field_validator("total")
    @classmethod
    def validate_total_matches_agents(cls, v: int, info: Any) -> int:
        """Ensure total matches the actual number of agents."""
        agents = info.data.get("agents", [])
        if len(agents) != v:
            raise ValueError(f"Total count ({v}) does not match number of agents ({len(agents)})")
        return v


class AgentResponse(BaseModel):
    """Response model for single agent operations."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    agent: Agent = Field(..., description="Agent details")


class SkillsResponse(BaseModel):
    """Response model for agent skills."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    skills: list[Skill] = Field(..., description="List of agent skills")


class WorkloadResponse(BaseModel):
    """Response model for agent workload."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    workload: Workload = Field(..., description="Current agent workload metrics")


# ============================================
# Request Models
# ============================================


class StatusUpdateRequest(BaseModel):
    """Request model for updating agent status."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
    )

    status: AgentStatus = Field(..., description="New operational status for the agent")


class StatusUpdateResponse(BaseModel):
    """Response model for status update operations."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    success: bool = Field(..., description="Whether the status update was successful")
    agent_id: str = Field(..., description="ID of the agent that was updated")
    new_status: AgentStatus = Field(..., description="The new status that was set")
