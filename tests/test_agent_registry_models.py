"""Unit tests for models/agent_registry.py Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models.agent_registry import (
    Agent,
    AgentFilter,
    AgentListResponse,
    AgentResponse,
    AgentStatus,
    Skill,
    SkillLevel,
    SkillsResponse,
    StatusUpdateRequest,
    StatusUpdateResponse,
    Workload,
    WorkloadResponse,
)


class TestSkill:
    def test_valid_skill(self):
        s = Skill(name="python", level=SkillLevel.EXPERT, domains=["backend"])
        assert s.name == "python"
        assert s.level == "expert"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Skill(name="   ", level=SkillLevel.BEGINNER)

    def test_strip_whitespace(self):
        s = Skill(name="  python  ", level=SkillLevel.INTERMEDIATE)
        assert s.name == "python"


class TestWorkload:
    def test_valid_workload(self):
        w = Workload(current_tasks=2, capacity=5, availability=0.6)
        assert w.current_tasks == 2
        assert w.availability == 0.6

    def test_negative_tasks_rejected(self):
        with pytest.raises(ValidationError):
            Workload(current_tasks=-1, capacity=5, availability=0.5)

    def test_zero_capacity_rejected(self):
        with pytest.raises(ValidationError):
            Workload(current_tasks=0, capacity=0, availability=0.5)

    def test_availability_out_of_range(self):
        with pytest.raises(ValidationError):
            Workload(current_tasks=0, capacity=5, availability=1.5)

    def test_availability_rounding(self):
        w = Workload(current_tasks=0, capacity=5, availability=0.333)
        assert w.availability == 0.33


class TestAgent:
    def test_valid_agent(self):
        a = Agent(
            id="agent-1",
            name="Test Agent",
            status=AgentStatus.AVAILABLE,
            capacity=10,
        )
        assert a.id == "agent-1"
        assert a.status == AgentStatus.AVAILABLE

    def test_with_skills_and_workload(self):
        a = Agent(
            id="a-2",
            name="Worker",
            skills=[Skill(name="coding", level=SkillLevel.ADVANCED)],
            status=AgentStatus.BUSY,
            capacity=5,
            current_workload=Workload(current_tasks=3, capacity=5, availability=0.4),
        )
        assert len(a.skills) == 1

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Agent(id="a-1", name="", status=AgentStatus.OFFLINE, capacity=1)


class TestAgentFilter:
    def test_empty_filter(self):
        f = AgentFilter()
        assert f.skills is None
        assert f.min_availability is None

    def test_with_filters(self):
        f = AgentFilter(
            skills=["python"],
            min_skill_level=SkillLevel.INTERMEDIATE,
            status=[AgentStatus.AVAILABLE],
            min_availability=0.5,
            min_capacity=3,
            domains=["backend"],
        )
        assert f.skills == ["python"]
        assert f.min_capacity == 3


class TestAgentListResponse:
    def test_valid_response(self):
        agent = Agent(id="a-1", name="A", status=AgentStatus.AVAILABLE, capacity=5)
        resp = AgentListResponse(agents=[agent], total=1)
        assert resp.total == 1

    def test_mismatched_total_rejected(self):
        agent = Agent(id="a-1", name="A", status=AgentStatus.AVAILABLE, capacity=5)
        with pytest.raises(ValidationError):
            AgentListResponse(agents=[agent], total=5)


class TestAgentResponse:
    def test_valid(self):
        agent = Agent(id="a-1", name="A", status=AgentStatus.OFFLINE, capacity=1)
        resp = AgentResponse(agent=agent)
        assert resp.agent.id == "a-1"


class TestSkillsResponse:
    def test_valid(self):
        resp = SkillsResponse(skills=[Skill(name="test", level=SkillLevel.BEGINNER)])
        assert len(resp.skills) == 1


class TestWorkloadResponse:
    def test_valid(self):
        resp = WorkloadResponse(workload=Workload(current_tasks=0, capacity=10, availability=1.0))
        assert resp.workload.capacity == 10


class TestStatusUpdateRequest:
    def test_valid(self):
        req = StatusUpdateRequest(status=AgentStatus.MAINTENANCE)
        assert req.status == "maintenance"


class TestStatusUpdateResponse:
    def test_valid(self):
        resp = StatusUpdateResponse(success=True, agent_id="a-1", new_status=AgentStatus.AVAILABLE)
        assert resp.success is True
