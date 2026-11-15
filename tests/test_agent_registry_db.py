"""Tests for agent registry database components (PostgreSQL).

This module tests SkillsDB and WorkloadDB with async fixtures.
Requires PostgreSQL connection pool to be initialized.
"""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from src.db.skills_db import (
    Skill,
    SkillLevel,
    SkillsDB,
    Workload,
    WorkloadDB,
)

# Test constants
TEST_CREATOR_ID = "test-creator-123"
TEST_WORKER_ID = "test-worker-456"
TEST_WORKER_ID_2 = "test-worker-789"


# Mock connection fixture
@pytest.fixture
def mock_pg_connection():
    """Create a mock PostgreSQL connection."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    return mock_conn


@pytest.fixture
def mock_get_pg_connection(mock_pg_connection):
    """Patch get_pg_connection to return mock connection."""
    with patch("src.db.skills_db.get_pg_connection") as mock:
        # Create async context manager
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_pg_connection
        async_cm.__aexit__.return_value = None
        mock.return_value = async_cm
        yield mock_pg_connection


# SkillsDB Tests
class TestSkillsDB:
    """Test suite for SkillsDB functionality."""

    @pytest.mark.asyncio
    async def test_add_skill_new(self, mock_get_pg_connection):
        """Test adding a new skill."""
        mock_get_pg_connection.fetchrow.return_value = None  # No existing skill

        skills_db = SkillsDB()
        skill = await skills_db.add_skill(
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            worker_id=TEST_WORKER_ID,
            skill_name="python",
            domains=["backend", "data"],
            skill_level="advanced",
        )

        assert skill.worker_id == TEST_WORKER_ID
        assert skill.skill_name == "python"
        assert skill.domains == ["backend", "data"]
        assert skill.skill_level == "advanced"
        assert skill.tenant_id == TEST_CREATOR_ID
        assert skill.created_at is not None

        # Verify INSERT was called
        mock_get_pg_connection.execute.assert_called_once()
        call_args = mock_get_pg_connection.execute.call_args[0]
        assert "INSERT INTO skills" in call_args[0]

    @pytest.mark.asyncio
    async def test_add_skill_upsert(self, mock_get_pg_connection):
        """Test that adding duplicate skill updates existing one (upsert)."""
        # Simulate existing skill found
        mock_get_pg_connection.fetchrow.return_value = {"id": 1}

        skills_db = SkillsDB()
        skill = await skills_db.add_skill(
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            worker_id=TEST_WORKER_ID,
            skill_name="python",
            domains=["backend", "ai"],
            skill_level="expert",
        )

        assert skill.skill_level == "expert"

        # Verify UPDATE was called
        mock_get_pg_connection.execute.assert_called_once()
        call_args = mock_get_pg_connection.execute.call_args[0]
        assert "UPDATE skills" in call_args[0]

    @pytest.mark.asyncio
    async def test_add_skill_validation(self, mock_get_pg_connection):
        """Test skill validation on add."""
        skills_db = SkillsDB()

        # Test empty skill name - Pydantic catches this
        with pytest.raises(Exception):  # ValidationError from Pydantic
            await skills_db.add_skill(
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
                worker_id=TEST_WORKER_ID,
                skill_name="",
                domains=["test"],
            )

        # Test null bytes in skill name
        with pytest.raises(ValueError, match="cannot contain null bytes"):
            await skills_db.add_skill(
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
                worker_id=TEST_WORKER_ID,
                skill_name="python\x00",
                domains=["test"],
            )

    @pytest.mark.asyncio
    async def test_get_skills(self, mock_get_pg_connection):
        """Test retrieving skills for a worker."""
        now = datetime.now(UTC)
        mock_get_pg_connection.fetch.return_value = [
            {
                "worker_id": TEST_WORKER_ID,
                "skill_name": "python",
                "skill_level": "advanced",
                "domains": ["backend"],
                "tenant_id": TEST_CREATOR_ID,
                "created_at": now,
                "updated_at": now,
            },
            {
                "worker_id": TEST_WORKER_ID,
                "skill_name": "javascript",
                "skill_level": "intermediate",
                "domains": ["frontend"],
                "tenant_id": TEST_CREATOR_ID,
                "created_at": now,
                "updated_at": now,
            },
        ]

        skills_db = SkillsDB()
        skills = await skills_db.get_skills(TEST_CREATOR_ID, TEST_WORKER_ID)

        assert len(skills) == 2
        skill_names = [s.skill_name for s in skills]
        assert "python" in skill_names
        assert "javascript" in skill_names

    @pytest.mark.asyncio
    async def test_get_skills_empty(self, mock_get_pg_connection):
        """Test getting skills for non-existent worker."""
        mock_get_pg_connection.fetch.return_value = []

        skills_db = SkillsDB()
        skills = await skills_db.get_skills(TEST_CREATOR_ID, "nonexistent")

        assert skills == []

    @pytest.mark.asyncio
    async def test_delete_skill(self, mock_get_pg_connection):
        """Test deleting a skill."""
        mock_get_pg_connection.execute.return_value = "DELETE 1"

        skills_db = SkillsDB()
        result = await skills_db.delete_skill(TEST_CREATOR_ID, TEST_WORKER_ID, "python")

        assert result is True
        mock_get_pg_connection.execute.assert_called_once()
        call_args = mock_get_pg_connection.execute.call_args[0]
        assert "DELETE FROM skills" in call_args[0]

    @pytest.mark.asyncio
    async def test_delete_skill_not_found(self, mock_get_pg_connection):
        """Test deleting non-existent skill."""
        mock_get_pg_connection.execute.return_value = "DELETE 0"

        skills_db = SkillsDB()
        result = await skills_db.delete_skill(TEST_CREATOR_ID, TEST_WORKER_ID, "nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_search_by_skill(self, mock_get_pg_connection):
        """Test searching workers by skill name."""
        mock_get_pg_connection.fetch.return_value = [
            {"worker_id": "worker1"},
            {"worker_id": "worker2"},
        ]

        skills_db = SkillsDB()
        workers = await skills_db.search_by_skill(TEST_CREATOR_ID, "python")

        assert len(workers) == 2
        assert "worker1" in workers
        assert "worker2" in workers

    @pytest.mark.asyncio
    async def test_search_by_skill_not_found(self, mock_get_pg_connection):
        """Test searching for non-existent skill."""
        mock_get_pg_connection.fetch.return_value = []

        skills_db = SkillsDB()
        workers = await skills_db.search_by_skill(TEST_CREATOR_ID, "nonexistent")

        assert workers == []


# WorkloadDB Tests
class TestWorkloadDB:
    """Test suite for WorkloadDB functionality."""

    @pytest.mark.asyncio
    async def test_init_workload(self, mock_get_pg_connection):
        """Test initializing workload for a worker."""
        mock_get_pg_connection.fetchrow.return_value = None  # No existing workload

        workload_db = WorkloadDB()
        workload = await workload_db.init_workload(
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            worker_id=TEST_WORKER_ID,
            capacity=10,
        )

        assert workload.worker_id == TEST_WORKER_ID
        assert workload.tenant_id == TEST_CREATOR_ID
        assert workload.current_tasks == 0
        assert workload.capacity == 10
        assert workload.availability == 1.0
        assert workload.last_updated is not None

        mock_get_pg_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_workload_default_capacity(self, mock_get_pg_connection):
        """Test initializing workload with default capacity."""
        mock_get_pg_connection.fetchrow.return_value = None

        workload_db = WorkloadDB()
        workload = await workload_db.init_workload(
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            worker_id=TEST_WORKER_ID,
        )

        assert workload.capacity == 5  # Default
        assert workload.current_tasks == 0
        assert workload.availability == 1.0

    @pytest.mark.asyncio
    async def test_init_workload_duplicate(self, mock_get_pg_connection):
        """Test that initializing duplicate workload raises error."""
        mock_get_pg_connection.fetchrow.return_value = {"id": 1}  # Exists

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="already exists"):
            await workload_db.init_workload(
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
                worker_id=TEST_WORKER_ID,
            )

    @pytest.mark.asyncio
    async def test_get_workload(self, mock_get_pg_connection):
        """Test retrieving workload for a worker."""
        now = datetime.now(UTC)
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 2,
            "capacity": 5,
            "availability": 0.6,
            "last_updated": now,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.get_workload(TEST_CREATOR_ID, TEST_WORKER_ID)

        assert workload is not None
        assert workload.worker_id == TEST_WORKER_ID
        assert workload.capacity == 5
        assert workload.current_tasks == 2

    @pytest.mark.asyncio
    async def test_get_workload_not_found(self, mock_get_pg_connection):
        """Test getting workload for non-existent worker."""
        mock_get_pg_connection.fetchrow.return_value = None

        workload_db = WorkloadDB()
        workload = await workload_db.get_workload(TEST_CREATOR_ID, "nonexistent")

        assert workload is None

    @pytest.mark.asyncio
    async def test_increment_tasks(self, mock_get_pg_connection):
        """Atomic increment returns the post-update row from UPDATE...RETURNING."""
        now = datetime.now(UTC)
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 2,
            "capacity": 5,
            "availability": 0.6,  # (5-2)/5, computed DB-side
            "last_updated": now,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.increment_tasks(TEST_CREATOR_ID, TEST_WORKER_ID)

        assert workload.current_tasks == 2
        assert workload.availability == 0.6
        # Single guarded UPDATE...RETURNING; no separate SELECT-then-UPDATE round-trip.
        mock_get_pg_connection.fetchrow.assert_awaited_once()
        mock_get_pg_connection.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_increment_tasks_uses_atomic_guarded_update(self, mock_get_pg_connection):
        """Regression guard for the read-then-write race: the increment must be a
        single ``UPDATE ... WHERE current_tasks < capacity ... RETURNING`` so two
        concurrent callers can never push a worker past its capacity."""
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 1,
            "capacity": 5,
            "availability": 0.8,
            "last_updated": None,
        }

        workload_db = WorkloadDB()
        await workload_db.increment_tasks(TEST_CREATOR_ID, TEST_WORKER_ID)

        sql = " ".join(mock_get_pg_connection.fetchrow.call_args[0][0].split()).lower()
        assert "update workload" in sql
        assert "current_tasks < capacity" in sql
        assert "returning" in sql
        mock_get_pg_connection.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_increment_tasks_at_capacity(self, mock_get_pg_connection):
        """At capacity: guarded UPDATE matches no row; existence probe finds it."""
        # 1st fetchrow = UPDATE...RETURNING (guard failed -> None);
        # 2nd fetchrow = _existing_or_not_found existence probe.
        mock_get_pg_connection.fetchrow.side_effect = [
            None,
            {"current_tasks": 5, "capacity": 5},
        ]

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="at capacity"):
            await workload_db.increment_tasks(TEST_CREATOR_ID, TEST_WORKER_ID)

    @pytest.mark.asyncio
    async def test_increment_tasks_not_found(self, mock_get_pg_connection):
        """Not found: both the UPDATE and the existence probe return None."""
        mock_get_pg_connection.fetchrow.return_value = None

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="not found"):
            await workload_db.increment_tasks(TEST_CREATOR_ID, "nonexistent")

    @pytest.mark.asyncio
    async def test_decrement_tasks(self, mock_get_pg_connection):
        """Atomic decrement returns the post-update row."""
        now = datetime.now(UTC)
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 1,
            "capacity": 5,
            "availability": 0.8,  # (5-1)/5, computed DB-side
            "last_updated": now,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.decrement_tasks(TEST_CREATOR_ID, TEST_WORKER_ID)

        assert workload.current_tasks == 1
        assert workload.availability == 0.8
        mock_get_pg_connection.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_decrement_tasks_at_zero(self, mock_get_pg_connection):
        """At zero: guarded UPDATE matches no row; existence probe finds it."""
        mock_get_pg_connection.fetchrow.side_effect = [
            None,
            {"current_tasks": 0, "capacity": 5},
        ]

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="no tasks"):
            await workload_db.decrement_tasks(TEST_CREATOR_ID, TEST_WORKER_ID)

    @pytest.mark.asyncio
    async def test_decrement_tasks_not_found(self, mock_get_pg_connection):
        """Test decrementing tasks for non-existent worker."""
        mock_get_pg_connection.fetchrow.return_value = None

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="not found"):
            await workload_db.decrement_tasks(TEST_CREATOR_ID, "nonexistent")

    @pytest.mark.asyncio
    async def test_update_capacity(self, mock_get_pg_connection):
        """Atomic capacity update returns the post-update row."""
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 0,
            "capacity": 10,
            "availability": 1.0,
            "last_updated": None,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.update_capacity(
            TEST_CREATOR_ID, TEST_WORKER_ID, new_capacity=10
        )

        assert workload.capacity == 10
        assert workload.current_tasks == 0
        assert workload.availability == 1.0
        mock_get_pg_connection.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_capacity_with_tasks(self, mock_get_pg_connection):
        """Test updating capacity when worker has active tasks."""
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 2,
            "capacity": 5,
            "availability": 0.6,  # (5-2)/5, computed DB-side
            "last_updated": None,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.update_capacity(
            TEST_CREATOR_ID, TEST_WORKER_ID, new_capacity=5
        )

        assert workload.capacity == 5
        assert workload.current_tasks == 2
        assert workload.availability == 0.6

    @pytest.mark.asyncio
    async def test_update_capacity_below_current_tasks(self, mock_get_pg_connection):
        """Capacity cannot be set below current tasks: guarded UPDATE matches no row."""
        mock_get_pg_connection.fetchrow.side_effect = [
            None,
            {"current_tasks": 3, "capacity": 10},
        ]

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="Cannot reduce capacity"):
            await workload_db.update_capacity(TEST_CREATOR_ID, TEST_WORKER_ID, new_capacity=2)

    @pytest.mark.asyncio
    async def test_update_capacity_invalid(self, mock_get_pg_connection):
        """Test updating capacity with invalid values."""
        workload_db = WorkloadDB()

        # Test capacity < 1
        with pytest.raises(ValueError, match="at least 1"):
            await workload_db.update_capacity(TEST_CREATOR_ID, TEST_WORKER_ID, new_capacity=0)

        with pytest.raises(ValueError, match="at least 1"):
            await workload_db.update_capacity(TEST_CREATOR_ID, TEST_WORKER_ID, new_capacity=-5)

    @pytest.mark.asyncio
    async def test_update_capacity_not_found(self, mock_get_pg_connection):
        """Test updating capacity for non-existent worker."""
        mock_get_pg_connection.fetchrow.return_value = None

        workload_db = WorkloadDB()
        with pytest.raises(ValueError, match="not found"):
            await workload_db.update_capacity(TEST_CREATOR_ID, "nonexistent", new_capacity=10)

    @pytest.mark.asyncio
    async def test_get_or_create_workload_existing(self, mock_get_pg_connection):
        """Test get_or_create returns existing workload."""
        now = datetime.now(UTC)
        mock_get_pg_connection.fetchrow.return_value = {
            "worker_id": TEST_WORKER_ID,
            "tenant_id": TEST_CREATOR_ID,
            "current_tasks": 2,
            "capacity": 5,
            "availability": 0.6,
            "last_updated": now,
        }

        workload_db = WorkloadDB()
        workload = await workload_db.get_or_create_workload(TEST_CREATOR_ID, TEST_WORKER_ID)

        assert workload.current_tasks == 2  # Existing values preserved

    @pytest.mark.asyncio
    async def test_get_or_create_workload_new(self, mock_get_pg_connection):
        """Test get_or_create creates new workload if not exists."""
        # First call returns None (not found), second call is for INSERT check
        mock_get_pg_connection.fetchrow.side_effect = [None, None]

        workload_db = WorkloadDB()
        workload = await workload_db.get_or_create_workload(
            TEST_CREATOR_ID, TEST_WORKER_ID, capacity=8
        )

        assert workload.capacity == 8
        assert workload.current_tasks == 0


# Pydantic Model Tests
class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_skill_model_validation(self):
        """Test Skill model validation."""
        # Valid skill
        skill = Skill(
            worker_id=TEST_WORKER_ID,
            skill_name="python",
            skill_level="advanced",
            domains=["backend"],
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
        )
        assert skill.worker_id == TEST_WORKER_ID
        assert skill.skill_name == "python"
        assert skill.skill_level == "advanced"

        # Test invalid skill_level - should be one of beginner/intermediate/advanced/expert
        with pytest.raises(ValueError):
            Skill(
                worker_id=TEST_WORKER_ID,
                skill_name="python",
                # Intentionally invalid; cast to satisfy ty so the runtime
                # validator (pydantic) is the thing under test.
                skill_level=cast("SkillLevel", "invalid_level"),
                domains=["backend"],
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
            )

    def test_workload_model_validation(self):
        """Test Workload model validation."""
        # Valid workload
        workload = Workload(
            worker_id=TEST_WORKER_ID,
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            current_tasks=3,
            capacity=5,
        )
        assert workload.worker_id == TEST_WORKER_ID
        assert workload.current_tasks == 3

        # Test negative current_tasks
        with pytest.raises(ValueError):
            Workload(
                worker_id=TEST_WORKER_ID,
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
                current_tasks=-1,
                capacity=5,
            )

        # Test invalid capacity
        with pytest.raises(ValueError):
            Workload(
                worker_id=TEST_WORKER_ID,
                tenant_id=TEST_CREATOR_ID,
                subject_id=TEST_CREATOR_ID,
                current_tasks=0,
                capacity=0,  # < 1
            )

    def test_workload_calculate_availability(self):
        """Test availability calculation method."""
        workload = Workload(
            worker_id=TEST_WORKER_ID,
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            current_tasks=2,
            capacity=10,
        )
        assert workload.calculate_availability() == 0.8

        workload = Workload(
            worker_id=TEST_WORKER_ID,
            tenant_id=TEST_CREATOR_ID,
            subject_id=TEST_CREATOR_ID,
            current_tasks=5,
            capacity=5,
        )
        assert workload.calculate_availability() == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys

    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
    sys.exit(result.returncode)
