"""Agent registry database module with skills and workload management.

This module provides database operations for managing worker skills and workloads
using PostgreSQL. All operations are scoped by (tenant_id, subject_id) for
per-user-within-tenant multi-tenancy.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.tenant_constants import tenant_label
from src.db.db_config import get_pg_connection
from src.db.workers_state_db import with_retry
from src.type_definitions.constraint_types import validate_identifier

logger = logging.getLogger(__name__)

# Skill level enum matching database constraint
SkillLevel = Literal["beginner", "intermediate", "advanced", "expert"]


def _effective_subject(subject_id: str, tenant_id: str) -> str:
    """Resolve the subject used for per-user scoping, defaulting to the tenant."""
    return subject_id or tenant_id


def _backfill_scope_fields(model: BaseModel) -> None:
    """Default ``subject_id`` to ``tenant_id`` when unset (per-user-within-tenant).

    Shared by :class:`Skill` and :class:`Workload`. A row with no explicit subject
    is tenant-scoped, so the subject defaults to the tenant id.
    """
    tenant_id = getattr(model, "tenant_id", "")
    subject_id = getattr(model, "subject_id", "")
    if not subject_id:
        object.__setattr__(model, "subject_id", tenant_id)


class Skill(BaseModel):
    """Skill model with validation."""

    worker_id: str = Field(..., min_length=1, max_length=200)
    skill_name: str = Field(..., min_length=1, max_length=200)
    skill_level: SkillLevel = Field(default="intermediate")
    domains: list[str] = Field(default_factory=list)
    # tenant_id / subject_id are the canonical scoping fields.
    tenant_id: str = Field(default="", min_length=0, max_length=200)
    subject_id: str = Field(default="", min_length=0, max_length=200)
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode="after")
    def _backfill_compat_fields(self) -> Skill:
        """Default subject_id to tenant_id when unset."""
        _backfill_scope_fields(self)
        return self

    @field_validator("skill_name")
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        """Validate skill name."""
        return validate_identifier(v, "Skill name")


class Workload(BaseModel):
    """Workload model with validation."""

    worker_id: str = Field(..., min_length=1, max_length=200)
    # tenant_id / subject_id are the canonical scoping fields.
    tenant_id: str = Field(default="", min_length=0, max_length=200)
    subject_id: str = Field(default="", min_length=0, max_length=200)
    current_tasks: int = Field(default=0, ge=0)
    capacity: int = Field(default=5, ge=1)
    availability: float | None = Field(default=None, ge=0.0, le=1.0)
    last_updated: str | None = None

    @model_validator(mode="after")
    def _backfill_compat_fields(self) -> Workload:
        """Default subject_id to tenant_id when unset."""
        _backfill_scope_fields(self)
        return self

    def calculate_availability(self) -> float:
        """Calculate availability based on current tasks and capacity."""
        if self.capacity == 0:
            return 0.0
        return max(0.0, (self.capacity - self.current_tasks) / self.capacity)


def _row_to_workload(row: Any) -> Workload:
    """Build a :class:`Workload` from a DB row (shared by read/mutate paths)."""
    availability = float(row["availability"]) if row["availability"] is not None else None
    return Workload(
        worker_id=str(row["worker_id"]),
        tenant_id=row["tenant_id"],
        current_tasks=row["current_tasks"] or 0,
        capacity=row["capacity"] or 5,
        availability=availability,
        last_updated=(row["last_updated"].isoformat() if row["last_updated"] else None),
    )


class SkillsDB:
    """API for managing worker skills in the PostgreSQL database."""

    @with_retry()
    async def add_skill(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        worker_id: str,
        skill_name: str,
        domains: list[str],
        skill_level: SkillLevel = "intermediate",
        visibility: str = "tenant",
    ) -> Skill:
        """Add or update a skill for a worker (upsert behavior).

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            subject_id: Subject (user) identifier for per-user scoping
            tenant_name: Human-readable tenant name (defaults to tenant_id)
            worker_id: The worker identifier
            skill_name: The skill name
            domains: List of domain strings
            skill_level: Skill level (beginner/intermediate/advanced/expert)
            visibility: Row visibility ('tenant' | 'user')

        Returns:
            The created or updated Skill
        """
        effective_subject = _effective_subject(subject_id, tenant_id)
        skill = Skill(
            worker_id=worker_id,
            skill_name=skill_name,
            skill_level=skill_level,
            domains=domains,
            tenant_id=tenant_id,
            subject_id=effective_subject,
        )

        # ⚠️ `domains` MUST carry an explicit `::text[]` cast on every write.
        # Production is `domains text[] NOT NULL DEFAULT '{}'::text[]` (measured
        # on the production primary). The asyncpg-compat shim in
        # db_config.py disambiguates a bare Python list ONLY by looking for an
        # array cast on the placeholder (`_array_param_indices`); without one it
        # wraps the list in `Jsonb()`, and postgres then rejects the statement
        # with `column "domains" is of type text[] but expression is of type
        # jsonb`. Same defect class as `tools.tags` (migrations/002_fix_tool_tags.sql).
        async with get_pg_connection() as conn:
            now = datetime.now(UTC)

            # Check if skill exists (scoped to tenant + worker)
            existing = await conn.fetchrow(
                """
                SELECT id FROM skills
                WHERE tenant_id = $1 AND worker_id = $2 AND skill_name = $3
                """,
                tenant_id,
                worker_id,
                skill_name,
            )

            if existing:
                # Update existing skill
                await conn.execute(
                    """
                    UPDATE skills
                    SET skill_level = $1, domains = $2::text[], updated_at = $3
                    WHERE tenant_id = $4 AND worker_id = $5 AND skill_name = $6
                    """,
                    skill_level,
                    domains,
                    now,
                    tenant_id,
                    worker_id,
                    skill_name,
                )
                logger.info(
                    "Updated skill: tenant_id=%s, worker_id=%s, skill=%s",
                    tenant_id,
                    worker_id,
                    skill_name,
                )
            else:
                # Insert new skill
                await conn.execute(
                    """
                    INSERT INTO skills
                    (worker_id, skill_name, skill_level, domains,
                     tenant_id, subject_id, visibility, tenant_name,
                     created_at, updated_at)
                    VALUES ($1, $2, $3, $4::text[], $5, $6, $7, $8, $9, $10)
                    """,
                    worker_id,
                    skill_name,
                    skill_level,
                    domains,
                    tenant_id,
                    effective_subject,
                    visibility,
                    tenant_label(tenant_id, tenant_name),
                    now,
                    now,
                )
                logger.info(
                    "Added skill: tenant_id=%s, worker_id=%s, skill=%s",
                    tenant_id,
                    worker_id,
                    skill_name,
                )

            skill.created_at = now.isoformat()
            skill.updated_at = now.isoformat()

        return skill

    @with_retry()
    async def get_skills(self, tenant_id: str, worker_id: str) -> list[Skill]:
        """Get all skills for a worker.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier

        Returns:
            List of Skill objects
        """
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT worker_id, skill_name, skill_level, domains, tenant_id, created_at, updated_at
                FROM skills
                WHERE tenant_id = $1 AND worker_id = $2
                ORDER BY skill_name
                """,
                tenant_id,
                worker_id,
            )

            skills = []
            for row in rows:
                skills.append(
                    Skill(
                        worker_id=str(row["worker_id"]),
                        skill_name=row["skill_name"],
                        skill_level=row["skill_level"],
                        domains=list(row["domains"]) if row["domains"] else [],
                        tenant_id=row["tenant_id"],
                        created_at=(row["created_at"].isoformat() if row["created_at"] else None),
                        updated_at=(row["updated_at"].isoformat() if row["updated_at"] else None),
                    )
                )

            return skills

    @with_retry()
    async def delete_skill(self, tenant_id: str, worker_id: str, skill_name: str) -> bool:
        """Delete a specific skill for a worker.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier
            skill_name: The skill name to delete

        Returns:
            True if skill was deleted, False if not found
        """
        async with get_pg_connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM skills
                WHERE tenant_id = $1 AND worker_id = $2 AND skill_name = $3
                """,
                tenant_id,
                worker_id,
                skill_name,
            )

            deleted_count = int(result.split()[-1]) if result else 0

            if deleted_count > 0:
                logger.info(
                    "Deleted skill: tenant_id=%s, worker_id=%s, skill=%s",
                    tenant_id,
                    worker_id,
                    skill_name,
                )
                return True

            return False

    @with_retry()
    async def search_by_skill(self, tenant_id: str, skill_name: str) -> list[str]:
        """Search for workers with a specific skill.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            skill_name: The skill name to search for

        Returns:
            List of worker IDs that have the skill
        """
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT worker_id
                FROM skills
                WHERE tenant_id = $1 AND skill_name = $2
                ORDER BY worker_id
                """,
                tenant_id,
                skill_name,
            )

            return [str(row["worker_id"]) for row in rows]


class WorkloadDB:
    """API for managing worker workloads in the PostgreSQL database."""

    @with_retry()
    async def init_workload(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        tenant_name: str = "",
        worker_id: str,
        capacity: int = 5,
        visibility: str = "tenant",
    ) -> Workload:
        """Initialize workload for a worker.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            subject_id: Subject (user) identifier for per-user scoping
            worker_id: The worker identifier
            capacity: Maximum task capacity (default: 5)
            visibility: Row visibility ('tenant' | 'user')

        Returns:
            The initialized Workload

        Raises:
            ValueError: If workload already exists for worker
        """
        effective_subject = _effective_subject(subject_id, tenant_id)

        async with get_pg_connection() as conn:
            now = datetime.now(UTC)

            # Check if workload exists
            existing = await conn.fetchrow(
                """
                SELECT id FROM workload
                WHERE tenant_id = $1 AND worker_id = $2
                """,
                tenant_id,
                worker_id,
            )

            if existing:
                raise ValueError(f"Workload already exists for worker {worker_id}")

            availability = 1.0  # No tasks yet

            await conn.execute(
                """
                INSERT INTO workload
                (worker_id, tenant_id, subject_id, visibility, tenant_name,
                 current_tasks, capacity, availability, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                worker_id,
                tenant_id,
                effective_subject,
                visibility,
                tenant_label(tenant_id, tenant_name),
                0,
                capacity,
                availability,
                now,
            )

            logger.info(
                "Initialized workload: tenant_id=%s, worker_id=%s, capacity=%d",
                tenant_id,
                worker_id,
                capacity,
            )

            return Workload(
                worker_id=worker_id,
                tenant_id=tenant_id,
                subject_id=effective_subject,
                current_tasks=0,
                capacity=capacity,
                availability=availability,
                last_updated=now.isoformat(),
            )

    @with_retry()
    async def get_workload(self, tenant_id: str, worker_id: str) -> Workload | None:
        """Get workload for a worker.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier

        Returns:
            Workload object or None if not found
        """
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT worker_id, tenant_id, current_tasks, capacity, availability, last_updated
                FROM workload
                WHERE tenant_id = $1 AND worker_id = $2
                """,
                tenant_id,
                worker_id,
            )

            if not row:
                return None

            return _row_to_workload(row)

    @staticmethod
    async def _existing_or_not_found(conn: Any, tenant_id: str, worker_id: str) -> Any:
        """Return the current workload row, or raise the 'not found' ValueError.

        Used after an atomic guarded UPDATE returns no row to distinguish a
        missing workload from a guard violation (capacity / zero-tasks).
        """
        existing = await conn.fetchrow(
            """
            SELECT current_tasks, capacity
            FROM workload
            WHERE tenant_id = $1 AND worker_id = $2
            """,
            tenant_id,
            worker_id,
        )
        if existing is None:
            raise ValueError(f"Workload not found for worker {worker_id}")
        return existing

    @with_retry()
    async def increment_tasks(self, tenant_id: str, worker_id: str) -> Workload:
        """Increment current tasks for a worker and recalculate availability.

        The check-and-increment is a single atomic ``UPDATE ... WHERE
        current_tasks < capacity`` so concurrent callers can never push a worker
        past its capacity (the prior read-then-write left a race window).

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier

        Returns:
            Updated Workload

        Raises:
            ValueError: If workload not found or at capacity
        """
        now = datetime.now(UTC)
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workload
                SET current_tasks = current_tasks + 1,
                    availability = GREATEST(
                        0.0,
                        (capacity - (current_tasks + 1))::float8 / capacity::float8
                    ),
                    last_updated = $3
                WHERE tenant_id = $1 AND worker_id = $2 AND current_tasks < capacity
                RETURNING worker_id, tenant_id, current_tasks, capacity, availability, last_updated
                """,
                tenant_id,
                worker_id,
                now,
            )

            if row is None:
                existing = await self._existing_or_not_found(conn, tenant_id, worker_id)
                raise ValueError(
                    f"Worker {worker_id} is at capacity ({existing['capacity']} tasks)"
                )

            logger.info(
                "Incremented tasks: tenant_id=%s, worker_id=%s, tasks=%d/%d",
                tenant_id,
                worker_id,
                row["current_tasks"],
                row["capacity"],
            )
            return _row_to_workload(row)

    @with_retry()
    async def decrement_tasks(self, tenant_id: str, worker_id: str) -> Workload:
        """Decrement current tasks for a worker and recalculate availability.

        Atomic guarded UPDATE (``WHERE current_tasks > 0``) so concurrent
        callers cannot drive ``current_tasks`` negative.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier

        Returns:
            Updated Workload

        Raises:
            ValueError: If workload not found or already at 0 tasks
        """
        now = datetime.now(UTC)
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workload
                SET current_tasks = current_tasks - 1,
                    availability = GREATEST(
                        0.0,
                        (capacity - (current_tasks - 1))::float8 / capacity::float8
                    ),
                    last_updated = $3
                WHERE tenant_id = $1 AND worker_id = $2 AND current_tasks > 0
                RETURNING worker_id, tenant_id, current_tasks, capacity, availability, last_updated
                """,
                tenant_id,
                worker_id,
                now,
            )

            if row is None:
                await self._existing_or_not_found(conn, tenant_id, worker_id)
                raise ValueError(f"Worker {worker_id} has no tasks to decrement")

            logger.info(
                "Decremented tasks: tenant_id=%s, worker_id=%s, tasks=%d/%d",
                tenant_id,
                worker_id,
                row["current_tasks"],
                row["capacity"],
            )
            return _row_to_workload(row)

    @with_retry()
    async def update_capacity(self, tenant_id: str, worker_id: str, new_capacity: int) -> Workload:
        """Update capacity for a worker and recalculate availability.

        Atomic guarded UPDATE (``WHERE current_tasks <= $new_capacity``) so a
        concurrent increment cannot slip the worker over the new capacity
        between the check and the write.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            worker_id: The worker identifier
            new_capacity: New capacity value (must be >= 1)

        Returns:
            Updated Workload

        Raises:
            ValueError: If workload not found, invalid capacity, or current_tasks > new_capacity
        """
        if new_capacity < 1:
            raise ValueError("Capacity must be at least 1")

        now = datetime.now(UTC)
        async with get_pg_connection() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workload
                SET capacity = $3,
                    availability = GREATEST(
                        0.0,
                        ($3 - current_tasks)::float8 / $3::float8
                    ),
                    last_updated = $4
                WHERE tenant_id = $1 AND worker_id = $2 AND current_tasks <= $3
                RETURNING worker_id, tenant_id, current_tasks, capacity, availability, last_updated
                """,
                tenant_id,
                worker_id,
                new_capacity,
                now,
            )

            if row is None:
                existing = await self._existing_or_not_found(conn, tenant_id, worker_id)
                raise ValueError(
                    f"Cannot reduce capacity to {new_capacity}: "
                    f"worker has {existing['current_tasks']} active tasks"
                )

            logger.info(
                "Updated capacity: tenant_id=%s, worker_id=%s, capacity=%d",
                tenant_id,
                worker_id,
                new_capacity,
            )
            return _row_to_workload(row)

    async def get_or_create_workload(
        self,
        tenant_id: str,
        worker_id: str,
        capacity: int = 5,
    ) -> Workload:
        """Get existing workload or create new one if not exists.

        Args:
            tenant_id: Tenant identifier (used as both tenant_id and subject_id
                for backward-compat callers that haven't migrated to the new
                (tenant_id, subject_id) API)
            worker_id: The worker identifier
            capacity: Default capacity if creating new (default: 5)

        Returns:
            Existing or newly created Workload
        """
        workload = await self.get_workload(tenant_id, worker_id)
        if workload:
            return workload
        return await self.init_workload(
            tenant_id=tenant_id,
            subject_id=tenant_id,
            worker_id=worker_id,
            capacity=capacity,
        )
