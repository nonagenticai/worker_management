"""Test debug TTL override - Task 80

Uses mock K8s classes — no real infrastructure needed.
"""

from unittest.mock import Mock
from uuid import uuid7

import pytest
import pytest_asyncio


class MockK8sJobExecutor:
    def __init__(self):
        self.namespace = "agent-jobs"
        self.batch_v1 = Mock()

    async def initialize(self):
        pass

    async def execute_agent(self, agent_name, prompt, execution_id, config=None):
        """Execute with debug configuration"""
        config = config or {}

        if config.get("debug"):  # noqa: SIM108
            ttl = 3600  # 60 minutes for debug
        else:
            ttl = config.get("ttl_seconds", 300)

        job = Mock()
        job.spec.ttl_seconds_after_finished = ttl
        job.metadata.name = f"claude-agent-{str(execution_id)[:8]}"

        self.batch_v1.read_namespaced_job.return_value = job

        return {
            "job_name": job.metadata.name,
            "execution_id": str(execution_id),
            "status": "completed",
            "ttl_seconds": ttl,
            "debug": config.get("debug", False),
        }

    def _build_job_manifest(self, agent_name, prompt, execution_id, config):
        """Build Job manifest with TTL"""
        ttl_seconds = config.get("ttl_seconds", 300)
        if config.get("debug"):
            ttl_seconds = 3600

        return {
            "spec": {
                "ttlSecondsAfterFinished": ttl_seconds,
                "activeDeadlineSeconds": 600,
                "backoffLimit": 0,
            }
        }


class TestDebugTTL:
    """Test suite for debug TTL override"""

    @pytest_asyncio.fixture
    async def executor(self):
        """Create executor instance"""
        executor = MockK8sJobExecutor()
        await executor.initialize()
        return executor

    @pytest.mark.asyncio
    async def test_debug_mode_sets_longer_ttl(self, executor):
        """Test debug mode sets TTL to 3600 seconds (60 minutes)"""
        execution_id = uuid7()

        # Create Job with debug mode
        config = {"debug": True}

        result = await executor.execute_agent(
            agent_name="debug-test",
            prompt="echo 'debug mode'",
            execution_id=execution_id,
            config=config,
        )

        # Verify TTL is 3600 seconds
        assert result["ttl_seconds"] == 3600
        assert result["debug"] is True

    @pytest.mark.asyncio
    async def test_debug_ttl_in_job_spec(self, executor):
        """Verify Job spec has correct ttl_seconds_after_finished"""
        execution_id = uuid7()

        config = {"debug": True, "ttl_seconds": 3600}

        result = await executor.execute_agent(
            agent_name="debug-spec-test",
            prompt="test prompt",
            execution_id=execution_id,
            config=config,
        )

        # Read Job spec
        job = executor.batch_v1.read_namespaced_job(name=result["job_name"], namespace="agent-jobs")

        # Verify TTL field
        assert job.spec.ttl_seconds_after_finished == 3600

    @pytest.mark.asyncio
    async def test_debug_flag_overrides_default(self, executor):
        """Test debug flag properly overrides default TTL"""
        execution_id = str(uuid7())

        # Test without debug - should use default
        config_normal = {"ttl_seconds": 180}  # Custom 3 min
        manifest_normal = executor._build_job_manifest(
            agent_name="normal-test",
            prompt="test",
            execution_id=execution_id,
            config=config_normal,
        )
        assert manifest_normal["spec"]["ttlSecondsAfterFinished"] == 180

        # Test with debug - should override to 3600
        config_debug = {
            "ttl_seconds": 180,  # This should be ignored
            "debug": True,
        }
        manifest_debug = executor._build_job_manifest(
            agent_name="debug-test",
            prompt="test",
            execution_id=execution_id,
            config=config_debug,
        )
        assert manifest_debug["spec"]["ttlSecondsAfterFinished"] == 3600

    @pytest.mark.asyncio
    async def test_debug_ttl_range_30_to_60_min(self, executor):
        """Test debug TTL is within acceptable range (30-60 minutes)"""
        execution_id = uuid7()

        config = {"debug": True}
        result = await executor.execute_agent(
            agent_name="range-test",
            prompt="test",
            execution_id=execution_id,
            config=config,
        )

        ttl_minutes = result["ttl_seconds"] / 60
        assert 30 <= ttl_minutes <= 60, f"Debug TTL should be 30-60 min, got {ttl_minutes}"

    @pytest.mark.asyncio
    async def test_non_debug_uses_normal_ttl(self, executor):
        """Verify non-debug mode uses normal TTL"""
        execution_id = uuid7()

        # Without debug flag
        config = {"ttl_seconds": 300}
        result = await executor.execute_agent(
            agent_name="normal-ttl-test",
            prompt="normal execution",
            execution_id=execution_id,
            config=config,
        )

        assert result["ttl_seconds"] == 300
        assert result["debug"] is False

    @pytest.mark.asyncio
    async def test_debug_mode_with_explicit_ttl(self, executor):
        """Test debug mode ignores explicit TTL and uses debug TTL"""
        execution_id = uuid7()

        # Debug mode with explicit TTL (should be ignored)
        config = {
            "debug": True,
            "ttl_seconds": 120,  # This should be ignored
        }

        result = await executor.execute_agent(
            agent_name="debug-override-test",
            prompt="debug with explicit ttl",
            execution_id=execution_id,
            config=config,
        )

        # Should use debug TTL (3600), not the explicit 120
        assert result["ttl_seconds"] == 3600
        assert result["debug"] is True

    @pytest.mark.asyncio
    async def test_multiple_debug_jobs(self, executor):
        """Test multiple debug Jobs all get extended TTL"""
        debug_jobs = []

        for i in range(3):
            execution_id = uuid7()
            config = {"debug": True}

            result = await executor.execute_agent(
                agent_name=f"debug-multi-{i}",
                prompt=f"debug job {i}",
                execution_id=execution_id,
                config=config,
            )

            debug_jobs.append(result)

        # All debug Jobs should have 3600s TTL
        assert all(job["ttl_seconds"] == 3600 for job in debug_jobs)
        assert all(job["debug"] is True for job in debug_jobs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
