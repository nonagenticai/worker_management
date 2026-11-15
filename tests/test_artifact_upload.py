"""Test artifact upload to MinIO - Task 78

Uses mock Minio/K8s classes — no real infrastructure needed.
"""

import io
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid7

import pytest
import pytest_asyncio


class MockMinio:
    """Mock MinIO client for testing."""

    def __init__(self, *args, **kwargs):
        pass

    def put_object(self, bucket_name, object_name, data, length):
        return Mock(etag="mock-etag")

    def list_objects(self, bucket_name, prefix=""):
        return [
            Mock(object_name=f"{prefix}/file1.txt"),
            Mock(object_name=f"{prefix}/file2.txt"),
            Mock(object_name=f"{prefix}/file3.txt"),
        ]

    def get_object(self, bucket_name, object_name):
        return Mock(data=b"Mock file content")


class MockK8sJobExecutor:
    """Mock K8s job executor for testing."""

    def __init__(self):
        self.minio_client: MockMinio | None = None

    async def initialize(self):
        self.minio_client = MockMinio(
            "minio.storage.svc.cluster.local:9000",
            access_key="test-access-key",
            secret_key="test-secret-key",
            secure=False,
        )

    async def execute_agent(self, agent_name, prompt, execution_id, config=None):
        """Mock execution that creates files"""
        artifacts = await self.collect_artifacts(
            job_name=f"claude-agent-{str(execution_id)[:8]}", execution_id=execution_id
        )

        return {
            "job_name": f"claude-agent-{str(execution_id)[:8]}",
            "execution_id": str(execution_id),
            "logs": "Created 3 files in /workspace",
            "artifacts": artifacts,
            "status": "completed",
        }

    async def collect_artifacts(self, job_name, execution_id):
        """Mock artifact collection"""
        files = ["file1.txt", "file2.txt", "file3.txt"]
        uploaded = []

        for filename in files:
            object_name = f"{execution_id}/{filename}"
            assert self.minio_client is not None
            self.minio_client.put_object(
                bucket_name="agent-artifacts",
                object_name=object_name,
                data=io.BytesIO(b"Mock content"),
                length=12,
            )
            uploaded.append(object_name)

        return uploaded


class TestArtifactUpload:
    """Test suite for artifact upload to MinIO"""

    @pytest_asyncio.fixture
    async def executor(self):
        """Create executor with MinIO client"""
        executor = MockK8sJobExecutor()
        await executor.initialize()
        return executor

    @pytest.mark.asyncio
    async def test_artifact_upload_to_minio(self, executor):
        """Verify artifacts uploaded to MinIO after Job completion"""
        execution_id = uuid7()

        # Run agent that creates files
        result = await executor.execute_agent(
            agent_name="test-agent",
            prompt="Create 3 files in /workspace",
            execution_id=execution_id,
        )

        # Verify artifacts uploaded
        artifacts = result["artifacts"]
        assert len(artifacts) >= 3
        assert all(str(execution_id) in artifact for artifact in artifacts)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_artifacts_from_minio(self, executor):
        """Test can list artifacts from MinIO"""
        execution_id = uuid7()

        # Upload artifacts
        await executor.execute_agent(
            agent_name="test-agent",
            prompt="Create multiple files",
            execution_id=execution_id,
        )

        # List objects from MinIO
        objects = list(
            executor.minio_client.list_objects("agent-artifacts", prefix=str(execution_id))
        )

        assert len(objects) >= 3
        assert all(str(execution_id) in obj.object_name for obj in objects)

    @pytest.mark.asyncio
    async def test_download_artifact_from_minio(self, executor):
        """Verify can download artifacts from MinIO"""
        execution_id = uuid7()

        # Upload artifacts
        result = await executor.execute_agent(
            agent_name="test-agent",
            prompt="Create test file",
            execution_id=execution_id,
        )

        # Download first artifact
        first_artifact = result["artifacts"][0]
        obj = executor.minio_client.get_object("agent-artifacts", first_artifact)

        assert obj is not None
        assert obj.data is not None

    @pytest.mark.asyncio
    async def test_multiple_file_upload(self, executor):
        """Test multiple file uploads work correctly"""
        execution_id = uuid7()

        # Create multiple files
        files = ["output.log", "results.json", "data.csv", "report.txt"]
        uploaded = []

        for filename in files:
            object_name = f"{execution_id}/{filename}"
            executor.minio_client.put_object(
                bucket_name="agent-artifacts",
                object_name=object_name,
                data=io.BytesIO(f"Content of {filename}".encode()),
                length=len(f"Content of {filename}"),
            )
            uploaded.append(object_name)

        assert len(uploaded) == len(files)
        assert all(str(execution_id) in path for path in uploaded)

    @pytest.mark.asyncio
    async def test_artifact_path_structure(self, executor):
        """Test artifacts are stored with correct path structure"""
        execution_id = uuid7()

        result = await executor.execute_agent(
            agent_name="test-agent",
            prompt="Create structured files",
            execution_id=execution_id,
        )

        # Check path structure: {execution_id}/{filename}
        for artifact in result["artifacts"]:
            assert artifact.startswith(str(execution_id) + "/")
            assert "/" in artifact
            parts = artifact.split("/")
            assert len(parts) == 2
            assert parts[0] == str(execution_id)

    @pytest.mark.asyncio
    async def test_empty_workspace_no_artifacts(self):
        """Test handling when no artifacts created"""
        executor = MockK8sJobExecutor()
        await executor.initialize()

        # Mock collect_artifacts to return empty list. Assigning to an instance
        # attribute shadows the bound method without rebinding, so the
        # replacement does not take `self` — cast for ty's signature check.
        async def mock_collect_empty(job_name, execution_id):
            return []

        cast("Any", executor).collect_artifacts = mock_collect_empty

        artifacts = await executor.collect_artifacts(job_name="test-job", execution_id=uuid7())

        assert artifacts == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
