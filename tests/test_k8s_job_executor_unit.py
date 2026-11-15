"""Unit tests for k8s_job_executor.py."""

import os
from unittest.mock import MagicMock, patch
from uuid import uuid7

import pytest

from src.k8s_job_executor import K8sJobExecutor


class TestK8sJobExecutorInit:
    def test_default_namespace(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXECUTOR_NAMESPACE", None)
            executor = K8sJobExecutor()
            assert executor.namespace == "agent-jobs"

    def test_custom_namespace(self):
        executor = K8sJobExecutor(namespace="custom-ns")
        assert executor.namespace == "custom-ns"

    def test_from_env(self):
        with patch.dict(os.environ, {"EXECUTOR_NAMESPACE": "env-ns"}):
            executor = K8sJobExecutor()
            assert executor.namespace == "env-ns"

    def test_initial_state(self):
        executor = K8sJobExecutor()
        assert executor.batch_v1 is None
        assert executor.core_v1 is None
        assert executor.minio_client is None


class TestBuildJobManifest:
    def test_basic_manifest(self):
        executor = K8sJobExecutor()
        manifest = executor._build_job_manifest(
            agent_name="test-agent",
            prompt="Do something",
            execution_id="12345678-abcd-1234-abcd-1234567890ab",
            config={},
        )
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["kind"] == "Job"
        assert "claude-agent-12345678" in manifest["metadata"]["name"]
        spec = manifest["spec"]["template"]["spec"]
        assert spec["restartPolicy"] == "Never"
        containers = spec["containers"]
        assert len(containers) == 1
        env_names = [e["name"] for e in containers[0]["env"]]
        assert "AGENT_PROMPT" in env_names
        assert "EXECUTION_ID" in env_names

    def test_debug_ttl(self):
        executor = K8sJobExecutor()
        manifest = executor._build_job_manifest(
            agent_name="agent",
            prompt="p",
            execution_id="12345678",
            config={"debug": True},
        )
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_custom_image(self):
        executor = K8sJobExecutor()
        manifest = executor._build_job_manifest(
            agent_name="agent",
            prompt="p",
            execution_id="12345678",
            config={},
            image_override="custom:v1",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "custom:v1"


class TestNotInitialized:
    @pytest.mark.asyncio
    async def test_execute_agent_not_initialized(self):
        executor = K8sJobExecutor()
        with pytest.raises(RuntimeError, match="not initialized"):
            await executor.execute_agent("agent", "prompt", uuid7())

    @pytest.mark.asyncio
    async def test_stream_logs_not_initialized(self):
        executor = K8sJobExecutor()
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in executor.stream_logs("job"):
                pass

    @pytest.mark.asyncio
    async def test_collect_artifacts_not_initialized(self):
        executor = K8sJobExecutor()
        with pytest.raises(RuntimeError, match="not initialized"):
            await executor.collect_artifacts("job", uuid7())

    @pytest.mark.asyncio
    async def test_get_pod_not_initialized(self):
        executor = K8sJobExecutor()
        with pytest.raises(RuntimeError, match="not initialized"):
            await executor._get_pod_for_job("job")

    @pytest.mark.asyncio
    async def test_wait_for_completion_not_initialized(self):
        executor = K8sJobExecutor()
        with pytest.raises(RuntimeError, match="not initialized"):
            await executor._wait_for_completion("job", timeout=1)


class TestGetPodForJob:
    @pytest.mark.asyncio
    async def test_returns_pod(self):
        executor = K8sJobExecutor()
        mock_pod = MagicMock()
        mock_pods = MagicMock()
        mock_pods.items = [mock_pod]
        executor.core_v1 = MagicMock()
        executor.core_v1.list_namespaced_pod = MagicMock(return_value=mock_pods)

        pod = await executor._get_pod_for_job("my-job")
        assert pod is mock_pod

    @pytest.mark.asyncio
    async def test_returns_none_when_no_pods(self):
        executor = K8sJobExecutor()
        mock_pods = MagicMock()
        mock_pods.items = []
        executor.core_v1 = MagicMock()
        executor.core_v1.list_namespaced_pod = MagicMock(return_value=mock_pods)

        pod = await executor._get_pod_for_job("my-job")
        assert pod is None
