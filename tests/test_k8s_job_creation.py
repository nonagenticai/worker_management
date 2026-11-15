"""Tests for K8sJobExecutor — job manifest generation and configuration.

Tests the real K8sJobExecutor._build_job_manifest() directly (pure function, no cluster needed).
"""

import os
from unittest.mock import patch

import pytest

from src.k8s_job_executor import K8sJobExecutor


class TestBuildJobManifest:
    """Test _build_job_manifest produces correct K8s Job specs."""

    @pytest.fixture
    def executor(self):
        return K8sJobExecutor(namespace="agent-jobs")

    @pytest.fixture
    def base_manifest(self, executor):
        return executor._build_job_manifest(
            agent_name="test-agent",
            prompt="Write hello world",
            execution_id="abcd1234-5678-9012-3456-789012345678",
            config={},
        )

    def _get_container(self, manifest):
        return manifest["spec"]["template"]["spec"]["containers"][0]

    def _get_template_spec(self, manifest):
        return manifest["spec"]["template"]["spec"]

    def test_mount_path_is_staging(self, base_manifest):
        """Credentials must mount at /tmp/claude-secrets (staging path, not directly at .claude)."""
        container = self._get_container(base_manifest)
        creds_mount = next(m for m in container["volumeMounts"] if m["name"] == "claude-creds")
        assert creds_mount["mountPath"] == "/tmp/claude-secrets"
        assert creds_mount["readOnly"] is True

    def test_no_unsatisfiable_arch_pin(self, base_manifest):
        """Jobs must not pin an architecture no node in the fleet provides.

        This test previously asserted the opposite — that nodeSelector was
        {"kubernetes.io/arch": "arm64"} — and so locked the defect in place.
        Every k3s node in the fleet is amd64, and
        both executor images (claude-executor, codex-executor) are built amd64,
        so an arm64 pin made every Job permanently Pending rather than Failed.
        Pending is the dangerous half: activeDeadlineSeconds never fires, so the
        Job neither runs nor reports an error.

        Asserting the absence of the pin, rather than asserting amd64, keeps the
        manifest portable if a genuinely arm64 node is ever added — but fails
        loudly if anyone re-introduces a hard architecture constraint.
        """
        spec = self._get_template_spec(base_manifest)
        assert "kubernetes.io/arch" not in spec.get("nodeSelector", {})

    def test_env_vars(self, base_manifest):
        """AGENT_PROMPT, EXECUTION_ID, AGENT_NAME must be set."""
        container = self._get_container(base_manifest)
        env_map = {e["name"]: e["value"] for e in container["env"]}
        assert env_map["AGENT_PROMPT"] == "Write hello world"
        assert env_map["EXECUTION_ID"] == "abcd1234-5678-9012-3456-789012345678"
        assert env_map["AGENT_NAME"] == "test-agent"

    def test_ttl_default_300(self, base_manifest):
        """Default TTL should be 300 seconds."""
        assert base_manifest["spec"]["ttlSecondsAfterFinished"] == 300

    def test_ttl_debug_3600(self, executor):
        """Debug mode should set TTL to 3600."""
        manifest = executor._build_job_manifest(
            agent_name="debug-agent",
            prompt="test",
            execution_id="abcd1234",
            config={"debug": True},
        )
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_security_context(self, base_manifest):
        """Pod security context must enforce non-root with uid 1000."""
        sec = self._get_template_spec(base_manifest)["securityContext"]
        assert sec["runAsUser"] == 1000
        assert sec["runAsNonRoot"] is True
        assert sec["fsGroup"] == 1000

    def test_workspace_emptydir(self, base_manifest):
        """Workspace volume must be emptyDir with 2Gi limit."""
        volumes = self._get_template_spec(base_manifest)["volumes"]
        workspace = next(v for v in volumes if v["name"] == "workspace")
        assert "emptyDir" in workspace
        assert workspace["emptyDir"]["sizeLimit"] == "2Gi"

    def test_creds_secret_volume(self, base_manifest):
        """Credentials volume must reference the correct secret."""
        volumes = self._get_template_spec(base_manifest)["volumes"]
        creds = next(v for v in volumes if v["name"] == "claude-creds")
        assert creds["secret"]["secretName"] == "claude-secrets"
        assert creds["secret"]["defaultMode"] == 0o400
        assert creds["secret"]["items"] == [
            {"key": ".credentials.json", "path": ".credentials.json"}
        ]

    def test_job_name_format(self, base_manifest):
        """Job name should be claude-agent-<first8chars>."""
        assert base_manifest["metadata"]["name"] == "claude-agent-abcd1234"

    def test_backoff_limit_zero(self, base_manifest):
        """No retries on failure."""
        assert base_manifest["spec"]["backoffLimit"] == 0

    def test_active_deadline_600(self, base_manifest):
        """Job must have 600s deadline."""
        assert base_manifest["spec"]["activeDeadlineSeconds"] == 600

    def test_custom_image_from_env(self):
        """K8S_EXECUTOR_IMAGE env var should set executor image."""
        with patch.dict(
            os.environ,
            {"K8S_EXECUTOR_IMAGE": "registry.example.com:5000/custom-executor:v1"},
        ):
            executor = K8sJobExecutor()
            assert executor.executor_image == "registry.example.com:5000/custom-executor:v1"

    def test_custom_namespace(self):
        """Namespace should be configurable."""
        executor = K8sJobExecutor(namespace="custom-ns")
        assert executor.namespace == "custom-ns"

    def test_namespace_from_env(self):
        """EXECUTOR_NAMESPACE env var should set namespace."""
        with patch.dict(os.environ, {"EXECUTOR_NAMESPACE": "env-ns"}):
            executor = K8sJobExecutor()
            assert executor.namespace == "env-ns"

    def test_image_override_parameter(self, executor):
        """image_override should take precedence over executor_image."""
        manifest = executor._build_job_manifest(
            agent_name="test",
            prompt="test",
            execution_id="abcd1234",
            config={},
            image_override="registry.example.com:5000/codex-executor:latest",
        )
        container = self._get_container(manifest)
        assert container["image"] == "registry.example.com:5000/codex-executor:latest"

    def test_default_image_when_no_override(self, executor):
        """Without image_override, should use self.executor_image."""
        manifest = executor._build_job_manifest(
            agent_name="test",
            prompt="test",
            execution_id="abcd1234",
            config={},
        )
        container = self._get_container(manifest)
        assert container["image"] == executor.executor_image

    def test_workspace_mount_at_workspace(self, base_manifest):
        """Workspace volume mount at /workspace."""
        container = self._get_container(base_manifest)
        ws_mount = next(m for m in container["volumeMounts"] if m["name"] == "workspace")
        assert ws_mount["mountPath"] == "/workspace"

    def test_resource_limits(self, base_manifest):
        """Container should have resource requests and limits."""
        container = self._get_container(base_manifest)
        assert container["resources"]["requests"]["cpu"] == "250m"
        assert container["resources"]["limits"]["memory"] == "2Gi"

    def test_labels(self, base_manifest):
        """Job should have correct labels."""
        labels = base_manifest["metadata"]["labels"]
        assert labels["app"] == "worker-management"
        assert labels["component"] == "agent-executor"
        assert labels["execution-id"] == "abcd1234-5678-9012-3456-789012345678"
