"""Kubernetes Job-based executor for running worker tasks as one-shot pods."""

import asyncio
import io
from typing import TYPE_CHECKING, Any, cast

from kubernetes import client, config
from kubernetes.stream import stream
from minio import Minio

from src.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID


class K8sJobExecutor:
    """Executor for running Claude agents in Kubernetes Jobs."""

    def __init__(self, namespace: str | None = None) -> None:
        settings = get_settings()
        self.namespace = namespace or settings.executor_namespace
        self.batch_v1: client.BatchV1Api | None = None
        self.core_v1: client.CoreV1Api | None = None
        self.minio_client: Minio | None = None
        self.minio_bucket = settings.minio_bucket
        self.service_account = settings.executor_service_account
        self.executor_image = settings.k8s_executor_image

    async def initialize(self) -> None:
        """Initialize K8s and MinIO clients."""
        config.load_incluster_config()

        # Workaround for kubernetes>=33: InClusterConfigLoader._set_config writes
        # the SA token to api_key['authorization'] but Configuration.auth_settings()
        # only reads api_key['BearerToken']. Without this bridge every request
        # goes anonymously and the apiserver returns 401. Remove when the lib pin
        # bumps past the upstream fix. Mirrors coding-containers' fix (2026-05-27).
        _cfg = client.Configuration.get_default_copy()
        if "authorization" in _cfg.api_key and "BearerToken" not in _cfg.api_key:
            _cfg.api_key["BearerToken"] = _cfg.api_key["authorization"]
            client.Configuration.set_default(_cfg)

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        settings = get_settings()
        self.minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key or None,
            secret_key=settings.minio_secret_key or None,
            secure=settings.minio_secure,
        )

    async def execute_agent(
        self,
        agent_name: str,
        prompt: str,
        execution_id: UUID,
        config: dict[str, Any] | None = None,
        image_override: str | None = None,
    ) -> dict[str, Any]:
        """Create and run Claude agent Job."""
        job_manifest = self._build_job_manifest(
            agent_name=agent_name,
            prompt=prompt,
            execution_id=str(execution_id),
            config=config or {},
            image_override=image_override,
        )

        # Create Job
        if self.batch_v1 is None:
            raise RuntimeError("K8sJobExecutor not initialized. Call initialize() first.")
        # kubernetes 37 types `body` as V1Job, but the client serialises a plain
        # manifest dict identically and this executor builds the manifest as a
        # dict (image, env, volumes, tolerations). Keep the dict and assert the
        # name we asked for rather than reading it back off an Optional
        # `V1Job.metadata`.
        self.batch_v1.create_namespaced_job(
            namespace=self.namespace, body=cast("Any", job_manifest)
        )
        job_name = (job_manifest.get("metadata") or {}).get("name")
        if not isinstance(job_name, str):
            raise TypeError("Job manifest has no metadata.name")

        # Wait for pod to start
        await self._wait_for_pod(job_name)

        # Stream logs
        logs = []
        async for line in self.stream_logs(job_name):
            logs.append(line)

        # Wait for completion
        job = await self._wait_for_completion(job_name, timeout=600)

        # Check for job failure
        if job and job.status is not None and job.status.failed:
            raise RuntimeError(f"K8s Job {job_name} failed. Check pod logs for details.")

        # Collect artifacts
        artifacts = await self.collect_artifacts(job_name, execution_id)

        return {
            "job_name": job_name,
            "execution_id": str(execution_id),
            "logs": "\n".join(logs),
            "artifacts": artifacts,
            "status": "completed",
        }

    def _build_job_manifest(
        self,
        agent_name: str,
        prompt: str,
        execution_id: str,
        config: dict,
        image_override: str | None = None,
    ) -> dict:
        """Build Job manifest."""
        ttl_seconds = config.get("ttl_seconds", 300)  # 5 min default
        if config.get("debug"):
            ttl_seconds = 3600  # 60 min for debug

        settings = get_settings()
        session_secret = settings.claude_session_secret_name

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"claude-agent-{execution_id[:8]}",
                "namespace": self.namespace,
                "labels": {
                    "app": "worker-management",
                    "component": "agent-executor",
                    "execution-id": execution_id,
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": ttl_seconds,
                "activeDeadlineSeconds": 600,
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "serviceAccountName": self.service_account,
                        "restartPolicy": "Never",
                        # No architecture nodeSelector. This used to pin
                        # kubernetes.io/arch=arm64, which no node has ever
                        # satisfied: every node in the cluster is amd64, so
                        # every Job produced here was
                        # permanently unschedulable (Pending, never Failed —
                        # so activeDeadlineSeconds never fires and nothing
                        # surfaces an error). The executor images are amd64
                        # too: claude-executor:latest and codex-executor:latest
                        # both report architecture=amd64 in the registry, so
                        # the pin contradicted the very images it scheduled.
                        # Leaving the selector out lets the scheduler place the
                        # Job on any node; every node matches the image arch.
                        "securityContext": {
                            "runAsUser": 1000,
                            "runAsNonRoot": True,
                            "fsGroup": 1000,
                        },
                        "containers": [
                            {
                                "name": "claude-executor",
                                "image": image_override or self.executor_image,
                                "env": [
                                    {"name": "AGENT_PROMPT", "value": prompt},
                                    {"name": "EXECUTION_ID", "value": execution_id},
                                    {"name": "AGENT_NAME", "value": agent_name},
                                    {
                                        "name": "MINIO_ENDPOINT",
                                        "value": settings.minio_endpoint,
                                    },
                                    {
                                        "name": "MINIO_BUCKET",
                                        "value": self.minio_bucket,
                                    },
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "250m",
                                        "memory": "512Mi",
                                        "ephemeral-storage": "512Mi",
                                    },
                                    "limits": {
                                        "cpu": "1000m",
                                        "memory": "2Gi",
                                        "ephemeral-storage": "2Gi",
                                    },
                                },
                                "volumeMounts": [
                                    {"name": "workspace", "mountPath": "/workspace"},
                                    {
                                        "name": "claude-creds",
                                        "mountPath": "/tmp/claude-secrets",  # nosec B108 — standard temp directory usage  # noqa: S108
                                        "readOnly": True,
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "workspace", "emptyDir": {"sizeLimit": "2Gi"}},
                            {
                                "name": "claude-creds",
                                "secret": {
                                    "secretName": session_secret,
                                    "defaultMode": 0o400,
                                    "items": [
                                        {
                                            "key": ".credentials.json",
                                            "path": ".credentials.json",
                                        }
                                    ],
                                },
                            },
                        ],
                    }
                },
            },
        }

    async def stream_logs(self, job_name: str) -> AsyncIterator[str]:
        """Stream logs from Job pod."""
        import logging

        _logger = logging.getLogger(__name__)

        if self.core_v1 is None:
            raise RuntimeError("K8sJobExecutor not initialized. Call initialize() first.")
        pod = await self._get_pod_for_job(job_name)
        if pod is None:
            _logger.warning(f"No pod found for job {job_name}, cannot stream logs")
            return

        try:
            logs = self.core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=self.namespace,
                follow=True,
                _preload_content=False,
            )

            for line in logs:
                if isinstance(line, bytes):
                    yield line.decode("utf-8")
                else:
                    yield str(line)
        except Exception:
            _logger.exception(f"Error streaming logs for job {job_name}")

    async def collect_artifacts(self, job_name: str, execution_id: UUID) -> list[str]:
        """Collect artifacts from Job workspace to MinIO."""
        import logging

        _logger = logging.getLogger(__name__)

        if self.core_v1 is None or self.minio_client is None:
            raise RuntimeError("K8sJobExecutor not initialized. Call initialize() first.")

        pod = await self._get_pod_for_job(job_name)
        if pod is None:
            _logger.warning(f"No pod found for job {job_name}, cannot collect artifacts")
            return []

        # List files in workspace using null-delimited output for safety with special chars
        exec_command = ["/bin/sh", "-c", "find /workspace -type f -print0"]

        try:
            files_output = stream(
                self.core_v1.connect_get_namespaced_pod_exec,
                pod.metadata.name,
                self.namespace,
                command=exec_command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
        except Exception:
            _logger.exception(f"Failed to list workspace files for job {job_name}")
            return []

        # Split on null byte for safe handling of file paths with special characters
        files = [f for f in files_output.split("\0") if f.strip()]
        uploaded = []

        for file_path in files:
            if not file_path or not file_path.startswith("/workspace/"):
                continue

            try:
                # Read file content
                read_command = ["/bin/sh", "-c", f"cat -- '{file_path}'"]
                content = stream(
                    self.core_v1.connect_get_namespaced_pod_exec,
                    pod.metadata.name,
                    self.namespace,
                    command=read_command,
                    stderr=False,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )

                # Upload to MinIO
                filename = file_path[len("/workspace/") :]
                object_name = f"{execution_id}/{filename}"

                self.minio_client.put_object(
                    bucket_name=self.minio_bucket,
                    object_name=object_name,
                    data=io.BytesIO(content.encode()),
                    length=len(content),
                )

                uploaded.append(object_name)
            except Exception:
                _logger.exception(f"Failed to collect artifact {file_path}")
                continue

        return uploaded

    async def _wait_for_pod(self, job_name: str, timeout: int = 60) -> Any:
        """Wait for Job pod to be created."""
        for _ in range(timeout):
            pod = await self._get_pod_for_job(job_name)
            if pod:
                return pod
            await asyncio.sleep(1)
        raise TimeoutError(f"Pod for job {job_name} not created")

    async def _wait_for_completion(self, job_name: str, timeout: int = 600) -> Any:
        """Wait for Job to complete."""
        if self.batch_v1 is None:
            raise RuntimeError("K8sJobExecutor not initialized. Call initialize() first.")
        for _ in range(timeout):
            try:
                job = self.batch_v1.read_namespaced_job(name=job_name, namespace=self.namespace)
                status = job.status
                if status is not None and (status.succeeded or status.failed):
                    return job
            except client.exceptions.ApiException:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(f"Job {job_name} did not complete within {timeout}s")

    async def _get_pod_for_job(self, job_name: str) -> Any:
        """Get pod for Job."""
        if self.core_v1 is None:
            raise RuntimeError("K8sJobExecutor not initialized. Call initialize() first.")
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace, label_selector=f"job-name={job_name}"
        )
        return pods.items[0] if pods.items else None
