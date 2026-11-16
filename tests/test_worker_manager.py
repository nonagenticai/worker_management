"""Unit tests for worker/worker_manager.py."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.worker.worker_manager import WorkerManager


class TestWorkerManagerInit:
    def test_default_url(self):
        wm = WorkerManager()
        assert wm.base_url == "http://localhost:8887"
        assert wm.client is None

    def test_custom_url(self):
        wm = WorkerManager(base_url="http://custom:9999")
        assert wm.base_url == "http://custom:9999"


class TestGetClient:
    @pytest.mark.asyncio
    async def test_creates_client_on_first_call(self):
        wm = WorkerManager()
        assert wm.client is None
        client = await wm._get_client()
        assert client is not None
        assert wm.client is client
        await wm.cleanup()

    @pytest.mark.asyncio
    async def test_reuses_existing_client(self):
        wm = WorkerManager()
        client1 = await wm._get_client()
        client2 = await wm._get_client()
        assert client1 is client2
        await wm.cleanup()


class TestSendMessageToWorker:
    @pytest.mark.asyncio
    async def test_success(self):
        wm = WorkerManager()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "hello",
            "metadata": {"k": "v"},
            "usage_metrics": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wm.client = mock_client

        result = await wm.send_message_to_worker("my_worker", "test message")
        assert result["response"] == "hello"
        assert result.get("error") is None

    @pytest.mark.asyncio
    async def test_success_structured_output_maps_output_field(self):
        # The /workers/{name}/message endpoint returns the agent output under
        # "output" (MessageResponse), not "response". A structured (dict) output
        # must be carried through under "output" AND rendered to text under
        # "response" so the graph WorkerNode can both route and lift quality_score.
        wm = WorkerManager()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"validation_passed": True, "quality_score": 0.87, "decision": "pass"},
            "output_type": "object",
            "metadata": {"k": "v"},
            "usage_metrics": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wm.client = mock_client

        result = await wm.send_message_to_worker("qa_validator", "validate this")
        assert result.get("error") is None
        # Raw structured value preserved for the quality_score lift.
        assert result["output"] == {
            "validation_passed": True,
            "quality_score": 0.87,
            "decision": "pass",
        }
        # Text form available for routing / End.
        assert json.loads(result["response"]) == result["output"]

    @pytest.mark.asyncio
    async def test_non_200_status(self):
        wm = WorkerManager()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wm.client = mock_client

        result = await wm.send_message_to_worker("worker", "msg")
        assert result["error"] is True
        assert result["status_code"] == 500

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        wm = WorkerManager()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("bad", "", 0)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wm.client = mock_client

        result = await wm.send_message_to_worker("worker", "msg")
        assert result["error"] is True
        assert "Invalid JSON" in result["response"]

    @pytest.mark.asyncio
    async def test_request_error(self):
        wm = WorkerManager()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("conn refused"))
        wm.client = mock_client

        result = await wm.send_message_to_worker("worker", "msg")
        assert result["error"] is True
        assert "Request error" in result["response"]

    @pytest.mark.asyncio
    async def test_value_error(self):
        wm = WorkerManager()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("bad value"))
        wm.client = mock_client

        result = await wm.send_message_to_worker("worker", "msg")
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        wm = WorkerManager()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=RuntimeError("unexpected"))
        wm.client = mock_client

        result = await wm.send_message_to_worker("worker", "msg")
        assert result["error"] is True
        assert "Unexpected error" in result["response"]

    @pytest.mark.asyncio
    async def test_with_context(self):
        wm = WorkerManager()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wm.client = mock_client

        result = await wm.send_message_to_worker("w", "m", context={"key": "val"})
        assert result["context"] == {"key": "val"}


class TestSendMessageIdentityHeaders:
    """The loopback POST must forward the run's tenant identity so the receiver's
    trusted-network auth branch enforces byo_required (not DEFAULT_TENANT_ID)."""

    @staticmethod
    def _mock_client() -> AsyncMock:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_no_identity_sends_no_tenant_header(self):
        # Absent identity ⇒ headers must be None (byte-identical to pre-fix
        # behavior: the receiver falls back to DEFAULT_TENANT_ID).
        wm = WorkerManager()
        wm.client = self._mock_client()

        await wm.send_message_to_worker("w", "m")

        _, kwargs = wm.client.post.call_args
        assert kwargs.get("headers") is None

    @pytest.mark.asyncio
    async def test_tenant_id_forwarded_as_raw_non_bearer_header(self):
        wm = WorkerManager()
        wm.client = self._mock_client()
        tenant = "aaaaaaaa-0000-4000-8000-000000000099"

        await wm.send_message_to_worker("w", "m", tenant_id=tenant)

        _, kwargs = wm.client.post.call_args
        headers = kwargs["headers"]
        assert headers["X-Tenant-Id"] == tenant
        # Must be a RAW tenant id, never a Bearer token — otherwise the receiver
        # would try to introspect it as a license token instead of taking the
        # trusted-network branch.
        assert not headers["X-Tenant-Id"].lower().startswith("bearer ")
        # No subject supplied ⇒ no X-Subject-Id.
        assert "X-Subject-Id" not in headers

    @pytest.mark.asyncio
    async def test_subject_id_forwarded_when_present(self):
        wm = WorkerManager()
        wm.client = self._mock_client()

        await wm.send_message_to_worker("w", "m", tenant_id="tenant-a", subject_id="subject-b")

        _, kwargs = wm.client.post.call_args
        headers = kwargs["headers"]
        assert headers["X-Tenant-Id"] == "tenant-a"
        assert headers["X-Subject-Id"] == "subject-b"

    @pytest.mark.asyncio
    async def test_subject_without_tenant_is_ignored(self):
        # subject_id alone (no tenant) must not produce any header — the tenant is
        # the signal that drives the trusted-network branch.
        wm = WorkerManager()
        wm.client = self._mock_client()

        await wm.send_message_to_worker("w", "m", subject_id="subject-only")

        _, kwargs = wm.client.post.call_args
        assert kwargs.get("headers") is None


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_closes_client(self):
        wm = WorkerManager()
        mock_client = AsyncMock()
        wm.client = mock_client
        await wm.cleanup()
        mock_client.aclose.assert_awaited_once()
        assert wm.client is None

    @pytest.mark.asyncio
    async def test_cleanup_noop_without_client(self):
        wm = WorkerManager()
        await wm.cleanup()  # Should not raise


class TestContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with WorkerManager() as wm:
            assert isinstance(wm, WorkerManager)
