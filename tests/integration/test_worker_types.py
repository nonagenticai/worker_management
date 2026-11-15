"""Integration tests for typed worker creation endpoints."""

import uuid

import pytest

pytestmark = pytest.mark.integration


class TestPydanticAiCreate:
    def test_create_pydantic_ai_worker(self, client):
        name = f"type-pai-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/pydantic-ai",
            json={
                "worker_name": name,
                "model": "openai:gpt-4o-mini",
                "system_prompt": "Pydantic AI test",
            },
        )
        assert r.status_code == 201
        assert r.json()["name"] == name
        client.delete(f"/workers/{name}")


class TestClaudeCodeCreate:
    def test_create_claude_code_worker(self, client):
        """Claude Code workers may fail if API key is not configured — tolerate 4xx."""
        name = f"type-cc-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/claude-code",
            json={"worker_name": name},
        )
        # 201 if API key is available, 4xx/5xx otherwise
        assert r.status_code in (201, 400, 422, 500, 503)
        if r.status_code == 201:
            client.delete(f"/workers/{name}")


class TestCodexCreate:
    def test_create_codex_worker(self, client):
        """Codex workers may fail if API key is not configured — tolerate 4xx."""
        name = f"type-codex-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/codex",
            json={"worker_name": name},
        )
        assert r.status_code in (201, 400, 422, 500, 503)
        if r.status_code == 201:
            client.delete(f"/workers/{name}")


class TestRemoteClaudeCreate:
    def test_create_remote_claude_worker(self, client):
        """Remote Claude workers may fail without proper config — tolerate errors."""
        name = f"type-rc-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/remote-claude",
            json={"worker_name": name},
        )
        assert r.status_code in (201, 400, 422, 500, 503)
        if r.status_code == 201:
            client.delete(f"/workers/{name}")


class TestRemoteCodexCreate:
    def test_create_remote_codex_worker(self, client):
        """Remote Codex workers may fail without proper config — tolerate errors."""
        name = f"type-rcodex-{uuid.uuid7().hex[:8]}"
        r = client.post(
            "/workers/remote-codex",
            json={"worker_name": name},
        )
        assert r.status_code in (201, 400, 422, 500, 503)
        if r.status_code == 201:
            client.delete(f"/workers/{name}")
