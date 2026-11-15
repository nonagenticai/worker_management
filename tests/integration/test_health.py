"""Integration tests for health check endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestHealthCheck:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestRoot:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_message(self, client):
        r = client.get("/")
        data = r.json()
        assert "message" in data


class TestLiveness:
    def test_liveness_returns_200(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200

    def test_liveness_has_status(self, client):
        data = client.get("/health/live").json()
        assert data["status"] == "alive"


class TestReadiness:
    def test_readiness_returns_200(self, client):
        r = client.get("/health/ready")
        assert r.status_code == 200

    def test_readiness_has_checks(self, client):
        data = client.get("/health/ready").json()
        assert "checks" in data
