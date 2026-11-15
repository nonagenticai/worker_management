"""Test worker states API validation, especially state_type enum validation."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.worker_states import router
from src.auth import get_identity_proven, get_tenant_id

# Test tenant ID (UUID format)
TEST_TENANT_ID = "550e8400-e29b-41d4-a716-446655440000"


# Override auth dependency to return test tenant_id
async def override_get_tenant_id():
    return TEST_TENANT_ID


# This module builds a bare FastAPI app with only the worker_states router —
# no AuthenticationMiddleware at all — so `get_identity_proven` falls back to
# its fail-closed default of False (src/auth/dependencies.py). These tests
# are validation-focused, not exercises of the tokenless write hole POST
# /workers_states now refuses, so simulate a proven caller explicitly.
async def override_get_identity_proven():
    return True


# Create default client for validation-only tests (will fail on actual DB ops)
app = FastAPI()
app.dependency_overrides[get_tenant_id] = override_get_tenant_id
app.dependency_overrides[get_identity_proven] = override_get_identity_proven
app.include_router(router)
client = TestClient(app)


class TestWorkerStatesValidation:
    """Test validation for worker states API."""

    def test_invalid_state_types_numeric(self):
        """Test that numeric state types are rejected."""
        invalid_types = ["0", "1", "2", "123", "-1"]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, (
                f"Invalid numeric state_type '{state_type}' was not rejected"
            )
            error_detail = response.json()["detail"]
            # Handle both string and array responses
            if isinstance(error_detail, list):
                assert any("Invalid state type" in str(error) for error in error_detail)
            else:
                assert "Invalid state type" in error_detail

    def test_invalid_state_types_special(self):
        """Test that special invalid values are rejected."""
        invalid_types = ["null", "None", "undefined", "", " ", "  "]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, (
                f"Invalid special state_type '{state_type}' was not rejected"
            )

    def test_invalid_state_types_wrong_enum(self):
        """Test that non-enum values are rejected."""
        invalid_types = [
            "pending",
            "running",
            "stopped",
            "paused",
            "finished",
            "unknown",
            "CREATED",
            "Processing",
        ]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, (
                f"Invalid enum state_type '{state_type}' was not rejected"
            )
            error_detail = response.json()["detail"]
            # Handle both string and array responses
            if isinstance(error_detail, list):
                assert any("Invalid state type" in str(error) for error in error_detail)
            else:
                assert "Invalid state type" in error_detail
            # Check for "Must be one of" in error messages
            if isinstance(error_detail, list):
                assert any("Must be one of" in str(error) for error in error_detail)
            else:
                assert "Must be one of" in error_detail

    def test_state_type_with_null_bytes(self):
        """Test that state types with null bytes are rejected."""
        payload = {
            "worker_states": [
                {
                    "worker_id": "test_worker",
                    "worker_name": "test",
                    "is_agent": False,
                    "state_type": "created\x00",
                    "worker_state": {"status": "test"},
                }
            ]
        }

        response = client.post("/workers_states", json=payload)
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        # Handle both string and array responses
        if isinstance(error_detail, list):
            assert any("cannot contain null bytes" in str(error) for error in error_detail)
        else:
            assert "cannot contain null bytes" in error_detail

    def test_state_type_non_ascii(self):
        """Test that non-ASCII state types are rejected."""
        invalid_types = ["créé", "处理中", "🚀", "test™"]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, (
                f"Non-ASCII state_type '{state_type}' was not rejected"
            )
            error_detail = response.json()["detail"]
            # Handle both string and array responses
            if isinstance(error_detail, list):
                assert any("Invalid state type" in str(error) for error in error_detail)
            else:
                assert "Invalid state type" in error_detail

    def test_state_type_with_special_chars(self):
        """Test that state types with special characters are rejected."""
        invalid_types = [
            "created!",
            "process-ing",
            "error@",
            "idle#",
            "active$",
            "default%",
        ]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, (
                f"State type with special chars '{state_type}' was not rejected"
            )
            error_detail = response.json()["detail"]
            # Handle both string and array responses
            if isinstance(error_detail, list):
                assert any("Invalid state type" in str(error) for error in error_detail)
            else:
                assert "Invalid state type" in error_detail

    def test_multiple_workers_mixed_valid_invalid(self):
        """Test batch request with mix of valid and invalid state types."""
        payload = {
            "worker_states": [
                {
                    "worker_id": "worker1",
                    "worker_name": "test1",
                    "state_type": "created",  # Valid
                    "worker_state": {"status": "test"},
                },
                {
                    "worker_id": "worker2",
                    "worker_name": "test2",
                    "state_type": "0",  # Invalid
                    "worker_state": {"status": "test"},
                },
            ]
        }

        response = client.post("/workers_states", json=payload)
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        # Handle both string and array responses
        if isinstance(error_detail, list):
            assert any("Invalid state type" in str(error) for error in error_detail)
        else:
            assert "Invalid state type" in error_detail

    def test_state_type_case_sensitivity(self):
        """Test that state_type validation is case-sensitive."""
        invalid_types = [
            "CREATED",
            "Processing",
            "COMPLETED",
            "Error",
            "IDLE",
            "Active",
            "DEFAULT",
        ]

        for state_type in invalid_types:
            payload = {
                "worker_states": [
                    {
                        "worker_id": "test_worker",
                        "worker_name": "test",
                        "state_type": state_type,
                        "worker_state": {"status": "test"},
                    }
                ]
            }

            response = client.post("/workers_states", json=payload)
            assert response.status_code == 422, f"Case variant '{state_type}' was not rejected"
            error_detail = response.json()["detail"]
            # Handle both string and array responses
            if isinstance(error_detail, list):
                assert any("Invalid state type" in str(error) for error in error_detail)
            else:
                assert "Invalid state type" in error_detail


if __name__ == "__main__":
    # Run tests without pytest
    test_instance = TestWorkerStatesValidation()
    test_methods = [method for method in dir(test_instance) if method.startswith("test_")]

    print("Running Worker States Validation Tests...")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method}...")
            getattr(test_instance, test_method)()
            print(f"✅ {test_method} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_method} FAILED: {e!s}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_method} ERROR: {type(e).__name__}: {e!s}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    sys.exit(0 if failed == 0 else 1)
