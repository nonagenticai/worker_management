#!/usr/bin/env python3
"""Quick test to check API directly with known worker ID."""

import requests

# Test creator ID (UUID format)
TEST_TENANT_ID = "550e8400-e29b-41d4-a716-446655440000"


def test_api_directly():
    worker_id = "test_states_worker_1_1753549513724_reanie"
    base_url = "http://localhost:8887"

    print(f"Testing API directly for worker: {worker_id}")

    # Test the worker-specific states endpoint
    try:
        response = requests.get(  # noqa: S113
            f"{base_url}/workers/{worker_id}/states?tenant_id={TEST_TENANT_ID}"
        )
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Returned {len(data)} states")

            if data:
                print("Sample state:")
                sample = data[0]
                print(f"  ID: {sample.get('id')}")
                print(f"  Type: {sample.get('state_type')}")
                print(f"  Timestamp: {sample.get('timestamp')}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")

    # Test with explicit limit
    try:
        response = requests.get(  # noqa: S113
            f"{base_url}/workers/{worker_id}/states?tenant_id={TEST_TENANT_ID}&limit=10"
        )
        print("\nWith limit=10:")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Returned {len(data)} states")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    test_api_directly()
