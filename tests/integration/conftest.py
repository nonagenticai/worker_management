"""Integration test fixtures for live Worker Management endpoints.

These tests hit a REAL running Worker Management. They require:
- WORKERS_URL set explicitly (no default: unset => every live test SKIPS)
- LICENSE_TOKEN set in the environment (no token is committed here)

WORKERS_URL on a production host (*.internal, *.nonagentic.ai,
*.svc.cluster.local) is REFUSED unless ALLOW_PROD_INTEGRATION=1 — hand runs
against the old in-cluster default left ~153 test workers on
prod. See tests/integration/target_guard.py.

Run with:  uv run pytest tests/integration/ -v --no-cov -m integration
Skip with: default `uv run pytest` excludes integration marker
"""

import os
import uuid

import httpx2 as httpx
import pytest

from .target_guard import integration_target

pytestmark = pytest.mark.integration

LICENSE_TOKEN = os.environ.get("LICENSE_TOKEN", "")


def _delete(client: httpx.Client, path: str) -> None:
    """Teardown delete that fails LOUDLY: a leaked row is a red run, not clutter."""
    r = client.delete(path)
    assert r.status_code in (200, 202, 204, 404), (
        f"integration cleanup LEAKED {path}: {r.status_code} {r.text[:200]}"
    )


@pytest.fixture(scope="session")
def client():
    """httpx client pointed at Worker Management with license token auth."""
    url = integration_target("WORKERS_URL")  # raises ProdTargetRefused for prod
    if url is None:
        pytest.skip("WORKERS_URL not set — integration tests never default to production")
    with httpx.Client(
        base_url=url,
        headers={
            "X-Tenant-Id": f"Bearer {LICENSE_TOKEN}",
        },
        verify=False,
        timeout=30.0,
    ) as c:
        yield c


@pytest.fixture
def test_worker(client):
    """Create a pydantic-ai worker with random name, delete on teardown."""
    name = f"int-test-{uuid.uuid7().hex[:8]}"
    r = client.post(
        "/workers/pydantic-ai",
        json={
            "worker_name": name,
            "model": "openai:gpt-4o-mini",
            "system_prompt": "You are a test worker.",
        },
    )
    assert r.status_code == 201, f"Failed to create worker {name}: {r.text}"
    worker = r.json()
    yield worker
    _delete(client, f"/workers/{name}")


@pytest.fixture
def test_graph(client):
    """Create a single-node graph, delete on teardown.

    Skips if graph creation fails (known UUID serialization bug in API).
    """
    graph_name = f"int-graph-{uuid.uuid7().hex[:8]}"
    r = client.post(
        "/graphs/",
        json={
            "name": graph_name,
            "nodes": [{"name": "node1", "worker_name": "dummy", "node_type": "worker"}],
            "edges": [],
        },
    )
    if r.status_code != 201:
        pytest.skip(f"Graph creation unavailable (status {r.status_code}): {r.text[:100]}")
    graph = r.json()
    yield graph
    _delete(client, f"/graphs/{graph['id']}")


@pytest.fixture
def two_node_graph(client):
    """Create a two-node connected graph for structure/execution tests.

    Skips if graph creation fails (known UUID serialization bug in API).
    """
    graph_name = f"int-graph2-{uuid.uuid7().hex[:8]}"
    r = client.post(
        "/graphs/",
        json={
            "name": graph_name,
            "nodes": [
                {"name": "start", "worker_name": "dummy", "node_type": "worker"},
                {"name": "end", "worker_name": "dummy", "node_type": "worker"},
            ],
            "edges": [],
        },
    )
    if r.status_code != 201:
        pytest.skip(f"Graph creation unavailable (status {r.status_code}): {r.text[:100]}")
    graph = r.json()
    # Add edge using actual node UUIDs if nodes are returned
    if graph.get("nodes") and len(graph["nodes"]) >= 2:
        node_ids = [n["id"] for n in graph["nodes"]]
        client.patch(
            f"/graphs/{graph['id']}",
            json={"edges": [{"source": node_ids[0], "target": node_ids[1]}]},
        )
        r2 = client.get(f"/graphs/{graph['id']}")
        if r2.status_code == 200:
            graph = r2.json()
    yield graph
    _delete(client, f"/graphs/{graph['id']}")
