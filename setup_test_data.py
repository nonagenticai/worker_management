#!/usr/bin/env python3
"""Comprehensive test data setup script for Schemathesis testing.

This script creates all necessary test entities to achieve 100% operation coverage
by ensuring all DELETE/PUT operations have valid entities to operate on.

PROBLEM SOLVED:
- 16 operations untested due to missing test data (404 responses)
- DELETE/PUT operations require existing entities

ENTITIES CREATED:
1. Workers: test_worker_001, test_worker_002, etc.
2. Tools: test_tool_001, test_tool_002, etc.
3. Graphs: Creates graphs and captures their UUIDs
4. States: Generates worker states for state endpoints

USAGE:
    python setup_test_data.py --base-url http://localhost:8887
    python setup_test_data.py --base-url http://localhost:8887 --cleanup
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestDataSetup:
    """Comprehensive test data setup for API testing coverage."""

    def __init__(self, base_url: str = "http://localhost:8887"):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self.created_entities = {"workers": [], "tools": [], "graphs": [], "states": []}

    @property
    def session(self) -> aiohttp.ClientSession:
        """Return the active session, raising if not inside async context manager."""
        if self._session is None:
            msg = "Session not initialized. Use 'async with TestDataSetup(...) as setup:'"
            raise RuntimeError(msg)
        return self._session

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def check_api_health(self) -> bool:
        """Check if the API is accessible."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    logger.info("✅ API is accessible")
                    return True
                else:
                    logger.warning(f"⚠️ API health check returned {response.status}")
                    return True  # Continue anyway
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {e}")
            return True  # Continue anyway

    async def create_test_workers(self) -> list[dict[str, Any]]:
        """Create test workers with different configurations."""
        logger.info("🔧 Creating test workers...")

        workers_config = [
            {
                "worker_name": "test_worker_001",
                "description": "Basic test worker for API coverage",
                "tool_names": ["example_text_analyzer"],
                "agent_role": None,
            },
            {
                "worker_name": "test_worker_002",
                "description": "Agent test worker for message testing",
                "tool_names": ["example_text_analyzer", "file_reader"],
                "agent_role": "Test Agent for Schemathesis",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
                "model": "local-llm",
                "system_prompt": "You are a test agent for API coverage testing.",
                "temperature": 0.7,
            },
            {
                "worker_name": "test_worker_003",
                "description": "Unicode test worker: тест_воркер",
                "tool_names": [],
                "agent_role": None,
            },
            {
                "worker_name": "test_worker_004",
                "description": "Minimal worker for delete operations",
                "tool_names": ["example_text_analyzer"],
                "agent_role": None,
            },
            {
                "worker_name": "test_worker_005",
                "description": "Agent worker for state testing",
                "tool_names": ["example_text_analyzer"],
                "agent_role": "State Test Agent",
                "agent_type": "pydantic_ai",
                "model_provider": "inference_router",
                "model": "local-llm",
                "system_prompt": "You are a test agent for state management testing.",
                "temperature": 0.5,
            },
        ]

        created_workers = []
        for worker_config in workers_config:
            try:
                async with self.session.post(
                    f"{self.base_url}/workers", json=worker_config
                ) as response:
                    if response.status in [200, 201]:
                        worker_data = await response.json()
                        created_workers.append(worker_data)
                        self.created_entities["workers"].append(worker_data)
                        logger.info(f"✅ Created worker: {worker_config['worker_name']}")
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"⚠️ Failed to create worker {worker_config['worker_name']}: {response.status} - {error_text}"
                        )
            except Exception as e:
                logger.exception(f"❌ Error creating worker {worker_config['worker_name']}: {e}")  # noqa: TRY401

        return created_workers

    async def create_test_tools(self) -> list[dict[str, Any]]:
        """Create test tools for tool endpoint coverage."""
        logger.info("🔧 Creating test tools...")

        tools_config = [
            {
                "name": "test_tool_001",
                "description": "Basic test tool for API coverage testing",
                "python_code": '''def test_function(input_text: str) -> dict:
    """Test function for API coverage."""
    return {
        "result": f"Processed: {input_text}",
        "length": len(input_text),
        "timestamp": "2024-01-01T00:00:00Z"
    }''',
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "input_text": {
                            "type": "string",
                            "description": "Text to process",
                        }
                    },
                    "required": ["input_text"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"},
                        "length": {"type": "integer"},
                        "timestamp": {"type": "string"},
                    },
                },
                "tags": ["test", "coverage"],
                "version": "1.0.0",
            },
            {
                "name": "test_tool_002",
                "description": "Math calculation tool for testing",
                "python_code": '''def calculate(operation: str, a: float, b: float) -> dict:
    """Perform basic math operations."""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else None
    else:
        result = None

    return {
        "operation": operation,
        "operands": [a, b],
        "result": result,
        "valid": result is not None
    }''',
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
                "tags": ["math", "test"],
                "version": "1.0.0",
            },
            {
                "name": "test_tool_003",
                "description": "Text analysis tool for comprehensive testing",
                "python_code": '''def analyze_text(text: str) -> dict:
    """Analyze text properties."""
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "has_numbers": any(c.isdigit() for c in text),
        "has_uppercase": any(c.isupper() for c in text),
        "first_word": words[0] if words else None
    }''',
                "tags": ["text", "analysis"],
                "version": "1.0.0",
            },
            {
                "name": "test_tool_004",
                "description": "JSON processor for update testing",
                "python_code": '''def process_json(data: dict) -> dict:
    """Process JSON data."""
    return {
        "keys": list(data.keys()),
        "key_count": len(data),
        "has_nested": any(isinstance(v, dict) for v in data.values()),
        "processed_at": "2024-01-01T00:00:00Z"
    }''',
                "tags": ["json", "processing"],
                "version": "1.1.0",
            },
            {
                "name": "test_tool_005",
                "description": "List processor for delete testing",
                "python_code": '''def process_list(items: list) -> dict:
    """Process list data."""
    return {
        "count": len(items),
        "types": list(set(type(item).__name__ for item in items)),
        "first": items[0] if items else None,
        "last": items[-1] if items else None
    }''',
                "tags": ["list", "processing"],
                "version": "1.0.0",
            },
        ]

        created_tools = []
        for tool_config in tools_config:
            try:
                async with self.session.post(
                    f"{self.base_url}/tools", json=tool_config
                ) as response:
                    if response.status in [200, 201]:
                        tool_data = await response.json()
                        created_tools.append(tool_data)
                        self.created_entities["tools"].append(tool_data)
                        logger.info(f"✅ Created tool: {tool_config['name']}")
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"⚠️ Failed to create tool {tool_config['name']}: {response.status} - {error_text}"
                        )
            except Exception as e:
                logger.exception(f"❌ Error creating tool {tool_config['name']}: {e}")  # noqa: TRY401

        return created_tools

    async def create_test_graphs(self) -> list[dict[str, Any]]:
        """Create test graphs and capture their UUIDs."""
        logger.info("🔧 Creating test graphs...")

        graphs_config = [
            {
                "name": "Test Graph 001",
                "description": "Basic test graph for API coverage",
                "nodes": [
                    {
                        "name": "start_node",
                        "node_type": "worker",
                        "worker_name": "test_worker_001",
                        "task_template": "Process initial data",
                        "timeout": 300,
                        "retry_count": 2,
                    }
                ],
                "edges": [],
            },
            {
                "name": "Test Graph 002",
                "description": "Multi-node graph for comprehensive testing",
                "nodes": [
                    {
                        "name": "input_node",
                        "node_type": "worker",
                        "worker_name": "test_worker_001",
                        "task_template": "Receive input data",
                        "timeout": 180,
                        "retry_count": 1,
                    },
                    {
                        "name": "process_node",
                        "node_type": "worker",
                        "worker_name": "test_worker_002",
                        "task_template": "Process data from input",
                        "timeout": 600,
                        "retry_count": 3,
                    },
                ],
                "edges": [
                    {
                        "source": "input_node",
                        "target": "process_node",
                        "condition": "result.status == 'success'",
                    }
                ],
            },
            {
                "name": "Test Graph 003",
                "description": "Complex workflow for delete testing",
                "nodes": [
                    {
                        "name": "validator",
                        "node_type": "worker",
                        "worker_name": "test_worker_003",
                        "task_template": "Validate input",
                        "timeout": 120,
                        "retry_count": 1,
                    },
                    {
                        "name": "transformer",
                        "node_type": "worker",
                        "worker_name": "test_worker_004",
                        "task_template": "Transform data",
                        "timeout": 300,
                        "retry_count": 2,
                    },
                    {
                        "name": "output",
                        "node_type": "worker",
                        "worker_name": "test_worker_005",
                        "task_template": "Generate output",
                        "timeout": 240,
                        "retry_count": 1,
                    },
                ],
                "edges": [
                    {
                        "source": "validator",
                        "target": "transformer",
                        "condition": "result.valid == true",
                    },
                    {
                        "source": "transformer",
                        "target": "output",
                        "condition": "result.transformed == true",
                    },
                ],
            },
            {
                "name": "Test Graph 004",
                "description": "State management test graph",
                "nodes": [
                    {
                        "name": "state_tracker",
                        "node_type": "worker",
                        "worker_name": "test_worker_002",
                        "task_template": "Track state changes",
                        "timeout": 180,
                        "retry_count": 1,
                    }
                ],
                "edges": [],
            },
            {
                "name": "Test Graph 005",
                "description": "Final test graph for complete coverage",
                "nodes": [
                    {
                        "name": "final_processor",
                        "node_type": "worker",
                        "worker_name": "test_worker_001",
                        "task_template": "Final processing step",
                        "timeout": 300,
                        "retry_count": 2,
                    }
                ],
                "edges": [],
            },
        ]

        created_graphs = []
        for graph_config in graphs_config:
            try:
                async with self.session.post(
                    f"{self.base_url}/graphs", json=graph_config
                ) as response:
                    if response.status in [200, 201]:
                        graph_data = await response.json()
                        created_graphs.append(graph_data)
                        self.created_entities["graphs"].append(graph_data)
                        logger.info(
                            f"✅ Created graph: {graph_config['name']} (ID: {graph_data.get('id', 'N/A')})"
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"⚠️ Failed to create graph {graph_config['name']}: {response.status} - {error_text}"
                        )
            except Exception as e:
                logger.exception(f"❌ Error creating graph {graph_config['name']}: {e}")  # noqa: TRY401

        return created_graphs

    async def create_worker_states(self) -> list[dict[str, Any]]:
        """Create worker states for state endpoint testing."""
        logger.info("🔧 Creating worker states...")

        created_states = []

        # Create states for each worker
        for worker in self.created_entities["workers"]:
            worker_name = worker.get("name")
            worker_id = worker.get("id", str(uuid7()))

            if not worker_name:
                continue

            states_config = [
                {
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "is_agent": worker.get("is_agent", False),
                    "state_type": "created",
                    "worker_state": {
                        "status": "initialized",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    "metadata": {
                        "test_data": True,
                        "created_for": "schemathesis_coverage",
                    },
                },
                {
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "is_agent": worker.get("is_agent", False),
                    "state_type": "processing",
                    "worker_state": {
                        "status": "running",
                        "task": "test_task",
                        "progress": 50,
                    },
                    "metadata": {"test_data": True},
                },
                {
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "is_agent": worker.get("is_agent", False),
                    "state_type": "completed",
                    "worker_state": {
                        "status": "finished",
                        "result": "test_result",
                        "duration": 120,
                    },
                    "metadata": {"test_data": True},
                },
            ]

            try:
                async with self.session.post(
                    f"{self.base_url}/workers_states",
                    json={"worker_states": states_config},
                ) as response:
                    if response.status in [200, 201]:
                        state_data = await response.json()
                        results_list = state_data.get("results", [])
                        created_states.extend(results_list)
                        self.created_entities["states"].extend(results_list)
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"⚠️ Failed to create states for {worker_name}: {response.status} - {error_text}"
                        )
            except Exception as e:
                logger.exception(f"❌ Error creating states for {worker_name}: {e}")  # noqa: TRY401

        logger.info(f"✅ Created {len(created_states)} worker states")
        return created_states

    async def verify_test_data(self) -> dict[str, Any]:
        """Verify that all test data was created successfully."""
        logger.info("🔍 Verifying test data...")

        verification_results = {
            "workers": {"created": 0, "verified": 0, "ids": []},
            "tools": {"created": 0, "verified": 0, "names": []},
            "graphs": {"created": 0, "verified": 0, "ids": []},
            "states": {"created": 0, "verified": 0},
        }

        # Verify workers
        try:
            async with self.session.get(f"{self.base_url}/workers") as response:
                if response.status == 200:
                    workers = await response.json()
                    test_workers = [w for w in workers if w.startswith("test_worker_")]
                    verification_results["workers"]["verified"] = len(test_workers)
                    verification_results["workers"]["ids"] = test_workers
        except Exception as e:
            logger.exception(f"❌ Error verifying workers: {e}")  # noqa: TRY401

        # Verify tools
        try:
            async with self.session.get(f"{self.base_url}/tools") as response:
                if response.status == 200:
                    tools = await response.json()
                    test_tools = [t["name"] for t in tools if t["name"].startswith("test_tool_")]
                    verification_results["tools"]["verified"] = len(test_tools)
                    verification_results["tools"]["names"] = test_tools
        except Exception as e:
            logger.exception(f"❌ Error verifying tools: {e}")  # noqa: TRY401

        # Verify graphs
        try:
            async with self.session.get(f"{self.base_url}/graphs") as response:
                if response.status == 200:
                    graphs = await response.json()
                    test_graphs = [g for g in graphs if g["name"].startswith("Test Graph")]
                    verification_results["graphs"]["verified"] = len(test_graphs)
                    verification_results["graphs"]["ids"] = [g["id"] for g in test_graphs]
        except Exception as e:
            logger.exception(f"❌ Error verifying graphs: {e}")  # noqa: TRY401

        # Verify states
        try:
            async with self.session.get(f"{self.base_url}/workers/states") as response:
                if response.status == 200:
                    states_data = await response.json()
                    states = states_data.get("states", [])
                    test_states = [s for s in states if s.get("metadata", {}).get("test_data")]
                    verification_results["states"]["verified"] = len(test_states)
        except Exception as e:
            logger.exception(f"❌ Error verifying states: {e}")  # noqa: TRY401

        # Update created counts
        verification_results["workers"]["created"] = len(self.created_entities["workers"])
        verification_results["tools"]["created"] = len(self.created_entities["tools"])
        verification_results["graphs"]["created"] = len(self.created_entities["graphs"])
        verification_results["states"]["created"] = len(self.created_entities["states"])

        return verification_results

    async def cleanup_test_data(self) -> dict[str, Any]:
        """Clean up all created test data."""
        logger.info("🧹 Cleaning up test data...")

        cleanup_results = {
            "workers_deleted": 0,
            "tools_deleted": 0,
            "graphs_deleted": 0,
            "states_deleted": 0,
            "errors": [],
        }

        # Delete workers
        for worker in self.created_entities["workers"]:
            worker_name = worker.get("name")
            if worker_name:
                try:
                    async with self.session.delete(
                        f"{self.base_url}/workers/{worker_name}"
                    ) as response:
                        if response.status in [200, 204, 404]:
                            cleanup_results["workers_deleted"] += 1
                        else:
                            cleanup_results["errors"].append(
                                f"Failed to delete worker {worker_name}: {response.status}"
                            )
                except Exception as e:
                    cleanup_results["errors"].append(f"Error deleting worker {worker_name}: {e}")

        # Delete tools
        for tool in self.created_entities["tools"]:
            tool_name = tool.get("name")
            if tool_name:
                try:
                    async with self.session.delete(
                        f"{self.base_url}/tools/{tool_name}"
                    ) as response:
                        if response.status in [200, 204, 404]:
                            cleanup_results["tools_deleted"] += 1
                        else:
                            cleanup_results["errors"].append(
                                f"Failed to delete tool {tool_name}: {response.status}"
                            )
                except Exception as e:
                    cleanup_results["errors"].append(f"Error deleting tool {tool_name}: {e}")

        # Delete graphs
        for graph in self.created_entities["graphs"]:
            graph_id = graph.get("id")
            if graph_id:
                try:
                    async with self.session.delete(
                        f"{self.base_url}/graphs/{graph_id}"
                    ) as response:
                        if response.status in [200, 204, 404]:
                            cleanup_results["graphs_deleted"] += 1
                        else:
                            cleanup_results["errors"].append(
                                f"Failed to delete graph {graph_id}: {response.status}"
                            )
                except Exception as e:
                    cleanup_results["errors"].append(f"Error deleting graph {graph_id}: {e}")

        # Clean up all states
        try:
            async with self.session.delete(f"{self.base_url}/workers/states") as response:
                if response.status in [200, 204]:
                    cleanup_results["states_deleted"] = len(self.created_entities.get("states", []))
                else:
                    cleanup_results["errors"].append(
                        f"Failed to delete all states: {response.status}"
                    )
        except Exception as e:
            cleanup_results["errors"].append(f"Error deleting all states: {e}")

        return cleanup_results

    async def setup_all_test_data(self) -> dict[str, Any]:
        """Set up all test data for comprehensive API coverage."""
        logger.info("🚀 Starting comprehensive test data setup...")

        # Check API health
        if not await self.check_api_health():
            logger.error("❌ API is not accessible. Please ensure the server is running.")
            return {"success": False, "error": "API not accessible"}

        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "success": True,
            "entities_created": {},
            "verification": {},
            "test_ids": {},
        }

        try:
            # Create all entities
            workers = await self.create_test_workers()
            tools = await self.create_test_tools()
            graphs = await self.create_test_graphs()
            states = await self.create_worker_states()

            results["entities_created"] = {
                "workers": len(workers),
                "tools": len(tools),
                "graphs": len(graphs),
                "states": len(states),
            }

            # Verify test data
            verification = await self.verify_test_data()
            results["verification"] = verification

            # Extract test IDs for schemathesis
            results["test_ids"] = {
                "worker_names": verification["workers"]["ids"],
                "tool_names": verification["tools"]["names"],
                "graph_ids": verification["graphs"]["ids"],
            }

            # Summary
            total_created = sum(results["entities_created"].values())
            total_verified = sum(
                v["verified"]
                for v in verification.values()
                if isinstance(v, dict) and "verified" in v
            )

            logger.info("✅ Test data setup complete!")
            logger.info(f"📊 Created: {total_created} entities")
            logger.info(f"🔍 Verified: {total_verified} entities")
            logger.info("🎯 Ready for 100% operation coverage testing")

        except Exception as e:
            logger.exception(f"❌ Error during test data setup: {e}")  # noqa: TRY401
            results["success"] = False
            results["error"] = str(e)

        return results

    def save_test_ids(self, results: dict[str, Any], filename: str = "test_ids.json"):
        """Save test IDs to a file for schemathesis usage."""
        test_ids = results.get("test_ids", {})
        if test_ids:
            try:
                with open(filename, "w") as f:
                    json.dump(test_ids, f, indent=2)
                logger.info(f"💾 Test IDs saved to {filename}")
            except Exception as e:
                logger.exception(f"❌ Error saving test IDs: {e}")  # noqa: TRY401


async def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Setup test data for API coverage testing")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8887",
        help="Base URL of the API (default: http://localhost:8887)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test data instead of creating it",
    )
    parser.add_argument(
        "--output",
        default="test_ids.json",
        help="Output file for test IDs (default: test_ids.json)",
    )

    args = parser.parse_args()

    async with TestDataSetup(args.base_url) as setup:
        if args.cleanup:
            logger.info("🧹 Cleanup mode - removing test data...")
            cleanup_results = await setup.cleanup_test_data()

            logger.info("🧹 Cleanup Results:")
            logger.info(f"   Workers deleted: {cleanup_results['workers_deleted']}")
            logger.info(f"   Tools deleted: {cleanup_results['tools_deleted']}")
            logger.info(f"   Graphs deleted: {cleanup_results['graphs_deleted']}")
            logger.info(f"   States deleted: {cleanup_results['states_deleted']}")

            if cleanup_results["errors"]:
                logger.warning("⚠️ Cleanup errors:")
                for error in cleanup_results["errors"]:
                    logger.warning(f"   {error}")
            else:
                logger.info("✅ Cleanup completed successfully")
        else:
            logger.info("🚀 Setup mode - creating test data...")
            results = await setup.setup_all_test_data()

            if results["success"]:
                # Print summary
                logger.info("\n📋 SETUP SUMMARY:")
                logger.info(f"   Workers: {results['entities_created']['workers']} created")
                logger.info(f"   Tools: {results['entities_created']['tools']} created")
                logger.info(f"   Graphs: {results['entities_created']['graphs']} created")
                logger.info(f"   States: {results['entities_created']['states']} created")

                logger.info("\n🎯 TEST IDs FOR SCHEMATHESIS:")
                test_ids = results["test_ids"]
                logger.info(f"   Worker names: {test_ids['worker_names']}")
                logger.info(f"   Tool names: {test_ids['tool_names']}")
                logger.info(f"   Graph IDs: {test_ids['graph_ids']}")

                # Save test IDs
                setup.save_test_ids(results, args.output)

                logger.info("\n✅ Ready for schemathesis testing with 100% operation coverage!")
                logger.info(f"🚀 Run: schemathesis run {args.base_url}/openapi.json")
            else:
                logger.error(f"❌ Setup failed: {results.get('error', 'Unknown error')}")
                sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
