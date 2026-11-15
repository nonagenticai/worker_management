# Graph Execution Examples - Complete usage examples for the graph execution system

import asyncio
from typing import Any, cast
from uuid import uuid7

import httpx

# Base URL for the API (adjust as needed)
API_BASE_URL = "http://localhost:8887"


class GraphExecutionClient:
    """Client for interacting with the graph execution API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def create_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new graph"""
        response = await self.client.post(f"{self.base_url}/graphs/", json=graph_data)
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def execute_graph_full(
        self, graph_id: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a graph in full mode"""
        request_data = {"variables": variables or {}, "execution_mode": "full"}
        response = await self.client.post(
            f"{self.base_url}/graphs/{graph_id}/execute", json=request_data
        )
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def execute_graph_stepwise(
        self, graph_id: str, variables: dict[str, Any] | None = None
    ) -> str:
        """Start stepwise execution and return session ID"""
        request_data = {"variables": variables or {}, "execution_mode": "stepwise"}
        response = await self.client.post(
            f"{self.base_url}/graphs/{graph_id}/execute", json=request_data
        )
        response.raise_for_status()
        json_response = cast("dict[str, Any]", response.json())
        return cast("str", json_response["session_id"])

    async def execute_next_step(self, graph_id: str, session_id: str) -> dict[str, Any]:
        """Execute next step in stepwise execution"""
        response = await self.client.post(
            f"{self.base_url}/graphs/{graph_id}/execute/{session_id}/next"
        )
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def get_execution_status(self, graph_id: str, session_id: str) -> dict[str, Any]:
        """Get execution status"""
        response = await self.client.get(
            f"{self.base_url}/graphs/{graph_id}/execute/{session_id}/status"
        )
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    # Template management methods removed - no longer supported
    # Graph execution now uses pydantic_ai only

    async def preview_templates(self, graph_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Preview template resolution for a graph"""
        request_data = {"variables": variables}
        response = await self.client.post(
            f"{self.base_url}/graphs/{graph_id}/preview", json=request_data
        )
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def validate_templates(self, graph_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Validate templates for a graph"""
        request_data = {"variables": variables}
        response = await self.client.post(
            f"{self.base_url}/graphs/{graph_id}/validate", json=request_data
        )
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Example 1: Simple Linear Processing Pipeline
async def example_linear_pipeline():
    """Example of a simple linear processing pipeline"""
    print("\n=== Example 1: Linear Processing Pipeline ===")

    client = GraphExecutionClient()

    try:
        # Define a simple 3-node linear pipeline
        graph_data = {
            "name": "Data Processing Pipeline",
            "nodes": [
                {
                    "id": str(uuid7()),
                    "name": "Data Ingestion",
                    "worker_name": "data_worker",
                    "task_template": "Ingest data from source: {data_source}",
                    "data": {"step": 1},
                    "timeout": 300,
                    "retry_count": 1,
                },
                {
                    "id": str(uuid7()),
                    "name": "Data Processing",
                    "worker_name": "processing_worker",
                    "task_template": "Process the data: {nodes[0].output} with parameters: {processing_params}",
                    "data": {"step": 2},
                    "timeout": 600,
                    "retry_count": 2,
                },
                {
                    "id": str(uuid7()),
                    "name": "Data Output",
                    "worker_name": "output_worker",
                    "task_template": "Output processed data: {nodes[1].output} to destination: {output_destination}",
                    "data": {"step": 3},
                    "timeout": 300,
                    "retry_count": 1,
                },
            ],
            "edges": [
                {
                    "source": "nodes[0].id",  # This would be replaced with actual UUIDs
                    "target": "nodes[1].id",
                },
                {"source": "nodes[1].id", "target": "nodes[2].id"},
            ],
        }

        # Create the graph
        graph = await client.create_graph(graph_data)
        graph_id = graph["id"]
        print(f"Created graph: {graph_id}")

        # Preview templates before execution
        variables = {
            "data_source": "database://production/users",
            "processing_params": "normalize=true,filter=active",
            "output_destination": "s3://processed-data/users/",
        }

        preview = await client.preview_templates(graph_id, variables)
        print("Template Preview:")
        for node_id, resolved_task in preview["resolved_tasks"].items():
            print(f"  Node {node_id}: {resolved_task}")

        # Execute the graph
        print("\nExecuting graph...")
        result = await client.execute_graph_full(graph_id, variables)

        print(f"Execution Status: {result['status']}")
        if result["status"] == "completed":
            print("All node outputs:")
            for node_id, output in result["all_outputs"].items():
                print(f"  {node_id}: {output}")

    except Exception as e:
        print(f"Error in linear pipeline example: {e}")
    finally:
        await client.close()


# Example 2: Parallel Processing with Convergence
async def example_parallel_processing():
    """Example of parallel processing with convergence"""
    print("\n=== Example 2: Parallel Processing ===")

    client = GraphExecutionClient()

    try:
        # Define a graph with parallel branches that converge
        graph_data = {
            "name": "Parallel Data Analysis",
            "nodes": [
                {
                    "id": str(uuid7()),
                    "name": "Data Source",
                    "worker_name": "data_worker",
                    "task_template": "Load dataset: {dataset_path}",
                    "data": {"type": "source"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Statistical Analysis",
                    "worker_name": "stats_worker",
                    "task_template": "Perform statistical analysis on: {nodes[0].output}",
                    "data": {"type": "analysis", "branch": "stats"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Machine Learning",
                    "worker_name": "ml_worker",
                    "task_template": "Train ML model on: {nodes[0].output} with algorithm: {ml_algorithm}",
                    "data": {"type": "analysis", "branch": "ml"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Visualization",
                    "worker_name": "viz_worker",
                    "task_template": "Create visualizations from: {nodes[0].output}",
                    "data": {"type": "analysis", "branch": "viz"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Report Generation",
                    "worker_name": "report_worker",
                    "task_template": "Generate report combining stats: {nodes[1].output}, ML results: {nodes[2].output}, and visualizations: {nodes[3].output}",
                    "data": {"type": "output"},
                },
            ],
            "edges": [
                # Source to all parallel branches
                {"source": "nodes[0].id", "target": "nodes[1].id"},
                {"source": "nodes[0].id", "target": "nodes[2].id"},
                {"source": "nodes[0].id", "target": "nodes[3].id"},
                # All branches converge to report
                {"source": "nodes[1].id", "target": "nodes[4].id"},
                {"source": "nodes[2].id", "target": "nodes[4].id"},
                {"source": "nodes[3].id", "target": "nodes[4].id"},
            ],
        }

        # Create graph
        graph = await client.create_graph(graph_data)
        graph_id = graph["id"]
        print(f"Created parallel processing graph: {graph_id}")

        # Validate templates
        validation = await client.validate_templates(graph_id, {})
        print(f"Template validation: {validation['summary']['overall_status']}")
        if validation["summary"]["invalid_nodes"] > 0:
            print("Invalid templates found:")
            for result in validation["validation_results"]:
                if result["status"] == "invalid":
                    print(f"  Node {result['node_id']}: {result.get('error', 'Unknown error')}")

        # Execute with variables
        variables = {
            "dataset_path": "/data/customer_analytics.csv",
            "ml_algorithm": "random_forest",
        }

        result = await client.execute_graph_full(graph_id, variables)
        print(f"Parallel execution status: {result['status']}")

    except Exception as e:
        print(f"Error in parallel processing example: {e}")
    finally:
        await client.close()


# Example 3: Stepwise Execution with Monitoring
async def example_stepwise_execution():
    """Example of stepwise execution with real-time monitoring"""
    print("\n=== Example 3: Stepwise Execution ===")

    client = GraphExecutionClient()

    try:
        # Simple 3-step workflow
        graph_data = {
            "name": "Content Creation Workflow",
            "nodes": [
                {
                    "id": str(uuid7()),
                    "name": "Research",
                    "worker_name": "research_worker",
                    "task_template": "Research topic: {topic} for audience: {audience}",
                    "data": {"phase": "research"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Content Creation",
                    "worker_name": "content_worker",
                    "task_template": "Create content based on research: {nodes[0].output} in format: {content_format}",
                    "data": {"phase": "creation"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Review and Publish",
                    "worker_name": "publish_worker",
                    "task_template": "Review and publish content: {nodes[1].output} to platform: {platform}",
                    "data": {"phase": "publish"},
                },
            ],
            "edges": [
                {"source": "nodes[0].id", "target": "nodes[1].id"},
                {"source": "nodes[1].id", "target": "nodes[2].id"},
            ],
        }

        # Create graph
        graph = await client.create_graph(graph_data)
        graph_id = graph["id"]
        print(f"Created content workflow graph: {graph_id}")

        # Start stepwise execution
        variables = {
            "topic": "AI in Healthcare",
            "audience": "medical professionals",
            "content_format": "blog_post",
            "platform": "company_blog",
        }

        session_id = await client.execute_graph_stepwise(graph_id, variables)
        print(f"Started stepwise execution: {session_id}")

        # Execute steps one by one with monitoring
        step_count = 0
        while True:
            step_count += 1
            print(f"\n--- Step {step_count} ---")

            # Get current status
            status = await client.get_execution_status(graph_id, session_id)
            print(f"Current status: {status['status']}")
            print(f"Completed nodes: {status.get('completed_nodes', [])}")

            # Execute next step
            result = await client.execute_next_step(graph_id, session_id)
            print(f"Step result: {result['status']}")

            if result.get("current_node"):
                print(f"Executed node: {result['current_node']}")

            # Show node outputs
            if result["node_outputs"]:
                print("Node outputs so far:")
                for node_id, output in result["node_outputs"].items():
                    print(f"  {node_id}: {output[:100]}...")  # Truncate long outputs

            # Check if done
            if not result.get("has_more_steps", False):
                print("\nWorkflow completed!")
                if result.get("final_output"):
                    print(f"Final output: {result['final_output']}")
                break

            # Simulate some delay between steps
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error in stepwise execution example: {e}")
    finally:
        await client.close()


# Example 4: Error Handling and Recovery
async def example_error_handling():
    """Example demonstrating error handling and recovery"""
    print("\n=== Example 4: Error Handling ===")

    client = GraphExecutionClient()

    try:
        # Graph with potential failure points
        graph_data = {
            "name": "Robust Processing Pipeline",
            "nodes": [
                {
                    "id": str(uuid7()),
                    "name": "Reliable Start",
                    "worker_name": "reliable_worker",
                    "task_template": "Start processing: {input_data}",
                    "data": {"reliability": "high"},
                    "timeout": 30,
                    "retry_count": 0,
                },
                {
                    "id": str(uuid7()),
                    "name": "Potentially Failing Step",
                    "worker_name": "unreliable_worker",  # This worker might not exist
                    "task_template": "Process with potential failure: {nodes[0].output}",
                    "data": {"reliability": "low"},
                    "timeout": 60,
                    "retry_count": 3,  # More retries for unreliable step
                },
                {
                    "id": str(uuid7()),
                    "name": "Cleanup and Finish",
                    "worker_name": "cleanup_worker",
                    "task_template": "Cleanup and finish processing: {nodes[1].output}",
                    "data": {"reliability": "medium"},
                    "timeout": 45,
                    "retry_count": 1,
                },
            ],
            "edges": [
                {"source": "nodes[0].id", "target": "nodes[1].id"},
                {"source": "nodes[1].id", "target": "nodes[2].id"},
            ],
        }

        # Create graph
        graph = await client.create_graph(graph_data)
        graph_id = graph["id"]
        print(f"Created error handling demo graph: {graph_id}")

        # Execute and handle potential errors
        variables = {"input_data": "critical_business_data.json"}

        try:
            result = await client.execute_graph_full(graph_id, variables)

            if result["status"] == "completed":
                print("✅ Execution completed successfully")
                print(f"Final output: {result.get('final_output', 'No output')}")
            elif result["status"] == "failed":
                print("❌ Execution failed")
                print(f"Error: {result.get('error_message', 'Unknown error')}")
                print(f"Failed nodes: {result.get('failed_nodes', [])}")
                print(f"Node errors: {result.get('node_errors', {})}")

                # Show what did complete successfully
                if result.get("completed_nodes"):
                    print(f"Successfully completed nodes: {result['completed_nodes']}")
                    print("Partial outputs:")
                    for node_id, output in result.get("all_outputs", {}).items():
                        print(f"  {node_id}: {output}")

        except Exception as api_error:
            print(f"❌ API Error: {api_error}")

            # Could implement recovery strategies here
            print("🔄 Could implement recovery strategies:")
            print("  - Retry with different workers")
            print("  - Skip failed nodes and continue")
            print("  - Rollback to last successful state")

    except Exception as e:
        print(f"Error in error handling example: {e}")
    finally:
        await client.close()


# Example 5: Complex Conditional Workflow
async def example_conditional_workflow():
    """Example of a workflow with conditional branching"""
    print("\n=== Example 5: Conditional Workflow ===")

    client = GraphExecutionClient()

    try:
        # Workflow that branches based on analysis results
        graph_data = {
            "name": "Conditional Document Processing",
            "nodes": [
                {
                    "id": str(uuid7()),
                    "name": "Document Analysis",
                    "worker_name": "analysis_worker",
                    "task_template": "Analyze document type and content: {document_path}",
                    "data": {"type": "analysis"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Text Processing",
                    "worker_name": "text_worker",
                    "task_template": "Process text document: {nodes[0].output} with extraction: {text_extraction_rules}",
                    "data": {"document_type": "text", "conditional": True},
                },
                {
                    "id": str(uuid7()),
                    "name": "Image Processing",
                    "worker_name": "image_worker",
                    "task_template": "Process image document: {nodes[0].output} with OCR settings: {ocr_settings}",
                    "data": {"document_type": "image", "conditional": True},
                },
                {
                    "id": str(uuid7()),
                    "name": "PDF Processing",
                    "worker_name": "pdf_worker",
                    "task_template": "Process PDF document: {nodes[0].output} with options: {pdf_options}",
                    "data": {"document_type": "pdf", "conditional": True},
                },
                {
                    "id": str(uuid7()),
                    "name": "Quality Check",
                    "worker_name": "quality_worker",
                    "task_template": "Quality check on processed content from any source",
                    "data": {"type": "quality_control"},
                },
                {
                    "id": str(uuid7()),
                    "name": "Final Output",
                    "worker_name": "output_worker",
                    "task_template": "Generate final output after quality check: {nodes[4].output}",
                    "data": {"type": "output"},
                },
            ],
            "edges": [
                # Analysis connects to all processing types
                {"source": "nodes[0].id", "target": "nodes[1].id"},
                {"source": "nodes[0].id", "target": "nodes[2].id"},
                {"source": "nodes[0].id", "target": "nodes[3].id"},
                # All processing types connect to quality check
                {"source": "nodes[1].id", "target": "nodes[4].id"},
                {"source": "nodes[2].id", "target": "nodes[4].id"},
                {"source": "nodes[3].id", "target": "nodes[4].id"},
                # Quality check to final output
                {"source": "nodes[4].id", "target": "nodes[5].id"},
            ],
        }

        # Create graph
        graph = await client.create_graph(graph_data)
        graph_id = graph["id"]
        print(f"Created conditional workflow graph: {graph_id}")

        # Test with different document types
        test_cases = [
            {
                "name": "Text Document",
                "variables": {
                    "document_path": "/docs/report.txt",
                    "text_extraction_rules": "extract_headers=true,preserve_formatting=false",
                    "ocr_settings": "language=en,dpi=300",
                    "pdf_options": "extract_images=false",
                },
            },
            {
                "name": "Image Document",
                "variables": {
                    "document_path": "/docs/scanned_form.png",
                    "text_extraction_rules": "extract_headers=true",
                    "ocr_settings": "language=en,dpi=600,preprocessing=true",
                    "pdf_options": "extract_images=true",
                },
            },
        ]

        for test_case in test_cases:
            test_case = cast("dict[str, Any]", test_case)
            print(f"\n--- Testing: {test_case['name']} ---")

            # Preview what will be executed
            variables = cast("dict[str, Any]", test_case["variables"])
            preview = await client.preview_templates(graph_id, variables)
            print("Resolved templates:")
            for task in preview["resolved_tasks"].values():
                print(f"  {task}")

            # Execute
            result = await client.execute_graph_full(graph_id, variables)
            print(f"Result: {result['status']}")

            if result["status"] == "completed":
                print(f"Completed nodes: {len(result['completed_nodes'])}")
            elif result["status"] == "failed":
                print(f"Failed at nodes: {result.get('failed_nodes', [])}")

    except Exception as e:
        print(f"Error in conditional workflow example: {e}")
    finally:
        await client.close()


# Main execution function
async def run_all_examples():
    """Run all examples"""
    print("🚀 Graph Execution System Examples")
    print("=" * 50)

    examples = [
        ("Linear Pipeline", example_linear_pipeline),
        ("Parallel Processing", example_parallel_processing),
        ("Stepwise Execution", example_stepwise_execution),
        ("Error Handling", example_error_handling),
        ("Conditional Workflow", example_conditional_workflow),
    ]

    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            await example_func()
            print(f"✅ Completed: {name}")
        except Exception as e:
            print(f"❌ Failed: {name} - {e}")

        print("-" * 50)

    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(run_all_examples())
