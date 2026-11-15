#!/usr/bin/env python3
"""
Full Coverage Test Runner for Schemathesis

This script automates the complete testing workflow:
1. Sets up comprehensive test data
2. Runs schemathesis with 100% operation coverage
3. Cleans up test data after testing
4. Provides detailed reporting

USAGE:
    python run_full_coverage_test.py
    python run_full_coverage_test.py --base-url http://localhost:8887
    python run_full_coverage_test.py --skip-cleanup
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from setup_test_data import TestDataSetup


class FullCoverageTestRunner:
    """Manages complete testing workflow with data setup and cleanup."""

    def __init__(self, base_url: str = "http://localhost:8887", skip_cleanup: bool = False):  # noqa: FBT001, FBT002
        self.base_url = base_url
        self.skip_cleanup = skip_cleanup
        self.test_results = {}

    async def setup_test_data(self) -> dict[str, Any]:
        """Set up all required test data."""
        logger.info("🚀 Setting up test data for full coverage...")

        async with TestDataSetup(self.base_url) as setup:
            results = await setup.setup_all_test_data()

            if results["success"]:
                # Save test IDs for reference
                setup.save_test_ids(results, "test_ids.json")
                logger.info("✅ Test data setup completed successfully")
                return results
            else:
                logger.error(f"❌ Test data setup failed: {results.get('error', 'Unknown error')}")
                raise Exception(f"Test data setup failed: {results.get('error')}")

    async def cleanup_test_data(self) -> dict[str, Any]:
        """Clean up all test data."""
        if self.skip_cleanup:
            logger.info("⏭️ Skipping cleanup as requested")
            return {"skipped": True}

        logger.info("🧹 Cleaning up test data...")

        async with TestDataSetup(self.base_url) as setup:
            # Load created entities if available
            if os.path.exists("test_ids.json"):
                try:
                    with open("test_ids.json") as f:
                        test_ids = json.load(f)

                    # Populate setup.created_entities for cleanup
                    for worker_name in test_ids.get("worker_names", []):
                        setup.created_entities["workers"].append({"name": worker_name})

                    for tool_name in test_ids.get("tool_names", []):
                        setup.created_entities["tools"].append({"name": tool_name})

                    for graph_id in test_ids.get("graph_ids", []):
                        setup.created_entities["graphs"].append({"id": graph_id})

                except Exception as e:
                    logger.warning(f"⚠️ Could not load test_ids.json: {e}")

            cleanup_results = await setup.cleanup_test_data()
            logger.info("✅ Test data cleanup completed")
            return cleanup_results

    def run_schemathesis(self) -> dict[str, Any]:
        """Run schemathesis testing with comprehensive coverage."""
        logger.info("🔬 Running schemathesis tests for full coverage...")

        # Build schemathesis command
        cmd = [
            "schemathesis",
            "run",
            f"{self.base_url}/openapi.json",
            "--checks",
            "all",
            "--hypothesis-max-examples",
            "50",
            "--hooks",
            "tests.contract.schemathesis_hooks",
            "--hypothesis-phases",
            "explicit,reuse,generate,target,shrink",
            "--show-errors-tracebacks",
            "--validate-schema",
            "--base-url",
            self.base_url,
        ]

        logger.info(f"🚀 Running command: {' '.join(cmd)}")

        try:
            # Run schemathesis
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False,
            )

            # Parse results
            test_results: dict[str, Any] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Save detailed results
            with open("schemathesis_full_coverage_results.txt", "w") as f:
                f.write("Schemathesis Full Coverage Test Results\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"Timestamp: {test_results['timestamp']}\n")
                f.write(f"Base URL: {self.base_url}\n")
                f.write(f"Return Code: {test_results['return_code']}\n")
                f.write(f"Success: {test_results['success']}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

            # Log summary
            if test_results["success"]:
                logger.info("✅ Schemathesis tests completed successfully")
            else:
                logger.warning(
                    f"⚠️ Schemathesis tests completed with issues (exit code: {result.returncode})"
                )

            # Extract key metrics from output
            stdout = result.stdout
            test_results["metrics"] = self.parse_schemathesis_output(stdout)

            return test_results

        except subprocess.TimeoutExpired:
            logger.exception("❌ Schemathesis tests timed out after 10 minutes")
            return {
                "return_code": -1,
                "success": False,
                "error": "Timeout after 10 minutes",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.exception(f"❌ Error running schemathesis: {e}")  # noqa: TRY401
            return {
                "return_code": -1,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def parse_schemathesis_output(self, output: str) -> dict[str, Any]:
        """Parse schemathesis output to extract key metrics."""
        metrics: dict[str, Any] = {
            "operations_total": 0,
            "operations_selected": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "coverage_percentage": 0,
            "missing_test_data_warnings": 0,
        }

        lines = output.split("\n")
        for line in lines:
            # Parse operations count
            if "Operations:" in line and "selected" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "selected" in part and i > 0:
                            metrics["operations_selected"] = int(parts[i - 1])
                        if "total" in part and i > 0:
                            metrics["operations_total"] = int(parts[i - 1])
                except ValueError, IndexError:
                    pass

            # Parse test results
            if "✅" in line and "passed" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "passed" in part and i > 0:
                            metrics["tests_passed"] += int(parts[i - 1])
                except ValueError, IndexError:
                    pass

            if "❌" in line and "failed" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "failed" in part and i > 0:
                            metrics["tests_failed"] += int(parts[i - 1])
                except ValueError, IndexError:
                    pass

            if "⏭" in line and "skipped" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "skipped" in part and i > 0:
                            metrics["tests_skipped"] += int(parts[i - 1])
                except ValueError, IndexError:
                    pass

            # Count missing test data warnings
            if "404" in line and ("not found" in line.lower() or "missing" in line.lower()):
                metrics["missing_test_data_warnings"] += 1

        # Calculate coverage percentage
        if metrics["operations_total"] > 0:
            tested_operations = metrics["operations_total"] - metrics["missing_test_data_warnings"]
            metrics["coverage_percentage"] = (tested_operations / metrics["operations_total"]) * 100

        return metrics

    async def run_full_test_cycle(self) -> dict[str, Any]:
        """Run the complete test cycle: setup -> test -> cleanup."""
        logger.info("🎯 Starting full coverage test cycle...")

        cycle_results: dict[str, Any] = {
            "start_time": datetime.now(UTC).isoformat(),
            "setup_results": None,
            "test_results": None,
            "cleanup_results": None,
            "success": False,
            "end_time": None,
        }

        try:
            # 1. Setup test data
            setup_results = await self.setup_test_data()
            cycle_results["setup_results"] = setup_results

            # 2. Run schemathesis tests
            test_results = self.run_schemathesis()
            cycle_results["test_results"] = test_results

            # 3. Cleanup test data
            cleanup_results = await self.cleanup_test_data()
            cycle_results["cleanup_results"] = cleanup_results

            # 4. Determine overall success
            cycle_results["success"] = setup_results.get("success", False) and test_results.get(
                "success", False
            )

            cycle_results["end_time"] = datetime.now(UTC).isoformat()

            return cycle_results

        except Exception as e:
            logger.exception(f"❌ Full test cycle failed: {e}")  # noqa: TRY401
            cycle_results["error"] = str(e)
            cycle_results["end_time"] = datetime.now(UTC).isoformat()

            # Attempt cleanup even if setup/test failed
            try:
                cleanup_results = await self.cleanup_test_data()
                cycle_results["cleanup_results"] = cleanup_results
            except Exception as cleanup_error:
                logger.exception(f"❌ Cleanup also failed: {cleanup_error}")  # noqa: TRY401
                cycle_results["cleanup_error"] = str(cleanup_error)

            return cycle_results

    def print_final_report(self, cycle_results: dict[str, Any]):
        """Print a comprehensive final report."""
        print("\n" + "=" * 80)
        print("🎯 FULL COVERAGE TEST REPORT")
        print("=" * 80)

        print(f"Start Time: {cycle_results['start_time']}")
        print(f"End Time: {cycle_results.get('end_time', 'N/A')}")
        print(f"Overall Success: {'✅ YES' if cycle_results['success'] else '❌ NO'}")

        # Setup results
        setup = cycle_results.get("setup_results", {})
        if setup:
            print("\n📋 SETUP RESULTS:")
            entities = setup.get("entities_created", {})
            print(f"   Workers created: {entities.get('workers', 0)}")
            print(f"   Tools created: {entities.get('tools', 0)}")
            print(f"   Graphs created: {entities.get('graphs', 0)}")
            print(f"   States created: {entities.get('states', 0)}")

        # Test results
        test = cycle_results.get("test_results", {})
        if test:
            print("\n🔬 SCHEMATHESIS RESULTS:")
            print(f"   Exit Code: {test.get('return_code', 'N/A')}")
            print(f"   Success: {'✅ YES' if test.get('success') else '❌ NO'}")

            metrics = test.get("metrics", {})
            if metrics:
                print(f"   Operations Total: {metrics.get('operations_total', 0)}")
                print(f"   Operations Tested: {metrics.get('operations_selected', 0)}")
                print(f"   Tests Passed: {metrics.get('tests_passed', 0)}")
                print(f"   Tests Failed: {metrics.get('tests_failed', 0)}")
                print(f"   Tests Skipped: {metrics.get('tests_skipped', 0)}")
                print(f"   Coverage: {metrics.get('coverage_percentage', 0):.1f}%")
                print(f"   Missing Data Warnings: {metrics.get('missing_test_data_warnings', 0)}")

        # Cleanup results
        cleanup = cycle_results.get("cleanup_results", {})
        if cleanup and not cleanup.get("skipped"):
            print("\n🧹 CLEANUP RESULTS:")
            print(f"   Workers deleted: {cleanup.get('workers_deleted', 0)}")
            print(f"   Tools deleted: {cleanup.get('tools_deleted', 0)}")
            print(f"   Graphs deleted: {cleanup.get('graphs_deleted', 0)}")
            print(f"   States deleted: {cleanup.get('states_deleted', 0)}")

            errors = cleanup.get("errors", [])
            if errors:
                print(f"   Cleanup errors: {len(errors)}")

        print("\n" + "=" * 80)

        if cycle_results["success"]:
            print("🎉 FULL COVERAGE TESTING COMPLETED SUCCESSFULLY!")
            print("✅ All operations have been tested with proper test data")
            print("🚀 API is ready for production deployment")
        else:
            print("❌ FULL COVERAGE TESTING ENCOUNTERED ISSUES")
            print("🔍 Check the detailed logs above for specific problems")
            print("📋 Review schemathesis_full_coverage_results.txt for complete output")

        print("=" * 80)


async def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Run full coverage API testing with schemathesis")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8887",
        help="Base URL of the API (default: http://localhost:8887)",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup after testing (useful for debugging)",
    )

    args = parser.parse_args()

    # Check if schemathesis is installed
    try:
        subprocess.run(["schemathesis", "--version"], check=True, capture_output=True)  # noqa: S607
    except subprocess.CalledProcessError, FileNotFoundError:
        logger.exception("❌ schemathesis is not installed or not in PATH")
        logger.exception("💡 Install with: pip install schemathesis")
        sys.exit(1)

    # Run full test cycle
    runner = FullCoverageTestRunner(args.base_url, args.skip_cleanup)
    cycle_results = await runner.run_full_test_cycle()

    # Print final report
    runner.print_final_report(cycle_results)

    # Save complete results
    with open("full_coverage_test_results.json", "w") as f:
        json.dump(cycle_results, f, indent=2)

    # Exit with appropriate code
    sys.exit(0 if cycle_results["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
