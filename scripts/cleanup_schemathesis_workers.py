#!/usr/bin/env python3
"""
Script to clean up schemathesis-generated workers from the SQLite database.

This script deletes workers with unicode character names that were created
during schemathesis testing.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add the project root to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.tenant_constants import maintenance_tenant_id
from src.db.worker_registry_db import WorkerRegistryDB

# The list of schemathesis-generated worker names to delete
SCHEMATHESIS_WORKERS = [
    "",
    "\u0002n£r\u0005B 𩎠HË:򫦫6𜇌×\u0012",
    "\u0004",
    "\u000bê\u00106¥\t",
    "\u0010",
    "\u0012",
    '"º\\â󠓻\u0014ÛðY.\\\u0005]򑆘 /𾐧',
    ",\u0007",
    "2𬩠Ç\t򊆦õ¿",
    "=>\u0015\b",
    "?󞱭ÊÚW¿^#ÝPx󒀢j󚍑􀒞\u000fG򆛍Ç_𩾹",
    "H*\u000b",
    "KÉ򁒟\\𚮃",
    "Pb\u0013Û\u0017\u000f",
    "T񷂂`Á\u0006",
    "\\\u001eÇ\u0001",
    "^K\b",
    "`f>UdT$\u00162󺵩󰷮Ø\u000e",
    "{µ?G򍯙ß",
    "~\u0006򭫐µú-ò򺋗\u001d",
    "F¶\u001cZãÏz\u0015",
    "/N𙦭©vø",
    "#ßý񢉷¶+²\u0001\u0017服f򃃺½¥",
    "2\u0015",
    "¢\u001a",
    "¨\u0010",
    "ªn񯓿\u000e",
    "ªN\t\u0007ë-\u0006\u0014\flÐ2\n",
    "´\nãì𖝅\u001aÿ\u0014𤱕Tì¬·",
    "¹\t\u001e󇭞t󥼈Í",
    "¾Ð􅎜±Mï#񞟎D6",
    "Ä¤#LÍ\u001bUª[",
    "Ê\u0010",
    "Ðp¿²®|2r \u0017\ro",
    "ÓÛSq񐚑tg±\u0006",
    "ò¤âvↆÄ\f",
    "øu%\u0019",
    "䣢a2bÇ\u0001",
    "𗕧Ë򩁓(Ê2\u0017",
    "𦺣ç򹁱L\u000b#",
    "𻱼â6ù\t,?",
    "񃠟󅏀񚃂Nì򂖗Ç±a򉷝\u001e",
    "񸚮#¬u񔆙5}s",
    "񻽾\u0003\u001a¤L𚋷\b°¨򺚋R{\ruv",
    "򩆦PS\u0002",
    "󃏼\u001a\rÓ򇔉=",
    "󍢾P𖈌\u0003",
    "󠁩򫄲𲔣ß\u0004\n",
]


async def delete_workers_batch(worker_names: list[str]) -> dict:
    """
    Delete a batch of workers from the database.

    Args:
        worker_names: List of worker names to delete

    Returns:
        Dictionary with deletion results
    """
    registry = WorkerRegistryDB()
    results: dict[str, list[Any]] = {"deleted": [], "not_found": [], "errors": []}

    for worker_name in worker_names:
        try:
            # Attempt to delete the worker
            deleted = await registry.delete_worker(
                tenant_id=maintenance_tenant_id(), subject_id="", worker_name=worker_name
            )

            if deleted:
                results["deleted"].append({"worker_name": worker_name})
                print(f"Deleted worker: {worker_name!r}")
            else:
                results["not_found"].append(worker_name)
                print(f"Worker not found: {worker_name!r}")

        except Exception as e:
            results["errors"].append({"worker_name": worker_name, "error": str(e)})
            print(f"Error deleting worker {worker_name!r}: {e}")

    return results


async def main():
    """Main function to delete schemathesis workers."""
    print("Starting cleanup of schemathesis-generated workers...")
    print(f"Workers to delete: {len(SCHEMATHESIS_WORKERS)}")
    print("=" * 60)

    # Get current worker count
    registry = WorkerRegistryDB()
    try:
        current_workers = await registry.list_workers(
            tenant_id=maintenance_tenant_id(), subject_id=""
        )
        print(f"Current workers in database: {len(current_workers)}")
    except Exception as e:
        print(f"Error getting current worker count: {e}")
        current_workers = []

    print("=" * 60)

    # Delete the workers
    results = await delete_workers_batch(SCHEMATHESIS_WORKERS)

    print("=" * 60)
    print("DELETION SUMMARY:")
    print(f"Successfully deleted: {len(results['deleted'])}")
    print(f"Not found: {len(results['not_found'])}")
    print(f"Errors: {len(results['errors'])}")

    if results["deleted"]:
        print("\nDeleted workers:")
        for worker in results["deleted"]:
            print(f"  - {worker['worker_name']!r}")

    if results["not_found"]:
        print(f"\nWorkers not found ({len(results['not_found'])}):")
        for worker_name in results["not_found"][:5]:  # Show first 5
            print(f"  - {worker_name!r}")
        if len(results["not_found"]) > 5:
            print(f"  ... and {len(results['not_found']) - 5} more")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error['worker_name']!r}: {error['error']}")

    # Get final worker count
    try:
        final_workers = await registry.list_workers(
            tenant_id=maintenance_tenant_id(), subject_id=""
        )
        print(f"\nFinal workers in database: {len(final_workers)}")
        print(f"Net change: {len(final_workers) - len(current_workers)}")
    except Exception as e:
        print(f"Error getting final worker count: {e}")

    print("=" * 60)
    print("Cleanup completed!")

    # Return exit code based on results
    if results["errors"]:
        return 1  # Exit with error if there were errors
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
