import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Imports must be after path manipulation
from src.api.tenant_constants import maintenance_tenant_id
from src.db.worker_registry_db import WorkerRegistryDB
from src.db.workers_state_db import StateDatabase

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting agent cleanup process...")

    # --- Delete all agents from Worker Registry ---
    worker_registry = WorkerRegistryDB()
    logger.info("Attempting to delete all agents from the worker registry...")
    try:
        workers = await worker_registry.list_workers(
            tenant_id=maintenance_tenant_id(), subject_id=""
        )
        reg_deleted_count = 0
        reg_success = True
        for worker_name in workers:
            try:
                deleted = await worker_registry.delete_worker(
                    tenant_id=maintenance_tenant_id(), subject_id="", worker_name=worker_name
                )
                if deleted:
                    reg_deleted_count += 1
            except Exception as e:
                logger.exception(f"Failed to delete worker {worker_name}: {e}")  # noqa: TRY401
                reg_success = False
        if reg_success:
            logger.info(
                f"Successfully deleted {reg_deleted_count} agent(s) from the worker registry."
            )
        else:
            logger.error(
                f"Failed to delete agents from the worker registry. {reg_deleted_count} agents were deleted before failure (if any)."
            )
    except Exception as e:
        logger.error(
            f"An error occurred while deleting agents from the worker registry: {e}",
            exc_info=True,
        )

    # --- Delete all agent states from State Database ---
    logger.info("Attempting to delete all agent states from the state database...")
    try:
        state_db = StateDatabase()
        state_deleted_count = await state_db.delete_all_worker_states(
            tenant_id=maintenance_tenant_id(), subject_id=""
        )
        logger.info(
            f"Successfully deleted {state_deleted_count} agent state(s) from the state database."
        )
    except Exception as e:
        logger.error(
            f"An error occurred while deleting agent states from the state database: {e}",
            exc_info=True,
        )

    logger.info("Agent cleanup process finished.")


if __name__ == "__main__":
    asyncio.run(main())
