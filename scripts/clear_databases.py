#!/usr/bin/env python3
"""
Script to clear all data from PostgreSQL database while preserving the schema.

Requires DATABASE_URL environment variable to be set.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tables to clear, ordered to respect foreign key constraints (children first)
TABLES_TO_CLEAR = [
    "execution_events",
    "node_execution_logs",
    "graph_execution_sessions",
    "graph_states",
    "graphs",
    "worker_states",
    "workload",
    "tools",
    "skills",
    "workers",
]


async def clear_postgres() -> bool:
    """Clear all data from PostgreSQL tables while preserving schema."""
    from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

    await init_pg_pool()
    try:
        async with get_pg_connection() as conn:
            for table in TABLES_TO_CLEAR:
                try:
                    result = await conn.execute(f"DELETE FROM {table}")
                    logger.info("Cleared table %s: %s", table, result)
                except Exception as e:
                    # Table may not exist yet if migrations haven't run
                    logger.warning("Could not clear table %s: %s", table, e)

        logger.info("Successfully cleared PostgreSQL database")
        return True
    except Exception as e:
        logger.exception("Error clearing PostgreSQL database: %s", e)  # noqa: TRY401
        return False
    finally:
        await close_pg_pool()


async def main() -> int:
    """Main function to clear the database."""
    logger.info("Starting database clearing process...")

    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required.")
        return 1

    success = await clear_postgres()

    if success:
        logger.info("Successfully cleared all databases")
        return 0
    else:
        logger.error("Failed to clear the database")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
