# scripts/init_db.py

# Initializes the PostgreSQL database by applying the schema.
#
# Usage:
#     python scripts/init_db.py
#
# Requires DATABASE_URL environment variable to be set.

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def init_postgres():
    """Initialize PostgreSQL database using the schema file."""
    from src.db.db_config import (
        close_pg_pool,
        get_pg_connection,
        init_pg_pool,
    )

    schema_path = project_root / "postgres_schema.sql"
    if not schema_path.exists():
        logger.error("Schema file not found: %s", schema_path)
        sys.exit(1)

    schema_sql = schema_path.read_text()

    await init_pg_pool()
    try:
        async with get_pg_connection() as conn:
            await conn.execute(schema_sql)
        logger.info("PostgreSQL schema applied successfully.")
    except Exception as e:
        logger.exception("Failed to apply PostgreSQL schema: %s", e)  # noqa: TRY401
        raise
    finally:
        await close_pg_pool()


if __name__ == "__main__":
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required.")
        sys.exit(1)
    asyncio.run(init_postgres())
    print("Database initialization complete.")
