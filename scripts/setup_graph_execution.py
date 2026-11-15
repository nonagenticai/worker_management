# Database Migration and Setup Utility

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

from src.api.tenant_constants import maintenance_tenant_id

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Utility for running database migrations"""

    def __init__(self, db_path: Path, migrations_dir: Path):
        self.db_path = db_path
        self.migrations_dir = migrations_dir

    def get_migration_files(self) -> list[Path]:
        """Get all migration files sorted by name"""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory {self.migrations_dir} does not exist")
            return []

        migration_files = list(self.migrations_dir.glob("*.sql"))
        return sorted(migration_files, key=lambda x: x.name)

    def create_migration_table(self, conn: sqlite3.Connection):
        """Create migration tracking table"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_file TEXT UNIQUE NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()

    def get_applied_migrations(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of already applied migrations"""
        try:
            cursor = conn.execute("SELECT migration_file FROM schema_migrations")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return []

    def apply_migration(self, conn: sqlite3.Connection, migration_file: Path):
        """Apply a single migration file"""
        logger.info(f"Applying migration: {migration_file.name}")

        with open(migration_file) as f:
            migration_sql = f.read()

        # Execute migration statements
        statements = migration_sql.split(";")
        for statement in statements:
            statement = statement.strip()  # noqa: PLW2901 -- intentional inline normalize
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.Error as e:
                    logger.exception(f"Error executing statement in {migration_file.name}: {e}")  # noqa: TRY401
                    logger.exception(f"Statement: {statement}")
                    raise

        # Record migration as applied
        conn.execute(
            "INSERT INTO schema_migrations (migration_file) VALUES (?)",
            (migration_file.name,),
        )
        conn.commit()
        logger.info(f"Successfully applied migration: {migration_file.name}")

    def run_migrations(self):
        """Run all pending migrations"""
        logger.info(f"Running migrations for database: {self.db_path}")

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        conn = sqlite3.connect(str(self.db_path))

        try:
            # Create migration tracking table
            self.create_migration_table(conn)

            # Get migration files and applied migrations
            migration_files = self.get_migration_files()
            applied_migrations = self.get_applied_migrations(conn)

            if not migration_files:
                logger.info("No migration files found")
                return

            # Apply pending migrations
            pending_migrations = [f for f in migration_files if f.name not in applied_migrations]

            if not pending_migrations:
                logger.info("All migrations are up to date")
                return

            logger.info(f"Found {len(pending_migrations)} pending migrations")

            for migration_file in pending_migrations:
                self.apply_migration(conn, migration_file)

            logger.info("All migrations completed successfully")

        except Exception as e:
            logger.exception(f"Migration failed: {e}")  # noqa: TRY401
            conn.rollback()
            raise
        finally:
            conn.close()

    def show_migration_status(self):
        """Show status of all migrations"""
        logger.info(f"Migration status for database: {self.db_path}")

        if not self.db_path.exists():
            logger.info("Database does not exist")
            return

        conn = sqlite3.connect(str(self.db_path))

        try:
            migration_files = self.get_migration_files()
            applied_migrations = self.get_applied_migrations(conn)

            if not migration_files:
                logger.info("No migration files found")
                return

            print("\nMigration Status:")
            print("-" * 50)

            for migration_file in migration_files:
                status = "✅ Applied" if migration_file.name in applied_migrations else "❌ Pending"
                print(f"{migration_file.name:<30} {status}")

            print(f"\nTotal migrations: {len(migration_files)}")
            print(f"Applied: {len(applied_migrations)}")
            print(f"Pending: {len(migration_files) - len(applied_migrations)}")

        finally:
            conn.close()


async def setup_databases():
    """Setup all required databases"""
    logger.info("Setting up databases for graph execution system")

    # Define database paths
    if sys.platform.startswith("win"):
        base_path = Path("src/db")
    else:
        base_path = Path(__file__).parent.parent / "src" / "db"

    databases = [
        ("Graph States", base_path / "graph_states.sqlite"),
        ("Graph Execution", base_path / "graph_execution.sqlite"),
        ("Worker Registry", base_path / "worker_registry.sqlite"),
        ("Workers State", base_path / "workers_state.sqlite"),
    ]

    migrations_dir = Path(__file__).parent.parent / "migrations"

    for db_name, db_path in databases:
        logger.info(f"Setting up {db_name} database: {db_path}")

        # Create database-specific migrator
        migrator = DatabaseMigrator(db_path, migrations_dir)

        try:
            migrator.run_migrations()
            logger.info(f"✅ {db_name} database setup completed")
        except Exception as e:
            logger.exception(f"❌ Failed to setup {db_name} database: {e}")  # noqa: TRY401
            raise


async def verify_system():
    """Verify the graph execution system is working"""
    logger.info("Verifying graph execution system")

    try:
        # Test database connections
        from src.db.graph_execution_db import GraphExecutionDatabase
        from src.db.graph_state_db import GraphStateDatabase

        db = GraphStateDatabase()
        graphs = await db.list_graphs(tenant_id=maintenance_tenant_id(), subject_id="")
        logger.info(f"✅ Graph states database: {len(graphs)} graphs found")

        exec_db = GraphExecutionDatabase()
        sessions = await exec_db.list_execution_sessions(
            tenant_id=maintenance_tenant_id(), subject_id="", limit=5
        )
        logger.info(f"✅ Graph execution database: {len(sessions)} sessions found")

        # Test execution service
        from src.worker.graphs.graph_execution_service import get_execution_service

        get_execution_service()
        logger.info("✅ Graph execution service initialized")

        # Test monitoring
        from src.worker.graphs.graph_monitoring import get_monitor, get_persistence

        get_monitor()
        get_persistence()
        logger.info("✅ Graph monitoring system initialized")

        logger.info("🎉 Graph execution system verification completed successfully")

    except Exception as e:
        logger.exception(f"❌ System verification failed: {e}")  # noqa: TRY401
        raise


def main():
    """Main function for database setup and verification"""
    import argparse

    parser = argparse.ArgumentParser(description="Database Migration and Setup Utility")
    parser.add_argument(
        "command",
        choices=["migrate", "status", "setup", "verify"],
        help="Command to run",
    )
    parser.add_argument(
        "--database",
        help="Specific database to operate on (graph_states, graph_execution, worker_registry, workers_state)",
    )

    args = parser.parse_args()

    # Define paths
    base_path = Path(__file__).parent.parent / "src" / "db"
    migrations_dir = Path(__file__).parent.parent / "migrations"

    if args.command == "migrate":
        if args.database:
            db_path = base_path / f"{args.database}.sqlite"
            migrator = DatabaseMigrator(db_path, migrations_dir)
            migrator.run_migrations()
        else:
            asyncio.run(setup_databases())

    elif args.command == "status":
        if args.database:
            db_path = base_path / f"{args.database}.sqlite"
            migrator = DatabaseMigrator(db_path, migrations_dir)
            migrator.show_migration_status()
        else:
            # Show status for all databases
            databases = [
                "graph_states",
                "graph_execution",
                "worker_registry",
                "workers_state",
            ]
            for db_name in databases:
                db_path = base_path / f"{db_name}.sqlite"
                migrator = DatabaseMigrator(db_path, migrations_dir)
                print(f"\n{db_name.upper()} DATABASE:")
                migrator.show_migration_status()

    elif args.command == "setup":
        asyncio.run(setup_databases())

    elif args.command == "verify":
        asyncio.run(verify_system())


if __name__ == "__main__":
    main()
