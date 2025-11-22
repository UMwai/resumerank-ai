"""Database connection management for Clinical Trial Signal Detection System."""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import connection as PgConnection

from config import config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling."""

    _instance: Optional["DatabaseConnection"] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pool is None:
            self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password,
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator[PgConnection, None, None]:
        """Get a connection from the pool."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Get a cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor_factory = cursor_factory or extras.RealDictCursor
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute(self, query: str, params: tuple = None) -> list:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return []

    def execute_many(self, query: str, params_list: list) -> None:
        """Execute a query with multiple parameter sets."""
        with self.get_cursor() as cursor:
            extras.execute_batch(cursor, query, params_list)

    def init_schema(self, schema_path: Optional[str] = None) -> None:
        """Initialize the database schema from SQL file."""
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(schema_sql)
                logger.info("Database schema initialized successfully")
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize schema: {e}")
                raise
            finally:
                cursor.close()

    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed")


# Singleton accessor
def get_db_connection() -> DatabaseConnection:
    """Get the database connection singleton."""
    return DatabaseConnection()


class MockDatabaseConnection:
    """Mock database connection for dry-run mode."""

    def __init__(self):
        self._data = {
            "companies": [],
            "trials": [],
            "trial_signals": [],
            "trial_scores": [],
        }

    @contextmanager
    def get_connection(self):
        yield self

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        yield MockCursor(self._data)

    def execute(self, query: str, params: tuple = None) -> list:
        logger.debug(f"[DRY RUN] Execute: {query[:100]}...")
        return []

    def execute_many(self, query: str, params_list: list) -> None:
        logger.debug(f"[DRY RUN] Execute many: {query[:100]}... ({len(params_list)} rows)")

    def init_schema(self, schema_path: Optional[str] = None) -> None:
        logger.info("[DRY RUN] Schema initialization skipped")

    def close(self):
        logger.info("[DRY RUN] Connection closed")


class MockCursor:
    """Mock cursor for dry-run mode."""

    def __init__(self, data: dict):
        self._data = data
        self.description = None

    def execute(self, query: str, params: tuple = None):
        logger.debug(f"[DRY RUN] Cursor execute: {query[:100]}...")

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def close(self):
        pass
