"""
Database connection and operations for the Patent Intelligence system.
Provides PostgreSQL connectivity using psycopg2 and SQLAlchemy.
"""

import contextlib
from datetime import date, datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import DictCursor, execute_batch, execute_values
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Database connection manager with connection pooling.
    Supports both raw psycopg2 and SQLAlchemy connections.
    """

    _instance: Optional["DatabaseConnection"] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None

    def __new__(cls, *args, **kwargs) -> "DatabaseConnection":
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "patent_intelligence",
        user: str = "patent_user",
        password: str = "",
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """
        Initialize database connection.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            min_connections: Minimum connections in pool.
            max_connections: Maximum connections in pool.
        """
        if self._initialized:
            return

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections

        self._engine = None
        self._session_factory = None

        self._initialized = True

    def _get_dsn(self) -> str:
        """Get database connection string."""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )

    def _get_connection_url(self) -> str:
        """Get SQLAlchemy connection URL."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def initialize_pool(self) -> None:
        """Initialize the connection pool."""
        if self._pool is None:
            try:
                self._pool = pool.ThreadedConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    self._get_dsn(),
                )
                logger.info(
                    f"Database connection pool initialized "
                    f"(min={self.min_connections}, max={self.max_connections})"
                )
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize connection pool: {e}")
                raise

    @contextlib.contextmanager
    def get_connection(self) -> Generator:
        """
        Get a database connection from the pool.

        Yields:
            Database connection.
        """
        if self._pool is None:
            self.initialize_pool()

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextlib.contextmanager
    def get_cursor(self, cursor_factory=DictCursor) -> Generator:
        """
        Get a database cursor.

        Args:
            cursor_factory: Cursor factory (default: DictCursor).

        Yields:
            Database cursor.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Database query error: {e}")
                raise
            finally:
                cursor.close()

    def get_engine(self):
        """Get SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self._get_connection_url(),
                pool_size=self.min_connections,
                max_overflow=self.max_connections - self.min_connections,
            )
        return self._engine

    def get_session(self):
        """Get SQLAlchemy session."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.get_engine())
        return self._session_factory()

    def execute_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of result dictionaries.
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results

    def execute_write(
        self, query: str, params: Optional[Tuple] = None
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            Number of affected rows.
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_batch_write(
        self, query: str, params_list: List[Tuple], page_size: int = 100
    ) -> int:
        """
        Execute batch INSERT/UPDATE operations.

        Args:
            query: SQL query string with placeholders.
            params_list: List of parameter tuples.
            page_size: Batch size for execution.

        Returns:
            Total number of affected rows.
        """
        with self.get_cursor() as cursor:
            execute_batch(cursor, query, params_list, page_size=page_size)
            return cursor.rowcount

    def execute_values_insert(
        self,
        table: str,
        columns: List[str],
        values: List[Tuple],
        on_conflict: Optional[str] = None,
    ) -> int:
        """
        Efficiently insert multiple rows using execute_values.

        Args:
            table: Table name.
            columns: Column names.
            values: List of value tuples.
            on_conflict: ON CONFLICT clause (e.g., "DO NOTHING").

        Returns:
            Number of inserted rows.
        """
        if not values:
            return 0

        columns_str = ", ".join(columns)
        query = f"INSERT INTO {table} ({columns_str}) VALUES %s"

        if on_conflict:
            query += f" {on_conflict}"

        with self.get_cursor() as cursor:
            execute_values(cursor, query, values)
            return len(values)

    def upsert(
        self,
        table: str,
        data: Dict[str, Any],
        conflict_columns: List[str],
        update_columns: Optional[List[str]] = None,
    ) -> int:
        """
        Insert or update a single row (upsert).

        Args:
            table: Table name.
            data: Dictionary of column-value pairs.
            conflict_columns: Columns to check for conflict.
            update_columns: Columns to update on conflict. If None, updates all.

        Returns:
            Number of affected rows.
        """
        columns = list(data.keys())
        values = list(data.values())

        if update_columns is None:
            update_columns = [c for c in columns if c not in conflict_columns]

        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        conflict_str = ", ".join(conflict_columns)

        update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        query = f"""
            INSERT INTO {table} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_str})
            DO UPDATE SET {update_str}
        """

        return self.execute_write(query, tuple(values))

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """
        result = self.execute_query(query, (table_name,))
        return result[0]["exists"] if result else False

    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]["count"] if result else 0

    def close(self) -> None:
        """Close all database connections."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed")

        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def get_database(config: Optional[Dict[str, Any]] = None) -> DatabaseConnection:
    """
    Get database connection instance.

    Args:
        config: Optional database configuration dictionary.

    Returns:
        DatabaseConnection instance.
    """
    if config:
        return DatabaseConnection(**config)
    return DatabaseConnection()
