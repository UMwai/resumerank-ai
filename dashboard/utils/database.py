"""
Database connection manager for the Investment Intelligence Dashboard.

Manages connections to all three PostgreSQL databases:
1. Clinical Trial Signals
2. Patent Intelligence
3. Insider Activity + Hiring Signals
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL connections for all three data systems.

    Uses connection pooling for efficient resource management.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connections from configuration.

        Args:
            config: Dictionary with database configurations for each system:
                    {
                        'clinical_trials': {...},
                        'patent_intelligence': {...},
                        'insider_hiring': {...}
                    }
        """
        self._pools: Dict[str, pool.ThreadedConnectionPool] = {}
        self._config = config
        self._initialized = False

    def initialize(self) -> None:
        """Initialize connection pools for all databases."""
        if self._initialized:
            return

        for db_name, db_config in self._config.items():
            try:
                self._pools[db_name] = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    database=db_config.get('database'),
                    user=db_config.get('user'),
                    password=db_config.get('password', ''),
                )
                logger.info(f"Connection pool initialized for {db_name}")
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize pool for {db_name}: {e}")
                # Continue with other databases even if one fails
                continue

        self._initialized = True

    @contextmanager
    def get_connection(self, db_name: str) -> Generator:
        """
        Get a database connection from the specified pool.

        Args:
            db_name: Name of the database ('clinical_trials', 'patent_intelligence', 'insider_hiring')

        Yields:
            Database connection
        """
        if not self._initialized:
            self.initialize()

        if db_name not in self._pools:
            raise ValueError(f"Unknown database: {db_name}")

        conn = None
        try:
            conn = self._pools[db_name].getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error in {db_name}: {e}")
            raise
        finally:
            if conn:
                self._pools[db_name].putconn(conn)

    @contextmanager
    def get_cursor(self, db_name: str) -> Generator:
        """
        Get a cursor with automatic connection management.

        Args:
            db_name: Name of the database

        Yields:
            Database cursor (RealDictCursor)
        """
        with self.get_connection(db_name) as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.

        Args:
            db_name: Name of the database
            query: SQL query string
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        try:
            with self.get_cursor(db_name) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query failed on {db_name}: {e}")
            return []

    def execute_query_one(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query and return a single result.

        Args:
            db_name: Name of the database
            query: SQL query string
            params: Query parameters

        Returns:
            Single result dictionary or None
        """
        results = self.execute_query(db_name, query, params)
        return results[0] if results else None

    def is_connected(self, db_name: str) -> bool:
        """
        Check if a database connection is available.

        Args:
            db_name: Name of the database

        Returns:
            True if connection is available
        """
        if not self._initialized:
            return False

        if db_name not in self._pools:
            return False

        try:
            with self.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
        except Exception:
            return False

    def get_connection_status(self) -> Dict[str, bool]:
        """
        Get connection status for all databases.

        Returns:
            Dictionary mapping database names to connection status
        """
        return {
            db_name: self.is_connected(db_name)
            for db_name in self._config.keys()
        }

    def close(self) -> None:
        """Close all connection pools."""
        for db_name, db_pool in self._pools.items():
            if db_pool:
                db_pool.closeall()
                logger.info(f"Connection pool closed for {db_name}")

        self._pools.clear()
        self._initialized = False


class MockDatabaseManager:
    """
    Mock database manager for testing and demo mode.

    Returns sample data without requiring actual database connections.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize mock database (no-op)."""
        self._initialized = True

    def is_connected(self, db_name: str) -> bool:
        """Always returns True for mock."""
        return True

    def get_connection_status(self) -> Dict[str, bool]:
        """Returns all connected for mock."""
        return {
            'clinical_trials': True,
            'patent_intelligence': True,
            'insider_hiring': True,
        }

    def execute_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Return empty list for mock queries."""
        return []

    def execute_query_one(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Return None for mock queries."""
        return None

    def close(self) -> None:
        """Close mock database (no-op)."""
        self._initialized = False
