"""
Database connection manager for the Investment Intelligence Dashboard.

Manages connections to all three PostgreSQL databases:
1. Clinical Trial Signals
2. Patent Intelligence
3. Insider Activity + Hiring Signals

Features:
- Connection pooling for efficient resource management
- Retry logic with exponential backoff
- Health checks and connection status monitoring
- Graceful error handling
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import psycopg2
from psycopg2 import pool, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def retry_on_connection_error(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator that retries database operations on connection errors.

    Uses exponential backoff between retries.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, InterfaceError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Database operation failed after {max_retries} attempts: {e}")
            raise last_error
        return wrapper
    return decorator


class DatabaseManager:
    """
    Manages PostgreSQL connections for all three data systems.

    Uses connection pooling for efficient resource management.
    Includes retry logic and health monitoring.
    """

    # Database identifiers
    CLINICAL_TRIALS = 'clinical_trials'
    PATENT_INTELLIGENCE = 'patent_intelligence'
    INSIDER_HIRING = 'insider_hiring'

    # Default pool configuration
    DEFAULT_MIN_CONN = 1
    DEFAULT_MAX_CONN = 10

    def __init__(
        self,
        config: Dict[str, Any],
        min_conn: int = DEFAULT_MIN_CONN,
        max_conn: int = DEFAULT_MAX_CONN,
    ):
        """
        Initialize database connections from configuration.

        Args:
            config: Dictionary with database configurations for each system:
                    {
                        'clinical_trials': {...},
                        'patent_intelligence': {...},
                        'insider_hiring': {...}
                    }
            min_conn: Minimum connections per pool
            max_conn: Maximum connections per pool
        """
        self._pools: Dict[str, pool.ThreadedConnectionPool] = {}
        self._config = config
        self._min_conn = min_conn
        self._max_conn = max_conn
        self._initialized = False
        self._health_status: Dict[str, Dict[str, Any]] = {}

    def initialize(self) -> Dict[str, bool]:
        """
        Initialize connection pools for all databases.

        Returns:
            Dictionary mapping database names to initialization success status
        """
        if self._initialized:
            return self.get_connection_status()

        results = {}
        for db_name, db_config in self._config.items():
            try:
                self._pools[db_name] = pool.ThreadedConnectionPool(
                    minconn=self._min_conn,
                    maxconn=self._max_conn,
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    database=db_config.get('database'),
                    user=db_config.get('user'),
                    password=db_config.get('password', ''),
                    connect_timeout=10,  # 10 second connection timeout
                )
                self._health_status[db_name] = {
                    'connected': True,
                    'last_check': time.time(),
                    'error': None,
                }
                logger.info(f"Connection pool initialized for {db_name}")
                results[db_name] = True
            except psycopg2.Error as e:
                self._health_status[db_name] = {
                    'connected': False,
                    'last_check': time.time(),
                    'error': str(e),
                }
                logger.error(f"Failed to initialize pool for {db_name}: {e}")
                results[db_name] = False
                # Continue with other databases even if one fails

        self._initialized = True
        return results

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
            raise ValueError(f"Unknown database: {db_name}. Available: {list(self._pools.keys())}")

        conn = None
        try:
            conn = self._pools[db_name].getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            self._health_status[db_name] = {
                'connected': False,
                'last_check': time.time(),
                'error': str(e),
            }
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

    @retry_on_connection_error(max_retries=3, base_delay=1.0)
    def execute_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None,
        fetch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.

        Args:
            db_name: Name of the database
            query: SQL query string
            params: Query parameters
            fetch_size: Optional limit on number of results to fetch

        Returns:
            List of result dictionaries
        """
        try:
            with self.get_cursor(db_name) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    if fetch_size:
                        results = cursor.fetchmany(fetch_size)
                    else:
                        results = cursor.fetchall()
                    return [dict(row) for row in results]
                return []
        except Exception as e:
            logger.error(f"Query failed on {db_name}: {e}")
            return []

    @retry_on_connection_error(max_retries=3, base_delay=1.0)
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
        results = self.execute_query(db_name, query, params, fetch_size=1)
        return results[0] if results else None

    def execute_query_paginated(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Execute a paginated query.

        Args:
            db_name: Name of the database
            query: SQL query string (without LIMIT/OFFSET)
            params: Query parameters
            page: Page number (1-indexed)
            page_size: Number of results per page

        Returns:
            Tuple of (results, total_count)
        """
        # First get total count
        count_query = f"SELECT COUNT(*) as total FROM ({query}) as subquery"
        count_result = self.execute_query_one(db_name, count_query, params)
        total_count = count_result.get('total', 0) if count_result else 0

        # Then get paginated results
        offset = (page - 1) * page_size
        paginated_query = f"{query} LIMIT %s OFFSET %s"
        paginated_params = params + (page_size, offset) if params else (page_size, offset)

        results = self.execute_query(db_name, paginated_query, paginated_params)

        return results, total_count

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
                self._health_status[db_name] = {
                    'connected': True,
                    'last_check': time.time(),
                    'error': None,
                }
                return True
        except Exception as e:
            self._health_status[db_name] = {
                'connected': False,
                'last_check': time.time(),
                'error': str(e),
            }
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

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed health status for all databases.

        Returns:
            Dictionary with health information for each database
        """
        # Update health status for all databases
        for db_name in self._config.keys():
            self.is_connected(db_name)

        return self._health_status.copy()

    def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool stats for each database
        """
        stats = {}
        for db_name, db_pool in self._pools.items():
            if db_pool:
                # Note: ThreadedConnectionPool doesn't expose detailed stats
                # This is a simplified version
                stats[db_name] = {
                    'min_conn': self._min_conn,
                    'max_conn': self._max_conn,
                }
        return stats

    def refresh_connection(self, db_name: str) -> bool:
        """
        Refresh a database connection by closing and reopening the pool.

        Args:
            db_name: Name of the database

        Returns:
            True if refresh was successful
        """
        if db_name not in self._config:
            return False

        # Close existing pool
        if db_name in self._pools and self._pools[db_name]:
            try:
                self._pools[db_name].closeall()
            except Exception as e:
                logger.warning(f"Error closing pool for {db_name}: {e}")

        # Reinitialize pool
        db_config = self._config[db_name]
        try:
            self._pools[db_name] = pool.ThreadedConnectionPool(
                minconn=self._min_conn,
                maxconn=self._max_conn,
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database'),
                user=db_config.get('user'),
                password=db_config.get('password', ''),
                connect_timeout=10,
            )
            self._health_status[db_name] = {
                'connected': True,
                'last_check': time.time(),
                'error': None,
            }
            logger.info(f"Connection pool refreshed for {db_name}")
            return True
        except psycopg2.Error as e:
            self._health_status[db_name] = {
                'connected': False,
                'last_check': time.time(),
                'error': str(e),
            }
            logger.error(f"Failed to refresh pool for {db_name}: {e}")
            return False

    def close(self) -> None:
        """Close all connection pools."""
        for db_name, db_pool in self._pools.items():
            if db_pool:
                try:
                    db_pool.closeall()
                    logger.info(f"Connection pool closed for {db_name}")
                except Exception as e:
                    logger.warning(f"Error closing pool for {db_name}: {e}")

        self._pools.clear()
        self._health_status.clear()
        self._initialized = False


class MockDatabaseManager:
    """
    Mock database manager for testing and demo mode.

    Returns sample data without requiring actual database connections.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._initialized = False

    def initialize(self) -> Dict[str, bool]:
        """Initialize mock database (no-op)."""
        self._initialized = True
        return {
            'clinical_trials': True,
            'patent_intelligence': True,
            'insider_hiring': True,
        }

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

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns healthy status for all databases."""
        return {
            'clinical_trials': {'connected': True, 'last_check': time.time(), 'error': None},
            'patent_intelligence': {'connected': True, 'last_check': time.time(), 'error': None},
            'insider_hiring': {'connected': True, 'last_check': time.time(), 'error': None},
        }

    def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """Returns mock pool stats."""
        return {
            'clinical_trials': {'min_conn': 1, 'max_conn': 5},
            'patent_intelligence': {'min_conn': 1, 'max_conn': 5},
            'insider_hiring': {'min_conn': 1, 'max_conn': 5},
        }

    def execute_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None,
        fetch_size: Optional[int] = None,
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

    def execute_query_paginated(
        self,
        db_name: str,
        query: str,
        params: Optional[Tuple] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return empty results for mock queries."""
        return [], 0

    def refresh_connection(self, db_name: str) -> bool:
        """Always returns True for mock."""
        return True

    def close(self) -> None:
        """Close mock database (no-op)."""
        self._initialized = False
