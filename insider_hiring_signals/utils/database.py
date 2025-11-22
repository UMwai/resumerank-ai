"""
Database connection and utilities for Insider Activity + Hiring Signals System
"""

import json
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor, Json, execute_values

from .config import get_config
from .logger import setup_logger

logger = setup_logger(__name__)


class Database:
    """PostgreSQL database connection manager."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self._conn = None

    @property
    def connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self):
        """Create a new database connection."""
        db_config = self.config.database
        try:
            conn = psycopg2.connect(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('name', 'insider_signals'),
                user=db_config.get('user', 'postgres'),
                password=db_config.get('password', ''),
            )
            conn.autocommit = False
            logger.info(f"Connected to database: {db_config.get('name')}")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @contextmanager
    def cursor(self, dict_cursor: bool = True) -> Generator:
        """Context manager for database cursor."""
        cursor_factory = DictCursor if dict_cursor else None
        cursor = self.connection.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()

    def execute(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """Execute a query and return results."""
        with self.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            return []

    def execute_one(self, query: str, params: Optional[Tuple] = None) -> Optional[Dict]:
        """Execute a query and return a single result."""
        results = self.execute(query, params)
        return results[0] if results else None

    def insert(self, table: str, data: Dict[str, Any], returning: str = 'id') -> Any:
        """Insert a single row and return the specified column."""
        columns = list(data.keys())
        values = [self._convert_value(v) for v in data.values()]

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING {}").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(values)),
            sql.Identifier(returning.split('_')[0] + '_id' if '_' not in returning else returning)
        )

        with self.cursor() as cur:
            cur.execute(query, values)
            result = cur.fetchone()
            return result[0] if result else None

    def insert_many(self, table: str, data: List[Dict[str, Any]],
                    on_conflict: Optional[str] = None) -> int:
        """Insert multiple rows efficiently."""
        if not data:
            return 0

        columns = list(data[0].keys())
        values = [
            tuple(self._convert_value(row.get(col)) for col in columns)
            for row in data
        ]

        query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns))
        )

        if on_conflict:
            query = sql.SQL("{} ON CONFLICT {} DO NOTHING").format(
                query, sql.SQL(on_conflict)
            )

        with self.cursor() as cur:
            execute_values(cur, query, values)
            return cur.rowcount

    def upsert(self, table: str, data: Dict[str, Any],
               conflict_columns: List[str], update_columns: List[str]) -> Optional[int]:
        """Insert or update a row."""
        columns = list(data.keys())
        values = [self._convert_value(v) for v in data.values()]

        conflict_clause = sql.SQL(', ').join(map(sql.Identifier, conflict_columns))
        update_clause = sql.SQL(', ').join([
            sql.SQL('{} = EXCLUDED.{}').format(sql.Identifier(col), sql.Identifier(col))
            for col in update_columns
        ])

        # Get the primary key column name
        pk_column = f"{table.rstrip('s')}_id" if not table.endswith('ies') else f"{table[:-3]}y_id"

        query = sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}) "
            "ON CONFLICT ({}) DO UPDATE SET {} "
            "RETURNING {}"
        ).format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(values)),
            conflict_clause,
            update_clause,
            sql.Identifier(pk_column)
        )

        with self.cursor() as cur:
            cur.execute(query, values)
            result = cur.fetchone()
            return result[0] if result else None

    def _convert_value(self, value: Any) -> Any:
        """Convert Python values to PostgreSQL-compatible types."""
        if isinstance(value, dict):
            return Json(value)
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (datetime, date)):
            return value
        return value

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a company ticker."""
        result = self.execute_one(
            "SELECT cik FROM companies WHERE ticker = %s",
            (ticker,)
        )
        return result['cik'] if result else None

    def update_company(self, ticker: str, data: Dict[str, Any]):
        """Update or insert company information."""
        data['ticker'] = ticker
        self.upsert(
            'companies',
            data,
            conflict_columns=['ticker'],
            update_columns=[k for k in data.keys() if k != 'ticker']
        )

    def log_scraper_run(self, scraper_name: str) -> int:
        """Log the start of a scraper run."""
        return self.insert('scraper_runs', {
            'scraper_name': scraper_name,
            'start_time': datetime.now(),
            'status': 'running'
        })

    def update_scraper_run(self, run_id: int, status: str,
                           records_processed: int = 0,
                           records_inserted: int = 0,
                           records_updated: int = 0,
                           errors_count: int = 0,
                           error_details: Optional[Dict] = None):
        """Update a scraper run record."""
        with self.cursor() as cur:
            cur.execute("""
                UPDATE scraper_runs SET
                    end_time = %s,
                    status = %s,
                    records_processed = %s,
                    records_inserted = %s,
                    records_updated = %s,
                    errors_count = %s,
                    error_details = %s
                WHERE run_id = %s
            """, (
                datetime.now(), status, records_processed,
                records_inserted, records_updated, errors_count,
                Json(error_details) if error_details else None, run_id
            ))

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function
def get_database(config_path: Optional[str] = None) -> Database:
    """Get a database instance."""
    return Database(config_path)
