"""Tests for the database module."""

import pytest
from unittest.mock import MagicMock, patch
from utils.database import (
    DatabaseManager,
    MockDatabaseManager,
    retry_on_connection_error,
)


class TestMockDatabaseManager:
    """Tests for MockDatabaseManager class."""

    def test_initialize(self):
        """Test mock database initialization."""
        mock_db = MockDatabaseManager()
        result = mock_db.initialize()
        assert result == {
            'clinical_trials': True,
            'patent_intelligence': True,
            'insider_hiring': True,
        }

    def test_is_connected(self):
        """Test mock database always returns connected."""
        mock_db = MockDatabaseManager()
        assert mock_db.is_connected('clinical_trials') is True
        assert mock_db.is_connected('any_db') is True

    def test_get_connection_status(self):
        """Test mock database connection status."""
        mock_db = MockDatabaseManager()
        status = mock_db.get_connection_status()
        assert status['clinical_trials'] is True
        assert status['patent_intelligence'] is True
        assert status['insider_hiring'] is True

    def test_execute_query(self):
        """Test mock database query returns empty list."""
        mock_db = MockDatabaseManager()
        result = mock_db.execute_query('clinical_trials', 'SELECT * FROM trials')
        assert result == []

    def test_execute_query_one(self):
        """Test mock database single query returns None."""
        mock_db = MockDatabaseManager()
        result = mock_db.execute_query_one('clinical_trials', 'SELECT * FROM trials LIMIT 1')
        assert result is None

    def test_execute_query_paginated(self):
        """Test mock database paginated query."""
        mock_db = MockDatabaseManager()
        results, total = mock_db.execute_query_paginated('clinical_trials', 'SELECT * FROM trials')
        assert results == []
        assert total == 0

    def test_refresh_connection(self):
        """Test mock database refresh always succeeds."""
        mock_db = MockDatabaseManager()
        assert mock_db.refresh_connection('clinical_trials') is True

    def test_get_health_status(self):
        """Test mock database health status."""
        mock_db = MockDatabaseManager()
        status = mock_db.get_health_status()
        assert 'clinical_trials' in status
        assert status['clinical_trials']['connected'] is True
        assert status['clinical_trials']['error'] is None

    def test_close(self):
        """Test mock database close."""
        mock_db = MockDatabaseManager()
        mock_db.initialize()
        mock_db.close()
        assert mock_db._initialized is False


class TestRetryDecorator:
    """Tests for retry_on_connection_error decorator."""

    def test_successful_call_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry_on_connection_error(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_operational_error(self):
        """Test retry on OperationalError."""
        from psycopg2 import OperationalError

        call_count = 0

        @retry_on_connection_error(max_retries=3, base_delay=0.01)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("Connection failed")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test exception raised after max retries."""
        from psycopg2 import OperationalError

        @retry_on_connection_error(max_retries=2, base_delay=0.01)
        def always_failing_func():
            raise OperationalError("Connection failed")

        with pytest.raises(OperationalError):
            always_failing_func()


class TestDatabaseManagerConfig:
    """Tests for DatabaseManager configuration."""

    def test_database_constants(self):
        """Test database name constants."""
        assert DatabaseManager.CLINICAL_TRIALS == 'clinical_trials'
        assert DatabaseManager.PATENT_INTELLIGENCE == 'patent_intelligence'
        assert DatabaseManager.INSIDER_HIRING == 'insider_hiring'

    def test_default_pool_config(self):
        """Test default pool configuration."""
        assert DatabaseManager.DEFAULT_MIN_CONN == 1
        assert DatabaseManager.DEFAULT_MAX_CONN == 10

    def test_initialization_without_initialize(self):
        """Test database manager initialization state."""
        config = {
            'clinical_trials': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'user': 'test_user',
                'password': 'test_pass',
            }
        }
        db_manager = DatabaseManager(config)
        assert db_manager._initialized is False
        assert len(db_manager._pools) == 0
