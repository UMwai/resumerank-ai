"""
Tests for the retry module.
"""
import pytest
import time
from unittest.mock import MagicMock, patch

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.retry import (
    RetryConfig,
    retry_with_backoff,
    retry_request,
    calculate_delay,
    RetryExhausted,
    RetryableSession,
    RETRYABLE_EXCEPTIONS,
    RETRYABLE_STATUS_CODES,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_increase(self):
        """Test that delay increases exponentially."""
        delay0 = calculate_delay(0, 1.0, 60.0, 2.0, False)
        delay1 = calculate_delay(1, 1.0, 60.0, 2.0, False)
        delay2 = calculate_delay(2, 1.0, 60.0, 2.0, False)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = calculate_delay(10, 1.0, 5.0, 2.0, False)
        assert delay == 5.0

    def test_jitter_applied(self):
        """Test that jitter varies the delay."""
        delays = set()
        for _ in range(10):
            delay = calculate_delay(0, 1.0, 60.0, 2.0, True)
            delays.add(delay)

        # With jitter, we should get different values
        assert len(delays) > 1

    def test_jitter_range(self):
        """Test that jitter stays within expected range."""
        for _ in range(100):
            delay = calculate_delay(0, 1.0, 60.0, 2.0, True)
            # Jitter is +/- 25%, so 0.75 to 1.25 for base delay of 1.0
            assert 0.75 <= delay <= 1.25


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3))
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Test retry on retryable exception."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3, base_delay=0.01))
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.ConnectionError("Connection failed")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_exhaust_retries(self):
        """Test RetryExhausted is raised after max retries."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.Timeout("Timeout")

        with pytest.raises(RetryExhausted):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback_calls = []

        def on_retry(attempt, exception, delay):
            callback_calls.append((attempt, type(exception).__name__))

        @retry_with_backoff(
            RetryConfig(max_retries=2, base_delay=0.01),
            on_retry=on_retry
        )
        def failing_func():
            raise requests.exceptions.ConnectionError("Failed")

        with pytest.raises(RetryExhausted):
            failing_func()

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 0
        assert callback_calls[1][0] == 1

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3))
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries


class TestRetryRequest:
    """Tests for retry_request function."""

    def test_successful_request(self, mock_requests_session):
        """Test successful request."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_session.request.return_value = mock_response

        response = retry_request("GET", "http://test.com", session=mock_session)
        assert response.status_code == 200

    def test_retry_on_500_error(self):
        """Test retry on 500 status code."""
        mock_session = MagicMock()
        responses = [
            MagicMock(status_code=500, headers={}),
            MagicMock(status_code=500, headers={}),
            MagicMock(status_code=200, headers={}),
        ]
        mock_session.request.side_effect = responses

        config = RetryConfig(max_retries=3, base_delay=0.01)
        response = retry_request("GET", "http://test.com", session=mock_session, config=config)
        assert response.status_code == 200
        assert mock_session.request.call_count == 3

    def test_retry_after_header_respected(self):
        """Test Retry-After header is respected."""
        mock_session = MagicMock()
        responses = [
            MagicMock(status_code=429, headers={"Retry-After": "0.01"}),
            MagicMock(status_code=200, headers={}),
        ]
        mock_session.request.side_effect = responses

        config = RetryConfig(max_retries=3, base_delay=0.01)
        response = retry_request("GET", "http://test.com", session=mock_session, config=config)
        assert response.status_code == 200


class TestRetryableSession:
    """Tests for RetryableSession class."""

    def test_get_request(self):
        """Test GET request through RetryableSession."""
        with patch("utils.retry.retry_request") as mock_retry:
            mock_retry.return_value = MagicMock(status_code=200)

            session = RetryableSession(max_retries=3)
            response = session.get("http://test.com")

            mock_retry.assert_called_once()
            assert response.status_code == 200

    def test_post_request(self):
        """Test POST request through RetryableSession."""
        with patch("utils.retry.retry_request") as mock_retry:
            mock_retry.return_value = MagicMock(status_code=201)

            session = RetryableSession(max_retries=3)
            response = session.post("http://test.com", json={"data": "test"})

            mock_retry.assert_called_once()

    def test_context_manager(self):
        """Test RetryableSession as context manager."""
        with RetryableSession() as session:
            assert session is not None

    def test_custom_headers(self):
        """Test custom headers are set."""
        session = RetryableSession(headers={"Authorization": "Bearer token"})
        assert session.session.headers.get("Authorization") == "Bearer token"


class TestRetryableExceptions:
    """Tests for retryable exception detection."""

    def test_timeout_is_retryable(self):
        """Test that Timeout is in retryable exceptions."""
        assert requests.exceptions.Timeout in RETRYABLE_EXCEPTIONS

    def test_connection_error_is_retryable(self):
        """Test that ConnectionError is in retryable exceptions."""
        assert requests.exceptions.ConnectionError in RETRYABLE_EXCEPTIONS


class TestRetryableStatusCodes:
    """Tests for retryable status codes."""

    def test_500_is_retryable(self):
        """Test that 500 is retryable."""
        assert 500 in RETRYABLE_STATUS_CODES

    def test_429_is_retryable(self):
        """Test that 429 (rate limit) is retryable."""
        assert 429 in RETRYABLE_STATUS_CODES

    def test_503_is_retryable(self):
        """Test that 503 is retryable."""
        assert 503 in RETRYABLE_STATUS_CODES

    def test_200_is_not_retryable(self):
        """Test that 200 is not retryable."""
        assert 200 not in RETRYABLE_STATUS_CODES
