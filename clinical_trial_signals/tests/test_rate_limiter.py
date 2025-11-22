"""
Tests for the rate limiting module.
"""
import pytest
import threading
import time
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimiterRegistry,
    get_rate_limiter,
    rate_limited,
    RateLimitExceeded,
    wait_for_rate_limit,
)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_initial_tokens_available(self):
        """Test that initial tokens are available."""
        limiter = TokenBucketRateLimiter(requests_per_second=10, burst_size=5)
        assert limiter.available_tokens == 5.0

    def test_acquire_reduces_tokens(self):
        """Test that acquire reduces available tokens."""
        limiter = TokenBucketRateLimiter(requests_per_second=10, burst_size=5)
        assert limiter.acquire(timeout=1)
        assert limiter.available_tokens < 5.0

    def test_try_acquire_non_blocking(self):
        """Test that try_acquire doesn't block."""
        limiter = TokenBucketRateLimiter(requests_per_second=10, burst_size=1)

        # First acquire should succeed
        assert limiter.try_acquire()

        # Second should fail (no tokens left)
        assert not limiter.try_acquire()

    def test_tokens_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(requests_per_second=100, burst_size=1)

        # Use the token
        assert limiter.acquire(timeout=1)
        assert limiter.available_tokens < 1.0

        # Wait for refill
        time.sleep(0.02)  # 20ms should give ~2 tokens at 100/s

        assert limiter.available_tokens > 0.5

    def test_timeout_exceeded_returns_false(self):
        """Test that timeout returns False when exceeded."""
        limiter = TokenBucketRateLimiter(requests_per_second=0.1, burst_size=1)

        # Use the token
        limiter.acquire(timeout=1)

        # Try to acquire with short timeout
        result = limiter.acquire(timeout=0.01)
        assert not result

    def test_burst_limit_respected(self):
        """Test that burst limit is respected."""
        limiter = TokenBucketRateLimiter(requests_per_second=100, burst_size=3)

        # Should be able to acquire burst_size tokens quickly
        for _ in range(3):
            assert limiter.try_acquire()

        # Fourth should fail
        assert not limiter.try_acquire()


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry."""

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        reg1 = RateLimiterRegistry()
        reg2 = RateLimiterRegistry()
        assert reg1 is reg2

    def test_get_or_create_limiter(self):
        """Test getting or creating a limiter."""
        registry = RateLimiterRegistry()
        registry.clear_all()

        limiter1 = registry.get_limiter("test_api", requests_per_second=5)
        limiter2 = registry.get_limiter("test_api", requests_per_second=10)

        # Should return the same limiter
        assert limiter1 is limiter2
        # Rate should be from first creation
        assert limiter1.rate == 5

    def test_remove_limiter(self):
        """Test removing a limiter."""
        registry = RateLimiterRegistry()
        registry.get_limiter("to_remove", requests_per_second=1)
        registry.remove_limiter("to_remove")

        # Getting again should create new one
        limiter = registry.get_limiter("to_remove", requests_per_second=10)
        assert limiter.rate == 10

    def test_clear_all(self):
        """Test clearing all limiters."""
        registry = RateLimiterRegistry()
        registry.get_limiter("test1")
        registry.get_limiter("test2")
        registry.clear_all()

        # Creating again should work
        limiter = registry.get_limiter("test1", requests_per_second=50)
        assert limiter.rate == 50


class TestGetRateLimiter:
    """Tests for get_rate_limiter function."""

    def test_get_known_api_limiter(self):
        """Test getting limiter for known API."""
        RateLimiterRegistry().clear_all()

        limiter = get_rate_limiter("clinicaltrials")
        assert limiter is not None
        assert limiter.rate == 2.0  # From DEFAULT_RATE_LIMITS

    def test_get_unknown_api_limiter(self):
        """Test getting limiter for unknown API uses defaults."""
        RateLimiterRegistry().clear_all()

        limiter = get_rate_limiter("unknown_api")
        assert limiter is not None
        assert limiter.rate == 1.0  # Default


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    def test_decorator_applies_rate_limit(self):
        """Test that decorator applies rate limiting."""
        RateLimiterRegistry().clear_all()

        call_count = 0

        @rate_limited("test_decorator_api", timeout=5)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        # First call should succeed immediately
        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_decorator_raises_on_timeout(self):
        """Test that decorator raises on timeout."""
        RateLimiterRegistry().clear_all()

        # Create limiter with very slow refill
        registry = RateLimiterRegistry()
        limiter = registry.get_limiter("slow_api", requests_per_second=0.001, burst_size=1)

        # Use up the token
        limiter.acquire(timeout=1)

        @rate_limited("slow_api", timeout=0.01)
        def slow_function():
            return "success"

        with pytest.raises(RateLimitExceeded):
            slow_function()


class TestWaitForRateLimit:
    """Tests for wait_for_rate_limit function."""

    def test_wait_succeeds(self):
        """Test that wait succeeds when tokens available."""
        RateLimiterRegistry().clear_all()

        result = wait_for_rate_limit("test_wait_api", timeout=1)
        assert result is True

    def test_wait_returns_false_on_timeout(self):
        """Test that wait returns False on timeout."""
        RateLimiterRegistry().clear_all()

        # Use up tokens
        limiter = get_rate_limiter("test_timeout_api")
        limiter._tokens = 0  # Force no tokens
        limiter._last_update = time.monotonic()

        result = wait_for_rate_limit("test_timeout_api", timeout=0.01)
        assert result is False


class TestThreadSafety:
    """Tests for thread safety of rate limiter."""

    def test_concurrent_access(self):
        """Test that concurrent access is thread-safe."""
        limiter = TokenBucketRateLimiter(requests_per_second=100, burst_size=10)
        acquired_count = 0
        lock = threading.Lock()

        def try_acquire():
            nonlocal acquired_count
            if limiter.try_acquire():
                with lock:
                    acquired_count += 1

        threads = [threading.Thread(target=try_acquire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have acquired at most burst_size tokens
        assert acquired_count <= 10
