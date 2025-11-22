"""
Rate limiting utilities for API requests.

Provides thread-safe rate limiting with configurable limits per endpoint.
"""
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 1.0
    burst_size: int = 1
    retry_after_seconds: float = 60.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Thread-safe implementation that allows burst traffic while
    maintaining average rate limits.
    """

    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: int = 1,
        name: str = "default"
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Average rate of requests allowed
            burst_size: Maximum burst of requests allowed
            name: Name for logging purposes
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.name = name

        self._tokens = float(burst_size)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.rate
        )
        self._last_update = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token, blocking if necessary.

        Args:
            timeout: Maximum time to wait for a token (None = wait forever)

        Returns:
            True if token acquired, False if timeout exceeded
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    logger.debug(f"[{self.name}] Token acquired, {self._tokens:.2f} remaining")
                    return True

                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self.rate

            # Check timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(f"[{self.name}] Rate limit timeout exceeded")
                    return False
                wait_time = min(wait_time, remaining)

            logger.debug(f"[{self.name}] Rate limited, waiting {wait_time:.2f}s")
            time.sleep(wait_time)

    def try_acquire(self) -> bool:
        """
        Try to acquire a token without blocking.

        Returns:
            True if token acquired, False otherwise
        """
        with self._lock:
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill_tokens()
            return self._tokens


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.

    Provides named rate limiters for different API endpoints.
    """

    _instance: Optional["RateLimiterRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._limiters: Dict[str, TokenBucketRateLimiter] = {}
                cls._instance._limiter_lock = threading.Lock()
        return cls._instance

    def get_limiter(
        self,
        name: str,
        requests_per_second: float = 1.0,
        burst_size: int = 1
    ) -> TokenBucketRateLimiter:
        """
        Get or create a rate limiter by name.

        Args:
            name: Unique name for the rate limiter
            requests_per_second: Rate limit (only used if creating new)
            burst_size: Burst size (only used if creating new)

        Returns:
            TokenBucketRateLimiter instance
        """
        with self._limiter_lock:
            if name not in self._limiters:
                self._limiters[name] = TokenBucketRateLimiter(
                    requests_per_second=requests_per_second,
                    burst_size=burst_size,
                    name=name
                )
                logger.info(f"Created rate limiter '{name}': {requests_per_second}/s, burst={burst_size}")
            return self._limiters[name]

    def remove_limiter(self, name: str) -> None:
        """Remove a rate limiter by name."""
        with self._limiter_lock:
            if name in self._limiters:
                del self._limiters[name]

    def clear_all(self) -> None:
        """Remove all rate limiters."""
        with self._limiter_lock:
            self._limiters.clear()


# Default rate limiter configurations for known APIs
DEFAULT_RATE_LIMITS = {
    "clinicaltrials": RateLimitConfig(requests_per_second=2.0, burst_size=5),
    "sec_edgar": RateLimitConfig(requests_per_second=10.0, burst_size=10),
    "pubmed": RateLimitConfig(requests_per_second=3.0, burst_size=10),
    "uspto": RateLimitConfig(requests_per_second=1.0, burst_size=3),
    "medrxiv": RateLimitConfig(requests_per_second=1.0, burst_size=5),
}


def get_rate_limiter(api_name: str) -> TokenBucketRateLimiter:
    """
    Get a rate limiter for a known API.

    Args:
        api_name: Name of the API (clinicaltrials, sec_edgar, pubmed, uspto, medrxiv)

    Returns:
        Configured TokenBucketRateLimiter
    """
    registry = RateLimiterRegistry()
    config = DEFAULT_RATE_LIMITS.get(api_name, RateLimitConfig())

    return registry.get_limiter(
        name=api_name,
        requests_per_second=config.requests_per_second,
        burst_size=config.burst_size
    )


def rate_limited(
    api_name: str,
    timeout: Optional[float] = 30.0
) -> Callable:
    """
    Decorator to apply rate limiting to a function.

    Args:
        api_name: Name of the API for rate limiting
        timeout: Maximum time to wait for rate limit

    Returns:
        Decorated function

    Example:
        @rate_limited("pubmed")
        def fetch_pubmed_article(pmid):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(api_name)
            if not limiter.acquire(timeout=timeout):
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {api_name} (timeout={timeout}s)"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit timeout is exceeded."""
    pass


# Convenience function for simple rate limiting
def wait_for_rate_limit(api_name: str, timeout: Optional[float] = None) -> bool:
    """
    Wait for rate limit to allow a request.

    Args:
        api_name: Name of the API
        timeout: Maximum time to wait

    Returns:
        True if allowed to proceed, False if timeout
    """
    limiter = get_rate_limiter(api_name)
    return limiter.acquire(timeout=timeout)
