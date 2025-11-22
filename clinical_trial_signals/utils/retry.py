"""
Retry utilities with exponential backoff for API calls.

Provides robust error handling for network requests.
"""
import logging
import random
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Union

import requests

logger = logging.getLogger(__name__)


# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        self.message = message
        self.last_exception = last_exception
        super().__init__(message)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
        retryable_status_codes: set = None,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types to retry
            retryable_status_codes: Set of HTTP status codes to retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.retryable_status_codes = retryable_status_codes or RETRYABLE_STATUS_CODES


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate delay for a retry attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add random jitter of +/- 25%
        jitter_factor = 0.75 + random.random() * 0.5
        delay *= jitter_factor

    return delay


def retry_with_backoff(
    config: RetryConfig = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        config: RetryConfig instance (uses defaults if None)
        on_retry: Optional callback called before each retry with
                  (attempt, exception, delay) arguments

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(RetryConfig(max_retries=5))
        def fetch_data(url):
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Retry exhausted for {func.__name__} after "
                            f"{attempt + 1} attempts: {e}"
                        )
                        raise RetryExhausted(
                            f"Failed after {attempt + 1} attempts",
                            last_exception=e
                        )

                    delay = calculate_delay(
                        attempt,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter
                    )

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for "
                        f"{func.__name__}: {e}. Waiting {delay:.2f}s"
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    time.sleep(delay)

            raise RetryExhausted(
                f"Failed after {config.max_retries + 1} attempts",
                last_exception=last_exception
            )

        return wrapper
    return decorator


def retry_request(
    method: str,
    url: str,
    session: Optional[requests.Session] = None,
    config: RetryConfig = None,
    **kwargs
) -> requests.Response:
    """
    Make an HTTP request with retry logic.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        session: Optional requests.Session to use
        config: RetryConfig instance
        **kwargs: Additional arguments for requests

    Returns:
        requests.Response object

    Raises:
        RetryExhausted: If all retries fail
    """
    if config is None:
        config = RetryConfig()

    requester = session or requests
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            response = requester.request(method, url, **kwargs)

            # Check if status code is retryable
            if response.status_code in config.retryable_status_codes:
                logger.warning(
                    f"Retryable status {response.status_code} from {url}"
                )

                if attempt >= config.max_retries:
                    response.raise_for_status()

                delay = calculate_delay(
                    attempt,
                    config.base_delay,
                    config.max_delay,
                    config.exponential_base,
                    config.jitter
                )

                # Check for Retry-After header
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries}: "
                    f"status {response.status_code}. Waiting {delay:.2f}s"
                )
                time.sleep(delay)
                continue

            return response

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt >= config.max_retries:
                raise RetryExhausted(
                    f"Request to {url} failed after {attempt + 1} attempts",
                    last_exception=e
                )

            delay = calculate_delay(
                attempt,
                config.base_delay,
                config.max_delay,
                config.exponential_base,
                config.jitter
            )

            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {url}: "
                f"{e}. Waiting {delay:.2f}s"
            )
            time.sleep(delay)

    raise RetryExhausted(
        f"Request to {url} failed after {config.max_retries + 1} attempts",
        last_exception=last_exception
    )


class RetryableSession:
    """
    Wrapper around requests.Session with built-in retry logic.

    Example:
        session = RetryableSession(max_retries=3)
        response = session.get("https://api.example.com/data")
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: int = 30,
        headers: Optional[dict] = None,
    ):
        """
        Initialize retryable session.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for backoff
            timeout: Default request timeout
            headers: Default headers for all requests
        """
        self.session = requests.Session()
        self.config = RetryConfig(max_retries=max_retries, base_delay=base_delay)
        self.timeout = timeout

        if headers:
            self.session.headers.update(headers)

    def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """Make a request with retry logic."""
        kwargs.setdefault('timeout', self.timeout)
        return retry_request(method, url, self.session, self.config, **kwargs)

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request with retry logic."""
        return self.request('GET', url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request with retry logic."""
        return self.request('POST', url, **kwargs)

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
