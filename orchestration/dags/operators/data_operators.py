"""
Data Fetch Operators for Investment Signals

Custom Airflow operators for fetching data from external sources with:
- Exponential backoff retry logic
- Rate limiting and throttling
- Request caching
- Error classification and handling
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import requests
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


class DataFetchOperator(BaseOperator):
    """
    Base operator for fetching data from external APIs.

    Features:
    - Exponential backoff retry with configurable parameters
    - Rate limiting per API
    - Request/response logging
    - Error classification (transient vs permanent)
    - Metrics collection

    Args:
        task_id: Unique task identifier
        api_name: Name of the API for logging/metrics
        base_url: Base URL for the API
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        params: Query parameters
        data: Request body data
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        retry_exponential_base: Base for exponential backoff
        max_retry_delay: Maximum retry delay in seconds
        rate_limit_requests: Max requests per rate_limit_period
        rate_limit_period: Rate limit time window in seconds
    """

    template_fields = ("endpoint", "params", "data", "headers")
    ui_color = "#e8f4ea"
    ui_fgcolor = "#1a5928"

    # Error classification
    TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
    PERMANENT_STATUS_CODES = {400, 401, 403, 404, 405, 422}

    @apply_defaults
    def __init__(
        self,
        api_name: str,
        base_url: str,
        endpoint: str = "",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_exponential_base: float = 2.0,
        max_retry_delay: float = 300.0,
        rate_limit_requests: int = 10,
        rate_limit_period: int = 60,
        parse_response: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_name = api_name
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params or {}
        self.data = data
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential_base = retry_exponential_base
        self.max_retry_delay = max_retry_delay
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_period = rate_limit_period
        self.parse_response = parse_response

        # Rate limiting state
        self._request_timestamps: List[float] = []

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exceeded."""
        now = time.time()
        cutoff = now - self.rate_limit_period

        # Remove old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]

        # Check if we need to wait
        if len(self._request_timestamps) >= self.rate_limit_requests:
            oldest = min(self._request_timestamps)
            wait_time = oldest + self.rate_limit_period - now
            if wait_time > 0:
                logger.info(
                    f"Rate limit reached for {self.api_name}, waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)

        self._request_timestamps.append(time.time())

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.retry_delay * (self.retry_exponential_base ** attempt)
        return min(delay, self.max_retry_delay)

    def _is_transient_error(self, status_code: int) -> bool:
        """Check if error is transient and should be retried."""
        return status_code in self.TRANSIENT_STATUS_CODES

    def _make_request(self) -> requests.Response:
        """Make HTTP request with rate limiting."""
        self._wait_for_rate_limit()

        url = f"{self.base_url}/{self.endpoint.lstrip('/')}" if self.endpoint else self.base_url

        logger.info(f"Making {self.method} request to {url}")

        response = requests.request(
            method=self.method,
            url=url,
            headers=self.headers,
            params=self.params,
            json=self.data if self.method in ("POST", "PUT", "PATCH") else None,
            timeout=self.timeout,
        )

        return response

    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the data fetch with retry logic."""
        start_time = time.time()
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._make_request()

                # Log response details
                logger.info(
                    f"{self.api_name} response: status={response.status_code}, "
                    f"time={response.elapsed.total_seconds():.2f}s"
                )

                # Handle successful response
                if response.status_code < 400:
                    result = response.json() if self.parse_response else response.text

                    # Record success metrics
                    duration = time.time() - start_time
                    self._record_metrics(context, "success", duration, attempt)

                    return result

                # Handle error response
                if self._is_transient_error(response.status_code):
                    if attempt < self.max_retries:
                        delay = self._calculate_backoff(attempt)
                        logger.warning(
                            f"Transient error {response.status_code} from {self.api_name}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        response.raise_for_status()
                else:
                    # Permanent error - don't retry
                    logger.error(
                        f"Permanent error {response.status_code} from {self.api_name}: "
                        f"{response.text[:500]}"
                    )
                    response.raise_for_status()

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Timeout from {self.api_name}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Connection error from {self.api_name}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error from {self.api_name}: {e}")
                break

        # Record failure metrics
        duration = time.time() - start_time
        self._record_metrics(context, "failure", duration, self.max_retries)

        raise Exception(
            f"Failed to fetch data from {self.api_name} after {self.max_retries} retries: "
            f"{last_exception}"
        )

    def _record_metrics(
        self,
        context: Dict[str, Any],
        status: str,
        duration: float,
        attempts: int,
    ) -> None:
        """Record execution metrics to XCom."""
        metrics = {
            "api_name": self.api_name,
            "status": status,
            "duration_seconds": duration,
            "attempts": attempts + 1,
            "timestamp": datetime.utcnow().isoformat(),
        }
        context["ti"].xcom_push(key=f"{self.task_id}_metrics", value=metrics)


class SECEdgarOperator(DataFetchOperator):
    """
    Specialized operator for SEC EDGAR API.

    Handles SEC-specific requirements:
    - User-Agent header requirement
    - Rate limiting (10 requests/second)
    - XML/JSON response handling
    """

    SEC_BASE_URL = "https://data.sec.gov"
    SEC_FILINGS_URL = "https://efts.sec.gov/LATEST/search-index"

    @apply_defaults
    def __init__(
        self,
        endpoint: str,
        user_agent: str = "InvestmentSignals/1.0 (contact@example.com)",
        **kwargs,
    ):
        # SEC requires specific rate limiting
        kwargs.setdefault("rate_limit_requests", 10)
        kwargs.setdefault("rate_limit_period", 1)
        kwargs.setdefault("timeout", 60)

        super().__init__(
            api_name="sec_edgar",
            base_url=self.SEC_BASE_URL,
            endpoint=endpoint,
            headers={"User-Agent": user_agent},
            **kwargs,
        )


class ClinicalTrialsOperator(DataFetchOperator):
    """
    Specialized operator for ClinicalTrials.gov API.

    Features:
    - Support for v2 API
    - Study search and filtering
    - Pagination handling
    """

    CT_BASE_URL = "https://clinicaltrials.gov/api/v2"

    @apply_defaults
    def __init__(
        self,
        endpoint: str = "studies",
        query_params: Optional[Dict[str, Any]] = None,
        page_size: int = 100,
        max_pages: int = 10,
        **kwargs,
    ):
        params = query_params or {}
        params.setdefault("pageSize", page_size)

        super().__init__(
            api_name="clinicaltrials_gov",
            base_url=self.CT_BASE_URL,
            endpoint=endpoint,
            params=params,
            rate_limit_requests=3,  # Conservative rate limit
            rate_limit_period=1,
            **kwargs,
        )
        self.max_pages = max_pages

    def execute(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute with pagination support."""
        all_studies = []
        page_token = None

        for page in range(self.max_pages):
            if page_token:
                self.params["pageToken"] = page_token

            result = super().execute(context)

            studies = result.get("studies", [])
            all_studies.extend(studies)

            # Check for next page
            page_token = result.get("nextPageToken")
            if not page_token:
                break

            logger.info(f"Fetched page {page + 1}, total studies: {len(all_studies)}")

        return all_studies
