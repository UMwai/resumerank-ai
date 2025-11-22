"""
Health Check System

Provides health monitoring for all pipeline components:
- Database connectivity
- Data source availability
- Data freshness
- Pipeline execution status
- Resource utilization
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# Project root for imports
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/Users/waiyang/Desktop/repo/dreamers-v2")


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0


@dataclass
class PipelineHealth:
    """Overall pipeline health status."""
    pipeline_name: str
    healthy: bool
    checks: List[HealthCheckResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthChecker:
    """Base class for health checkers."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def check(self) -> HealthCheckResult:
        """Run the health check. Override in subclasses."""
        raise NotImplementedError


class DatabaseHealthChecker(HealthChecker):
    """Check database connectivity and health."""

    def __init__(self, connection_string: str = None):
        super().__init__()
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'investment_signals')}"
        )

    def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        start_time = time.time()
        try:
            import psycopg2

            conn = psycopg2.connect(self.connection_string, connect_timeout=10)
            cursor = conn.cursor()

            # Test query
            cursor.execute("SELECT 1")
            cursor.fetchone()

            # Get connection info
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="database",
                healthy=True,
                message="Database connection successful",
                details={"version": version[:50] if version else "unknown"},
                duration_ms=duration_ms,
            )

        except ImportError:
            return HealthCheckResult(
                name="database",
                healthy=False,
                message="psycopg2 not installed",
                details={"error": "Missing dependency"},
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                healthy=False,
                message=f"Database connection failed: {str(e)[:100]}",
                details={"error": str(e)},
                duration_ms=(time.time() - start_time) * 1000,
            )


class ExternalAPIHealthChecker(HealthChecker):
    """Check external API availability."""

    # API endpoints to check
    ENDPOINTS = {
        "clinicaltrials_gov": "https://clinicaltrials.gov/api/v2/studies",
        "sec_edgar": "https://www.sec.gov/cgi-bin/browse-edgar",
        "fda_orange_book": "https://www.accessdata.fda.gov/scripts/cder/ob/",
        "uspto": "https://developer.uspto.gov/",
    }

    def __init__(self, api_name: str, endpoint: str = None):
        super().__init__()
        self.api_name = api_name
        self.endpoint = endpoint or self.ENDPOINTS.get(api_name)

    def check(self) -> HealthCheckResult:
        """Check if external API is reachable."""
        start_time = time.time()

        if not self.endpoint:
            return HealthCheckResult(
                name=f"api_{self.api_name}",
                healthy=False,
                message=f"Unknown API: {self.api_name}",
                duration_ms=0,
            )

        try:
            response = requests.get(
                self.endpoint,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "Investment-Signals-HealthCheck/1.0"},
            )

            duration_ms = (time.time() - start_time) * 1000
            is_healthy = response.status_code < 500

            return HealthCheckResult(
                name=f"api_{self.api_name}",
                healthy=is_healthy,
                message=f"API responded with status {response.status_code}",
                details={
                    "status_code": response.status_code,
                    "response_time_ms": duration_ms,
                },
                duration_ms=duration_ms,
            )

        except requests.Timeout:
            return HealthCheckResult(
                name=f"api_{self.api_name}",
                healthy=False,
                message=f"API timeout after {self.timeout_seconds}s",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"api_{self.api_name}",
                healthy=False,
                message=f"API check failed: {str(e)[:100]}",
                duration_ms=(time.time() - start_time) * 1000,
            )


class DataFreshnessChecker(HealthChecker):
    """Check if data is fresh (updated within threshold)."""

    def __init__(
        self,
        data_type: str,
        threshold_hours: int = 24,
        connection_string: str = None,
    ):
        super().__init__()
        self.data_type = data_type
        self.threshold_hours = threshold_hours
        self.connection_string = connection_string or os.getenv("DATABASE_URL")

    def check(self) -> HealthCheckResult:
        """Check data freshness."""
        start_time = time.time()

        # Table and timestamp column mapping
        table_mapping = {
            "clinical_trials": ("trials", "last_updated"),
            "form4_filings": ("form4_filings", "filing_date"),
            "form13f_holdings": ("form13f_holdings", "report_date"),
            "job_postings": ("job_postings", "posted_date"),
            "patents": ("patents", "updated_at"),
        }

        if self.data_type not in table_mapping:
            return HealthCheckResult(
                name=f"freshness_{self.data_type}",
                healthy=False,
                message=f"Unknown data type: {self.data_type}",
                duration_ms=0,
            )

        table, timestamp_col = table_mapping[self.data_type]

        try:
            import psycopg2

            conn = psycopg2.connect(self.connection_string, connect_timeout=10)
            cursor = conn.cursor()

            # Get most recent record
            cursor.execute(f"SELECT MAX({timestamp_col}) FROM {table}")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            duration_ms = (time.time() - start_time) * 1000

            if not result or not result[0]:
                return HealthCheckResult(
                    name=f"freshness_{self.data_type}",
                    healthy=False,
                    message=f"No data found in {table}",
                    details={"table": table},
                    duration_ms=duration_ms,
                )

            last_update = result[0]
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)

            cutoff = datetime.utcnow() - timedelta(hours=self.threshold_hours)
            is_fresh = last_update > cutoff
            hours_old = (datetime.utcnow() - last_update).total_seconds() / 3600

            return HealthCheckResult(
                name=f"freshness_{self.data_type}",
                healthy=is_fresh,
                message=f"Data is {hours_old:.1f} hours old (threshold: {self.threshold_hours}h)",
                details={
                    "last_update": last_update.isoformat(),
                    "hours_old": round(hours_old, 1),
                    "threshold_hours": self.threshold_hours,
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            return HealthCheckResult(
                name=f"freshness_{self.data_type}",
                healthy=False,
                message=f"Freshness check failed: {str(e)[:100]}",
                duration_ms=(time.time() - start_time) * 1000,
            )


class AirflowHealthChecker(HealthChecker):
    """Check Airflow scheduler and webserver health."""

    def __init__(self, webserver_url: str = None):
        super().__init__()
        self.webserver_url = webserver_url or os.getenv(
            "AIRFLOW_WEBSERVER_URL", "http://localhost:8080"
        )

    def check(self) -> HealthCheckResult:
        """Check Airflow health endpoint."""
        start_time = time.time()

        try:
            response = requests.get(
                f"{self.webserver_url}/health",
                timeout=10,
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                health_data = response.json()
                scheduler_healthy = health_data.get("scheduler", {}).get("status") == "healthy"
                metadatabase_healthy = health_data.get("metadatabase", {}).get("status") == "healthy"

                is_healthy = scheduler_healthy and metadatabase_healthy

                return HealthCheckResult(
                    name="airflow",
                    healthy=is_healthy,
                    message="Airflow is healthy" if is_healthy else "Airflow has issues",
                    details=health_data,
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheckResult(
                    name="airflow",
                    healthy=False,
                    message=f"Airflow returned status {response.status_code}",
                    details={"status_code": response.status_code},
                    duration_ms=duration_ms,
                )

        except Exception as e:
            return HealthCheckResult(
                name="airflow",
                healthy=False,
                message=f"Airflow health check failed: {str(e)[:100]}",
                duration_ms=(time.time() - start_time) * 1000,
            )


# Pipeline-specific health check functions

def check_clinical_trial_health() -> Dict[str, Any]:
    """
    Run health checks for clinical trial pipeline.

    Returns:
        Dictionary with health status
    """
    checks = []

    # Database check
    db_checker = DatabaseHealthChecker()
    checks.append(db_checker.check())

    # ClinicalTrials.gov API check
    ct_checker = ExternalAPIHealthChecker("clinicaltrials_gov")
    checks.append(ct_checker.check())

    # SEC EDGAR API check
    sec_checker = ExternalAPIHealthChecker("sec_edgar")
    checks.append(sec_checker.check())

    # Data freshness check
    freshness_checker = DataFreshnessChecker("clinical_trials", threshold_hours=48)
    checks.append(freshness_checker.check())

    # Compile results
    issues = [c.message for c in checks if not c.healthy]

    return {
        "pipeline": "clinical_trial_signals",
        "healthy": len(issues) == 0,
        "checks": [
            {
                "name": c.name,
                "healthy": c.healthy,
                "message": c.message,
                "duration_ms": c.duration_ms,
            }
            for c in checks
        ],
        "issues": issues,
        "timestamp": datetime.utcnow().isoformat(),
    }


def check_patent_ip_health() -> Dict[str, Any]:
    """
    Run health checks for patent/IP pipeline.

    Returns:
        Dictionary with health status
    """
    checks = []

    # Database check
    db_checker = DatabaseHealthChecker()
    checks.append(db_checker.check())

    # FDA Orange Book check
    fda_checker = ExternalAPIHealthChecker("fda_orange_book")
    checks.append(fda_checker.check())

    # USPTO check
    uspto_checker = ExternalAPIHealthChecker("uspto")
    checks.append(uspto_checker.check())

    # Data freshness check
    freshness_checker = DataFreshnessChecker("patents", threshold_hours=168)  # 7 days
    checks.append(freshness_checker.check())

    issues = [c.message for c in checks if not c.healthy]

    return {
        "pipeline": "patent_ip_intelligence",
        "healthy": len(issues) == 0,
        "checks": [
            {
                "name": c.name,
                "healthy": c.healthy,
                "message": c.message,
                "duration_ms": c.duration_ms,
            }
            for c in checks
        ],
        "issues": issues,
        "timestamp": datetime.utcnow().isoformat(),
    }


def check_insider_hiring_health() -> Dict[str, Any]:
    """
    Run health checks for insider/hiring pipeline.

    Returns:
        Dictionary with health status
    """
    checks = []

    # Database check
    db_checker = DatabaseHealthChecker()
    checks.append(db_checker.check())

    # SEC EDGAR API check (Form 4, 13F)
    sec_checker = ExternalAPIHealthChecker("sec_edgar")
    checks.append(sec_checker.check())

    # Form 4 data freshness (should be updated every 30 min during market hours)
    form4_checker = DataFreshnessChecker("form4_filings", threshold_hours=24)
    checks.append(form4_checker.check())

    # Job postings freshness
    jobs_checker = DataFreshnessChecker("job_postings", threshold_hours=48)
    checks.append(jobs_checker.check())

    issues = [c.message for c in checks if not c.healthy]

    return {
        "pipeline": "insider_hiring_signals",
        "healthy": len(issues) == 0,
        "checks": [
            {
                "name": c.name,
                "healthy": c.healthy,
                "message": c.message,
                "duration_ms": c.duration_ms,
            }
            for c in checks
        ],
        "issues": issues,
        "timestamp": datetime.utcnow().isoformat(),
    }


def check_orchestration_health() -> Dict[str, Any]:
    """
    Run health checks for orchestration system (Airflow).

    Returns:
        Dictionary with health status
    """
    checks = []

    # Airflow health
    airflow_checker = AirflowHealthChecker()
    checks.append(airflow_checker.check())

    # Database check
    db_checker = DatabaseHealthChecker()
    checks.append(db_checker.check())

    issues = [c.message for c in checks if not c.healthy]

    return {
        "pipeline": "orchestration",
        "healthy": len(issues) == 0,
        "checks": [
            {
                "name": c.name,
                "healthy": c.healthy,
                "message": c.message,
                "duration_ms": c.duration_ms,
            }
            for c in checks
        ],
        "issues": issues,
        "timestamp": datetime.utcnow().isoformat(),
    }


def run_all_health_checks() -> Dict[str, Any]:
    """
    Run health checks for all pipelines.

    Returns:
        Dictionary with all health check results
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipelines": {},
        "overall_healthy": True,
        "total_issues": [],
    }

    # Run checks for each pipeline
    pipelines = [
        ("clinical_trial_signals", check_clinical_trial_health),
        ("patent_ip_intelligence", check_patent_ip_health),
        ("insider_hiring_signals", check_insider_hiring_health),
        ("orchestration", check_orchestration_health),
    ]

    for name, check_func in pipelines:
        try:
            result = check_func()
            results["pipelines"][name] = result

            if not result["healthy"]:
                results["overall_healthy"] = False
                results["total_issues"].extend(
                    [f"[{name}] {issue}" for issue in result.get("issues", [])]
                )
        except Exception as e:
            results["pipelines"][name] = {
                "healthy": False,
                "error": str(e),
            }
            results["overall_healthy"] = False
            results["total_issues"].append(f"[{name}] Health check failed: {str(e)}")

    return results


if __name__ == "__main__":
    import json

    # Run all health checks
    results = run_all_health_checks()
    print(json.dumps(results, indent=2))

    # Exit with appropriate code
    sys.exit(0 if results["overall_healthy"] else 1)
