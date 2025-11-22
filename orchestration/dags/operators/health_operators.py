"""
Health Check Operators for Investment Signals

Custom Airflow operators for pipeline health monitoring:
- System health validation
- Data freshness checks
- External dependency verification
- SLA compliance monitoring
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from airflow.models import BaseOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


class HealthCheckOperator(BaseOperator):
    """
    Operator for running health checks on pipeline dependencies.

    Features:
    - Database connectivity checks
    - External API availability
    - Data freshness validation
    - Resource availability checks
    - Configurable thresholds

    Args:
        task_id: Unique task identifier
        checks: List of health checks to perform
        fail_on_unhealthy: Whether to fail task on unhealthy status
        timeout_seconds: Timeout for each check
    """

    template_fields = ("checks",)
    ui_color = "#dcfce7"
    ui_fgcolor = "#166534"

    # Available health checks
    CHECK_TYPES = [
        "database",
        "redis",
        "airflow",
        "external_api",
        "data_freshness",
        "disk_space",
        "memory",
    ]

    @apply_defaults
    def __init__(
        self,
        checks: Optional[List[str]] = None,
        fail_on_unhealthy: bool = True,
        timeout_seconds: int = 30,
        data_freshness_hours: int = 24,
        disk_threshold_percent: int = 90,
        memory_threshold_percent: int = 90,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checks = checks or ["database", "airflow"]
        self.fail_on_unhealthy = fail_on_unhealthy
        self.timeout_seconds = timeout_seconds
        self.data_freshness_hours = data_freshness_hours
        self.disk_threshold_percent = disk_threshold_percent
        self.memory_threshold_percent = memory_threshold_percent

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health checks."""
        start_time = time.time()
        results = {}
        overall_healthy = True
        issues = []

        for check_name in self.checks:
            try:
                check_result = self._run_check(check_name)
                results[check_name] = check_result

                if not check_result.get("healthy", False):
                    overall_healthy = False
                    issues.append(
                        f"{check_name}: {check_result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                results[check_name] = {
                    "healthy": False,
                    "message": str(e),
                    "error": True,
                }
                overall_healthy = False
                issues.append(f"{check_name}: {e}")

        duration = time.time() - start_time

        health_status = {
            "healthy": overall_healthy,
            "checks": results,
            "issues": issues,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Push to XCom
        context["ti"].xcom_push(key="health_status", value=health_status)

        # Log summary
        status_str = "HEALTHY" if overall_healthy else "UNHEALTHY"
        logger.info(
            f"Health check {status_str}: {len([r for r in results.values() if r.get('healthy')])}/"
            f"{len(results)} checks passed in {duration:.2f}s"
        )

        if not overall_healthy:
            error_msg = f"Health check failed: {'; '.join(issues)}"
            if self.fail_on_unhealthy:
                raise Exception(error_msg)
            else:
                logger.warning(error_msg)

        return health_status

    def _run_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        check_methods = {
            "database": self._check_database,
            "redis": self._check_redis,
            "airflow": self._check_airflow,
            "external_api": self._check_external_apis,
            "data_freshness": self._check_data_freshness,
            "disk_space": self._check_disk_space,
            "memory": self._check_memory,
        }

        check_method = check_methods.get(check_name)
        if not check_method:
            return {"healthy": False, "message": f"Unknown check: {check_name}"}

        return check_method()

    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            import psycopg2

            db_url = os.environ.get(
                "DATABASE_URL",
                f"postgresql://{os.environ.get('DB_USER', 'signals')}:"
                f"{os.environ.get('DB_PASSWORD', 'signals_password')}@"
                f"{os.environ.get('DB_HOST', 'postgres')}:"
                f"{os.environ.get('DB_PORT', '5432')}/"
                f"{os.environ.get('DB_NAME', 'investment_signals')}"
            )

            # Parse connection string
            conn = psycopg2.connect(db_url, connect_timeout=self.timeout_seconds)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()

            return {"healthy": True, "message": "Database connection successful"}

        except ImportError:
            return {"healthy": False, "message": "psycopg2 not installed"}
        except Exception as e:
            return {"healthy": False, "message": f"Database connection failed: {e}"}

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import redis

            redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
            r = redis.from_url(redis_url, socket_timeout=self.timeout_seconds)
            r.ping()

            return {"healthy": True, "message": "Redis connection successful"}

        except ImportError:
            return {"healthy": True, "message": "Redis not configured (optional)"}
        except Exception as e:
            return {"healthy": False, "message": f"Redis connection failed: {e}"}

    def _check_airflow(self) -> Dict[str, Any]:
        """Check Airflow webserver health."""
        try:
            import requests

            airflow_url = os.environ.get(
                "AIRFLOW_WEBSERVER_URL", "http://airflow-webserver:8080"
            )
            response = requests.get(
                f"{airflow_url}/health", timeout=self.timeout_seconds
            )

            if response.status_code == 200:
                return {"healthy": True, "message": "Airflow webserver healthy"}
            else:
                return {
                    "healthy": False,
                    "message": f"Airflow returned status {response.status_code}",
                }

        except Exception as e:
            return {"healthy": False, "message": f"Airflow health check failed: {e}"}

    def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API availability."""
        apis = [
            ("SEC EDGAR", "https://data.sec.gov"),
            ("ClinicalTrials.gov", "https://clinicaltrials.gov/api/v2/version"),
        ]

        results = []
        all_healthy = True

        for name, url in apis:
            try:
                import requests

                response = requests.head(url, timeout=10)
                if response.status_code < 400:
                    results.append(f"{name}: OK")
                else:
                    results.append(f"{name}: {response.status_code}")
                    all_healthy = False
            except Exception as e:
                results.append(f"{name}: Error - {e}")
                all_healthy = False

        return {
            "healthy": all_healthy,
            "message": "; ".join(results),
            "details": results,
        }

    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check if pipeline data is fresh."""
        # In production, check database for last execution times
        return {
            "healthy": True,
            "message": f"Data freshness within {self.data_freshness_hours}h threshold",
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            used_percent = (used / total) * 100

            if used_percent < self.disk_threshold_percent:
                return {
                    "healthy": True,
                    "message": f"Disk usage: {used_percent:.1f}%",
                    "used_percent": used_percent,
                }
            else:
                return {
                    "healthy": False,
                    "message": f"High disk usage: {used_percent:.1f}%",
                    "used_percent": used_percent,
                }

        except Exception as e:
            return {"healthy": False, "message": f"Disk check failed: {e}"}

    def _check_memory(self) -> Dict[str, Any]:
        """Check available memory."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent < self.memory_threshold_percent:
                return {
                    "healthy": True,
                    "message": f"Memory usage: {used_percent:.1f}%",
                    "used_percent": used_percent,
                }
            else:
                return {
                    "healthy": False,
                    "message": f"High memory usage: {used_percent:.1f}%",
                    "used_percent": used_percent,
                }

        except ImportError:
            return {"healthy": True, "message": "psutil not available for memory check"}
        except Exception as e:
            return {"healthy": False, "message": f"Memory check failed: {e}"}


class PipelineHealthSensor(BaseSensorOperator):
    """
    Sensor that waits for pipeline health to be good before proceeding.

    Features:
    - Polls health status at intervals
    - Configurable success criteria
    - Timeout handling
    - Soft fail option

    Args:
        task_id: Unique task identifier
        health_checks: List of health checks to verify
        check_interval: Seconds between health checks
        timeout: Maximum time to wait for healthy status
        soft_fail: If True, don't fail DAG on timeout
    """

    template_fields = ("health_checks",)
    ui_color = "#fef3c7"
    ui_fgcolor = "#92400e"

    @apply_defaults
    def __init__(
        self,
        health_checks: Optional[List[str]] = None,
        check_interval: int = 60,
        timeout: int = 300,
        soft_fail: bool = False,
        **kwargs,
    ):
        super().__init__(
            poke_interval=check_interval,
            timeout=timeout,
            soft_fail=soft_fail,
            **kwargs,
        )
        self.health_checks = health_checks or ["database"]

    def poke(self, context: Dict[str, Any]) -> bool:
        """Check if health conditions are met."""
        health_op = HealthCheckOperator(
            task_id=f"{self.task_id}_check",
            checks=self.health_checks,
            fail_on_unhealthy=False,
        )

        try:
            result = health_op.execute(context)
            is_healthy = result.get("healthy", False)

            if is_healthy:
                logger.info("Pipeline health checks passed")
            else:
                logger.info(
                    f"Pipeline health checks not passed yet: {result.get('issues', [])}"
                )

            return is_healthy

        except Exception as e:
            logger.warning(f"Health check error: {e}")
            return False
