"""
Metrics and SLA Monitoring Operators

Custom Airflow operators for metrics collection and SLA monitoring:
- Pipeline execution metrics
- Data quality metrics
- SLA compliance tracking
- Performance benchmarking
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


class MetricsRecordOperator(BaseOperator):
    """
    Operator for recording pipeline execution metrics.

    Features:
    - Aggregates metrics from all pipeline tasks
    - Records to database and Prometheus
    - Calculates derived metrics
    - Supports custom metric functions

    Args:
        task_id: Unique task identifier
        pipeline_name: Name of the pipeline
        input_task_ids: Task IDs to collect metrics from
        custom_metrics: Dict of custom metric functions
        push_to_prometheus: Whether to push to Prometheus
    """

    template_fields = ("pipeline_name", "input_task_ids")
    ui_color = "#e0e7ff"
    ui_fgcolor = "#3730a3"

    @apply_defaults
    def __init__(
        self,
        pipeline_name: str,
        input_task_ids: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        push_to_prometheus: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.input_task_ids = input_task_ids or []
        self.custom_metrics = custom_metrics or {}
        self.push_to_prometheus = push_to_prometheus

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute metrics recording."""
        ti = context["ti"]
        dag_run = context["dag_run"]

        # Collect base metrics
        metrics = {
            "pipeline_name": self.pipeline_name,
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "execution_date": str(context["execution_date"]),
            "start_time": str(dag_run.start_date) if dag_run.start_date else None,
            "recorded_at": datetime.utcnow().isoformat(),
        }

        # Calculate execution duration
        if dag_run.start_date:
            duration = (datetime.utcnow() - dag_run.start_date.replace(tzinfo=None)).total_seconds()
            metrics["execution_duration_seconds"] = round(duration, 2)

        # Collect metrics from input tasks
        task_metrics = {}
        for task_id in self.input_task_ids:
            task_data = ti.xcom_pull(task_ids=task_id)
            if task_data:
                task_metrics[task_id] = task_data

        metrics["task_metrics"] = task_metrics

        # Aggregate key metrics
        metrics.update(self._aggregate_metrics(task_metrics))

        # Run custom metrics
        for metric_name, metric_func in self.custom_metrics.items():
            try:
                metrics[metric_name] = metric_func(task_metrics, context)
            except Exception as e:
                logger.warning(f"Custom metric '{metric_name}' failed: {e}")

        # Push to Prometheus if enabled
        if self.push_to_prometheus:
            self._push_to_prometheus(metrics)

        # Store in database
        self._store_metrics(metrics)

        # Push to XCom
        ti.xcom_push(key="pipeline_metrics", value=metrics)

        logger.info(f"Recorded metrics for {self.pipeline_name}: {metrics}")

        return metrics

    def _aggregate_metrics(
        self, task_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate metrics from tasks."""
        aggregated = {
            "records_processed": 0,
            "signals_found": 0,
            "errors": 0,
            "alerts_sent": 0,
        }

        for task_id, data in task_metrics.items():
            if isinstance(data, dict):
                aggregated["records_processed"] += data.get("records_processed", 0)
                aggregated["signals_found"] += data.get("signals_found", 0)
                aggregated["errors"] += data.get("errors", 0)
                aggregated["alerts_sent"] += data.get("alerts_sent", 0)

        return aggregated

    def _push_to_prometheus(self, metrics: Dict[str, Any]) -> None:
        """Push metrics to Prometheus."""
        try:
            import sys
            import os
            sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
            from orchestration.monitoring.metrics import record_pipeline_execution

            status = "success" if metrics.get("errors", 0) == 0 else "partial"
            record_pipeline_execution(
                pipeline_name=self.pipeline_name,
                status=status,
                duration_seconds=metrics.get("execution_duration_seconds", 0),
                records_processed=metrics.get("records_processed", 0),
                signals_found=metrics.get("signals_found", 0),
                errors=metrics.get("errors", 0),
            )
            logger.info(f"Pushed metrics to Prometheus for {self.pipeline_name}")

        except Exception as e:
            logger.warning(f"Failed to push metrics to Prometheus: {e}")

    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store metrics in database."""
        try:
            import os
            import psycopg2
            import json

            db_url = os.environ.get(
                "DATABASE_URL",
                f"postgresql://{os.environ.get('DB_USER', 'signals')}:"
                f"{os.environ.get('DB_PASSWORD', 'signals_password')}@"
                f"{os.environ.get('DB_HOST', 'postgres')}:5432/"
                f"{os.environ.get('DB_NAME', 'investment_signals')}"
            )

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO orchestration.pipeline_runs
                (pipeline_name, dag_id, run_type, status, start_time,
                 duration_seconds, records_processed, signals_found, errors)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    metrics.get("pipeline_name"),
                    metrics.get("dag_id"),
                    "scheduled",
                    "success" if metrics.get("errors", 0) == 0 else "partial",
                    metrics.get("start_time"),
                    metrics.get("execution_duration_seconds"),
                    metrics.get("records_processed"),
                    metrics.get("signals_found"),
                    metrics.get("errors"),
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Stored metrics in database for {self.pipeline_name}")

        except Exception as e:
            logger.warning(f"Failed to store metrics in database: {e}")


class SLAMonitorOperator(BaseOperator):
    """
    Operator for monitoring SLA compliance.

    Features:
    - Track execution time against SLA targets
    - Monitor data freshness SLAs
    - Alert on SLA violations
    - Historical SLA reporting

    Args:
        task_id: Unique task identifier
        pipeline_name: Name of the pipeline
        execution_sla_seconds: Maximum allowed execution time
        data_freshness_sla_hours: Maximum age for data
        alert_on_violation: Whether to send alerts on SLA miss
    """

    template_fields = ("pipeline_name",)
    ui_color = "#fecaca"
    ui_fgcolor = "#991b1b"

    @apply_defaults
    def __init__(
        self,
        pipeline_name: str,
        execution_sla_seconds: int = 3600,  # 1 hour default
        data_freshness_sla_hours: int = 24,
        alert_on_violation: bool = True,
        sla_targets: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.execution_sla_seconds = execution_sla_seconds
        self.data_freshness_sla_hours = data_freshness_sla_hours
        self.alert_on_violation = alert_on_violation
        self.sla_targets = sla_targets or {}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SLA monitoring."""
        ti = context["ti"]
        dag_run = context["dag_run"]

        sla_status = {
            "pipeline_name": self.pipeline_name,
            "checked_at": datetime.utcnow().isoformat(),
            "violations": [],
            "warnings": [],
            "metrics": {},
        }

        # Check execution time SLA
        if dag_run.start_date:
            execution_time = (
                datetime.utcnow() - dag_run.start_date.replace(tzinfo=None)
            ).total_seconds()
            sla_status["metrics"]["execution_seconds"] = round(execution_time, 2)

            if execution_time > self.execution_sla_seconds:
                sla_status["violations"].append({
                    "type": "execution_time",
                    "target": self.execution_sla_seconds,
                    "actual": round(execution_time, 2),
                    "severity": "high",
                })
            elif execution_time > self.execution_sla_seconds * 0.8:
                # Warning at 80% of SLA
                sla_status["warnings"].append({
                    "type": "execution_time",
                    "message": f"Execution at {execution_time / self.execution_sla_seconds * 100:.0f}% of SLA",
                })

        # Check data freshness SLA
        freshness_result = self._check_data_freshness()
        sla_status["metrics"]["data_freshness_hours"] = freshness_result.get("hours", 0)

        if not freshness_result.get("within_sla", True):
            sla_status["violations"].append({
                "type": "data_freshness",
                "target_hours": self.data_freshness_sla_hours,
                "actual_hours": freshness_result.get("hours", 0),
                "severity": "medium",
            })

        # Check custom SLA targets
        for target_name, target_config in self.sla_targets.items():
            target_result = self._check_custom_target(target_name, target_config, context)
            if target_result:
                if target_result.get("violation"):
                    sla_status["violations"].append(target_result)
                elif target_result.get("warning"):
                    sla_status["warnings"].append(target_result)

        # Determine overall status
        sla_status["passed"] = len(sla_status["violations"]) == 0
        sla_status["violation_count"] = len(sla_status["violations"])
        sla_status["warning_count"] = len(sla_status["warnings"])

        # Alert on violations
        if sla_status["violations"] and self.alert_on_violation:
            self._send_sla_alert(sla_status)

        # Push results
        ti.xcom_push(key="sla_status", value=sla_status)

        # Log summary
        status_str = "PASSED" if sla_status["passed"] else "VIOLATED"
        logger.info(
            f"SLA check {status_str} for {self.pipeline_name}: "
            f"{sla_status['violation_count']} violations, "
            f"{sla_status['warning_count']} warnings"
        )

        return sla_status

    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check data freshness against SLA."""
        try:
            import os
            import psycopg2

            db_url = os.environ.get(
                "DATABASE_URL",
                f"postgresql://{os.environ.get('DB_USER', 'signals')}:"
                f"{os.environ.get('DB_PASSWORD', 'signals_password')}@"
                f"{os.environ.get('DB_HOST', 'postgres')}:5432/"
                f"{os.environ.get('DB_NAME', 'investment_signals')}"
            )

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # Check last successful run time
            cur.execute(
                """
                SELECT MAX(start_time) FROM orchestration.pipeline_runs
                WHERE pipeline_name = %s AND status = 'success'
                """,
                (self.pipeline_name,),
            )
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result and result[0]:
                last_run = result[0]
                hours_ago = (datetime.utcnow() - last_run).total_seconds() / 3600
                return {
                    "hours": round(hours_ago, 2),
                    "within_sla": hours_ago <= self.data_freshness_sla_hours,
                    "last_run": str(last_run),
                }

            return {"hours": 0, "within_sla": True, "message": "No previous runs"}

        except Exception as e:
            logger.warning(f"Failed to check data freshness: {e}")
            return {"hours": 0, "within_sla": True, "error": str(e)}

    def _check_custom_target(
        self,
        target_name: str,
        target_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check a custom SLA target."""
        # Placeholder for custom target checking
        return None

    def _send_sla_alert(self, sla_status: Dict[str, Any]) -> None:
        """Send alert for SLA violations."""
        try:
            import sys
            import os
            sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
            from orchestration.alerts.alert_manager import AlertManager

            violations = sla_status.get("violations", [])
            subject = f"[SLA VIOLATION] {self.pipeline_name} - {len(violations)} violations"

            body_parts = [
                f"Pipeline: {self.pipeline_name}",
                f"Time: {sla_status.get('checked_at')}",
                f"Violations:",
            ]

            for v in violations:
                body_parts.append(
                    f"  - {v['type']}: target={v.get('target')}, actual={v.get('actual')}"
                )

            manager = AlertManager()
            manager.send_email_alert(
                subject=subject,
                body="\n".join(body_parts),
                signal_type="sla_violation",
                priority="high",
            )

            logger.info(f"Sent SLA violation alert for {self.pipeline_name}")

        except Exception as e:
            logger.error(f"Failed to send SLA alert: {e}")
