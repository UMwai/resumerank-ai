"""
Clinical Trial Signal Detection DAG

Orchestrates the clinical trial signal detection pipeline.
Schedule: Daily at 6:00 PM ET (after ClinicalTrials.gov updates)

Tasks:
1. Fetch trial data from ClinicalTrials.gov
2. Fetch SEC 8-K filings
3. Run change detection
4. Calculate composite scores
5. Send alerts for high-confidence signals
6. Generate daily digest
"""

from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Add project root to path
PROJECT_ROOT = Variable.get("PROJECT_ROOT", default_var="/Users/waiyang/Desktop/repo/dreamers-v2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))

from alerts.alert_manager import AlertManager
from monitoring.health_checks import check_clinical_trial_health

# Default arguments for the DAG
default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=1),
    "execution_timeout": timedelta(hours=2),
}

# DAG definition
dag = DAG(
    dag_id="clinical_trial_signal_detection",
    default_args=default_args,
    description="Daily clinical trial signal detection pipeline",
    schedule_interval="0 18 * * *",  # 6:00 PM ET daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["clinical-trials", "signals", "biotech", "investment"],
    doc_md=__doc__,
)


def fetch_trials(**context):
    """Fetch clinical trial data from ClinicalTrials.gov."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
    from main import ClinicalTrialPipeline

    pipeline = ClinicalTrialPipeline(dry_run=False)
    stats = pipeline.fetch_trials(limit=50)

    # Push results to XCom for downstream tasks
    context["ti"].xcom_push(key="fetch_trials_stats", value=stats)

    if stats.get("errors"):
        raise Exception(f"Fetch trials errors: {stats['errors']}")

    return stats


def fetch_sec_filings(**context):
    """Fetch SEC 8-K filings for monitored companies."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
    from main import ClinicalTrialPipeline

    pipeline = ClinicalTrialPipeline(dry_run=False)
    stats = pipeline.fetch_sec_filings(days_back=7)

    context["ti"].xcom_push(key="fetch_sec_stats", value=stats)

    if stats.get("errors"):
        raise Exception(f"Fetch SEC errors: {stats['errors']}")

    return stats


def run_change_detection(**context):
    """Run change detection on all monitored trials."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
    from main import ClinicalTrialPipeline

    pipeline = ClinicalTrialPipeline(dry_run=False)
    stats = pipeline.run_change_detection()

    context["ti"].xcom_push(key="change_detection_stats", value=stats)

    return stats


def calculate_scores(**context):
    """Calculate composite scores for all trials."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
    from main import ClinicalTrialPipeline

    pipeline = ClinicalTrialPipeline(dry_run=False)
    stats = pipeline.calculate_scores()

    context["ti"].xcom_push(key="scoring_stats", value=stats)

    # Check for high-confidence signals
    high_confidence_signals = []
    if stats.get("strong_buys", 0) > 0 or stats.get("shorts", 0) > 0:
        # These are candidates for alerting
        high_confidence_signals = {
            "strong_buys": stats.get("strong_buys", 0),
            "shorts": stats.get("shorts", 0),
        }
        context["ti"].xcom_push(key="high_confidence_signals", value=high_confidence_signals)

    return stats


def send_high_confidence_alerts(**context):
    """Send alerts for high-confidence signals (score >= 7)."""
    ti = context["ti"]

    # Get scoring results from previous task
    scoring_stats = ti.xcom_pull(key="scoring_stats", task_ids="calculate_scores")
    high_confidence = ti.xcom_pull(key="high_confidence_signals", task_ids="calculate_scores")

    if not high_confidence:
        return {"alerts_sent": 0, "message": "No high-confidence signals found"}

    # Initialize alert manager
    alert_manager = AlertManager()

    # Send alerts based on signal type
    alerts_sent = 0

    if high_confidence.get("strong_buys", 0) > 0:
        alert_manager.send_email_alert(
            subject="[HIGH CONFIDENCE] Clinical Trial BUY Signals Detected",
            body=f"Detected {high_confidence['strong_buys']} strong buy signals from clinical trial analysis.",
            signal_type="clinical_trial",
            priority="high"
        )
        alerts_sent += 1

    if high_confidence.get("shorts", 0) > 0:
        alert_manager.send_email_alert(
            subject="[HIGH CONFIDENCE] Clinical Trial SHORT Signals Detected",
            body=f"Detected {high_confidence['shorts']} short signals from clinical trial analysis.",
            signal_type="clinical_trial",
            priority="high"
        )
        alerts_sent += 1

    return {"alerts_sent": alerts_sent}


def send_daily_digest(**context):
    """Generate and send daily email digest."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
    from main import ClinicalTrialPipeline

    pipeline = ClinicalTrialPipeline(dry_run=False)
    stats = pipeline.send_email_digest()

    context["ti"].xcom_push(key="email_stats", value=stats)

    return stats


def check_health(**context):
    """Run health check for clinical trial pipeline."""
    health_status = check_clinical_trial_health()

    if not health_status["healthy"]:
        raise Exception(f"Health check failed: {health_status['issues']}")

    context["ti"].xcom_push(key="health_status", value=health_status)
    return health_status


def record_metrics(**context):
    """Record pipeline execution metrics."""
    ti = context["ti"]

    # Collect stats from all tasks
    fetch_trials_stats = ti.xcom_pull(key="fetch_trials_stats", task_ids="fetch_trials") or {}
    fetch_sec_stats = ti.xcom_pull(key="fetch_sec_stats", task_ids="fetch_sec_filings") or {}
    change_detection_stats = ti.xcom_pull(key="change_detection_stats", task_ids="run_change_detection") or {}
    scoring_stats = ti.xcom_pull(key="scoring_stats", task_ids="calculate_scores") or {}
    email_stats = ti.xcom_pull(key="email_stats", task_ids="send_daily_digest") or {}

    metrics = {
        "execution_date": str(context["execution_date"]),
        "run_id": context["run_id"],
        "new_trials": fetch_trials_stats.get("new_trials", 0),
        "updated_trials": fetch_trials_stats.get("updated_trials", 0),
        "filings_processed": fetch_sec_stats.get("filings_processed", 0),
        "signals_found": fetch_sec_stats.get("signals_found", 0),
        "changes_detected": change_detection_stats.get("changes_detected", 0),
        "trials_scored": scoring_stats.get("trials_scored", 0),
        "strong_buys": scoring_stats.get("strong_buys", 0),
        "shorts": scoring_stats.get("shorts", 0),
        "email_sent": email_stats.get("sent", False),
    }

    # Log metrics (in production, send to monitoring system)
    print(f"Pipeline metrics: {metrics}")

    return metrics


# Task definitions
with dag:
    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    # Health check before starting
    health_check = PythonOperator(
        task_id="health_check",
        python_callable=check_health,
        provide_context=True,
    )

    # Data fetching tasks (can run in parallel)
    with TaskGroup(group_id="data_fetching") as data_fetching:
        fetch_trials_task = PythonOperator(
            task_id="fetch_trials",
            python_callable=fetch_trials,
            provide_context=True,
        )

        fetch_sec_task = PythonOperator(
            task_id="fetch_sec_filings",
            python_callable=fetch_sec_filings,
            provide_context=True,
        )

    # Change detection
    change_detection_task = PythonOperator(
        task_id="run_change_detection",
        python_callable=run_change_detection,
        provide_context=True,
    )

    # Scoring
    scoring_task = PythonOperator(
        task_id="calculate_scores",
        python_callable=calculate_scores,
        provide_context=True,
    )

    # Alerting tasks (can run in parallel)
    with TaskGroup(group_id="alerting") as alerting:
        high_confidence_alerts_task = PythonOperator(
            task_id="send_high_confidence_alerts",
            python_callable=send_high_confidence_alerts,
            provide_context=True,
        )

        daily_digest_task = PythonOperator(
            task_id="send_daily_digest",
            python_callable=send_daily_digest,
            provide_context=True,
        )

    # Record metrics
    record_metrics_task = PythonOperator(
        task_id="record_metrics",
        python_callable=record_metrics,
        provide_context=True,
    )

    # Define task dependencies
    start >> health_check >> data_fetching >> change_detection_task >> scoring_task >> alerting >> record_metrics_task >> end
