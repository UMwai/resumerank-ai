"""
Clinical Trial Signal Detection Flow (Prefect)

Alternative to Airflow DAG using Prefect for orchestration.
Schedule: Daily at 6:00 PM ET (after ClinicalTrials.gov updates)

Usage:
    # Run locally
    python clinical_trial_flow.py

    # Deploy to Prefect Cloud
    prefect deployment build clinical_trial_flow.py:clinical_trial_flow -n production
    prefect deployment apply clinical_trial_flow-deployment.yaml
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Add project root to path
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/Users/waiyang/Desktop/repo/dreamers-v2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))


@task(
    name="fetch_clinical_trials",
    retries=3,
    retry_delay_seconds=300,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def fetch_trials(limit: int = 50) -> Dict[str, Any]:
    """Fetch clinical trial data from ClinicalTrials.gov."""
    logger = get_run_logger()
    logger.info(f"Fetching top {limit} Phase 3 trials...")

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
        from main import ClinicalTrialPipeline

        pipeline = ClinicalTrialPipeline(dry_run=False)
        stats = pipeline.fetch_trials(limit=limit)

        logger.info(f"Fetched {stats.get('new_trials', 0)} new trials")
        return stats

    except Exception as e:
        logger.error(f"Failed to fetch trials: {e}")
        raise


@task(
    name="fetch_sec_filings",
    retries=3,
    retry_delay_seconds=300,
)
def fetch_sec_filings(days_back: int = 7) -> Dict[str, Any]:
    """Fetch SEC 8-K filings for monitored companies."""
    logger = get_run_logger()
    logger.info(f"Fetching SEC filings from last {days_back} days...")

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
        from main import ClinicalTrialPipeline

        pipeline = ClinicalTrialPipeline(dry_run=False)
        stats = pipeline.fetch_sec_filings(days_back=days_back)

        logger.info(f"Processed {stats.get('filings_processed', 0)} filings")
        return stats

    except Exception as e:
        logger.error(f"Failed to fetch SEC filings: {e}")
        raise


@task(name="run_change_detection")
def run_change_detection() -> Dict[str, Any]:
    """Run change detection on all monitored trials."""
    logger = get_run_logger()
    logger.info("Running change detection...")

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
        from main import ClinicalTrialPipeline

        pipeline = ClinicalTrialPipeline(dry_run=False)
        stats = pipeline.run_change_detection()

        logger.info(f"Detected {stats.get('changes_detected', 0)} changes")
        return stats

    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        raise


@task(name="calculate_scores")
def calculate_scores() -> Dict[str, Any]:
    """Calculate composite scores for all trials."""
    logger = get_run_logger()
    logger.info("Calculating scores...")

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
        from main import ClinicalTrialPipeline

        pipeline = ClinicalTrialPipeline(dry_run=False)
        stats = pipeline.calculate_scores()

        logger.info(f"Scored {stats.get('trials_scored', 0)} trials")
        return stats

    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise


@task(name="send_alerts")
def send_alerts(scoring_stats: Dict[str, Any]) -> Dict[str, bool]:
    """Send alerts for high-confidence signals."""
    logger = get_run_logger()

    from alerts.alert_manager import AlertManager

    alert_manager = AlertManager()
    results = {"email": False, "slack": False}

    strong_buys = scoring_stats.get("strong_buys", 0)
    shorts = scoring_stats.get("shorts", 0)

    if strong_buys > 0:
        results["email"] = alert_manager.send_email_alert(
            subject="[HIGH CONFIDENCE] Clinical Trial BUY Signals",
            body=f"Detected {strong_buys} strong buy signals from clinical trial analysis.",
            signal_type="clinical_trial",
            priority="high",
        )
        logger.info(f"Sent BUY alert: {results['email']}")

    if shorts > 0:
        results["email"] = alert_manager.send_email_alert(
            subject="[HIGH CONFIDENCE] Clinical Trial SHORT Signals",
            body=f"Detected {shorts} short signals from clinical trial analysis.",
            signal_type="clinical_trial",
            priority="high",
        )
        logger.info(f"Sent SHORT alert: {results['email']}")

    return results


@task(name="send_daily_digest")
def send_daily_digest() -> Dict[str, Any]:
    """Generate and send daily email digest."""
    logger = get_run_logger()
    logger.info("Generating daily digest...")

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "clinical_trial_signals"))
        from main import ClinicalTrialPipeline

        pipeline = ClinicalTrialPipeline(dry_run=False)
        stats = pipeline.send_email_digest()

        logger.info(f"Daily digest sent: {stats.get('sent', False)}")
        return stats

    except Exception as e:
        logger.error(f"Failed to send digest: {e}")
        raise


@task(name="record_metrics")
def record_metrics(
    fetch_stats: Dict[str, Any],
    sec_stats: Dict[str, Any],
    change_stats: Dict[str, Any],
    score_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Record pipeline execution metrics."""
    logger = get_run_logger()

    from monitoring.metrics import record_pipeline_execution

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "new_trials": fetch_stats.get("new_trials", 0),
        "updated_trials": fetch_stats.get("updated_trials", 0),
        "filings_processed": sec_stats.get("filings_processed", 0),
        "signals_found": sec_stats.get("signals_found", 0),
        "changes_detected": change_stats.get("changes_detected", 0),
        "trials_scored": score_stats.get("trials_scored", 0),
        "strong_buys": score_stats.get("strong_buys", 0),
        "shorts": score_stats.get("shorts", 0),
    }

    # Record to metrics system
    record_pipeline_execution(
        "clinical_trial_signals",
        "success",
        0,  # Duration calculated by Prefect
        records_processed=metrics["new_trials"] + metrics["updated_trials"],
        signals_found=metrics["signals_found"],
    )

    logger.info(f"Recorded metrics: {metrics}")
    return metrics


@flow(
    name="clinical_trial_signal_detection",
    description="Daily clinical trial signal detection pipeline",
    version="1.0.0",
)
def clinical_trial_flow(
    trial_limit: int = 50,
    sec_days_back: int = 7,
) -> Dict[str, Any]:
    """
    Main clinical trial signal detection flow.

    Args:
        trial_limit: Maximum number of trials to fetch
        sec_days_back: Days to look back for SEC filings

    Returns:
        Dictionary with pipeline execution results
    """
    logger = get_run_logger()
    logger.info("Starting Clinical Trial Signal Detection Pipeline")

    # Data fetching (parallel)
    fetch_stats = fetch_trials(limit=trial_limit)
    sec_stats = fetch_sec_filings(days_back=sec_days_back)

    # Wait for data fetching to complete
    fetch_stats_result = fetch_stats
    sec_stats_result = sec_stats

    # Sequential processing
    change_stats = run_change_detection()
    score_stats = calculate_scores()

    # Alerting (parallel)
    alert_results = send_alerts(score_stats)
    digest_result = send_daily_digest()

    # Record metrics
    metrics = record_metrics(
        fetch_stats_result,
        sec_stats_result,
        change_stats,
        score_stats,
    )

    logger.info("Pipeline completed successfully")

    return {
        "status": "success",
        "fetch_stats": fetch_stats_result,
        "sec_stats": sec_stats_result,
        "change_stats": change_stats,
        "score_stats": score_stats,
        "alerts": alert_results,
        "metrics": metrics,
    }


def create_deployment():
    """Create Prefect deployment for the flow."""
    deployment = Deployment.build_from_flow(
        flow=clinical_trial_flow,
        name="clinical-trial-signals-production",
        schedule=CronSchedule(cron="0 18 * * *", timezone="America/New_York"),
        work_queue_name="default",
        tags=["clinical-trials", "signals", "production"],
    )
    deployment.apply()
    print("Deployment created successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Create deployment")
    parser.add_argument("--limit", type=int, default=50, help="Trial limit")
    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        # Run flow locally
        result = clinical_trial_flow(trial_limit=args.limit)
        print(f"Result: {result}")
