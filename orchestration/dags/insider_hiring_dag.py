"""
Insider Activity + Hiring Signals DAG

Orchestrates multiple sub-pipelines for insider trading and hiring signals.

Sub-DAGs:
1. Form 4 Insider Trading - Every 30 minutes during market hours (9:30 AM - 4:00 PM ET)
2. 13F Institutional Holdings - Quarterly (45 days after quarter end)
3. Job Postings - Daily at 9:00 AM ET

This is the main orchestrator that includes all sub-DAGs.
"""

from datetime import datetime, timedelta
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.sensors.time_delta import TimeDeltaSensor

# Project configuration
PROJECT_ROOT = Variable.get("PROJECT_ROOT", default_var="/Users/waiyang/Desktop/repo/dreamers-v2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))

from alerts.alert_manager import AlertManager

# =============================================================================
# FORM 4 DAG - Every 30 minutes during market hours
# =============================================================================

form4_default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=25),
}

form4_dag = DAG(
    dag_id="insider_form4_signals",
    default_args=form4_default_args,
    description="Real-time Form 4 insider trading signal detection",
    schedule_interval="*/30 9-16 * * 1-5",  # Every 30 min, Mon-Fri, 9 AM - 4 PM ET
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["insider-trading", "form4", "real-time", "signals", "investment"],
    doc_md="""
## Form 4 Insider Trading Signals

Runs every 30 minutes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri).

Detects:
- Large insider purchases (>$100K)
- Cluster buying (multiple insiders)
- C-suite transactions
- Options exercises
""",
)


def fetch_form4_filings(**context):
    """Fetch recent Form 4 filings from SEC EDGAR."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from scrapers import Form4Scraper

    scraper = Form4Scraper()
    result = scraper.run()

    context["ti"].xcom_push(key="form4_result", value=result)
    return result


def analyze_form4_signals(**context):
    """Analyze Form 4 filings for trading signals."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from models import SignalScorer

    ti = context["ti"]
    form4_result = ti.xcom_pull(key="form4_result", task_ids="fetch_form4_filings")

    if not form4_result or form4_result.get("filings_count", 0) == 0:
        return {"signals_found": 0, "message": "No new filings"}

    scorer = SignalScorer()

    # Get unique tickers from filings
    tickers = form4_result.get("tickers", [])
    signals = []
    high_priority_signals = []

    for ticker in tickers[:20]:  # Limit to prevent timeout
        try:
            score = scorer.calculate_score(ticker)
            if score and score.insider_score > 0:
                signal_data = {
                    "ticker": ticker,
                    "insider_score": score.insider_score,
                    "composite_score": score.composite_score,
                    "recommendation": score.recommendation,
                    "confidence": score.confidence,
                }
                signals.append(signal_data)

                # Track high-priority signals
                if score.composite_score >= 7.0 or score.insider_score >= 8.0:
                    high_priority_signals.append(signal_data)

                scorer.save_score(score)
        except Exception as e:
            print(f"Error scoring {ticker}: {e}")
            continue

    context["ti"].xcom_push(key="signals", value=signals)
    context["ti"].xcom_push(key="high_priority_signals", value=high_priority_signals)

    return {"signals_found": len(signals), "high_priority": len(high_priority_signals)}


def send_form4_alerts(**context):
    """Send real-time alerts for significant Form 4 filings."""
    ti = context["ti"]
    high_priority = ti.xcom_pull(key="high_priority_signals", task_ids="analyze_form4_signals")

    if not high_priority:
        return {"alerts_sent": 0}

    alert_manager = AlertManager()
    alerts_sent = 0

    for signal in high_priority:
        # Send Slack alert for real-time notification
        alert_manager.send_slack_alert(
            message=f"""
*Insider Trading Alert*
Ticker: `{signal['ticker']}`
Insider Score: {signal['insider_score']:.1f}/10
Composite Score: {signal['composite_score']:.1f}/10
Recommendation: {signal['recommendation']}
Confidence: {signal['confidence']:.0%}
            """,
            signal_type="form4",
            priority="high" if signal['composite_score'] >= 8.0 else "medium"
        )
        alerts_sent += 1

        # Also send email for high-confidence signals
        if signal['composite_score'] >= 8.0:
            alert_manager.send_email_alert(
                subject=f"[URGENT] High-Confidence Insider Signal: {signal['ticker']}",
                body=f"""
High-Confidence Insider Trading Signal Detected

Ticker: {signal['ticker']}
Insider Score: {signal['insider_score']:.1f}/10
Composite Score: {signal['composite_score']:.1f}/10
Recommendation: {signal['recommendation']}
Confidence: {signal['confidence']:.0%}

This signal has been flagged for immediate review.
                """,
                signal_type="form4",
                priority="critical"
            )

    return {"alerts_sent": alerts_sent}


with form4_dag:
    form4_start = DummyOperator(task_id="start")
    form4_end = DummyOperator(task_id="end")

    fetch_task = PythonOperator(
        task_id="fetch_form4_filings",
        python_callable=fetch_form4_filings,
        provide_context=True,
    )

    analyze_task = PythonOperator(
        task_id="analyze_form4_signals",
        python_callable=analyze_form4_signals,
        provide_context=True,
    )

    alert_task = PythonOperator(
        task_id="send_form4_alerts",
        python_callable=send_form4_alerts,
        provide_context=True,
    )

    form4_start >> fetch_task >> analyze_task >> alert_task >> form4_end


# =============================================================================
# 13F DAG - Quarterly (45 days after quarter end)
# =============================================================================

form13f_default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 3,
    "retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=6),
}

form13f_dag = DAG(
    dag_id="institutional_13f_signals",
    default_args=form13f_default_args,
    description="Quarterly 13F institutional holdings analysis",
    # Schedule: 8 AM on Feb 14, May 15, Aug 14, Nov 14 (45 days after quarter end)
    schedule_interval="0 8 14 2,8,11 * | 0 8 15 5 *",  # Complex schedule handled separately
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["institutional", "13f", "quarterly", "signals", "investment"],
    doc_md="""
## 13F Institutional Holdings Analysis

Runs quarterly, 45 days after each quarter end:
- February 14 (Q4 filings)
- May 15 (Q1 filings)
- August 14 (Q2 filings)
- November 14 (Q3 filings)

Analyzes institutional position changes from hedge funds and large investors.
""",
)


def fetch_13f_filings(**context):
    """Fetch 13F institutional holdings filings."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from scrapers import Form13FScraper

    scraper = Form13FScraper()
    result = scraper.run()

    context["ti"].xcom_push(key="form13f_result", value=result)
    return result


def analyze_institutional_changes(**context):
    """Analyze institutional position changes."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from models import SignalScorer

    ti = context["ti"]
    result = ti.xcom_pull(key="form13f_result", task_ids="fetch_13f_filings")

    if not result:
        return {"status": "no_data"}

    scorer = SignalScorer()
    significant_changes = []

    # Analyze position changes
    for holding in result.get("holdings", [])[:100]:
        ticker = holding.get("ticker")
        if not ticker:
            continue

        try:
            score = scorer.calculate_score(ticker)
            if score and score.institutional_score >= 6.0:
                significant_changes.append({
                    "ticker": ticker,
                    "institutional_score": score.institutional_score,
                    "composite_score": score.composite_score,
                    "recommendation": score.recommendation,
                })
                scorer.save_score(score)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

    context["ti"].xcom_push(key="significant_changes", value=significant_changes)
    return {"analyzed": len(result.get("holdings", [])), "significant": len(significant_changes)}


def generate_13f_report(**context):
    """Generate quarterly 13F analysis report."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from reports import ReportGenerator

    ti = context["ti"]
    significant_changes = ti.xcom_pull(key="significant_changes", task_ids="analyze_institutional_changes")

    generator = ReportGenerator()
    generator.print_console_report()

    return {"report_generated": True}


def send_13f_alerts(**context):
    """Send alerts for significant institutional changes."""
    ti = context["ti"]
    significant_changes = ti.xcom_pull(key="significant_changes", task_ids="analyze_institutional_changes")

    if not significant_changes:
        return {"alerts_sent": 0}

    alert_manager = AlertManager()

    # Group by recommendation
    buys = [c for c in significant_changes if "BUY" in c['recommendation'].upper()]
    sells = [c for c in significant_changes if "SELL" in c['recommendation'].upper() or "SHORT" in c['recommendation'].upper()]

    if buys:
        buy_list = "\n".join([f"- {c['ticker']}: Score {c['composite_score']:.1f}" for c in buys[:10]])
        alert_manager.send_email_alert(
            subject=f"[13F] Institutional BUY Signals - {len(buys)} Opportunities",
            body=f"""
Quarterly 13F Analysis Complete

{len(buys)} BUY signals detected from institutional filings:

{buy_list}

Review the full report for details.
            """,
            signal_type="13f",
            priority="high"
        )

    if sells:
        sell_list = "\n".join([f"- {c['ticker']}: Score {c['composite_score']:.1f}" for c in sells[:10]])
        alert_manager.send_email_alert(
            subject=f"[13F] Institutional SELL Signals - {len(sells)} Warnings",
            body=f"""
Quarterly 13F Analysis Complete

{len(sells)} SELL/SHORT signals detected from institutional filings:

{sell_list}

Review the full report for details.
            """,
            signal_type="13f",
            priority="high"
        )

    return {"alerts_sent": int(bool(buys)) + int(bool(sells))}


with form13f_dag:
    f13_start = DummyOperator(task_id="start")
    f13_end = DummyOperator(task_id="end")

    fetch_13f_task = PythonOperator(
        task_id="fetch_13f_filings",
        python_callable=fetch_13f_filings,
        provide_context=True,
    )

    analyze_13f_task = PythonOperator(
        task_id="analyze_institutional_changes",
        python_callable=analyze_institutional_changes,
        provide_context=True,
    )

    report_13f_task = PythonOperator(
        task_id="generate_13f_report",
        python_callable=generate_13f_report,
        provide_context=True,
    )

    alert_13f_task = PythonOperator(
        task_id="send_13f_alerts",
        python_callable=send_13f_alerts,
        provide_context=True,
    )

    f13_start >> fetch_13f_task >> analyze_13f_task >> [report_13f_task, alert_13f_task] >> f13_end


# =============================================================================
# JOB POSTINGS DAG - Daily at 9:00 AM ET
# =============================================================================

jobs_default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 3,
    "retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=3),
}

jobs_dag = DAG(
    dag_id="hiring_signals",
    default_args=jobs_default_args,
    description="Daily job posting analysis for hiring signals",
    schedule_interval="0 9 * * *",  # 9:00 AM ET daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["hiring", "jobs", "signals", "investment"],
    doc_md="""
## Hiring Signals from Job Postings

Runs daily at 9:00 AM ET.

Analyzes job postings to detect:
- Rapid hiring (growth signal)
- Leadership changes
- New product/market expansion
- Cost cutting (mass layoffs)
""",
)


def fetch_job_postings(**context):
    """Fetch job postings from various sources."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from scrapers import JobScraper

    scraper = JobScraper()
    result = scraper.run()

    context["ti"].xcom_push(key="jobs_result", value=result)
    return result


def analyze_hiring_signals(**context):
    """Analyze job postings for hiring signals."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from models import SignalScorer

    ti = context["ti"]
    result = ti.xcom_pull(key="jobs_result", task_ids="fetch_job_postings")

    if not result:
        return {"status": "no_data"}

    scorer = SignalScorer()
    hiring_signals = []

    # Get companies with notable hiring activity
    companies = result.get("companies", [])

    for company in companies[:50]:
        ticker = company.get("ticker")
        if not ticker:
            continue

        try:
            score = scorer.calculate_score(ticker)
            if score and score.hiring_score >= 5.0:
                hiring_signals.append({
                    "ticker": ticker,
                    "company": company.get("name"),
                    "hiring_score": score.hiring_score,
                    "composite_score": score.composite_score,
                    "recommendation": score.recommendation,
                    "job_count": company.get("job_count", 0),
                })
                scorer.save_score(score)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

    context["ti"].xcom_push(key="hiring_signals", value=hiring_signals)
    return {"analyzed": len(companies), "signals": len(hiring_signals)}


def send_hiring_alerts(**context):
    """Send alerts for significant hiring signals."""
    ti = context["ti"]
    hiring_signals = ti.xcom_pull(key="hiring_signals", task_ids="analyze_hiring_signals")

    if not hiring_signals:
        return {"alerts_sent": 0}

    # Filter high-confidence signals
    high_confidence = [s for s in hiring_signals if s['composite_score'] >= 7.0]

    if not high_confidence:
        return {"alerts_sent": 0}

    alert_manager = AlertManager()

    signal_list = "\n".join([
        f"- {s['ticker']} ({s['company']}): {s['job_count']} jobs, Score {s['composite_score']:.1f}"
        for s in high_confidence[:10]
    ])

    alert_manager.send_email_alert(
        subject=f"[HIRING] {len(high_confidence)} High-Confidence Hiring Signals",
        body=f"""
Daily Hiring Signal Analysis Complete

{len(high_confidence)} high-confidence signals detected:

{signal_list}

These companies show significant hiring activity that may indicate growth opportunities.
        """,
        signal_type="hiring",
        priority="medium"
    )

    return {"alerts_sent": 1}


def run_daily_scoring(**context):
    """Run comprehensive daily scoring for all tracked companies."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from main import run_scoring

    scores = run_scoring()
    return {"companies_scored": len(scores) if scores else 0}


def generate_daily_digest(**context):
    """Generate and send daily insider/hiring digest."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "insider_hiring_signals"))
    from main import run_digest

    result = run_digest(send=True, save=True)
    return result


with jobs_dag:
    jobs_start = DummyOperator(task_id="start")
    jobs_end = DummyOperator(task_id="end")

    fetch_jobs_task = PythonOperator(
        task_id="fetch_job_postings",
        python_callable=fetch_job_postings,
        provide_context=True,
    )

    analyze_jobs_task = PythonOperator(
        task_id="analyze_hiring_signals",
        python_callable=analyze_hiring_signals,
        provide_context=True,
    )

    alert_jobs_task = PythonOperator(
        task_id="send_hiring_alerts",
        python_callable=send_hiring_alerts,
        provide_context=True,
    )

    scoring_task = PythonOperator(
        task_id="run_daily_scoring",
        python_callable=run_daily_scoring,
        provide_context=True,
    )

    digest_task = PythonOperator(
        task_id="generate_daily_digest",
        python_callable=generate_daily_digest,
        provide_context=True,
    )

    jobs_start >> fetch_jobs_task >> analyze_jobs_task >> alert_jobs_task >> scoring_task >> digest_task >> jobs_end


# =============================================================================
# MASTER DAG - Orchestrates all insider/hiring sub-DAGs
# =============================================================================

master_default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

master_dag = DAG(
    dag_id="insider_hiring_master",
    default_args=master_default_args,
    description="Master orchestrator for all insider/hiring signal pipelines",
    schedule_interval=None,  # Triggered manually or by external scheduler
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["master", "orchestrator", "signals", "investment"],
    doc_md="""
## Master Insider/Hiring Signals Orchestrator

This DAG can be used to manually trigger all sub-DAGs or run a comprehensive
analysis across all signal types.

Sub-DAGs:
- insider_form4_signals (Every 30 min during market hours)
- institutional_13f_signals (Quarterly)
- hiring_signals (Daily)
""",
)

with master_dag:
    master_start = DummyOperator(task_id="start")
    master_end = DummyOperator(task_id="end")

    trigger_form4 = TriggerDagRunOperator(
        task_id="trigger_form4_dag",
        trigger_dag_id="insider_form4_signals",
        wait_for_completion=False,
    )

    trigger_jobs = TriggerDagRunOperator(
        task_id="trigger_jobs_dag",
        trigger_dag_id="hiring_signals",
        wait_for_completion=False,
    )

    master_start >> [trigger_form4, trigger_jobs] >> master_end
