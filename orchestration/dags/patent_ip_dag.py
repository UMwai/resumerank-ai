"""
Patent/IP Intelligence DAG

Orchestrates the patent intelligence pipeline for drug patent cliff analysis.
Schedule: Weekly on Monday at 8:00 AM ET

Tasks:
1. Extract FDA Orange Book data
2. Extract USPTO patent data
3. Extract ANDA (generic competition) data
4. Enrich patent data with calculations
5. Generate patent cliff calendar
6. Score opportunities
7. Load to database
8. Send weekly digest
"""

from datetime import datetime, timedelta
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Project configuration
PROJECT_ROOT = Variable.get("PROJECT_ROOT", default_var="/Users/waiyang/Desktop/repo/dreamers-v2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))

from alerts.alert_manager import AlertManager

# Default arguments
default_args = {
    "owner": "investment-signals",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": Variable.get("ALERT_EMAIL_RECIPIENTS", default_var="").split(","),
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=2),
    "execution_timeout": timedelta(hours=4),
}

# DAG definition
dag = DAG(
    dag_id="patent_ip_intelligence",
    default_args=default_args,
    description="Weekly patent/IP intelligence pipeline for patent cliff analysis",
    schedule_interval="0 8 * * 1",  # Monday 8:00 AM ET
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["patents", "ip", "pharma", "signals", "investment"],
    doc_md=__doc__,
)


def extract_orange_book_data(**context):
    """Extract drug and patent data from FDA Orange Book."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    pipeline = PatentIntelligencePipeline()
    pipeline.extract_orange_book_data(top_n=100)

    stats = {
        "drugs_extracted": len(pipeline.drugs),
        "patents_extracted": len(pipeline.patents),
    }

    # Store pipeline instance data for downstream tasks
    context["ti"].xcom_push(key="drugs", value=pipeline.drugs)
    context["ti"].xcom_push(key="patents", value=pipeline.patents)
    context["ti"].xcom_push(key="orange_book_stats", value=stats)

    return stats


def extract_uspto_data(**context):
    """Enrich patent data with USPTO information."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    patents = ti.xcom_pull(key="patents", task_ids="extract_orange_book_data")

    if not patents:
        return {"status": "skipped", "message": "No patents to enrich"}

    pipeline = PatentIntelligencePipeline()
    pipeline.patents = patents
    pipeline.enrich_patent_data()

    # Push enriched patents
    context["ti"].xcom_push(key="enriched_patents", value=pipeline.patents)

    return {"patents_enriched": len(pipeline.patents)}


def extract_anda_data(**context):
    """Extract ANDA (generic application) data."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")

    if not drugs:
        return {"status": "skipped", "message": "No drugs to analyze"}

    pipeline = PatentIntelligencePipeline()
    pipeline.drugs = drugs
    pipeline.extract_anda_data()

    context["ti"].xcom_push(key="andas", value=pipeline.andas)

    return {"andas_extracted": len(pipeline.andas)}


def generate_calendar(**context):
    """Generate patent cliff calendar from extracted data."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")
    patents = ti.xcom_pull(key="enriched_patents", task_ids="enrich_patent_data")
    andas = ti.xcom_pull(key="andas", task_ids="extract_anda_data")

    pipeline = PatentIntelligencePipeline()
    pipeline.drugs = drugs or []
    pipeline.patents = patents or []
    pipeline.andas = andas or []
    pipeline.generate_calendar()

    context["ti"].xcom_push(key="calendar_events", value=pipeline.calendar_events)

    return {"events_generated": len(pipeline.calendar_events)}


def score_opportunities(**context):
    """Score patent cliff opportunities."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from transformers.scoring import PatentCliffScorer, DrugPatentData

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")
    patents = ti.xcom_pull(key="enriched_patents", task_ids="enrich_patent_data")
    calendar_events = ti.xcom_pull(key="calendar_events", task_ids="generate_calendar")

    scorer = PatentCliffScorer()
    scored_opportunities = []
    high_confidence_opportunities = []

    # Score each drug
    for drug in (drugs or []):
        drug_patents = [p for p in (patents or []) if p.get("drug_id") == drug.get("drug_id")]

        if not drug_patents:
            continue

        # Create DrugPatentData object
        try:
            drug_data = DrugPatentData(
                brand_name=drug.get("brand_name", "Unknown"),
                generic_name=drug.get("generic_name", "Unknown"),
                company=drug.get("branded_company", "Unknown"),
                annual_revenue=drug.get("annual_revenue_millions", 0),
                patent_expiry_date=drug_patents[0].get("final_expiration_date"),
                generic_competition_expected=bool(drug.get("anda_count", 0) > 0),
                therapeutic_area=drug.get("therapeutic_area", "Unknown"),
            )

            score_result = scorer.score_patent_cliff(drug_data)
            scored_opportunities.append({
                "drug": drug.get("brand_name"),
                "company": drug.get("branded_company"),
                "score": score_result.total_score,
                "recommendation": score_result.recommendation,
            })

            # Track high-confidence opportunities
            if score_result.total_score >= 7.0:
                high_confidence_opportunities.append({
                    "drug": drug.get("brand_name"),
                    "company": drug.get("branded_company"),
                    "score": score_result.total_score,
                    "recommendation": score_result.recommendation,
                    "factors": score_result.factors,
                })
        except Exception as e:
            print(f"Error scoring {drug.get('brand_name')}: {e}")
            continue

    context["ti"].xcom_push(key="scored_opportunities", value=scored_opportunities)
    context["ti"].xcom_push(key="high_confidence_opportunities", value=high_confidence_opportunities)

    return {
        "opportunities_scored": len(scored_opportunities),
        "high_confidence_count": len(high_confidence_opportunities),
    }


def load_to_database(**context):
    """Load all extracted and transformed data to database."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")
    patents = ti.xcom_pull(key="enriched_patents", task_ids="enrich_patent_data")
    andas = ti.xcom_pull(key="andas", task_ids="extract_anda_data")
    calendar_events = ti.xcom_pull(key="calendar_events", task_ids="generate_calendar")

    pipeline = PatentIntelligencePipeline()
    pipeline.drugs = drugs or []
    pipeline.patents = patents or []
    pipeline.andas = andas or []
    pipeline.calendar_events = calendar_events or []

    stats = pipeline.load_to_database()
    context["ti"].xcom_push(key="db_stats", value=stats)

    return stats


def send_alerts(**context):
    """Send alerts for high-confidence patent cliff opportunities."""
    ti = context["ti"]
    high_confidence = ti.xcom_pull(key="high_confidence_opportunities", task_ids="score_opportunities")

    if not high_confidence:
        return {"alerts_sent": 0, "message": "No high-confidence opportunities found"}

    alert_manager = AlertManager()
    alerts_sent = 0

    for opportunity in high_confidence:
        # Send email alert
        alert_manager.send_email_alert(
            subject=f"[PATENT CLIFF] High-Confidence Opportunity: {opportunity['drug']}",
            body=f"""
Patent Cliff Alert

Drug: {opportunity['drug']}
Company: {opportunity['company']}
Score: {opportunity['score']:.1f}/10
Recommendation: {opportunity['recommendation']}

Key Factors:
{chr(10).join(f"- {k}: {v}" for k, v in opportunity.get('factors', {}).items())}
            """,
            signal_type="patent_cliff",
            priority="high" if opportunity['score'] >= 8.0 else "medium"
        )
        alerts_sent += 1

    return {"alerts_sent": alerts_sent}


def send_weekly_digest(**context):
    """Send weekly patent cliff digest."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")
    calendar_events = ti.xcom_pull(key="calendar_events", task_ids="generate_calendar")

    pipeline = PatentIntelligencePipeline()
    pipeline.drugs = drugs or []
    pipeline.calendar_events = calendar_events or []

    result = pipeline.send_weekly_digest()
    return {"email_sent": result}


def export_reports(**context):
    """Export patent cliff reports."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "patent_intelligence", "src"))
    from pipeline import PatentIntelligencePipeline

    ti = context["ti"]
    drugs = ti.xcom_pull(key="drugs", task_ids="extract_orange_book_data")
    patents = ti.xcom_pull(key="enriched_patents", task_ids="enrich_patent_data")
    andas = ti.xcom_pull(key="andas", task_ids="extract_anda_data")

    pipeline = PatentIntelligencePipeline()
    pipeline.drugs = drugs or []
    pipeline.patents = patents or []
    pipeline.andas = andas or []

    output_dir = os.path.join(PROJECT_ROOT, "patent_intelligence", "output")
    files = pipeline.export_calendar(output_dir=output_dir)

    return {"exported_files": files}


def record_metrics(**context):
    """Record pipeline execution metrics."""
    ti = context["ti"]

    orange_book_stats = ti.xcom_pull(key="orange_book_stats", task_ids="extract_orange_book_data") or {}
    db_stats = ti.xcom_pull(key="db_stats", task_ids="load_to_database") or {}
    scored = ti.xcom_pull(key="scored_opportunities", task_ids="score_opportunities") or []
    high_confidence = ti.xcom_pull(key="high_confidence_opportunities", task_ids="score_opportunities") or []

    metrics = {
        "execution_date": str(context["execution_date"]),
        "run_id": context["run_id"],
        "drugs_extracted": orange_book_stats.get("drugs_extracted", 0),
        "patents_extracted": orange_book_stats.get("patents_extracted", 0),
        "opportunities_scored": len(scored),
        "high_confidence_count": len(high_confidence),
        "db_drugs_inserted": db_stats.get("drugs_inserted", 0),
        "db_patents_inserted": db_stats.get("patents_inserted", 0),
    }

    print(f"Pipeline metrics: {metrics}")
    return metrics


# Task definitions
with dag:
    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    # Extraction tasks
    with TaskGroup(group_id="extraction") as extraction:
        extract_orange_book = PythonOperator(
            task_id="extract_orange_book_data",
            python_callable=extract_orange_book_data,
            provide_context=True,
        )

    # Enrichment tasks (must run after extraction)
    with TaskGroup(group_id="enrichment") as enrichment:
        enrich_patents = PythonOperator(
            task_id="enrich_patent_data",
            python_callable=extract_uspto_data,
            provide_context=True,
        )

        extract_andas = PythonOperator(
            task_id="extract_anda_data",
            python_callable=extract_anda_data,
            provide_context=True,
        )

    # Calendar generation
    calendar_task = PythonOperator(
        task_id="generate_calendar",
        python_callable=generate_calendar,
        provide_context=True,
    )

    # Scoring
    scoring_task = PythonOperator(
        task_id="score_opportunities",
        python_callable=score_opportunities,
        provide_context=True,
    )

    # Database loading
    db_load_task = PythonOperator(
        task_id="load_to_database",
        python_callable=load_to_database,
        provide_context=True,
    )

    # Alerting and reporting
    with TaskGroup(group_id="output") as output:
        alerts_task = PythonOperator(
            task_id="send_alerts",
            python_callable=send_alerts,
            provide_context=True,
        )

        digest_task = PythonOperator(
            task_id="send_weekly_digest",
            python_callable=send_weekly_digest,
            provide_context=True,
        )

        export_task = PythonOperator(
            task_id="export_reports",
            python_callable=export_reports,
            provide_context=True,
        )

    # Metrics
    metrics_task = PythonOperator(
        task_id="record_metrics",
        python_callable=record_metrics,
        provide_context=True,
    )

    # Define task dependencies
    start >> extraction >> enrichment >> calendar_task >> scoring_task >> db_load_task >> output >> metrics_task >> end
