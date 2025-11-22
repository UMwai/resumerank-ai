"""
Health Check Service

Provides HTTP endpoints for health monitoring and Prometheus metrics.

Endpoints:
- /health - Overall health status
- /health/<pipeline> - Pipeline-specific health
- /metrics - Prometheus metrics
- /ready - Kubernetes readiness probe
- /live - Kubernetes liveness probe
"""

import json
import logging
import os
import time
from datetime import datetime
from flask import Flask, jsonify, Response
import psycopg2
import requests
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://signals:signals_password@postgres:5432/investment_signals"
)
AIRFLOW_URL = os.getenv("AIRFLOW_WEBSERVER_URL", "http://airflow-webserver:8080")

# Prometheus metrics
HEALTH_CHECK_DURATION = Histogram(
    "health_check_duration_seconds",
    "Time spent performing health checks",
    ["check_type"]
)
HEALTH_CHECK_STATUS = Gauge(
    "health_check_status",
    "Health check status (1=healthy, 0=unhealthy)",
    ["pipeline", "check_name"]
)
HEALTH_CHECK_TOTAL = Counter(
    "health_check_total",
    "Total number of health checks performed",
    ["pipeline", "status"]
)


def check_database():
    """Check database connectivity."""
    start = time.time()
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()

        duration = time.time() - start
        HEALTH_CHECK_DURATION.labels(check_type="database").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline="system", check_name="database").set(1)

        return {
            "name": "database",
            "healthy": True,
            "message": "Database connection successful",
            "duration_ms": round(duration * 1000, 2),
        }
    except Exception as e:
        duration = time.time() - start
        HEALTH_CHECK_DURATION.labels(check_type="database").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline="system", check_name="database").set(0)

        return {
            "name": "database",
            "healthy": False,
            "message": f"Database connection failed: {str(e)[:100]}",
            "duration_ms": round(duration * 1000, 2),
        }


def check_airflow():
    """Check Airflow health."""
    start = time.time()
    try:
        response = requests.get(f"{AIRFLOW_URL}/health", timeout=10)
        duration = time.time() - start

        if response.status_code == 200:
            health_data = response.json()
            scheduler_ok = health_data.get("scheduler", {}).get("status") == "healthy"
            db_ok = health_data.get("metadatabase", {}).get("status") == "healthy"
            is_healthy = scheduler_ok and db_ok

            HEALTH_CHECK_DURATION.labels(check_type="airflow").observe(duration)
            HEALTH_CHECK_STATUS.labels(pipeline="system", check_name="airflow").set(1 if is_healthy else 0)

            return {
                "name": "airflow",
                "healthy": is_healthy,
                "message": "Airflow healthy" if is_healthy else "Airflow has issues",
                "details": health_data,
                "duration_ms": round(duration * 1000, 2),
            }
        else:
            HEALTH_CHECK_STATUS.labels(pipeline="system", check_name="airflow").set(0)
            return {
                "name": "airflow",
                "healthy": False,
                "message": f"Airflow returned status {response.status_code}",
                "duration_ms": round(duration * 1000, 2),
            }
    except Exception as e:
        duration = time.time() - start
        HEALTH_CHECK_DURATION.labels(check_type="airflow").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline="system", check_name="airflow").set(0)

        return {
            "name": "airflow",
            "healthy": False,
            "message": f"Airflow check failed: {str(e)[:100]}",
            "duration_ms": round(duration * 1000, 2),
        }


def check_external_api(name, url):
    """Check external API availability."""
    start = time.time()
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Investment-Signals-HealthCheck/1.0"}
        )
        duration = time.time() - start

        is_healthy = response.status_code < 500

        HEALTH_CHECK_DURATION.labels(check_type=f"api_{name}").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline="system", check_name=f"api_{name}").set(1 if is_healthy else 0)

        return {
            "name": f"api_{name}",
            "healthy": is_healthy,
            "message": f"API responded with status {response.status_code}",
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        }
    except Exception as e:
        duration = time.time() - start
        HEALTH_CHECK_DURATION.labels(check_type=f"api_{name}").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline="system", check_name=f"api_{name}").set(0)

        return {
            "name": f"api_{name}",
            "healthy": False,
            "message": f"API check failed: {str(e)[:100]}",
            "duration_ms": round(duration * 1000, 2),
        }


def check_data_freshness(table, timestamp_col, threshold_hours):
    """Check data freshness in a table."""
    start = time.time()
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX({timestamp_col}) FROM {table}")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        duration = time.time() - start

        if not result or not result[0]:
            HEALTH_CHECK_STATUS.labels(pipeline=table, check_name="freshness").set(0)
            return {
                "name": f"freshness_{table}",
                "healthy": False,
                "message": f"No data in {table}",
                "duration_ms": round(duration * 1000, 2),
            }

        last_update = result[0]
        hours_old = (datetime.utcnow() - last_update).total_seconds() / 3600
        is_fresh = hours_old <= threshold_hours

        HEALTH_CHECK_DURATION.labels(check_type=f"freshness_{table}").observe(duration)
        HEALTH_CHECK_STATUS.labels(pipeline=table, check_name="freshness").set(1 if is_fresh else 0)

        return {
            "name": f"freshness_{table}",
            "healthy": is_fresh,
            "message": f"Data is {hours_old:.1f}h old (threshold: {threshold_hours}h)",
            "hours_old": round(hours_old, 1),
            "threshold_hours": threshold_hours,
            "duration_ms": round(duration * 1000, 2),
        }
    except Exception as e:
        duration = time.time() - start
        HEALTH_CHECK_STATUS.labels(pipeline=table, check_name="freshness").set(0)
        return {
            "name": f"freshness_{table}",
            "healthy": False,
            "message": f"Freshness check failed: {str(e)[:100]}",
            "duration_ms": round(duration * 1000, 2),
        }


@app.route("/health")
def health():
    """Overall health endpoint."""
    checks = [
        check_database(),
        check_airflow(),
    ]

    overall_healthy = all(c["healthy"] for c in checks)
    status_code = 200 if overall_healthy else 503

    HEALTH_CHECK_TOTAL.labels(pipeline="overall", status="healthy" if overall_healthy else "unhealthy").inc()

    return jsonify({
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }), status_code


@app.route("/health/clinical_trials")
def health_clinical_trials():
    """Clinical trials pipeline health."""
    checks = [
        check_database(),
        check_external_api("clinicaltrials_gov", "https://clinicaltrials.gov/api/v2/studies"),
        check_external_api("sec_edgar", "https://www.sec.gov/cgi-bin/browse-edgar"),
        check_data_freshness("clinical_trials.trials", "last_updated", 48),
    ]

    overall_healthy = all(c["healthy"] for c in checks)
    HEALTH_CHECK_TOTAL.labels(pipeline="clinical_trials", status="healthy" if overall_healthy else "unhealthy").inc()

    return jsonify({
        "pipeline": "clinical_trials",
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }), 200 if overall_healthy else 503


@app.route("/health/patent_ip")
def health_patent_ip():
    """Patent/IP pipeline health."""
    checks = [
        check_database(),
        check_external_api("fda_orange_book", "https://www.accessdata.fda.gov/scripts/cder/ob/"),
        check_data_freshness("patent_intelligence.drugs", "updated_at", 168),  # 7 days
    ]

    overall_healthy = all(c["healthy"] for c in checks)
    HEALTH_CHECK_TOTAL.labels(pipeline="patent_ip", status="healthy" if overall_healthy else "unhealthy").inc()

    return jsonify({
        "pipeline": "patent_ip",
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }), 200 if overall_healthy else 503


@app.route("/health/insider_hiring")
def health_insider_hiring():
    """Insider/hiring pipeline health."""
    checks = [
        check_database(),
        check_external_api("sec_edgar", "https://www.sec.gov/cgi-bin/browse-edgar"),
        check_data_freshness("insider_hiring.form4_filings", "filing_date", 24),
        check_data_freshness("insider_hiring.job_postings", "posted_date", 48),
    ]

    overall_healthy = all(c["healthy"] for c in checks)
    HEALTH_CHECK_TOTAL.labels(pipeline="insider_hiring", status="healthy" if overall_healthy else "unhealthy").inc()

    return jsonify({
        "pipeline": "insider_hiring",
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }), 200 if overall_healthy else 503


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/ready")
def ready():
    """Kubernetes readiness probe."""
    db_check = check_database()
    if db_check["healthy"]:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready", "reason": db_check["message"]}), 503


@app.route("/live")
def live():
    """Kubernetes liveness probe."""
    return jsonify({"status": "alive", "timestamp": datetime.utcnow().isoformat()}), 200


@app.route("/")
def index():
    """Root endpoint with service info."""
    return jsonify({
        "service": "Investment Signals Health Check",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Overall health status",
            "/health/clinical_trials": "Clinical trials pipeline health",
            "/health/patent_ip": "Patent/IP pipeline health",
            "/health/insider_hiring": "Insider/hiring pipeline health",
            "/metrics": "Prometheus metrics",
            "/ready": "Kubernetes readiness probe",
            "/live": "Kubernetes liveness probe",
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9091, debug=False)
