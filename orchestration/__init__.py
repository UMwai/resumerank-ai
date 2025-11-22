"""
Investment Signals Orchestration System

Automated scheduling and orchestration for investment intelligence pipelines:
- Clinical Trial Signal Detection
- Patent/IP Intelligence
- Insider Activity + Hiring Signals

This package provides:
- Apache Airflow DAGs for pipeline orchestration
- Multi-channel alerting (Email, Slack, SMS)
- Health monitoring and metrics collection
- Docker-based local deployment
- AWS/GCP cloud deployment configurations

Quick Start:
    # Run setup script for local deployment
    ./scripts/setup.sh

    # Access Airflow UI
    open http://localhost:8080

Configuration:
    Copy config/env.example to config/.env and configure your settings.

Usage:
    # Import alert manager
    from orchestration.alerts import AlertManager, get_alert_manager

    # Import health checks
    from orchestration.monitoring import run_all_health_checks

    # Import metrics
    from orchestration.monitoring import record_pipeline_execution
"""

__version__ = "1.0.0"
__author__ = "Investment Signals Team"

# Convenience imports
from .config import get_config, OrchestrationConfig
from .alerts import AlertManager, get_alert_manager
from .monitoring import (
    run_all_health_checks,
    check_clinical_trial_health,
    check_patent_ip_health,
    check_insider_hiring_health,
    record_pipeline_execution,
    record_signal,
    get_prometheus_metrics,
)

__all__ = [
    # Config
    "get_config",
    "OrchestrationConfig",
    # Alerts
    "AlertManager",
    "get_alert_manager",
    # Monitoring
    "run_all_health_checks",
    "check_clinical_trial_health",
    "check_patent_ip_health",
    "check_insider_hiring_health",
    "record_pipeline_execution",
    "record_signal",
    "get_prometheus_metrics",
]
