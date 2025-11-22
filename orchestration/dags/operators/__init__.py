"""
Custom Airflow Operators for Investment Signals Orchestration

This module provides specialized operators for:
- Data fetching with retry logic and rate limiting
- Signal detection and scoring
- Alert management with deduplication
- Health checks and monitoring
- Pipeline metrics collection
"""

from dags.operators.data_operators import (
    DataFetchOperator,
    SECEdgarOperator,
    ClinicalTrialsOperator,
)
from dags.operators.signal_operators import (
    SignalDetectionOperator,
    SignalScoringOperator,
    SignalAggregationOperator,
)
from dags.operators.alert_operators import (
    AlertOperator,
    SlackAlertOperator,
    EmailAlertOperator,
)
from dags.operators.health_operators import (
    HealthCheckOperator,
    PipelineHealthSensor,
)
from dags.operators.metrics_operators import (
    MetricsRecordOperator,
    SLAMonitorOperator,
)

__all__ = [
    # Data operators
    "DataFetchOperator",
    "SECEdgarOperator",
    "ClinicalTrialsOperator",
    # Signal operators
    "SignalDetectionOperator",
    "SignalScoringOperator",
    "SignalAggregationOperator",
    # Alert operators
    "AlertOperator",
    "SlackAlertOperator",
    "EmailAlertOperator",
    # Health operators
    "HealthCheckOperator",
    "PipelineHealthSensor",
    # Metrics operators
    "MetricsRecordOperator",
    "SLAMonitorOperator",
]
