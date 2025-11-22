"""
Metrics Collection System

Collects and exposes metrics for monitoring:
- Pipeline execution metrics
- Data quality metrics
- Cost tracking
- Resource utilization

Supports Prometheus exposition format.
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A metric value with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsRegistry:
    """
    Thread-safe metrics registry.

    Stores metrics and provides Prometheus-compatible exposition.
    """

    def __init__(self):
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._lock = Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
        """
        with self._lock:
            metric = MetricValue(
                name=name,
                value=value,
                labels=labels or {},
                metric_type="gauge",
            )
            # Replace existing metric with same labels
            self._metrics[name] = [
                m for m in self._metrics[name]
                if m.labels != (labels or {})
            ]
            self._metrics[name].append(metric)

    def counter(self, name: str, value: float = 1, labels: Dict[str, str] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels
        """
        with self._lock:
            label_key = str(sorted((labels or {}).items()))
            key = f"{name}:{label_key}"
            self._counters[key] += value

            metric = MetricValue(
                name=name,
                value=self._counters[key],
                labels=labels or {},
                metric_type="counter",
            )
            self._metrics[name] = [
                m for m in self._metrics[name]
                if m.labels != (labels or {})
            ]
            self._metrics[name].append(metric)

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        with self._lock:
            label_key = str(sorted((labels or {}).items()))
            key = f"{name}:{label_key}"
            self._histograms[key].append(value)

            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus exposition format.

        Returns:
            String with metrics in Prometheus format
        """
        lines = []

        with self._lock:
            for name, metrics in self._metrics.items():
                # Add help and type comments
                lines.append(f"# HELP {name} {name}")
                lines.append(f"# TYPE {name} {metrics[0].metric_type if metrics else 'gauge'}")

                for metric in metrics:
                    # Format labels
                    if metric.labels:
                        label_str = ",".join([
                            f'{k}="{v}"' for k, v in metric.labels.items()
                        ])
                        lines.append(f"{name}{{{label_str}}} {metric.value}")
                    else:
                        lines.append(f"{name} {metric.value}")

            # Add histogram summaries
            for key, values in self._histograms.items():
                name = key.split(":")[0]
                if values:
                    lines.append(f"# HELP {name}_sum Sum of {name}")
                    lines.append(f"# TYPE {name}_sum gauge")
                    lines.append(f"{name}_sum {sum(values)}")

                    lines.append(f"# HELP {name}_count Count of {name}")
                    lines.append(f"# TYPE {name}_count counter")
                    lines.append(f"{name}_count {len(values)}")

                    lines.append(f"# HELP {name}_avg Average of {name}")
                    lines.append(f"# TYPE {name}_avg gauge")
                    lines.append(f"{name}_avg {sum(values) / len(values)}")

        return "\n".join(lines)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get metrics as a dictionary.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            result = {}
            for name, metrics in self._metrics.items():
                result[name] = [
                    {
                        "value": m.value,
                        "labels": m.labels,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in metrics
                ]
            return result


# Global registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


class PipelineMetrics:
    """
    Pipeline-specific metrics collection.

    Records execution metrics for investment signal pipelines.
    """

    def __init__(self, pipeline_name: str, registry: MetricsRegistry = None):
        self.pipeline_name = pipeline_name
        self.registry = registry or get_registry()

    def record_execution(
        self,
        status: str,
        duration_seconds: float,
        records_processed: int = 0,
        signals_found: int = 0,
        errors: int = 0,
    ) -> None:
        """
        Record pipeline execution metrics.

        Args:
            status: Execution status (success, failure, partial)
            duration_seconds: Execution duration in seconds
            records_processed: Number of records processed
            signals_found: Number of signals found
            errors: Number of errors encountered
        """
        labels = {"pipeline": self.pipeline_name}

        self.registry.counter(
            "pipeline_executions_total",
            labels={**labels, "status": status}
        )

        self.registry.histogram(
            "pipeline_duration_seconds",
            duration_seconds,
            labels=labels
        )

        self.registry.gauge(
            "pipeline_last_execution_timestamp",
            time.time(),
            labels=labels
        )

        self.registry.gauge(
            "pipeline_records_processed",
            records_processed,
            labels=labels
        )

        self.registry.gauge(
            "pipeline_signals_found",
            signals_found,
            labels=labels
        )

        if errors > 0:
            self.registry.counter(
                "pipeline_errors_total",
                errors,
                labels=labels
            )

    def record_signal(
        self,
        signal_type: str,
        ticker: str,
        score: float,
        recommendation: str,
    ) -> None:
        """
        Record a detected investment signal.

        Args:
            signal_type: Type of signal (clinical_trial, form4, etc.)
            ticker: Stock ticker
            score: Signal score
            recommendation: Investment recommendation
        """
        labels = {
            "pipeline": self.pipeline_name,
            "signal_type": signal_type,
        }

        self.registry.counter(
            "signals_detected_total",
            labels=labels
        )

        self.registry.histogram(
            "signal_score",
            score,
            labels=labels
        )

        self.registry.counter(
            "signals_by_recommendation_total",
            labels={**labels, "recommendation": recommendation}
        )

    def record_alert_sent(self, channel: str, priority: str) -> None:
        """
        Record an alert being sent.

        Args:
            channel: Alert channel (email, slack, sms)
            priority: Alert priority
        """
        self.registry.counter(
            "alerts_sent_total",
            labels={
                "pipeline": self.pipeline_name,
                "channel": channel,
                "priority": priority,
            }
        )

    def record_api_call(
        self,
        api_name: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """
        Record an external API call.

        Args:
            api_name: Name of the API
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        labels = {
            "pipeline": self.pipeline_name,
            "api": api_name,
        }

        self.registry.counter(
            "api_calls_total",
            labels={**labels, "status": str(status_code)}
        )

        self.registry.histogram(
            "api_call_duration_ms",
            duration_ms,
            labels=labels
        )


class CostTracker:
    """
    Track infrastructure and API costs.

    Provides cost estimation and alerting for budget management.
    """

    # Cost estimates (per unit)
    COST_ESTIMATES = {
        # Cloud resources (hourly)
        "ec2_t3_medium": 0.0416,
        "ec2_t3_large": 0.0832,
        "rds_db_t3_medium": 0.068,
        "fargate_vcpu_hour": 0.04048,
        "fargate_gb_hour": 0.004445,

        # API calls (per call)
        "sec_edgar_call": 0.0,  # Free
        "clinicaltrials_gov_call": 0.0,  # Free
        "linkedin_api_call": 0.0,  # Varies by plan

        # Storage (per GB/month)
        "s3_storage_gb": 0.023,
        "rds_storage_gb": 0.115,

        # Data transfer (per GB)
        "data_transfer_out_gb": 0.09,
    }

    def __init__(self, registry: MetricsRegistry = None):
        self.registry = registry or get_registry()
        self._daily_costs: Dict[str, float] = defaultdict(float)
        self._monthly_costs: Dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def record_cost(
        self,
        resource_type: str,
        quantity: float,
        unit_cost: float = None,
    ) -> float:
        """
        Record a cost event.

        Args:
            resource_type: Type of resource
            quantity: Quantity used
            unit_cost: Cost per unit (uses estimate if not provided)

        Returns:
            Calculated cost
        """
        if unit_cost is None:
            unit_cost = self.COST_ESTIMATES.get(resource_type, 0)

        cost = quantity * unit_cost

        with self._lock:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            month = datetime.utcnow().strftime("%Y-%m")

            self._daily_costs[f"{today}:{resource_type}"] += cost
            self._monthly_costs[f"{month}:{resource_type}"] += cost

        # Record metrics
        self.registry.counter(
            "cost_dollars_total",
            cost,
            labels={"resource_type": resource_type}
        )

        return cost

    def get_daily_cost(self, date: str = None) -> float:
        """Get total cost for a day."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        with self._lock:
            total = sum(
                v for k, v in self._daily_costs.items()
                if k.startswith(date)
            )
        return total

    def get_monthly_cost(self, month: str = None) -> float:
        """Get total cost for a month."""
        if month is None:
            month = datetime.utcnow().strftime("%Y-%m")

        with self._lock:
            total = sum(
                v for k, v in self._monthly_costs.items()
                if k.startswith(month)
            )
        return total

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")

        with self._lock:
            daily_breakdown = {}
            for k, v in self._daily_costs.items():
                if k.startswith(today):
                    resource = k.split(":", 1)[1]
                    daily_breakdown[resource] = v

            monthly_breakdown = {}
            for k, v in self._monthly_costs.items():
                if k.startswith(month):
                    resource = k.split(":", 1)[1]
                    monthly_breakdown[resource] = monthly_breakdown.get(resource, 0) + v

        return {
            "daily": {
                "date": today,
                "total": sum(daily_breakdown.values()),
                "breakdown": daily_breakdown,
            },
            "monthly": {
                "month": month,
                "total": sum(monthly_breakdown.values()),
                "breakdown": monthly_breakdown,
            },
        }


# Convenience functions

def record_pipeline_execution(
    pipeline_name: str,
    status: str,
    duration_seconds: float,
    **kwargs,
) -> None:
    """Record pipeline execution metrics."""
    metrics = PipelineMetrics(pipeline_name)
    metrics.record_execution(status, duration_seconds, **kwargs)


def record_signal(
    pipeline_name: str,
    signal_type: str,
    ticker: str,
    score: float,
    recommendation: str,
) -> None:
    """Record a detected signal."""
    metrics = PipelineMetrics(pipeline_name)
    metrics.record_signal(signal_type, ticker, score, recommendation)


def record_cost(resource_type: str, quantity: float, unit_cost: float = None) -> float:
    """Record a cost event."""
    tracker = CostTracker()
    return tracker.record_cost(resource_type, quantity, unit_cost)


def get_prometheus_metrics() -> str:
    """Get metrics in Prometheus format."""
    return get_registry().get_prometheus_metrics()


if __name__ == "__main__":
    # Example usage
    import json

    # Record some test metrics
    record_pipeline_execution(
        "clinical_trial_signals",
        "success",
        120.5,
        records_processed=100,
        signals_found=5,
    )

    record_signal(
        "clinical_trial_signals",
        "phase_advancement",
        "MRNA",
        8.5,
        "STRONG_BUY",
    )

    record_cost("ec2_t3_medium", 24)  # 24 hours
    record_cost("s3_storage_gb", 10)  # 10 GB

    # Print metrics
    print("=== Prometheus Format ===")
    print(get_prometheus_metrics())

    print("\n=== JSON Format ===")
    print(json.dumps(get_registry().get_metrics_dict(), indent=2))

    print("\n=== Cost Breakdown ===")
    tracker = CostTracker()
    print(json.dumps(tracker.get_cost_breakdown(), indent=2))
