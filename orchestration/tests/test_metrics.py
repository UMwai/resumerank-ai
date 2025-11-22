"""
Tests for the metrics collection system.

Tests:
- MetricsRegistry operations
- PipelineMetrics recording
- SLA compliance checks
- Data quality metrics
- API health tracking
- Prometheus format output
"""

import pytest
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.metrics import (
    MetricsRegistry,
    PipelineMetrics,
    CostTracker,
    SLAMetrics,
    DataQualityMetrics,
    APIHealthMetrics,
    HealthCheckMetrics,
    get_registry,
    record_pipeline_execution,
    record_signal,
    record_cost,
    record_sla_check,
    record_data_quality,
    record_api_health,
    record_health_check,
    get_prometheus_metrics,
    timed_metric,
    counted_metric,
)


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""

    def setup_method(self):
        """Create a fresh registry for each test."""
        self.registry = MetricsRegistry()

    def test_gauge_set_and_retrieve(self):
        """Test setting and retrieving gauge metrics."""
        self.registry.gauge("test_gauge", 42.0)
        metrics = self.registry.get_metrics_dict()

        assert "test_gauge" in metrics
        assert metrics["test_gauge"][0]["value"] == 42.0

    def test_gauge_with_labels(self):
        """Test gauge metrics with labels."""
        self.registry.gauge("test_gauge", 10.0, labels={"env": "prod"})
        self.registry.gauge("test_gauge", 20.0, labels={"env": "dev"})

        metrics = self.registry.get_metrics_dict()
        assert len(metrics["test_gauge"]) == 2

    def test_gauge_replace_same_labels(self):
        """Test that gauge replaces value for same labels."""
        self.registry.gauge("test_gauge", 10.0, labels={"env": "prod"})
        self.registry.gauge("test_gauge", 20.0, labels={"env": "prod"})

        metrics = self.registry.get_metrics_dict()
        assert len(metrics["test_gauge"]) == 1
        assert metrics["test_gauge"][0]["value"] == 20.0

    def test_counter_increment(self):
        """Test counter increments correctly."""
        self.registry.counter("test_counter")
        self.registry.counter("test_counter")
        self.registry.counter("test_counter", 5)

        metrics = self.registry.get_metrics_dict()
        assert metrics["test_counter"][0]["value"] == 7.0

    def test_counter_with_labels(self):
        """Test counter with different labels."""
        self.registry.counter("test_counter", labels={"status": "success"})
        self.registry.counter("test_counter", labels={"status": "failure"})
        self.registry.counter("test_counter", labels={"status": "success"})

        metrics = self.registry.get_metrics_dict()
        assert len(metrics["test_counter"]) == 2

    def test_histogram_recording(self):
        """Test histogram records values."""
        for i in range(10):
            self.registry.histogram("test_histogram", float(i))

        prometheus_output = self.registry.get_prometheus_metrics()
        assert "test_histogram_count 10" in prometheus_output
        assert "test_histogram_sum 45" in prometheus_output

    def test_prometheus_format_output(self):
        """Test Prometheus format output."""
        self.registry.gauge("test_metric", 100.0, labels={"pipeline": "test"})

        output = self.registry.get_prometheus_metrics()
        assert "# HELP test_metric" in output
        assert "# TYPE test_metric gauge" in output
        assert 'test_metric{pipeline="test"} 100.0' in output


class TestPipelineMetrics:
    """Tests for PipelineMetrics class."""

    def setup_method(self):
        """Create fresh metrics instance."""
        self.registry = MetricsRegistry()
        self.metrics = PipelineMetrics("test_pipeline", self.registry)

    def test_record_execution_success(self):
        """Test recording successful execution."""
        self.metrics.record_execution(
            status="success",
            duration_seconds=120.5,
            records_processed=100,
            signals_found=5,
        )

        metrics_dict = self.registry.get_metrics_dict()
        assert "pipeline_executions_total" in metrics_dict
        assert "pipeline_records_processed" in metrics_dict
        assert "pipeline_signals_found" in metrics_dict

    def test_record_execution_with_errors(self):
        """Test recording execution with errors."""
        self.metrics.record_execution(
            status="partial",
            duration_seconds=60.0,
            records_processed=50,
            signals_found=2,
            errors=3,
        )

        metrics_dict = self.registry.get_metrics_dict()
        assert "pipeline_errors_total" in metrics_dict

    def test_record_signal(self):
        """Test recording a detected signal."""
        self.metrics.record_signal(
            signal_type="clinical_trial",
            ticker="MRNA",
            score=8.5,
            recommendation="STRONG_BUY",
        )

        metrics_dict = self.registry.get_metrics_dict()
        assert "signals_detected_total" in metrics_dict
        assert "signals_by_recommendation_total" in metrics_dict

    def test_record_alert_sent(self):
        """Test recording alert sent."""
        self.metrics.record_alert_sent(channel="email", priority="high")

        metrics_dict = self.registry.get_metrics_dict()
        assert "alerts_sent_total" in metrics_dict

    def test_record_api_call(self):
        """Test recording API call."""
        self.metrics.record_api_call(
            api_name="sec_edgar",
            status_code=200,
            duration_ms=150.5,
        )

        metrics_dict = self.registry.get_metrics_dict()
        assert "api_calls_total" in metrics_dict


class TestCostTracker:
    """Tests for CostTracker class."""

    def setup_method(self):
        """Create fresh cost tracker."""
        self.registry = MetricsRegistry()
        self.tracker = CostTracker(self.registry)

    def test_record_cost_with_estimate(self):
        """Test recording cost with built-in estimate."""
        cost = self.tracker.record_cost("ec2_t3_medium", 24)  # 24 hours

        assert cost == pytest.approx(24 * 0.0416, rel=0.01)

    def test_record_cost_with_custom_rate(self):
        """Test recording cost with custom unit cost."""
        cost = self.tracker.record_cost("custom_resource", 10, unit_cost=0.50)

        assert cost == 5.0

    def test_get_daily_cost(self):
        """Test getting daily cost."""
        self.tracker.record_cost("ec2_t3_medium", 24)
        self.tracker.record_cost("s3_storage_gb", 100)

        daily_cost = self.tracker.get_daily_cost()
        assert daily_cost > 0

    def test_get_cost_breakdown(self):
        """Test getting cost breakdown."""
        self.tracker.record_cost("ec2_t3_medium", 24)
        self.tracker.record_cost("s3_storage_gb", 100)

        breakdown = self.tracker.get_cost_breakdown()

        assert "daily" in breakdown
        assert "monthly" in breakdown
        assert breakdown["daily"]["total"] > 0


class TestSLAMetrics:
    """Tests for SLAMetrics class."""

    def setup_method(self):
        """Create fresh SLA metrics instance."""
        self.registry = MetricsRegistry()
        self.sla = SLAMetrics(self.registry)
        self.sla.set_sla_target(
            "test_pipeline",
            execution_sla_seconds=3600,
            freshness_sla_hours=24,
            success_rate_target=0.95,
        )

    def test_sla_compliant(self):
        """Test SLA check when all SLAs are met."""
        result = self.sla.record_sla_check(
            pipeline_name="test_pipeline",
            execution_seconds=1800,  # 30 minutes - under 1 hour SLA
            data_age_hours=12,  # Under 24 hour freshness
            success_rate=0.98,  # Above 95% target
        )

        assert result["execution_compliant"] is True
        assert result["freshness_compliant"] is True
        assert result["success_rate_compliant"] is True

    def test_sla_violation_execution_time(self):
        """Test SLA violation for execution time."""
        result = self.sla.record_sla_check(
            pipeline_name="test_pipeline",
            execution_seconds=5000,  # Over 1 hour SLA
            data_age_hours=12,
            success_rate=0.98,
        )

        assert result["execution_compliant"] is False

    def test_sla_violation_freshness(self):
        """Test SLA violation for data freshness."""
        result = self.sla.record_sla_check(
            pipeline_name="test_pipeline",
            execution_seconds=1800,
            data_age_hours=30,  # Over 24 hour freshness SLA
            success_rate=0.98,
        )

        assert result["freshness_compliant"] is False

    def test_sla_violation_success_rate(self):
        """Test SLA violation for success rate."""
        result = self.sla.record_sla_check(
            pipeline_name="test_pipeline",
            execution_seconds=1800,
            data_age_hours=12,
            success_rate=0.90,  # Below 95% target
        )

        assert result["success_rate_compliant"] is False


class TestDataQualityMetrics:
    """Tests for DataQualityMetrics class."""

    def setup_method(self):
        """Create fresh data quality metrics instance."""
        self.registry = MetricsRegistry()
        self.dq = DataQualityMetrics(self.registry)

    def test_perfect_quality(self):
        """Test perfect data quality score."""
        result = self.dq.record_data_quality(
            pipeline_name="test_pipeline",
            total_records=100,
            valid_records=100,
            null_fields=0,
            duplicate_records=0,
            validation_errors=0,
        )

        assert result["completeness"] == 1.0
        assert result["quality_score"] == pytest.approx(1.0, rel=0.01)

    def test_quality_with_issues(self):
        """Test data quality score with issues."""
        result = self.dq.record_data_quality(
            pipeline_name="test_pipeline",
            total_records=100,
            valid_records=90,
            null_fields=50,
            duplicate_records=5,
            validation_errors=3,
        )

        assert result["completeness"] == 0.9
        assert result["duplicate_rate"] == 0.05
        assert result["validation_error_rate"] == 0.03
        assert result["quality_score"] < 1.0


class TestAPIHealthMetrics:
    """Tests for APIHealthMetrics class."""

    def setup_method(self):
        """Create fresh API health metrics instance."""
        self.registry = MetricsRegistry()
        self.api = APIHealthMetrics(self.registry)

    def test_record_successful_calls(self):
        """Test recording successful API calls."""
        for _ in range(10):
            self.api.record_api_call(
                api_name="test_api",
                success=True,
                latency_ms=100.0,
                status_code=200,
            )

        stats = self.api.get_api_stats("test_api")
        assert stats["total_calls"] == 10
        assert stats["successes"] == 10
        assert stats["availability"] == 1.0

    def test_record_failed_calls(self):
        """Test recording failed API calls."""
        for _ in range(8):
            self.api.record_api_call("test_api", True, 100.0)
        for _ in range(2):
            self.api.record_api_call("test_api", False, 500.0, status_code=500)

        stats = self.api.get_api_stats("test_api")
        assert stats["availability"] == pytest.approx(0.8, rel=0.01)

    def test_latency_statistics(self):
        """Test latency statistics calculation."""
        latencies = [100, 150, 200, 250, 300]
        for lat in latencies:
            self.api.record_api_call("test_api", True, float(lat))

        stats = self.api.get_api_stats("test_api")
        assert stats["avg_latency_ms"] == pytest.approx(200.0, rel=0.01)
        assert stats["p50_latency_ms"] == 200.0


class TestHealthCheckMetrics:
    """Tests for HealthCheckMetrics class."""

    def setup_method(self):
        """Create fresh health check metrics instance."""
        self.registry = MetricsRegistry()
        self.health = HealthCheckMetrics(self.registry)

    def test_record_healthy_check(self):
        """Test recording healthy check."""
        self.health.record_health_check(
            component="database",
            healthy=True,
            duration_seconds=0.05,
        )

        metrics = self.registry.get_metrics_dict()
        assert "health_check_status" in metrics
        assert metrics["health_check_status"][0]["value"] == 1

    def test_record_unhealthy_check(self):
        """Test recording unhealthy check."""
        self.health.record_health_check(
            component="database",
            healthy=False,
            duration_seconds=5.0,
            details="Connection timeout",
        )

        metrics = self.registry.get_metrics_dict()
        assert metrics["health_check_status"][0]["value"] == 0
        assert "health_check_failures_total" in metrics


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_record_pipeline_execution(self):
        """Test convenience function for pipeline execution."""
        record_pipeline_execution(
            "test_pipeline",
            "success",
            60.0,
            records_processed=50,
            signals_found=3,
        )

        output = get_prometheus_metrics()
        assert "pipeline_executions_total" in output

    def test_record_signal_function(self):
        """Test convenience function for signal recording."""
        record_signal(
            "test_pipeline",
            "form4",
            "AAPL",
            7.5,
            "BUY",
        )

        output = get_prometheus_metrics()
        assert "signals_detected_total" in output


class TestDecorators:
    """Tests for metric decorators."""

    def setup_method(self):
        """Reset registry for clean tests."""
        # Note: Decorators use the global registry

    def test_timed_metric_decorator(self):
        """Test timed_metric decorator."""

        @timed_metric("test_function_duration")
        def slow_function():
            time.sleep(0.1)
            return "done"

        result = slow_function()
        assert result == "done"

        # Check metric was recorded
        output = get_prometheus_metrics()
        assert "test_function_duration" in output

    def test_counted_metric_decorator(self):
        """Test counted_metric decorator."""

        @counted_metric("test_function_calls")
        def simple_function():
            return 42

        result = simple_function()
        assert result == 42

        output = get_prometheus_metrics()
        assert "test_function_calls" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
