"""
Tests for Real-time FDA/PACER Monitoring module.
"""

import pytest
from datetime import datetime, timedelta
import tempfile
import json
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitors.realtime_monitor import (
    RealtimeMonitor,
    FDAMonitor,
    PACERMonitor,
    SlackNotifier,
    MonitorEvent,
    BudgetTracker,
    DEFAULT_WATCHED_DRUGS,
    DEFAULT_WATCHED_COMPANIES,
)


class TestMonitorEvent:
    """Tests for MonitorEvent class."""

    def test_create_event(self):
        """Test event creation."""
        event = MonitorEvent(
            event_type="ANDA_APPROVAL",
            drug_name="TestDrug",
            company="TestCompany",
            description="Generic approved",
            source="FDA",
            source_url="https://example.com",
            severity="HIGH",
        )

        assert event.event_type == "ANDA_APPROVAL"
        assert event.drug_name == "TestDrug"
        assert event.severity == "HIGH"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = MonitorEvent(
            event_type="LITIGATION_UPDATE",
            drug_name="TestDrug",
            company="TestCompany",
            description="Motion filed",
            source="PACER",
            source_url=None,
            severity="MEDIUM",
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert "event_type" in event_dict
        assert "detected_at" in event_dict

    def test_event_to_slack_message(self):
        """Test Slack message formatting."""
        event = MonitorEvent(
            event_type="ANDA_APPROVAL",
            drug_name="TestDrug",
            company="TestCompany",
            description="Generic drug approved",
            source="FDA",
            source_url="https://example.com",
            severity="HIGH",
        )

        message = event.to_slack_message()

        assert "attachments" in message
        assert len(message["attachments"]) > 0


class TestBudgetTracker:
    """Tests for BudgetTracker class."""

    def test_init_default(self):
        """Test default initialization."""
        tracker = BudgetTracker(monthly_budget=30.0)
        assert tracker.monthly_budget == 30.0

    def test_can_query_within_budget(self):
        """Test can_query when within budget."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker = BudgetTracker(monthly_budget=30.0, state_file=f.name)
            assert tracker.can_query(estimated_pages=10)
            os.unlink(f.name)

    def test_can_query_exceeds_budget(self):
        """Test can_query when exceeding budget."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker = BudgetTracker(monthly_budget=1.0, state_file=f.name)

            # Record queries until budget exceeded
            tracker.record_query(pages=50, description="Test query")

            assert not tracker.can_query(estimated_pages=100)
            os.unlink(f.name)

    def test_record_query(self):
        """Test query recording."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker = BudgetTracker(monthly_budget=30.0, state_file=f.name)

            cost = tracker.record_query(pages=10, description="Test")

            assert cost == 1.0  # 10 pages * $0.10
            assert tracker.state["total_spent"] == 1.0
            assert tracker.state["total_pages"] == 10

            os.unlink(f.name)

    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker = BudgetTracker(monthly_budget=30.0, state_file=f.name)
            tracker.record_query(pages=100, description="Test")

            remaining = tracker.get_remaining_budget()
            assert remaining == 20.0  # 30 - 10

            os.unlink(f.name)

    def test_get_status(self):
        """Test status retrieval."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker = BudgetTracker(monthly_budget=30.0, state_file=f.name)
            tracker.record_query(pages=50, description="Test")

            status = tracker.get_status()

            assert "budget" in status
            assert "spent" in status
            assert "remaining" in status
            assert "percent_used" in status

            os.unlink(f.name)


class TestFDAMonitor:
    """Tests for FDAMonitor class."""

    def test_init(self):
        """Test initialization."""
        monitor = FDAMonitor(
            watched_drugs=["humira", "eliquis"],
            watched_companies=["abbvie", "pfizer"],
        )

        assert len(monitor.watched_drugs) == 2
        assert len(monitor.watched_companies) == 2

    def test_is_relevant_drug_match(self):
        """Test relevance detection for drugs."""
        monitor = FDAMonitor(
            watched_drugs=["humira"],
            watched_companies=[],
        )

        assert monitor._is_relevant("FDA approves generic Humira")
        assert not monitor._is_relevant("FDA approves unrelated drug")

    def test_is_relevant_company_match(self):
        """Test relevance detection for companies."""
        monitor = FDAMonitor(
            watched_drugs=[],
            watched_companies=["abbvie"],
        )

        assert monitor._is_relevant("AbbVie announces new drug")
        assert not monitor._is_relevant("Other company news")

    def test_is_relevant_anda_keyword(self):
        """Test relevance detection for ANDA keywords."""
        monitor = FDAMonitor(
            watched_drugs=[],
            watched_companies=[],
        )

        assert monitor._is_relevant("FDA approves ANDA for generic drug")
        assert monitor._is_relevant("New biosimilar application filed")

    def test_classify_entry_anda_approval(self):
        """Test entry classification for ANDA approval."""
        monitor = FDAMonitor()

        event_type, severity = monitor._classify_entry(
            "Generic Drug ANDA Approved",
            "FDA approves abbreviated new drug application"
        )

        assert event_type == "ANDA_APPROVAL"
        assert severity == "HIGH"

    def test_classify_entry_safety(self):
        """Test entry classification for safety alerts."""
        monitor = FDAMonitor()

        event_type, severity = monitor._classify_entry(
            "Drug Safety Warning",
            "FDA issues safety warning for medication"
        )

        assert event_type == "SAFETY_ALERT"
        assert severity == "HIGH"


class TestPACERMonitor:
    """Tests for PACERMonitor class."""

    def test_init(self):
        """Test initialization."""
        monitor = PACERMonitor(
            watched_cases=["1:23-cv-01234"],
            watched_patents=["6090382"],
        )

        assert len(monitor.watched_cases) == 1
        assert len(monitor.watched_patents) == 1

    def test_check_cases_mock(self):
        """Test case checking with mock data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            budget = BudgetTracker(monthly_budget=30.0, state_file=f.name)
            monitor = PACERMonitor(budget_tracker=budget)

            events = monitor.check_cases()

            # Should return events from mock data
            assert isinstance(events, list)

            os.unlink(f.name)

    def test_budget_prevents_query(self):
        """Test that budget prevents queries when exceeded."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            budget = BudgetTracker(monthly_budget=0.01, state_file=f.name)
            monitor = PACERMonitor(budget_tracker=budget)

            events = monitor.check_cases()

            # Should return empty due to budget
            assert events == []

            os.unlink(f.name)

    def test_classify_filing_severity(self):
        """Test filing severity classification."""
        monitor = PACERMonitor()

        assert monitor._classify_filing_severity("ORDER") == "HIGH"
        assert monitor._classify_filing_severity("JUDGMENT") == "HIGH"
        assert monitor._classify_filing_severity("MOTION") == "MEDIUM"
        assert monitor._classify_filing_severity("NOTICE") == "LOW"

    def test_get_budget_status(self):
        """Test budget status retrieval."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            budget = BudgetTracker(monthly_budget=30.0, state_file=f.name)
            monitor = PACERMonitor(budget_tracker=budget)

            status = monitor.get_budget_status()

            assert "budget" in status
            assert status["budget"] == 30.0

            os.unlink(f.name)


class TestSlackNotifier:
    """Tests for SlackNotifier class."""

    def test_init_no_config(self):
        """Test initialization without configuration."""
        notifier = SlackNotifier()

        # Should not fail without config
        assert notifier is not None

    def test_send_event_no_config(self):
        """Test sending event without configuration."""
        notifier = SlackNotifier()

        event = MonitorEvent(
            event_type="TEST",
            drug_name="TestDrug",
            company="TestCompany",
            description="Test event",
            source="TEST",
            source_url=None,
            severity="LOW",
        )

        # Should return False when no config
        result = notifier.send_event(event)
        assert result is False

    def test_send_summary_empty(self):
        """Test sending empty summary."""
        notifier = SlackNotifier()

        # Empty events should return True
        result = notifier.send_summary([])
        assert result is True


class TestRealtimeMonitor:
    """Tests for RealtimeMonitor class."""

    def test_init(self):
        """Test initialization."""
        monitor = RealtimeMonitor(
            watched_drugs=["humira"],
            watched_companies=["abbvie"],
            pacer_budget=30.0,
        )

        assert monitor is not None
        assert monitor.fda_monitor is not None
        assert monitor.pacer_monitor is not None

    def test_get_status(self):
        """Test status retrieval."""
        monitor = RealtimeMonitor(
            watched_drugs=["humira"],
            pacer_budget=30.0,
        )

        status = monitor.get_status()

        assert "pacer_budget" in status
        assert "fda_feeds" in status
        assert "slack_configured" in status

    def test_add_handler(self):
        """Test adding custom event handler."""
        monitor = RealtimeMonitor()

        events_received = []

        def handler(event):
            events_received.append(event)

        monitor.add_handler(handler)

        assert len(monitor.event_handlers) == 1


class TestDefaultWatchedItems:
    """Tests for default watched items."""

    def test_default_drugs_exist(self):
        """Test that default drugs are defined."""
        assert len(DEFAULT_WATCHED_DRUGS) > 0
        assert "humira" in DEFAULT_WATCHED_DRUGS

    def test_default_companies_exist(self):
        """Test that default companies are defined."""
        assert len(DEFAULT_WATCHED_COMPANIES) > 0
        assert "abbvie" in DEFAULT_WATCHED_COMPANIES
