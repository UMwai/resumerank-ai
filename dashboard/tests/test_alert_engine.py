"""
Tests for Alert Engine Module

Tests cover:
- AlertRule dataclass
- Alert dataclass
- AlertEngine rule management
- Alert delivery channels
- Alert triggering and history
"""

import os
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.alert_engine import (
    AlertChannel,
    AlertPriority,
    AlertStatus,
    AlertRule,
    Alert,
    AlertEngine,
    DashboardChannel,
    EmailChannel,
    SlackChannel,
    SMSChannel,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def engine(temp_db):
    """Create an AlertEngine with temporary database."""
    return AlertEngine(db_path=temp_db)


class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_rule_creation(self):
        """Test creating an AlertRule."""
        rule = AlertRule(
            name="High Score Alert",
            description="Alert when score > 0.8",
            min_score=0.8,
            min_confidence=0.7,
            channels="dashboard,email",
        )

        assert rule.name == "High Score Alert"
        assert rule.min_score == 0.8
        assert rule.min_confidence == 0.7
        assert rule.enabled is True

    def test_rule_to_dict(self):
        """Test AlertRule to_dict conversion."""
        rule = AlertRule(
            id=1,
            name="Test Rule",
            min_score=0.7,
            channels="slack",
        )

        data = rule.to_dict()

        assert data["id"] == 1
        assert data["name"] == "Test Rule"
        assert data["min_score"] == 0.7
        assert data["channels"] == "slack"

    def test_rule_from_dict(self):
        """Test AlertRule from_dict creation."""
        data = {
            "name": "Imported Rule",
            "min_score": 0.85,
            "ticker": "MRNA",
            "channels": "email,sms",
            "priority": "high",
        }

        rule = AlertRule.from_dict(data)

        assert rule.name == "Imported Rule"
        assert rule.min_score == 0.85
        assert rule.ticker == "MRNA"
        assert rule.priority == "high"

    def test_get_channels_list(self):
        """Test getting channels as list."""
        rule = AlertRule(channels="dashboard,email,slack")
        channels = rule.get_channels_list()

        assert channels == ["dashboard", "email", "slack"]


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an Alert."""
        alert = Alert(
            ticker="MRNA",
            company_name="Moderna Inc.",
            source="clinical_trial",
            signal_type="bullish",
            score=0.85,
            confidence=0.90,
            title="Phase 3 Results",
            message="Positive results announced",
        )

        assert alert.ticker == "MRNA"
        assert alert.score == 0.85
        assert alert.status == "active"

    def test_alert_to_dict(self):
        """Test Alert to_dict conversion."""
        alert = Alert(
            id=1,
            ticker="PFE",
            score=0.75,
            confidence=0.80,
            title="Test Alert",
        )

        data = alert.to_dict()

        assert data["id"] == 1
        assert data["ticker"] == "PFE"
        assert data["score"] == 0.75


class TestDeliveryChannels:
    """Test delivery channel implementations."""

    def test_dashboard_channel(self):
        """Test DashboardChannel."""
        callback = MagicMock()
        channel = DashboardChannel(callback=callback)

        alert = Alert(ticker="MRNA", title="Test Alert")
        success, error = channel.send(alert)

        assert success is True
        assert error is None
        assert alert in channel.pending_alerts
        callback.assert_called_once_with(alert)

    def test_dashboard_channel_get_pending(self):
        """Test getting and clearing pending alerts."""
        channel = DashboardChannel()

        alert1 = Alert(ticker="MRNA", title="Alert 1")
        alert2 = Alert(ticker="PFE", title="Alert 2")

        channel.send(alert1)
        channel.send(alert2)

        # Get and clear
        pending = channel.get_pending(clear=True)
        assert len(pending) == 2

        # Should be empty now
        pending = channel.get_pending()
        assert len(pending) == 0

    def test_email_channel_no_recipients(self):
        """Test EmailChannel with no recipients."""
        channel = EmailChannel()
        alert = Alert(ticker="MRNA", title="Test")

        success, error = channel.send(alert)

        assert success is False
        assert "No recipients" in error

    def test_email_channel_no_credentials(self):
        """Test EmailChannel with no credentials."""
        channel = EmailChannel(to_addrs=["test@example.com"])
        alert = Alert(ticker="MRNA", title="Test")

        success, error = channel.send(alert)

        assert success is False
        assert "credentials" in error.lower()

    def test_slack_channel_no_webhook(self):
        """Test SlackChannel with no webhook."""
        channel = SlackChannel()
        alert = Alert(ticker="MRNA", title="Test")

        success, error = channel.send(alert)

        assert success is False
        assert "webhook" in error.lower()

    def test_sms_channel_no_numbers(self):
        """Test SMSChannel with no phone numbers."""
        channel = SMSChannel()
        alert = Alert(ticker="MRNA", title="Test")

        success, error = channel.send(alert)

        assert success is False
        assert "No phone numbers" in error


class TestAlertEngineRules:
    """Test AlertEngine rule management."""

    def test_default_rule_created(self, engine):
        """Test that a default rule is created."""
        rules = engine.get_all_rules()
        assert len(rules) >= 1

    def test_create_rule(self, engine):
        """Test creating a new rule."""
        rule = AlertRule(
            name="Custom Rule",
            min_score=0.75,
            ticker="MRNA",
            channels="dashboard,email",
        )

        created = engine.create_rule(rule)

        assert created.id is not None
        assert created.name == "Custom Rule"
        assert created.min_score == 0.75

    def test_get_rule(self, engine):
        """Test getting a rule by ID."""
        rule = engine.create_rule(AlertRule(name="Test Rule"))
        retrieved = engine.get_rule(rule.id)

        assert retrieved is not None
        assert retrieved.name == "Test Rule"

    def test_get_all_rules(self, engine):
        """Test getting all rules."""
        engine.create_rule(AlertRule(name="Rule 1"))
        engine.create_rule(AlertRule(name="Rule 2"))

        rules = engine.get_all_rules()
        assert len(rules) >= 3  # Default + 2 created

    def test_get_enabled_rules_only(self, engine):
        """Test getting only enabled rules."""
        engine.create_rule(AlertRule(name="Enabled", enabled=True))
        disabled = engine.create_rule(AlertRule(name="Disabled", enabled=False))

        # Disable it
        disabled.enabled = False
        engine.update_rule(disabled)

        enabled_rules = engine.get_all_rules(enabled_only=True)
        names = [r.name for r in enabled_rules]

        assert "Enabled" in names

    def test_update_rule(self, engine):
        """Test updating a rule."""
        rule = engine.create_rule(AlertRule(name="Original"))

        rule.name = "Updated"
        rule.min_score = 0.9
        success = engine.update_rule(rule)

        assert success is True

        updated = engine.get_rule(rule.id)
        assert updated.name == "Updated"
        assert updated.min_score == 0.9

    def test_delete_rule(self, engine):
        """Test deleting a rule."""
        rule = engine.create_rule(AlertRule(name="To Delete"))
        deleted = engine.delete_rule(rule.id)

        assert deleted is True
        assert engine.get_rule(rule.id) is None


class TestAlertEngineSignalMatching:
    """Test signal matching against rules."""

    def test_matches_score_threshold(self, engine):
        """Test matching score threshold."""
        rule = AlertRule(name="Score Test", min_score=0.8)

        signal_high = {"score": 0.85, "confidence": 0.7}
        signal_low = {"score": 0.5, "confidence": 0.7}

        assert engine._matches_rule(signal_high, rule) is True
        assert engine._matches_rule(signal_low, rule) is False

    def test_matches_confidence_threshold(self, engine):
        """Test matching confidence threshold."""
        rule = AlertRule(name="Confidence Test", min_confidence=0.8)

        signal_high = {"confidence": 0.9}
        signal_low = {"confidence": 0.5}

        assert engine._matches_rule(signal_high, rule) is True
        assert engine._matches_rule(signal_low, rule) is False

    def test_matches_ticker_filter(self, engine):
        """Test matching ticker filter."""
        rule = AlertRule(name="Ticker Test", ticker="MRNA")

        signal_match = {"ticker": "MRNA", "score": 0.5}
        signal_no_match = {"ticker": "PFE", "score": 0.5}

        assert engine._matches_rule(signal_match, rule) is True
        assert engine._matches_rule(signal_no_match, rule) is False

    def test_matches_source_filter(self, engine):
        """Test matching source filter."""
        rule = AlertRule(name="Source Test", source="clinical_trial")

        signal_match = {"source": "clinical_trial"}
        signal_no_match = {"source": "patent"}

        assert engine._matches_rule(signal_match, rule) is True
        assert engine._matches_rule(signal_no_match, rule) is False

    def test_matches_signal_type_filter(self, engine):
        """Test matching signal type filter."""
        rule = AlertRule(name="Type Test", signal_type="bullish")

        signal_match = {"signal_type": "bullish"}
        signal_no_match = {"signal_type": "bearish"}

        assert engine._matches_rule(signal_match, rule) is True
        assert engine._matches_rule(signal_no_match, rule) is False


class TestAlertEngineAlertManagement:
    """Test alert management operations."""

    def test_check_signal_triggers_alert(self, engine):
        """Test that check_signal triggers alerts."""
        engine.create_rule(AlertRule(
            name="High Score",
            min_score=0.8,
            min_confidence=0.7,
        ))

        signal = {
            "ticker": "MRNA",
            "company_name": "Moderna",
            "source": "clinical_trial",
            "signal_type": "bullish",
            "score": 0.85,
            "confidence": 0.9,
            "title": "Test Signal",
            "description": "Test description",
        }

        alerts = engine.check_signal(signal)

        assert len(alerts) >= 1
        assert any(a.ticker == "MRNA" for a in alerts)

    def test_get_alerts(self, engine):
        """Test getting alerts."""
        # Create an alert via signal
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        engine.check_signal({
            "ticker": "PFE",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        alerts = engine.get_alerts()
        assert len(alerts) >= 1

    def test_get_active_alerts(self, engine):
        """Test getting active alerts."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        engine.check_signal({
            "ticker": "GILD",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        active = engine.get_active_alerts()
        assert all(a.status == "active" for a in active)

    def test_acknowledge_alert(self, engine):
        """Test acknowledging an alert."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        alerts = engine.check_signal({
            "ticker": "ABBV",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        if alerts:
            success = engine.acknowledge_alert(alerts[0].id)
            assert success is True

            # Verify status changed
            updated_alerts = engine.get_alerts()
            alert = next((a for a in updated_alerts if a.id == alerts[0].id), None)
            assert alert is not None
            assert alert.status == "acknowledged"

    def test_snooze_alert(self, engine):
        """Test snoozing an alert."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        alerts = engine.check_signal({
            "ticker": "REGN",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        if alerts:
            success = engine.snooze_alert(alerts[0].id, minutes=30)
            assert success is True

    def test_dismiss_alert(self, engine):
        """Test dismissing an alert."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        alerts = engine.check_signal({
            "ticker": "BIIB",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        if alerts:
            success = engine.dismiss_alert(alerts[0].id)
            assert success is True

    def test_record_outcome(self, engine):
        """Test recording alert outcome."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))
        alerts = engine.check_signal({
            "ticker": "AMGN",
            "score": 0.8,
            "confidence": 0.8,
            "title": "Test",
        })

        if alerts:
            success = engine.record_outcome(
                alerts[0].id,
                outcome="positive",
                notes="Price increased 10%",
            )
            assert success is True


class TestAlertEnginePerformance:
    """Test performance tracking."""

    def test_get_performance_stats(self, engine):
        """Test getting performance statistics."""
        stats = engine.get_performance_stats()

        assert "total_alerts" in stats
        assert "outcomes" in stats
        assert "win_rate" in stats
        assert "by_source" in stats

    def test_performance_with_outcomes(self, engine):
        """Test performance stats with recorded outcomes."""
        engine.create_rule(AlertRule(name="Test", min_score=0.5))

        # Create alerts and record outcomes
        outcomes_recorded = 0
        for i, outcome in enumerate(["positive", "positive", "negative"]):
            alerts = engine.check_signal({
                "ticker": f"TEST{i}",
                "score": 0.8,
                "confidence": 0.8,
                "title": f"Test {i}",
            })
            if alerts:
                engine.record_outcome(alerts[0].id, outcome)
                outcomes_recorded += 1

        stats = engine.get_performance_stats()

        # Check that we recorded outcomes for the alerts we created
        assert stats["total_alerts"] >= outcomes_recorded
