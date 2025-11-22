"""
Tests for the Real-time Slack Alerts module.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alerts.slack_alerts import (
    SlackAlerter,
    MockSlackAlerter,
    SlackAlert,
    AlertPriority,
    AlertType,
    get_alerter,
)


class TestSlackAlert:
    """Tests for SlackAlert dataclass."""

    def test_alert_creation(self):
        """Test creating a SlackAlert."""
        alert = SlackAlert(
            company="BioTech Corp",
            ticker="BTCH",
            trial_id="NCT12345678",
            signal_type=AlertType.STRONG_BUY,
            score=8.5,
            confidence=0.85,
            summary="Positive trial results",
            recommendation="STRONG_BUY",
            priority=AlertPriority.HIGH,
        )

        assert alert.company == "BioTech Corp"
        assert alert.ticker == "BTCH"
        assert alert.score == 8.5
        assert alert.priority == AlertPriority.HIGH
        assert isinstance(alert.timestamp, datetime)


class TestMockSlackAlerter:
    """Tests for MockSlackAlerter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.alerter = MockSlackAlerter(
            max_alerts_per_hour=5,
            min_score_threshold=8.0,
            min_confidence=0.7
        )

    def test_should_alert_high_score(self):
        """Test that high scores trigger alerts."""
        assert self.alerter.should_alert(8.5, 0.8) is True
        assert self.alerter.should_alert(9.0, 0.75) is True

    def test_should_alert_low_score(self):
        """Test that low scores (short signals) trigger alerts."""
        assert self.alerter.should_alert(2.0, 0.8) is True
        assert self.alerter.should_alert(1.5, 0.75) is True

    def test_should_not_alert_medium_score(self):
        """Test that medium scores don't trigger alerts."""
        assert self.alerter.should_alert(5.0, 0.8) is False
        assert self.alerter.should_alert(6.0, 0.8) is False
        assert self.alerter.should_alert(7.0, 0.8) is False

    def test_should_not_alert_low_confidence(self):
        """Test that low confidence doesn't trigger alerts."""
        assert self.alerter.should_alert(8.5, 0.5) is False
        assert self.alerter.should_alert(9.0, 0.6) is False

    def test_send_alert_records(self):
        """Test that mock alerter records alerts."""
        alert = SlackAlert(
            company="Test Co",
            ticker="TEST",
            trial_id="NCT00000001",
            signal_type=AlertType.BUY,
            score=8.5,
            confidence=0.85,
            summary="Test alert",
            recommendation="BUY",
            priority=AlertPriority.HIGH,
        )

        result = self.alerter.send_alert(alert)

        assert result is True
        assert len(self.alerter.sent_alerts) == 1
        assert self.alerter.sent_alerts[0].trial_id == "NCT00000001"

    def test_send_signal_alert(self):
        """Test sending alert from signal data."""
        result = self.alerter.send_signal_alert(
            company="BioTech Corp",
            ticker="BTCH",
            trial_id="NCT12345678",
            score=8.7,
            confidence=0.85,
            recommendation="STRONG_BUY",
            summary="Positive results"
        )

        assert result is True
        assert len(self.alerter.sent_alerts) == 1

    def test_send_signal_alert_below_threshold(self):
        """Test that alerts below threshold are not sent."""
        result = self.alerter.send_signal_alert(
            company="BioTech Corp",
            ticker="BTCH",
            trial_id="NCT12345678",
            score=6.0,  # Below threshold
            confidence=0.85,
            recommendation="HOLD",
            summary="Neutral results"
        )

        assert result is False
        assert len(self.alerter.sent_alerts) == 0

    def test_rate_limiting(self):
        """Test rate limiting works."""
        # Send max alerts
        for i in range(5):
            self.alerter.send_signal_alert(
                company=f"Test Co {i}",
                ticker=f"TST{i}",
                trial_id=f"NCT0000000{i}",
                score=9.0,
                confidence=0.9,
                recommendation="STRONG_BUY",
                summary="Test"
            )

        # Check rate limit status
        status = self.alerter.get_rate_limit_status()
        assert status["alerts_sent_last_hour"] == 5
        assert status["alerts_remaining"] == 0
        assert status["rate_limited"] is True

        # Try to send another - should fail
        result = self.alerter.send_signal_alert(
            company="Blocked Co",
            ticker="BLCK",
            trial_id="NCT99999999",
            score=9.0,
            confidence=0.9,
            recommendation="STRONG_BUY",
            summary="Should be blocked"
        )

        assert result is False

    def test_batch_alerts(self):
        """Test sending batch alerts."""
        signals = [
            {
                "company": "BioTech Corp",
                "ticker": "BTCH",
                "trial_id": "NCT12345678",
                "score": 8.7,
                "confidence": 0.85,
                "recommendation": "STRONG_BUY",
                "summary": "Positive results"
            },
            {
                "company": "PharmaCo",
                "ticker": "PHRM",
                "trial_id": "NCT87654321",
                "score": 1.5,
                "confidence": 0.78,
                "recommendation": "STRONG_SHORT",
                "summary": "Negative results"
            },
            {
                "company": "MedDev",
                "ticker": "MDEV",
                "trial_id": "NCT11111111",
                "score": 6.5,  # Below threshold
                "confidence": 0.65,
                "recommendation": "HOLD",
                "summary": "Mixed signals"
            },
        ]

        sent = self.alerter.send_batch_alerts(signals)

        assert sent == 2  # Only 2 should be sent (third is below threshold)

    def test_clear_alerts(self):
        """Test clearing recorded alerts."""
        self.alerter.send_signal_alert(
            company="Test Co",
            ticker="TEST",
            trial_id="NCT00000001",
            score=9.0,
            confidence=0.9,
            recommendation="STRONG_BUY",
            summary="Test"
        )

        assert len(self.alerter.sent_alerts) == 1

        self.alerter.clear_alerts()

        assert len(self.alerter.sent_alerts) == 0


class TestSlackAlerter:
    """Tests for the real SlackAlerter."""

    def test_init_without_webhook(self):
        """Test initialization without webhook URL."""
        with patch.dict('os.environ', {'SLACK_WEBHOOK_URL': ''}):
            alerter = SlackAlerter(webhook_url=None)
            assert alerter.webhook_url == ""

    def test_init_with_webhook(self):
        """Test initialization with webhook URL."""
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        assert alerter.webhook_url == "https://hooks.slack.com/test"

    def test_determine_priority_critical(self):
        """Test priority determination for critical scores."""
        alerter = SlackAlerter()
        assert alerter._determine_priority(9.5) == AlertPriority.CRITICAL
        assert alerter._determine_priority(0.5) == AlertPriority.CRITICAL

    def test_determine_priority_high(self):
        """Test priority determination for high scores."""
        alerter = SlackAlerter()
        assert alerter._determine_priority(8.5) == AlertPriority.HIGH
        assert alerter._determine_priority(1.5) == AlertPriority.HIGH

    def test_determine_priority_medium(self):
        """Test priority determination for medium scores."""
        alerter = SlackAlerter()
        assert alerter._determine_priority(7.5) == AlertPriority.MEDIUM
        assert alerter._determine_priority(2.5) == AlertPriority.MEDIUM

    def test_determine_alert_type_buy(self):
        """Test alert type determination for buy signals."""
        alerter = SlackAlerter()
        assert alerter._determine_alert_type(8.5, "STRONG_BUY") == AlertType.STRONG_BUY
        assert alerter._determine_alert_type(7.5, "BUY") == AlertType.BUY

    def test_determine_alert_type_short(self):
        """Test alert type determination for short signals."""
        alerter = SlackAlerter()
        assert alerter._determine_alert_type(1.5, "STRONG_SHORT") == AlertType.STRONG_SHORT
        assert alerter._determine_alert_type(2.5, "SHORT") == AlertType.SHORT

    def test_build_message_blocks(self):
        """Test message block building."""
        alerter = SlackAlerter()
        alert = SlackAlert(
            company="Test Co",
            ticker="TEST",
            trial_id="NCT12345678",
            signal_type=AlertType.STRONG_BUY,
            score=8.5,
            confidence=0.85,
            summary="Test summary",
            recommendation="STRONG_BUY",
            priority=AlertPriority.HIGH,
        )

        blocks = alerter._build_message_blocks(alert)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"

    @patch('requests.post')
    def test_send_alert_success(self, mock_post):
        """Test successful alert sending."""
        mock_post.return_value = Mock(status_code=200)

        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        alert = SlackAlert(
            company="Test Co",
            ticker="TEST",
            trial_id="NCT12345678",
            signal_type=AlertType.BUY,
            score=8.0,
            confidence=0.8,
            summary="Test",
            recommendation="BUY",
            priority=AlertPriority.HIGH,
        )

        result = alerter.send_alert(alert)

        assert result is True
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_send_alert_failure(self, mock_post):
        """Test failed alert sending."""
        mock_post.return_value = Mock(status_code=500, text="Error")

        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        alert = SlackAlert(
            company="Test Co",
            ticker="TEST",
            trial_id="NCT12345678",
            signal_type=AlertType.BUY,
            score=8.0,
            confidence=0.8,
            summary="Test",
            recommendation="BUY",
            priority=AlertPriority.HIGH,
        )

        result = alerter.send_alert(alert)

        assert result is False


class TestGetAlerter:
    """Tests for the alerter factory function."""

    def test_get_mock_alerter(self):
        """Test getting mock alerter."""
        alerter = get_alerter(use_mock=True)
        assert isinstance(alerter, MockSlackAlerter)

    def test_get_real_alerter(self):
        """Test getting real alerter."""
        alerter = get_alerter(use_mock=False)
        assert isinstance(alerter, SlackAlerter)

    def test_get_alerter_with_kwargs(self):
        """Test getting alerter with custom kwargs."""
        alerter = get_alerter(
            use_mock=True,
            max_alerts_per_hour=10,
            min_score_threshold=7.0
        )

        assert alerter.max_alerts_per_hour == 10
        assert alerter.min_score_threshold == 7.0
