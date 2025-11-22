"""
Real-time Slack Alerts for Clinical Trial Signal Detection System.

Provides instant Slack notifications for high-confidence signals:
- Integrates with Slack webhooks
- Rate limiting (max 5 alerts/hour by default)
- Rich message formatting with signal details
- Configurable thresholds for alert triggering
"""
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = "critical"  # Score >= 9 or <= 1
    HIGH = "high"  # Score >= 8 or <= 2
    MEDIUM = "medium"  # Score >= 7 or <= 3
    LOW = "low"  # Score >= 6 or <= 4


class AlertType(Enum):
    """Types of alerts."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    STRONG_SHORT = "strong_short"
    SHORT = "short"
    ENDPOINT_CHANGE = "endpoint_change"
    ENROLLMENT_MILESTONE = "enrollment_milestone"
    FDA_UPDATE = "fda_update"
    SEC_FILING = "sec_filing"


@dataclass
class SlackAlert:
    """Represents a Slack alert to be sent."""
    company: str
    ticker: str
    trial_id: str
    signal_type: AlertType
    score: float
    confidence: float
    summary: str
    recommendation: str
    priority: AlertPriority
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SlackAlerter:
    """
    Sends real-time alerts to Slack for high-confidence signals.

    Features:
    - Rate limiting to prevent spam
    - Rich message formatting with blocks
    - Configurable thresholds
    - Priority-based formatting
    """

    # Default configuration
    DEFAULT_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    DEFAULT_MAX_ALERTS_PER_HOUR = 5
    DEFAULT_MIN_SCORE_THRESHOLD = 8.0
    DEFAULT_MIN_CONFIDENCE = 0.7

    # Color codes for Slack attachments
    COLORS = {
        AlertPriority.CRITICAL: "#FF0000",  # Red
        AlertPriority.HIGH: "#FF6B00",  # Orange
        AlertPriority.MEDIUM: "#FFC107",  # Yellow
        AlertPriority.LOW: "#4CAF50",  # Green
    }

    # Emoji for different alert types
    EMOJI = {
        AlertType.STRONG_BUY: ":rocket:",
        AlertType.BUY: ":chart_with_upwards_trend:",
        AlertType.STRONG_SHORT: ":warning:",
        AlertType.SHORT: ":chart_with_downwards_trend:",
        AlertType.ENDPOINT_CHANGE: ":rotating_light:",
        AlertType.ENROLLMENT_MILESTONE: ":dart:",
        AlertType.FDA_UPDATE: ":pill:",
        AlertType.SEC_FILING: ":page_facing_up:",
    }

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        max_alerts_per_hour: int = DEFAULT_MAX_ALERTS_PER_HOUR,
        min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        channel: Optional[str] = None,
    ):
        """
        Initialize the Slack alerter.

        Args:
            webhook_url: Slack webhook URL
            max_alerts_per_hour: Maximum alerts allowed per hour
            min_score_threshold: Minimum score to trigger alert (8 = high confidence)
            min_confidence: Minimum confidence level (0-1)
            channel: Override channel (if webhook allows)
        """
        self.webhook_url = webhook_url or self.DEFAULT_WEBHOOK_URL
        self.max_alerts_per_hour = max_alerts_per_hour
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence
        self.channel = channel

        # Rate limiting: track timestamps of sent alerts
        self._alert_timestamps: deque = deque(maxlen=100)

        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not configured - alerts will be logged only")

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if alert can be sent, False if rate limited
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Remove old timestamps
        while self._alert_timestamps and self._alert_timestamps[0] < cutoff:
            self._alert_timestamps.popleft()

        # Check if under limit
        return len(self._alert_timestamps) < self.max_alerts_per_hour

    def _record_alert(self) -> None:
        """Record that an alert was sent."""
        self._alert_timestamps.append(datetime.now())

    def _determine_priority(self, score: float) -> AlertPriority:
        """Determine alert priority based on score."""
        if score >= 9 or score <= 1:
            return AlertPriority.CRITICAL
        elif score >= 8 or score <= 2:
            return AlertPriority.HIGH
        elif score >= 7 or score <= 3:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _determine_alert_type(self, score: float, recommendation: str) -> AlertType:
        """Determine alert type based on score and recommendation."""
        if "STRONG_BUY" in recommendation.upper():
            return AlertType.STRONG_BUY
        elif "BUY" in recommendation.upper():
            return AlertType.BUY
        elif "STRONG_SHORT" in recommendation.upper():
            return AlertType.STRONG_SHORT
        elif "SHORT" in recommendation.upper():
            return AlertType.SHORT
        elif score >= 7:
            return AlertType.BUY
        else:
            return AlertType.SHORT

    def should_alert(self, score: float, confidence: float) -> bool:
        """
        Determine if a signal should trigger an alert.

        Args:
            score: Signal composite score (0-10)
            confidence: Confidence level (0-1)

        Returns:
            True if alert should be sent
        """
        # Check score threshold (high scores >= 8 or low scores <= 2)
        score_triggers = score >= self.min_score_threshold or score <= (10 - self.min_score_threshold)

        # Check confidence threshold
        confidence_ok = confidence >= self.min_confidence

        return score_triggers and confidence_ok

    def _build_message_blocks(self, alert: SlackAlert) -> List[Dict]:
        """
        Build Slack Block Kit message blocks.

        Args:
            alert: SlackAlert object

        Returns:
            List of Slack block dicts
        """
        emoji = self.EMOJI.get(alert.signal_type, ":bell:")
        priority_text = f"*Priority: {alert.priority.value.upper()}*"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Clinical Trial Signal Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Company:*\n{alert.company} ({alert.ticker})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Trial ID:*\n{alert.trial_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Score:*\n{alert.score:.1f}/10"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{alert.confidence:.0%}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recommendation:* {alert.recommendation}\n{priority_text}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{alert.summary}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Detected at {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    }
                ]
            }
        ]

        # Add additional data if present
        if alert.additional_data:
            extra_fields = []
            for key, value in list(alert.additional_data.items())[:4]:
                extra_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })

            if extra_fields:
                blocks.insert(-1, {
                    "type": "section",
                    "fields": extra_fields
                })

        # Add divider at the end
        blocks.append({"type": "divider"})

        return blocks

    def _build_fallback_text(self, alert: SlackAlert) -> str:
        """Build fallback text for notifications."""
        return (
            f"Clinical Trial Alert: {alert.company} ({alert.ticker}) - "
            f"{alert.recommendation} - Score: {alert.score:.1f}/10 - "
            f"{alert.summary[:100]}"
        )

    def send_alert(self, alert: SlackAlert) -> bool:
        """
        Send a Slack alert.

        Args:
            alert: SlackAlert object to send

        Returns:
            True if sent successfully
        """
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning(
                f"Rate limit exceeded - skipping alert for {alert.trial_id}. "
                f"Max {self.max_alerts_per_hour} alerts/hour allowed."
            )
            return False

        # Build message
        message = {
            "text": self._build_fallback_text(alert),
            "blocks": self._build_message_blocks(alert),
            "attachments": [
                {
                    "color": self.COLORS.get(alert.priority, "#808080"),
                    "fallback": self._build_fallback_text(alert),
                }
            ]
        }

        if self.channel:
            message["channel"] = self.channel

        # Log the alert
        logger.info(
            f"Slack Alert: {alert.company} ({alert.ticker}) - "
            f"{alert.trial_id} - Score: {alert.score:.1f} - {alert.recommendation}"
        )

        # Send to Slack if webhook configured
        if not self.webhook_url:
            logger.info("Slack webhook not configured - alert logged only")
            return True

        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                self._record_alert()
                logger.info(f"Slack alert sent successfully for {alert.trial_id}")
                return True
            else:
                logger.error(
                    f"Slack API error: {response.status_code} - {response.text}"
                )
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def send_signal_alert(
        self,
        company: str,
        ticker: str,
        trial_id: str,
        score: float,
        confidence: float,
        recommendation: str,
        summary: str,
        additional_data: Optional[Dict] = None
    ) -> bool:
        """
        Convenience method to send an alert from signal data.

        Args:
            company: Company name
            ticker: Stock ticker
            trial_id: NCT trial ID
            score: Composite score (0-10)
            confidence: Confidence level (0-1)
            recommendation: Recommendation string
            summary: Signal summary
            additional_data: Extra data to include

        Returns:
            True if alert should be sent and was sent successfully
        """
        # Check if alert should be sent
        if not self.should_alert(score, confidence):
            logger.debug(
                f"Signal for {trial_id} does not meet alert threshold "
                f"(score={score:.1f}, confidence={confidence:.2f})"
            )
            return False

        # Create alert object
        alert = SlackAlert(
            company=company,
            ticker=ticker,
            trial_id=trial_id,
            signal_type=self._determine_alert_type(score, recommendation),
            score=score,
            confidence=confidence,
            summary=summary,
            recommendation=recommendation,
            priority=self._determine_priority(score),
            additional_data=additional_data or {}
        )

        return self.send_alert(alert)

    def send_batch_alerts(self, signals: List[Dict]) -> int:
        """
        Send alerts for a batch of signals (respecting rate limits).

        Args:
            signals: List of signal dicts with required fields

        Returns:
            Number of alerts sent
        """
        sent_count = 0

        # Sort by score (most extreme first)
        sorted_signals = sorted(
            signals,
            key=lambda s: abs(s.get("score", 5) - 5),
            reverse=True
        )

        for signal in sorted_signals:
            if not self._check_rate_limit():
                logger.warning(
                    f"Rate limit reached - {len(sorted_signals) - sent_count} alerts skipped"
                )
                break

            success = self.send_signal_alert(
                company=signal.get("company", "Unknown"),
                ticker=signal.get("ticker", "N/A"),
                trial_id=signal.get("trial_id", "Unknown"),
                score=signal.get("score", 5.0),
                confidence=signal.get("confidence", 0.5),
                recommendation=signal.get("recommendation", "HOLD"),
                summary=signal.get("summary", ""),
                additional_data=signal.get("additional_data")
            )

            if success:
                sent_count += 1

        logger.info(f"Sent {sent_count}/{len(signals)} alerts")
        return sent_count

    def send_test_alert(self) -> bool:
        """
        Send a test alert to verify configuration.

        Returns:
            True if test alert sent successfully
        """
        test_alert = SlackAlert(
            company="Test Company Inc",
            ticker="TEST",
            trial_id="NCT00000000",
            signal_type=AlertType.BUY,
            score=8.5,
            confidence=0.85,
            summary="This is a test alert to verify Slack integration is working correctly.",
            recommendation="STRONG_BUY",
            priority=AlertPriority.HIGH,
            additional_data={
                "Drug": "TestDrug-123",
                "Indication": "Test Condition",
                "Phase": "Phase 3",
                "Status": "Active"
            }
        )

        return self.send_alert(test_alert)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Dict with rate limit information
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Clean old timestamps
        while self._alert_timestamps and self._alert_timestamps[0] < cutoff:
            self._alert_timestamps.popleft()

        alerts_sent = len(self._alert_timestamps)
        remaining = max(0, self.max_alerts_per_hour - alerts_sent)

        # Calculate when rate limit resets
        if self._alert_timestamps:
            oldest = self._alert_timestamps[0]
            reset_at = oldest + timedelta(hours=1)
        else:
            reset_at = now

        return {
            "alerts_sent_last_hour": alerts_sent,
            "alerts_remaining": remaining,
            "max_per_hour": self.max_alerts_per_hour,
            "rate_limited": remaining == 0,
            "reset_at": reset_at.isoformat(),
        }


class MockSlackAlerter(SlackAlerter):
    """
    Mock Slack alerter for testing.

    Records alerts instead of sending them.
    """

    def __init__(self, **kwargs):
        """Initialize mock alerter."""
        super().__init__(webhook_url="http://mock.slack.webhook", **kwargs)
        self.sent_alerts: List[SlackAlert] = []

    def send_alert(self, alert: SlackAlert) -> bool:
        """Record alert instead of sending."""
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded (mock)")
            return False

        self.sent_alerts.append(alert)
        self._record_alert()
        logger.info(f"Mock alert recorded: {alert.trial_id} - {alert.recommendation}")
        return True

    def clear_alerts(self) -> None:
        """Clear recorded alerts."""
        self.sent_alerts.clear()
        self._alert_timestamps.clear()


# Factory function
def get_alerter(use_mock: bool = False, **kwargs) -> SlackAlerter:
    """
    Get appropriate alerter instance.

    Args:
        use_mock: If True, return mock alerter
        **kwargs: Additional arguments for alerter

    Returns:
        SlackAlerter instance
    """
    if use_mock:
        return MockSlackAlerter(**kwargs)
    return SlackAlerter(**kwargs)


if __name__ == "__main__":
    # Test the Slack alerter
    logging.basicConfig(level=logging.INFO)

    print("Testing Slack Alerter...")
    print("=" * 60)

    # Use mock alerter for testing
    alerter = MockSlackAlerter(
        max_alerts_per_hour=5,
        min_score_threshold=8.0,
        min_confidence=0.7
    )

    # Test threshold checking
    print("\nTesting threshold logic:")
    test_cases = [
        (8.5, 0.8, True, "High score, high confidence"),
        (7.0, 0.8, False, "Score below threshold"),
        (8.5, 0.5, False, "Confidence below threshold"),
        (2.0, 0.8, True, "Low score (short signal), high confidence"),
        (5.0, 0.9, False, "Neutral score"),
    ]

    for score, confidence, expected, description in test_cases:
        result = alerter.should_alert(score, confidence)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {description}: score={score}, conf={confidence} -> {result}")

    # Test sending alerts
    print("\nTesting alert sending:")
    signals = [
        {
            "company": "BioTech Corp",
            "ticker": "BTCH",
            "trial_id": "NCT12345678",
            "score": 8.7,
            "confidence": 0.85,
            "recommendation": "STRONG_BUY",
            "summary": "Phase 3 trial showing strong enrollment and positive interim data.",
            "additional_data": {"Drug": "Drug-123", "Indication": "Lung Cancer"}
        },
        {
            "company": "PharmaCo Inc",
            "ticker": "PHRM",
            "trial_id": "NCT87654321",
            "score": 1.5,
            "confidence": 0.78,
            "recommendation": "STRONG_SHORT",
            "summary": "Primary endpoint change detected, enrollment significantly behind.",
            "additional_data": {"Drug": "Drug-456", "Indication": "Alzheimer's"}
        },
        {
            "company": "MedDev LLC",
            "ticker": "MDEV",
            "trial_id": "NCT11111111",
            "score": 6.5,  # Below threshold
            "confidence": 0.65,
            "recommendation": "HOLD",
            "summary": "Mixed signals, monitoring continues.",
        },
    ]

    sent = alerter.send_batch_alerts(signals)
    print(f"  Sent {sent} alerts out of {len(signals)} signals")

    print(f"\nRecorded alerts:")
    for alert in alerter.sent_alerts:
        print(f"  - {alert.company}: {alert.recommendation} (score={alert.score:.1f})")

    # Test rate limiting
    print("\nTesting rate limiting:")
    status = alerter.get_rate_limit_status()
    print(f"  Alerts sent: {status['alerts_sent_last_hour']}/{status['max_per_hour']}")
    print(f"  Remaining: {status['alerts_remaining']}")

    # Send more alerts to test rate limit
    for i in range(5):
        alerter.send_signal_alert(
            company=f"Test Co {i}",
            ticker=f"TST{i}",
            trial_id=f"NCT0000000{i}",
            score=9.0,
            confidence=0.9,
            recommendation="STRONG_BUY",
            summary="Test alert"
        )

    status = alerter.get_rate_limit_status()
    print(f"\n  After more alerts:")
    print(f"  Alerts sent: {status['alerts_sent_last_hour']}/{status['max_per_hour']}")
    print(f"  Rate limited: {status['rate_limited']}")

    print("\n" + "=" * 60)
    print("Test complete!")
