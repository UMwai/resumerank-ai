"""
Alert Manager

Centralized alert management system supporting:
- Email alerts (SMTP)
- Slack webhooks
- SMS alerts (Twilio)
- Alert deduplication
"""

import hashlib
import json
import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types by signal source."""
    CLINICAL_TRIAL = "clinical_trial"
    PATENT_CLIFF = "patent_cliff"
    FORM4 = "form4"
    FORM13F = "13f"
    HIRING = "hiring"
    SYSTEM = "system"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dedup_key: Optional[str] = None
    sent_channels: List[str] = field(default_factory=list)


class AlertDeduplicator:
    """
    Alert deduplication to prevent spam.

    Uses in-memory cache with configurable TTL.
    For production, consider using Redis.
    """

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.sent_alerts: Dict[str, datetime] = {}

    def _generate_dedup_key(self, alert: Alert) -> str:
        """Generate deduplication key for an alert."""
        if alert.dedup_key:
            return alert.dedup_key

        # Create key from alert type, title, and key data
        key_data = {
            "type": alert.alert_type.value,
            "title": alert.title,
            "ticker": alert.data.get("ticker"),
            "signal_id": alert.data.get("signal_id"),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within the dedup window."""
        self._cleanup_expired()

        dedup_key = self._generate_dedup_key(alert)
        return dedup_key in self.sent_alerts

    def mark_sent(self, alert: Alert) -> None:
        """Mark an alert as sent."""
        dedup_key = self._generate_dedup_key(alert)
        self.sent_alerts[dedup_key] = datetime.utcnow()

    def _cleanup_expired(self) -> None:
        """Remove expired dedup entries."""
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        self.sent_alerts = {
            k: v for k, v in self.sent_alerts.items()
            if v > cutoff
        }


class EmailAlertSender:
    """Email alert sender using SMTP."""

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        from_email: str = None,
        use_tls: bool = True,
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("EMAIL_FROM", self.smtp_user)
        self.use_tls = use_tls

    def send(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
    ) -> bool:
        """
        Send an email alert.

        Args:
            recipients: List of email addresses
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body

        Returns:
            True if sent successfully
        """
        if not self.smtp_user or not self.smtp_password:
            logger.warning("Email credentials not configured, skipping email alert")
            return False

        if not recipients:
            logger.warning("No email recipients configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = ", ".join(recipients)

            # Attach plain text
            msg.attach(MIMEText(body, "plain"))

            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, recipients, msg.as_string())

            logger.info(f"Email sent to {len(recipients)} recipients: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class SlackAlertSender:
    """Slack alert sender using webhooks."""

    # Priority to emoji mapping
    PRIORITY_EMOJI = {
        AlertPriority.LOW: ":information_source:",
        AlertPriority.MEDIUM: ":warning:",
        AlertPriority.HIGH: ":rotating_light:",
        AlertPriority.CRITICAL: ":fire:",
    }

    # Alert type to emoji mapping
    TYPE_EMOJI = {
        AlertType.CLINICAL_TRIAL: ":microscope:",
        AlertType.PATENT_CLIFF: ":page_facing_up:",
        AlertType.FORM4: ":chart_with_upwards_trend:",
        AlertType.FORM13F: ":bank:",
        AlertType.HIRING: ":office:",
        AlertType.SYSTEM: ":gear:",
    }

    def __init__(
        self,
        webhook_url: str = None,
        channel: str = None,
        mention_users: List[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel or os.getenv("SLACK_CHANNEL", "#investment-signals")
        self.mention_users = mention_users or []

    def send(
        self,
        message: str,
        alert_type: AlertType = AlertType.SYSTEM,
        priority: AlertPriority = AlertPriority.MEDIUM,
        blocks: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Send a Slack alert.

        Args:
            message: Alert message
            alert_type: Type of alert
            priority: Alert priority
            blocks: Optional Slack blocks for rich formatting

        Returns:
            True if sent successfully
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured, skipping Slack alert")
            return False

        try:
            # Build message with emojis
            type_emoji = self.TYPE_EMOJI.get(alert_type, ":bell:")
            priority_emoji = self.PRIORITY_EMOJI.get(priority, "")

            # Add mentions for high priority
            mentions = ""
            if priority in (AlertPriority.HIGH, AlertPriority.CRITICAL) and self.mention_users:
                mentions = " ".join([f"<@{u}>" for u in self.mention_users]) + "\n"

            formatted_message = f"{priority_emoji} {type_emoji} {mentions}{message}"

            payload = {
                "channel": self.channel,
                "text": formatted_message,
                "unfurl_links": False,
                "unfurl_media": False,
            }

            if blocks:
                payload["blocks"] = blocks

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Slack alert sent to {self.channel}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class SMSAlertSender:
    """SMS alert sender using Twilio."""

    def __init__(
        self,
        account_sid: str = None,
        auth_token: str = None,
        from_number: str = None,
    ):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")

    def send(self, recipients: List[str], message: str) -> bool:
        """
        Send SMS alert via Twilio.

        Args:
            recipients: List of phone numbers (E.164 format)
            message: SMS message (max 160 chars for single SMS)

        Returns:
            True if at least one message sent successfully
        """
        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not configured, skipping SMS alert")
            return False

        if not recipients:
            logger.warning("No SMS recipients configured")
            return False

        try:
            # Truncate message if too long
            if len(message) > 160:
                message = message[:157] + "..."

            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"

            success_count = 0
            for recipient in recipients:
                try:
                    response = requests.post(
                        url,
                        data={
                            "From": self.from_number,
                            "To": recipient,
                            "Body": message,
                        },
                        auth=(self.account_sid, self.auth_token),
                        timeout=10,
                    )
                    response.raise_for_status()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send SMS to {recipient}: {e}")

            logger.info(f"SMS sent to {success_count}/{len(recipients)} recipients")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False


class AlertManager:
    """
    Centralized alert management.

    Coordinates sending alerts across multiple channels with deduplication.
    """

    def __init__(self):
        self.deduplicator = AlertDeduplicator(
            window_hours=int(os.getenv("ALERT_DEDUP_WINDOW_HOURS", "24"))
        )
        self.email_sender = EmailAlertSender()
        self.slack_sender = SlackAlertSender()
        self.sms_sender = SMSAlertSender()

        # Configuration
        self.email_enabled = os.getenv("EMAIL_ALERTS_ENABLED", "true").lower() == "true"
        self.slack_enabled = os.getenv("SLACK_ALERTS_ENABLED", "true").lower() == "true"
        self.sms_enabled = os.getenv("SMS_ALERTS_ENABLED", "false").lower() == "true"
        self.sms_critical_only = os.getenv("SMS_CRITICAL_ONLY", "true").lower() == "true"

        # Recipients
        self.email_recipients = [
            r.strip() for r in os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
            if r.strip()
        ]
        self.sms_recipients = [
            r.strip() for r in os.getenv("SMS_RECIPIENTS", "").split(",")
            if r.strip()
        ]

    def send_alert(
        self,
        title: str,
        message: str,
        alert_type: AlertType = AlertType.SYSTEM,
        priority: AlertPriority = AlertPriority.MEDIUM,
        data: Dict[str, Any] = None,
        channels: List[str] = None,
        skip_dedup: bool = False,
    ) -> Dict[str, bool]:
        """
        Send an alert across configured channels.

        Args:
            title: Alert title
            message: Alert message
            alert_type: Type of alert
            priority: Alert priority
            data: Additional data for the alert
            channels: Specific channels to use (default: all enabled)
            skip_dedup: Skip deduplication check

        Returns:
            Dictionary of channel -> success status
        """
        alert = Alert(
            alert_id=hashlib.md5(f"{title}{datetime.utcnow()}".encode()).hexdigest()[:12],
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            data=data or {},
        )

        # Check for duplicates
        if not skip_dedup and self.deduplicator.is_duplicate(alert):
            logger.info(f"Duplicate alert suppressed: {title}")
            return {"deduplicated": True}

        results = {}

        # Determine channels
        if channels is None:
            channels = []
            if self.email_enabled:
                channels.append("email")
            if self.slack_enabled:
                channels.append("slack")
            if self.sms_enabled and (
                not self.sms_critical_only or priority == AlertPriority.CRITICAL
            ):
                channels.append("sms")

        # Send to each channel
        if "email" in channels:
            results["email"] = self.email_sender.send(
                recipients=self.email_recipients,
                subject=title,
                body=message,
            )

        if "slack" in channels:
            results["slack"] = self.slack_sender.send(
                message=f"*{title}*\n\n{message}",
                alert_type=alert_type,
                priority=priority,
            )

        if "sms" in channels:
            # SMS gets abbreviated message
            sms_message = f"{title}: {message[:100]}"
            results["sms"] = self.sms_sender.send(
                recipients=self.sms_recipients,
                message=sms_message,
            )

        # Mark as sent if any channel succeeded
        if any(results.values()):
            self.deduplicator.mark_sent(alert)

        return results

    def send_email_alert(
        self,
        subject: str,
        body: str,
        signal_type: str = "system",
        priority: str = "medium",
        data: Dict[str, Any] = None,
    ) -> bool:
        """
        Convenience method for sending email alerts.

        Args:
            subject: Email subject
            body: Email body
            signal_type: Signal type (clinical_trial, patent_cliff, form4, etc.)
            priority: Priority level (low, medium, high, critical)
            data: Additional data

        Returns:
            True if sent successfully
        """
        alert_type = AlertType(signal_type) if signal_type in [t.value for t in AlertType] else AlertType.SYSTEM
        alert_priority = AlertPriority(priority) if priority in [p.value for p in AlertPriority] else AlertPriority.MEDIUM

        results = self.send_alert(
            title=subject,
            message=body,
            alert_type=alert_type,
            priority=alert_priority,
            data=data,
            channels=["email"],
        )
        return results.get("email", False)

    def send_slack_alert(
        self,
        message: str,
        signal_type: str = "system",
        priority: str = "medium",
        data: Dict[str, Any] = None,
    ) -> bool:
        """
        Convenience method for sending Slack alerts.

        Args:
            message: Slack message (supports markdown)
            signal_type: Signal type
            priority: Priority level
            data: Additional data

        Returns:
            True if sent successfully
        """
        alert_type = AlertType(signal_type) if signal_type in [t.value for t in AlertType] else AlertType.SYSTEM
        alert_priority = AlertPriority(priority) if priority in [p.value for p in AlertPriority] else AlertPriority.MEDIUM

        results = self.send_alert(
            title="Slack Alert",
            message=message,
            alert_type=alert_type,
            priority=alert_priority,
            data=data,
            channels=["slack"],
        )
        return results.get("slack", False)

    def send_sms_alert(
        self,
        message: str,
        priority: str = "critical",
    ) -> bool:
        """
        Convenience method for sending SMS alerts.

        Args:
            message: SMS message
            priority: Priority level (SMS typically for critical only)

        Returns:
            True if sent successfully
        """
        alert_priority = AlertPriority(priority) if priority in [p.value for p in AlertPriority] else AlertPriority.CRITICAL

        results = self.send_alert(
            title="SMS Alert",
            message=message,
            alert_type=AlertType.SYSTEM,
            priority=alert_priority,
            channels=["sms"],
        )
        return results.get("sms", False)

    def send_high_confidence_signal_alert(
        self,
        ticker: str,
        signal_type: str,
        score: float,
        recommendation: str,
        details: str = "",
    ) -> Dict[str, bool]:
        """
        Send alert for high-confidence investment signal.

        Args:
            ticker: Stock ticker
            signal_type: Type of signal
            score: Confidence score (0-10)
            recommendation: BUY/SELL/HOLD recommendation
            details: Additional details

        Returns:
            Dictionary of channel -> success status
        """
        title = f"[{signal_type.upper()}] High-Confidence Signal: {ticker}"
        message = f"""
Investment Signal Alert

Ticker: {ticker}
Signal Type: {signal_type}
Score: {score:.1f}/10
Recommendation: {recommendation}

{details}

Timestamp: {datetime.utcnow().isoformat()}
        """

        # Determine priority based on score
        if score >= 9.0:
            priority = AlertPriority.CRITICAL
        elif score >= 8.0:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM

        alert_type_map = {
            "clinical_trial": AlertType.CLINICAL_TRIAL,
            "patent_cliff": AlertType.PATENT_CLIFF,
            "form4": AlertType.FORM4,
            "13f": AlertType.FORM13F,
            "hiring": AlertType.HIRING,
        }
        alert_type = alert_type_map.get(signal_type.lower(), AlertType.SYSTEM)

        return self.send_alert(
            title=title,
            message=message,
            alert_type=alert_type,
            priority=priority,
            data={
                "ticker": ticker,
                "signal_type": signal_type,
                "score": score,
                "recommendation": recommendation,
            },
        )


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
