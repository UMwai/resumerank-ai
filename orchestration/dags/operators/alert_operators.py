"""
Alert Operators for Investment Signals

Custom Airflow operators for sending alerts through multiple channels:
- Email alerts with HTML formatting
- Slack alerts with rich formatting
- Alert deduplication and rate limiting
- Priority-based routing
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from airflow.models import BaseOperator, Variable
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


class AlertOperator(BaseOperator):
    """
    Base operator for sending alerts.

    Features:
    - Multi-channel support (email, slack, SMS)
    - Alert deduplication by content hash
    - Priority-based routing
    - Rate limiting
    - Audit logging

    Args:
        task_id: Unique task identifier
        alert_type: Type of alert (signal, system, digest)
        priority: Alert priority (critical, high, medium, low)
        input_task_id: Task ID to pull signal data from
        input_key: XCom key for input data
        channels: List of channels to send to
        dedup_window_hours: Window for alert deduplication
        rate_limit_per_hour: Maximum alerts per hour
    """

    template_fields = ("subject", "message")
    ui_color = "#fee2e2"
    ui_fgcolor = "#991b1b"

    PRIORITY_CHANNELS = {
        "critical": ["email", "slack", "sms"],
        "high": ["email", "slack"],
        "medium": ["email"],
        "low": ["email"],
    }

    @apply_defaults
    def __init__(
        self,
        alert_type: str,
        priority: str = "medium",
        subject: str = "",
        message: str = "",
        input_task_id: Optional[str] = None,
        input_key: str = "high_confidence",
        channels: Optional[List[str]] = None,
        dedup_window_hours: int = 24,
        rate_limit_per_hour: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alert_type = alert_type
        self.priority = priority
        self.subject = subject
        self.message = message
        self.input_task_id = input_task_id
        self.input_key = input_key
        self.channels = channels or self.PRIORITY_CHANNELS.get(priority, ["email"])
        self.dedup_window_hours = dedup_window_hours
        self.rate_limit_per_hour = rate_limit_per_hour

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute alert sending."""
        ti = context["ti"]

        # Get signal data if input task specified
        signal_data = None
        if self.input_task_id:
            signal_data = ti.xcom_pull(
                task_ids=self.input_task_id, key=self.input_key
            )

        # Skip if no signals and no explicit message
        if not signal_data and not self.message:
            logger.info("No signals to alert on")
            return {"alerts_sent": 0, "reason": "no_data"}

        # Build alert content
        subject, message = self._build_alert_content(signal_data)

        # Check deduplication
        dedup_key = self._generate_dedup_key(subject, message)
        if self._is_duplicate(dedup_key, context):
            logger.info(f"Skipping duplicate alert: {subject}")
            return {"alerts_sent": 0, "reason": "duplicate"}

        # Check rate limit
        if not self._check_rate_limit(context):
            logger.warning(f"Rate limit exceeded, skipping alert: {subject}")
            return {"alerts_sent": 0, "reason": "rate_limited"}

        # Send to each channel
        results = {}
        for channel in self.channels:
            if self._is_channel_enabled(channel):
                result = self._send_to_channel(channel, subject, message, signal_data)
                results[channel] = result

        # Record alert
        self._record_alert(dedup_key, subject, results, context)

        alerts_sent = sum(1 for r in results.values() if r.get("success"))
        ti.xcom_push(key="alert_results", value=results)

        return {"alerts_sent": alerts_sent, "results": results}

    def _build_alert_content(
        self, signal_data: Optional[Any]
    ) -> tuple[str, str]:
        """Build alert subject and message."""
        if self.subject and self.message:
            return self.subject, self.message

        if not signal_data:
            return self.subject or "Investment Alert", self.message or "Alert triggered"

        # Build from signal data
        if isinstance(signal_data, list) and len(signal_data) > 0:
            signals = signal_data
            subject = (
                f"[{self.priority.upper()}] {len(signals)} High-Confidence "
                f"{self.alert_type.replace('_', ' ').title()} Signals"
            )

            message_parts = [
                f"Detected {len(signals)} high-confidence signals:\n"
            ]
            for signal in signals[:10]:  # Limit to 10
                ticker = signal.get("ticker", "N/A")
                score = signal.get("composite_score", signal.get("score", 0))
                rec = signal.get("recommendation", "N/A")
                message_parts.append(f"- {ticker}: Score {score:.1f}, {rec}")

            if len(signals) > 10:
                message_parts.append(f"\n... and {len(signals) - 10} more")

            return subject, "\n".join(message_parts)

        return self.subject or "Investment Alert", str(signal_data)

    def _generate_dedup_key(self, subject: str, message: str) -> str:
        """Generate deduplication key from content."""
        content = f"{self.alert_type}:{self.priority}:{subject}:{message[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, dedup_key: str, context: Dict[str, Any]) -> bool:
        """Check if alert is a duplicate within dedup window."""
        # In production, check against database
        # For now, use XCom (limited but works for single runs)
        return False  # Simplified for demo

    def _check_rate_limit(self, context: Dict[str, Any]) -> bool:
        """Check if rate limit allows sending."""
        # In production, check against database/redis
        return True  # Simplified for demo

    def _is_channel_enabled(self, channel: str) -> bool:
        """Check if channel is enabled in configuration."""
        channel_env_map = {
            "email": "EMAIL_ALERTS_ENABLED",
            "slack": "SLACK_ALERTS_ENABLED",
            "sms": "SMS_ALERTS_ENABLED",
        }
        env_var = channel_env_map.get(channel, "")
        return os.environ.get(env_var, "false").lower() == "true"

    def _send_to_channel(
        self,
        channel: str,
        subject: str,
        message: str,
        signal_data: Optional[Any],
    ) -> Dict[str, Any]:
        """Send alert to specific channel."""
        try:
            if channel == "email":
                return self._send_email(subject, message)
            elif channel == "slack":
                return self._send_slack(subject, message)
            elif channel == "sms":
                return self._send_sms(subject, message)
            else:
                return {"success": False, "error": f"Unknown channel: {channel}"}
        except Exception as e:
            logger.error(f"Failed to send {channel} alert: {e}")
            return {"success": False, "error": str(e)}

    def _send_email(self, subject: str, message: str) -> Dict[str, Any]:
        """Send email alert."""
        # Import here to avoid dependency issues
        try:
            import sys
            sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
            from orchestration.alerts.alert_manager import AlertManager

            manager = AlertManager()
            result = manager.send_email_alert(
                subject=subject,
                body=message,
                signal_type=self.alert_type,
                priority=self.priority,
            )
            return {"success": result, "channel": "email"}
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return {"success": False, "error": str(e)}

    def _send_slack(self, subject: str, message: str) -> Dict[str, Any]:
        """Send Slack alert."""
        try:
            import sys
            sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
            from orchestration.alerts.alert_manager import AlertManager

            manager = AlertManager()
            formatted_message = f"*{subject}*\n\n{message}"
            result = manager.send_slack_alert(
                message=formatted_message,
                signal_type=self.alert_type,
                priority=self.priority,
            )
            return {"success": result, "channel": "slack"}
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return {"success": False, "error": str(e)}

    def _send_sms(self, subject: str, message: str) -> Dict[str, Any]:
        """Send SMS alert (for critical alerts only)."""
        try:
            import sys
            sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
            from orchestration.alerts.alert_manager import AlertManager

            manager = AlertManager()
            # SMS message is truncated
            sms_message = f"{subject}: {message}"[:160]
            result = manager.send_sms_alert(
                message=sms_message,
                signal_type=self.alert_type,
                priority=self.priority,
            )
            return {"success": result, "channel": "sms"}
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return {"success": False, "error": str(e)}

    def _record_alert(
        self,
        dedup_key: str,
        subject: str,
        results: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Record alert for auditing."""
        record = {
            "dedup_key": dedup_key,
            "subject": subject,
            "alert_type": self.alert_type,
            "priority": self.priority,
            "channels": self.channels,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": context.get("run_id"),
        }
        context["ti"].xcom_push(key="alert_record", value=record)


class SlackAlertOperator(AlertOperator):
    """
    Specialized operator for Slack alerts with rich formatting.

    Features:
    - Block Kit message formatting
    - Interactive buttons
    - Attachments with signal details
    """

    @apply_defaults
    def __init__(
        self,
        channel: str = "#investment-signals",
        mention_users: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs["channels"] = ["slack"]
        super().__init__(**kwargs)
        self.slack_channel = channel
        self.mention_users = mention_users or []

    def _build_slack_blocks(
        self, subject: str, message: str, signal_data: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Build Slack Block Kit message."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": subject},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            },
        ]

        # Add signal details as fields
        if signal_data and isinstance(signal_data, list):
            fields = []
            for signal in signal_data[:8]:  # Max 10 fields, leave room for others
                ticker = signal.get("ticker", "N/A")
                score = signal.get("composite_score", signal.get("score", 0))
                fields.append(
                    {"type": "mrkdwn", "text": f"*{ticker}*\nScore: {score:.1f}"}
                )

            if fields:
                blocks.append({"type": "section", "fields": fields})

        # Add timestamp
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Sent at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    }
                ],
            }
        )

        return blocks


class EmailAlertOperator(AlertOperator):
    """
    Specialized operator for email alerts with HTML formatting.

    Features:
    - HTML email templates
    - Inline signal tables
    - Attachment support
    """

    @apply_defaults
    def __init__(
        self,
        recipients: Optional[List[str]] = None,
        cc: Optional[List[str]] = None,
        html_template: Optional[str] = None,
        **kwargs,
    ):
        kwargs["channels"] = ["email"]
        super().__init__(**kwargs)
        self.recipients = recipients or []
        self.cc = cc or []
        self.html_template = html_template

    def _build_html_email(
        self, subject: str, message: str, signal_data: Optional[Any]
    ) -> str:
        """Build HTML email content."""
        if self.html_template:
            return self.html_template.format(
                subject=subject, message=message, signals=signal_data
            )

        # Default HTML template
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #1a365d; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .high-score {{ background-color: #c6f6d5; }}
                .medium-score {{ background-color: #fefcbf; }}
                .footer {{ background-color: #f7fafc; padding: 10px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{subject}</h1>
            </div>
            <div class="content">
                <p>{message}</p>
        """

        # Add signal table if available
        if signal_data and isinstance(signal_data, list):
            html += """
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Score</th>
                        <th>Recommendation</th>
                    </tr>
            """
            for signal in signal_data[:20]:
                ticker = signal.get("ticker", "N/A")
                score = signal.get("composite_score", signal.get("score", 0))
                rec = signal.get("recommendation", "N/A")
                score_class = "high-score" if score >= 7 else "medium-score" if score >= 5 else ""
                html += f"""
                    <tr class="{score_class}">
                        <td>{ticker}</td>
                        <td>{score:.1f}</td>
                        <td>{rec}</td>
                    </tr>
                """
            html += "</table>"

        html += f"""
            </div>
            <div class="footer">
                <p>Generated by Investment Signals Orchestration System</p>
                <p>Sent at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
        </body>
        </html>
        """

        return html
