"""
Alerts Module

Provides alerting capabilities across multiple channels:
- Email (SMTP)
- Slack (Webhooks)
- SMS (Twilio)
"""

from .alert_manager import (
    AlertManager,
    AlertPriority,
    AlertType,
    Alert,
    AlertDeduplicator,
    EmailAlertSender,
    SlackAlertSender,
    SMSAlertSender,
    get_alert_manager,
)

__all__ = [
    "AlertManager",
    "AlertPriority",
    "AlertType",
    "Alert",
    "AlertDeduplicator",
    "EmailAlertSender",
    "SlackAlertSender",
    "SMSAlertSender",
    "get_alert_manager",
]
