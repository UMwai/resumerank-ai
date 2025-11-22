"""
Alert Engine with Multi-Channel Delivery

Configurable alert system supporting email, Slack, SMS (Twilio),
and in-app notifications. Includes alert history, snooze/dismiss,
and performance tracking.
"""

import json
import logging
import os
import smtplib
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert delivery channels."""
    DASHBOARD = "dashboard"
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


@dataclass
class AlertRule:
    """Defines a condition for triggering alerts."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    enabled: bool = True

    # Conditions
    condition_type: str = "score_threshold"  # score_threshold, price_target, signal_match
    ticker: Optional[str] = None  # None = all tickers
    source: Optional[str] = None  # clinical_trial, patent, insider, or None for all
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    min_confidence: Optional[float] = None
    signal_type: Optional[str] = None  # bullish, bearish, or None for both

    # Delivery
    channels: str = "dashboard"  # comma-separated
    priority: str = "medium"

    # Throttling
    cooldown_minutes: int = 60  # Minimum time between alerts for same ticker
    max_daily_alerts: int = 10

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_triggered: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "condition_type": self.condition_type,
            "ticker": self.ticker,
            "source": self.source,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "min_confidence": self.min_confidence,
            "signal_type": self.signal_type,
            "channels": self.channels,
            "priority": self.priority,
            "cooldown_minutes": self.cooldown_minutes,
            "max_daily_alerts": self.max_daily_alerts,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRule":
        """Create from dictionary."""
        def parse_datetime(val):
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return val

        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            condition_type=data.get("condition_type", "score_threshold"),
            ticker=data.get("ticker"),
            source=data.get("source"),
            min_score=data.get("min_score"),
            max_score=data.get("max_score"),
            min_confidence=data.get("min_confidence"),
            signal_type=data.get("signal_type"),
            channels=data.get("channels", "dashboard"),
            priority=data.get("priority", "medium"),
            cooldown_minutes=data.get("cooldown_minutes", 60),
            max_daily_alerts=data.get("max_daily_alerts", 10),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            last_triggered=parse_datetime(data.get("last_triggered")),
        )

    def get_channels_list(self) -> List[str]:
        """Get channels as list."""
        return [c.strip() for c in self.channels.split(",") if c.strip()]


@dataclass
class Alert:
    """Represents a triggered alert."""
    id: Optional[int] = None
    rule_id: Optional[int] = None
    rule_name: str = ""

    # Signal info
    ticker: str = ""
    company_name: str = ""
    source: str = ""
    signal_type: str = ""
    score: float = 0.0
    confidence: float = 0.0
    title: str = ""
    message: str = ""

    # Status
    status: str = "active"
    priority: str = "medium"

    # Delivery tracking
    channels_sent: str = ""  # comma-separated channels that were notified
    delivery_errors: str = ""

    # Timestamps
    created_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    snoozed_until: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None

    # Performance tracking
    outcome: Optional[str] = None  # positive, negative, neutral
    outcome_timestamp: Optional[datetime] = None
    outcome_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "source": self.source,
            "signal_type": self.signal_type,
            "score": self.score,
            "confidence": self.confidence,
            "title": self.title,
            "message": self.message,
            "status": self.status,
            "priority": self.priority,
            "channels_sent": self.channels_sent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "outcome": self.outcome,
        }


class AlertDeliveryChannel(ABC):
    """Abstract base class for alert delivery channels."""

    @abstractmethod
    def send(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """
        Send an alert.

        Args:
            alert: Alert to send

        Returns:
            Tuple of (success, error_message)
        """
        pass


class EmailChannel(AlertDeliveryChannel):
    """Email alert delivery via SMTP."""

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        to_addrs: Optional[List[str]] = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username
        self.to_addrs = to_addrs or []

    def send(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """Send alert via email."""
        if not self.to_addrs:
            return False, "No recipients configured"

        if not self.username or not self.password:
            return False, "Email credentials not configured"

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = f"[{alert.priority.upper()}] {alert.title}"

            body = f"""
Investment Alert: {alert.ticker}

{alert.title}

Company: {alert.company_name}
Source: {alert.source}
Signal Type: {alert.signal_type}
Score: {alert.score:.2f}
Confidence: {alert.confidence:.0%}

{alert.message}

---
Alert generated at {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True, None

        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
            return False, str(e)


class SlackChannel(AlertDeliveryChannel):
    """Slack alert delivery via webhook."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """Send alert via Slack webhook."""
        if not self.webhook_url:
            return False, "Slack webhook URL not configured"

        try:
            import requests

            color = {
                "bullish": "#10b981",
                "bearish": "#ef4444",
            }.get(alert.signal_type, "#6b7280")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": f"{alert.ticker}: {alert.title}",
                                },
                            },
                            {
                                "type": "section",
                                "fields": [
                                    {"type": "mrkdwn", "text": f"*Company:*\n{alert.company_name}"},
                                    {"type": "mrkdwn", "text": f"*Source:*\n{alert.source}"},
                                    {"type": "mrkdwn", "text": f"*Score:*\n{alert.score:.2f}"},
                                    {"type": "mrkdwn", "text": f"*Confidence:*\n{alert.confidence:.0%}"},
                                ],
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": alert.message},
                            },
                        ],
                    }
                ]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            return True, None

        except ImportError:
            return False, "requests package not installed"
        except Exception as e:
            logger.error(f"Slack delivery failed: {e}")
            return False, str(e)


class SMSChannel(AlertDeliveryChannel):
    """SMS alert delivery via Twilio."""

    def __init__(
        self,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        to_numbers: Optional[List[str]] = None,
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers or []

    def send(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """Send alert via Twilio SMS."""
        if not self.to_numbers:
            return False, "No phone numbers configured"

        if not self.account_sid or not self.auth_token:
            return False, "Twilio credentials not configured"

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)

            message = (
                f"[{alert.priority.upper()}] {alert.ticker}: {alert.title}\n"
                f"Score: {alert.score:.2f} | Confidence: {alert.confidence:.0%}"
            )

            errors = []
            for to_number in self.to_numbers:
                try:
                    client.messages.create(
                        body=message,
                        from_=self.from_number,
                        to=to_number,
                    )
                except Exception as e:
                    errors.append(f"{to_number}: {e}")

            if errors:
                return False, "; ".join(errors)

            return True, None

        except ImportError:
            return False, "twilio package not installed"
        except Exception as e:
            logger.error(f"SMS delivery failed: {e}")
            return False, str(e)


class DashboardChannel(AlertDeliveryChannel):
    """Dashboard in-app notification (stores for display)."""

    def __init__(self, callback: Optional[Callable[[Alert], None]] = None):
        self.callback = callback
        self.pending_alerts: List[Alert] = []

    def send(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """Store alert for dashboard display."""
        self.pending_alerts.append(alert)

        if self.callback:
            try:
                self.callback(alert)
            except Exception as e:
                logger.warning(f"Dashboard callback error: {e}")

        return True, None

    def get_pending(self, clear: bool = True) -> List[Alert]:
        """Get and optionally clear pending alerts."""
        alerts = self.pending_alerts.copy()
        if clear:
            self.pending_alerts = []
        return alerts


class AlertEngine:
    """
    Main alert engine managing rules, delivery, and history.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize alert engine.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            data_dir = Path.home() / ".investment_dashboard"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "alerts.db")

        self.db_path = db_path
        self._channels: Dict[str, AlertDeliveryChannel] = {}
        self._initialize_db()

        # Register default dashboard channel
        self.register_channel("dashboard", DashboardChannel())

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Alert rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    enabled INTEGER DEFAULT 1,
                    condition_type TEXT DEFAULT 'score_threshold',
                    ticker TEXT,
                    source TEXT,
                    min_score REAL,
                    max_score REAL,
                    min_confidence REAL,
                    signal_type TEXT,
                    channels TEXT DEFAULT 'dashboard',
                    priority TEXT DEFAULT 'medium',
                    cooldown_minutes INTEGER DEFAULT 60,
                    max_daily_alerts INTEGER DEFAULT 10,
                    created_at TEXT,
                    updated_at TEXT,
                    last_triggered TEXT
                )
            """)

            # Alerts history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER,
                    rule_name TEXT,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    source TEXT,
                    signal_type TEXT,
                    score REAL,
                    confidence REAL,
                    title TEXT,
                    message TEXT,
                    status TEXT DEFAULT 'active',
                    priority TEXT DEFAULT 'medium',
                    channels_sent TEXT,
                    delivery_errors TEXT,
                    created_at TEXT,
                    acknowledged_at TEXT,
                    snoozed_until TEXT,
                    dismissed_at TEXT,
                    outcome TEXT,
                    outcome_timestamp TEXT,
                    outcome_notes TEXT,
                    FOREIGN KEY (rule_id) REFERENCES alert_rules(id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")

            # Create default rule if none exists
            cursor.execute("SELECT COUNT(*) FROM alert_rules")
            if cursor.fetchone()[0] == 0:
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO alert_rules (
                        name, description, min_score, min_confidence,
                        channels, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    "High Score Alert",
                    "Alert when score exceeds 0.8 with high confidence",
                    0.8, 0.7, "dashboard", now, now
                ))

            conn.commit()

    def register_channel(self, name: str, channel: AlertDeliveryChannel) -> None:
        """
        Register a delivery channel.

        Args:
            name: Channel name
            channel: Channel instance
        """
        self._channels[name] = channel

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        to_addrs: List[str],
    ) -> None:
        """Configure email channel."""
        self.register_channel("email", EmailChannel(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            username=username,
            password=password,
            to_addrs=to_addrs,
        ))

    def configure_slack(self, webhook_url: str) -> None:
        """Configure Slack channel."""
        self.register_channel("slack", SlackChannel(webhook_url=webhook_url))

    def configure_sms(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_numbers: List[str],
    ) -> None:
        """Configure SMS channel."""
        self.register_channel("sms", SMSChannel(
            account_sid=account_sid,
            auth_token=auth_token,
            from_number=from_number,
            to_numbers=to_numbers,
        ))

    # Rule management

    def create_rule(self, rule: AlertRule) -> AlertRule:
        """Create a new alert rule."""
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alert_rules (
                    name, description, enabled, condition_type, ticker,
                    source, min_score, max_score, min_confidence, signal_type,
                    channels, priority, cooldown_minutes, max_daily_alerts,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.name, rule.description, int(rule.enabled), rule.condition_type,
                rule.ticker, rule.source, rule.min_score, rule.max_score,
                rule.min_confidence, rule.signal_type, rule.channels, rule.priority,
                rule.cooldown_minutes, rule.max_daily_alerts, now, now
            ))

            rule.id = cursor.lastrowid
            rule.created_at = datetime.fromisoformat(now)
            rule.updated_at = datetime.fromisoformat(now)

        return rule

    def get_rule(self, rule_id: int) -> Optional[AlertRule]:
        """Get a rule by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_rule(row)
        return None

    def get_all_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """Get all rules."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if enabled_only:
                cursor.execute("SELECT * FROM alert_rules WHERE enabled = 1")
            else:
                cursor.execute("SELECT * FROM alert_rules")
            return [self._row_to_rule(row) for row in cursor.fetchall()]

    def update_rule(self, rule: AlertRule) -> bool:
        """Update a rule."""
        if not rule.id:
            return False

        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alert_rules SET
                    name = ?, description = ?, enabled = ?, condition_type = ?,
                    ticker = ?, source = ?, min_score = ?, max_score = ?,
                    min_confidence = ?, signal_type = ?, channels = ?, priority = ?,
                    cooldown_minutes = ?, max_daily_alerts = ?, updated_at = ?
                WHERE id = ?
            """, (
                rule.name, rule.description, int(rule.enabled), rule.condition_type,
                rule.ticker, rule.source, rule.min_score, rule.max_score,
                rule.min_confidence, rule.signal_type, rule.channels, rule.priority,
                rule.cooldown_minutes, rule.max_daily_alerts, now, rule.id
            ))
            return cursor.rowcount > 0

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
            return cursor.rowcount > 0

    def _row_to_rule(self, row: sqlite3.Row) -> AlertRule:
        """Convert database row to AlertRule."""
        def parse_dt(val):
            return datetime.fromisoformat(val) if val else None

        return AlertRule(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            enabled=bool(row["enabled"]),
            condition_type=row["condition_type"] or "score_threshold",
            ticker=row["ticker"],
            source=row["source"],
            min_score=row["min_score"],
            max_score=row["max_score"],
            min_confidence=row["min_confidence"],
            signal_type=row["signal_type"],
            channels=row["channels"] or "dashboard",
            priority=row["priority"] or "medium",
            cooldown_minutes=row["cooldown_minutes"] or 60,
            max_daily_alerts=row["max_daily_alerts"] or 10,
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
            last_triggered=parse_dt(row["last_triggered"]),
        )

    # Alert processing

    def check_signal(self, signal: Dict[str, Any]) -> List[Alert]:
        """
        Check if a signal triggers any rules.

        Args:
            signal: Signal data dictionary

        Returns:
            List of triggered alerts
        """
        rules = self.get_all_rules(enabled_only=True)
        triggered = []

        for rule in rules:
            if self._matches_rule(signal, rule):
                if self._can_trigger(rule, signal.get("ticker", "")):
                    alert = self._create_alert(signal, rule)
                    triggered.append(alert)

        return triggered

    def _matches_rule(self, signal: Dict[str, Any], rule: AlertRule) -> bool:
        """Check if signal matches rule conditions."""
        # Ticker filter
        if rule.ticker and rule.ticker.upper() != signal.get("ticker", "").upper():
            return False

        # Source filter
        if rule.source and rule.source != signal.get("source"):
            return False

        # Signal type filter
        if rule.signal_type and rule.signal_type != signal.get("signal_type"):
            return False

        # Score threshold
        score = signal.get("score", 0)
        if rule.min_score is not None and score < rule.min_score:
            return False
        if rule.max_score is not None and score > rule.max_score:
            return False

        # Confidence threshold
        confidence = signal.get("confidence", 0)
        if rule.min_confidence is not None and confidence < rule.min_confidence:
            return False

        return True

    def _can_trigger(self, rule: AlertRule, ticker: str) -> bool:
        """Check if rule can trigger (cooldown and daily limits)."""
        # Check cooldown
        if rule.last_triggered:
            cooldown = timedelta(minutes=rule.cooldown_minutes)
            if datetime.now() - rule.last_triggered < cooldown:
                return False

        # Check daily limit
        with self._get_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().date().isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE rule_id = ? AND ticker = ? AND created_at >= ?
            """, (rule.id, ticker, today))

            if cursor.fetchone()[0] >= rule.max_daily_alerts:
                return False

        return True

    def _create_alert(self, signal: Dict[str, Any], rule: AlertRule) -> Alert:
        """Create and store an alert."""
        now = datetime.now()

        alert = Alert(
            rule_id=rule.id,
            rule_name=rule.name,
            ticker=signal.get("ticker", ""),
            company_name=signal.get("company_name", ""),
            source=signal.get("source", ""),
            signal_type=signal.get("signal_type", ""),
            score=signal.get("score", 0),
            confidence=signal.get("confidence", 0),
            title=signal.get("title", "Alert Triggered"),
            message=signal.get("description", ""),
            priority=rule.priority,
            created_at=now,
        )

        # Deliver to channels
        channels_sent = []
        errors = []

        for channel_name in rule.get_channels_list():
            if channel_name in self._channels:
                success, error = self._channels[channel_name].send(alert)
                if success:
                    channels_sent.append(channel_name)
                elif error:
                    errors.append(f"{channel_name}: {error}")

        alert.channels_sent = ",".join(channels_sent)
        alert.delivery_errors = "; ".join(errors)

        # Store alert
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (
                    rule_id, rule_name, ticker, company_name, source,
                    signal_type, score, confidence, title, message,
                    status, priority, channels_sent, delivery_errors, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.rule_id, alert.rule_name, alert.ticker, alert.company_name,
                alert.source, alert.signal_type, alert.score, alert.confidence,
                alert.title, alert.message, alert.status, alert.priority,
                alert.channels_sent, alert.delivery_errors, now.isoformat()
            ))
            alert.id = cursor.lastrowid

            # Update rule last_triggered
            cursor.execute("""
                UPDATE alert_rules SET last_triggered = ? WHERE id = ?
            """, (now.isoformat(), rule.id))

        return alert

    # Alert management

    def get_alerts(
        self,
        status: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM alerts WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker.upper())

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.get_alerts(status="active")

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark alert as acknowledged."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'acknowledged', acknowledged_at = ?
                WHERE id = ?
            """, (now, alert_id))
            return cursor.rowcount > 0

    def snooze_alert(self, alert_id: int, minutes: int = 60) -> bool:
        """Snooze an alert."""
        until = (datetime.now() + timedelta(minutes=minutes)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'snoozed', snoozed_until = ?
                WHERE id = ?
            """, (until, alert_id))
            return cursor.rowcount > 0

    def dismiss_alert(self, alert_id: int) -> bool:
        """Dismiss an alert."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'dismissed', dismissed_at = ?
                WHERE id = ?
            """, (now, alert_id))
            return cursor.rowcount > 0

    def record_outcome(
        self,
        alert_id: int,
        outcome: str,
        notes: str = "",
    ) -> bool:
        """Record alert outcome for performance tracking."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET outcome = ?, outcome_timestamp = ?, outcome_notes = ?
                WHERE id = ?
            """, (outcome, now, notes, alert_id))
            return cursor.rowcount > 0

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        """Convert database row to Alert."""
        def parse_dt(val):
            return datetime.fromisoformat(val) if val else None

        return Alert(
            id=row["id"],
            rule_id=row["rule_id"],
            rule_name=row["rule_name"] or "",
            ticker=row["ticker"],
            company_name=row["company_name"] or "",
            source=row["source"] or "",
            signal_type=row["signal_type"] or "",
            score=row["score"] or 0,
            confidence=row["confidence"] or 0,
            title=row["title"] or "",
            message=row["message"] or "",
            status=row["status"] or "active",
            priority=row["priority"] or "medium",
            channels_sent=row["channels_sent"] or "",
            delivery_errors=row["delivery_errors"] or "",
            created_at=parse_dt(row["created_at"]),
            acknowledged_at=parse_dt(row["acknowledged_at"]),
            snoozed_until=parse_dt(row["snoozed_until"]),
            dismissed_at=parse_dt(row["dismissed_at"]),
            outcome=row["outcome"],
            outcome_timestamp=parse_dt(row["outcome_timestamp"]),
            outcome_notes=row["outcome_notes"] or "",
        )

    # Performance tracking

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get alert performance statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total alerts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total = cursor.fetchone()[0]

            # By outcome
            cursor.execute("""
                SELECT outcome, COUNT(*) as count FROM alerts
                WHERE outcome IS NOT NULL
                GROUP BY outcome
            """)
            outcomes = {row["outcome"]: row["count"] for row in cursor.fetchall()}

            # Win rate
            positive = outcomes.get("positive", 0)
            negative = outcomes.get("negative", 0)
            tracked = positive + negative
            win_rate = positive / tracked if tracked > 0 else 0

            # By source
            cursor.execute("""
                SELECT source, outcome, COUNT(*) as count FROM alerts
                WHERE outcome IS NOT NULL
                GROUP BY source, outcome
            """)
            by_source = {}
            for row in cursor.fetchall():
                source = row["source"] or "unknown"
                if source not in by_source:
                    by_source[source] = {"positive": 0, "negative": 0, "neutral": 0}
                by_source[source][row["outcome"]] = row["count"]

            return {
                "total_alerts": total,
                "outcomes": outcomes,
                "win_rate": win_rate,
                "tracked_count": tracked,
                "by_source": by_source,
            }

    def get_pending_dashboard_alerts(self) -> List[Alert]:
        """Get alerts pending for dashboard display."""
        channel = self._channels.get("dashboard")
        if isinstance(channel, DashboardChannel):
            return channel.get_pending(clear=True)
        return []


# Global instance
_engine: Optional[AlertEngine] = None


def get_alert_engine(db_path: Optional[str] = None) -> AlertEngine:
    """Get or create the global alert engine."""
    global _engine
    if _engine is None:
        _engine = AlertEngine(db_path)
    return _engine
