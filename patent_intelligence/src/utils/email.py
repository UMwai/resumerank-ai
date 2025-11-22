"""
Email Notification System for Patent Intelligence

Sends weekly digest emails with patent cliff calendar updates
and alerts for important events.
"""

import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """
    Sends email notifications for patent cliff events.

    Supports:
    - Weekly digest emails
    - Immediate alerts for high-priority events
    - HTML and plain text formats
    """

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "",
        sender_password: str = "",
        use_tls: bool = True,
    ):
        """
        Initialize the email notifier.

        Args:
            smtp_server: SMTP server hostname.
            smtp_port: SMTP server port.
            sender_email: Sender email address.
            sender_password: Sender email password or app password.
            use_tls: Whether to use TLS encryption.
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.use_tls = use_tls

    def _connect(self) -> smtplib.SMTP:
        """
        Establish SMTP connection.

        Returns:
            SMTP connection object.
        """
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)

        if self.use_tls:
            server.starttls()

        if self.sender_email and self.sender_password:
            server.login(self.sender_email, self.sender_password)

        return server

    def send_email(
        self,
        recipients: List[str],
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
    ) -> bool:
        """
        Send an email to specified recipients.

        Args:
            recipients: List of recipient email addresses.
            subject: Email subject.
            body_html: HTML email body.
            body_text: Plain text email body (optional).

        Returns:
            True if email sent successfully, False otherwise.
        """
        if not self.sender_email or not self.sender_password:
            logger.warning("Email credentials not configured. Skipping email send.")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            # Add plain text version
            if body_text:
                part1 = MIMEText(body_text, "plain")
                msg.attach(part1)

            # Add HTML version
            part2 = MIMEText(body_html, "html")
            msg.attach(part2)

            # Send email
            server = self._connect()
            server.sendmail(self.sender_email, recipients, msg.as_string())
            server.quit()

            logger.info(f"Email sent to {len(recipients)} recipients: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def format_weekly_digest_html(
        self, events: List[Dict[str, Any]], start_date: date, end_date: date
    ) -> str:
        """
        Format weekly digest as HTML email.

        Args:
            events: List of calendar events.
            start_date: Digest period start date.
            end_date: Digest period end date.

        Returns:
            HTML formatted email body.
        """
        # Calculate summary stats
        total_events = len(events)
        high_conf = len([e for e in events if e.get("recommendation_confidence") == "HIGH"])
        total_opportunity = sum(e.get("market_opportunity", 0) for e in events)

        # Group events by month
        events_by_month = {}
        for event in events:
            event_date = event.get("event_date")
            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date).date()
            month_key = event_date.strftime("%B %Y") if event_date else "Unknown"
            if month_key not in events_by_month:
                events_by_month[month_key] = []
            events_by_month[month_key].append(event)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #1a365d;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .summary {{
            background-color: #f7fafc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .summary-stat {{
            display: inline-block;
            margin: 0 20px;
            text-align: center;
        }}
        .summary-number {{
            font-size: 24px;
            font-weight: bold;
            color: #1a365d;
        }}
        .month-header {{
            background-color: #e2e8f0;
            padding: 10px 15px;
            margin-top: 20px;
            font-weight: bold;
            border-left: 4px solid #1a365d;
        }}
        .event-card {{
            border: 1px solid #e2e8f0;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .event-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .drug-name {{
            font-size: 18px;
            font-weight: bold;
            color: #1a365d;
        }}
        .event-date {{
            color: #666;
            font-size: 14px;
        }}
        .confidence-high {{
            background-color: #38a169;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .confidence-medium {{
            background-color: #d69e2e;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .confidence-low {{
            background-color: #e53e3e;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .tier-blockbuster {{
            color: #805ad5;
            font-weight: bold;
        }}
        .tier-high_value {{
            color: #2b6cb0;
        }}
        .metric {{
            margin: 5px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 13px;
        }}
        .recommendation {{
            background-color: #ebf8ff;
            padding: 10px;
            margin-top: 10px;
            border-radius: 3px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding: 20px;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Patent Cliff Weekly Digest</h1>
        <p>Week of {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}</p>
    </div>

    <div class="summary">
        <div class="summary-stat">
            <div class="summary-number">{total_events}</div>
            <div>Upcoming Events</div>
        </div>
        <div class="summary-stat">
            <div class="summary-number">{high_conf}</div>
            <div>High Confidence</div>
        </div>
        <div class="summary-stat">
            <div class="summary-number">${total_opportunity/1e9:.1f}B</div>
            <div>Market at Risk</div>
        </div>
    </div>

    <h2>Upcoming Patent Cliff Events</h2>
"""

        for month, month_events in events_by_month.items():
            html += f'<div class="month-header">{month}</div>'

            for event in sorted(month_events, key=lambda x: x.get("event_date", "")):
                event_date = event.get("event_date")
                if isinstance(event_date, str):
                    event_date = datetime.fromisoformat(event_date).date()

                confidence = event.get("recommendation_confidence", "LOW").lower()
                confidence_class = f"confidence-{confidence}"

                tier = event.get("opportunity_tier", "SMALL").lower()
                tier_class = f"tier-{tier}"

                market_opp = event.get("market_opportunity", 0)
                if market_opp >= 1e9:
                    market_str = f"${market_opp/1e9:.1f}B"
                elif market_opp >= 1e6:
                    market_str = f"${market_opp/1e6:.0f}M"
                else:
                    market_str = f"${market_opp:,.0f}"

                html += f"""
    <div class="event-card">
        <div class="event-header">
            <div>
                <span class="drug-name">{event.get('brand_name', 'Unknown')}</span>
                <span class="event-date"> - {event_date.strftime('%b %d, %Y') if event_date else 'TBD'}</span>
            </div>
            <span class="{confidence_class}">{event.get('recommendation_confidence', 'N/A')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Generic Name:</span> {event.get('generic_name', 'N/A')}
        </div>
        <div class="metric">
            <span class="metric-label">Company:</span> {event.get('branded_company', 'N/A')} ({event.get('branded_company_ticker', 'N/A')})
        </div>
        <div class="metric">
            <span class="metric-label">Certainty Score:</span> {event.get('certainty_score', 0):.1f}%
        </div>
        <div class="metric">
            <span class="metric-label">Market Opportunity:</span>
            <span class="{tier_class}">{market_str} ({event.get('opportunity_tier', 'N/A')})</span>
        </div>
        <div class="metric">
            <span class="metric-label">Days Until Event:</span> {event.get('days_until_event', 'N/A')}
        </div>
        <div class="recommendation">
            <strong>Recommendation:</strong> {event.get('trade_recommendation', 'N/A')}<br>
            <small>{event.get('notes', '')}</small>
        </div>
    </div>
"""

        html += """
    <div class="footer">
        <p>This report is generated automatically by the Patent Intelligence System.</p>
        <p>For questions or to unsubscribe, contact the system administrator.</p>
    </div>
</body>
</html>
"""
        return html

    def format_weekly_digest_text(
        self, events: List[Dict[str, Any]], start_date: date, end_date: date
    ) -> str:
        """
        Format weekly digest as plain text email.

        Args:
            events: List of calendar events.
            start_date: Digest period start date.
            end_date: Digest period end date.

        Returns:
            Plain text formatted email body.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PATENT CLIFF WEEKLY DIGEST")
        lines.append(
            f"Week of {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}"
        )
        lines.append("=" * 60)
        lines.append("")

        # Summary
        total_events = len(events)
        high_conf = len([e for e in events if e.get("recommendation_confidence") == "HIGH"])
        total_opportunity = sum(e.get("market_opportunity", 0) for e in events)

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Upcoming Events: {total_events}")
        lines.append(f"High Confidence: {high_conf}")
        lines.append(f"Total Market at Risk: ${total_opportunity:,.0f}")
        lines.append("")

        # Events
        lines.append("UPCOMING EVENTS")
        lines.append("-" * 40)

        for event in sorted(events, key=lambda x: x.get("event_date", "")):
            event_date = event.get("event_date")
            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date).date()

            lines.append("")
            lines.append(
                f"[{event_date.strftime('%Y-%m-%d') if event_date else 'TBD'}] "
                f"{event.get('brand_name', 'Unknown')}"
            )
            lines.append(f"  Generic: {event.get('generic_name', 'N/A')}")
            lines.append(
                f"  Company: {event.get('branded_company', 'N/A')} "
                f"({event.get('branded_company_ticker', 'N/A')})"
            )
            lines.append(f"  Certainty: {event.get('certainty_score', 0):.1f}%")
            lines.append(
                f"  Market Opportunity: ${event.get('market_opportunity', 0):,.0f} "
                f"({event.get('opportunity_tier', 'N/A')})"
            )
            lines.append(
                f"  Recommendation: {event.get('trade_recommendation', 'N/A')} "
                f"[{event.get('recommendation_confidence', 'N/A')}]"
            )

        lines.append("")
        lines.append("=" * 60)
        lines.append("Generated by Patent Intelligence System")

        return "\n".join(lines)

    def send_weekly_digest(
        self,
        recipients: List[str],
        events: List[Dict[str, Any]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> bool:
        """
        Send the weekly patent cliff digest.

        Args:
            recipients: List of recipient email addresses.
            events: List of calendar events.
            start_date: Digest period start date.
            end_date: Digest period end date.

        Returns:
            True if sent successfully, False otherwise.
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            from dateutil.relativedelta import relativedelta
            end_date = start_date + relativedelta(months=12)

        subject = f"Patent Cliff Weekly Digest - {start_date.strftime('%B %d, %Y')}"

        html_body = self.format_weekly_digest_html(events, start_date, end_date)
        text_body = self.format_weekly_digest_text(events, start_date, end_date)

        return self.send_email(recipients, subject, html_body, text_body)

    def send_high_priority_alert(
        self,
        recipients: List[str],
        event: Dict[str, Any],
        alert_type: str = "NEW_EVENT",
    ) -> bool:
        """
        Send immediate alert for high-priority events.

        Args:
            recipients: List of recipient email addresses.
            event: Calendar event dictionary.
            alert_type: Type of alert (NEW_EVENT, LITIGATION_UPDATE, etc.)

        Returns:
            True if sent successfully, False otherwise.
        """
        brand_name = event.get("brand_name", "Unknown Drug")
        subject = f"[ALERT] Patent Cliff: {brand_name} - {alert_type}"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .alert-box {{
            background-color: #fed7d7;
            border: 2px solid #c53030;
            padding: 20px;
            margin: 20px;
            border-radius: 5px;
        }}
        .header {{ color: #c53030; }}
    </style>
</head>
<body>
    <div class="alert-box">
        <h2 class="header">Patent Cliff Alert</h2>
        <p><strong>Drug:</strong> {brand_name} ({event.get('generic_name', 'N/A')})</p>
        <p><strong>Company:</strong> {event.get('branded_company', 'N/A')}</p>
        <p><strong>Event Date:</strong> {event.get('event_date', 'TBD')}</p>
        <p><strong>Certainty Score:</strong> {event.get('certainty_score', 0):.1f}%</p>
        <p><strong>Market Opportunity:</strong> ${event.get('market_opportunity', 0):,.0f}</p>
        <p><strong>Recommendation:</strong> {event.get('trade_recommendation', 'N/A')}</p>
        <p><strong>Alert Type:</strong> {alert_type}</p>
    </div>
</body>
</html>
"""

        return self.send_email(recipients, subject, html_body)


if __name__ == "__main__":
    # Test email formatting
    print("\n=== Testing Email Notifier ===")

    notifier = EmailNotifier()

    # Sample events
    sample_events = [
        {
            "brand_name": "Humira",
            "generic_name": "adalimumab",
            "branded_company": "AbbVie",
            "branded_company_ticker": "ABBV",
            "event_date": "2025-01-15",
            "certainty_score": 85.5,
            "market_opportunity": 15_000_000_000,
            "opportunity_tier": "BLOCKBUSTER",
            "trade_recommendation": "EXECUTE TRADE",
            "recommendation_confidence": "HIGH",
            "days_until_event": 60,
            "notes": "Multiple generics approved. High certainty of market erosion.",
        },
        {
            "brand_name": "Eliquis",
            "generic_name": "apixaban",
            "branded_company": "Bristol-Myers Squibb",
            "branded_company_ticker": "BMY",
            "event_date": "2025-06-30",
            "certainty_score": 65.0,
            "market_opportunity": 8_000_000_000,
            "opportunity_tier": "HIGH_VALUE",
            "trade_recommendation": "MONITOR CLOSELY",
            "recommendation_confidence": "MEDIUM",
            "days_until_event": 220,
            "notes": "Litigation ongoing. Wait for resolution.",
        },
    ]

    # Generate HTML
    html = notifier.format_weekly_digest_html(
        sample_events,
        date(2024, 12, 1),
        date(2025, 12, 1),
    )

    # Save for preview
    with open("/tmp/email_preview.html", "w") as f:
        f.write(html)

    print("HTML email preview saved to /tmp/email_preview.html")

    # Generate text version
    text = notifier.format_weekly_digest_text(
        sample_events,
        date(2024, 12, 1),
        date(2025, 12, 1),
    )
    print("\n--- Plain Text Version ---")
    print(text)
