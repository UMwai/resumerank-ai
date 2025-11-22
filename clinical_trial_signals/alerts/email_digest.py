"""
Email digest system for Clinical Trial Signal Detection System.

Sends daily digest emails with detected signals and recommendations.
Supports SendGrid for email delivery.
"""
import hashlib
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database.models import TrialSignal, Trial
from database.connection import get_db_connection
from scoring.signal_scorer import SignalScorer

logger = logging.getLogger(__name__)


@dataclass
class DigestContent:
    """Content for email digest."""
    subject: str
    html_body: str
    text_body: str
    signals_count: int
    content_hash: str


class EmailDigest:
    """
    Generates and sends daily email digests with clinical trial signals.
    """

    def __init__(self):
        self.config = config.email
        self.scorer = SignalScorer()

    def generate_digest(self, days: int = 1) -> DigestContent:
        """
        Generate digest content for recent signals.

        Args:
            days: Number of days to include in digest

        Returns:
            DigestContent with formatted email content
        """
        # Get recent signals
        recent_signals = TrialSignal.get_recent(days=days)

        # Get scoring summary
        summary = self.scorer.generate_summary(lookback_days=30)

        # Build content
        subject = self._build_subject(recent_signals, summary)
        html_body = self._build_html_body(recent_signals, summary)
        text_body = self._build_text_body(recent_signals, summary)

        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(
            (subject + text_body).encode()
        ).hexdigest()[:16]

        return DigestContent(
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            signals_count=len(recent_signals),
            content_hash=content_hash
        )

    def _build_subject(
        self,
        signals: List[Dict],
        summary: Dict
    ) -> str:
        """Build email subject line."""
        date_str = date.today().strftime("%Y-%m-%d")

        strong_buys = len(summary.get("strong_buys", []))
        shorts = len(summary.get("shorts", []))

        if strong_buys > 0 and shorts > 0:
            action = f"{strong_buys} Buy, {shorts} Short Opportunities"
        elif strong_buys > 0:
            action = f"{strong_buys} Buy Opportunities"
        elif shorts > 0:
            action = f"{shorts} Short Opportunities"
        else:
            action = f"{len(signals)} Signals Detected"

        return f"Clinical Trial Signals [{date_str}]: {action}"

    def _build_html_body(
        self,
        signals: List[Dict],
        summary: Dict
    ) -> str:
        """Build HTML email body."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .header { background: #1a365d; color: white; padding: 20px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .opportunity { margin: 10px 0; padding: 10px; background: #f9f9f9; }
        .buy { border-left: 4px solid #38a169; }
        .short { border-left: 4px solid #e53e3e; }
        .signal { margin: 5px 0; padding: 5px; font-size: 0.9em; }
        .positive { color: #38a169; }
        .negative { color: #e53e3e; }
        .score { font-size: 1.2em; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f4f4f4; }
        .footer { font-size: 0.8em; color: #666; margin-top: 30px; padding-top: 15px; border-top: 1px solid #ddd; }
    </style>
</head>
<body>
"""

        # Header
        html += f"""
<div class="header">
    <h1>Clinical Trial Signal Report</h1>
    <p>{date.today().strftime("%B %d, %Y")}</p>
</div>
"""

        # Summary stats
        html += f"""
<div class="section">
    <h2>Summary</h2>
    <p>Monitoring <strong>{summary.get('total_trials', 0)}</strong> trials |
       <strong>{summary.get('scored_trials', 0)}</strong> with active signals |
       <strong>{len(signals)}</strong> new signals today</p>
</div>
"""

        # Top Buy Opportunities
        strong_buys = summary.get("strong_buys", [])
        if strong_buys:
            html += """
<div class="section">
    <h2>Top Buy Opportunities</h2>
"""
            for opp in strong_buys[:5]:
                score_class = "positive" if opp["score"] >= 5 else "negative"
                html += f"""
    <div class="opportunity buy">
        <strong>{opp.get('drug_name', 'Unknown')} ({opp.get('company', 'N/A')})</strong>
        <span class="score {score_class}" style="float:right;">{opp['score']:.1f}/10</span>
        <br>
        <small>Trial: {opp['trial_id']} | {opp.get('indication', 'N/A')}</small>
        <br>
        <span>Recommendation: <strong>{opp['recommendation']}</strong> |
              Confidence: {opp['confidence']:.0%} |
              {opp['signal_count']} signals</span>
    </div>
"""
            html += "</div>"

        # Short Opportunities
        shorts = summary.get("shorts", [])
        if shorts:
            html += """
<div class="section">
    <h2>Short Opportunities</h2>
"""
            for opp in shorts[:5]:
                html += f"""
    <div class="opportunity short">
        <strong>{opp.get('drug_name', 'Unknown')} ({opp.get('company', 'N/A')})</strong>
        <span class="score negative" style="float:right;">{opp['score']:.1f}/10</span>
        <br>
        <small>Trial: {opp['trial_id']} | {opp.get('indication', 'N/A')}</small>
        <br>
        <span>Recommendation: <strong>{opp['recommendation']}</strong> |
              Confidence: {opp['confidence']:.0%} |
              {opp['signal_count']} signals</span>
    </div>
"""
            html += "</div>"

        # Recent Signals Table
        if signals:
            html += """
<div class="section">
    <h2>Recent Signals</h2>
    <table>
        <tr>
            <th>Trial ID</th>
            <th>Company</th>
            <th>Signal Type</th>
            <th>Weight</th>
            <th>Details</th>
        </tr>
"""
            for sig in signals[:20]:  # Limit to 20 most recent
                weight = sig.get("signal_weight", 0)
                weight_class = "positive" if weight > 0 else "negative"
                weight_display = f"+{weight}" if weight > 0 else str(weight)

                html += f"""
        <tr>
            <td>{sig.get('trial_id', 'N/A')}</td>
            <td>{sig.get('company_name', 'N/A')}</td>
            <td>{sig.get('signal_type', 'N/A')}</td>
            <td class="{weight_class}">{weight_display}</td>
            <td>{sig.get('signal_value', '')[:50]}</td>
        </tr>
"""
            html += """
    </table>
</div>
"""

        # Footer
        html += f"""
<div class="footer">
    <p>This report was generated automatically by the Clinical Trial Signal Detection System.</p>
    <p>Data sources: ClinicalTrials.gov, SEC EDGAR</p>
    <p><strong>Disclaimer:</strong> This is not financial advice. All investment decisions should be made
       after conducting your own research. Past performance does not guarantee future results.</p>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
</div>
</body>
</html>
"""

        return html

    def _build_text_body(
        self,
        signals: List[Dict],
        summary: Dict
    ) -> str:
        """Build plain text email body."""
        lines = []
        lines.append("=" * 60)
        lines.append("CLINICAL TRIAL SIGNAL REPORT")
        lines.append(date.today().strftime("%B %d, %Y"))
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Monitoring: {summary.get('total_trials', 0)} trials")
        lines.append(f"With active signals: {summary.get('scored_trials', 0)}")
        lines.append(f"New signals today: {len(signals)}")
        lines.append("")

        # Buy Opportunities
        strong_buys = summary.get("strong_buys", [])
        if strong_buys:
            lines.append("TOP BUY OPPORTUNITIES")
            lines.append("-" * 40)
            for i, opp in enumerate(strong_buys[:5], 1):
                lines.append(f"{i}. {opp.get('drug_name', 'Unknown')} ({opp.get('company', 'N/A')})")
                lines.append(f"   Score: {opp['score']:.1f}/10 | {opp['recommendation']}")
                lines.append(f"   Trial: {opp['trial_id']}")
                lines.append(f"   Indication: {opp.get('indication', 'N/A')}")
                lines.append("")

        # Short Opportunities
        shorts = summary.get("shorts", [])
        if shorts:
            lines.append("SHORT OPPORTUNITIES")
            lines.append("-" * 40)
            for i, opp in enumerate(shorts[:5], 1):
                lines.append(f"{i}. {opp.get('drug_name', 'Unknown')} ({opp.get('company', 'N/A')})")
                lines.append(f"   Score: {opp['score']:.1f}/10 | {opp['recommendation']}")
                lines.append(f"   Trial: {opp['trial_id']}")
                lines.append("")

        # Recent Signals
        if signals:
            lines.append("RECENT SIGNALS")
            lines.append("-" * 40)
            for sig in signals[:10]:
                weight = sig.get("signal_weight", 0)
                weight_str = f"+{weight}" if weight > 0 else str(weight)
                lines.append(
                    f"[{weight_str}] {sig.get('trial_id', 'N/A')} - "
                    f"{sig.get('signal_type', 'N/A')}: {sig.get('signal_value', '')[:40]}"
                )
            lines.append("")

        # Footer
        lines.append("=" * 60)
        lines.append("DISCLAIMER: This is not financial advice.")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        return "\n".join(lines)

    def send_digest(self, digest: DigestContent = None) -> bool:
        """
        Send the digest email using SendGrid.

        Args:
            digest: DigestContent to send (generates if None)

        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            logger.info("Email sending disabled in config")
            return False

        if not self.config.sendgrid_api_key:
            logger.error("SendGrid API key not configured")
            return False

        if not self.config.to_emails or self.config.to_emails == [""]:
            logger.error("No recipient emails configured")
            return False

        if digest is None:
            digest = self.generate_digest()

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Email, To, Content

            sg = SendGridAPIClient(api_key=self.config.sendgrid_api_key)

            message = Mail(
                from_email=Email(self.config.from_email),
                to_emails=[To(email) for email in self.config.to_emails],
                subject=digest.subject,
            )
            message.add_content(Content("text/plain", digest.text_body))
            message.add_content(Content("text/html", digest.html_body))

            response = sg.send(message)

            if response.status_code in (200, 201, 202):
                logger.info(f"Email sent successfully to {len(self.config.to_emails)} recipients")
                self._log_digest_sent(digest)
                return True
            else:
                logger.error(f"Email send failed with status {response.status_code}")
                return False

        except ImportError:
            logger.error("SendGrid package not installed. Run: pip install sendgrid")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _log_digest_sent(self, digest: DigestContent) -> None:
        """Log digest to database for tracking."""
        try:
            db = get_db_connection()
            db.execute("""
                INSERT INTO email_digests (recipients, subject, signals_count, content_hash)
                VALUES (%s, %s, %s, %s)
            """, (
                self.config.to_emails,
                digest.subject,
                digest.signals_count,
                digest.content_hash
            ))
        except Exception as e:
            logger.warning(f"Failed to log digest: {e}")

    def print_digest(self, days: int = 1) -> None:
        """
        Print digest to console (for testing/dry-run).

        Args:
            days: Number of days to include
        """
        digest = self.generate_digest(days)

        print("\n" + "=" * 60)
        print("EMAIL DIGEST PREVIEW")
        print("=" * 60)
        print(f"Subject: {digest.subject}")
        print(f"Signals: {digest.signals_count}")
        print(f"Hash: {digest.content_hash}")
        print("-" * 60)
        print(digest.text_body)


if __name__ == "__main__":
    # Test email digest
    logging.basicConfig(level=logging.INFO)

    print("Testing email digest generation...")

    digest_sender = EmailDigest()

    # Generate and print digest
    digest_sender.print_digest(days=7)

    # Check if sending is enabled
    if config.email.enabled:
        print("\nEmail sending is ENABLED")
        print(f"  From: {config.email.from_email}")
        print(f"  To: {config.email.to_emails}")
    else:
        print("\nEmail sending is DISABLED")
        print("Set EMAIL_ENABLED=true in .env to enable")
