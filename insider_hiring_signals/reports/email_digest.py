"""
Daily Email Digest for Insider Activity + Hiring Signals System

Generates and sends daily digest emails with:
- Top bullish and bearish signals
- New insider transactions
- Institutional position changes
- Job posting trends
"""

import smtplib
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from jinja2 import Template

from models.signal_scorer import SignalScore, SignalScorer
from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailDigest:
    """
    Generates and sends daily email digests with signal summaries.
    """

    # HTML email template
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biotech Signal Digest - {{ date }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a1a2e;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 {
            color: #1a1a2e;
            margin-top: 30px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }
        .bullish { color: #4CAF50; }
        .bearish { color: #f44336; }
        .neutral { color: #9e9e9e; }
        .score-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            padding: 15px 20px;
            margin: 10px 0;
            border-left: 4px solid #ddd;
        }
        .score-card.bullish { border-left-color: #4CAF50; }
        .score-card.bearish { border-left-color: #f44336; }
        .ticker {
            font-size: 1.4em;
            font-weight: bold;
            color: #1a1a2e;
        }
        .score {
            font-size: 1.8em;
            font-weight: bold;
        }
        .recommendation {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .recommendation.strong-buy { background: #4CAF50; color: white; }
        .recommendation.buy { background: #8BC34A; color: white; }
        .recommendation.neutral { background: #9e9e9e; color: white; }
        .recommendation.sell { background: #FF9800; color: white; }
        .recommendation.strong-sell { background: #f44336; color: white; }
        .signal-list {
            margin: 10px 0;
            padding-left: 20px;
        }
        .signal-item {
            margin: 5px 0;
            font-size: 0.9em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #1a1a2e;
        }
        .stat-label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
        }
        .transaction-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .transaction-table th, .transaction-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .transaction-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.85em;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.85em;
            color: #666;
            text-align: center;
        }
        .disclaimer {
            font-size: 0.75em;
            color: #999;
            margin-top: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Biotech Signal Digest</h1>
        <p>Daily summary for <strong>{{ date }}</strong></p>

        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{{ total_signals }}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-box">
                <div class="stat-value bullish">{{ bullish_count }}</div>
                <div class="stat-label">Bullish</div>
            </div>
            <div class="stat-box">
                <div class="stat-value bearish">{{ bearish_count }}</div>
                <div class="stat-label">Bearish</div>
            </div>
        </div>

        <!-- Top Bullish -->
        {% if top_bullish %}
        <h2 class="bullish">Top Bullish Signals</h2>
        {% for score in top_bullish %}
        <div class="score-card bullish">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="ticker">{{ score.company_ticker }}</span>
                    <span class="recommendation {{ score.recommendation | lower | replace(' ', '-') }}">
                        {{ score.recommendation }}
                    </span>
                </div>
                <div class="score bullish">+{{ "%.1f" | format(score.composite_score) }}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                Confidence: {{ "%.0f" | format(score.confidence * 100) }}% |
                Signals: {{ score.signal_count }}
            </div>
            <ul class="signal-list">
            {% for signal in score.signals[:3] %}
                <li class="signal-item">{{ signal.description }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
        {% endif %}

        <!-- Top Bearish -->
        {% if top_bearish %}
        <h2 class="bearish">Top Bearish Signals</h2>
        {% for score in top_bearish %}
        <div class="score-card bearish">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="ticker">{{ score.company_ticker }}</span>
                    <span class="recommendation {{ score.recommendation | lower | replace(' ', '-') }}">
                        {{ score.recommendation }}
                    </span>
                </div>
                <div class="score bearish">{{ "%.1f" | format(score.composite_score) }}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                Confidence: {{ "%.0f" | format(score.confidence * 100) }}% |
                Signals: {{ score.signal_count }}
            </div>
            <ul class="signal-list">
            {% for signal in score.signals[:3] %}
                <li class="signal-item">{{ signal.description }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
        {% endif %}

        <!-- Recent Insider Transactions -->
        {% if insider_transactions %}
        <h2>Recent Insider Transactions</h2>
        <table class="transaction-table">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Insider</th>
                    <th>Type</th>
                    <th>Value</th>
                    <th>Date</th>
                </tr>
            </thead>
            <tbody>
            {% for trans in insider_transactions %}
                <tr>
                    <td><strong>{{ trans.ticker }}</strong></td>
                    <td>{{ trans.name }} ({{ trans.title }})</td>
                    <td class="{{ 'bullish' if trans.type == 'Purchase' else 'bearish' }}">
                        {{ trans.type }}
                    </td>
                    <td>${{ "{:,.0f}".format(trans.value) }}</td>
                    <td>{{ trans.date }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <!-- Institutional Changes -->
        {% if institutional_changes %}
        <h2>Institutional Position Changes</h2>
        <table class="transaction-table">
            <thead>
                <tr>
                    <th>Fund</th>
                    <th>Ticker</th>
                    <th>Change</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
            {% for change in institutional_changes %}
                <tr>
                    <td>{{ change.fund }}</td>
                    <td><strong>{{ change.ticker }}</strong></td>
                    <td class="{{ 'bullish' if change.pct_change > 0 else 'bearish' }}">
                        {{ "{:+.1f}%".format(change.pct_change) }}
                    </td>
                    <td>{{ change.type }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <!-- Job Posting Highlights -->
        {% if job_highlights %}
        <h2>Hiring Highlights</h2>
        {% for ticker, stats in job_highlights.items() %}
        <div class="score-card">
            <span class="ticker">{{ ticker }}</span>
            <span style="margin-left: 10px; color: #666;">
                {{ stats.total }} open positions
                ({{ stats.commercial }} commercial, {{ stats.clinical }} clinical)
            </span>
        </div>
        {% endfor %}
        {% endif %}

        <div class="footer">
            <p>Generated by Insider Activity + Hiring Signals System</p>
            <p>{{ generated_at }}</p>
        </div>

        <div class="disclaimer">
            <strong>Disclaimer:</strong> This report is for informational purposes only and
            does not constitute investment advice. Past performance does not guarantee future
            results. Always conduct your own due diligence before making investment decisions.
        </div>
    </div>
</body>
</html>
"""

    # Plain text template
    TEXT_TEMPLATE = """
BIOTECH SIGNAL DIGEST - {{ date }}
======================================

SUMMARY
-------
Total Signals: {{ total_signals }}
Bullish: {{ bullish_count }}
Bearish: {{ bearish_count }}

{% if top_bullish %}
TOP BULLISH SIGNALS
-------------------
{% for score in top_bullish %}
{{ score.company_ticker }}: +{{ "%.1f" | format(score.composite_score) }} ({{ score.recommendation }})
  Confidence: {{ "%.0f" | format(score.confidence * 100) }}% | Signals: {{ score.signal_count }}
{% for signal in score.signals[:3] %}
  - {{ signal.description }}
{% endfor %}

{% endfor %}
{% endif %}

{% if top_bearish %}
TOP BEARISH SIGNALS
-------------------
{% for score in top_bearish %}
{{ score.company_ticker }}: {{ "%.1f" | format(score.composite_score) }} ({{ score.recommendation }})
  Confidence: {{ "%.0f" | format(score.confidence * 100) }}% | Signals: {{ score.signal_count }}
{% for signal in score.signals[:3] %}
  - {{ signal.description }}
{% endfor %}

{% endfor %}
{% endif %}

{% if insider_transactions %}
RECENT INSIDER TRANSACTIONS
---------------------------
{% for trans in insider_transactions %}
{{ trans.ticker }}: {{ trans.name }} ({{ trans.title }}) - {{ trans.type }} ${{ "{:,.0f}".format(trans.value) }} on {{ trans.date }}
{% endfor %}
{% endif %}

---
Generated: {{ generated_at }}

DISCLAIMER: This report is for informational purposes only and does not constitute investment advice.
"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.scorer = SignalScorer(config_path)

    def get_recent_insider_transactions(self, days: int = 7) -> List[Dict]:
        """Get recent significant insider transactions."""
        cutoff = date.today() - timedelta(days=days)

        transactions = self.db.execute("""
            SELECT
                company_ticker as ticker,
                insider_name as name,
                insider_title as title,
                transaction_type as type,
                transaction_value as value,
                transaction_date as date
            FROM insider_transactions
            WHERE transaction_date >= %s
            AND is_10b5_1_plan = FALSE
            AND transaction_value >= %s
            ORDER BY transaction_date DESC, transaction_value DESC
            LIMIT 10
        """, (cutoff, self.config.signals.get('min_transaction_value', 50000)))

        return transactions

    def get_recent_institutional_changes(self) -> List[Dict]:
        """Get recent significant institutional position changes."""
        # Get most recent quarter
        latest_quarter = self.db.execute_one("""
            SELECT MAX(quarter_end) as quarter FROM institutional_holdings
        """)

        if not latest_quarter or not latest_quarter['quarter']:
            return []

        changes = self.db.execute("""
            SELECT
                fund_name as fund,
                company_ticker as ticker,
                pct_change_shares as pct_change,
                CASE
                    WHEN is_new_position THEN 'New Position'
                    WHEN is_exit THEN 'Exit'
                    WHEN pct_change_shares > 0 THEN 'Increased'
                    ELSE 'Decreased'
                END as type
            FROM institutional_holdings
            WHERE quarter_end = %s
            AND (is_new_position OR is_exit OR ABS(pct_change_shares) > 25)
            ORDER BY ABS(pct_change_shares) DESC
            LIMIT 10
        """, (latest_quarter['quarter'],))

        return changes

    def get_job_highlights(self) -> Dict[str, Dict]:
        """Get job posting highlights by company."""
        stats = self.db.execute("""
            SELECT
                company_ticker,
                COUNT(*) as total,
                SUM(CASE WHEN is_commercial_role THEN 1 ELSE 0 END) as commercial,
                SUM(CASE WHEN is_clinical_role THEN 1 ELSE 0 END) as clinical
            FROM job_postings
            WHERE removal_date IS NULL
            AND first_seen_date >= %s
            GROUP BY company_ticker
            HAVING COUNT(*) >= 5
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """, (date.today() - timedelta(days=30),))

        return {
            s['company_ticker']: {
                'total': s['total'],
                'commercial': s['commercial'],
                'clinical': s['clinical']
            }
            for s in stats
        }

    def generate_digest(self) -> Dict[str, Any]:
        """Generate the daily digest content."""
        logger.info("Generating daily digest...")

        # Get top signals
        top_bullish, top_bearish = self.scorer.get_top_signals(n_bullish=5, n_bearish=5)

        # Prepare signal data for template
        bullish_data = [
            {
                'company_ticker': s.company_ticker,
                'composite_score': s.composite_score,
                'confidence': s.confidence,
                'signal_count': s.signal_count,
                'recommendation': s.recommendation,
                'signals': [
                    {'description': sig.description}
                    for sig in s.contributing_signals[:3]
                ]
            }
            for s in top_bullish
        ]

        bearish_data = [
            {
                'company_ticker': s.company_ticker,
                'composite_score': s.composite_score,
                'confidence': s.confidence,
                'signal_count': s.signal_count,
                'recommendation': s.recommendation,
                'signals': [
                    {'description': sig.description}
                    for sig in s.contributing_signals[:3]
                ]
            }
            for s in top_bearish
        ]

        # Get supporting data
        insider_transactions = self.get_recent_insider_transactions()
        institutional_changes = self.get_recent_institutional_changes()
        job_highlights = self.get_job_highlights()

        # Count signals
        all_scores = self.scorer.score_all_companies()
        total_signals = sum(s.signal_count for s in all_scores)
        bullish_count = len([s for s in all_scores if s.composite_score > 0])
        bearish_count = len([s for s in all_scores if s.composite_score < 0])

        # Template context
        context = {
            'date': date.today().strftime('%B %d, %Y'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signals': total_signals,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'top_bullish': bullish_data,
            'top_bearish': bearish_data,
            'insider_transactions': insider_transactions,
            'institutional_changes': institutional_changes,
            'job_highlights': job_highlights,
        }

        # Render templates
        html_template = Template(self.HTML_TEMPLATE)
        text_template = Template(self.TEXT_TEMPLATE)

        html_content = html_template.render(**context)
        text_content = text_template.render(**context)

        return {
            'html': html_content,
            'text': text_content,
            'context': context
        }

    def send_email(self, subject: str, html_content: str, text_content: str) -> bool:
        """Send the digest email."""
        email_config = self.config.email

        if not email_config.get('sender_email') or not email_config.get('recipients'):
            logger.warning("Email not configured, skipping send")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])

            # Attach plain text and HTML versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)

            # Connect and send
            with smtplib.SMTP(
                email_config.get('smtp_server', 'smtp.gmail.com'),
                email_config.get('smtp_port', 587)
            ) as server:
                server.starttls()
                server.login(
                    email_config['sender_email'],
                    email_config['sender_password']
                )
                server.send_message(msg)

            logger.info(f"Email sent to {len(email_config['recipients'])} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def save_digest(self, context: Dict, html_content: str,
                    text_content: str, status: str = 'pending') -> int:
        """Save digest to database."""
        data = {
            'digest_date': date.today(),
            'recipient_count': len(self.config.email.get('recipients', [])),
            'top_bullish': context.get('top_bullish'),
            'top_bearish': context.get('top_bearish'),
            'new_signals_count': context.get('total_signals', 0),
            'content_html': html_content,
            'content_text': text_content,
            'status': status
        }

        try:
            digest_id = self.db.insert('email_digests', data)
            return digest_id
        except Exception as e:
            logger.error(f"Failed to save digest: {e}")
            return None

    def run(self, send: bool = True, save_html: bool = False) -> Dict[str, Any]:
        """
        Generate and optionally send the daily digest.

        Args:
            send: Whether to send the email
            save_html: Whether to save HTML to file

        Returns:
            Dictionary with run results
        """
        # Generate content
        digest = self.generate_digest()

        # Save to file if requested
        if save_html:
            html_path = f"reports/digest_{date.today().isoformat()}.html"
            with open(html_path, 'w') as f:
                f.write(digest['html'])
            logger.info(f"Saved HTML digest to {html_path}")

        # Save to database
        digest_id = self.save_digest(
            digest['context'],
            digest['html'],
            digest['text'],
            status='pending'
        )

        # Send email
        sent = False
        if send:
            subject = f"Biotech Signal Digest - {date.today().strftime('%B %d, %Y')}"
            sent = self.send_email(subject, digest['html'], digest['text'])

            # Update status
            if digest_id:
                self.db.execute("""
                    UPDATE email_digests
                    SET status = %s, sent_at = %s
                    WHERE digest_id = %s
                """, ('sent' if sent else 'failed', datetime.now() if sent else None, digest_id))

        return {
            'digest_id': digest_id,
            'sent': sent,
            'top_bullish': len(digest['context'].get('top_bullish', [])),
            'top_bearish': len(digest['context'].get('top_bearish', [])),
            'total_signals': digest['context'].get('total_signals', 0)
        }


if __name__ == '__main__':
    # Test the digest
    digest = EmailDigest()
    result = digest.run(send=False, save_html=True)
    print(f"Digest result: {result}")
